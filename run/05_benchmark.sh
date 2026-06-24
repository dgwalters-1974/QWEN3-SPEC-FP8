#!/usr/bin/env bash
# Step 5 — serve ONE config and benchmark it. Runs in vllm_venv.
#
# Usage:
#   ./run/05_benchmark.sh <baseline|spec|fp8|fp8_spec> [num_speculative_tokens]
#
# Examples:
#   ./run/05_benchmark.sh baseline
#   ./run/05_benchmark.sh spec 2          # tune draft tokens for the unquantized run
#   ./run/05_benchmark.sh spec 3
#   ./run/05_benchmark.sh fp8
#   ./run/05_benchmark.sh fp8_spec 1      # tune draft tokens for the FP8 run
#
# It starts the server in the background, waits until it's healthy, runs
# `vllm bench serve` with the shared settings, saves the output to
# results/<label>.txt, then shuts the server down. All four configs use the
# same dataset, concurrency, prompt count, seed, and prefix-caching=off.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
source "$VLLM_VENV/bin/activate"

CONFIG="${1:?usage: 05_benchmark.sh <baseline|spec|fp8|fp8_spec> [num_speculative_tokens]}"
SPEC_OVERRIDE="${2:-}"

mkdir -p "$RESULTS_DIR"
SERVE_ARGS=()

case "$CONFIG" in
  baseline)
    TARGET="$MODEL"
    ;;
  spec)
    TARGET="$MODEL"
    NTOK="${SPEC_OVERRIDE:-$SPEC_TOKENS_UNQUANT}"
    SERVE_ARGS=(--speculative-config "{\"model\": \"$BEST_CKPT\", \"num_speculative_tokens\": $NTOK, \"method\": \"eagle3\"}")
    ;;
  fp8)
    TARGET="$FP8_DIR"
    ;;
  fp8_spec)
    TARGET="$FP8_DIR"
    NTOK="${SPEC_OVERRIDE:-$SPEC_TOKENS_FP8}"
    SERVE_ARGS=(--speculative-config "{\"model\": \"$BEST_CKPT\", \"num_speculative_tokens\": $NTOK, \"method\": \"eagle3\"}")
    ;;
  *)
    echo "unknown config: $CONFIG (expected baseline|spec|fp8|fp8_spec)" >&2
    exit 1
    ;;
esac

LABEL="$CONFIG${SPEC_OVERRIDE:+_ntok$SPEC_OVERRIDE}"
SERVER_LOG="$RESULTS_DIR/${LABEL}_server.log"
RESULT_FILE="$RESULTS_DIR/${LABEL}.txt"

echo "[05] config=$CONFIG  target=$TARGET  ${NTOK:+draft_tokens=$NTOK}"
echo "[05] starting server (log: $SERVER_LOG)"
CUDA_VISIBLE_DEVICES=0 vllm serve "$TARGET" \
  -tp 1 \
  --port "$PORT" \
  --seed "$SEED" \
  --no-enable-prefix-caching \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "${SERVE_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Always shut the server down on exit (success, failure, or Ctrl+C).
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[05] waiting for server health on port $PORT ..."
for _ in $(seq 1 600); do          # up to ~20 min for model load
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "[05] server is up."
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[05] server died during startup — see $SERVER_LOG" >&2
    exit 1
  fi
  sleep 2
done

echo "[05] benchmarking -> $RESULT_FILE"
vllm bench serve \
  --model "$TARGET" \
  --dataset-name hf \
  --dataset-path "$DATASET_PATH" \
  --max-concurrency "$CONCURRENCY" \
  --temperature 0 --num-prompts "$NUM_PROMPTS" \
  | tee "$RESULT_FILE"

echo "[05] done. Saved $RESULT_FILE"
echo "[05] (server will be shut down now)"
