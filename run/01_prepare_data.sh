#!/usr/bin/env bash
# Step 1 — preprocess ShareGPT data for Qwen3-8B.  Runs in speculators_venv.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
source "$SPECULATORS_VENV/bin/activate"

cd "$SPECULATORS_DIR"
python scripts/prepare_data.py \
  --model "$MODEL" \
  --data sharegpt \
  --output "$PREP_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --seq-length "$SEQ_LEN"

echo "[01] Prepared data written to $PREP_DIR"
