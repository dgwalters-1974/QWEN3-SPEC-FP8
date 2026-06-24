#!/usr/bin/env bash
# Step 2 — launch the vLLM server that exposes verifier hidden states.
# Runs in vllm_venv. This BLOCKS (runs in the foreground): leave it running
# in this terminal and run step 3 in a SECOND terminal. Ctrl+C to stop when
# hidden-state generation is finished.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
source "$VLLM_VENV/bin/activate"

cd "$SPECULATORS_DIR"
# Single H100 -> single GPU, no data-parallelism.
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py \
  "$MODEL" \
  -- --port "$PORT" --gpu-memory-utilization "$GPU_MEM_UTIL"
