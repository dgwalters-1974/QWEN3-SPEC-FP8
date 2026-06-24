#!/usr/bin/env bash
# Step 3 — generate + cache hidden states offline.  Runs in speculators_venv.
# Run this in a SECOND terminal while step 2's server is up.
# Safe to re-run: it skips samples that already exist (resumes where it left off).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
source "$SPECULATORS_VENV/bin/activate"

cd "$SPECULATORS_DIR"
python scripts/data_generation_offline.py \
  --preprocessed-data "$PREP_DIR" \
  --endpoint "http://localhost:$PORT/v1" \
  --output "$HS_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --concurrency 32 \
  --validate-outputs

echo "[03] Hidden states written to $HS_DIR"
echo "[03] You can now Ctrl+C the vLLM server (step 2) — it isn't needed for training."
