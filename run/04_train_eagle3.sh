#!/usr/bin/env bash
# Step 4 — train the EAGLE-3 draft head from cached hidden states.
# Runs in speculators_venv. The vLLM server is NOT needed here.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
source "$SPECULATORS_VENV/bin/activate"

cd "$SPECULATORS_DIR"
python scripts/train.py \
  --verifier-name-or-path "$MODEL" \
  --data-path "$PREP_DIR" \
  --hidden-states-path "$HS_DIR" \
  --save-path "$CKPT_DIR" \
  --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --total-seq-len "$SEQ_LEN" \
  --on-missing raise

echo "[04] Training done. Best checkpoint: $BEST_CKPT"
echo "[04] Watch the per-position val accuracy (full_acc_0/1/2, cond_acc_0/1/2),"
echo "     not just total loss. If full_acc_0 is very low, suspect data gen (step 3)."
