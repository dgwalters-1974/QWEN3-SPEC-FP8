#!/usr/bin/env bash
#
# setup.sh — one-shot environment setup for the
# Speculative Decoding + FP8 Quantization homework (Qwen/Qwen3-8B).
#
# Builds three isolated Python 3.12 environments using `uv`:
#   speculators_venv  -> data prep, hidden-state generation, EAGLE-3 training
#   vllm_venv         -> serving + benchmarking
#   comp_venv         -> FP8 quantization (llmcompressor)
#
# Safe to re-run: if the box restarts and wipes everything, just run it again.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Optional: point DATA_ROOT at your big disk BEFORE running, e.g.
#   DATA_ROOT=/workspace ./setup.sh
# (hidden states need ~140GB+, so this should NOT be a small boot disk.)

set -euo pipefail

# ----------------------------------------------------------------------
# Config — pinned versions come straight from the homework spec
# ----------------------------------------------------------------------
PYTHON_VERSION="3.12"
SPECULATORS_TAG="v0.5.0"
VLLM_VERSION="0.20.0"
FASTAPI_CONSTRAINT="<0.137"
LLMCOMPRESSOR_VERSION="0.12.0"

PROJECT_ROOT="$(pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT}"          # where big data + HF cache live
HF_CACHE="$DATA_ROOT/hf_cache"                   # model downloads go here

echo "=============================================="
echo " Project root : $PROJECT_ROOT"
echo " Data root    : $DATA_ROOT   (large files + HF cache)"
echo "=============================================="

# ----------------------------------------------------------------------
# 1. Install uv if it isn't already present
#    (uv is a fast venv/pip replacement that can also fetch Python 3.12)
# ----------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version

# Make sure the big disk dirs exist, and route HF downloads there
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"
echo "[setup] HF_HOME set to $HF_HOME"

# ----------------------------------------------------------------------
# 2. speculators_venv  — clone the repo and install it editable
#    (we need the repo's scripts/ folder, so a plain pip install won't do)
# ----------------------------------------------------------------------
echo "[setup] === speculators_venv ==="
if [ ! -d "$PROJECT_ROOT/speculators" ]; then
  git clone --branch "$SPECULATORS_TAG" --depth 1 \
    https://github.com/vllm-project/speculators.git "$PROJECT_ROOT/speculators"
fi

uv venv --python "$PYTHON_VERSION" "$PROJECT_ROOT/speculators_venv"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/speculators_venv/bin/activate"
uv pip install -e "$PROJECT_ROOT/speculators"
uv pip install tensorboard            # optional: lets you watch training curves
deactivate

# ----------------------------------------------------------------------
# 3. vllm_venv  — serving + benchmarking runtime
# ----------------------------------------------------------------------
echo "[setup] === vllm_venv ==="
uv venv --python "$PYTHON_VERSION" "$PROJECT_ROOT/vllm_venv"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/vllm_venv/bin/activate"
uv pip install "vllm==$VLLM_VERSION" "fastapi$FASTAPI_CONSTRAINT"
deactivate

# ----------------------------------------------------------------------
# 4. comp_venv  — FP8 dynamic quantization
# ----------------------------------------------------------------------
echo "[setup] === comp_venv ==="
uv venv --python "$PYTHON_VERSION" "$PROJECT_ROOT/comp_venv"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/comp_venv/bin/activate"
uv pip install "llmcompressor==$LLMCOMPRESSOR_VERSION"
deactivate

# ----------------------------------------------------------------------
# 5. Summary
# ----------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Setup complete. Three environments created:"
echo "   speculators_venv  (repo cloned at tag $SPECULATORS_TAG, editable)"
echo "   vllm_venv         (vllm $VLLM_VERSION, fastapi $FASTAPI_CONSTRAINT)"
echo "   comp_venv         (llmcompressor $LLMCOMPRESSOR_VERSION)"
echo ""
echo " Activate one with, e.g.:"
echo "   source speculators_venv/bin/activate"
echo ""
echo " IMPORTANT: re-export HF_HOME in any NEW shell so model"
echo " downloads land on the big disk, not the boot disk:"
echo "   export HF_HOME=$HF_CACHE"
echo "=============================================="
