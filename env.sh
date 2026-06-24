#!/usr/bin/env bash
#
# env.sh — single source of truth for the whole homework.
# Every run/ script sources this, so all four benchmark configs share
# identical settings. Change a value here once, not in five places.
#
# This file is SOURCED, not executed. Don't add `set -e` here.

# Resolve the repo root from this file's location, so paths work
# regardless of which directory you run a script from.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------
# Where big files live. Override before running if your large disk is
# mounted elsewhere, e.g.  export DATA_ROOT=/workspace
# ----------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"

# If Qwen3-8B ever needs auth, export this in your shell (do NOT commit it):
# export HF_TOKEN=hf_xxx

# ----------------------------------------------------------------------
# Environments + cloned repo (created by setup.sh)
# ----------------------------------------------------------------------
SPECULATORS_DIR="$PROJECT_ROOT/speculators"
SPECULATORS_VENV="$PROJECT_ROOT/speculators_venv"
VLLM_VENV="$PROJECT_ROOT/vllm_venv"
COMP_VENV="$PROJECT_ROOT/comp_venv"

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
MODEL="Qwen/Qwen3-8B"

# ----------------------------------------------------------------------
# Generated artifacts (all under the big disk)
# ----------------------------------------------------------------------
PREP_DIR="$DATA_ROOT/output"                       # preprocessed dataset
HS_DIR="$DATA_ROOT/output/hidden_states"           # cached hidden states (~140GB+)
CKPT_DIR="$DATA_ROOT/output/checkpoints"           # EAGLE-3 training checkpoints
BEST_CKPT="$CKPT_DIR/checkpoint_best"              # best draft head (symlink)
FP8_DIR="$DATA_ROOT/Qwen3-8B-FP8-Dynamic"          # quantized verifier
RESULTS_DIR="$PROJECT_ROOT/results"                # benchmark text output (small; commit)

# ----------------------------------------------------------------------
# Data prep + training
# ----------------------------------------------------------------------
MAX_SAMPLES=3000          # tutorial-recommended starting point
SEQ_LEN=2048              # sequence length for prep + training
DRAFT_VOCAB_SIZE=32000    # reduced draft-head vocab (tutorial default)
EPOCHS=5
LR=1e-4

# ----------------------------------------------------------------------
# Serving + benchmarking (kept identical across all four configs)
# ----------------------------------------------------------------------
PORT=8000                 # if you change this, also pass --port to `vllm bench serve`
SEED=42
GPU_MEM_UTIL=0.9
CONCURRENCY=8             # --max-concurrency
NUM_PROMPTS=80
DATASET_PATH="philschmid/mt-bench"

# Speculative draft tokens — TUNE THESE. The reference used different
# values for the unquantized vs FP8 runs, so they are separate on purpose.
SPEC_TOKENS_UNQUANT=2
SPEC_TOKENS_FP8=1
