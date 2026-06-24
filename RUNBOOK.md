# Day-of runbook — Qwen3-8B spec-decoding + FP8 on one H100

A time-budgeted checklist for the server session. We'll work through this together live.
Goal: spend GPU time only on GPU-bound work; everything else is already done.

## Budget the day

Realistic wall-clock on a single H100 (model download on a fast link):

| Step | What | venv | Rough time |
| --- | --- | --- | ---: |
| 0 | `setup.sh` (build 3 venvs, clone speculators) | — | 10–25 min |
| 1 | `01_prepare_data.sh` (tokenize 3000 ShareGPT) | speculators | 5–15 min |
| 2+3 | launch vLLM + generate hidden states (~140GB) | vllm + speculators | **1–3 hr** ⟵ long pole |
| Q | `quantize_fp8.py` (FP8 verifier) | comp | 5–15 min |
| 4 | `04_train_eagle3.sh` (5 epochs) | speculators | **30 min–2 hr** ⟵ long pole |
| 5 | benchmark sweeps (6–8 server starts) | vllm | 45–90 min |
| — | paste results, finish answers | — | 30 min |

**Plan for a full day (~5–8 hr).** Two long poles: hidden-state generation and training.
Single GPU = everything is serial; there's no GPU parallelism to exploit.

## Before the first command (do these in order, once)

```bash
export DATA_ROOT=/workspace            # <- your ≥200GB disk; CONFIRM the real mount first
export HF_HOME=$DATA_ROOT/hf_cache     # keep the 16GB model off the boot disk
export HF_TOKEN=hf_...                 # the token you made in prep
df -h "$DATA_ROOT"                     # sanity: ≥200GB free?
chmod +x setup.sh quantize_fp8.py run/*.sh   # (already done in prep, harmless to repeat)
```

> ⚠️ The #1 day-of failure is the 140GB hidden-state dump landing on a small boot disk.
> If `DATA_ROOT` is wrong, everything downstream is in the wrong place. Confirm `df -h` first.
> ⚠️ A **fresh terminal forgets these exports.** Re-run all four `export`s in every new shell
> (the two-terminal hidden-states step especially).

## Recommended order (note: quantize slotted at step Q, after the server frees the GPU)

### 0 — Build environments
```bash
./setup.sh
```
Healthy: ends with "Setup complete. Three environments created". If `uv` install of vllm
fails, it's almost always a version/network hiccup — re-run, it's idempotent.

### 1 — Prepare data
```bash
./run/01_prepare_data.sh
```
Healthy: writes a preprocessed dataset under `$DATA_ROOT/output`, prints sample count.

### 2 + 3 — Generate hidden states (TWO terminals)
Terminal A (leave running):
```bash
./run/02_launch_vllm.sh     # wait for "Application startup complete"
```
Terminal B (re-export the 4 vars first!):
```bash
./run/03_gen_hidden_states.sh
```
Healthy: B shows a progress bar; `du -sh $DATA_ROOT/output/hidden_states` climbs toward
~140GB. When B finishes, **Ctrl+C the server in A** (frees the GPU for the next steps).
- Missing temp files? `rm -rf /tmp/hidden_states/*` and re-run B — it resumes.
- Hidden-state length ≠ tokenized length? Check the vLLM version, not the recipe.
- Disk filling too fast? Lower `MAX_SAMPLES` in `env.sh` before other knobs.

### Q — Quantize the verifier (GPU is now free)
```bash
source comp_venv/bin/activate
python quantize_fp8.py
deactivate
```
Healthy: prints "OK — quantization section present" and a `quantization_config` block.
Writes `$DATA_ROOT/Qwen3-8B-FP8-Dynamic/`. Original model untouched. **Do this before
training so it's banked** — it's independent of the draft head and cheap.

### 4 — Train the draft head
```bash
./run/04_train_eagle3.sh
```
Watch the **per-position** metrics, not just total loss:
- `full_acc_0` should reach **~0.45+** (reference 0.463). If it's near zero, stop —
  that's a data-gen problem (step 3), not training. See suggestions.md Task 2 Q3.
- `full_acc_1/2` and `cond_acc_1/2` will be lower; that's expected (later positions harder).
- Best checkpoint symlinks to `$DATA_ROOT/output/checkpoints/checkpoint_best`.

### 5 — Benchmark + tune draft tokens
```bash
./run/05_benchmark.sh baseline          # -> results/baseline.txt   (sanity vs ~841 tok/s)
./run/05_benchmark.sh spec 2            # -> results/spec_ntok2.txt
./run/05_benchmark.sh spec 3            # -> results/spec_ntok3.txt   (tune unquantized)
./run/05_benchmark.sh fp8               # -> results/fp8.txt
./run/05_benchmark.sh fp8_spec 1        # -> results/fp8_spec_ntok1.txt
./run/05_benchmark.sh fp8_spec 2        # -> results/fp8_spec_ntok2.txt  (tune FP8)
```
Each run starts a server, waits for health, benchmarks, tears down. Pick the winner of each
sweep by **output tok/s first**, then justify with acceptance length + TPOT (see suggestions.md).

**Gates (each pass/fail):** spec **> 1250**, fp8 **> 1550**, fp8+spec **> 1750**.
The spec gate is tight (reference 1258) — if a sweep value lands just under, try the other
draft-token count before moving on.

### Finish — assemble the submission
- Paste the three raw `============ Serving Benchmark Result ============` blocks into the
  notebook's TODO cells (cell 6 = spec, 7 = fp8, 8 = fp8+spec).
- Write the question answers using `suggestions.md`, swapping in your real numbers.
- Make the quantize-first case (see suggestions.md headline section).

## If we're behind on time — triage
- **Hidden-state gen too slow / disk tight:** lower `MAX_SAMPLES` (e.g. 2000) in `env.sh`.
  More samples mainly help draft-head *quality*; you can still clear gates with fewer.
- **Training slow:** the 5 epochs are the cost; the best checkpoint may already pass gates
  before epoch 5 — check the per-epoch metrics.
- **Don't skip:** the three graded benchmark configs and at least one tuning comparison per
  spec config (you must *show* you tuned draft tokens, not just assert it).

## Quick map (which venv does what)
| Step | Script | venv |
| --- | --- | --- |
| prep / hidden states / train | `run/01,03,04` | speculators_venv |
| serve + benchmark | `run/02,05` | vllm_venv |
| quantize | `quantize_fp8.py` | comp_venv |
