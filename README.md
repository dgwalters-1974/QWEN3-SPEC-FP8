# Qwen3-8B — Speculative Decoding + FP8 Quantization

Scaffold for the homework: train an EAGLE-3 draft head for `Qwen/Qwen3-8B`,
FP8-quantize the verifier, and benchmark four serving configurations on a
single H100.

Everything heavy (the model, hidden states, checkpoints, the cloned
`speculators` repo, the venvs) is **regenerated on the box** and is
git-ignored. The repo holds only the things you author.

## Repo layout

```
.
├── README.md            <- you are here (the runbook)
├── setup.sh             <- builds the 3 venvs + clones speculators
├── env.sh               <- all shared settings (edit here, sourced everywhere)
├── quantize_fp8.py      <- FP8 dynamic quantization (comp_venv)
├── run/
│   ├── 01_prepare_data.sh
│   ├── 02_launch_vllm.sh
│   ├── 03_gen_hidden_states.sh
│   ├── 04_train_eagle3.sh
│   └── 05_benchmark.sh
├── results/             <- benchmark output per config (commit the .txt files)
└── submission.ipynb     <- the notebook you fill in and hand back
```

## Which environment each step uses

| Step | Script | venv |
| --- | --- | --- |
| Prepare data | `run/01_prepare_data.sh` | `speculators_venv` |
| Launch vLLM (for hidden states) | `run/02_launch_vllm.sh` | `vllm_venv` |
| Generate hidden states | `run/03_gen_hidden_states.sh` | `speculators_venv` |
| Train draft head | `run/04_train_eagle3.sh` | `speculators_venv` |
| Quantize verifier | `quantize_fp8.py` | `comp_venv` |
| Serve + benchmark | `run/05_benchmark.sh` | `vllm_venv` |

The scripts activate the right venv for you. Run everything **from the repo
root**.

## Before you start

- This needs a real GPU box (1× H100 80GB) with **≥ ~200GB of disk** for the
  hidden states. If your big disk is mounted somewhere like `/workspace`,
  export `DATA_ROOT` so all heavy files land there:
  ```
  export DATA_ROOT=/workspace
  ```
- Make the scripts executable once:
  ```
  chmod +x setup.sh quantize_fp8.py run/*.sh
  ```

## Run order

### 0. Build environments
```
DATA_ROOT=/workspace ./setup.sh      # omit DATA_ROOT to use the current folder
```

### 1. Prepare data
```
./run/01_prepare_data.sh
```

### 2–3. Generate hidden states (two terminals)
Hidden-state generation talks to a running vLLM server, so use two terminals.

Terminal A (leave it running):
```
./run/02_launch_vllm.sh
```
Terminal B (once the server prints "Application startup complete"):
```
./run/03_gen_hidden_states.sh
```
When step 3 finishes, Ctrl+C the server in Terminal A.

> Disk warning: a few thousand samples can reach ~140GB here. If generation
> complains about missing temp files, clear `/tmp/hidden_states/*` and re-run
> step 3 — it resumes from where it stopped.

### 4. Train the draft head
```
./run/04_train_eagle3.sh
```
The best checkpoint is symlinked at `output/checkpoints/checkpoint_best`.
Watch the per-position accuracy (`full_acc_0/1/2`, `cond_acc_0/1/2`), not just
total loss. If first-position accuracy is very low, the problem is almost
always in data generation (step 3), not the training recipe.

### 5. Quantize the verifier
```
source comp_venv/bin/activate
python quantize_fp8.py
deactivate
```
This writes `Qwen3-8B-FP8-Dynamic/` and verifies the saved config has a
quantization section. The original model is never touched.

### 6. Benchmark all four configs
Each call starts a server, benchmarks it with identical settings, saves the
output to `results/`, and shuts the server down.
```
./run/05_benchmark.sh baseline
./run/05_benchmark.sh spec 2
./run/05_benchmark.sh fp8
./run/05_benchmark.sh fp8_spec 1
```

## Tuning the speculative draft tokens

The homework requires tuning the number of draft tokens **separately** for the
unquantized and FP8 runs — don't assume one value is best for both. Sweep by
re-running step 5 with different values; each lands in its own results file:
```
./run/05_benchmark.sh spec 2        # -> results/spec_ntok2.txt
./run/05_benchmark.sh spec 3        # -> results/spec_ntok3.txt
./run/05_benchmark.sh fp8_spec 1    # -> results/fp8_spec_ntok1.txt
./run/05_benchmark.sh fp8_spec 2    # -> results/fp8_spec_ntok2.txt
```
Pick the winner using **output token throughput** first, then justify it with
acceptance rate, acceptance length, and TPOT. A setting that drafts more but
accepts little is the wrong setting.

## What you're graded on (output tok/s)

| Config | Threshold |
| --- | --- |
| Speculative decoding (with trained head) | > 1250 tok/s |
| FP8 dynamic quantization | > 1550 tok/s |
| Best FP8 + speculative decoding | > 1750 tok/s |

## Submission

Paste the raw `results/*.txt` blocks into `submission.ipynb` for the three
required configs (speculative decoding, FP8, FP8 + speculative decoding),
answer the written questions, and make the case for **quantize-first**. Do not
submit the venvs.

## Gotchas

- Hidden states not matching tokenized lengths → check the vLLM version first.
- A fresh terminal won't have `HF_HOME` set; re-export `DATA_ROOT` (or
  `HF_HOME`) so model downloads don't fill the small boot disk.
- If `vllm bench serve` can't reach the server, confirm the port in `env.sh`
  matches and the server log under `results/` shows a clean startup.
