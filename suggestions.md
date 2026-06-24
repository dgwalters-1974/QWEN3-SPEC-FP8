# Suggested answers — reference notes for the written questions

These are **draft answers to lean on, not text to paste verbatim.** Re-word them in
your own voice and, where a question asks for it, back the claim with your *own*
benchmark numbers once you have them. Reference numbers quoted below come from the
homework sheet's reference run, so swap in yours where they differ.

Key hardware/model facts used throughout:
- H100 (Hopper) has **native FP8 (E4M3) tensor cores** — FP8 is a compute speedup, not just memory savings.
- Qwen3-8B: hidden size 4096, ~36 layers, vocab ~151k. EAGLE-3 draws on **multiple layers'** hidden states.
- Decoding is **memory-bandwidth bound**: weights are streamed from HBM every token.

---

## The headline question — quantize first, or train the draft head first?

**Answer: quantize first.** Quantize the verifier to FP8, *then* build the EAGLE-3
draft head on top of the already-quantized verifier. Three independent reasons:

1. **Train against the verifier you actually deploy (alignment).**
   The draft head is trained to predict the verifier's hidden states and next-token
   distribution. Acceptance during serving is decided by comparing draft tokens to the
   *verifier's* distribution. If you train against BF16 and then serve FP8, the verifier
   the head was matched to has moved underneath it → acceptance drops. Evidence in the
   reference run: acceptance **length falls from 1.45 (unquantized) to 1.36 (FP8)** and
   the optimal draft-token count drops from 2 → 1. Quantizing first removes that
   train/serve mismatch.

2. **Cost asymmetry — do the cheap, foundational change first.**
   Quantization is minutes, needs no calibration data, and is fully reproducible.
   Training (plus the ~140GB hidden-state generation) is the expensive, slow step you
   don't want to repeat. Fixing the verifier *before* the expensive step means you never
   redo it; doing it after risks "now retrain everything against the new verifier."

3. **Generate hidden states once, from the final verifier.**
   The ~140GB hidden-state dump should come from the model you'll serve. Quantize-first
   means hidden states, training target, and serving verifier are all the same model —
   no distribution shift baked into the most expensive artifact.

> Honesty note on *this* scaffold: as written, `run/02–04` generate hidden states and
> train against the **BF16** model (`$MODEL`), then serving applies the head under both
> BF16 and FP8. That is exactly why the FP8+spec run wants fewer draft tokens and shows a
> shorter acceptance length — it's live evidence *for* the quantize-first argument. If you
> want to push the combined number higher, the principled change is to quantize first and
> train the head against the FP8 verifier. State the argument as the recommendation; cite
> the acceptance-length drop as the symptom of doing it in the other order.

---

## Task 1 — Why do hidden states need far more disk than the text dataset?

Text is stored as **token IDs**: compact integers, ~2–4 bytes per token. A 2048-token
sample is a few KB.

Hidden states are the verifier's **internal activation vectors** — dense floats. For each
token you store a hidden-size vector (4096 for Qwen3-8B) in bf16 (2 bytes), and EAGLE-3
needs hidden states from **several layers** (low/mid/high), so it's multiple such vectors
per token (plus auxiliary tensors). Roughly:

- text: ~2–4 bytes / token
- hidden states: ~3 layers × 4096 × 2 bytes ≈ **~24 KB / token**

That's roughly a **~6,000× expansion** (24 KB vs a few bytes per token: 24,000 ÷ 4 ≈ 6,000).
Across 3000 samples × ~2048 tokens × ~24 KB ≈ **~140 GB** (we measured ~122 GB on the box),
matching the homework's warning. You're trading disk for not recomputing the verifier
forward pass during training (the whole point of *offline* EAGLE-3).

---

## Task 2 — Training the draft head

**Q1. What do `full_acc` and `cond_acc` measure?**
Both measure how often the draft head's predicted token at speculative position *k* matches
the verifier's ground-truth token, but with different conditioning:

- **`cond_acc_k` (conditional):** accuracy at position *k* **given that positions 0…k-1 were
  predicted correctly** — the per-step difficulty in isolation.
- **`full_acc_k` (full / cumulative):** accuracy at position *k* counting the whole chain
  0…k as correct — i.e. roughly the **product** of the per-step accuracies. This is the one
  that maps to real **acceptance length**.

Sanity check against the reference run: at position 0 they're identical (0.463 = 0.463 —
nothing to condition on). Then `full_acc_1` (0.181) ≈ `cond_acc_0 × cond_acc_1`
(0.463 × 0.364 = 0.169), and `full_acc_2` (0.069) ≈ 0.181 × 0.320 = 0.058. So `full` is the
cumulative measure and `cond` is per-step-given-prefix-correct.

**Q2. Why does accuracy fall for later positions?**
Two compounding effects: (a) later draft tokens are generated **autoregressively on the
draft head's own earlier guesses**, so any early error propagates and the context diverges
from what the verifier would have seen; (b) predicting **further into the future is
intrinsically harder** — more plausible continuations. `full_acc` drops fastest because it's
a product of per-step accuracies; even `cond_acc` declines because position *k+2* is harder
than *k+1*.

**Q3. What if first-position accuracy is very low?**
A low **position-0** accuracy means the head can't predict the *immediate* next token from
the verifier's hidden state — that points **upstream to data generation, not the training
recipe** (the homework hint says exactly this). Check, in order:
- hidden-state/token **misalignment** (off-by-one, wrong layers extracted, sequence-length
  mismatch) — verify the **vLLM version** first, per the troubleshooting note;
- **stale/corrupt temp files** in `/tmp/hidden_states/*` — clear and regenerate;
- **tokenizer / chat template / assistant-mask** mismatch corrupting the training targets;
- same Qwen3-8B used for hidden states *and* as training verifier.
Only once data is verified clean: add more samples (more data helps draft quality more than
hyperparameter tweaks), then revisit LR/epochs.

---

## Task 3 — FP8 dynamic quantization

**Q1. Why is FP8 dynamic quantization useful on H100?**
- H100 has **native FP8 tensor cores**, so FP8 matmuls run at ~2× BF16 throughput — a real
  compute win, not only memory.
- Decoding is **memory-bandwidth bound**; FP8 halves weight bytes vs BF16, so ~2× less
  weight traffic from HBM per token → higher throughput, lower TPOT (reference: TPOT
  4.90 ms vs 7.28 ms baseline; output 1566 vs 841 tok/s).
- **Dynamic** activation quantization computes activation scales **per token at runtime** —
  no calibration dataset needed, and per-token scales track outliers better than a single
  static scale, so quality holds up.
- Smaller weights free HBM for **more KV cache** → higher concurrency.

**Q2. Why exclude `lm_head`?**
`lm_head` maps the hidden state to **full-vocab logits** (~151k). It's numerically sensitive:
small logit errors shift the argmax / sampling distribution and directly change which tokens
are emitted — and which draft tokens get **accepted**. It's a single layer, so quantizing it
buys almost no speedup while disproportionately hurting output quality and acceptance.
Standard FP8 recipes ignore it (often embeddings too).

**Q3. How can quantization affect speculative acceptance rate?**
Acceptance compares draft tokens to the **verifier's** distribution. Quantizing the verifier
shifts that distribution slightly from the BF16 model the head was trained against → the
draft's guesses match a bit less often → **shorter acceptance length** for the same draft
length (reference: 1.45 → 1.36). That's why the FP8 run's optimal draft-token count is lower.
Caveat on reading the numbers: acceptance **rate** = accepted/drafted *isn't* directly
comparable across different draft lengths — the FP8 run shows a *higher* rate (36.5% vs
22.5%) only because it drafts a single, easy first-position token. **Acceptance length** is
the cleaner cross-config signal, and it went down.

---

## Task 4 — Serve and benchmark

**Q1. Why can speculative decoding help even when acceptance is well below 100%?**
One verifier forward pass **verifies all draft tokens in parallel**, and verification is
memory-bandwidth bound, so checking a few extra positions is nearly free. Drafting is cheap
(small head). So as long as the **expected accepted tokens per verify step > 1** (i.e.
acceptance length > 1), you emit more than one token per expensive verifier pass and amortize
its HBM cost → higher throughput, lower TPOT. The breakeven is low; you don't need high
acceptance. Reference: accept length 1.45 → output 1258 vs 841 tok/s, TPOT 5.76 vs 7.28.

**Q2. How many speculative tokens are optimal, and why?**
The trade-off: more draft tokens = more potential tokens per verify, **but** each added
position has lower (compounding) acceptance and adds draft+verify overhead. Past a point you
do more draft work for little accepted gain.
- **Method:** pick the count that **maximizes output tok/s first**, then justify with
  acceptance length (should rise enough per added position to be worth it) and TPOT (should
  fall). If 1→2 barely lifts acceptance length or doesn't lower TPOT, stop at 1.
- **This setup (reference, confirm with your sweep):** unquantized optimum = **2** draft
  tokens (accept length 1.45); FP8 optimum = **1** (accept length 1.36). FP8 prefers fewer
  because quantization lowered acceptance, so the 2nd draft position rarely pays off.
- **Sweep to actually run:** `spec 2` vs `spec 3`; `fp8_spec 1` vs `fp8_spec 2`. Report the
  winner of each by output tok/s and explain with acceptance length + TPOT.

---

## Quick reference — the numbers you're chasing (reference run; thresholds are the gates)

| Config | Output tok/s | Pass threshold | Draft tok | Accept len | Accept rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 841 | — | — | — | — |
| Speculative decoding | 1258 | **> 1250** | 2 | 1.45 | 22.5% |
| FP8 | 1566 | **> 1550** | — | — | — |
| FP8 + spec | 1766 | **> 1750** | 1 | 1.36 | 36.5% |

The thresholds are tight — e.g. spec must clear 1250 and the reference only hit 1258. That's
the practical reason draft-token tuning matters: a mistuned count can drop you under the gate.
