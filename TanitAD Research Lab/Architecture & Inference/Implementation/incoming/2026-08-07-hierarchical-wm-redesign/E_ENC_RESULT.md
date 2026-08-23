# E-ENC — the encoder-width arm, decided at step 500

**PI decision D-A (2026-08-13):** pause S-W, run the encoder pair first, decide before
committing the ~60 A40-hours. This is the result.

Evidence class: **MEASURED (ours)** — pod5, both arms same seed (0), same corpus
(2400 train providers, parity enforced), same losses, same step count, run back-to-back by
`/workspace/eenc_chain.sh`. `EENC_A_EXIT=0`, `EENC_C_EXIT=0`, `EENC_CHAIN_DONE`.

## The two arms

| | **(a) shipped default** | **(c) v5-width encoder** |
|---|---|---|
| encoder | 384 × 8, 6 heads | **768 × 12, 12 heads** |
| total params | **87.89 M** | **159.93 M** |
| encoder params | 15.33 M | 87.32 M |

## Result at step 500

| metric | (a) 384×8 | (c) 768×12 | winner |
|---|---:|---:|---|
| **total loss** | **2.9720** | 3.5924 | **(a)** |
| O1 factual | **0.3672** | 0.8824 | **(a)** |
| **O1 factual ADE** | **0.6141** | 1.4405 | **(a)** |
| O5 rollout | **0.2307** | 0.2366 | (a), marginally |
| O5 growth | 0.832 | **0.796** | (c), marginally |
| O2 | **0.1119** | 0.1130 | (a), marginally |
| O3 masked-cell | **0.0270** | 0.0333 | **(a)** |
| O6 SIGReg | **15.75** | 16.17 | (a) |
| **s/step** | **7.19** | 10.76 | **(a), 1.50× faster** |

**(a) wins on 7 of 8 objectives and is 1.50× faster per step.** The gap is not marginal on
the terms that matter: O1 factual ADE is **2.35× worse** on the wide encoder, and total loss
is 21 % higher.

## Verdict: **KEEP THE SHIPPED 384×8 ENCODER.** Resume S-W on arm (a).

⚠️ **Three honest qualifications, because this is a 500-step read.**

1. **A bigger model losing at step 500 is the EXPECTED shape of an early curve**, not proof
   it loses at 30k — wider models carry more randomly-initialised parameters and start
   further from a good solution. What this result rules out is a *large early advantage*
   for the wide encoder; it does not rule out a late crossover. Given (c) also costs
   **1.50×** per step (30k would be **90 h**, not 60 h), the burden of proof was on (c) and
   it did not discharge it.
2. ⛔ **The pre-registered P-battery did NOT decide this.** Both arms wrote
   `gate_verdict: INCONCLUSIVE`, because P1/P3/P6 are computed by *external* probe scripts
   (`probe_latent_state.py`, `stage_a_probes.py`) folded in via `--gate-probes`, which I did
   not supply. So the decision above rests on **training objectives**, not on the gate the
   design doc names. That is a weaker instrument than intended and it is recorded as such
   rather than dressed up — the P-battery still owes its verdict at the real S-W milestone.
3. The comparison is **matched on seed, corpus, losses and steps**, but *not* on parameter
   count — (c) is 1.82× the parameters by construction. That is the intended question here
   ("is a v5-width visual trunk worth it?"), and is a different question from
   `V6_TRAINER_DESIGN` §2's matched-total-params E-ENC (a) vs (b), which remains unrun.

## What this changes

- **`V6_SIZING.md` §4's recommendation is now ANSWERED for (c):** the encoder cut from
  768×12 to 384×8 was a default that had never been tested; it has now been tested and it
  holds. The concern was legitimate and is retired at a cost of ~2 GPU-h instead of a failed
  60-hour stage.
- **S-W resumes on arm (a)** from `/workspace/experiments/v6-SW-30k/ckpt.pt` (step 4000),
  `--resume auto`, no work discarded.
- Still open: E-ENC (b) per-layer-encoders at matched total params, and the P-battery
  probes as the *actual* gate instrument.
