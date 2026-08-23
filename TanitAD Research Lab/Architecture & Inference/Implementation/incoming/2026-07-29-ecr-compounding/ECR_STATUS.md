# E-CR — the driver runs, and the metric is NOT yet trustworthy

**Status 2026-07-29 ~02:2x UTC: SMOKE ONLY (4 episodes / 48 windows). ⛔ NO VERDICT ON C61.**
Pre-registration: `Project Steering/PREREG_deep_research_2026-07-29.md`.

## What ran

`ecr_sweep.py` (banked beside this file; lives on pod3 at `/workspace/ecr_sweep.py`) mirrors
`train_flagship_v4.canary_rollout` exactly — same `load_v1_from_ck`, same
`build_val_dataset_base`, same `val40cache`, same action assembly, same `grounding.step["op"]`,
same autocast placement — and adds one arm whose window advances with the **true** latent.

**v1 `flagship4b-speedjerk-30k`, step 29,999 · 4 episodes · 48 windows · MEASURED:**

| k | e_rollout | e_teacher_forced | **CR** | ER |
|---|---|---|---|---|
| 4 | 0.04491 | 0.06160 | **0.7291** | 0.00930 |
| 8 | 0.08582 | 0.13581 | **0.6319** | 0.01481 |
| 16 | 0.29171 | 0.44242 | **0.6594** | 0.03840 |
| 20 | 0.53253 | 0.70582 | **0.7545** | 0.07279 |

## ⛔ CR < 1 EVERYWHERE — teacher forcing is WORSE. Do not report this as "no compounding".

`CR < 1` says the arm fed the **true** latent predicts **worse** than the arm fed its own
prediction. Under the pre-registered reading that is not even one of the two registered outcomes:
H-TASK predicts CR ≈ 1, H-COMPOUND predicts CR > 1. **An out-of-range result is a specification
problem until proven otherwise, not a discovery.**

### Two candidate bugs RULED OUT by measurement (not by reasoning)

1. **Wrong latent space.** `z_true` is built with `world.encode` (single frame) while the predictor
   window is filled by `world.encode_window`. If those differ, teacher forcing injects
   out-of-distribution states. **MEASURED: they agree** — `cosine = 0.999995`,
   `rel L2 = 0.0031`, `max|diff| = 0.0156` (bf16 rounding). *(`z_check.py`, pod3.)*
2. **Off-by-one alignment.** At step `j = 0` the predictor sees frames `t … t+W-1` and predicts the
   latent of frame `t+W`. The driver takes `z_true[:, j]` = last frame of window `(e, t+j+1)`
   = frame `t+j+W`; at `j = 0` that is `t+W`. **Correct by construction.**

### ⭐ The mechanism that probably invalidates CR AS DEFINED

`step_readout` decodes a **pair** `(z_prev, z_hat)` into a displacement. The two arms feed it
**different pair distributions**:

| arm | z_prev at step j>0 | z_hat |
|---|---|---|
| rollout | the model's own previous prediction | predicted from a predicted window |
| teacher-forced | the **true** latent | predicted from a **true** window |

The rollout arm's pair is **self-consistent** — both halves come from the model's own (possibly
biased, e.g. mean-contracted) output distribution, which is what the readout was trained on. The
teacher-forced arm hands the readout a **mixed** `(true, predicted)` pair it never saw in training.
So CR may be measuring **readout input-distribution mismatch**, not recursion cost — and a
distribution mismatch of the wrong sign would push CR below 1 exactly as observed.

⇒ **CR as currently defined does not cleanly isolate compounding.** It was supposed to remove a
confound from the C61 numbers; if this mechanism is real it has introduced a different one.

## What has to happen before E-CR can adjudicate C61

1. **Discriminate the readout-mismatch hypothesis.** Cheapest test: score a *pure one-step* arm —
   decode `(z_true_{j-1}, z_true_j)`, i.e. hand the readout two TRUE latents. If that also scores
   worse than the self-consistent rollout pair, the readout is the confound and CR must be
   redefined (e.g. on latent error, not decoded displacement).
2. **Only then** scale to 40 episodes / 881 windows and run the **paired episode-cluster bootstrap**
   over `/workspace/ecr_arrays.npz` (already emitted per-window and per-episode for exactly this).
3. Report `CR` with its interval **on the difference**, never on the ratio (no ratio CI is computed).

⚠️ **C61 remains OPEN.** The retraction stands: the 1.03 → 1.91 exponent rise may not be used to
justify an architecture change, and E-ROLL / rollout-recovery / the Koopman lever all remain
blocked — not because CR came back flat, but because **CR has not yet produced an admissible
number at all**.

## Evidence class

| item | class |
|---|---|
| the four CR values above | **MEASURED**, but n=48 windows / 4 episodes, **no interval**, and specification in doubt |
| encode ≡ encode_window agreement | **MEASURED** (`z_check.py`) |
| alignment correctness | **MEASURED by construction** (index arithmetic, stated above) |
| the readout-mismatch mechanism | **HYPOTHESIS** — not yet tested |
