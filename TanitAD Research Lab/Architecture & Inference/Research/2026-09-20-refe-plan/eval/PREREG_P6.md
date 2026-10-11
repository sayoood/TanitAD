# P6: the paper-faithful full REFe run on one A40 (pre-registration, DRAFT)

> ⛔ **DRAFT. It is not in force until both of these hold:**
> 1. P2 has a verdict under `eval/PREREG_P2.md`.
> 2. The PI has decided P6-T, P6-C, P6-N and P6-Y (`SPEC_P6_PAPER_FAITHFUL.md` §7).
>
> It is then finalised and **landed in git before the first P6 training step**. After landing, nothing below may
> change without a dated amendment written before the read.
>
> Design: `SPEC_P6_PAPER_FAITHFUL.md`. Confidence: `P6_CONFIDENCE.md`.
> Written 2026-10-11.

## The arm

**Data and schedule.**
- REFe ViT-L/16 trained from DINOv3 with fresh rank-32 Q/V LoRA and fresh heads.
- The final run's bank: 189,888 tuples over 103,037 scenes, full from step 0.
- 25 scene-epochs = 10,063 optimiser steps at 4 × 64.

**Optimiser.** AdamW, lr 2e-4 cosine to 0, no warm-up, wd 0.01, clip 1.0, bf16 + TF32 (declared departure), seed 0.

**Scorer.**
- Online NAVSIM PDM targets on each sample's own current proposals, at lag 0 (labeller L1).
- Context un-detached; candidates detached.
- Six components, selected by `navsim_v1`.

**Inputs and losses.** Command nav input (if P6-N = command) and unwrapped yaw L1 (if P6-Y = unwrapped). The final
text names the decided values, and each one is also named in the argv.

**Launch.** Only on a launch-gate PASS token (SPEC §5).

## Primary endpoint

**Full navtest PDMS:** 12,146 tokens over 136 logs, NAVSIM v1.1.

**Eval settings.**
- The harness is the one that produced `eval/RESULT_SFT4.md`, on the dev box and never on the pod.
- The checkpoint is `model_final.pt`.
- The plan transform is the same one used for the labels (A7 repair ON).
- Route fix: N/A in command mode, ON in goal-point mode.

**Tier.** Pseudo-simulation of an open-loop plan against logged agents. **Not closed loop.**

**Comparators**, both paired on the identical tokens:

| comparator | PDMS on the full navtest | source |
|---|---|---|
| **C1: SFT-4 arm B u400** | **88.25** | `eval/RESULT_SFT4.md`, MEASURED, with its own seam: route fix ON, A7 ON, `navsim_v1_lane` |
| **C2: P2** | its full-navtest number, if P2 reached that stage | – |

The comparison is **system against system**: each arm runs with its own registered selection rule.

**Estimator.**
- Paired log-cluster bootstrap over the 136 logs.
- **10,000 resamples, seed 20260927.**
- Reported for the difference and for P6's level.

## Rule (committed now)

**Versus C1 (and, separately, C2):**
- **ADOPT** iff the lower bound of Δ > 0 AND no longitudinal or lateral family component separates adversely.
- **REFUTED** iff the upper bound of Δ < 0.
- Otherwise **NOT PROVEN**.

**Against the goal of 94.55** (PUBLISHED, Table A13):
- **GOAL MET** iff P6's point estimate ≥ 94.55 AND ADOPT vs C1.
- **GOAL NOT EXCLUDED** iff the point estimate < 94.55 ≤ the upper bound of P6's level CI.
- **BELOW GOAL** iff the upper bound < 94.55.

⚠️ REVIEW_7 §1.2 bounds the harness difference to the paper at ≤ ~0.25 PDMS (ESTIMATED, our harness stricter). It is
**reported and never subtracted**.

## Four families (binding; per family, never pooled)

The instrument is the one that produced `sft4_B_u400_families.json`. It runs on the identical tokens, and each tested
component carries a paired CI.

| family | components |
|---|---|
| **LONGITUDINAL** | speed MAE and bias; along-track MAE and bias; EP and TTC sub-scores; headway/time-gap to the lead agent where a lead exists. Each comes with its n and, when there is no lead, the reason. |
| **LATERAL** | heading MAE; cross-track MAE; curvature MAE; yaw-rate MAE; DAC sub-score. **Last-pose heading error** is reported separately, as the D5 check. |
| **TACTICAL** | lateral, longitudinal and 5-way decision accuracy/κ; **command adherence** of the selected proposal; **selection quality** as random / actual / best-of-64 on the same tokens (a random control is mandatory). |
| **STRATEGIC** | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25). This is not a pass, and it is not omitted. |

## What the interval answers, and the replicates

**The bootstrap answers one question only:** *would another draw of logs say this?*

**Inference variance.**
- REFe inference is deterministic. This is **asserted**: a second full eval of the same checkpoint must be
  bit-identical in per-token PDMS.
- A mismatch is a defect, investigated before any read.

**Training variance (H-ESTIM-SEED-1).**
- A second 7-day P6 seed is not in budget.
- The seed floor is therefore taken from the **cheapest available replicate**, in this order:
  1. P2's training-seed replicate (`eval/PREREG_P2.md`, on ADOPT);
  2. otherwise an SFT-4 u400 seed replicate on the dev box.
- A lever claim needs **Δ's lower bound > that floor**.
- If the floor is unmeasured at read time, P6 is reported as **"separated, single seed, not yet a lever claim"**.
- A P6 seed-1 replicate is requested **only if P6 is ADOPT and the delta is within 2× the floor**.

**Reported, not gating:**
- the per-epoch snapshots on the 923 confirmation tokens;
- the in-run coverage, lag, stall and `op_drift` series;
- all six sub-scores.

## Next arm, by outcome (Rule Zero; committed now)

**GOAL MET:**
1. the seed replicate;
2. then the first beyond-paper arm, P7.1: a training-only map-supervised drivable-area/occupancy aux head (admissible,
   PI 2026-10-11);
3. then the scorer ensemble (P7.2).

**ADOPT, below or not excluding goal:**
- Deploy P6.
- The next lever is set by the sub-score gap to Table A13:
  - **EP gap largest** ⇒ the selection rule and progress calibration on P6's scorer (0 GPU, dev box);
  - **DAC/NC/TTC gap largest** ⇒ P7.1, then P7.2.

**NOT PROVEN:**
- Read the in-run series against the SPEC's expectations: coverage ≈ 100 %, lag 0, the scorer's held-out ranking.
- The cheapest next experiment is a **scorer-only PDM fine-tune on P6** (the SFT-4 recipe), which took SFT-4 +3.79.

**REFUTED:**
- Bisect P6 against SFT-4 by the deltas that differ: nav mode, yaw form, from-scratch vs warm start.
- Use 0-GPU dev-box reads first: the selection rule on P6's proposals, and goal vs command on the same tokens.
