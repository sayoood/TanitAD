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
>
> **State on 2026-10-14 (still a DRAFT, so edited in place):**
> - Precondition 1 **holds**. P2 R2 is **GO**: +0.57 PDMS [+0.02, +1.18] over u400 on the full navtest (`eval/RESULT_P2.md`, landed `78e9bc9`).
> - Precondition 2: the PI decided P6-T (25 epochs), P6-N (command), P6-Y (unwrapped) and P6-D on 2026-10-11.
>   **P6-C was decided 2026-10-14:** the PI said, verbatim, *"Find a solution with our 11 cpu pod"*. P6 runs on the existing A40 pod, which has an 11-vCPU cgroup.
>   - The feasibility plan and a CPU-only pod probe are in build (`eval/P6_ON_11VCPU.md`, `refe/p6_cpu_probe.sh`).
>   - The probe runs in a non-training window before P6. The chain's own G-LIVE-LAB gate (≥ 6.735 sets/s at the launch worker count) is unchanged.
> - **The P2 seed replicate is ordered** (PI 2026-10-14, *"Replicate p2"*; `eval/PREREG_P2_REPLICATE.md`). It supplies P6's training-seed floor F.
> - Pod order (PI): P2 ✅ → RL-S1 (training since 2026-10-14 04:33 Berlin) → S1 aux fine-tunes → **P2-s1** → P6.
> - Still to do before P6's first step: finalise this file (the 11-vCPU configuration, once the probe has measured it) and land it.

## The arm

**Data and schedule.**
- REFe ViT-L/16 trained from DINOv3 with fresh rank-32 Q/V LoRA and fresh heads.
- The final run's bank: 189,888 tuples over 103,037 scenes, full from step 0.
- 25 scene-epochs = 10,075 optimiser steps at 4 × 64 (25 × ⌈103,037 / 256⌉; corrected pre-data on 2026-10-11 from 10,063).

**Optimiser.** AdamW, lr 2e-4 cosine to 0, no warm-up, wd 0.01, clip 1.0, bf16 + TF32 (declared departure), seed 0.

**Scorer.**
- Online NAVSIM PDM targets on each sample's own current proposals, at lag 0 (labeller L1).
- Context un-detached; candidates detached.
- Six components, selected by `navsim_v1`.

**Inputs and losses.**
- Command nav input: P6-N, decided. It is `--nav-mode command`, hard-wired in the argv of `refe/p6_pod.sh`.
- Unwrapped yaw L1: P6-Y, decided. It is `--yaw-wrap off`, from `P6_YAW_WRAP`, whose default is `off`.
- The launch-gate token binds both, because it is minted over the exact argv.

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
| **C2: P2** | **88.82** | `eval/RESULT_P2.md`, MEASURED 2026-10-14, with its own seam: goal fix `pdm_route`, A7 ON, `navsim_v1` |

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
  1. P2's training-seed replicate (`eval/PREREG_P2.md`, on ADOPT).
     - ⚠️ 2026-10-14: P2 was ADOPT and GO. `PREREG_P2.md`'s ADOPT clause ("the seed replicate, then P6") still stands, because Amendment 1 replaced only the NO-GO clause.
     - **Ordered by the PI on 2026-10-14** as P2-s1 (`eval/PREREG_P2_REPLICATE.md`). It runs after S1 and before P6. Its floor is **F = |mean Δ_seed|**, a one-draw estimate.
     - Cost: about **21 A40-h** (MEASURED for P2: 42 half-hourly egress ticks), plus a 5.3 h dev-box navtest read.
  2. Only if P2-s1 yields no valid read: the three S1 no-head arms (`N-s0/N-s1/N-s2`: the SFT-4 recipe on u400, 400 updates, seeds 0–2; `PREREG_AUX_DRIVABLE_S1.md`).
     - They are read on the full navtest on the dev box, at about 5.3 h each (MEASURED for u400).
     - This read belongs to P6, not to S1: S1 computes no navtest PDMS (its §4.2), and N carries no head, so the read cannot touch S1's H2 question.
     - ⚠️ It is a **scorer-only** floor (frozen trunk). It may understate the seed variance of a joint fine-tune, and is reported as such.
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
