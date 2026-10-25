# RESULT_V8F_SPEEDAVG: T's speed error is NOT seed variance: the two seeds make the same errors (2026-10-25, MM)

**Outcome (`PREREG_V8F_SPEEDAVG.md` + Amendment 1, applied verbatim): MIXED.** Committed next step, quoted: *"reported per seed; seeds before windows"*.

* **Evidence class:** MEASURED (ours). Dev-box CPU, zero training.
  * The driver is tip `3950bdbb`'s `v8f_mini_eval_speedavg.py` (the pair-check fix, landed before any number). The runner is `code/devbox_ops/run_speedavg.sh`.
  * X_0 = `M2b-mixo_s0` (sha c6a62876…); X_1 = `M2b-mixo_s1` (sha 68a0f335…). Both are unchanged after the evals.
  * Artefacts: `raw/2026-10-25-speedavg/run/`.
* **Tier:** T1-class open loop (own held outputs, no recorded future action). Not a driving claim.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, 141 episodes / 2,399 windows. F = 0.1207 m.
* **Condition met:** KINDX's seed-1 read is CARRIER (speed): gain_speed +0.668 / +0.716 m (`RESULT_V8F_KINDX.md`).

| read | seed 0 | seed 1 | bar |
|---|---|---|---|
| **PRIMARY** ADE6 SPEEDAVG − X | −0.0414 [−0.0730, −0.0110] * | +0.0189 [−0.0116, +0.0483] | ≤ −F = −0.1207, separated, at BOTH seeds: **not met** |
| CAPTURE (fraction of the speed oracle's headroom) | 0.058 | −0.028 | beside |
| SECONDARY not separated-worse | yes | **no**: accel MAE +0.0169 [+0.0022, +0.0283] * | both |

**The table, row by row:**
* **VARIANCE PAYS** needs PRIMARY at both seeds: no.
* **PARTIAL** needs separated < 0 at both: no, seed 1 is not separated.
* **BIAS, NOT VARIANCE** needs "not separated at either seed": no, because seed 0 is separated, by −0.041 m.
* ⇒ **MIXED.**

**Controls (all PASS; `run/READ_controls.json`):**
* Self-average = own: max |dev| 0.0 at both seeds.
* GT speed cells reproduce KINDX's speed-only oracle per window: 4.3e-6 / 1.5e-5 m.
* Own forward reproduces the banked evals: ≤ 2.4e-5 m.
* Non-speed cells are untouched, and both average arms averaged the same pair.

## What it means: the mechanism, read from the descriptives (no bar was registered on them)

Per speed cell (m/s, on the windows with GT; `run/READ.json` → `beside.descriptives`):

| cell | RMSE X_0 | RMSE X_1 | RMSE averaged | corr of the two seeds' ERRORS | RMSE ratio measured / if errors were pure variance |
|---|---|---|---|---|---|
| `lon_v_target_ms` | 2.894 | 2.747 | 2.677 | **0.80** | 0.949 / 0.950 |
| `v_a_ms` | 1.919 | 1.302 | 1.441 | **0.71** | 0.879 / 0.925 |
| `v_end_ms` | 2.795 | 2.692 | 2.648 | **0.85** | 0.965 / 0.962 |
| `v_lo_ms` | 2.257 | 2.143 | 2.074 | **0.78** | 0.942 / 0.943 |

* **The two independently trained T's make the same speed errors: their errors correlate at 0.71–0.85.** Averaging can remove only the uncorrelated part, so RMSE falls by only 3–12 %, and the measured ratios match the pure-variance prediction for that correlation.
* **T's speed error is therefore systematic.** It is shared by every training run of this model on these inputs. Either it is a bias of the speed head (for example, shrinkage toward the mean speed) or the right speed is not recoverable from T's inputs. **SPEEDAVG cannot tell these two apart, and no number here decides between them.**
* The headroom is unchanged and real (T0): GT speed cells give ADE6 1.481 / 1.453 m against X's own 2.198 / 2.120 m.

**Rule Zero (the next lever, which this result routes but does not decide).** The cheapest experiment that separates the two readings is zero-training: recalibrate T's five speed cells with a map fitted on TRAINING windows only, and score it on eval.
* If the error is head bias (shrinkage), a calibration recovers part of the 0.67–0.72 m headroom.
* If it is information-limited, a calibration recovers nothing, and the lever moves to perception (what T sees).

This is pre-registered as `PREREG_V8F_SPEEDCAL` before any calibration number exists.

## Four families, SPEEDAVG − X (paired; `run/READ.json` → `beside.families`)

| family | metric | seed 0 | seed 1 |
|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | −0.0196 [−0.0361, −0.0029] * | +0.0128 [−0.0043, +0.0290] |
| LONGITUDINAL | accel MAE (m/s²) | −0.0177 [−0.0321, −0.0048] * | **+0.0169 [+0.0022, +0.0283] *** |
| LONGITUDINAL | along MAE (m) | −0.0394 [−0.0715, −0.0076] * | +0.0296 [−0.0036, +0.0615] |
| LONGITUDINAL | distance keeping, abs error vs the human: headway (m) / time gap (s) / min TTC (s) | −0.011 / −0.0004 / −0.110 * | −0.006 / −0.002 / +0.025 (none separated) |
| LONGITUDINAL | LEAD-VIOL | 0.0 | not separated-worse |
| LATERAL | heading MAE (deg) | −0.0071 [−0.0430, +0.0303] | −0.0056 [−0.0339, +0.0168] |
| LATERAL | yaw-rate MAE (deg/s) | +0.0289 [−0.0747, +0.1411] | +0.0059 [−0.0783, +0.0769] |
| LATERAL | curvature MAE (1/m) | +0.0009 [−0.0015, +0.0037] | −0.0002 [−0.0021, +0.0014] |
| LATERAL | cross-track MAE (m) | −0.0054 [−0.0108, −0.0003] * | −0.0043 [−0.0093, +0.0004] |
| TACTICAL | T's decisions (lat / lon macro-F1) | identical to X_s by construction (bit-equal): 0.827 / 0.502 | identical (bit-equal): 0.820 / 0.496 |
| TACTICAL | plan-derived manoeuvre (paired, `v8f_paired_families.py`) | lat / lon / 5-way accuracy +0.005 / −0.010 / −0.004 (none separated); goal-point error −0.103 [−0.197, −0.010] *, goal lateral bias −0.028 [−0.054, −0.003] *, goal bearing MAE **+0.136 [+0.002, +0.344] * (worse)** | lat / lon / 5-way accuracy +0.001 / −0.004 / −0.001 (none separated); goal-point error +0.045 [−0.045, +0.133]; goal longitudinal bias **−0.428 [−0.526, −0.326] *** (the plan's goal falls 0.43 m shorter), range ratio −0.013 [−0.022, −0.005] * |
| STRATEGIC | S's checkpoint error (pos @30 / @50 m) | identical to X_s by construction (bit-equal): 7.01 / 8.04 m | identical (bit-equal): 7.01 / 8.01 m |

* Other horizons: ADE2 −0.0035 (s0, not separated) / −0.0070 * (s1); FDE6 −0.103 * (s0) / +0.045 (s1, not separated). The seeds disagree in sign at 6 s, which is the MIXED reading again.
* **STRATEGIC** is not paused for v8F (`PI-V8F-2026-10-17`). S is X_s's own in every arm, so the family is identical to X_s by construction.
* **TACTICAL, plan-derived.** No decision cell moves at either seed. The goal-setting cells that do separate disagree between the seeds (s0: the goal point is 0.10 m closer to the human's and its bearing 0.14° worse; s1: the goal lands 0.43 m shorter), which is the MIXED reading a third time. The paired STRATEGIC cells are exactly 0.0 [0.0, 0.0] on all six (S is X_s's own).
* **The paired tool's JSON-reproduction control could not run on 22 of its 79 values, at both seeds.** These are the plan-derived and goal-setting cells: SPEEDAVG's driver does not write `raw.families` into its eval JSON, so the reference is absent (`pairfam/pairfam_speedavg_s*.json` → `controls`). The other 57 reproduce at max |diff| 0.0, and the identity control passes on all 45 cells. **Substitute control, run before these rows were written into this file:** the same tool on SPEEDAVG's `own` dump against the BANKED X_s eval (`eval_M2b-mixo_s0.json`; `eval_M2b-mixo_s1-kindx-own.json`), whose JSON does carry the block: **PASS at both seeds**: all 119 reference values reproduce at max |diff| 0.0 (to the JSON's 1e-6 rounding), and the paired own − banked delta is at most 8e-7 on every cell (`pairfam/subctl_own_vs_banked_s*.json`). A few cells read "separated" at the 1e-7 scale; that is float noise from the ≤ 2.4e-5 m forward difference, resolved by the bootstrap, not an effect. The tactical rows above are therefore read through a pipeline that reproduces the banked family blocks. This is an instrument gap in the SPEEDAVG driver (it omits a block the base eval writes), not a defect in the paired tool.

## Provenance

* The first run (21:40) refused at the pair check before any eval: `o_interface` was `<MISSING>` in X_0 and `full` in X_1. Fix `3950bdbb`; logs `run/logs/*.refused_pair_cfg_2140.*`.
* The rerun was 21:42–21:50. Its logs are `run/logs/{run.out,evals0.log,evals1.log,read.log}`, and the per-mode evals with their speed-cell dumps are in `run/eval/`.
