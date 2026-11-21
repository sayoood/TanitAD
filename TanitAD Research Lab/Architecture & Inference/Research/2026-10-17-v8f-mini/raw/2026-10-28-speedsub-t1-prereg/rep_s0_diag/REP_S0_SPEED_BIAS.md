# The DAG-ORC replicate floor is one slow-speed training draw: REP s0's T predicts every speed cell about 1 m/s slow

Written 2026-10-28 ~09:20 by the Master Mind. MEASURED with `rep_s0_speedcells.py` (this folder), output `rep_s0_speedcells.json`. It reads only
banked artifacts: the box-2 T0 evals of CTRL (DAG-OACC), REP and A, at both seeds. The input sha256s are in the JSON. This is a diagnosis,
beside `RESULT_V8F_DAG_ORC` (landed `ea5e3d3b`); it changes no outcome, bar or floor of that read.

**Tier.** T0 (box 2's standard eval, eval-141, ages cycle). It is a diagnostic, never driving performance.

**Estimator.** Per-window paired difference over the same 2,399 windows, with an episode-cluster bootstrap over 141 episodes (B = 2,000,
seed 0). The window list is deterministic: the eval grid plus age = f(window index), `stack/scripts/v8f_mini_eval.py:269` and `:292-294`.
The episode ids are asserted equal across all six evals.

## What the read found

`RESULT_V8F_DAG_ORC` section 4 (c): the replicate pair sets every crawl bar, and REP s0 sets RF on 17 of 19 rows. REP s0 − CTRL at T1-kin
reads ADE6 +0.4025 m, yaw-rate MAE +15.86 and curvature MAE +0.1324, all separated-worse. REP s1 − CTRL reads ≈ 0 on the same rows.

## What REP s0 is (T0, paired, mean [95 % CI], m/s)

| T speed cell | CTRL s0 mean | REP s0 − CTRL s0 | A s0 − CTRL s0 | REP s1 − CTRL s1 | A s1 − CTRL s1 |
|---|---|---|---|---|---|
| lon_v_target_ms | 11.5014 | **−1.1097 [−1.2814, −0.9345]** | +0.0085 [−0.0702, +0.0864] | +0.0597 [−0.0752, +0.2026] | −0.0084 [−0.0898, +0.0726] |
| v_a_ms | 11.6807 | **−0.7315 [−0.8635, −0.5901]** | −0.1131 [−0.1797, −0.0479] | +0.2498 [+0.1185, +0.3844] | +0.1272 [+0.0504, +0.2081] |
| v_end_ms | 11.6941 | **−1.0056 [−1.1436, −0.8686]** | +0.0035 [−0.0616, +0.0646] | −0.0635 [−0.2021, +0.0785] | +0.0180 [−0.0586, +0.0992] |
| v_lo_ms | 10.2247 | **−1.1700 [−1.3210, −1.0175]** | −0.0473 [−0.1181, +0.0186] | +0.1672 [+0.0264, +0.3103] | +0.0001 [−0.0762, +0.0744] |
| v_hi_ms | 12.8535 | **−0.9913 [−1.1383, −0.8487]** | −0.0429 [−0.1154, +0.0323] | +0.0225 [−0.1290, +0.1797] | −0.0313 [−0.1072, +0.0442] |

The executed plan follows the cells. These are the eval JSON's longitudinal leaves, means (T0):

| row | CTRL s0 | REP s0 | A s0 | CTRL s1 | REP s1 | A s1 |
|---|---|---|---|---|---|---|
| speed bias (m/s) | −0.1762 | **−0.5440** | −0.0480 | −0.2070 | −0.1830 | −0.1334 |
| along-track bias (m) | −0.2347 | **−1.0525** | +0.1120 | −0.2662 | −0.1729 | −0.0348 |
| along-track final bias (m) | −0.7531 | **−3.0806** | −0.0123 | −0.9342 | −0.5804 | −0.5032 |
| ADE6 (m) | 1.9635 | 2.2473 | 2.0742 | 2.0770 | 2.0159 | 2.1493 |

## Reading (arithmetic, deciding nothing)

1. **REP s0's draw is a slow T, not general noise.** All five speed cells are separated-low by 0.73–1.17 m/s, about 9 % of the mean. REP s1's
   cells move by at most 0.25 m/s. The plan inherits the bias: speed bias −0.544 against −0.176 m/s, final along-track bias −3.08 against
   −0.75 m. At T1-kin, where the model rolls on its own plan, a slow plan compounds into the crawl. The crawl rows (yaw-rate, curvature) are
   the rows REP s0 inflates most: +15.86 yaw-rate at T1-kin against +0.30 deg/s at T0 (4.9016 − 4.5966). That fits compounding, and is not a
   proof of it (INFERENCE).
2. **So the floor that blocked DAG-ORC's R-MOVE measures how much the speed cells vary across training runs.** The bootstrap cannot see that
   variance (`H-ESTIM-SEED-1`), and the floor caught it.
3. **The lever this points at is already measured to work.** `PREREG_V8F_SPEEDSUB` read PASS on 2026-10-28. It replaced exactly these five
   cells with the banked ego-only read-out `mlp_ego`, fitted on TRAIN clips from measured ego history, and ADE6 fell at both seeds
   (T1-class open loop, on M2b-mixo). A read-out that does not depend on T's training draw would, if the cells carry the draw, also shrink the
   REP − CTRL spread.
   **This is the HYPOTHESIS to pre-register:** under `mlp_ego` substitution, the ADE6 penalty of REP s0 against CTRL s0 shrinks, and the
   substituted base beats its own unsubstituted form at T1-kin.

## What this does not show

* It shows the speed cells differ. It does not show that the cells *cause* the T1-kin outlier: that needs the substitution read at T1-kin
  (the next PREREG).
* Two seeds give two draws. Whether slow-T draws are rare or common on this rig is not known from n = 2.
* A s0's T speed cells sit with CTRL's. A's T0 cost (R-T0) is not a speed-cell shift.
