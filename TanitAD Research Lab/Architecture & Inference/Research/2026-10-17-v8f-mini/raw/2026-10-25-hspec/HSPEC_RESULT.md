# HSPEC_RESULT: PREREG_V8F_HSPEC executed exactly as written (2026-10-25)

Dev box, CPU only, ZERO training, no GPU, no box-2 access. Every number is **MEASURED (ours)** from the artifact paths named; tier **T1-class open loop (own held outputs; not closed loop)**; estimator **paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0**. The bootstrap answers 'would another draw of EPISODES say this?' only; training-seed variance enters only through the two seeds side by side and F = |ADE6(s0) - ADE6(s1)| = 0.12074 m (`H-ESTIM-SEED-1`). Inference is deterministic. `OUT` = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hspec`.

## 0. Outcome (the PREREG outcome table, applied verbatim)

**SMOOTHER, NOT DOMINANT** (final stage: 2; stage 2 ran automatically because PRIMARY-DOM failed at stage 1).

| stage | PRIMARY-DOM s0 | PRIMARY-DOM s1 | PRIMARY-KEEP s0 | PRIMARY-KEEP s1 | accel AND curvature HS - H separated-negative, s0 / s1 |
|---|---|---|---|---|---|
| 1 | FAIL | FAIL | PASS (hi -0.0003) | PASS (hi +0.0003) | True / False |
| 2 | FAIL | FAIL | PASS (hi +0.0247) | PASS (hi +0.0292) | True / True |

Condition reading: PRIMARY-DOM needs HS - P10 separated-better on ADE2, ADE6, accel MAE AND curvature MAE at BOTH seeds and not separated-worse anywhere else. **It fails at stage 2 on exactly two rows, accel MAE and curvature MAE**, which are not separated-better than P10 at either seed (accel: s0 separated-WORSE, s1 not separated; curvature: not separated at both). Every other row of the dominance row holds at both seeds (section 3). PRIMARY-KEEP passes at both seeds and both stages (the ADE6 gain over H is kept; HS - H upper bound <= +0.06037 m). Accel and curvature HS - H are separated-negative at both seeds at stage 2 (section 5), so the table lands on SMOOTHER, NOT DOMINANT.

**Committed next step (PREREG, verbatim):** *"the roughness is partly in the coefficients' amplitudes, and partly elsewhere. Next: a training arm with a second-difference penalty on O's plan, on box 2 (a loss term: G-DVB entry and launch gate)"*.

**Rule Zero (a refutation is a waypoint).** The dominance claim is not delivered by this arm; the section 9 lever map says what is left and what blocks the committed next step. Short form: (i) the committed training arm is blocked on a PREREG for the loss term with its G-DVB entry, the launch gate, and a box-2 slot (box 2 is outside this read; GPU/spend is the PI's call); (ii) a zero-training re-registration of lambda alone cannot dominate at seed 0 on the grid examined (accel is never separated-better than P10, and ADE2 turns separated-worse than P10 from x4); (iii) the per-step weights w carry part of the roughness (smoothing w alone moves H's accel gap to P10 from +0.104 to +0.080 at s0), but not enough.

## 1. What was run

* **Route.** HBLEND's banked machinery, imported read-only (`v8f_hblend.py`: loaders, `apply_w`, `arm_pw`, `rows_for`, `deep_diff`, `single`; the eval's own `evaluate()`; `taniteval.v8f_beta_eval.paired_row`; the DK reader `v8f_dk_read.read_dk`). Fit set: HBLEND's 9,582 TRAIN windows (141 clips, k = 32; `hblend/fit_windows_M2b_s{0,1}.npz`); weights w: `hblend/weights_M2b_s{0,1}.json` (final = clip then PAV), unchanged. Eval-141 (2,399 windows) never enters a fit. B's plan tau = the KIND run's banked `pred`; P = the checkpoint's own frozen P10 (CPU forward, == HBLEND's saved P bit for bit); H = HBLEND's saved H (recomputed: max abs diff 0.0).
* **Coefficients.** c = least squares of (traj30 - p30) on `basis[1::2]` (the ramped DCT of `v8f_mini.py`, rows at tau = 0.2 j, [30, 12]; cond 22.5, rank 12), per axis (along = axis 0, cross = axis 1). The formula basis equals both checkpoints' `basis` buffer to 2.1e-06; `up(P)[1::2] = P` and `subsample_02` are pinned by a test against the real `v8f_mini` functions.
* **Arm.** tau_HS = P + w (.) B30 (v (.) c) on the 0.2 s knots, v in [0, 1]^(12 x 2). **Stage 1:** unweighted bounded least squares of tau_HS vs the GT, `scipy.optimize.lsq_linear(bounds=(0,1), method='bvls', tol=1e-12)`, one 12-parameter problem per axis, KKT-checked (relative projected gradient <= 1e-6; observed <= 2e-14). **Stage 2:** the same + lambda ||D2 tau - D2 y||^2.
* **Declared interpretations (fixed in the script's docstring BEFORE any HSPEC number; none changed after).** I1 channels: v[k, a], a = 0 along / 1 cross, the axis order of `coef [B, 2, nb]`. I2 scored grid: the 0.2 s knots only (the 0.1 s midpoints are neither fitted nor scored, as in HBLEND). I3 stage-1 objective unweighted over windows x 30 steps x axes. I4 stage 2: e(t) = tau_HS - y with the ORIGIN prepended as e(0) = 0, so D2 has 29 rows per axis; ONE lambda per seed, pooled over windows, steps and axes, lambda = S1/S2 at v = 1 on TRAIN (the two terms equal at v = 1). Computed once: **lambda = 2499.59 (s0), 2308.21 (s1)**; check S1 = lambda S2 at v = 1: 4.0165e+06 vs 4.0165e+06 (s0). I5 'separated-better / worse' = `paired_row`'s flags; distance keeping WORSE = headway or time-gap separated NEGATIVE, LEAD-VIOL at-risk separated POSITIVE (a two-sided sensitivity is stated after the section 4 tables and does not change the outcome). I6 PRIMARY-KEEP bar = F/2 with the exact F = 0.12074 m, i.e. +0.06037 m (the printed +0.060 variant is reported beside; same verdict). I7 control thresholds (below). I8 table order DOMINATES, GAIN LOST, SMOOTHER, NO EFFECT; 'SMOOTHER' = accel and curvature HS - H separated-negative at both seeds of the final stage. I9 stage 2 runs for both seeds iff PRIMARY-DOM fails at either seed of stage 1.

## 2. Controls (run BEFORE any read; the script refuses to read if one fails; 15 tests, `pytest_final.log`)

| control | bar | s0 stage 1 | s1 stage 1 | s0 stage 2 | s1 stage 2 |
|---|---|---|---|---|---|
| 1. coefficient recovery residual, train (m) | <= 1e-4 | 1.52e-05 | 1.49e-05 | 1.52e-05 | 1.49e-05 |
| 1. coefficient recovery residual, eval (m) | <= 1e-4 | 2.02e-05 | 2.09e-05 | 2.02e-05 | 2.09e-05 |
| 1. v = 1: max abs plan diff vs H (m) | <= 1e-4 | 1.61e-05 | 1.80e-05 | 1.61e-05 | 1.80e-05 |
| 1. v = 1: evaluate() block mismatches vs H (rel tol 1e-6) | 0 (see note) | 1 (0 excl. threshold diagnostic) | 1 (0 excl. threshold diagnostic) | 1 (0 excl. threshold diagnostic) | 1 (0 excl. threshold diagnostic) |
| 1. v = 1: per-window ADE / family metric max abs diff vs H (each metric in its own unit) | - | 2.10e-03 | 3.89e-03 | 2.10e-03 | 3.89e-03 |
| 2. v = 0: max abs plan diff vs the P10 file (m) | <= 1e-4 | 1.53e-05 | 1.53e-05 | 1.53e-05 | 1.53e-05 |
| 2. v = 0: per-window ADE6 vs floors file (m) | <= 1e-5 | 1.96e-06 | 1.96e-06 | 1.96e-06 | 1.96e-06 |
| 2. v = 0: ADE6 mean (m) | 2.4870 +- 1e-3 | 2.48699 | 2.48699 | 2.48699 | 2.48699 |
| 2. v = 0: evaluate() block mismatches vs P10 (rel tol 1e-4) | 0 | 0 | 0 | 0 | 0 |
| 3. regression (GT := P10 in the fit): max v | <= 1e-12 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 3. regression: ADE6 minus P10's (m) (reads as P10, never a gain) | <= 1e-4 abs | -6.0e-10 | -6.0e-10 | -6.0e-10 | -6.0e-10 |
| 4. noise (c permuted across windows): mean v / max v | <= 0.10 / <= 0.30 | 0.040 / 0.136 | 0.030 / 0.121 | 0.042 / 0.156 | 0.042 / 0.187 |
| optimality: KKT residual of the fit | <= 1e-6 | True | True | True | True |
| all controls pass |  | True | True | True | True |

* **Note on control 1 (found at control time, before any read, disclosed).** HS(v = 1) vs H through the eval's own `evaluate()` matches on every entry except **one**: `traj.acts.zero_accel_frac`, the share of (window, step) cells with |a| < 1e-3 m/s^2 (a HARD threshold in `floors_kin.acts_check`). M2b_s0: 7 of 69,571 cells straddle it under a 1.6e-05 m plan perturbation (max |dA| on a flip 1.4e-04, Lipschitz bound 2.4e-03); M2b_s1: 7 of 69,571 cells straddle it under a 1.8e-05 m plan perturbation (max |dA| on a flip 5.4e-04, Lipschitz bound 2.7e-03). The perturbation is the float32 rounding of the banked arrays, inside the PREREG's own 1e-4 m recovery bar, so a bit-exact 'Delta = 0.0' is not attainable from banked float32 data. The control therefore passes iff every other entry matches AND every threshold flip is explained by the perturbation (script docstring I7; test `test_control1_v_ones_equals_H`; mutation `test_MUTATION_a_real_plan_difference_is_not_excused...` shows a real 3e-3 m plan difference is still caught). `zero_accel_frac` is not a HSPEC reading quantity. Accel MAE itself matches.
* **Mutations that go RED** (`test_v8f_hspec.py`): the wrong subsample phase (basis rows 0::2) breaks the coefficient recovery (> 1e-2 m vs the 1e-4 bar); swapped w axes cannot recover a planted v; a planted v is recovered to 1e-5 at stage 1 and stage 2; the regression fit collapses to v = 0 while the real fit does not; a 3e-3 m plan perturbation is caught by the evaluate() block beyond the threshold diagnostic.
* **Noise control, descriptive** (eval, stage-1 c-permuted v, s0 / s1): ADE6 minus P10 = -0.0069 [-0.0082, -0.0057] * / -0.0033 [-0.0040, -0.0026] *, against HS's -0.5225 [-0.6360, -0.4125] * / -0.5050 [-0.6240, -0.3886] *. The permuted arm is P10-like (v ~ 0.03-0.04) and keeps 1.3 % / 0.7 % of HS's ADE6 gain (a small positive v on a residual that is better than P10 at 6 s still moves ADE6 slightly the right way); the control passes its registered bar on v, and it shows the HS gain is not what a noise fit produces.

## 3. PRIMARY-DOM and PRIMARY-KEEP, stage 1 (position objective)

### Stage 1: the dominance row, HS - P10 (paired; lower is better except the two tactical agreements; `*` = CI excludes 0)

| metric | requirement | s0: HS - P10 [95 % CI] | s0 | s1: HS - P10 [95 % CI] | s1 |
|---|---|---|---|---|---|
| ADE2 | separated-better | -0.0119 [-0.0163, -0.0071] | better / OK | -0.0090 [-0.0137, -0.0037] | better / OK |
| ADE6 | separated-better | -0.5225 [-0.6360, -0.4125] | better / OK | -0.5050 [-0.6240, -0.3886] | better / OK |
| accel MAE | separated-better | +0.0942 [+0.0684, +0.1269] | worse / **FAIL** | +0.0898 [+0.0645, +0.1221] | worse / **FAIL** |
| curvature MAE | separated-better | +0.00570 [+0.00250, +0.00930] | worse / **FAIL** | +0.00890 [+0.00340, +0.01560] | worse / **FAIL** |
| speed MAE | not separated-worse | -0.1192 [-0.1650, -0.0687] | better / OK | -0.1463 [-0.1905, -0.0995] | better / OK |
| along MAE | not separated-worse | -0.2337 [-0.3158, -0.1499] | better / OK | -0.2671 [-0.3469, -0.1816] | better / OK |
| cross-track MAE | not separated-worse | -0.4298 [-0.5250, -0.3365] | better / OK | -0.3485 [-0.4504, -0.2540] | better / OK |
| heading MAE | not separated-worse | -1.6763 [-2.1084, -1.2706] | better / OK | -1.3466 [-1.8707, -0.8206] | better / OK |
| yaw-rate MAE | not separated-worse | +0.0179 [-0.1755, +0.2137] | not sep. / OK | +0.0425 [-0.2397, +0.3255] | not sep. / OK |
| plan-derived lateral decision agreement | not separated-worse | +0.0942 [+0.0709, +0.1180] | better / OK | +0.0688 [+0.0391, +0.0988] | better / OK |
| plan-derived longitudinal decision agreement | not separated-worse | +0.0800 [+0.0479, +0.1105] | better / OK | +0.0900 [+0.0480, +0.1318] | better / OK |
| CONSISTENCY(a) 'any' | not separated-worse | -0.0409 [-0.0704, -0.0100] | better / OK | -0.0200 [-0.0479, +0.0087] | not sep. / OK |
| DK headway min (m) | not separated-worse | +0.3207 [-0.0347, +0.6860] | not sep. / OK | +0.3698 [-0.0516, +0.8113] | not sep. / OK |
| DK time-gap min (s) | not separated-worse | +0.1124 [+0.0116, +0.2313] | better / OK | +0.1067 [+0.0009, +0.2270] | better / OK |
| DK LEAD-VIOL at-risk share | not separated-worse | -0.0303 [-0.0559, -0.0081] | better / OK | -0.0202 [-0.0492, +0.0036] | not sep. / OK |

* PRIMARY-DOM M2b_s0: **FAIL** (separated-better rows False, no-separated-worse rows True). PRIMARY-KEEP M2b_s0: ADE6 HS - H = -0.0040 [-0.0081, -0.0003], bar hi <= +0.06037 -> **PASS** (printed +0.060: PASS).
* PRIMARY-DOM M2b_s1: **FAIL** (separated-better rows False, no-separated-worse rows True). PRIMARY-KEEP M2b_s1: ADE6 HS - H = -0.0028 [-0.0081, +0.0003], bar hi <= +0.06037 -> **PASS** (printed +0.060: PASS).

## 4. PRIMARY-DOM and PRIMARY-KEEP, stage 2 (position + second-difference objective, lambda by the train-only rule)

### Stage 2: the dominance row, HS - P10 (paired; lower is better except the two tactical agreements; `*` = CI excludes 0)

| metric | requirement | s0: HS - P10 [95 % CI] | s0 | s1: HS - P10 [95 % CI] | s1 |
|---|---|---|---|---|---|
| ADE2 | separated-better | -0.0051 [-0.0081, -0.0019] | better / OK | -0.0053 [-0.0090, -0.0012] | better / OK |
| ADE6 | separated-better | -0.5114 [-0.6174, -0.4108] | better / OK | -0.4904 [-0.5977, -0.3846] | better / OK |
| accel MAE | separated-better | +0.0258 [+0.0105, +0.0444] | worse / **FAIL** | +0.0057 [-0.0110, +0.0283] | not sep. / **FAIL** |
| curvature MAE | separated-better | +0.00120 [-0.00090, +0.00360] | not sep. / **FAIL** | +0.00310 [-0.00010, +0.00660] | not sep. / **FAIL** |
| speed MAE | not separated-worse | -0.1183 [-0.1605, -0.0732] | better / OK | -0.1414 [-0.1827, -0.0961] | better / OK |
| along MAE | not separated-worse | -0.2258 [-0.2944, -0.1536] | better / OK | -0.2496 [-0.3145, -0.1790] | better / OK |
| cross-track MAE | not separated-worse | -0.4230 [-0.5161, -0.3320] | better / OK | -0.3484 [-0.4471, -0.2558] | better / OK |
| heading MAE | not separated-worse | -1.6633 [-2.0719, -1.2786] | better / OK | -1.5176 [-1.9730, -1.0943] | better / OK |
| yaw-rate MAE | not separated-worse | -0.3249 [-0.4906, -0.1691] | better / OK | -0.3278 [-0.5404, -0.1342] | better / OK |
| plan-derived lateral decision agreement | not separated-worse | +0.0888 [+0.0675, +0.1114] | better / OK | +0.0867 [+0.0621, +0.1122] | better / OK |
| plan-derived longitudinal decision agreement | not separated-worse | +0.0575 [+0.0254, +0.0880] | better / OK | +0.0759 [+0.0358, +0.1150] | better / OK |
| CONSISTENCY(a) 'any' | not separated-worse | -0.0504 [-0.0751, -0.0267] | better / OK | -0.0325 [-0.0555, -0.0088] | better / OK |
| DK headway min (m) | not separated-worse | +0.2650 [-0.0721, +0.6097] | not sep. / OK | +0.3789 [+0.0072, +0.7929] | better / OK |
| DK time-gap min (s) | not separated-worse | +0.1002 [+0.0076, +0.2139] | better / OK | +0.1176 [+0.0072, +0.2479] | better / OK |
| DK LEAD-VIOL at-risk share | not separated-worse | -0.0319 [-0.0588, -0.0085] | better / OK | -0.0218 [-0.0496, +0.0017] | not sep. / OK |

* PRIMARY-DOM M2b_s0: **FAIL** (separated-better rows False, no-separated-worse rows True). PRIMARY-KEEP M2b_s0: ADE6 HS - H = +0.0070 [-0.0118, +0.0247], bar hi <= +0.06037 -> **PASS** (printed +0.060: PASS).
* PRIMARY-DOM M2b_s1: **FAIL** (separated-better rows False, no-separated-worse rows True). PRIMARY-KEEP M2b_s1: ADE6 HS - H = +0.0117 [-0.0064, +0.0292], bar hi <= +0.06037 -> **PASS** (printed +0.060: PASS).

**Distance-keeping direction sensitivity (I5).** Read two-sidedly (any separated difference from P10 counts), these rows would be flagged: stage 1 M2b_s0 DK time-gap min (s) (+0.112, separated BETTER); stage 1 M2b_s1 DK time-gap min (s) (+0.107, separated BETTER); stage 2 M2b_s0 DK time-gap min (s) (+0.100, separated BETTER); stage 2 M2b_s1 DK headway min (m) (+0.379, separated BETTER); stage 2 M2b_s1 DK time-gap min (s) (+0.118, separated BETTER). Each is a separated difference in the MORE-margin direction (longer time-gap / headway than P10); no distance-keeping row is separated in the less-margin direction, and LEAD-VIOL is never separated-worse. The outcome does not depend on this reading: PRIMARY-DOM already fails on accel and curvature.

## 5. HS - H and HS - B on everything (paired; final = stage 2; stage 1 beside)

**Stage 2, M2b_s0** (lower is better except the tactical rows; `*` = CI excludes 0)

| family | metric | B | H | HS | P10 | HS - H | HS - B |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 1.0674 | 0.8709 | 0.8620 | 0.9802 | -0.0089 [-0.0205, +0.0015] | -0.2055 [-0.2652, -0.1520] * |
| LONGITUDINAL | accel MAE (m/s^2) | 1.2058 | 0.7414 | 0.6635 | 0.6376 | -0.0779 [-0.0908, -0.0663] * | -0.5424 [-0.6431, -0.4467] * |
| LONGITUDINAL | along MAE (m) | 1.8097 | 1.5925 | 1.5959 | 1.8217 | +0.0035 [-0.0160, +0.0217] | -0.2138 [-0.3049, -0.1303] * |
| LATERAL | cross-track MAE (m) | 0.7662 | 0.7557 | 0.7628 | 1.1858 | +0.0071 [+0.0007, +0.0133] * | -0.0034 [-0.0225, +0.0138] |
| LATERAL | heading MAE (deg) | 4.7804 | 4.0259 | 4.0669 | 5.7302 | +0.0410 [-0.0004, +0.0878] | -0.6804 [-1.1730, -0.3055] * |
| LATERAL | yaw-rate MAE (deg/s) | 8.1167 | 6.4244 | 6.0050 | 6.3299 | -0.4193 [-0.5678, -0.2933] * | -2.1855 [-3.0858, -1.5078] * |
| LATERAL | curvature MAE (1/m) | 0.08150 | 0.05360 | 0.04740 | 0.04620 | -0.00620 [-0.00910, -0.00380] * | -0.03760 [-0.06180, -0.02110] * |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8458 | 0.8629 | 0.8616 | 0.7728 | -0.0013 [-0.0108, +0.0088] | +0.0158 [-0.0004, +0.0334] |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5790 | 0.5969 | 0.5911 | 0.5336 | -0.0058 [-0.0217, +0.0113] | +0.0121 [-0.0154, +0.0404] |
| CONSISTENCY | CONSISTENCY(a) 'any' (share of windows violating; lower better) | 0.1596 | 0.2601 | 0.2605 | 0.3110 | +0.0004 [-0.0096, +0.0104] | +0.1009 [+0.0667, +0.1350] * |
| CONSISTENCY | CONSISTENCY(a) SPEED | 0.0955 | 0.1955 | 0.1997 | 0.2418 | +0.0042 [-0.0058, +0.0138] | +0.1042 [+0.0713, +0.1359] * |
| CONSISTENCY | CONSISTENCY(a) STOP | 0.0188 | 0.0292 | 0.0317 | 0.0338 | +0.0025 [+0.0000, +0.0054] | +0.0129 [+0.0067, +0.0200] * |
| CONSISTENCY | CONSISTENCY(a) TURN | 0.0042 | 0.0071 | 0.0075 | 0.0342 | +0.0004 [-0.0008, +0.0025] | +0.0033 [+0.0004, +0.0075] * |
| CONSISTENCY | CONSISTENCY(a) LEAD | 0.0725 | 0.1042 | 0.1075 | 0.1384 | +0.0033 [-0.0021, +0.0088] | +0.0350 [+0.0188, +0.0530] * |
| CONSISTENCY | self-consistency (b) 'any' | 0.2297 | 0.3697 | 0.3760 | n/a | +0.0063 [+0.0000, +0.0129] | +0.1463 [+0.1159, +0.1780] * |

**Stage 2, M2b_s1** (lower is better except the tactical rows; `*` = CI excludes 0)

| family | metric | B | H | HS | P10 | HS - H | HS - B |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.9112 | 0.8389 | 0.8388 | 0.9802 | -0.0001 [-0.0089, +0.0085] | -0.0725 [-0.1058, -0.0463] * |
| LONGITUDINAL | accel MAE (m/s^2) | 0.9313 | 0.7036 | 0.6433 | 0.6376 | -0.0603 [-0.0751, -0.0484] * | -0.2880 [-0.3644, -0.2258] * |
| LONGITUDINAL | along MAE (m) | 1.6081 | 1.5573 | 1.5722 | 1.8217 | +0.0149 [-0.0041, +0.0333] | -0.0359 [-0.0797, +0.0044] |
| LATERAL | cross-track MAE (m) | 0.8644 | 0.8375 | 0.8373 | 1.1858 | -0.0002 [-0.0049, +0.0045] | -0.0271 [-0.0514, -0.0042] * |
| LATERAL | heading MAE (deg) | 4.4383 | 4.3276 | 4.2126 | 5.7302 | -0.1149 [-0.3301, +0.0265] | -0.2257 [-0.3781, -0.0929] * |
| LATERAL | yaw-rate MAE (deg/s) | 6.9864 | 6.4447 | 6.0021 | 6.3299 | -0.4399 [-0.6664, -0.2495] * | -1.0519 [-1.5023, -0.6813] * |
| LATERAL | curvature MAE (1/m) | 0.06260 | 0.05720 | 0.04930 | 0.04620 | -0.00790 [-0.01330, -0.00340] * | -0.01630 [-0.02790, -0.00680] * |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8287 | 0.8470 | 0.8595 | 0.7728 | +0.0125 [+0.0033, +0.0237] * | +0.0308 [+0.0162, +0.0492] * |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5919 | 0.6132 | 0.6094 | 0.5336 | -0.0038 [-0.0184, +0.0113] | +0.0175 [+0.0050, +0.0308] * |
| CONSISTENCY | CONSISTENCY(a) 'any' (share of windows violating; lower better) | 0.2097 | 0.2851 | 0.2784 | 0.3110 | -0.0067 [-0.0158, +0.0025] | +0.0688 [+0.0458, +0.0918] * |
| CONSISTENCY | CONSISTENCY(a) SPEED | 0.1255 | 0.2238 | 0.2197 | 0.2418 | -0.0042 [-0.0117, +0.0025] | +0.0942 [+0.0680, +0.1209] * |
| CONSISTENCY | CONSISTENCY(a) STOP | 0.0271 | 0.0296 | 0.0317 | 0.0338 | +0.0021 [+0.0000, +0.0050] | +0.0046 [+0.0013, +0.0087] * |
| CONSISTENCY | CONSISTENCY(a) TURN | 0.0067 | 0.0071 | 0.0063 | 0.0342 | -0.0008 [-0.0021, +0.0000] | -0.0004 [-0.0025, +0.0013] |
| CONSISTENCY | CONSISTENCY(a) LEAD | 0.1084 | 0.1263 | 0.1267 | 0.1384 | +0.0004 [-0.0059, +0.0067] | +0.0183 [+0.0067, +0.0312] * |
| CONSISTENCY | self-consistency (b) 'any' | 0.1430 | 0.2172 | 0.2176 | n/a | +0.0004 [-0.0058, +0.0063] | +0.0746 [+0.0551, +0.0958] * |

**Stage 1, M2b_s0** (lower is better except the tactical rows; `*` = CI excludes 0)

| family | metric | B | H | HS | P10 | HS - H | HS - B |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 1.0674 | 0.8709 | 0.8610 | 0.9802 | -0.0098 [-0.0166, -0.0041] * | -0.2064 [-0.2620, -0.1573] * |
| LONGITUDINAL | accel MAE (m/s^2) | 1.2058 | 0.7414 | 0.7319 | 0.6376 | -0.0095 [-0.0172, -0.0017] * | -0.4740 [-0.5621, -0.3896] * |
| LONGITUDINAL | along MAE (m) | 1.8097 | 1.5925 | 1.5881 | 1.8217 | -0.0044 [-0.0088, -0.0003] * | -0.2217 [-0.2983, -0.1507] * |
| LATERAL | cross-track MAE (m) | 0.7662 | 0.7557 | 0.7559 | 1.1858 | +0.0002 [-0.0002, +0.0007] | -0.0102 [-0.0236, +0.0016] |
| LATERAL | heading MAE (deg) | 4.7804 | 4.0259 | 4.0538 | 5.7302 | +0.0279 [+0.0049, +0.0532] * | -0.7131 [-1.2037, -0.3458] * |
| LATERAL | yaw-rate MAE (deg/s) | 8.1167 | 6.4244 | 6.3479 | 6.3299 | -0.0765 [-0.1796, +0.0164] | -1.9065 [-2.7450, -1.2708] * |
| LATERAL | curvature MAE (1/m) | 0.08150 | 0.05360 | 0.05180 | 0.04620 | -0.00170 [-0.00360, -0.00020] * | -0.03490 [-0.05780, -0.01930] * |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8458 | 0.8629 | 0.8670 | 0.7728 | +0.0042 [-0.0050, +0.0142] | +0.0213 [+0.0042, +0.0400] * |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5790 | 0.5969 | 0.6136 | 0.5336 | +0.0167 [+0.0000, +0.0371] | +0.0346 [+0.0067, +0.0633] * |
| CONSISTENCY | CONSISTENCY(a) 'any' (share of windows violating; lower better) | 0.1596 | 0.2601 | 0.2701 | 0.3110 | +0.0100 [+0.0054, +0.0150] * | +0.1105 [+0.0796, +0.1426] * |
| CONSISTENCY | CONSISTENCY(a) SPEED | 0.0955 | 0.1955 | 0.2043 | 0.2418 | +0.0088 [+0.0050, +0.0129] * | +0.1088 [+0.0808, +0.1376] * |
| CONSISTENCY | CONSISTENCY(a) STOP | 0.0188 | 0.0292 | 0.0308 | 0.0338 | +0.0017 [+0.0004, +0.0033] * | +0.0121 [+0.0058, +0.0188] * |
| CONSISTENCY | CONSISTENCY(a) TURN | 0.0042 | 0.0071 | 0.0075 | 0.0342 | +0.0004 [-0.0008, +0.0017] | +0.0033 [+0.0004, +0.0079] * |
| CONSISTENCY | CONSISTENCY(a) LEAD | 0.0725 | 0.1042 | 0.1100 | 0.1384 | +0.0058 [+0.0025, +0.0100] * | +0.0375 [+0.0225, +0.0538] * |
| CONSISTENCY | self-consistency (b) 'any' | 0.2297 | 0.3697 | 0.3781 | n/a | +0.0083 [+0.0046, +0.0125] * | +0.1484 [+0.1188, +0.1797] * |

**Stage 1, M2b_s1** (lower is better except the tactical rows; `*` = CI excludes 0)

| family | metric | B | H | HS | P10 | HS - H | HS - B |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.9112 | 0.8389 | 0.8339 | 0.9802 | -0.0050 [-0.0162, +0.0016] | -0.0773 [-0.1140, -0.0511] * |
| LONGITUDINAL | accel MAE (m/s^2) | 0.9313 | 0.7036 | 0.7274 | 0.6376 | +0.0238 [+0.0169, +0.0303] * | -0.2039 [-0.2682, -0.1517] * |
| LONGITUDINAL | along MAE (m) | 1.6081 | 1.5573 | 1.5546 | 1.8217 | -0.0026 [-0.0081, +0.0003] | -0.0534 [-0.0856, -0.0260] * |
| LATERAL | cross-track MAE (m) | 0.8644 | 0.8375 | 0.8373 | 1.1858 | -0.0003 [-0.0014, +0.0009] | -0.0271 [-0.0480, -0.0075] * |
| LATERAL | heading MAE (deg) | 4.4383 | 4.3276 | 4.3836 | 5.7302 | +0.0545 [+0.0101, +0.1119] * | -0.0546 [-0.1754, +0.0967] |
| LATERAL | yaw-rate MAE (deg/s) | 6.9864 | 6.4447 | 6.3724 | 6.3299 | -0.0712 [-0.1802, +0.0164] | -0.7111 [-1.0427, -0.4412] * |
| LATERAL | curvature MAE (1/m) | 0.06260 | 0.05720 | 0.05510 | 0.04620 | -0.00220 [-0.00440, -0.00030] * | -0.01200 [-0.02060, -0.00490] * |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8287 | 0.8470 | 0.8416 | 0.7728 | -0.0054 [-0.0196, +0.0063] | +0.0129 [+0.0025, +0.0254] * |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5919 | 0.6132 | 0.6236 | 0.5336 | +0.0104 [-0.0004, +0.0217] | +0.0317 [+0.0162, +0.0475] * |
| CONSISTENCY | CONSISTENCY(a) 'any' (share of windows violating; lower better) | 0.2097 | 0.2851 | 0.2910 | 0.3110 | +0.0058 [+0.0021, +0.0096] * | +0.0813 [+0.0629, +0.1009] * |
| CONSISTENCY | CONSISTENCY(a) SPEED | 0.1255 | 0.2238 | 0.2301 | 0.2418 | +0.0063 [+0.0025, +0.0100] * | +0.1046 [+0.0817, +0.1289] * |
| CONSISTENCY | CONSISTENCY(a) STOP | 0.0271 | 0.0296 | 0.0296 | 0.0338 | +0.0000 [-0.0013, +0.0013] | +0.0025 [+0.0004, +0.0050] * |
| CONSISTENCY | CONSISTENCY(a) TURN | 0.0067 | 0.0071 | 0.0075 | 0.0342 | +0.0004 [+0.0000, +0.0013] | +0.0008 [-0.0012, +0.0029] |
| CONSISTENCY | CONSISTENCY(a) LEAD | 0.1084 | 0.1263 | 0.1280 | 0.1384 | +0.0017 [+0.0000, +0.0038] | +0.0196 [+0.0113, +0.0288] * |
| CONSISTENCY | self-consistency (b) 'any' | 0.1430 | 0.2172 | 0.2213 | n/a | +0.0042 [+0.0013, +0.0071] * | +0.0784 [+0.0580, +0.1010] * |

**TACTICAL (T's own heads) and STRATEGIC.** HS never touches T's logits or S: T's own decision-head block equals B's banked block (json equality, s0 / s1, stage 2: True / True). **STRATEGIC: PAUSED** per the PI ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` ITEM 25), not a pass; S's banked `ck_pos_err` is carried unchanged. The plan-derived tactical agreements above DO move with the plan.

### 5a. Distance keeping beside (K = 30, 6 s lead block, `v8f_dk_read.read_dk`; min-TTC is a CENSORED mean and is not read as better)

**Stage 2.** Window states {'LEAD': 743, 'NO_LEAD': 552, 'NOT_STRAIGHT': 931, 'NO_LABEL': 173}.

| arm | headway min (m) | time-gap min (s) | LEAD-VIOL share of at-risk |
|---|---|---|---|
| B_s0 | 25.622 | 4.022 | 0.0353 |
| H_s0 | 25.553 | 3.920 | 0.0487 |
| HS_s0 | 25.494 | 3.937 | 0.0487 |
| B_s1 | 25.544 | 3.950 | 0.0605 |
| H_s1 | 25.606 | 3.950 | 0.0588 |
| HS_s1 | 25.663 | 3.951 | 0.0588 |
| P10 | 25.116 | 3.832 | 0.0807 |

| paired contrast | headway (m) | time-gap (s) | LEAD-VIOL at-risk (share) |
|---|---|---|---|
| HS_s0 - P10 | +0.2650 [-0.0721, +0.6097] | +0.1002 [+0.0076, +0.2139] * | -0.0319 [-0.0588, -0.0085] * |
| HS_s1 - P10 | +0.3789 [+0.0072, +0.7929] * | +0.1176 [+0.0072, +0.2479] * | -0.0218 [-0.0496, +0.0017] |
| HS_s0 - H_s0 | -0.1187 [-0.2790, +0.0037] | +0.0045 [-0.0451, +0.0798] | +0.0000 [+0.0000, +0.0000] |
| HS_s1 - H_s1 | +0.0455 [-0.0811, +0.2136] | -0.0016 [-0.0432, +0.0540] | +0.0000 [-0.0047, +0.0050] |
| HS_s0 - B_s0 | -0.2787 [-0.4953, -0.0764] * | -0.0915 [-0.1718, -0.0299] * | +0.0134 [+0.0032, +0.0246] * |
| HS_s1 - B_s1 | +0.0226 [-0.1145, +0.1687] | +0.0010 [-0.0654, +0.0859] | -0.0017 [-0.0092, +0.0053] |

**Stage 1.** Window states {'LEAD': 743, 'NO_LEAD': 552, 'NOT_STRAIGHT': 931, 'NO_LABEL': 173}.

| arm | headway min (m) | time-gap min (s) | LEAD-VIOL share of at-risk |
|---|---|---|---|
| B_s0 | 25.622 | 4.022 | 0.0353 |
| H_s0 | 25.553 | 3.920 | 0.0487 |
| HS_s0 | 25.546 | 3.943 | 0.0504 |
| B_s1 | 25.544 | 3.950 | 0.0605 |
| H_s1 | 25.606 | 3.950 | 0.0588 |
| HS_s1 | 25.640 | 3.938 | 0.0605 |
| P10 | 25.116 | 3.832 | 0.0807 |

| paired contrast | headway (m) | time-gap (s) | LEAD-VIOL at-risk (share) |
|---|---|---|---|
| HS_s0 - P10 | +0.3207 [-0.0347, +0.6860] | +0.1124 [+0.0116, +0.2313] * | -0.0303 [-0.0559, -0.0081] * |
| HS_s1 - P10 | +0.3698 [-0.0516, +0.8113] | +0.1067 [+0.0009, +0.2270] * | -0.0202 [-0.0492, +0.0036] |
| HS_s0 - H_s0 | -0.0161 [-0.1402, +0.0978] | +0.0219 [-0.0240, +0.0953] | +0.0017 [+0.0000, +0.0053] |
| HS_s1 - H_s1 | +0.0348 [-0.0558, +0.1913] | -0.0122 [-0.0214, -0.0048] * | +0.0017 [+0.0000, +0.0053] |
| HS_s0 - B_s0 | -0.1913 [-0.3854, -0.0019] * | -0.0737 [-0.1405, -0.0222] * | +0.0151 [+0.0048, +0.0269] * |
| HS_s1 - B_s1 | +0.0311 [-0.1058, +0.1812] | -0.0097 [-0.0407, +0.0188] | +0.0000 [-0.0051, +0.0050] |

## 6. The fitted v per coefficient (the entire deployable artifact beside w: 24 numbers per seed and stage)

v[k, a]: k = DCT coefficient index 0..11 (higher k = higher frequency), a = along / cross. Source: `fit_hs{1,2}_M2b_s{0,1}.json` (train windows only).

**Stage 1** (lambda = 0)

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 along | 1.000 | 0.914 | 0.955 | 1.000 | 0.662 | 1.000 | 1.000 | 0.799 | 0.759 | 0.388 | 0.335 | 0.471 | 0.774 |
| M2b_s0 cross | 1.000 | 0.986 | 1.000 | 1.000 | 1.000 | 0.705 | 1.000 | 0.931 | 1.000 | 0.324 | 1.000 | 0.909 | 0.905 |
| M2b_s1 along | 1.000 | 1.000 | 0.644 | 1.000 | 1.000 | 0.659 | 1.000 | 0.680 | 0.311 | 0.000 | 0.000 | 0.704 | 0.666 |
| M2b_s1 cross | 0.982 | 1.000 | 1.000 | 1.000 | 0.602 | 1.000 | 1.000 | 0.903 | 1.000 | 0.409 | 1.000 | 1.000 | 0.908 |

**Stage 2** (lambda = 2499.59 s0, 2308.21 s1)

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 along | 0.702 | 0.918 | 0.535 | 0.425 | 0.450 | 0.815 | 0.809 | 0.792 | 0.780 | 0.590 | 0.415 | 0.343 | 0.631 |
| M2b_s0 cross | 0.872 | 1.000 | 1.000 | 0.698 | 0.779 | 1.000 | 0.969 | 0.839 | 0.761 | 0.552 | 0.970 | 0.535 | 0.831 |
| M2b_s1 along | 0.629 | 0.988 | 0.520 | 0.433 | 0.616 | 0.663 | 0.820 | 0.794 | 0.686 | 0.466 | 0.231 | 0.372 | 0.602 |
| M2b_s1 cross | 0.923 | 1.000 | 1.000 | 0.822 | 0.836 | 1.000 | 1.000 | 0.915 | 0.890 | 0.582 | 0.902 | 0.557 | 0.869 |

Reading (descriptive): stage 1 keeps most of every coefficient (mean v 0.84 / 0.79) because the position objective has no reason to shrink; at seed 1, along, coefficients 9 and 10 go to exactly 0.0. Stage 2 (mean v 0.73 / 0.74) shrinks the along axis hardest at k = 2-4 (0.43-0.62) and k = 9-11 (0.23-0.59); the cross axis stays near 1 at k = 1, 2 (and 5) and shrinks at k = 9 and 11 (0.53-0.58). The two seeds agree in shape, not in the exact values.

## 7. Per-horizon ADE (m), 1-6 s: B, H, HS, P10 (mean [95 % CI]; stage 2 and stage 1)

**Stage 2, M2b_s0**

| horizon | B | H | HS | P10 | HS - P10 | HS - H | HS - B |
|---|---|---|---|---|---|---|---|
| ADE@1s | 0.0819 [0.0714, 0.0953] | 0.0598 [0.0511, 0.0728] | 0.0639 [0.0551, 0.0773] | 0.0633 [0.0545, 0.0764] | +0.0006 [-0.0000, +0.0012] | +0.0042 [+0.0032, +0.0051] * | -0.0179 [-0.0231, -0.0133] * |
| ADE@2s | 0.3155 [0.2765, 0.3621] | 0.2123 [0.1916, 0.2392] | 0.2170 [0.1959, 0.2437] | 0.2221 [0.2008, 0.2491] | -0.0051 [-0.0081, -0.0019] * | +0.0047 [+0.0026, +0.0065] * | -0.0986 [-0.1374, -0.0694] * |
| ADE@3s | 0.5772 [0.5254, 0.6401] | 0.4734 [0.4324, 0.5198] | 0.4784 [0.4375, 0.5249] | 0.5337 [0.4865, 0.5871] | -0.0553 [-0.0673, -0.0437] * | +0.0050 [-0.0001, +0.0092] | -0.0989 [-0.1430, -0.0674] * |
| ADE@4s | 0.9838 [0.9078, 1.0702] | 0.8541 [0.7859, 0.9292] | 0.8550 [0.7850, 0.9316] | 1.0129 [0.9246, 1.1071] | -0.1579 [-0.1897, -0.1275] * | +0.0008 [-0.0077, +0.0084] | -0.1288 [-0.1757, -0.0885] * |
| ADE@5s | 1.5179 [1.4027, 1.6470] | 1.3530 [1.2458, 1.4681] | 1.3539 [1.2452, 1.4701] | 1.6628 [1.5163, 1.8107] | -0.3089 [-0.3731, -0.2488] * | +0.0008 [-0.0115, +0.0120] | -0.1641 [-0.2261, -0.1075] * |
| ADE@6s | 2.1753 [2.0071, 2.3568] | 1.9685 [1.8106, 2.1306] | 1.9756 [1.8190, 2.1397] | 2.4870 [2.2606, 2.7135] | -0.5114 [-0.6174, -0.4108] * | +0.0070 [-0.0118, +0.0247] | -0.1997 [-0.2864, -0.1208] * |

**Stage 2, M2b_s1**

| horizon | B | H | HS | P10 | HS - P10 | HS - H | HS - B |
|---|---|---|---|---|---|---|---|
| ADE@1s | 0.0706 [0.0613, 0.0832] | 0.0601 [0.0516, 0.0731] | 0.0630 [0.0540, 0.0762] | 0.0633 [0.0545, 0.0764] | -0.0004 [-0.0009, +0.0001] | +0.0028 [+0.0021, +0.0035] * | -0.0076 [-0.0118, -0.0045] * |
| ADE@2s | 0.2914 [0.2541, 0.3397] | 0.2137 [0.1931, 0.2400] | 0.2168 [0.1962, 0.2434] | 0.2221 [0.2008, 0.2491] | -0.0053 [-0.0090, -0.0012] * | +0.0031 [+0.0008, +0.0048] * | -0.0746 [-0.1139, -0.0481] * |
| ADE@3s | 0.5525 [0.4983, 0.6177] | 0.4814 [0.4414, 0.5274] | 0.4864 [0.4456, 0.5331] | 0.5337 [0.4865, 0.5871] | -0.0473 [-0.0600, -0.0344] * | +0.0049 [-0.0020, +0.0099] | -0.0661 [-0.1138, -0.0354] * |
| ADE@4s | 0.9361 [0.8585, 1.0205] | 0.8663 [0.7980, 0.9396] | 0.8693 [0.7994, 0.9445] | 1.0129 [0.9246, 1.1071] | -0.1436 [-0.1766, -0.1123] * | +0.0031 [-0.0076, +0.0121] | -0.0668 [-0.1095, -0.0349] * |
| ADE@5s | 1.4403 [1.3246, 1.5633] | 1.3687 [1.2598, 1.4832] | 1.3727 [1.2628, 1.4876] | 1.6628 [1.5163, 1.8107] | -0.2901 [-0.3547, -0.2265] * | +0.0040 [-0.0089, +0.0164] | -0.0676 [-0.1084, -0.0315] * |
| ADE@6s | 2.0546 [1.8859, 2.2323] | 1.9848 [1.8199, 2.1587] | 1.9966 [1.8332, 2.1694] | 2.4870 [2.2606, 2.7135] | -0.4904 [-0.5977, -0.3846] * | +0.0117 [-0.0064, +0.0292] | -0.0580 [-0.1050, -0.0148] * |

**Stage 1, M2b_s0**

| horizon | B | H | HS | P10 | HS - P10 | HS - H | HS - B |
|---|---|---|---|---|---|---|---|
| ADE@1s | 0.0819 [0.0714, 0.0953] | 0.0598 [0.0511, 0.0728] | 0.0595 [0.0509, 0.0725] | 0.0633 [0.0545, 0.0764] | -0.0038 [-0.0043, -0.0033] * | -0.0002 [-0.0004, -0.0001] * | -0.0223 [-0.0274, -0.0176] * |
| ADE@2s | 0.3155 [0.2765, 0.3621] | 0.2123 [0.1916, 0.2392] | 0.2101 [0.1895, 0.2370] | 0.2221 [0.2008, 0.2491] | -0.0119 [-0.0163, -0.0071] * | -0.0021 [-0.0028, -0.0016] * | -0.1054 [-0.1432, -0.0770] * |
| ADE@3s | 0.5772 [0.5254, 0.6401] | 0.4734 [0.4324, 0.5198] | 0.4719 [0.4310, 0.5182] | 0.5337 [0.4865, 0.5871] | -0.0618 [-0.0760, -0.0464] * | -0.0014 [-0.0018, -0.0011] * | -0.1053 [-0.1447, -0.0758] * |
| ADE@4s | 0.9838 [0.9078, 1.0702] | 0.8541 [0.7859, 0.9292] | 0.8520 [0.7835, 0.9274] | 1.0129 [0.9246, 1.1071] | -0.1609 [-0.1962, -0.1262] * | -0.0021 [-0.0030, -0.0013] * | -0.1318 [-0.1719, -0.0976] * |
| ADE@5s | 1.5179 [1.4027, 1.6470] | 1.3530 [1.2458, 1.4681] | 1.3520 [1.2452, 1.4669] | 1.6628 [1.5163, 1.8107] | -0.3108 [-0.3795, -0.2462] * | -0.0010 [-0.0025, +0.0004] | -0.1659 [-0.2201, -0.1174] * |
| ADE@6s | 2.1753 [2.0071, 2.3568] | 1.9685 [1.8106, 2.1306] | 1.9645 [1.8073, 2.1265] | 2.4870 [2.2606, 2.7135] | -0.5225 [-0.6360, -0.4125] * | -0.0040 [-0.0081, -0.0003] * | -0.2108 [-0.2836, -0.1434] * |

**Stage 1, M2b_s1**

| horizon | B | H | HS | P10 | HS - P10 | HS - H | HS - B |
|---|---|---|---|---|---|---|---|
| ADE@1s | 0.0706 [0.0613, 0.0832] | 0.0601 [0.0516, 0.0731] | 0.0606 [0.0519, 0.0738] | 0.0633 [0.0545, 0.0764] | -0.0027 [-0.0032, -0.0023] * | +0.0004 [-0.0000, +0.0008] | -0.0100 [-0.0141, -0.0068] * |
| ADE@2s | 0.2914 [0.2541, 0.3397] | 0.2137 [0.1931, 0.2400] | 0.2130 [0.1926, 0.2394] | 0.2221 [0.2008, 0.2491] | -0.0090 [-0.0137, -0.0037] * | -0.0006 [-0.0012, -0.0002] * | -0.0784 [-0.1160, -0.0528] * |
| ADE@3s | 0.5525 [0.4983, 0.6177] | 0.4814 [0.4414, 0.5274] | 0.4797 [0.4400, 0.5250] | 0.5337 [0.4865, 0.5871] | -0.0540 [-0.0694, -0.0374] * | -0.0017 [-0.0043, -0.0002] * | -0.0728 [-0.1164, -0.0450] * |
| ADE@4s | 0.9361 [0.8585, 1.0205] | 0.8663 [0.7980, 0.9396] | 0.8638 [0.7962, 0.9362] | 1.0129 [0.9246, 1.1071] | -0.1491 [-0.1868, -0.1115] * | -0.0024 [-0.0064, +0.0002] | -0.0723 [-0.1090, -0.0460] * |
| ADE@5s | 1.4403 [1.3246, 1.5633] | 1.3687 [1.2598, 1.4832] | 1.3667 [1.2575, 1.4819] | 1.6628 [1.5163, 1.8107] | -0.2961 [-0.3680, -0.2252] * | -0.0021 [-0.0054, +0.0001] | -0.0737 [-0.1072, -0.0465] * |
| ADE@6s | 2.0546 [1.8859, 2.2323] | 1.9848 [1.8199, 2.1587] | 1.9820 [1.8160, 2.1571] | 2.4870 [2.2606, 2.7135] | -0.5050 [-0.6240, -0.3886] * | -0.0028 [-0.0081, +0.0003] | -0.0725 [-0.1083, -0.0420] * |

FDE@1-6 s and every other row are in `score_hs{1,2}_M2b_s{0,1}.json::single / paired`.

## 8. Consistency by rule, beside (descriptive; see section 5 for the rates)

| stage | seed | rule | HS - P10 | HS - H | HS - B |
|---|---|---|---|---|---|
| 2 | M2b_s0 | any | -0.0504 [-0.0751, -0.0267] * | +0.0004 [-0.0096, +0.0104] | +0.1009 [+0.0667, +0.1350] * |
| 2 | M2b_s0 | SPEED | -0.0421 [-0.0617, -0.0229] * | +0.0042 [-0.0058, +0.0138] | +0.1042 [+0.0713, +0.1359] * |
| 2 | M2b_s0 | STOP | -0.0021 [-0.0046, -0.0004] * | +0.0025 [+0.0000, +0.0054] | +0.0129 [+0.0067, +0.0200] * |
| 2 | M2b_s0 | TURN | -0.0267 [-0.0395, -0.0150] * | +0.0004 [-0.0008, +0.0025] | +0.0033 [+0.0004, +0.0075] * |
| 2 | M2b_s0 | LEAD | -0.0308 [-0.0446, -0.0183] * | +0.0033 [-0.0021, +0.0088] | +0.0350 [+0.0188, +0.0530] * |
| 2 | M2b_s1 | any | -0.0325 [-0.0555, -0.0088] * | -0.0067 [-0.0158, +0.0025] | +0.0688 [+0.0458, +0.0918] * |
| 2 | M2b_s1 | SPEED | -0.0221 [-0.0392, -0.0046] * | -0.0042 [-0.0117, +0.0025] | +0.0942 [+0.0680, +0.1209] * |
| 2 | M2b_s1 | STOP | -0.0021 [-0.0050, +0.0000] | +0.0021 [+0.0000, +0.0050] | +0.0046 [+0.0013, +0.0087] * |
| 2 | M2b_s1 | TURN | -0.0279 [-0.0417, -0.0154] * | -0.0008 [-0.0021, +0.0000] | -0.0004 [-0.0025, +0.0013] |
| 2 | M2b_s1 | LEAD | -0.0117 [-0.0225, +0.0008] | +0.0004 [-0.0059, +0.0067] | +0.0183 [+0.0067, +0.0312] * |
| 1 | M2b_s0 | any | -0.0409 [-0.0704, -0.0100] * | +0.0100 [+0.0054, +0.0150] * | +0.1105 [+0.0796, +0.1426] * |
| 1 | M2b_s0 | SPEED | -0.0375 [-0.0617, -0.0117] * | +0.0088 [+0.0050, +0.0129] * | +0.1088 [+0.0808, +0.1376] * |
| 1 | M2b_s0 | STOP | -0.0029 [-0.0067, -0.0004] * | +0.0017 [+0.0004, +0.0033] * | +0.0121 [+0.0058, +0.0188] * |
| 1 | M2b_s0 | TURN | -0.0267 [-0.0396, -0.0146] * | +0.0004 [-0.0008, +0.0017] | +0.0033 [+0.0004, +0.0079] * |
| 1 | M2b_s0 | LEAD | -0.0283 [-0.0442, -0.0133] * | +0.0058 [+0.0025, +0.0100] * | +0.0375 [+0.0225, +0.0538] * |
| 1 | M2b_s1 | any | -0.0200 [-0.0479, +0.0087] | +0.0058 [+0.0021, +0.0096] * | +0.0813 [+0.0629, +0.1009] * |
| 1 | M2b_s1 | SPEED | -0.0117 [-0.0333, +0.0113] | +0.0063 [+0.0025, +0.0100] * | +0.1046 [+0.0817, +0.1289] * |
| 1 | M2b_s1 | STOP | -0.0042 [-0.0088, -0.0008] * | +0.0000 [-0.0013, +0.0013] | +0.0025 [+0.0004, +0.0050] * |
| 1 | M2b_s1 | TURN | -0.0267 [-0.0404, -0.0142] * | +0.0004 [+0.0000, +0.0013] | +0.0008 [-0.0012, +0.0029] |
| 1 | M2b_s1 | LEAD | -0.0104 [-0.0254, +0.0062] | +0.0017 [+0.0000, +0.0038] | +0.0196 [+0.0113, +0.0288] * |

## 9. Rule Zero: where the roughness still is (DESCRIPTIVE, computed AFTER the outcome was fixed; not a bar, not an arm)

These maps use the same banked arrays and machinery (`lever_map.py`, `lever_map_d3.py`). They change nothing in sections 0-8. Looking at lambda values other than the registered one AFTER seeing the HSPEC numbers is exactly the move the PREREG forbids as a result; so they are a map for the NEXT pre-registration, not a rescue of this one.

**D1. T0 diagnostic, eval-oracle v (fitted ON eval-141, reads GT): the ceiling of 'shrink per cosine coefficient' at the registered objectives (HS_oracle - P10).**

| seed | objective | mean v | ADE6 | accel MAE | curvature MAE |
|---|---|---|---|---|---|
| M2b_s0 | stage 1 | 0.923 | -0.5268 [-0.6360, -0.4217] * | +0.0922 [+0.0668, +0.1242] * | +0.00330 [+0.00060, +0.00630] * |
| M2b_s0 | stage 2 (train lambda) | 0.722 | -0.5047 [-0.6078, -0.4070] * | +0.0169 [+0.0022, +0.0353] * | +0.00060 [-0.00130, +0.00270] |
| M2b_s1 | stage 1 | 0.834 | -0.5041 [-0.6178, -0.3922] * | +0.0892 [+0.0622, +0.1259] * | +0.00780 [+0.00260, +0.01400] * |
| M2b_s1 | stage 2 (train lambda) | 0.753 | -0.4823 [-0.5854, -0.3803] * | +0.0030 [-0.0145, +0.0262] | +0.00280 [-0.00020, +0.00620] |

Even with the eval GT in the fit, the registered stage-2 objective does not make accel separated-better than P10 at either seed (s0 still separated-worse) and does not make curvature separated-better: the lever is exhausted at its registered form, not under-fitted.

**D2. Lambda map (TRAIN fits, m x the registered lambda; eval read; HS_m - P10). m = 1 is the registered arm; m = 0 is stage 1.**

| seed | m | mean v | ADE2 | ADE6 | accel MAE | curvature MAE | yaw-rate MAE (deg/s) |
|---|---|---|---|---|---|---|---|
| M2b_s0 | 0 | 0.839 | -0.0119 [-0.0163, -0.0071] * | -0.5225 [-0.6360, -0.4125] * | +0.0942 [+0.0684, +0.1269] * | +0.00570 [+0.00250, +0.00930] * | +0.018 [-0.175, +0.214] |
| M2b_s0 | 0.25 | 0.780 | -0.0114 [-0.0152, -0.0072] * | -0.5225 [-0.6339, -0.4129] * | +0.0480 [+0.0307, +0.0691] * | +0.00320 [+0.00060, +0.00620] * | -0.229 [-0.420, -0.056] * |
| M2b_s0 | 1 | 0.731 | -0.0051 [-0.0081, -0.0019] * | -0.5114 [-0.6174, -0.4108] * | +0.0258 [+0.0105, +0.0444] * | +0.00120 [-0.00090, +0.00360] | -0.325 [-0.491, -0.169] * |
| M2b_s0 | 4 | 0.685 | +0.0058 [+0.0039, +0.0080] * | -0.4639 [-0.5560, -0.3796] * | +0.0043 [-0.0086, +0.0201] | -0.00110 [-0.00270, +0.00060] | -0.440 [-0.595, -0.301] * |
| M2b_s0 | 16 | 0.642 | +0.0109 [+0.0083, +0.0135] * | -0.3867 [-0.4644, -0.3168] * | -0.0071 [-0.0185, +0.0062] | -0.00190 [-0.00340, -0.00040] * | -0.484 [-0.626, -0.357] * |
| M2b_s0 | 64 | 0.623 | +0.0133 [+0.0104, +0.0164] * | -0.3411 [-0.4107, -0.2782] * | -0.0093 [-0.0200, +0.0032] | -0.00260 [-0.00430, -0.00100] * | -0.508 [-0.656, -0.378] * |
| M2b_s0 | 256 | 0.617 | +0.0141 [+0.0110, +0.0174] * | -0.3260 [-0.3927, -0.2656] * | -0.0095 [-0.0200, +0.0028] | -0.00270 [-0.00430, -0.00100] * | -0.508 [-0.655, -0.378] * |
| M2b_s1 | 0 | 0.787 | -0.0090 [-0.0137, -0.0037] * | -0.5050 [-0.6240, -0.3886] * | +0.0898 [+0.0645, +0.1221] * | +0.00890 [+0.00340, +0.01560] * | +0.043 [-0.240, +0.326] |
| M2b_s1 | 0.25 | 0.771 | -0.0083 [-0.0124, -0.0036] * | -0.5028 [-0.6170, -0.3895] * | +0.0222 [+0.0039, +0.0465] * | +0.00460 [+0.00080, +0.00900] * | -0.241 [-0.472, -0.017] * |
| M2b_s1 | 1 | 0.735 | -0.0053 [-0.0090, -0.0012] * | -0.4904 [-0.5977, -0.3846] * | +0.0057 [-0.0110, +0.0283] | +0.00310 [-0.00010, +0.00660] | -0.328 [-0.540, -0.134] * |
| M2b_s1 | 4 | 0.696 | -0.0018 [-0.0045, +0.0010] | -0.4464 [-0.5409, -0.3558] * | -0.0124 [-0.0272, +0.0065] | +0.00030 [-0.00190, +0.00280] | -0.436 [-0.615, -0.275] * |
| M2b_s1 | 16 | 0.663 | +0.0018 [-0.0001, +0.0039] | -0.3883 [-0.4712, -0.3095] * | -0.0202 [-0.0334, -0.0037] * | -0.00230 [-0.00380, -0.00110] * | -0.509 [-0.676, -0.361] * |
| M2b_s1 | 64 | 0.648 | +0.0033 [+0.0016, +0.0052] * | -0.3577 [-0.4349, -0.2845] * | -0.0211 [-0.0338, -0.0054] * | -0.00240 [-0.00400, -0.00080] * | -0.506 [-0.668, -0.362] * |
| M2b_s1 | 256 | 0.643 | +0.0039 [+0.0023, +0.0058] * | -0.3478 [-0.4230, -0.2764] * | -0.0211 [-0.0336, -0.0056] * | -0.00240 [-0.00410, -0.00090] * | -0.507 [-0.668, -0.363] * |

Reading: raising lambda trades ADE2 for smoothness. Curvature becomes separated-better than P10 at both seeds from x16, but accel is separated-better only at seed 1 (x16 and above) and NEVER at seed 0 on this grid, while ADE2 is separated-WORSE than P10 at seed 0 from x4 (and at x64 and above at seed 1). So no single lambda on this grid dominates at seed 0 on ADE2 + accel + curvature together. Amplitude shrinkage alone is therefore not enough, which is the PREREG's 'partly elsewhere'.

**D3. Is part of the roughness in the per-step weights w (HBLEND's PAV plateaus and steps)? w replaced by one fixed 5-knot moving average (clipped to [0, 1]; no sweep); HS - P10.**

| seed | arm | ADE2 | ADE6 | accel MAE | curvature MAE |
|---|---|---|---|---|---|
| M2b_s0 | H (HBLEND: w as banked, v = 1) | -0.0098 [-0.0144, -0.0046] * | -0.5185 [-0.6349, -0.4066] * | +0.1037 [+0.0795, +0.1344] * | +0.00740 [+0.00360, +0.01190] * |
| M2b_s0 | H with smoothed w (v = 1) | -0.0096 [-0.0145, -0.0039] * | -0.5186 [-0.6351, -0.4068] * | +0.0799 [+0.0592, +0.1032] * | +0.00980 [+0.00520, +0.01530] * |
| M2b_s0 | HS stage-2 v with smoothed w | -0.0049 [-0.0082, -0.0014] * | -0.5111 [-0.6170, -0.4105] * | +0.0060 [-0.0063, +0.0198] | +0.00200 [-0.00030, +0.00470] |
| M2b_s1 | H (HBLEND: w as banked, v = 1) | -0.0084 [-0.0134, -0.0025] * | -0.5022 [-0.6214, -0.3845] * | +0.0660 [+0.0401, +0.1016] * | +0.01100 [+0.00450, +0.01900] * |
| M2b_s1 | H with smoothed w (v = 1) | -0.0082 [-0.0134, -0.0020] * | -0.5022 [-0.6212, -0.3846] * | +0.0574 [+0.0370, +0.0822] * | +0.01240 [+0.00550, +0.02070] * |
| M2b_s1 | HS stage-2 v with smoothed w | -0.0051 [-0.0090, -0.0009] * | -0.4900 [-0.5971, -0.3842] * | +0.0082 [-0.0057, +0.0254] | +0.00360 [+0.00060, +0.00700] * |

Smoothing w alone lowers H's accel gap to P10 (s0 +0.104 -> +0.080, s1 +0.066 -> +0.057) and does not close it; with the stage-2 v accel is not separated from P10 at either seed, curvature is not separated at s0 and separated-WORSE at s1. A real contributor, ranked below the lambda lever and below the committed training arm by measured effect.

**Levers, ranked by measured effect (each needs its own PREREG; nothing below is a new bar):**

1. **The committed next step: a training arm with a second-difference penalty on O's plan (box 2).** It is the only lever that changes what O emits rather than how much of it is kept; D1 shows the post-hoc coefficient lever is exhausted at seed 0. **Blocked on:** a PREREG for the loss term, its G-DVB entry and a launch-gate PASS token (binding 2026-09-26), a box-2 slot (compute/spend is the PI's call), and no box-2 access in this read. Unblocking = Master Mind drafts `PREREG_V8F_D2PEN` with the lambda rule stated and the dominance row (this PREREG's table) as the bar; the PI authorises the slot.
2. **A re-registered lambda rule (zero training, ~1 minute of fit).** D2: from about x16 the curvature row is separated-better at both seeds and accel at seed 1. At x16 the cost against P10 is ADE2 +0.011 m (s0) / +0.002 m (s1) and ADE6 about -0.39 m (against -0.51 / -0.49 m at the registered lambda). It could clear the curvature row at both seeds and accel at seed 1 only, not ADE2 at seed 0, so it cannot meet this PREREG's dominance row as written; it would need a train-only selection rule (e.g. episode-held-out cross-validation on the 141 train clips) and a bar that admits the ADE2 trade, which is a PI/Master Mind call.
3. **Smooth w (D3).** Smaller effect than 1-2; cheap; worth folding into whichever arm is registered next rather than a stand-alone read.

## 10. What this does and does not show

* **Evidence class.** All rows MEASURED (this run) on B's banked KIND-run plans (INHERITED from box 2's GPU run; HBLEND re-derived their metrics with 0 mismatches) and on HBLEND's banked P, H and fit windows. Section 9 is MEASURED but POST-HOC (read after the outcome was fixed).
* **Tier.** T1-class open loop on 2,399 windows / 141 episodes. Nothing here says HS drives better; only that its open-loop plan is closer to the logged future than P10 on position, speed, along, cross-track, heading, yaw rate, both plan-derived tactical agreements and consistency, and that its second-derivative metrics are NOT shown to beat the kinematic floor.
* **Variance.** The bootstrap resamples episodes with the models fixed; training-run variance enters only through the two seeds, which agree on the failing rows (accel and curvature not separated-better than P10 at either seed) but differ in size. Two seeds are a floor check, not an interval.
* **Fit-set caveat (declared in HBLEND and the PREREG).** The fit windows are windows B was trained on; the eval-oracle D1 shows the registered form is at its ceiling anyway.
* **Not done.** No H + hard-speed-projection read; no FLAT comparison (the PREREG's next step only if DOMINATES); no training.

## 11. Deliverable manifest

All under `OUT` (dev-box scratchpad; **nothing staged to git, nothing committed or pushed, nothing written outside `OUT`, nothing on box 2, nothing deleted**). The Master Mind lands them.

| artifact | note |
|---|---|
| `HSPEC_RESULT.md` | this report (generated by build_result.py from the JSONs) |
| `READ.json` | outcome, both stages' PRIMARY rows, controls summary, manifest |
| `v8f_hspec.py` | the executor (fit / score / dk / auto); declared interpretations in its docstring |
| `test_v8f_hspec.py, pytest_final.log` | controls as tests + mutations that go RED (15 tests) |
| `pytest_run1.log, pytest_run2.log` | the first test run (control 1 RED on the zero-accel threshold diagnostic: 12 passed, 2 failed) and the run after the declared control-1 clause (15 passed) |
| `fit_hs{1,2}_M2b_s{0,1}.json` | THE fitted v (12 x 2) per seed and stage, lambda, train objective terms, controls 3-4 fits |
| `score_hs{1,2}_M2b_s{0,1}.json` | every row, CI, controls, blocks (evaluate()), consistency, structural identity |
| `score_hs{1,2}_M2b_s{0,1}_perwindow.npz` | per-window HS plan, ADE/FDE/family components, violations |
| `dk_read_hs{1,2}.json, dk_perwindow_hs{1,2}.npz` | distance keeping via read_dk |
| `lever_map.py, lever_map.json, lever_map.log` | D1 eval-oracle and D2 lambda map (post-hoc, descriptive) |
| `lever_map_d3.py, lever_map_d3.json, lever_map_d3.log` | D3 smoothed w (post-hoc, descriptive) |
| `build_result.py` | report generator |
| `auto.log` | the read's log |
| `_v8f_mini_8d6449a7.py` | v8f_mini.py at the PREREG commit (reference copy of the basis/subsample code read) |

## 12. Reproduce

```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY -B v8f_hspec.py auto --n-boot 2000      # ~8 min: controls first, stage 1, stage 2 iff PRIMARY-DOM fails, READ.json
$PY -B lever_map.py ; $PY -B lever_map_d3.py  # post-hoc, descriptive
$PY -B build_result.py
$PY -B -m pytest -q -p no:cacheprovider --basetemp <OUT>/pt_basetemp test_v8f_hspec.py
```
Paths are hard-coded to the scratchpad (`SCR` in `v8f_hspec.py`); relocate before landing. Inputs are HBLEND's (read-only): `hblend/{fit_windows,weights,score}_*`, `hblend/inputs/`, `hblend/deps/`, `viz/work/{ckpt,inputs}`, `viz/tree/stack`.
