# RESULT_V8F_HACC: PREREG_V8F_HACC executed exactly as written (2026-10-25)

Dev box, CPU only, ZERO training, no GPU, no box-2 access. Every number is **MEASURED (ours)** from the artifact paths named; tier **T1-class open loop (own held outputs; no recorded future action)** (the CEILING is T0); estimator **paired episode-cluster bootstrap, B=2000, 141 episodes, seed 0 (taniteval/ci.py). Answers 'another draw of EPISODES' ONLY; training-seed variance only via the two seeds side by side and F = |ADE6(s0)-ADE6(s1)| (H-ESTIM-SEED-1); inference is deterministic (no inference-seed floor).** Eval-141: 2,399 windows. `OUT` = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hacc`.

## 0. Outcome (the PREREG outcome table, applied in its order)

Artifacts: `READ.json` (this table), `score_<seed>.json`, `dk_read.json`, `fit_<seed>.json`, `control_X1.json`; per-window plans in `score_<seed>_perwindow.npz`.

**DOMINATES P10**  (PRIMARY-DOM at both seeds: True; PRIMARY-KEEP at both seeds: True; HACC - H accel AND curvature separated-negative at both seeds: True; HACC - H accel separated-negative at either seed: True; all controls pass: True).

| seed | lambda* (m*) | PRIMARY-DOM | rows that fail it | PRIMARY-KEEP (ADE6 HACC - H) | HACC - H accel MAE | HACC - H curvature MAE |
|---|---|---|---|---|---|---|
| M2b_s0 | 9998.4 (m*=4) | PASS | none | PASS: +0.0151 [-0.0030, +0.0320], bar hi <= +0.06037 | -0.1253 [-0.1504, -0.1065] * (sep-neg: True) | -0.01310 [-0.01960, -0.00820] * (sep-neg: True) |
| M2b_s1 | 36931 (m*=16) | PASS | none | PASS: +0.0313 [+0.0026, +0.0586], bar hi <= +0.06037 | -0.1062 [-0.1360, -0.0861] * (sep-neg: True) | -0.01640 [-0.02470, -0.00930] * (sep-neg: True) |

**Committed next step (PREREG, verbatim):** *"HACC becomes the deployed post-map: 30 x 2 buffers per model, zero training, folded into `forward_infer` under its own landing with tests. The evidence videos are re-rendered on HACC. The T1-kin closed-loop read on HACC (`D-V8F-T1KIN-1`) is the next tier and needs a PI go for the pooled eval cache"*.

**Rule Zero (a refutation is a waypoint).** The dominance claim over the programme's kinematic floor is delivered at both seeds (T1-class open loop; not state of the art, and not closed loop). What blocks the committed next step: (i) folding 30 x 2 buffers into `forward_infer` is a landing with tests under its own brief (not done here; nothing is committed or pushed); (ii) the T1-kin closed-loop read (`D-V8F-T1KIN-1`) needs a PI go for the pooled eval cache.

## 1. What was run

* **Route.** HBLEND's / HSPEC's banked machinery, imported read-only (`v8f_hblend.py`: loaders, `apply_w`, `arm_pw`, `rows_for`, `single`, `fit_weights`, `pav_nondecreasing`; `v8f_hspec.py`: `load_world`, `cons_viol_arrays`, `dk_flags`, `DOM_BETTER`, `DOM_NOT_WORSE`; the eval's own `evaluate()`; the DK reader `v8f_dk_read.read_dk`; `taniteval.v8f_beta_eval.paired_row`). Fit set: HBLEND's 9,582 TRAIN windows (141 clips, k = 32); eval-141 (2,399 windows) never enters a fit or a selection. B's plan tau = the KIND run's banked `pred`; P = the checkpoint's own frozen P10; H = HBLEND's saved H.
* **Arm.** HACC = P + w (.) (tau - P), w in [0,1]^(30 x 2) a fixed function of the step index, from a bounded least squares with the position term plus lambda x the second-difference term (origin prepended, 29 cells), `scipy.optimize.lsq_linear(method='bvls', bounds=(0,1), tol=1e-12)` on the literal stacked system (565,338 x 30 rows at lambda > 0), KKT-checked (relative projected gradient <= 1e-6), NO isotonic step. Inference stays vision-only: the weight is a function of the step index; P is the deployed v8F prior; no situation-classifier output is in the goal path.
* **Declared interpretations (fixed in the script's docstring BEFORE any HACC number; none changed after).** I1 axes (0 along / 1 cross); I2 one lambda per seed shared by both axes, lambda0 = S1/S2 pooled over windows, steps and axes at HBLEND's banked w; I3 the D2 term is D2 e with e_0 = 0 (29 rows per axis); I4 the solver and the KKT definition (HSPEC's); I5 the train metrics of the lambda rule are per-window `arm_pw` means over the windows where both arms are finite, point comparisons (`<= 0.98 x` for accel and curvature, `<` for ADE2 and ADE6); I6 outcome row 3 reads 'not separated-negative at either seed' as 'at neither'; I7 separated flags = `paired_row`'s, distance keeping per HSPEC's `dk_flags`; I8 PRIMARY-KEEP bar = F/2 with the exact F = 0.12074 m = +0.06037 m (the printed +0.0604 variant is reported beside); I9 control thresholds; I10 the added machinery tie X1 (my dominance reader reproduces HSPEC's banked verdicts and every per-row flag); I11 tactical / strategic by construction.

## 2. Controls (each read its known value BEFORE any number was read; a failure is rc 2 and no outcome)

| control | bar | s0 | s1 |
|---|---|---|---|
| 1. lambda = 0 + HBLEND's isotonic step vs HBLEND's banked w, max abs | <= 1e-12 | 3.18e-14 | 4.17e-14 |
| 2. lambda = 0 (no isotonic) vs the clipped closed form, max abs | <= 1e-12 | 3.65e-14 | 4.56e-14 |
| 3. w = 1: plan vs banked pred (m) | <= 1e-4 | 0.00e+00 | 0.00e+00 |
| 3. w = 1: per-window ADE2 / ADE6 / FDE6 vs banked, max abs (m) | <= 1e-4 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| 3. w = 0: plan vs the P10 file (m) | <= 1e-4 | 1.53e-05 | 1.53e-05 |
| 3. w = 0: ADE6 mean (m) | 2.4870 +- 1e-3 | 2.48699 | 2.48699 |
| 4. KKT residual, max over EVERY train fit (lambda = 0, the 6 grid fits, both regression families at every grid lambda) | <= 1e-6 | 8.3e-12 | 7.4e-12 |
| 4. KKT residual of the eval-ceiling fit (T0) | <= 1e-6 | 2.4e-12 | 1.4e-12 |
| 5. regression GT := P10: max w, worst over the grid | <= 1e-12 | 0.0e+00 | 0.0e+00 |
| 5. regression: eval ADE6 minus the P10 file's (m) | <= 1e-4 abs | -6.0e-10 | -6.0e-10 |
| 5. r permuted across training windows: mean w at lambda* | <= 0.10 | 0.062 | 0.065 |
| 6. the lambda selection refuses an eval input (dict world / bare array / eval-split bundle) | all True | True | True |
| 7. accel instrument vs banked fw_lon_accel_mae, H / P10 (float32) | <= 2e-6 | 9.5e-07 / 4.8e-07 | 7.2e-07 / 4.8e-07 |
| X1 (added, I10). dominance reader reproduces HSPEC's banked verdicts and per-row flags | 0 of 60 rows differ | 0 differ | 0 differ |
| all controls pass |  | True | True |

* **Unit tests** (`test_v8f_hacc.py`, synthetic data plus the two TRAIN-window machinery ties on the real banked arrays): `40 passed in 10.91s`.
* **Mutations** (`mutate_hacc.py`, `mutation_results.json`; baseline GREEN: True): 15 of 15 mutants went RED by an assertion: M1_drop_the_origin_in_D2 (3 failed); M2_keep_the_isotonic_step (9 failed); M3_select_lambda_on_eval_guards_removed (1 failed); M4_swap_axes (9 failed); M5_skip_the_KKT_check (1 failed); M6a_invert_the_SMOOTHER_row (10 failed); M6b_DOMINATES_needs_only_one_of_DOM_KEEP (9 failed); M6c_NO_SMOOTHING_reads_all_not_any (2 failed); M7_rule_bar_098_becomes_1 (1 failed); M8_ADE_bar_not_strict (1 failed); M9_lambda_enters_unsquared (1 failed); M10_grid_changed (1 failed); M11_fallback2_is_not_lambda_zero (1 failed); M12_lambda0_not_at_banked_w (2 failed); M13_SMOOTHER_reads_accel_only (1 failed).

## 3. The fit and the train-only lambda rule

Arm HACC, tier T1-class open loop; every number here is on the 9,582 TRAIN windows (in-sample; the rule's only input); artifact `fit_<seed>.json`, weights in `weights_HACC_<seed>.json`.

**M2b_s0.**  lambda0 = 2499.6 (position term / D2 term at HBLEND's banked w, pooled over axes; train).  **lambda\* = 9998.4 (m\* = 4)**, rule branch: *qualified: the smallest grid lambda meeting all four train conditions*.

| m | lambda | train accel MAE | P10 | x P10 | train curvature MAE | P10 | x P10 | train ADE2 | P10 | train ADE6 | P10 | qualifies | w mean | KKT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.8187 | 0.7046 | 1.162 | 0.05983 | 0.05204 | 1.150 | 0.2073 | 0.2186 | 2.0332 | 2.5502 | False | 0.604 | 2.5e-14 |
| 1 | 2499.6 | 0.6762 | 0.7046 | 0.960 | 0.05234 | 0.05204 | 1.006 | 0.2120 | 0.2186 | 2.0463 | 2.5502 | False | 0.561 | 3.9e-12 |
| 4 | 9998.4 | 0.6711 | 0.7046 | 0.952 | 0.04914 | 0.05204 | 0.944 | 0.2117 | 0.2186 | 2.0644 | 2.5502 | True | 0.526 | 6.5e-12 |
| 16 | 39993 | 0.6680 | 0.7046 | 0.948 | 0.04732 | 0.05204 | 0.909 | 0.2115 | 0.2186 | 2.0796 | 2.5502 | True | 0.503 | 2.5e-12 |
| 64 | 1.5997e+05 | 0.6670 | 0.7046 | 0.947 | 0.04693 | 0.05204 | 0.902 | 0.2114 | 0.2186 | 2.0856 | 2.5502 | True | 0.495 | 8.3e-12 |
| 256 | 6.3989e+05 | 0.6667 | 0.7046 | 0.946 | 0.04673 | 0.05204 | 0.898 | 0.2114 | 0.2186 | 2.0873 | 2.5502 | True | 0.493 | 3.2e-12 |

w\* by step (t = 0.2 (i+1) s), along / cross: i=1: 0.000 / 1.000; i=5: 0.015 / 0.374; i=10: 0.200 / 0.856; i=11: 0.257 / 0.882; i=12: 0.315 / 0.898; i=13: 0.367 / 0.907; i=14: 0.407 / 0.903; i=15: 0.438 / 0.889; i=20: 0.511 / 0.773; i=25: 0.523 / 0.701; i=26: 0.528 / 0.694; i=29: 0.520 / 0.673; i=30: 0.529 / 0.684.  Mean along 0.334, cross 0.718; non-monotone steps (decreases > 1e-9): along 4, cross 18. Banked in `weights_HACC_M2b_s0.json`.

**M2b_s1.**  lambda0 = 2308.2 (position term / D2 term at HBLEND's banked w, pooled over axes; train).  **lambda\* = 36931 (m\* = 16)**, rule branch: *qualified: the smallest grid lambda meeting all four train conditions*.

| m | lambda | train accel MAE | P10 | x P10 | train curvature MAE | P10 | x P10 | train ADE2 | P10 | train ADE6 | P10 | qualifies | w mean | KKT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.8203 | 0.7046 | 1.164 | 0.06731 | 0.05192 | 1.296 | 0.2077 | 0.2186 | 1.9779 | 2.5502 | False | 0.640 | 1.4e-14 |
| 1 | 2308.2 | 0.6670 | 0.7046 | 0.947 | 0.05867 | 0.05185 | 1.132 | 0.2144 | 0.2186 | 1.9946 | 2.5502 | False | 0.610 | 3.2e-12 |
| 4 | 9232.8 | 0.6607 | 0.7046 | 0.938 | 0.05306 | 0.05204 | 1.020 | 0.2138 | 0.2186 | 2.0174 | 2.5502 | False | 0.571 | 7.4e-12 |
| 16 | 36931 | 0.6579 | 0.7046 | 0.934 | 0.04975 | 0.05204 | 0.956 | 0.2134 | 0.2186 | 2.0378 | 2.5502 | True | 0.544 | 4.4e-12 |
| 64 | 1.4773e+05 | 0.6572 | 0.7046 | 0.933 | 0.04898 | 0.05204 | 0.941 | 0.2133 | 0.2186 | 2.0462 | 2.5502 | True | 0.534 | 4.4e-12 |
| 256 | 5.909e+05 | 0.6571 | 0.7046 | 0.933 | 0.04875 | 0.05204 | 0.937 | 0.2132 | 0.2186 | 2.0486 | 2.5502 | True | 0.531 | 5.1e-12 |

w\* by step (t = 0.2 (i+1) s), along / cross: i=1: 0.000 / 1.000; i=5: 0.010 / 0.350; i=10: 0.220 / 0.858; i=11: 0.282 / 0.879; i=12: 0.347 / 0.886; i=13: 0.409 / 0.889; i=14: 0.460 / 0.884; i=15: 0.503 / 0.871; i=20: 0.580 / 0.776; i=25: 0.575 / 0.707; i=26: 0.576 / 0.701; i=29: 0.576 / 0.678; i=30: 0.590 / 0.687.  Mean along 0.374, cross 0.714; non-monotone steps (decreases > 1e-9): along 7, cross 18. Banked in `weights_HACC_M2b_s1.json`.

## 4. PRIMARY-DOM and PRIMARY-KEEP: the dominance row, HACC - P10 (paired; lower is better except the two tactical agreements; `*` = CI excludes 0)

Artifacts: `score_<seed>.json` (paired.HACC_minus_P10, consistency.paired.HACC_minus_P10.viol_any), `dk_read.json` (pairs HACC_<s>__minus__P10), `READ.json` (per_seed.PRIMARY_DOM).

| metric | requirement | s0: HACC - P10 [95 % CI] | s0 | s1: HACC - P10 [95 % CI] | s1 |
|---|---|---|---|---|---|
| ADE2 | separated-better | -0.0106 [-0.0150, -0.0056] * | better / OK | -0.0083 [-0.0133, -0.0026] * | better / OK |
| ADE6 | separated-better | -0.5034 [-0.6073, -0.4050] * | better / OK | -0.4708 [-0.5695, -0.3748] * | better / OK |
| accel MAE | separated-better | -0.0215 [-0.0345, -0.0087] * | better / OK | -0.0402 [-0.0509, -0.0293] * | better / OK |
| curvature MAE | separated-better | -0.00560 [-0.00960, -0.00210] * | better / OK | -0.00510 [-0.00830, -0.00230] * | better / OK |
| speed MAE | not separated-worse | -0.1132 [-0.1547, -0.0666] * | better / OK | -0.1306 [-0.1646, -0.0934] * | better / OK |
| along MAE | not separated-worse | -0.2323 [-0.3042, -0.1569] * | better / OK | -0.2390 [-0.2981, -0.1754] * | better / OK |
| cross-track MAE | not separated-worse | -0.4027 [-0.4906, -0.3173] * | better / OK | -0.3344 [-0.4239, -0.2484] * | better / OK |
| heading MAE | not separated-worse | -1.5548 [-1.9150, -1.2099] * | better / OK | -1.4346 [-1.8027, -1.0956] * | better / OK |
| yaw-rate MAE | not separated-worse | -0.8374 [-1.1086, -0.6108] * | better / OK | -0.7927 [-1.0245, -0.5881] * | better / OK |
| plan-derived lateral decision agreement | not separated-worse | +0.0809 [+0.0617, +0.1017] * | better / OK | +0.0704 [+0.0496, +0.0917] * | better / OK |
| plan-derived longitudinal decision agreement | not separated-worse | +0.0479 [+0.0117, +0.0821] * | better / OK | +0.0604 [+0.0242, +0.0963] * | better / OK |
| CONSISTENCY(a) 'any' | not separated-worse | -0.0750 [-0.1017, -0.0496] * | better / OK | -0.0463 [-0.0684, -0.0242] * | better / OK |
| DK headway min (m) | not separated-worse | +0.2132 [-0.0879, +0.5363] | not sep. / OK | +0.1986 [-0.1648, +0.5916] | not sep. / OK |
| DK time-gap min (s) | not separated-worse | +0.0702 [-0.0086, +0.1593] | not sep. / OK | +0.0457 [-0.0489, +0.1481] | not sep. / OK |
| DK LEAD-VIOL at-risk share | not separated-worse | -0.0353 [-0.0625, -0.0124] * | better / OK | -0.0218 [-0.0487, +0.0000] | not sep. / OK |

* PRIMARY-DOM M2b_s0: **PASS** (separated-better rows ok: True, no separated-worse rows: True). PRIMARY-KEEP M2b_s0: ADE6 HACC - H = +0.0151 [-0.0030, +0.0320], bar hi <= +0.06037 -> **PASS** (printed +0.0604: PASS).
* PRIMARY-DOM M2b_s1: **PASS** (separated-better rows ok: True, no separated-worse rows: True). PRIMARY-KEEP M2b_s1: ADE6 HACC - H = +0.0313 [+0.0026, +0.0586], bar hi <= +0.06037 -> **PASS** (printed +0.0604: PASS).

## 5. The four families and ADE: B / H / HACC / P10 and the paired reads (per-family, never pooled)

Tier T1-class open loop; means are unpaired window means, the three right-hand columns are PAIRED episode-cluster bootstrap differences; artifacts `score_<seed>.json` (single, paired, consistency) and `dk_read.json`.

**M2b_s0** (lower is better except the tactical rows; `*` = CI excludes 0)

| family | metric | B | H | HACC | P10 | HACC - H | HACC - B | HACC - P10 |
|---|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 1.0674 | 0.8709 | 0.8670 | 0.9802 | -0.0038 [-0.0153, +0.0071] | -0.2004 [-0.2591, -0.1473] * | -0.1132 [-0.1547, -0.0666] * |
| LONGITUDINAL | accel MAE (m/s^2) | 1.2058 | 0.7414 | 0.6161 | 0.6376 | -0.1253 [-0.1504, -0.1065] * | -0.5897 [-0.6994, -0.4871] * | -0.0215 [-0.0345, -0.0087] * |
| LONGITUDINAL | along MAE (m) | 1.8097 | 1.5925 | 1.5894 | 1.8217 | -0.0031 [-0.0184, +0.0115] | -0.2203 [-0.3080, -0.1396] * | -0.2323 [-0.3042, -0.1569] * |
| LATERAL | cross-track MAE (m) | 0.7662 | 0.7557 | 0.7831 | 1.1858 | +0.0274 [+0.0148, +0.0409] * | +0.0169 [-0.0068, +0.0403] | -0.4027 [-0.4906, -0.3173] * |
| LATERAL | heading MAE (deg) | 4.7804 | 4.0259 | 4.1754 | 5.7302 | +0.1495 [+0.0653, +0.2369] * | -0.5718 [-1.0701, -0.1843] * | -1.5548 [-1.9150, -1.2099] * |
| LATERAL | yaw-rate MAE (deg/s) | 8.1167 | 6.4244 | 5.4528 | 6.3299 | -0.9352 [-1.2809, -0.6495] * | -2.6641 [-3.7395, -1.8625] * | -0.8374 [-1.1086, -0.6108] * |
| LATERAL | curvature MAE (1/m) | 0.08150 | 0.05360 | 0.03920 | 0.04620 | -0.01310 [-0.01960, -0.00820] * | -0.04350 [-0.07100, -0.02520] * | -0.00560 [-0.00960, -0.00210] * |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8458 | 0.8629 | 0.8537 | 0.7728 | -0.0092 [-0.0209, +0.0025] | +0.0079 [-0.0075, +0.0238] | +0.0809 [+0.0617, +0.1017] * |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5790 | 0.5969 | 0.5815 | 0.5336 | -0.0154 [-0.0254, -0.0071] * | +0.0025 [-0.0129, +0.0183] | +0.0479 [+0.0117, +0.0821] * |
| ADE/FDE | ADE1 (m) | 0.0819 | 0.0598 | 0.0606 | 0.0633 | +0.0008 [+0.0004, +0.0013] * | -0.0213 [-0.0265, -0.0166] * | -0.0027 [-0.0030, -0.0024] * |
| ADE/FDE | ADE2 (m) | 0.3155 | 0.2123 | 0.2114 | 0.2221 | -0.0008 [-0.0013, -0.0004] * | -0.1041 [-0.1418, -0.0757] * | -0.0106 [-0.0150, -0.0056] * |
| ADE/FDE | ADE3 (m) | 0.5772 | 0.4734 | 0.4728 | 0.5337 | -0.0006 [-0.0019, +0.0008] | -0.1044 [-0.1443, -0.0749] * | -0.0609 [-0.0748, -0.0461] * |
| ADE/FDE | ADE4 (m) | 0.9838 | 0.8541 | 0.8520 | 1.0129 | -0.0021 [-0.0061, +0.0016] | -0.1318 [-0.1753, -0.0953] * | -0.1609 [-0.1946, -0.1283] * |
| ADE/FDE | ADE5 (m) | 1.5179 | 1.3530 | 1.3542 | 1.6628 | +0.0011 [-0.0083, +0.0097] | -0.1638 [-0.2236, -0.1100] * | -0.3086 [-0.3732, -0.2476] * |
| ADE/FDE | ADE6 (m) | 2.1753 | 1.9685 | 1.9836 | 2.4870 | +0.0151 [-0.0030, +0.0320] | -0.1917 [-0.2759, -0.1139] * | -0.5034 [-0.6073, -0.4050] * |
| ADE/FDE | FDE6 (m) | 6.2810 | 5.8185 | 5.9381 | 7.6673 | +0.1195 [+0.0411, +0.1966] * | -0.3429 [-0.6018, -0.1055] * | -1.7292 [-2.1014, -1.3781] * |
| CONSISTENCY | viol_any (share of windows violating; lower better) | 0.1596 | 0.2601 | 0.2359 | 0.3110 | -0.0242 [-0.0350, -0.0133] * | +0.0763 [+0.0450, +0.1092] * | -0.0750 [-0.1017, -0.0496] * |
| CONSISTENCY | viol_SPEED (share of windows violating; lower better) | 0.0955 | 0.1955 | 0.1688 | 0.2418 | -0.0267 [-0.0375, -0.0171] * | +0.0734 [+0.0450, +0.1025] * | -0.0729 [-0.0958, -0.0500] * |
| CONSISTENCY | viol_STOP (share of windows violating; lower better) | 0.0188 | 0.0292 | 0.0321 | 0.0338 | +0.0029 [+0.0008, +0.0058] * | +0.0133 [+0.0067, +0.0208] * | -0.0017 [-0.0038, +0.0000] |
| CONSISTENCY | viol_TURN (share of windows violating; lower better) | 0.0042 | 0.0071 | 0.0063 | 0.0342 | -0.0008 [-0.0029, +0.0017] | +0.0021 [-0.0008, +0.0058] | -0.0279 [-0.0408, -0.0163] * |
| CONSISTENCY | viol_LEAD (share of windows violating; lower better) | 0.0725 | 0.1042 | 0.1013 | 0.1384 | -0.0029 [-0.0087, +0.0025] | +0.0288 [+0.0142, +0.0446] * | -0.0371 [-0.0529, -0.0225] * |
| CONSISTENCY | self_viol_any (share of windows violating; lower better) | 0.2297 | 0.3697 | 0.3522 | n/a | -0.0175 [-0.0246, -0.0112] * | +0.1226 [+0.0938, +0.1517] * | n/a |

**M2b_s1** (lower is better except the tactical rows; `*` = CI excludes 0)

| family | metric | B | H | HACC | P10 | HACC - H | HACC - B | HACC - P10 |
|---|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.9112 | 0.8389 | 0.8497 | 0.9802 | +0.0108 [-0.0060, +0.0271] | -0.0616 [-0.1015, -0.0304] * | -0.1306 [-0.1646, -0.0934] * |
| LONGITUDINAL | accel MAE (m/s^2) | 0.9313 | 0.7036 | 0.5974 | 0.6376 | -0.1062 [-0.1360, -0.0861] * | -0.3339 [-0.4234, -0.2660] * | -0.0402 [-0.0509, -0.0293] * |
| LONGITUDINAL | along MAE (m) | 1.6081 | 1.5573 | 1.5828 | 1.8217 | +0.0255 [-0.0017, +0.0516] | -0.0253 [-0.0777, +0.0222] | -0.2390 [-0.2981, -0.1754] * |
| LATERAL | cross-track MAE (m) | 0.8644 | 0.8375 | 0.8514 | 1.1858 | +0.0138 [+0.0004, +0.0272] * | -0.0131 [-0.0441, +0.0166] | -0.3344 [-0.4239, -0.2484] * |
| LATERAL | heading MAE (deg) | 4.4383 | 4.3276 | 4.2955 | 5.7302 | -0.0328 [-0.3314, +0.1867] | -0.1427 [-0.3632, +0.0599] | -1.4346 [-1.8027, -1.0956] * |
| LATERAL | yaw-rate MAE (deg/s) | 6.9864 | 6.4447 | 5.4975 | 6.3299 | -0.9166 [-1.3001, -0.5791] * | -1.5242 [-2.1155, -1.0191] * | -0.7927 [-1.0245, -0.5881] * |
| LATERAL | curvature MAE (1/m) | 0.06260 | 0.05720 | 0.03970 | 0.04620 | -0.01640 [-0.02470, -0.00930] * | -0.02420 [-0.03820, -0.01210] * | -0.00510 [-0.00830, -0.00230] * |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8287 | 0.8470 | 0.8433 | 0.7728 | -0.0038 [-0.0167, +0.0092] | +0.0146 [-0.0013, +0.0329] | +0.0704 [+0.0496, +0.0917] * |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5919 | 0.6132 | 0.5940 | 0.5336 | -0.0192 [-0.0375, -0.0013] * | +0.0021 [-0.0104, +0.0150] | +0.0604 [+0.0242, +0.0963] * |
| ADE/FDE | ADE1 (m) | 0.0706 | 0.0601 | 0.0607 | 0.0633 | +0.0006 [+0.0001, +0.0009] * | -0.0099 [-0.0139, -0.0067] * | -0.0026 [-0.0029, -0.0023] * |
| ADE/FDE | ADE2 (m) | 0.2914 | 0.2137 | 0.2137 | 0.2221 | +0.0000 [-0.0007, +0.0007] | -0.0777 [-0.1152, -0.0522] * | -0.0083 [-0.0133, -0.0026] * |
| ADE/FDE | ADE3 (m) | 0.5525 | 0.4814 | 0.4843 | 0.5337 | +0.0028 [+0.0006, +0.0054] * | -0.0682 [-0.1092, -0.0412] * | -0.0494 [-0.0660, -0.0310] * |
| ADE/FDE | ADE4 (m) | 0.9361 | 0.8663 | 0.8699 | 1.0129 | +0.0036 [-0.0031, +0.0104] | -0.0662 [-0.1037, -0.0377] * | -0.1430 [-0.1788, -0.1077] * |
| ADE/FDE | ADE5 (m) | 1.4403 | 1.3687 | 1.3788 | 1.6628 | +0.0101 [-0.0047, +0.0243] | -0.0615 [-0.1025, -0.0258] * | -0.2840 [-0.3468, -0.2226] * |
| ADE/FDE | ADE6 (m) | 2.0546 | 1.9848 | 2.0161 | 2.4870 | +0.0313 [+0.0026, +0.0586] * | -0.0384 [-0.0948, +0.0137] | -0.4708 [-0.5695, -0.3748] * |
| ADE/FDE | FDE6 (m) | 5.8582 | 5.8168 | 6.0054 | 7.6673 | +0.1886 [+0.0652, +0.3092] * | +0.1472 [-0.0364, +0.3198] | -1.6619 [-2.0017, -1.3386] * |
| CONSISTENCY | viol_any (share of windows violating; lower better) | 0.2097 | 0.2851 | 0.2647 | 0.3110 | -0.0204 [-0.0330, -0.0088] * | +0.0550 [+0.0317, +0.0779] * | -0.0463 [-0.0684, -0.0242] * |
| CONSISTENCY | viol_SPEED (share of windows violating; lower better) | 0.1255 | 0.2238 | 0.1972 | 0.2418 | -0.0267 [-0.0367, -0.0171] * | +0.0717 [+0.0467, +0.0967] * | -0.0446 [-0.0621, -0.0275] * |
| CONSISTENCY | viol_STOP (share of windows violating; lower better) | 0.0271 | 0.0296 | 0.0325 | 0.0338 | +0.0029 [+0.0008, +0.0062] * | +0.0054 [+0.0017, +0.0100] * | -0.0013 [-0.0033, +0.0000] |
| CONSISTENCY | viol_TURN (share of windows violating; lower better) | 0.0067 | 0.0071 | 0.0079 | 0.0342 | +0.0008 [-0.0017, +0.0038] | +0.0013 [-0.0017, +0.0050] | -0.0263 [-0.0392, -0.0142] * |
| CONSISTENCY | viol_LEAD (share of windows violating; lower better) | 0.1084 | 0.1263 | 0.1246 | 0.1384 | -0.0017 [-0.0104, +0.0062] | +0.0163 [+0.0033, +0.0300] * | -0.0138 [-0.0234, -0.0033] * |
| CONSISTENCY | self_viol_any (share of windows violating; lower better) | 0.1430 | 0.2172 | 0.2076 | n/a | -0.0096 [-0.0183, +0.0000] | +0.0646 [+0.0462, +0.0850] * | n/a |

**LONGITUDINAL distance keeping** (the programme's own reader `v8f_dk_read.read_dk`, K = 30, B = 2000; headway / time-gap: higher = more margin; LEAD-VIOL: share of at-risk windows, lower better)

| arm | headway min (m) | time-gap min (s) | LEAD-VIOL at-risk |
|---|---|---|---|
| B_s0 | 25.6223 | 4.0223 | 0.035294 |
| H_s0 | 25.5529 | 3.92 | 0.048739 |
| HACC_s0 | 25.491 | 3.9006 | 0.045378 |
| B_s1 | 25.5441 | 3.9505 | 0.060504 |
| H_s1 | 25.6057 | 3.9498 | 0.058824 |
| HACC_s1 | 25.3954 | 3.8731 | 0.058824 |
| P10 | 25.1162 | 3.8316 | 0.080672 |

| pair | headway min (m) | time-gap min (s) | LEAD-VIOL at-risk |
|---|---|---|---|
| HACC_s0 - P10 | +0.2132 [-0.0879, +0.5363] | +0.0702 [-0.0086, +0.1593] | -0.0353 [-0.0625, -0.0124] * |
| HACC_s0 - H_s0 | -0.0670 [-0.1722, +0.0635] | -0.0202 [-0.0585, +0.0141] | -0.0034 [-0.0104, +0.0030] |
| HACC_s0 - B_s0 | -0.2763 [-0.4925, -0.0498] * | -0.1158 [-0.2570, -0.0112] * | +0.0101 [+0.0000, +0.0213] |
| HACC_s1 - P10 | +0.1986 [-0.1648, +0.5916] | +0.0457 [-0.0489, +0.1481] | -0.0218 [-0.0487, +0.0000] |
| HACC_s1 - H_s1 | -0.1860 [-0.3265, -0.0555] * | -0.0735 [-0.1497, -0.0175] * | +0.0000 [-0.0049, +0.0049] |
| HACC_s1 - B_s1 | -0.1900 [-0.3649, -0.0228] * | -0.0710 [-0.1419, -0.0142] * | -0.0017 [-0.0090, +0.0055] |

## 6. The per-step accel profile (beside; `accel_profile.py`'s float32 instrument, tied to the banked metric by control 7)

Tier T1-class open loop, eval-141, all 2,399 windows, unpaired cell means (descriptive, no bar); artifact `accel_profile.json` (= `score_<seed>.json` accel_profile).

**M2b_s0**: mean |a_plan - a_gt| (m/s^2) per band; the gap columns are minus P10 (negative = better than P10).

| band | HACC | H | P10 | B | HACC - P10 | H - P10 |
|---|---|---|---|---|---|---|
| 0.8-1.6 s (cells 4-8) | 0.707 | 0.856 | 0.747 | 2.598 | -0.041 | +0.109 |
| 2.2-2.6 s (cells 11-13) | 0.643 | 1.107 | 0.670 | 0.672 | -0.027 | +0.437 |
| 2.8-5.0 s (cells 14-25) | 0.568 | 0.564 | 0.653 | 0.579 | -0.085 | -0.089 |
| 5.2-5.8 s (cells 26-29) | 0.799 | 1.091 | 0.565 | 1.536 | +0.233 | +0.526 |
| all 29 cells | 0.616 | 0.741 | 0.638 | 1.206 | -0.022 | +0.104 |

per cell (k: t_s HACC-P10 / H-P10): 1: +0.00 / -0.01; 2: -0.01 / +0.03; 3: +0.03 / +0.00; 4: -0.01 / +0.23; 5: +0.02 / +0.13; 6: -0.03 / -0.02; 7: -0.09 / +0.12; 8: -0.09 / +0.08; 9: -0.00 / -0.04; 10: -0.27 / +0.14; 11: -0.08 / +0.59; 12: +0.07 / +0.63; 13: -0.07 / +0.10; 14: -0.18 / -0.16; 15: -0.10 / -0.08; 16: -0.08 / -0.07; 17: -0.16 / -0.17; 18: -0.10 / -0.14; 19: -0.09 / -0.11; 20: -0.08 / -0.09; 21: -0.03 / -0.03; 22: -0.04 / -0.03; 23: -0.06 / -0.07; 24: -0.06 / -0.06; 25: -0.05 / -0.05; 26: +0.18 / +0.31; 27: +0.34 / +0.65; 28: +0.12 / +0.23; 29: +0.30 / +0.91.  Full arrays: `accel_profile.json`.

**M2b_s1**: mean |a_plan - a_gt| (m/s^2) per band; the gap columns are minus P10 (negative = better than P10).

| band | HACC | H | P10 | B | HACC - P10 | H - P10 |
|---|---|---|---|---|---|---|
| 0.8-1.6 s (cells 4-8) | 0.714 | 0.811 | 0.747 | 1.850 | -0.034 | +0.064 |
| 2.2-2.6 s (cells 11-13) | 0.596 | 1.200 | 0.670 | 0.612 | -0.074 | +0.530 |
| 2.8-5.0 s (cells 14-25) | 0.578 | 0.585 | 0.653 | 0.552 | -0.075 | -0.068 |
| 5.2-5.8 s (cells 26-29) | 0.692 | 0.817 | 0.565 | 1.081 | +0.127 | +0.252 |
| all 29 cells | 0.597 | 0.704 | 0.638 | 0.931 | -0.040 | +0.066 |

per cell (k: t_s HACC-P10 / H-P10): 1: +0.00 / -0.00; 2: -0.01 / +0.00; 3: +0.01 / -0.00; 4: -0.01 / +0.18; 5: +0.02 / +0.09; 6: -0.03 / -0.03; 7: -0.09 / +0.05; 8: -0.05 / +0.03; 9: -0.07 / -0.04; 10: -0.32 / -0.14; 11: -0.11 / +0.70; 12: +0.01 / +0.60; 13: -0.12 / +0.29; 14: -0.21 / -0.11; 15: -0.10 / +0.00; 16: -0.05 / +0.04; 17: -0.14 / -0.13; 18: -0.07 / -0.10; 19: -0.03 / -0.07; 20: -0.04 / -0.09; 21: -0.04 / -0.08; 22: -0.05 / -0.07; 23: -0.05 / -0.07; 24: -0.05 / -0.07; 25: -0.05 / -0.07; 26: +0.08 / +0.18; 27: +0.19 / +0.34; 28: +0.03 / +0.16; 29: +0.19 / +0.32.  Full arrays: `accel_profile.json`.

## 7. CEILING (T0 diagnostic, never a result): the same objective at lambda*, fitted on the EVAL GT

Tier **T0 diagnostic** (the fit reads the eval GT); paired episode-cluster bootstrap; artifact `score_<seed>.json` (ceiling_T0).

| seed | metric | CEILING - P10 | CEILING - H | CEILING - HACC |
|---|---|---|---|---|
| M2b_s0 | ADE2 | -0.0098 [-0.0140, -0.0050] * | +0.0000 [-0.0005, +0.0005] | +0.0008 [+0.0002, +0.0014] * |
| M2b_s0 | ADE6 | -0.5040 [-0.6070, -0.4065] * | +0.0144 [-0.0090, +0.0354] | -0.0007 [-0.0070, +0.0053] |
| M2b_s0 | accel MAE | -0.0329 [-0.0451, -0.0207] * | -0.1366 [-0.1621, -0.1165] * | -0.0113 [-0.0141, -0.0088] * |
| M2b_s0 | curvature MAE | -0.00560 [-0.00960, -0.00210] * | -0.01310 [-0.01940, -0.00820] * | -0.00000 [-0.00050, +0.00060] |
| M2b_s0 | yaw-rate MAE | -0.8729 [-1.1363, -0.6464] * | -0.9707 [-1.3000, -0.6994] * | -0.0355 [-0.0717, +0.0079] |
| M2b_s1 | ADE2 | -0.0080 [-0.0132, -0.0015] * | +0.0003 [-0.0002, +0.0011] | +0.0003 [-0.0007, +0.0015] |
| M2b_s1 | ADE6 | -0.4732 [-0.5718, -0.3767] * | +0.0289 [-0.0006, +0.0566] | -0.0024 [-0.0044, -0.0006] * |
| M2b_s1 | accel MAE | -0.0474 [-0.0589, -0.0353] * | -0.1133 [-0.1411, -0.0938] * | -0.0072 [-0.0095, -0.0047] * |
| M2b_s1 | curvature MAE | -0.00480 [-0.00790, -0.00210] * | -0.01610 [-0.02410, -0.00930] * | +0.00030 [-0.00010, +0.00080] |
| M2b_s1 | yaw-rate MAE | -0.8293 [-1.0565, -0.6286] * | -0.9540 [-1.3180, -0.6293] * | -0.0366 [-0.0692, -0.0046] * |

The ceiling reads the eval GT, so it bounds what a per-step weight can do at this lambda; it is not a deployable arm and enters no row above.

## 8. TACTICAL and STRATEGIC families (identical to H by construction)

* M2b_s0: T's decision heads and S's checkpoint outputs are the banked B arrays; the eval's own `evaluate()` on the HACC plan returns a tactical block equal to the banked B's: **True**; strategic block equal to the banked B's: **None** (`None` = the evaluate() block carries no strategic entry, in which case S's `ck_pos_err` is carried unchanged from the banked dump).
* M2b_s1: T's decision heads and S's checkpoint outputs are the banked B arrays; the eval's own `evaluate()` on the HACC plan returns a tactical block equal to the banked B's: **True**; strategic block equal to the banked B's: **None** (`None` = the evaluate() block carries no strategic entry, in which case S's `ck_pos_err` is carried unchanged from the banked dump).
* STRATEGIC is **not paused** for v8F (PI-V8F-2026-10-17; R72/R73): HACC - H is exactly 0 on it because the post-map touches the plan only. `v8f_paired_families.py` was not run: it reads eval `run` dumps and HACC produces none through `evaluate()`. The plan-derived tactical agreements (lateral / longitudinal decision) move with the plan and are in sections 4 and 5.

## 9. Deliverable manifest

Everything is under `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hacc`. Nothing was committed, pushed or staged. The list with sizes and raw-bytes git blob sha1 is `FILES.txt`. The coordinator lands the files.


## MM independent re-derivation (2026-10-25, Master Mind; descriptive; it changes no row of the read)

The verdict above came from a build agent. Before it was quoted anywhere, the Master Mind re-derived the headline with code that shares nothing with the agent's fit or bootstrap (`raw/2026-10-25-hacc/mm_verify/verify_mm.{py,log,json}`):
* its own normal equations;
* a **different solver**: L-BFGS-B on ½wᵀAw − bᵀw, bounds [0, 1];
* its own D2 with the origin prepended;
* its own λ0, ADE and float32 accel;
* its own episode-cluster bootstrap (numpy, B = 2,000, rng 20261025).

It reuses only HSPEC's `load_world` (the eval arrays HBLEND and HSPEC already read) and the banked training fit windows.

| check | s0 | s1 |
|---|---|---|
| λ0 at HBLEND's w, train windows (mine / banked) | 2499.588 / 2499.588 | 2308.211 / 2308.211 |
| w at λ\* re-fitted from the TRAIN windows alone, different solver: max \|w_mine − w_banked\| | 1.5e-5 | 7.2e-6 |
| my ADE6 / accel vs the eval's `arm_pw` (P10), max abs | 0.0 / 4.8e-7 | 0.0 / 4.8e-7 |
| HACC − P10 ADE2 | −0.0106 [−0.0151, −0.0062] * | −0.0083 [−0.0132, −0.0032] * |
| HACC − P10 ADE6 | −0.5034 [−0.6082, −0.4030] * | −0.4708 [−0.5704, −0.3777] * |
| HACC − P10 accel MAE (m/s²) | −0.0215 [−0.0354, −0.0085] * | −0.0402 [−0.0518, −0.0291] * |
| KEEP: HACC − H ADE6, upper bound (bar ≤ +0.06037) | +0.0151 [−0.0029, **+0.0331**] | +0.0313 [+0.0031, **+0.0581**] |

* **The λ rule was re-applied by hand** to the agent's train grid (`fit_M2b_s*.json`). The smallest grid λ meeting all four train conditions is m = 4 at s0 (m = 0 and 1 fail curvature) and m = 16 at s1 (m = 0, 1 and 4 fail curvature). This matches λ\* as banked.
* **Every number agrees with the read to the bootstrap draw.**

**What the result is, and what it is not:**
* **The claim.** At both seeds, v8F's own output (M2b), blended with HACC's 30 × 2 train-fitted weights, is separated-better than its kinematic prior P10-yaw-4512 on every row of HSPEC's dominance row except distance keeping. That covers ADE2, ADE6, accel, curvature, speed, along, cross-track, heading, yaw rate, both plan-derived decision agreements and CONSISTENCY. Distance keeping (headway, time gap) is not separated, and LEAD-VIOL is separated-better at s0 and not separated at s1. Nothing is separated-worse.
* ⚠ **The tier is T1-class open loop:** the model's own held outputs, with no recorded future action and no re-perception. It is **not** a closed-loop driving result. The committed next tier is the T1-kin read, which needs a PI go.
* ⚠ **The baseline is the programme's kinematic floor, not the state of the art.**
* ⚠ **The KEEP margin at s1 is thin:** the upper bound +0.0581 (mine) / +0.0586 (the read) against +0.0604. HACC is separated-worse than H on ADE6 at s1, by +0.031 m. It passes a pre-registered F/2 bar, and that is how it is reported.
* ⚠ **One estimator question is answered:** another draw of episodes. Training-seed variance enters through the two seeds, both of which pass. Inference is deterministic.
