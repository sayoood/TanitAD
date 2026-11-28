# COMBO_ENS_RETIME -- step30000ema_unc / gains: eval

Tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  Estimator: paired episode-cluster bootstrap, models FIXED, B 2000, seed 0: answers 'another draw of EPISODES' only; the disjoint draw sets A/B are the inference-run replicate; NOT another training run (single training seed).
Status: EXPLORATORY (eval-fitted gains, post-hoc arm, milestone checkpoint); bar of record = single-draw `os`.
Bar of record: single-draw `os` (BAR-R8-E1, vs `ha0_ext_pose`, D-R8-E1-REF).  A combo is a NEW arm (inference cost up to xK).

prereg sha256 `a888e7cc99c3346dcbe71244f11c1d6f6ce654bccec9aeb7ec764b7050701287`; script sha256 `83ff0a56c162`; gains `C:\Users\Admin\qland\hold_r8batt\stage\FlyWheels\TanitAD_EvalFlyWheel\incoming\2026-10-15-refcv8-milestone-battery\raw\a0k0_retime\FROZEN_GAINS_RETIME_ema30k.json` sha256 `4ecde275554b`.

## OUTCOME (K=4, sets A and B): **STACKS**

* qualifier: EXPLORATORY (eval-fitted gains, post-hoc arm, milestone checkpoint); bar of record = single-draw `os`
* qualifier: EVAL-FITTED GAINS (in-sample on the eval episodes; EXPLORATORY ONLY)
* qualifier: tree step30000ema_unc
* qualifier: single training seed
* qualifier: S6 K=4 UNTESTED (S6 draws exist for inference seeds 0,1 only)
* qualifier: transfer unmodelled: gains fitted on single-draw plans

## Controls

| control | PASS |
|---|---|
| C-MASK | True |
| C-ALIGN | True |
| C-ID | True |
| C-DIST | True |
| C-R0926 | True |
| C-G1 | True |
| C-G0 | True |
| C-K1 | True |
| C-K1R | True |
| C-ONPATH | True |
| C-TRI | True |
| C-ENS | True |

Echo `ha0_ext` mean ADE (compliant) 0.2807 m; pose `ha0_ext_pose` 0.2858 m.

## E1 cells vs `ha0_ext_pose` (PRIMARY, D-R8-E1-REF) and vs `ha0_ext` (beside), ADE 0-2 s, compliant S2

| set | seeds | arm | vs pose | vs echo | mean ADE |
|---|---|---|---|---|---|
| 4A | [0, 1, 2, 3] | ENS-K | -0.0225 [-0.0373, -0.0071] | -0.0174 [-0.0314, -0.0021] | 0.2633 |
| 4A | [0, 1, 2, 3] | COMBO-K | -0.0353 [-0.0458, -0.0244] | -0.0302 [-0.0405, -0.0194] | 0.2505 |
| 4A | [0, 1, 2, 3] | COMBO-G-K (beside) | -0.0400 [-0.0506, -0.0289] | -0.0348 [-0.0452, -0.0239] | 0.2459 |
| 4A | [0, 1, 2, 3] | RT-THEN-ENS-K (beside) | -0.0353 [-0.0458, -0.0244] | -0.0302 [-0.0405, -0.0194] | 0.2505 |
| 4A | [0, 1, 2, 3] | RET-alone (mean of 4 single draws) | -0.0322 [-0.0428, -0.0215] | -0.0271 [-0.0373, -0.0165] | 0.2536 |
| 4B | [4, 5, 6, 7] | ENS-K | -0.0210 [-0.0363, -0.0056] | -0.0158 [-0.0304, -0.0006] | 0.2648 |
| 4B | [4, 5, 6, 7] | COMBO-K | -0.0348 [-0.0454, -0.0240] | -0.0296 [-0.0401, -0.0189] | 0.2511 |
| 4B | [4, 5, 6, 7] | COMBO-G-K (beside) | -0.0395 [-0.0507, -0.0285] | -0.0344 [-0.0451, -0.0233] | 0.2463 |
| 4B | [4, 5, 6, 7] | RT-THEN-ENS-K (beside) | -0.0347 [-0.0454, -0.0240] | -0.0296 [-0.0401, -0.0188] | 0.2511 |
| 4B | [4, 5, 6, 7] | RET-alone (mean of 4 single draws) | -0.0318 [-0.0424, -0.0210] | -0.0266 [-0.0371, -0.0159] | 0.2541 |
| 2A | [0, 1] | ENS-K | -0.0189 [-0.0335, -0.0035] | -0.0138 [-0.0279, +0.0012] | 0.2669 |
| 2A | [0, 1] | COMBO-K | -0.0338 [-0.0445, -0.0230] | -0.0287 [-0.0389, -0.0180] | 0.2520 |
| 2A | [0, 1] | COMBO-G-K (beside) | -0.0384 [-0.0491, -0.0275] | -0.0333 [-0.0436, -0.0224] | 0.2474 |
| 2A | [0, 1] | RT-THEN-ENS-K (beside) | -0.0338 [-0.0445, -0.0230] | -0.0287 [-0.0389, -0.0180] | 0.2520 |
| 2A | [0, 1] | RET-alone (mean of 2 single draws) | -0.0318 [-0.0425, -0.0210] | -0.0266 [-0.0369, -0.0160] | 0.2540 |
| 2B | [4, 5] | ENS-K | -0.0179 [-0.0333, -0.0026] | -0.0128 [-0.0275, +0.0026] | 0.2679 |
| 2B | [4, 5] | COMBO-K | -0.0335 [-0.0443, -0.0227] | -0.0284 [-0.0389, -0.0175] | 0.2523 |
| 2B | [4, 5] | COMBO-G-K (beside) | -0.0383 [-0.0496, -0.0272] | -0.0331 [-0.0439, -0.0221] | 0.2476 |
| 2B | [4, 5] | RT-THEN-ENS-K (beside) | -0.0335 [-0.0443, -0.0227] | -0.0284 [-0.0389, -0.0175] | 0.2523 |
| 2B | [4, 5] | RET-alone (mean of 2 single draws) | -0.0314 [-0.0423, -0.0206] | -0.0263 [-0.0368, -0.0156] | 0.2544 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | ENS-K | -0.0232 [-0.0381, -0.0076] | -0.0180 [-0.0323, -0.0027] | 0.2627 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | COMBO-K | -0.0356 [-0.0462, -0.0247] | -0.0304 [-0.0408, -0.0197] | 0.2503 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | COMBO-G-K (beside) | -0.0403 [-0.0513, -0.0292] | -0.0352 [-0.0457, -0.0242] | 0.2455 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | RT-THEN-ENS-K (beside) | -0.0355 [-0.0462, -0.0247] | -0.0304 [-0.0408, -0.0197] | 0.2503 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | RET-alone (mean of 8 single draws) | -0.0320 [-0.0426, -0.0212] | -0.0269 [-0.0373, -0.0162] | 0.2538 |

## STACKING (paired, ADE 0-2 s, compliant S2)

| set | COMBO - ENS-K (adds RETIME) | COMBO - RET-alone (adds ENS) | COMBO - best single (which) | COMBO - RT-THEN-ENS (transfer) |
|---|---|---|---|---|
| 4A | -0.0128 [-0.0197, -0.0064] | -0.0030 [-0.0033, -0.0028] | -0.0030 [-0.0033, -0.0028] (RET_alone) | +0.0000 [-0.0000, +0.0000] |
| 4B | -0.0138 [-0.0208, -0.0074] | -0.0030 [-0.0032, -0.0028] | -0.0030 [-0.0032, -0.0028] (RET_alone) | -0.0000 [-0.0000, +0.0000] |
| 2A | -0.0149 [-0.0218, -0.0086] | -0.0020 [-0.0022, -0.0019] | -0.0020 [-0.0022, -0.0019] (RET_alone) | +0.0000 [-0.0000, +0.0000] |
| 2B | -0.0155 [-0.0225, -0.0090] | -0.0020 [-0.0022, -0.0019] | -0.0020 [-0.0022, -0.0019] (RET_alone) | +0.0000 [-0.0000, +0.0000] |
| 8 | -0.0124 [-0.0195, -0.0060] | -0.0035 [-0.0038, -0.0033] | -0.0035 [-0.0038, -0.0033] (RET_alone) | -0.0000 [-0.0000, +0.0000] |

## ADDITIVITY (mean ADE, compliant S2; negative = better than the single-draw `os` of the same draws)

| set | single os | dENS | dRET | additive combo | observed combo | share of additive | ENS on top of RET (m, share of dENS) | RET on top of ENS (m, share of dRET) |
|---|---|---|---|---|---|---|---|---|
| 4A | 0.2719 | -0.0086 | -0.0183 | 0.2450 | 0.2505 | 0.79 | -0.0030 (0.35) | -0.0128 (0.70) |
| 4B | 0.2731 | -0.0083 | -0.0191 | 0.2458 | 0.2511 | 0.81 | -0.0030 (0.36) | -0.0138 (0.72) |
| 2A | 0.2726 | -0.0057 | -0.0186 | 0.2484 | 0.2520 | 0.85 | -0.0020 (0.36) | -0.0149 (0.80) |
| 2B | 0.2732 | -0.0054 | -0.0189 | 0.2490 | 0.2523 | 0.86 | -0.0020 (0.38) | -0.0155 (0.82) |
| 8 | 0.2725 | -0.0098 | -0.0187 | 0.2440 | 0.2503 | 0.78 | -0.0035 (0.36) | -0.0124 (0.66) |

K=4 floors (m): {'COMBO': 0.0005325973033905029, 'ENS': 0.0015491843223571777, 'RET_alone': 0.000481386805619044}.  Stacking verdict K=4: {"ST_ens": {"stacks": true, "worse_on_any_set": false, "separated_better_by_set": [true, true], "beyond_2x_floor_by_set": [true, true], "two_x_floor_m": 0.0030983686447143555}, "ST_ret": {"stacks": true, "worse_on_any_set": false, "separated_better_by_set": [true, true], "beyond_2x_floor_by_set": [true, true], "two_x_floor_m": 0.0010651946067810059}}
E1 K=4 vs pose: PASS ; vs echo: PASS
K=2 descriptive label: STACKING CELLS STACK (K=2 descriptive; NR is not evaluated at K=2, so no STACKS label is issued) ([])

## ACC-LAT

* S2 K=4 (COMBO - ENS-K, five LATERAL rows, sets A and B): {'HOLDS': True, 'moved': True, 'separated_worse_rows': [], 'undetermined': []}
* S6 K=2 draws {0,1} (n 3251 w / 138 ep): {'HOLDS': True, 'moved': True, 'separated_worse_rows': [], 'undetermined': []}; ADE combo - ens -0.2194 [-0.3172, -0.1189]; combo - pose -1.0254 [-1.3032, -0.7762]
  * LAT_heading_mae_deg: +0.0076 [-0.0871, +0.1351]
  * LAT_curvature_mae_1pm_valid: +0.0000 [-0.0001, +0.0001]
  * LAT_yaw_rate_mae_radps_valid: -0.0004 [-0.0010, +0.0002]
  * LAT_cross_mae_m: -0.0136 [-0.0485, +0.0212]
  * LAT_cross_abs_final_m: -0.0381 [-0.1376, +0.0637]

## NR vs refcv7-50,400 (S2 headline rows; PASS = not separated worse at BOTH refcv7 draws)

* COMBO-4A: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)
* ENS-4A: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)
* COMBO-4B: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)
* ENS-4B: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)

## Diagnostics

* 4A: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.110, on-path max 7.11e-15 m
* 4B: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.108, on-path max 7.11e-15 m
* 2A: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.110, on-path max 7.11e-15 m
* 2B: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.001, extrapolated 0.110, on-path max 7.11e-15 m
* 8: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.109, on-path max 7.11e-15 m
* mode disagreement 4A: {'frac_spread_gt_1m_compliant': 0.017311853811012263, 'n_disagree_compliant': 72, 'combo_minus_ens_curvature_on_disagree': {'delta': 0.0002, 'lo': -0.0006, 'hi': 0.0014, 'ci95': 0.001, 'p_delta_gt0': 0.5875, 'separated': False, 'reducer': 'mean', 'n_windows': 60, 'n_episodes': 41, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'n_dropped_nonfinite': 12, 'mean_new': 0.012424036905940739, 'mean_ref': 0.012228696063903043}, 'combo_minus_ens_ade_on_disagree': {'delta': 0.0101, 'lo': -0.0331, 'hi': 0.0579, 'ci95': 0.0455, 'p_delta_gt0': 0.6845, 'separated': False, 'reducer': 'mean', 'n_windows': 72, 'n_episodes': 45, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'n_dropped_nonfinite': 0, 'mean_new': 0.4613624363620248, 'mean_ref': 0.45130950569485623}}
* mode disagreement 4B: {'frac_spread_gt_1m_compliant': 0.016109641740803076, 'n_disagree_compliant': 67, 'combo_minus_ens_curvature_on_disagree': {'delta': 0.0002, 'lo': -0.0004, 'hi': 0.0011, 'ci95': 0.0008, 'p_delta_gt0': 0.603, 'separated': False, 'reducer': 'mean', 'n_windows': 59, 'n_episodes': 42, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'n_dropped_nonfinite': 8, 'mean_new': 0.012052553473595408, 'mean_ref': 0.011880465066250338}, 'combo_minus_ens_ade_on_disagree': {'delta': -0.0106, 'lo': -0.0594, 'hi': 0.0411, 'ci95': 0.0503, 'p_delta_gt0': 0.331, 'separated': False, 'reducer': 'mean', 'n_windows': 67, 'n_episodes': 45, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'n_dropped_nonfinite': 0, 'mean_new': 0.43841972513430155, 'mean_ref': 0.4490508632873421}}
