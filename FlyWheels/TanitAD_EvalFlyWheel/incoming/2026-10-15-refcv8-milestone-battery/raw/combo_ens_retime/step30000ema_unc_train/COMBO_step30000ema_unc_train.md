# COMBO_ENS_RETIME -- step30000ema_unc / gains: train

Tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  Estimator: paired episode-cluster bootstrap, models FIXED, B 2000, seed 0: answers 'another draw of EPISODES' only; the disjoint draw sets A/B are the inference-run replicate; NOT another training run (single training seed).
Status: CLEAN-GAINS read of a milestone checkpoint (not the registered read).
Bar of record: single-draw `os` (BAR-R8-E1, vs `ha0_ext_pose`, D-R8-E1-REF).  A combo is a NEW arm (inference cost up to xK).

prereg sha256 `a888e7cc99c3346dcbe71244f11c1d6f6ce654bccec9aeb7ec764b7050701287`; script sha256 `83ff0a56c162`; gains `C:\Users\Admin\qland\hold_r8batt\stage\FlyWheels\TanitAD_EvalFlyWheel\incoming\2026-10-15-refcv8-milestone-battery\raw\retime_trainfit\FROZEN_GAINS_RETIME_train.json` sha256 `b208c9c1e423`.

## OUTCOME (K=4, sets A and B): **STACKS**

* qualifier: CLEAN-GAINS read of a milestone checkpoint (not the registered read)
* qualifier: CLEAN GAINS (train-fitted)
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
| C-K1R | None (N/A: literals exist only for the EMA-30k tree with the eval-fitted gains file) |
| C-ONPATH | True |
| C-TRI | True |
| C-ENS | True |

Echo `ha0_ext` mean ADE (compliant) 0.2807 m; pose `ha0_ext_pose` 0.2858 m.

## E1 cells vs `ha0_ext_pose` (PRIMARY, D-R8-E1-REF) and vs `ha0_ext` (beside), ADE 0-2 s, compliant S2

| set | seeds | arm | vs pose | vs echo | mean ADE |
|---|---|---|---|---|---|
| 4A | [0, 1, 2, 3] | ENS-K | -0.0225 [-0.0373, -0.0071] | -0.0174 [-0.0314, -0.0021] | 0.2633 |
| 4A | [0, 1, 2, 3] | COMBO-K | -0.0361 [-0.0476, -0.0241] | -0.0309 [-0.0421, -0.0192] | 0.2498 |
| 4A | [0, 1, 2, 3] | COMBO-G-K (beside) | -0.0389 [-0.0505, -0.0271] | -0.0338 [-0.0447, -0.0223] | 0.2469 |
| 4A | [0, 1, 2, 3] | RT-THEN-ENS-K (beside) | -0.0361 [-0.0476, -0.0241] | -0.0309 [-0.0421, -0.0192] | 0.2498 |
| 4A | [0, 1, 2, 3] | RET-alone (mean of 4 single draws) | -0.0322 [-0.0437, -0.0203] | -0.0270 [-0.0380, -0.0154] | 0.2537 |
| 4B | [4, 5, 6, 7] | ENS-K | -0.0210 [-0.0363, -0.0056] | -0.0158 [-0.0304, -0.0006] | 0.2648 |
| 4B | [4, 5, 6, 7] | COMBO-K | -0.0353 [-0.0470, -0.0234] | -0.0302 [-0.0415, -0.0183] | 0.2505 |
| 4B | [4, 5, 6, 7] | COMBO-G-K (beside) | -0.0383 [-0.0500, -0.0263] | -0.0331 [-0.0445, -0.0215] | 0.2476 |
| 4B | [4, 5, 6, 7] | RT-THEN-ENS-K (beside) | -0.0353 [-0.0470, -0.0234] | -0.0302 [-0.0415, -0.0183] | 0.2505 |
| 4B | [4, 5, 6, 7] | RET-alone (mean of 4 single draws) | -0.0315 [-0.0431, -0.0197] | -0.0264 [-0.0377, -0.0146] | 0.2543 |
| 2A | [0, 1] | ENS-K | -0.0189 [-0.0335, -0.0035] | -0.0138 [-0.0279, +0.0012] | 0.2669 |
| 2A | [0, 1] | COMBO-K | -0.0341 [-0.0455, -0.0222] | -0.0290 [-0.0399, -0.0173] | 0.2517 |
| 2A | [0, 1] | COMBO-G-K (beside) | -0.0371 [-0.0485, -0.0254] | -0.0320 [-0.0428, -0.0205] | 0.2487 |
| 2A | [0, 1] | RT-THEN-ENS-K (beside) | -0.0342 [-0.0455, -0.0222] | -0.0290 [-0.0399, -0.0173] | 0.2517 |
| 2A | [0, 1] | RET-alone (mean of 2 single draws) | -0.0316 [-0.0429, -0.0197] | -0.0264 [-0.0373, -0.0148] | 0.2543 |
| 2B | [4, 5] | ENS-K | -0.0179 [-0.0333, -0.0026] | -0.0128 [-0.0275, +0.0026] | 0.2679 |
| 2B | [4, 5] | COMBO-K | -0.0337 [-0.0456, -0.0219] | -0.0286 [-0.0399, -0.0168] | 0.2521 |
| 2B | [4, 5] | COMBO-G-K (beside) | -0.0367 [-0.0485, -0.0248] | -0.0316 [-0.0429, -0.0200] | 0.2491 |
| 2B | [4, 5] | RT-THEN-ENS-K (beside) | -0.0337 [-0.0456, -0.0219] | -0.0286 [-0.0399, -0.0168] | 0.2521 |
| 2B | [4, 5] | RET-alone (mean of 2 single draws) | -0.0312 [-0.0431, -0.0194] | -0.0260 [-0.0373, -0.0142] | 0.2546 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | ENS-K | -0.0232 [-0.0381, -0.0076] | -0.0180 [-0.0323, -0.0027] | 0.2627 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | COMBO-K | -0.0363 [-0.0481, -0.0244] | -0.0312 [-0.0424, -0.0195] | 0.2495 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | COMBO-G-K (beside) | -0.0393 [-0.0510, -0.0274] | -0.0341 [-0.0453, -0.0226] | 0.2466 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | RT-THEN-ENS-K (beside) | -0.0363 [-0.0480, -0.0244] | -0.0312 [-0.0424, -0.0195] | 0.2495 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | RET-alone (mean of 8 single draws) | -0.0318 [-0.0434, -0.0200] | -0.0267 [-0.0379, -0.0150] | 0.2540 |

## STACKING (paired, ADE 0-2 s, compliant S2)

| set | COMBO - ENS-K (adds RETIME) | COMBO - RET-alone (adds ENS) | COMBO - best single (which) | COMBO - RT-THEN-ENS (transfer) |
|---|---|---|---|---|
| 4A | -0.0135 [-0.0190, -0.0086] | -0.0039 [-0.0042, -0.0036] | -0.0039 [-0.0042, -0.0036] (RET_alone) | +0.0000 [-0.0000, +0.0000] |
| 4B | -0.0144 [-0.0199, -0.0093] | -0.0038 [-0.0041, -0.0036] | -0.0038 [-0.0041, -0.0036] (RET_alone) | -0.0000 [-0.0000, +0.0000] |
| 2A | -0.0153 [-0.0206, -0.0103] | -0.0026 [-0.0028, -0.0023] | -0.0026 [-0.0028, -0.0023] (RET_alone) | +0.0000 [-0.0000, +0.0000] |
| 2B | -0.0158 [-0.0214, -0.0107] | -0.0026 [-0.0028, -0.0024] | -0.0026 [-0.0028, -0.0024] (RET_alone) | +0.0000 [-0.0000, +0.0000] |
| 8 | -0.0132 [-0.0187, -0.0082] | -0.0045 [-0.0048, -0.0042] | -0.0045 [-0.0048, -0.0042] (RET_alone) | -0.0000 [-0.0000, +0.0000] |

## ADDITIVITY (mean ADE, compliant S2; negative = better than the single-draw `os` of the same draws)

| set | single os | dENS | dRET | additive combo | observed combo | share of additive | ENS on top of RET (m, share of dENS) | RET on top of ENS (m, share of dRET) |
|---|---|---|---|---|---|---|---|---|
| 4A | 0.2719 | -0.0086 | -0.0182 | 0.2451 | 0.2498 | 0.83 | -0.0039 (0.45) | -0.0135 (0.74) |
| 4B | 0.2731 | -0.0083 | -0.0188 | 0.2460 | 0.2505 | 0.84 | -0.0038 (0.46) | -0.0144 (0.76) |
| 2A | 0.2726 | -0.0057 | -0.0184 | 0.2486 | 0.2517 | 0.87 | -0.0026 (0.45) | -0.0153 (0.83) |
| 2B | 0.2732 | -0.0054 | -0.0186 | 0.2493 | 0.2521 | 0.88 | -0.0026 (0.48) | -0.0158 (0.85) |
| 8 | 0.2725 | -0.0098 | -0.0185 | 0.2442 | 0.2495 | 0.81 | -0.0045 (0.46) | -0.0132 (0.71) |

K=4 floors (m): {'COMBO': 0.0007302165031433105, 'ENS': 0.0015491843223571777, 'RET_alone': 0.0006444106671336658}.  Stacking verdict K=4: {"ST_ens": {"stacks": true, "worse_on_any_set": false, "separated_better_by_set": [true, true], "beyond_2x_floor_by_set": [true, true], "two_x_floor_m": 0.0030983686447143555}, "ST_ret": {"stacks": true, "worse_on_any_set": false, "separated_better_by_set": [true, true], "beyond_2x_floor_by_set": [true, true], "two_x_floor_m": 0.001460433006286621}}
E1 K=4 vs pose: PASS ; vs echo: PASS
K=2 descriptive label: STACKING CELLS STACK (K=2 descriptive; NR is not evaluated at K=2, so no STACKS label is issued) ([])

## ACC-LAT

* S2 K=4 (COMBO - ENS-K, five LATERAL rows, sets A and B): {'HOLDS': True, 'moved': True, 'separated_worse_rows': [], 'undetermined': []}
* S6 K=2 draws {0,1} (n 3251 w / 138 ep): {'HOLDS': True, 'moved': True, 'separated_worse_rows': [], 'undetermined': []}; ADE combo - ens -0.1883 [-0.2478, -0.1289]; combo - pose -0.9942 [-1.2935, -0.7168]
  * LAT_heading_mae_deg: -0.0066 [-0.0582, +0.0616]
  * LAT_curvature_mae_1pm_valid: -0.0001 [-0.0001, -0.0000]
  * LAT_yaw_rate_mae_radps_valid: -0.0004 [-0.0008, -0.0001]
  * LAT_cross_mae_m: -0.0206 [-0.0415, +0.0000]
  * LAT_cross_abs_final_m: -0.0595 [-0.1165, -0.0029]

## NR vs refcv7-50,400 (S2 headline rows; PASS = not separated worse at BOTH refcv7 draws)

* COMBO-4A: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)
* ENS-4A: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)
* COMBO-4B: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)
* ENS-4B: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)

## Diagnostics

* 4A: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.110, on-path max 7.11e-15 m
* 4B: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.108, on-path max 7.11e-15 m
* 2A: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.110, on-path max 7.12e-15 m
* 2B: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.110, on-path max 7.11e-15 m
* 8: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.108, on-path max 7.11e-15 m
* mode disagreement 4A: {'frac_spread_gt_1m_compliant': 0.017311853811012263, 'n_disagree_compliant': 72, 'combo_minus_ens_curvature_on_disagree': {'delta': 0.0, 'lo': -0.0003, 'hi': 0.0004, 'ci95': 0.0003, 'p_delta_gt0': 0.468, 'separated': False, 'reducer': 'mean', 'n_windows': 60, 'n_episodes': 41, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'n_dropped_nonfinite': 12, 'mean_new': 0.012234543918990919, 'mean_ref': 0.012228696063903043}, 'combo_minus_ens_ade_on_disagree': {'delta': -0.0035, 'lo': -0.0385, 'hi': 0.0352, 'ci95': 0.0368, 'p_delta_gt0': 0.428, 'separated': False, 'reducer': 'mean', 'n_windows': 72, 'n_episodes': 45, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'n_dropped_nonfinite': 0, 'mean_new': 0.44776144380577737, 'mean_ref': 0.45130950569485623}}
* mode disagreement 4B: {'frac_spread_gt_1m_compliant': 0.016109641740803076, 'n_disagree_compliant': 67, 'combo_minus_ens_curvature_on_disagree': {'delta': 0.0, 'lo': -0.0002, 'hi': 0.0004, 'ci95': 0.0003, 'p_delta_gt0': 0.522, 'separated': False, 'reducer': 'mean', 'n_windows': 59, 'n_episodes': 42, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'n_dropped_nonfinite': 8, 'mean_new': 0.011911897946709483, 'mean_ref': 0.011880465066250338}, 'combo_minus_ens_ade_on_disagree': {'delta': -0.0224, 'lo': -0.0613, 'hi': 0.0175, 'ci95': 0.0394, 'p_delta_gt0': 0.1285, 'separated': False, 'reducer': 'mean', 'n_windows': 67, 'n_episodes': 45, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'n_dropped_nonfinite': 0, 'mean_new': 0.42668774223594524, 'mean_ref': 0.4490508632873421}}
