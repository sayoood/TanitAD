# COMBO_ENS_RETIME -- step48852ema_unc / gains: train

Tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  Estimator: paired episode-cluster bootstrap, models FIXED, B 2000, seed 0: answers 'another draw of EPISODES' only; the disjoint draw sets A/B are the inference-run replicate; NOT another training run (single training seed).
Status: REGISTERED READ (final checkpoint, train-fitted gains).
Bar of record: single-draw `os` (BAR-R8-E1, vs `ha0_ext_pose`, D-R8-E1-REF).  A combo is a NEW arm (inference cost up to xK).

prereg sha256 `a888e7cc99c3346dcbe71244f11c1d6f6ce654bccec9aeb7ec764b7050701287`; script sha256 `83ff0a56c162`; gains `C:\Users\Admin\qland\hold_r8batt\stage\FlyWheels\TanitAD_EvalFlyWheel\incoming\2026-10-15-refcv8-milestone-battery\raw\retime_trainfit\FROZEN_GAINS_RETIME_train.json` sha256 `b208c9c1e423`.

## OUTCOME (K=4, sets A and B): **VOID (controls failed: C-K1)**

* qualifier: REGISTERED READ (final checkpoint, train-fitted gains)
* qualifier: CLEAN GAINS (train-fitted)
* qualifier: tree step48852ema_unc
* qualifier: single training seed
* qualifier: S6 K=4 UNTESTED (S6 draws exist for inference seeds 0,1 only)
* qualifier: transfer unmodelled: gains fitted on single-draw plans; fitted on EMA-30k plans, applied to step48852ema_unc
* **VOID: controls failed -- ['C-K1']** (would have been {'label': 'STACKS', 'reasons': []})

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
| C-K1 | False |
| C-K1R | None (N/A: literals exist only for the EMA-30k tree with the eval-fitted gains file) |
| C-ONPATH | True |
| C-TRI | True |
| C-ENS | True |

Echo `ha0_ext` mean ADE (compliant) 0.2807 m; pose `ha0_ext_pose` 0.2858 m.

## E1 cells vs `ha0_ext_pose` (PRIMARY, D-R8-E1-REF) and vs `ha0_ext` (beside), ADE 0-2 s, compliant S2

| set | seeds | arm | vs pose | vs echo | mean ADE |
|---|---|---|---|---|---|
| 4A | [0, 1, 2, 3] | ENS-K | -0.0035 [-0.0210, +0.0147] | +0.0017 [-0.0151, +0.0195] | 0.2824 |
| 4A | [0, 1, 2, 3] | COMBO-K | -0.0307 [-0.0441, -0.0179] | -0.0256 [-0.0383, -0.0131] | 0.2551 |
| 4A | [0, 1, 2, 3] | COMBO-G-K (beside) | -0.0301 [-0.0437, -0.0163] | -0.0250 [-0.0382, -0.0113] | 0.2557 |
| 4A | [0, 1, 2, 3] | RT-THEN-ENS-K (beside) | -0.0307 [-0.0440, -0.0179] | -0.0256 [-0.0383, -0.0131] | 0.2551 |
| 4A | [0, 1, 2, 3] | RET-alone (mean of 4 single draws) | -0.0260 [-0.0394, -0.0132] | -0.0209 [-0.0334, -0.0084] | 0.2598 |
| 4B | [4, 5, 6, 7] | ENS-K | -0.0028 [-0.0205, +0.0153] | +0.0024 [-0.0148, +0.0205] | 0.2831 |
| 4B | [4, 5, 6, 7] | COMBO-K | -0.0303 [-0.0435, -0.0174] | -0.0251 [-0.0377, -0.0126] | 0.2556 |
| 4B | [4, 5, 6, 7] | COMBO-G-K (beside) | -0.0299 [-0.0434, -0.0163] | -0.0247 [-0.0378, -0.0113] | 0.2560 |
| 4B | [4, 5, 6, 7] | RT-THEN-ENS-K (beside) | -0.0303 [-0.0436, -0.0175] | -0.0251 [-0.0377, -0.0125] | 0.2556 |
| 4B | [4, 5, 6, 7] | RET-alone (mean of 4 single draws) | -0.0254 [-0.0388, -0.0124] | -0.0202 [-0.0331, -0.0076] | 0.2605 |
| 2A | [0, 1] | ENS-K | -0.0007 [-0.0178, +0.0171] | +0.0044 [-0.0122, +0.0219] | 0.2851 |
| 2A | [0, 1] | COMBO-K | -0.0292 [-0.0423, -0.0165] | -0.0241 [-0.0366, -0.0119] | 0.2566 |
| 2A | [0, 1] | COMBO-G-K (beside) | -0.0287 [-0.0420, -0.0150] | -0.0236 [-0.0365, -0.0100] | 0.2571 |
| 2A | [0, 1] | RT-THEN-ENS-K (beside) | -0.0292 [-0.0423, -0.0164] | -0.0240 [-0.0366, -0.0118] | 0.2567 |
| 2A | [0, 1] | RET-alone (mean of 2 single draws) | -0.0260 [-0.0390, -0.0132] | -0.0208 [-0.0333, -0.0084] | 0.2599 |
| 2B | [4, 5] | ENS-K | +0.0011 [-0.0160, +0.0190] | +0.0062 [-0.0106, +0.0244] | 0.2869 |
| 2B | [4, 5] | COMBO-K | -0.0283 [-0.0414, -0.0156] | -0.0232 [-0.0357, -0.0105] | 0.2575 |
| 2B | [4, 5] | COMBO-G-K (beside) | -0.0281 [-0.0414, -0.0147] | -0.0230 [-0.0361, -0.0097] | 0.2577 |
| 2B | [4, 5] | RT-THEN-ENS-K (beside) | -0.0283 [-0.0414, -0.0155] | -0.0232 [-0.0356, -0.0105] | 0.2575 |
| 2B | [4, 5] | RET-alone (mean of 2 single draws) | -0.0250 [-0.0383, -0.0120] | -0.0199 [-0.0324, -0.0073] | 0.2608 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | ENS-K | -0.0048 [-0.0225, +0.0134] | +0.0004 [-0.0166, +0.0184] | 0.2811 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | COMBO-K | -0.0314 [-0.0446, -0.0188] | -0.0263 [-0.0389, -0.0137] | 0.2544 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | COMBO-G-K (beside) | -0.0309 [-0.0444, -0.0172] | -0.0257 [-0.0390, -0.0121] | 0.2550 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | RT-THEN-ENS-K (beside) | -0.0313 [-0.0446, -0.0188] | -0.0262 [-0.0388, -0.0137] | 0.2545 |
| 8 | [0, 1, 2, 3, 4, 5, 6, 7] | RET-alone (mean of 8 single draws) | -0.0257 [-0.0391, -0.0128] | -0.0205 [-0.0333, -0.0080] | 0.2601 |

## STACKING (paired, ADE 0-2 s, compliant S2)

| set | COMBO - ENS-K (adds RETIME) | COMBO - RET-alone (adds ENS) | COMBO - best single (which) | COMBO - RT-THEN-ENS (transfer) |
|---|---|---|---|---|
| 4A | -0.0273 [-0.0345, -0.0206] | -0.0047 [-0.0053, -0.0043] | -0.0047 [-0.0053, -0.0043] (RET_alone) | -0.0000 [-0.0001, +0.0000] |
| 4B | -0.0275 [-0.0345, -0.0210] | -0.0049 [-0.0056, -0.0043] | -0.0049 [-0.0056, -0.0043] (RET_alone) | -0.0000 [-0.0000, +0.0000] |
| 2A | -0.0285 [-0.0355, -0.0219] | -0.0032 [-0.0036, -0.0029] | -0.0032 [-0.0036, -0.0029] (RET_alone) | -0.0000 [-0.0001, +0.0000] |
| 2B | -0.0294 [-0.0364, -0.0229] | -0.0033 [-0.0039, -0.0028] | -0.0033 [-0.0039, -0.0028] (RET_alone) | -0.0000 [-0.0000, +0.0000] |
| 8 | -0.0266 [-0.0338, -0.0199] | -0.0057 [-0.0064, -0.0051] | -0.0057 [-0.0064, -0.0051] (RET_alone) | -0.0000 [-0.0001, +0.0000] |

## ADDITIVITY (mean ADE, compliant S2; negative = better than the single-draw `os` of the same draws)

| set | single os | dENS | dRET | additive combo | observed combo | share of additive | ENS on top of RET (m, share of dENS) | RET on top of ENS (m, share of dRET) |
|---|---|---|---|---|---|---|---|---|
| 4A | 0.2912 | -0.0088 | -0.0313 | 0.2510 | 0.2551 | 0.90 | -0.0047 (0.54) | -0.0273 (0.87) |
| 4B | 0.2922 | -0.0092 | -0.0318 | 0.2513 | 0.2556 | 0.90 | -0.0049 (0.54) | -0.0275 (0.87) |
| 2A | 0.2910 | -0.0059 | -0.0312 | 0.2540 | 0.2566 | 0.93 | -0.0032 (0.55) | -0.0285 (0.91) |
| 2B | 0.2930 | -0.0061 | -0.0322 | 0.2547 | 0.2575 | 0.93 | -0.0033 (0.54) | -0.0294 (0.91) |
| 8 | 0.2917 | -0.0107 | -0.0316 | 0.2495 | 0.2544 | 0.88 | -0.0057 (0.54) | -0.0266 (0.84) |

K=4 floors (m): {'COMBO': 0.000457763671875, 'ENS': 0.0007097423076629639, 'RET_alone': 0.0006218673135886532}.  Stacking verdict K=4: {"ST_ens": {"stacks": true, "worse_on_any_set": false, "separated_better_by_set": [true, true], "beyond_2x_floor_by_set": [true, true], "two_x_floor_m": 0.0014194846153259277}, "ST_ret": {"stacks": true, "worse_on_any_set": false, "separated_better_by_set": [true, true], "beyond_2x_floor_by_set": [true, true], "two_x_floor_m": 0.0012437346271773064}}
E1 K=4 vs pose: PASS ; vs echo: PASS
K=2 descriptive label: None (None)

## ACC-LAT

* S2 K=4 (COMBO - ENS-K, five LATERAL rows, sets A and B): {'HOLDS': True, 'moved': True, 'separated_worse_rows': [], 'undetermined': []}
* S6 K=2 draws {0,1} (n 3251 w / 138 ep): {'HOLDS': True, 'moved': True, 'separated_worse_rows': [], 'undetermined': []}; ADE combo - ens -0.3244 [-0.3945, -0.2550]; combo - pose -0.8686 [-1.1904, -0.5758]
  * LAT_heading_mae_deg: -0.0618 [-0.1160, -0.0136]
  * LAT_curvature_mae_1pm_valid: -0.0000 [-0.0001, +0.0000]
  * LAT_yaw_rate_mae_radps_valid: -0.0007 [-0.0012, -0.0003]
  * LAT_cross_mae_m: -0.0397 [-0.0658, -0.0145]
  * LAT_cross_abs_final_m: -0.1098 [-0.1804, -0.0449]

## NR vs refcv7-50,400 (S2 headline rows; PASS = not separated worse at BOTH refcv7 draws)

* COMBO-4A: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)
* ENS-4A: {'ADE': 'PASS', 'LONGITUDINAL': 'FAIL: LON_speed_mae_mps', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'FAIL: LON_speed_mae_mps, LON_along_abs_final_m', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)
* COMBO-4B: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)
* ENS-4B: {'ADE': 'PASS', 'LONGITUDINAL': 'FAIL: LON_speed_mae_mps, LON_along_abs_final_m', 'LATERAL': 'PASS'}   (vs refcv7 2-draw mean: {'ADE': 'PASS', 'LONGITUDINAL': 'FAIL: LON_speed_mae_mps, LON_along_abs_final_m', 'LATERAL': 'PASS'})   (STRATEGIC: NOT APPLICABLE, n = 0)

## Diagnostics

* 4A: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.001, extrapolated 0.107, on-path max 7.11e-15 m
* 4B: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.108, on-path max 7.11e-15 m
* 2A: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.001, extrapolated 0.107, on-path max 7.11e-15 m
* 2B: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.108, on-path max 7.11e-15 m
* 8: arc-length shrinkage of the averaged plan 1.0000; combo geometry nonmonotone 0.000, extrapolated 0.107, on-path max 7.11e-15 m
* mode disagreement 4A: {'frac_spread_gt_1m_compliant': 0.05001202212070209, 'n_disagree_compliant': 208, 'combo_minus_ens_curvature_on_disagree': {'delta': -0.00026, 'lo': -0.00062, 'hi': -1e-05, 'ci95': 0.00031, 'p_delta_gt0': 0.019, 'separated': True, 'reducer': 'mean', 'n_windows': 199, 'n_episodes': 64, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'display_dp': 5, 'display_note': 'rendered at 5 dp, not the usual 4: at 4 dp the printed interval would have contradicted separated=True. The extra digits are a RENDERING fix, not extra precision — the estimator and the separation test are unchanged.', 'n_dropped_nonfinite': 9, 'mean_new': 0.006751632317717845, 'mean_ref': 0.007012723786911422}, 'combo_minus_ens_ade_on_disagree': {'delta': -0.109, 'lo': -0.1498, 'hi': -0.0674, 'ci95': 0.0412, 'p_delta_gt0': 0.0, 'separated': True, 'reducer': 'mean', 'n_windows': 208, 'n_episodes': 67, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'n_dropped_nonfinite': 0, 'mean_new': 0.4521611905298554, 'mean_ref': 0.5611512970465881}}
* mode disagreement 4B: {'frac_spread_gt_1m_compliant': 0.049771579706660254, 'n_disagree_compliant': 207, 'combo_minus_ens_curvature_on_disagree': {'delta': -0.0001, 'lo': -0.0004, 'hi': 0.0003, 'ci95': 0.0003, 'p_delta_gt0': 0.3215, 'separated': False, 'reducer': 'mean', 'n_windows': 194, 'n_episodes': 65, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'n_dropped_nonfinite': 13, 'mean_new': 0.007451235235308071, 'mean_ref': 0.00751420749326929}, 'combo_minus_ens_ade_on_disagree': {'delta': -0.1127, 'lo': -0.1441, 'hi': -0.0798, 'ci95': 0.0321, 'p_delta_gt0': 0.0, 'separated': True, 'reducer': 'mean', 'n_windows': 207, 'n_episodes': 68, 'n_boot': 2000, 'estimator': 'paired_episode_cluster_bootstrap', 'n_dropped_nonfinite': 0, 'mean_new': 0.45746239011990275, 'mean_ref': 0.5701995112253848}}
