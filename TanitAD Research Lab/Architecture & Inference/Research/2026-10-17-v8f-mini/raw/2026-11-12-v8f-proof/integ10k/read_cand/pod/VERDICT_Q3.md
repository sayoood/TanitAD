# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 20}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RLcK"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RLcK minus I10R0G0X

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0130 / +0.0230 | +0.0180 | [-0.0205, +0.0603] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1011 / +0.1251 | +0.1131 | [-0.0020, +0.2548] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0162 / -0.0211 | -0.0186 | [-0.0325, -0.0072] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2046 / +0.2639 | +0.2342 | [+0.0699, +0.4280] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0042 / +0.0051 | +0.0046 | [+0.0007, +0.0095] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.2164 / +0.2184 | +0.2174 | [+0.0309, +0.4409] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0054 / -0.0083 | -0.0069 | [-0.0169, +0.0029] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0088 / -0.0075 | -0.0081 | [-0.0140, -0.0033] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0046 / -0.0079 | -0.0063 | [-0.0154, +0.0029] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.1066 / +0.1339 | +0.1203 | [-0.0308, +0.3048] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0089 / +0.0118 | +0.0014 | [-0.0937, +0.1177] | 0.0229 | OPEN |
| ade6_m|bulk | +0.2123 / +0.2271 | +0.2197 | [+0.0674, +0.3962] | 0.1243 | WORSE |
| ade6_m|all | +0.0920 / +0.1100 | +0.1010 | [-0.0139, +0.2402] | 0.0442 | OPEN |
| ade_tac_m|all | +0.1066 / +0.1339 | +0.1203 | [-0.0308, +0.3048] | 0.0699 | OPEN |

## I10RQ50K minus I10R0G0X

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0359 / +0.0256 | +0.0307 | [+0.0011, +0.0657] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.1766 / +0.1387 | +0.1576 | [+0.0616, +0.2694] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0198 / -0.0193 | -0.0196 | [-0.0313, -0.0098] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2977 / +0.1422 | +0.2199 | [+0.0429, +0.4794] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0064 / +0.0057 | +0.0061 | [+0.0026, +0.0106] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.3119 / +0.2186 | +0.2653 | [+0.0864, +0.5149] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0083 / -0.0050 | -0.0067 | [-0.0156, +0.0019] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0067 / -0.0067 | -0.0067 | [-0.0121, -0.0023] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0071 / -0.0042 | -0.0056 | [-0.0135, +0.0017] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.1965 / +0.1458 | +0.1711 | [+0.0499, +0.3105] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1121 / +0.0247 | +0.0684 | [-0.0210, +0.1847] | 0.0229 | OPEN |
| ade6_m|bulk | +0.2145 / +0.2305 | +0.2225 | [+0.0978, +0.3596] | 0.1243 | WORSE |
| ade6_m|all | +0.1588 / +0.1185 | +0.1387 | [+0.0460, +0.2472] | 0.0442 | WORSE |
| ade_tac_m|all | +0.1965 / +0.1458 | +0.1711 | [+0.0499, +0.3105] | 0.0699 | WORSE |

## I10RLcKF5 minus I10RLcK

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0099 / +0.0108 | +0.0103 | [+0.0020, +0.0206] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0243 / +0.0264 | +0.0254 | [+0.0070, +0.0467] | 0.0298 (0.0298) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0001 / -0.0002 | -0.0000 | [-0.0008, +0.0006] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.0001 / -0.0024 | -0.0013 | [-0.0111, +0.0085] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0002 / +0.0003 | +0.0003 | [-0.0002, +0.0010] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0052 / +0.0098 | +0.0075 | [-0.0135, +0.0408] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0029 / +0.0046 | +0.0038 | [+0.0000, +0.0088] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0008 / +0.0008 | +0.0008 | [-0.0004, +0.0021] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0029 / +0.0050 | +0.0040 | [-0.0000, +0.0094] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0369 / +0.0384 | +0.0376 | [+0.0115, +0.0681] | 0.0699 (0.0699) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0068 / +0.0069 | +0.0069 | [-0.0033, +0.0199] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0474 / +0.0497 | +0.0486 | [+0.0170, +0.0855] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | +0.0253 / +0.0264 | +0.0259 | [+0.0078, +0.0467] | 0.0442 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0369 / +0.0384 | +0.0376 | [+0.0115, +0.0681] | 0.0699 | INSIDE-SPREAD |

## I10RQ50KF5 minus I10RQ50K

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0034 / +0.0055 | +0.0044 | [-0.0035, +0.0132] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0289 / +0.0263 | +0.0276 | [+0.0052, +0.0509] | 0.0298 (0.0298) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0012 / -0.0004 | -0.0008 | [-0.0021, +0.0002] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.0130 / -0.0036 | -0.0083 | [-0.0624, +0.0395] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0024 / +0.0001 | +0.0013 | [-0.0001, +0.0038] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.1054 / -0.0016 | +0.0519 | [-0.0185, +0.1817] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0008 / -0.0025 | -0.0017 | [-0.0056, +0.0029] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0008 / +0.0000 | -0.0004 | [-0.0017, +0.0002] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0008 / -0.0008 | -0.0008 | [-0.0046, +0.0033] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0392 / +0.0367 | +0.0380 | [+0.0072, +0.0697] | 0.0699 (0.0699) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0003 / -0.0106 | -0.0054 | [-0.0251, +0.0167] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0628 / +0.0696 | +0.0662 | [+0.0337, +0.1036] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | +0.0285 / +0.0260 | +0.0272 | [+0.0055, +0.0496] | 0.0442 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0392 / +0.0367 | +0.0380 | [+0.0072, +0.0697] | 0.0699 | INSIDE-SPREAD |

## I10RLcKF5 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.6222 / -0.6479 | -0.6350 | [-0.7434, -0.5349] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.4828 / -0.4369 | -0.4599 | [-0.5844, -0.3550] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3577 / -0.2366 | -0.2971 | [-0.4230, -0.1773] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.8010 / -2.8526 | -2.8268 | [-3.3900, -2.3187] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0639 / -0.0811 | -0.0725 | [-0.0946, -0.0548] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -16.3883 / -16.9286 | -16.6585 | [-20.0783, -13.7654] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1251 / -0.1822 | -0.1536 | [-0.2063, -0.1024] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1809 / -0.1326 | -0.1567 | [-0.2023, -0.1167] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1638 / -0.2155 | -0.1897 | [-0.2400, -0.1428] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.9288 / -0.7549 | -0.8418 | [-1.1120, -0.6269] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2717 / -0.2127 | -0.2422 | [-0.3890, -0.1083] | 0.0229 | BETTER |
| ade6_m|bulk | -1.0321 / -0.8481 | -0.9401 | [-1.2192, -0.7178] | 0.1243 | BETTER |
| ade6_m|all | -0.6184 / -0.5024 | -0.5604 | [-0.7403, -0.4173] | 0.0442 | BETTER |
| ade_tac_m|all | -0.9288 / -0.7549 | -0.8418 | [-1.1120, -0.6269] | 0.0699 | BETTER |

## I10RQ50KF5 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.6095 / -0.6613 | -0.6354 | [-0.7494, -0.5291] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.4837 / -0.5413 | -0.5125 | [-0.6578, -0.3807] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3497 / -0.2360 | -0.2929 | [-0.4122, -0.1744] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.8319 / -3.1469 | -2.9894 | [-3.6532, -2.4417] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0615 / -0.0786 | -0.0701 | [-0.0918, -0.0526] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -16.1183 / -16.4905 | -16.3044 | [-19.2383, -13.8493] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1296 / -0.1830 | -0.1563 | [-0.2081, -0.1078] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1851 / -0.1421 | -0.1636 | [-0.2053, -0.1263] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1751 / -0.2163 | -0.1957 | [-0.2428, -0.1517] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.9336 / -0.9011 | -0.9174 | [-1.1544, -0.6986] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2646 / -0.2850 | -0.2748 | [-0.4182, -0.1425] | 0.0229 | BETTER |
| ade6_m|bulk | -1.0441 / -0.9741 | -1.0091 | [-1.2758, -0.7895] | 0.1243 | BETTER |
| ade6_m|all | -0.6201 / -0.5993 | -0.6097 | [-0.7666, -0.4642] | 0.0442 | BETTER |
| ade_tac_m|all | -0.9336 / -0.9011 | -0.9174 | [-1.1544, -0.6986] | 0.0699 | BETTER |

## I10RLcK minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.6321 / -0.6587 | -0.6454 | [-0.7572, -0.5416] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.5071 / -0.4634 | -0.4852 | [-0.6167, -0.3721] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3578 / -0.2363 | -0.2971 | [-0.4230, -0.1770] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.8009 / -2.8501 | -2.8255 | [-3.3895, -2.3138] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0641 / -0.0814 | -0.0728 | [-0.0949, -0.0549] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -16.3935 / -16.9384 | -16.6660 | [-20.0905, -13.7819] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1280 / -0.1867 | -0.1574 | [-0.2110, -0.1061] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1817 / -0.1334 | -0.1576 | [-0.2033, -0.1177] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1667 / -0.2205 | -0.1936 | [-0.2452, -0.1448] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.9656 / -0.7933 | -0.8795 | [-1.1439, -0.6533] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2785 / -0.2196 | -0.2490 | [-0.3957, -0.1099] | 0.0229 | BETTER |
| ade6_m|bulk | -1.0794 / -0.8978 | -0.9886 | [-1.2672, -0.7623] | 0.1243 | BETTER |
| ade6_m|all | -0.6437 / -0.5289 | -0.5863 | [-0.7626, -0.4355] | 0.0442 | BETTER |
| ade_tac_m|all | -0.9656 / -0.7933 | -0.8795 | [-1.1439, -0.6533] | 0.0699 | BETTER |

## I10RQ50K minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.6129 / -0.6668 | -0.6398 | [-0.7540, -0.5313] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.5126 / -0.5676 | -0.5401 | [-0.6959, -0.3984] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3485 / -0.2356 | -0.2920 | [-0.4106, -0.1742] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.8189 / -3.1432 | -2.9811 | [-3.6451, -2.4246] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0640 / -0.0787 | -0.0713 | [-0.0924, -0.0541] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -16.2237 / -16.4889 | -16.3563 | [-19.3298, -13.9087] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1288 / -0.1805 | -0.1546 | [-0.2065, -0.1060] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1842 / -0.1421 | -0.1632 | [-0.2047, -0.1260] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1742 / -0.2155 | -0.1949 | [-0.2421, -0.1508] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.9728 / -0.9379 | -0.9554 | [-1.2071, -0.7213] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2643 / -0.2744 | -0.2694 | [-0.4187, -0.1308] | 0.0229 | BETTER |
| ade6_m|bulk | -1.1069 / -1.0437 | -1.0753 | [-1.3554, -0.8422] | 0.1243 | BETTER |
| ade6_m|all | -0.6486 / -0.6252 | -0.6369 | [-0.8048, -0.4808] | 0.0442 | BETTER |
| ade_tac_m|all | -0.9728 / -0.9379 | -0.9554 | [-1.2071, -0.7213] | 0.0699 | BETTER |

## I10RLcKF5 minus I10RLcF5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.6350 / -0.6565 | -0.6458 | [-0.7538, -0.5450] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.5056 / -0.4610 | -0.4833 | [-0.6140, -0.3762] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3576 / -0.2379 | -0.2978 | [-0.4245, -0.1777] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.8488 / -2.8912 | -2.8700 | [-3.4349, -2.3501] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0674 / -0.0850 | -0.0762 | [-0.0991, -0.0573] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -16.6813 / -17.1927 | -16.9370 | [-20.3763, -14.0843] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1242 / -0.1822 | -0.1532 | [-0.2062, -0.1022] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1822 / -0.1355 | -0.1588 | [-0.2029, -0.1196] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1634 / -0.2172 | -0.1903 | [-0.2419, -0.1418] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.9586 / -0.7894 | -0.8740 | [-1.1423, -0.6579] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2847 / -0.2242 | -0.2545 | [-0.4026, -0.1148] | 0.0229 | BETTER |
| ade6_m|bulk | -1.0617 / -0.8866 | -0.9741 | [-1.2546, -0.7492] | 0.1243 | BETTER |
| ade6_m|all | -0.6390 / -0.5263 | -0.5827 | [-0.7615, -0.4386] | 0.0442 | BETTER |
| ade_tac_m|all | -0.9586 / -0.7894 | -0.8740 | [-1.1423, -0.6579] | 0.0699 | BETTER |

## I10RQ50KF5 minus I10RQ50F5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.6404 / -0.6910 | -0.6657 | [-0.7806, -0.5543] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.5606 / -0.6098 | -0.5852 | [-0.7456, -0.4375] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3468 / -0.2432 | -0.2950 | [-0.4096, -0.1793] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -3.0647 / -3.2537 | -3.1592 | [-3.7884, -2.6081] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0687 / -0.0830 | -0.0758 | [-0.0974, -0.0577] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -17.1277 / -17.1729 | -17.1503 | [-20.1536, -14.5837] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1330 / -0.1797 | -0.1563 | [-0.2058, -0.1101] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1859 / -0.1526 | -0.1692 | [-0.2061, -0.1340] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1792 / -0.2209 | -0.2001 | [-0.2476, -0.1559] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -1.0368 / -1.0033 | -1.0201 | [-1.2799, -0.7732] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2955 / -0.3128 | -0.3041 | [-0.4626, -0.1626] | 0.0229 | BETTER |
| ade6_m|bulk | -1.1633 / -1.0936 | -1.1284 | [-1.4173, -0.8864] | 0.1243 | BETTER |
| ade6_m|all | -0.6912 / -0.6689 | -0.6800 | [-0.8533, -0.5155] | 0.0442 | BETTER |
| ade_tac_m|all | -1.0368 / -1.0033 | -1.0201 | [-1.2799, -0.7732] | 0.0699 | BETTER |

## I10RQ50KF5 minus I10RLcKF5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0164 / -0.0027 | +0.0068 | [-0.0204, +0.0330] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0801 / +0.0135 | +0.0468 | [-0.0334, +0.1340] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0050 / +0.0015 | -0.0017 | [-0.0102, +0.0090] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.1941 / -0.0039 | +0.0951 | [-0.1386, +0.3974] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0088 / +0.0050 | +0.0069 | [+0.0020, +0.0127] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.4143 / +0.1928 | +0.3035 | [+0.0458, +0.6045] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0067 / -0.0038 | -0.0052 | [-0.0123, +0.0021] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0004 / +0.0000 | +0.0002 | [-0.0042, +0.0042] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0063 / -0.0021 | -0.0042 | [-0.0115, +0.0033] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0922 / +0.0101 | +0.0512 | [-0.0552, +0.1618] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1139 / -0.0046 | +0.0547 | [-0.0396, +0.1681] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0177 / +0.0232 | +0.0204 | [-0.0693, +0.1089] | 0.1243 | OPEN |
| ade6_m|all | +0.0700 / +0.0081 | +0.0390 | [-0.0394, +0.1215] | 0.0442 | OPEN |
| ade_tac_m|all | +0.0922 / +0.0101 | +0.0512 | [-0.0552, +0.1618] | 0.0699 | OPEN |

## I10RLcKF5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.5679 / -0.6021 | -0.5850 | [-0.6983, -0.4799] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.2304 / -0.2204 | -0.2254 | [-0.3857, -0.0544] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3849 / -0.2615 | -0.3232 | [-0.4510, -0.1984] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.3907 / -2.5817 | -2.4862 | [-3.0570, -2.0069] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0484 / -0.0664 | -0.0574 | [-0.0779, -0.0406] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -15.2391 / -16.7875 | -16.0133 | [-19.8433, -12.9513] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1284 / -0.1792 | -0.1538 | [-0.2056, -0.1042] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1755 / -0.1259 | -0.1507 | [-0.1957, -0.1115] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1626 / -0.2080 | -0.1853 | [-0.2335, -0.1397] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.6370 / -0.5025 | -0.5698 | [-0.8394, -0.3183] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1426 / -0.1075 | -0.1251 | [-0.2715, +0.0266] | 0.0229 | OPEN |
| ade6_m|bulk | -0.7139 / -0.5591 | -0.6365 | [-0.9455, -0.3567] | 0.1243 | BETTER |
| ade6_m|all | -0.4031 / -0.3135 | -0.3583 | [-0.5460, -0.1770] | 0.0442 | BETTER |
| ade_tac_m|all | -0.6370 / -0.5025 | -0.5698 | [-0.8394, -0.3183] | 0.0699 | BETTER |

## I10RQ50KF5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.5516 / -0.6048 | -0.5782 | [-0.6899, -0.4717] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1504 / -0.2070 | -0.1787 | [-0.3263, -0.0345] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3899 / -0.2600 | -0.3249 | [-0.4539, -0.1983] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.2256 / -2.6313 | -2.4285 | [-3.0942, -1.8913] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0411 / -0.0647 | -0.0529 | [-0.0759, -0.0334] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -14.3146 / -15.8318 | -15.0732 | [-18.7634, -12.2100] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1351 / -0.1830 | -0.1590 | [-0.2100, -0.1115] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1751 / -0.1259 | -0.1505 | [-0.1951, -0.1113] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1688 / -0.2101 | -0.1895 | [-0.2356, -0.1458] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.5448 / -0.4923 | -0.5186 | [-0.7536, -0.3108] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0287 / -0.1121 | -0.0704 | [-0.2171, +0.0735] | 0.0229 | OPEN |
| ade6_m|bulk | -0.6962 / -0.5359 | -0.6160 | [-0.9107, -0.3592] | 0.1243 | BETTER |
| ade6_m|all | -0.3331 / -0.3054 | -0.3192 | [-0.4770, -0.1718] | 0.0442 | BETTER |
| ade_tac_m|all | -0.5448 / -0.4923 | -0.5186 | [-0.7536, -0.3108] | 0.0699 | BETTER |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0G0X_s0 | 2.2145 | 2.9722 | 1.0060 | 0.9256 | 0.4860 |
| I10R0G0X_s1 | 2.2407 | 3.0094 | 1.0121 | 0.9484 | 0.4664 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RLcF5_s0 | 2.9708 | 3.2549 | 1.6639 | 1.2672 | 0.6478 |
| I10RLcF5_s1 | 2.9034 | 3.2523 | 1.7024 | 1.1650 | 0.6807 |
| I10RLcK_s0 | 2.3064 | 2.9633 | 1.0189 | 0.9094 | 0.4815 |
| I10RLcK_s1 | 2.3507 | 3.0212 | 1.0351 | 0.9274 | 0.4585 |
| I10RLcKF5_s0 | 2.3317 | 2.9701 | 1.0288 | 0.9096 | 0.4844 |
| I10RLcKF5_s1 | 2.3771 | 3.0281 | 1.0459 | 0.9271 | 0.4635 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50F5_s0 | 3.0929 | 3.3795 | 1.6856 | 1.2514 | 0.6574 |
| I10RQ50F5_s1 | 3.0541 | 3.3363 | 1.7342 | 1.1719 | 0.6824 |
| I10RQ50K_s0 | 2.3733 | 3.0843 | 1.0418 | 0.9058 | 0.4789 |
| I10RQ50K_s1 | 2.3592 | 3.0341 | 1.0377 | 0.9291 | 0.4623 |
| I10RQ50KF5_s0 | 2.4017 | 3.0840 | 1.0452 | 0.9046 | 0.4781 |
| I10RQ50KF5_s1 | 2.3852 | 3.0235 | 1.0432 | 0.9287 | 0.4614 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
