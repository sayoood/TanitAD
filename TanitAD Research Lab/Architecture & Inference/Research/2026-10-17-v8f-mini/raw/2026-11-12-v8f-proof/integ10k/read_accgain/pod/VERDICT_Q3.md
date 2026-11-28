# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0G50"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867476, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0G50 minus I10R0G0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4340 / +0.6409 | +0.5375 | [+0.3653, +0.7130] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.4706 / +0.6092 | +0.5399 | [+0.3736, +0.7165] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0477 / -0.0506 | -0.0492 | [-0.0757, -0.0255] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +3.0338 / +1.9365 | +2.4851 | [+1.6477, +3.3282] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.2049 / +0.2382 | +0.2215 | [+0.1806, +0.2712] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +30.1380 / +29.7253 | +29.9316 | [+25.5693, +34.7441] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.1680 / +0.2614 | +0.2147 | [+0.1475, +0.2868] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1055 / +0.0321 | +0.0688 | [+0.0236, +0.1173] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.1292 / +0.2059 | +0.1676 | [+0.1125, +0.2281] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +0.5493 / +0.7246 | +0.6369 | [+0.4304, +0.8479] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2574 / +0.3223 | +0.2898 | [+0.1905, +0.3927] | 0.0229 | WORSE |
| ade6_m|bulk | +0.4961 / +0.6747 | +0.5854 | [+0.3477, +0.8375] | 0.1243 | WORSE |
| ade6_m|all | +0.3662 / +0.4830 | +0.4246 | [+0.2869, +0.5652] | 0.0442 | WORSE |
| ade_tac_m|all | +0.5493 / +0.7246 | +0.6369 | [+0.4304, +0.8479] | 0.0699 | WORSE |

## I10R0C10 minus I10R0G50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.2399 / +1.5264 | +1.3832 | [+1.1430, +1.6598] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.7943 / +0.9652 | +0.8797 | [+0.7179, +1.0550] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.1680 / +0.0941 | +0.1311 | [+0.0592, +0.1994] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +2.3319 / +1.9585 | +2.1452 | [+1.6715, +2.6409] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0147 / -0.0021 | +0.0063 | [-0.0243, +0.0327] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +22.7577 / +28.2407 | +25.4992 | [+21.1885, +30.4523] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0930 / +0.0725 | +0.0827 | [+0.0540, +0.1130] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.0671 / +0.0546 | +0.0609 | [+0.0413, +0.0805] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.0921 / +0.1000 | +0.0961 | [+0.0733, +0.1175] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.1554 / +1.4014 | +1.2784 | [+1.0526, +1.5237] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.7261 / +0.7299 | +0.7280 | [+0.6146, +0.8400] | 0.0229 | WORSE |
| ade6_m|bulk | +0.8231 / +1.1780 | +1.0005 | [+0.7177, +1.3168] | 0.1243 | WORSE |
| ade6_m|all | +0.7703 / +0.9343 | +0.8523 | [+0.7017, +1.0158] | 0.0442 | WORSE |
| ade_tac_m|all | +1.1554 / +1.4014 | +1.2784 | [+1.0526, +1.5237] | 0.0699 | WORSE |

## I10R0G0 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.5975 / -0.6260 | -0.6118 | [-0.7147, -0.5207] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.3219 / -0.3394 | -0.3307 | [-0.4393, -0.2451] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | +0.0346 / -0.0369 | -0.0011 | [-0.0702, +0.0624] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.1100 / +0.6225 | +0.3662 | [-0.0907, +0.9064] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0212 / -0.0030 | -0.0121 | [-0.0267, +0.0056] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -12.6649 / -12.8225 | -12.7437 | [-15.7253, -10.3028] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1501 / -0.1684 | -0.1592 | [-0.1989, -0.1215] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.0092 / +0.0067 | -0.0013 | [-0.0265, +0.0229] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0834 / -0.1267 | -0.1050 | [-0.1486, -0.0640] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.2985 / -0.4299 | -0.3642 | [-0.5097, -0.2285] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0732 / -0.1448 | -0.1090 | [-0.1971, -0.0255] | 0.0229 | BETTER |
| ade6_m|bulk | -0.3491 / -0.4558 | -0.4025 | [-0.5689, -0.2540] | 0.1243 | BETTER |
| ade6_m|all | -0.1990 / -0.2866 | -0.2428 | [-0.3398, -0.1523] | 0.0442 | BETTER |
| ade_tac_m|all | -0.2985 / -0.4299 | -0.3642 | [-0.5097, -0.2285] | 0.0699 | BETTER |

## I10R0G0X minus I10R0G0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0067 / -0.0099 | -0.0016 | [-0.0174, +0.0131] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0339 / -0.0325 | -0.0332 | [-0.0645, -0.0008] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.4034 / -0.2033 | -0.3034 | [-0.4834, -0.1377] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.7682 / -3.6260 | -3.1971 | [-4.3181, -2.3781] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0337 / -0.0702 | -0.0520 | [-0.0848, -0.0286] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -3.6106 / -5.0329 | -4.3217 | [-5.9513, -3.1367] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | +0.0242 / -0.0071 | +0.0085 | [-0.0154, +0.0302] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.1584 / -0.1259 | -0.1421 | [-0.1816, -0.1037] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0775 / -0.0784 | -0.0779 | [-0.1080, -0.0492] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.4820 / -0.2449 | -0.3635 | [-0.5827, -0.1520] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0673 / +0.0185 | -0.0244 | [-0.1234, +0.0844] | 0.0229 | OPEN |
| ade6_m|bulk | -0.6244 / -0.3801 | -0.5023 | [-0.7441, -0.2833] | 0.1243 | BETTER |
| ade6_m|all | -0.3213 / -0.1633 | -0.2423 | [-0.3885, -0.1013] | 0.0442 | BETTER |
| ade_tac_m|all | -0.4820 / -0.2449 | -0.3635 | [-0.5827, -0.1520] | 0.0699 | BETTER |

## I10R0Gm25 minus I10R0G0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1257 / +0.0499 | +0.0878 | [+0.0288, +0.1603] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.0986 / +0.0331 | +0.0658 | [-0.0111, +0.1573] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.0694 / +0.0543 | +0.0619 | [+0.0389, +0.0873] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.3499 / -1.0754 | -0.7127 | [-1.2697, -0.2426] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0041 / -0.0104 | -0.0073 | [-0.0172, +0.0012] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +3.1818 / +2.7838 | +2.9828 | [+2.2618, +3.8200] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.0150 / -0.0392 | -0.0271 | [-0.0532, -0.0031] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0313 / -0.0175 | -0.0244 | [-0.0377, -0.0117] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0258 / -0.0200 | -0.0229 | [-0.0359, -0.0111] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1930 / +0.0873 | +0.1402 | [+0.0403, +0.2452] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1777 / +0.0814 | +0.1295 | [+0.0393, +0.2129] | 0.0229 | WORSE |
| ade6_m|bulk | +0.0702 / +0.0306 | +0.0504 | [-0.0287, +0.1502] | 0.1243 | OPEN |
| ade6_m|all | +0.1287 / +0.0582 | +0.0934 | [+0.0268, +0.1635] | 0.0442 | WORSE |
| ade_tac_m|all | +0.1930 / +0.0873 | +0.1402 | [+0.0403, +0.2452] | 0.0699 | WORSE |

## I10R0Gm25 minus I10R0Bk10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0168 / -0.0655 | -0.0412 | [-0.0905, +0.0054] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0566 / +0.0082 | +0.0324 | [-0.0393, +0.1104] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.4634 / +0.2312 | +0.3473 | [+0.1672, +0.5341] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +2.4255 / +1.6165 | +2.0210 | [+1.2948, +2.7689] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0106 / -0.0422 | -0.0264 | [-0.0598, -0.0017] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.9497 / +1.2165 | +2.0831 | [+0.5096, +3.6384] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.1496 / -0.1663 | -0.1580 | [-0.1995, -0.1204] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.0980 / +0.0684 | +0.0832 | [+0.0415, +0.1229] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | -0.0513 / -0.0863 | -0.0688 | [-0.1073, -0.0325] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | +0.5826 / +0.2590 | +0.4208 | [+0.1488, +0.6978] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1990 / +0.0176 | +0.1083 | [-0.0494, +0.2609] | 0.0229 | OPEN |
| ade6_m|bulk | +0.6143 / +0.3577 | +0.4860 | [+0.2218, +0.7626] | 0.1243 | WORSE |
| ade6_m|all | +0.3884 / +0.1727 | +0.2805 | [+0.0992, +0.4652] | 0.0442 | WORSE |
| ade_tac_m|all | +0.5826 / +0.2590 | +0.4208 | [+0.1488, +0.6978] | 0.0699 | WORSE |

## I10R0G0X minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.5908 / -0.6359 | -0.6133 | [-0.7220, -0.5146] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.3558 / -0.3719 | -0.3639 | [-0.4714, -0.2728] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3688 / -0.2402 | -0.3045 | [-0.4326, -0.1796] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.6581 / -3.0036 | -2.8309 | [-3.5343, -2.2817] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0549 / -0.0733 | -0.0641 | [-0.0859, -0.0463] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -16.2755 / -17.8554 | -17.0655 | [-21.1298, -13.7195] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1259 / -0.1755 | -0.1507 | [-0.2026, -0.1027] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1676 / -0.1192 | -0.1434 | [-0.1880, -0.1047] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1609 / -0.2051 | -0.1830 | [-0.2312, -0.1382] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.7805 / -0.6748 | -0.7277 | [-0.9478, -0.5450] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1405 / -0.1262 | -0.1334 | [-0.2461, -0.0210] | 0.0229 | BETTER |
| ade6_m|bulk | -0.9735 / -0.8360 | -0.9047 | [-1.1715, -0.6947] | 0.1243 | BETTER |
| ade6_m|all | -0.5204 / -0.4499 | -0.4851 | [-0.6318, -0.3633] | 0.0442 | BETTER |
| ade_tac_m|all | -0.7805 / -0.6748 | -0.7277 | [-0.9478, -0.5450] | 0.0699 | BETTER |

## I10R0G0 minus I10R0C10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -1.6739 / -2.1673 | -1.9206 | [-2.3293, -1.5519] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -1.2649 / -1.5743 | -1.4196 | [-1.7535, -1.1099] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.1203 / -0.0435 | -0.0819 | [-0.1614, +0.0039] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -5.3657 / -3.8949 | -4.6303 | [-5.7703, -3.5018] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.2195 / -0.2361 | -0.2278 | [-0.2676, -0.1949] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -52.8957 / -57.9659 | -55.4308 | [-63.6112, -48.6545] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.2609 / -0.3339 | -0.2974 | [-0.3669, -0.2322] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1726 / -0.0867 | -0.1296 | [-0.1882, -0.0738] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.2213 / -0.3060 | -0.2637 | [-0.3324, -0.1988] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -1.7048 / -2.1259 | -1.9154 | [-2.3399, -1.5060] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.9834 / -1.0522 | -1.0178 | [-1.1747, -0.8625] | 0.0229 | BETTER |
| ade6_m|bulk | -1.3191 / -1.8528 | -1.5859 | [-2.1324, -1.0900] | 0.1243 | BETTER |
| ade6_m|all | -1.1365 / -1.4173 | -1.2769 | [-1.5599, -1.0040] | 0.0442 | BETTER |
| ade_tac_m|all | -1.7048 / -2.1259 | -1.9154 | [-2.3399, -1.5060] | 0.0699 | BETTER |

## I10R0G50 minus I10R0C10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -1.2399 / -1.5264 | -1.3832 | [-1.6598, -1.1430] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.7943 / -0.9652 | -0.8797 | [-1.0550, -0.7179] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.1680 / -0.0941 | -0.1311 | [-0.1994, -0.0592] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.3319 / -1.9585 | -2.1452 | [-2.6409, -1.6715] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0147 / +0.0021 | -0.0063 | [-0.0327, +0.0243] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -22.7577 / -28.2407 | -25.4992 | [-30.4523, -21.1885] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0930 / -0.0725 | -0.0827 | [-0.1130, -0.0540] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.0671 / -0.0546 | -0.0609 | [-0.0805, -0.0413] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0921 / -0.1000 | -0.0961 | [-0.1175, -0.0733] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -1.1554 / -1.4014 | -1.2784 | [-1.5237, -1.0526] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.7261 / -0.7299 | -0.7280 | [-0.8400, -0.6146] | 0.0229 | BETTER |
| ade6_m|bulk | -0.8231 / -1.1780 | -1.0005 | [-1.3168, -0.7177] | 0.1243 | BETTER |
| ade6_m|all | -0.7703 / -0.9343 | -0.8523 | [-1.0158, -0.7017] | 0.0442 | BETTER |
| ade_tac_m|all | -1.1554 / -1.4014 | -1.2784 | [-1.5237, -1.0526] | 0.0699 | BETTER |

## I10R0Gm25 minus I10R0C10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -1.5482 / -2.1174 | -1.8328 | [-2.2755, -1.4439] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -1.1663 / -1.5413 | -1.3538 | [-1.6893, -1.0453] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.0509 / +0.0108 | -0.0201 | [-0.0874, +0.0595] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -5.7157 / -4.9704 | -5.3430 | [-6.2145, -4.5092] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.2237 / -0.2469 | -0.2353 | [-0.2779, -0.2013] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -49.7139 / -55.2163 | -52.4651 | [-60.4592, -45.8595] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.2759 / -0.3731 | -0.3245 | [-0.4062, -0.2482] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2038 / -0.1042 | -0.1540 | [-0.2222, -0.0906] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.2472 / -0.3260 | -0.2866 | [-0.3534, -0.2234] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -1.5117 / -2.0386 | -1.7752 | [-2.2332, -1.3444] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.8057 / -0.9709 | -0.8883 | [-1.0867, -0.6965] | 0.0229 | BETTER |
| ade6_m|bulk | -1.2490 / -1.8222 | -1.5356 | [-2.0401, -1.0594] | 0.1243 | BETTER |
| ade6_m|all | -1.0078 / -1.3591 | -1.1835 | [-1.4888, -0.8963] | 0.0442 | BETTER |
| ade_tac_m|all | -1.5117 / -2.0386 | -1.7752 | [-2.2332, -1.3444] | 0.0699 | BETTER |

## I10R0G0X minus I10R0C10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -1.6672 / -2.1772 | -1.9222 | [-2.3385, -1.5432] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -1.2988 / -1.6068 | -1.4528 | [-1.7797, -1.1522] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.5237 / -0.2468 | -0.3853 | [-0.5851, -0.2074] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -8.1339 / -7.5210 | -7.8274 | [-8.8598, -6.8323] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.2532 / -0.3063 | -0.2798 | [-0.3379, -0.2319] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -56.5063 / -62.9988 | -59.7526 | [-69.0747, -52.1593] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.2368 / -0.3410 | -0.2889 | [-0.3737, -0.2081] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.3310 / -0.2126 | -0.2718 | [-0.3556, -0.1932] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.2989 / -0.3843 | -0.3416 | [-0.4131, -0.2757] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -2.1868 / -2.3708 | -2.2788 | [-2.7005, -1.9147] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -1.0507 / -1.0337 | -1.0422 | [-1.2218, -0.8618] | 0.0229 | BETTER |
| ade6_m|bulk | -1.9435 / -2.2329 | -2.0882 | [-2.6246, -1.6437] | 0.1243 | BETTER |
| ade6_m|all | -1.4579 / -1.5806 | -1.5192 | [-1.8003, -1.2765] | 0.0442 | BETTER |
| ade_tac_m|all | -2.1868 / -2.3708 | -2.2788 | [-2.7005, -1.9147] | 0.0699 | BETTER |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0Bk10_s0 | 2.2761 | 3.0182 | 1.1418 | 0.9350 | 0.5890 |
| I10R0Bk10_s1 | 2.2895 | 3.0546 | 1.1373 | 0.9748 | 0.6111 |
| I10R0C10_s0 | 3.6723 | 4.0229 | 2.6732 | 1.4494 | 0.7849 |
| I10R0C10_s1 | 3.8213 | 4.0431 | 3.1893 | 1.1953 | 0.8508 |
| I10R0G0_s0 | 2.5358 | 3.0395 | 0.9992 | 1.3290 | 0.5636 |
| I10R0G0_s1 | 2.4040 | 2.9908 | 1.0219 | 1.1518 | 0.5448 |
| I10R0G0X_s0 | 2.2145 | 2.9722 | 1.0060 | 0.9256 | 0.4860 |
| I10R0G0X_s1 | 2.2407 | 3.0094 | 1.0121 | 0.9484 | 0.4664 |
| I10R0G50_s0 | 2.9020 | 3.2969 | 1.4333 | 1.2814 | 0.6928 |
| I10R0G50_s1 | 2.8870 | 3.3132 | 1.6629 | 1.1011 | 0.7507 |
| I10R0Gm25_s0 | 2.6645 | 3.2172 | 1.1250 | 1.3985 | 0.5377 |
| I10R0Gm25_s1 | 2.4622 | 3.0722 | 1.0718 | 1.2060 | 0.5248 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
