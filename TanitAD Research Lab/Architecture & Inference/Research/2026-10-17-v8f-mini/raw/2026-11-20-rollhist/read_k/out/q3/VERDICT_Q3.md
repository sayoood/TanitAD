# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 28}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RHK_R0C10XHb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.23423522057961366, "D_read": 0.23423522057961368, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10RHK_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RHK R0C10XHb20 minus I10B0 R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0053 / +0.0045 | +0.0049 | [-0.0122, +0.0237] | 0.0781 (0.0781) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0317 / +0.0147 | +0.0232 | [-0.0087, +0.0574] | 0.0832 (0.0832) | OPEN |
| LATERAL:cross_mae_m | -0.0360 / -0.0313 | -0.0336 | [-0.0509, -0.0174] | 0.0422 (0.0422) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.2476 / -0.3530 | -0.3003 | [-0.4584, -0.1346] | 0.8003 (0.8003) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0164 / -0.0280 | -0.0222 | [-0.0427, -0.0036] | 0.0350 (0.0350) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.6870 / -1.4127 | -1.0498 | [-1.9245, -0.1204] | 3.3179 (3.3179) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0008 / -0.0142 | -0.0075 | [-0.0206, +0.0052] | 0.0463 (0.0463) | OPEN |
| TACTICAL:lat_err_rate | -0.0208 / -0.0038 | -0.0123 | [-0.0271, +0.0019] | 0.0013 (0.0013) | OPEN |
| TACTICAL:joint_err_rate | -0.0088 / -0.0196 | -0.0142 | [-0.0263, -0.0010] | 0.0467 (0.0467) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0074 / -0.0203 | -0.0065 | [-0.0572, +0.0469] | 0.0497 (0.0497) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0422 / -0.0332 | -0.0377 | [-0.0807, +0.0031] | 0.0560 | OPEN |
| ade6_m|bulk | +0.0658 / +0.0139 | +0.0399 | [-0.0135, +0.1051] | 0.0054 | OPEN |
| ade6_m|all | +0.0071 / -0.0117 | -0.0023 | [-0.0365, +0.0335] | 0.0329 | OPEN |
| ade_tac_m|all | +0.0074 / -0.0203 | -0.0065 | [-0.0572, +0.0469] | 0.0497 | OPEN |

## I10RHK R0C10XHb20 minus I10RHK R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4222 / +0.5788 | +0.5005 | [+0.3590, +0.6544] | 0.0781 (0.0781) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5415 / +0.6893 | +0.6154 | [+0.4520, +0.7970] | 0.0832 (0.0832) | WORSE |
| LATERAL:cross_mae_m | +0.0168 / +0.0303 | +0.0235 | [+0.0081, +0.0415] | 0.0422 (0.0422) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.6580 / +2.2552 | +1.9566 | [+1.4802, +2.5504] | 0.8003 (0.8003) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1691 / +0.1872 | +0.1781 | [+0.1430, +0.2210] | 0.0350 (0.0350) | WORSE |
| LATERAL:yaw_rate_mae_degps | +13.3509 / +15.4654 | +14.4081 | [+11.9018, +17.3771] | 3.3179 (3.3179) | WORSE |
| TACTICAL:lon_err_rate | +0.1722 / +0.2393 | +0.2057 | [+0.1496, +0.2657] | 0.0463 (0.0463) | WORSE |
| TACTICAL:lat_err_rate | +0.0896 / +0.0892 | +0.0894 | [+0.0708, +0.1105] | 0.0013 (0.0013) | WORSE |
| TACTICAL:joint_err_rate | +0.1859 / +0.2522 | +0.2190 | [+0.1642, +0.2786] | 0.0467 (0.0467) | WORSE |
| TACTICAL:ade_tac_m | +0.7615 / +0.9914 | +0.8764 | [+0.6291, +1.1513] | 0.0497 (0.0497) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1912 / +0.2963 | +0.2438 | [+0.1283, +0.3627] | 0.0560 | WORSE |
| ade6_m|bulk | +0.8852 / +1.0958 | +0.9905 | [+0.7202, +1.3174] | 0.0054 | WORSE |
| ade6_m|all | +0.5076 / +0.6609 | +0.5843 | [+0.4194, +0.7675] | 0.0329 | WORSE |
| ade_tac_m|all | +0.7615 / +0.9914 | +0.8764 | [+0.6291, +1.1513] | 0.0497 | WORSE |

## I10RHK R0G50Hb20 minus I10RHK R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1492 / +0.1874 | +0.1683 | [+0.1172, +0.2225] | 0.0781 (0.0781) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1967 / +0.2377 | +0.2172 | [+0.1506, +0.2862] | 0.0832 (0.0832) | WORSE |
| LATERAL:cross_mae_m | -0.0245 / -0.0369 | -0.0307 | [-0.0522, -0.0144] | 0.0422 (0.0422) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.5899 / +0.9626 | +0.7763 | [+0.4972, +1.1363] | 0.8003 (0.8003) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0718 / +0.0903 | +0.0810 | [+0.0618, +0.1051] | 0.0350 (0.0350) | WORSE |
| LATERAL:yaw_rate_mae_degps | +6.2405 / +8.0095 | +7.1250 | [+5.5497, +9.1231] | 3.3179 (3.3179) | WORSE |
| TACTICAL:lon_err_rate | +0.0725 / +0.1463 | +0.1094 | [+0.0560, +0.1681] | 0.0463 (0.0463) | WORSE |
| TACTICAL:lat_err_rate | +0.0050 / +0.0008 | +0.0029 | [-0.0113, +0.0154] | 0.0013 (0.0013) | OPEN |
| TACTICAL:joint_err_rate | +0.0475 / +0.1117 | +0.0796 | [+0.0358, +0.1304] | 0.0467 (0.0467) | WORSE |
| TACTICAL:ade_tac_m | +0.2176 / +0.2652 | +0.2414 | [+0.1626, +0.3240] | 0.0497 (0.0497) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0306 / +0.0415 | +0.0361 | [-0.0202, +0.0955] | 0.0560 | OPEN |
| ade6_m|bulk | +0.2816 / +0.3381 | +0.3099 | [+0.2345, +0.3975] | 0.0054 | WORSE |
| ade6_m|all | +0.1451 / +0.1768 | +0.1609 | [+0.1084, +0.2160] | 0.0329 | WORSE |
| ade_tac_m|all | +0.2176 / +0.2652 | +0.2414 | [+0.1626, +0.3240] | 0.0497 | WORSE |

## I10RHK R0C10Hb20 minus I10RHK R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2891 / +0.4350 | +0.3621 | [+0.2538, +0.4777] | 0.0781 (0.0781) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3627 / +0.5078 | +0.4353 | [+0.3126, +0.5658] | 0.0832 (0.0832) | WORSE |
| LATERAL:cross_mae_m | +0.0033 / +0.0139 | +0.0086 | [-0.0120, +0.0276] | 0.0422 (0.0422) | OPEN |
| LATERAL:heading_mae_deg | +1.5211 / +1.8050 | +1.6631 | [+1.3538, +2.0190] | 0.8003 (0.8003) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1142 / +0.1171 | +0.1156 | [+0.0933, +0.1403] | 0.0350 (0.0350) | WORSE |
| LATERAL:yaw_rate_mae_degps | +11.8650 / +13.3130 | +12.5890 | [+10.5746, +14.8275] | 3.3179 (3.3179) | WORSE |
| TACTICAL:lon_err_rate | +0.0988 / +0.0992 | +0.0990 | [+0.0759, +0.1236] | 0.0463 (0.0463) | WORSE |
| TACTICAL:lat_err_rate | +0.0650 / +0.0888 | +0.0769 | [+0.0544, +0.1043] | 0.0013 (0.0013) | WORSE |
| TACTICAL:joint_err_rate | +0.0934 / +0.1188 | +0.1061 | [+0.0814, +0.1359] | 0.0467 (0.0467) | WORSE |
| TACTICAL:ade_tac_m | +0.4573 / +0.6881 | +0.5727 | [+0.3927, +0.7670] | 0.0497 (0.0497) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1692 / +0.2874 | +0.2283 | [+0.1410, +0.3179] | 0.0560 | WORSE |
| ade6_m|bulk | +0.4667 / +0.6631 | +0.5649 | [+0.3839, +0.7752] | 0.0054 | WORSE |
| ade6_m|all | +0.3049 / +0.4588 | +0.3818 | [+0.2618, +0.5114] | 0.0329 | WORSE |
| ade_tac_m|all | +0.4573 / +0.6881 | +0.5727 | [+0.3927, +0.7670] | 0.0497 | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4253 / +0.5783 | +0.5018 | [+0.3633, +0.6525] | 0.0781 (0.0781) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5148 / +0.6779 | +0.5963 | [+0.4294, +0.7851] | 0.0832 (0.0832) | WORSE |
| LATERAL:cross_mae_m | +0.0021 / +0.0281 | +0.0151 | [-0.0064, +0.0392] | 0.0422 (0.0422) | OPEN |
| LATERAL:heading_mae_deg | +1.2984 / +2.1302 | +1.7143 | [+1.1487, +2.3855] | 0.8003 (0.8003) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1632 / +0.1844 | +0.1738 | [+0.1396, +0.2133] | 0.0350 (0.0350) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.8306 / +15.7625 | +14.2966 | [+11.4789, +17.5884] | 3.3179 (3.3179) | WORSE |
| TACTICAL:lon_err_rate | +0.1734 / +0.2543 | +0.2138 | [+0.1512, +0.2812] | 0.0463 (0.0463) | WORSE |
| TACTICAL:lat_err_rate | +0.0646 / +0.0638 | +0.0642 | [+0.0500, +0.0794] | 0.0013 (0.0013) | WORSE |
| TACTICAL:joint_err_rate | +0.1788 / +0.2580 | +0.2184 | [+0.1561, +0.2826] | 0.0467 (0.0467) | WORSE |
| TACTICAL:ade_tac_m | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0497 (0.0497) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1535 / +0.2743 | +0.2139 | [+0.0940, +0.3374] | 0.0560 | WORSE |
| ade6_m|bulk | +0.8602 / +1.1053 | +0.9827 | [+0.7095, +1.3144] | 0.0054 | WORSE |
| ade6_m|all | +0.4758 / +0.6532 | +0.5645 | [+0.3925, +0.7571] | 0.0329 | WORSE |
| ade_tac_m|all | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0497 | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1390 / +0.1740 | +0.1565 | [+0.1087, +0.2098] | 0.0781 (0.0781) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1801 / +0.2231 | +0.2016 | [+0.1391, +0.2668] | 0.0832 (0.0832) | WORSE |
| LATERAL:cross_mae_m | -0.0540 / -0.0648 | -0.0594 | [-0.0833, -0.0397] | 0.0422 (0.0422) | BETTER |
| LATERAL:heading_mae_deg | +0.2941 / +0.8362 | +0.5652 | [+0.2123, +0.9892] | 0.8003 (0.8003) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0682 / +0.0883 | +0.0783 | [+0.0594, +0.1021] | 0.0350 (0.0350) | WORSE |
| LATERAL:yaw_rate_mae_degps | +5.7579 / +7.9876 | +6.8727 | [+5.1833, +9.1092] | 3.3179 (3.3179) | WORSE |
| TACTICAL:lon_err_rate | +0.0767 / +0.1463 | +0.1115 | [+0.0594, +0.1676] | 0.0463 (0.0463) | WORSE |
| TACTICAL:lat_err_rate | -0.0113 / -0.0129 | -0.0121 | [-0.0231, -0.0017] | 0.0013 (0.0013) | BETTER |
| TACTICAL:joint_err_rate | +0.0429 / +0.0713 | +0.0571 | [+0.0304, +0.0869] | 0.0467 (0.0467) | WORSE |
| TACTICAL:ade_tac_m | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0497 (0.0497) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0033 / +0.0210 | +0.0122 | [-0.0397, +0.0695] | 0.0560 | OPEN |
| ade6_m|bulk | +0.2261 / +0.2684 | +0.2473 | [+0.1874, +0.3092] | 0.0054 | WORSE |
| ade6_m|all | +0.1049 / +0.1338 | +0.1194 | [+0.0744, +0.1683] | 0.0329 | WORSE |
| ade_tac_m|all | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0497 | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2919 / +0.4426 | +0.3672 | [+0.2558, +0.4843] | 0.0781 (0.0781) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3451 / +0.5030 | +0.4241 | [+0.2983, +0.5604] | 0.0832 (0.0832) | WORSE |
| LATERAL:cross_mae_m | -0.0080 / -0.0177 | -0.0128 | [-0.0332, +0.0072] | 0.0422 (0.0422) | OPEN |
| LATERAL:heading_mae_deg | +1.4153 / +1.5714 | +1.4933 | [+1.2106, +1.8094] | 0.8003 (0.8003) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1058 / +0.1205 | +0.1131 | [+0.0895, +0.1417] | 0.0350 (0.0350) | WORSE |
| LATERAL:yaw_rate_mae_degps | +11.2146 / +14.0424 | +12.6285 | [+10.0108, +15.6291] | 3.3179 (3.3179) | WORSE |
| TACTICAL:lon_err_rate | +0.0942 / +0.1105 | +0.1023 | [+0.0763, +0.1301] | 0.0463 (0.0463) | WORSE |
| TACTICAL:lat_err_rate | +0.0538 / +0.0396 | +0.0467 | [+0.0311, +0.0628] | 0.0013 (0.0013) | WORSE |
| TACTICAL:joint_err_rate | +0.0763 / +0.1159 | +0.0961 | [+0.0650, +0.1312] | 0.0467 (0.0467) | WORSE |
| TACTICAL:ade_tac_m | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0497 (0.0497) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1666 / +0.2602 | +0.2134 | [+0.1412, +0.2884] | 0.0560 | WORSE |
| ade6_m|bulk | +0.4298 / +0.6170 | +0.5234 | [+0.3460, +0.7226] | 0.0054 | WORSE |
| ade6_m|all | +0.2866 / +0.4229 | +0.3548 | [+0.2432, +0.4718] | 0.0329 | WORSE |
| ade_tac_m|all | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0497 | WORSE |

## I10RHK R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4306 / +0.5828 | +0.5067 | [+0.3641, +0.6625] | 0.0781 (0.0781) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5464 / +0.6926 | +0.6195 | [+0.4482, +0.8093] | 0.0832 (0.0832) | WORSE |
| LATERAL:cross_mae_m | -0.0339 / -0.0032 | -0.0185 | [-0.0469, +0.0135] | 0.0422 (0.0422) | OPEN |
| LATERAL:heading_mae_deg | +1.0504 / +1.7980 | +1.4242 | [+0.8870, +2.0634] | 0.8003 (0.8003) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1490 / +0.1651 | +0.1571 | [+0.1260, +0.1938] | 0.0350 (0.0350) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.2420 / +14.6060 | +13.4240 | [+10.9136, +16.3788] | 3.3179 (3.3179) | WORSE |
| TACTICAL:lon_err_rate | +0.1726 / +0.2401 | +0.2063 | [+0.1506, +0.2687] | 0.0463 (0.0463) | WORSE |
| TACTICAL:lat_err_rate | +0.0438 / +0.0600 | +0.0519 | [+0.0317, +0.0725] | 0.0013 (0.0013) | WORSE |
| TACTICAL:joint_err_rate | +0.1701 / +0.2384 | +0.2043 | [+0.1478, +0.2645] | 0.0467 (0.0467) | WORSE |
| TACTICAL:ade_tac_m | +0.7210 / +0.9595 | +0.8403 | [+0.5825, +1.1321] | 0.0497 (0.0497) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1114 / +0.2411 | +0.1763 | [+0.0432, +0.3177] | 0.0560 | WORSE |
| ade6_m|bulk | +0.9260 / +1.1192 | +1.0226 | [+0.7475, +1.3537] | 0.0054 | WORSE |
| ade6_m|all | +0.4829 / +0.6416 | +0.5622 | [+0.3905, +0.7565] | 0.0329 | WORSE |
| ade_tac_m|all | +0.7210 / +0.9595 | +0.8403 | [+0.5825, +1.1321] | 0.0497 | WORSE |

## I10RHK R0Hb20 minus I10B0 R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0391 / +0.0365 | +0.0378 | [+0.0213, +0.0557] | 0.0781 (0.0781) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0482 / +0.0612 | +0.0547 | [+0.0301, +0.0803] | 0.0832 (0.0832) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0368 / -0.1048 | -0.0708 | [-0.1217, -0.0261] | 0.0422 (0.0422) | BETTER |
| LATERAL:heading_mae_deg | -0.4086 / -0.3302 | -0.3694 | [-0.4904, -0.2400] | 0.8003 (0.8003) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0038 / -0.0033 | -0.0035 | [-0.0112, +0.0039] | 0.0350 (0.0350) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.7279 / +0.3797 | -0.1741 | [-1.0238, +0.7169] | 3.3179 (3.3179) | OPEN |
| TACTICAL:lon_err_rate | +0.0325 / +0.0079 | +0.0202 | [-0.0013, +0.0423] | 0.0463 (0.0463) | OPEN |
| TACTICAL:lat_err_rate | -0.0438 / -0.0317 | -0.0377 | [-0.0559, -0.0221] | 0.0013 (0.0013) | BETTER |
| TACTICAL:joint_err_rate | +0.0042 / -0.0142 | -0.0050 | [-0.0244, +0.0127] | 0.0467 (0.0467) | OPEN |
| TACTICAL:ade_tac_m | +0.0361 / -0.0314 | +0.0023 | [-0.0554, +0.0605] | 0.0497 (0.0497) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0286 / +0.0297 | +0.0291 | [-0.0006, +0.0605] | 0.0560 | OPEN |
| ade6_m|bulk | +0.0235 / -0.0773 | -0.0269 | [-0.1011, +0.0506] | 0.0054 | OPEN |
| ade6_m|all | +0.0263 / -0.0191 | +0.0036 | [-0.0353, +0.0428] | 0.0329 | OPEN |
| ade_tac_m|all | +0.0361 / -0.0314 | +0.0023 | [-0.0554, +0.0605] | 0.0497 | OPEN |

## I10RHK R0C10Hb20 minus I10B0 R0C10Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0163 / +0.0194 | +0.0178 | [-0.0013, +0.0379] | 0.0781 (0.0781) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0562 / +0.0335 | +0.0449 | [+0.0139, +0.0801] | 0.0832 (0.0832) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0087 / -0.0722 | -0.0405 | [-0.0852, +0.0027] | 0.0422 (0.0422) | OPEN |
| LATERAL:heading_mae_deg | -0.2145 / -0.2547 | -0.2346 | [-0.3932, -0.0667] | 0.8003 (0.8003) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0080 / -0.0184 | -0.0132 | [-0.0293, +0.0032] | 0.0350 (0.0350) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0218 / -1.4235 | -0.7008 | [-1.8988, +0.6101] | 3.3179 (3.3179) | OPEN |
| TACTICAL:lon_err_rate | +0.0092 / -0.0113 | -0.0010 | [-0.0190, +0.0158] | 0.0463 (0.0463) | OPEN |
| TACTICAL:lat_err_rate | -0.0417 / -0.0367 | -0.0392 | [-0.0530, -0.0258] | 0.0013 (0.0013) | BETTER |
| TACTICAL:joint_err_rate | +0.0008 / -0.0200 | -0.0096 | [-0.0277, +0.0083] | 0.0467 (0.0467) | OPEN |
| TACTICAL:ade_tac_m | +0.0654 / -0.0131 | +0.0261 | [-0.0406, +0.0951] | 0.0497 (0.0497) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0188 / +0.0111 | +0.0149 | [-0.0182, +0.0482] | 0.0560 | OPEN |
| ade6_m|bulk | +0.0779 / -0.0283 | +0.0248 | [-0.0557, +0.1147] | 0.0054 | OPEN |
| ade6_m|all | +0.0458 / -0.0069 | +0.0194 | [-0.0253, +0.0657] | 0.0329 | OPEN |
| ade_tac_m|all | +0.0654 / -0.0131 | +0.0261 | [-0.0406, +0.0951] | 0.0497 | OPEN |

## I10RHK R0G50Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0190 / +0.0270 | +0.0230 | [+0.0099, +0.0372] | 0.0781 (0.0781) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0387 / +0.0287 | +0.0337 | [+0.0108, +0.0576] | 0.0832 (0.0832) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0200 / -0.1037 | -0.0619 | [-0.1202, -0.0094] | 0.0422 (0.0422) | BETTER |
| LATERAL:heading_mae_deg | -0.3157 / -0.4526 | -0.3841 | [-0.5303, -0.2347] | 0.8003 (0.8003) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0110 / -0.0107 | -0.0109 | [-0.0195, -0.0028] | 0.0350 (0.0350) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.4337 / -0.5657 | -0.4997 | [-0.9675, -0.0676] | 3.3179 (3.3179) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0046 / +0.0000 | +0.0023 | [-0.0094, +0.0159] | 0.0463 (0.0463) | OPEN |
| TACTICAL:lat_err_rate | -0.0529 / -0.0859 | -0.0694 | [-0.1025, -0.0429] | 0.0013 (0.0013) | BETTER |
| TACTICAL:joint_err_rate | -0.0163 / -0.0229 | -0.0196 | [-0.0331, -0.0058] | 0.0467 (0.0467) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0380 / -0.0668 | -0.0144 | [-0.0892, +0.0611] | 0.0497 (0.0497) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0161 / -0.0161 | -0.0000 | [-0.0369, +0.0394] | 0.0560 | OPEN |
| ade6_m|bulk | +0.0411 / -0.0744 | -0.0166 | [-0.0971, +0.0630] | 0.0054 | OPEN |
| ade6_m|all | +0.0275 / -0.0427 | -0.0076 | [-0.0576, +0.0430] | 0.0329 | OPEN |
| ade_tac_m|all | +0.0380 / -0.0668 | -0.0144 | [-0.0892, +0.0611] | 0.0497 | OPEN |

## I10RHK R0G0Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0088 / +0.0136 | +0.0112 | [+0.0020, +0.0208] | 0.0781 (0.0781) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0221 / +0.0142 | +0.0181 | [+0.0003, +0.0348] | 0.0832 (0.0832) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0495 / -0.1317 | -0.0906 | [-0.1516, -0.0379] | 0.0422 (0.0422) | BETTER |
| LATERAL:heading_mae_deg | -0.6150 / -0.5855 | -0.6003 | [-0.7323, -0.4882] | 0.8003 (0.8003) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0163 / -0.0135 | -0.0149 | [-0.0217, -0.0093] | 0.0350 (0.0350) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.9278 / -0.5977 | -0.7627 | [-1.1707, -0.4631] | 3.3179 (3.3179) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0088 / +0.0000 | +0.0044 | [-0.0069, +0.0190] | 0.0463 (0.0463) | OPEN |
| TACTICAL:lat_err_rate | -0.0692 / -0.0996 | -0.0844 | [-0.1153, -0.0582] | 0.0013 (0.0013) | BETTER |
| TACTICAL:joint_err_rate | -0.0208 / -0.0634 | -0.0421 | [-0.0757, -0.0117] | 0.0467 (0.0467) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.0222 / -0.1313 | -0.0767 | [-0.1580, -0.0043] | 0.0497 (0.0497) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0112 / -0.0366 | -0.0239 | [-0.0566, +0.0086] | 0.0560 | OPEN |
| ade6_m|bulk | -0.0144 / -0.1441 | -0.0793 | [-0.1703, +0.0020] | 0.0054 | OPEN |
| ade6_m|all | -0.0126 / -0.0857 | -0.0491 | [-0.1035, -0.0005] | 0.0329 | BETTER |
| ade_tac_m|all | -0.0222 / -0.1313 | -0.0767 | [-0.1580, -0.0043] | 0.0497 | BETTER |

## I10RHK R0KHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0084 / +0.0039 | +0.0062 | [-0.0021, +0.0144] | 0.0781 (0.0781) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0049 / +0.0033 | +0.0041 | [-0.0163, +0.0238] | 0.0832 (0.0832) | OPEN |
| LATERAL:cross_mae_m | -0.0506 / -0.0335 | -0.0421 | [-0.0628, -0.0202] | 0.0422 (0.0422) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.6120 / -0.4484 | -0.5302 | [-0.7101, -0.3705] | 0.8003 (0.8003) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0190 / -0.0142 | -0.0166 | [-0.0239, -0.0101] | 0.0350 (0.0350) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.0273 / -0.6116 | -0.8195 | [-1.2780, -0.4638] | 3.3179 (3.3179) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0004 / +0.0008 | +0.0006 | [-0.0090, +0.0102] | 0.0463 (0.0463) | OPEN |
| TACTICAL:lat_err_rate | -0.0459 / -0.0292 | -0.0375 | [-0.0544, -0.0223] | 0.0013 (0.0013) | BETTER |
| TACTICAL:joint_err_rate | -0.0158 / -0.0138 | -0.0148 | [-0.0267, -0.0040] | 0.0467 (0.0467) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.0404 / -0.0319 | -0.0361 | [-0.0734, -0.0003] | 0.0497 (0.0497) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0798 / -0.0552 | -0.0675 | [-0.1092, -0.0234] | 0.0560 | BETTER |
| ade6_m|bulk | +0.0409 / +0.0234 | +0.0321 | [+0.0131, +0.0549] | 0.0054 | WORSE |
| ade6_m|all | -0.0248 / -0.0194 | -0.0221 | [-0.0468, +0.0018] | 0.0329 | OPEN |
| ade_tac_m|all | -0.0404 / -0.0319 | -0.0361 | [-0.0734, -0.0003] | 0.0497 | INSIDE-SPREAD |

## I10RHK R0Hb20 minus I10RH R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0762 / +0.1288 | +0.1025 | [+0.0513, +0.1570] | 0.0781 (0.0781) | WORSE |
| LONGITUDINAL:along_mae_m | +0.0248 / +0.1397 | +0.0823 | [-0.0109, +0.1778] | 0.0832 (0.0832) | OPEN |
| LATERAL:cross_mae_m | -0.0200 / -0.1051 | -0.0625 | [-0.1234, -0.0093] | 0.0422 (0.0422) | BETTER |
| LATERAL:heading_mae_deg | +0.5471 / +0.7732 | +0.6601 | [+0.4550, +0.9170] | 0.8003 (0.8003) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0583 / +0.0599 | +0.0591 | [+0.0460, +0.0741] | 0.0350 (0.0350) | WORSE |
| LATERAL:yaw_rate_mae_degps | +3.4337 / +4.8601 | +4.1469 | [+2.8860, +5.5842] | 3.3179 (3.3179) | WORSE |
| TACTICAL:lon_err_rate | +0.0830 / +0.0771 | +0.0800 | [+0.0569, +0.1055] | 0.0463 (0.0463) | WORSE |
| TACTICAL:lat_err_rate | +0.0079 / +0.0263 | +0.0171 | [-0.0048, +0.0373] | 0.0013 (0.0013) | OPEN |
| TACTICAL:joint_err_rate | +0.0600 / +0.0738 | +0.0669 | [+0.0463, +0.0911] | 0.0467 (0.0467) | WORSE |
| TACTICAL:ade_tac_m | +0.0282 / +0.0856 | +0.0569 | [-0.0250, +0.1384] | 0.0497 (0.0497) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0757 / +0.0535 | -0.0111 | [-0.1246, +0.0907] | 0.0560 | OPEN |
| ade6_m|bulk | +0.1233 / +0.0521 | +0.0877 | [+0.0057, +0.1575] | 0.0054 | WORSE |
| ade6_m|all | +0.0151 / +0.0529 | +0.0340 | [-0.0215, +0.0887] | 0.0329 | OPEN |
| ade_tac_m|all | +0.0282 / +0.0856 | +0.0569 | [-0.0250, +0.1384] | 0.0497 | OPEN |

## I10RHK R0C10XHb20 minus I10RH R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2013 / +0.3372 | +0.2692 | [+0.1490, +0.3999] | 0.0781 (0.0781) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1545 / +0.3407 | +0.2476 | [+0.0798, +0.4244] | 0.0832 (0.0832) | WORSE |
| LATERAL:cross_mae_m | -0.0252 / +0.0014 | -0.0119 | [-0.0334, +0.0108] | 0.0422 (0.0422) | OPEN |
| LATERAL:heading_mae_deg | +0.5350 / +0.5070 | +0.5210 | [+0.3604, +0.6955] | 0.8003 (0.8003) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0730 / +0.0712 | +0.0721 | [+0.0567, +0.0894] | 0.0350 (0.0350) | WORSE |
| LATERAL:yaw_rate_mae_degps | +4.4885 / +4.5206 | +4.5045 | [+3.6069, +5.5428] | 3.3179 (3.3179) | WORSE |
| TACTICAL:lon_err_rate | +0.0659 / +0.0855 | +0.0757 | [+0.0512, +0.1034] | 0.0463 (0.0463) | WORSE |
| TACTICAL:lat_err_rate | +0.0354 / +0.0388 | +0.0371 | [+0.0257, +0.0497] | 0.0013 (0.0013) | WORSE |
| TACTICAL:joint_err_rate | +0.0684 / +0.0946 | +0.0815 | [+0.0544, +0.1094] | 0.0467 (0.0467) | WORSE |
| TACTICAL:ade_tac_m | +0.2130 / +0.4994 | +0.3562 | [+0.1051, +0.6184] | 0.0497 (0.0497) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1143 / +0.0938 | -0.0103 | [-0.1611, +0.1370] | 0.0560 | OPEN |
| ade6_m|bulk | +0.4396 / +0.6089 | +0.5243 | [+0.2960, +0.8131] | 0.0054 | WORSE |
| ade6_m|all | +0.1383 / +0.3287 | +0.2335 | [+0.0664, +0.4081] | 0.0329 | WORSE |
| ade_tac_m|all | +0.2130 / +0.4994 | +0.3562 | [+0.1051, +0.6184] | 0.0497 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10B0_R0C10Hb20_s0 | 2.7607 | 3.1459 | 1.3815 | 1.0227 | 0.6590 |
| I10B0_R0C10Hb20_s1 | 2.8982 | 3.2241 | 1.5476 | 1.0286 | 0.7278 |
| I10B0_R0C10XHb20_s0 | 2.6464 | 3.1096 | 1.3833 | 0.8429 | 0.6432 |
| I10B0_R0C10XHb20_s1 | 2.7558 | 3.1537 | 1.5089 | 0.8373 | 0.7053 |
| I10B0_R0G0Hb20_s0 | 2.3692 | 2.9760 | 0.9507 | 1.0847 | 0.5398 |
| I10B0_R0G0Hb20_s1 | 2.3414 | 2.9428 | 0.9310 | 1.1110 | 0.5406 |
| I10B0_R0G50Hb20_s0 | 2.4741 | 2.9794 | 1.0897 | 1.0307 | 0.5827 |
| I10B0_R0G50Hb20_s1 | 2.4752 | 2.9638 | 1.1050 | 1.0462 | 0.6119 |
| I10B0_R0Hb20_s0 | 2.4794 | 2.9596 | 1.1540 | 1.0323 | 0.5940 |
| I10B0_R0Hb20_s1 | 2.5576 | 3.0145 | 1.2346 | 1.0582 | 0.6590 |
| I10B0_R0KHb20_s0 | 2.1706 | 2.9561 | 0.9580 | 0.8408 | 0.4644 |
| I10B0_R0KHb20_s1 | 2.1026 | 2.8794 | 0.9306 | 0.8092 | 0.4473 |
| I10RHK_R0C10Hb20_s0 | 2.8065 | 3.1647 | 1.3978 | 1.0140 | 0.6599 |
| I10RHK_R0C10Hb20_s1 | 2.8913 | 3.2351 | 1.5670 | 0.9564 | 0.7078 |
| I10RHK_R0C10XHb20_s0 | 2.6535 | 3.0675 | 1.3886 | 0.8069 | 0.6344 |
| I10RHK_R0C10XHb20_s1 | 2.7441 | 3.1205 | 1.5134 | 0.8060 | 0.6857 |
| I10RHK_R0G0Hb20_s0 | 2.3566 | 2.9649 | 0.9595 | 1.0351 | 0.5190 |
| I10RHK_R0G0Hb20_s1 | 2.2558 | 2.9062 | 0.9446 | 0.9794 | 0.4773 |
| I10RHK_R0G50Hb20_s0 | 2.5016 | 2.9955 | 1.1087 | 1.0107 | 0.5665 |
| I10RHK_R0G50Hb20_s1 | 2.4325 | 2.9477 | 1.1320 | 0.9425 | 0.5890 |
| I10RHK_R0Hb20_s0 | 2.5056 | 2.9882 | 1.1930 | 0.9955 | 0.5982 |
| I10RHK_R0Hb20_s1 | 2.5386 | 3.0442 | 1.2711 | 0.9533 | 0.6449 |
| I10RHK_R0KHb20_s0 | 2.1459 | 2.8763 | 0.9664 | 0.7901 | 0.4485 |
| I10RHK_R0KHb20_s1 | 2.0832 | 2.8242 | 0.9345 | 0.7757 | 0.4335 |
| I10RH_R0C10XHb20_s0 | 2.5152 | 3.1818 | 1.1873 | 0.8321 | 0.5661 |
| I10RH_R0C10XHb20_s1 | 2.4154 | 3.0267 | 1.1762 | 0.8046 | 0.5911 |
| I10RH_R0Hb20_s0 | 2.4906 | 3.0639 | 1.1169 | 1.0155 | 0.5381 |
| I10RH_R0Hb20_s1 | 2.4857 | 2.9907 | 1.1424 | 1.0584 | 0.5711 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
