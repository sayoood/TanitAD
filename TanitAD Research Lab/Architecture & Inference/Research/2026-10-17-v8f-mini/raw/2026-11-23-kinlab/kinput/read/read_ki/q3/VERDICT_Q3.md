# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 28}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10KI_R0C10XHb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.2307178299135948, "D_read": 0.23071782991359482, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10KI_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10KI R0C10XHb20 minus I10B0 R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL GAIN-LONGITUDINAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0427 / -0.1265 | -0.0846 | [-0.1784, -0.0084] | 0.0769 (0.0769) | BETTER |
| LONGITUDINAL:along_mae_m | -0.0132 / -0.0963 | -0.0548 | [-0.1574, +0.0265] | 0.0453 (0.0453) | OPEN |
| LATERAL:cross_mae_m | +0.0306 / +0.0051 | +0.0178 | [-0.0020, +0.0421] | 0.0596 (0.0596) | OPEN |
| LATERAL:heading_mae_deg | +0.8893 / +1.3913 | +1.1403 | [+0.7493, +1.6347] | 0.9181 (0.9181) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0202 / -0.0069 | +0.0066 | [-0.0233, +0.0295] | 0.0194 (0.0194) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.8299 / -0.1502 | +0.3399 | [-0.9420, +1.3869] | 0.5567 (0.5567) | OPEN |
| TACTICAL:lon_err_rate | +0.0046 / -0.0117 | -0.0035 | [-0.0309, +0.0204] | 0.0725 (0.0725) | OPEN |
| TACTICAL:lat_err_rate | +0.0200 / +0.0538 | +0.0369 | [+0.0123, +0.0657] | 0.0129 (0.0129) | WORSE |
| TACTICAL:joint_err_rate | -0.0017 / -0.0238 | -0.0127 | [-0.0409, +0.0131] | 0.0442 (0.0442) | OPEN |
| TACTICAL:ade_tac_m | +0.0170 / -0.1313 | -0.0571 | [-0.2157, +0.0738] | 0.1131 (0.1131) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1418 / +0.0701 | +0.1059 | [+0.0430, +0.1660] | 0.0544 | WORSE |
| ade6_m|bulk | -0.1423 / -0.2745 | -0.2084 | [-0.3901, -0.0629] | 0.0997 | BETTER |
| ade6_m|all | +0.0122 / -0.0871 | -0.0374 | [-0.1435, +0.0500] | 0.0751 | OPEN |
| ade_tac_m|all | +0.0170 / -0.1313 | -0.0571 | [-0.2157, +0.0738] | 0.1131 | OPEN |

## I10KI R0C10XHb20 minus I10KI R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3647 / +0.4352 | +0.3999 | [+0.3180, +0.4823] | 0.0769 (0.0769) | WORSE |
| LONGITUDINAL:along_mae_m | +0.4845 / +0.5624 | +0.5235 | [+0.4138, +0.6391] | 0.0453 (0.0453) | WORSE |
| LATERAL:cross_mae_m | +0.0098 / +0.0166 | +0.0132 | [-0.0036, +0.0307] | 0.0596 (0.0596) | OPEN |
| LATERAL:heading_mae_deg | +1.9450 / +3.2649 | +2.6049 | [+1.7042, +3.6445] | 0.9181 (0.9181) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1662 / +0.1519 | +0.1591 | [+0.1296, +0.1902] | 0.0194 (0.0194) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.9556 / +14.3631 | +13.6593 | [+11.4166, +15.9903] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | +0.1688 / +0.2313 | +0.2001 | [+0.1485, +0.2589] | 0.0725 (0.0725) | WORSE |
| TACTICAL:lat_err_rate | +0.0838 / +0.1130 | +0.0984 | [+0.0694, +0.1288] | 0.0129 (0.0129) | WORSE |
| TACTICAL:joint_err_rate | +0.1617 / +0.2230 | +0.1924 | [+0.1420, +0.2476] | 0.0442 (0.0442) | WORSE |
| TACTICAL:ade_tac_m | +0.6802 / +0.8040 | +0.7421 | [+0.5735, +0.9156] | 0.1131 (0.1131) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2529 / +0.3051 | +0.2790 | [+0.1794, +0.3834] | 0.0544 | WORSE |
| ade6_m|bulk | +0.6928 / +0.8114 | +0.7521 | [+0.5784, +0.9321] | 0.0997 | WORSE |
| ade6_m|all | +0.4535 / +0.5360 | +0.4947 | [+0.3824, +0.6104] | 0.0751 | WORSE |
| ade_tac_m|all | +0.6802 / +0.8040 | +0.7421 | [+0.5735, +0.9156] | 0.1131 | WORSE |

## I10KI R0G50Hb20 minus I10KI R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1425 / +0.1583 | +0.1504 | [+0.1124, +0.1911] | 0.0769 (0.0769) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1981 / +0.2149 | +0.2065 | [+0.1564, +0.2597] | 0.0453 (0.0453) | WORSE |
| LATERAL:cross_mae_m | -0.0298 / -0.0591 | -0.0444 | [-0.0710, -0.0217] | 0.0596 (0.0596) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.3375 / +0.8261 | +0.5818 | [+0.2200, +0.9602] | 0.9181 (0.9181) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0683 / +0.0747 | +0.0715 | [+0.0567, +0.0898] | 0.0194 (0.0194) | WORSE |
| LATERAL:yaw_rate_mae_degps | +5.7321 / +7.4011 | +6.5666 | [+5.1176, +8.3395] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | +0.0875 / +0.1246 | +0.1061 | [+0.0690, +0.1451] | 0.0725 (0.0725) | WORSE |
| TACTICAL:lat_err_rate | +0.0021 / +0.0004 | +0.0013 | [-0.0083, +0.0106] | 0.0129 (0.0129) | OPEN |
| TACTICAL:joint_err_rate | +0.0563 / +0.0650 | +0.0607 | [+0.0400, +0.0829] | 0.0442 (0.0442) | WORSE |
| TACTICAL:ade_tac_m | +0.2268 / +0.2108 | +0.2188 | [+0.1560, +0.2850] | 0.1131 (0.1131) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0566 / +0.0272 | +0.0419 | [-0.0189, +0.1041] | 0.0544 | OPEN |
| ade6_m|bulk | +0.2640 / +0.2758 | +0.2699 | [+0.2168, +0.3249] | 0.0997 | WORSE |
| ade6_m|all | +0.1512 / +0.1405 | +0.1459 | [+0.1040, +0.1900] | 0.0751 | WORSE |
| ade_tac_m|all | +0.2268 / +0.2108 | +0.2188 | [+0.1560, +0.2850] | 0.1131 | WORSE |

## I10KI R0C10Hb20 minus I10KI R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2376 / +0.3543 | +0.2959 | [+0.2113, +0.3814] | 0.0769 (0.0769) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3330 / +0.4367 | +0.3849 | [+0.2947, +0.4729] | 0.0453 (0.0453) | WORSE |
| LATERAL:cross_mae_m | +0.0110 / +0.0060 | +0.0085 | [-0.0048, +0.0207] | 0.0596 (0.0596) | OPEN |
| LATERAL:heading_mae_deg | +2.2814 / +2.8449 | +2.5632 | [+2.0144, +3.1929] | 0.9181 (0.9181) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1292 / +0.1321 | +0.1307 | [+0.1063, +0.1569] | 0.0194 (0.0194) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.4354 / +14.8453 | +13.6404 | [+11.0571, +16.4392] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | +0.0796 / +0.1488 | +0.1142 | [+0.0674, +0.1663] | 0.0725 (0.0725) | WORSE |
| TACTICAL:lat_err_rate | +0.0654 / +0.0446 | +0.0550 | [+0.0346, +0.0773] | 0.0129 (0.0129) | WORSE |
| TACTICAL:joint_err_rate | +0.0734 / +0.1150 | +0.0942 | [+0.0644, +0.1294] | 0.0442 (0.0442) | WORSE |
| TACTICAL:ade_tac_m | +0.4534 / +0.5835 | +0.5184 | [+0.4023, +0.6317] | 0.1131 (0.1131) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2262 / +0.3138 | +0.2700 | [+0.2016, +0.3472] | 0.0544 | WORSE |
| ade6_m|bulk | +0.3930 / +0.4787 | +0.4358 | [+0.3304, +0.5409] | 0.0997 | WORSE |
| ade6_m|all | +0.3023 / +0.3890 | +0.3456 | [+0.2682, +0.4212] | 0.0751 | WORSE |
| ade_tac_m|all | +0.4534 / +0.5835 | +0.5184 | [+0.4023, +0.6317] | 0.1131 | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4253 / +0.5783 | +0.5018 | [+0.3633, +0.6525] | 0.0769 (0.0769) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5148 / +0.6779 | +0.5963 | [+0.4294, +0.7851] | 0.0453 (0.0453) | WORSE |
| LATERAL:cross_mae_m | +0.0021 / +0.0281 | +0.0151 | [-0.0064, +0.0392] | 0.0596 (0.0596) | OPEN |
| LATERAL:heading_mae_deg | +1.2984 / +2.1302 | +1.7143 | [+1.1487, +2.3855] | 0.9181 (0.9181) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1632 / +0.1844 | +0.1738 | [+0.1396, +0.2133] | 0.0194 (0.0194) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.8306 / +15.7625 | +14.2966 | [+11.4789, +17.5884] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | +0.1734 / +0.2543 | +0.2138 | [+0.1512, +0.2812] | 0.0725 (0.0725) | WORSE |
| TACTICAL:lat_err_rate | +0.0646 / +0.0638 | +0.0642 | [+0.0500, +0.0794] | 0.0129 (0.0129) | WORSE |
| TACTICAL:joint_err_rate | +0.1788 / +0.2580 | +0.2184 | [+0.1561, +0.2826] | 0.0442 (0.0442) | WORSE |
| TACTICAL:ade_tac_m | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.1131 (0.1131) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1535 / +0.2743 | +0.2139 | [+0.0940, +0.3374] | 0.0544 | WORSE |
| ade6_m|bulk | +0.8602 / +1.1053 | +0.9827 | [+0.7095, +1.3144] | 0.0997 | WORSE |
| ade6_m|all | +0.4758 / +0.6532 | +0.5645 | [+0.3925, +0.7571] | 0.0751 | WORSE |
| ade_tac_m|all | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.1131 | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1390 / +0.1740 | +0.1565 | [+0.1087, +0.2098] | 0.0769 (0.0769) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1801 / +0.2231 | +0.2016 | [+0.1391, +0.2668] | 0.0453 (0.0453) | WORSE |
| LATERAL:cross_mae_m | -0.0540 / -0.0648 | -0.0594 | [-0.0833, -0.0397] | 0.0596 (0.0596) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2941 / +0.8362 | +0.5652 | [+0.2123, +0.9892] | 0.9181 (0.9181) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0682 / +0.0883 | +0.0783 | [+0.0594, +0.1021] | 0.0194 (0.0194) | WORSE |
| LATERAL:yaw_rate_mae_degps | +5.7579 / +7.9876 | +6.8727 | [+5.1833, +9.1092] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | +0.0767 / +0.1463 | +0.1115 | [+0.0594, +0.1676] | 0.0725 (0.0725) | WORSE |
| TACTICAL:lat_err_rate | -0.0113 / -0.0129 | -0.0121 | [-0.0231, -0.0017] | 0.0129 (0.0129) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0429 / +0.0713 | +0.0571 | [+0.0304, +0.0869] | 0.0442 (0.0442) | WORSE |
| TACTICAL:ade_tac_m | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.1131 (0.1131) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0033 / +0.0210 | +0.0122 | [-0.0397, +0.0695] | 0.0544 | OPEN |
| ade6_m|bulk | +0.2261 / +0.2684 | +0.2473 | [+0.1874, +0.3092] | 0.0997 | WORSE |
| ade6_m|all | +0.1049 / +0.1338 | +0.1194 | [+0.0744, +0.1683] | 0.0751 | WORSE |
| ade_tac_m|all | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.1131 | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2919 / +0.4426 | +0.3672 | [+0.2558, +0.4843] | 0.0769 (0.0769) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3451 / +0.5030 | +0.4241 | [+0.2983, +0.5604] | 0.0453 (0.0453) | WORSE |
| LATERAL:cross_mae_m | -0.0080 / -0.0177 | -0.0128 | [-0.0332, +0.0072] | 0.0596 (0.0596) | OPEN |
| LATERAL:heading_mae_deg | +1.4153 / +1.5714 | +1.4933 | [+1.2106, +1.8094] | 0.9181 (0.9181) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1058 / +0.1205 | +0.1131 | [+0.0895, +0.1417] | 0.0194 (0.0194) | WORSE |
| LATERAL:yaw_rate_mae_degps | +11.2146 / +14.0424 | +12.6285 | [+10.0108, +15.6291] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | +0.0942 / +0.1105 | +0.1023 | [+0.0763, +0.1301] | 0.0725 (0.0725) | WORSE |
| TACTICAL:lat_err_rate | +0.0538 / +0.0396 | +0.0467 | [+0.0311, +0.0628] | 0.0129 (0.0129) | WORSE |
| TACTICAL:joint_err_rate | +0.0763 / +0.1159 | +0.0961 | [+0.0650, +0.1312] | 0.0442 (0.0442) | WORSE |
| TACTICAL:ade_tac_m | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.1131 (0.1131) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1666 / +0.2602 | +0.2134 | [+0.1412, +0.2884] | 0.0544 | WORSE |
| ade6_m|bulk | +0.4298 / +0.6170 | +0.5234 | [+0.3460, +0.7226] | 0.0997 | WORSE |
| ade6_m|all | +0.2866 / +0.4229 | +0.3548 | [+0.2432, +0.4718] | 0.0751 | WORSE |
| ade_tac_m|all | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.1131 | WORSE |

## I10KI R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3826 / +0.4518 | +0.4172 | [+0.3341, +0.5011] | 0.0769 (0.0769) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5015 / +0.5816 | +0.5416 | [+0.4255, +0.6624] | 0.0453 (0.0453) | WORSE |
| LATERAL:cross_mae_m | +0.0327 / +0.0332 | +0.0329 | [+0.0122, +0.0535] | 0.0596 (0.0596) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +2.1782 / +3.5071 | +2.8426 | [+1.9196, +3.8938] | 0.9181 (0.9181) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1751 / +0.1710 | +0.1730 | [+0.1423, +0.2053] | 0.0194 (0.0194) | WORSE |
| LATERAL:yaw_rate_mae_degps | +13.2871 / +15.3333 | +14.3102 | [+11.8591, +16.9604] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | +0.1780 / +0.2426 | +0.2103 | [+0.1555, +0.2704] | 0.0725 (0.0725) | WORSE |
| TACTICAL:lat_err_rate | +0.0846 / +0.1175 | +0.1011 | [+0.0713, +0.1342] | 0.0129 (0.0129) | WORSE |
| TACTICAL:joint_err_rate | +0.1772 / +0.2343 | +0.2057 | [+0.1585, +0.2593] | 0.0442 (0.0442) | WORSE |
| TACTICAL:ade_tac_m | +0.7307 / +0.8486 | +0.7896 | [+0.6180, +0.9684] | 0.1131 (0.1131) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2953 / +0.3444 | +0.3198 | [+0.2170, +0.4296] | 0.0544 | WORSE |
| ade6_m|bulk | +0.7178 / +0.8308 | +0.7743 | [+0.6028, +0.9626] | 0.0997 | WORSE |
| ade6_m|all | +0.4880 / +0.5662 | +0.5271 | [+0.4128, +0.6460] | 0.0751 | WORSE |
| ade_tac_m|all | +0.7307 / +0.8486 | +0.7896 | [+0.6180, +0.9684] | 0.1131 | WORSE |

## I10KI R0Hb20 minus I10B0 R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0188 / +0.0151 | +0.0170 | [-0.0136, +0.0462] | 0.0769 (0.0769) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0631 / +0.0382 | +0.0506 | [+0.0123, +0.0868] | 0.0453 (0.0453) | WORSE |
| LATERAL:cross_mae_m | -0.0062 / +0.0276 | +0.0107 | [-0.0184, +0.0400] | 0.0596 (0.0596) | OPEN |
| LATERAL:heading_mae_deg | +0.6167 / +0.7647 | +0.6907 | [+0.4757, +0.9703] | 0.9181 (0.9181) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0311 / +0.0127 | +0.0219 | [+0.0026, +0.0376] | 0.0194 (0.0194) | WORSE |
| LATERAL:yaw_rate_mae_degps | +2.4037 / +0.6409 | +1.5223 | [-0.0617, +2.9411] | 0.5567 (0.5567) | OPEN |
| TACTICAL:lon_err_rate | -0.0121 / -0.0104 | -0.0113 | [-0.0359, +0.0125] | 0.0725 (0.0725) | OPEN |
| TACTICAL:lat_err_rate | +0.0221 / +0.0225 | +0.0223 | [+0.0071, +0.0381] | 0.0129 (0.0129) | WORSE |
| TACTICAL:joint_err_rate | -0.0046 / -0.0254 | -0.0150 | [-0.0396, +0.0058] | 0.0442 (0.0442) | OPEN |
| TACTICAL:ade_tac_m | +0.0897 / +0.0856 | +0.0877 | [+0.0459, +0.1297] | 0.1131 (0.1131) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0741 / +0.0736 | +0.0738 | [+0.0377, +0.1098] | 0.0544 | WORSE |
| ade6_m|bulk | +0.0447 / +0.0384 | +0.0415 | [+0.0006, +0.0818] | 0.0997 | INSIDE-SPREAD |
| ade6_m|all | +0.0607 / +0.0575 | +0.0591 | [+0.0310, +0.0871] | 0.0751 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0897 / +0.0856 | +0.0877 | [+0.0459, +0.1297] | 0.1131 | INSIDE-SPREAD |

## I10KI R0C10Hb20 minus I10B0 R0C10Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0415 / -0.0899 | -0.0657 | [-0.1341, -0.0081] | 0.0769 (0.0769) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0298 / -0.0582 | -0.0142 | [-0.1055, +0.0643] | 0.0453 (0.0453) | OPEN |
| LATERAL:cross_mae_m | +0.0205 / +0.0572 | +0.0388 | [+0.0031, +0.0778] | 0.0596 (0.0596) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.0668 / +1.5307 | +1.2988 | [+0.8995, +1.7888] | 0.9181 (0.9181) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0393 / +0.0201 | +0.0297 | [+0.0043, +0.0484] | 0.0194 (0.0194) | WORSE |
| LATERAL:yaw_rate_mae_degps | +1.8695 / +1.4302 | +1.6498 | [+0.4224, +2.7229] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | -0.0004 / +0.0292 | +0.0144 | [-0.0127, +0.0400] | 0.0725 (0.0725) | OPEN |
| TACTICAL:lat_err_rate | +0.0133 / +0.0383 | +0.0258 | [+0.0033, +0.0503] | 0.0129 (0.0129) | WORSE |
| TACTICAL:joint_err_rate | +0.0079 / +0.0079 | +0.0079 | [-0.0058, +0.0211] | 0.0442 (0.0442) | OPEN |
| TACTICAL:ade_tac_m | +0.0999 / +0.0166 | +0.0582 | [-0.0381, +0.1341] | 0.1131 (0.1131) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1244 / +0.0879 | +0.1062 | [+0.0562, +0.1501] | 0.0544 | WORSE |
| ade6_m|bulk | -0.0005 / -0.0796 | -0.0400 | [-0.1475, +0.0426] | 0.0997 | OPEN |
| ade6_m|all | +0.0675 / +0.0115 | +0.0395 | [-0.0249, +0.0903] | 0.0751 | OPEN |
| ade_tac_m|all | +0.0999 / +0.0166 | +0.0582 | [-0.0381, +0.1341] | 0.1131 | OPEN |

## I10KI R0G50Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0128 / -0.0015 | +0.0056 | [-0.0163, +0.0251] | 0.0769 (0.0769) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0418 / +0.0081 | +0.0250 | [-0.0079, +0.0580] | 0.0453 (0.0453) | OPEN |
| LATERAL:cross_mae_m | +0.0015 / +0.0336 | +0.0176 | [-0.0129, +0.0489] | 0.0596 (0.0596) | OPEN |
| LATERAL:heading_mae_deg | +0.1657 / +0.2850 | +0.2253 | [+0.0362, +0.4381] | 0.9181 (0.9181) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0114 / +0.0076 | +0.0095 | [-0.0004, +0.0181] | 0.0194 (0.0194) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.4257 / +0.5264 | +0.4761 | [-0.0563, +1.0283] | 0.5567 (0.5567) | OPEN |
| TACTICAL:lon_err_rate | +0.0142 / -0.0092 | +0.0025 | [-0.0238, +0.0221] | 0.0725 (0.0725) | OPEN |
| TACTICAL:lat_err_rate | +0.0017 / +0.0333 | +0.0175 | [-0.0075, +0.0417] | 0.0129 (0.0129) | OPEN |
| TACTICAL:joint_err_rate | +0.0108 / +0.0088 | +0.0098 | [-0.0027, +0.0219] | 0.0442 (0.0442) | OPEN |
| TACTICAL:ade_tac_m | +0.0764 / +0.0675 | +0.0720 | [+0.0388, +0.1044] | 0.1131 (0.1131) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0648 / +0.0343 | +0.0496 | [+0.0122, +0.0848] | 0.0544 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0364 / +0.0587 | +0.0475 | [+0.0192, +0.0805] | 0.0997 | INSIDE-SPREAD |
| ade6_m|all | +0.0518 / +0.0455 | +0.0486 | [+0.0262, +0.0704] | 0.0751 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0764 / +0.0675 | +0.0720 | [+0.0388, +0.1044] | 0.1131 | INSIDE-SPREAD |

## I10KI R0G0Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0093 / +0.0142 | +0.0117 | [+0.0044, +0.0200] | 0.0769 (0.0769) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0239 / +0.0163 | +0.0201 | [+0.0039, +0.0363] | 0.0453 (0.0453) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0227 / +0.0279 | +0.0026 | [-0.0345, +0.0406] | 0.0596 (0.0596) | OPEN |
| LATERAL:heading_mae_deg | +0.1131 / +0.3479 | +0.2305 | [+0.0488, +0.4111] | 0.9181 (0.9181) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0112 / +0.0225 | +0.0168 | [+0.0086, +0.0287] | 0.0194 (0.0194) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.4700 / +1.1708 | +0.8204 | [+0.3640, +1.4357] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | +0.0033 / +0.0125 | +0.0079 | [-0.0015, +0.0179] | 0.0725 (0.0725) | OPEN |
| TACTICAL:lat_err_rate | -0.0117 / +0.0200 | +0.0042 | [-0.0169, +0.0254] | 0.0129 (0.0129) | OPEN |
| TACTICAL:joint_err_rate | -0.0025 / +0.0150 | +0.0063 | [-0.0071, +0.0202] | 0.0442 (0.0442) | OPEN |
| TACTICAL:ade_tac_m | +0.0070 / +0.0574 | +0.0322 | [-0.0098, +0.0775] | 0.1131 (0.1131) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0115 / +0.0282 | +0.0199 | [-0.0079, +0.0494] | 0.0544 | OPEN |
| ade6_m|bulk | -0.0015 / +0.0513 | +0.0249 | [-0.0151, +0.0696] | 0.0997 | OPEN |
| ade6_m|all | +0.0056 / +0.0387 | +0.0222 | [-0.0059, +0.0522] | 0.0751 | OPEN |
| ade_tac_m|all | +0.0070 / +0.0574 | +0.0322 | [-0.0098, +0.0775] | 0.1131 | OPEN |

## I10KI R0KHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0179 / +0.0166 | +0.0173 | [+0.0101, +0.0241] | 0.0769 (0.0769) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0170 / +0.0192 | +0.0181 | [+0.0039, +0.0313] | 0.0453 (0.0453) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0229 / +0.0166 | +0.0197 | [+0.0073, +0.0352] | 0.0596 (0.0596) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2009 / +0.2480 | +0.2245 | [+0.1419, +0.3107] | 0.9181 (0.9181) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0096 / +0.0236 | +0.0166 | [+0.0075, +0.0299] | 0.0194 (0.0194) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.4110 / +1.1259 | +0.7684 | [+0.3226, +1.3881] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | +0.0092 / +0.0113 | +0.0102 | [+0.0021, +0.0188] | 0.0725 (0.0725) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0008 / +0.0046 | +0.0027 | [-0.0042, +0.0096] | 0.0129 (0.0129) | OPEN |
| TACTICAL:joint_err_rate | +0.0154 / +0.0113 | +0.0133 | [+0.0052, +0.0221] | 0.0442 (0.0442) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0504 / +0.0446 | +0.0475 | [+0.0227, +0.0715] | 0.1131 (0.1131) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0424 / +0.0393 | +0.0408 | [+0.0151, +0.0670] | 0.0544 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0250 / +0.0194 | +0.0222 | [+0.0069, +0.0382] | 0.0997 | INSIDE-SPREAD |
| ade6_m|all | +0.0345 / +0.0302 | +0.0324 | [+0.0156, +0.0486] | 0.0751 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0504 / +0.0446 | +0.0475 | [+0.0227, +0.0715] | 0.1131 | INSIDE-SPREAD |

## I10KI R0Hb20 minus I10RH R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0559 / +0.1073 | +0.0816 | [+0.0348, +0.1285] | 0.0769 (0.0769) | WORSE |
| LONGITUDINAL:along_mae_m | +0.0397 / +0.1167 | +0.0782 | [-0.0001, +0.1504] | 0.0453 (0.0453) | OPEN |
| LATERAL:cross_mae_m | +0.0107 / +0.0274 | +0.0190 | [-0.0050, +0.0455] | 0.0596 (0.0596) | OPEN |
| LATERAL:heading_mae_deg | +1.5548 / +1.8681 | +1.7114 | [+1.3729, +2.1250] | 0.9181 (0.9181) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0926 / +0.0772 | +0.0849 | [+0.0646, +0.1050] | 0.0194 (0.0194) | WORSE |
| LATERAL:yaw_rate_mae_degps | +6.5334 / +5.1640 | +5.8487 | [+4.4392, +7.3213] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | +0.0383 / +0.0588 | +0.0486 | [+0.0256, +0.0694] | 0.0725 (0.0725) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0738 / +0.0805 | +0.0771 | [+0.0600, +0.0969] | 0.0129 (0.0129) | WORSE |
| TACTICAL:joint_err_rate | +0.0513 / +0.0625 | +0.0569 | [+0.0396, +0.0755] | 0.0442 (0.0442) | WORSE |
| TACTICAL:ade_tac_m | +0.0818 / +0.2026 | +0.1422 | [+0.0298, +0.2447] | 0.1131 (0.1131) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0302 / +0.0974 | +0.0336 | [-0.0874, +0.1463] | 0.0544 | OPEN |
| ade6_m|bulk | +0.1446 / +0.1677 | +0.1562 | [+0.1114, +0.2045] | 0.0997 | WORSE |
| ade6_m|all | +0.0495 / +0.1295 | +0.0895 | [+0.0144, +0.1583] | 0.0751 | WORSE |
| ade_tac_m|all | +0.0818 / +0.2026 | +0.1422 | [+0.0298, +0.2447] | 0.1131 | WORSE |

## I10KI R0C10XHb20 minus I10RH R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1533 / +0.2062 | +0.1797 | [+0.1200, +0.2363] | 0.0769 (0.0769) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1096 / +0.2297 | +0.1697 | [+0.0594, +0.2746] | 0.0453 (0.0453) | WORSE |
| LATERAL:cross_mae_m | +0.0414 / +0.0378 | +0.0396 | [+0.0183, +0.0633] | 0.0596 (0.0596) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.6682 / +2.2711 | +1.9697 | [+1.4599, +2.5599] | 0.9181 (0.9181) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0987 / +0.0774 | +0.0880 | [+0.0626, +0.1145] | 0.0194 (0.0194) | WORSE |
| LATERAL:yaw_rate_mae_degps | +5.6137 / +5.2793 | +5.4465 | [+4.2614, +6.6452] | 0.5567 (0.5567) | WORSE |
| TACTICAL:lon_err_rate | +0.0713 / +0.0880 | +0.0796 | [+0.0563, +0.1055] | 0.0725 (0.0725) | WORSE |
| TACTICAL:lat_err_rate | +0.0763 / +0.0963 | +0.0863 | [+0.0644, +0.1098] | 0.0129 (0.0129) | WORSE |
| TACTICAL:joint_err_rate | +0.0754 / +0.0905 | +0.0830 | [+0.0608, +0.1076] | 0.0442 (0.0442) | WORSE |
| TACTICAL:ade_tac_m | +0.2226 / +0.3884 | +0.3055 | [+0.1477, +0.4539] | 0.1131 (0.1131) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0696 / +0.1971 | +0.1333 | [+0.0140, +0.2452] | 0.0544 | WORSE |
| ade6_m|bulk | +0.2314 / +0.3205 | +0.2759 | [+0.1472, +0.4092] | 0.0997 | WORSE |
| ade6_m|all | +0.1434 / +0.2534 | +0.1984 | [+0.0925, +0.2969] | 0.0751 | WORSE |
| ade_tac_m|all | +0.2226 / +0.3884 | +0.3055 | [+0.1477, +0.4539] | 0.1131 | WORSE |

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
| I10KI_R0C10Hb20_s0 | 2.8282 | 3.2703 | 1.3400 | 1.0432 | 0.6669 |
| I10KI_R0C10Hb20_s1 | 2.9097 | 3.3120 | 1.4578 | 1.0858 | 0.7357 |
| I10KI_R0C10XHb20_s0 | 2.6586 | 3.2514 | 1.3406 | 0.8735 | 0.6415 |
| I10KI_R0C10XHb20_s1 | 2.6688 | 3.2238 | 1.3824 | 0.8424 | 0.6815 |
| I10KI_R0G0Hb20_s0 | 2.3747 | 2.9875 | 0.9600 | 1.0620 | 0.5373 |
| I10KI_R0G0Hb20_s1 | 2.3801 | 2.9710 | 0.9452 | 1.1389 | 0.5556 |
| I10KI_R0G50Hb20_s0 | 2.5259 | 3.0441 | 1.1025 | 1.0322 | 0.5936 |
| I10KI_R0G50Hb20_s1 | 2.5207 | 2.9982 | 1.1035 | 1.0798 | 0.6207 |
| I10KI_R0Hb20_s0 | 2.5401 | 3.0337 | 1.1728 | 1.0262 | 0.5894 |
| I10KI_R0Hb20_s1 | 2.6152 | 3.0881 | 1.2497 | 1.0857 | 0.6336 |
| I10KI_R0KHb20_s0 | 2.2051 | 2.9985 | 0.9760 | 0.8636 | 0.4798 |
| I10KI_R0KHb20_s1 | 2.1328 | 2.9187 | 0.9472 | 0.8258 | 0.4585 |
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
