# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 18}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "IPlx_RLA3"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045083, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## IPlx RLA3 minus IPox RLA3

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1774 / +0.1408 | +0.1591 | [+0.0055, +0.3381] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.6412 / +0.6103 | +0.6258 | [+0.2314, +1.0966] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0465 / -0.0489 | -0.0477 | [-0.0701, -0.0259] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.3590 / +0.2016 | +0.2803 | [+0.0515, +0.5187] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0271 / +0.0198 | +0.0234 | [+0.0124, +0.0370] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.9437 / -0.2862 | +0.3288 | [-1.1108, +1.8150] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0046 / +0.0121 | +0.0038 | [-0.0088, +0.0194] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0021 / +0.0075 | +0.0048 | [-0.0058, +0.0186] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0017 / +0.0113 | +0.0048 | [-0.0048, +0.0188] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.8519 / +0.8099 | +0.8309 | [+0.2777, +1.4897] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3446 / +0.3149 | +0.3297 | [+0.0557, +0.6916] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.9024 / +0.8780 | +0.8902 | [+0.3442, +1.5395] | 0.3535 | WORSE |
| ade6_m|all | +0.5990 / +0.5717 | +0.5853 | [+0.1968, +1.0477] | 0.4002 | WORSE |
| ade_tac_m|all | +0.8519 / +0.8099 | +0.8309 | [+0.2777, +1.4897] | 0.5955 | WORSE |

## IPix RLA3 minus IPox RLA3

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0724 / +0.4025 | +0.2374 | [+0.0287, +0.5620] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.2479 / +1.2586 | +0.7532 | [+0.1351, +1.6288] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0276 / -0.0632 | -0.0454 | [-0.0825, -0.0178] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.0393 / +0.5089 | +0.2741 | [+0.0125, +0.7122] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0052 / +0.0352 | +0.0202 | [+0.0033, +0.0442] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.1309 / +1.8961 | +1.0135 | [-0.0497, +2.8314] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0013 / +0.0383 | +0.0185 | [-0.0027, +0.0482] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0013 / +0.0146 | +0.0067 | [-0.0046, +0.0252] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0004 / +0.0329 | +0.0167 | [-0.0015, +0.0425] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.3233 / +1.7060 | +1.0147 | [+0.1701, +2.2248] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1653 / +0.7425 | +0.4539 | [+0.0681, +1.0470] | 0.4517 | WORSE |
| ade6_m|bulk | +0.2966 / +1.7436 | +1.0201 | [+0.1455, +2.2658] | 0.3535 | WORSE |
| ade6_m|all | +0.2252 / +1.1990 | +0.7121 | [+0.1190, +1.5633] | 0.4002 | WORSE |
| ade_tac_m|all | +0.3233 / +1.7060 | +1.0147 | [+0.1701, +2.2248] | 0.5955 | WORSE |

## IPix RLA3 minus IPcx RLA3

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0264 / +0.2003 | +0.0869 | [-0.0680, +0.2618] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.2849 / +0.4164 | +0.0657 | [-0.4244, +0.5613] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0305 / -0.0016 | +0.0145 | [-0.0094, +0.0404] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.2538 / +0.0638 | -0.0950 | [-0.3612, +0.2091] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0144 / +0.0084 | -0.0030 | [-0.0200, +0.0133] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0740 / +1.1682 | +0.6211 | [-0.5928, +1.6860] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0054 / +0.0154 | +0.0104 | [-0.0010, +0.0236] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0033 / +0.0079 | +0.0056 | [-0.0013, +0.0137] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0033 / +0.0158 | +0.0096 | [+0.0004, +0.0231] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.3648 / +0.5815 | +0.1084 | [-0.5619, +0.7791] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1593 / +0.1349 | -0.0122 | [-0.2498, +0.1794] | 0.4517 | OPEN |
| ade6_m|bulk | -0.3801 / +0.7288 | +0.1743 | [-0.5852, +1.0303] | 0.3535 | OPEN |
| ade6_m|all | -0.2600 / +0.4058 | +0.0729 | [-0.3986, +0.5440] | 0.4002 | OPEN |
| ade_tac_m|all | -0.3648 / +0.5815 | +0.1084 | [-0.5619, +0.7791] | 0.5955 | OPEN |

## IPix RLA3 minus IPlx RLA3

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1050 / +0.2618 | +0.0784 | [-0.1584, +0.3553] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.3934 / +0.6483 | +0.1274 | [-0.5698, +0.8750] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0189 / -0.0143 | +0.0023 | [-0.0284, +0.0290] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.3031 / +0.3297 | +0.0133 | [-0.4134, +0.5275] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0213 / +0.0185 | -0.0014 | [-0.0296, +0.0272] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.0409 / +1.7555 | +0.3573 | [-1.6820, +2.5020] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0033 / +0.0263 | +0.0148 | [-0.0008, +0.0340] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0033 / +0.0071 | +0.0019 | [-0.0081, +0.0138] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0021 / +0.0217 | +0.0119 | [-0.0006, +0.0288] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.5286 / +0.8961 | +0.1838 | [-0.7709, +1.2109] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1793 / +0.4276 | +0.1242 | [-0.2730, +0.6098] | 0.4517 | OPEN |
| ade6_m|bulk | -0.6058 / +0.8656 | +0.1299 | [-0.8955, +1.2651] | 0.3535 | OPEN |
| ade6_m|all | -0.3738 / +0.6274 | +0.1268 | [-0.5455, +0.8472] | 0.4002 | OPEN |
| ade_tac_m|all | -0.5286 / +0.8961 | +0.1838 | [-0.7709, +1.2109] | 0.5955 | OPEN |

## IPcx RLA3 minus IPox RLA3

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0987 / +0.2023 | +0.1505 | [-0.0055, +0.3448] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.5328 / +0.8422 | +0.6875 | [+0.2663, +1.2017] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0582 / -0.0616 | -0.0599 | [-0.0861, -0.0364] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2920 / +0.4434 | +0.3677 | [+0.1432, +0.6287] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0195 / +0.0264 | +0.0229 | [+0.0128, +0.0342] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.0301 / +0.7960 | +0.4131 | [-0.8601, +1.6824] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0067 / +0.0229 | +0.0081 | [-0.0125, +0.0304] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0046 / +0.0067 | +0.0010 | [-0.0108, +0.0173] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0029 / +0.0171 | +0.0071 | [-0.0061, +0.0240] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.6881 / +1.1245 | +0.9063 | [+0.3140, +1.6217] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3247 / +0.6075 | +0.4661 | [+0.1109, +0.9178] | 0.4517 | WORSE |
| ade6_m|bulk | +0.6768 / +1.0148 | +0.8458 | [+0.3368, +1.4519] | 0.3535 | WORSE |
| ade6_m|all | +0.4852 / +0.7933 | +0.6392 | [+0.2222, +1.1426] | 0.4002 | WORSE |
| ade_tac_m|all | +0.6881 / +1.1245 | +0.9063 | [+0.3140, +1.6217] | 0.5955 | WORSE |

## IPsx RLA3 minus IPix RLA3

Seeds read [0, 1] of [0, 1]. **Q3: COST-LATERAL COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.0332 / +1.3668 | +1.2000 | [+0.8791, +1.5370] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +3.4004 / +4.1695 | +3.7850 | [+2.9572, +4.6164] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0294 / +0.1184 | +0.0445 | [-0.0787, +0.1884] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +3.0542 / +3.8150 | +3.4346 | [+2.4151, +4.5556] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.1857 / +0.2001 | +0.1929 | [+0.1518, +0.2356] | 0.1528 (0.0760) | WORSE |
| LATERAL:yaw_rate_mae_degps | +13.3231 / +14.9976 | +14.1603 | [+10.5658, +17.8499] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0584 / +0.0709 | +0.0646 | [+0.0423, +0.0855] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0042 / +0.0258 | +0.0150 | [-0.0175, +0.0477] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0333 / +0.0425 | +0.0379 | [+0.0231, +0.0533] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +4.6004 / +5.7522 | +5.1763 | [+3.9820, +6.3788] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +2.9473 / +4.0057 | +3.4765 | [+2.5023, +4.4975] | 0.4517 | WORSE |
| ade6_m|bulk | +3.6995 / +4.2729 | +3.9862 | [+3.0031, +5.0508] | 0.3535 | WORSE |
| ade6_m|all | +3.2903 / +4.1275 | +3.7089 | [+2.8593, +4.5702] | 0.4002 | WORSE |
| ade_tac_m|all | +4.6004 / +5.7522 | +5.1763 | [+3.9820, +6.3788] | 0.5955 | WORSE |

## IPkx RLA3 minus IPix RLA3

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0016 / +0.0158 | +0.0071 | [-0.0086, +0.0273] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0069 / +0.0376 | +0.0154 | [-0.0247, +0.0642] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0009 / +0.0031 | +0.0020 | [+0.0000, +0.0041] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0012 / +0.0826 | +0.0407 | [-0.0075, +0.1280] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0002 / +0.0011 | +0.0006 | [-0.0019, +0.0038] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.0034 / +0.2325 | +0.1145 | [-0.0309, +0.3899] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0008 / -0.0004 | +0.0002 | [-0.0017, +0.0017] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0008 / +0.0013 | +0.0010 | [-0.0006, +0.0031] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0008 / +0.0013 | +0.0010 | [-0.0002, +0.0027] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0080 / +0.0563 | +0.0242 | [-0.0329, +0.0939] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0240 / +0.0191 | -0.0024 | [-0.0517, +0.0394] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0164 / +0.0639 | +0.0401 | [-0.0000, +0.1005] | 0.3535 | OPEN |
| ade6_m|all | -0.0055 / +0.0395 | +0.0170 | [-0.0231, +0.0657] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0080 / +0.0563 | +0.0242 | [-0.0329, +0.0939] | 0.5955 | OPEN |

## IPkx RLA3 minus IPox RLA3

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0708 / +0.4183 | +0.2445 | [+0.0284, +0.5811] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.2410 / +1.2962 | +0.7686 | [+0.1326, +1.6775] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0268 / -0.0600 | -0.0434 | [-0.0795, -0.0171] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.0380 / +0.5915 | +0.3148 | [+0.0133, +0.7919] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0054 / +0.0363 | +0.0209 | [+0.0036, +0.0455] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.1275 / +2.1286 | +1.1280 | [-0.0396, +3.0280] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0004 / +0.0379 | +0.0188 | [-0.0017, +0.0475] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0004 / +0.0158 | +0.0077 | [-0.0035, +0.0263] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0013 / +0.0342 | +0.0177 | [-0.0004, +0.0440] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.3154 / +1.7623 | +1.0388 | [+0.1639, +2.2920] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1413 / +0.7616 | +0.4515 | [+0.0545, +1.0615] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.3131 / +1.8075 | +1.0603 | [+0.1509, +2.3517] | 0.3535 | WORSE |
| ade6_m|all | +0.2196 / +1.2385 | +0.7291 | [+0.1148, +1.6109] | 0.4002 | WORSE |
| ade_tac_m|all | +0.3154 / +1.7623 | +1.0388 | [+0.1639, +2.2920] | 0.5955 | WORSE |

## IPkx RLA3 minus IPcx RLA3

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0280 / +0.2160 | +0.0940 | [-0.0732, +0.2805] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.2918 / +0.4540 | +0.0811 | [-0.4418, +0.6127] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0314 / +0.0016 | +0.0165 | [-0.0058, +0.0418] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.2550 / +0.1465 | -0.0543 | [-0.3564, +0.2906] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0142 / +0.0096 | -0.0023 | [-0.0195, +0.0147] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0706 / +1.3974 | +0.7340 | [-0.5665, +1.8570] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0063 / +0.0150 | +0.0106 | [-0.0002, +0.0235] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0042 / +0.0092 | +0.0067 | [+0.0000, +0.0150] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0042 / +0.0171 | +0.0106 | [+0.0010, +0.0248] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.3727 / +0.6378 | +0.1325 | [-0.5829, +0.8620] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1833 / +0.1540 | -0.0146 | [-0.2919, +0.2081] | 0.4517 | OPEN |
| ade6_m|bulk | -0.3637 / +0.7927 | +0.2145 | [-0.5712, +1.1177] | 0.3535 | OPEN |
| ade6_m|all | -0.2656 / +0.4453 | +0.0898 | [-0.4143, +0.6022] | 0.4002 | OPEN |
| ade_tac_m|all | -0.3727 / +0.6378 | +0.1325 | [-0.5829, +0.8620] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| IPcx_RLA3_s0 | 3.2749 | 3.5892 | 1.8535 | 1.2264 | 0.7028 |
| IPcx_RLA3_s1 | 3.5155 | 3.8140 | 1.9722 | 1.1108 | 0.6561 |
| IPix_RLA3_s0 | 3.0149 | 3.4299 | 1.8271 | 1.2569 | 0.7061 |
| IPix_RLA3_s1 | 3.9213 | 3.9490 | 2.1725 | 1.1092 | 0.6719 |
| IPkx_RLA3_s0 | 3.0093 | 3.4059 | 1.8255 | 1.2578 | 0.7070 |
| IPkx_RLA3_s1 | 3.9608 | 3.9681 | 2.1883 | 1.1123 | 0.6732 |
| IPlx_RLA3_s0 | 3.3887 | 3.6092 | 1.9321 | 1.2381 | 0.7040 |
| IPlx_RLA3_s1 | 3.2939 | 3.5214 | 1.9107 | 1.1235 | 0.6503 |
| IPox_RLA3_s0 | 2.7897 | 3.2645 | 1.7548 | 1.2846 | 0.7057 |
| IPox_RLA3_s1 | 2.7222 | 3.2065 | 1.7700 | 1.1724 | 0.6390 |
| IPsx_RLA3_s0 | 6.3052 | 6.3771 | 2.8603 | 1.2275 | 0.7395 |
| IPsx_RLA3_s1 | 8.0488 | 7.9547 | 3.5393 | 1.2276 | 0.7145 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
