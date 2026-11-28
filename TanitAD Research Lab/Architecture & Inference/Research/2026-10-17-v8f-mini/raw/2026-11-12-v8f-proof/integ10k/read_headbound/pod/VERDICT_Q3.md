# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 22}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0Hb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867476, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0Hb20 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2354 / -0.2438 | -0.2396 | [-0.3174, -0.1728] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.0414 / -0.0841 | -0.0628 | [-0.0994, -0.0280] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.1410 / -0.1690 | -0.1550 | [-0.2037, -0.1129] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -1.3582 / -2.0402 | -1.6992 | [-2.4893, -1.0494] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0082 / -0.0058 | +0.0012 | [-0.0120, +0.0141] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -4.4756 / -4.6906 | -4.5831 | [-6.5492, -2.9709] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0042 / -0.0046 | -0.0044 | [-0.0163, +0.0069] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0409 / -0.0417 | -0.0413 | [-0.0559, -0.0281] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0138 / -0.0154 | -0.0146 | [-0.0229, -0.0069] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1831 / -0.2651 | -0.2241 | [-0.3118, -0.1522] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1572 / -0.2386 | -0.1979 | [-0.2862, -0.1238] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0801 / -0.1030 | -0.0915 | [-0.1343, -0.0568] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | -0.1220 / -0.1767 | -0.1494 | [-0.2079, -0.1015] | 0.0442 | BETTER |
| ade_tac_m|all | -0.1831 / -0.2651 | -0.2241 | [-0.3118, -0.1522] | 0.0699 | BETTER |

## I10R0KHb20 minus I10R0G0X

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0146 / -0.0141 | -0.0144 | [-0.0261, -0.0040] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0139 / -0.0257 | -0.0198 | [-0.0333, -0.0071] | 0.0298 (0.0298) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0680 / -0.0767 | -0.0724 | [-0.0988, -0.0489] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.5904 / -1.3094 | -0.9499 | [-1.6017, -0.3589] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0249 / +0.0245 | +0.0247 | [+0.0146, +0.0362] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.5677 / +1.1592 | +1.3634 | [+0.6996, +2.0655] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0038 / -0.0083 | -0.0060 | [-0.0183, +0.0065] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0438 / -0.0438 | -0.0438 | [-0.0600, -0.0296] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0163 / -0.0175 | -0.0169 | [-0.0269, -0.0075] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.0729 / -0.1028 | -0.0878 | [-0.1270, -0.0553] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0767 / -0.0988 | -0.0878 | [-0.1221, -0.0568] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0150 / -0.0325 | -0.0237 | [-0.0446, -0.0045] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | -0.0486 / -0.0685 | -0.0586 | [-0.0847, -0.0368] | 0.0442 | BETTER |
| ade_tac_m|all | -0.0729 / -0.1028 | -0.0878 | [-0.1270, -0.0553] | 0.0699 | BETTER |

## I10R0KHb10 minus I10R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0026 / -0.0038 | -0.0006 | [-0.0065, +0.0050] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0032 / -0.0103 | -0.0067 | [-0.0158, +0.0013] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0082 / -0.0080 | -0.0081 | [-0.0206, +0.0033] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.1728 / -0.2923 | -0.2325 | [-0.3894, -0.0661] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0082 / -0.0206 | -0.0144 | [-0.0252, -0.0059] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.4956 / -1.0391 | -0.7673 | [-1.2969, -0.3260] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0004 / -0.0021 | -0.0013 | [-0.0062, +0.0033] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0058 / -0.0046 | +0.0006 | [-0.0096, +0.0098] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0021 / -0.0017 | -0.0019 | [-0.0063, +0.0023] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -0.0119 / -0.0230 | -0.0174 | [-0.0356, -0.0018] | 0.0699 (0.0699) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0094 / -0.0321 | -0.0207 | [-0.0458, +0.0015] | 0.0229 | OPEN |
| ade6_m|bulk | -0.0061 / +0.0046 | -0.0008 | [-0.0097, +0.0073] | 0.1243 | OPEN |
| ade6_m|all | -0.0079 / -0.0153 | -0.0116 | [-0.0237, -0.0012] | 0.0442 | INSIDE-SPREAD |
| ade_tac_m|all | -0.0119 / -0.0230 | -0.0174 | [-0.0356, -0.0018] | 0.0699 | INSIDE-SPREAD |

## I10R0KHb20 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.6054 / -0.6500 | -0.6277 | [-0.7390, -0.5240] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.3698 / -0.3976 | -0.3837 | [-0.4910, -0.2923] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.4368 / -0.3169 | -0.3769 | [-0.5069, -0.2483] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.8646 / -3.8443 | -3.3545 | [-4.4616, -2.5330] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0205 / -0.0370 | -0.0288 | [-0.0448, -0.0138] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -10.2430 / -12.0353 | -11.1391 | [-13.9520, -8.8335] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1296 / -0.1838 | -0.1567 | [-0.2116, -0.1040] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2113 / -0.1630 | -0.1872 | [-0.2348, -0.1466] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1772 / -0.2226 | -0.1999 | [-0.2489, -0.1531] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.8534 / -0.7776 | -0.8155 | [-1.0282, -0.6321] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2172 / -0.2250 | -0.2211 | [-0.3420, -0.0989] | 0.0229 | BETTER |
| ade6_m|bulk | -0.9885 / -0.8684 | -0.9284 | [-1.1885, -0.7231] | 0.1243 | BETTER |
| ade6_m|all | -0.5689 / -0.5184 | -0.5437 | [-0.6855, -0.4214] | 0.0442 | BETTER |
| ade_tac_m|all | -0.8534 / -0.7776 | -0.8155 | [-1.0282, -0.6321] | 0.0699 | BETTER |

## I10RLcKF5Hb20 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.6291 / -0.6543 | -0.6417 | [-0.7519, -0.5366] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.4931 / -0.4597 | -0.4764 | [-0.5948, -0.3727] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.4121 / -0.2949 | -0.3535 | [-0.4802, -0.2305] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -3.1680 / -3.8691 | -3.5185 | [-4.4732, -2.7873] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0315 / -0.0472 | -0.0394 | [-0.0575, -0.0236] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -11.8593 / -13.0247 | -12.4420 | [-15.0540, -10.2552] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1280 / -0.1867 | -0.1574 | [-0.2132, -0.1042] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2205 / -0.1722 | -0.1963 | [-0.2431, -0.1562] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1797 / -0.2288 | -0.2043 | [-0.2541, -0.1554] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.9831 / -0.8333 | -0.9082 | [-1.1700, -0.6921] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.3347 / -0.2905 | -0.3126 | [-0.4571, -0.1697] | 0.0229 | BETTER |
| ade6_m|bulk | -1.0363 / -0.8699 | -0.9531 | [-1.2371, -0.7350] | 0.1243 | BETTER |
| ade6_m|all | -0.6547 / -0.5547 | -0.6047 | [-0.7789, -0.4609] | 0.0442 | BETTER |
| ade_tac_m|all | -0.9831 / -0.8333 | -0.9082 | [-1.1700, -0.6921] | 0.0699 | BETTER |

## I10RQ50KF5Hb20 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.6180 / -0.6696 | -0.6438 | [-0.7588, -0.5356] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.4999 / -0.5658 | -0.5329 | [-0.6785, -0.4007] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.4005 / -0.2973 | -0.3489 | [-0.4691, -0.2308] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -3.3719 / -4.0177 | -3.6948 | [-4.6060, -2.9633] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0395 / -0.0526 | -0.0460 | [-0.0628, -0.0318] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -12.4164 / -13.3547 | -12.8855 | [-15.3326, -10.7216] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1342 / -0.1901 | -0.1622 | [-0.2194, -0.1086] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2222 / -0.1813 | -0.2018 | [-0.2435, -0.1644] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1888 / -0.2318 | -0.2103 | [-0.2577, -0.1637] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.9938 / -0.9850 | -0.9894 | [-1.2314, -0.7728] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.3340 / -0.3728 | -0.3534 | [-0.5073, -0.2155] | 0.0229 | BETTER |
| ade6_m|bulk | -1.0493 / -0.9920 | -1.0207 | [-1.2793, -0.7986] | 0.1243 | BETTER |
| ade6_m|all | -0.6602 / -0.6552 | -0.6577 | [-0.8183, -0.5138] | 0.0442 | BETTER |
| ade_tac_m|all | -0.9938 / -0.9850 | -0.9894 | [-1.2314, -0.7728] | 0.0699 | BETTER |

## I10RLcKF5Hb20 minus I10RLcKF5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0069 / -0.0065 | -0.0067 | [-0.0187, +0.0041] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0104 / -0.0227 | -0.0166 | [-0.0296, -0.0039] | 0.0298 (0.0298) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0544 / -0.0583 | -0.0564 | [-0.0775, -0.0370] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.5991 / -1.2481 | -0.9236 | [-1.5360, -0.3658] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0245 / +0.0210 | +0.0228 | [+0.0132, +0.0331] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.5066 / +1.0085 | +1.2575 | [+0.6098, +1.9372] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0029 / -0.0046 | -0.0038 | [-0.0152, +0.0075] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0396 / -0.0396 | -0.0396 | [-0.0546, -0.0269] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0158 / -0.0133 | -0.0146 | [-0.0246, -0.0060] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.0544 / -0.0784 | -0.0664 | [-0.0984, -0.0379] | 0.0699 (0.0699) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0631 / -0.0778 | -0.0704 | [-0.0986, -0.0443] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0042 / -0.0218 | -0.0130 | [-0.0330, +0.0055] | 0.1243 | OPEN |
| ade6_m|all | -0.0362 / -0.0523 | -0.0443 | [-0.0656, -0.0253] | 0.0442 | BETTER |
| ade_tac_m|all | -0.0544 / -0.0784 | -0.0664 | [-0.0984, -0.0379] | 0.0699 | INSIDE-SPREAD |

## I10RQ50KF5Hb20 minus I10RQ50KF5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0085 / -0.0083 | -0.0084 | [-0.0199, +0.0020] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0162 / -0.0245 | -0.0204 | [-0.0321, -0.0088] | 0.0298 (0.0298) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0508 / -0.0613 | -0.0560 | [-0.0795, -0.0356] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.7415 / -1.2177 | -0.9796 | [-1.5192, -0.4444] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0186 / +0.0170 | +0.0178 | [+0.0093, +0.0276] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.3357 / +0.8754 | +1.1056 | [+0.5105, +1.7783] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0046 / -0.0071 | -0.0058 | [-0.0173, +0.0050] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0371 / -0.0392 | -0.0381 | [-0.0533, -0.0246] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0138 / -0.0154 | -0.0146 | [-0.0244, -0.0054] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.0602 / -0.0839 | -0.0720 | [-0.1046, -0.0429] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0694 / -0.0878 | -0.0786 | [-0.1105, -0.0504] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0052 / -0.0180 | -0.0116 | [-0.0273, +0.0040] | 0.1243 | OPEN |
| ade6_m|all | -0.0401 / -0.0559 | -0.0480 | [-0.0697, -0.0286] | 0.0442 | BETTER |
| ade_tac_m|all | -0.0602 / -0.0839 | -0.0720 | [-0.1046, -0.0429] | 0.0699 | BETTER |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0G0X_s0 | 2.2145 | 2.9722 | 1.0060 | 0.9256 | 0.4860 |
| I10R0G0X_s1 | 2.2407 | 3.0094 | 1.0121 | 0.9484 | 0.4664 |
| I10R0Hb20_s0 | 2.6128 | 2.9555 | 1.3614 | 1.1534 | 0.6332 |
| I10R0Hb20_s1 | 2.5138 | 2.8970 | 1.4041 | 1.0196 | 0.6561 |
| I10R0KHb10_s0 | 2.1580 | 2.8861 | 0.9940 | 0.8495 | 0.4677 |
| I10R0KHb10_s1 | 2.1568 | 2.8785 | 0.9941 | 0.8636 | 0.4473 |
| I10R0KHb20_s0 | 2.1659 | 2.8955 | 0.9914 | 0.8576 | 0.4698 |
| I10R0KHb20_s1 | 2.1722 | 2.9106 | 0.9979 | 0.8717 | 0.4489 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RLcKF5_s0 | 2.3317 | 2.9701 | 1.0288 | 0.9096 | 0.4844 |
| I10RLcKF5_s1 | 2.3771 | 3.0281 | 1.0459 | 0.9271 | 0.4635 |
| I10RLcKF5Hb20_s0 | 2.2955 | 2.9071 | 1.0220 | 0.8551 | 0.4685 |
| I10RLcKF5Hb20_s1 | 2.3249 | 2.9503 | 1.0394 | 0.8688 | 0.4502 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50KF5_s0 | 2.4017 | 3.0840 | 1.0452 | 0.9046 | 0.4781 |
| I10RQ50KF5_s1 | 2.3852 | 3.0235 | 1.0432 | 0.9287 | 0.4614 |
| I10RQ50KF5Hb20_s0 | 2.3616 | 3.0146 | 1.0367 | 0.8537 | 0.4644 |
| I10RQ50KF5Hb20_s1 | 2.3293 | 2.9357 | 1.0349 | 0.8674 | 0.4460 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
