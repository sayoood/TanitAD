# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 18}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "IPlx_RLAm"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045083, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## IPlx RLAm minus IPox RLAm

Seeds read [2, 3] of [2, 3]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1287 / +0.1558 | +0.1422 | [-0.0141, +0.3224] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.6037 / +0.6353 | +0.6195 | [+0.2298, +1.0881] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0772 / -0.0678 | -0.0725 | [-0.1057, -0.0419] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.3236 / +0.2852 | +0.3044 | [+0.1265, +0.5228] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0146 / +0.0241 | +0.0193 | [+0.0075, +0.0334] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.3665 / +0.1268 | +0.2466 | [-0.8280, +1.4493] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0075 / +0.0092 | +0.0083 | [-0.0019, +0.0194] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0067 / +0.0021 | +0.0044 | [-0.0069, +0.0169] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0096 / +0.0104 | +0.0100 | [+0.0010, +0.0202] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.7508 / +0.8140 | +0.7824 | [+0.2509, +1.4243] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2784 / +0.3193 | +0.2989 | [+0.0313, +0.6477] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.8347 / +0.8785 | +0.8566 | [+0.3132, +1.4863] | 0.3535 | WORSE |
| ade6_m|all | +0.5321 / +0.5743 | +0.5532 | [+0.1792, +1.0042] | 0.4002 | WORSE |
| ade_tac_m|all | +0.7508 / +0.8140 | +0.7824 | [+0.2509, +1.4243] | 0.5955 | WORSE |

## IPkx RLAm minus IPox RLAm

Seeds read [2, 3] of [2, 3]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1072 / +0.1895 | +0.1483 | [+0.0019, +0.3146] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.5517 / +0.6530 | +0.6024 | [+0.2470, +1.0127] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0670 / -0.0295 | -0.0482 | [-0.0888, -0.0111] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.1382 / -0.1171 | +0.0106 | [-0.2307, +0.2464] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0111 / +0.0176 | +0.0144 | [+0.0070, +0.0237] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.1849 / -0.1770 | -0.1809 | [-0.9283, +0.5663] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0133 / +0.0138 | +0.0135 | [+0.0044, +0.0233] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0029 / +0.0008 | +0.0019 | [-0.0075, +0.0110] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0121 / +0.0138 | +0.0129 | [+0.0046, +0.0227] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.6867 / +0.8736 | +0.7802 | [+0.2945, +1.3448] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3479 / +0.4692 | +0.4086 | [+0.1242, +0.7499] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.6521 / +0.7885 | +0.7203 | [+0.2358, +1.2756] | 0.3535 | WORSE |
| ade6_m|all | +0.4866 / +0.6148 | +0.5507 | [+0.2091, +0.9463] | 0.4002 | WORSE |
| ade_tac_m|all | +0.6867 / +0.8736 | +0.7802 | [+0.2945, +1.3448] | 0.5955 | WORSE |

## IPkx RLAm minus IPcx RLAm

Seeds read [2, 3] of [2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0476 / +0.0723 | +0.0600 | [+0.0266, +0.0958] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0458 / +0.0613 | +0.0535 | [-0.0110, +0.1293] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0188 / +0.0316 | +0.0252 | [+0.0099, +0.0408] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.1467 / -0.2361 | -0.1914 | [-0.3868, -0.0253] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0060 / -0.0076 | -0.0068 | [-0.0129, -0.0010] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.0557 / +0.4334 | +0.2445 | [-0.2813, +0.7675] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0092 / +0.0063 | +0.0077 | [+0.0012, +0.0142] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0017 / +0.0042 | +0.0029 | [-0.0027, +0.0083] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0033 / +0.0025 | +0.0029 | [-0.0013, +0.0071] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0853 / +0.1238 | +0.1046 | [+0.0107, +0.2172] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0313 / +0.0880 | +0.0597 | [-0.0129, +0.1577] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0885 / +0.0812 | +0.0849 | [-0.0071, +0.1875] | 0.3535 | OPEN |
| ade6_m|all | +0.0574 / +0.0849 | +0.0711 | [+0.0046, +0.1499] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0853 / +0.1238 | +0.1046 | [+0.0107, +0.2172] | 0.5955 | INSIDE-SPREAD |

## IPkx RLAm minus IPlx RLAm

Seeds read [2, 3] of [2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0215 / +0.0337 | +0.0061 | [-0.0664, +0.0864] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0520 / +0.0177 | -0.0171 | [-0.1760, +0.1551] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0102 / +0.0383 | +0.0243 | [-0.0071, +0.0557] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1567 / -0.2654 | -0.2110 | [-0.4144, -0.0264] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0034 / -0.0076 | -0.0055 | [-0.0160, +0.0042] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.5230 / +0.1592 | -0.1819 | [-1.1275, +0.7181] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0058 / +0.0046 | +0.0052 | [-0.0019, +0.0131] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0038 / -0.0013 | -0.0025 | [-0.0104, +0.0054] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0025 / +0.0033 | +0.0029 | [-0.0031, +0.0090] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0641 / +0.0596 | -0.0022 | [-0.2326, +0.2543] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0696 / +0.1499 | +0.1097 | [-0.0292, +0.3140] | 0.4517 | OPEN |
| ade6_m|bulk | -0.1827 / -0.0900 | -0.1364 | [-0.3860, +0.0970] | 0.3535 | OPEN |
| ade6_m|all | -0.0455 / +0.0405 | -0.0025 | [-0.1642, +0.1771] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0641 / +0.0596 | -0.0022 | [-0.2326, +0.2543] | 0.5955 | OPEN |

## IPcx RLAm minus IPox RLAm

Seeds read [2, 3] of [2, 3]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0596 / +0.1172 | +0.0884 | [-0.0494, +0.2445] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.5060 / +0.5917 | +0.5488 | [+0.2191, +0.9381] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0857 / -0.0612 | -0.0735 | [-0.1116, -0.0394] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2567 / +0.2094 | +0.2330 | [+0.0306, +0.4511] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0159 / +0.0230 | +0.0194 | [+0.0097, +0.0304] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.1355 / -0.5925 | -0.3640 | [-1.2889, +0.5048] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0042 / +0.0075 | +0.0058 | [-0.0063, +0.0188] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0013 / -0.0033 | -0.0010 | [-0.0106, +0.0096] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0088 / +0.0113 | +0.0100 | [+0.0000, +0.0212] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.6014 / +0.7498 | +0.6756 | [+0.2258, +1.2141] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3166 / +0.3813 | +0.3489 | [+0.0927, +0.6545] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.5635 / +0.7073 | +0.6354 | [+0.1878, +1.1851] | 0.3535 | WORSE |
| ade6_m|all | +0.4292 / +0.5299 | +0.4796 | [+0.1649, +0.8554] | 0.4002 | WORSE |
| ade_tac_m|all | +0.6014 / +0.7498 | +0.6756 | [+0.2258, +1.2141] | 0.5955 | WORSE |

## IPksx RLAm minus IPkx RLAm

Seeds read [2, 3] of [2, 3]. **Q3: COST-LATERAL COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.2408 / +1.3721 | +1.3064 | [+1.0421, +1.5938] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +3.8647 / +4.1588 | +4.0117 | [+3.3065, +4.7738] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0926 / +0.0679 | +0.0802 | [-0.0257, +0.1986] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +3.2451 / +4.3633 | +3.8042 | [+2.5239, +5.1318] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.1627 / +0.2363 | +0.1995 | [+0.1388, +0.2669] | 0.1528 (0.0760) | WORSE |
| LATERAL:yaw_rate_mae_degps | +11.4821 / +15.7409 | +13.6115 | [+9.4304, +18.0579] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0684 / +0.0604 | +0.0644 | [+0.0446, +0.0850] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0246 / +0.0038 | +0.0142 | [-0.0207, +0.0450] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0438 / +0.0342 | +0.0390 | [+0.0223, +0.0571] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +5.2677 / +5.6697 | +5.4687 | [+4.4901, +6.5278] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.7145 / +3.9768 | +3.8456 | [+3.0588, +4.6577] | 0.4517 | WORSE |
| ade6_m|bulk | +3.8757 / +4.1735 | +4.0246 | [+3.0647, +5.1043] | 0.3535 | WORSE |
| ade6_m|all | +3.7880 / +4.0665 | +3.9273 | [+3.2263, +4.6705] | 0.4002 | WORSE |
| ade_tac_m|all | +5.2677 / +5.6697 | +5.4687 | [+4.4901, +6.5278] | 0.5955 | WORSE |

## IPkx RLAm minus IPix RLAm

Seeds read [2, 3] of [2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0202 / +0.0119 | +0.0161 | [+0.0017, +0.0320] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0256 / +0.0111 | +0.0183 | [-0.0122, +0.0543] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0071 / +0.0044 | +0.0058 | [+0.0028, +0.0097] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0158 / -0.0233 | -0.0196 | [-0.0748, +0.0292] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0024 / -0.0014 | -0.0019 | [-0.0042, +0.0006] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.0414 / +0.5245 | +0.2415 | [-0.1339, +0.7757] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0013 / +0.0008 | +0.0010 | [-0.0010, +0.0033] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0004 / +0.0008 | +0.0006 | [-0.0019, +0.0031] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0008 / -0.0013 | -0.0010 | [-0.0025, +0.0002] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0463 / +0.0206 | +0.0335 | [-0.0113, +0.0845] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0064 / +0.0276 | +0.0170 | [-0.0119, +0.0693] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0627 / -0.0018 | +0.0305 | [-0.0162, +0.1061] | 0.3535 | OPEN |
| ade6_m|all | +0.0321 / +0.0142 | +0.0231 | [-0.0083, +0.0589] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0463 / +0.0206 | +0.0335 | [-0.0113, +0.0845] | 0.5955 | OPEN |

## IPix RLAm minus IPox RLAm

Seeds read [2, 3] of [2, 3]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0869 / +0.1776 | +0.1323 | [-0.0143, +0.2959] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.5261 / +0.6419 | +0.5840 | [+0.2383, +0.9878] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0740 / -0.0339 | -0.0540 | [-0.0961, -0.0151] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.1540 / -0.0937 | +0.0302 | [-0.2116, +0.2663] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0136 / +0.0190 | +0.0163 | [+0.0092, +0.0256] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.1435 / -0.7014 | -0.4224 | [-1.3007, +0.4131] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0121 / +0.0129 | +0.0125 | [+0.0025, +0.0231] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0025 / +0.0000 | +0.0013 | [-0.0081, +0.0106] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0129 / +0.0150 | +0.0140 | [+0.0052, +0.0236] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.6404 / +0.8530 | +0.7467 | [+0.2665, +1.3093] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3415 / +0.4416 | +0.3916 | [+0.1146, +0.7188] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.5893 / +0.7903 | +0.6898 | [+0.2269, +1.2401] | 0.3535 | WORSE |
| ade6_m|all | +0.4545 / +0.6006 | +0.5276 | [+0.1898, +0.9228] | 0.4002 | WORSE |
| ade_tac_m|all | +0.6404 / +0.8530 | +0.7467 | [+0.2665, +1.3093] | 0.5955 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| IPcx_RLAm_s2 | 3.1283 | 3.5466 | 1.6313 | 1.1845 | 0.6261 |
| IPcx_RLAm_s3 | 3.3484 | 3.6755 | 1.8916 | 1.2213 | 0.6932 |
| IPix_RLAm_s2 | 3.1536 | 3.5716 | 1.6586 | 1.1962 | 0.6303 |
| IPix_RLAm_s3 | 3.4191 | 3.7359 | 1.9521 | 1.2485 | 0.6970 |
| IPksx_RLAm_s2 | 6.9737 | 7.2925 | 2.9196 | 1.2959 | 0.6732 |
| IPksx_RLAm_s3 | 7.4998 | 7.7403 | 3.3360 | 1.3208 | 0.7299 |
| IPkx_RLAm_s2 | 3.1857 | 3.5779 | 1.6788 | 1.2033 | 0.6294 |
| IPkx_RLAm_s3 | 3.4333 | 3.7635 | 1.9640 | 1.2529 | 0.6957 |
| IPlx_RLAm_s2 | 3.2311 | 3.5084 | 1.7003 | 1.1931 | 0.6269 |
| IPlx_RLAm_s3 | 3.3928 | 3.6136 | 1.9303 | 1.2146 | 0.6924 |
| IPox_RLAm_s2 | 2.6990 | 3.2300 | 1.5717 | 1.2703 | 0.6173 |
| IPox_RLAm_s3 | 2.8184 | 3.2943 | 1.7745 | 1.2825 | 0.6820 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
