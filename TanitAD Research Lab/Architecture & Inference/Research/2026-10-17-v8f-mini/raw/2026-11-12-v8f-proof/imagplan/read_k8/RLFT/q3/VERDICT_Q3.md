# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 18}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "IPlx_RLFT"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045083, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## IPlx RLFT minus IPox RLFT

Seeds read [2, 3] of [2, 3]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1513 / +0.1687 | +0.1600 | [+0.0033, +0.3416] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.6086 / +0.6346 | +0.6216 | [+0.2326, +1.0877] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0699 / -0.0602 | -0.0650 | [-0.0963, -0.0367] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2702 / +0.3755 | +0.3228 | [+0.1245, +0.5646] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0173 / +0.0239 | +0.0206 | [+0.0108, +0.0342] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.1036 / +0.3558 | +0.1261 | [-0.9867, +1.4670] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0004 / +0.0042 | +0.0019 | [-0.0081, +0.0127] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0092 / +0.0063 | +0.0077 | [-0.0040, +0.0211] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0054 / +0.0071 | +0.0063 | [-0.0025, +0.0165] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.7722 / +0.8227 | +0.7975 | [+0.2585, +1.4391] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2863 / +0.3230 | +0.3047 | [+0.0362, +0.6553] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.8567 / +0.8866 | +0.8717 | [+0.3269, +1.5061] | 0.3535 | WORSE |
| ade6_m|all | +0.5464 / +0.5800 | +0.5632 | [+0.1842, +1.0142] | 0.4002 | WORSE |
| ade_tac_m|all | +0.7722 / +0.8227 | +0.7975 | [+0.2585, +1.4391] | 0.5955 | WORSE |

## IPkx RLFT minus IPox RLFT

Seeds read [2, 3] of [2, 3]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1232 / +0.2050 | +0.1641 | [+0.0181, +0.3265] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.5359 / +0.6605 | +0.5982 | [+0.2412, +1.0162] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0647 / -0.0334 | -0.0490 | [-0.0848, -0.0143] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.1543 / +0.0831 | +0.1187 | [-0.0420, +0.2958] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0148 / +0.0154 | +0.0151 | [+0.0080, +0.0235] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.3145 / -0.2695 | -0.2920 | [-1.0555, +0.4687] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0063 / +0.0100 | +0.0081 | [-0.0006, +0.0179] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0025 / -0.0038 | -0.0006 | [-0.0098, +0.0088] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0088 / +0.0092 | +0.0090 | [+0.0010, +0.0181] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.6714 / +0.8825 | +0.7769 | [+0.2894, +1.3429] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3200 / +0.4481 | +0.3840 | [+0.1142, +0.7037] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.6618 / +0.8261 | +0.7439 | [+0.2563, +1.3074] | 0.3535 | WORSE |
| ade6_m|all | +0.4758 / +0.6204 | +0.5481 | [+0.2057, +0.9481] | 0.4002 | WORSE |
| ade_tac_m|all | +0.6714 / +0.8825 | +0.7769 | [+0.2894, +1.3429] | 0.5955 | WORSE |

## IPkx RLFT minus IPcx RLFT

Seeds read [2, 3] of [2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0333 / +0.0679 | +0.0506 | [+0.0127, +0.0914] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0249 / +0.0557 | +0.0403 | [-0.0325, +0.1262] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0093 / +0.0201 | +0.0147 | [+0.0015, +0.0288] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.2102 / -0.1505 | -0.1803 | [-0.3649, +0.0062] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0102 / -0.0087 | -0.0094 | [-0.0156, -0.0040] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.7135 / +0.4215 | -0.1460 | [-1.0421, +0.8287] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0046 / +0.0038 | +0.0042 | [-0.0012, +0.0100] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0008 / -0.0033 | -0.0021 | [-0.0081, +0.0035] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0025 / +0.0004 | +0.0015 | [-0.0025, +0.0065] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0461 / +0.1040 | +0.0750 | [-0.0303, +0.1981] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0026 / +0.0400 | +0.0187 | [-0.0575, +0.1010] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0711 / +0.1072 | +0.0891 | [-0.0271, +0.2190] | 0.3535 | OPEN |
| ade6_m|all | +0.0310 / +0.0706 | +0.0508 | [-0.0222, +0.1362] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0461 / +0.1040 | +0.0750 | [-0.0303, +0.1981] | 0.5955 | OPEN |

## IPkx RLFT minus IPlx RLFT

Seeds read [2, 3] of [2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0281 / +0.0363 | +0.0041 | [-0.0710, +0.0837] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0727 / +0.0259 | -0.0234 | [-0.1980, +0.1499] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0052 / +0.0268 | +0.0160 | [-0.0110, +0.0431] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1098 / -0.2263 | -0.1681 | [-0.3607, -0.0038] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0021 / -0.0097 | -0.0059 | [-0.0170, +0.0044] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.3523 / -0.3487 | -0.3505 | [-1.2392, +0.4342] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0067 / +0.0058 | +0.0063 | [+0.0000, +0.0125] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0067 / -0.0100 | -0.0083 | [-0.0194, +0.0017] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0033 / +0.0021 | +0.0027 | [-0.0035, +0.0090] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.1009 / +0.0597 | -0.0206 | [-0.2673, +0.2358] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0337 / +0.1250 | +0.0794 | [-0.0613, +0.2601] | 0.4517 | OPEN |
| ade6_m|bulk | -0.1950 / -0.0605 | -0.1278 | [-0.3941, +0.1292] | 0.3535 | OPEN |
| ade6_m|all | -0.0706 / +0.0404 | -0.0151 | [-0.1881, +0.1649] | 0.4002 | OPEN |
| ade_tac_m|all | -0.1009 / +0.0597 | -0.0206 | [-0.2673, +0.2358] | 0.5955 | OPEN |

## IPcx RLFT minus IPox RLFT

Seeds read [2, 3] of [2, 3]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0899 / +0.1371 | +0.1135 | [-0.0230, +0.2660] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.5110 / +0.6048 | +0.5579 | [+0.2290, +0.9551] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0740 / -0.0536 | -0.0638 | [-0.0998, -0.0294] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.3299 / +0.3084 | +0.3191 | [+0.1072, +0.5535] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0224 / +0.0228 | +0.0226 | [+0.0133, +0.0334] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.3756 / -0.4350 | -0.0297 | [-1.1314, +0.9551] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0017 / +0.0063 | +0.0040 | [-0.0073, +0.0156] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0033 / -0.0004 | +0.0015 | [-0.0067, +0.0108] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0063 / +0.0088 | +0.0075 | [-0.0015, +0.0179] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.6253 / +0.7785 | +0.7019 | [+0.2538, +1.2475] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3226 / +0.4081 | +0.3653 | [+0.1014, +0.6719] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.5906 / +0.7189 | +0.6548 | [+0.2045, +1.2042] | 0.3535 | WORSE |
| ade6_m|all | +0.4448 / +0.5498 | +0.4973 | [+0.1834, +0.8790] | 0.4002 | WORSE |
| ade_tac_m|all | +0.6253 / +0.7785 | +0.7019 | [+0.2538, +1.2475] | 0.5955 | WORSE |

## IPksx RLFT minus IPkx RLFT

Seeds read [2, 3] of [2, 3]. **Q3: COST-LATERAL COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.3538 / +1.4038 | +1.3788 | [+1.1188, +1.6536] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +4.0800 / +4.2288 | +4.1544 | [+3.4445, +4.9095] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0927 / +0.1015 | +0.0971 | [-0.0022, +0.2074] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +3.3323 / +4.1787 | +3.7555 | [+2.5761, +4.9216] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.1893 / +0.2300 | +0.2097 | [+0.1575, +0.2644] | 0.1528 (0.0760) | WORSE |
| LATERAL:yaw_rate_mae_degps | +11.8407 / +16.9671 | +14.4039 | [+9.7849, +19.3361] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0792 / +0.0629 | +0.0711 | [+0.0500, +0.0946] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0308 / +0.0250 | +0.0279 | [-0.0042, +0.0567] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0488 / +0.0417 | +0.0452 | [+0.0298, +0.0636] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +5.5737 / +5.7980 | +5.6858 | [+4.6908, +6.7514] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.8714 / +4.0692 | +3.9703 | [+3.1694, +4.7870] | 0.4517 | WORSE |
| ade6_m|bulk | +4.1581 / +4.2575 | +4.2078 | [+3.2031, +5.3002] | 0.3535 | WORSE |
| ade6_m|all | +4.0022 / +4.1551 | +4.0786 | [+3.3741, +4.8322] | 0.4002 | WORSE |
| ade_tac_m|all | +5.5737 / +5.7980 | +5.6858 | [+4.6908, +6.7514] | 0.5955 | WORSE |

## IPkx RLFT minus IPix RLFT

Seeds read [2, 3] of [2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0979 / -0.1242 | -0.1111 | [-0.1772, -0.0598] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.2668 / -0.3313 | -0.2991 | [-0.4641, -0.1708] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0012 / -0.0057 | -0.0023 | [-0.0113, +0.0068] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1206 / -0.2444 | -0.1825 | [-0.3583, -0.0229] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0055 / -0.0147 | -0.0101 | [-0.0208, -0.0005] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.5443 / -0.9937 | -0.7690 | [-1.6516, -0.0736] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0013 / -0.0008 | -0.0010 | [-0.0033, +0.0010] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0063 / -0.0117 | -0.0090 | [-0.0167, -0.0019] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0029 / -0.0029 | -0.0029 | [-0.0063, -0.0002] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.3674 / -0.4677 | -0.4175 | [-0.6541, -0.2357] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1493 / -0.2146 | -0.1819 | [-0.3311, -0.0660] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.3914 / -0.4653 | -0.4283 | [-0.6900, -0.2321] | 0.3535 | BETTER |
| ade6_m|all | -0.2597 / -0.3289 | -0.2943 | [-0.4600, -0.1664] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.3674 / -0.4677 | -0.4175 | [-0.6541, -0.2357] | 0.5955 | INSIDE-SPREAD |

## IPix RLFT minus IPox RLFT

Seeds read [2, 3] of [2, 3]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2211 / +0.3292 | +0.2752 | [+0.0889, +0.4822] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +0.8027 / +0.9918 | +0.8973 | [+0.4456, +1.4223] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0658 / -0.0277 | -0.0468 | [-0.0879, -0.0047] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2483 / +0.3294 | +0.2888 | [+0.0370, +0.5681] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0201 / +0.0305 | +0.0253 | [+0.0118, +0.0415] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.3079 / +0.7425 | +0.5252 | [-0.8196, +1.9900] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0075 / +0.0108 | +0.0092 | [-0.0004, +0.0190] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0088 / +0.0079 | +0.0083 | [-0.0015, +0.0198] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0117 / +0.0121 | +0.0119 | [+0.0029, +0.0225] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +1.0387 / +1.3502 | +1.1944 | [+0.5638, +1.9138] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.4693 / +0.6626 | +0.5660 | [+0.2064, +1.0017] | 0.4517 | WORSE |
| ade6_m|bulk | +1.0531 / +1.2914 | +1.1723 | [+0.5318, +1.9072] | 0.3535 | WORSE |
| ade6_m|all | +0.7355 / +0.9494 | +0.8424 | [+0.4010, +1.3503] | 0.4002 | WORSE |
| ade_tac_m|all | +1.0387 / +1.3502 | +1.1944 | [+0.5638, +1.9138] | 0.5955 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| IPcx_RLFT_s2 | 3.1271 | 3.5479 | 1.6296 | 1.1852 | 0.6169 |
| IPcx_RLFT_s3 | 3.3898 | 3.6811 | 1.9452 | 1.1893 | 0.6953 |
| IPix_RLFT_s2 | 3.4179 | 3.6946 | 1.7609 | 1.1934 | 0.6223 |
| IPix_RLFT_s3 | 3.7893 | 3.9357 | 2.1373 | 1.2151 | 0.6986 |
| IPksx_RLFT_s2 | 7.1603 | 7.4167 | 3.0167 | 1.2873 | 0.6682 |
| IPksx_RLFT_s3 | 7.6155 | 7.7904 | 3.4170 | 1.3109 | 0.7374 |
| IPkx_RLFT_s2 | 3.1582 | 3.5453 | 1.6629 | 1.1945 | 0.6194 |
| IPkx_RLFT_s3 | 3.4604 | 3.7211 | 2.0131 | 1.2094 | 0.6957 |
| IPlx_RLFT_s2 | 3.2288 | 3.5116 | 1.6911 | 1.1893 | 0.6161 |
| IPlx_RLFT_s3 | 3.4200 | 3.5961 | 1.9768 | 1.1826 | 0.6936 |
| IPox_RLFT_s2 | 2.6823 | 3.2253 | 1.5398 | 1.2592 | 0.6107 |
| IPox_RLFT_s3 | 2.8400 | 3.2731 | 1.8081 | 1.2428 | 0.6865 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
