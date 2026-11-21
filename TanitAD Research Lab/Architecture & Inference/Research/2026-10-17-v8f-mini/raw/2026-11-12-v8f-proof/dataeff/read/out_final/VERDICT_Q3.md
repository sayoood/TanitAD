# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 18}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "R1_2"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8990895386859552, "D_read": 0.8990895386859552, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## R1 2 minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1461 / -0.0090 | -0.0776 | [-0.2194, +0.0571] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0857 / -0.0230 | -0.0543 | [-0.1806, +0.0615] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | +0.0566 / -0.1736 | -0.0585 | [-0.2194, +0.0923] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | -0.9223 / -1.6602 | -1.2912 | [-1.9837, -0.5428] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0096 / +0.0026 | -0.0035 | [-0.0264, +0.0141] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -3.0786 / -2.7746 | -2.9266 | [-4.8931, -1.0362] | 10.6485 (4.6266) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0325 / -0.0183 | +0.0071 | [-0.0365, +0.0511] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0021 / -0.1542 | -0.0782 | [-0.1725, +0.0148] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0413 / -0.0725 | -0.0156 | [-0.0905, +0.0573] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.0646 / -0.1489 | -0.1067 | [-0.2666, +0.0634] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1375 / -0.0789 | -0.1082 | [-0.2279, +0.0159] | 0.4093 | OPEN |
| ade6_m|bulk | +0.0554 / -0.1260 | -0.0353 | [-0.2346, +0.1881] | 0.3850 | OPEN |
| ade6_m|all | -0.0495 / -0.1004 | -0.0750 | [-0.1818, +0.0388] | 0.2845 | OPEN |
| ade_tac_m|all | -0.0646 / -0.1489 | -0.1067 | [-0.2666, +0.0634] | 0.4169 | OPEN |

## R1 4 minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2119 / +0.1573 | -0.0273 | [-0.2677, +0.2209] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0722 / +0.3649 | +0.1464 | [-0.1522, +0.4519] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | +0.0448 / -0.1005 | -0.0278 | [-0.1472, +0.0878] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | -1.6004 / -0.9078 | -1.2541 | [-1.9952, -0.6007] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0924 / -0.0066 | -0.0495 | [-0.1072, +0.0009] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -8.1196 / -1.8479 | -4.9838 | [-9.5824, -0.7882] | 10.6485 (4.6266) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0425 / +0.0229 | -0.0098 | [-0.0661, +0.0453] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0217 / -0.1021 | -0.0619 | [-0.1236, -0.0069] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0217 / -0.0017 | -0.0117 | [-0.0480, +0.0229] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.0402 / +0.4005 | +0.1801 | [-0.1535, +0.5305] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0423 / +0.2639 | +0.1108 | [-0.1505, +0.3822] | 0.4093 | OPEN |
| ade6_m|bulk | -0.0398 / +0.2664 | +0.1133 | [-0.1521, +0.3787] | 0.3850 | OPEN |
| ade6_m|all | -0.0411 / +0.2650 | +0.1119 | [-0.1178, +0.3529] | 0.2845 | OPEN |
| ade_tac_m|all | -0.0402 / +0.4005 | +0.1801 | [-0.1535, +0.5305] | 0.4169 | OPEN |

## R1 8 minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0986 / +0.0373 | -0.0307 | [-0.1703, +0.1149] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0453 / +0.4466 | +0.2460 | [-0.0489, +0.5411] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | +0.2417 / -0.1142 | +0.0638 | [-0.1684, +0.2945] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | -2.6904 / -1.2483 | -1.9693 | [-3.1476, -0.9432] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.1292 / -0.0029 | -0.0660 | [-0.1468, +0.0092] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -11.2820 / -2.6813 | -6.9817 | [-13.2084, -1.3456] | 10.6485 (4.6266) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0567 / -0.0467 | -0.0517 | [-0.0884, -0.0150] | 0.0746 (0.0329) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0404 / -0.1546 | -0.0975 | [-0.1748, -0.0242] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0592 / -0.0975 | -0.0784 | [-0.1248, -0.0386] | 0.1205 (0.0438) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.3233 / +0.5174 | +0.4204 | [+0.1617, +0.6664] | 0.4169 (0.1897) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3014 / +0.5609 | +0.4311 | [+0.1627, +0.6845] | 0.4093 | WORSE |
| ade6_m|bulk | +0.1045 / +0.0949 | +0.0997 | [-0.0800, +0.2830] | 0.3850 | OPEN |
| ade6_m|all | +0.2116 / +0.3484 | +0.2800 | [+0.1036, +0.4477] | 0.2845 | INSIDE-SPREAD |
| ade_tac_m|all | +0.3233 / +0.5174 | +0.4204 | [+0.1617, +0.6664] | 0.4169 | WORSE |

## R1 8 minus R1 4

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1132 / -0.1200 | -0.0034 | [-0.1902, +0.1659] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1176 / +0.0817 | +0.0996 | [-0.0429, +0.2482] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | +0.1969 / -0.0137 | +0.0916 | [-0.0613, +0.2430] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | -1.0900 / -0.3405 | -0.7153 | [-1.5851, +0.0491] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0370 / +0.0038 | -0.0166 | [-0.0494, +0.0146] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -3.1664 / -0.8334 | -1.9999 | [-4.7109, +0.4691] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | -0.0142 / -0.0696 | -0.0419 | [-0.0916, +0.0058] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0188 / -0.0525 | -0.0356 | [-0.0682, -0.0027] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0375 / -0.0959 | -0.0667 | [-0.1176, -0.0177] | 0.1205 (0.0438) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.3635 / +0.1169 | +0.2402 | [-0.0418, +0.4947] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3436 / +0.2970 | +0.3203 | [+0.1374, +0.4989] | 0.4093 | INSIDE-SPREAD |
| ade6_m|bulk | +0.1443 / -0.1715 | -0.0136 | [-0.3051, +0.2576] | 0.3850 | OPEN |
| ade6_m|all | +0.2527 / +0.0834 | +0.1680 | [-0.0234, +0.3413] | 0.2845 | OPEN |
| ade_tac_m|all | +0.3635 / +0.1169 | +0.2402 | [-0.0418, +0.4947] | 0.4169 | OPEN |

## S1 4 minus R1 4

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3478 / +0.3186 | +0.3332 | [+0.1999, +0.4691] | 0.2997 (0.1392) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3290 / +0.3751 | +0.3521 | [+0.1673, +0.5590] | 0.4550 (0.1815) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.1182 / -0.0340 | +0.0421 | [-0.0938, +0.1597] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | -0.6503 / +0.8805 | +0.1151 | [-1.0891, +1.3443] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0007 / +0.0387 | +0.0190 | [-0.0157, +0.0530] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.5513 / +2.5316 | +0.4901 | [-2.7587, +4.2959] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0642 / -0.0067 | +0.0288 | [-0.0261, +0.0862] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | +0.0371 / +0.0246 | +0.0308 | [-0.0013, +0.0644] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0567 / -0.0283 | +0.0142 | [-0.0469, +0.0779] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | +0.5644 / +0.5218 | +0.5431 | [+0.2561, +0.8412] | 0.4169 (0.1897) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0225 / +0.0469 | +0.0347 | [-0.1750, +0.2608] | 0.4093 | OPEN |
| ade6_m|bulk | +0.8026 / +0.7078 | +0.7552 | [+0.4619, +1.0573] | 0.3850 | WORSE |
| ade6_m|all | +0.3782 / +0.3483 | +0.3633 | [+0.1705, +0.5617] | 0.2845 | WORSE |
| ade_tac_m|all | +0.5644 / +0.5218 | +0.5431 | [+0.2561, +0.8412] | 0.4169 | WORSE |

## R1 4 minus F1 4

Seeds read [0, 1] of [0, 1]. **Q3: COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1201 / +0.2735 | +0.1968 | [+0.0660, +0.3605] | 0.2997 (0.1392) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.1566 / +0.3057 | +0.2311 | [+0.0841, +0.4167] | 0.4550 (0.1815) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.2126 / -0.1367 | -0.1746 | [-0.2647, -0.0787] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.2618 / -1.8692 | -1.5655 | [-2.2857, -0.8342] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0296 / -0.0605 | -0.0450 | [-0.0745, -0.0165] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.7103 / -5.4615 | -3.0859 | [-6.9231, +0.7625] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0604 / +0.1234 | +0.0919 | [+0.0392, +0.1456] | 0.0746 (0.0329) | WORSE |
| TACTICAL:lat_err_rate | -0.0417 / -0.0167 | -0.0292 | [-0.0605, +0.0048] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0242 / +0.1050 | +0.0646 | [+0.0069, +0.1249] | 0.1205 (0.0438) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.0210 / +0.2381 | +0.1086 | [-0.1393, +0.4087] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2457 / -0.0722 | -0.1590 | [-0.3452, +0.0551] | 0.4093 | OPEN |
| ade6_m|bulk | +0.2589 / +0.4370 | +0.3479 | [+0.1257, +0.6224] | 0.3850 | INSIDE-SPREAD |
| ade6_m|all | -0.0156 / +0.1600 | +0.0722 | [-0.0947, +0.2746] | 0.2845 | OPEN |
| ade_tac_m|all | -0.0210 / +0.2381 | +0.1086 | [-0.1393, +0.4087] | 0.4169 | OPEN |

## HIER minus FLAT

Seeds read [0, 1] of [0, 1]. **Q3: COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4999 / +0.0966 | +0.2982 | [+0.0195, +0.5748] | 0.2997 (0.1392) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.4551 / +0.0577 | +0.2564 | [-0.0108, +0.5393] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.1788 / -0.1410 | -0.1599 | [-0.2291, -0.0926] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.7751 / -2.6272 | -1.7011 | [-2.9895, -0.4489] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0619 / -0.0957 | -0.0169 | [-0.1124, +0.0747] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | +6.9265 / -8.2270 | -0.6503 | [-9.6394, +8.3090] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.1167 / +0.0409 | +0.0788 | [+0.0156, +0.1376] | 0.0746 (0.0329) | WORSE |
| TACTICAL:lat_err_rate | +0.0088 / -0.0909 | -0.0411 | [-0.1064, +0.0244] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0900 / -0.0233 | +0.0333 | [-0.0429, +0.1101] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | +0.3976 / -0.0959 | +0.1508 | [-0.1805, +0.5287] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1085 / -0.1863 | -0.0389 | [-0.2574, +0.1951] | 0.4093 | OPEN |
| ade6_m|bulk | +0.4817 / +0.0817 | +0.2817 | [-0.0070, +0.6148] | 0.3850 | OPEN |
| ade6_m|all | +0.2787 / -0.0641 | +0.1073 | [-0.1209, +0.3701] | 0.2845 | OPEN |
| ade_tac_m|all | +0.3976 / -0.0959 | +0.1508 | [-0.1805, +0.5287] | 0.4169 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| F1_4_s0 | 3.1676 | 3.9067 | 1.6678 | 1.5328 | 0.6499 |
| F1_4_s1 | 3.0820 | 3.7613 | 1.6588 | 1.5239 | 0.6244 |
| FLAT_s0 | 2.9144 | 3.5947 | 1.5000 | 1.4542 | 0.6057 |
| FLAT_s1 | 3.0410 | 3.6115 | 1.6784 | 1.6287 | 0.7545 |
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| HIER_s3 | 2.9201 | 3.2939 | 1.8914 | 1.1934 | 0.7724 |
| HIER_s4 | 2.9086 | 3.3309 | 1.7001 | 1.4618 | 0.7524 |
| HIER_s5 | 3.0344 | 3.4157 | 1.8393 | 1.4161 | 0.7624 |
| R1_2_s0 | 3.1435 | 3.5657 | 1.8537 | 1.3320 | 0.7370 |
| R1_2_s1 | 2.8766 | 3.3463 | 1.7660 | 1.3141 | 0.6586 |
| R1_4_s0 | 3.1520 | 3.6609 | 1.7880 | 1.3202 | 0.6740 |
| R1_4_s1 | 3.2420 | 3.6891 | 1.9323 | 1.3872 | 0.7295 |
| R1_8_s0 | 3.4047 | 4.0046 | 1.9012 | 1.5171 | 0.6365 |
| R1_8_s1 | 3.3253 | 3.9861 | 1.8123 | 1.3735 | 0.6336 |
| S1_4_s0 | 3.5302 | 3.6835 | 2.1357 | 1.4384 | 0.7307 |
| S1_4_s1 | 3.5903 | 3.7360 | 2.2509 | 1.3532 | 0.7011 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
