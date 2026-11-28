# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "PE"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8990895386859552, "D_read": 0.8990895386859552, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## PE minus P

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0061 / -0.1839 | -0.0950 | [-0.2281, +0.0397] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0065 / -0.2013 | -0.0974 | [-0.2492, +0.0582] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.1242 / -0.0819 | -0.1031 | [-0.1642, -0.0479] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.1791 / -1.4900 | -0.8346 | [-1.7213, +0.1995] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0167 / -0.0304 | -0.0068 | [-0.0385, +0.0250] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | +4.0165 / -4.3115 | -0.1475 | [-5.3136, +5.0331] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | -0.0438 / -0.0888 | -0.0663 | [-0.1036, -0.0275] | 0.0746 (0.0329) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0859 / -0.0404 | -0.0632 | [-0.1022, -0.0217] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0692 / -0.1138 | -0.0915 | [-0.1282, -0.0540] | 0.1205 (0.0438) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1355 / -0.3516 | -0.2436 | [-0.4227, -0.0585] | 0.4169 (0.1897) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0311 / -0.2904 | -0.1607 | [-0.3394, +0.0329] | 0.4093 | OPEN |
| ade6_m|bulk | -0.1649 / -0.1710 | -0.1679 | [-0.3055, -0.0388] | 0.3850 | INSIDE-SPREAD |
| ade6_m|all | -0.0921 / -0.2359 | -0.1640 | [-0.2850, -0.0409] | 0.2845 | INSIDE-SPREAD |
| ade_tac_m|all | -0.1355 / -0.3516 | -0.2436 | [-0.4227, -0.0585] | 0.4169 | INSIDE-SPREAD |

## PE minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0925 / +0.0368 | -0.0278 | [-0.1595, +0.0814] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1589 / +0.0399 | -0.0595 | [-0.2255, +0.0984] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.0275 / -0.2075 | -0.1175 | [-0.2343, +0.0078] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | +0.0995 / -0.9685 | -0.4345 | [-1.1726, +0.5042] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0112 / +0.0039 | -0.0037 | [-0.0221, +0.0116] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.8474 / -0.7121 | +0.0676 | [-1.4837, +1.9348] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | -0.0183 / -0.0058 | -0.0121 | [-0.0377, +0.0117] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0233 / -0.0454 | -0.0344 | [-0.0613, -0.0094] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0204 / -0.0300 | -0.0252 | [-0.0490, -0.0002] | 0.1205 (0.0438) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.2308 / -0.1528 | -0.1918 | [-0.3369, -0.0460] | 0.4169 (0.1897) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1357 / -0.1888 | -0.1623 | [-0.2561, -0.0537] | 0.4093 | INSIDE-SPREAD |
| ade6_m|bulk | -0.1907 / +0.0011 | -0.0948 | [-0.2968, +0.1081] | 0.3850 | OPEN |
| ade6_m|all | -0.1608 / -0.1022 | -0.1315 | [-0.2317, -0.0302] | 0.2845 | INSIDE-SPREAD |
| ade_tac_m|all | -0.2308 / -0.1528 | -0.1918 | [-0.3369, -0.0460] | 0.4169 | INSIDE-SPREAD |

## PR minus P

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1863 / -0.3037 | -0.0587 | [-0.3496, +0.2489] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2492 / -0.2265 | +0.0114 | [-0.2787, +0.3262] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.1721 / -0.0704 | -0.1212 | [-0.2108, -0.0425] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.4613 / +0.2847 | +0.8730 | [-0.0467, +1.7859] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0701 / +0.0184 | +0.0443 | [+0.0090, +0.0863] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +7.2653 / -0.3248 | +3.4702 | [-1.4447, +8.3001] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0367 / -0.0913 | -0.0273 | [-0.1076, +0.0530] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0225 / -0.0263 | -0.0244 | [-0.0462, -0.0036] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0267 / -0.1030 | -0.0381 | [-0.1182, +0.0429] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | +0.1601 / -0.3799 | -0.1099 | [-0.4627, +0.2713] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0281 / -0.1737 | -0.0728 | [-0.2346, +0.0899] | 0.4093 | OPEN |
| ade6_m|bulk | +0.2062 / -0.3573 | -0.0756 | [-0.4437, +0.3442] | 0.3850 | OPEN |
| ade6_m|all | +0.1093 / -0.2575 | -0.0741 | [-0.3118, +0.1847] | 0.2845 | OPEN |
| ade_tac_m|all | +0.1601 / -0.3799 | -0.1099 | [-0.4627, +0.2713] | 0.4169 | OPEN |

## PR minus G

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1879 / -0.0396 | +0.0741 | [-0.0836, +0.2573] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1709 / +0.0633 | +0.1171 | [-0.0082, +0.2677] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.0271 / -0.0300 | -0.0285 | [-0.0773, +0.0215] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | -0.1800 / +2.0900 | +0.9550 | [-0.4718, +2.5210] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0047 / +0.0613 | +0.0330 | [-0.0056, +0.0717] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.7494 / +6.0644 | +2.6575 | [-1.9584, +7.3994] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0454 / -0.0192 | +0.0131 | [-0.0367, +0.0606] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0246 / +0.0642 | +0.0198 | [-0.0417, +0.0798] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0325 / +0.0079 | +0.0202 | [-0.0088, +0.0481] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | +0.2119 / +0.0220 | +0.1170 | [-0.0717, +0.3329] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0535 / +0.1130 | +0.0298 | [-0.1265, +0.1811] | 0.4093 | OPEN |
| ade6_m|bulk | +0.3746 / -0.1027 | +0.1360 | [-0.1861, +0.5294] | 0.3850 | OPEN |
| ade6_m|all | +0.1417 / +0.0147 | +0.0782 | [-0.0486, +0.2246] | 0.2845 | OPEN |
| ade_tac_m|all | +0.2119 / +0.0220 | +0.1170 | [-0.0717, +0.3329] | 0.4169 | OPEN |

## PR minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0999 / -0.0829 | +0.0085 | [-0.1340, +0.1632] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0838 / +0.0146 | +0.0492 | [-0.0574, +0.1757] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.0753 / -0.1960 | -0.1357 | [-0.2253, -0.0500] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.7399 / +0.8062 | +1.2730 | [+0.4013, +2.2047] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0420 / +0.0527 | +0.0473 | [+0.0279, +0.0659] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +4.0918 / +3.2733 | +3.6825 | [+1.5144, +5.9755] | 10.6485 (4.6266) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0621 / -0.0083 | +0.0269 | [-0.0248, +0.0796] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | +0.0400 / -0.0313 | +0.0044 | [-0.0500, +0.0529] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0754 / -0.0192 | +0.0281 | [-0.0342, +0.0926] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | +0.0648 / -0.1810 | -0.0581 | [-0.2660, +0.1813] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0764 / -0.0722 | -0.0743 | [-0.1715, +0.0316] | 0.4093 | OPEN |
| ade6_m|bulk | +0.1803 / -0.1853 | -0.0025 | [-0.2698, +0.3173] | 0.3850 | OPEN |
| ade6_m|all | +0.0406 / -0.1237 | -0.0416 | [-0.1813, +0.1206] | 0.2845 | OPEN |
| ade_tac_m|all | +0.0648 / -0.1810 | -0.0581 | [-0.2660, +0.1813] | 0.4169 | OPEN |

## P minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0864 / +0.2207 | +0.0672 | [-0.1329, +0.2676] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1654 / +0.2411 | +0.0379 | [-0.2180, +0.2952] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | +0.0968 / -0.1256 | -0.0144 | [-0.1544, +0.1369] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | +0.2786 / +0.5215 | +0.4001 | [+0.0026, +0.7801] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0279 / +0.0342 | +0.0032 | [-0.0372, +0.0426] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -3.1691 / +3.5982 | +0.2146 | [-4.1922, +4.6944] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0254 / +0.0830 | +0.0542 | [+0.0106, +0.0997] | 0.0746 (0.0329) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0625 / -0.0050 | +0.0288 | [-0.0211, +0.0777] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0488 / +0.0838 | +0.0663 | [+0.0346, +0.1007] | 0.1205 (0.0438) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.0953 / +0.1989 | +0.0518 | [-0.1757, +0.2822] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1046 / +0.1016 | -0.0015 | [-0.1722, +0.1634] | 0.4093 | OPEN |
| ade6_m|bulk | -0.0259 / +0.1721 | +0.0731 | [-0.1162, +0.2544] | 0.3850 | OPEN |
| ade6_m|all | -0.0687 / +0.1337 | +0.0325 | [-0.1213, +0.1899] | 0.2845 | OPEN |
| ade_tac_m|all | -0.0953 / +0.1989 | +0.0518 | [-0.1757, +0.2822] | 0.4169 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| HIER_s3 | 2.9201 | 3.2939 | 1.8914 | 1.1934 | 0.7724 |
| HIER_s4 | 2.9086 | 3.3309 | 1.7001 | 1.4618 | 0.7524 |
| HIER_s5 | 3.0344 | 3.4157 | 1.8393 | 1.4161 | 0.7624 |
| P_s0 | 3.1244 | 3.5986 | 1.9134 | 1.3721 | 0.7445 |
| P_s1 | 3.1107 | 3.5268 | 1.9957 | 1.3621 | 0.8149 |
| PE_s0 | 3.0323 | 3.5675 | 1.9073 | 1.2479 | 0.6753 |
| PE_s1 | 2.8747 | 3.2364 | 1.8118 | 1.2802 | 0.7011 |
| PR_s0 | 3.2337 | 3.6268 | 2.0997 | 1.2000 | 0.7712 |
| PR_s1 | 2.8532 | 3.3530 | 1.6921 | 1.2917 | 0.7120 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
