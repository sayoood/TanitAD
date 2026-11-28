# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "G"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8990895386859552, "D_read": 0.8990895386859552, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## G minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0880 / -0.0433 | -0.0657 | [-0.1507, +0.0067] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0871 / -0.0486 | -0.0679 | [-0.1519, +0.0112] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.0482 / -0.1660 | -0.1071 | [-0.1964, -0.0173] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.9199 / -1.2838 | +0.3180 | [-1.4903, +2.3335] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0373 / -0.0086 | +0.0144 | [-0.0138, +0.0492] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | +4.8412 / -2.7912 | +1.0250 | [-3.4980, +6.2493] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0167 / +0.0108 | +0.0138 | [-0.0098, +0.0381] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | +0.0646 / -0.0955 | -0.0154 | [-0.1130, +0.0817] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0429 / -0.0271 | +0.0079 | [-0.0425, +0.0594] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.1470 / -0.2031 | -0.1750 | [-0.2930, -0.0483] | 0.4169 (0.1897) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0230 / -0.1852 | -0.1041 | [-0.2334, +0.0436] | 0.4093 | OPEN |
| ade6_m|bulk | -0.1943 / -0.0826 | -0.1384 | [-0.2958, -0.0052] | 0.3850 | INSIDE-SPREAD |
| ade6_m|all | -0.1011 / -0.1384 | -0.1198 | [-0.1982, -0.0349] | 0.2845 | INSIDE-SPREAD |
| ade_tac_m|all | -0.1470 / -0.2031 | -0.1750 | [-0.2930, -0.0483] | 0.4169 | INSIDE-SPREAD |

## W minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1745 / +0.0807 | -0.0469 | [-0.2313, +0.1272] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1850 / +0.0158 | -0.0846 | [-0.2403, +0.0680] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | +0.1500 / -0.0653 | +0.0424 | [-0.1052, +0.1883] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | +0.8137 / -0.0466 | +0.3836 | [-0.2834, +1.0844] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0275 / +0.0296 | +0.0285 | [+0.0115, +0.0468] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.9637 / +1.1535 | +2.0586 | [-0.1076, +4.4662] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | -0.0113 / -0.0363 | -0.0238 | [-0.0540, +0.0063] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | +0.0971 / +0.0204 | +0.0588 | [+0.0054, +0.1146] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0267 / -0.0233 | +0.0017 | [-0.0390, +0.0440] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.0805 / -0.0459 | -0.0632 | [-0.1891, +0.0706] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1397 / -0.0676 | -0.1037 | [-0.2037, -0.0007] | 0.4093 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0364 / +0.0104 | +0.0234 | [-0.1220, +0.1777] | 0.3850 | OPEN |
| ade6_m|all | -0.0594 / -0.0320 | -0.0457 | [-0.1312, +0.0456] | 0.2845 | OPEN |
| ade_tac_m|all | -0.0805 / -0.0459 | -0.0632 | [-0.1891, +0.0706] | 0.4169 | OPEN |

## G minus W

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0865 / -0.1240 | -0.0187 | [-0.1637, +0.1294] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0979 / -0.0644 | +0.0167 | [-0.1136, +0.1487] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.1983 / -0.1007 | -0.1495 | [-0.2356, -0.0601] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.1061 / -1.2372 | -0.0655 | [-1.4672, +1.4855] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0099 / -0.0383 | -0.0142 | [-0.0482, +0.0240] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.8803 / -3.9446 | -1.0322 | [-4.9986, +3.2229] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0279 / +0.0471 | +0.0375 | [+0.0090, +0.0661] | 0.0746 (0.0329) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0325 / -0.1159 | -0.0742 | [-0.1361, -0.0160] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0163 / -0.0038 | +0.0063 | [-0.0237, +0.0340] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.0665 / -0.1572 | -0.1118 | [-0.2497, +0.0265] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1168 / -0.1176 | -0.0004 | [-0.1717, +0.1850] | 0.4093 | OPEN |
| ade6_m|bulk | -0.2307 / -0.0930 | -0.1618 | [-0.3059, -0.0046] | 0.3850 | INSIDE-SPREAD |
| ade6_m|all | -0.0417 / -0.1064 | -0.0740 | [-0.1691, +0.0197] | 0.2845 | OPEN |
| ade_tac_m|all | -0.0665 / -0.1572 | -0.1118 | [-0.2497, +0.0265] | 0.4169 | OPEN |

## P minus G

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0016 / +0.2640 | +0.1328 | [-0.0417, +0.3036] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0783 / +0.2898 | +0.1057 | [-0.1324, +0.3336] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | +0.1450 / +0.0404 | +0.0927 | [+0.0149, +0.1709] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.6413 / +1.8053 | +0.0820 | [-1.9599, +2.0185] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0654 / +0.0429 | -0.0113 | [-0.0768, +0.0507] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -8.0131 / +6.3893 | -0.8119 | [-9.4408, +7.4093] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0088 / +0.0721 | +0.0404 | [-0.0098, +0.0881] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0021 / +0.0905 | +0.0442 | [-0.0183, +0.1073] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0058 / +0.1109 | +0.0584 | [-0.0106, +0.1247] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | +0.0517 / +0.4020 | +0.2268 | [-0.0217, +0.4673] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0816 / +0.2868 | +0.1026 | [-0.1431, +0.3465] | 0.4093 | OPEN |
| ade6_m|bulk | +0.1684 / +0.2546 | +0.2115 | [+0.0950, +0.3301] | 0.3850 | INSIDE-SPREAD |
| ade6_m|all | +0.0324 / +0.2721 | +0.1523 | [-0.0174, +0.3164] | 0.2845 | OPEN |
| ade_tac_m|all | +0.0517 / +0.4020 | +0.2268 | [-0.0217, +0.4673] | 0.4169 | OPEN |

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
| W_s0 | 3.1337 | 3.5635 | 1.8253 | 1.4254 | 0.7224 |
| W_s1 | 2.9449 | 3.3576 | 1.8557 | 1.4224 | 0.7078 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
