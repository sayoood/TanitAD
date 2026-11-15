# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 18}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LFTRL"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LFTRL minus LFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0008 / +0.0654 | +0.0331 | [-0.0485, +0.1239] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1907 / +0.2095 | +0.2001 | [+0.0515, +0.3800] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0226 / -0.0178 | -0.0202 | [-0.0438, +0.0065] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.3457 / +0.4608 | +0.4032 | [+0.2425, +0.5971] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0163 / +0.0171 | +0.0167 | [+0.0092, +0.0256] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.3052 / +1.5944 | +0.9498 | [-0.2285, +2.0579] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0108 / +0.0075 | -0.0017 | [-0.0173, +0.0146] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0054 / +0.0138 | +0.0096 | [-0.0019, +0.0236] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0000 / +0.0079 | +0.0040 | [-0.0058, +0.0148] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2231 / +0.2400 | +0.2315 | [+0.0302, +0.4763] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0789 / +0.0945 | +0.0867 | [-0.0476, +0.2510] | 0.4517 | OPEN |
| ade6_m|bulk | +0.2775 / +0.2844 | +0.2809 | [+0.0971, +0.4939] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.1694 / +0.1811 | +0.1753 | [+0.0271, +0.3556] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2231 / +0.2400 | +0.2315 | [+0.0302, +0.4763] | 0.5955 | INSIDE-SPREAD |

## LCORL minus LFTRL

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.5365 / +0.5734 | +0.5550 | [+0.3768, +0.7516] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +1.3945 / +1.5011 | +1.4478 | [+0.9553, +2.0027] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.2701 / +0.1199 | +0.1950 | [+0.0945, +0.3127] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.9361 / +1.7646 | +1.8503 | [+1.3277, +2.4132] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0411 / +0.0175 | +0.0293 | [+0.0081, +0.0499] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +4.0548 / +3.9633 | +4.0091 | [+2.3639, +5.7014] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0054 / +0.0267 | +0.0160 | [-0.0077, +0.0413] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0604 / +0.0775 | +0.0690 | [+0.0454, +0.0949] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0096 / +0.0438 | +0.0267 | [-0.0033, +0.0575] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +2.0438 / +2.1721 | +2.1080 | [+1.4320, +2.8693] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.1214 / +1.1372 | +1.1293 | [+0.7342, +1.5889] | 0.4517 | WORSE |
| ade6_m|bulk | +1.8199 / +2.0052 | +1.9125 | [+1.1502, +2.8174] | 0.3535 | WORSE |
| ade6_m|all | +1.4400 / +1.5330 | +1.4865 | [+1.0041, +2.0333] | 0.4002 | WORSE |
| ade_tac_m|all | +2.0438 / +2.1721 | +2.1080 | [+1.4320, +2.8693] | 0.5955 | WORSE |

## LORL minus LFTRL

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3506 / +0.6736 | +0.5121 | [+0.2765, +0.8237] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +0.9113 / +1.5873 | +1.2493 | [+0.7163, +1.9477] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0907 / +0.1478 | +0.1193 | [+0.0587, +0.1886] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.4579 / +2.6161 | +2.0370 | [+1.1805, +3.0695] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0405 / +0.0362 | +0.0384 | [+0.0182, +0.0601] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +4.8613 / +6.5674 | +5.7143 | [+3.3963, +8.4897] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0288 / +0.0200 | -0.0044 | [-0.0429, +0.0329] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0454 / +0.1163 | +0.0809 | [+0.0333, +0.1328] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0088 / +0.0500 | +0.0206 | [-0.0229, +0.0634] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.2804 / +2.3302 | +1.8053 | [+1.0261, +2.8248] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.8332 / +1.4064 | +1.1198 | [+0.6505, +1.7940] | 0.4517 | WORSE |
| ade6_m|bulk | +1.0410 / +1.8980 | +1.4695 | [+0.7377, +2.3940] | 0.3535 | WORSE |
| ade6_m|all | +0.9280 / +1.6306 | +1.2793 | [+0.7392, +1.9800] | 0.4002 | WORSE |
| ade_tac_m|all | +1.2804 / +2.3302 | +1.8053 | [+1.0261, +2.8248] | 0.5955 | WORSE |

## LCORL minus LFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.5373 / +0.6389 | +0.5881 | [+0.3624, +0.8575] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +1.5852 / +1.7106 | +1.6479 | [+1.0257, +2.3442] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.2475 / +0.1022 | +0.1749 | [+0.0718, +0.2948] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +2.2939 / +2.2218 | +2.2578 | [+1.7068, +2.8828] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0567 / +0.0342 | +0.0455 | [+0.0233, +0.0677] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +4.3015 / +5.5188 | +4.9101 | [+3.0187, +6.9401] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0054 / +0.0342 | +0.0144 | [-0.0196, +0.0506] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0659 / +0.0913 | +0.0786 | [+0.0496, +0.1119] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0096 / +0.0517 | +0.0306 | [-0.0048, +0.0677] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +2.2669 / +2.4121 | +2.3395 | [+1.4854, +3.2773] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.2003 / +1.2317 | +1.2160 | [+0.7372, +1.7790] | 0.4517 | WORSE |
| ade6_m|bulk | +2.0974 / +2.2896 | +2.1935 | [+1.2772, +3.2505] | 0.3535 | WORSE |
| ade6_m|all | +1.6094 / +1.7141 | +1.6617 | [+1.0494, +2.3362] | 0.4002 | WORSE |
| ade_tac_m|all | +2.2669 / +2.4121 | +2.3395 | [+1.4854, +3.2773] | 0.5955 | WORSE |

## LCO minus LC

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.5517 / +0.6361 | +0.5939 | [+0.3676, +0.8542] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +1.5158 / +1.5909 | +1.5534 | [+0.9455, +2.2365] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.2701 / +0.1144 | +0.1922 | [+0.0873, +0.3232] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +2.4527 / +1.1105 | +1.7816 | [+0.7082, +2.8688] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0433 / -0.0123 | +0.0155 | [-0.0264, +0.0542] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +5.5499 / +2.6443 | +4.0971 | [+1.1949, +6.9354] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0188 / +0.0246 | +0.0029 | [-0.0329, +0.0396] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0671 / +0.0800 | +0.0736 | [+0.0486, +0.1019] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0121 / +0.0454 | +0.0167 | [-0.0250, +0.0606] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +2.1932 / +2.2607 | +2.2270 | [+1.3934, +3.1273] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.1701 / +1.1446 | +1.1574 | [+0.6688, +1.7079] | 0.4517 | WORSE |
| ade6_m|bulk | +2.0248 / +2.1694 | +2.0971 | [+1.1792, +3.1631] | 0.3535 | WORSE |
| ade6_m|all | +1.5599 / +1.6119 | +1.5859 | [+0.9844, +2.2356] | 0.4002 | WORSE |
| ade_tac_m|all | +2.1932 / +2.2607 | +2.2270 | [+1.3934, +3.1273] | 0.5955 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LC_s0 | 2.7007 | 3.1225 | 1.7669 | 1.1882 | 0.7153 |
| LC_s1 | 2.6943 | 3.2161 | 1.5949 | 1.1787 | 0.6674 |
| LCO_s0 | 4.2606 | 4.2926 | 2.3186 | 1.4582 | 0.7032 |
| LCO_s1 | 4.3062 | 4.3607 | 2.2309 | 1.2931 | 0.7128 |
| LCORL_s0 | 4.2471 | 4.2747 | 2.2349 | 1.4365 | 0.6982 |
| LCORL_s1 | 4.2966 | 4.3227 | 2.1335 | 1.2595 | 0.7057 |
| LFT_s0 | 2.6377 | 3.0744 | 1.6976 | 1.1890 | 0.6886 |
| LFT_s1 | 2.5825 | 3.0910 | 1.4947 | 1.1573 | 0.6540 |
| LFTRL_s0 | 2.8072 | 3.1533 | 1.6984 | 1.1664 | 0.6886 |
| LFTRL_s1 | 2.7636 | 3.1854 | 1.5601 | 1.1395 | 0.6619 |
| LORL_s0 | 3.7351 | 3.9866 | 2.0490 | 1.2572 | 0.6799 |
| LORL_s1 | 4.3942 | 4.5919 | 2.2337 | 1.2873 | 0.7120 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
