# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 22}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LSRZ"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LSRZ minus LSRH

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0321 / -0.0096 | -0.0209 | [-0.0511, +0.0047] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0648 / -0.0369 | -0.0508 | [-0.1067, +0.0035] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0059 / -0.0079 | -0.0069 | [-0.0186, +0.0042] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1215 / -0.0662 | -0.0938 | [-0.2024, +0.0311] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0006 / +0.0033 | +0.0014 | [-0.0022, +0.0064] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.6298 / +0.1208 | -0.2545 | [-0.9217, +0.5308] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0083 / +0.0025 | -0.0029 | [-0.0117, +0.0048] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0054 / +0.0000 | -0.0027 | [-0.0077, +0.0033] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0046 / +0.0033 | -0.0006 | [-0.0069, +0.0050] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0947 / -0.0567 | -0.0757 | [-0.1574, +0.0035] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0070 / -0.0054 | +0.0008 | [-0.0533, +0.0622] | 0.4517 | OPEN |
| ade6_m|bulk | -0.1486 / -0.0795 | -0.1140 | [-0.2257, +0.0151] | 0.3535 | OPEN |
| ade6_m|all | -0.0639 / -0.0392 | -0.0516 | [-0.1067, +0.0022] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0947 / -0.0567 | -0.0757 | [-0.1574, +0.0035] | 0.5955 | OPEN |

## LSDRZ minus LSDRH

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0176 / -0.0495 | -0.0336 | [-0.0634, -0.0033] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0253 / -0.0851 | -0.0552 | [-0.1200, +0.0102] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0077 / -0.0195 | -0.0136 | [-0.0263, +0.0009] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.0423 / -0.2270 | -0.0924 | [-0.3129, +0.1190] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0006 / +0.0097 | +0.0052 | [-0.0006, +0.0162] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.2419 / +0.0505 | -0.0957 | [-0.5453, +0.3856] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0054 / -0.0025 | -0.0040 | [-0.0092, +0.0006] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0021 / -0.0025 | -0.0002 | [-0.0052, +0.0050] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0025 / -0.0029 | -0.0027 | [-0.0054, -0.0000] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.0424 / -0.1368 | -0.0896 | [-0.1874, +0.0095] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0183 / -0.0751 | -0.0467 | [-0.1247, +0.0399] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0414 / -0.1141 | -0.0777 | [-0.1763, -0.0020] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | -0.0288 / -0.0929 | -0.0609 | [-0.1269, +0.0064] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0424 / -0.1368 | -0.0896 | [-0.1874, +0.0095] | 0.5955 | OPEN |

## LSQZ minus LSQC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0371 / -0.0373 | -0.0372 | [-0.0677, -0.0074] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0464 / -0.1002 | -0.0733 | [-0.1407, -0.0003] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0148 / -0.0102 | -0.0125 | [-0.0323, +0.0056] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.0107 / -0.0771 | -0.0439 | [-0.2188, +0.1525] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0066 / +0.0007 | +0.0036 | [-0.0030, +0.0123] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.7035 / +0.2424 | -0.2305 | [-1.1490, +0.7484] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0025 / +0.0050 | +0.0038 | [-0.0027, +0.0098] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0029 / +0.0000 | -0.0015 | [-0.0085, +0.0044] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0004 / +0.0004 | +0.0004 | [-0.0042, +0.0052] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0731 / -0.1380 | -0.1055 | [-0.2023, -0.0018] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0286 / -0.0206 | -0.0246 | [-0.1045, +0.0503] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0772 / -0.1854 | -0.1313 | [-0.2600, -0.0291] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | -0.0508 / -0.0958 | -0.0733 | [-0.1386, -0.0031] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.0731 / -0.1380 | -0.1055 | [-0.2023, -0.0018] | 0.5955 | INSIDE-SPREAD |

## LSDQZ minus LSDQC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0271 / -0.0508 | -0.0390 | [-0.0751, -0.0040] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0629 / -0.1025 | -0.0827 | [-0.1588, -0.0094] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0200 / -0.0010 | +0.0095 | [-0.0151, +0.0358] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1370 / +0.0440 | -0.0465 | [-0.2821, +0.2069] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0044 / +0.0059 | +0.0052 | [-0.0021, +0.0140] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.9174 / +0.1159 | -0.4008 | [-1.5471, +0.8055] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0075 / -0.0042 | -0.0058 | [-0.0140, +0.0023] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0063 / -0.0079 | -0.0071 | [-0.0167, +0.0010] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0063 / -0.0042 | -0.0052 | [-0.0117, +0.0010] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0759 / -0.1475 | -0.1117 | [-0.2308, +0.0034] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0129 / +0.0027 | -0.0051 | [-0.0887, +0.0709] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0965 / -0.2227 | -0.1596 | [-0.3188, -0.0021] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | -0.0510 / -0.1001 | -0.0755 | [-0.1569, +0.0033] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0759 / -0.1475 | -0.1117 | [-0.2308, +0.0034] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LSDQC_s0 | 4.0576 | 4.2657 | 2.1910 | 1.2630 | 0.7595 |
| LSDQC_s1 | 3.7752 | 3.9137 | 2.1116 | 1.3035 | 0.7486 |
| LSDQZ_s0 | 4.0066 | 4.2528 | 2.1639 | 1.2829 | 0.7532 |
| LSDQZ_s1 | 3.6751 | 3.9164 | 2.0608 | 1.3025 | 0.7445 |
| LSDRH_s0 | 3.4232 | 3.8238 | 1.9855 | 1.2252 | 0.7570 |
| LSDRH_s1 | 3.2663 | 3.6549 | 1.9746 | 1.3191 | 0.7478 |
| LSDRZ_s0 | 3.3944 | 3.8055 | 1.9679 | 1.2175 | 0.7545 |
| LSDRZ_s1 | 3.1735 | 3.5798 | 1.9251 | 1.2996 | 0.7449 |
| LSQC_s0 | 3.7300 | 4.0354 | 2.1348 | 1.2867 | 0.6865 |
| LSQC_s1 | 3.5927 | 3.7853 | 1.8469 | 1.2885 | 0.7074 |
| LSQZ_s0 | 3.6792 | 4.0067 | 2.0977 | 1.2719 | 0.6870 |
| LSQZ_s1 | 3.4970 | 3.7646 | 1.8097 | 1.2783 | 0.7078 |
| LSRH_s0 | 3.3109 | 3.6586 | 1.9849 | 1.2439 | 0.6769 |
| LSRH_s1 | 3.1634 | 3.5230 | 1.6962 | 1.2410 | 0.7040 |
| LSRZ_s0 | 3.2469 | 3.6656 | 1.9528 | 1.2379 | 0.6724 |
| LSRZ_s1 | 3.1242 | 3.5175 | 1.6866 | 1.2331 | 0.7074 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
