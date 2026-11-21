# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 24}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LSRH"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LSRH minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0370 / +0.1153 | +0.0762 | [-0.0267, +0.1800] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3386 / +0.4948 | +0.4167 | [+0.1928, +0.6760] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0189 / +0.0044 | -0.0073 | [-0.0354, +0.0287] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.3873 / +0.9271 | +0.6572 | [+0.2764, +1.1458] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0074 / +0.0307 | +0.0190 | [+0.0052, +0.0400] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.2974 / +3.5942 | +1.6484 | [-1.1657, +4.3975] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0029 / -0.0058 | -0.0044 | [-0.0152, +0.0063] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0071 / -0.0025 | +0.0023 | [-0.0119, +0.0146] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0063 / -0.0004 | +0.0029 | [-0.0063, +0.0119] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.4203 / +0.6548 | +0.5376 | [+0.2233, +0.8891] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1445 / +0.3525 | +0.2485 | [+0.0522, +0.5040] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.4889 / +0.6148 | +0.5518 | [+0.2369, +0.9145] | 0.3535 | WORSE |
| ade6_m|all | +0.3015 / +0.4721 | +0.3868 | [+0.1603, +0.6446] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.4203 / +0.6548 | +0.5376 | [+0.2233, +0.8891] | 0.5955 | INSIDE-SPREAD |

## LSRL minus LS

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0298 / +0.0434 | +0.0366 | [-0.0287, +0.1123] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2402 / +0.2110 | +0.2256 | [+0.0722, +0.4075] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0151 / -0.0131 | -0.0141 | [-0.0357, +0.0077] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.2145 / +0.3105 | +0.2625 | [+0.0596, +0.4669] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0141 / +0.0163 | +0.0152 | [+0.0074, +0.0238] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.0637 / +1.1760 | +0.5562 | [-0.6349, +1.7005] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0042 / -0.0083 | -0.0021 | [-0.0140, +0.0104] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0021 / +0.0004 | -0.0008 | [-0.0115, +0.0108] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0067 / -0.0013 | +0.0027 | [-0.0060, +0.0123] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2917 / +0.2574 | +0.2746 | [+0.0692, +0.5198] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1342 / +0.1047 | +0.1194 | [-0.0178, +0.2880] | 0.4517 | OPEN |
| ade6_m|bulk | +0.3121 / +0.2949 | +0.3035 | [+0.1169, +0.5190] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.2153 / +0.1914 | +0.2034 | [+0.0537, +0.3854] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2917 / +0.2574 | +0.2746 | [+0.0692, +0.5198] | 0.5955 | INSIDE-SPREAD |

## LSHRH minus LSH

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0778 / +0.0321 | +0.0550 | [-0.0413, +0.1488] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2336 / +0.4175 | +0.3256 | [+0.1253, +0.5879] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0072 / -0.0804 | -0.0438 | [-0.1001, -0.0005] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.1413 / +0.4942 | +0.3177 | [+0.0687, +0.6895] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0032 / +0.0275 | +0.0154 | [+0.0019, +0.0357] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.1157 / +1.1403 | +0.5123 | [-0.7740, +1.8969] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0063 / -0.0121 | -0.0092 | [-0.0179, -0.0021] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0217 / +0.0046 | +0.0131 | [-0.0052, +0.0302] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0013 / -0.0038 | -0.0013 | [-0.0079, +0.0040] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2990 / +0.4740 | +0.3865 | [+0.1331, +0.7202] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1424 / +0.1732 | +0.1578 | [+0.0275, +0.3205] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.2968 / +0.5532 | +0.4250 | [+0.1384, +0.8025] | 0.3535 | WORSE |
| ade6_m|all | +0.2128 / +0.3465 | +0.2797 | [+0.0973, +0.5216] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2990 / +0.4740 | +0.3865 | [+0.1331, +0.7202] | 0.5955 | INSIDE-SPREAD |

## LSHRL minus LSH

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0501 / +0.0024 | +0.0263 | [-0.0498, +0.1077] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2281 / +0.2146 | +0.2214 | [+0.0708, +0.3992] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0255 / -0.0366 | -0.0310 | [-0.0567, -0.0097] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.0734 / +0.2739 | +0.1736 | [-0.0747, +0.3923] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0127 / +0.0170 | +0.0148 | [+0.0063, +0.0239] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.0306 / +0.4642 | +0.2474 | [-0.5577, +1.1159] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0067 / -0.0063 | -0.0065 | [-0.0131, +0.0000] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0104 / +0.0033 | +0.0069 | [-0.0050, +0.0202] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0054 / -0.0033 | -0.0044 | [-0.0108, +0.0023] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2693 / +0.2387 | +0.2540 | [+0.0543, +0.4975] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0985 / +0.0749 | +0.0867 | [-0.0421, +0.2516] | 0.4517 | OPEN |
| ade6_m|bulk | +0.3215 / +0.3047 | +0.3131 | [+0.1225, +0.5318] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.2002 / +0.1797 | +0.1900 | [+0.0435, +0.3661] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2693 / +0.2387 | +0.2540 | [+0.0543, +0.4975] | 0.5955 | INSIDE-SPREAD |

## LSDRH minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0519 / -0.0311 | +0.0104 | [-0.0867, +0.1117] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3657 / +0.2971 | +0.3314 | [+0.1552, +0.5352] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0169 / -0.0458 | -0.0145 | [-0.0610, +0.0295] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.5927 / +0.4496 | +0.5211 | [+0.3064, +0.7486] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0095 / +0.0072 | +0.0083 | [+0.0037, +0.0140] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.2231 / -0.3254 | -0.2743 | [-1.4941, +0.8947] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0017 / -0.0083 | -0.0050 | [-0.0133, +0.0031] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0296 / +0.0029 | +0.0163 | [-0.0029, +0.0394] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0092 / -0.0004 | +0.0044 | [-0.0035, +0.0129] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.4932 / +0.3578 | +0.4255 | [+0.1747, +0.7144] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2437 / +0.1483 | +0.1960 | [+0.0307, +0.3945] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.4824 / +0.3873 | +0.4348 | [+0.1979, +0.7144] | 0.3535 | WORSE |
| ade6_m|all | +0.3525 / +0.2573 | +0.3049 | [+0.1258, +0.5132] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.4932 / +0.3578 | +0.4255 | [+0.1747, +0.7144] | 0.5955 | INSIDE-SPREAD |

## LSDRL minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0284 / -0.0256 | +0.0014 | [-0.0802, +0.0853] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2342 / +0.2129 | +0.2235 | [+0.0643, +0.4114] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0188 / -0.0556 | -0.0372 | [-0.0743, -0.0044] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.1959 / +0.3317 | +0.2638 | [+0.0249, +0.4721] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0148 / +0.0093 | +0.0121 | [+0.0049, +0.0218] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.0149 / +0.1528 | +0.0690 | [-0.8304, +0.8984] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0088 / -0.0125 | -0.0106 | [-0.0177, -0.0029] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0208 / -0.0025 | +0.0092 | [-0.0087, +0.0290] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0008 / -0.0054 | -0.0023 | [-0.0094, +0.0052] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2843 / +0.2271 | +0.2557 | [+0.0344, +0.5164] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0964 / +0.0823 | +0.0893 | [-0.0525, +0.2629] | 0.4517 | OPEN |
| ade6_m|bulk | +0.3452 / +0.2753 | +0.3103 | [+0.1095, +0.5561] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.2098 / +0.1703 | +0.1901 | [+0.0300, +0.3802] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2843 / +0.2271 | +0.2557 | [+0.0344, +0.5164] | 0.5955 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LS_s0 | 3.0093 | 3.5142 | 1.9479 | 1.2628 | 0.6707 |
| LS_s1 | 2.6913 | 3.1704 | 1.5809 | 1.2366 | 0.7045 |
| LSD_s0 | 3.0707 | 3.5802 | 1.9336 | 1.2084 | 0.7478 |
| LSD_s1 | 3.0090 | 3.5066 | 2.0057 | 1.3649 | 0.7482 |
| LSDRH_s0 | 3.4232 | 3.8238 | 1.9855 | 1.2252 | 0.7570 |
| LSDRH_s1 | 3.2663 | 3.6549 | 1.9746 | 1.3191 | 0.7478 |
| LSDRL_s0 | 3.2805 | 3.6765 | 1.9621 | 1.1895 | 0.7486 |
| LSDRL_s1 | 3.1794 | 3.5889 | 1.9801 | 1.3092 | 0.7428 |
| LSH_s0 | 3.0224 | 3.6093 | 1.8153 | 1.2967 | 0.7457 |
| LSH_s1 | 2.8723 | 3.4420 | 1.7069 | 1.4529 | 0.7799 |
| LSHRH_s0 | 3.2352 | 3.7517 | 1.8931 | 1.2896 | 0.7470 |
| LSHRH_s1 | 3.2188 | 3.6153 | 1.7391 | 1.3725 | 0.7762 |
| LSHRL_s0 | 3.2226 | 3.7078 | 1.8654 | 1.2713 | 0.7403 |
| LSHRL_s1 | 3.0520 | 3.5169 | 1.7094 | 1.4163 | 0.7766 |
| LSRH_s0 | 3.3109 | 3.6586 | 1.9849 | 1.2439 | 0.6769 |
| LSRH_s1 | 3.1634 | 3.5230 | 1.6962 | 1.2410 | 0.7040 |
| LSRL_s0 | 3.2247 | 3.6484 | 1.9776 | 1.2477 | 0.6774 |
| LSRL_s1 | 2.8827 | 3.2751 | 1.6243 | 1.2236 | 0.7032 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
