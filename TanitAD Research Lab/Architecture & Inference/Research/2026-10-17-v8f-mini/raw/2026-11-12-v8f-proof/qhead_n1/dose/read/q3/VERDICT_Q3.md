# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LSDQF"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LSDQF minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1216 / +0.0817 | +0.1016 | [-0.0328, +0.2285] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.7363 / +0.7268 | +0.7316 | [+0.4597, +1.0360] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0281 / -0.0542 | -0.0131 | [-0.0842, +0.0511] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.1439 / +1.0608 | +1.1023 | [+0.7299, +1.4889] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0256 / +0.0276 | +0.0266 | [+0.0164, +0.0380] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.1509 / +2.5791 | +1.8650 | [-0.5285, +3.8505] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0021 / -0.0096 | -0.0058 | [-0.0169, +0.0050] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0488 / +0.0179 | +0.0333 | [+0.0081, +0.0605] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0108 / +0.0054 | +0.0081 | [+0.0000, +0.0167] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.9746 / +0.9111 | +0.9429 | [+0.5532, +1.3493] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.4064 / +0.4118 | +0.4091 | [+0.1729, +0.6779] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +1.0427 / +0.9528 | +0.9977 | [+0.5868, +1.4332] | 0.3535 | WORSE |
| ade6_m|all | +0.6965 / +0.6585 | +0.6775 | [+0.4004, +0.9696] | 0.4002 | WORSE |
| ade_tac_m|all | +0.9746 / +0.9111 | +0.9429 | [+0.5532, +1.3493] | 0.5955 | WORSE |

## LSDQF minus LSDQC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1358 / -0.0242 | -0.0800 | [-0.1739, +0.0117] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.2886 / -0.1149 | -0.2017 | [-0.3904, -0.0246] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0265 / +0.0072 | -0.0097 | [-0.0450, +0.0218] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.2726 / -0.0588 | -0.1657 | [-0.4847, +0.1309] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0079 / -0.0031 | -0.0055 | [-0.0137, +0.0023] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.4748 / -0.3681 | -0.9214 | [-2.2871, +0.3740] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0058 / +0.0050 | +0.0054 | [-0.0021, +0.0127] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0146 / -0.0054 | -0.0100 | [-0.0221, +0.0029] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0008 / +0.0050 | +0.0021 | [-0.0048, +0.0094] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.4007 / -0.1497 | -0.2752 | [-0.5448, -0.0219] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2791 / +0.0047 | -0.1372 | [-0.3572, +0.0593] | 0.4517 | OPEN |
| ade6_m|bulk | -0.3038 / -0.2417 | -0.2727 | [-0.5209, -0.0414] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | -0.2904 / -0.1076 | -0.1990 | [-0.3911, -0.0171] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.4007 / -0.1497 | -0.2752 | [-0.5448, -0.0219] | 0.5955 | INSIDE-SPREAD |

## LSDQF minus LSDMC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1077 / -0.0134 | -0.0605 | [-0.1379, +0.0228] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.2478 / -0.1279 | -0.1879 | [-0.3459, -0.0375] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0250 / +0.0284 | +0.0017 | [-0.0428, +0.0430] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.2306 / -0.1409 | +0.0448 | [-0.2853, +0.4584] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0010 / -0.0183 | -0.0087 | [-0.0248, +0.0061] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0993 / -1.2108 | -0.5557 | [-2.0503, +0.8807] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0100 / +0.0067 | +0.0083 | [-0.0002, +0.0171] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0021 / +0.0163 | +0.0092 | [-0.0046, +0.0235] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0017 / +0.0117 | +0.0067 | [-0.0025, +0.0154] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.3330 / -0.1300 | -0.2315 | [-0.4687, -0.0128] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.3157 / -0.0562 | -0.1860 | [-0.4047, +0.0119] | 0.4517 | OPEN |
| ade6_m|bulk | -0.1837 / -0.1652 | -0.1744 | [-0.3788, +0.0226] | 0.3535 | OPEN |
| ade6_m|all | -0.2555 / -0.1059 | -0.1807 | [-0.3493, -0.0225] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.3330 / -0.1300 | -0.2315 | [-0.4687, -0.0128] | 0.5955 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LSD_s0 | 3.0707 | 3.5802 | 1.9336 | 1.2084 | 0.7478 |
| LSD_s1 | 3.0090 | 3.5066 | 2.0057 | 1.3649 | 0.7482 |
| LSDMC_s0 | 4.0227 | 4.3023 | 2.1629 | 1.2615 | 0.7570 |
| LSDMC_s1 | 3.7735 | 3.9747 | 2.1008 | 1.2822 | 0.7420 |
| LSDQC_s0 | 4.0576 | 4.2657 | 2.1910 | 1.2630 | 0.7595 |
| LSDQC_s1 | 3.7752 | 3.9137 | 2.1116 | 1.3035 | 0.7486 |
| LSDQF_s0 | 3.7672 | 3.9866 | 2.0552 | 1.2364 | 0.7586 |
| LSDQF_s1 | 3.6676 | 3.9184 | 2.0874 | 1.3107 | 0.7536 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
