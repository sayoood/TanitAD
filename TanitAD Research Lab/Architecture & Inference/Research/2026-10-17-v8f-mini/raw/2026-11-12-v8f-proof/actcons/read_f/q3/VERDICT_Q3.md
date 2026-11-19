# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "RLA3"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045083, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## RLA3 minus RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0682 / +0.0153 | +0.0418 | [-0.0055, +0.0948] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0371 / -0.0173 | +0.0099 | [-0.0336, +0.0527] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0268 / +0.0129 | +0.0199 | [+0.0012, +0.0398] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.0530 / -0.2739 | -0.1104 | [-0.4204, +0.1682] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0016 / -0.0209 | -0.0097 | [-0.0264, +0.0073] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.3764 / -0.3068 | +0.5348 | [-1.0890, +2.5715] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0004 / +0.0000 | +0.0002 | [-0.0104, +0.0108] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0250 / +0.0046 | -0.0102 | [-0.0325, +0.0115] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0033 / +0.0083 | +0.0025 | [-0.0111, +0.0156] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0704 / -0.0110 | +0.0297 | [-0.0340, +0.0940] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0368 / -0.0296 | +0.0036 | [-0.0475, +0.0582] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0605 / +0.0221 | +0.0413 | [-0.0031, +0.0847] | 0.3535 | OPEN |
| ade6_m|all | +0.0476 / -0.0060 | +0.0208 | [-0.0216, +0.0634] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0704 / -0.0110 | +0.0297 | [-0.0340, +0.0940] | 0.5955 | OPEN |

## RLA3 minus RLAm

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0790 / -0.0513 | +0.0139 | [-0.0707, +0.0979] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0260 / -0.0264 | -0.0002 | [-0.0464, +0.0478] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0272 / -0.0114 | +0.0079 | [-0.0209, +0.0388] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.2373 / -0.3048 | -0.0337 | [-0.4935, +0.4067] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0046 / -0.0065 | -0.0056 | [-0.0131, +0.0026] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.1856 / -0.8378 | +0.1739 | [-1.3536, +1.8462] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0183 / -0.0083 | -0.0133 | [-0.0263, -0.0004] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0242 / +0.0088 | -0.0077 | [-0.0313, +0.0156] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0200 / +0.0054 | -0.0073 | [-0.0260, +0.0110] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0579 / -0.0405 | +0.0087 | [-0.0687, +0.0900] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0594 / -0.0385 | +0.0104 | [-0.0630, +0.0864] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0144 / -0.0115 | +0.0014 | [-0.0428, +0.0446] | 0.3535 | OPEN |
| ade6_m|all | +0.0389 / -0.0262 | +0.0063 | [-0.0449, +0.0604] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0579 / -0.0405 | +0.0087 | [-0.0687, +0.0900] | 0.5955 | OPEN |

## RLA2 minus RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0570 / +0.0054 | +0.0312 | [-0.0160, +0.0856] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0148 / -0.0657 | -0.0255 | [-0.0819, +0.0319] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0456 / +0.0531 | +0.0493 | [+0.0289, +0.0679] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.2316 / -0.8512 | -0.5414 | [-1.0610, -0.0937] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0162 / -0.0386 | -0.0274 | [-0.0457, -0.0115] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -2.4179 / -2.3337 | -2.3758 | [-3.9884, -0.8553] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0029 / -0.0108 | -0.0040 | [-0.0179, +0.0106] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0263 / +0.0071 | -0.0096 | [-0.0358, +0.0135] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0042 / -0.0038 | +0.0002 | [-0.0115, +0.0131] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0705 / -0.0483 | +0.0111 | [-0.0715, +0.0962] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0461 / -0.0750 | -0.0144 | [-0.0966, +0.0711] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0477 / +0.0192 | +0.0334 | [-0.0015, +0.0695] | 0.3535 | OPEN |
| ade6_m|all | +0.0468 / -0.0321 | +0.0074 | [-0.0476, +0.0639] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0705 / -0.0483 | +0.0111 | [-0.0715, +0.0962] | 0.5955 | OPEN |

## RLA2 minus RLAm

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0678 / -0.0612 | +0.0033 | [-0.0807, +0.0946] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0037 / -0.0749 | -0.0356 | [-0.0935, +0.0229] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0460 / +0.0288 | +0.0374 | [+0.0158, +0.0621] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0473 / -0.8821 | -0.4647 | [-1.0704, +0.1219] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0224 / -0.0242 | -0.0233 | [-0.0321, -0.0147] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -2.6067 / -2.8664 | -2.7366 | [-4.1660, -1.4462] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0158 / -0.0192 | -0.0175 | [-0.0285, -0.0067] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0254 / +0.0113 | -0.0071 | [-0.0367, +0.0190] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0125 / -0.0067 | -0.0096 | [-0.0208, +0.0008] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0580 / -0.0777 | -0.0099 | [-0.1017, +0.0878] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0686 / -0.0839 | -0.0076 | [-0.1057, +0.0977] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0015 / -0.0144 | -0.0064 | [-0.0428, +0.0317] | 0.3535 | OPEN |
| ade6_m|all | +0.0380 / -0.0522 | -0.0071 | [-0.0681, +0.0581] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0580 / -0.0777 | -0.0099 | [-0.1017, +0.0878] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| RLA2_s0 | 2.7888 | 3.2738 | 1.7435 | 1.3034 | 0.7132 |
| RLA2_s1 | 2.6962 | 3.1611 | 1.7601 | 1.2125 | 0.6269 |
| RLA3_s0 | 2.7897 | 3.2645 | 1.7548 | 1.2846 | 0.7057 |
| RLA3_s1 | 2.7222 | 3.2065 | 1.7700 | 1.1724 | 0.6390 |
| RLAm_s0 | 2.7508 | 3.2052 | 1.6757 | 1.2574 | 0.7257 |
| RLAm_s1 | 2.7484 | 3.2450 | 1.8212 | 1.1837 | 0.6336 |
| RLFT_s0 | 2.7420 | 3.2277 | 1.6865 | 1.2578 | 0.7090 |
| RLFT_s1 | 2.7283 | 3.2361 | 1.7547 | 1.1595 | 0.6307 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
