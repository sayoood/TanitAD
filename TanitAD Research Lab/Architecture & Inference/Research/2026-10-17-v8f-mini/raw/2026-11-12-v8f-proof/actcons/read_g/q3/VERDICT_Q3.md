# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "RLAm"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045083, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## RLAm minus RLFT

Seeds read [2, 3] of [2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0319 / -0.0336 | -0.0009 | [-0.0538, +0.0508] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0097 / -0.0489 | -0.0196 | [-0.0686, +0.0267] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0111 / +0.0396 | +0.0253 | [-0.0011, +0.0507] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.2182 / +0.5952 | +0.4067 | [+0.1052, +0.7576] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0063 / -0.0075 | -0.0006 | [-0.0110, +0.0097] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.4176 / -0.1544 | +0.1316 | [-0.7586, +1.0758] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0008 / -0.0179 | -0.0085 | [-0.0235, +0.0063] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0183 / +0.0304 | +0.0244 | [+0.0119, +0.0397] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0067 / -0.0046 | +0.0010 | [-0.0106, +0.0125] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0253 / -0.0301 | -0.0024 | [-0.0607, +0.0572] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0047 / +0.0212 | +0.0130 | [-0.0267, +0.0517] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0310 / -0.0725 | -0.0207 | [-0.0939, +0.0621] | 0.3535 | OPEN |
| ade6_m|all | +0.0167 / -0.0215 | -0.0024 | [-0.0424, +0.0382] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0253 / -0.0301 | -0.0024 | [-0.0607, +0.0572] | 0.5955 | OPEN |

## RLA3 minus RLFT

Seeds read [2, 3] of [2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0352 / -0.0162 | +0.0095 | [-0.0393, +0.0515] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0200 / +0.0018 | +0.0109 | [-0.0215, +0.0368] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0310 / +0.0240 | +0.0275 | [+0.0099, +0.0456] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.0088 / +0.4608 | +0.2348 | [-0.1389, +0.6396] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0044 / -0.0053 | -0.0004 | [-0.0089, +0.0091] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.3671 / -1.0106 | -0.6889 | [-2.1316, +0.4033] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0121 / -0.0146 | -0.0133 | [-0.0265, -0.0008] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0467 / +0.0046 | +0.0256 | [-0.0010, +0.0571] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0079 / -0.0133 | -0.0027 | [-0.0217, +0.0156] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0627 / +0.0167 | +0.0397 | [-0.0164, +0.0884] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0553 / +0.0535 | +0.0544 | [+0.0140, +0.0920] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0258 / -0.0382 | -0.0062 | [-0.0639, +0.0436] | 0.3535 | OPEN |
| ade6_m|all | +0.0418 / +0.0117 | +0.0267 | [-0.0109, +0.0590] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0627 / +0.0167 | +0.0397 | [-0.0164, +0.0884] | 0.5955 | OPEN |

## RLA2 minus RLFT

Seeds read [2, 3] of [2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0234 / -0.0494 | -0.0364 | [-0.0707, -0.0049] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0348 / -0.0375 | -0.0362 | [-0.0682, -0.0079] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0131 / +0.0086 | -0.0023 | [-0.0231, +0.0197] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.2292 / +0.2378 | +0.0043 | [-0.3731, +0.4395] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0059 / -0.0114 | -0.0028 | [-0.0152, +0.0108] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.2142 / -0.8423 | -0.5283 | [-1.4002, +0.3607] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0054 / -0.0154 | -0.0050 | [-0.0233, +0.0119] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0088 / -0.0088 | +0.0000 | [-0.0156, +0.0142] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0163 / -0.0154 | +0.0004 | [-0.0229, +0.0223] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0673 / -0.0340 | -0.0506 | [-0.1035, -0.0021] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0466 / +0.0096 | -0.0185 | [-0.0703, +0.0352] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0436 / -0.0644 | -0.0540 | [-0.0988, -0.0086] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | -0.0452 / -0.0241 | -0.0347 | [-0.0699, -0.0022] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.0673 / -0.0340 | -0.0506 | [-0.1035, -0.0021] | 0.5955 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| RLA2_s2 | 2.6371 | 3.1787 | 1.5164 | 1.2461 | 0.6269 |
| RLA2_s3 | 2.8159 | 3.2826 | 1.7587 | 1.2514 | 0.6711 |
| RLA3_s2 | 2.7242 | 3.2806 | 1.5750 | 1.2902 | 0.6186 |
| RLA3_s3 | 2.8516 | 3.3265 | 1.7919 | 1.2668 | 0.6732 |
| RLAm_s2 | 2.6990 | 3.2300 | 1.5717 | 1.2703 | 0.6173 |
| RLAm_s3 | 2.8184 | 3.2943 | 1.7745 | 1.2825 | 0.6820 |
| RLFT_s2 | 2.6823 | 3.2253 | 1.5398 | 1.2592 | 0.6107 |
| RLFT_s3 | 2.8400 | 3.2731 | 1.8081 | 1.2428 | 0.6865 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
