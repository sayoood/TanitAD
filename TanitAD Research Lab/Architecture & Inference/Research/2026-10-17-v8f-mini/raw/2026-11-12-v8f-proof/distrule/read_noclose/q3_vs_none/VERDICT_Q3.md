# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 18}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LSRZ"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LSRZ minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0049 / +0.1057 | +0.0553 | [-0.0553, +0.1655] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2739 / +0.4579 | +0.3659 | [+0.1496, +0.6189] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0249 / -0.0035 | -0.0142 | [-0.0413, +0.0154] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.2323 / +0.8512 | +0.5418 | [+0.1119, +1.0666] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0057 / +0.0336 | +0.0196 | [+0.0042, +0.0428] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.7097 / +3.6836 | +1.4869 | [-1.5450, +4.5512] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0113 / -0.0033 | -0.0073 | [-0.0177, +0.0035] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0017 / -0.0025 | -0.0004 | [-0.0125, +0.0113] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0017 / +0.0029 | +0.0023 | [-0.0046, +0.0104] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.3257 / +0.5981 | +0.4619 | [+0.1587, +0.8057] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1515 / +0.3471 | +0.2493 | [+0.0425, +0.4906] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.3403 / +0.5353 | +0.4378 | [+0.1648, +0.7582] | 0.3535 | WORSE |
| ade6_m|all | +0.2376 / +0.4329 | +0.3353 | [+0.1177, +0.5853] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.3257 / +0.5981 | +0.4619 | [+0.1587, +0.8057] | 0.5955 | INSIDE-SPREAD |

## LSDRZ minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0343 / -0.0806 | -0.0232 | [-0.1331, +0.0856] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3404 / +0.2121 | +0.2762 | [+0.0984, +0.4886] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0092 / -0.0653 | -0.0281 | [-0.0832, +0.0210] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.6579 / +0.2226 | +0.4402 | [+0.1228, +0.8144] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0105 / +0.0168 | +0.0136 | [+0.0060, +0.0252] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.2256 / -0.1672 | -0.1964 | [-1.2406, +0.8703] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0071 / -0.0108 | -0.0090 | [-0.0165, -0.0019] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0317 / +0.0004 | +0.0160 | [-0.0052, +0.0409] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0067 / -0.0033 | +0.0017 | [-0.0067, +0.0102] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.4508 / +0.2210 | +0.3359 | [+0.0804, +0.6410] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2254 / +0.0732 | +0.1493 | [-0.0170, +0.3503] | 0.4517 | OPEN |
| ade6_m|bulk | +0.4410 / +0.2732 | +0.3571 | [+0.1291, +0.6469] | 0.3535 | WORSE |
| ade6_m|all | +0.3237 / +0.1644 | +0.2441 | [+0.0602, +0.4631] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.4508 / +0.2210 | +0.3359 | [+0.0804, +0.6410] | 0.5955 | INSIDE-SPREAD |

## LSQZ minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1498 / +0.2287 | +0.1893 | [+0.0480, +0.3309] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.7199 / +0.8191 | +0.7695 | [+0.4542, +1.1193] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0091 / +0.0417 | +0.0254 | [-0.0220, +0.0704] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.0118 / +1.4025 | +1.2072 | [+0.7544, +1.6859] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0322 / +0.0437 | +0.0379 | [+0.0248, +0.0554] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.3654 / +5.9833 | +4.1744 | [+0.9992, +7.1845] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0054 / -0.0038 | +0.0008 | [-0.0133, +0.0152] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0171 / +0.0167 | +0.0169 | [+0.0008, +0.0342] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0163 / +0.0033 | +0.0098 | [-0.0031, +0.0252] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.9329 / +1.1156 | +1.0242 | [+0.5841, +1.5047] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.4926 / +0.5942 | +0.5434 | [+0.2705, +0.8564] | 0.4517 | WORSE |
| ade6_m|bulk | +0.8813 / +1.0580 | +0.9696 | [+0.5141, +1.4837] | 0.3535 | WORSE |
| ade6_m|all | +0.6698 / +0.8057 | +0.7378 | [+0.4191, +1.0905] | 0.4002 | WORSE |
| ade_tac_m|all | +0.9329 / +1.1156 | +1.0242 | [+0.5841, +1.5047] | 0.5955 | WORSE |

## LSDQZ minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2303 / +0.0550 | +0.1427 | [-0.0307, +0.3240] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.9620 / +0.7392 | +0.8506 | [+0.5177, +1.2217] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0746 / -0.0624 | +0.0061 | [-0.0921, +0.1034] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.3093 / +1.0399 | +1.1746 | [+0.8171, +1.5812] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0383 / +0.0310 | +0.0346 | [+0.0230, +0.0482] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.6342 / +3.1494 | +2.8918 | [+1.3098, +4.3962] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0154 / -0.0188 | -0.0171 | [-0.0283, -0.0058] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0571 / +0.0154 | +0.0363 | [+0.0054, +0.0704] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0054 / -0.0038 | +0.0008 | [-0.0092, +0.0117] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.2994 / +0.9133 | +1.1063 | [+0.6200, +1.6376] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.6726 / +0.4098 | +0.5412 | [+0.2418, +0.9168] | 0.4517 | WORSE |
| ade6_m|bulk | +1.2500 / +0.9718 | +1.1109 | [+0.6097, +1.6739] | 0.3535 | WORSE |
| ade6_m|all | +0.9359 / +0.6661 | +0.8010 | [+0.4530, +1.1823] | 0.4002 | WORSE |
| ade_tac_m|all | +1.2994 / +0.9133 | +1.1063 | [+0.6200, +1.6376] | 0.5955 | WORSE |

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
| LSDQZ_s0 | 4.0066 | 4.2528 | 2.1639 | 1.2829 | 0.7532 |
| LSDQZ_s1 | 3.6751 | 3.9164 | 2.0608 | 1.3025 | 0.7445 |
| LSDRZ_s0 | 3.3944 | 3.8055 | 1.9679 | 1.2175 | 0.7545 |
| LSDRZ_s1 | 3.1735 | 3.5798 | 1.9251 | 1.2996 | 0.7449 |
| LSQZ_s0 | 3.6792 | 4.0067 | 2.0977 | 1.2719 | 0.6870 |
| LSQZ_s1 | 3.4970 | 3.7646 | 1.8097 | 1.2783 | 0.7078 |
| LSRZ_s0 | 3.2469 | 3.6656 | 1.9528 | 1.2379 | 0.6724 |
| LSRZ_s1 | 3.1242 | 3.5175 | 1.6866 | 1.2331 | 0.7074 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
