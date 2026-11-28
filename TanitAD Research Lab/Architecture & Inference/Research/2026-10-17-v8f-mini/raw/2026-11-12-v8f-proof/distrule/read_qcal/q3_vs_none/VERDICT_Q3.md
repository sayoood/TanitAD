# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LSQC"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LSQC minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1869 / +0.2660 | +0.2265 | [+0.0873, +0.3714] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.7663 / +0.9193 | +0.8428 | [+0.5180, +1.2142] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0239 / +0.0519 | +0.0379 | [-0.0093, +0.0856] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.0113 / +1.4742 | +1.2428 | [+0.7402, +1.7827] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0314 / +0.0471 | +0.0392 | [+0.0231, +0.0589] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.5074 / +5.8287 | +4.1680 | [+0.9790, +6.9779] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0029 / -0.0088 | -0.0029 | [-0.0181, +0.0133] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0200 / +0.0167 | +0.0183 | [+0.0027, +0.0346] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0158 / +0.0029 | +0.0094 | [-0.0046, +0.0252] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.0059 / +1.2536 | +1.1298 | [+0.6829, +1.6503] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.5212 / +0.6149 | +0.5680 | [+0.2999, +0.8865] | 0.4517 | WORSE |
| ade6_m|bulk | +0.9585 / +1.2433 | +1.1009 | [+0.6128, +1.7019] | 0.3535 | WORSE |
| ade6_m|all | +0.7206 / +0.9015 | +0.8110 | [+0.4883, +1.1888] | 0.4002 | WORSE |
| ade_tac_m|all | +1.0059 / +1.2536 | +1.1298 | [+0.6829, +1.6503] | 0.5955 | WORSE |

## LSDQC minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2574 / +0.1059 | +0.1816 | [+0.0159, +0.3612] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +1.0249 / +0.8417 | +0.9333 | [+0.5800, +1.3081] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0546 / -0.0614 | -0.0034 | [-0.0866, +0.0812] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.3758 / +1.1509 | +1.2634 | [+0.8947, +1.6716] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0336 / +0.0335 | +0.0336 | [+0.0239, +0.0445] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.8004 / +2.8213 | +2.8108 | [+0.8504, +4.5870] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0079 / -0.0146 | -0.0113 | [-0.0221, -0.0010] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0634 / +0.0233 | +0.0434 | [+0.0144, +0.0771] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0117 / +0.0004 | +0.0060 | [-0.0033, +0.0171] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.3753 / +1.0608 | +1.2180 | [+0.7097, +1.7509] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.6855 / +0.4071 | +0.5463 | [+0.2300, +0.9161] | 0.4517 | WORSE |
| ade6_m|bulk | +1.3465 / +1.1945 | +1.2705 | [+0.7485, +1.8420] | 0.3535 | WORSE |
| ade6_m|all | +0.9869 / +0.7662 | +0.8765 | [+0.5107, +1.2607] | 0.4002 | WORSE |
| ade_tac_m|all | +1.3753 / +1.0608 | +1.2180 | [+0.7097, +1.7509] | 0.5955 | WORSE |

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
| LSDQC_s0 | 4.0576 | 4.2657 | 2.1910 | 1.2630 | 0.7595 |
| LSDQC_s1 | 3.7752 | 3.9137 | 2.1116 | 1.3035 | 0.7486 |
| LSQC_s0 | 3.7300 | 4.0354 | 2.1348 | 1.2867 | 0.6865 |
| LSQC_s1 | 3.5927 | 3.7853 | 1.8469 | 1.2885 | 0.7074 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
