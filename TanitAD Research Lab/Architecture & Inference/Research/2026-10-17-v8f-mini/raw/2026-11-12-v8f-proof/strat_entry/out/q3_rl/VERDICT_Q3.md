# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "RLA"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## RLA minus RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0701 / +0.0583 | +0.0642 | [-0.0262, +0.1519] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0321 / +0.2509 | +0.1415 | [-0.0114, +0.3186] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0992 / +0.3528 | +0.2260 | [+0.0714, +0.3983] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.0461 / +0.5368 | +0.2914 | [-0.1826, +0.8622] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0068 / +0.0035 | -0.0017 | [-0.0154, +0.0152] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.9693 / -1.0530 | +0.4582 | [-2.2606, +3.0697] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0150 / +0.1017 | +0.0434 | [-0.0273, +0.1238] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0017 / +0.0963 | +0.0490 | [-0.0112, +0.1145] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0058 / +0.1221 | +0.0581 | [-0.0179, +0.1421] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.1525 / +0.7026 | +0.4276 | [+0.0898, +0.8011] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0856 / +0.5524 | +0.3190 | [+0.0328, +0.6558] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.1236 / +0.3676 | +0.2456 | [+0.0591, +0.4523] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.1030 / +0.4681 | +0.2855 | [+0.0608, +0.5336] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.1525 / +0.7026 | +0.4276 | [+0.0898, +0.8011] | 0.5955 | INSIDE-SPREAD |

## RLAm minus RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0108 / +0.0666 | +0.0279 | [-0.0287, +0.0891] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0111 / +0.0091 | +0.0101 | [-0.0171, +0.0383] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0003 / +0.0243 | +0.0120 | [-0.0132, +0.0363] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1843 / +0.0309 | -0.0767 | [-0.3142, +0.2046] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0062 / -0.0144 | -0.0041 | [-0.0192, +0.0113] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.1903 / +0.5327 | +0.3615 | [-0.8330, +1.8376] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0188 / +0.0083 | +0.0135 | [+0.0002, +0.0263] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0008 / -0.0042 | -0.0025 | [-0.0131, +0.0090] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0167 / +0.0029 | +0.0098 | [-0.0037, +0.0242] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0125 / +0.0295 | +0.0210 | [-0.0185, +0.0601] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0225 / +0.0089 | -0.0068 | [-0.0499, +0.0337] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0461 / +0.0336 | +0.0399 | [+0.0035, +0.0826] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.0088 / +0.0202 | +0.0145 | [-0.0121, +0.0405] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0125 / +0.0295 | +0.0210 | [-0.0185, +0.0601] | 0.5955 | OPEN |

## RLFTb minus RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0249 / +0.0254 | +0.0251 | [-0.0070, +0.0591] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0133 / +0.0300 | +0.0083 | [-0.0273, +0.0454] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0566 / +0.0221 | +0.0393 | [+0.0132, +0.0683] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0910 / +0.2808 | +0.0949 | [-0.2785, +0.4047] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0015 / -0.0034 | -0.0010 | [-0.0104, +0.0066] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.7303 / +0.8066 | +0.7685 | [-0.3362, +1.9825] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0058 / +0.0050 | -0.0004 | [-0.0123, +0.0117] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0033 / +0.0254 | +0.0110 | [-0.0100, +0.0321] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0029 / +0.0071 | +0.0021 | [-0.0100, +0.0135] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0286 / +0.0600 | +0.0443 | [+0.0038, +0.0843] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0223 / +0.0482 | +0.0353 | [-0.0044, +0.0708] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0159 / +0.0316 | +0.0237 | [-0.0062, +0.0572] | 0.3535 | OPEN |
| ade6_m|all | +0.0194 / +0.0406 | +0.0300 | [+0.0024, +0.0568] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0286 / +0.0600 | +0.0443 | [+0.0038, +0.0843] | 0.5955 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| RLA_s0 | 2.8450 | 3.3133 | 1.7566 | 1.3570 | 0.7032 |
| RLA_s1 | 3.1964 | 3.7885 | 1.8130 | 1.5122 | 0.7528 |
| RLAm_s0 | 2.7508 | 3.2052 | 1.6757 | 1.2574 | 0.7257 |
| RLAm_s1 | 2.7484 | 3.2450 | 1.8212 | 1.1837 | 0.6336 |
| RLFT_s0 | 2.7420 | 3.2277 | 1.6865 | 1.2578 | 0.7090 |
| RLFT_s1 | 2.7283 | 3.2361 | 1.7547 | 1.1595 | 0.6307 |
| RLFTb_s0 | 2.7614 | 3.2500 | 1.7114 | 1.3143 | 0.7061 |
| RLFTb_s1 | 2.7689 | 3.2843 | 1.7801 | 1.1815 | 0.6378 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
