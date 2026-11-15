# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 10}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LCS"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LCS minus LCSZ

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0043 / +0.0182 | +0.0113 | [-0.0068, +0.0262] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0218 / +0.0266 | +0.0242 | [+0.0114, +0.0370] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0208 / -0.0006 | -0.0107 | [-0.0267, +0.0018] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.0970 / +0.0387 | +0.0679 | [-0.0170, +0.1686] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0072 / +0.0034 | +0.0053 | [+0.0001, +0.0103] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.4484 / +0.1551 | +0.3018 | [-0.1158, +0.7017] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0067 / -0.0038 | +0.0015 | [-0.0073, +0.0110] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0038 / -0.0025 | -0.0031 | [-0.0096, +0.0027] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0104 / -0.0008 | +0.0048 | [-0.0044, +0.0146] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0106 / +0.0341 | +0.0224 | [-0.0014, +0.0431] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0334 / +0.0274 | +0.0304 | [+0.0187, +0.0425] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.0228 / +0.0196 | -0.0016 | [-0.0368, +0.0301] | 0.3535 | OPEN |
| ade6_m|all | +0.0078 / +0.0239 | +0.0158 | [-0.0006, +0.0299] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0106 / +0.0341 | +0.0224 | [-0.0014, +0.0431] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LCS_s0 | 2.8466 | 3.3814 | 1.7981 | 1.2131 | 0.7120 |
| LCS_s1 | 2.8738 | 3.3924 | 1.7879 | 1.3888 | 0.7336 |
| LCSZ_s0 | 2.8388 | 3.3480 | 1.7937 | 1.2339 | 0.7015 |
| LCSZ_s1 | 2.8499 | 3.3649 | 1.7697 | 1.3894 | 0.7345 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
