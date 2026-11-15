# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 10}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "RL"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## RL minus LS

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1374 / +0.2958 | +0.2166 | [+0.0662, +0.3591] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.2730 / +0.1656 | +0.2193 | [+0.0977, +0.3465] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.1004 / +0.0721 | +0.0862 | [+0.0348, +0.1374] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.5184 / -0.0598 | -0.2891 | [-0.8097, +0.2701] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0153 / -0.0150 | -0.0151 | [-0.0320, +0.0001] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -3.4094 / +2.2761 | -0.5667 | [-4.5867, +3.2282] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.1080 / -0.0071 | +0.0504 | [-0.0240, +0.1274] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0217 / -0.0058 | -0.0138 | [-0.0398, +0.0100] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.1067 / -0.0017 | +0.0525 | [-0.0186, +0.1254] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.4692 / +0.3066 | +0.3879 | [+0.2077, +0.5673] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1964 / +0.1440 | +0.1702 | [+0.0552, +0.2827] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.4518 / +0.2729 | +0.3623 | [+0.1706, +0.5601] | 0.3535 | WORSE |
| ade6_m|all | +0.3129 / +0.2028 | +0.2578 | [+0.1356, +0.3789] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.4692 / +0.3066 | +0.3879 | [+0.2077, +0.5673] | 0.5955 | INSIDE-SPREAD |

## RL minus G

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1734 / +0.1450 | +0.1592 | [+0.0764, +0.2515] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.1646 / +0.0781 | +0.1213 | [+0.0058, +0.2492] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.1361 / -0.0130 | +0.0616 | [-0.0429, +0.1665] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -2.1379 / +0.7890 | -0.6745 | [-2.5105, +0.9986] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0674 / +0.0230 | -0.0222 | [-0.0813, +0.0301] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -8.1627 / +3.9158 | -2.1234 | [-9.6631, +4.6451] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0421 / -0.0121 | +0.0150 | [-0.0296, +0.0584] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0179 / +0.0213 | +0.0017 | [-0.0338, +0.0383] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0388 / -0.0013 | +0.0188 | [-0.0177, +0.0554] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.3484 / +0.0798 | +0.2141 | [-0.0117, +0.4561] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0303 / +0.0744 | +0.0524 | [-0.0612, +0.1707] | 0.4517 | OPEN |
| ade6_m|bulk | +0.4687 / +0.0329 | +0.2508 | [-0.0631, +0.5793] | 0.3535 | OPEN |
| ade6_m|all | +0.2302 / +0.0555 | +0.1429 | [-0.0068, +0.3030] | 0.4002 | OPEN |
| ade_tac_m|all | +0.3484 / +0.0798 | +0.2141 | [-0.0117, +0.4561] | 0.5955 | OPEN |

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
| RL_s0 | 3.3222 | 3.7106 | 2.0852 | 1.3632 | 0.7774 |
| RL_s1 | 2.8940 | 3.3144 | 1.8767 | 1.3087 | 0.7028 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
