# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 10}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "PF"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## PF minus P

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST GAIN-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1342 / -0.2151 | -0.1747 | [-0.2720, -0.0834] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0023 / -0.1998 | -0.1011 | [-0.2614, +0.0471] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0055 / -0.1242 | -0.0649 | [-0.1484, +0.0219] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.0914 / -0.5930 | +0.2492 | [-0.8607, +1.3698] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0626 / +0.0348 | +0.0487 | [+0.0247, +0.0717] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +7.0505 / +0.1516 | +3.6011 | [-1.0517, +8.5607] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0459 / -0.0750 | -0.0604 | [-0.0906, -0.0308] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0484 / -0.0325 | +0.0079 | [-0.0490, +0.0630] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0246 / -0.0913 | -0.0579 | [-0.1051, -0.0127] | 0.0492 (0.0263) | BETTER |
| TACTICAL:ade_tac_m | -0.0227 / -0.3934 | -0.2080 | [-0.4844, +0.0408] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0178 / -0.1461 | -0.0820 | [-0.2049, +0.0359] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0133 / -0.4016 | -0.2075 | [-0.4998, +0.0543] | 0.3535 | OPEN |
| ade6_m|all | -0.0158 / -0.2627 | -0.1392 | [-0.3240, +0.0275] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0227 / -0.3934 | -0.2080 | [-0.4844, +0.0408] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| P_s0 | 3.1244 | 3.5986 | 1.9134 | 1.3721 | 0.7445 |
| P_s1 | 3.1107 | 3.5268 | 1.9957 | 1.3621 | 0.8149 |
| PF_s0 | 3.1086 | 3.5808 | 1.7792 | 1.3666 | 0.7199 |
| PF_s1 | 2.8480 | 3.3806 | 1.7806 | 1.2379 | 0.7236 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
