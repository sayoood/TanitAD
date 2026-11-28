# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 10}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LAm"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LAm minus LFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0444 / +0.0874 | +0.0659 | [+0.0257, +0.1112] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0001 / +0.0854 | +0.0428 | [-0.0143, +0.1046] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0550 / +0.0133 | +0.0342 | [+0.0005, +0.0648] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.0130 / +0.5138 | +0.2634 | [-0.1018, +0.7219] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0023 / +0.0158 | +0.0067 | [-0.0061, +0.0235] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.1727 / +2.1026 | +0.9649 | [-0.6090, +3.0671] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0021 / +0.0192 | +0.0106 | [-0.0048, +0.0275] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0225 / +0.0129 | +0.0177 | [+0.0050, +0.0321] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0104 / +0.0242 | +0.0173 | [+0.0050, +0.0313] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0611 / +0.1233 | +0.0922 | [+0.0385, +0.1543] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0668 / +0.0629 | +0.0649 | [+0.0266, +0.0989] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0076 / +0.1056 | +0.0566 | [-0.0140, +0.1353] | 0.3535 | OPEN |
| ade6_m|all | +0.0398 / +0.0823 | +0.0611 | [+0.0246, +0.1032] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0611 / +0.1233 | +0.0922 | [+0.0385, +0.1543] | 0.5955 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LAm_s0 | 2.6775 | 3.1413 | 1.7420 | 1.2440 | 0.6990 |
| LAm_s1 | 2.6648 | 3.1538 | 1.5820 | 1.1706 | 0.6782 |
| LFT_s0 | 2.6377 | 3.0744 | 1.6976 | 1.1890 | 0.6886 |
| LFT_s1 | 2.5825 | 3.0910 | 1.4947 | 1.1573 | 0.6540 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
