# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "GRW6"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## GRW6 minus GFT6

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0654 / +0.0352 | -0.0151 | [-0.1019, +0.0651] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0245 / +0.0998 | +0.0622 | [-0.0356, +0.1617] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0038 / +0.0148 | +0.0055 | [-0.0230, +0.0356] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.0425 / -0.3507 | -0.1966 | [-0.5068, +0.1125] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0046 / -0.0174 | -0.0064 | [-0.0226, +0.0132] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.0048 / -2.8741 | -1.4395 | [-3.7937, +0.5976] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0371 / -0.0063 | -0.0217 | [-0.0452, +0.0027] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0192 / -0.0017 | -0.0104 | [-0.0284, +0.0056] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0363 / -0.0067 | -0.0215 | [-0.0452, +0.0021] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0443 / +0.1700 | +0.1071 | [-0.0290, +0.2507] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0513 / +0.0580 | +0.0546 | [-0.0120, +0.1309] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0078 / +0.1846 | +0.0962 | [-0.0689, +0.2610] | 0.3535 | OPEN |
| ade6_m|all | +0.0315 / +0.1157 | +0.0736 | [-0.0207, +0.1723] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0443 / +0.1700 | +0.1071 | [-0.0290, +0.2507] | 0.5955 | OPEN |

## GRW6 minus GRT6

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0808 / +0.0201 | -0.0303 | [-0.1136, +0.0604] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0198 / +0.0382 | +0.0092 | [-0.0702, +0.0980] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0435 / +0.0053 | -0.0191 | [-0.0578, +0.0224] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.3276 / -0.3177 | +0.0049 | [-0.5084, +0.5209] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0254 / -0.0173 | +0.0041 | [-0.0231, +0.0341] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.0512 / -1.9260 | -0.4374 | [-2.8359, +1.7045] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0017 / -0.0292 | -0.0138 | [-0.0373, +0.0102] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0092 / -0.0238 | -0.0165 | [-0.0323, -0.0008] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0050 / -0.0358 | -0.0154 | [-0.0444, +0.0133] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0577 / +0.0543 | -0.0017 | [-0.1235, +0.1389] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0027 / +0.0086 | +0.0030 | [-0.0566, +0.0596] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0739 / +0.0742 | +0.0001 | [-0.1341, +0.1611] | 0.3535 | OPEN |
| ade6_m|all | -0.0352 / +0.0385 | +0.0017 | [-0.0768, +0.0960] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0577 / +0.0543 | -0.0017 | [-0.1235, +0.1389] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| GFT6_s0 | 2.7051 | 3.1752 | 1.7108 | 1.1587 | 0.7036 |
| GFT6_s1 | 2.6642 | 3.1115 | 1.6200 | 1.1726 | 0.6478 |
| GRT6_s0 | 2.7718 | 3.2292 | 1.7261 | 1.1984 | 0.6624 |
| GRT6_s1 | 2.7414 | 3.1608 | 1.6352 | 1.1820 | 0.6769 |
| GRW6_s0 | 2.7366 | 3.2265 | 1.6454 | 1.1549 | 0.6674 |
| GRW6_s1 | 2.7799 | 3.1694 | 1.6553 | 1.1874 | 0.6411 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
