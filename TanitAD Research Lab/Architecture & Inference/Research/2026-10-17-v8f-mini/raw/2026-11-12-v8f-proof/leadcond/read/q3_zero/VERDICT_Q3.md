# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LC"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LC minus LCZ

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0186 / +0.0036 | +0.0111 | [+0.0009, +0.0230] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0098 / +0.0008 | +0.0053 | [-0.0013, +0.0122] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0004 / -0.0007 | -0.0006 | [-0.0031, +0.0016] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.0377 / +0.0223 | +0.0300 | [-0.0102, +0.0741] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0064 / -0.0035 | -0.0050 | [-0.0112, -0.0015] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.1061 / -0.2772 | -0.0855 | [-0.3946, +0.2610] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0029 / -0.0008 | +0.0010 | [-0.0027, +0.0048] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0008 / +0.0013 | +0.0002 | [-0.0027, +0.0033] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0029 / -0.0013 | +0.0008 | [-0.0027, +0.0046] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0125 / -0.0005 | +0.0060 | [-0.0032, +0.0158] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0072 / +0.0018 | +0.0045 | [-0.0002, +0.0096] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0103 / -0.0025 | +0.0039 | [-0.0048, +0.0135] | 0.3535 | OPEN |
| ade6_m|all | +0.0086 / -0.0001 | +0.0042 | [-0.0019, +0.0109] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0125 / -0.0005 | +0.0060 | [-0.0032, +0.0158] | 0.5955 | OPEN |

## LCO minus LCOZ

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0160 / +0.0180 | +0.0170 | [+0.0086, +0.0267] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0283 / +0.0275 | +0.0279 | [+0.0200, +0.0356] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0035 / +0.0026 | +0.0031 | [-0.0001, +0.0059] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.0509 / +0.0313 | +0.0411 | [-0.0092, +0.0875] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0046 / +0.0022 | +0.0034 | [+0.0000, +0.0071] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.0541 / +0.5735 | +0.2597 | [-0.2400, +0.7291] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0029 / +0.0033 | +0.0031 | [-0.0000, +0.0065] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0017 / +0.0046 | +0.0031 | [-0.0006, +0.0071] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0021 / +0.0050 | +0.0035 | [+0.0004, +0.0073] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0410 / +0.0411 | +0.0410 | [+0.0296, +0.0523] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0216 / +0.0253 | +0.0235 | [+0.0157, +0.0328] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0350 / +0.0319 | +0.0335 | [+0.0217, +0.0465] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.0277 / +0.0283 | +0.0280 | [+0.0202, +0.0358] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0410 / +0.0411 | +0.0410 | [+0.0296, +0.0523] | 0.5955 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LC_s0 | 2.7007 | 3.1225 | 1.7669 | 1.1882 | 0.7153 |
| LC_s1 | 2.6943 | 3.2161 | 1.5949 | 1.1787 | 0.6674 |
| LCO_s0 | 4.2606 | 4.2926 | 2.3186 | 1.4582 | 0.7032 |
| LCO_s1 | 4.3062 | 4.3607 | 2.2309 | 1.2931 | 0.7128 |
| LCOZ_s0 | 4.2329 | 4.2710 | 2.3027 | 1.4547 | 0.7011 |
| LCOZ_s1 | 4.2779 | 4.3353 | 2.2129 | 1.2904 | 0.7078 |
| LCZ_s0 | 2.6921 | 3.1152 | 1.7484 | 1.1886 | 0.7124 |
| LCZ_s1 | 2.6944 | 3.2142 | 1.5912 | 1.1794 | 0.6686 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
