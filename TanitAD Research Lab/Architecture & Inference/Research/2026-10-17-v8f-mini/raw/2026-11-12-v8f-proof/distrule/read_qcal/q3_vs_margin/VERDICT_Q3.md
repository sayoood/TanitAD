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

## LSQC minus LSMC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0063 / -0.0620 | -0.0341 | [-0.0898, +0.0178] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0868 / -0.1953 | -0.1411 | [-0.2640, -0.0377] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0009 / +0.0043 | +0.0026 | [-0.0204, +0.0251] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.0130 / -0.7445 | -0.3787 | [-1.0155, +0.1365] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0111 / -0.0330 | -0.0221 | [-0.0418, -0.0059] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.0597 / -3.9003 | -1.9203 | [-4.8403, +0.5631] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0017 / +0.0000 | -0.0008 | [-0.0085, +0.0069] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0058 / +0.0000 | +0.0029 | [-0.0071, +0.0123] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0008 / -0.0017 | -0.0013 | [-0.0069, +0.0044] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.1169 / -0.2344 | -0.1757 | [-0.3377, -0.0408] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1500 / -0.2165 | -0.1833 | [-0.2981, -0.0789] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.0045 / -0.1471 | -0.0758 | [-0.2732, +0.0708] | 0.3535 | OPEN |
| ade6_m|all | -0.0837 / -0.1848 | -0.1343 | [-0.2553, -0.0332] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.1169 / -0.2344 | -0.1757 | [-0.3377, -0.0408] | 0.5955 | INSIDE-SPREAD |

## LSDQC minus LSDMC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0281 / +0.0108 | +0.0195 | [-0.0107, +0.0500] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0408 / -0.0131 | +0.0138 | [-0.0712, +0.0987] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0015 / +0.0212 | +0.0114 | [-0.0153, +0.0372] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.4338 / -0.1085 | +0.1626 | [-0.2125, +0.6114] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0070 / -0.0137 | -0.0034 | [-0.0203, +0.0120] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.6413 / -0.9201 | +0.3606 | [-1.4736, +2.2712] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0042 / +0.0017 | +0.0029 | [-0.0054, +0.0115] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0167 / +0.0217 | +0.0192 | [+0.0090, +0.0300] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0025 / +0.0067 | +0.0046 | [-0.0012, +0.0108] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0677 / +0.0196 | +0.0437 | [-0.0702, +0.1646] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0366 / -0.0610 | -0.0488 | [-0.1392, +0.0468] | 0.4517 | OPEN |
| ade6_m|bulk | +0.1201 / +0.0765 | +0.0983 | [-0.0188, +0.2218] | 0.3535 | OPEN |
| ade6_m|all | +0.0349 / +0.0017 | +0.0183 | [-0.0607, +0.1006] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0677 / +0.0196 | +0.0437 | [-0.0702, +0.1646] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LSDMC_s0 | 4.0227 | 4.3023 | 2.1629 | 1.2615 | 0.7570 |
| LSDMC_s1 | 3.7735 | 3.9747 | 2.1008 | 1.2822 | 0.7420 |
| LSDQC_s0 | 4.0576 | 4.2657 | 2.1910 | 1.2630 | 0.7595 |
| LSDQC_s1 | 3.7752 | 3.9137 | 2.1116 | 1.3035 | 0.7486 |
| LSMC_s0 | 3.8137 | 4.1854 | 2.1411 | 1.2858 | 0.6874 |
| LSMC_s1 | 3.7776 | 4.0018 | 1.9089 | 1.2842 | 0.7090 |
| LSQC_s0 | 3.7300 | 4.0354 | 2.1348 | 1.2867 | 0.6865 |
| LSQC_s1 | 3.5927 | 3.7853 | 1.8469 | 1.2885 | 0.7074 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
