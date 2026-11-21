# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "RLVc"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## RLVc minus RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0217 / +0.0251 | +0.0017 | [-0.0420, +0.0485] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0057 / -0.0044 | -0.0051 | [-0.0335, +0.0235] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0318 / +0.0191 | +0.0255 | [+0.0060, +0.0461] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.2308 / -0.2894 | -0.2601 | [-0.5525, -0.0053] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0041 / -0.0128 | -0.0044 | [-0.0176, +0.0090] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.3193 / +0.8325 | +0.2566 | [-1.3550, +1.9659] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0088 / +0.0083 | -0.0002 | [-0.0146, +0.0158] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0208 / +0.0096 | -0.0056 | [-0.0263, +0.0162] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0129 / +0.0067 | -0.0031 | [-0.0192, +0.0125] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0203 / +0.0088 | +0.0145 | [-0.0303, +0.0570] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0443 / -0.0202 | +0.0120 | [-0.0418, +0.0649] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0198 / +0.0381 | +0.0092 | [-0.0384, +0.0779] | 0.3535 | OPEN |
| ade6_m|all | +0.0151 / +0.0064 | +0.0107 | [-0.0197, +0.0389] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0203 / +0.0088 | +0.0145 | [-0.0303, +0.0570] | 0.5955 | OPEN |

## RLVc minus RLAm

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0109 / -0.0415 | -0.0262 | [-0.0652, +0.0099] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0168 / -0.0136 | -0.0152 | [-0.0485, +0.0215] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0321 / -0.0051 | +0.0135 | [-0.0158, +0.0437] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.0465 / -0.3204 | -0.1835 | [-0.5606, +0.1635] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0022 / +0.0016 | -0.0003 | [-0.0084, +0.0071] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.5103 / +0.3005 | -0.1049 | [-1.5016, +1.0044] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0275 / +0.0000 | -0.0138 | [-0.0342, +0.0077] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0200 / +0.0138 | -0.0031 | [-0.0281, +0.0219] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0296 / +0.0038 | -0.0129 | [-0.0359, +0.0111] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0078 / -0.0207 | -0.0064 | [-0.0603, +0.0481] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0668 / -0.0292 | +0.0188 | [-0.0571, +0.0923] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0659 / +0.0045 | -0.0307 | [-0.1001, +0.0420] | 0.3535 | OPEN |
| ade6_m|all | +0.0063 / -0.0138 | -0.0038 | [-0.0402, +0.0331] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0078 / -0.0207 | -0.0064 | [-0.0603, +0.0481] | 0.5955 | OPEN |

## RLVd minus RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0408 / +0.0273 | +0.0340 | [+0.0023, +0.0668] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0531 / +0.0077 | +0.0304 | [-0.0084, +0.0702] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0158 / +0.0196 | +0.0177 | [+0.0004, +0.0348] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0792 / -0.0198 | -0.0495 | [-0.2801, +0.1668] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0146 / -0.0112 | +0.0017 | [-0.0167, +0.0208] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0511 / +0.6220 | +0.3365 | [-0.8419, +1.5073] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0179 / +0.0188 | +0.0183 | [+0.0069, +0.0302] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0158 / +0.0058 | -0.0050 | [-0.0229, +0.0106] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0142 / +0.0108 | +0.0125 | [+0.0017, +0.0236] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0810 / +0.0259 | +0.0534 | [+0.0042, +0.1064] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0550 / +0.0152 | +0.0351 | [-0.0024, +0.0801] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0544 / +0.0194 | +0.0369 | [-0.0034, +0.0752] | 0.3535 | OPEN |
| ade6_m|all | +0.0547 / +0.0171 | +0.0359 | [+0.0026, +0.0718] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0810 / +0.0259 | +0.0534 | [+0.0042, +0.1064] | 0.5955 | INSIDE-SPREAD |

## RLVd minus RLAm

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0516 / -0.0393 | +0.0062 | [-0.0655, +0.0724] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0420 / -0.0014 | +0.0203 | [-0.0185, +0.0621] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0162 / -0.0047 | +0.0057 | [-0.0162, +0.0288] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.1051 / -0.0507 | +0.0272 | [-0.2197, +0.2951] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0083 / +0.0031 | +0.0057 | [-0.0026, +0.0154] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.1392 / +0.0887 | -0.0253 | [-1.3579, +1.1374] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0008 / +0.0104 | +0.0048 | [-0.0071, +0.0169] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0150 / +0.0100 | -0.0025 | [-0.0235, +0.0177] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0025 / +0.0079 | +0.0027 | [-0.0081, +0.0142] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0685 / -0.0036 | +0.0325 | [-0.0274, +0.0977] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0775 / +0.0062 | +0.0419 | [-0.0161, +0.1020] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0082 / -0.0142 | -0.0030 | [-0.0457, +0.0432] | 0.3535 | OPEN |
| ade6_m|all | +0.0459 / -0.0031 | +0.0214 | [-0.0188, +0.0655] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0685 / -0.0036 | +0.0325 | [-0.0274, +0.0977] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| RLAm_s0 | 2.7508 | 3.2052 | 1.6757 | 1.2574 | 0.7257 |
| RLAm_s1 | 2.7484 | 3.2450 | 1.8212 | 1.1837 | 0.6336 |
| RLFT_s0 | 2.7420 | 3.2277 | 1.6865 | 1.2578 | 0.7090 |
| RLFT_s1 | 2.7283 | 3.2361 | 1.7547 | 1.1595 | 0.6307 |
| RLVc_s0 | 2.7571 | 3.2720 | 1.6648 | 1.2895 | 0.6961 |
| RLVc_s1 | 2.7346 | 3.2158 | 1.7797 | 1.1786 | 0.6373 |
| RLVd_s0 | 2.7968 | 3.2827 | 1.7273 | 1.2736 | 0.7232 |
| RLVd_s1 | 2.7454 | 3.2513 | 1.7820 | 1.1790 | 0.6415 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
