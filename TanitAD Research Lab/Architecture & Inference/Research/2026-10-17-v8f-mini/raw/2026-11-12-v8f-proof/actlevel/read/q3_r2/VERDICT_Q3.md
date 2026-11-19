# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 16}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "RLVa"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## RLVa minus RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0452 / +0.0075 | -0.0189 | [-0.0645, +0.0295] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0526 / -0.0333 | -0.0430 | [-0.0753, -0.0091] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0001 / -0.0169 | -0.0084 | [-0.0255, +0.0118] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.3148 / -0.4567 | -0.3857 | [-0.6410, -0.1547] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0027 / -0.0193 | -0.0110 | [-0.0257, +0.0016] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.2589 / -0.8083 | -0.5336 | [-1.6056, +0.6320] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0050 / +0.0092 | +0.0021 | [-0.0108, +0.0173] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0271 / -0.0042 | -0.0156 | [-0.0357, +0.0019] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0042 / +0.0029 | -0.0006 | [-0.0119, +0.0111] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0714 / -0.0545 | -0.0629 | [-0.1085, -0.0164] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0491 / -0.0690 | -0.0591 | [-0.0982, -0.0226] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.0468 / +0.0038 | -0.0215 | [-0.0752, +0.0384] | 0.3535 | OPEN |
| ade6_m|all | -0.0480 / -0.0358 | -0.0419 | [-0.0724, -0.0104] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.0714 / -0.0545 | -0.0629 | [-0.1085, -0.0164] | 0.5955 | INSIDE-SPREAD |

## RLVa minus RLAm

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0344 / -0.0591 | -0.0468 | [-0.0861, -0.0072] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0637 / -0.0424 | -0.0531 | [-0.0930, -0.0184] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0004 / -0.0411 | -0.0204 | [-0.0526, +0.0109] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1305 / -0.4876 | -0.3091 | [-0.6691, +0.0628] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0089 / -0.0050 | -0.0069 | [-0.0160, +0.0018] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.4489 / -1.3410 | -0.8949 | [-2.2530, +0.2491] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0238 / +0.0008 | -0.0115 | [-0.0298, +0.0077] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0263 / +0.0000 | -0.0131 | [-0.0340, +0.0081] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0208 / +0.0000 | -0.0104 | [-0.0257, +0.0067] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0839 / -0.0839 | -0.0839 | [-0.1323, -0.0342] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0265 / -0.0780 | -0.0523 | [-0.0996, +0.0001] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0929 / -0.0298 | -0.0613 | [-0.1391, +0.0011] | 0.3535 | OPEN |
| ade6_m|all | -0.0568 / -0.0560 | -0.0564 | [-0.0885, -0.0230] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.0839 / -0.0839 | -0.0839 | [-0.1323, -0.0342] | 0.5955 | INSIDE-SPREAD |

## RLVb minus RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0459 / +0.0654 | +0.0097 | [-0.0659, +0.0888] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0457 / +0.0027 | -0.0215 | [-0.0616, +0.0197] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0545 / +0.0684 | +0.0614 | [+0.0405, +0.0809] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.1900 / +0.0537 | -0.0681 | [-0.4088, +0.1583] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0088 / -0.0251 | -0.0170 | [-0.0322, -0.0039] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -2.1793 / -1.1046 | -1.6419 | [-3.4870, -0.3834] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0075 / +0.0071 | +0.0073 | [-0.0031, +0.0181] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0238 / +0.0267 | +0.0015 | [-0.0329, +0.0331] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0038 / +0.0171 | +0.0104 | [-0.0040, +0.0238] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0025 / +0.0634 | +0.0304 | [-0.0263, +0.0870] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0157 / +0.0506 | +0.0331 | [-0.0080, +0.0748] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0240 / +0.0344 | +0.0052 | [-0.0447, +0.0563] | 0.3535 | OPEN |
| ade6_m|all | -0.0024 / +0.0432 | +0.0204 | [-0.0185, +0.0587] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0025 / +0.0634 | +0.0304 | [-0.0263, +0.0870] | 0.5955 | OPEN |

## RLVb minus RLAm

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0351 / -0.0012 | -0.0181 | [-0.0570, +0.0187] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0568 / -0.0064 | -0.0316 | [-0.0787, +0.0080] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0548 / +0.0441 | +0.0495 | [+0.0325, +0.0685] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0057 / +0.0228 | +0.0086 | [-0.2869, +0.2867] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0150 / -0.0107 | -0.0129 | [-0.0221, -0.0044] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -2.3668 / -1.6364 | -2.0016 | [-3.7028, -0.6929] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0113 / -0.0013 | -0.0063 | [-0.0182, +0.0056] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0229 / +0.0308 | +0.0040 | [-0.0329, +0.0381] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0129 / +0.0142 | +0.0006 | [-0.0194, +0.0204] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0150 / +0.0339 | +0.0095 | [-0.0479, +0.0554] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0382 / +0.0416 | +0.0399 | [+0.0046, +0.0769] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.0701 / +0.0008 | -0.0346 | [-0.1058, +0.0181] | 0.3535 | OPEN |
| ade6_m|all | -0.0112 / +0.0230 | +0.0059 | [-0.0333, +0.0372] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0150 / +0.0339 | +0.0095 | [-0.0479, +0.0554] | 0.5955 | OPEN |

## RLAh minus RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0011 / +0.0435 | +0.0212 | [-0.0204, +0.0677] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0046 / +0.0296 | +0.0125 | [-0.0215, +0.0535] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0545 / +0.0039 | +0.0292 | [-0.0057, +0.0683] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1203 / +0.0535 | -0.0334 | [-0.3558, +0.2020] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0077 / -0.0085 | -0.0081 | [-0.0149, -0.0018] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.8848 / +0.5005 | -0.6921 | [-2.8975, +0.9281] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0004 / +0.0238 | +0.0117 | [-0.0081, +0.0317] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0058 / -0.0200 | -0.0129 | [-0.0277, +0.0027] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0088 / +0.0154 | +0.0121 | [-0.0006, +0.0256] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0451 / +0.0438 | +0.0445 | [+0.0042, +0.0849] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0475 / +0.0050 | +0.0262 | [-0.0147, +0.0693] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0095 / +0.0609 | +0.0352 | [-0.0101, +0.0999] | 0.3535 | OPEN |
| ade6_m|all | +0.0302 / +0.0305 | +0.0303 | [+0.0038, +0.0572] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0451 / +0.0438 | +0.0445 | [+0.0042, +0.0849] | 0.5955 | INSIDE-SPREAD |

## RLAh minus RLAm

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0097 / -0.0231 | -0.0067 | [-0.0513, +0.0305] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0158 / +0.0205 | +0.0024 | [-0.0364, +0.0441] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0549 / -0.0204 | +0.0172 | [-0.0307, +0.0669] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.0640 / +0.0226 | +0.0433 | [-0.2773, +0.3578] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0139 / +0.0059 | -0.0040 | [-0.0195, +0.0105] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -2.0745 / -0.0322 | -1.0534 | [-3.1421, +0.7437] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0192 / +0.0154 | -0.0019 | [-0.0254, +0.0221] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0050 / -0.0158 | -0.0104 | [-0.0254, +0.0037] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0079 / +0.0125 | +0.0023 | [-0.0142, +0.0188] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0326 / +0.0143 | +0.0235 | [-0.0225, +0.0732] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0700 / -0.0039 | +0.0330 | [-0.0260, +0.0929] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0366 / +0.0273 | -0.0046 | [-0.0719, +0.0678] | 0.3535 | OPEN |
| ade6_m|all | +0.0214 / +0.0103 | +0.0159 | [-0.0147, +0.0490] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0326 / +0.0143 | +0.0235 | [-0.0225, +0.0732] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| RLAh_s0 | 2.7722 | 3.2752 | 1.6855 | 1.3123 | 0.7178 |
| RLAh_s1 | 2.7588 | 3.2411 | 1.7981 | 1.1634 | 0.6461 |
| RLAm_s0 | 2.7508 | 3.2052 | 1.6757 | 1.2574 | 0.7257 |
| RLAm_s1 | 2.7484 | 3.2450 | 1.8212 | 1.1837 | 0.6336 |
| RLFT_s0 | 2.7420 | 3.2277 | 1.6865 | 1.2578 | 0.7090 |
| RLFT_s1 | 2.7283 | 3.2361 | 1.7547 | 1.1595 | 0.6307 |
| RLVa_s0 | 2.6940 | 3.1786 | 1.6413 | 1.2578 | 0.7049 |
| RLVa_s1 | 2.6924 | 3.1670 | 1.7621 | 1.1426 | 0.6336 |
| RLVb_s0 | 2.7396 | 3.2434 | 1.6406 | 1.3122 | 0.7128 |
| RLVb_s1 | 2.7714 | 3.2867 | 1.8200 | 1.2278 | 0.6478 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
