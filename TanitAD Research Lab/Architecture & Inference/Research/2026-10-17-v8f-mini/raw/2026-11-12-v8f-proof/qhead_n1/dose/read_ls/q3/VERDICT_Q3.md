# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LSQF"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LSQF minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2041 / +0.2098 | +0.2069 | [+0.0858, +0.3297] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.8812 / +0.7503 | +0.8157 | [+0.5298, +1.1328] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0316 / +0.0384 | +0.0350 | [-0.0101, +0.0815] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.2813 / +1.4874 | +1.3843 | [+0.9500, +1.8462] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0304 / +0.0466 | +0.0385 | [+0.0234, +0.0580] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +3.5951 / +6.1381 | +4.8666 | [+1.9774, +7.4231] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0050 / -0.0167 | -0.0058 | [-0.0265, +0.0144] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0146 / +0.0113 | +0.0129 | [-0.0029, +0.0283] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0163 / -0.0008 | +0.0077 | [-0.0075, +0.0244] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.1772 / +1.0165 | +1.0968 | [+0.7005, +1.5270] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.5750 / +0.5036 | +0.5393 | [+0.2960, +0.8057] | 0.4517 | WORSE |
| ade6_m|bulk | +1.1573 / +1.0113 | +1.0843 | [+0.6287, +1.5645] | 0.3535 | WORSE |
| ade6_m|all | +0.8405 / +0.7351 | +0.7878 | [+0.5057, +1.0996] | 0.4002 | WORSE |
| ade_tac_m|all | +1.1772 / +1.0165 | +1.0968 | [+0.7005, +1.5270] | 0.5955 | WORSE |

## LSQF minus LSQC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0172 / -0.0562 | -0.0195 | [-0.0882, +0.0489] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1149 / -0.1690 | -0.0271 | [-0.2478, +0.1945] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0077 / -0.0135 | -0.0029 | [-0.0290, +0.0244] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.3197 / -0.0085 | +0.1556 | [-0.1374, +0.4581] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0035 / +0.0040 | +0.0038 | [-0.0014, +0.0098] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.5945 / +0.3902 | +0.4923 | [-0.4570, +1.4662] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0021 / -0.0079 | -0.0029 | [-0.0127, +0.0073] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0054 / -0.0054 | -0.0054 | [-0.0142, +0.0031] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0004 / -0.0038 | -0.0017 | [-0.0085, +0.0054] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.1712 / -0.2371 | -0.0329 | [-0.3464, +0.2795] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0538 / -0.1113 | -0.0288 | [-0.1841, +0.1222] | 0.4517 | OPEN |
| ade6_m|bulk | +0.1987 / -0.2320 | -0.0167 | [-0.3655, +0.3365] | 0.3535 | OPEN |
| ade6_m|all | +0.1199 / -0.1663 | -0.0232 | [-0.2455, +0.1962] | 0.4002 | OPEN |
| ade_tac_m|all | +0.1712 / -0.2371 | -0.0329 | [-0.3464, +0.2795] | 0.5955 | OPEN |

## LSQF minus LSMC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0109 / -0.1182 | -0.0537 | [-0.1536, +0.0432] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0281 / -0.3643 | -0.1681 | [-0.4437, +0.1196] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0086 / -0.0091 | -0.0003 | [-0.0257, +0.0266] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.2371 / -0.7518 | -0.2573 | [-1.0074, +0.4184] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0185 / -0.0294 | -0.0239 | [-0.0395, -0.0097] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.1899 / -3.5707 | -1.6904 | [-4.4505, +0.9001] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0004 / -0.0079 | -0.0038 | [-0.0129, +0.0056] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0004 / -0.0054 | -0.0025 | [-0.0125, +0.0069] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0004 / -0.0054 | -0.0029 | [-0.0104, +0.0042] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0543 / -0.4716 | -0.2086 | [-0.5881, +0.1827] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0963 / -0.3278 | -0.2120 | [-0.4319, -0.0115] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.1942 / -0.3791 | -0.0925 | [-0.4916, +0.3393] | 0.3535 | OPEN |
| ade6_m|all | +0.0362 / -0.3512 | -0.1575 | [-0.4335, +0.1251] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0543 / -0.4716 | -0.2086 | [-0.5881, +0.1827] | 0.5955 | OPEN |

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
| LSMC_s0 | 3.8137 | 4.1854 | 2.1411 | 1.2858 | 0.6874 |
| LSMC_s1 | 3.7776 | 4.0018 | 1.9089 | 1.2842 | 0.7090 |
| LSQC_s0 | 3.7300 | 4.0354 | 2.1348 | 1.2867 | 0.6865 |
| LSQC_s1 | 3.5927 | 3.7853 | 1.8469 | 1.2885 | 0.7074 |
| LSQF_s0 | 3.8498 | 4.0891 | 2.1520 | 1.2944 | 0.6870 |
| LSQF_s1 | 3.4264 | 3.6740 | 1.7907 | 1.2751 | 0.7036 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
