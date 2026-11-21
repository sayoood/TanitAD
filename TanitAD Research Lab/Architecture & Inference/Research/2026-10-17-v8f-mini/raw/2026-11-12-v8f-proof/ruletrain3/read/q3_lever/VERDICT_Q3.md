# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 16}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "GOL"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## GOL minus GFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.9195 / +0.7665 | +0.8430 | [+0.5220, +1.1721] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +2.1737 / +2.1263 | +2.1500 | [+1.3509, +3.0126] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.1653 / +0.1139 | +0.1396 | [+0.0692, +0.2150] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +4.1276 / +3.0356 | +3.5816 | [+2.4832, +4.7262] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.1049 / +0.0959 | +0.1004 | [+0.0773, +0.1252] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +16.9405 / +13.5223 | +15.2314 | [+10.3118, +20.3912] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0333 / -0.0071 | +0.0131 | [-0.0202, +0.0471] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.1276 / +0.0546 | +0.0911 | [+0.0392, +0.1444] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0613 / +0.0146 | +0.0379 | [+0.0008, +0.0738] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +3.1051 / +2.9620 | +3.0335 | [+1.9289, +4.2294] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.7223 / +1.6786 | +1.7004 | [+1.0644, +2.4015] | 0.4517 | WORSE |
| ade6_m|bulk | +2.6775 / +2.5773 | +2.6274 | [+1.4909, +3.8708] | 0.3535 | WORSE |
| ade6_m|all | +2.1579 / +2.0884 | +2.1231 | [+1.3430, +2.9635] | 0.4002 | WORSE |
| ade_tac_m|all | +3.1051 / +2.9620 | +3.0335 | [+1.9289, +4.2294] | 0.5955 | WORSE |

## GOL4 minus GFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LATERAL COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.3953 / +1.3070 | +1.3512 | [+1.0129, +1.6927] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +3.3300 / +3.3105 | +3.3202 | [+2.4478, +4.2829] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.2398 / +0.2683 | +0.2541 | [+0.1796, +0.3360] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +7.4775 / +6.4950 | +6.9862 | [+5.5562, +8.3896] | 4.5570 (1.9414) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1837 / +0.1518 | +0.1678 | [+0.1317, +0.2068] | 0.1528 (0.0760) | WORSE |
| LATERAL:yaw_rate_mae_degps | +23.8214 / +18.3146 | +21.0680 | [+15.6386, +26.7483] | 15.8479 (7.7515) | WORSE |
| TACTICAL:lon_err_rate | +0.0688 / +0.0434 | +0.0561 | [+0.0258, +0.0859] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.2276 / +0.1576 | +0.1926 | [+0.1394, +0.2464] | 0.1042 (0.0534) | WORSE |
| TACTICAL:joint_err_rate | +0.1055 / +0.1021 | +0.1038 | [+0.0788, +0.1282] | 0.0492 (0.0263) | WORSE |
| TACTICAL:ade_tac_m | +4.7138 / +4.6915 | +4.7027 | [+3.4966, +6.0323] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +2.5633 / +2.6879 | +2.6256 | [+1.8633, +3.4606] | 0.4517 | WORSE |
| ade6_m|bulk | +4.1611 / +4.0178 | +4.0894 | [+2.8084, +5.5080] | 0.3535 | WORSE |
| ade6_m|all | +3.2919 / +3.2943 | +3.2931 | [+2.4300, +4.2344] | 0.4002 | WORSE |
| ade_tac_m|all | +4.7138 / +4.6915 | +4.7027 | [+3.4966, +6.0323] | 0.5955 | WORSE |

## GFTb minus GFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2084 / -0.1002 | +0.0541 | [-0.1332, +0.2575] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1888 / -0.0037 | +0.0926 | [-0.0378, +0.2546] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0951 / -0.0176 | -0.0564 | [-0.1160, +0.0027] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.6049 / +0.1168 | -0.2440 | [-0.8328, +0.3033] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0049 / +0.0043 | +0.0046 | [-0.0072, +0.0161] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.7391 / -0.1098 | -0.4245 | [-1.8243, +0.8546] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0021 / -0.0129 | -0.0054 | [-0.0290, +0.0167] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0242 / -0.0042 | -0.0142 | [-0.0377, +0.0073] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0054 / +0.0100 | +0.0023 | [-0.0202, +0.0269] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.1677 / -0.0348 | +0.0665 | [-0.0867, +0.2623] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0350 / -0.0319 | -0.0335 | [-0.1064, +0.0357] | 0.4517 | OPEN |
| ade6_m|bulk | +0.3037 / -0.0219 | +0.1409 | [-0.0726, +0.4335] | 0.3535 | OPEN |
| ade6_m|all | +0.1194 / -0.0274 | +0.0460 | [-0.0620, +0.1833] | 0.4002 | OPEN |
| ade_tac_m|all | +0.1677 / -0.0348 | +0.0665 | [-0.0867, +0.2623] | 0.5955 | OPEN |

## GOL minus GOW

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.4849 / -0.2896 | -0.3873 | [-0.5539, -0.2282] | 0.2693 (0.1381) | BETTER |
| LONGITUDINAL:along_mae_m | -0.6795 / -0.3958 | -0.5376 | [-0.8224, -0.2773] | 0.3530 (0.2400) | BETTER |
| LATERAL:cross_mae_m | +0.0407 / -0.1466 | -0.0530 | [-0.1839, +0.0606] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.6793 / -1.1093 | -0.8943 | [-1.4097, -0.3052] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0151 / +0.0192 | +0.0020 | [-0.0291, +0.0304] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.8143 / +1.0034 | +0.9089 | [-1.6979, +3.4902] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0625 / -0.0258 | -0.0442 | [-0.0727, -0.0154] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0521 / -0.0725 | -0.0623 | [-0.0867, -0.0356] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0479 / -0.0421 | -0.0450 | [-0.0621, -0.0294] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.9438 / -0.7054 | -0.8246 | [-1.1670, -0.5160] | 0.5955 (0.2199) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2656 / -0.3312 | -0.2984 | [-0.4866, -0.1107] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -1.0584 / -0.6386 | -0.8485 | [-1.2753, -0.4524] | 0.3535 | BETTER |
| ade6_m|all | -0.6271 / -0.4714 | -0.5493 | [-0.7883, -0.3363] | 0.4002 | BETTER |
| ade_tac_m|all | -0.9438 / -0.7054 | -0.8246 | [-1.1670, -0.5160] | 0.5955 | BETTER |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| GFT_s0 | 2.8462 | 3.3885 | 1.6906 | 1.2622 | 0.7449 |
| GFT_s1 | 2.6830 | 3.1140 | 1.6070 | 1.2180 | 0.6474 |
| GFTb_s0 | 2.9656 | 3.3535 | 1.8990 | 1.1671 | 0.7395 |
| GFTb_s1 | 2.6556 | 3.0821 | 1.5067 | 1.2004 | 0.6574 |
| GOL_s0 | 5.0040 | 5.1108 | 2.6102 | 1.4275 | 0.8062 |
| GOL_s1 | 4.7714 | 4.7926 | 2.3735 | 1.3319 | 0.6619 |
| GOL4_s0 | 6.1381 | 5.9518 | 3.0860 | 1.5020 | 0.8504 |
| GOL4_s1 | 5.9773 | 5.8019 | 2.9140 | 1.4863 | 0.7495 |
| GOW_s0 | 5.6312 | 5.3764 | 3.0951 | 1.3868 | 0.8541 |
| GOW_s1 | 5.2428 | 5.1238 | 2.6631 | 1.4785 | 0.7040 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
