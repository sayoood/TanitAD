# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 18}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "H_R0A"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.1272268626789459, "D_read": 0.1272268626789459, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm U_RLcK, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## H R0A minus H R0K

Seeds read [0, 1, 2] of [0, 1, 2]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2100 / +0.2354 / +0.1569 | +0.2008 | [+0.1534, +0.2481] | 0.0424 (0.0250) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3082 / +0.3005 / +0.2013 | +0.2700 | [+0.1982, +0.3414] | 0.0348 (0.0275) | WORSE |
| LATERAL:cross_mae_m | +0.0006 / +0.0064 / +0.0032 | +0.0034 | [-0.0056, +0.0120] | 0.0276 (0.0140) | OPEN |
| LATERAL:heading_mae_deg | +0.6883 / +1.4563 / +0.6957 | +0.9467 | [+0.6007, +1.4451] | 0.1789 (0.1056) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0922 / +0.1055 / +0.0839 | +0.0938 | [+0.0732, +0.1170] | 0.0247 (0.0154) | WORSE |
| LATERAL:yaw_rate_mae_degps | +8.5999 / +10.6197 / +7.3891 | +8.8696 | [+7.0800, +10.9218] | 0.8110 (0.4509) | WORSE |
| TACTICAL:lon_err_rate | +0.1088 / +0.1601 / +0.1000 | +0.1230 | [+0.0876, +0.1646] | 0.0158 (0.0113) | WORSE |
| TACTICAL:lat_err_rate | +0.0179 / +0.0367 / +0.0192 | +0.0246 | [+0.0126, +0.0388] | 0.0133 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.1021 / +0.1555 / +0.0975 | +0.1184 | [+0.0842, +0.1598] | 0.0183 (0.0125) | WORSE |
| TACTICAL:ade_tac_m | +0.4135 / +0.4168 / +0.2719 | +0.3674 | [+0.2673, +0.4659] | 0.0703 (0.0406) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1189 / +0.0873 / +0.0437 | +0.0833 | [+0.0186, +0.1527] | 0.0585 | WORSE |
| ade6_m|bulk | +0.4628 / +0.5052 / +0.3454 | +0.4378 | [+0.3418, +0.5354] | 0.0441 | WORSE |
| ade6_m|all | +0.2757 / +0.2778 / +0.1813 | +0.2449 | [+0.1782, +0.3106] | 0.0519 | WORSE |
| ade_tac_m|all | +0.4135 / +0.4168 / +0.2719 | +0.3674 | [+0.2673, +0.4659] | 0.0703 | WORSE |

## H RLcA minus U RLcK

Seeds read [0, 1, 2] of [0, 1, 2]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1893 / +0.1645 / +0.1135 | +0.1558 | [+0.1065, +0.2038] | 0.0424 (0.0250) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3765 / +0.2817 / +0.1871 | +0.2818 | [+0.1716, +0.3946] | 0.0348 (0.0275) | WORSE |
| LATERAL:cross_mae_m | -0.0206 / -0.0613 / -0.0523 | -0.0447 | [-0.0761, -0.0130] | 0.0276 (0.0140) | BETTER |
| LATERAL:heading_mae_deg | +0.3261 / +0.8911 / -0.0920 | +0.3751 | [-0.0654, +0.8736] | 0.1789 (0.1056) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0843 / +0.0850 / +0.0446 | +0.0713 | [+0.0438, +0.0977] | 0.0247 (0.0154) | WORSE |
| LATERAL:yaw_rate_mae_degps | +8.1512 / +9.3990 / +5.7479 | +7.7660 | [+5.7349, +9.7698] | 0.8110 (0.4509) | WORSE |
| TACTICAL:lon_err_rate | +0.1000 / +0.1326 / +0.0796 | +0.1041 | [+0.0710, +0.1405] | 0.0158 (0.0113) | WORSE |
| TACTICAL:lat_err_rate | +0.0129 / +0.0217 / -0.0079 | +0.0089 | [-0.0079, +0.0253] | 0.0133 (0.0117) | OPEN |
| TACTICAL:joint_err_rate | +0.0855 / +0.1313 / +0.0659 | +0.0942 | [+0.0581, +0.1366] | 0.0183 (0.0125) | WORSE |
| TACTICAL:ade_tac_m | +0.5005 / +0.3302 / +0.2090 | +0.3465 | [+0.1826, +0.5220] | 0.0703 (0.0406) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2918 / +0.1222 / +0.0698 | +0.1612 | [+0.0462, +0.2982] | 0.0585 | WORSE |
| ade6_m|bulk | +0.3565 / +0.3064 / +0.2023 | +0.2884 | [+0.1681, +0.4113] | 0.0441 | WORSE |
| ade6_m|all | +0.3213 / +0.2062 / +0.1302 | +0.2192 | [+0.1112, +0.3354] | 0.0519 | WORSE |
| ade_tac_m|all | +0.5005 / +0.3302 / +0.2090 | +0.3465 | [+0.1826, +0.5220] | 0.0703 | WORSE |

## H RQ50A minus U RQ50K

Seeds read [0, 1, 2] of [0, 1, 2]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1881 / +0.1777 / +0.1251 | +0.1636 | [+0.1167, +0.2075] | 0.0424 (0.0250) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3530 / +0.2810 / +0.1942 | +0.2761 | [+0.1735, +0.3743] | 0.0348 (0.0275) | WORSE |
| LATERAL:cross_mae_m | -0.0120 / -0.0607 / -0.0457 | -0.0395 | [-0.0725, -0.0050] | 0.0276 (0.0140) | BETTER |
| LATERAL:heading_mae_deg | +0.3117 / +0.8554 / -0.1558 | +0.3371 | [-0.1160, +0.8383] | 0.1789 (0.1056) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0748 / +0.0837 / +0.0366 | +0.0651 | [+0.0363, +0.0919] | 0.0247 (0.0154) | WORSE |
| LATERAL:yaw_rate_mae_degps | +7.8752 / +9.4579 / +5.5534 | +7.6288 | [+5.5238, +9.7826] | 0.8110 (0.4509) | WORSE |
| TACTICAL:lon_err_rate | +0.1059 / +0.1388 / +0.0905 | +0.1117 | [+0.0803, +0.1461] | 0.0158 (0.0113) | WORSE |
| TACTICAL:lat_err_rate | +0.0113 / +0.0213 / -0.0075 | +0.0083 | [-0.0071, +0.0246] | 0.0133 (0.0117) | OPEN |
| TACTICAL:joint_err_rate | +0.0892 / +0.1367 / +0.0775 | +0.1012 | [+0.0667, +0.1418] | 0.0183 (0.0125) | WORSE |
| TACTICAL:ade_tac_m | +0.4762 / +0.3307 / +0.2291 | +0.3453 | [+0.1963, +0.5028] | 0.0703 (0.0406) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2551 / +0.1105 / +0.0468 | +0.1375 | [+0.0231, +0.2668] | 0.0585 | WORSE |
| ade6_m|bulk | +0.3669 / +0.3223 / +0.2603 | +0.3165 | [+0.2096, +0.4288] | 0.0441 | WORSE |
| ade6_m|all | +0.3061 / +0.2071 / +0.1441 | +0.2191 | [+0.1190, +0.3229] | 0.0519 | WORSE |
| ade_tac_m|all | +0.4762 / +0.3307 / +0.2291 | +0.3453 | [+0.1963, +0.5028] | 0.0703 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| H_R0A_s0 | 2.5152 | 3.1818 | 1.1873 | 0.8321 | 0.5661 |
| H_R0A_s1 | 2.4154 | 3.0267 | 1.1762 | 0.8046 | 0.5911 |
| H_R0A_s2 | 2.2916 | 2.9383 | 1.0788 | 0.7904 | 0.5211 |
| H_R0K_s0 | 2.2395 | 3.0629 | 0.9774 | 0.8314 | 0.4639 |
| H_R0K_s1 | 2.1375 | 2.9394 | 0.9408 | 0.7982 | 0.4356 |
| H_R0K_s2 | 2.1103 | 2.8946 | 0.9219 | 0.7872 | 0.4235 |
| H_RLcA_s0 | 2.6168 | 3.1988 | 1.2113 | 0.8346 | 0.5540 |
| H_RLcA_s1 | 2.5310 | 3.0725 | 1.2039 | 0.8075 | 0.5815 |
| H_RLcA_s2 | 2.4031 | 2.9616 | 1.1105 | 0.7889 | 0.5219 |
| H_RQ50A_s0 | 2.6678 | 3.2698 | 1.2248 | 0.8417 | 0.5536 |
| H_RQ50A_s1 | 2.5364 | 3.0463 | 1.2125 | 0.8067 | 0.5827 |
| H_RQ50A_s2 | 2.4766 | 3.0422 | 1.1353 | 0.7935 | 0.5177 |
| U_RLcK_s0 | 2.2955 | 2.9071 | 1.0220 | 0.8551 | 0.4685 |
| U_RLcK_s1 | 2.3249 | 2.9503 | 1.0394 | 0.8688 | 0.4502 |
| U_RLcK_s2 | 2.2729 | 2.8918 | 0.9970 | 0.8411 | 0.4560 |
| U_RQ50K_s0 | 2.3617 | 3.0147 | 1.0367 | 0.8537 | 0.4644 |
| U_RQ50K_s1 | 2.3293 | 2.9357 | 1.0349 | 0.8674 | 0.4460 |
| U_RQ50K_s2 | 2.3325 | 2.9954 | 1.0102 | 0.8392 | 0.4402 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
