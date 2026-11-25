# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "H_RLcE1"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.16015711299059687, "D_read": 0.16015711299059687, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm H_RLcK, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## H RLcE1 minus H RLcK

Seeds read [0, 1, 2] of [0, 1, 2]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0074 / +0.0074 / +0.0079 | +0.0076 | [+0.0029, +0.0123] | 0.0534 (0.0281) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0248 / +0.0230 / +0.0241 | +0.0240 | [+0.0127, +0.0362] | 0.0944 (0.0543) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0014 / +0.0017 / +0.0014 | +0.0015 | [+0.0006, +0.0027] | 0.0492 (0.0312) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.0012 / +0.0035 / +0.0037 | +0.0028 | [-0.0006, +0.0064] | 0.3536 (0.2640) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0002 / -0.0001 / -0.0001 | -0.0000 | [-0.0004, +0.0004] | 0.0070 (0.0045) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0037 / -0.0084 / -0.0085 | -0.0044 | [-0.0258, +0.0147] | 0.2763 (0.1738) | OPEN |
| TACTICAL:lon_err_rate | +0.0021 / -0.0004 / +0.0004 | +0.0007 | [-0.0019, +0.0032] | 0.0338 (0.0246) | OPEN |
| TACTICAL:lat_err_rate | +0.0000 / +0.0004 / +0.0000 | +0.0001 | [-0.0007, +0.0010] | 0.0129 (0.0129) | OPEN |
| TACTICAL:joint_err_rate | +0.0025 / +0.0017 / +0.0004 | +0.0015 | [-0.0008, +0.0042] | 0.0338 (0.0221) | OPEN |
| TACTICAL:ade_tac_m | +0.0333 / +0.0307 / +0.0322 | +0.0321 | [+0.0165, +0.0487] | 0.1844 (0.1249) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0172 / +0.0149 / +0.0147 | +0.0156 | [+0.0063, +0.0263] | 0.1623 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0338 / +0.0328 / +0.0352 | +0.0339 | [+0.0148, +0.0562] | 0.0748 | INSIDE-SPREAD |
| ade6_m|all | +0.0248 / +0.0231 / +0.0241 | +0.0240 | [+0.0130, +0.0357] | 0.1224 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0333 / +0.0307 / +0.0322 | +0.0321 | [+0.0165, +0.0487] | 0.1844 | INSIDE-SPREAD |

## H RLcFC minus H RLcK

Seeds read [0, 1, 2] of [0, 1, 2]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0106 / +0.0118 / +0.0123 | +0.0116 | [+0.0072, +0.0166] | 0.0534 (0.0281) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0302 / +0.0278 / +0.0305 | +0.0295 | [+0.0184, +0.0428] | 0.0944 (0.0543) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0018 / +0.0026 / +0.0020 | +0.0021 | [+0.0005, +0.0043] | 0.0492 (0.0312) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0032 / -0.0086 / -0.0052 | -0.0057 | [-0.0132, +0.0000] | 0.3536 (0.2640) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0001 / -0.0014 / -0.0005 | -0.0007 | [-0.0022, +0.0005] | 0.0070 (0.0045) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0156 / -0.0356 / -0.0039 | -0.0080 | [-0.0588, +0.0338] | 0.2763 (0.1738) | OPEN |
| TACTICAL:lon_err_rate | +0.0013 / +0.0025 / +0.0029 | +0.0022 | [-0.0001, +0.0047] | 0.0338 (0.0246) | OPEN |
| TACTICAL:lat_err_rate | +0.0000 / -0.0021 / -0.0017 | -0.0013 | [-0.0026, +0.0000] | 0.0129 (0.0129) | OPEN |
| TACTICAL:joint_err_rate | +0.0021 / +0.0013 / +0.0017 | +0.0017 | [-0.0006, +0.0039] | 0.0338 (0.0221) | OPEN |
| TACTICAL:ade_tac_m | +0.0419 / +0.0399 / +0.0424 | +0.0414 | [+0.0257, +0.0599] | 0.1844 (0.1249) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0271 / +0.0256 / +0.0271 | +0.0266 | [+0.0137, +0.0407] | 0.1623 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0306 / +0.0296 / +0.0314 | +0.0305 | [+0.0177, +0.0453] | 0.0748 | INSIDE-SPREAD |
| ade6_m|all | +0.0287 / +0.0274 / +0.0290 | +0.0284 | [+0.0177, +0.0410] | 0.1224 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0419 / +0.0399 / +0.0424 | +0.0414 | [+0.0257, +0.0599] | 0.1844 | INSIDE-SPREAD |

## H RLcE1FC minus H RLcK

Seeds read [0, 1, 2] of [0, 1, 2]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0226 / +0.0245 / +0.0246 | +0.0239 | [+0.0147, +0.0340] | 0.0534 (0.0281) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0655 / +0.0658 / +0.0666 | +0.0660 | [+0.0424, +0.0928] | 0.0944 (0.0543) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0045 / +0.0051 / +0.0048 | +0.0048 | [+0.0017, +0.0084] | 0.0492 (0.0312) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0018 / +0.0013 / +0.0019 | +0.0005 | [-0.0074, +0.0091] | 0.3536 (0.2640) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0000 / -0.0006 / -0.0000 | -0.0002 | [-0.0015, +0.0008] | 0.0070 (0.0045) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0154 / +0.0074 / +0.0110 | +0.0113 | [-0.0374, +0.0567] | 0.2763 (0.1738) | OPEN |
| TACTICAL:lon_err_rate | +0.0050 / +0.0017 / +0.0058 | +0.0042 | [+0.0007, +0.0078] | 0.0338 (0.0246) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0004 / +0.0000 / -0.0008 | -0.0001 | [-0.0014, +0.0012] | 0.0129 (0.0129) | OPEN |
| TACTICAL:joint_err_rate | +0.0063 / +0.0025 / +0.0058 | +0.0049 | [+0.0014, +0.0085] | 0.0338 (0.0221) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0903 / +0.0912 / +0.0916 | +0.0910 | [+0.0584, +0.1281] | 0.1844 (0.1249) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0471 / +0.0498 / +0.0495 | +0.0488 | [+0.0276, +0.0722] | 0.1623 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0850 / +0.0830 / +0.0840 | +0.0840 | [+0.0445, +0.1331] | 0.0748 | WORSE |
| ade6_m|all | +0.0644 / +0.0649 / +0.0652 | +0.0648 | [+0.0420, +0.0911] | 0.1224 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0903 / +0.0912 / +0.0916 | +0.0910 | [+0.0584, +0.1281] | 0.1844 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| H_RLcE1_s0 | 2.3812 | 3.0633 | 1.0081 | 0.8291 | 0.4594 |
| H_RLcE1_s1 | 2.2970 | 2.9653 | 0.9801 | 0.7982 | 0.4364 |
| H_RLcE1_s2 | 2.2581 | 2.8985 | 0.9552 | 0.7800 | 0.4235 |
| H_RLcE1FC_s0 | 2.4208 | 3.0932 | 1.0233 | 0.8322 | 0.4631 |
| H_RLcE1FC_s1 | 2.3389 | 3.0001 | 0.9972 | 0.8015 | 0.4373 |
| H_RLcE1FC_s2 | 2.2993 | 2.9333 | 0.9719 | 0.7833 | 0.4289 |
| H_RLcFC_s0 | 2.3852 | 3.0732 | 1.0113 | 0.8295 | 0.4589 |
| H_RLcFC_s1 | 2.3014 | 2.9759 | 0.9844 | 0.7991 | 0.4360 |
| H_RLcFC_s2 | 2.2631 | 2.9109 | 0.9596 | 0.7805 | 0.4248 |
| H_RLcK_s0 | 2.3564 | 3.0461 | 1.0007 | 0.8277 | 0.4569 |
| H_RLcK_s1 | 2.2740 | 2.9504 | 0.9726 | 0.7965 | 0.4348 |
| H_RLcK_s2 | 2.2341 | 2.8838 | 0.9473 | 0.7785 | 0.4231 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
