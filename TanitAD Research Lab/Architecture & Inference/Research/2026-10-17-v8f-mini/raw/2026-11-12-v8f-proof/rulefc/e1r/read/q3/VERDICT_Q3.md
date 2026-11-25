# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 6}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "H_RQ50KE1"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15711934041139608, "D_read": 0.15711934041139608, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm H_RQ50K, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## H RQ50KE1 minus H RQ50K

Seeds read [0, 1, 2] of [0, 1, 2]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0073 / +0.0138 / +0.0069 | +0.0094 | [+0.0030, +0.0171] | 0.0524 (0.0393) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0363 / +0.0440 / +0.0375 | +0.0393 | [+0.0209, +0.0599] | 0.1037 (0.0967) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0009 / -0.0003 / -0.0019 | -0.0010 | [-0.0023, +0.0001] | 0.0486 (0.0300) | OPEN |
| LATERAL:heading_mae_deg | -0.0219 / -0.0053 / -0.0130 | -0.0134 | [-0.0275, -0.0040] | 0.3583 (0.2524) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0008 / -0.0000 / -0.0002 | -0.0003 | [-0.0019, +0.0008] | 0.0069 (0.0066) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.0542 / -0.0168 / -0.0305 | -0.0339 | [-0.0874, -0.0007] | 0.1387 (0.1039) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0013 / +0.0054 / +0.0025 | +0.0031 | [-0.0013, +0.0076] | 0.0333 (0.0238) | OPEN |
| TACTICAL:lat_err_rate | -0.0004 / -0.0013 / +0.0004 | -0.0004 | [-0.0018, +0.0008] | 0.0138 (0.0100) | OPEN |
| TACTICAL:joint_err_rate | +0.0021 / +0.0050 / +0.0029 | +0.0033 | [-0.0008, +0.0076] | 0.0358 (0.0192) | OPEN |
| TACTICAL:ade_tac_m | +0.0407 / +0.0559 / +0.0402 | +0.0456 | [+0.0211, +0.0736] | 0.1895 (0.1883) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0134 / +0.0049 / +0.0136 | +0.0106 | [-0.0107, +0.0330] | 0.2085 | OPEN |
| ade6_m|bulk | +0.0609 / +0.0898 / +0.0603 | +0.0704 | [+0.0451, +0.1051] | 0.0810 | INSIDE-SPREAD |
| ade6_m|all | +0.0351 / +0.0436 / +0.0349 | +0.0379 | [+0.0199, +0.0583] | 0.1332 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0407 / +0.0559 / +0.0402 | +0.0456 | [+0.0211, +0.0736] | 0.1895 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| H_RQ50K_s0 | 2.4191 | 3.1521 | 1.0136 | 0.8273 | 0.4510 |
| H_RQ50K_s1 | 2.2859 | 2.9436 | 0.9743 | 0.7973 | 0.4318 |
| H_RQ50K_s2 | 2.2941 | 2.9901 | 0.9612 | 0.7787 | 0.4152 |
| H_RQ50KE1_s0 | 2.4542 | 3.1655 | 1.0209 | 0.8264 | 0.4531 |
| H_RQ50KE1_s1 | 2.3296 | 2.9485 | 0.9881 | 0.7970 | 0.4368 |
| H_RQ50KE1_s2 | 2.3290 | 3.0037 | 0.9681 | 0.7768 | 0.4181 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
