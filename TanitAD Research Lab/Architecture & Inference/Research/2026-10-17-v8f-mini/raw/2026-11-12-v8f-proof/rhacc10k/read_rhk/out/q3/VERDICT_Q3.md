# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "H_RLcK"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.1272268626789459, "D_read": 0.1272268626789459, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm U_RLcK, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## H RLcK minus U RLcK

Seeds read [0, 1, 2] of [0, 1, 2]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0212 / -0.0668 / -0.0497 | -0.0459 | [-0.0746, -0.0175] | 0.0424 (0.0250) | BETTER |
| LONGITUDINAL:along_mae_m | +0.0880 / +0.0063 / +0.0010 | +0.0318 | [-0.0336, +0.0977] | 0.0348 (0.0275) | OPEN |
| LATERAL:cross_mae_m | -0.0274 / -0.0723 / -0.0626 | -0.0541 | [-0.0852, -0.0228] | 0.0276 (0.0140) | BETTER |
| LATERAL:heading_mae_deg | -0.4013 / -0.5862 / -0.8442 | -0.6105 | [-0.8532, -0.3960] | 0.1789 (0.1056) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0085 / -0.0183 / -0.0417 | -0.0228 | [-0.0415, -0.0082] | 0.0247 (0.0154) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.8068 / -1.3703 / -2.0717 | -1.4163 | [-2.1421, -0.7859] | 0.8110 (0.4509) | BETTER |
| TACTICAL:lon_err_rate | -0.0025 / -0.0158 / -0.0204 | -0.0129 | [-0.0279, +0.0018] | 0.0158 (0.0113) | OPEN |
| TACTICAL:lat_err_rate | -0.0067 / -0.0179 / -0.0313 | -0.0186 | [-0.0330, -0.0042] | 0.0133 (0.0117) | BETTER |
| TACTICAL:joint_err_rate | -0.0117 / -0.0154 / -0.0329 | -0.0200 | [-0.0367, -0.0050] | 0.0183 (0.0125) | BETTER |
| TACTICAL:ade_tac_m | +0.1100 / -0.0554 / -0.0446 | +0.0033 | [-0.1005, +0.1165] | 0.0703 (0.0406) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1391 / +0.0001 / -0.0080 | +0.0437 | [-0.0422, +0.1479] | 0.0585 | OPEN |
| ade6_m|bulk | -0.0322 / -0.1117 / -0.0757 | -0.0732 | [-0.1661, +0.0048] | 0.0441 | OPEN |
| ade6_m|all | +0.0609 / -0.0509 / -0.0389 | -0.0096 | [-0.0807, +0.0675] | 0.0519 | OPEN |
| ade_tac_m|all | +0.1100 / -0.0554 / -0.0446 | +0.0033 | [-0.1005, +0.1165] | 0.0703 | OPEN |

## H RQ50K minus U RQ50K

Seeds read [0, 1, 2] of [0, 1, 2]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0231 / -0.0606 / -0.0490 | -0.0442 | [-0.0711, -0.0182] | 0.0424 (0.0250) | BETTER |
| LONGITUDINAL:along_mae_m | +0.0821 / +0.0136 / -0.0002 | +0.0318 | [-0.0328, +0.0936] | 0.0348 (0.0275) | OPEN |
| LATERAL:cross_mae_m | -0.0264 / -0.0701 / -0.0605 | -0.0523 | [-0.0831, -0.0215] | 0.0276 (0.0140) | BETTER |
| LATERAL:heading_mae_deg | -0.3996 / -0.5829 / -0.8405 | -0.6076 | [-0.8444, -0.3895] | 0.1789 (0.1056) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0117 / -0.0185 / -0.0392 | -0.0231 | [-0.0396, -0.0100] | 0.0247 (0.0154) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.9640 / -1.4105 / -2.0442 | -1.4729 | [-2.1292, -0.9038] | 0.8110 (0.4509) | BETTER |
| TACTICAL:lon_err_rate | -0.0013 / -0.0158 / -0.0146 | -0.0106 | [-0.0249, +0.0033] | 0.0158 (0.0113) | OPEN |
| TACTICAL:lat_err_rate | -0.0113 / -0.0171 / -0.0308 | -0.0197 | [-0.0326, -0.0076] | 0.0133 (0.0117) | BETTER |
| TACTICAL:joint_err_rate | -0.0133 / -0.0142 / -0.0250 | -0.0175 | [-0.0310, -0.0051] | 0.0183 (0.0125) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1033 / -0.0450 / -0.0448 | +0.0045 | [-0.0973, +0.1117] | 0.0703 (0.0406) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1374 / +0.0079 / -0.0053 | +0.0466 | [-0.0365, +0.1472] | 0.0585 | OPEN |
| ade6_m|bulk | -0.0378 / -0.1045 / -0.0779 | -0.0734 | [-0.1657, +0.0036] | 0.0441 | OPEN |
| ade6_m|all | +0.0575 / -0.0434 / -0.0384 | -0.0081 | [-0.0791, +0.0655] | 0.0519 | OPEN |
| ade_tac_m|all | +0.1033 / -0.0450 / -0.0448 | +0.0045 | [-0.0973, +0.1117] | 0.0703 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| H_RLcK_s0 | 2.3564 | 3.0461 | 1.0007 | 0.8277 | 0.4569 |
| H_RLcK_s1 | 2.2740 | 2.9504 | 0.9726 | 0.7965 | 0.4348 |
| H_RLcK_s2 | 2.2341 | 2.8838 | 0.9473 | 0.7785 | 0.4231 |
| H_RQ50K_s0 | 2.4191 | 3.1521 | 1.0136 | 0.8273 | 0.4510 |
| H_RQ50K_s1 | 2.2859 | 2.9436 | 0.9743 | 0.7973 | 0.4318 |
| H_RQ50K_s2 | 2.2941 | 2.9901 | 0.9612 | 0.7787 | 0.4152 |
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
