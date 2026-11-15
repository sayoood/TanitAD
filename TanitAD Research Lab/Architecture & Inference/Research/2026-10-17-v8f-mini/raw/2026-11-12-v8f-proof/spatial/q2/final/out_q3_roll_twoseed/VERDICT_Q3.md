# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 8}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "ROLL"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045083, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## ROLL minus G

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0670 / -0.0443 | -0.0556 | [-0.1247, +0.0116] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0731 / -0.0523 | -0.0627 | [-0.1444, +0.0140] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0009 / +0.0460 | +0.0225 | [-0.0333, +0.0783] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -1.2900 / +1.2375 | -0.0262 | [-1.6202, +1.4309] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0072 / +0.0270 | +0.0099 | [-0.0167, +0.0323] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -3.3797 / +4.5654 | +0.5928 | [-4.5631, +5.4094] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0321 / -0.0167 | -0.0244 | [-0.0506, +0.0013] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0742 / +0.0238 | -0.0252 | [-0.0902, +0.0394] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0484 / +0.0000 | -0.0242 | [-0.0650, +0.0140] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0781 / -0.0399 | -0.0590 | [-0.1807, +0.0503] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1393 / +0.0647 | -0.0373 | [-0.2044, +0.1200] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0516 / -0.1437 | -0.0461 | [-0.1997, +0.1467] | 0.3535 | OPEN |
| ade6_m|all | -0.0523 / -0.0303 | -0.0413 | [-0.1229, +0.0325] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0781 / -0.0399 | -0.0590 | [-0.1807, +0.0503] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| ROLL_s0 | 3.0397 | 3.5409 | 1.8449 | 1.2262 | 0.6903 |
| ROLL_s1 | 2.8082 | 3.3047 | 1.6874 | 1.3677 | 0.7040 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
