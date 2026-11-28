# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 6}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "H_RQ50KV1"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.1585212049604997, "D_read": 0.15852120496049965, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm H_RQ50KE1, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## H RQ50KV1 minus H RQ50KE1

Seeds read [0, 1, 2] of [0, 1, 2]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0136 / +0.0335 / +0.0134 | +0.0202 | [+0.0060, +0.0416] | 0.0528 (0.0328) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0305 / +0.0761 / +0.0330 | +0.0466 | [+0.0111, +0.0998] | 0.0960 (0.0956) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0004 / -0.0002 / -0.0009 | -0.0002 | [-0.0017, +0.0012] | 0.0496 (0.0293) | OPEN |
| LATERAL:heading_mae_deg | -0.0011 / -0.0036 / +0.0040 | -0.0002 | [-0.0075, +0.0088] | 0.3494 (0.2358) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0000 / -0.0002 / -0.0005 | -0.0002 | [-0.0009, +0.0004] | 0.0087 (0.0085) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0192 / -0.0099 / -0.0126 | -0.0011 | [-0.0419, +0.0379] | 0.1843 (0.1341) | OPEN |
| TACTICAL:lon_err_rate | +0.0079 / +0.0092 / -0.0008 | +0.0054 | [-0.0024, +0.0142] | 0.0321 (0.0196) | OPEN |
| TACTICAL:lat_err_rate | +0.0013 / +0.0008 / +0.0000 | +0.0007 | [-0.0011, +0.0025] | 0.0129 (0.0108) | OPEN |
| TACTICAL:joint_err_rate | +0.0071 / +0.0063 / +0.0008 | +0.0047 | [-0.0019, +0.0125] | 0.0350 (0.0188) | OPEN |
| TACTICAL:ade_tac_m | +0.0408 / +0.1074 / +0.0439 | +0.0641 | [+0.0150, +0.1384] | 0.1888 (0.1743) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0097 / -0.0004 / -0.0180 | -0.0094 | [-0.0279, +0.0080] | 0.2170 | OPEN |
| ade6_m|bulk | +0.0756 / +0.1651 / +0.0899 | +0.1102 | [+0.0421, +0.2134] | 0.0816 | WORSE |
| ade6_m|all | +0.0292 / +0.0751 / +0.0312 | +0.0451 | [+0.0108, +0.0969] | 0.1252 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0408 / +0.1074 / +0.0439 | +0.0641 | [+0.0150, +0.1384] | 0.1888 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| H_RQ50KE1_s0 | 2.4542 | 3.1655 | 1.0209 | 0.8264 | 0.4531 |
| H_RQ50KE1_s1 | 2.3296 | 2.9485 | 0.9881 | 0.7970 | 0.4368 |
| H_RQ50KE1_s2 | 2.3290 | 3.0037 | 0.9681 | 0.7768 | 0.4181 |
| H_RQ50KV1_s0 | 2.4834 | 3.1557 | 1.0346 | 0.8268 | 0.4602 |
| H_RQ50KV1_s1 | 2.4046 | 2.9481 | 1.0216 | 0.7969 | 0.4431 |
| H_RQ50KV1_s2 | 2.3602 | 2.9857 | 0.9815 | 0.7758 | 0.4189 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
