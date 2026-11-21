# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 24}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "HIERRL"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8990895386859552, "D_read": 0.8990895386859553, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## HIERRL minus HIER

Seeds read [0, 1, 2, 3, 4, 5] of [0, 1, 2, 3, 4, 5]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0326 / +0.0058 / -0.0002 / +0.0355 / +0.0296 / +0.0286 | +0.0220 | [-0.0199, +0.0622] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2295 / +0.2004 / +0.2110 / +0.2217 / +0.2190 / +0.2260 | +0.2179 | [+0.1301, +0.3195] | 0.4550 (0.1815) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0136 / -0.0373 / -0.0658 / -0.0095 / -0.0569 / -0.0480 | -0.0385 | [-0.0609, -0.0163] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2065 / +0.3094 / +0.1443 / +0.1967 / +0.2577 / -0.0478 | +0.1778 | [+0.0331, +0.3078] | 2.5508 (1.1975) | MIXED-SIGN |
| LATERAL:curvature_mae_1pm | +0.0237 / +0.0171 / +0.0102 / +0.0168 / +0.0161 / +0.0207 | +0.0174 | [+0.0118, +0.0245] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.5948 / +0.1718 / -0.4671 / +0.6774 / +0.1725 / +0.7800 | +0.3216 | [-0.2967, +0.8894] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | -0.0021 / -0.0104 / -0.0058 / -0.0029 / -0.0054 / -0.0063 | -0.0055 | [-0.0106, -0.0004] | 0.0746 (0.0329) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0063 / +0.0013 / +0.0071 / +0.0104 / -0.0017 / +0.0033 | +0.0044 | [-0.0029, +0.0125] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0013 / -0.0083 / -0.0013 / +0.0021 / -0.0017 / -0.0013 | -0.0015 | [-0.0063, +0.0033] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | +0.2762 / +0.2167 / +0.1990 / +0.2742 / +0.2233 / +0.2353 | +0.2374 | [+0.1224, +0.3761] | 0.4169 (0.1897) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1262 / +0.0534 / +0.0573 / +0.1112 / +0.0782 / +0.0898 | +0.0860 | [+0.0065, +0.1768] | 0.4093 | INSIDE-SPREAD |
| ade6_m|bulk | +0.2988 / +0.2966 / +0.2685 / +0.3136 / +0.2773 / +0.2818 | +0.2894 | [+0.1839, +0.4095] | 0.3850 | INSIDE-SPREAD |
| ade6_m|all | +0.2049 / +0.1643 / +0.1536 / +0.2035 / +0.1690 / +0.1774 | +0.1788 | [+0.0952, +0.2801] | 0.2845 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2762 / +0.2167 / +0.1990 / +0.2742 / +0.2233 / +0.2353 | +0.2374 | [+0.1224, +0.3761] | 0.4169 | INSIDE-SPREAD |

## GRL minus G

Seeds read [0, 1, 2, 3, 4, 5] of [0, 1, 2, 3, 4, 5]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0372 / -0.0096 / +0.0218 / +0.0230 / -0.0014 / +0.0289 | +0.0166 | [-0.0241, +0.0609] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2636 / +0.1890 / +0.2013 / +0.2004 / +0.2094 / +0.2200 | +0.2139 | [+0.1277, +0.3167] | 0.4550 (0.1815) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0376 / -0.0276 / -0.0434 / -0.0067 / -0.0609 / -0.0172 | -0.0322 | [-0.0508, -0.0127] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0211 / +0.4552 / +0.1861 / +0.3601 / +0.3501 / +0.1712 | +0.2503 | [+0.0818, +0.4017] | 2.5508 (1.1975) | MIXED-SIGN |
| LATERAL:curvature_mae_1pm | +0.0130 / +0.0178 / +0.0140 / +0.0200 / +0.0174 / +0.0113 | +0.0156 | [+0.0101, +0.0209] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.5882 / +0.9469 / -0.2987 / +1.0773 / +0.7895 / -0.3344 | +0.2654 | [-0.4625, +0.9604] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0013 / -0.0063 / +0.0017 / -0.0063 / -0.0033 / -0.0088 | -0.0036 | [-0.0088, +0.0024] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0025 / +0.0088 / +0.0063 / +0.0150 / +0.0017 / +0.0050 | +0.0057 | [-0.0021, +0.0145] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | -0.0004 / -0.0025 / +0.0017 / +0.0021 / +0.0017 / -0.0033 | -0.0001 | [-0.0044, +0.0045] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | +0.3071 / +0.2102 / +0.2146 / +0.2403 / +0.2021 / +0.2606 | +0.2392 | [+0.1255, +0.3796] | 0.4169 (0.1897) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1299 / +0.0715 / +0.0848 / +0.0956 / +0.0521 / +0.0953 | +0.0882 | [+0.0069, +0.1832] | 0.4093 | INSIDE-SPREAD |
| ade6_m|bulk | +0.3401 / +0.2664 / +0.2579 / +0.2816 / +0.2785 / +0.3124 | +0.2895 | [+0.1883, +0.4142] | 0.3850 | INSIDE-SPREAD |
| ade6_m|all | +0.2258 / +0.1604 / +0.1637 / +0.1804 / +0.1554 / +0.1943 | +0.1800 | [+0.0978, +0.2823] | 0.2845 | INSIDE-SPREAD |
| ade_tac_m|all | +0.3071 / +0.2102 / +0.2146 / +0.2403 / +0.2021 / +0.2606 | +0.2392 | [+0.1255, +0.3796] | 0.4169 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| GRL_s0 | 3.3178 | 3.8102 | 1.9490 | 1.1895 | 0.7382 |
| GRL_s1 | 2.9990 | 3.3116 | 1.7221 | 1.2941 | 0.7015 |
| GRL_s2 | 3.3666 | 3.7766 | 1.9893 | 1.3865 | 0.7382 |
| GRL_s3 | 2.9831 | 3.3417 | 1.7967 | 1.1427 | 0.7370 |
| GRL_s4 | 3.0937 | 3.3981 | 1.7213 | 1.4552 | 0.6911 |
| GRL_s5 | 3.0007 | 3.3544 | 1.7270 | 1.1794 | 0.7270 |
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| HIER_s3 | 2.9201 | 3.2939 | 1.8914 | 1.1934 | 0.7724 |
| HIER_s4 | 2.9086 | 3.3309 | 1.7001 | 1.4618 | 0.7524 |
| HIER_s5 | 3.0344 | 3.4157 | 1.8393 | 1.4161 | 0.7624 |
| HIERRL_s0 | 3.3980 | 3.8294 | 2.0325 | 1.2618 | 0.6970 |
| HIERRL_s1 | 3.1413 | 3.4787 | 1.7808 | 1.4504 | 0.7228 |
| HIERRL_s2 | 3.3143 | 3.5290 | 1.9855 | 1.3948 | 0.8149 |
| HIERRL_s3 | 3.1235 | 3.4051 | 1.9269 | 1.1839 | 0.7745 |
| HIERRL_s4 | 3.0776 | 3.4091 | 1.7297 | 1.4049 | 0.7507 |
| HIERRL_s5 | 3.2118 | 3.5056 | 1.8680 | 1.3680 | 0.7612 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
