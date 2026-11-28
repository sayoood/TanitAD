# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "PF"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8990895386859552, "D_read": 0.8990895386859552, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## PF minus P

Seeds read [0] of [0]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1342 | -0.1342 | [-0.2432, -0.0335] | 0.2997 (0.1392) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0023 | -0.0023 | [-0.1128, +0.1017] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.0055 | -0.0055 | [-0.0705, +0.0653] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | +1.0914 | +1.0914 | [+0.5344, +1.6864] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0626 | +0.0626 | [+0.0430, +0.0839] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +7.0505 | +7.0505 | [+4.0388, +10.1873] | 10.6485 (4.6266) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0459 | -0.0459 | [-0.0783, -0.0166] | 0.0746 (0.0329) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0484 | +0.0484 | [+0.0154, +0.0812] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0246 | -0.0246 | [-0.0555, +0.0033] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.0227 | -0.0227 | [-0.1766, +0.1295] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0178 | -0.0178 | [-0.1370, +0.1018] | 0.4093 | OPEN |
| ade6_m|bulk | -0.0133 | -0.0133 | [-0.1724, +0.1376] | 0.3850 | OPEN |
| ade6_m|all | -0.0158 | -0.0158 | [-0.1176, +0.0866] | 0.2845 | OPEN |
| ade_tac_m|all | -0.0227 | -0.0227 | [-0.1766, +0.1295] | 0.4169 | OPEN |

## PF minus HIER

Seeds read [0] of [0]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2207 | -0.2207 | [-0.3477, -0.1004] | 0.2997 (0.1392) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.1677 | -0.1677 | [-0.2952, -0.0480] | 0.4550 (0.1815) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0912 | +0.0912 | [+0.0241, +0.1653] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.3700 | +1.3700 | [+0.6910, +2.0976] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0346 | +0.0346 | [+0.0097, +0.0621] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +3.8799 | +3.8799 | [+0.3338, +7.3359] | 10.6485 (4.6266) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0204 | -0.0204 | [-0.0525, +0.0129] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | +0.1109 | +0.1109 | [+0.0742, +0.1468] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0242 | +0.0242 | [-0.0096, +0.0555] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.1180 | -0.1180 | [-0.3109, +0.0676] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1224 | -0.1224 | [-0.2910, +0.0404] | 0.4093 | OPEN |
| ade6_m|bulk | -0.0392 | -0.0392 | [-0.2221, +0.1260] | 0.3850 | OPEN |
| ade6_m|all | -0.0845 | -0.0845 | [-0.2134, +0.0411] | 0.2845 | OPEN |
| ade_tac_m|all | -0.1180 | -0.1180 | [-0.3109, +0.0676] | 0.4169 | OPEN |

## PF minus PR

Seeds read [0] of [0]. **Q3: NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3205 | -0.3205 | [-0.5074, -0.1625] | 0.2997 (0.1392) | BETTER |
| LONGITUDINAL:along_mae_m | -0.2515 | -0.2515 | [-0.4744, -0.0659] | 0.4550 (0.1815) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.1665 | +0.1665 | [+0.1095, +0.2281] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.3699 | -0.3699 | [-1.2086, +0.4256] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0074 | -0.0074 | [-0.0389, +0.0193] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.2119 | -0.2119 | [-3.4269, +2.5333] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | -0.0825 | -0.0825 | [-0.1189, -0.0500] | 0.0746 (0.0329) | BETTER |
| TACTICAL:lat_err_rate | +0.0709 | +0.0709 | [+0.0371, +0.1046] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0513 | -0.0513 | [-0.0889, -0.0188] | 0.1205 (0.0438) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1829 | -0.1829 | [-0.4724, +0.0747] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0460 | -0.0460 | [-0.2331, +0.1242] | 0.4093 | OPEN |
| ade6_m|bulk | -0.2195 | -0.2195 | [-0.5496, +0.0787] | 0.3850 | OPEN |
| ade6_m|all | -0.1251 | -0.1251 | [-0.3212, +0.0492] | 0.2845 | OPEN |
| ade_tac_m|all | -0.1829 | -0.1829 | [-0.4724, +0.0747] | 0.4169 | OPEN |

## RES minus G

Seeds read [0] of [0]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0246 | +0.0246 | [-0.0788, +0.1300] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0220 | -0.0220 | [-0.1336, +0.0904] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.0183 | -0.0183 | [-0.0809, +0.0459] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | -1.4610 | -1.4610 | [-2.1774, -0.8248] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0450 | -0.0450 | [-0.0719, -0.0201] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -3.1503 | -3.1503 | [-6.1465, -0.3944] | 10.6485 (4.6266) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0038 | +0.0038 | [-0.0288, +0.0371] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0275 | -0.0275 | [-0.0592, +0.0029] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | -0.0121 | -0.0121 | [-0.0417, +0.0188] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.0715 | -0.0715 | [-0.2279, +0.0862] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0728 | -0.0728 | [-0.2029, +0.0528] | 0.4093 | OPEN |
| ade6_m|bulk | -0.0061 | -0.0061 | [-0.1496, +0.1489] | 0.3850 | OPEN |
| ade6_m|all | -0.0424 | -0.0424 | [-0.1464, +0.0627] | 0.2845 | OPEN |
| ade_tac_m|all | -0.0715 | -0.0715 | [-0.2279, +0.0862] | 0.4169 | OPEN |

## RES minus HIER

Seeds read [0] of [0]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0635 | -0.0635 | [-0.1739, +0.0470] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1091 | -0.1091 | [-0.2407, +0.0165] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.0665 | -0.0665 | [-0.1356, +0.0019] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | +0.4589 | +0.4589 | [-0.2502, +1.1513] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0077 | -0.0077 | [-0.0284, +0.0123] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.6909 | +1.6909 | [-1.1934, +4.3717] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0204 | +0.0204 | [-0.0100, +0.0509] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | +0.0371 | +0.0371 | [-0.0017, +0.0742] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0308 | +0.0308 | [-0.0004, +0.0608] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.2185 | -0.2185 | [-0.4145, -0.0304] | 0.4169 (0.1897) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0958 | -0.0958 | [-0.2512, +0.0576] | 0.4093 | OPEN |
| ade6_m|bulk | -0.2004 | -0.2004 | [-0.4145, +0.0082] | 0.3850 | OPEN |
| ade6_m|all | -0.1435 | -0.1435 | [-0.2764, -0.0157] | 0.2845 | INSIDE-SPREAD |
| ade_tac_m|all | -0.2185 | -0.2185 | [-0.4145, -0.0304] | 0.4169 | INSIDE-SPREAD |

## RR minus G

Seeds read [0] of [0]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0479 | -0.0479 | [-0.1407, +0.0475] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0979 | -0.0979 | [-0.2095, +0.0223] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.0254 | -0.0254 | [-0.1022, +0.0567] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | -1.9430 | -1.9430 | [-2.7195, -1.2011] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0486 | -0.0486 | [-0.0732, -0.0263] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -6.0283 | -6.0283 | [-8.7629, -3.5109] | 10.6485 (4.6266) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0133 | -0.0133 | [-0.0483, +0.0213] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0521 | -0.0521 | [-0.0864, -0.0171] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0313 | -0.0313 | [-0.0621, -0.0004] | 0.1205 (0.0438) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1612 | -0.1612 | [-0.3151, +0.0264] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1697 | -0.1697 | [-0.2887, -0.0435] | 0.4093 | INSIDE-SPREAD |
| ade6_m|bulk | -0.0303 | -0.0303 | [-0.1897, +0.1834] | 0.3850 | OPEN |
| ade6_m|all | -0.1061 | -0.1061 | [-0.2112, +0.0216] | 0.2845 | OPEN |
| ade_tac_m|all | -0.1612 | -0.1612 | [-0.3151, +0.0264] | 0.4169 | OPEN |

## RR minus HIER

Seeds read [0] of [0]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1360 | -0.1360 | [-0.2605, -0.0197] | 0.2997 (0.1392) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.1849 | -0.1849 | [-0.2986, -0.0759] | 0.4550 (0.1815) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0736 | -0.0736 | [-0.1498, +0.0044] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | -0.0231 | -0.0231 | [-0.5034, +0.4793] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0112 | -0.0112 | [-0.0318, +0.0091] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.1843 | -1.1843 | [-3.3412, +1.1348] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0033 | +0.0033 | [-0.0287, +0.0346] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | +0.0125 | +0.0125 | [-0.0267, +0.0496] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0117 | +0.0117 | [-0.0188, +0.0421] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.3082 | -0.3082 | [-0.4896, -0.1420] | 0.4169 (0.1897) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1927 | -0.1927 | [-0.3307, -0.0531] | 0.4093 | INSIDE-SPREAD |
| ade6_m|bulk | -0.2246 | -0.2246 | [-0.4234, -0.0436] | 0.3850 | INSIDE-SPREAD |
| ade6_m|all | -0.2072 | -0.2072 | [-0.3292, -0.0950] | 0.2845 | INSIDE-SPREAD |
| ade_tac_m|all | -0.3082 | -0.3082 | [-0.4896, -0.1420] | 0.4169 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| HIER_s3 | 2.9201 | 3.2939 | 1.8914 | 1.1934 | 0.7724 |
| HIER_s4 | 2.9086 | 3.3309 | 1.7001 | 1.4618 | 0.7524 |
| HIER_s5 | 3.0344 | 3.4157 | 1.8393 | 1.4161 | 0.7624 |
| P_s0 | 3.1244 | 3.5986 | 1.9134 | 1.3721 | 0.7445 |
| PF_s0 | 3.1086 | 3.5808 | 1.7792 | 1.3666 | 0.7199 |
| PR_s0 | 3.2337 | 3.6268 | 2.0997 | 1.2000 | 0.7712 |
| RES_s0 | 3.0496 | 3.6074 | 1.9364 | 1.2088 | 0.7266 |
| RR_s0 | 2.9859 | 3.5105 | 1.8639 | 1.2017 | 0.7074 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
