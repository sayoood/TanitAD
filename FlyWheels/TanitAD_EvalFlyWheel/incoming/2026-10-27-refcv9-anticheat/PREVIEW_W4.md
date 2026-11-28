# PREVIEW (UNREGISTERED, PARTIAL) -- W4 on the first 1,500 of 4,754 windows of the blind roll; NOT a verdict, controls (a)(b)(c) against the full identity roll are pending

**Tier** T1 self-action OPEN LOOP on logged eval139 frames (per-frame nav feed) -- not driving performance.  **Estimator** pair_read_r8.paired_cell -> taniteval.ci.paired_episode_cluster_bootstrap, B 2000, bootstrap seed 0, over EPISODES, models fixed; answers 'another draw of EPISODES' only (not another training run: one seed; not another inference run: seed 0 here).  
**Prereg** `PREREG_ANTICHEAT_REFCV8.md` sha256 `16a6ed68c5d815afa74335455ae1b67f7a620e917e5bbca831413c4a725f2ab2` (sealed before any score; `PREREG_ANTICHEAT_REFCV8.sha256`).  
**Disclosures** refcv8's speed band (fed ceiling `lim`) is FUTURE-DERIVED (ego max speed 2-6 s ahead; leak 0.224 vs 0.05 bar); refcv8's per-frame nav token is ego-future-derived (optimistic by construction); one training seed.

**W4 status:** PREVIEW ONLY (unregistered, partial windows; no verdict).  **W5 status:** CONTROLS PASS.

## Inference seed 0

Cells are `new - ref` on the ceiling-compliant windows, paired episode-cluster bootstrap (B 2000, seed 0). REAL better = delta < 0 on error rows, delta > 0 on TAC rows; `BETTER`/`WORSE` = the 95 % CI excludes 0 on that side, `n.s.` = not separated.

| contrast | S2 `ade_m` | S6 `ade_m` | S2 `LON_speed_mae_mps` | S2 `LAT_heading_mae_deg` | S2 `TAC_traj_lat_correct` | S2 `TAC_traj_lon_correct` | S6 `TAC_traj_lat_correct` | S6 `TAC_traj_lon_correct` |
|---|---|---|---|---|---|---|---|---|
| W4  REAL_OS - CAMERA-BLIND  (binding) | -0.3948 [-0.4914, -0.3039] BETTER | -2.9865 [-3.7421, -2.2704] BETTER | -0.0456 [-0.0646, -0.0276] BETTER | -2.5563 [-3.2485, -1.9429] BETTER | +0.1486 [+0.0955, +0.2082] BETTER | +0.0430 [+0.0203, +0.0697] BETTER | +0.2651 [+0.1952, +0.3359] BETTER | +0.2076 [+0.1124, +0.3060] BETTER |
| W5  REAL_REG - EGO-ONLY FLOOR  (binding) | -0.0211 [-0.0381, -0.0046] BETTER | -0.7935 [-1.0263, -0.5638] BETTER | -0.0222 [-0.0354, -0.0088] BETTER | -0.1373 [-0.2532, -0.0364] BETTER | +0.0281 [+0.0100, +0.0477] BETTER | +0.0360 [+0.0170, +0.0579] BETTER | +0.0858 [+0.0503, +0.1239] BETTER | +0.2008 [+0.1098, +0.2977] BETTER |
| beside  REAL_REG - CAMERA-BLIND | -0.3947 [-0.4909, -0.3042] BETTER | -2.9901 [-3.7404, -2.2799] BETTER | -0.0455 [-0.0645, -0.0276] BETTER | -2.5658 [-3.2572, -1.9471] BETTER | +0.1501 [+0.0973, +0.2088] BETTER | +0.0430 [+0.0203, +0.0697] BETTER | +0.2651 [+0.1950, +0.3367] BETTER | +0.2076 [+0.1124, +0.3060] BETTER |
| beside  REAL_OS - EGO-ONLY FLOOR | -0.0212 [-0.0376, -0.0052] BETTER | -0.7900 [-1.0195, -0.5577] BETTER | -0.0222 [-0.0354, -0.0089] BETTER | -0.1292 [-0.2424, -0.0252] BETTER | +0.0266 [+0.0073, +0.0465] BETTER | +0.0360 [+0.0170, +0.0579] BETTER | +0.0858 [+0.0511, +0.1241] BETTER | +0.2008 [+0.1098, +0.2977] BETTER |
| beside  REAL_REG - CONST (train-mean trajectory) | -7.8462 [-9.5724, -6.2822] BETTER | -20.8597 [-25.7933, -16.4720] BETTER | -6.2390 [-7.6229, -4.9694] BETTER | -1.0978 [-1.5795, -0.6831] BETTER | +0.0633 [+0.0274, +0.1104] BETTER | +0.1290 [+0.0803, +0.1865] BETTER | +0.1267 [+0.0697, +0.1964] BETTER | +0.2027 [+0.1182, +0.2985] BETTER |
| beside  REAL_REG - FLOOR_NL (boosted, same features) | -0.0732 [-0.1129, -0.0407] BETTER | -1.0895 [-1.3632, -0.8443] BETTER | -0.0928 [-0.1504, -0.0552] BETTER | -0.2362 [-0.3631, -0.1185] BETTER | +0.0266 [+0.0077, +0.0464] BETTER | +0.0711 [+0.0352, +0.1191] BETTER | +0.1881 [+0.1322, +0.2470] BETTER | +0.1940 [+0.1289, +0.2593] BETTER |
| beside  REAL_REG - ha0_ext_pose (kinematic prior) | -0.0492 [-0.0645, -0.0326] BETTER | -1.4132 [-1.9189, -0.9501] BETTER | -0.0443 [-0.0557, -0.0328] BETTER | -0.1503 [-0.2847, -0.0236] BETTER | +0.0281 [+0.0149, +0.0427] BETTER | +0.0321 [+0.0152, +0.0497] BETTER | +0.1228 [+0.0777, +0.1662] BETTER | +0.0965 [+0.0347, +0.1554] BETTER |
| beside  FLOOR - CONST | -7.8251 [-9.5522, -6.2574] BETTER | -20.0662 [-25.0151, -15.6026] BETTER | -6.2168 [-7.6086, -4.9410] BETTER | -0.9614 [-1.3703, -0.6148] BETTER | +0.0352 [-0.0033, +0.0811] n.s. | +0.0930 [+0.0522, +0.1410] BETTER | +0.0409 [-0.0146, +0.1003] n.s. | +0.0019 [-0.1292, +0.1402] n.s. |
| beside  FLOOR - ha0_ext_pose | -0.0281 [-0.0444, -0.0109] BETTER | -0.6197 [-1.0204, -0.2438] BETTER | -0.0221 [-0.0373, -0.0051] BETTER | -0.0132 [-0.1158, +0.0871] n.s. | +0.0000 [-0.0151, +0.0148] n.s. | -0.0039 [-0.0265, +0.0176] n.s. | +0.0370 [+0.0020, +0.0724] BETTER | -0.1043 [-0.1942, -0.0101] WORSE |
| draw floor  REAL_REG - REAL_OS (battery-tree plan vs os-only plan) | +0.0001 [-0.0015, +0.0018] n.s. | -0.0035 [-0.0136, +0.0062] n.s. | +0.0000 [-0.0000, +0.0001] n.s. | -0.0083 [-0.0232, +0.0057] n.s. | +0.0016 [-0.0015, +0.0048] n.s. | +0.0000 [+0.0000, +0.0000] n.s. | +0.0000 [-0.0066, +0.0069] n.s. | +0.0000 [+0.0000, +0.0000] n.s. |
| beside  RAW_REAL - RAW_BLIND (no stage) | -0.6340 [-0.7662, -0.5136] BETTER | -6.7551 [-8.3919, -5.1728] BETTER | -0.5119 [-0.6774, -0.3605] BETTER | -2.5035 [-3.1458, -1.9179] BETTER | +0.1470 [+0.0959, +0.2050] BETTER | +0.3761 [+0.2744, +0.4800] BETTER | +0.2739 [+0.1988, +0.3499] BETTER | +0.2758 [+0.1582, +0.3934] BETTER |
| beside  BLIND - FLOOR | +0.3736 [+0.2821, +0.4719] WORSE | +2.1966 [+1.5385, +2.9508] WORSE | +0.0234 [+0.0048, +0.0442] WORSE | +2.4286 [+1.8382, +3.1024] WORSE | -0.1220 [-0.1780, -0.0700] WORSE | -0.0070 [-0.0330, +0.0125] n.s. | -0.1793 [-0.2457, -0.1131] WORSE | -0.0068 [-0.1084, +0.0943] n.s. |

**W4_REAL_OS_vs_BLIND**  OVERALL: **PASS**  
* ADE: **PASS**  (S2|ade_m=BETTER_SEP, S6|ade_m=BETTER_SEP)
* LONGITUDINAL: **PASS**  (S2|LON_speed_mae_mps=BETTER_SEP)
* LATERAL: **PASS**  (S2|LAT_heading_mae_deg=BETTER_SEP)
* TACTICAL: **PASS**  (S2|TAC_traj_lat_correct=BETTER_SEP, S2|TAC_traj_lon_correct=BETTER_SEP, S6|TAC_traj_lat_correct=BETTER_SEP, S6|TAC_traj_lon_correct=BETTER_SEP)
* STRATEGIC: **N/A (n = 0)**

**W5_REAL_REG_vs_FLOOR**  OVERALL: **PASS**  
* ADE: **PASS**  (S2|ade_m=BETTER_SEP, S6|ade_m=BETTER_SEP)
* LONGITUDINAL: **PASS**  (S2|LON_speed_mae_mps=BETTER_SEP)
* LATERAL: **PASS**  (S2|LAT_heading_mae_deg=BETTER_SEP)
* TACTICAL: **PASS**  (S2|TAC_traj_lat_correct=BETTER_SEP, S2|TAC_traj_lon_correct=BETTER_SEP, S6|TAC_traj_lat_correct=BETTER_SEP, S6|TAC_traj_lon_correct=BETTER_SEP)
* STRATEGIC: **N/A (n = 0)**

**beside_REAL_REG_vs_BLIND**  OVERALL: **PASS**  
* ADE: **PASS**  (S2|ade_m=BETTER_SEP, S6|ade_m=BETTER_SEP)
* LONGITUDINAL: **PASS**  (S2|LON_speed_mae_mps=BETTER_SEP)
* LATERAL: **PASS**  (S2|LAT_heading_mae_deg=BETTER_SEP)
* TACTICAL: **PASS**  (S2|TAC_traj_lat_correct=BETTER_SEP, S2|TAC_traj_lon_correct=BETTER_SEP, S6|TAC_traj_lat_correct=BETTER_SEP, S6|TAC_traj_lon_correct=BETTER_SEP)
* STRATEGIC: **N/A (n = 0)**

### Arm means on the compliant set (ade_m, m)

| arm | S2 ade_m | S6 ade_m |
|---|---|---|
| REAL_REG | 0.2370 | 2.1650 |
| REAL_OS | 0.2369 | 2.1685 |
| RAW_REAL | 0.2681 | 2.4908 |
| HP | 0.2862 | 3.5782 |
| FLOOR | 0.2581 | 2.9585 |
| FLOOR_NL | 0.3102 | 3.2544 |
| CONST | 8.0833 | 23.0246 |
| BLIND | 0.6318 | 5.1550 |
| RAW_BLIND | 0.9021 | 9.2459 |

n S2: 1279 compliant windows / 44 episodes (all: 1500 / 44).
n S6: 1026 compliant windows / 44 episodes (all: 1161 / 44).

### W4  REAL_OS - CAMERA-BLIND, EVERY registered row (compliant)

| family | surface | metric [unit] | REAL | other | delta [95 % CI] | |
|---|---|---|---|---|---|---|
| ADE | S2 | ADE (mean over the surface's instants) [m] | 0.23694 | 0.63176 | -0.3948 [-0.4914, -0.3039] | BETTER |
| ADE | S6 | ADE (mean over the surface's instants) [m] | 2.1685 | 5.155 | -2.9865 [-3.7421, -2.2704] | BETTER |
| ADE | S2 | FDE (final instant) [m] | 0.51129 | 1.3652 | -0.8539 [-1.0654, -0.6586] | BETTER |
| ADE | S6 | FDE (final instant) [m] | 5.3117 | 11.998 | -6.6858 [-8.2978, -5.1842] | BETTER |
| LONGITUDINAL | S2 | speed MAE [m/s] | 0.21491 | 0.26048 | -0.0456 [-0.0646, -0.0276] | BETTER |
| LONGITUDINAL | S6 | speed MAE [m/s] | 0.76245 | 1.0994 | -0.3369 [-0.4665, -0.2139] | BETTER |
| LONGITUDINAL | S2 | along-track MAE [m] | 0.18475 | 0.22963 | -0.0449 [-0.0641, -0.0260] | BETTER |
| LONGITUDINAL | S6 | along-track MAE [m] | 1.7397 | 2.5133 | -0.7735 [-1.0686, -0.4928] | BETTER |
| LONGITUDINAL | S2 | |along-track| at the final instant [m] | 0.40954 | 0.51706 | -0.1075 [-0.1533, -0.0624] | BETTER |
| LONGITUDINAL | S6 | |along-track| at the final instant [m] | 4.1929 | 6.305 | -2.1121 [-2.9481, -1.3263] | BETTER |
| LONGITUDINAL | S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.88976 | 0.84343 | +0.0463 [+0.0288, +0.0661] | BETTER |
| LONGITUDINAL | S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.56693 | 0.46004 | +0.1069 [+0.0583, +0.1610] | BETTER |
| LONGITUDINAL | S2 | accel MAE [m/s2] | 0.28253 | 0.34899 | -0.0665 [-0.0906, -0.0419] | BETTER |
| LONGITUDINAL | S6 | accel MAE [m/s2] | 0.39191 | 0.5346 | -0.1427 [-0.1927, -0.0947] | BETTER |
| LATERAL | S2 | heading MAE [deg] | 0.69463 | 3.2509 | -2.5563 [-3.2485, -1.9429] | BETTER |
| LATERAL | S6 | heading MAE [deg] | 2.2701 | 8.2071 | -5.9370 [-7.5691, -4.5410] | BETTER |
| LATERAL | S2 | curvature MAE (masked) [1/m] | 0.0025817 | 0.0089094 | -0.0063 [-0.0086, -0.0043] | BETTER |
| LATERAL | S6 | curvature MAE (masked) [1/m] | 0.0037151 | 0.0079947 | -0.0043 [-0.0058, -0.0030] | BETTER |
| LATERAL | S2 | yaw-rate MAE (masked) [rad/s] | 0.015424 | 0.060301 | -0.0449 [-0.0562, -0.0346] | BETTER |
| LATERAL | S6 | yaw-rate MAE (masked) [rad/s] | 0.024354 | 0.053178 | -0.0288 [-0.0365, -0.0219] | BETTER |
| LATERAL | S2 | cross-track MAE [m] | 0.099617 | 0.51652 | -0.4169 [-0.5194, -0.3159] | BETTER |
| LATERAL | S6 | cross-track MAE [m] | 0.84647 | 3.8016 | -2.9551 [-3.7352, -2.2221] | BETTER |
| LATERAL | S2 | |cross-track| at the final instant [m] | 0.20196 | 1.1048 | -0.9028 [-1.1298, -0.6858] | BETTER |
| LATERAL | S6 | |cross-track| at the final instant [m] | 2.1609 | 8.53 | -6.3691 [-8.0405, -4.8108] | BETTER |
| TACTICAL | S2 | trajectory-derived lateral decision correct [frac] | 0.97342 | 0.82486 | +0.1486 [+0.0955, +0.2082] | BETTER |
| TACTICAL | S6 | trajectory-derived lateral decision correct [frac] | 0.89571 | 0.6306 | +0.2651 [+0.1952, +0.3359] | BETTER |
| TACTICAL | S2 | trajectory-derived longitudinal decision correct [frac] | 0.87803 | 0.83503 | +0.0430 [+0.0203, +0.0697] | BETTER |
| TACTICAL | S6 | trajectory-derived longitudinal decision correct [frac] | 0.66862 | 0.46101 | +0.2076 [+0.1124, +0.3060] | BETTER |

### W5  REAL_REG - EGO-ONLY FLOOR, EVERY registered row (compliant)

| family | surface | metric [unit] | REAL | other | delta [95 % CI] | |
|---|---|---|---|---|---|---|
| ADE | S2 | ADE (mean over the surface's instants) [m] | 0.23705 | 0.25811 | -0.0211 [-0.0381, -0.0046] | BETTER |
| ADE | S6 | ADE (mean over the surface's instants) [m] | 2.165 | 2.9585 | -0.7935 [-1.0263, -0.5638] | BETTER |
| ADE | S2 | FDE (final instant) [m] | 0.5109 | 0.59151 | -0.0806 [-0.1216, -0.0421] | BETTER |
| ADE | S6 | FDE (final instant) [m] | 5.3097 | 7.3787 | -2.0690 [-2.6723, -1.4704] | BETTER |
| LONGITUDINAL | S2 | speed MAE [m/s] | 0.21493 | 0.2371 | -0.0222 [-0.0354, -0.0088] | BETTER |
| LONGITUDINAL | S6 | speed MAE [m/s] | 0.76251 | 0.92405 | -0.1615 [-0.2344, -0.0866] | BETTER |
| LONGITUDINAL | S2 | along-track MAE [m] | 0.18473 | 0.20434 | -0.0196 [-0.0310, -0.0075] | BETTER |
| LONGITUDINAL | S6 | along-track MAE [m] | 1.738 | 2.1368 | -0.3988 [-0.5795, -0.2186] | BETTER |
| LONGITUDINAL | S2 | |along-track| at the final instant [m] | 0.40932 | 0.46009 | -0.0508 [-0.0790, -0.0214] | BETTER |
| LONGITUDINAL | S6 | |along-track| at the final instant [m] | 4.1846 | 5.2535 | -1.0689 [-1.5531, -0.5903] | BETTER |
| LONGITUDINAL | S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.88976 | 0.86689 | +0.0229 [+0.0113, +0.0340] | BETTER |
| LONGITUDINAL | S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.56676 | 0.48538 | +0.0814 [+0.0361, +0.1235] | BETTER |
| LONGITUDINAL | S2 | accel MAE [m/s2] | 0.28261 | 0.30699 | -0.0244 [-0.0414, -0.0071] | BETTER |
| LONGITUDINAL | S6 | accel MAE [m/s2] | 0.39209 | 0.4126 | -0.0205 [-0.0476, +0.0075] | n.s. |
| LATERAL | S2 | heading MAE [deg] | 0.69341 | 0.83075 | -0.1373 [-0.2532, -0.0364] | BETTER |
| LATERAL | S6 | heading MAE [deg] | 2.2701 | 4.0373 | -1.7671 [-2.5979, -1.0637] | BETTER |
| LATERAL | S2 | curvature MAE (masked) [1/m] | 0.0025445 | 0.003714 | -0.0012 [-0.0019, -0.0006] | BETTER |
| LATERAL | S6 | curvature MAE (masked) [1/m] | 0.0036857 | 0.0054546 | -0.0018 [-0.0033, -0.0007] | BETTER |
| LATERAL | S2 | yaw-rate MAE (masked) [rad/s] | 0.015211 | 0.020375 | -0.0052 [-0.0081, -0.0027] | BETTER |
| LATERAL | S6 | yaw-rate MAE (masked) [rad/s] | 0.024486 | 0.031043 | -0.0066 [-0.0109, -0.0026] | BETTER |
| LATERAL | S2 | cross-track MAE [m] | 0.09934 | 0.10303 | -0.0037 [-0.0189, +0.0098] | n.s. |
| LATERAL | S6 | cross-track MAE [m] | 0.84425 | 1.4457 | -0.6014 [-0.8595, -0.3946] | BETTER |
| LATERAL | S2 | |cross-track| at the final instant [m] | 0.20065 | 0.24706 | -0.0464 [-0.0864, -0.0130] | BETTER |
| LATERAL | S6 | |cross-track| at the final instant [m] | 2.1676 | 3.6865 | -1.5188 [-2.1321, -1.0127] | BETTER |
| TACTICAL | S2 | trajectory-derived lateral decision correct [frac] | 0.97498 | 0.94683 | +0.0281 [+0.0100, +0.0477] | BETTER |
| TACTICAL | S6 | trajectory-derived lateral decision correct [frac] | 0.89571 | 0.80994 | +0.0858 [+0.0503, +0.1239] | BETTER |
| TACTICAL | S2 | trajectory-derived longitudinal decision correct [frac] | 0.87803 | 0.84206 | +0.0360 [+0.0170, +0.0579] | BETTER |
| TACTICAL | S6 | trajectory-derived longitudinal decision correct [frac] | 0.66862 | 0.46784 | +0.2008 [+0.1098, +0.2977] | BETTER |

## The EGO-ONLY floor (fit on TRAIN windows only)

| surface | n train windows | episodes | d | parametrisation | lambda | CV SSE / window (m^2) | train R^2 (positions) |
|---|---|---|---|---|---|---|---|
| S2 | 4211 | 137 | 10 | DIRECT | 0.1 | 0.9797 | 0.9978 |
| S6 | 3301 | 137 | 10 | DIRECT | 10 | 155.5544 | 0.9713 |

Features: progress_scalar.feats: [1, v0, a0, omega0, v0*omega0, v0^2, a0*v0, min(lim,40), min(lim,40)-v0, lim_valid].  Train set: A6 train139 ceiling-compliant windows (l1_fit_score.load_train).  Function class: ridge (linear in these features).  FLOOR_NL (HistGradientBoostingRegressor, same features) is beside.  Tactical-HEAD (logit) rows: n = 0 for a ridge floor (no class logits); the TAC rows above are trajectory-derived. STRATEGIC: n = 0.

## Controls (verbatim)

```
floor_known_values (K-FL1 recovers v0, K-FL2 shuffled target): True  | S2: K_FL1 R2=1.000000000000 K_FL2 R2=-0.00220; S6: K_FL1 R2=1.000000000000 K_FL2 R2=-0.00348
K_BASE_all_positions_bit_equal (REAL_REG == landed H2CN_DEPLOY8 package): True  | {"S2|s0": {"aligned": true, "bit_equal": true, "max_abs_m": 0.0}, "S6|s0": {"aligned": true, "bit_equal": true, "max_abs_m": 0.0}}
K_BASE_cells_equal (E1/E3 reproduce landed): None  | {"s0": {"E1_S2": {"delta": -0.0492, "lo": -0.0645, "hi": -0.0326, "separated": true, "n_windows": 1279, "n_episodes": 44, "mean_new": 0.2370454830061455, "mean_ref": 0.28622226216164043, "status": null}, "E3_S6": {"delta": -1.4132, "lo": -1.9189, "hi": -0.9501, "separated": true, "n_windows": 1026, "n_episodes": 44, "mean_new": 2.1649517481511213, "mean_ref": 3.578172889797264, "status": null}, "landed_S2": {"delta": -0.0471, "lo": -0.0585, "hi": -0.0363, "n_windows": 4159, "n_episodes": 138}, "cell_equal_S2": false, "landed_S6": {"delta": -1.1975, "lo": -1.4942, "hi": -0.9168, "n_windows": 3251, "n_episodes": 138}, "cell_equal_S6": false}}
K_ID (REAL-vs-REAL = 0 with zero-width CI; GT TAC = 1.0, ADE = 0): True
W4 roll controls: {"blind_roll_final_4754": false, "derangement_ok": true, "b_own_sample": true, "a_identity_bit_equal_all": null, "b_changed_fraction_100": null, "c_ego_inputs_equal": null}
derangement: {"n_episodes": 139, "n_windows": 4754, "seed": 0, "fixed_points": 0, "each_episode_a_donor_once": true, "mapping_sha256": "48ff98cb395ef572ccbbadfcad59fd8be883666b4d485631817f3f1b0d33ba43"}
(a) identity vs BASE bank: {"PASS": null, "status": "PENDING (identity roll not provided)"}
(b) changed fraction vs identity roll: null
(b) own-frame sample inside the blind roll: {"n": 60, "n_own_differs_fed": 60, "PASS": true}
(c) ego inputs equal vs identity roll: null
```
