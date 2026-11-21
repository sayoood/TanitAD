# RESULT -- refcv8 (EMA-48,852 + shipped H2cn stage) vs CAMERA-BLIND (W4) and EGO-ONLY FLOOR (W5)

**Tier** T1 self-action OPEN LOOP on logged eval139 frames (per-frame nav feed) -- not driving performance.  **Estimator** pair_read_r8.paired_cell -> taniteval.ci.paired_episode_cluster_bootstrap, B 2000, bootstrap seed 0, over EPISODES, models fixed; answers 'another draw of EPISODES' only (not another training run: one seed; not another inference run: seed 0 here).  
**Prereg** `PREREG_ANTICHEAT_REFCV8.md` sha256 `16a6ed68c5d815afa74335455ae1b67f7a620e917e5bbca831413c4a725f2ab2` (sealed before any score; `PREREG_ANTICHEAT_REFCV8.sha256`).  
**Disclosures** refcv8's speed band (fed ceiling `lim`) is FUTURE-DERIVED (ego max speed 2-6 s ahead; leak 0.224 vs 0.05 bar); refcv8's per-frame nav token is ego-future-derived (optimistic by construction); one training seed.

**W4 status:** CONTROLS PASS.  **W5 status:** CONTROLS PASS.

## Inference seed 0

Cells are `new - ref` on the ceiling-compliant windows, paired episode-cluster bootstrap (B 2000, seed 0). REAL better = delta < 0 on error rows, delta > 0 on TAC rows; `BETTER`/`WORSE` = the 95 % CI excludes 0 on that side, `n.s.` = not separated.

| contrast | S2 `ade_m` | S6 `ade_m` | S2 `LON_speed_mae_mps` | S2 `LAT_heading_mae_deg` | S2 `TAC_traj_lat_correct` | S2 `TAC_traj_lon_correct` | S6 `TAC_traj_lat_correct` | S6 `TAC_traj_lon_correct` |
|---|---|---|---|---|---|---|---|---|
| W4  REAL_OS - CAMERA-BLIND  (binding) | -0.4366 [-0.5263, -0.3580] BETTER | -3.2889 [-3.9860, -2.6674] BETTER | -0.0425 [-0.0533, -0.0314] BETTER | -2.8426 [-3.3213, -2.3978] BETTER | +0.1871 [+0.1493, +0.2288] BETTER | +0.0255 [+0.0081, +0.0421] BETTER | +0.2636 [+0.2204, +0.3121] BETTER | +0.1455 [+0.0955, +0.1957] BETTER |
| W5  REAL_REG - EGO-ONLY FLOOR  (binding) | -0.0280 [-0.0385, -0.0175] BETTER | -0.9131 [-1.1501, -0.7017] BETTER | -0.0198 [-0.0280, -0.0113] BETTER | -0.2603 [-0.3503, -0.1745] BETTER | +0.0368 [+0.0252, +0.0493] BETTER | +0.0209 [+0.0096, +0.0333] BETTER | +0.1037 [+0.0824, +0.1284] BETTER | +0.1387 [+0.0907, +0.1900] BETTER |
| beside  REAL_REG - CAMERA-BLIND | -0.4372 [-0.5277, -0.3586] BETTER | -3.2966 [-3.9952, -2.6725] BETTER | -0.0425 [-0.0533, -0.0314] BETTER | -2.8524 [-3.3347, -2.4066] BETTER | +0.1875 [+0.1502, +0.2291] BETTER | +0.0257 [+0.0082, +0.0424] BETTER | +0.2642 [+0.2203, +0.3128] BETTER | +0.1455 [+0.0955, +0.1957] BETTER |
| beside  REAL_OS - EGO-ONLY FLOOR | -0.0273 [-0.0376, -0.0171] BETTER | -0.9053 [-1.1389, -0.6966] BETTER | -0.0198 [-0.0280, -0.0113] BETTER | -0.2506 [-0.3379, -0.1664] BETTER | +0.0363 [+0.0246, +0.0490] BETTER | +0.0207 [+0.0092, +0.0333] BETTER | +0.1030 [+0.0813, +0.1279] BETTER | +0.1387 [+0.0907, +0.1900] BETTER |
| beside  REAL_REG - CONST (train-mean trajectory) | -7.3681 [-8.3054, -6.4436] BETTER | -19.6359 [-22.2110, -17.0600] BETTER | -5.8239 [-6.5785, -5.0751] BETTER | -1.5236 [-1.9236, -1.1714] BETTER | +0.0983 [+0.0660, +0.1356] BETTER | +0.0923 [+0.0628, +0.1217] BETTER | +0.1775 [+0.1321, +0.2301] BETTER | +0.1676 [+0.1139, +0.2218] BETTER |
| beside  REAL_REG - FLOOR_NL (boosted, same features) | -0.0835 [-0.1036, -0.0664] BETTER | -1.2289 [-1.4732, -1.0067] BETTER | -0.0771 [-0.0967, -0.0611] BETTER | -0.4683 [-0.6179, -0.3320] BETTER | +0.0392 [+0.0271, +0.0526] BETTER | +0.0522 [+0.0339, +0.0714] BETTER | +0.2058 [+0.1702, +0.2432] BETTER | +0.1476 [+0.1101, +0.1854] BETTER |
| beside  REAL_REG - ha0_ext_pose (kinematic prior) | -0.0471 [-0.0585, -0.0363] BETTER | -1.1975 [-1.4942, -0.9168] BETTER | -0.0343 [-0.0428, -0.0259] BETTER | -0.1940 [-0.2910, -0.1003] BETTER | +0.0269 [+0.0173, +0.0371] BETTER | +0.0127 [-0.0045, +0.0281] n.s. | +0.1120 [+0.0850, +0.1399] BETTER | +0.0975 [+0.0605, +0.1362] BETTER |
| beside  FLOOR - CONST | -7.3401 [-8.2764, -6.4080] BETTER | -18.7228 [-21.3140, -16.0903] BETTER | -5.8041 [-6.5563, -5.0534] BETTER | -1.2679 [-1.6110, -0.9644] BETTER | +0.0616 [+0.0352, +0.0931] BETTER | +0.0714 [+0.0456, +0.0976] BETTER | +0.0738 [+0.0360, +0.1177] BETTER | +0.0289 [-0.0398, +0.0951] n.s. |
| beside  FLOOR - ha0_ext_pose | -0.0192 [-0.0298, -0.0088] BETTER | -0.2844 [-0.5481, -0.0066] BETTER | -0.0145 [-0.0257, -0.0041] BETTER | +0.0648 [+0.0024, +0.1358] WORSE | -0.0099 [-0.0189, -0.0016] WORSE | -0.0082 [-0.0270, +0.0086] n.s. | +0.0083 [-0.0122, +0.0289] n.s. | -0.0412 [-0.0936, +0.0109] n.s. |
| draw floor  REAL_REG - REAL_OS (battery-tree plan vs os-only plan) | -0.0006 [-0.0015, +0.0002] n.s. | -0.0077 [-0.0140, -0.0019] BETTER | +0.0000 [-0.0000, +0.0000] n.s. | -0.0096 [-0.0185, -0.0009] BETTER | +0.0005 [-0.0014, +0.0022] n.s. | +0.0002 [+0.0000, +0.0007] n.s. | +0.0006 [-0.0045, +0.0064] n.s. | +0.0000 [+0.0000, +0.0000] n.s. |
| beside  RAW_REAL - RAW_BLIND (no stage) | -0.6335 [-0.7298, -0.5475] BETTER | -6.1304 [-7.1354, -5.1589] BETTER | -0.4216 [-0.5102, -0.3398] BETTER | -2.8015 [-3.2691, -2.3677] BETTER | +0.1863 [+0.1494, +0.2287] BETTER | +0.3114 [+0.2523, +0.3702] BETTER | +0.2701 [+0.2235, +0.3211] BETTER | +0.2372 [+0.1779, +0.2932] BETTER |
| beside  BLIND - FLOOR | +0.4092 [+0.3330, +0.4963] WORSE | +2.3835 [+1.8429, +3.0140] WORSE | +0.0226 [+0.0117, +0.0334] WORSE | +2.5976 [+2.1743, +3.0419] WORSE | -0.1508 [-0.1903, -0.1150] WORSE | -0.0048 [-0.0218, +0.0133] n.s. | -0.1606 [-0.2081, -0.1166] WORSE | -0.0068 [-0.0635, +0.0518] n.s. |

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
| REAL_REG | 0.2387 | 2.2123 |
| REAL_OS | 0.2393 | 2.2201 |
| RAW_REAL | 0.2805 | 2.6676 |
| HP | 0.2858 | 3.4098 |
| FLOOR | 0.2667 | 3.1254 |
| FLOOR_NL | 0.3222 | 3.4412 |
| CONST | 7.6068 | 21.8482 |
| BLIND | 0.6759 | 5.5089 |
| RAW_BLIND | 0.9139 | 8.7980 |

n S2: 4159 compliant windows / 138 episodes (all: 4754 / 139).
n S6: 3251 compliant windows / 138 episodes (all: 3668 / 139).

### W4  REAL_OS - CAMERA-BLIND, EVERY registered row (compliant)

| family | surface | metric [unit] | REAL | other | delta [95 % CI] | |
|---|---|---|---|---|---|---|
| ADE | S2 | ADE (mean over the surface's instants) [m] | 0.23932 | 0.6759 | -0.4366 [-0.5263, -0.3580] | BETTER |
| ADE | S6 | ADE (mean over the surface's instants) [m] | 2.2201 | 5.5089 | -3.2889 [-3.9860, -2.6674] | BETTER |
| ADE | S2 | FDE (final instant) [m] | 0.51665 | 1.4641 | -0.9474 [-1.1431, -0.7758] | BETTER |
| ADE | S6 | FDE (final instant) [m] | 5.4412 | 12.861 | -7.4198 [-9.0105, -6.0068] | BETTER |
| LONGITUDINAL | S2 | speed MAE [m/s] | 0.21295 | 0.25544 | -0.0425 [-0.0533, -0.0314] | BETTER |
| LONGITUDINAL | S6 | speed MAE [m/s] | 0.78262 | 1.0595 | -0.2769 [-0.3464, -0.2031] | BETTER |
| LONGITUDINAL | S2 | along-track MAE [m] | 0.18982 | 0.23327 | -0.0435 [-0.0542, -0.0329] | BETTER |
| LONGITUDINAL | S6 | along-track MAE [m] | 1.8047 | 2.6647 | -0.8600 [-1.0814, -0.6538] | BETTER |
| LONGITUDINAL | S2 | |along-track| at the final instant [m] | 0.42013 | 0.52642 | -0.1063 [-0.1332, -0.0803] | BETTER |
| LONGITUDINAL | S6 | |along-track| at the final instant [m] | 4.385 | 6.7301 | -2.3451 [-2.9579, -1.7738] | BETTER |
| LONGITUDINAL | S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.88759 | 0.85075 | +0.0368 [+0.0260, +0.0476] | BETTER |
| LONGITUDINAL | S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.5669 | 0.48734 | +0.0796 [+0.0508, +0.1082] | BETTER |
| LONGITUDINAL | S2 | accel MAE [m/s2] | 0.29183 | 0.35217 | -0.0603 [-0.0768, -0.0436] | BETTER |
| LONGITUDINAL | S6 | accel MAE [m/s2] | 0.42324 | 0.53015 | -0.1069 [-0.1381, -0.0749] | BETTER |
| LATERAL | S2 | heading MAE [deg] | 1.505 | 4.3476 | -2.8426 [-3.3213, -2.3978] | BETTER |
| LATERAL | S6 | heading MAE [deg] | 3.1987 | 9.808 | -6.6093 [-7.8965, -5.4789] | BETTER |
| LATERAL | S2 | curvature MAE (masked) [1/m] | 0.0037432 | 0.010251 | -0.0065 [-0.0077, -0.0053] | BETTER |
| LATERAL | S6 | curvature MAE (masked) [1/m] | 0.0051347 | 0.0095676 | -0.0044 [-0.0054, -0.0036] | BETTER |
| LATERAL | S2 | yaw-rate MAE (masked) [rad/s] | 0.017692 | 0.068254 | -0.0506 [-0.0592, -0.0428] | BETTER |
| LATERAL | S6 | yaw-rate MAE (masked) [rad/s] | 0.02729 | 0.060805 | -0.0335 [-0.0401, -0.0276] | BETTER |
| LATERAL | S2 | cross-track MAE [m] | 0.096379 | 0.5534 | -0.4570 [-0.5520, -0.3739] | BETTER |
| LATERAL | S6 | cross-track MAE [m] | 0.84943 | 4.0561 | -3.2067 [-3.9198, -2.5663] | BETTER |
| LATERAL | S2 | |cross-track| at the final instant [m] | 0.19671 | 1.1877 | -0.9909 [-1.1971, -0.8112] | BETTER |
| LATERAL | S6 | |cross-track| at the final instant [m] | 2.1355 | 9.1449 | -7.0094 [-8.6066, -5.5827] | BETTER |
| TACTICAL | S2 | trajectory-derived lateral decision correct [frac] | 0.96321 | 0.77615 | +0.1871 [+0.1493, +0.2288] | BETTER |
| TACTICAL | S6 | trajectory-derived lateral decision correct [frac] | 0.88926 | 0.62565 | +0.2636 [+0.2204, +0.3121] | BETTER |
| TACTICAL | S2 | trajectory-derived longitudinal decision correct [frac] | 0.86679 | 0.84131 | +0.0255 [+0.0081, +0.0421] | BETTER |
| TACTICAL | S6 | trajectory-derived longitudinal decision correct [frac] | 0.64042 | 0.49492 | +0.1455 [+0.0955, +0.1957] | BETTER |

### W5  REAL_REG - EGO-ONLY FLOOR, EVERY registered row (compliant)

| family | surface | metric [unit] | REAL | other | delta [95 % CI] | |
|---|---|---|---|---|---|---|
| ADE | S2 | ADE (mean over the surface's instants) [m] | 0.2387 | 0.26666 | -0.0280 [-0.0385, -0.0175] | BETTER |
| ADE | S6 | ADE (mean over the surface's instants) [m] | 2.2123 | 3.1254 | -0.9131 [-1.1501, -0.7017] | BETTER |
| ADE | S2 | FDE (final instant) [m] | 0.51508 | 0.61061 | -0.0955 [-0.1223, -0.0684] | BETTER |
| ADE | S6 | FDE (final instant) [m] | 5.4271 | 7.8169 | -2.3899 [-3.0522, -1.8075] | BETTER |
| LONGITUDINAL | S2 | speed MAE [m/s] | 0.21297 | 0.2328 | -0.0198 [-0.0280, -0.0113] | BETTER |
| LONGITUDINAL | S6 | speed MAE [m/s] | 0.78264 | 0.9244 | -0.1418 [-0.1923, -0.0904] | BETTER |
| LONGITUDINAL | S2 | along-track MAE [m] | 0.18976 | 0.20974 | -0.0200 [-0.0272, -0.0128] | BETTER |
| LONGITUDINAL | S6 | along-track MAE [m] | 1.8042 | 2.3399 | -0.5357 [-0.7221, -0.3685] | BETTER |
| LONGITUDINAL | S2 | |along-track| at the final instant [m] | 0.41993 | 0.47175 | -0.0518 [-0.0708, -0.0326] | BETTER |
| LONGITUDINAL | S6 | |along-track| at the final instant [m] | 4.3827 | 5.8157 | -1.4330 [-1.9691, -0.9489] | BETTER |
| LONGITUDINAL | S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.88759 | 0.87046 | +0.0171 [+0.0091, +0.0250] | BETTER |
| LONGITUDINAL | S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.56695 | 0.48775 | +0.0792 [+0.0536, +0.1064] | BETTER |
| LONGITUDINAL | S2 | accel MAE [m/s2] | 0.29187 | 0.30941 | -0.0175 [-0.0294, -0.0044] | BETTER |
| LONGITUDINAL | S6 | accel MAE [m/s2] | 0.42329 | 0.41458 | +0.0087 [-0.0119, +0.0319] | n.s. |
| LATERAL | S2 | heading MAE [deg] | 1.4983 | 1.7586 | -0.2603 [-0.3503, -0.1745] | BETTER |
| LATERAL | S6 | heading MAE [deg] | 3.2072 | 5.4374 | -2.2301 [-2.7924, -1.7232] | BETTER |
| LATERAL | S2 | curvature MAE (masked) [1/m] | 0.003715 | 0.0059283 | -0.0022 [-0.0032, -0.0014] | BETTER |
| LATERAL | S6 | curvature MAE (masked) [1/m] | 0.0055427 | 0.0079624 | -0.0024 [-0.0033, -0.0016] | BETTER |
| LATERAL | S2 | yaw-rate MAE (masked) [rad/s] | 0.017481 | 0.025564 | -0.0081 [-0.0103, -0.0060] | BETTER |
| LATERAL | S6 | yaw-rate MAE (masked) [rad/s] | 0.027634 | 0.038691 | -0.0111 [-0.0147, -0.0078] | BETTER |
| LATERAL | S2 | cross-track MAE [m] | 0.095503 | 0.10879 | -0.0133 [-0.0230, -0.0033] | BETTER |
| LATERAL | S6 | cross-track MAE [m] | 0.83905 | 1.4716 | -0.6326 [-0.8108, -0.4616] | BETTER |
| LATERAL | S2 | |cross-track| at the final instant [m] | 0.19453 | 0.26139 | -0.0669 [-0.0929, -0.0420] | BETTER |
| LATERAL | S6 | |cross-track| at the final instant [m] | 2.1181 | 3.7506 | -1.6325 [-2.1144, -1.1837] | BETTER |
| TACTICAL | S2 | trajectory-derived lateral decision correct [frac] | 0.96369 | 0.92691 | +0.0368 [+0.0252, +0.0493] | BETTER |
| TACTICAL | S6 | trajectory-derived lateral decision correct [frac] | 0.88988 | 0.78622 | +0.1037 [+0.0824, +0.1284] | BETTER |
| TACTICAL | S2 | trajectory-derived longitudinal decision correct [frac] | 0.86704 | 0.84612 | +0.0209 [+0.0096, +0.0333] | BETTER |
| TACTICAL | S6 | trajectory-derived longitudinal decision correct [frac] | 0.64042 | 0.50169 | +0.1387 [+0.0907, +0.1900] | BETTER |

## Inference seed 1

Cells are `new - ref` on the ceiling-compliant windows, paired episode-cluster bootstrap (B 2000, seed 0). REAL better = delta < 0 on error rows, delta > 0 on TAC rows; `BETTER`/`WORSE` = the 95 % CI excludes 0 on that side, `n.s.` = not separated.

| contrast | S2 `ade_m` | S6 `ade_m` | S2 `LON_speed_mae_mps` | S2 `LAT_heading_mae_deg` | S2 `TAC_traj_lat_correct` | S2 `TAC_traj_lon_correct` | S6 `TAC_traj_lat_correct` | S6 `TAC_traj_lon_correct` |
|---|---|---|---|---|---|---|---|---|
| W4  REAL_OS - CAMERA-BLIND  (binding) | -0.4373 [-0.5279, -0.3579] BETTER | -3.2712 [-3.9572, -2.6471] BETTER | -0.0425 [-0.0532, -0.0313] BETTER | -2.8563 [-3.3394, -2.4138] BETTER | +0.1892 [+0.1517, +0.2304] BETTER | +0.0255 [+0.0080, +0.0420] BETTER | +0.2645 [+0.2200, +0.3135] BETTER | +0.1455 [+0.0955, +0.1957] BETTER |
| W5  REAL_REG - EGO-ONLY FLOOR  (binding) | -0.0272 [-0.0372, -0.0174] BETTER | -0.9065 [-1.1380, -0.7005] BETTER | -0.0199 [-0.0280, -0.0113] BETTER | -0.2496 [-0.3382, -0.1658] BETTER | +0.0368 [+0.0252, +0.0491] BETTER | +0.0209 [+0.0096, +0.0333] BETTER | +0.1064 [+0.0845, +0.1309] BETTER | +0.1387 [+0.0907, +0.1900] BETTER |
| beside  REAL_REG - CAMERA-BLIND | -0.4373 [-0.5279, -0.3579] BETTER | -3.2712 [-3.9572, -2.6471] BETTER | -0.0425 [-0.0532, -0.0313] BETTER | -2.8563 [-3.3394, -2.4138] BETTER | +0.1892 [+0.1517, +0.2304] BETTER | +0.0255 [+0.0080, +0.0420] BETTER | +0.2645 [+0.2200, +0.3135] BETTER | +0.1455 [+0.0955, +0.1957] BETTER |
| beside  REAL_OS - EGO-ONLY FLOOR | -0.0272 [-0.0372, -0.0174] BETTER | -0.9065 [-1.1380, -0.7005] BETTER | -0.0199 [-0.0280, -0.0113] BETTER | -0.2496 [-0.3382, -0.1658] BETTER | +0.0368 [+0.0252, +0.0491] BETTER | +0.0209 [+0.0096, +0.0333] BETTER | +0.1064 [+0.0845, +0.1309] BETTER | +0.1387 [+0.0907, +0.1900] BETTER |
| beside  REAL_REG - CONST (train-mean trajectory) | -7.3674 [-8.3058, -6.4422] BETTER | -19.6293 [-22.2022, -17.0486] BETTER | -5.8239 [-6.5785, -5.0751] BETTER | -1.5131 [-1.9143, -1.1609] BETTER | +0.0983 [+0.0658, +0.1347] BETTER | +0.0923 [+0.0628, +0.1217] BETTER | +0.1803 [+0.1359, +0.2323] BETTER | +0.1676 [+0.1139, +0.2218] BETTER |
| beside  REAL_REG - FLOOR_NL (boosted, same features) | -0.0828 [-0.1033, -0.0655] BETTER | -1.2223 [-1.4580, -1.0037] BETTER | -0.0771 [-0.0967, -0.0611] BETTER | -0.4565 [-0.6064, -0.3230] BETTER | +0.0392 [+0.0271, +0.0533] BETTER | +0.0522 [+0.0339, +0.0714] BETTER | +0.2086 [+0.1724, +0.2460] BETTER | +0.1476 [+0.1101, +0.1854] BETTER |
| beside  REAL_REG - ha0_ext_pose (kinematic prior) | -0.0464 [-0.0576, -0.0360] BETTER | -1.1909 [-1.4800, -0.9163] BETTER | -0.0344 [-0.0428, -0.0259] BETTER | -0.1837 [-0.2772, -0.0927] BETTER | +0.0269 [+0.0171, +0.0370] BETTER | +0.0127 [-0.0045, +0.0281] n.s. | +0.1147 [+0.0888, +0.1422] BETTER | +0.0975 [+0.0605, +0.1362] BETTER |
| beside  FLOOR - CONST | -7.3401 [-8.2764, -6.4080] BETTER | -18.7228 [-21.3140, -16.0903] BETTER | -5.8041 [-6.5563, -5.0534] BETTER | -1.2679 [-1.6110, -0.9644] BETTER | +0.0616 [+0.0352, +0.0931] BETTER | +0.0714 [+0.0456, +0.0976] BETTER | +0.0738 [+0.0360, +0.1177] BETTER | +0.0289 [-0.0398, +0.0951] n.s. |
| beside  FLOOR - ha0_ext_pose | -0.0192 [-0.0298, -0.0088] BETTER | -0.2844 [-0.5481, -0.0066] BETTER | -0.0145 [-0.0257, -0.0041] BETTER | +0.0648 [+0.0024, +0.1358] WORSE | -0.0099 [-0.0189, -0.0016] WORSE | -0.0082 [-0.0270, +0.0086] n.s. | +0.0083 [-0.0122, +0.0289] n.s. | -0.0412 [-0.0936, +0.0109] n.s. |
| draw floor  REAL_REG - REAL_OS (battery-tree plan vs os-only plan) | +0.0000 [+0.0000, +0.0000] n.s. | +0.0000 [+0.0000, +0.0000] n.s. | +0.0000 [+0.0000, +0.0000] n.s. | +0.0000 [+0.0000, +0.0000] n.s. | +0.0000 [+0.0000, +0.0000] n.s. | +0.0000 [+0.0000, +0.0000] n.s. | +0.0000 [+0.0000, +0.0000] n.s. | +0.0000 [+0.0000, +0.0000] n.s. |
| beside  RAW_REAL - RAW_BLIND (no stage) | -0.6348 [-0.7303, -0.5486] BETTER | -6.1430 [-7.1403, -5.1801] BETTER | -0.4249 [-0.5118, -0.3434] BETTER | -2.8038 [-3.2720, -2.3708] BETTER | +0.1899 [+0.1521, +0.2323] BETTER | +0.3078 [+0.2492, +0.3670] BETTER | +0.2688 [+0.2237, +0.3197] BETTER | +0.2258 [+0.1659, +0.2840] BETTER |
| beside  BLIND - FLOOR | +0.4100 [+0.3351, +0.4979] WORSE | +2.3647 [+1.8291, +2.9979] WORSE | +0.0226 [+0.0117, +0.0333] WORSE | +2.6115 [+2.1863, +3.0570] WORSE | -0.1524 [-0.1910, -0.1164] WORSE | -0.0046 [-0.0215, +0.0136] n.s. | -0.1581 [-0.2060, -0.1148] WORSE | -0.0068 [-0.0635, +0.0518] n.s. |

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
| REAL_REG | 0.2394 | 2.2189 |
| REAL_OS | 0.2394 | 2.2189 |
| RAW_REAL | 0.2810 | 2.6810 |
| HP | 0.2858 | 3.4098 |
| FLOOR | 0.2667 | 3.1254 |
| FLOOR_NL | 0.3222 | 3.4412 |
| CONST | 7.6068 | 21.8482 |
| BLIND | 0.6767 | 5.4901 |
| RAW_BLIND | 0.9158 | 8.8240 |

n S2: 4159 compliant windows / 138 episodes (all: 4754 / 139).
n S6: 3251 compliant windows / 138 episodes (all: 3668 / 139).

### W4  REAL_OS - CAMERA-BLIND, EVERY registered row (compliant)

| family | surface | metric [unit] | REAL | other | delta [95 % CI] | |
|---|---|---|---|---|---|---|
| ADE | S2 | ADE (mean over the surface's instants) [m] | 0.23942 | 0.67671 | -0.4373 [-0.5279, -0.3579] | BETTER |
| ADE | S6 | ADE (mean over the surface's instants) [m] | 2.2189 | 5.4901 | -3.2712 [-3.9572, -2.6471] | BETTER |
| ADE | S2 | FDE (final instant) [m] | 0.51655 | 1.4655 | -0.9489 [-1.1495, -0.7761] | BETTER |
| ADE | S6 | FDE (final instant) [m] | 5.4451 | 12.808 | -7.3626 [-8.9356, -5.9579] | BETTER |
| LONGITUDINAL | S2 | speed MAE [m/s] | 0.21295 | 0.25541 | -0.0425 [-0.0532, -0.0313] | BETTER |
| LONGITUDINAL | S6 | speed MAE [m/s] | 0.78255 | 1.0595 | -0.2769 [-0.3464, -0.2032] | BETTER |
| LONGITUDINAL | S2 | along-track MAE [m] | 0.18993 | 0.23339 | -0.0435 [-0.0543, -0.0328] | BETTER |
| LONGITUDINAL | S6 | along-track MAE [m] | 1.8068 | 2.6641 | -0.8573 [-1.0816, -0.6513] | BETTER |
| LONGITUDINAL | S2 | |along-track| at the final instant [m] | 0.42053 | 0.52661 | -0.1061 [-0.1331, -0.0801] | BETTER |
| LONGITUDINAL | S6 | |along-track| at the final instant [m] | 4.3913 | 6.7246 | -2.3333 [-2.9455, -1.7681] | BETTER |
| LONGITUDINAL | S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.88759 | 0.85069 | +0.0369 [+0.0261, +0.0477] | BETTER |
| LONGITUDINAL | S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.5669 | 0.48749 | +0.0794 [+0.0507, +0.1081] | BETTER |
| LONGITUDINAL | S2 | accel MAE [m/s2] | 0.29183 | 0.35212 | -0.0603 [-0.0768, -0.0435] | BETTER |
| LONGITUDINAL | S6 | accel MAE [m/s2] | 0.42329 | 0.5301 | -0.1068 [-0.1381, -0.0748] | BETTER |
| LATERAL | S2 | heading MAE [deg] | 1.5058 | 4.3621 | -2.8563 [-3.3394, -2.4138] | BETTER |
| LATERAL | S6 | heading MAE [deg] | 3.1902 | 9.8115 | -6.6213 [-7.9218, -5.4869] | BETTER |
| LATERAL | S2 | curvature MAE (masked) [1/m] | 0.0037089 | 0.01039 | -0.0067 [-0.0079, -0.0055] | BETTER |
| LATERAL | S6 | curvature MAE (masked) [1/m] | 0.0051594 | 0.0096665 | -0.0045 [-0.0055, -0.0036] | BETTER |
| LATERAL | S2 | yaw-rate MAE (masked) [rad/s] | 0.017536 | 0.068496 | -0.0510 [-0.0595, -0.0431] | BETTER |
| LATERAL | S6 | yaw-rate MAE (masked) [rad/s] | 0.02733 | 0.061038 | -0.0337 [-0.0403, -0.0279] | BETTER |
| LATERAL | S2 | cross-track MAE [m] | 0.096437 | 0.55403 | -0.4576 [-0.5530, -0.3744] | BETTER |
| LATERAL | S6 | cross-track MAE [m] | 0.8461 | 4.0316 | -3.1855 [-3.8923, -2.5431] | BETTER |
| LATERAL | S2 | |cross-track| at the final instant [m] | 0.19662 | 1.1887 | -0.9920 [-1.2009, -0.8092] | BETTER |
| LATERAL | S6 | |cross-track| at the final instant [m] | 2.1385 | 9.0777 | -6.9391 [-8.5430, -5.5092] | BETTER |
| TACTICAL | S2 | trajectory-derived lateral decision correct [frac] | 0.96369 | 0.77447 | +0.1892 [+0.1517, +0.2304] | BETTER |
| TACTICAL | S6 | trajectory-derived lateral decision correct [frac] | 0.89265 | 0.62811 | +0.2645 [+0.2200, +0.3135] | BETTER |
| TACTICAL | S2 | trajectory-derived longitudinal decision correct [frac] | 0.86704 | 0.84155 | +0.0255 [+0.0080, +0.0420] | BETTER |
| TACTICAL | S6 | trajectory-derived longitudinal decision correct [frac] | 0.64042 | 0.49492 | +0.1455 [+0.0955, +0.1957] | BETTER |

### W5  REAL_REG - EGO-ONLY FLOOR, EVERY registered row (compliant)

| family | surface | metric [unit] | REAL | other | delta [95 % CI] | |
|---|---|---|---|---|---|---|
| ADE | S2 | ADE (mean over the surface's instants) [m] | 0.23942 | 0.26666 | -0.0272 [-0.0372, -0.0174] | BETTER |
| ADE | S6 | ADE (mean over the surface's instants) [m] | 2.2189 | 3.1254 | -0.9065 [-1.1380, -0.7005] | BETTER |
| ADE | S2 | FDE (final instant) [m] | 0.51655 | 0.61061 | -0.0941 [-0.1201, -0.0676] | BETTER |
| ADE | S6 | FDE (final instant) [m] | 5.4451 | 7.8169 | -2.3718 [-3.0177, -1.7872] | BETTER |
| LONGITUDINAL | S2 | speed MAE [m/s] | 0.21295 | 0.2328 | -0.0199 [-0.0280, -0.0113] | BETTER |
| LONGITUDINAL | S6 | speed MAE [m/s] | 0.78255 | 0.9244 | -0.1419 [-0.1924, -0.0905] | BETTER |
| LONGITUDINAL | S2 | along-track MAE [m] | 0.18993 | 0.20974 | -0.0198 [-0.0270, -0.0126] | BETTER |
| LONGITUDINAL | S6 | along-track MAE [m] | 1.8068 | 2.3399 | -0.5331 [-0.7169, -0.3663] | BETTER |
| LONGITUDINAL | S2 | |along-track| at the final instant [m] | 0.42053 | 0.47175 | -0.0512 [-0.0702, -0.0318] | BETTER |
| LONGITUDINAL | S6 | |along-track| at the final instant [m] | 4.3913 | 5.8157 | -1.4244 [-1.9553, -0.9484] | BETTER |
| LONGITUDINAL | S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.88759 | 0.87046 | +0.0171 [+0.0091, +0.0250] | BETTER |
| LONGITUDINAL | S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.5669 | 0.48775 | +0.0792 [+0.0536, +0.1063] | BETTER |
| LONGITUDINAL | S2 | accel MAE [m/s2] | 0.29183 | 0.30941 | -0.0176 [-0.0293, -0.0044] | BETTER |
| LONGITUDINAL | S6 | accel MAE [m/s2] | 0.42329 | 0.41458 | +0.0087 [-0.0119, +0.0319] | n.s. |
| LATERAL | S2 | heading MAE [deg] | 1.509 | 1.7586 | -0.2496 [-0.3382, -0.1658] | BETTER |
| LATERAL | S6 | heading MAE [deg] | 3.2323 | 5.4374 | -2.2051 [-2.7579, -1.7071] | BETTER |
| LATERAL | S2 | curvature MAE (masked) [1/m] | 0.0037074 | 0.0059283 | -0.0022 [-0.0032, -0.0014] | BETTER |
| LATERAL | S6 | curvature MAE (masked) [1/m] | 0.0055461 | 0.0079624 | -0.0024 [-0.0034, -0.0016] | BETTER |
| LATERAL | S2 | yaw-rate MAE (masked) [rad/s] | 0.017533 | 0.025564 | -0.0080 [-0.0102, -0.0059] | BETTER |
| LATERAL | S6 | yaw-rate MAE (masked) [rad/s] | 0.027694 | 0.038691 | -0.0110 [-0.0146, -0.0078] | BETTER |
| LATERAL | S2 | cross-track MAE [m] | 0.096437 | 0.10879 | -0.0124 [-0.0216, -0.0034] | BETTER |
| LATERAL | S6 | cross-track MAE [m] | 0.8461 | 1.4716 | -0.6255 [-0.7983, -0.4580] | BETTER |
| LATERAL | S2 | |cross-track| at the final instant [m] | 0.19662 | 0.26139 | -0.0648 [-0.0901, -0.0412] | BETTER |
| LATERAL | S6 | |cross-track| at the final instant [m] | 2.1385 | 3.7506 | -1.6120 [-2.0747, -1.1697] | BETTER |
| TACTICAL | S2 | trajectory-derived lateral decision correct [frac] | 0.96369 | 0.92691 | +0.0368 [+0.0252, +0.0491] | BETTER |
| TACTICAL | S6 | trajectory-derived lateral decision correct [frac] | 0.89265 | 0.78622 | +0.1064 [+0.0845, +0.1309] | BETTER |
| TACTICAL | S2 | trajectory-derived longitudinal decision correct [frac] | 0.86704 | 0.84612 | +0.0209 [+0.0096, +0.0333] | BETTER |
| TACTICAL | S6 | trajectory-derived longitudinal decision correct [frac] | 0.64042 | 0.50169 | +0.1387 [+0.0907, +0.1900] | BETTER |

## The EGO-ONLY floor (fit on TRAIN windows only)

| surface | n train windows | episodes | d | parametrisation | lambda | CV SSE / window (m^2) | train R^2 (positions) |
|---|---|---|---|---|---|---|---|
| S2 | 4211 | 137 | 10 | DIRECT | 0.1 | 0.9797 | 0.9978 |
| S6 | 3301 | 137 | 10 | DIRECT | 10 | 155.5544 | 0.9713 |

Features: progress_scalar.feats: [1, v0, a0, omega0, v0*omega0, v0^2, a0*v0, min(lim,40), min(lim,40)-v0, lim_valid].  Train set: A6 train139 ceiling-compliant windows (l1_fit_score.load_train).  Function class: ridge (linear in these features).  FLOOR_NL (HistGradientBoostingRegressor, same features) is beside.  Tactical-HEAD (logit) rows: n = 0 for a ridge floor (no class logits); the TAC rows above are trajectory-derived. STRATEGIC: n = 0.

## Controls (verbatim)

```
floor_known_values (K-FL1 recovers v0, K-FL2 shuffled target): True  | S2: K_FL1 R2=1.000000000000 K_FL2 R2=-0.00220; S6: K_FL1 R2=1.000000000000 K_FL2 R2=-0.00348
K_BASE_all_positions_bit_equal (REAL_REG == landed H2CN_DEPLOY8 package): True  | {"S2|s0": {"aligned": true, "bit_equal": true, "max_abs_m": 0.0}, "S6|s0": {"aligned": true, "bit_equal": true, "max_abs_m": 0.0}, "S2|s1": {"aligned": true, "bit_equal": true, "max_abs_m": 0.0}, "S6|s1": {"aligned": true, "bit_equal": true, "max_abs_m": 0.0}}
K_BASE_cells_equal (E1/E3 reproduce landed): True  | {"s0": {"E1_S2": {"delta": -0.0471, "lo": -0.0585, "hi": -0.0363, "separated": true, "n_windows": 4159, "n_episodes": 138, "mean_new": 0.23869576250182253, "mean_ref": 0.28581937894529424, "status": null}, "E3_S6": {"delta": -1.1975, "lo": -1.4942, "hi": -0.9168, "separated": true, "n_windows": 3251, "n_episodes": 138, "mean_new": 2.2123268279881523, "mean_ref": 3.4097815215823526, "status": null}, "landed_S2": {"delta": -0.0471, "lo": -0.0585, "hi": -0.0363, "n_windows": 4159, "n_episodes": 138}, "cell_equal_S2": true, "landed_S6": {"delta": -1.1975, "lo": -1.4942, "hi": -0.9168, "n_windows": 3251, "n_episodes": 138}, "cell_equal_S6": true}, "s1": {"E1_S2": {"delta": -0.0464, "lo": -0.0576, "hi": -0.036, "separated": true, "n_windows": 4159, "n_episodes": 138, "mean_new": 0.2394174276644018, "mean_ref": 0.28581937894529424, "status": null}, "E3_S6": {"delta": -1.1909, "lo": -1.48, "hi": -0.9163, "separated": true, "n_windows": 3251, "n_episodes": 138, "mean_new": 2.2189281648927306, "mean_ref": 3.4097815215823526, "status": null}, "landed_S2": {"delta": -0.0464, "lo": -0.0576, "hi": -0.036, "n_windows": 4159, "n_episodes": 138}, "cell_equal_S2": true, "landed_S6": {"delta": -1.1909, "lo": -1.48, "hi": -0.9163, "n_windows": 3251, "n_episodes": 138}, "cell_equal_S6": true}}
K_ID (REAL-vs-REAL = 0 with zero-width CI; GT TAC = 1.0, ADE = 0): True
W4 roll controls: {"blind_roll_final_4754": true, "derangement_ok": true, "b_own_sample": true, "a_identity_bit_equal_all": true, "b_changed_fraction_100": true, "c_ego_inputs_equal": true}
derangement: {"n_episodes": 139, "n_windows": 4754, "seed": 0, "fixed_points": 0, "each_episode_a_donor_once": true, "mapping_sha256": "48ff98cb395ef572ccbbadfcad59fd8be883666b4d485631817f3f1b0d33ba43"}
(a) identity vs BASE bank: {"n_base": 4754, "n_new": 4754, "keys_equal": true, "arrays": {"traj_own": {"shape": [4754, 8, 2], "max_abs_diff": 0.0, "bytes_equal": true}, "z_tac": {"shape": [4754, 512], "max_abs_diff": 0.0, "bytes_equal": true}, "sel_own": {"shape": [4754], "max_abs_diff": 0.0, "bytes_equal": true}}, "PASS": true, "identity_final_4754": true}
(b) changed fraction vs identity roll: {"PASS": true, "n": 4754, "n_changed": 4754, "fraction_changed": 1.0}
(b) own-frame sample inside the blind roll: {"n": 191, "n_own_differs_fed": 191, "PASS": true}
(c) ego inputs equal vs identity roll: {"PASS": true, "n": 4754, "n_equal": 4754}
```
