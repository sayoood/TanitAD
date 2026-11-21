# PERC2PLAN-R8 — does refcv8's planner USE its perception inputs?

* Rendered 2026-10-22 03:31:43 +0200 from `C:/Users/Admin/refcv8_pf/perc2plan/PERC2PLAN.json`; prereg `PREREG_PERC2PLAN.md` sha256 `d1a2eb53d4488dde404b4c30b84ddb9d35bfa0aead8564fa07d729d4e2b68568`.
* **Tier: T1 self-action OPEN LOOP on logged eval139 frames (the SHIP-R8 tier); ONE training seed.**
* **Interval:** paired episode-cluster bootstrap (refav1_arm._paired_families), B 2000 -- answers 'another draw of EPISODES' only; inference-run variance is read from the second inference seed (floor).
* Model: refcv8 EMA-48,852, per-frame nav feed, eval139 (4,754 windows / 139 episodes), os-only arm set, inference seeds 0 and 1, Thor.

## Verdicts (mechanical, PREREG §4)

| arm | headline | S2 verdict | S6 verdict | S2 ADE Δ s0 | S2 ADE Δ s1 | S6 ADE Δ s0 | S6 ADE Δ s1 |
|---|---|---|---|---|---|---|---|
| AGT_ZERO | **USED (on S2,S6)** | USED | USED | +0.0153 [+0.0091, +0.0221] | +0.0157 [+0.0095, +0.0225] | +0.1139 [+0.0443, +0.1921] | +0.0940 [+0.0338, +0.1579] |
| AGT_SHUF | **USED (on S2,S6)** | USED | USED | +0.0092 [+0.0012, +0.0170] | +0.0105 [+0.0031, +0.0186] | +0.0937 [+0.0269, +0.1706] | +0.0794 [+0.0057, +0.1692] |
| BEV_ZERO | **USED (on S2,S6)** | USED | USED | +0.2060 [+0.1813, +0.2329] | +0.2018 [+0.1768, +0.2284] | +1.1505 [+0.9510, +1.3660] | +1.1282 [+0.9333, +1.3484] |
| BEV_SHUF | **USED (on S2,S6)** | USED | USED | +0.2083 [+0.1730, +0.2449] | +0.2065 [+0.1719, +0.2431] | +1.2821 [+1.0458, +1.5314] | +1.2369 [+1.0083, +1.4717] |
| BOTH_ZERO | **USED (on S2,S6)** | USED | USED | +0.2113 [+0.1857, +0.2384] | +0.2129 [+0.1869, +0.2411] | +1.2030 [+1.0002, +1.4277] | +1.1993 [+0.9978, +1.4260] |

ADE Δ = arm − BASE at the same inference seed, metres, 95 % paired episode-cluster CI.

### Inference-seed floor (BASE_s1 − BASE_s0)

* S2: floor = **0.0018 m** (block +0.0018 [-0.0019, +0.0055]); seed plan displacement mean 0.0983 m, p95 0.2888 m
* S6: floor = **0.0237 m** (block +0.0237 [-0.0100, +0.0577]); seed plan displacement mean 0.6664 m, p95 2.7743 m

### Separated components at BOTH seeds

* AGT_ZERO S2: worse ['ADE.ade_m', 'ADE.fde_m', 'lateral.LAT_cross_mae_m', 'lateral.LAT_heading_mae_deg', 'lateral.LAT_yaw_rate_mae_masked_radps', 'lateral.LAT_yaw_rate_mae_radps', 'longitudinal.LON_accel_mae_mps2', 'longitudinal.LON_along_mae_m', 'longitudinal.LON_speed_mae_mps', 'tactical.TAC_traj_lat_correct', 'tactical.TAC_traj_lon_correct'] · better []
* AGT_ZERO S6: worse ['ADE.ade_m', 'ADE.fde_m', 'lateral.LAT_curvature_mae_1pm', 'lateral.LAT_heading_mae_deg', 'longitudinal.LON_accel_mae_mps2', 'longitudinal.LON_along_mae_m', 'longitudinal.LON_speed_mae_mps'] · better []
* AGT_SHUF S2: worse ['ADE.ade_m', 'ADE.fde_m', 'lateral.LAT_cross_mae_m', 'lateral.LAT_heading_mae_deg', 'longitudinal.LON_accel_mae_mps2', 'longitudinal.LON_speed_mae_mps'] · better []
* AGT_SHUF S6: worse ['ADE.ade_m', 'ADE.fde_m', 'longitudinal.LON_accel_mae_mps2', 'longitudinal.LON_speed_mae_mps'] · better []
* BEV_ZERO S2: worse ['ADE.ade_m', 'ADE.fde_m', 'lateral.LAT_cross_mae_m', 'lateral.LAT_curvature_mae_1pm', 'lateral.LAT_heading_mae_deg', 'lateral.LAT_yaw_rate_mae_masked_radps', 'lateral.LAT_yaw_rate_mae_radps', 'longitudinal.LON_accel_mae_mps2', 'longitudinal.LON_along_mae_m', 'longitudinal.LON_speed_mae_mps', 'tactical.TAC_traj_lat_correct', 'tactical.TAC_traj_lon_correct'] · better []
* BEV_ZERO S6: worse ['ADE.ade_m', 'ADE.fde_m', 'lateral.LAT_cross_mae_m', 'lateral.LAT_curvature_mae_1pm', 'lateral.LAT_heading_mae_deg', 'lateral.LAT_yaw_rate_mae_masked_radps', 'lateral.LAT_yaw_rate_mae_radps', 'longitudinal.LON_accel_mae_mps2', 'longitudinal.LON_along_mae_m', 'longitudinal.LON_speed_mae_mps', 'tactical.TAC_traj_lat_correct', 'tactical.TAC_traj_lon_correct'] · better []
* BEV_SHUF S2: worse ['ADE.ade_m', 'ADE.fde_m', 'lateral.LAT_cross_mae_m', 'lateral.LAT_curvature_mae_1pm', 'lateral.LAT_heading_mae_deg', 'lateral.LAT_yaw_rate_mae_masked_radps', 'lateral.LAT_yaw_rate_mae_radps', 'longitudinal.LON_accel_mae_mps2', 'longitudinal.LON_along_mae_m', 'longitudinal.LON_speed_mae_mps', 'tactical.TAC_traj_lat_correct', 'tactical.TAC_traj_lon_correct'] · better []
* BEV_SHUF S6: worse ['ADE.ade_m', 'ADE.fde_m', 'lateral.LAT_cross_mae_m', 'lateral.LAT_curvature_mae_1pm', 'lateral.LAT_heading_mae_deg', 'lateral.LAT_yaw_rate_mae_masked_radps', 'lateral.LAT_yaw_rate_mae_radps', 'longitudinal.LON_accel_mae_mps2', 'longitudinal.LON_along_mae_m', 'longitudinal.LON_speed_mae_mps', 'tactical.TAC_traj_lat_correct', 'tactical.TAC_traj_lon_correct'] · better []
* BOTH_ZERO S2: worse ['ADE.ade_m', 'ADE.fde_m', 'lateral.LAT_cross_mae_m', 'lateral.LAT_curvature_mae_1pm', 'lateral.LAT_heading_mae_deg', 'lateral.LAT_yaw_rate_mae_masked_radps', 'lateral.LAT_yaw_rate_mae_radps', 'longitudinal.LON_accel_mae_mps2', 'longitudinal.LON_along_mae_m', 'longitudinal.LON_speed_mae_mps', 'tactical.TAC_traj_lat_correct', 'tactical.TAC_traj_lon_correct'] · better []
* BOTH_ZERO S6: worse ['ADE.ade_m', 'ADE.fde_m', 'lateral.LAT_cross_mae_m', 'lateral.LAT_curvature_mae_1pm', 'lateral.LAT_heading_mae_deg', 'lateral.LAT_yaw_rate_mae_masked_radps', 'lateral.LAT_yaw_rate_mae_radps', 'longitudinal.LON_accel_mae_mps2', 'longitudinal.LON_along_mae_m', 'longitudinal.LON_speed_mae_mps', 'tactical.TAC_traj_lat_correct', 'tactical.TAC_traj_lon_correct'] · better []

## All four families, per seed (arm − BASE)

### s0 — n_windows S2 4754, S6 3668, episodes 139

**S2**

| family.metric | BASE mean | AGT_ZERO | AGT_SHUF | BEV_ZERO | BEV_SHUF | BOTH_ZERO |
|---|---|---|---|---|---|---|
| ADE.ade_m | 0.29727 | +0.0153 [+0.0091, +0.0221] **sep** | +0.0092 [+0.0012, +0.0170] **sep** | +0.2060 [+0.1813, +0.2329] **sep** | +0.2083 [+0.1730, +0.2449] **sep** | +0.2113 [+0.1857, +0.2384] **sep** |
| ADE.fde_m | 0.644666 | +0.0342 [+0.0199, +0.0498] **sep** | +0.0213 [+0.0029, +0.0392] **sep** | +0.3782 [+0.3241, +0.4355] **sep** | +0.4386 [+0.3657, +0.5158] **sep** | +0.3833 [+0.3268, +0.4419] **sep** |
| longitudinal.LON_speed_mae_mps | 0.286846 | +0.0172 [+0.0103, +0.0250] **sep** | +0.0089 [+0.0002, +0.0175] **sep** | +0.1363 [+0.1150, +0.1596] **sep** | +0.0203 [+0.0113, +0.0298] **sep** | +0.1463 [+0.1230, +0.1718] **sep** |
| longitudinal.LON_along_mae_m | 0.251182 | +0.0152 [+0.0090, +0.0219] **sep** | +0.0072 [-0.0005, +0.0147] | +0.1360 [+0.1134, +0.1592] **sep** | +0.0197 [+0.0114, +0.0283] **sep** | +0.1454 [+0.1212, +0.1700] **sep** |
| longitudinal.LON_accel_mae_mps2 | 0.400485 | +0.0193 [+0.0109, +0.0289] **sep** | +0.0115 [+0.0017, +0.0219] **sep** | +0.1312 [+0.1088, +0.1557] **sep** | +0.0155 [+0.0055, +0.0259] **sep** | +0.1488 [+0.1237, +0.1768] **sep** |
| lateral.LAT_cross_mae_m | 0.101592 | +0.0037 [+0.0014, +0.0060] **sep** | +0.0050 [+0.0024, +0.0076] **sep** | +0.1252 [+0.1082, +0.1433] **sep** | +0.2320 [+0.1940, +0.2727] **sep** | +0.1209 [+0.1049, +0.1385] **sep** |
| lateral.LAT_heading_mae_deg | 1.425503 | +0.0321 [+0.0142, +0.0504] **sep** | +0.0262 [+0.0015, +0.0473] **sep** | +0.9434 [+0.8169, +1.0929] **sep** | +1.3776 [+1.1657, +1.5900] **sep** | +0.9241 [+0.8036, +1.0669] **sep** |
| lateral.LAT_yaw_rate_mae_radps | 0.018216 | +0.0005 [+0.0002, +0.0009] **sep** | +0.0004 [-0.0001, +0.0007] | +0.0190 [+0.0167, +0.0218] **sep** | +0.0228 [+0.0193, +0.0263] **sep** | +0.0186 [+0.0164, +0.0213] **sep** |
| lateral.LAT_curvature_mae_1pm | 0.00364 | +0.0001 [-0.0000, +0.0002] | +0.0001 [-0.0000, +0.0002] | +0.0040 [+0.0031, +0.0049] **sep** | +0.0029 [+0.0024, +0.0035] **sep** | +0.0040 [+0.0031, +0.0050] **sep** |
| lateral.LAT_yaw_rate_mae_masked_radps | 0.018216 | +0.0005 [+0.0002, +0.0009] **sep** | +0.0004 [-0.0001, +0.0007] | +0.0190 [+0.0167, +0.0218] **sep** | +0.0228 [+0.0193, +0.0263] **sep** | +0.0186 [+0.0164, +0.0213] **sep** |
| tactical.TAC_traj_lat_correct | 0.961927 | -0.0038 [-0.0068, -0.0006] **sep** | -0.0015 [-0.0050, +0.0025] | -0.0400 [-0.0535, -0.0276] **sep** | -0.0797 [-0.1022, -0.0578] **sep** | -0.0400 [-0.0522, -0.0290] **sep** |
| tactical.TAC_traj_lon_correct | 0.816155 | -0.0082 [-0.0152, -0.0013] **sep** | -0.0046 [-0.0120, +0.0023] | -0.0785 [-0.1010, -0.0571] **sep** | -0.0158 [-0.0269, -0.0046] **sep** | -0.0806 [-0.1041, -0.0593] **sep** |

**S6**

| family.metric | BASE mean | AGT_ZERO | AGT_SHUF | BEV_ZERO | BEV_SHUF | BOTH_ZERO |
|---|---|---|---|---|---|---|
| ADE.ade_m | 2.775706 | +0.1139 [+0.0443, +0.1921] **sep** | +0.0937 [+0.0269, +0.1706] **sep** | +1.1505 [+0.9510, +1.3660] **sep** | +1.2821 [+1.0458, +1.5314] **sep** | +1.2030 [+1.0002, +1.4277] **sep** |
| ADE.fde_m | 6.823358 | +0.2736 [+0.0970, +0.4581] **sep** | +0.2458 [+0.0792, +0.4370] **sep** | +2.5030 [+2.0399, +2.9998] **sep** | +2.6128 [+2.0938, +3.1462] **sep** | +2.6518 [+2.1659, +3.1740] **sep** |
| longitudinal.LON_speed_mae_mps | 1.030058 | +0.0483 [+0.0191, +0.0801] **sep** | +0.0387 [+0.0107, +0.0696] **sep** | +0.2344 [+0.1562, +0.3165] **sep** | +0.0569 [+0.0240, +0.0900] **sep** | +0.2650 [+0.1855, +0.3528] **sep** |
| longitudinal.LON_along_mae_m | 2.425353 | +0.1166 [+0.0454, +0.1975] **sep** | +0.0817 [+0.0123, +0.1589] **sep** | +0.6068 [+0.4205, +0.8074] **sep** | +0.2157 [+0.1081, +0.3377] **sep** | +0.6456 [+0.4485, +0.8602] **sep** |
| longitudinal.LON_accel_mae_mps2 | 0.468092 | +0.0178 [+0.0070, +0.0292] **sep** | +0.0154 [+0.0044, +0.0270] **sep** | +0.0809 [+0.0568, +0.1067] **sep** | +0.0186 [+0.0071, +0.0302] **sep** | +0.0938 [+0.0681, +0.1220] **sep** |
| lateral.LAT_cross_mae_m | 0.843858 | +0.0108 [-0.0075, +0.0274] | +0.0273 [+0.0090, +0.0485] **sep** | +0.9736 [+0.8501, +1.1074] **sep** | +1.4621 [+1.2107, +1.7286] **sep** | +0.9859 [+0.8635, +1.1181] **sep** |
| lateral.LAT_heading_mae_deg | 3.150987 | +0.0731 [+0.0031, +0.1474] **sep** | +0.0772 [+0.0094, +0.1468] **sep** | +2.7527 [+2.3249, +3.2255] **sep** | +2.9658 [+2.4736, +3.4621] **sep** | +2.7817 [+2.4053, +3.1923] **sep** |
| lateral.LAT_yaw_rate_mae_radps | 0.027973 | +0.0003 [-0.0001, +0.0007] | +0.0004 [-0.0002, +0.0009] | +0.0171 [+0.0145, +0.0200] **sep** | +0.0133 [+0.0107, +0.0160] **sep** | +0.0171 [+0.0144, +0.0202] **sep** |
| lateral.LAT_curvature_mae_1pm | 0.005111 | +0.0002 [+0.0000, +0.0003] **sep** | +0.0001 [-0.0000, +0.0003] | +0.0030 [+0.0023, +0.0038] **sep** | +0.0019 [+0.0014, +0.0024] **sep** | +0.0032 [+0.0023, +0.0040] **sep** |
| lateral.LAT_yaw_rate_mae_masked_radps | 0.027973 | +0.0003 [-0.0001, +0.0007] | +0.0004 [-0.0002, +0.0009] | +0.0171 [+0.0145, +0.0200] **sep** | +0.0133 [+0.0107, +0.0160] **sep** | +0.0171 [+0.0144, +0.0202] **sep** |
| tactical.TAC_traj_lat_correct | 0.882497 | +0.0000 [-0.0055, +0.0054] | -0.0003 [-0.0063, +0.0052] | -0.1050 [-0.1345, -0.0791] **sep** | -0.1213 [-0.1534, -0.0907] **sep** | -0.1071 [-0.1357, -0.0828] **sep** |
| tactical.TAC_traj_lon_correct | 0.602781 | +0.0076 [-0.0008, +0.0160] | -0.0068 [-0.0164, +0.0027] | -0.0608 [-0.0897, -0.0310] **sep** | -0.0314 [-0.0486, -0.0125] **sep** | -0.0573 [-0.0869, -0.0288] **sep** |

Plan displacement vs BASE (mean / p95, m) · sel_idx changed · tac_decoder_v6 argmax changed:

* S2|AGT_ZERO: 0.0712 / 0.3333
* S2|AGT_SHUF: 0.0659 / 0.3064
* S2|BEV_ZERO: 0.4267 / 1.1503
* S2|BEV_SHUF: 0.3797 / 1.3338
* S2|BOTH_ZERO: 0.4376 / 1.1795
* S6|AGT_ZERO: 0.6067 / 3.0775
* S6|AGT_SHUF: 0.5634 / 3.096
* S6|BEV_ZERO: 3.1203 / 8.3551
* S6|BEV_SHUF: 2.8336 / 9.194
* S6|BOTH_ZERO: 3.198 / 8.6348
* sel_idx changed: {'AGT_ZERO': 0.2408, 'AGT_SHUF': 0.2209, 'BEV_ZERO': 0.8591, 'BEV_SHUF': 0.4987, 'BOTH_ZERO': 0.862}
* tacv6: {'AGT_ZERO|nav_true.tacv6_lat_logits': 0.0368, 'AGT_ZERO|nav_true.tacv6_lon_logits': 0.6418, 'AGT_ZERO|nav_true.tacv6_goal_logits|max_abs_logit_diff': 9.604748725891113, 'AGT_SHUF|nav_true.tacv6_lat_logits': 0.0442, 'AGT_SHUF|nav_true.tacv6_lon_logits': 0.5429, 'AGT_SHUF|nav_true.tacv6_goal_logits|max_abs_logit_diff': 11.609400749206543, 'BEV_ZERO|nav_true.tacv6_lat_logits': 0.1024, 'BEV_ZERO|nav_true.tacv6_lon_logits': 0.4777, 'BEV_ZERO|nav_true.tacv6_goal_logits|max_abs_logit_diff': 10.582782745361328, 'BEV_SHUF|nav_true.tacv6_lat_logits': 0.0862, 'BEV_SHUF|nav_true.tacv6_lon_logits': 0.4323, 'BEV_SHUF|nav_true.tacv6_goal_logits|max_abs_logit_diff': 10.654619216918945, 'BOTH_ZERO|nav_true.tacv6_lat_logits': 0.1081, 'BOTH_ZERO|nav_true.tacv6_lon_logits': 0.6904, 'BOTH_ZERO|nav_true.tacv6_goal_logits|max_abs_logit_diff': 11.606718063354492}

### s1 — n_windows S2 4754, S6 3668, episodes 139

**S2**

| family.metric | BASE mean | AGT_ZERO | AGT_SHUF | BEV_ZERO | BEV_SHUF | BOTH_ZERO |
|---|---|---|---|---|---|---|
| ADE.ade_m | 0.299033 | +0.0157 [+0.0095, +0.0225] **sep** | +0.0105 [+0.0031, +0.0186] **sep** | +0.2018 [+0.1768, +0.2284] **sep** | +0.2065 [+0.1719, +0.2431] **sep** | +0.2129 [+0.1869, +0.2411] **sep** |
| ADE.fde_m | 0.647401 | +0.0362 [+0.0215, +0.0516] **sep** | +0.0245 [+0.0071, +0.0443] **sep** | +0.3734 [+0.3192, +0.4302] **sep** | +0.4355 [+0.3620, +0.5116] **sep** | +0.3855 [+0.3294, +0.4452] **sep** |
| longitudinal.LON_speed_mae_mps | 0.288041 | +0.0167 [+0.0095, +0.0241] **sep** | +0.0100 [+0.0018, +0.0197] **sep** | +0.1351 [+0.1141, +0.1576] **sep** | +0.0192 [+0.0111, +0.0283] **sep** | +0.1496 [+0.1266, +0.1744] **sep** |
| longitudinal.LON_along_mae_m | 0.252895 | +0.0146 [+0.0082, +0.0214] **sep** | +0.0079 [+0.0005, +0.0161] **sep** | +0.1321 [+0.1109, +0.1547] **sep** | +0.0188 [+0.0109, +0.0272] **sep** | +0.1472 [+0.1229, +0.1725] **sep** |
| longitudinal.LON_accel_mae_mps2 | 0.400455 | +0.0191 [+0.0105, +0.0282] **sep** | +0.0123 [+0.0023, +0.0239] **sep** | +0.1329 [+0.1107, +0.1567] **sep** | +0.0157 [+0.0063, +0.0260] **sep** | +0.1510 [+0.1271, +0.1785] **sep** |
| lateral.LAT_cross_mae_m | 0.102003 | +0.0042 [+0.0020, +0.0065] **sep** | +0.0056 [+0.0028, +0.0086] **sep** | +0.1228 [+0.1056, +0.1419] **sep** | +0.2306 [+0.1934, +0.2711] **sep** | +0.1199 [+0.1039, +0.1374] **sep** |
| lateral.LAT_heading_mae_deg | 1.428462 | +0.0447 [+0.0263, +0.0654] **sep** | +0.0371 [+0.0153, +0.0608] **sep** | +0.9278 [+0.8036, +1.0737] **sep** | +1.3597 [+1.1501, +1.5681] **sep** | +0.9136 [+0.7948, +1.0488] **sep** |
| lateral.LAT_yaw_rate_mae_radps | 0.018082 | +0.0008 [+0.0005, +0.0012] **sep** | +0.0006 [+0.0001, +0.0010] **sep** | +0.0190 [+0.0167, +0.0217] **sep** | +0.0226 [+0.0192, +0.0260] **sep** | +0.0188 [+0.0165, +0.0212] **sep** |
| lateral.LAT_curvature_mae_1pm | 0.003658 | +0.0002 [+0.0001, +0.0002] **sep** | +0.0001 [+0.0000, +0.0002] **sep** | +0.0039 [+0.0031, +0.0049] **sep** | +0.0029 [+0.0024, +0.0034] **sep** | +0.0040 [+0.0032, +0.0050] **sep** |
| lateral.LAT_yaw_rate_mae_masked_radps | 0.018082 | +0.0008 [+0.0005, +0.0012] **sep** | +0.0006 [+0.0001, +0.0010] **sep** | +0.0190 [+0.0167, +0.0217] **sep** | +0.0226 [+0.0192, +0.0260] **sep** | +0.0188 [+0.0165, +0.0212] **sep** |
| tactical.TAC_traj_lat_correct | 0.962768 | -0.0057 [-0.0090, -0.0025] **sep** | -0.0023 [-0.0055, +0.0008] | -0.0395 [-0.0537, -0.0272] **sep** | -0.0822 [-0.1047, -0.0611] **sep** | -0.0416 [-0.0551, -0.0295] **sep** |
| tactical.TAC_traj_lon_correct | 0.815734 | -0.0088 [-0.0151, -0.0025] **sep** | -0.0067 [-0.0158, +0.0008] | -0.0831 [-0.1056, -0.0628] **sep** | -0.0111 [-0.0213, -0.0004] **sep** | -0.0833 [-0.1057, -0.0627] **sep** |

**S6**

| family.metric | BASE mean | AGT_ZERO | AGT_SHUF | BEV_ZERO | BEV_SHUF | BOTH_ZERO |
|---|---|---|---|---|---|---|
| ADE.ade_m | 2.799412 | +0.0940 [+0.0338, +0.1579] **sep** | +0.0794 [+0.0057, +0.1692] **sep** | +1.1282 [+0.9333, +1.3484] **sep** | +1.2369 [+1.0083, +1.4717] **sep** | +1.1993 [+0.9978, +1.4260] **sep** |
| ADE.fde_m | 6.888274 | +0.2098 [+0.0566, +0.3663] **sep** | +0.2085 [+0.0205, +0.4374] **sep** | +2.4445 [+1.9897, +2.9593] **sep** | +2.5031 [+2.0204, +3.0165] **sep** | +2.6282 [+2.1407, +3.1535] **sep** |
| longitudinal.LON_speed_mae_mps | 1.042302 | +0.0302 [+0.0051, +0.0565] **sep** | +0.0371 [+0.0056, +0.0756] **sep** | +0.2200 [+0.1425, +0.3034] **sep** | +0.0432 [+0.0141, +0.0734] **sep** | +0.2578 [+0.1754, +0.3468] **sep** |
| longitudinal.LON_along_mae_m | 2.451119 | +0.0837 [+0.0217, +0.1499] **sep** | +0.0685 [-0.0067, +0.1609] | +0.5665 [+0.3869, +0.7628] **sep** | +0.1942 [+0.0940, +0.3095] **sep** | +0.6330 [+0.4370, +0.8502] **sep** |
| longitudinal.LON_accel_mae_mps2 | 0.470881 | +0.0116 [+0.0021, +0.0214] **sep** | +0.0167 [+0.0046, +0.0312] **sep** | +0.0774 [+0.0532, +0.1037] **sep** | +0.0158 [+0.0053, +0.0262] **sep** | +0.0938 [+0.0679, +0.1219] **sep** |
| lateral.LAT_cross_mae_m | 0.843799 | +0.0250 [+0.0053, +0.0440] **sep** | +0.0215 [-0.0007, +0.0426] | +0.9843 [+0.8550, +1.1346] **sep** | +1.4319 [+1.1873, +1.6903] **sep** | +1.0000 [+0.8719, +1.1389] **sep** |
| lateral.LAT_heading_mae_deg | 3.143155 | +0.1115 [+0.0388, +0.1879] **sep** | +0.0421 [-0.0252, +0.1133] | +2.8291 [+2.3678, +3.3099] **sep** | +2.9376 [+2.4693, +3.4145] **sep** | +2.8215 [+2.3890, +3.2878] **sep** |
| lateral.LAT_yaw_rate_mae_radps | 0.027906 | +0.0006 [+0.0001, +0.0012] **sep** | +0.0003 [-0.0003, +0.0010] | +0.0175 [+0.0147, +0.0208] **sep** | +0.0128 [+0.0102, +0.0153] **sep** | +0.0173 [+0.0146, +0.0204] **sep** |
| lateral.LAT_curvature_mae_1pm | 0.00512 | +0.0001 [+0.0001, +0.0002] **sep** | +0.0001 [-0.0001, +0.0003] | +0.0031 [+0.0023, +0.0040] **sep** | +0.0018 [+0.0014, +0.0022] **sep** | +0.0032 [+0.0023, +0.0041] **sep** |
| lateral.LAT_yaw_rate_mae_masked_radps | 0.027906 | +0.0006 [+0.0001, +0.0012] **sep** | +0.0003 [-0.0003, +0.0010] | +0.0175 [+0.0147, +0.0208] **sep** | +0.0128 [+0.0102, +0.0153] **sep** | +0.0173 [+0.0146, +0.0204] **sep** |
| tactical.TAC_traj_lat_correct | 0.881952 | -0.0052 [-0.0114, +0.0003] | +0.0005 [-0.0063, +0.0071] | -0.1014 [-0.1316, -0.0755] **sep** | -0.1186 [-0.1492, -0.0903] **sep** | -0.1031 [-0.1320, -0.0780] **sep** |
| tactical.TAC_traj_lon_correct | 0.597601 | +0.0104 [+0.0005, +0.0204] **sep** | -0.0101 [-0.0218, +0.0016] | -0.0564 [-0.0849, -0.0286] **sep** | -0.0289 [-0.0469, -0.0096] **sep** | -0.0491 [-0.0793, -0.0199] **sep** |

Plan displacement vs BASE (mean / p95, m) · sel_idx changed · tac_decoder_v6 argmax changed:

* S2|AGT_ZERO: 0.0687 / 0.3163
* S2|AGT_SHUF: 0.0682 / 0.3213
* S2|BEV_ZERO: 0.4264 / 1.1607
* S2|BEV_SHUF: 0.3803 / 1.3383
* S2|BOTH_ZERO: 0.4414 / 1.1944
* S6|AGT_ZERO: 0.5822 / 3.165
* S6|AGT_SHUF: 0.5808 / 3.1745
* S6|BEV_ZERO: 3.1387 / 8.1887
* S6|BEV_SHUF: 2.7731 / 9.2456
* S6|BOTH_ZERO: 3.2446 / 8.5995
* sel_idx changed: {'AGT_ZERO': 0.2343, 'AGT_SHUF': 0.2276, 'BEV_ZERO': 0.8576, 'BEV_SHUF': 0.4979, 'BOTH_ZERO': 0.8683}
* tacv6: {'AGT_ZERO|nav_true.tacv6_lat_logits': 0.0368, 'AGT_ZERO|nav_true.tacv6_lon_logits': 0.6418, 'AGT_ZERO|nav_true.tacv6_goal_logits|max_abs_logit_diff': 9.604748725891113, 'AGT_SHUF|nav_true.tacv6_lat_logits': 0.0442, 'AGT_SHUF|nav_true.tacv6_lon_logits': 0.5429, 'AGT_SHUF|nav_true.tacv6_goal_logits|max_abs_logit_diff': 11.609400749206543, 'BEV_ZERO|nav_true.tacv6_lat_logits': 0.1024, 'BEV_ZERO|nav_true.tacv6_lon_logits': 0.4777, 'BEV_ZERO|nav_true.tacv6_goal_logits|max_abs_logit_diff': 10.582782745361328, 'BEV_SHUF|nav_true.tacv6_lat_logits': 0.0862, 'BEV_SHUF|nav_true.tacv6_lon_logits': 0.4323, 'BEV_SHUF|nav_true.tacv6_goal_logits|max_abs_logit_diff': 10.654619216918945, 'BOTH_ZERO|nav_true.tacv6_lat_logits': 0.1081, 'BOTH_ZERO|nav_true.tacv6_lon_logits': 0.6904, 'BOTH_ZERO|nav_true.tacv6_goal_logits|max_abs_logit_diff': 11.606718063354492}

**Strategic family:** NOT COMPUTABLE by this instrument (n = 0) — _paired_families carries no strategic component; the nav-obedience block exists only in the full-arm-set acceptance stage, not on an os-only roll.

## Controls

* VOID gates all pass: **True**; failed: []
* C2 (BASE_s1 vs banked dev-box battery s1, ≤ 1e-5): **C2-FAIL (cross-device reproduction not established)**
* s0: C2 {"binding": false, "n_windows": 4754, "max_abs_m": 3.7815208435058594, "frac_windows_bitexact": 0.0284, "frac_windows_le_1e-5": 0.0284, "pass": false, "g_equal": false} · C3 {"status": "ABSENT", "fan_dump": "/c/Users/Admin/refcv8_pf/perc2plan/thor_ref/fan_s0"} · S6 GT independent max |Δ| 0.0
  * C4/C5/C6: C5_aligned_BASE=True, C5_aligned_AGT_ZERO=True, C5_aligned_AGT_SHUF=True, C5_aligned_BEV_ZERO=True, C5_aligned_BEV_SHUF=True, C5_aligned_BOTH_ZERO=True, C4_BASE=True, C4_BASE_donor_same_clip=0, C4_AGT_ZERO=True, C4_AGT_ZERO_donor_same_clip=0, C4_AGT_SHUF=True, C4_AGT_SHUF_donor_same_clip=0, C4_AGT_SHUF_donor_same_clip_zero=True, C4_BEV_ZERO=True, C4_BEV_ZERO_donor_same_clip=0, C4_BEV_SHUF=True, C4_BEV_SHUF_donor_same_clip=0, C4_BEV_SHUF_donor_same_clip_zero=True, C4_BOTH_ZERO=True, C4_BOTH_ZERO_donor_same_clip=0, C6_self_pair_zero_S2=True, C6_self_pair_zero_S6=True
* s1: C2 {"binding": true, "n_windows": 4754, "max_abs_m": 2.6255531311035156, "frac_windows_bitexact": 0.0299, "frac_windows_le_1e-5": 0.0307, "pass": false, "g_equal": false} · C3 {"aligned": true, "os_bitexact": true, "plan_full_bitexact": true, "max_abs_m": 0.0} · S6 GT independent max |Δ| 0.0
  * C4/C5/C6: C5_aligned_BASE=True, C5_aligned_AGT_ZERO=True, C5_aligned_AGT_SHUF=True, C5_aligned_BEV_ZERO=True, C5_aligned_BEV_SHUF=True, C5_aligned_BOTH_ZERO=True, C4_BASE=True, C4_BASE_donor_same_clip=0, C4_AGT_ZERO=True, C4_AGT_ZERO_donor_same_clip=0, C4_AGT_SHUF=True, C4_AGT_SHUF_donor_same_clip=0, C4_AGT_SHUF_donor_same_clip_zero=True, C4_BEV_ZERO=True, C4_BEV_ZERO_donor_same_clip=0, C4_BEV_SHUF=True, C4_BEV_SHUF_donor_same_clip=0, C4_BEV_SHUF_donor_same_clip_zero=True, C4_BOTH_ZERO=True, C4_BOTH_ZERO_donor_same_clip=0, C6_self_pair_zero_S2=True, C6_self_pair_zero_S6=True

### CPU smoke (dev box, 6 windows — controls only, not a result)

* C1 as registered (bit-exact vs the unmodified child): **FAIL as registered (bit-exact); PASS at the brief's tolerance (max |d| <= 1e-5)** — max |Δ| os 4.84e-08 m, plan_full 3.81e-06 m, sel_idx identical True
* Attribution: the HOOKS are a bit-exact identity at BASE (hooked == un-hooked wrapper, bitwise); the residual 3.8e-6 m is the WRAPPER PROCESS vs the directly-run child (same code, same seed, same device, both deterministic run-to-run) -- a CPU float-reduction difference from a different process history, not the intervention. Every arm and BASE go through the SAME wrapper, so arm - BASE isolates the hook mode.
* Zero is exactly zero at the hook: {'AGT_ZERO': [True, None], 'BEV_ZERO': [None, True], 'BOTH_ZERO': [True, True]}
