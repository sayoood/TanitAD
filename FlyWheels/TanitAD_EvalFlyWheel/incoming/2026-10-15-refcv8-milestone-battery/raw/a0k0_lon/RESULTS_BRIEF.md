# RESULTS_BRIEF -- longitudinal-only residual scaling (all cells T1 self-action OPEN LOOP on logged eval139 frames; paired episode-cluster bootstrap B 2000, 'another draw of EPISODES' only, single training seed; n windows / episodes in each header)

controls ALL_PASS = True; PREREG_LON sha256 cdeb894ad912; script sha256 9ec2af28f3ae

## step30000ema_unc / S2 / compliant -- n 4159 windows / 138 episodes

| arm | - ha0_ext (LE-E1) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RS_LON_XF | -0.0258 [-0.0364, -0.0144] **SEP** | -0.0254 [-0.0358, -0.0140] **SEP** | -0.0309 [-0.0421, -0.0196] **SEP** | -0.0305 [-0.0415, -0.0189] **SEP** | -0.0173 [-0.0232, -0.0118] **SEP** | -0.0178 [-0.0237, -0.0125] **SEP** |
| GATED_RS_LON_XF | -0.0298 [-0.0405, -0.0186] **SEP** | -0.0291 [-0.0390, -0.0181] **SEP** | -0.0349 [-0.0460, -0.0237] **SEP** | -0.0342 [-0.0448, -0.0230] **SEP** | -0.0213 [-0.0281, -0.0152] **SEP** | -0.0215 [-0.0283, -0.0153] **SEP** |
| RS_LAT1_XF | -0.0258 [-0.0364, -0.0144] **SEP** | -0.0254 [-0.0358, -0.0140] **SEP** | -0.0309 [-0.0421, -0.0196] **SEP** | -0.0305 [-0.0415, -0.0189] **SEP** | -0.0173 [-0.0232, -0.0118] **SEP** | -0.0178 [-0.0237, -0.0125] **SEP** |
| RS_XF | -0.0428 [-0.0538, -0.0321] **SEP** | -0.0427 [-0.0534, -0.0321] **SEP** | -0.0479 [-0.0596, -0.0367] **SEP** | -0.0479 [-0.0593, -0.0368] **SEP** | -0.0343 [-0.0419, -0.0274] **SEP** | -0.0351 [-0.0425, -0.0284] **SEP** |

* RS_LON_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* GATED_RS_LON_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* RS_LAT1_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* share of RS_XF - os (ADE, s0 point) retained by RS_LON: 0.5043731778425656

## step30000ema_unc / S6 / compliant -- n 3251 windows / 138 episodes

| arm | - ha0_ext (LE-E3) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RS_LON_XF | -0.9017 [-1.1719, -0.6542] **SEP** | -0.9008 [-1.1651, -0.6543] **SEP** | -0.9305 [-1.1994, -0.6785] **SEP** | -0.9295 [-1.1963, -0.6774] **SEP** | -0.1482 [-0.2309, -0.0647] **SEP** | -0.1578 [-0.2399, -0.0726] **SEP** |
| GATED_RS_LON_XF | -0.9381 [-1.2001, -0.7000] **SEP** | -0.9387 [-1.2023, -0.7011] **SEP** | -0.9669 [-1.2368, -0.7225] **SEP** | -0.9675 [-1.2392, -0.7224] **SEP** | -0.1846 [-0.2820, -0.0880] **SEP** | -0.1958 [-0.3006, -0.0982] **SEP** |
| RS_LAT1_XF | -0.9017 [-1.1719, -0.6542] **SEP** | -0.9008 [-1.1651, -0.6543] **SEP** | -0.9305 [-1.1994, -0.6785] **SEP** | -0.9295 [-1.1963, -0.6774] **SEP** | -0.1482 [-0.2309, -0.0647] **SEP** | -0.1578 [-0.2399, -0.0726] **SEP** |
| RS_XF | -0.9162 [-1.1794, -0.6723] **SEP** | -0.9161 [-1.1758, -0.6742] **SEP** | -0.9450 [-1.2100, -0.6967] **SEP** | -0.9449 [-1.2098, -0.6947] **SEP** | -0.1627 [-0.2464, -0.0763] **SEP** | -0.1732 [-0.2554, -0.0869] **SEP** |

S6 LATERAL rows (step30000ema_unc / compliant) -- arm - os and arm - refcv7-50,400, s0 | s1:

| row | RS_LON - os | GATED_RS_LON - os | RS_LAT1 - os | RS_LON - refcv7 | GATED_RS_LON - refcv7 |
|---|---|---|---|---|---|
| LAT_heading_mae_deg | +0.2346 [+0.1220, +0.3707] **SEP** ; +0.2326 [+0.1218, +0.3682] **SEP** | +0.1961 [+0.0942, +0.3123] **SEP** ; +0.1562 [+0.0677, +0.2576] **SEP** | +0.2346 [+0.1220, +0.3707] **SEP** ; +0.2326 [+0.1218, +0.3682] **SEP** | +0.3621 [+0.1416, +0.6059] **SEP** ; +0.4188 [+0.1789, +0.6863] **SEP** | +0.3187 [+0.1064, +0.5424] **SEP** ; +0.3788 [+0.1352, +0.6357] **SEP** |
| LAT_curvature_mae_1pm_valid | +0.0007 [+0.0003, +0.0012] **SEP** ; +0.0007 [+0.0003, +0.0013] **SEP** | +0.0006 [+0.0002, +0.0011] **SEP** ; +0.0007 [+0.0002, +0.0013] **SEP** | +0.0007 [+0.0003, +0.0012] **SEP** ; +0.0007 [+0.0003, +0.0013] **SEP** | +0.0008 [+0.0003, +0.0013] **SEP** ; +0.0009 [+0.0004, +0.0015] **SEP** | +0.0006 [+0.0002, +0.0011] **SEP** ; +0.0009 [+0.0004, +0.0015] **SEP** |
| LAT_yaw_rate_mae_radps_valid | +0.0021 [+0.0009, +0.0035] **SEP** ; +0.0020 [+0.0009, +0.0035] **SEP** | +0.0016 [+0.0006, +0.0027] **SEP** ; +0.0015 [+0.0005, +0.0026] **SEP** | +0.0021 [+0.0009, +0.0035] **SEP** ; +0.0020 [+0.0009, +0.0035] **SEP** | +0.0023 [+0.0004, +0.0043] **SEP** ; +0.0027 [+0.0009, +0.0047] **SEP** | +0.0018 [+0.0002, +0.0035] **SEP** ; +0.0022 [+0.0005, +0.0039] **SEP** |
| LAT_cross_mae_m | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0222 [-0.0284, +0.0712] ; +0.0252 [-0.0296, +0.0802] | +0.0222 [-0.0284, +0.0712] ; +0.0252 [-0.0296, +0.0802] |
| LAT_cross_abs_final_m | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0737 [-0.0627, +0.2105] ; +0.0769 [-0.0647, +0.2223] | +0.0737 [-0.0627, +0.2105] ; +0.0769 [-0.0647, +0.2223] |

* RS_LON_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* GATED_RS_LON_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* RS_LAT1_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* share of RS_XF - os (ADE, s0 point) retained by RS_LON: 0.910878918254456

## step30000ema_unc / S2 / all -- n 4754 windows / 139 episodes

| arm | - ha0_ext (LE-E1) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RS_LON_XF | -0.0237 [-0.0340, -0.0134] **SEP** | -0.0225 [-0.0329, -0.0124] **SEP** | -0.0292 [-0.0399, -0.0184] **SEP** | -0.0280 [-0.0389, -0.0174] **SEP** | -0.0229 [-0.0295, -0.0165] **SEP** | -0.0239 [-0.0309, -0.0175] **SEP** |
| GATED_RS_LON_XF | -0.0285 [-0.0389, -0.0179] **SEP** | -0.0270 [-0.0374, -0.0165] **SEP** | -0.0340 [-0.0449, -0.0229] **SEP** | -0.0325 [-0.0436, -0.0217] **SEP** | -0.0277 [-0.0349, -0.0211] **SEP** | -0.0285 [-0.0356, -0.0219] **SEP** |
| RS_LAT1_XF | -0.0237 [-0.0340, -0.0134] **SEP** | -0.0225 [-0.0329, -0.0124] **SEP** | -0.0292 [-0.0399, -0.0184] **SEP** | -0.0280 [-0.0389, -0.0174] **SEP** | -0.0229 [-0.0295, -0.0165] **SEP** | -0.0239 [-0.0309, -0.0175] **SEP** |
| RS_XF | -0.0412 [-0.0511, -0.0317] **SEP** | -0.0401 [-0.0498, -0.0307] **SEP** | -0.0467 [-0.0573, -0.0362] **SEP** | -0.0457 [-0.0561, -0.0354] **SEP** | -0.0403 [-0.0484, -0.0319] **SEP** | -0.0416 [-0.0500, -0.0331] **SEP** |

* RS_LON_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* GATED_RS_LON_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* RS_LAT1_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* share of RS_XF - os (ADE, s0 point) retained by RS_LON: 0.5682382133995036

## step30000ema_unc / S6 / all -- n 3668 windows / 139 episodes

| arm | - ha0_ext (LE-E3) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RS_LON_XF | -0.9881 [-1.2546, -0.7365] **SEP** | -0.9751 [-1.2370, -0.7241] **SEP** | -1.0167 [-1.2943, -0.7572] **SEP** | -1.0036 [-1.2789, -0.7449] **SEP** | -0.1736 [-0.2544, -0.0901] **SEP** | -0.1849 [-0.2683, -0.1015] **SEP** |
| GATED_RS_LON_XF | -1.0286 [-1.3013, -0.7744] **SEP** | -1.0164 [-1.2828, -0.7702] **SEP** | -1.0571 [-1.3415, -0.7929] **SEP** | -1.0450 [-1.3279, -0.7881] **SEP** | -0.2140 [-0.3055, -0.1181] **SEP** | -0.2262 [-0.3222, -0.1263] **SEP** |
| RS_LAT1_XF | -0.9881 [-1.2546, -0.7365] **SEP** | -0.9751 [-1.2370, -0.7241] **SEP** | -1.0167 [-1.2943, -0.7572] **SEP** | -1.0036 [-1.2789, -0.7449] **SEP** | -0.1736 [-0.2544, -0.0901] **SEP** | -0.1849 [-0.2683, -0.1015] **SEP** |
| RS_XF | -1.0018 [-1.2678, -0.7533] **SEP** | -0.9895 [-1.2504, -0.7423] **SEP** | -1.0304 [-1.3075, -0.7709] **SEP** | -1.0181 [-1.2915, -0.7621] **SEP** | -0.1872 [-0.2689, -0.1026] **SEP** | -0.1994 [-0.2823, -0.1141] **SEP** |

S6 LATERAL rows (step30000ema_unc / all) -- arm - os and arm - refcv7-50,400, s0 | s1:

| row | RS_LON - os | GATED_RS_LON - os | RS_LAT1 - os | RS_LON - refcv7 | GATED_RS_LON - refcv7 |
|---|---|---|---|---|---|
| LAT_heading_mae_deg | +0.2624 [+0.1489, +0.3910] **SEP** ; +0.2518 [+0.1417, +0.3752] **SEP** | +0.1589 [+0.0757, +0.2507] **SEP** ; +0.1462 [+0.0681, +0.2336] **SEP** | +0.2624 [+0.1489, +0.3910] **SEP** ; +0.2518 [+0.1417, +0.3752] **SEP** | +0.3983 [+0.1967, +0.6157] **SEP** ; +0.4514 [+0.2337, +0.6942] **SEP** | +0.2916 [+0.1045, +0.4912] **SEP** ; +0.3771 [+0.1566, +0.6127] **SEP** |
| LAT_curvature_mae_1pm_valid | +0.0008 [+0.0004, +0.0013] **SEP** ; +0.0008 [+0.0004, +0.0014] **SEP** | +0.0006 [+0.0002, +0.0010] **SEP** ; +0.0007 [+0.0003, +0.0011] **SEP** | +0.0008 [+0.0004, +0.0013] **SEP** ; +0.0008 [+0.0004, +0.0014] **SEP** | +0.0009 [+0.0004, +0.0014] **SEP** ; +0.0011 [+0.0006, +0.0016] **SEP** | +0.0006 [+0.0002, +0.0011] **SEP** ; +0.0009 [+0.0004, +0.0014] **SEP** |
| LAT_yaw_rate_mae_radps_valid | +0.0024 [+0.0013, +0.0037] **SEP** ; +0.0023 [+0.0012, +0.0038] **SEP** | +0.0018 [+0.0008, +0.0029] **SEP** ; +0.0017 [+0.0007, +0.0029] **SEP** | +0.0024 [+0.0013, +0.0037] **SEP** ; +0.0023 [+0.0012, +0.0038] **SEP** | +0.0027 [+0.0011, +0.0046] **SEP** ; +0.0032 [+0.0016, +0.0050] **SEP** | +0.0021 [+0.0006, +0.0037] **SEP** ; +0.0025 [+0.0010, +0.0043] **SEP** |
| LAT_cross_mae_m | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0209 [-0.0251, +0.0685] ; +0.0261 [-0.0236, +0.0773] | +0.0209 [-0.0251, +0.0685] ; +0.0261 [-0.0236, +0.0773] |
| LAT_cross_abs_final_m | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0546 [-0.0678, +0.1827] ; +0.0662 [-0.0631, +0.2032] | +0.0546 [-0.0678, +0.1827] ; +0.0662 [-0.0631, +0.2032] |

* RS_LON_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* GATED_RS_LON_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* RS_LAT1_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* share of RS_XF - os (ADE, s0 point) retained by RS_LON: 0.9273504273504274

## step30000_unc / S2 / compliant -- n 4159 windows / 138 episodes

| arm | - ha0_ext (LE-E1) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RS_LON_XF | -0.0120 [-0.0221, -0.0019] **SEP** | -0.0115 [-0.0216, -0.0009] **SEP** | -0.0171 [-0.0275, -0.0067] **SEP** | -0.0166 [-0.0271, -0.0057] **SEP** | -0.0294 [-0.0388, -0.0212] **SEP** | -0.0275 [-0.0367, -0.0191] **SEP** |
| GATED_RS_LON_XF | -0.0162 [-0.0261, -0.0057] **SEP** | -0.0154 [-0.0253, -0.0044] **SEP** | -0.0213 [-0.0316, -0.0109] **SEP** | -0.0205 [-0.0308, -0.0096] **SEP** | -0.0336 [-0.0436, -0.0251] **SEP** | -0.0314 [-0.0412, -0.0225] **SEP** |
| RS_LAT1_XF | -0.0120 [-0.0221, -0.0019] **SEP** | -0.0115 [-0.0216, -0.0009] **SEP** | -0.0171 [-0.0275, -0.0067] **SEP** | -0.0166 [-0.0271, -0.0057] **SEP** | -0.0294 [-0.0388, -0.0212] **SEP** | -0.0275 [-0.0367, -0.0191] **SEP** |
| RS_XF | -0.0289 [-0.0382, -0.0199] **SEP** | -0.0286 [-0.0381, -0.0197] **SEP** | -0.0340 [-0.0441, -0.0244] **SEP** | -0.0338 [-0.0439, -0.0243] **SEP** | -0.0463 [-0.0567, -0.0368] **SEP** | -0.0446 [-0.0549, -0.0352] **SEP** |

* RS_LON_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* GATED_RS_LON_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* RS_LAT1_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* share of RS_XF - os (ADE, s0 point) retained by RS_LON: 0.6349892008639308

## step30000_unc / S6 / compliant -- n 3251 windows / 138 episodes

| arm | - ha0_ext (LE-E3) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RS_LON_XF | -0.8275 [-1.0752, -0.5940] **SEP** | -0.8293 [-1.0852, -0.5895] **SEP** | -0.8562 [-1.1121, -0.6195] **SEP** | -0.8581 [-1.1187, -0.6148] **SEP** | -0.2534 [-0.3635, -0.1403] **SEP** | -0.2443 [-0.3551, -0.1316] **SEP** |
| GATED_RS_LON_XF | -0.8828 [-1.1359, -0.6521] **SEP** | -0.8873 [-1.1453, -0.6478] **SEP** | -0.9116 [-1.1697, -0.6746] **SEP** | -0.9161 [-1.1806, -0.6717] **SEP** | -0.3088 [-0.4414, -0.1861] **SEP** | -0.3022 [-0.4353, -0.1784] **SEP** |
| RS_LAT1_XF | -0.8275 [-1.0752, -0.5940] **SEP** | -0.8293 [-1.0852, -0.5895] **SEP** | -0.8562 [-1.1121, -0.6195] **SEP** | -0.8581 [-1.1187, -0.6148] **SEP** | -0.2534 [-0.3635, -0.1403] **SEP** | -0.2443 [-0.3551, -0.1316] **SEP** |
| RS_XF | -0.8431 [-1.0855, -0.6161] **SEP** | -0.8444 [-1.0949, -0.6111] **SEP** | -0.8718 [-1.1246, -0.6370] **SEP** | -0.8732 [-1.1285, -0.6328] **SEP** | -0.2690 [-0.3804, -0.1547] **SEP** | -0.2594 [-0.3723, -0.1447] **SEP** |

S6 LATERAL rows (step30000_unc / compliant) -- arm - os and arm - refcv7-50,400, s0 | s1:

| row | RS_LON - os | GATED_RS_LON - os | RS_LAT1 - os | RS_LON - refcv7 | GATED_RS_LON - refcv7 |
|---|---|---|---|---|---|
| LAT_heading_mae_deg | +0.2955 [+0.1334, +0.4904] **SEP** ; +0.2690 [+0.1071, +0.4643] **SEP** | +0.2124 [+0.0438, +0.4545] **SEP** ; +0.1863 [+0.0255, +0.4233] **SEP** | +0.2955 [+0.1334, +0.4904] **SEP** ; +0.2690 [+0.1071, +0.4643] **SEP** | +0.6895 [+0.4230, +1.0156] **SEP** ; +0.7456 [+0.4326, +1.1231] **SEP** | +0.6421 [+0.3402, +1.0098] **SEP** ; +0.7002 [+0.3520, +1.1104] **SEP** |
| LAT_curvature_mae_1pm_valid | +0.0015 [+0.0007, +0.0024] **SEP** ; +0.0013 [+0.0006, +0.0023] **SEP** | +0.0007 [+0.0002, +0.0013] **SEP** ; +0.0006 [+0.0001, +0.0012] **SEP** | +0.0015 [+0.0007, +0.0024] **SEP** ; +0.0013 [+0.0006, +0.0023] **SEP** | +0.0020 [+0.0012, +0.0030] **SEP** ; +0.0020 [+0.0012, +0.0031] **SEP** | +0.0022 [+0.0008, +0.0048] **SEP** ; +0.0022 [+0.0008, +0.0047] **SEP** |
| LAT_yaw_rate_mae_radps_valid | +0.0031 [+0.0012, +0.0054] **SEP** ; +0.0028 [+0.0010, +0.0048] **SEP** | +0.0013 [+0.0002, +0.0027] **SEP** ; +0.0010 [-0.0000, +0.0022] | +0.0031 [+0.0012, +0.0054] **SEP** ; +0.0028 [+0.0010, +0.0048] **SEP** | +0.0056 [+0.0032, +0.0085] **SEP** ; +0.0051 [+0.0029, +0.0077] **SEP** | +0.0046 [+0.0023, +0.0074] **SEP** ; +0.0042 [+0.0019, +0.0069] **SEP** |
| LAT_cross_mae_m | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0785 [+0.0151, +0.1425] **SEP** ; +0.0657 [+0.0032, +0.1270] **SEP** | +0.0785 [+0.0151, +0.1425] **SEP** ; +0.0657 [+0.0032, +0.1270] **SEP** |
| LAT_cross_abs_final_m | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.2237 [+0.0606, +0.3896] **SEP** ; +0.1844 [+0.0221, +0.3471] **SEP** | +0.2237 [+0.0606, +0.3896] **SEP** ; +0.1844 [+0.0221, +0.3471] **SEP** |

* RS_LON_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* GATED_RS_LON_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* RS_LAT1_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* share of RS_XF - os (ADE, s0 point) retained by RS_LON: 0.9420074349442379

## step30000_unc / S2 / all -- n 4754 windows / 139 episodes

| arm | - ha0_ext (LE-E1) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RS_LON_XF | -0.0112 [-0.0209, -0.0017] **SEP** | -0.0096 [-0.0193, +0.0001] | -0.0167 [-0.0275, -0.0066] **SEP** | -0.0151 [-0.0259, -0.0046] **SEP** | -0.0354 [-0.0443, -0.0269] **SEP** | -0.0340 [-0.0430, -0.0254] **SEP** |
| GATED_RS_LON_XF | -0.0159 [-0.0257, -0.0061] **SEP** | -0.0142 [-0.0241, -0.0043] **SEP** | -0.0214 [-0.0321, -0.0108] **SEP** | -0.0198 [-0.0304, -0.0089] **SEP** | -0.0401 [-0.0491, -0.0316] **SEP** | -0.0386 [-0.0480, -0.0300] **SEP** |
| RS_LAT1_XF | -0.0112 [-0.0209, -0.0017] **SEP** | -0.0096 [-0.0193, +0.0001] | -0.0167 [-0.0275, -0.0066] **SEP** | -0.0151 [-0.0259, -0.0046] **SEP** | -0.0354 [-0.0443, -0.0269] **SEP** | -0.0340 [-0.0430, -0.0254] **SEP** |
| RS_XF | -0.0292 [-0.0381, -0.0209] **SEP** | -0.0282 [-0.0371, -0.0200] **SEP** | -0.0347 [-0.0443, -0.0255] **SEP** | -0.0337 [-0.0433, -0.0247] **SEP** | -0.0534 [-0.0639, -0.0432] **SEP** | -0.0525 [-0.0629, -0.0421] **SEP** |

* RS_LON_XF: LE-E1 verdict form FAILED; ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* GATED_RS_LON_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* RS_LAT1_XF: LE-E1 verdict form FAILED; ACC-LAT FAILS (separated worse at any seed: ['LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* share of RS_XF - os (ADE, s0 point) retained by RS_LON: 0.6629213483146067

## step30000_unc / S6 / all -- n 3668 windows / 139 episodes

| arm | - ha0_ext (LE-E3) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RS_LON_XF | -0.9041 [-1.1618, -0.6645] **SEP** | -0.9018 [-1.1620, -0.6583] **SEP** | -0.9327 [-1.1940, -0.6870] **SEP** | -0.9304 [-1.1963, -0.6777] **SEP** | -0.2752 [-0.3794, -0.1725] **SEP** | -0.2760 [-0.3810, -0.1741] **SEP** |
| GATED_RS_LON_XF | -0.9698 [-1.2306, -0.7306] **SEP** | -0.9689 [-1.2370, -0.7264] **SEP** | -0.9984 [-1.2705, -0.7522] **SEP** | -0.9974 [-1.2726, -0.7486] **SEP** | -0.3409 [-0.4694, -0.2225] **SEP** | -0.3430 [-0.4721, -0.2226] **SEP** |
| RS_LAT1_XF | -0.9041 [-1.1618, -0.6645] **SEP** | -0.9018 [-1.1620, -0.6583] **SEP** | -0.9327 [-1.1940, -0.6870] **SEP** | -0.9304 [-1.1963, -0.6777] **SEP** | -0.2752 [-0.3794, -0.1725] **SEP** | -0.2760 [-0.3810, -0.1741] **SEP** |
| RS_XF | -0.9202 [-1.1782, -0.6834] **SEP** | -0.9178 [-1.1749, -0.6751] **SEP** | -0.9488 [-1.2063, -0.7037] **SEP** | -0.9464 [-1.2099, -0.6971] **SEP** | -0.2913 [-0.3990, -0.1864] **SEP** | -0.2920 [-0.4033, -0.1848] **SEP** |

S6 LATERAL rows (step30000_unc / all) -- arm - os and arm - refcv7-50,400, s0 | s1:

| row | RS_LON - os | GATED_RS_LON - os | RS_LAT1 - os | RS_LON - refcv7 | GATED_RS_LON - refcv7 |
|---|---|---|---|---|---|
| LAT_heading_mae_deg | +0.2648 [+0.1217, +0.4308] **SEP** ; +0.2637 [+0.1173, +0.4390] **SEP** | +0.1381 [+0.0285, +0.2840] **SEP** ; +0.1313 [+0.0216, +0.2746] **SEP** | +0.2648 [+0.1217, +0.4308] **SEP** ; +0.2637 [+0.1173, +0.4390] **SEP** | +0.7251 [+0.4701, +1.0035] **SEP** ; +0.7572 [+0.4769, +1.0921] **SEP** | +0.6289 [+0.3786, +0.8960] **SEP** ; +0.6545 [+0.3669, +0.9740] **SEP** |
| LAT_curvature_mae_1pm_valid | +0.0013 [+0.0007, +0.0022] **SEP** ; +0.0013 [+0.0006, +0.0022] **SEP** | +0.0005 [+0.0001, +0.0010] **SEP** ; +0.0005 [+0.0001, +0.0010] **SEP** | +0.0013 [+0.0007, +0.0022] **SEP** ; +0.0013 [+0.0006, +0.0022] **SEP** | +0.0020 [+0.0013, +0.0029] **SEP** ; +0.0020 [+0.0012, +0.0030] **SEP** | +0.0019 [+0.0008, +0.0037] **SEP** ; +0.0019 [+0.0008, +0.0037] **SEP** |
| LAT_yaw_rate_mae_radps_valid | +0.0030 [+0.0013, +0.0050] **SEP** ; +0.0029 [+0.0012, +0.0048] **SEP** | +0.0011 [+0.0001, +0.0023] **SEP** ; +0.0011 [+0.0001, +0.0021] **SEP** | +0.0030 [+0.0013, +0.0050] **SEP** ; +0.0029 [+0.0012, +0.0048] **SEP** | +0.0059 [+0.0037, +0.0085] **SEP** ; +0.0055 [+0.0034, +0.0077] **SEP** | +0.0044 [+0.0025, +0.0065] **SEP** ; +0.0041 [+0.0023, +0.0060] **SEP** |
| LAT_cross_mae_m | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0947 [+0.0328, +0.1594] **SEP** ; +0.0790 [+0.0164, +0.1430] **SEP** | +0.0947 [+0.0328, +0.1594] **SEP** ; +0.0790 [+0.0164, +0.1430] **SEP** |
| LAT_cross_abs_final_m | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] ; +0.0000 [+0.0000, +0.0000] | +0.2476 [+0.0931, +0.4068] **SEP** ; +0.1999 [+0.0401, +0.3645] **SEP** | +0.2476 [+0.0931, +0.4068] **SEP** ; +0.1999 [+0.0401, +0.3645] **SEP** |

* RS_LON_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* GATED_RS_LON_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* RS_LAT1_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid'])
* share of RS_XF - os (ADE, s0 point) retained by RS_LON: 0.9447305183659458

## FITTED ALONG GAINS (EMA-30k, ALL windows, both seeds pooled; cross == 1.0)

* S2 instants [0.5, 1.0, 1.5, 2.0] s: RS_LON along = [0.994, 0.58, 0.43, 0.411]; RS_LAT1 [along, cross] = [[0.994, 1.0], [0.58, 1.0], [0.43, 1.0], [0.411, 1.0]]
  * GATED v0<2 (n 465): along [0.12, 0.091, 0.135, 0.169]
  * GATED v0 2-8 (n 1306): along [0.622, 0.286, 0.358, 0.481]
  * GATED v0 8-15 (n 1792): along [0.82, 0.407, 0.318, 0.341]
  * GATED v0 15-25 (n 856): along [1.278, 0.824, 0.729, 0.483]
  * GATED v0>=25 (n 335): along [1.375, 0.867, 0.835, 0.819]
* S6 instants [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s: RS_LON along = [0.599, 0.441, 0.53, 0.59, 0.636, 0.65]; RS_LAT1 [along, cross] = [[0.599, 1.0], [0.441, 1.0], [0.53, 1.0], [0.59, 1.0], [0.636, 1.0], [0.65, 1.0]]
  * GATED v0<2 (n 330): along [0.148, 0.263, 0.345, 0.436, 0.463, 0.51]
  * GATED v0 2-8 (n 1017): along [0.322, 0.516, 0.701, 0.806, 0.83, 0.83]
  * GATED v0 8-15 (n 1397): along [0.432, 0.356, 0.428, 0.484, 0.513, 0.54]
  * GATED v0 15-25 (n 679): along [0.828, 0.415, 0.16, 0.175, 0.208, 0.227]
  * GATED v0>=25 (n 245): along [0.864, 0.854, 0.72, 0.535, 0.426, 0.352]
