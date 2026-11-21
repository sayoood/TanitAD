# RESULTS_BRIEF -- arc-length re-timing (all cells T1 self-action OPEN LOOP on logged eval139 frames; paired episode-cluster bootstrap B 2000, 'another draw of EPISODES' only, single training seed; n windows / episodes in each header)

controls ALL_PASS = True; PREREG_RETIME sha256 dc02f704fd09; script sha256 842889130005

## step30000ema_unc / S2 / compliant -- n 4159 windows / 138 episodes

| arm | - ha0_ext (LE-E1) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RETIME_XF | -0.0269 [-0.0376, -0.0156] **SEP** | -0.0265 [-0.0371, -0.0153] **SEP** | -0.0321 [-0.0432, -0.0208] **SEP** | -0.0316 [-0.0425, -0.0202] **SEP** | -0.0184 [-0.0245, -0.0127] **SEP** | -0.0189 [-0.0249, -0.0134] **SEP** |
| GATED_RETIME_XF | -0.0310 [-0.0419, -0.0201] **SEP** | -0.0301 [-0.0402, -0.0191] **SEP** | -0.0362 [-0.0472, -0.0252] **SEP** | -0.0352 [-0.0458, -0.0242] **SEP** | -0.0225 [-0.0296, -0.0161] **SEP** | -0.0225 [-0.0294, -0.0163] **SEP** |
| RS_XF | -0.0428 [-0.0538, -0.0321] **SEP** | -0.0427 [-0.0534, -0.0321] **SEP** | -0.0479 [-0.0596, -0.0367] **SEP** | -0.0479 [-0.0593, -0.0368] **SEP** | -0.0343 [-0.0419, -0.0274] **SEP** | -0.0351 [-0.0425, -0.0284] **SEP** |

* RETIME_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* GATED_RETIME_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 0.5364431486880467

## step30000ema_unc / S6 / compliant -- n 3251 windows / 138 episodes

| arm | - ha0_ext (LE-E3) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RETIME_XF | -0.9825 [-1.2567, -0.7375] **SEP** | -0.9808 [-1.2490, -0.7395] **SEP** | -1.0113 [-1.2913, -0.7596] **SEP** | -1.0095 [-1.2813, -0.7640] **SEP** | -0.2290 [-0.3254, -0.1316] **SEP** | -0.2378 [-0.3367, -0.1368] **SEP** |
| GATED_RETIME_XF | -0.9932 [-1.2680, -0.7501] **SEP** | -0.9907 [-1.2577, -0.7507] **SEP** | -1.0219 [-1.3024, -0.7778] **SEP** | -1.0195 [-1.2912, -0.7746] **SEP** | -0.2396 [-0.3416, -0.1369] **SEP** | -0.2478 [-0.3539, -0.1419] **SEP** |
| RS_XF | -0.9162 [-1.1794, -0.6723] **SEP** | -0.9161 [-1.1758, -0.6742] **SEP** | -0.9450 [-1.2100, -0.6967] **SEP** | -0.9449 [-1.2098, -0.6947] **SEP** | -0.1627 [-0.2464, -0.0763] **SEP** | -0.1732 [-0.2554, -0.0869] **SEP** |

S6 LATERAL rows (step30000ema_unc / compliant) -- arm - os and arm - refcv7-50,400, s0 ; s1:

| row | RETIME - os | GATED_RETIME - os | RETIME - refcv7 | GATED_RETIME - refcv7 |
|---|---|---|---|---|
| LAT_heading_mae_deg | -0.0372 [-0.1053, +0.0265] ; -0.0466 [-0.1122, +0.0149] | -0.0275 [-0.0836, +0.0267] ; -0.0357 [-0.0877, +0.0138] | +0.0894 [-0.0981, +0.2637] ; +0.1393 [-0.0492, +0.3333] | +0.0975 [-0.0971, +0.2777] ; +0.1897 [-0.0237, +0.4136] |
| LAT_curvature_mae_1pm_valid | -0.0000 [-0.0001, +0.0001] ; -0.0000 [-0.0002, +0.0001] | -0.0001 [-0.0002, +0.0000] ; -0.0001 [-0.0002, +0.0000] | -0.0000 [-0.0004, +0.0004] ; +0.0002 [-0.0002, +0.0005] | -0.0001 [-0.0004, +0.0003] ; +0.0001 [-0.0003, +0.0005] |
| LAT_yaw_rate_mae_radps_valid | -0.0005 [-0.0011, +0.0002] ; -0.0006 [-0.0012, +0.0000] | -0.0005 [-0.0010, -0.0000] **SEP** ; -0.0006 [-0.0010, -0.0001] **SEP** | -0.0002 [-0.0016, +0.0010] ; +0.0001 [-0.0012, +0.0014] | -0.0003 [-0.0016, +0.0010] ; +0.0001 [-0.0012, +0.0014] |
| LAT_cross_mae_m | -0.0139 [-0.0478, +0.0200] ; -0.0168 [-0.0529, +0.0187] | -0.0136 [-0.0430, +0.0164] ; -0.0157 [-0.0469, +0.0161] | +0.0083 [-0.0543, +0.0692] ; +0.0084 [-0.0552, +0.0730] | +0.0087 [-0.0536, +0.0682] ; +0.0095 [-0.0545, +0.0735] |
| LAT_cross_abs_final_m | -0.0398 [-0.1370, +0.0588] ; -0.0490 [-0.1501, +0.0534] | -0.0370 [-0.1199, +0.0504] ; -0.0429 [-0.1305, +0.0495] | +0.0339 [-0.1322, +0.1946] ; +0.0280 [-0.1351, +0.1910] | +0.0367 [-0.1256, +0.1946] ; +0.0340 [-0.1299, +0.2006] |

* RETIME_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* GATED_RETIME_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 1.407498463429625

## step30000ema_unc / S2 / all -- n 4754 windows / 139 episodes

| arm | - ha0_ext (LE-E1) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RETIME_XF | -0.0249 [-0.0355, -0.0145] **SEP** | -0.0236 [-0.0344, -0.0133] **SEP** | -0.0304 [-0.0416, -0.0194] **SEP** | -0.0291 [-0.0403, -0.0182] **SEP** | -0.0241 [-0.0308, -0.0175] **SEP** | -0.0250 [-0.0320, -0.0184] **SEP** |
| GATED_RETIME_XF | -0.0297 [-0.0403, -0.0187] **SEP** | -0.0279 [-0.0388, -0.0171] **SEP** | -0.0352 [-0.0464, -0.0240] **SEP** | -0.0335 [-0.0448, -0.0221] **SEP** | -0.0288 [-0.0359, -0.0220] **SEP** | -0.0294 [-0.0364, -0.0228] **SEP** |
| RS_XF | -0.0412 [-0.0511, -0.0317] **SEP** | -0.0401 [-0.0498, -0.0307] **SEP** | -0.0467 [-0.0573, -0.0362] **SEP** | -0.0457 [-0.0561, -0.0354] **SEP** | -0.0403 [-0.0484, -0.0319] **SEP** | -0.0416 [-0.0500, -0.0331] **SEP** |

* RETIME_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* GATED_RETIME_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 0.598014888337469

## step30000ema_unc / S6 / all -- n 3668 windows / 139 episodes

| arm | - ha0_ext (LE-E3) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RETIME_XF | -1.0717 [-1.3539, -0.8089] **SEP** | -1.0591 [-1.3353, -0.7999] **SEP** | -1.1003 [-1.3924, -0.8258] **SEP** | -1.0877 [-1.3760, -0.8174] **SEP** | -0.2571 [-0.3473, -0.1679] **SEP** | -0.2689 [-0.3633, -0.1755] **SEP** |
| GATED_RETIME_XF | -1.0813 [-1.3688, -0.8175] **SEP** | -1.0694 [-1.3522, -0.8107] **SEP** | -1.1099 [-1.4039, -0.8384] **SEP** | -1.0979 [-1.3906, -0.8334] **SEP** | -0.2667 [-0.3644, -0.1702] **SEP** | -0.2792 [-0.3793, -0.1787] **SEP** |
| RS_XF | -1.0018 [-1.2678, -0.7533] **SEP** | -0.9895 [-1.2504, -0.7423] **SEP** | -1.0304 [-1.3075, -0.7709] **SEP** | -1.0181 [-1.2915, -0.7621] **SEP** | -0.1872 [-0.2689, -0.1026] **SEP** | -0.1994 [-0.2823, -0.1141] **SEP** |

S6 LATERAL rows (step30000ema_unc / all) -- arm - os and arm - refcv7-50,400, s0 ; s1:

| row | RETIME - os | GATED_RETIME - os | RETIME - refcv7 | GATED_RETIME - refcv7 |
|---|---|---|---|---|
| LAT_heading_mae_deg | -0.0192 [-0.0804, +0.0402] ; -0.0252 [-0.0869, +0.0327] | -0.0064 [-0.0577, +0.0420] ; -0.0150 [-0.0632, +0.0292] | +0.1150 [-0.0474, +0.2885] ; +0.1730 [-0.0001, +0.3527] | +0.1282 [-0.0373, +0.3076] ; +0.2182 [+0.0192, +0.4214] **SEP** |
| LAT_curvature_mae_1pm_valid | -0.0000 [-0.0001, +0.0001] ; -0.0000 [-0.0002, +0.0001] | -0.0001 [-0.0001, +0.0000] ; -0.0001 [-0.0002, -0.0000] **SEP** | +0.0000 [-0.0003, +0.0004] ; +0.0002 [-0.0001, +0.0005] | -0.0000 [-0.0003, +0.0003] ; +0.0001 [-0.0002, +0.0005] |
| LAT_yaw_rate_mae_radps_valid | -0.0003 [-0.0009, +0.0003] ; -0.0005 [-0.0011, +0.0001] | -0.0003 [-0.0008, +0.0002] ; -0.0005 [-0.0010, -0.0000] **SEP** | +0.0000 [-0.0012, +0.0013] ; +0.0004 [-0.0008, +0.0017] | +0.0000 [-0.0012, +0.0014] ; +0.0004 [-0.0009, +0.0017] |
| LAT_cross_mae_m | -0.0002 [-0.0347, +0.0318] ; -0.0014 [-0.0378, +0.0324] | +0.0022 [-0.0280, +0.0304] ; +0.0018 [-0.0306, +0.0324] | +0.0207 [-0.0381, +0.0806] ; +0.0248 [-0.0360, +0.0836] | +0.0231 [-0.0348, +0.0818] ; +0.0279 [-0.0332, +0.0872] |
| LAT_cross_abs_final_m | -0.0006 [-0.0971, +0.0907] ; -0.0069 [-0.1097, +0.0877] | +0.0098 [-0.0756, +0.0926] ; +0.0069 [-0.0811, +0.0942] | +0.0540 [-0.0932, +0.2076] ; +0.0593 [-0.0923, +0.2100] | +0.0644 [-0.0868, +0.2213] ; +0.0730 [-0.0828, +0.2255] |

* RETIME_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* GATED_RETIME_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 1.373397435897436

## step30000_unc / S2 / compliant -- n 4159 windows / 138 episodes

| arm | - ha0_ext (LE-E1) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RETIME_XF | -0.0130 [-0.0233, -0.0027] **SEP** | -0.0122 [-0.0225, -0.0016] **SEP** | -0.0181 [-0.0288, -0.0073] **SEP** | -0.0173 [-0.0278, -0.0061] **SEP** | -0.0304 [-0.0396, -0.0223] **SEP** | -0.0282 [-0.0370, -0.0198] **SEP** |
| GATED_RETIME_XF | -0.0170 [-0.0273, -0.0065] **SEP** | -0.0159 [-0.0261, -0.0051] **SEP** | -0.0221 [-0.0327, -0.0115] **SEP** | -0.0210 [-0.0315, -0.0100] **SEP** | -0.0344 [-0.0442, -0.0261] **SEP** | -0.0319 [-0.0415, -0.0233] **SEP** |
| RS_XF | -0.0289 [-0.0382, -0.0199] **SEP** | -0.0286 [-0.0381, -0.0197] **SEP** | -0.0340 [-0.0441, -0.0244] **SEP** | -0.0338 [-0.0439, -0.0243] **SEP** | -0.0463 [-0.0567, -0.0368] **SEP** | -0.0446 [-0.0549, -0.0352] **SEP** |

* RETIME_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* GATED_RETIME_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 0.6565874730021598

## step30000_unc / S6 / compliant -- n 3251 windows / 138 episodes

| arm | - ha0_ext (LE-E3) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RETIME_XF | -0.9052 [-1.1680, -0.6618] **SEP** | -0.8992 [-1.1682, -0.6570] **SEP** | -0.9340 [-1.2058, -0.6833] **SEP** | -0.9280 [-1.1996, -0.6782] **SEP** | -0.3312 [-0.4497, -0.2201] **SEP** | -0.3142 [-0.4327, -0.2047] **SEP** |
| GATED_RETIME_XF | -0.9279 [-1.1950, -0.6916] **SEP** | -0.9219 [-1.1888, -0.6847] **SEP** | -0.9567 [-1.2303, -0.7145] **SEP** | -0.9506 [-1.2253, -0.7062] **SEP** | -0.3539 [-0.4873, -0.2348] **SEP** | -0.3368 [-0.4678, -0.2165] **SEP** |
| RS_XF | -0.8431 [-1.0855, -0.6161] **SEP** | -0.8444 [-1.0949, -0.6111] **SEP** | -0.8718 [-1.1246, -0.6370] **SEP** | -0.8732 [-1.1285, -0.6328] **SEP** | -0.2690 [-0.3804, -0.1547] **SEP** | -0.2594 [-0.3723, -0.1447] **SEP** |

S6 LATERAL rows (step30000_unc / compliant) -- arm - os and arm - refcv7-50,400, s0 ; s1:

| row | RETIME - os | GATED_RETIME - os | RETIME - refcv7 | GATED_RETIME - refcv7 |
|---|---|---|---|---|
| LAT_heading_mae_deg | -0.0091 [-0.0911, +0.0672] ; -0.0188 [-0.1186, +0.0670] | -0.0029 [-0.0732, +0.0646] ; -0.0165 [-0.1251, +0.0652] | +0.3747 [+0.1659, +0.5944] **SEP** ; +0.4521 [+0.2260, +0.6984] **SEP** | +0.4206 [+0.1908, +0.6797] **SEP** ; +0.4938 [+0.2339, +0.7829] **SEP** |
| LAT_curvature_mae_1pm_valid | -0.0002 [-0.0005, -0.0000] **SEP** ; -0.0002 [-0.0005, -0.0000] **SEP** | -0.0003 [-0.0005, -0.0001] **SEP** ; -0.0003 [-0.0005, -0.0001] **SEP** | +0.0004 [+0.0000, +0.0007] **SEP** ; +0.0004 [+0.0000, +0.0009] **SEP** | +0.0003 [-0.0001, +0.0006] ; +0.0004 [-0.0000, +0.0008] |
| LAT_yaw_rate_mae_radps_valid | -0.0008 [-0.0016, +0.0001] ; -0.0005 [-0.0014, +0.0003] | -0.0008 [-0.0015, -0.0001] **SEP** ; -0.0005 [-0.0012, +0.0002] | +0.0017 [+0.0003, +0.0031] **SEP** ; +0.0018 [+0.0005, +0.0032] **SEP** | +0.0016 [+0.0002, +0.0030] **SEP** ; +0.0018 [+0.0004, +0.0031] **SEP** |
| LAT_cross_mae_m | +0.0121 [-0.0267, +0.0519] ; +0.0190 [-0.0201, +0.0576] | +0.0127 [-0.0203, +0.0480] ; +0.0196 [-0.0138, +0.0545] | +0.0906 [+0.0174, +0.1688] **SEP** ; +0.0847 [+0.0147, +0.1573] **SEP** | +0.0912 [+0.0174, +0.1685] **SEP** ; +0.0853 [+0.0161, +0.1570] **SEP** |
| LAT_cross_abs_final_m | +0.0242 [-0.0829, +0.1389] ; +0.0396 [-0.0718, +0.1501] | +0.0295 [-0.0634, +0.1329] ; +0.0445 [-0.0478, +0.1426] | +0.2479 [+0.0543, +0.4502] **SEP** ; +0.2241 [+0.0413, +0.4065] **SEP** | +0.2532 [+0.0517, +0.4529] **SEP** ; +0.2289 [+0.0412, +0.4136] **SEP** |

* RETIME_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* GATED_RETIME_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 1.2312267657992564

## step30000_unc / S2 / all -- n 4754 windows / 139 episodes

| arm | - ha0_ext (LE-E1) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RETIME_XF | -0.0120 [-0.0222, -0.0019] **SEP** | -0.0103 [-0.0206, +0.0001] | -0.0175 [-0.0287, -0.0068] **SEP** | -0.0158 [-0.0271, -0.0047] **SEP** | -0.0362 [-0.0446, -0.0278] **SEP** | -0.0347 [-0.0433, -0.0260] **SEP** |
| GATED_RETIME_XF | -0.0165 [-0.0268, -0.0061] **SEP** | -0.0148 [-0.0251, -0.0043] **SEP** | -0.0220 [-0.0332, -0.0109] **SEP** | -0.0203 [-0.0316, -0.0091] **SEP** | -0.0407 [-0.0492, -0.0324] **SEP** | -0.0392 [-0.0480, -0.0309] **SEP** |
| RS_XF | -0.0292 [-0.0381, -0.0209] **SEP** | -0.0282 [-0.0371, -0.0200] **SEP** | -0.0347 [-0.0443, -0.0255] **SEP** | -0.0337 [-0.0433, -0.0247] **SEP** | -0.0534 [-0.0639, -0.0432] **SEP** | -0.0525 [-0.0629, -0.0421] **SEP** |

* RETIME_XF: LE-E1 verdict form FAILED; ACC-LAT HOLDS (separated worse at any seed: none)
* GATED_RETIME_XF: LE-E1 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_cross_mae_m'])
* share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 0.6779026217228464

## step30000_unc / S6 / all -- n 3668 windows / 139 episodes

| arm | - ha0_ext (LE-E3) s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |
|---|---|---|---|---|---|---|
| RETIME_XF | -0.9780 [-1.2464, -0.7207] **SEP** | -0.9742 [-1.2514, -0.7164] **SEP** | -1.0066 [-1.2872, -0.7443] **SEP** | -1.0028 [-1.2885, -0.7371] **SEP** | -0.3491 [-0.4539, -0.2481] **SEP** | -0.3484 [-0.4535, -0.2437] **SEP** |
| GATED_RETIME_XF | -1.0071 [-1.2818, -0.7462] **SEP** | -1.0038 [-1.2819, -0.7458] **SEP** | -1.0357 [-1.3190, -0.7696] **SEP** | -1.0324 [-1.3151, -0.7664] **SEP** | -0.3781 [-0.5023, -0.2632] **SEP** | -0.3780 [-0.5065, -0.2593] **SEP** |
| RS_XF | -0.9202 [-1.1782, -0.6834] **SEP** | -0.9178 [-1.1749, -0.6751] **SEP** | -0.9488 [-1.2063, -0.7037] **SEP** | -0.9464 [-1.2099, -0.6971] **SEP** | -0.2913 [-0.3990, -0.1864] **SEP** | -0.2920 [-0.4033, -0.1848] **SEP** |

S6 LATERAL rows (step30000_unc / all) -- arm - os and arm - refcv7-50,400, s0 ; s1:

| row | RETIME - os | GATED_RETIME - os | RETIME - refcv7 | GATED_RETIME - refcv7 |
|---|---|---|---|---|
| LAT_heading_mae_deg | +0.0092 [-0.0605, +0.0771] ; +0.0147 [-0.0561, +0.0799] | +0.0161 [-0.0435, +0.0756] ; +0.0261 [-0.0321, +0.0813] | +0.4607 [+0.2578, +0.6712] **SEP** ; +0.5021 [+0.2856, +0.7312] **SEP** | +0.5025 [+0.2804, +0.7233] **SEP** ; +0.5485 [+0.3003, +0.8358] **SEP** |
| LAT_curvature_mae_1pm_valid | -0.0002 [-0.0005, -0.0001] **SEP** ; -0.0002 [-0.0005, -0.0000] **SEP** | -0.0003 [-0.0005, -0.0001] **SEP** ; -0.0002 [-0.0005, -0.0001] **SEP** | +0.0004 [+0.0001, +0.0007] **SEP** ; +0.0005 [+0.0001, +0.0008] **SEP** | +0.0003 [+0.0000, +0.0007] **SEP** ; +0.0004 [-0.0000, +0.0008] |
| LAT_yaw_rate_mae_radps_valid | -0.0007 [-0.0014, +0.0000] ; -0.0004 [-0.0012, +0.0003] | -0.0006 [-0.0013, -0.0001] **SEP** ; -0.0004 [-0.0010, +0.0002] | +0.0023 [+0.0010, +0.0037] **SEP** ; +0.0022 [+0.0009, +0.0035] **SEP** | +0.0022 [+0.0009, +0.0037] **SEP** ; +0.0021 [+0.0009, +0.0035] **SEP** |
| LAT_cross_mae_m | +0.0278 [-0.0066, +0.0607] ; +0.0329 [-0.0033, +0.0672] | +0.0314 [+0.0009, +0.0609] **SEP** ; +0.0365 [+0.0055, +0.0656] **SEP** | +0.1225 [+0.0493, +0.1999] **SEP** ; +0.1119 [+0.0437, +0.1863] **SEP** | +0.1262 [+0.0531, +0.2038] **SEP** ; +0.1155 [+0.0457, +0.1902] **SEP** |
| LAT_cross_abs_final_m | +0.0685 [-0.0293, +0.1650] ; +0.0797 [-0.0227, +0.1763] | +0.0842 [-0.0027, +0.1682] ; +0.0958 [+0.0074, +0.1783] **SEP** | +0.3160 [+0.1335, +0.5080] **SEP** ; +0.2796 [+0.1054, +0.4601] **SEP** | +0.3318 [+0.1457, +0.5314] **SEP** ; +0.2957 [+0.1167, +0.4830] **SEP** |

* RETIME_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT HOLDS (separated worse at any seed: none)
* GATED_RETIME_XF: LE-E3 verdict form PASS (single training seed; training-seed floor unmeasured); ACC-LAT FAILS (separated worse at any seed: ['LAT_cross_mae_m', 'LAT_cross_abs_final_m'])
* share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 1.1984208719533127

## FITTED GAINS (EMA-30k, ALL windows, both seeds pooled; one scalar g per instant)

* S2 instants [0.5, 1.0, 1.5, 2.0] s: RETIME g = [0.997, 0.583, 0.426, 0.382]
  * GATED v0<2 (n 465): g [0.118, 0.108, 0.128, 0.168]
  * GATED v0 2-8 (n 1306): g [0.629, 0.289, 0.339, 0.422]
  * GATED v0 8-15 (n 1792): g [0.823, 0.421, 0.324, 0.334]
  * GATED v0 15-25 (n 856): g [1.279, 0.823, 0.721, 0.459]
  * GATED v0>=25 (n 335): g [1.376, 0.867, 0.834, 0.819]
* S6 instants [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s: RETIME g = [0.604, 0.421, 0.464, 0.508, 0.53, 0.555]
  * GATED v0<2 (n 330): g [0.152, 0.262, 0.335, 0.43, 0.452, 0.473]
  * GATED v0 2-8 (n 1017): g [0.326, 0.472, 0.59, 0.638, 0.663, 0.669]
  * GATED v0 8-15 (n 1397): g [0.45, 0.363, 0.424, 0.479, 0.512, 0.534]
  * GATED v0 15-25 (n 679): g [0.828, 0.403, 0.142, 0.169, 0.203, 0.22]
  * GATED v0>=25 (n 245): g [0.864, 0.855, 0.725, 0.543, 0.446, 0.386]
