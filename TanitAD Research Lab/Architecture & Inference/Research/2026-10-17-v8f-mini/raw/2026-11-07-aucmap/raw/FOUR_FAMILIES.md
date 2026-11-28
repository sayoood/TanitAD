# FOUR_FAMILIES (AUCMAP, CEILING-TYPE: oracle gate AUC 1.0 draw 0, ANCHOR scope v0<8; tier T1-kin; STRATEGIC PAUSED (PI ITEM 25); rows are X (gated) minus R (RIC0), delta [lo, hi], seed 0 | seed 1; estimator: paired episode-cluster bootstrap B=2000)

## LF050 (ADMISSIBLE heads level), q = 1.0 (smallest q with point share_closed >= 0.5 at AUC 1.0; 1.0 if none), share_closed 0.054 | 0.045, M1P False, M1FULL False, GUARD False
| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse (direction-aware) s0/s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 | +0.1078 [+0.0125, +0.2312] | +0.0654 [-0.0072, +0.1535] | True/False |
| LONGITUDINAL | speed MAE | +0.0318 [-0.0252, +0.0987] | +0.0170 [-0.0192, +0.0604] | False/False |
| LONGITUDINAL | along-track MAE | +0.0781 [-0.0049, +0.1841] | +0.0425 [-0.0149, +0.1099] | False/False |
| LONGITUDINAL | accel MAE | -0.0508 [-0.1568, +0.0518] | -0.0150 [-0.0720, +0.0399] | False/False |
| LONGITUDINAL | DK-VIOL | +0.0151 [-0.0016, +0.0387] | +0.0134 [+0.0032, +0.0271] | False/True |
| LONGITUDINAL | DK-HW | +0.0741 [-0.0266, +0.2152] | +0.0533 [-0.0568, +0.1747] | False/False |
| LONGITUDINAL | DK-TG | +0.0419 [-0.0500, +0.1701] | +0.0119 [-0.1030, +0.1128] | False/False |
| LATERAL | heading MAE | -0.6036 [-1.1622, -0.1518] | -0.7979 [-1.5469, -0.2817] | False/False |
| LATERAL | yaw-rate MAE | -4.5635 [-8.1257, -2.1202] | -4.8997 [-8.8509, -2.1376] | False/False |
| LATERAL | curvature MAE | -0.0542 [-0.0828, -0.0315] | -0.0555 [-0.0928, -0.0277] | False/False |
| LATERAL | cross-track MAE | +0.0556 [+0.0160, +0.1032] | +0.0386 [+0.0007, +0.0938] | True/True |
| LATERAL | L1 lateral-accel MAE | -0.1890 [-0.6198, +0.1587] | -0.1144 [-0.4002, +0.0956] | False/False |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0262 [-0.0456, -0.0120] | -0.0325 [-0.0585, -0.0100] | False/False |
| TACTICAL | TAC-LAT | +0.0142 [+0.0063, +0.0225] | +0.0117 [+0.0050, +0.0187] | False/False |
| TACTICAL | TAC-LON | -0.0008 [-0.0075, +0.0058] | +0.0063 [+0.0000, +0.0133] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@30 | -0.1531 [-0.3185, -0.0064] | -0.2876 [-0.4655, -0.1282] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@50 | -0.2817 [-0.5794, -0.0285] | -0.4146 [-0.6798, -0.1726] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@80 | -0.4910 [-0.9662, -0.0957] | -0.6000 [-1.0421, -0.2122] | False/False |
| crawl | M1 crawl share | -0.0029 [-0.0066, +0.0005] | -0.0027 [-0.0054, -0.0002] | False/False |

## LF075 (ADMISSIBLE heads level), q = 1.0 (smallest q with point share_closed >= 0.5 at AUC 1.0; 1.0 if none), share_closed 0.357 | 0.338, M1P False, M1FULL False, GUARD False
| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse (direction-aware) s0/s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 | +0.2853 [+0.1448, +0.4332] | +0.1693 [+0.0584, +0.2812] | True/True |
| LONGITUDINAL | speed MAE | +0.0876 [+0.0044, +0.1801] | +0.0255 [-0.0267, +0.0782] | True/False |
| LONGITUDINAL | along-track MAE | +0.2045 [+0.0818, +0.3317] | +0.1309 [+0.0185, +0.2392] | True/True |
| LONGITUDINAL | accel MAE | -0.0603 [-0.1751, +0.0630] | -0.0463 [-0.1094, +0.0160] | False/False |
| LONGITUDINAL | DK-VIOL | +0.0252 [+0.0049, +0.0526] | +0.0252 [+0.0075, +0.0485] | True/True |
| LONGITUDINAL | DK-HW | +0.1477 [+0.0317, +0.3026] | +0.0801 [-0.0491, +0.2253] | True/False |
| LONGITUDINAL | DK-TG | +0.0659 [-0.0219, +0.1944] | +0.0258 [-0.0904, +0.1312] | False/False |
| LATERAL | heading MAE | -0.9031 [-1.5226, -0.3417] | -0.9153 [-1.7182, -0.3708] | False/False |
| LATERAL | yaw-rate MAE | -7.1592 [-10.8906, -4.3722] | -7.1833 [-11.3999, -4.0238] | False/False |
| LATERAL | curvature MAE | -0.0733 [-0.1053, -0.0460] | -0.0767 [-0.1178, -0.0433] | False/False |
| LATERAL | cross-track MAE | +0.1370 [+0.0672, +0.2227] | +0.0717 [+0.0324, +0.1177] | True/True |
| LATERAL | L1 lateral-accel MAE | -0.2935 [-0.7649, +0.1102] | -0.3221 [-0.6176, -0.1071] | False/False |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0376 [-0.0649, -0.0181] | -0.0394 [-0.0681, -0.0141] | False/False |
| TACTICAL | TAC-LAT | +0.0267 [+0.0146, +0.0404] | +0.0263 [+0.0146, +0.0392] | False/False |
| TACTICAL | TAC-LON | +0.0058 [-0.0100, +0.0230] | +0.0196 [+0.0038, +0.0367] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@30 | -0.3539 [-0.5657, -0.1603] | -0.4862 [-0.7406, -0.2692] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@50 | -0.6407 [-1.0083, -0.3094] | -0.7694 [-1.2042, -0.4116] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@80 | -1.0949 [-1.7016, -0.5712] | -1.2027 [-1.9353, -0.5924] | False/False |
| crawl | M1 crawl share | -0.0191 [-0.0312, -0.0092] | -0.0201 [-0.0327, -0.0101] | False/False |

## LF100 (ADMISSIBLE heads level), q = 0.5 (smallest q with point share_closed >= 0.5 at AUC 1.0; 1.0 if none), share_closed 0.580 | 0.455, M1P True, M1FULL False, GUARD False
| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse (direction-aware) s0/s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 | +0.0135 [-0.0369, +0.0627] | +0.0024 [-0.0436, +0.0460] | False/False |
| LONGITUDINAL | speed MAE | -0.0407 [-0.0693, -0.0128] | -0.0359 [-0.0627, -0.0116] | False/False |
| LONGITUDINAL | along-track MAE | -0.0124 [-0.0695, +0.0416] | -0.0273 [-0.0845, +0.0249] | False/False |
| LONGITUDINAL | accel MAE | -0.0675 [-0.1284, -0.0110] | -0.0354 [-0.0877, +0.0155] | False/False |
| LONGITUDINAL | DK-VIOL | +0.0034 [+0.0000, +0.0085] | +0.0034 [+0.0000, +0.0084] | False/False |
| LONGITUDINAL | DK-HW | +0.0581 [-0.0227, +0.1585] | -0.0156 [-0.1384, +0.0969] | False/False |
| LONGITUDINAL | DK-TG | +0.0156 [-0.0499, +0.0914] | -0.0257 [-0.1289, +0.0485] | False/False |
| LATERAL | heading MAE | -0.7316 [-1.1747, -0.3620] | -0.7761 [-1.3047, -0.3700] | False/False |
| LATERAL | yaw-rate MAE | -4.4601 [-7.0550, -2.4325] | -4.5869 [-7.6908, -2.3806] | False/False |
| LATERAL | curvature MAE | -0.0385 [-0.0582, -0.0217] | -0.0428 [-0.0686, -0.0220] | False/False |
| LATERAL | cross-track MAE | +0.0294 [+0.0076, +0.0547] | +0.0307 [+0.0078, +0.0577] | True/True |
| LATERAL | L1 lateral-accel MAE | -0.3661 [-0.7143, -0.1097] | -0.2297 [-0.5033, -0.0164] | False/False |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0326 [-0.0582, -0.0144] | -0.0319 [-0.0574, -0.0106] | False/False |
| TACTICAL | TAC-LAT | +0.0188 [+0.0088, +0.0304] | +0.0208 [+0.0117, +0.0312] | False/False |
| TACTICAL | TAC-LON | +0.0509 [+0.0342, +0.0691] | +0.0454 [+0.0304, +0.0626] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@30 | -0.1629 [-0.3043, -0.0303] | -0.3009 [-0.4915, -0.1398] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@50 | -0.3084 [-0.5770, -0.0677] | -0.4787 [-0.8103, -0.2056] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@80 | -0.5350 [-0.9796, -0.1492] | -0.7478 [-1.3092, -0.2938] | False/False |
| crawl | M1 crawl share | -0.0309 [-0.0444, -0.0190] | -0.0271 [-0.0412, -0.0156] | False/False |

## LFB05 (ADMISSIBLE heads level), q = 1.0 (smallest q with point share_closed >= 0.5 at AUC 1.0; 1.0 if none), share_closed 0.335 | 0.291, M1P False, M1FULL False, GUARD False
| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse (direction-aware) s0/s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 | +0.0235 [-0.0392, +0.0882] | +0.0064 [-0.0459, +0.0626] | False/False |
| LONGITUDINAL | speed MAE | -0.0243 [-0.0697, +0.0209] | -0.0321 [-0.0590, -0.0026] | False/False |
| LONGITUDINAL | along-track MAE | +0.0191 [-0.0453, +0.0824] | +0.0095 [-0.0460, +0.0693] | False/False |
| LONGITUDINAL | accel MAE | -0.1275 [-0.2236, -0.0473] | -0.0726 [-0.1290, -0.0278] | False/False |
| LONGITUDINAL | DK-VIOL | +0.0118 [+0.0000, +0.0285] | +0.0118 [+0.0017, +0.0253] | False/True |
| LONGITUDINAL | DK-HW | +0.0919 [-0.0057, +0.1960] | -0.0070 [-0.1026, +0.0754] | False/False |
| LONGITUDINAL | DK-TG | +0.0222 [-0.0366, +0.0857] | -0.0249 [-0.1219, +0.0380] | False/False |
| LATERAL | heading MAE | -0.7626 [-1.2538, -0.3371] | -0.7332 [-1.2094, -0.3381] | False/False |
| LATERAL | yaw-rate MAE | -5.9010 [-9.1570, -3.4794] | -5.2098 [-8.2215, -2.9034] | False/False |
| LATERAL | curvature MAE | -0.0510 [-0.0746, -0.0323] | -0.0486 [-0.0751, -0.0264] | False/False |
| LATERAL | cross-track MAE | +0.0002 [-0.0195, +0.0217] | -0.0080 [-0.0242, +0.0076] | False/False |
| LATERAL | L1 lateral-accel MAE | -0.4744 [-0.9168, -0.1628] | -0.3453 [-0.6151, -0.1515] | False/False |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0317 [-0.0543, -0.0156] | -0.0283 [-0.0501, -0.0094] | False/False |
| TACTICAL | TAC-LAT | +0.0200 [+0.0088, +0.0321] | +0.0354 [+0.0208, +0.0530] | False/False |
| TACTICAL | TAC-LON | +0.0000 [-0.0183, +0.0175] | +0.0129 [-0.0008, +0.0275] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@30 | -0.3507 [-0.5633, -0.1603] | -0.4787 [-0.7082, -0.2786] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@50 | -0.5867 [-0.9528, -0.2632] | -0.7450 [-1.1615, -0.3927] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@80 | -0.8283 [-1.4166, -0.3309] | -1.1708 [-1.8753, -0.5696] | False/False |
| crawl | M1 crawl share | -0.0179 [-0.0256, -0.0107] | -0.0174 [-0.0264, -0.0094] | False/False |

## LFB10 (ADMISSIBLE heads level), q = 1.0 (smallest q with point share_closed >= 0.5 at AUC 1.0; 1.0 if none), share_closed 0.510 | 0.428, M1P False, M1FULL False, GUARD False
| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse (direction-aware) s0/s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 | +0.1213 [+0.0275, +0.2222] | +0.1027 [+0.0144, +0.1896] | True/True |
| LONGITUDINAL | speed MAE | +0.0041 [-0.0455, +0.0561] | +0.0046 [-0.0379, +0.0458] | False/False |
| LONGITUDINAL | along-track MAE | +0.1082 [+0.0112, +0.2145] | +0.0841 [-0.0082, +0.1796] | True/False |
| LONGITUDINAL | accel MAE | -0.0982 [-0.1911, -0.0139] | -0.0291 [-0.1025, +0.0484] | False/False |
| LONGITUDINAL | DK-VIOL | +0.0286 [+0.0112, +0.0503] | +0.0168 [+0.0035, +0.0348] | True/True |
| LONGITUDINAL | DK-HW | +0.2298 [+0.0979, +0.3735] | +0.1058 [-0.0069, +0.2363] | True/False |
| LONGITUDINAL | DK-TG | +0.0619 [-0.0052, +0.1506] | +0.0107 [-0.0947, +0.0991] | False/False |
| LATERAL | heading MAE | -0.7651 [-1.2944, -0.2865] | -0.9073 [-1.5382, -0.3940] | False/False |
| LATERAL | yaw-rate MAE | -6.5315 [-10.0392, -3.8806] | -6.3161 [-9.6733, -3.7478] | False/False |
| LATERAL | curvature MAE | -0.0636 [-0.0920, -0.0394] | -0.0662 [-0.0996, -0.0389] | False/False |
| LATERAL | cross-track MAE | +0.0241 [-0.0008, +0.0518] | +0.0255 [+0.0021, +0.0522] | False/True |
| LATERAL | L1 lateral-accel MAE | -0.3636 [-0.8075, -0.0305] | -0.2777 [-0.5564, -0.0610] | False/False |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0357 [-0.0617, -0.0176] | -0.0342 [-0.0589, -0.0130] | False/False |
| TACTICAL | TAC-LAT | +0.0242 [+0.0117, +0.0379] | +0.0392 [+0.0246, +0.0563] | False/False |
| TACTICAL | TAC-LON | +0.0079 [-0.0146, +0.0309] | +0.0188 [-0.0004, +0.0379] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@30 | -0.3535 [-0.5800, -0.1321] | -0.5714 [-0.8365, -0.3451] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@50 | -0.5618 [-0.9546, -0.1891] | -0.9030 [-1.3691, -0.5029] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@80 | -0.8069 [-1.4700, -0.2119] | -1.4092 [-2.1819, -0.7381] | False/False |
| crawl | M1 crawl share | -0.0272 [-0.0384, -0.0168] | -0.0255 [-0.0375, -0.0155] | False/False |

## LFO075 (CEILING-LEVEL logged level), q = 1.0 (smallest q with point share_closed >= 0.5 at AUC 1.0; 1.0 if none), share_closed 0.099 | 0.097, M1P False, M1FULL False, GUARD False
| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse (direction-aware) s0/s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 | -0.0141 [-0.0407, +0.0170] | -0.0130 [-0.0426, +0.0227] | False/False |
| LONGITUDINAL | speed MAE | +0.0085 [-0.0291, +0.0556] | -0.0036 [-0.0302, +0.0295] | False/False |
| LONGITUDINAL | along-track MAE | -0.0292 [-0.0573, -0.0000] | -0.0284 [-0.0538, -0.0013] | False/False |
| LONGITUDINAL | accel MAE | +0.0573 [-0.0068, +0.1417] | +0.0697 [+0.0089, +0.1445] | False/True |
| LONGITUDINAL | DK-VIOL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | False/False |
| LONGITUDINAL | DK-HW | +0.0060 [-0.0065, +0.0207] | +0.0015 [-0.0012, +0.0053] | False/False |
| LONGITUDINAL | DK-TG | +0.0042 [-0.0011, +0.0132] | +0.0005 [-0.0003, +0.0014] | False/False |
| LATERAL | heading MAE | -0.0111 [-0.0961, +0.0893] | +0.0239 [-0.1123, +0.2430] | False/False |
| LATERAL | yaw-rate MAE | -1.3936 [-2.2313, -0.6887] | -1.1423 [-1.7642, -0.5958] | False/False |
| LATERAL | curvature MAE | -0.0214 [-0.0335, -0.0116] | -0.0204 [-0.0325, -0.0104] | False/False |
| LATERAL | cross-track MAE | +0.0275 [+0.0096, +0.0494] | +0.0238 [+0.0019, +0.0530] | True/True |
| LATERAL | L1 lateral-accel MAE | +0.1398 [-0.0248, +0.3382] | +0.1456 [-0.0040, +0.3562] | False/False |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0287 [-0.0480, -0.0134] | -0.0346 [-0.0596, -0.0132] | False/False |
| TACTICAL | TAC-LAT | +0.0008 [-0.0029, +0.0046] | +0.0075 [+0.0033, +0.0125] | False/False |
| TACTICAL | TAC-LON | +0.0117 [+0.0058, +0.0184] | +0.0154 [+0.0079, +0.0238] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@30 | -0.0704 [-0.1262, -0.0196] | -0.0602 [-0.1503, +0.0274] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@50 | -0.1365 [-0.2399, -0.0436] | -0.0704 [-0.2280, +0.0872] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@80 | -0.2352 [-0.4270, -0.0690] | -0.1036 [-0.3643, +0.1670] | False/False |
| crawl | M1 crawl share | -0.0053 [-0.0088, -0.0020] | -0.0058 [-0.0095, -0.0025] | False/False |

## LFO100 (CEILING-LEVEL logged level), q = 0.7 (smallest q with point share_closed >= 0.5 at AUC 1.0; 1.0 if none), share_closed 0.522 | 0.510, M1P True, M1FULL False, GUARD False
| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse (direction-aware) s0/s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 | -0.0829 [-0.1398, -0.0312] | -0.0839 [-0.1360, -0.0369] | False/False |
| LONGITUDINAL | speed MAE | -0.0759 [-0.1093, -0.0450] | -0.0636 [-0.1013, -0.0284] | False/False |
| LONGITUDINAL | along-track MAE | -0.1031 [-0.1601, -0.0494] | -0.1128 [-0.1694, -0.0646] | False/False |
| LONGITUDINAL | accel MAE | -0.0424 [-0.0747, -0.0124] | +0.0175 [-0.0220, +0.0699] | False/False |
| LONGITUDINAL | DK-VIOL | +0.0084 [+0.0016, +0.0176] | +0.0017 [+0.0000, +0.0054] | True/False |
| LONGITUDINAL | DK-HW | +0.0163 [-0.0401, +0.0671] | -0.0338 [-0.1148, +0.0331] | False/False |
| LONGITUDINAL | DK-TG | +0.0059 [-0.0050, +0.0181] | -0.0082 [-0.0273, +0.0060] | False/False |
| LATERAL | heading MAE | -0.2761 [-0.4217, -0.1382] | -0.1989 [-0.3298, -0.0914] | False/False |
| LATERAL | yaw-rate MAE | -2.5480 [-3.5628, -1.6465] | -2.2249 [-3.3094, -1.3257] | False/False |
| LATERAL | curvature MAE | -0.0264 [-0.0392, -0.0159] | -0.0274 [-0.0422, -0.0153] | False/False |
| LATERAL | cross-track MAE | +0.0204 [+0.0021, +0.0395] | +0.0234 [+0.0088, +0.0392] | True/True |
| LATERAL | L1 lateral-accel MAE | -0.1098 [-0.1764, -0.0514] | +0.0001 [-0.0678, +0.1016] | False/False |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0320 [-0.0564, -0.0147] | -0.0352 [-0.0609, -0.0134] | False/False |
| TACTICAL | TAC-LAT | +0.0158 [+0.0088, +0.0238] | +0.0158 [+0.0071, +0.0254] | False/False |
| TACTICAL | TAC-LON | +0.0438 [+0.0284, +0.0605] | +0.0434 [+0.0279, +0.0605] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@30 | -0.2536 [-0.3540, -0.1525] | -0.2643 [-0.3966, -0.1465] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@50 | -0.4362 [-0.6246, -0.2490] | -0.4178 [-0.6460, -0.2161] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@80 | -0.6992 [-1.0371, -0.3878] | -0.6883 [-1.0799, -0.3478] | False/False |
| crawl | M1 crawl share | -0.0279 [-0.0392, -0.0183] | -0.0304 [-0.0440, -0.0185] | False/False |

## LFOB05 (CEILING-LEVEL logged level), q = 1.0 (smallest q with point share_closed >= 0.5 at AUC 1.0; 1.0 if none), share_closed 0.334 | 0.278, M1P False, M1FULL False, GUARD False
| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse (direction-aware) s0/s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 | -0.0643 [-0.1003, -0.0318] | -0.0632 [-0.0995, -0.0280] | False/False |
| LONGITUDINAL | speed MAE | -0.0363 [-0.0592, -0.0149] | -0.0380 [-0.0610, -0.0137] | False/False |
| LONGITUDINAL | along-track MAE | -0.0727 [-0.1093, -0.0404] | -0.0790 [-0.1159, -0.0431] | False/False |
| LONGITUDINAL | accel MAE | -0.0118 [-0.0373, +0.0123] | +0.0086 [-0.0192, +0.0443] | False/False |
| LONGITUDINAL | DK-VIOL | +0.0034 [+0.0000, +0.0108] | +0.0034 [+0.0000, +0.0088] | False/False |
| LONGITUDINAL | DK-HW | +0.0213 [-0.0259, +0.0652] | -0.0024 [-0.0707, +0.0648] | False/False |
| LONGITUDINAL | DK-TG | +0.0065 [-0.0031, +0.0175] | -0.0007 [-0.0156, +0.0155] | False/False |
| LATERAL | heading MAE | -0.1980 [-0.3238, -0.0820] | -0.1184 [-0.2093, -0.0363] | False/False |
| LATERAL | yaw-rate MAE | -1.8505 [-2.7518, -1.0613] | -1.2546 [-1.8702, -0.7064] | False/False |
| LATERAL | curvature MAE | -0.0222 [-0.0333, -0.0130] | -0.0150 [-0.0258, -0.0067] | False/False |
| LATERAL | cross-track MAE | +0.0069 [-0.0058, +0.0196] | +0.0137 [+0.0009, +0.0281] | False/True |
| LATERAL | L1 lateral-accel MAE | -0.0455 [-0.1170, +0.0235] | -0.0115 [-0.0576, +0.0338] | False/False |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0283 [-0.0494, -0.0138] | -0.0194 [-0.0377, -0.0056] | False/False |
| TACTICAL | TAC-LAT | +0.0117 [+0.0054, +0.0188] | +0.0125 [+0.0054, +0.0204] | False/False |
| TACTICAL | TAC-LON | +0.0142 [+0.0042, +0.0250] | +0.0158 [+0.0075, +0.0254] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@30 | -0.2225 [-0.3117, -0.1450] | -0.2605 [-0.3914, -0.1472] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@50 | -0.3914 [-0.5654, -0.2414] | -0.4325 [-0.6794, -0.2245] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@80 | -0.5782 [-0.9066, -0.3040] | -0.6976 [-1.1437, -0.3378] | False/False |
| crawl | M1 crawl share | -0.0178 [-0.0256, -0.0114] | -0.0165 [-0.0255, -0.0092] | False/False |

## LFOB10 (CEILING-LEVEL logged level), q = 1.0 (smallest q with point share_closed >= 0.5 at AUC 1.0; 1.0 if none), share_closed 0.480 | 0.418, M1P False, M1FULL False, GUARD False
| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse (direction-aware) s0/s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 | -0.0838 [-0.1367, -0.0354] | -0.0873 [-0.1403, -0.0396] | False/False |
| LONGITUDINAL | speed MAE | -0.0561 [-0.0923, -0.0208] | -0.0537 [-0.0845, -0.0246] | False/False |
| LONGITUDINAL | along-track MAE | -0.1048 [-0.1612, -0.0551] | -0.1156 [-0.1715, -0.0658] | False/False |
| LONGITUDINAL | accel MAE | -0.0017 [-0.0350, +0.0328] | +0.0298 [+0.0026, +0.0620] | False/True |
| LONGITUDINAL | DK-VIOL | +0.0101 [+0.0016, +0.0225] | +0.0034 [+0.0000, +0.0086] | True/False |
| LONGITUDINAL | DK-HW | +0.0341 [-0.0383, +0.1065] | -0.0313 [-0.1174, +0.0411] | False/False |
| LONGITUDINAL | DK-TG | +0.0088 [-0.0055, +0.0234] | -0.0089 [-0.0277, +0.0060] | False/False |
| LATERAL | heading MAE | -0.1718 [-0.2922, -0.0490] | -0.1371 [-0.2298, -0.0469] | False/False |
| LATERAL | yaw-rate MAE | -2.0281 [-2.8957, -1.2758] | -1.8409 [-2.7857, -1.0805] | False/False |
| LATERAL | curvature MAE | -0.0263 [-0.0395, -0.0158] | -0.0266 [-0.0402, -0.0152] | False/False |
| LATERAL | cross-track MAE | +0.0217 [+0.0018, +0.0425] | +0.0242 [+0.0070, +0.0420] | True/True |
| LATERAL | L1 lateral-accel MAE | -0.0129 [-0.0965, +0.0722] | +0.0133 [-0.0404, +0.0809] | False/False |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0314 [-0.0543, -0.0147] | -0.0308 [-0.0541, -0.0114] | False/False |
| TACTICAL | TAC-LAT | +0.0108 [+0.0033, +0.0192] | +0.0188 [+0.0104, +0.0284] | False/False |
| TACTICAL | TAC-LON | +0.0283 [+0.0129, +0.0438] | +0.0279 [+0.0150, +0.0421] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@30 | -0.2531 [-0.3730, -0.1459] | -0.2730 [-0.3961, -0.1632] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@50 | -0.4282 [-0.6397, -0.2342] | -0.4493 [-0.6996, -0.2432] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@80 | -0.6340 [-1.0075, -0.2987] | -0.7202 [-1.1494, -0.3685] | False/False |
| crawl | M1 crawl share | -0.0256 [-0.0366, -0.0166] | -0.0249 [-0.0377, -0.0146] | False/False |
