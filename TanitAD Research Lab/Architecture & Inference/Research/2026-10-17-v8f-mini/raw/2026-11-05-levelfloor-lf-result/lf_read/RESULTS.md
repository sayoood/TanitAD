# LF read: arms LF050 and LF075 (the LLR heads as a floor on O's along-track progress, planner C0), T1-kin, generated 2026-11-05T23:39:51+0100

**Headline: P fails at both doses and M1 is separated-better at dose 050 and 075: the floor binds but the head's estimate is too weak to lift the crawl; the level estimator is the bottleneck. Its levers are motion from vision (option (c): T2 re-perception, the A40) or a stronger estimator trained on more data, reported to the PI with the binding and v_f-error instruments. Two doses were read (kappa 0.50 and kappa 0.75): a counted DOMINANT names its dose and says that two were read (A3 card section 4, multiplicity).**

* dose 050 (kappa 0.50): P-FAILS-M1-MOVED -> the floor binds but the head's estimate is too weak to lift the crawl: the level estimator is the bottleneck (levers: motion from vision, option (c) T2 re-perception / the A40, or a stronger estimator on more data)
* dose 075 (kappa 0.75): P-FAILS-M1-MOVED -> the floor binds but the head's estimate is too weak to lift the crawl: the level estimator is the bottleneck (levers: motion from vision, option (c) T2 re-perception / the A40, or a stronger estimator on more data)

Tier: T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2). Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose); answers another draw of EPISODES only. `*` = interval excludes 0. STRATEGIC: PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it

`hold_now` is not read (a frozen frame makes v_hat constant: no information about a frame-reading floor; A3 card section 3).


## Dose 050 (kappa 0.50): **P-FAILS-M1-MOVED**

Next: the floor binds but the head's estimate is too weak to lift the crawl: the level estimator is the bottleneck (levers: motion from vision, option (c) T2 re-perception / the A40, or a stronger estimator on more data).


Reach control (F binds on >= 1 % of the replans inside the M1 crawl ticks, each seed): s0 0.250 of 244 ticks; s1 0.324 of 247 ticks -> **True**

| rule | s0 | s1 | holds |
|---|---|---|---|
| P: M1 separated-better and share >= 0.5 | sep False, share 0.062 | sep True, share 0.056 | **False** |
| G: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 +0.1089, worse: ADE6, cross-track MAE | ADE6 +0.0636, worse: DK-VIOL, cross-track MAE | **False** |
| F: abs(RILF-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0033 vs 0.0362 | 0.0033 vs 0.0038 | **False** |
| LEAK-BORNE: logged share - reindexed share > half the logged share | logged 0.041, reindexed 0.062 | logged 0.057, reindexed 0.056 | **False** |
| D: RILFC - P10 read A (D1, D2, D3 four families) + P + G | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | **False** |

Beside: P on the logged scene False; D three-family form False; M1 moved (RILF - RIC0 separated-better at >= 1 seed) True.

D failing terms: s0: D2 (separated-worse: M1 crawl share); s1: D2 (separated-worse: M1 crawl share)

Harm rows: none


### 19 rows, dose 050: RILF - RIC0 (reindexed, the deciding contrast)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.1089 [+0.0126, +0.2330] * | +0.0636 [-0.0109, +0.1509] |
| speed MAE | LONGITUDINAL | +0.0372 [-0.0213, +0.1037] | +0.0164 [-0.0210, +0.0608] |
| along-track MAE | LONGITUDINAL | +0.0792 [-0.0030, +0.1847] | +0.0390 [-0.0191, +0.1069] |
| accel MAE | LONGITUDINAL | -0.0369 [-0.1435, +0.0678] | -0.0107 [-0.0682, +0.0439] |
| DK-VIOL | LONGITUDINAL | +0.0168 [-0.0015, +0.0415] | +0.0134 [+0.0032, +0.0271] * |
| DK-HW | LONGITUDINAL | +0.0691 [-0.0303, +0.2087] | +0.0533 [-0.0568, +0.1747] |
| DK-TG | LONGITUDINAL | +0.0413 [-0.0501, +0.1691] | +0.0119 [-0.1030, +0.1128] |
| heading MAE | LATERAL | -0.6351 [-1.1832, -0.1782] * | -0.8080 [-1.5576, -0.2906] * |
| yaw-rate MAE | LATERAL | -4.6934 [-8.2309, -2.2024] * | -5.0548 [-9.0122, -2.2982] * |
| curvature MAE | LATERAL | -0.0560 [-0.0853, -0.0328] * | -0.0575 [-0.0949, -0.0298] * |
| cross-track MAE | LATERAL | +0.0569 [+0.0150, +0.1055] * | +0.0404 [+0.0022, +0.0956] * |
| L1 lateral-accel MAE | LATERAL | -0.1490 [-0.5985, +0.2011] | -0.1097 [-0.3973, +0.1000] |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0263 [-0.0456, -0.0122] * | -0.0330 [-0.0589, -0.0105] * |
| TAC-LAT | TACTICAL | +0.0150 [+0.0071, +0.0234] * | +0.0113 [+0.0046, +0.0183] * |
| TAC-LON | TACTICAL | -0.0008 [-0.0075, +0.0058] | +0.0054 [-0.0008, +0.0125] |
| S-POS@30 | STRATEGIC | -0.1537 [-0.3231, -0.0034] * | -0.3017 [-0.4766, -0.1476] * |
| S-POS@50 | STRATEGIC | -0.2840 [-0.5835, -0.0258] * | -0.4403 [-0.7029, -0.2010] * |
| S-POS@80 | STRATEGIC | -0.4956 [-0.9704, -0.0967] * | -0.6414 [-1.0832, -0.2618] * |
| M1 crawl share | crawl | -0.0033 [-0.0072, +0.0002] | -0.0033 [-0.0065, -0.0005] * |

### 19 rows, dose 050: RILFC - P10 (reindexed composite, D)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -1.0020 [-1.1991, -0.8223] * | -0.9639 [-1.1531, -0.7813] * |
| speed MAE | LONGITUDINAL | -0.7743 [-0.9845, -0.5920] * | -0.7848 [-0.9912, -0.5986] * |
| along-track MAE | LONGITUDINAL | -0.5278 [-0.6881, -0.3770] * | -0.4670 [-0.6371, -0.3111] * |
| accel MAE | LONGITUDINAL | -1.0575 [-1.4967, -0.6742] * | -1.0995 [-1.5404, -0.7259] * |
| DK-VIOL | LONGITUDINAL | +0.0286 [+0.0000, +0.0654] | +0.0084 [-0.0146, +0.0361] |
| DK-HW | LONGITUDINAL | -0.2802 [-0.7035, +0.0913] | -0.3778 [-0.7353, -0.0956] * |
| DK-TG | LONGITUDINAL | -0.0199 [-0.1636, +0.1725] | -0.0446 [-0.1446, +0.0729] |
| heading MAE | LATERAL | -2.2027 [-3.2680, -1.0222] * | -1.9493 [-3.0048, -0.8582] * |
| yaw-rate MAE | LATERAL | -12.8209 [-18.2818, -8.2164] * | -12.6836 [-18.0626, -8.1445] * |
| curvature MAE | LATERAL | -0.0415 [-0.0595, -0.0262] * | -0.0391 [-0.0566, -0.0241] * |
| cross-track MAE | LATERAL | -0.8634 [-1.0119, -0.7290] * | -0.8690 [-1.0051, -0.7440] * |
| L1 lateral-accel MAE | LATERAL | -3.4573 [-5.0047, -2.1341] * | -3.5298 [-5.0640, -2.2088] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0018 [-0.0046, +0.0010] | -0.0011 [-0.0042, +0.0021] |
| TAC-LAT | TACTICAL | +0.2059 [+0.1682, +0.2450] * | +0.1551 [+0.1124, +0.1987] * |
| TAC-LON | TACTICAL | +0.1501 [+0.0867, +0.2109] * | +0.1476 [+0.0892, +0.2043] * |
| S-POS@30 | STRATEGIC | -0.8046 [-1.0986, -0.5328] * | -0.8829 [-1.1801, -0.6092] * |
| S-POS@50 | STRATEGIC | -3.1347 [-4.0292, -2.3209] * | -3.3025 [-4.1376, -2.5121] * |
| S-POS@80 | STRATEGIC | -7.9355 [-9.8526, -6.1010] * | -8.7764 [-10.6831, -6.9184] * |
| M1 crawl share | crawl | +0.0549 [+0.0356, +0.0775] * | +0.0601 [+0.0375, +0.0866] * |

### 19 rows, dose 050: LF - C0 (logged, LEAK-BORNE only)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.1109 [+0.0153, +0.2341] * | +0.0641 [-0.0110, +0.1514] |
| speed MAE | LONGITUDINAL | +0.0374 [-0.0214, +0.1030] | +0.0163 [-0.0198, +0.0597] |
| along-track MAE | LONGITUDINAL | +0.0818 [-0.0001, +0.1867] | +0.0389 [-0.0206, +0.1083] |
| accel MAE | LONGITUDINAL | -0.0380 [-0.1479, +0.0711] | -0.0221 [-0.0788, +0.0300] |
| DK-VIOL | LONGITUDINAL | +0.0168 [-0.0015, +0.0415] | +0.0134 [+0.0032, +0.0271] * |
| DK-HW | LONGITUDINAL | +0.0775 [-0.0249, +0.2127] | +0.0499 [-0.0598, +0.1715] |
| DK-TG | LONGITUDINAL | +0.0441 [-0.0457, +0.1663] | +0.0076 [-0.1073, +0.1087] |
| heading MAE | LATERAL | -0.6228 [-1.1701, -0.1628] * | -0.8147 [-1.5544, -0.3084] * |
| yaw-rate MAE | LATERAL | -4.8866 [-8.3496, -2.4172] * | -5.3587 [-9.4298, -2.4715] * |
| curvature MAE | LATERAL | -0.0542 [-0.0818, -0.0319] * | -0.0593 [-0.0980, -0.0311] * |
| cross-track MAE | LATERAL | +0.0557 [+0.0134, +0.1048] * | +0.0420 [+0.0038, +0.0973] * |
| L1 lateral-accel MAE | LATERAL | -0.1581 [-0.6163, +0.2008] | -0.1456 [-0.4431, +0.0661] |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0261 [-0.0443, -0.0125] * | -0.0279 [-0.0522, -0.0083] * |
| TAC-LAT | TACTICAL | +0.0129 [+0.0050, +0.0209] * | +0.0113 [+0.0050, +0.0187] * |
| TAC-LON | TACTICAL | +0.0000 [-0.0079, +0.0079] | +0.0021 [-0.0042, +0.0088] |
| S-POS@30 | STRATEGIC | -0.1550 [-0.3242, -0.0054] * | -0.2964 [-0.4632, -0.1446] * |
| S-POS@50 | STRATEGIC | -0.2847 [-0.5762, -0.0306] * | -0.4195 [-0.6902, -0.1767] * |
| S-POS@80 | STRATEGIC | -0.4975 [-0.9594, -0.1113] * | -0.5955 [-1.0331, -0.2106] * |
| M1 crawl share | crawl | -0.0022 [-0.0059, +0.0014] | -0.0034 [-0.0063, -0.0010] * |

## Dose 075 (kappa 0.75): **P-FAILS-M1-MOVED**

Next: the floor binds but the head's estimate is too weak to lift the crawl: the level estimator is the bottleneck (levers: motion from vision, option (c) T2 re-perception / the A40, or a stronger estimator on more data).


Reach control (F binds on >= 1 % of the replans inside the M1 crawl ticks, each seed): s0 0.455 of 200 ticks; s1 0.462 of 199 ticks -> **True**

| rule | s0 | s1 | holds |
|---|---|---|---|
| P: M1 separated-better and share >= 0.5 | sep True, share 0.395 | sep True, share 0.392 | **False** |
| G: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 +0.3683, worse: ADE6, DK-HW, DK-VIOL, along-track MAE, cross-track MAE, speed MAE | ADE6 +0.2172, worse: ADE6, DK-HW, DK-VIOL, along-track MAE, cross-track MAE | **False** |
| F: abs(RILF-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0211 vs 0.0362 | 0.0234 vs 0.0038 | **False** |
| LEAK-BORNE: logged share - reindexed share > half the logged share | logged 0.381, reindexed 0.395 | logged 0.400, reindexed 0.392 | **False** |
| D: RILFC - P10 read A (D1, D2, D3 four families) + P + G | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | **False** |

Beside: P on the logged scene False; D three-family form False; M1 moved (RILF - RIC0 separated-better at >= 1 seed) True.

D failing terms: s0: D2 (separated-worse: DK-VIOL, M1 crawl share); s1: D2 (separated-worse: M1 crawl share)

Harm rows: none


### 19 rows, dose 075: RILF - RIC0 (reindexed, the deciding contrast)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.3683 [+0.2134, +0.5400] * | +0.2172 [+0.0971, +0.3406] * |
| speed MAE | LONGITUDINAL | +0.1250 [+0.0378, +0.2269] * | +0.0379 [-0.0199, +0.0938] |
| along-track MAE | LONGITUDINAL | +0.2753 [+0.1348, +0.4183] * | +0.1657 [+0.0462, +0.2827] * |
| accel MAE | LONGITUDINAL | -0.0357 [-0.1515, +0.0847] | -0.0379 [-0.1012, +0.0252] |
| DK-VIOL | LONGITUDINAL | +0.0269 [+0.0058, +0.0541] * | +0.0269 [+0.0084, +0.0513] * |
| DK-HW | LONGITUDINAL | +0.2961 [+0.0513, +0.6715] * | +0.1978 [+0.0052, +0.4163] * |
| DK-TG | LONGITUDINAL | +0.0779 [-0.0138, +0.2087] | +0.0351 [-0.0819, +0.1391] |
| heading MAE | LATERAL | -0.9023 [-1.5232, -0.3415] * | -0.9247 [-1.7145, -0.3716] * |
| yaw-rate MAE | LATERAL | -7.4724 [-11.1972, -4.6626] * | -7.5591 [-11.8067, -4.3665] * |
| curvature MAE | LATERAL | -0.0762 [-0.1084, -0.0485] * | -0.0800 [-0.1217, -0.0458] * |
| cross-track MAE | LATERAL | +0.1589 [+0.0843, +0.2533] * | +0.0868 [+0.0403, +0.1426] * |
| L1 lateral-accel MAE | LATERAL | -0.2673 [-0.7324, +0.1396] | -0.3303 [-0.6195, -0.1128] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0380 [-0.0654, -0.0184] * | -0.0403 [-0.0695, -0.0152] * |
| TAC-LAT | TACTICAL | +0.0283 [+0.0158, +0.0421] * | +0.0313 [+0.0184, +0.0446] * |
| TAC-LON | TACTICAL | +0.0013 [-0.0204, +0.0229] | +0.0275 [+0.0096, +0.0475] * |
| S-POS@30 | STRATEGIC | -0.3952 [-0.6138, -0.1979] * | -0.5271 [-0.7941, -0.2974] * |
| S-POS@50 | STRATEGIC | -0.7201 [-1.1097, -0.3769] * | -0.8314 [-1.2779, -0.4553] * |
| S-POS@80 | STRATEGIC | -1.2356 [-1.8567, -0.6888] * | -1.2768 [-2.0227, -0.6579] * |
| M1 crawl share | crawl | -0.0211 [-0.0334, -0.0112] * | -0.0234 [-0.0363, -0.0131] * |

### 19 rows, dose 075: RILFC - P10 (reindexed composite, D)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.7526 [-0.9765, -0.5462] * | -0.8043 [-1.0031, -0.6152] * |
| speed MAE | LONGITUDINAL | -0.7013 [-0.9073, -0.5211] * | -0.7503 [-0.9614, -0.5681] * |
| along-track MAE | LONGITUDINAL | -0.3066 [-0.4944, -0.1278] * | -0.3404 [-0.5153, -0.1736] * |
| accel MAE | LONGITUDINAL | -1.0515 [-1.4899, -0.6825] * | -1.1017 [-1.5373, -0.7323] * |
| DK-VIOL | LONGITUDINAL | +0.0387 [+0.0064, +0.0768] * | +0.0185 [-0.0068, +0.0452] |
| DK-HW | LONGITUDINAL | -0.0652 [-0.5703, +0.4591] | -0.2033 [-0.5906, +0.1355] |
| DK-TG | LONGITUDINAL | +0.0128 [-0.1325, +0.2118] | -0.0154 [-0.1145, +0.0964] |
| heading MAE | LATERAL | -2.3322 [-3.3822, -1.1386] * | -1.9624 [-3.0219, -0.8111] * |
| yaw-rate MAE | LATERAL | -13.3281 [-18.7954, -8.6686] * | -13.1525 [-18.4533, -8.6073] * |
| curvature MAE | LATERAL | -0.0464 [-0.0646, -0.0306] * | -0.0441 [-0.0617, -0.0286] * |
| cross-track MAE | LATERAL | -0.7971 [-0.9497, -0.6572] * | -0.8188 [-0.9576, -0.6892] * |
| L1 lateral-accel MAE | LATERAL | -3.4544 [-4.9967, -2.1195] * | -3.5205 [-5.0108, -2.2177] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0045 [-0.0072, -0.0019] * | -0.0036 [-0.0064, -0.0009] * |
| TAC-LAT | TACTICAL | +0.2134 [+0.1751, +0.2519] * | +0.1701 [+0.1284, +0.2147] * |
| TAC-LON | TACTICAL | +0.1538 [+0.0939, +0.2107] * | +0.1726 [+0.1159, +0.2280] * |
| S-POS@30 | STRATEGIC | -0.8442 [-1.1577, -0.5656] * | -0.9916 [-1.2938, -0.7119] * |
| S-POS@50 | STRATEGIC | -3.1794 [-4.0872, -2.3779] * | -3.4455 [-4.3221, -2.6290] * |
| S-POS@80 | STRATEGIC | -7.9963 [-9.8842, -6.1361] * | -9.0507 [-11.0584, -7.1729] * |
| M1 crawl share | crawl | +0.0380 [+0.0206, +0.0575] * | +0.0401 [+0.0211, +0.0619] * |

### 19 rows, dose 075: LF - C0 (logged, LEAK-BORNE only)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.3674 [+0.2135, +0.5384] * | +0.2162 [+0.0947, +0.3374] * |
| speed MAE | LONGITUDINAL | +0.1230 [+0.0372, +0.2250] * | +0.0356 [-0.0217, +0.0922] |
| along-track MAE | LONGITUDINAL | +0.2766 [+0.1357, +0.4192] * | +0.1634 [+0.0445, +0.2799] * |
| accel MAE | LONGITUDINAL | -0.0355 [-0.1515, +0.0865] | -0.0347 [-0.0990, +0.0288] |
| DK-VIOL | LONGITUDINAL | +0.0286 [+0.0065, +0.0574] * | +0.0269 [+0.0084, +0.0513] * |
| DK-HW | LONGITUDINAL | +0.2988 [+0.0559, +0.6675] * | +0.1617 [-0.0220, +0.3745] |
| DK-TG | LONGITUDINAL | +0.0751 [-0.0140, +0.2007] | +0.0336 [-0.0835, +0.1396] |
| heading MAE | LATERAL | -0.8930 [-1.5167, -0.3304] * | -0.9089 [-1.6773, -0.3659] * |
| yaw-rate MAE | LATERAL | -7.4450 [-11.1778, -4.6507] * | -7.5748 [-11.8718, -4.3680] * |
| curvature MAE | LATERAL | -0.0744 [-0.1047, -0.0478] * | -0.0805 [-0.1226, -0.0465] * |
| cross-track MAE | LATERAL | +0.1559 [+0.0813, +0.2494] * | +0.0882 [+0.0401, +0.1436] * |
| L1 lateral-accel MAE | LATERAL | -0.2601 [-0.7239, +0.1381] | -0.3165 [-0.6090, -0.0977] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0373 [-0.0650, -0.0180] * | -0.0360 [-0.0638, -0.0136] * |
| TAC-LAT | TACTICAL | +0.0254 [+0.0125, +0.0388] * | +0.0300 [+0.0171, +0.0434] * |
| TAC-LON | TACTICAL | +0.0042 [-0.0175, +0.0254] | +0.0267 [+0.0083, +0.0475] * |
| S-POS@30 | STRATEGIC | -0.4201 [-0.6336, -0.2234] * | -0.5331 [-0.7886, -0.3108] * |
| S-POS@50 | STRATEGIC | -0.7591 [-1.1359, -0.4211] * | -0.8443 [-1.2811, -0.4757] * |
| S-POS@80 | STRATEGIC | -1.2884 [-1.9069, -0.7443] * | -1.2962 [-2.0224, -0.6956] * |
| M1 crawl share | crawl | -0.0203 [-0.0327, -0.0102] * | -0.0238 [-0.0370, -0.0133] * |

## Instruments (from the tick dumps; beside, never deciding)

| rollout | bind overall | bind in M1 ticks (n) | cap overall | cap in M1 ticks | mean lift where bound, m/s | v_f vs logged: MAE m/s, share v_f > logged |
|---|---|---|---|---|---|---|
| RILF050_s0 | 0.075 | 0.250 (244) | 0.013 | 0.000 | 0.563 | 6.322, 0.066 (n 5793) |
| RILF050_s1 | 0.079 | 0.324 (247) | 0.017 | 0.000 | 0.509 | 6.467, 0.055 (n 5707) |
| RILF050C_s0 | 0.079 | 0.254 (244) | 0.014 | 0.000 | 0.555 | 6.324, 0.065 (n 5790) |
| RILF050C_s1 | 0.081 | 0.332 (244) | 0.017 | 0.000 | 0.517 | 6.465, 0.054 (n 5709) |
| RILF075_s0 | 0.187 | 0.455 (200) | 0.069 | 0.005 | 0.666 | 3.894, 0.191 (n 5793) |
| RILF075_s1 | 0.187 | 0.462 (199) | 0.050 | 0.000 | 0.662 | 3.979, 0.174 (n 5710) |
| RILF075C_s0 | 0.188 | 0.458 (201) | 0.068 | 0.005 | 0.674 | 3.894, 0.190 (n 5789) |
| RILF075C_s1 | 0.187 | 0.475 (200) | 0.050 | 0.000 | 0.661 | 3.978, 0.173 (n 5708) |
| LF050_s0 | 0.076 | 0.257 (245) | 0.013 | 0.000 | 0.552 | 6.309, 0.066 (n 5809) |
| LF050_s1 | 0.079 | 0.305 (246) | 0.017 | 0.000 | 0.520 | 6.449, 0.055 (n 5725) |
| LF050C_s0 | 0.081 | 0.273 (245) | 0.013 | 0.000 | 0.542 | 6.309, 0.066 (n 5809) |
| LF050C_s1 | 0.081 | 0.332 (244) | 0.017 | 0.000 | 0.535 | 6.449, 0.055 (n 5725) |
| LF075_s0 | 0.186 | 0.468 (201) | 0.068 | 0.000 | 0.674 | 3.887, 0.193 (n 5809) |
| LF075_s1 | 0.186 | 0.442 (197) | 0.050 | 0.000 | 0.662 | 3.972, 0.177 (n 5725) |
| LF075C_s0 | 0.189 | 0.493 (201) | 0.068 | 0.000 | 0.678 | 3.887, 0.193 (n 5809) |
| LF075C_s1 | 0.187 | 0.457 (197) | 0.050 | 0.000 | 0.665 | 3.972, 0.177 (n 5725) |

## Controls

| control | PASS |
|---|---|
| patched_tool_pin | True |
| rollout_RILF050_s0 | True |
| rollout_RILF050_s1 | True |
| rollout_RILF050C_s0 | True |
| rollout_RILF050C_s1 | True |
| rollout_RILF075_s0 | True |
| rollout_RILF075_s1 | True |
| rollout_RILF075C_s0 | True |
| rollout_RILF075C_s1 | True |
| rollout_LF050_s0 | True |
| rollout_LF050_s1 | True |
| rollout_LF050C_s0 | True |
| rollout_LF050C_s1 | True |
| rollout_LF075_s0 | True |
| rollout_LF075_s1 | True |
| rollout_LF075C_s0 | True |
| rollout_LF075C_s1 | True |
| one_checkpoint_per_seed | True |
| LF_PRECHECK | True |
| card_a3 | True |
| cl_card_rules | True |
| llr_read_py | True |
| reused_arrays | True |

## Variance named

* episodes: answered by the paired episode-cluster bootstrap (every interval here)
* training_run: O is C0's own checkpoint in both arms of every LF - C0 contrast, so O's training variance is absent from it; the heads come from one LLR run per seed (not measured); rule F reads against the C0REP floor
* inference: the planner and F are deterministic: no inference seed applies
