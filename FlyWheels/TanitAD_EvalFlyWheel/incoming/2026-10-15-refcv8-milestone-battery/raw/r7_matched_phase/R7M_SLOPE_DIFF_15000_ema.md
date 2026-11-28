# R7 matched-phase contrasts at N = 15000 (S2/S6 heading-error slope; PREREG_R7_MATCHED_PHASE_TURN.md)

* generated 2026-10-19T18:27:35; trees: `R7` = C:/Users/Admin/refcv8_eval_archive/battery/r7step15000_unc; `EMA8` = C:/Users/Admin/refcv8_eval_archive/battery/step15000ema_unc
* estimator: episode-cluster bootstrap over the eval episodes, models FIXED (not another training run; not another inference run); B 2000; tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
* CONTROL vs `heading_drift.shrinkage` on the same arms (must be < 1e-9): S2|EMA8: 0.00e+00 (ok); S6|EMA8: 0.00e+00 (ok)
* baseline identity (embedded refcv7-50,400 arm, max abs diff vs the R7 tree; must be 0): S2|EMA8|s0 0.0e+00, S2|EMA8|s1 0.0e+00, S6|EMA8|s0 0.0e+00, S6|EMA8|s1 0.0e+00

## S2 ([0.5, 1.0, 1.5, 2.0] s), 4754 windows / 139 episodes


### S2 -- EMA8 -- curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1575 / 1575 | -0.0883 | -0.2164 | -0.1324 | -0.1281 [-0.1709, -0.0868] SEP | +0.0840 [+0.0483, +0.1225] SEP | -0.0440 [-0.0750, -0.0190] SEP | 2.91 [1.79, 6.28] |
| seed 1 | 1575 / 1575 | -0.0804 | -0.2188 | -0.1330 | -0.1384 [-0.1805, -0.0984] SEP | +0.0858 [+0.0502, +0.1236] SEP | -0.0526 [-0.0864, -0.0256] SEP | 2.63 [1.70, 5.02] |
| pooled (2-seed mean heading error) | 1575 / 1575 | -0.0844 | -0.2176 | -0.1327 | -0.1332 [-0.1748, -0.0923] SEP | +0.0849 [+0.0492, +0.1230] SEP | -0.0483 [-0.0800, -0.0225] SEP | 2.76 [1.74, 5.46] |

### S2 -- EMA8 -- curved (GT |yaw| >= 0.02) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1593 / 4745 | -0.1501 | -0.2588 | -0.1918 | -0.1087 [-0.1504, -0.0694] SEP | +0.0670 [+0.0335, +0.1041] SEP | -0.0418 [-0.0688, -0.0188] SEP | 2.60 [1.63, 5.37] |
| seed 1 | 1593 / 4745 | -0.1430 | -0.2617 | -0.1921 | -0.1187 [-0.1589, -0.0825] SEP | +0.0695 [+0.0371, +0.1052] SEP | -0.0492 [-0.0790, -0.0243] SEP | 2.41 [1.59, 4.45] |
| pooled (2-seed mean heading error) | 1593 / 4743 | -0.1465 | -0.2602 | -0.1920 | -0.1137 [-0.1540, -0.0763] SEP | +0.0683 [+0.0355, +0.1046] SEP | -0.0455 [-0.0737, -0.0221] SEP | 2.50 [1.61, 4.80] |

### S2 -- EMA8 -- sharp (>= 0.15) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 363 / 363 | -0.0968 | -0.2336 | -0.1397 | -0.1368 [-0.1870, -0.0905] SEP | +0.0939 [+0.0520, +0.1386] SEP | -0.0429 [-0.0791, -0.0130] SEP | 3.19 [1.82, 8.99] |
| seed 1 | 363 / 363 | -0.0874 | -0.2366 | -0.1422 | -0.1491 [-0.1980, -0.1049] SEP | +0.0944 [+0.0553, +0.1365] SEP | -0.0547 [-0.0937, -0.0227] SEP | 2.73 [1.72, 5.74] |
| pooled (2-seed mean heading error) | 363 / 363 | -0.0921 | -0.2351 | -0.1409 | -0.1430 [-0.1924, -0.0983] SEP | +0.0942 [+0.0541, +0.1366] SEP | -0.0488 [-0.0867, -0.0176] SEP | 2.93 [1.78, 6.76] |

### S2 -- EMA8 -- sharp (>= 0.15) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 366 / 1094 | -0.1084 | -0.2283 | -0.1512 | -0.1199 [-0.1682, -0.0745] SEP | +0.0771 [+0.0361, +0.1206] SEP | -0.0428 [-0.0771, -0.0141] SEP | 2.80 [1.61, 7.18] |
| seed 1 | 366 / 1094 | -0.0998 | -0.2327 | -0.1534 | -0.1329 [-0.1791, -0.0907] SEP | +0.0793 [+0.0413, +0.1206] SEP | -0.0536 [-0.0909, -0.0227] SEP | 2.48 [1.56, 5.17] |
| pooled (2-seed mean heading error) | 366 / 1094 | -0.1041 | -0.2305 | -0.1523 | -0.1264 [-0.1732, -0.0841] SEP | +0.0782 [+0.0396, +0.1197] SEP | -0.0482 [-0.0838, -0.0187] SEP | 2.62 [1.60, 5.97] |

### S2 -- EMA8 -- curved & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 627 / 627 | -0.0906 | -0.2027 | -0.1383 | -0.1121 [-0.1509, -0.0719] SEP | +0.0643 [+0.0260, +0.0965] SEP | -0.0478 [-0.1011, -0.0015] SEP | 2.35 [1.11, 12.26] |
| seed 1 | 627 / 627 | -0.0945 | -0.2025 | -0.1411 | -0.1080 [-0.1421, -0.0705] SEP | +0.0614 [+0.0211, +0.0954] SEP | -0.0466 [-0.0974, -0.0031] SEP | 2.32 [1.15, 13.40] |
| pooled (2-seed mean heading error) | 627 / 627 | -0.0925 | -0.2026 | -0.1397 | -0.1100 [-0.1454, -0.0722] SEP | +0.0629 [+0.0245, +0.0955] SEP | -0.0472 [-0.0991, -0.0027] SEP | 2.33 [1.12, 12.11] |

### S2 -- EMA8 -- curved & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 627 / 1881 | -0.0979 | -0.1992 | -0.1453 | -0.1014 [-0.1367, -0.0651] SEP | +0.0540 [+0.0177, +0.0850] SEP | -0.0474 [-0.0966, -0.0041] SEP | 2.14 [1.08, 9.01] |
| seed 1 | 627 / 1881 | -0.1007 | -0.2008 | -0.1488 | -0.1001 [-0.1326, -0.0656] SEP | +0.0520 [+0.0131, +0.0841] SEP | -0.0481 [-0.0958, -0.0072] SEP | 2.08 [1.11, 8.73] |
| pooled (2-seed mean heading error) | 627 / 1881 | -0.0993 | -0.2000 | -0.1470 | -0.1008 [-0.1343, -0.0660] SEP | +0.0530 [+0.0157, +0.0841] SEP | -0.0478 [-0.0961, -0.0057] SEP | 2.11 [1.10, 8.81] |

### S2 -- EMA8 -- sharp & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 164 / 164 | -0.0901 | -0.2005 | -0.1319 | -0.1104 [-0.1540, -0.0672] SEP | +0.0686 [+0.0266, +0.1039] SEP | -0.0419 [-0.1011, +0.0074] | 2.64 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 164 / 164 | -0.0924 | -0.2002 | -0.1354 | -0.1078 [-0.1465, -0.0668] SEP | +0.0648 [+0.0202, +0.1012] SEP | -0.0430 [-0.0991, +0.0046] | 2.51 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 164 / 164 | -0.0912 | -0.2004 | -0.1337 | -0.1091 [-0.1504, -0.0674] SEP | +0.0667 [+0.0239, +0.1018] SEP | -0.0424 [-0.1008, +0.0060] | 2.57 (CI undefined: the CI of the denominator (gap) includes 0) |

### S2 -- EMA8 -- sharp & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 164 / 492 | -0.0947 | -0.1881 | -0.1351 | -0.0934 [-0.1336, -0.0532] SEP | +0.0531 [+0.0126, +0.0872] SEP | -0.0404 [-0.0972, +0.0069] | 2.31 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 164 / 492 | -0.0963 | -0.1897 | -0.1388 | -0.0933 [-0.1296, -0.0546] SEP | +0.0509 [+0.0087, +0.0849] SEP | -0.0424 [-0.0955, +0.0034] | 2.20 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 164 / 492 | -0.0955 | -0.1889 | -0.1369 | -0.0934 [-0.1317, -0.0545] SEP | +0.0520 [+0.0109, +0.0860] SEP | -0.0414 [-0.0964, +0.0052] | 2.26 (CI undefined: the CI of the denominator (gap) includes 0) |

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s), 3668 windows / 139 episodes


### S6 -- EMA8 -- curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1317 / 1317 | -0.2400 | -0.3728 | -0.2585 | -0.1328 [-0.1888, -0.0825] SEP | +0.1143 [+0.0593, +0.1751] SEP | -0.0185 [-0.0658, +0.0158] | 7.18 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 1312 / 1312 | -0.2428 | -0.3441 | -0.2550 | -0.1014 [-0.1515, -0.0530] SEP | +0.0891 [+0.0320, +0.1466] SEP | -0.0122 [-0.0609, +0.0258] | 8.28 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 1311 / 1311 | -0.2414 | -0.3581 | -0.2562 | -0.1167 [-0.1665, -0.0703] SEP | +0.1018 [+0.0481, +0.1585] SEP | -0.0149 [-0.0625, +0.0213] | 7.85 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- curved (GT |yaw| >= 0.02) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1414 / 6832 | -0.1689 | -0.2979 | -0.2279 | -0.1290 [-0.1875, -0.0680] SEP | +0.0700 [-0.0528, +0.1596] | -0.0590 [-0.1351, -0.0044] SEP | 2.19 [0.45, 16.82] |
| seed 1 | 1414 / 6826 | -0.1800 | -0.2965 | -0.1690 | -0.1164 [-0.1659, -0.0610] SEP | +0.1275 [+0.0627, +0.2031] SEP | +0.0111 [-0.0651, +0.1183] | -10.50 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 1414 / 6821 | -0.1744 | -0.2970 | -0.1982 | -0.1225 [-0.1765, -0.0654] SEP | +0.0988 [+0.0456, +0.1542] SEP | -0.0238 [-0.0719, +0.0156] | 5.15 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp (>= 0.15) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 284 / 284 | -0.2540 | -0.4088 | -0.2897 | -0.1547 [-0.2275, -0.0929] SEP | +0.1191 [+0.0517, +0.1959] SEP | -0.0357 [-0.1038, +0.0082] | 4.34 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 283 / 283 | -0.2586 | -0.3742 | -0.2827 | -0.1155 [-0.1794, -0.0582] SEP | +0.0915 [+0.0177, +0.1648] SEP | -0.0240 [-0.0924, +0.0249] | 4.80 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 283 / 283 | -0.2564 | -0.3910 | -0.2857 | -0.1346 [-0.1969, -0.0812] SEP | +0.1053 [+0.0389, +0.1751] SEP | -0.0294 [-0.0976, +0.0167] | 4.58 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp (>= 0.15) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 304 / 1459 | -0.1897 | -0.3707 | -0.2403 | -0.1809 [-0.2522, -0.1234] SEP | +0.1304 [+0.0663, +0.2036] SEP | -0.0506 [-0.1180, -0.0053] SEP | 3.58 [1.60, 22.03] |
| seed 1 | 304 / 1459 | -0.1869 | -0.3513 | -0.2359 | -0.1644 [-0.2283, -0.1154] SEP | +0.1154 [+0.0487, +0.1848] SEP | -0.0490 [-0.1222, +0.0007] | 3.35 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 304 / 1458 | -0.1883 | -0.3608 | -0.2379 | -0.1725 [-0.2366, -0.1204] SEP | +0.1229 [+0.0579, +0.1922] SEP | -0.0496 [-0.1202, -0.0020] SEP | 3.48 [1.40, 23.78] |

### S6 -- EMA8 -- curved & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 491 / 491 | -0.2552 | -0.3526 | -0.2816 | -0.0974 [-0.1637, -0.0401] SEP | +0.0710 [+0.0030, +0.1488] SEP | -0.0264 [-0.1218, +0.0501] | 3.69 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 488 / 488 | -0.2618 | -0.3368 | -0.2840 | -0.0750 [-0.1287, -0.0181] SEP | +0.0528 [-0.0372, +0.1415] | -0.0222 [-0.1117, +0.0563] | 3.38 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 487 / 487 | -0.2585 | -0.3446 | -0.2825 | -0.0861 [-0.1420, -0.0308] SEP | +0.0621 [-0.0165, +0.1445] | -0.0241 [-0.1162, +0.0504] | 3.58 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- curved & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 547 / 2634 | -0.1972 | -0.3152 | -0.2366 | -0.1180 [-0.1778, -0.0602] SEP | +0.0786 [+0.0249, +0.1387] SEP | -0.0393 [-0.1276, +0.0304] | 3.00 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 547 / 2629 | -0.2001 | -0.3028 | -0.2372 | -0.1027 [-0.1501, -0.0509] SEP | +0.0655 [-0.0048, +0.1359] | -0.0372 [-0.1188, +0.0314] | 2.76 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 547 / 2626 | -0.1986 | -0.3089 | -0.2367 | -0.1103 [-0.1618, -0.0568] SEP | +0.0722 [+0.0127, +0.1368] SEP | -0.0381 [-0.1224, +0.0300] | 2.89 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 130 / 130 | -0.2530 | -0.3620 | -0.2985 | -0.1090 [-0.1811, -0.0565] SEP | +0.0636 [-0.0162, +0.1528] | -0.0455 [-0.1663, +0.0450] | 2.40 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 130 / 130 | -0.2599 | -0.3485 | -0.2949 | -0.0887 [-0.1449, -0.0361] SEP | +0.0537 [-0.0536, +0.1584] | -0.0350 [-0.1451, +0.0578] | 2.53 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 130 / 130 | -0.2565 | -0.3553 | -0.2967 | -0.0988 [-0.1591, -0.0506] SEP | +0.0586 [-0.0322, +0.1557] | -0.0402 [-0.1551, +0.0493] | 2.46 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 139 / 679 | -0.2012 | -0.3381 | -0.2609 | -0.1369 [-0.2085, -0.0794] SEP | +0.0773 [+0.0123, +0.1491] SEP | -0.0597 [-0.1721, +0.0243] | 2.29 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 139 / 679 | -0.2050 | -0.3269 | -0.2561 | -0.1218 [-0.1750, -0.0712] SEP | +0.0707 [-0.0128, +0.1537] | -0.0511 [-0.1549, +0.0329] | 2.38 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 139 / 679 | -0.2031 | -0.3325 | -0.2585 | -0.1294 [-0.1891, -0.0774] SEP | +0.0740 [+0.0027, +0.1515] SEP | -0.0554 [-0.1632, +0.0279] | 2.34 (CI undefined: the CI of the denominator (gap) includes 0) |

## MECHANICAL READ of the pre-registered outcome table (cell S2/sharp (>= 0.15)/last segment)

* contrast **EMA8**: per-seed outcomes ['PHASE', 'PHASE']; both seeds agree = True; **PHASE**
