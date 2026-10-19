# R7 matched-phase contrasts at N = 20000 (S2/S6 heading-error slope; PREREG_R7_MATCHED_PHASE_TURN.md)

* generated 2026-10-18T06:08:33; trees: `R7` = C:/Users/Admin/refcv8_eval_archive/battery/r7step20000_unc; `EMA8` = C:/Users/Admin/refcv8_eval_archive/battery/step20000ema_unc
* estimator: episode-cluster bootstrap over the eval episodes, models FIXED (not another training run; not another inference run); B 2000; tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
* CONTROL vs `heading_drift.shrinkage` on the same arms (must be < 1e-9): S2|EMA8: 0.00e+00 (ok); S6|EMA8: 0.00e+00 (ok)
* baseline identity (embedded refcv7-50,400 arm, max abs diff vs the R7 tree; must be 0): S2|EMA8|s0 0.0e+00, S2|EMA8|s1 0.0e+00, S6|EMA8|s0 0.0e+00, S6|EMA8|s1 0.0e+00

## S2 ([0.5, 1.0, 1.5, 2.0] s), 4754 windows / 139 episodes


### S2 -- EMA8 -- curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1576 / 1576 | -0.0883 | -0.0986 | -0.1291 | -0.0103 [-0.0453, +0.0242] | -0.0305 [-0.0596, -0.0041] SEP | -0.0408 [-0.0682, -0.0172] SEP | 0.25 [-1.02, 0.88] |
| seed 1 | 1576 / 1576 | -0.0805 | -0.0996 | -0.1310 | -0.0191 [-0.0535, +0.0132] | -0.0314 [-0.0613, -0.0051] SEP | -0.0505 [-0.0819, -0.0247] SEP | 0.38 [-0.40, 0.90] |
| pooled (2-seed mean heading error) | 1576 / 1576 | -0.0844 | -0.0991 | -0.1301 | -0.0147 [-0.0488, +0.0184] | -0.0310 [-0.0606, -0.0053] SEP | -0.0457 [-0.0740, -0.0216] SEP | 0.32 [-0.60, 0.88] |

### S2 -- EMA8 -- curved (GT |yaw| >= 0.02) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1593 / 4752 | -0.1501 | -0.1521 | -0.1887 | -0.0020 [-0.0357, +0.0327] | -0.0367 [-0.0678, -0.0103] SEP | -0.0387 [-0.0632, -0.0172] SEP | 0.05 [-1.38, 0.74] |
| seed 1 | 1595 / 4752 | -0.1430 | -0.1528 | -0.1903 | -0.0098 [-0.0417, +0.0211] | -0.0375 [-0.0689, -0.0109] SEP | -0.0473 [-0.0742, -0.0239] SEP | 0.21 [-0.63, 0.74] |
| pooled (2-seed mean heading error) | 1593 / 4750 | -0.1466 | -0.1524 | -0.1895 | -0.0059 [-0.0373, +0.0261] | -0.0370 [-0.0684, -0.0111] SEP | -0.0429 [-0.0680, -0.0209] SEP | 0.14 [-0.93, 0.74] |

### S2 -- EMA8 -- sharp (>= 0.15) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 364 / 364 | -0.0968 | -0.1050 | -0.1405 | -0.0082 [-0.0490, +0.0336] | -0.0356 [-0.0696, -0.0053] SEP | -0.0438 [-0.0750, -0.0180] SEP | 0.19 [-1.43, 0.89] |
| seed 1 | 364 / 364 | -0.0876 | -0.1067 | -0.1424 | -0.0192 [-0.0594, +0.0193] | -0.0357 [-0.0705, -0.0054] SEP | -0.0548 [-0.0925, -0.0245] SEP | 0.35 [-0.53, 0.89] |
| pooled (2-seed mean heading error) | 364 / 364 | -0.0922 | -0.1058 | -0.1415 | -0.0137 [-0.0530, +0.0250] | -0.0356 [-0.0699, -0.0056] SEP | -0.0493 [-0.0834, -0.0218] SEP | 0.28 [-0.84, 0.87] |

### S2 -- EMA8 -- sharp (>= 0.15) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 367 / 1096 | -0.1084 | -0.1054 | -0.1517 | +0.0030 [-0.0392, +0.0468] | -0.0463 [-0.0830, -0.0134] SEP | -0.0433 [-0.0726, -0.0182] SEP | -0.07 [-1.93, 0.72] |
| seed 1 | 367 / 1095 | -0.0998 | -0.1067 | -0.1533 | -0.0069 [-0.0472, +0.0323] | -0.0466 [-0.0845, -0.0141] SEP | -0.0534 [-0.0886, -0.0243] SEP | 0.13 [-0.90, 0.73] |
| pooled (2-seed mean heading error) | 367 / 1095 | -0.1041 | -0.1061 | -0.1525 | -0.0020 [-0.0422, +0.0386] | -0.0464 [-0.0835, -0.0147] SEP | -0.0484 [-0.0797, -0.0221] SEP | 0.04 [-1.28, 0.71] |

### S2 -- EMA8 -- curved & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 627 / 627 | -0.0906 | -0.0914 | -0.1270 | -0.0009 [-0.0281, +0.0272] | -0.0355 [-0.0633, -0.0088] SEP | -0.0364 [-0.0776, +0.0014] | 0.02 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 627 / 627 | -0.0945 | -0.0936 | -0.1323 | +0.0009 [-0.0200, +0.0236] | -0.0387 [-0.0703, -0.0126] SEP | -0.0379 [-0.0744, -0.0038] SEP | -0.02 [-1.75, 0.50] |
| pooled (2-seed mean heading error) | 627 / 627 | -0.0925 | -0.0925 | -0.1296 | -0.0000 [-0.0233, +0.0247] | -0.0371 [-0.0656, -0.0112] SEP | -0.0371 [-0.0759, -0.0017] SEP | 0.00 [-2.50, 0.62] |

### S2 -- EMA8 -- curved & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 627 / 1881 | -0.0979 | -0.0918 | -0.1355 | +0.0060 [-0.0187, +0.0307] | -0.0436 [-0.0686, -0.0185] SEP | -0.0376 [-0.0754, -0.0014] SEP | -0.16 [-4.01, 0.43] |
| seed 1 | 627 / 1881 | -0.1007 | -0.0945 | -0.1405 | +0.0061 [-0.0136, +0.0276] | -0.0460 [-0.0753, -0.0210] SEP | -0.0398 [-0.0748, -0.0062] SEP | -0.15 [-2.16, 0.27] |
| pooled (2-seed mean heading error) | 627 / 1881 | -0.0993 | -0.0932 | -0.1380 | +0.0061 [-0.0160, +0.0286] | -0.0448 [-0.0712, -0.0201] SEP | -0.0387 [-0.0751, -0.0039] SEP | -0.16 [-2.56, 0.31] |

### S2 -- EMA8 -- sharp & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 164 / 164 | -0.0901 | -0.0849 | -0.1276 | +0.0052 [-0.0228, +0.0351] | -0.0427 [-0.0731, -0.0134] SEP | -0.0375 [-0.0806, +0.0022] | -0.14 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 164 / 164 | -0.0924 | -0.0858 | -0.1304 | +0.0066 [-0.0131, +0.0303] | -0.0446 [-0.0790, -0.0175] SEP | -0.0380 [-0.0775, -0.0015] SEP | -0.17 [-3.19, 0.33] |
| pooled (2-seed mean heading error) | 164 / 164 | -0.0912 | -0.0853 | -0.1290 | +0.0059 [-0.0173, +0.0316] | -0.0436 [-0.0744, -0.0163] SEP | -0.0377 [-0.0795, +0.0003] | -0.16 (CI undefined: the CI of the denominator (gap) includes 0) |

### S2 -- EMA8 -- sharp & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 164 / 492 | -0.0947 | -0.0788 | -0.1320 | +0.0159 [-0.0109, +0.0441] | -0.0532 [-0.0815, -0.0264] SEP | -0.0373 [-0.0792, +0.0017] | -0.43 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 164 / 492 | -0.0963 | -0.0802 | -0.1342 | +0.0161 [-0.0046, +0.0391] | -0.0540 [-0.0871, -0.0287] SEP | -0.0379 [-0.0768, -0.0009] SEP | -0.43 [-4.75, 0.17] |
| pooled (2-seed mean heading error) | 164 / 492 | -0.0955 | -0.0795 | -0.1331 | +0.0160 [-0.0068, +0.0412] | -0.0536 [-0.0828, -0.0281] SEP | -0.0376 [-0.0783, -0.0004] SEP | -0.43 [-5.49, 0.27] |

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s), 3668 windows / 139 episodes


### S6 -- EMA8 -- curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1315 / 1315 | -0.2424 | -0.2748 | -0.2293 | -0.0324 [-0.0736, +0.0031] | +0.0455 [+0.0018, +0.0894] SEP | +0.0131 [-0.0230, +0.0453] | -2.48 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 1318 / 1318 | -0.2439 | -0.2804 | -0.2392 | -0.0365 [-0.0777, -0.0039] SEP | +0.0412 [-0.0033, +0.0859] | +0.0047 [-0.0286, +0.0315] | -7.84 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 1313 / 1313 | -0.2431 | -0.2764 | -0.2347 | -0.0333 [-0.0738, +0.0005] | +0.0417 [-0.0016, +0.0852] | +0.0084 [-0.0257, +0.0365] | -3.97 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- curved (GT |yaw| >= 0.02) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1414 / 6832 | -0.1709 | -0.1954 | -0.1654 | -0.0245 [-0.0679, +0.0227] | +0.0300 [-0.0039, +0.0628] | +0.0055 [-0.0352, +0.0495] | -4.46 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 1414 / 6836 | -0.1812 | -0.1873 | -0.1573 | -0.0061 [-0.0735, +0.1013] | +0.0300 [-0.0071, +0.0666] | +0.0239 [-0.0425, +0.1263] | -0.26 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 1414 / 6825 | -0.1762 | -0.1901 | -0.1617 | -0.0139 [-0.0679, +0.0607] | +0.0284 [-0.0057, +0.0627] | +0.0146 [-0.0388, +0.0874] | -0.95 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp (>= 0.15) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 284 / 284 | -0.2571 | -0.3019 | -0.2496 | -0.0448 [-0.1027, +0.0019] | +0.0523 [-0.0016, +0.1081] | +0.0075 [-0.0407, +0.0468] | -6.01 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 285 / 285 | -0.2596 | -0.3115 | -0.2629 | -0.0519 [-0.1093, -0.0125] SEP | +0.0486 [-0.0080, +0.1061] | -0.0033 [-0.0498, +0.0301] | 15.92 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 284 / 284 | -0.2586 | -0.3054 | -0.2571 | -0.0468 [-0.1037, -0.0044] SEP | +0.0484 [-0.0056, +0.1029] | +0.0015 [-0.0447, +0.0371] | -30.62 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp (>= 0.15) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 304 / 1462 | -0.1925 | -0.2530 | -0.2109 | -0.0605 [-0.1151, -0.0188] SEP | +0.0421 [-0.0009, +0.0863] | -0.0184 [-0.0674, +0.0177] | 3.28 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 304 / 1465 | -0.1882 | -0.2622 | -0.2197 | -0.0740 [-0.1276, -0.0351] SEP | +0.0425 [-0.0043, +0.0905] | -0.0315 [-0.0817, +0.0025] | 2.35 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 304 / 1461 | -0.1907 | -0.2562 | -0.2160 | -0.0654 [-0.1181, -0.0267] SEP | +0.0402 [-0.0037, +0.0848] | -0.0252 [-0.0726, +0.0085] | 2.59 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- curved & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 492 / 492 | -0.2597 | -0.2643 | -0.2482 | -0.0046 [-0.0649, +0.0425] | +0.0161 [-0.0465, +0.0987] | +0.0115 [-0.0535, +0.0701] | -0.40 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 492 / 492 | -0.2662 | -0.2639 | -0.2559 | +0.0024 [-0.0436, +0.0383] | +0.0080 [-0.0635, +0.0937] | +0.0103 [-0.0455, +0.0663] | 0.23 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 490 / 490 | -0.2628 | -0.2640 | -0.2520 | -0.0012 [-0.0513, +0.0386] | +0.0119 [-0.0542, +0.0948] | +0.0107 [-0.0492, +0.0674] | -0.11 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- curved & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 547 / 2635 | -0.2004 | -0.2201 | -0.2085 | -0.0197 [-0.0678, +0.0233] | +0.0116 [-0.0395, +0.0736] | -0.0081 [-0.0692, +0.0430] | 2.42 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 547 / 2636 | -0.2033 | -0.2194 | -0.2139 | -0.0161 [-0.0502, +0.0138] | +0.0055 [-0.0523, +0.0706] | -0.0106 [-0.0647, +0.0376] | 1.51 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 547 / 2630 | -0.2018 | -0.2197 | -0.2112 | -0.0179 [-0.0573, +0.0172] | +0.0084 [-0.0463, +0.0710] | -0.0095 [-0.0668, +0.0404] | 1.89 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 131 / 131 | -0.2580 | -0.2605 | -0.2511 | -0.0025 [-0.0776, +0.0533] | +0.0094 [-0.0598, +0.1054] | +0.0069 [-0.0675, +0.0715] | -0.37 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 131 / 131 | -0.2646 | -0.2600 | -0.2601 | +0.0046 [-0.0476, +0.0455] | -0.0001 [-0.0792, +0.0968] | +0.0045 [-0.0622, +0.0661] | 1.03 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 131 / 131 | -0.2613 | -0.2603 | -0.2556 | +0.0011 [-0.0594, +0.0468] | +0.0047 [-0.0695, +0.1000] | +0.0057 [-0.0623, +0.0669] | 0.18 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 139 / 682 | -0.2050 | -0.2318 | -0.2205 | -0.0268 [-0.0900, +0.0253] | +0.0113 [-0.0447, +0.0834] | -0.0156 [-0.0856, +0.0433] | 1.72 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 139 / 683 | -0.2087 | -0.2303 | -0.2266 | -0.0216 [-0.0679, +0.0140] | +0.0036 [-0.0620, +0.0786] | -0.0180 [-0.0818, +0.0364] | 1.20 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 139 / 682 | -0.2068 | -0.2310 | -0.2236 | -0.0242 [-0.0773, +0.0182] | +0.0074 [-0.0521, +0.0816] | -0.0168 [-0.0833, +0.0394] | 1.44 (CI undefined: the CI of the denominator (gap) includes 0) |

## MECHANICAL READ of the pre-registered outcome table (cell S2/sharp (>= 0.15)/last segment)

* contrast **EMA8**: per-seed outcomes ['DESIGN', 'DESIGN']; both seeds agree = True; **DESIGN**
