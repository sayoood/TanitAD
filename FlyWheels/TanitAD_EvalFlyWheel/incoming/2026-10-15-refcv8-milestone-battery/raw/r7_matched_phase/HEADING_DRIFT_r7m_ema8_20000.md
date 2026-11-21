# refcv8 heading-drift localisation: EMA8-20k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-18T06:01:57; trees: `EMA8-20k` = C:/Users/Admin/refcv8_eval_archive/battery/step20000ema_unc; `R7-20k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step20000_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **True**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

### S2 -- EMA8-20k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4513 | 139 | 1.4584 | 1.4072 | 0.0512 [0.0113, 0.0933] | **SEP** |  |
| turn: straight <0.02 | 2900 | 132 | 1.2044 | 1.1591 | 0.0452 [0.0044, 0.0959] | **SEP** | 0.643 / 0.568 / -0.0166 [-0.1567, 0.1125] |
| turn: gentle 0.02-0.06 | 720 | 98 | 1.0225 | 1.0094 | 0.0130 [-0.0528, 0.0874] |  | 0.160 / 0.041 / -0.0454 [-0.1668, 0.0854] |
| turn: turn 0.06-0.15 | 508 | 72 | 1.9144 | 1.9207 | -0.0063 [-0.1057, 0.0952] |  | 0.113 / -0.014 / -0.0647 [-0.2278, 0.0846] |
| turn: sharp >=0.15 | 367 | 42 | 3.2434 | 2.9870 | 0.2564 [0.0149, 0.5588] | **SEP** | 0.081 / 0.407 / 0.2234 [-0.0945, 0.7253] |
| turn: undef (GT crawling) (low n) | 18 | 14 | 10.5542 | 10.5810 | -0.0268 [-0.1649, 0.1039] |  | 0.004 / -0.002 / -0.0783 [-0.2867, 0.1270] |
| speed: v0 <2 | 224 | 37 | 13.1927 | 13.0629 | 0.1298 [-0.0049, 0.2867] |  | 0.050 / 0.126 / 0.0827 [-0.1427, 0.3736] |
| speed: v0 2-5 | 516 | 58 | 1.9321 | 1.8694 | 0.0627 [-0.1638, 0.3273] |  | 0.114 / 0.140 / 0.0130 [-0.2823, 0.4510] |
| speed: v0 5-10 | 1390 | 84 | 1.1454 | 1.0092 | 0.1362 [0.0521, 0.2286] | **SEP** | 0.308 / 0.819 / 0.1228 [-0.0251, 0.2724] |
| speed: v0 10-15 | 1192 | 64 | 0.5857 | 0.5744 | 0.0113 [-0.0260, 0.0486] |  | 0.264 / 0.058 / -0.0542 [-0.1470, 0.0309] |
| speed: v0 15-25 | 856 | 35 | 0.3358 | 0.3734 | -0.0376 [-0.0682, -0.0099] | **SEP** | 0.190 / -0.139 / -0.1096 [-0.1995, -0.0336] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1547 | 0.1576 | -0.0029 [-0.0266, 0.0185] |  | 0.074 / -0.004 / -0.0584 [-0.1362, 0.0091] |
| nav: nav=follow | 2880 | 88 | 0.6568 | 0.5982 | 0.0586 [0.0168, 0.1092] | **SEP** | 0.638 / 0.731 / 0.0205 [-0.1278, 0.1674] |
| nav: nav=left (low n) | 405 | 13 | 6.8879 | 6.8793 | 0.0087 [-0.0538, 0.0628] |  | 0.090 / 0.015 / -0.0467 [-0.1695, 0.0537] |
| nav: nav=right | 1228 | 38 | 1.5476 | 1.4998 | 0.0478 [-0.0526, 0.1553] |  | 0.272 / 0.254 / -0.0047 [-0.1746, 0.1815] |

episode concentration: 0.568 of episodes have mean delta > 0; top-5 episodes carry 0.693 of the summed delta; delta excluding the top 5 = 0.0163 (overall 0.0512).

### S2 -- EMA8-20k - R7-20k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4503 | 139 | 1.4216 | 1.4436 | -0.0219 [-0.0739, 0.0339] |  |  |
| turn: straight <0.02 | 2895 | 132 | 1.2061 | 1.1943 | 0.0118 [-0.0377, 0.0640] |  | 0.643 / -0.346 / 0.0944 [-0.1035, 0.2825] |
| turn: gentle 0.02-0.06 | 720 | 98 | 1.0225 | 0.9825 | 0.0400 [-0.0484, 0.1482] |  | 0.160 / -0.292 / 0.0737 [-0.0659, 0.2448] |
| turn: turn 0.06-0.15 | 508 | 72 | 1.9144 | 2.1916 | -0.2772 [-0.4489, -0.1422] | **SEP** | 0.113 / 1.427 / -0.2878 [-0.5414, -0.0786] BONF-SEP |
| turn: sharp >=0.15 | 367 | 42 | 3.2434 | 3.3091 | -0.0657 [-0.3959, 0.3484] |  | 0.082 / 0.244 / -0.0477 [-0.5416, 0.5743] |
| turn: undef (GT crawling) (low n) | 13 | 12 | 0.8340 | 0.5793 | 0.2546 [-0.2044, 0.8334] |  | 0.003 / -0.034 / 0.2773 [-0.3636, 1.3472] |
| speed: v0 <2 | 214 | 37 | 12.9681 | 12.9331 | 0.0351 [-0.1167, 0.2321] |  | 0.048 / -0.076 / 0.0598 [-0.1839, 0.4073] |
| speed: v0 2-5 | 516 | 58 | 1.9321 | 2.0874 | -0.1553 [-0.4607, 0.1694] |  | 0.115 / 0.812 / -0.1506 [-0.5843, 0.3689] |
| speed: v0 5-10 | 1390 | 84 | 1.1454 | 1.1092 | 0.0362 [-0.0691, 0.1426] |  | 0.309 / -0.510 / 0.0841 [-0.0765, 0.2496] |
| speed: v0 10-15 | 1192 | 64 | 0.5857 | 0.6040 | -0.0183 [-0.0623, 0.0278] |  | 0.265 / 0.222 / 0.0048 [-0.1133, 0.1053] |
| speed: v0 15-25 | 856 | 35 | 0.3358 | 0.3964 | -0.0606 [-0.0990, -0.0270] | **SEP** | 0.190 / 0.526 / -0.0477 [-0.1649, 0.0525] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1547 | 0.1627 | -0.0080 [-0.0319, 0.0140] |  | 0.074 / 0.027 / 0.0150 [-0.0793, 0.1004] |
| nav: nav=follow | 2871 | 88 | 0.6583 | 0.6508 | 0.0076 [-0.0428, 0.0613] |  | 0.638 / -0.220 / 0.0813 [-0.1354, 0.2735] |
| nav: nav=left (low n) | 405 | 13 | 6.8879 | 6.9100 | -0.0220 [-0.2619, 0.2682] |  | 0.090 / 0.090 / -0.0001 [-0.3774, 0.4780] |
| nav: nav=right | 1227 | 38 | 1.4034 | 1.4942 | -0.0908 [-0.2200, 0.0503] |  | 0.272 / 1.130 / -0.0948 [-0.3006, 0.1437] |

episode concentration: 0.381 of episodes have mean delta > 0; top-5 episodes carry -1.800 of the summed delta; delta excluding the top 5 = -0.0636 (overall -0.0219).

### S2 -- R7-20k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4520 | 139 | 1.4855 | 1.4107 | 0.0748 [0.0298, 0.1207] | **SEP** |  |
| turn: straight <0.02 | 2898 | 132 | 1.1935 | 1.1596 | 0.0339 [0.0124, 0.0591] | **SEP** | 0.641 / 0.291 / -0.1139 [-0.3208, 0.0621] |
| turn: gentle 0.02-0.06 | 721 | 98 | 0.9847 | 1.0095 | -0.0248 [-0.1099, 0.0403] |  | 0.160 / -0.053 / -0.1185 [-0.2785, -0.0092] BONF-SEP |
| turn: turn 0.06-0.15 | 512 | 72 | 2.2085 | 1.9319 | 0.2766 [0.1421, 0.4417] | **SEP** | 0.113 / 0.419 / 0.2276 [0.0336, 0.4964] BONF-SEP |
| turn: sharp >=0.15 | 368 | 42 | 3.3254 | 3.0070 | 0.3185 [-0.0559, 0.7023] |  | 0.081 / 0.347 / 0.2653 [-0.3030, 0.9886] |
| turn: undef (GT crawling) (low n) | 21 | 17 | 9.0951 | 9.1504 | -0.0553 [-0.3755, 0.1854] |  | 0.005 / -0.003 / -0.1307 [-0.6283, 0.1948] |
| speed: v0 <2 | 231 | 38 | 12.9074 | 12.7780 | 0.1294 [-0.0116, 0.2725] |  | 0.051 / 0.088 / 0.0575 [-0.1563, 0.2727] |
| speed: v0 2-5 | 516 | 58 | 2.0874 | 1.8694 | 0.2180 [-0.0860, 0.5492] |  | 0.114 / 0.333 / 0.1616 [-0.3265, 0.7098] |
| speed: v0 5-10 | 1390 | 84 | 1.1092 | 1.0092 | 0.1000 [0.0416, 0.1655] | **SEP** | 0.308 / 0.411 / 0.0364 [-0.0810, 0.1531] |
| speed: v0 10-15 | 1192 | 64 | 0.6040 | 0.5744 | 0.0296 [-0.0108, 0.0696] |  | 0.264 / 0.104 / -0.0613 [-0.1738, 0.0382] |
| speed: v0 15-25 | 856 | 35 | 0.3964 | 0.3734 | 0.0230 [-0.0030, 0.0579] |  | 0.189 / 0.058 / -0.0639 [-0.1638, 0.0329] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1627 | 0.1576 | 0.0051 [-0.0069, 0.0182] |  | 0.074 / 0.005 / -0.0753 [-0.1564, -0.0009] BONF-SEP |
| nav: nav=follow | 2872 | 88 | 0.6506 | 0.5988 | 0.0518 [0.0196, 0.0903] | **SEP** | 0.635 / 0.441 / -0.0629 [-0.2319, 0.1204] |
| nav: nav=left (low n) | 412 | 13 | 7.2617 | 7.2314 | 0.0302 [-0.2494, 0.2404] |  | 0.091 / 0.037 / -0.0490 [-0.5216, 0.2698] |
| nav: nav=right | 1236 | 38 | 1.4999 | 1.3570 | 0.1429 [0.0189, 0.2619] | **SEP** | 0.273 / 0.523 / 0.0938 [-0.0821, 0.2815] |

episode concentration: 0.683 of episodes have mean delta > 0; top-5 episodes carry 0.461 of the summed delta; delta excluding the top 5 = 0.0418 (overall 0.0748).

### S2 -- EMA8-20k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4473 | 139 | 0.7168 | 0.6286 | 0.0882 [-0.0023, 0.2565] |  | 0.2188 [-0.0353, 0.7173] |
| seg1 | straight | 2878 | 132 | 0.7020 | 0.5699 | 0.1321 [-0.0014, 0.4092] |  | 0.3484 [-0.0390, 1.1846] |
| seg1 | curved | 1585 | 101 | 0.6319 | 0.6231 | 0.0087 [-0.0168, 0.0349] |  | -0.0147 [-0.0608, 0.0256] |
| seg2 | all | 4470 | 139 | 1.1410 | 0.9389 | 0.2021 [0.0129, 0.5419] | **SEP** | 0.0906 [-0.1096, 0.4422] |
| seg2 | straight | 2880 | 132 | 0.9523 | 0.6574 | 0.2948 [0.0169, 0.8579] | **SEP** | 0.1606 [-0.1276, 0.7381] |
| seg2 | curved | 1590 | 101 | 1.4829 | 1.4487 | 0.0342 [-0.0294, 0.1031] |  | -0.0362 [-0.1585, 0.0752] |
| seg3 | all | 4463 | 139 | 1.3533 | 1.2034 | 0.1499 [0.0283, 0.3387] | **SEP** | 0.0591 [-0.1632, 0.4276] |
| seg3 | straight | 2877 | 132 | 0.9679 | 0.7731 | 0.1948 [0.0251, 0.5087] | **SEP** | 0.1178 [-0.1907, 0.7068] |
| seg3 | curved | 1586 | 101 | 2.0524 | 1.9840 | 0.0684 [-0.0335, 0.1772] |  | -0.0474 [-0.2242, 0.1130] |
| seg4 | all | 4445 | 139 | 1.7508 | 1.4923 | 0.2585 [0.0083, 0.6923] | **SEP** | 0.1996 [-0.1614, 0.8133] |
| seg4 | straight | 2860 | 131 | 1.1852 | 0.8396 | 0.3456 [-0.0103, 1.0487] |  | 0.3391 [-0.1497, 1.3024] |
| seg4 | curved | 1577 | 101 | 2.7820 | 2.6798 | 0.1023 [-0.0412, 0.2584] |  | -0.0512 [-0.3259, 0.1901] |

### S2 -- EMA8-20k - R7-20k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4472 | 139 | 0.6770 | 0.6292 | 0.0479 [-0.0460, 0.2188] |  | -0.0360 [-0.0634, -0.0104] SEP |
| seg1 | straight | 2878 | 132 | 0.7020 | 0.5840 | 0.1180 [-0.0179, 0.3966] |  | -0.0417 [-0.0618, -0.0233] SEP |
| seg1 | curved | 1585 | 101 | 0.6319 | 0.7118 | -0.0800 [-0.1184, -0.0388] | **SEP** | -0.0246 [-0.0935, 0.0379] |
| seg2 | all | 4468 | 139 | 1.1404 | 1.1053 | 0.0352 [-0.0814, 0.2162] |  | -0.1248 [-0.1931, -0.0625] SEP |
| seg2 | straight | 2879 | 132 | 0.9526 | 0.8177 | 0.1349 [-0.0230, 0.4417] |  | -0.1159 [-0.1726, -0.0646] SEP |
| seg2 | curved | 1589 | 101 | 1.4808 | 1.6263 | -0.1455 [-0.2534, -0.0231] | **SEP** | -0.1408 [-0.3038, 0.0063] |
| seg3 | all | 4458 | 139 | 1.3537 | 1.2901 | 0.0637 [-0.0701, 0.2482] |  | -0.0838 [-0.2313, 0.1234] |
| seg3 | straight | 2873 | 132 | 0.9680 | 0.8073 | 0.1608 [-0.0173, 0.4855] |  | 0.0097 [-0.1788, 0.3191] |
| seg3 | curved | 1585 | 101 | 2.0529 | 2.1652 | -0.1123 [-0.2643, 0.0552] |  | -0.2534 [-0.4507, -0.0782] SEP |
| seg4 | all | 4435 | 139 | 1.7532 | 1.7673 | -0.0141 [-0.0998, 0.0727] |  | -0.0819 [-0.1942, 0.0244] |
| seg4 | straight | 2855 | 131 | 1.1862 | 1.1938 | -0.0076 [-0.0731, 0.0586] |  | -0.0138 [-0.1003, 0.0652] |
| seg4 | curved | 1576 | 101 | 2.7817 | 2.8090 | -0.0273 [-0.2169, 0.1823] |  | -0.2042 [-0.4745, 0.0505] |

### S2 -- R7-20k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4483 | 139 | 0.6686 | 0.6280 | 0.0406 [0.0246, 0.0559] | **SEP** | 0.2541 [-0.0036, 0.7510] |
| seg1 | straight | 2880 | 132 | 0.5838 | 0.5695 | 0.0143 [0.0056, 0.0240] | **SEP** | 0.3898 [0.0024, 1.2288] SEP |
| seg1 | curved | 1588 | 101 | 0.7117 | 0.6229 | 0.0888 [0.0507, 0.1229] | **SEP** | 0.0098 [-0.0456, 0.0712] |
| seg2 | all | 4478 | 139 | 1.1048 | 0.9384 | 0.1665 [0.0530, 0.3494] | **SEP** | 0.2129 [0.0126, 0.5602] SEP |
| seg2 | straight | 2886 | 132 | 0.8169 | 0.6569 | 0.1600 [0.0202, 0.4464] | **SEP** | 0.2754 [-0.0039, 0.8447] |
| seg2 | curved | 1592 | 101 | 1.6268 | 1.4487 | 0.1782 [0.0727, 0.2766] | **SEP** | 0.0997 [-0.0402, 0.2522] |
| seg3 | all | 4465 | 139 | 1.2910 | 1.2053 | 0.0858 [0.0258, 0.1474] | **SEP** | 0.1440 [0.0106, 0.3285] SEP |
| seg3 | straight | 2876 | 132 | 0.8073 | 0.7741 | 0.0333 [0.0026, 0.0697] | **SEP** | 0.1089 [-0.0537, 0.4104] |
| seg3 | curved | 1589 | 101 | 2.1665 | 1.9857 | 0.1808 [0.0307, 0.3259] | **SEP** | 0.2075 [0.0256, 0.3916] SEP |
| seg4 | all | 4449 | 139 | 1.7720 | 1.4975 | 0.2745 [0.0066, 0.7239] | **SEP** | 0.2838 [-0.0685, 0.8901] |
| seg4 | straight | 2862 | 131 | 1.1923 | 0.8393 | 0.3530 [0.0164, 1.0423] | **SEP** | 0.3517 [-0.1307, 1.3262] |
| seg4 | curved | 1581 | 101 | 2.8250 | 2.6914 | 0.1336 [-0.0669, 0.3364] |  | 0.1607 [-0.1063, 0.4168] |

### S2 -- EMA8-20k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 441 | 47 | 5.9804 | 5.9418 | 0.0387 [-0.1174, 0.2689] |  |  |
| straight <0.02 | v0 5-15 | 1519 | 93 | 0.4474 | 0.3674 | 0.0799 [0.0219, 0.1531] | **SEP** |  |
| straight <0.02 | v0 >=15 | 940 | 43 | 0.1870 | 0.1947 | -0.0077 [-0.0193, 0.0040] |  |  |
| gentle 0.02-0.06 | v0 <5 | 111 | 29 | 1.7029 | 1.7693 | -0.0664 [-0.3519, 0.2183] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 461 | 74 | 1.0329 | 0.9711 | 0.0618 [-0.0227, 0.1542] |  |  |
| gentle 0.02-0.06 | v0 >=15 | 148 | 20 | 0.4798 | 0.5590 | -0.0791 [-0.1639, -0.0123] | **SEP** |  |
| turn 0.06-0.15 | v0 <5 (low n) | 78 | 25 | 5.1855 | 5.1476 | 0.0379 [-0.3617, 0.3846] |  |  |
| turn 0.06-0.15 | v0 5-15 | 343 | 54 | 1.4413 | 1.4275 | 0.0137 [-0.1099, 0.1411] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 87 | 11 | 0.8471 | 0.9719 | -0.1248 [-0.2332, 0.0002] |  |  |
| sharp >=0.15 | v0 <5 (low n) | 92 | 16 | 5.7750 | 5.2396 | 0.5355 [-0.0474, 1.4998] |  |  |
| sharp >=0.15 | v0 5-15 | 259 | 33 | 2.4717 | 2.2857 | 0.1860 [-0.0367, 0.4660] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 16 | 3 | 1.1778 | 1.3866 | -0.2088 [-0.3405, 0.0375] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 18 | 14 | 10.5542 | 10.5810 | -0.0268 [-0.1649, 0.1039] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S2 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | EMA8-20k s0-s1 [CI] | R7-20k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0071 [-0.0216, 0.0069] | -0.0015 [-0.0157, 0.0150] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | -0.0099 [-0.0238, 0.0016] | 0.0079 [-0.0056, 0.0266] | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | -0.0023 [-0.0574, 0.0491] | -0.0271 [-0.0663, 0.0095] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | -0.0009 [-0.0407, 0.0384] | -0.0187 [-0.0706, 0.0340] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | -0.0045 [-0.0915, 0.0728] | -0.0092 [-0.1307, 0.1113] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | 0.0306 [-0.0652, 0.1367] | 0.1339 [-0.0506, 0.3279] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | 0.0031 [-0.1096, 0.1086] | 0.0129 [-0.1193, 0.1887] | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | -0.0682 [-0.1287, -0.0169] SEP | 0.0189 [-0.0664, 0.1076] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | 0.0029 [-0.0281, 0.0299] | -0.0115 [-0.0432, 0.0284] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | -0.0063 [-0.0185, 0.0052] | 0.0005 [-0.0131, 0.0140] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | 0.0069 [-0.0002, 0.0155] | -0.0017 [-0.0070, 0.0041] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | 0.0003 [-0.0058, 0.0063] | -0.0075 [-0.0163, -0.0008] SEP | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | -0.0038 [-0.0165, 0.0082] | -0.0062 [-0.0176, 0.0054] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | 0.0051 [-0.0507, 0.0632] | -0.0472 [-0.1021, -0.0024] SEP | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | -0.0187 [-0.0561, 0.0153] | 0.0250 [-0.0162, 0.0755] | 0.0283 [0.0035, 0.0597] SEP |

### S2 -- EMA8-20k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0512 | 0.0424 / 0.0600 | True / True / True | 0.0071 / 0.0110 | 4.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0452 | 0.0393 / 0.0512 | False / True / True | 0.0099 / 0.0019 | 4.6 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|gentle 0.02-0.06 | 0.0130 | 0.0197 / 0.0064 | False / False / False | 0.0023 / 0.0161 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.0063 | -0.0213 / 0.0087 | False / False / False | 0.0009 / 0.0331 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.2564 | 0.2001 / 0.3127 | False / True / True | 0.0045 / 0.1080 | 2.4 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|undef (GT crawling) | -0.0268 | 0.0241 / -0.0712 | False / False / False | 0.0306 / 0.0237 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.1298 | 0.1043 / 0.1541 | False / False / False | 0.0031 / 0.0573 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.0627 | -0.0074 / 0.1328 | False / False / False | 0.0682 / 0.0719 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1362 | 0.1373 / 0.1351 | True / True / True | 0.0029 / 0.0006 | 46.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0113 | 0.0082 / 0.0144 | False / False / False | 0.0063 / 0.0001 | 1.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0376 | -0.0330 / -0.0422 | True / True / True | 0.0069 / 0.0023 | 5.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0029 | -0.0025 / -0.0033 | False / False / False | 0.0003 / 0.0005 | 5.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0586 | 0.0585 / 0.0587 | True / True / True | 0.0038 / 0.0036 | 15.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.0087 | -0.0172 / 0.0345 | False / False / False | 0.0051 / 0.0608 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0478 | 0.0241 / 0.0714 | False / False / False | 0.0187 / 0.0283 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- EMA8-20k - R7-20k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.0219 | -0.0240 / -0.0199 | False / False / False | 0.0071 / 0.0015 | 3.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0118 | 0.0034 / 0.0201 | False / False / False | 0.0099 / 0.0079 | 1.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.0400 | 0.0527 / 0.0272 | False / False / False | 0.0023 / 0.0271 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.2772 | -0.2681 / -0.2863 | True / True / True | 0.0009 / 0.0187 | 14.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | -0.0657 | -0.0636 / -0.0678 | False / False / False | 0.0045 / 0.0092 | 7.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.2546 | 0.2596 / 0.2237 | False / False / False | 0.0306 / 0.1339 | 1.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.0351 | 0.0433 / 0.0240 | False / False / False | 0.0031 / 0.0129 | 2.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.1553 | -0.1988 / -0.1117 | False / False / False | 0.0682 / 0.0189 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.0362 | 0.0435 / 0.0290 | False / False / False | 0.0029 / 0.0115 | 3.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.0183 | -0.0218 / -0.0149 | False / False / False | 0.0063 / 0.0005 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0606 | -0.0563 / -0.0649 | True / True / True | 0.0069 / 0.0017 | 8.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0080 | -0.0041 / -0.0119 | False / False / False | 0.0003 / 0.0075 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0076 | 0.0087 / 0.0063 | False / False / False | 0.0038 / 0.0062 | 1.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.0220 | 0.0087 / -0.0528 | False / False / False | 0.0051 / 0.0472 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.0908 | -0.1112 / -0.0703 | False / False / False | 0.0187 / 0.0250 | 3.6 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- R7-20k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0748 | 0.0681 / 0.0822 | True / True / True | 0.0015 / 0.0110 | 6.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0339 | 0.0364 / 0.0314 | True / True / True | 0.0079 / 0.0019 | 4.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | -0.0248 | -0.0303 / -0.0194 | False / False / False | 0.0271 / 0.0161 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.2766 | 0.2510 / 0.3076 | True / True / True | 0.0187 / 0.0331 | 8.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 0.3185 | 0.2599 / 0.3771 | False / True / False | 0.0092 / 0.1080 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | -0.0553 | 0.0054 / -0.0831 | False / False / False | 0.1339 / 0.0237 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.1294 | 0.0965 / 0.1757 | False / True / False | 0.0129 / 0.0573 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.2180 | 0.1914 / 0.2445 | False / False / False | 0.0189 / 0.0719 | 3.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1000 | 0.0939 / 0.1061 | True / True / True | 0.0115 / 0.0006 | 8.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0296 | 0.0299 / 0.0293 | False / False / False | 0.0005 / 0.0001 | 54.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0230 | 0.0233 / 0.0227 | False / False / False | 0.0017 / 0.0023 | 10.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0051 | 0.0016 / 0.0086 | False / False / False | 0.0075 / 0.0005 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0518 | 0.0505 / 0.0531 | True / True / True | 0.0062 / 0.0036 | 8.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.0302 | -0.0271 / 0.0947 | False / False / False | 0.0472 / 0.0608 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.1429 | 0.1406 / 0.1455 | True / True / True | 0.0250 / 0.0283 | 5.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S2 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1585): EMA8-20k bias -0.012, refcv7-50400 bias +0.082, R7-20k bias +0.138; EMA8-20k MAE 0.632, refcv7-50400 MAE 0.623, R7-20k MAE 0.712
* seg2 (n 1589): EMA8-20k bias -0.305, refcv7-50400 bias -0.097, R7-20k bias -0.004; EMA8-20k MAE 1.481, refcv7-50400 MAE 1.448, R7-20k MAE 1.626
* seg3 (n 1585): EMA8-20k bias -0.854, refcv7-50400 bias -0.617, R7-20k bias -0.662; EMA8-20k MAE 2.053, refcv7-50400 MAE 1.984, R7-20k MAE 2.165
* seg4 (n 1576): EMA8-20k bias -1.097, refcv7-50400 bias -0.715, R7-20k bias -0.880; EMA8-20k MAE 2.782, refcv7-50400 MAE 2.681, R7-20k MAE 2.809
**sharp (>= 0.15)**

* seg1 (n 364): EMA8-20k bias -0.269, refcv7-50400 bias +0.010, R7-20k bias +0.286; EMA8-20k MAE 0.915, refcv7-50400 MAE 0.917, R7-20k MAE 1.108
* seg2 (n 366): EMA8-20k bias -1.139, refcv7-50400 bias -0.528, R7-20k bias +0.086; EMA8-20k MAE 2.515, refcv7-50400 MAE 2.413, R7-20k MAE 2.812
* seg3 (n 365): EMA8-20k bias -2.533, refcv7-50400 bias -1.863, R7-20k bias -1.602; EMA8-20k MAE 3.982, refcv7-50400 MAE 3.649, R7-20k MAE 4.038
* seg4 (n 364): EMA8-20k bias -2.976, refcv7-50400 bias -1.837, R7-20k bias -2.010; EMA8-20k MAE 5.428, refcv7-50400 MAE 4.878, R7-20k MAE 5.145
**curved & v0 5-10**

* seg1 (n 627): EMA8-20k bias -0.130, refcv7-50400 bias -0.038, R7-20k bias +0.030; EMA8-20k MAE 0.560, refcv7-50400 MAE 0.535, R7-20k MAE 0.661
* seg2 (n 627): EMA8-20k bias -0.507, refcv7-50400 bias -0.282, R7-20k bias -0.169; EMA8-20k MAE 1.466, refcv7-50400 MAE 1.363, R7-20k MAE 1.598
* seg3 (n 627): EMA8-20k bias -1.135, refcv7-50400 bias -0.871, R7-20k bias -0.896; EMA8-20k MAE 2.185, refcv7-50400 MAE 2.019, R7-20k MAE 2.199
* seg4 (n 627): EMA8-20k bias -1.553, refcv7-50400 bias -1.194, R7-20k bias -1.283; EMA8-20k MAE 2.963, refcv7-50400 MAE 2.730, R7-20k MAE 2.839
**sharp & v0 5-10**

* seg1 (n 164): EMA8-20k bias -0.121, refcv7-50400 bias +0.103, R7-20k bias +0.482; EMA8-20k MAE 0.818, refcv7-50400 MAE 0.857, R7-20k MAE 1.100
* seg2 (n 164): EMA8-20k bias -0.950, refcv7-50400 bias -0.422, R7-20k bias +0.462; EMA8-20k MAE 2.326, refcv7-50400 MAE 2.243, R7-20k MAE 2.699
* seg3 (n 164): EMA8-20k bias -2.660, refcv7-50400 bias -2.032, R7-20k bias -1.436; EMA8-20k MAE 3.852, refcv7-50400 MAE 3.465, R7-20k MAE 3.740
* seg4 (n 164): EMA8-20k bias -3.410, refcv7-50400 bias -2.403, R7-20k bias -2.138; EMA8-20k MAE 5.449, refcv7-50400 MAE 4.635, R7-20k MAE 4.837

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1576 / 1576 | 10.2 | EMA8-20k -0.1301 [-0.2148, -0.0635]; refcv7-50400 -0.0844 [-0.1557, -0.0274]; R7-20k -0.0991 [-0.1813, -0.0377] | EMA8-20k -0.0457 [-0.0740, -0.0216] SEP; R7-20k -0.0147 [-0.0488, +0.0184] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1593 / 4750 | 7.2 | EMA8-20k -0.1895 [-0.3127, -0.0888]; refcv7-50400 -0.1466 [-0.2654, -0.0522]; R7-20k -0.1524 [-0.2789, -0.0497] | EMA8-20k -0.0429 [-0.0680, -0.0209] SEP; R7-20k -0.0059 [-0.0373, +0.0261] |
| sharp (>= 0.15) | last segment | 364 / 364 | 25.4 | EMA8-20k -0.1415 [-0.2412, -0.0636]; refcv7-50400 -0.0922 [-0.1765, -0.0257]; R7-20k -0.1058 [-0.2040, -0.0350] | EMA8-20k -0.0493 [-0.0834, -0.0218] SEP; R7-20k -0.0137 [-0.0530, +0.0250] |
| sharp (>= 0.15) | segments 2..K | 367 / 1095 | 17.9 | EMA8-20k -0.1525 [-0.2543, -0.0740]; refcv7-50400 -0.1041 [-0.1873, -0.0361]; R7-20k -0.1061 [-0.2062, -0.0315] | EMA8-20k -0.0484 [-0.0797, -0.0221] SEP; R7-20k -0.0020 [-0.0422, +0.0386] |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | EMA8-20k -0.1296 [-0.2102, -0.0686]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; R7-20k -0.0925 [-0.1693, -0.0394] | EMA8-20k -0.0371 [-0.0759, -0.0017] SEP; R7-20k -0.0000 [-0.0233, +0.0247] |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | EMA8-20k -0.1380 [-0.2164, -0.0774]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; R7-20k -0.0932 [-0.1722, -0.0382] | EMA8-20k -0.0387 [-0.0751, -0.0039] SEP; R7-20k +0.0061 [-0.0160, +0.0286] |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | EMA8-20k -0.1290 [-0.2198, -0.0600]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; R7-20k -0.0853 [-0.1691, -0.0256] | EMA8-20k -0.0377 [-0.0795, +0.0003]; R7-20k +0.0059 [-0.0173, +0.0316] |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | EMA8-20k -0.1331 [-0.2225, -0.0650]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; R7-20k -0.0795 [-0.1657, -0.0181] | EMA8-20k -0.0376 [-0.0783, -0.0004] SEP; R7-20k +0.0160 [-0.0068, +0.0412] |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**EMA8-20k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4513 | 0.0512 | -0.00140 | 0.00070 | 0.495 / 0.505 / 0.000 | 308.8 / 231.0 |
| turn|straight <0.02 | 2900 | 0.0452 | 0.00002 | -0.00069 | 0.500 / 0.500 / 0.000 | 163.1 / 131.2 |
| turn|gentle 0.02-0.06 | 720 | 0.0130 | -0.01668 | -0.00700 | 0.471 / 0.529 / 0.000 | 38.4 / 9.4 |
| turn|turn 0.06-0.15 | 508 | -0.0063 | -0.01846 | -0.00782 | 0.478 / 0.522 / 0.000 | 25.3 / -3.2 |
| turn|sharp >=0.15 | 367 | 0.2564 | 0.03635 | 0.08815 | 0.529 / 0.471 / 0.000 | 52.8 / 94.1 |
| turn|undef (GT crawling) | 18 | -0.0268 | -0.02645 | -0.01649 | 0.389 / 0.611 / 0.000 | 0.6 / -0.5 |
| speed|v0 <2 | 224 | 0.1298 | -0.00624 | 0.01860 | 0.478 / 0.522 / 0.000 | 16.3 / 29.1 |
| speed|v0 2-5 | 516 | 0.0627 | -0.01308 | 0.00808 | 0.488 / 0.512 / 0.000 | 74.6 / 32.4 |
| speed|v0 5-10 | 1390 | 0.1362 | 0.00410 | 0.02668 | 0.508 / 0.492 / 0.000 | 100.3 / 189.3 |
| speed|v0 10-15 | 1192 | 0.0113 | 0.00215 | 0.00146 | 0.508 / 0.492 / 0.000 | 29.2 / 13.4 |
| speed|v0 15-25 | 856 | -0.0376 | -0.01039 | -0.01892 | 0.444 / 0.556 / 0.000 | 6.6 / -32.2 |
| speed|v0 >=25 | 335 | -0.0029 | 0.00550 | 0.00036 | 0.549 / 0.451 / 0.000 | 1.4 / -1.0 |
| nav|nav=follow | 2880 | 0.0586 | -0.00201 | -0.00448 | 0.493 / 0.507 / 0.000 | 182.8 / 168.8 |
| nav|nav=left | 405 | 0.0087 | 0.01287 | 0.02489 | 0.536 / 0.464 / 0.000 | 18.5 / 3.5 |
| nav|nav=right | 1228 | 0.0478 | -0.00303 | 0.00831 | 0.487 / 0.513 / 0.000 | 97.4 / 58.7 |
**EMA8-20k - R7-20k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4503 | -0.0219 | -0.01776 | -0.03514 | 0.438 / 0.562 / 0.000 | 307.0 / -98.7 |
| turn|straight <0.02 | 2895 | 0.0118 | -0.01149 | -0.01866 | 0.445 / 0.555 / 0.000 | 162.0 / 34.2 |
| turn|gentle 0.02-0.06 | 720 | 0.0400 | -0.02352 | -0.02886 | 0.454 / 0.546 / 0.000 | 55.3 / 28.8 |
| turn|turn 0.06-0.15 | 508 | -0.2772 | -0.12875 | -0.15716 | 0.364 / 0.636 / 0.000 | 14.2 / -140.8 |
| turn|sharp >=0.15 | 367 | -0.0657 | -0.13408 | -0.15994 | 0.452 / 0.548 / 0.000 | 42.5 / -24.1 |
| turn|undef (GT crawling) | 13 | 0.2546 | -0.02634 | 0.06095 | 0.462 / 0.538 / 0.000 | 2.8 / 3.3 |
| speed|v0 <2 | 214 | 0.0351 | -0.01425 | -0.02836 | 0.449 / 0.551 / 0.000 | 15.6 / 7.5 |
| speed|v0 2-5 | 516 | -0.1553 | -0.05572 | -0.11061 | 0.415 / 0.585 / 0.000 | 70.2 / -80.1 |
| speed|v0 5-10 | 1390 | 0.0362 | -0.03419 | -0.02015 | 0.451 / 0.549 / 0.000 | 98.0 / 50.4 |
| speed|v0 10-15 | 1192 | -0.0183 | -0.01536 | -0.02684 | 0.442 / 0.558 / 0.000 | 34.4 / -21.9 |
| speed|v0 15-25 | 856 | -0.0606 | -0.01907 | -0.03347 | 0.397 / 0.603 / 0.000 | 9.4 / -51.9 |
| speed|v0 >=25 | 335 | -0.0080 | 0.00043 | -0.00469 | 0.504 / 0.496 / 0.000 | 2.2 / -2.7 |
| nav|nav=follow | 2871 | 0.0076 | -0.01478 | -0.02795 | 0.441 / 0.559 / 0.000 | 175.2 / 21.7 |
| nav|nav=left | 405 | -0.0220 | -0.00169 | -0.00414 | 0.496 / 0.504 / 0.000 | 38.1 / -8.9 |
| nav|nav=right | 1227 | -0.0908 | -0.04261 | -0.06842 | 0.412 / 0.588 / 0.000 | 82.1 / -111.5 |
**R7-20k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4520 | 0.0748 | 0.01970 | 0.03531 | 0.567 / 0.433 / 0.000 | 264.4 / 338.0 |
| turn|straight <0.02 | 2898 | 0.0339 | 0.01302 | 0.01926 | 0.565 / 0.435 / 0.000 | 82.4 / 98.2 |
| turn|gentle 0.02-0.06 | 721 | -0.0248 | 0.02203 | 0.01421 | 0.530 / 0.470 / 0.000 | 15.5 / -17.9 |
| turn|turn 0.06-0.15 | 512 | 0.2766 | 0.11826 | 0.16205 | 0.619 / 0.381 / 0.000 | 34.9 / 141.6 |
| turn|sharp >=0.15 | 368 | 0.3185 | 0.16131 | 0.22209 | 0.584 / 0.416 / 0.000 | 54.9 / 117.2 |
| turn|undef (GT crawling) | 21 | -0.0553 | 0.03690 | 0.06354 | 0.571 / 0.429 / 0.000 | 1.0 / -1.2 |
| speed|v0 <2 | 231 | 0.1294 | 0.02261 | 0.05327 | 0.550 / 0.450 / 0.000 | 10.0 / 29.9 |
| speed|v0 2-5 | 516 | 0.2180 | 0.07220 | 0.11227 | 0.576 / 0.424 / 0.000 | 83.4 / 112.5 |
| speed|v0 5-10 | 1390 | 0.1000 | 0.03468 | 0.04677 | 0.573 / 0.427 / 0.000 | 72.8 / 138.9 |
| speed|v0 10-15 | 1192 | 0.0296 | 0.01950 | 0.03041 | 0.580 / 0.420 / 0.000 | 18.5 / 35.3 |
| speed|v0 15-25 | 856 | 0.0230 | 0.01076 | 0.01462 | 0.556 / 0.444 / 0.000 | 11.7 / 19.7 |
| speed|v0 >=25 | 335 | 0.0051 | 0.00395 | 0.00567 | 0.525 / 0.475 / 0.000 | 1.8 / 1.7 |
| nav|nav=follow | 2872 | 0.0518 | 0.01277 | 0.02296 | 0.554 / 0.446 / 0.000 | 107.9 / 148.9 |
| nav|nav=left | 412 | 0.0302 | 0.03076 | 0.03300 | 0.602 / 0.398 / 0.000 | 43.6 / 12.5 |
| nav|nav=right | 1236 | 0.1429 | 0.03541 | 0.07311 | 0.587 / 0.413 / 0.000 | 89.8 / 176.6 |

### S2 -- the episodes carrying the summed delta (panel file index only)

**EMA8-20k - refcv7-50400** (sum 231.0 over 139 episodes)

* ep024: n 35, share 0.170, mean delta 1.1207, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep012: n 34, share 0.144, mean delta 0.9757, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep013: n 34, share 0.139, mean delta 0.9421, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep108: n 24, share 0.137, mean delta 1.3179, sharp-window frac 0.00, curved frac 0.54, mean v0 8.9, nav {'nav=follow': 24, 'nav=left': 0, 'nav=right': 0}
* ep075: n 34, share 0.104, mean delta 0.7071, sharp-window frac 0.00, curved frac 0.38, mean v0 7.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep111: n 34, share 0.092, mean delta 0.6234, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
**EMA8-20k - R7-20k** (sum -98.7 over 139 episodes)

* ep104: n 35, share -0.445, mean delta 1.2557, sharp-window frac 0.29, curved frac 0.31, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 35, 'nav=right': 0}
* ep012: n 34, share -0.404, mean delta 1.1735, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep108: n 24, share -0.354, mean delta 1.4564, sharp-window frac 0.00, curved frac 0.54, mean v0 8.9, nav {'nav=follow': 24, 'nav=left': 0, 'nav=right': 0}
* ep024: n 35, share -0.349, mean delta 0.9838, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep075: n 34, share -0.246, mean delta 0.7146, sharp-window frac 0.00, curved frac 0.38, mean v0 7.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep078: n 34, share -0.197, mean delta 0.5705, sharp-window frac 0.00, curved frac 0.29, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
**R7-20k - refcv7-50400** (sum 338.0 over 139 episodes)

* ep067: n 34, share 0.140, mean delta 1.3928, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep041: n 34, share 0.101, mean delta 1.0037, sharp-window frac 0.18, curved frac 0.47, mean v0 9.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.088, mean delta 0.8721, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep082: n 31, share 0.076, mean delta 0.8308, sharp-window frac 0.00, curved frac 0.45, mean v0 8.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 31}
* ep124: n 34, share 0.056, mean delta 0.5595, sharp-window frac 0.32, curved frac 0.88, mean v0 9.7, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep129: n 31, share 0.053, mean delta 0.5812, sharp-window frac 0.32, curved frac 0.52, mean v0 4.7, nav {'nav=follow': 31, 'nav=left': 0, 'nav=right': 0}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**EMA8-20k vs R7-20k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1628 / -0.0457 [-0.0696, -0.0251] | 71 / -0.1444 [-0.4994, 0.2220] |
| only new agrees | 364 / -0.3515 [-0.5340, -0.1943] | 35 / -1.0834 [-1.9409, -0.4571] |
| only earlier agrees | 823 / 0.0994 [0.0206, 0.1960] | 38 / 0.1808 [-0.3594, 0.7947] |
| neither agrees | 1692 / 0.0073 [-0.0985, 0.1222] | 223 / 0.0805 [-0.4920, 0.6736] |
**EMA8-20k vs R7-20k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1624 / -0.0366 [-0.0593, -0.0170] | 67 / -0.0919 [-0.4838, 0.3126] |
| only new agrees | 382 / -0.3110 [-0.4897, -0.1642] | 42 / -0.4627 [-0.9772, -0.0646] |
| only earlier agrees | 818 / 0.0992 [0.0043, 0.2012] | 35 / 0.0136 [-0.8282, 0.9192] |
| neither agrees | 1679 / 0.0045 [-0.1066, 0.1231] | 223 / 0.0011 [-0.5028, 0.5907] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**EMA8-20k vs R7-20k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4481 | 0.0601 [-0.0044, 0.1267] | 0.0589 [-0.0136, 0.1443] | 0.0048 [-0.0770, 0.0848] |
| turn|straight <0.02 | 2888 | 0.0515 [-0.0001, 0.1194] | -0.0207 [-0.0518, 0.0116] | 0.0711 [0.0168, 0.1437] SEP |
| turn|gentle 0.02-0.06 | 719 | 0.0093 [-0.0985, 0.1050] | 0.0215 [-0.0281, 0.0865] | -0.0125 [-0.1268, 0.0868] |
| turn|turn 0.06-0.15 | 500 | 0.1092 [-0.0484, 0.3130] | 0.0702 [-0.1827, 0.3249] | 0.0502 [-0.1531, 0.2901] |
| turn|sharp >=0.15 | 365 | 0.1617 [-0.3897, 0.5809] | 0.7306 [0.0919, 1.5744] | -0.5425 [-1.3820, 0.0323] |
| turn|undef (GT crawling) | 9 | 0.0456 [-0.1348, 0.2315] | 0.2725 [-0.0434, 0.8481] | -0.2103 [-0.9272, 0.2267] |
| speed|v0 <2 | 192 | 0.0608 [-0.1203, 0.2194] | 0.1807 [-0.0229, 0.3663] | -0.0630 [-0.3432, 0.2134] |
| speed|v0 2-5 | 516 | 0.2683 [-0.1632, 0.7050] | 0.0293 [-0.2999, 0.4019] | 0.2390 [-0.1942, 0.6862] |
| speed|v0 5-10 | 1390 | 0.0807 [-0.0149, 0.1771] | 0.1443 [-0.0058, 0.3801] | -0.0636 [-0.2928, 0.0907] |
| speed|v0 10-15 | 1192 | 0.0029 [-0.0205, 0.0254] | 0.0070 [-0.0086, 0.0222] | -0.0041 [-0.0327, 0.0211] |
| speed|v0 15-25 | 856 | 0.0027 [-0.0143, 0.0175] | -0.0014 [-0.0103, 0.0073] | 0.0041 [-0.0124, 0.0179] |
| speed|v0 >=25 | 335 | 0.0034 [-0.0137, 0.0186] | 0.0033 [-0.0054, 0.0136] | 0.0000 [-0.0181, 0.0178] |
**EMA8-20k vs R7-20k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* EMA8-20k vs R7-20k|seed0: anchor flip rate 0.435; heading delta on flipped windows 0.0210 [-0.0875, 0.1333] (n 1952) vs unflipped -0.0584 [-0.0882, -0.0308]; goal_dist_sel proxy new-earlier -0.4187 [-0.6440, -0.1993]; corr(window dHeading, window dGoalDist) 0.016
* EMA8-20k vs R7-20k|seed1: anchor flip rate 0.439; heading delta on flipped windows 0.0241 [-0.0920, 0.1416] (n 1967) vs unflipped -0.0540 [-0.0813, -0.0270]; goal_dist_sel proxy new-earlier -0.4216 [-0.6482, -0.2026]; corr(window dHeading, window dGoalDist) 0.020

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

### S6 -- EMA8-20k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3536 | 139 | 3.3437 | 3.2404 | 0.1033 [-0.0340, 0.2507] |  |  |
| turn: straight <0.02 | 2106 | 122 | 1.6260 | 1.4581 | 0.1679 [0.0513, 0.3173] | **SEP** | 0.596 / 0.968 / 0.1597 [-0.3378, 0.5876] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.9633 | 2.8900 | 0.0733 [-0.1280, 0.2841] |  | 0.191 / 0.135 / -0.0371 [-0.4695, 0.3059] |
| turn: turn 0.06-0.15 | 435 | 59 | 6.2672 | 6.3633 | -0.0962 [-0.4387, 0.2205] |  | 0.123 / -0.115 / -0.2274 [-0.7242, 0.2391] |
| turn: sharp >=0.15 | 305 | 33 | 11.4057 | 11.4182 | -0.0126 [-0.9838, 1.1595] |  | 0.086 / -0.010 / -0.1268 [-1.4718, 1.8460] |
| turn: undef (GT crawling) (low n) | 15 | 11 | 12.9133 | 12.3899 | 0.5234 [0.1989, 1.0751] | **SEP** | 0.004 / 0.021 / 0.4220 [-0.0661, 1.4837] |
| speed: v0 <2 | 198 | 25 | 12.9008 | 13.1128 | -0.2121 [-0.9467, 0.2399] |  | 0.056 / -0.115 / -0.3340 [-1.7363, 0.3550] |
| speed: v0 2-5 | 422 | 52 | 4.3238 | 4.1501 | 0.1737 [-0.4062, 0.7927] |  | 0.119 / 0.201 / 0.0800 [-0.8277, 1.1728] |
| speed: v0 5-10 | 1040 | 79 | 3.7040 | 3.4941 | 0.2099 [-0.1055, 0.5671] |  | 0.294 / 0.598 / 0.1510 [-0.3219, 0.7422] |
| speed: v0 10-15 | 952 | 62 | 2.5479 | 2.4870 | 0.0609 [-0.1093, 0.2566] |  | 0.269 / 0.159 / -0.0581 [-0.3961, 0.3317] |
| speed: v0 15-25 | 679 | 34 | 1.5469 | 1.4456 | 0.1013 [0.0063, 0.2100] | **SEP** | 0.192 / 0.188 / -0.0024 [-0.2835, 0.2794] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4740 | 0.5193 | -0.0453 [-0.0914, 0.0039] |  | 0.069 / -0.030 / -0.1597 [-0.4062, 0.0490] |
| nav: nav=follow | 2253 | 88 | 2.1310 | 1.9769 | 0.1541 [0.0212, 0.2972] | **SEP** | 0.637 / 0.951 / 0.1401 [-0.3878, 0.5963] |
| nav: nav=left (low n) | 322 | 13 | 9.5230 | 9.6993 | -0.1763 [-0.7091, 0.3144] |  | 0.091 / -0.155 / -0.3076 [-1.2799, 0.4741] |
| nav: nav=right | 961 | 38 | 4.1162 | 4.0383 | 0.0778 [-0.2505, 0.4484] |  | 0.272 / 0.205 / -0.0349 [-0.5536, 0.5721] |

episode concentration: 0.568 of episodes have mean delta > 0; top-5 episodes carry 1.136 of the summed delta; delta excluding the top 5 = -0.0146 (overall 0.1033).

### S6 -- EMA8-20k - R7-20k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3519 | 139 | 3.3536 | 3.6854 | -0.3318 [-0.4976, -0.1616] | **SEP** |  |
| turn: straight <0.02 | 2090 | 122 | 1.6298 | 1.6658 | -0.0360 [-0.1606, 0.0974] |  | 0.594 / 0.064 / 0.7283 [0.2098, 1.2419] BONF-SEP |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.9633 | 2.9577 | 0.0056 [-0.2693, 0.3127] |  | 0.192 / -0.003 / 0.4174 [-0.0544, 0.9797] |
| turn: turn 0.06-0.15 | 435 | 59 | 6.2672 | 7.1454 | -0.8782 [-1.3914, -0.4083] | **SEP** | 0.124 / 0.327 / -0.6235 [-1.4396, 0.0850] |
| turn: sharp >=0.15 | 305 | 33 | 11.4057 | 13.7509 | -2.3452 [-3.5459, -1.2883] | **SEP** | 0.087 / 0.613 / -2.2044 [-4.3137, -0.6859] BONF-SEP |
| turn: undef (GT crawling) (low n) | 14 | 10 | 13.5628 | 13.4710 | 0.0918 [-0.2361, 0.4602] |  | 0.004 / -0.001 / 0.4253 [-0.1412, 1.0755] |
| speed: v0 <2 | 181 | 25 | 13.9914 | 14.7458 | -0.7545 [-1.1711, -0.3738] | **SEP** | 0.051 / 0.117 / -0.4456 [-1.1869, 0.2516] |
| speed: v0 2-5 | 422 | 52 | 4.3238 | 5.2980 | -0.9741 [-1.7909, -0.1837] | **SEP** | 0.120 / 0.352 / -0.7299 [-2.1448, 0.5630] |
| speed: v0 5-10 | 1040 | 79 | 3.7040 | 4.0421 | -0.3381 [-0.7224, 0.0454] |  | 0.296 / 0.301 / -0.0090 [-0.6546, 0.5819] |
| speed: v0 10-15 | 952 | 62 | 2.5479 | 2.7810 | -0.2331 [-0.4170, -0.0624] | **SEP** | 0.271 / 0.190 / 0.1353 [-0.2753, 0.5210] |
| speed: v0 15-25 | 679 | 34 | 1.5469 | 1.6197 | -0.0728 [-0.2110, 0.0537] |  | 0.193 / 0.042 / 0.3209 [-0.0323, 0.7006] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4740 | 0.4613 | 0.0126 [-0.0477, 0.0735] |  | 0.070 / -0.003 / 0.3702 [0.0604, 0.6472] BONF-SEP |
| nav: nav=follow | 2236 | 88 | 2.1374 | 2.3286 | -0.1912 [-0.3466, -0.0561] | **SEP** | 0.635 / 0.366 / 0.3855 [-0.2289, 1.0563] |
| nav: nav=left (low n) | 322 | 13 | 9.5230 | 10.0309 | -0.5080 [-1.2477, 0.2110] |  | 0.092 / 0.140 / -0.1939 [-1.7360, 0.8863] |
| nav: nav=right | 961 | 38 | 4.1162 | 4.7160 | -0.5998 [-1.0339, -0.1534] | **SEP** | 0.273 / 0.494 / -0.3687 [-1.1508, 0.4050] |

episode concentration: 0.367 of episodes have mean delta > 0; top-5 episodes carry -0.235 of the summed delta; delta excluding the top 5 = -0.4258 (overall -0.3318).

### S6 -- R7-20k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3553 | 139 | 3.7779 | 3.3013 | 0.4766 [0.3208, 0.6392] | **SEP** |  |
| turn: straight <0.02 | 2099 | 123 | 1.7062 | 1.4684 | 0.2379 [0.1461, 0.3431] | **SEP** | 0.591 / 0.295 / -0.5834 [-1.1715, -0.0085] BONF-SEP |
| turn: gentle 0.02-0.06 | 679 | 77 | 3.0078 | 2.8937 | 0.1141 [-0.2425, 0.3726] |  | 0.191 / 0.046 / -0.4481 [-1.0838, -0.0448] BONF-SEP |
| turn: turn 0.06-0.15 | 442 | 59 | 7.1717 | 6.3377 | 0.8340 [0.3322, 1.3347] | **SEP** | 0.124 / 0.218 / 0.4082 [-0.3512, 1.2173] |
| turn: sharp >=0.15 | 312 | 33 | 14.1420 | 11.7318 | 2.4102 [1.2515, 3.7058] | **SEP** | 0.088 / 0.444 / 2.1197 [0.5754, 4.3614] BONF-SEP |
| turn: undef (GT crawling) (low n) | 21 | 11 | 10.3328 | 10.5237 | -0.1909 [-0.5355, 0.1873] |  | 0.006 / -0.002 / -0.6715 [-1.2239, -0.0640] BONF-SEP |
| speed: v0 <2 | 215 | 28 | 14.5255 | 13.3387 | 1.1869 [0.4042, 2.2910] | **SEP** | 0.061 / 0.151 / 0.7560 [-0.4419, 2.6216] |
| speed: v0 2-5 | 422 | 52 | 5.2980 | 4.1501 | 1.1479 [0.2733, 2.0555] | **SEP** | 0.119 / 0.286 / 0.7617 [-0.6335, 2.4793] |
| speed: v0 5-10 | 1040 | 79 | 4.0421 | 3.4941 | 0.5480 [0.2695, 0.8686] | **SEP** | 0.293 / 0.337 / 0.1009 [-0.4337, 0.6905] |
| speed: v0 10-15 | 952 | 62 | 2.7810 | 2.4870 | 0.2940 [0.0543, 0.5494] | **SEP** | 0.268 / 0.165 / -0.2495 [-0.6955, 0.2878] |
| speed: v0 15-25 | 679 | 34 | 1.6197 | 1.4456 | 0.1741 [0.0571, 0.3164] | **SEP** | 0.191 / 0.070 / -0.3740 [-0.6823, -0.0098] BONF-SEP |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4613 | 0.5193 | -0.0580 [-0.1242, 0.0010] |  | 0.069 / -0.008 / -0.5742 [-0.8548, -0.2880] BONF-SEP |
| nav: nav=follow | 2240 | 88 | 2.3251 | 1.9803 | 0.3448 [0.1893, 0.5162] | **SEP** | 0.630 / 0.456 / -0.3566 [-0.9083, 0.1819] |
| nav: nav=left (low n) | 334 | 13 | 10.4774 | 9.8401 | 0.6373 [0.1485, 1.2859] | **SEP** | 0.094 / 0.126 / 0.1773 [-0.5691, 1.3100] |
| nav: nav=right | 979 | 38 | 4.8163 | 4.0930 | 0.7233 [0.3019, 1.1034] | **SEP** | 0.276 / 0.418 / 0.3405 [-0.3174, 0.9958] |

episode concentration: 0.705 of episodes have mean delta > 0; top-5 episodes carry 0.275 of the summed delta; delta excluding the top 5 = 0.3589 (overall 0.4766).

### S6 -- EMA8-20k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3493 | 139 | 0.9902 | 0.7640 | 0.2262 [-0.0019, 0.6521] |  | 0.1554 [-0.0808, 0.5883] |
| seg1 | straight | 2071 | 122 | 0.8381 | 0.5434 | 0.2947 [0.0137, 0.8712] | **SEP** | 0.2009 [-0.0881, 0.7789] |
| seg1 | curved | 1408 | 88 | 1.0891 | 0.9644 | 0.1247 [-0.0470, 0.4277] |  | 0.0935 [-0.0959, 0.3734] |
| seg2 | all | 3482 | 139 | 1.5801 | 1.4581 | 0.1220 [0.0144, 0.2547] | **SEP** | 0.2575 [-0.1795, 1.0321] |
| seg2 | straight | 2076 | 122 | 1.0071 | 0.8389 | 0.1681 [0.0247, 0.3858] | **SEP** | 0.2924 [-0.2097, 1.2545] |
| seg2 | curved | 1406 | 88 | 2.4261 | 2.3722 | 0.0539 [-0.0924, 0.2029] |  | 0.2060 [-0.2134, 0.7616] |
| seg3 | all | 3450 | 139 | 2.3655 | 2.0611 | 0.3044 [0.0152, 0.7635] | **SEP** | 0.5519 [-0.1898, 1.8745] |
| seg3 | straight | 2057 | 121 | 1.4049 | 0.9779 | 0.4270 [0.0129, 1.2323] | **SEP** | 0.7835 [-0.1705, 2.6921] |
| seg3 | curved | 1393 | 88 | 3.7839 | 3.6606 | 0.1233 [-0.1215, 0.3992] |  | 0.2098 [-0.3199, 0.8023] |
| seg4 | all | 3412 | 139 | 3.2834 | 2.9583 | 0.3250 [0.0259, 0.7160] | **SEP** | 0.4619 [-0.1579, 1.4946] |
| seg4 | straight | 2042 | 121 | 1.5693 | 1.1384 | 0.4310 [0.0661, 1.0729] | **SEP** | 0.7452 [-0.1632, 2.4925] |
| seg4 | curved | 1370 | 87 | 5.8382 | 5.6710 | 0.1672 [-0.2093, 0.5839] |  | 0.0396 [-0.4128, 0.4796] |
| seg5 | all | 3368 | 139 | 4.4800 | 4.2813 | 0.1987 [-0.0586, 0.4595] |  | 0.3789 [-0.1855, 1.2258] |
| seg5 | straight | 2019 | 121 | 1.6274 | 1.3080 | 0.3194 [0.0859, 0.6242] | **SEP** | 0.5487 [-0.2372, 1.9786] |
| seg5 | curved | 1349 | 88 | 8.7494 | 8.7314 | 0.0180 [-0.4732, 0.5547] |  | 0.1247 [-0.4285, 0.6868] |
| seg6 | all | 3327 | 138 | 5.9880 | 5.8836 | 0.1044 [-0.2178, 0.4153] |  | 0.0319 [-0.4044, 0.5672] |
| seg6 | straight | 2009 | 119 | 1.9036 | 1.7240 | 0.1796 [-0.1114, 0.5039] |  | 0.1763 [-0.3357, 0.8317] |
| seg6 | curved | 1317 | 88 | 12.2223 | 12.2331 | -0.0108 [-0.6952, 0.6823] |  | -0.1873 [-0.8871, 0.5587] |

### S6 -- EMA8-20k - R7-20k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3491 | 139 | 0.9907 | 0.9853 | 0.0054 [-0.0732, 0.1203] |  | -0.1342 [-0.2583, -0.0478] SEP |
| seg1 | straight | 2069 | 122 | 0.8388 | 0.6578 | 0.1810 [-0.0139, 0.5684] |  | -0.0891 [-0.1341, -0.0505] SEP |
| seg1 | curved | 1408 | 88 | 1.0891 | 1.3432 | -0.2541 [-0.5487, -0.0649] | **SEP** | -0.1977 [-0.5021, 0.0063] |
| seg2 | all | 3476 | 139 | 1.5826 | 1.6529 | -0.0704 [-0.2216, 0.0640] |  | -0.0915 [-0.2270, 0.0620] |
| seg2 | straight | 2070 | 122 | 1.0096 | 1.0553 | -0.0457 [-0.2742, 0.1033] |  | -0.0377 [-0.1970, 0.1850] |
| seg2 | curved | 1406 | 88 | 2.4261 | 2.5329 | -0.1067 [-0.2992, 0.1011] |  | -0.1707 [-0.3735, 0.0254] |
| seg3 | all | 3438 | 139 | 2.3721 | 2.4268 | -0.0547 [-0.2144, 0.1038] |  | 0.0414 [-0.1515, 0.2373] |
| seg3 | straight | 2045 | 121 | 1.4105 | 1.3329 | 0.0776 [-0.0805, 0.3025] |  | 0.1701 [-0.0083, 0.4053] |
| seg3 | curved | 1393 | 88 | 3.7839 | 4.0326 | -0.2488 [-0.5471, 0.0596] |  | -0.1476 [-0.5348, 0.2141] |
| seg4 | all | 3391 | 139 | 3.2926 | 3.5476 | -0.2551 [-0.4695, -0.0472] | **SEP** | 0.1784 [-0.0951, 0.4446] |
| seg4 | straight | 2025 | 121 | 1.5741 | 1.5241 | 0.0500 [-0.1509, 0.3079] |  | 0.3411 [0.1160, 0.6121] SEP |
| seg4 | curved | 1366 | 87 | 5.8401 | 6.5473 | -0.7072 [-1.1509, -0.2717] | **SEP** | -0.0628 [-0.6541, 0.5407] |
| seg5 | all | 3349 | 139 | 4.4929 | 5.1355 | -0.6426 [-0.9296, -0.3531] | **SEP** | 0.1386 [-0.3121, 0.5239] |
| seg5 | straight | 2002 | 121 | 1.6237 | 1.7575 | -0.1338 [-0.3386, 0.0939] |  | 0.2099 [-0.2821, 0.5764] |
| seg5 | curved | 1347 | 88 | 8.7572 | 10.1560 | -1.3989 [-2.0014, -0.7632] | **SEP** | 0.0326 [-0.7896, 0.8170] |
| seg6 | all | 3306 | 138 | 5.9991 | 6.9578 | -0.9587 [-1.3455, -0.5896] | **SEP** | 0.2890 [-0.2281, 0.7961] |
| seg6 | straight | 1992 | 119 | 1.8903 | 2.1150 | -0.2247 [-0.4987, 0.0732] |  | 0.3645 [-0.1705, 0.8070] |
| seg6 | curved | 1313 | 88 | 12.2370 | 14.3087 | -2.0717 [-2.8840, -1.2490] | **SEP** | 0.1735 [-0.8455, 1.1690] |

### S6 -- R7-20k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3501 | 139 | 1.0349 | 0.8146 | 0.2203 [0.0403, 0.5454] | **SEP** | 0.2892 [0.0013, 0.8230] SEP |
| seg1 | straight | 2072 | 122 | 0.6572 | 0.5433 | 0.1139 [0.0175, 0.3106] | **SEP** | 0.2903 [0.0082, 0.8694] SEP |
| seg1 | curved | 1410 | 88 | 1.4700 | 1.0917 | 0.3782 [0.0662, 0.9424] | **SEP** | 0.2908 [-0.0309, 0.8461] |
| seg2 | all | 3485 | 139 | 1.6554 | 1.4614 | 0.1940 [0.0195, 0.4512] | **SEP** | 0.3507 [-0.0159, 0.9989] |
| seg2 | straight | 2074 | 122 | 1.0543 | 0.8393 | 0.2151 [0.0246, 0.5990] | **SEP** | 0.3294 [-0.0552, 1.0995] |
| seg2 | curved | 1411 | 88 | 2.5389 | 2.3758 | 0.1632 [-0.0471, 0.3748] |  | 0.3822 [-0.0005, 0.9312] |
| seg3 | all | 3458 | 139 | 2.4449 | 2.0719 | 0.3730 [0.1132, 0.7424] | **SEP** | 0.5079 [-0.1707, 1.7158] |
| seg3 | straight | 2053 | 122 | 1.3310 | 0.9779 | 0.3531 [0.0590, 0.9477] | **SEP** | 0.6138 [-0.2312, 2.3653] |
| seg3 | curved | 1405 | 88 | 4.0726 | 3.6705 | 0.4021 [0.0589, 0.7412] | **SEP** | 0.3532 [-0.1600, 0.9748] |
| seg4 | all | 3428 | 139 | 3.5949 | 2.9802 | 0.6147 [0.3368, 0.9412] | **SEP** | 0.2839 [-0.3123, 1.2199] |
| seg4 | straight | 2041 | 122 | 1.5269 | 1.1359 | 0.3910 [0.1465, 0.8095] | **SEP** | 0.3913 [-0.4062, 1.9697] |
| seg4 | curved | 1387 | 87 | 6.6381 | 5.6943 | 0.9438 [0.4438, 1.4316] | **SEP** | 0.1257 [-0.4377, 0.6708] |
| seg5 | all | 3396 | 139 | 5.2468 | 4.3430 | 0.9037 [0.6252, 1.2007] | **SEP** | 0.2115 [-0.5311, 1.3192] |
| seg5 | straight | 2020 | 122 | 1.7976 | 1.3137 | 0.4839 [0.2685, 0.7550] | **SEP** | 0.2696 [-0.7013, 2.1164] |
| seg5 | curved | 1376 | 88 | 10.3102 | 8.7902 | 1.5201 [0.8910, 2.1643] | **SEP** | 0.1261 [-0.6410, 0.9120] |
| seg6 | all | 3361 | 138 | 7.1408 | 6.0071 | 1.1337 [0.7956, 1.4877] | **SEP** | -0.2801 [-0.8634, 0.4655] |
| seg6 | straight | 2009 | 121 | 2.1736 | 1.7386 | 0.4351 [0.1206, 0.7248] | **SEP** | -0.2906 [-0.9830, 0.8212] |
| seg6 | curved | 1350 | 88 | 14.5231 | 12.3433 | 2.1798 [1.4171, 2.9309] | **SEP** | -0.2697 [-1.1622, 0.6428] |

### S6 -- EMA8-20k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 372 | 39 | 5.2853 | 5.0849 | 0.2004 [-0.0378, 0.5365] |  |  |
| straight <0.02 | v0 5-15 | 1050 | 84 | 1.0418 | 0.7632 | 0.2786 [0.0854, 0.5379] | **SEP** |  |
| straight <0.02 | v0 >=15 | 684 | 36 | 0.5326 | 0.5523 | -0.0197 [-0.0512, 0.0159] |  |  |
| gentle 0.02-0.06 | v0 <5 (low n) | 98 | 22 | 5.4244 | 5.5283 | -0.1038 [-0.8781, 0.3679] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 424 | 59 | 2.5581 | 2.5310 | 0.0271 [-0.2102, 0.2997] |  |  |
| gentle 0.02-0.06 | v0 >=15 (low n) | 153 | 14 | 2.5100 | 2.1952 | 0.3148 [0.0447, 0.6125] | **SEP** |  |
| turn 0.06-0.15 | v0 <5 (low n) | 64 | 20 | 10.5390 | 11.2985 | -0.7595 [-1.4768, -0.1315] | **SEP** |  |
| turn 0.06-0.15 | v0 5-15 | 293 | 43 | 5.7557 | 5.8272 | -0.0715 [-0.4992, 0.3383] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 78 | 10 | 4.6835 | 4.3281 | 0.3555 [0.0051, 0.7544] | **SEP** |  |
| sharp >=0.15 | v0 <5 (low n) | 71 | 16 | 14.2687 | 14.1603 | 0.1084 [-2.3029, 3.5733] |  |  |
| sharp >=0.15 | v0 5-15 | 225 | 26 | 10.7238 | 10.7541 | -0.0303 [-1.1171, 1.1010] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 9 | 3 | 5.8672 | 6.3909 | -0.5236 [-1.0584, 0.9880] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 15 | 11 | 12.9133 | 12.3899 | 0.5234 [0.1989, 1.0751] | **SEP** |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S6 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | EMA8-20k s0-s1 [CI] | R7-20k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0423 [-0.0944, 0.0107] | 0.0032 [-0.0496, 0.0598] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | -0.0091 [-0.0673, 0.0348] | 0.0262 [-0.0207, 0.0960] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | -0.0273 [-0.1687, 0.1055] | 0.0292 [-0.0744, 0.1218] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | -0.1166 [-0.2760, 0.0260] | -0.0675 [-0.2282, 0.1022] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | -0.1986 [-0.4504, 0.0589] | -0.1077 [-0.5568, 0.3422] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | -0.0504 [-0.6076, 0.5167] | 0.0091 [-0.1871, 0.2769] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | -0.0378 [-0.2881, 0.2203] | 0.4396 [0.0260, 1.0407] SEP | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | -0.0458 [-0.2084, 0.0837] | -0.1181 [-0.4121, 0.0983] | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | -0.0501 [-0.2012, 0.0771] | 0.0358 [-0.1045, 0.1956] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | -0.0731 [-0.1471, -0.0100] SEP | -0.0728 [-0.1371, -0.0117] SEP | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | -0.0120 [-0.0465, 0.0218] | 0.0008 [-0.0513, 0.0624] | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | 0.0290 [-0.0024, 0.0730] | -0.0123 [-0.0357, 0.0129] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | -0.0247 [-0.0719, 0.0256] | -0.0214 [-0.0692, 0.0290] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | -0.0168 [-0.2558, 0.2619] | 0.0776 [0.0039, 0.1823] SEP | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | -0.0922 [-0.2292, 0.0195] | 0.0340 [-0.1064, 0.1917] | 0.0235 [-0.1175, 0.1775] |

### S6 -- EMA8-20k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1033 | 0.0722 / 0.1344 | False / False / False | 0.0423 / 0.0128 | 2.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.1679 | 0.1765 / 0.1592 | True / True / True | 0.0091 / 0.0216 | 7.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0733 | 0.0235 / 0.1231 | False / False / False | 0.0273 / 0.0713 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.0962 | -0.1765 / -0.0158 | False / False / False | 0.1166 / 0.0611 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | -0.0126 | -0.2047 / 0.1796 | False / False / False | 0.1986 / 0.2004 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.5234 | 0.4620 / 0.5849 | True / True / True | 0.0504 / 1.9278 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.2121 | -0.3123 / -0.1118 | False / False / False | 0.0378 / 0.0328 | 5.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.1737 | 0.0503 / 0.2971 | False / False / False | 0.0458 / 0.2009 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.2099 | 0.1970 / 0.2228 | False / False / False | 0.0501 / 0.0243 | 4.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.0609 | 0.0377 / 0.0840 | False / False / False | 0.0731 / 0.0268 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.1013 | 0.0926 / 0.1101 | True / True / True | 0.0120 / 0.0055 | 8.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0453 | -0.0312 / -0.0594 | False / True / False | 0.0290 / 0.0008 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.1541 | 0.1373 / 0.1710 | False / True / True | 0.0247 / 0.0080 | 6.2 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| nav|nav=left | -0.1763 | -0.1819 / -0.1707 | False / False / False | 0.0168 / 0.0140 | 10.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0778 | 0.0049 / 0.1508 | False / False / False | 0.0922 / 0.0235 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- EMA8-20k - R7-20k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.3318 | -0.3492 / -0.3146 | True / True / True | 0.0423 / 0.0032 | 7.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | -0.0360 | -0.0464 / -0.0263 | False / False / False | 0.0091 / 0.0262 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.0056 | -0.0279 / 0.0391 | False / False / False | 0.0273 / 0.0292 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.8782 | -0.8994 / -0.8570 | True / True / True | 0.1166 / 0.0675 | 7.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | -2.3452 | -2.3682 / -2.3221 | True / True / True | 0.1986 / 0.1077 | 11.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.0918 | -0.0088 / 0.1925 | False / False / False | 0.0504 / 0.0091 | 1.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.7545 | -0.9269 / -0.5796 | True / True / True | 0.0378 / 0.4396 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.9741 | -0.9380 / -1.0103 | True / True / True | 0.0458 / 0.1181 | 8.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 5-10 | -0.3381 | -0.3811 / -0.2952 | True / False / False | 0.0501 / 0.0358 | 6.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.2331 | -0.2333 / -0.2330 | True / True / True | 0.0731 / 0.0728 | 3.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | -0.0728 | -0.0792 / -0.0664 | False / False / False | 0.0120 / 0.0008 | 6.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0126 | 0.0333 / -0.0080 | False / False / False | 0.0290 / 0.0123 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | -0.1912 | -0.1929 / -0.1900 | True / True / True | 0.0247 / 0.0214 | 7.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -0.5080 | -0.5251 / -0.4908 | False / False / False | 0.0168 / 0.0776 | 6.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.5998 | -0.6542 / -0.5454 | True / True / True | 0.0922 / 0.0340 | 6.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S6 -- R7-20k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.4766 | 0.4724 / 0.4773 | True / True / True | 0.0032 / 0.0128 | 37.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.2379 | 0.2626 / 0.2136 | True / True / True | 0.0262 / 0.0216 | 9.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.1141 | 0.0930 / 0.1352 | False / False / False | 0.0292 / 0.0713 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.8340 | 0.7697 / 0.8982 | True / True / True | 0.0675 / 0.0611 | 12.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 2.4102 | 2.2609 / 2.5126 | True / True / True | 0.1077 / 0.2004 | 12.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | -0.1909 | 0.9116 / -1.2935 | True / False / False | 0.0091 / 1.9278 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 1.1869 | 1.3893 / 0.9127 | True / True / True | 0.4396 / 0.0328 | 2.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 2-5 | 1.1479 | 0.9884 / 1.3073 | True / True / True | 0.1181 / 0.2009 | 5.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 5-10 | 0.5480 | 0.5780 / 0.5180 | True / True / True | 0.0358 / 0.0243 | 15.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.2940 | 0.2710 / 0.3170 | True / True / True | 0.0728 / 0.0268 | 4.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.1741 | 0.1718 / 0.1765 | True / True / True | 0.0008 / 0.0055 | 31.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0580 | -0.0645 / -0.0514 | False / False / False | 0.0123 / 0.0008 | 4.7 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.3448 | 0.3299 / 0.3599 | True / True / True | 0.0214 / 0.0080 | 16.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.6373 | 0.6777 / 0.5589 | True / True / True | 0.0776 / 0.0140 | 8.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=right | 0.7233 | 0.7284 / 0.7182 | True / True / True | 0.0340 / 0.0235 | 21.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S6 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1408): EMA8-20k bias +0.254, refcv7-50400 bias +0.166, R7-20k bias +0.524; EMA8-20k MAE 1.089, refcv7-50400 MAE 0.964, R7-20k MAE 1.343
* seg2 (n 1406): EMA8-20k bias -0.094, refcv7-50400 bias -0.232, R7-20k bias -0.034; EMA8-20k MAE 2.426, refcv7-50400 MAE 2.372, R7-20k MAE 2.533
* seg3 (n 1393): EMA8-20k bias -1.066, refcv7-50400 bias -1.097, R7-20k bias -1.123; EMA8-20k MAE 3.784, refcv7-50400 MAE 3.661, R7-20k MAE 4.033
* seg4 (n 1366): EMA8-20k bias -2.746, refcv7-50400 bias -2.511, R7-20k bias -3.066; EMA8-20k MAE 5.840, refcv7-50400 MAE 5.653, R7-20k MAE 6.547
* seg5 (n 1347): EMA8-20k bias -4.577, refcv7-50400 bias -4.451, R7-20k bias -5.318; EMA8-20k MAE 8.757, refcv7-50400 MAE 8.718, R7-20k MAE 10.156
* seg6 (n 1313): EMA8-20k bias -6.533, refcv7-50400 bias -6.810, R7-20k bias -7.351; EMA8-20k MAE 12.237, refcv7-50400 MAE 12.227, R7-20k MAE 14.309
**sharp (>= 0.15)**

* seg1 (n 303): EMA8-20k bias -0.028, refcv7-50400 bias +0.175, R7-20k bias +0.983; EMA8-20k MAE 1.372, refcv7-50400 MAE 1.438, R7-20k MAE 2.394
* seg2 (n 298): EMA8-20k bias -1.195, refcv7-50400 bias -0.532, R7-20k bias -1.205; EMA8-20k MAE 4.018, refcv7-50400 MAE 3.819, R7-20k MAE 4.326
* seg3 (n 297): EMA8-20k bias -3.795, refcv7-50400 bias -2.411, R7-20k bias -4.737; EMA8-20k MAE 7.239, refcv7-50400 MAE 6.658, R7-20k MAE 8.039
* seg4 (n 293): EMA8-20k bias -8.157, refcv7-50400 bias -6.504, R7-20k bias -10.262; EMA8-20k MAE 12.266, refcv7-50400 MAE 11.524, R7-20k MAE 14.498
* seg5 (n 289): EMA8-20k bias -12.744, refcv7-50400 bias -11.640, R7-20k bias -16.477; EMA8-20k MAE 18.618, refcv7-50400 MAE 18.692, R7-20k MAE 22.890
* seg6 (n 284): EMA8-20k bias -17.180, refcv7-50400 bias -17.056, R7-20k bias -20.621; EMA8-20k MAE 24.400, refcv7-50400 MAE 24.727, R7-20k MAE 30.485
**curved & v0 5-10**

* seg1 (n 547): EMA8-20k bias +0.044, refcv7-50400 bias +0.049, R7-20k bias +0.199; EMA8-20k MAE 0.933, refcv7-50400 MAE 0.896, R7-20k MAE 1.009
* seg2 (n 547): EMA8-20k bias -0.520, refcv7-50400 bias -0.516, R7-20k bias -0.417; EMA8-20k MAE 2.427, refcv7-50400 MAE 2.231, R7-20k MAE 2.356
* seg3 (n 545): EMA8-20k bias -1.731, refcv7-50400 bias -1.649, R7-20k bias -1.729; EMA8-20k MAE 4.066, refcv7-50400 MAE 3.822, R7-20k MAE 4.200
* seg4 (n 532): EMA8-20k bias -3.620, refcv7-50400 bias -3.559, R7-20k bias -3.753; EMA8-20k MAE 6.243, refcv7-50400 MAE 6.219, R7-20k MAE 7.039
* seg5 (n 516): EMA8-20k bias -5.934, refcv7-50400 bias -5.853, R7-20k bias -6.105; EMA8-20k MAE 9.193, refcv7-50400 MAE 9.361, R7-20k MAE 10.664
* seg6 (n 490): EMA8-20k bias -7.846, refcv7-50400 bias -8.180, R7-20k bias -7.856; EMA8-20k MAE 12.448, refcv7-50400 MAE 12.522, R7-20k MAE 14.535
**sharp & v0 5-10**

* seg1 (n 139): EMA8-20k bias +0.001, refcv7-50400 bias +0.062, R7-20k bias +0.642; EMA8-20k MAE 1.201, refcv7-50400 MAE 1.251, R7-20k MAE 1.538
* seg2 (n 139): EMA8-20k bias -1.658, refcv7-50400 bias -1.267, R7-20k bias -0.900; EMA8-20k MAE 3.936, refcv7-50400 MAE 3.564, R7-20k MAE 3.694
* seg3 (n 139): EMA8-20k bias -5.142, refcv7-50400 bias -4.133, R7-20k bias -5.173; EMA8-20k MAE 7.646, refcv7-50400 MAE 7.029, R7-20k MAE 7.618
* seg4 (n 137): EMA8-20k bias -10.402, refcv7-50400 bias -9.127, R7-20k bias -11.030; EMA8-20k MAE 12.764, refcv7-50400 MAE 12.522, R7-20k MAE 14.329
* seg5 (n 136): EMA8-20k bias -15.117, refcv7-50400 bias -14.401, R7-20k bias -16.099; EMA8-20k MAE 19.682, refcv7-50400 MAE 19.814, R7-20k MAE 22.609
* seg6 (n 131): EMA8-20k bias -17.595, refcv7-50400 bias -18.177, R7-20k bias -17.414; EMA8-20k MAE 25.083, refcv7-50400 MAE 25.484, R7-20k MAE 29.278

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1313 / 1313 | 26.1 | EMA8-20k -0.2347 [-0.3198, -0.1617]; refcv7-50400 -0.2431 [-0.3124, -0.1770]; R7-20k -0.2764 [-0.3547, -0.2089] | EMA8-20k +0.0084 [-0.0257, +0.0365]; R7-20k -0.0333 [-0.0738, +0.0005] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1414 / 6825 | 16.3 | EMA8-20k -0.1617 [-0.2543, -0.0766]; refcv7-50400 -0.1762 [-0.2407, -0.1195]; R7-20k -0.1901 [-0.2730, -0.1100] | EMA8-20k +0.0146 [-0.0388, +0.0874]; R7-20k -0.0139 [-0.0679, +0.0607] |
| sharp (>= 0.15) | last segment | 284 / 284 | 63.1 | EMA8-20k -0.2571 [-0.3655, -0.1698]; refcv7-50400 -0.2586 [-0.3419, -0.1838]; R7-20k -0.3054 [-0.4089, -0.2261] | EMA8-20k +0.0015 [-0.0447, +0.0371]; R7-20k -0.0468 [-0.1037, -0.0044] SEP |
| sharp (>= 0.15) | segments 2..K | 304 / 1461 | 39.8 | EMA8-20k -0.2160 [-0.3236, -0.1297]; refcv7-50400 -0.1907 [-0.2738, -0.1208]; R7-20k -0.2562 [-0.3611, -0.1777] | EMA8-20k -0.0252 [-0.0726, +0.0085]; R7-20k -0.0654 [-0.1181, -0.0267] SEP |
| curved & v0 5-10 | last segment | 490 / 490 | 28.2 | EMA8-20k -0.2520 [-0.3678, -0.1461]; refcv7-50400 -0.2628 [-0.3845, -0.1667]; R7-20k -0.2640 [-0.3859, -0.1587] | EMA8-20k +0.0107 [-0.0492, +0.0674]; R7-20k -0.0012 [-0.0513, +0.0386] |
| curved & v0 5-10 | segments 2..K | 547 / 2630 | 17.8 | EMA8-20k -0.2112 [-0.3187, -0.1216]; refcv7-50400 -0.2018 [-0.3144, -0.1183]; R7-20k -0.2197 [-0.3216, -0.1337] | EMA8-20k -0.0095 [-0.0668, +0.0404]; R7-20k -0.0179 [-0.0573, +0.0172] |
| sharp & v0 5-10 | last segment | 131 / 131 | 67.2 | EMA8-20k -0.2556 [-0.3920, -0.1404]; refcv7-50400 -0.2613 [-0.3923, -0.1569]; R7-20k -0.2603 [-0.3919, -0.1458] | EMA8-20k +0.0057 [-0.0623, +0.0669]; R7-20k +0.0011 [-0.0594, +0.0468] |
| sharp & v0 5-10 | segments 2..K | 139 / 682 | 43.8 | EMA8-20k -0.2236 [-0.3517, -0.1189]; refcv7-50400 -0.2068 [-0.3341, -0.1122]; R7-20k -0.2310 [-0.3494, -0.1293] | EMA8-20k -0.0168 [-0.0833, +0.0394]; R7-20k -0.0242 [-0.0773, +0.0182] |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**EMA8-20k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3536 | 0.1033 | 0.03082 | 0.04277 | 0.540 / 0.460 / 0.000 | 625.9 / 365.2 |
| turn|straight <0.02 | 2106 | 0.1679 | 0.03879 | 0.05037 | 0.568 / 0.432 / 0.000 | 257.9 / 353.6 |
| turn|gentle 0.02-0.06 | 675 | 0.0733 | -0.00447 | -0.00590 | 0.496 / 0.504 / 0.000 | 71.0 / 49.5 |
| turn|turn 0.06-0.15 | 435 | -0.0962 | 0.03403 | 0.02648 | 0.520 / 0.480 / 0.000 | 46.7 / -41.8 |
| turn|sharp >=0.15 | 305 | -0.0126 | -0.31191 | -0.20886 | 0.452 / 0.548 / 0.000 | 158.1 / -3.8 |
| turn|undef (GT crawling) | 15 | 0.5234 | 0.25985 | 0.33172 | 0.933 / 0.067 / 0.000 | 3.6 / 7.9 |
| speed|v0 <2 | 198 | -0.2121 | 0.02118 | 0.01645 | 0.545 / 0.455 / 0.000 | 26.7 / -42.0 |
| speed|v0 2-5 | 422 | 0.1737 | 0.07554 | 0.11699 | 0.559 / 0.441 / 0.000 | 111.8 / 73.3 |
| speed|v0 5-10 | 1040 | 0.2099 | 0.00755 | 0.02545 | 0.510 / 0.490 / 0.000 | 295.0 / 218.3 |
| speed|v0 10-15 | 952 | 0.0609 | 0.06176 | 0.07668 | 0.605 / 0.395 / 0.000 | 76.7 / 57.9 |
| speed|v0 15-25 | 679 | 0.1013 | 0.01786 | 0.01799 | 0.530 / 0.470 / 0.000 | 40.1 / 68.8 |
| speed|v0 >=25 | 245 | -0.0453 | -0.03599 | -0.04080 | 0.404 / 0.596 / 0.000 | 3.5 / -11.1 |
| nav|nav=follow | 2253 | 0.1541 | 0.02219 | 0.02472 | 0.528 / 0.472 / 0.000 | 306.4 / 347.2 |
| nav|nav=left | 322 | -0.1763 | 0.08208 | 0.11383 | 0.602 / 0.398 / 0.000 | 69.6 / -56.8 |
| nav|nav=right | 961 | 0.0778 | 0.06243 | 0.04787 | 0.546 / 0.454 / 0.000 | 231.5 / 74.8 |
**EMA8-20k - R7-20k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3519 | -0.3318 | -0.07325 | -0.17574 | 0.422 / 0.578 / 0.000 | 654.1 / -1167.5 |
| turn|straight <0.02 | 2090 | -0.0360 | -0.02480 | -0.05189 | 0.459 / 0.541 / 0.000 | 246.2 / -75.3 |
| turn|gentle 0.02-0.06 | 675 | 0.0056 | -0.13815 | -0.17429 | 0.409 / 0.591 / 0.000 | 130.8 / 3.8 |
| turn|turn 0.06-0.15 | 435 | -0.8782 | -0.49286 | -0.62302 | 0.340 / 0.660 / 0.000 | 82.4 / -382.0 |
| turn|sharp >=0.15 | 305 | -2.3452 | -1.11644 | -1.76768 | 0.315 / 0.685 / 0.000 | 113.5 / -715.3 |
| turn|undef (GT crawling) | 14 | 0.0918 | -0.00286 | 0.10357 | 0.429 / 0.571 / 0.000 | 0.9 / 1.3 |
| speed|v0 <2 | 181 | -0.7545 | -0.45639 | -0.63684 | 0.287 / 0.713 / 0.000 | 27.5 / -136.6 |
| speed|v0 2-5 | 422 | -0.9741 | -0.39078 | -0.54749 | 0.341 / 0.659 / 0.000 | 119.4 / -411.1 |
| speed|v0 5-10 | 1040 | -0.3381 | -0.15330 | -0.22555 | 0.406 / 0.594 / 0.000 | 258.5 / -351.6 |
| speed|v0 10-15 | 952 | -0.2331 | -0.03634 | -0.09502 | 0.458 / 0.542 / 0.000 | 106.2 / -221.9 |
| speed|v0 15-25 | 679 | -0.0728 | -0.02230 | -0.04657 | 0.455 / 0.545 / 0.000 | 38.9 / -49.4 |
| speed|v0 >=25 | 245 | 0.0126 | -0.00416 | -0.00220 | 0.498 / 0.502 / 0.000 | 4.2 / 3.1 |
| nav|nav=follow | 2236 | -0.1912 | -0.05244 | -0.12278 | 0.428 / 0.572 / 0.000 | 299.0 / -427.6 |
| nav|nav=left | 322 | -0.5080 | 0.01146 | -0.06037 | 0.506 / 0.494 / 0.000 | 78.3 / -163.6 |
| nav|nav=right | 961 | -0.5998 | -0.18378 | -0.38450 | 0.380 / 0.620 / 0.000 | 238.7 / -576.4 |
**R7-20k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3553 | 0.4766 | 0.10028 | 0.20807 | 0.620 / 0.380 / 0.000 | 855.2 / 1693.4 |
| turn|straight <0.02 | 2099 | 0.2379 | 0.07444 | 0.12359 | 0.627 / 0.373 / 0.000 | 182.9 / 499.3 |
| turn|gentle 0.02-0.06 | 679 | 0.1141 | 0.10517 | 0.15251 | 0.589 / 0.411 / 0.000 | 53.7 / 77.5 |
| turn|turn 0.06-0.15 | 442 | 0.8340 | 0.43138 | 0.64440 | 0.631 / 0.369 / 0.000 | 106.9 / 368.6 |
| turn|sharp >=0.15 | 312 | 2.4102 | 0.75762 | 1.57327 | 0.625 / 0.375 / 0.000 | 176.8 / 752.0 |
| turn|undef (GT crawling) | 21 | -0.1909 | 0.01568 | 0.03616 | 0.524 / 0.476 / 0.000 | 1.0 / -4.0 |
| speed|v0 <2 | 215 | 1.1869 | 0.64327 | 0.82983 | 0.712 / 0.288 / 0.000 | 54.3 / 255.2 |
| speed|v0 2-5 | 422 | 1.1479 | 0.55297 | 0.70166 | 0.732 / 0.268 / 0.000 | 215.5 / 484.4 |
| speed|v0 5-10 | 1040 | 0.5480 | 0.16613 | 0.23231 | 0.627 / 0.373 / 0.000 | 287.0 / 569.9 |
| speed|v0 10-15 | 952 | 0.2940 | 0.08256 | 0.13515 | 0.633 / 0.367 / 0.000 | 111.5 / 279.8 |
| speed|v0 15-25 | 679 | 0.1741 | 0.02646 | 0.04804 | 0.558 / 0.442 / 0.000 | 40.0 / 118.2 |
| speed|v0 >=25 | 245 | -0.0580 | -0.03136 | -0.03594 | 0.433 / 0.567 / 0.000 | 2.1 / -14.2 |
| nav|nav=follow | 2240 | 0.3448 | 0.07247 | 0.13962 | 0.604 / 0.396 / 0.000 | 362.0 / 772.4 |
| nav|nav=left | 334 | 0.6373 | 0.08693 | 0.19686 | 0.608 / 0.392 / 0.000 | 117.6 / 212.9 |
| nav|nav=right | 979 | 0.7233 | 0.22736 | 0.42770 | 0.659 / 0.341 / 0.000 | 300.7 / 708.1 |

### S6 -- the episodes carrying the summed delta (panel file index only)

**EMA8-20k - refcv7-50400** (sum 365.2 over 139 episodes)

* ep059: n 27, share 0.311, mean delta 4.2016, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep111: n 26, share 0.258, mean delta 3.6240, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep013: n 27, share 0.223, mean delta 3.0101, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep085: n 26, share 0.206, mean delta 2.8935, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep121: n 27, share 0.139, mean delta 1.8795, sharp-window frac 0.00, curved frac 0.67, mean v0 6.8, nav {'nav=follow': 27, 'nav=left': 0, 'nav=right': 0}
* ep024: n 27, share 0.127, mean delta 1.7243, sharp-window frac 0.00, curved frac 0.00, mean v0 5.1, nav {'nav=follow': 27, 'nav=left': 0, 'nav=right': 0}
**EMA8-20k - R7-20k** (sum -1167.5 over 139 episodes)

* ep059: n 27, share -0.078, mean delta 3.3581, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep078: n 26, share -0.051, mean delta 2.3025, sharp-window frac 0.00, curved frac 0.54, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep048: n 27, share -0.038, mean delta 1.6396, sharp-window frac 0.00, curved frac 0.67, mean v0 6.9, nav {'nav=follow': 0, 'nav=left': 27, 'nav=right': 0}
* ep024: n 27, share -0.035, mean delta 1.5063, sharp-window frac 0.00, curved frac 0.00, mean v0 5.1, nav {'nav=follow': 27, 'nav=left': 0, 'nav=right': 0}
* ep012: n 26, share -0.033, mean delta 1.4927, sharp-window frac 0.46, curved frac 0.77, mean v0 7.0, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep029: n 26, share -0.032, mean delta 1.4389, sharp-window frac 0.00, curved frac 1.00, mean v0 12.1, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
**R7-20k - refcv7-50400** (sum 1693.4 over 139 episodes)

* ep085: n 26, share 0.058, mean delta 3.7941, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep100: n 26, share 0.056, mean delta 3.6358, sharp-window frac 0.35, curved frac 0.85, mean v0 3.6, nav {'nav=follow': 0, 'nav=left': 26, 'nav=right': 0}
* ep013: n 27, share 0.055, mean delta 3.4561, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep041: n 26, share 0.054, mean delta 3.5224, sharp-window frac 0.12, curved frac 0.38, mean v0 8.6, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.051, mean delta 3.3460, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep067: n 26, share 0.045, mean delta 2.9568, sharp-window frac 0.35, curved frac 0.96, mean v0 4.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**EMA8-20k vs R7-20k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1307 / -0.1615 [-0.2336, -0.1001] | 54 / -0.1776 [-0.6979, 0.3212] |
| only new agrees | 302 / -2.1435 [-2.9710, -1.4316] | 34 / -9.7049 [-14.5827, -5.9154] |
| only earlier agrees | 584 / 0.2197 [0.0097, 0.4310] | 21 / -1.3440 [-3.8871, 0.6275] |
| neither agrees | 1328 / -0.3761 [-0.6920, -0.0573] | 196 / -1.8088 [-3.3652, -0.4350] |
**EMA8-20k vs R7-20k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1303 / -0.1734 [-0.2536, -0.1103] | 54 / -0.3676 [-1.1532, 0.3296] |
| only new agrees | 316 / -1.9811 [-2.7804, -1.2679] | 37 / -9.0143 [-14.4040, -4.8576] |
| only earlier agrees | 582 / 0.2379 [0.0014, 0.5096] | 21 / -0.8849 [-2.8971, 2.0889] |
| neither agrees | 1318 / -0.2985 [-0.6280, 0.0446] | 193 / -1.7424 [-3.3605, -0.3872] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**EMA8-20k vs R7-20k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3480 | 0.1955 [-0.0370, 0.4418] | 0.2188 [-0.0458, 0.5550] | -0.0104 [-0.2794, 0.2130] |
| turn|straight <0.02 | 2073 | -0.0600 [-0.1772, 0.0325] | -0.1226 [-0.2141, -0.0514] | 0.0415 [-0.0711, 0.1650] |
| turn|gentle 0.02-0.06 | 674 | -0.2499 [-0.7483, 0.1135] | -0.2909 [-0.6428, 0.0101] | 0.0230 [-0.2545, 0.2706] |
| turn|turn 0.06-0.15 | 424 | 0.2901 [-0.0791, 0.6478] | -0.1136 [-0.5918, 0.3094] | 0.5046 [0.1531, 0.8822] SEP |
| turn|sharp >=0.15 | 301 | 2.8396 [0.8589, 5.1385] | 4.1074 [1.6804, 7.1485] | -1.1710 [-3.8971, 1.0394] |
| turn|undef (GT crawling) | 8 | -0.0798 [-0.4826, 0.3063] | -0.3843 [-0.8422, 0.1214] | 0.1011 [-0.4531, 0.8121] |
| speed|v0 <2 | 142 | 0.2469 [-0.0543, 0.6313] | -0.0013 [-0.8575, 0.8521] | 0.6796 [0.3844, 1.1002] SEP |
| speed|v0 2-5 | 422 | 0.4059 [-0.7976, 1.6074] | 0.5277 [-0.6093, 1.9391] | -0.1218 [-1.0663, 0.6883] |
| speed|v0 5-10 | 1040 | 0.4998 [0.0184, 1.0453] | 0.5521 [-0.0683, 1.4855] | -0.0523 [-0.8913, 0.5435] |
| speed|v0 10-15 | 952 | -0.0572 [-0.1820, 0.0451] | 0.0021 [-0.0834, 0.0813] | -0.0593 [-0.2021, 0.0545] |
| speed|v0 15-25 | 679 | 0.0121 [-0.0378, 0.0610] | -0.0380 [-0.1045, 0.0137] | 0.0501 [-0.0076, 0.1206] |
| speed|v0 >=25 | 245 | -0.0011 [-0.0349, 0.0384] | 0.0175 [-0.0140, 0.0529] | -0.0185 [-0.0758, 0.0379] |
**EMA8-20k vs R7-20k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* EMA8-20k vs R7-20k|seed0: anchor flip rate 0.417; heading delta on flipped windows -0.5348 [-0.9099, -0.1384] (n 1461) vs unflipped -0.2175 [-0.2908, -0.1572]; goal_dist_sel proxy new-earlier -0.3917 [-0.6377, -0.1470]; corr(window dHeading, window dGoalDist) -0.020
* EMA8-20k vs R7-20k|seed1: anchor flip rate 0.424; heading delta on flipped windows -0.4404 [-0.8431, -0.0387] (n 1485) vs unflipped -0.2227 [-0.2954, -0.1588]; goal_dist_sel proxy new-earlier -0.3976 [-0.6409, -0.1499]; corr(window dHeading, window dGoalDist) -0.019
