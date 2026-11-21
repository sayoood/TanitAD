# refcv8 heading-drift localisation: R7-15k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-19T18:11:00; trees: `R7-15k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step15000_unc; `R7-30k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step30000_unc; `R7-20k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step20000_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **None**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

### S2 -- R7-15k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4526 | 139 | 1.6096 | 1.4461 | 0.1635 [0.1035, 0.2261] | **SEP** |  |
| turn: straight <0.02 | 2901 | 132 | 1.1998 | 1.1585 | 0.0412 [0.0093, 0.0729] | **SEP** | 0.641 / 0.162 / -0.3405 [-0.5623, -0.1322] BONF-SEP |
| turn: gentle 0.02-0.06 | 721 | 98 | 1.0470 | 1.0095 | 0.0374 [-0.0610, 0.1252] |  | 0.159 / 0.036 / -0.1499 [-0.3216, -0.0073] BONF-SEP |
| turn: turn 0.06-0.15 | 512 | 72 | 2.2671 | 1.9319 | 0.3352 [0.2047, 0.4766] | **SEP** | 0.113 / 0.232 / 0.1936 [0.0196, 0.4046] BONF-SEP |
| turn: sharp >=0.15 | 367 | 42 | 4.1379 | 2.9870 | 1.1509 [0.6812, 1.6913] | **SEP** | 0.081 / 0.571 / 1.0745 [0.3241, 1.8708] BONF-SEP |
| turn: undef (GT crawling) (low n) | 25 | 20 | 14.8139 | 14.8398 | -0.0259 [-0.3863, 0.2094] |  | 0.006 / -0.001 / -0.1904 [-0.6872, 0.1378] |
| speed: v0 <2 | 237 | 38 | 13.5014 | 13.1669 | 0.3346 [0.1393, 0.5755] | **SEP** | 0.052 / 0.107 / 0.1805 [-0.1326, 0.5912] |
| speed: v0 2-5 | 516 | 58 | 2.1713 | 1.8694 | 0.3018 [-0.0861, 0.6533] |  | 0.114 / 0.210 / 0.1562 [-0.4184, 0.7190] |
| speed: v0 5-10 | 1390 | 84 | 1.2623 | 1.0092 | 0.2531 [0.1492, 0.3660] | **SEP** | 0.307 / 0.476 / 0.1294 [-0.0527, 0.3202] |
| speed: v0 10-15 | 1192 | 64 | 0.6836 | 0.5744 | 0.1092 [0.0467, 0.1829] | **SEP** | 0.263 / 0.176 / -0.0737 [-0.2228, 0.0870] |
| speed: v0 15-25 | 856 | 35 | 0.3947 | 0.3734 | 0.0213 [-0.0072, 0.0524] |  | 0.189 / 0.025 / -0.1754 [-0.2949, -0.0602] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1716 | 0.1576 | 0.0140 [0.0030, 0.0260] | **SEP** | 0.074 / 0.006 / -0.1614 [-0.2556, -0.0630] BONF-SEP |
| nav: nav=follow | 2877 | 88 | 0.7017 | 0.5980 | 0.1037 [0.0560, 0.1544] | **SEP** | 0.636 / 0.403 / -0.1642 [-0.4024, 0.0271] |
| nav: nav=left (low n) | 412 | 13 | 7.4995 | 7.2301 | 0.2694 [-0.0227, 0.6493] |  | 0.091 / 0.150 / 0.1165 [-0.3220, 0.8226] |
| nav: nav=right | 1237 | 38 | 1.7595 | 1.4922 | 0.2673 [0.0938, 0.4149] | **SEP** | 0.273 / 0.447 / 0.1429 [-0.1121, 0.3899] |

episode concentration: 0.755 of episodes have mean delta > 0; top-5 episodes carry 0.304 of the summed delta; delta excluding the top 5 = 0.1181 (overall 0.1635).

### S2 -- R7-15k - R7-30k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4518 | 139 | 1.6123 | 1.5033 | 0.1089 [0.0576, 0.1608] | **SEP** |  |
| turn: straight <0.02 | 2896 | 132 | 1.2016 | 1.1927 | 0.0089 [-0.0141, 0.0307] |  | 0.641 / 0.052 / -0.2786 [-0.4482, -0.1269] BONF-SEP |
| turn: gentle 0.02-0.06 | 721 | 98 | 1.0470 | 1.0278 | 0.0192 [-0.0728, 0.0987] |  | 0.160 / 0.028 / -0.1068 [-0.2586, 0.0229] |
| turn: turn 0.06-0.15 | 512 | 72 | 2.2671 | 2.0092 | 0.2579 [0.1246, 0.3937] | **SEP** | 0.113 / 0.268 / 0.1680 [-0.0152, 0.3791] |
| turn: sharp >=0.15 | 367 | 42 | 4.1379 | 3.2666 | 0.8713 [0.4848, 1.3186] | **SEP** | 0.081 / 0.650 / 0.8298 [0.2435, 1.4945] BONF-SEP |
| turn: undef (GT crawling) (low n) | 22 | 18 | 16.8208 | 16.7855 | 0.0353 [-0.3055, 0.3083] |  | 0.005 / 0.002 / -0.0740 [-0.5407, 0.3244] |
| speed: v0 <2 | 229 | 38 | 13.9696 | 13.6458 | 0.3238 [0.1301, 0.5356] | **SEP** | 0.051 / 0.151 / 0.2264 [-0.0443, 0.5551] |
| speed: v0 2-5 | 516 | 58 | 2.1713 | 1.9548 | 0.2165 [-0.0730, 0.5123] |  | 0.114 / 0.227 / 0.1214 [-0.3298, 0.5895] |
| speed: v0 5-10 | 1390 | 84 | 1.2623 | 1.1066 | 0.1557 [0.0761, 0.2439] | **SEP** | 0.308 / 0.440 / 0.0676 [-0.0793, 0.2310] |
| speed: v0 10-15 | 1192 | 64 | 0.6836 | 0.6061 | 0.0775 [0.0266, 0.1346] | **SEP** | 0.264 / 0.188 / -0.0427 [-0.1628, 0.0776] |
| speed: v0 15-25 | 856 | 35 | 0.3947 | 0.3950 | -0.0003 [-0.0336, 0.0320] |  | 0.189 / -0.000 / -0.1347 [-0.2466, -0.0405] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1716 | 0.1784 | -0.0068 [-0.0248, 0.0120] |  | 0.074 / -0.005 / -0.1250 [-0.2113, -0.0428] BONF-SEP |
| nav: nav=follow | 2870 | 88 | 0.7031 | 0.6379 | 0.0653 [0.0317, 0.1026] | **SEP** | 0.635 / 0.381 / -0.1197 [-0.3053, 0.0539] |
| nav: nav=left (low n) | 412 | 13 | 7.4995 | 7.3746 | 0.1249 [-0.1550, 0.4814] |  | 0.091 / 0.105 / 0.0175 [-0.3934, 0.7028] |
| nav: nav=right | 1236 | 38 | 1.7609 | 1.5558 | 0.2050 [0.0759, 0.3143] | **SEP** | 0.274 / 0.515 / 0.1323 [-0.0631, 0.3177] |

episode concentration: 0.712 of episodes have mean delta > 0; top-5 episodes carry 0.367 of the summed delta; delta excluding the top 5 = 0.0716 (overall 0.1089).

### S2 -- R7-30k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4527 | 139 | 1.5031 | 1.4485 | 0.0546 [0.0265, 0.0834] | **SEP** |  |
| turn: straight <0.02 | 2898 | 132 | 1.1921 | 1.1597 | 0.0323 [0.0166, 0.0496] | **SEP** | 0.640 / 0.379 / -0.0620 [-0.1925, 0.0424] |
| turn: gentle 0.02-0.06 | 721 | 98 | 1.0278 | 1.0095 | 0.0182 [-0.0155, 0.0493] |  | 0.159 / 0.053 / -0.0433 [-0.1147, 0.0278] |
| turn: turn 0.06-0.15 | 512 | 72 | 2.0092 | 1.9319 | 0.0773 [-0.0342, 0.1987] |  | 0.113 / 0.160 / 0.0256 [-0.1460, 0.2053] |
| turn: sharp >=0.15 | 368 | 42 | 3.2871 | 3.0070 | 0.2802 [0.0169, 0.5718] | **SEP** | 0.081 / 0.417 / 0.2455 [-0.1160, 0.6978] |
| turn: undef (GT crawling) (low n) | 28 | 22 | 13.2368 | 13.3146 | -0.0778 [-0.2004, 0.0202] |  | 0.006 / -0.009 / -0.1333 [-0.3174, 0.0243] |
| speed: v0 <2 | 238 | 39 | 13.1830 | 13.1626 | 0.0203 [-0.1078, 0.1516] |  | 0.053 / 0.020 / -0.0362 [-0.2413, 0.1696] |
| speed: v0 2-5 | 516 | 58 | 1.9548 | 1.8694 | 0.0853 [-0.0437, 0.2063] |  | 0.114 / 0.178 / 0.0346 [-0.1492, 0.2258] |
| speed: v0 5-10 | 1390 | 84 | 1.1066 | 1.0092 | 0.0974 [0.0383, 0.1648] | **SEP** | 0.307 / 0.547 / 0.0617 [-0.0313, 0.1862] |
| speed: v0 10-15 | 1192 | 64 | 0.6061 | 0.5744 | 0.0317 [-0.0180, 0.0720] |  | 0.263 / 0.153 / -0.0312 [-0.1187, 0.0452] |
| speed: v0 15-25 | 856 | 35 | 0.3950 | 0.3734 | 0.0215 [-0.0067, 0.0530] |  | 0.189 / 0.075 / -0.0408 [-0.1081, 0.0281] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1784 | 0.1576 | 0.0208 [0.0018, 0.0382] | **SEP** | 0.074 / 0.028 / -0.0365 [-0.0907, 0.0161] |
| nav: nav=follow | 2876 | 88 | 0.6369 | 0.5986 | 0.0384 [0.0058, 0.0734] | **SEP** | 0.635 / 0.446 / -0.0447 [-0.1283, 0.0402] |
| nav: nav=left (low n) | 414 | 13 | 7.3407 | 7.1974 | 0.1433 [0.0770, 0.2275] | **SEP** | 0.091 / 0.240 / 0.0975 [-0.0159, 0.2373] |
| nav: nav=right | 1237 | 38 | 1.5633 | 1.5005 | 0.0629 [-0.0003, 0.1199] |  | 0.273 / 0.314 / 0.0113 [-0.0876, 0.1210] |

episode concentration: 0.647 of episodes have mean delta > 0; top-5 episodes carry 0.357 of the summed delta; delta excluding the top 5 = 0.0365 (overall 0.0546).

### S2 -- R7-15k - R7-20k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4520 | 139 | 1.5732 | 1.4849 | 0.0883 [0.0407, 0.1424] | **SEP** |  |
| turn: straight <0.02 | 2899 | 132 | 1.2005 | 1.1938 | 0.0068 [-0.0181, 0.0290] |  | 0.641 / 0.049 / -0.2274 [-0.4243, -0.0332] BONF-SEP |
| turn: gentle 0.02-0.06 | 722 | 99 | 1.0487 | 0.9861 | 0.0626 [0.0117, 0.1134] | **SEP** | 0.160 / 0.113 / -0.0306 [-0.1571, 0.1039] |
| turn: turn 0.06-0.15 | 513 | 72 | 2.2685 | 2.2101 | 0.0584 [-0.1016, 0.1840] |  | 0.113 / 0.075 / -0.0337 [-0.2886, 0.1556] |
| turn: sharp >=0.15 | 367 | 42 | 4.1379 | 3.3091 | 0.8288 [0.3610, 1.3983] | **SEP** | 0.081 / 0.762 / 0.8059 [0.0612, 1.6965] BONF-SEP |
| turn: undef (GT crawling) (low n) | 19 | 16 | 10.0594 | 10.0459 | 0.0135 [-0.1974, 0.2284] |  | 0.004 / 0.001 / -0.0751 [-0.3774, 0.2359] |
| speed: v0 <2 | 231 | 37 | 13.0985 | 12.8965 | 0.2021 [0.0196, 0.4143] | **SEP** | 0.051 / 0.117 / 0.1199 [-0.1944, 0.4847] |
| speed: v0 2-5 | 516 | 58 | 2.1713 | 2.0874 | 0.0839 [-0.2366, 0.3897] |  | 0.114 / 0.108 / -0.0050 [-0.4640, 0.4793] |
| speed: v0 5-10 | 1390 | 84 | 1.2623 | 1.1092 | 0.1532 [0.0737, 0.2456] | **SEP** | 0.308 / 0.533 / 0.0936 [-0.0663, 0.2466] |
| speed: v0 10-15 | 1192 | 64 | 0.6836 | 0.6040 | 0.0796 [0.0241, 0.1543] | **SEP** | 0.264 / 0.238 / -0.0119 [-0.1464, 0.1185] |
| speed: v0 15-25 | 856 | 35 | 0.3947 | 0.3964 | -0.0017 [-0.0256, 0.0217] |  | 0.189 / -0.004 / -0.1111 [-0.2124, -0.0222] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1716 | 0.1627 | 0.0089 [-0.0086, 0.0266] |  | 0.074 / 0.007 / -0.0858 [-0.1816, -0.0065] BONF-SEP |
| nav: nav=follow | 2871 | 88 | 0.7029 | 0.6508 | 0.0521 [0.0142, 0.0953] | **SEP** | 0.635 / 0.375 / -0.0992 [-0.2859, 0.0636] |
| nav: nav=left (low n) | 414 | 13 | 7.4759 | 7.2431 | 0.2328 [0.0009, 0.5342] | **SEP** | 0.092 / 0.241 / 0.1590 [-0.2104, 0.7595] |
| nav: nav=right | 1235 | 38 | 1.6176 | 1.4936 | 0.1240 [0.0070, 0.2328] | **SEP** | 0.273 / 0.384 / 0.0491 [-0.1400, 0.2414] |

episode concentration: 0.712 of episodes have mean delta > 0; top-5 episodes carry 0.452 of the summed delta; delta excluding the top 5 = 0.0503 (overall 0.0883).

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

### S2 -- R7-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4483 | 139 | 0.6582 | 0.6675 | -0.0093 [-0.1754, 0.0817] |  | 0.3106 [0.0532, 0.8035] SEP |
| seg1 | straight | 2881 | 132 | 0.5435 | 0.5693 | -0.0258 [-0.1671, 0.0460] |  | 0.3875 [0.0608, 1.0848] SEP |
| seg1 | curved | 1585 | 101 | 0.7572 | 0.6233 | 0.1339 [0.0994, 0.1699] | **SEP** | 0.0611 [0.0014, 0.1218] SEP |
| seg2 | all | 4471 | 139 | 1.0888 | 0.9387 | 0.1501 [0.1001, 0.1998] | **SEP** | 0.2064 [0.0806, 0.3816] SEP |
| seg2 | straight | 2882 | 131 | 0.7121 | 0.6572 | 0.0549 [0.0274, 0.0850] | **SEP** | 0.2655 [0.1103, 0.5489] SEP |
| seg2 | curved | 1589 | 101 | 1.7720 | 1.4492 | 0.3228 [0.2144, 0.4298] | **SEP** | 0.0993 [-0.0618, 0.2542] |
| seg3 | all | 4460 | 139 | 1.4370 | 1.2055 | 0.2316 [0.1177, 0.3674] | **SEP** | 0.2145 [0.0254, 0.4889] SEP |
| seg3 | straight | 2875 | 131 | 0.8564 | 0.7742 | 0.0822 [-0.0059, 0.2317] |  | 0.2766 [0.0514, 0.7008] SEP |
| seg3 | curved | 1585 | 101 | 2.4902 | 1.9877 | 0.5026 [0.3083, 0.7004] | **SEP** | 0.1019 [-0.1714, 0.3495] |
| seg4 | all | 4453 | 139 | 1.8374 | 1.4941 | 0.3433 [0.1339, 0.6482] | **SEP** | 0.4392 [0.1330, 0.8943] SEP |
| seg4 | straight | 2865 | 131 | 1.0765 | 0.8387 | 0.2377 [0.0139, 0.6664] | **SEP** | 0.4898 [0.1276, 1.1999] SEP |
| seg4 | curved | 1580 | 101 | 3.2230 | 2.6866 | 0.5364 [0.2775, 0.7963] | **SEP** | 0.3461 [-0.0859, 0.7443] |

### S2 -- R7-15k - R7-30k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4480 | 139 | 0.6586 | 0.5746 | 0.0840 [0.0324, 0.1723] | **SEP** | 0.1169 [-0.0190, 0.3673] |
| seg1 | straight | 2879 | 132 | 0.5438 | 0.4624 | 0.0814 [0.0122, 0.2252] | **SEP** | 0.2045 [0.0068, 0.6239] SEP |
| seg1 | curved | 1585 | 101 | 0.7572 | 0.6681 | 0.0891 [0.0571, 0.1239] | **SEP** | -0.0404 [-0.0927, 0.0131] |
| seg2 | all | 4469 | 139 | 1.0892 | 0.9116 | 0.1776 [0.0653, 0.3579] | **SEP** | 0.1152 [-0.0878, 0.4560] |
| seg2 | straight | 2880 | 131 | 0.7125 | 0.5597 | 0.1529 [0.0131, 0.4280] | **SEP** | 0.2626 [-0.0116, 0.8133] |
| seg2 | curved | 1589 | 101 | 1.7720 | 1.5494 | 0.2225 [0.1325, 0.3158] | **SEP** | -0.1519 [-0.3002, -0.0179] SEP |
| seg3 | all | 4458 | 139 | 1.4376 | 1.2648 | 0.1728 [0.0750, 0.2926] | **SEP** | 0.0916 [-0.1837, 0.5235] |
| seg3 | straight | 2873 | 131 | 0.8569 | 0.8073 | 0.0497 [-0.0320, 0.1955] |  | 0.2740 [-0.0726, 0.9630] |
| seg3 | curved | 1585 | 101 | 2.4902 | 2.0941 | 0.3961 [0.2378, 0.5648] | **SEP** | -0.2389 [-0.4820, -0.0186] SEP |
| seg4 | all | 4450 | 139 | 1.8388 | 1.6552 | 0.1836 [0.0695, 0.3102] | **SEP** | -0.1458 [-0.3208, 0.0197] |
| seg4 | straight | 2864 | 131 | 1.0772 | 1.0142 | 0.0629 [-0.0255, 0.2124] |  | -0.1015 [-0.2640, 0.0112] |
| seg4 | curved | 1580 | 101 | 3.2230 | 2.8200 | 0.4030 [0.2117, 0.6129] | **SEP** | -0.2298 [-0.5830, 0.1099] |

### S2 -- R7-30k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4487 | 139 | 0.5743 | 0.6674 | -0.0931 [-0.3383, 0.0332] |  | 0.1938 [0.0645, 0.4428] SEP |
| seg1 | straight | 2879 | 132 | 0.4624 | 0.5697 | -0.1073 [-0.3865, 0.0230] |  | 0.1833 [0.0513, 0.4658] SEP |
| seg1 | curved | 1588 | 101 | 0.6680 | 0.6229 | 0.0451 [0.0238, 0.0649] | **SEP** | 0.1017 [0.0678, 0.1344] SEP |
| seg2 | all | 4479 | 139 | 0.9112 | 0.9386 | -0.0274 [-0.1946, 0.0683] |  | 0.0910 [-0.0901, 0.2029] |
| seg2 | straight | 2886 | 132 | 0.5592 | 0.6568 | -0.0976 [-0.3756, 0.0382] |  | 0.0032 [-0.2804, 0.1458] |
| seg2 | curved | 1593 | 101 | 1.5489 | 1.4490 | 0.0999 [0.0418, 0.1542] | **SEP** | 0.2501 [0.1593, 0.3354] SEP |
| seg3 | all | 4468 | 139 | 1.2635 | 1.2049 | 0.0586 [0.0245, 0.0944] | **SEP** | 0.1226 [-0.0604, 0.2474] |
| seg3 | straight | 2878 | 132 | 0.8063 | 0.7737 | 0.0326 [0.0125, 0.0536] | **SEP** | 0.0029 [-0.2798, 0.1460] |
| seg3 | curved | 1590 | 101 | 2.0911 | 1.9852 | 0.1058 [0.0155, 0.1967] | **SEP** | 0.3392 [0.2022, 0.4652] SEP |
| seg4 | all | 4454 | 139 | 1.6564 | 1.4968 | 0.1596 [0.0323, 0.3455] | **SEP** | 0.5845 [0.2990, 1.0855] SEP |
| seg4 | straight | 2864 | 131 | 1.0140 | 0.8394 | 0.1746 [0.0279, 0.4642] | **SEP** | 0.5910 [0.1924, 1.4137] SEP |
| seg4 | curved | 1582 | 101 | 2.8243 | 2.6906 | 0.1337 [-0.0235, 0.2923] |  | 0.5754 [0.3835, 0.7649] SEP |

### S2 -- R7-15k - R7-20k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4477 | 139 | 0.6191 | 0.6690 | -0.0499 [-0.2166, 0.0404] |  | 0.0567 [0.0342, 0.0789] SEP |
| seg1 | straight | 2880 | 132 | 0.5437 | 0.5838 | -0.0401 [-0.1803, 0.0283] |  | -0.0022 [-0.1499, 0.0706] |
| seg1 | curved | 1585 | 101 | 0.7572 | 0.7122 | 0.0451 [0.0092, 0.0820] | **SEP** | 0.0514 [-0.0053, 0.1055] |
| seg2 | all | 4468 | 139 | 1.0889 | 1.1052 | -0.0163 [-0.1901, 0.0918] |  | -0.0068 [-0.1994, 0.1268] |
| seg2 | straight | 2880 | 131 | 0.7127 | 0.8177 | -0.1050 [-0.3811, 0.0354] |  | -0.0105 [-0.2970, 0.1450] |
| seg2 | curved | 1588 | 101 | 1.7712 | 1.6266 | 0.1446 [0.0391, 0.2540] | **SEP** | 0.0000 [-0.1611, 0.1535] |
| seg3 | all | 4459 | 139 | 1.4376 | 1.2917 | 0.1459 [0.0536, 0.2621] | **SEP** | 0.0698 [-0.0456, 0.1985] |
| seg3 | straight | 2873 | 131 | 0.8569 | 0.8079 | 0.0491 [-0.0347, 0.1951] |  | 0.1675 [0.0661, 0.3225] SEP |
| seg3 | curved | 1586 | 101 | 2.4894 | 2.1683 | 0.3211 [0.1576, 0.4903] | **SEP** | -0.1072 [-0.3505, 0.1352] |
| seg4 | all | 4453 | 139 | 1.8388 | 1.7701 | 0.0687 [-0.1239, 0.2207] |  | 0.1553 [-0.0801, 0.3626] |
| seg4 | straight | 2864 | 131 | 1.0767 | 1.1926 | -0.1159 [-0.4095, 0.0413] |  | 0.1397 [-0.1730, 0.3282] |
| seg4 | curved | 1582 | 101 | 3.2232 | 2.8197 | 0.4035 [0.1718, 0.6547] | **SEP** | 0.1816 [-0.1857, 0.5358] |

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

### S2 -- R7-15k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 442 | 47 | 6.0018 | 5.9271 | 0.0748 [-0.1072, 0.2062] |  |  |
| straight <0.02 | v0 5-15 | 1519 | 93 | 0.4184 | 0.3674 | 0.0510 [0.0198, 0.0817] | **SEP** |  |
| straight <0.02 | v0 >=15 | 940 | 43 | 0.2044 | 0.1947 | 0.0097 [0.0013, 0.0188] | **SEP** |  |
| gentle 0.02-0.06 | v0 <5 | 112 | 29 | 1.5356 | 1.7630 | -0.2274 [-0.7917, 0.1726] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 461 | 74 | 1.0828 | 0.9711 | 0.1116 [0.0368, 0.1866] | **SEP** |  |
| gentle 0.02-0.06 | v0 >=15 | 148 | 20 | 0.5656 | 0.5590 | 0.0067 [-0.0707, 0.0917] |  |  |
| turn 0.06-0.15 | v0 <5 (low n) | 82 | 25 | 5.3285 | 5.0600 | 0.2685 [-0.2822, 0.8477] |  |  |
| turn 0.06-0.15 | v0 5-15 | 343 | 54 | 1.8385 | 1.4275 | 0.4110 [0.2756, 0.5760] | **SEP** |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 87 | 11 | 1.0712 | 0.9719 | 0.0993 [-0.0440, 0.2121] |  |  |
| sharp >=0.15 | v0 <5 (low n) | 92 | 16 | 7.4797 | 5.2396 | 2.2401 [0.8503, 3.8726] | **SEP** |  |
| sharp >=0.15 | v0 5-15 | 259 | 33 | 3.1047 | 2.2857 | 0.8190 [0.3748, 1.2680] | **SEP** |  |
| sharp >=0.15 | v0 >=15 (low n) | 16 | 3 | 1.6464 | 1.3866 | 0.2598 [-0.0725, 1.4302] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 25 | 20 | 14.8139 | 14.8398 | -0.0259 [-0.3863, 0.2094] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S2 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | R7-15k s0-s1 [CI] | R7-30k s0-s1 [CI] | R7-20k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|---|
| overall | 0.0102 [-0.0033, 0.0246] | -0.0060 [-0.0196, 0.0079] | -0.0015 [-0.0157, 0.0150] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | 0.0073 [0.0005, 0.0153] SEP | -0.0050 [-0.0158, 0.0047] | 0.0079 [-0.0056, 0.0266] | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | 0.0177 [-0.0188, 0.0628] | 0.0210 [-0.0180, 0.0668] | -0.0271 [-0.0663, 0.0095] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | 0.0313 [-0.0224, 0.0874] | 0.0090 [-0.0310, 0.0579] | -0.0187 [-0.0706, 0.0340] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | -0.0152 [-0.1389, 0.1082] | -0.0870 [-0.1984, 0.0173] | -0.0092 [-0.1307, 0.1113] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | 0.0587 [-0.0663, 0.2319] | -0.0060 [-0.0909, 0.0632] | 0.1339 [-0.0506, 0.3279] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | 0.0943 [0.0353, 0.1870] SEP | 0.0226 [-0.0617, 0.1081] | 0.0129 [-0.1193, 0.1887] | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | 0.0373 [-0.0539, 0.1359] | -0.0262 [-0.1192, 0.0801] | 0.0189 [-0.0664, 0.1076] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | -0.0004 [-0.0255, 0.0239] | -0.0190 [-0.0394, 0.0006] | -0.0115 [-0.0432, 0.0284] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | 0.0077 [-0.0040, 0.0196] | 0.0070 [-0.0025, 0.0174] | 0.0005 [-0.0131, 0.0140] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | -0.0047 [-0.0124, 0.0024] | -0.0017 [-0.0088, 0.0049] | -0.0017 [-0.0070, 0.0041] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | -0.0024 [-0.0100, 0.0042] | 0.0017 [-0.0086, 0.0125] | -0.0075 [-0.0163, -0.0008] SEP | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | 0.0077 [-0.0004, 0.0163] | 0.0003 [-0.0067, 0.0069] | -0.0062 [-0.0176, 0.0054] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | 0.0155 [-0.0735, 0.1327] | -0.0076 [-0.0615, 0.0391] | -0.0472 [-0.1021, -0.0024] SEP | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | 0.0142 [-0.0181, 0.0514] | -0.0201 [-0.0627, 0.0255] | 0.0250 [-0.0162, 0.0755] | 0.0283 [0.0035, 0.0597] SEP |

### S2 -- R7-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1635 | 0.1624 / 0.1652 | True / True / True | 0.0102 / 0.0110 | 14.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0412 | 0.0439 / 0.0385 | True / True / True | 0.0073 / 0.0019 | 5.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0374 | 0.0542 / 0.0206 | False / False / False | 0.0177 / 0.0161 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.3352 | 0.3288 / 0.3410 | True / True / True | 0.0313 / 0.0331 | 10.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 1.1509 | 1.0893 / 1.2161 | True / True / True | 0.0152 / 0.1080 | 10.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | -0.0259 | 0.0077 / -0.0316 | False / False / False | 0.0587 / 0.0237 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.3346 | 0.3402 / 0.3369 | True / True / True | 0.0943 / 0.0573 | 3.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 2-5 | 0.3018 | 0.2845 / 0.3191 | False / False / False | 0.0373 / 0.0719 | 4.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.2531 | 0.2526 / 0.2536 | True / True / True | 0.0004 / 0.0006 | 390.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.1092 | 0.1131 / 0.1053 | True / True / True | 0.0077 / 0.0001 | 14.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.0213 | 0.0201 / 0.0225 | False / False / False | 0.0047 / 0.0023 | 4.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0140 | 0.0130 / 0.0150 | True / True / True | 0.0024 / 0.0005 | 5.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.1037 | 0.1093 / 0.0980 | True / True / True | 0.0077 / 0.0036 | 13.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.2694 | 0.2392 / 0.2978 | False / True / False | 0.0155 / 0.0608 | 4.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.2673 | 0.2602 / 0.2767 | True / True / True | 0.0142 / 0.0283 | 9.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S2 -- R7-15k - R7-30k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1089 | 0.1162 / 0.1022 | True / True / True | 0.0102 / 0.0060 | 10.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0089 | 0.0150 / 0.0027 | False / False / False | 0.0073 / 0.0050 | 1.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.0192 | 0.0174 / 0.0209 | False / False / False | 0.0177 / 0.0210 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.2579 | 0.2636 / 0.2522 | True / True / True | 0.0313 / 0.0090 | 8.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 0.8713 | 0.9071 / 0.8385 | True / True / True | 0.0152 / 0.0870 | 10.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.0353 | 0.0624 / 0.0623 | False / False / False | 0.0587 / 0.0060 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.3238 | 0.3450 / 0.3089 | True / True / True | 0.0943 / 0.0226 | 3.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 2-5 | 0.2165 | 0.2483 / 0.1848 | False / False / False | 0.0373 / 0.0262 | 5.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1557 | 0.1650 / 0.1464 | True / True / True | 0.0004 / 0.0190 | 8.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0775 | 0.0778 / 0.0772 | True / True / True | 0.0077 / 0.0070 | 10.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | -0.0003 | -0.0018 / 0.0012 | False / False / False | 0.0047 / 0.0017 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0068 | -0.0089 / -0.0048 | False / False / False | 0.0024 / 0.0017 | 2.8 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0653 | 0.0688 / 0.0616 | True / True / True | 0.0077 / 0.0003 | 8.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.1249 | 0.1290 / 0.1204 | False / False / False | 0.0155 / 0.0076 | 8.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.2050 | 0.2221 / 0.1903 | True / True / True | 0.0142 / 0.0201 | 10.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S2 -- R7-30k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0546 | 0.0462 / 0.0631 | True / True / True | 0.0060 / 0.0110 | 5.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0323 | 0.0289 / 0.0359 | True / True / True | 0.0050 / 0.0019 | 6.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0182 | 0.0368 / -0.0003 | True / False / False | 0.0210 / 0.0161 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.0773 | 0.0653 / 0.0894 | False / False / False | 0.0090 / 0.0331 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.2802 | 0.1827 / 0.3776 | False / True / True | 0.0870 / 0.1080 | 2.6 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|undef (GT crawling) | -0.0778 | -0.0663 / -0.1043 | False / False / False | 0.0060 / 0.0237 | 3.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.0203 | 0.0025 / 0.0372 | False / False / False | 0.0226 / 0.0573 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.0853 | 0.0363 / 0.1344 | False / False / False | 0.0262 / 0.0719 | 1.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.0974 | 0.0876 / 0.1072 | True / True / True | 0.0190 / 0.0006 | 5.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0317 | 0.0352 / 0.0281 | False / False / False | 0.0070 / 0.0001 | 4.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0215 | 0.0219 / 0.0212 | False / False / False | 0.0017 / 0.0023 | 9.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0208 | 0.0219 / 0.0197 | True / True / True | 0.0017 / 0.0005 | 12.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.0384 | 0.0404 / 0.0365 | True / True / True | 0.0003 / 0.0036 | 10.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.1433 | 0.1091 / 0.1774 | True / True / True | 0.0076 / 0.0608 | 2.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=right | 0.0629 | 0.0386 / 0.0866 | False / True / False | 0.0201 / 0.0283 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- R7-15k - R7-20k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0883 | 0.0941 / 0.0832 | True / True / True | 0.0102 / 0.0015 | 8.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0068 | 0.0064 / 0.0070 | False / False / False | 0.0073 / 0.0079 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.0626 | 0.0851 / 0.0402 | True / False / True | 0.0177 / 0.0271 | 2.3 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|turn 0.06-0.15 | 0.0584 | 0.0834 / 0.0334 | False / False / False | 0.0313 / 0.0187 | 1.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.8288 | 0.8255 / 0.8390 | True / True / True | 0.0152 / 0.0092 | 54.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.0135 | -0.0170 / 0.0416 | False / False / False | 0.0587 / 0.1339 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.2021 | 0.2419 / 0.1709 | True / True / True | 0.0943 / 0.0129 | 2.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 2-5 | 0.0839 | 0.0931 / 0.0747 | False / False / False | 0.0373 / 0.0189 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1532 | 0.1587 / 0.1476 | True / True / True | 0.0004 / 0.0115 | 13.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0796 | 0.0831 / 0.0760 | True / True / True | 0.0077 / 0.0005 | 10.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | -0.0017 | -0.0032 / -0.0002 | False / False / False | 0.0047 / 0.0017 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0089 | 0.0115 / 0.0064 | False / False / False | 0.0024 / 0.0075 | 1.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0521 | 0.0591 / 0.0451 | True / True / True | 0.0077 / 0.0062 | 6.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.2328 | 0.2637 / 0.2010 | True / False / True | 0.0155 / 0.0472 | 4.9 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| nav|nav=right | 0.1240 | 0.1186 / 0.1319 | True / True / True | 0.0142 / 0.0250 | 5.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

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

* seg1 (n 1585): R7-15k bias +0.036, refcv7-50400 bias +0.082, R7-30k bias +0.046, R7-20k bias +0.137; R7-15k MAE 0.757, refcv7-50400 MAE 0.623, R7-30k MAE 0.668, R7-20k MAE 0.712
* seg2 (n 1588): R7-15k bias -0.336, refcv7-50400 bias -0.096, R7-30k bias -0.187, R7-20k bias -0.003; R7-15k MAE 1.771, refcv7-50400 MAE 1.449, R7-30k MAE 1.548, R7-20k MAE 1.627
* seg3 (n 1585): R7-15k bias -1.262, refcv7-50400 bias -0.622, R7-30k bias -0.810, R7-20k bias -0.665; R7-15k MAE 2.490, refcv7-50400 MAE 1.988, R7-30k MAE 2.094, R7-20k MAE 2.168
* seg4 (n 1580): R7-15k bias -1.872, refcv7-50400 bias -0.724, R7-30k bias -1.138, R7-20k bias -0.894; R7-15k MAE 3.223, refcv7-50400 MAE 2.687, R7-30k MAE 2.820, R7-20k MAE 2.819
**sharp (>= 0.15)**

* seg1 (n 364): R7-15k bias -0.125, refcv7-50400 bias +0.010, R7-30k bias -0.123, R7-20k bias +0.286; R7-15k MAE 1.147, refcv7-50400 MAE 0.917, R7-30k MAE 0.996, R7-20k MAE 1.108
* seg2 (n 366): R7-15k bias -1.213, refcv7-50400 bias -0.528, R7-30k bias -0.822, R7-20k bias +0.086; R7-15k MAE 3.135, refcv7-50400 MAE 2.413, R7-30k MAE 2.645, R7-20k MAE 2.812
* seg3 (n 366): R7-15k bias -3.752, refcv7-50400 bias -1.862, R7-30k bias -2.418, R7-20k bias -1.596; R7-15k MAE 5.219, refcv7-50400 MAE 3.643, R7-30k MAE 3.959, R7-20k MAE 4.029
* seg4 (n 364): R7-15k bias -5.288, refcv7-50400 bias -1.837, R7-30k bias -3.170, R7-20k bias -2.010; R7-15k MAE 6.898, refcv7-50400 MAE 4.878, R7-30k MAE 5.319, R7-20k MAE 5.145
**curved & v0 5-10**

* seg1 (n 627): R7-15k bias -0.095, refcv7-50400 bias -0.038, R7-30k bias -0.061, R7-20k bias +0.030; R7-15k MAE 0.735, refcv7-50400 MAE 0.535, R7-30k MAE 0.598, R7-20k MAE 0.661
* seg2 (n 627): R7-15k bias -0.534, refcv7-50400 bias -0.282, R7-30k bias -0.325, R7-20k bias -0.169; R7-15k MAE 1.843, refcv7-50400 MAE 1.363, R7-30k MAE 1.525, R7-20k MAE 1.598
* seg3 (n 627): R7-15k bias -1.477, refcv7-50400 bias -0.871, R7-30k bias -0.963, R7-20k bias -0.896; R7-15k MAE 2.653, refcv7-50400 MAE 2.019, R7-30k MAE 2.199, R7-20k MAE 2.199
* seg4 (n 627): R7-15k bias -2.210, refcv7-50400 bias -1.194, R7-30k bias -1.457, R7-20k bias -1.283; R7-15k MAE 3.352, refcv7-50400 MAE 2.730, R7-30k MAE 2.978, R7-20k MAE 2.839
**sharp & v0 5-10**

* seg1 (n 164): R7-15k bias +0.093, refcv7-50400 bias +0.103, R7-30k bias +0.060, R7-20k bias +0.482; R7-15k MAE 1.156, refcv7-50400 MAE 0.857, R7-30k MAE 0.971, R7-20k MAE 1.100
* seg2 (n 164): R7-15k bias -0.744, refcv7-50400 bias -0.422, R7-30k bias -0.444, R7-20k bias +0.462; R7-15k MAE 3.001, refcv7-50400 MAE 2.243, R7-30k MAE 2.557, R7-20k MAE 2.699
* seg3 (n 164): R7-15k bias -3.460, refcv7-50400 bias -2.032, R7-30k bias -2.198, R7-20k bias -1.436; R7-15k MAE 4.827, refcv7-50400 MAE 3.465, R7-30k MAE 3.740, R7-20k MAE 3.740
* seg4 (n 164): R7-15k bias -5.287, refcv7-50400 bias -2.403, R7-30k bias -3.228, R7-20k bias -2.138; R7-15k MAE 6.586, refcv7-50400 MAE 4.635, R7-30k MAE 5.151, R7-20k MAE 4.837

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1580 / 1580 | 10.2 | R7-15k -0.2187 [-0.3021, -0.1510]; refcv7-50400 -0.0849 [-0.1562, -0.0278]; R7-30k -0.1294 [-0.2020, -0.0670]; R7-20k -0.0997 [-0.1820, -0.0386] | R7-15k -0.1339 [-0.1762, -0.0924] SEP; R7-30k -0.0445 [-0.0652, -0.0226] SEP; R7-20k -0.0149 [-0.0488, +0.0182] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1598 / 4753 | 7.2 | R7-15k -0.2610 [-0.3762, -0.1633]; refcv7-50400 -0.1468 [-0.2660, -0.0526]; R7-30k -0.1806 [-0.3000, -0.0842]; R7-20k -0.1528 [-0.2792, -0.0498] | R7-15k -0.1141 [-0.1545, -0.0764] SEP; R7-30k -0.0337 [-0.0539, -0.0136] SEP; R7-20k -0.0060 [-0.0374, +0.0261] |
| sharp (>= 0.15) | last segment | 364 / 364 | 25.4 | R7-15k -0.2358 [-0.3357, -0.1562]; refcv7-50400 -0.0922 [-0.1765, -0.0257]; R7-30k -0.1434 [-0.2287, -0.0705]; R7-20k -0.1058 [-0.2040, -0.0350] | R7-15k -0.1436 [-0.1937, -0.0985] SEP; R7-30k -0.0512 [-0.0759, -0.0268] SEP; R7-20k -0.0137 [-0.0530, +0.0250] |
| sharp (>= 0.15) | segments 2..K | 367 / 1096 | 17.9 | R7-15k -0.2310 [-0.3316, -0.1494]; refcv7-50400 -0.1041 [-0.1873, -0.0361]; R7-30k -0.1454 [-0.2337, -0.0702]; R7-20k -0.1060 [-0.2062, -0.0315] | R7-15k -0.1269 [-0.1740, -0.0844] SEP; R7-30k -0.0413 [-0.0649, -0.0171] SEP; R7-20k -0.0019 [-0.0421, +0.0386] |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | R7-15k -0.2026 [-0.2773, -0.1468]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; R7-30k -0.1170 [-0.1981, -0.0539]; R7-20k -0.0925 [-0.1693, -0.0394] | R7-15k -0.1100 [-0.1454, -0.0722] SEP; R7-30k -0.0245 [-0.0621, +0.0105]; R7-20k -0.0000 [-0.0233, +0.0247] |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | R7-15k -0.2000 [-0.2767, -0.1423]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; R7-30k -0.1147 [-0.1962, -0.0510]; R7-20k -0.0932 [-0.1722, -0.0382] | R7-15k -0.1008 [-0.1343, -0.0660] SEP; R7-30k -0.0154 [-0.0515, +0.0188]; R7-20k +0.0061 [-0.0160, +0.0286] |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | R7-15k -0.2004 [-0.2813, -0.1383]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; R7-30k -0.1195 [-0.2080, -0.0502]; R7-20k -0.0853 [-0.1691, -0.0256] | R7-15k -0.1091 [-0.1504, -0.0674] SEP; R7-30k -0.0282 [-0.0696, +0.0083]; R7-20k +0.0059 [-0.0173, +0.0316] |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | R7-15k -0.1889 [-0.2718, -0.1251]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; R7-30k -0.1131 [-0.2023, -0.0424]; R7-20k -0.0795 [-0.1657, -0.0181] | R7-15k -0.0934 [-0.1317, -0.0545] SEP; R7-30k -0.0176 [-0.0578, +0.0180]; R7-20k +0.0160 [-0.0068, +0.0412] |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**R7-15k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4526 | 0.1635 | 0.03533 | 0.07819 | 0.615 / 0.385 / 0.000 | 338.4 / 739.9 |
| turn|straight <0.02 | 2901 | 0.0412 | 0.02299 | 0.03728 | 0.609 / 0.391 / 0.000 | 56.8 / 119.6 |
| turn|gentle 0.02-0.06 | 721 | 0.0374 | 0.05040 | 0.06975 | 0.583 / 0.417 / 0.000 | 24.3 / 27.0 |
| turn|turn 0.06-0.15 | 512 | 0.3352 | 0.18964 | 0.26054 | 0.641 / 0.359 / 0.000 | 29.8 / 171.6 |
| turn|sharp >=0.15 | 367 | 1.1509 | 0.59272 | 0.88909 | 0.695 / 0.305 / 0.000 | 70.1 / 422.4 |
| turn|undef (GT crawling) | 25 | -0.0259 | 0.03725 | 0.02737 | 0.520 / 0.480 / 0.000 | 1.9 / -0.6 |
| speed|v0 <2 | 237 | 0.3346 | 0.02631 | 0.11500 | 0.574 / 0.426 / 0.000 | 15.6 / 79.3 |
| speed|v0 2-5 | 516 | 0.3018 | 0.18655 | 0.24091 | 0.655 / 0.345 / 0.000 | 97.8 / 155.7 |
| speed|v0 5-10 | 1390 | 0.2531 | 0.06923 | 0.12250 | 0.618 / 0.382 / 0.000 | 88.2 / 351.8 |
| speed|v0 10-15 | 1192 | 0.1092 | 0.04512 | 0.06968 | 0.655 / 0.345 / 0.000 | 32.2 / 130.1 |
| speed|v0 15-25 | 856 | 0.0213 | 0.01273 | 0.01539 | 0.564 / 0.436 / 0.000 | 11.6 / 18.2 |
| speed|v0 >=25 | 335 | 0.0140 | 0.00753 | 0.00874 | 0.552 / 0.448 / 0.000 | 3.0 / 4.7 |
| nav|nav=follow | 2877 | 0.1037 | 0.02413 | 0.05110 | 0.597 / 0.403 / 0.000 | 128.0 / 298.3 |
| nav|nav=left | 412 | 0.2694 | 0.02697 | 0.07506 | 0.597 / 0.403 / 0.000 | 50.8 / 111.0 |
| nav|nav=right | 1237 | 0.2673 | 0.08259 | 0.16826 | 0.662 / 0.338 / 0.000 | 126.6 / 330.7 |
**R7-15k - R7-30k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4518 | 0.1089 | 0.01800 | 0.04637 | 0.556 / 0.444 / 0.000 | 304.6 / 492.2 |
| turn|straight <0.02 | 2896 | 0.0089 | 0.00395 | 0.01103 | 0.521 / 0.479 / 0.000 | 46.7 / 25.8 |
| turn|gentle 0.02-0.06 | 721 | 0.0192 | 0.04613 | 0.04418 | 0.566 / 0.434 / 0.000 | 25.0 / 13.8 |
| turn|turn 0.06-0.15 | 512 | 0.2579 | 0.13443 | 0.21853 | 0.646 / 0.354 / 0.000 | 26.8 / 132.0 |
| turn|sharp >=0.15 | 367 | 0.8713 | 0.40988 | 0.62059 | 0.689 / 0.311 / 0.000 | 68.2 / 319.8 |
| turn|undef (GT crawling) | 22 | 0.0353 | -0.00500 | 0.04541 | 0.455 / 0.545 / 0.000 | 2.3 / 0.8 |
| speed|v0 <2 | 229 | 0.3238 | 0.05926 | 0.15364 | 0.629 / 0.371 / 0.000 | 16.4 / 74.2 |
| speed|v0 2-5 | 516 | 0.2165 | 0.12707 | 0.15997 | 0.626 / 0.374 / 0.000 | 94.9 / 111.7 |
| speed|v0 5-10 | 1390 | 0.1557 | 0.03970 | 0.07171 | 0.582 / 0.418 / 0.000 | 76.1 / 216.4 |
| speed|v0 10-15 | 1192 | 0.0775 | 0.01647 | 0.03765 | 0.559 / 0.441 / 0.000 | 30.5 / 92.4 |
| speed|v0 15-25 | 856 | -0.0003 | -0.00064 | -0.00325 | 0.495 / 0.505 / 0.000 | 12.0 / -0.2 |
| speed|v0 >=25 | 335 | -0.0068 | -0.00863 | -0.00889 | 0.433 / 0.567 / 0.000 | 2.0 / -2.3 |
| nav|nav=follow | 2870 | 0.0653 | 0.01345 | 0.03191 | 0.549 / 0.451 / 0.000 | 107.2 / 187.3 |
| nav|nav=left | 412 | 0.1249 | -0.00405 | -0.00898 | 0.493 / 0.507 / 0.000 | 53.4 / 51.5 |
| nav|nav=right | 1236 | 0.2050 | 0.05689 | 0.11951 | 0.591 / 0.409 / 0.000 | 109.3 / 253.4 |
**R7-30k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4527 | 0.0546 | 0.02944 | 0.03463 | 0.593 / 0.407 / 0.000 | 188.0 / 247.4 |
| turn|straight <0.02 | 2898 | 0.0323 | 0.02691 | 0.02930 | 0.607 / 0.393 / 0.000 | 45.8 / 93.7 |
| turn|gentle 0.02-0.06 | 721 | 0.0182 | 0.02245 | 0.02962 | 0.556 / 0.444 / 0.000 | 12.0 / 13.1 |
| turn|turn 0.06-0.15 | 512 | 0.0773 | 0.04893 | 0.04963 | 0.568 / 0.432 / 0.000 | 34.8 / 39.6 |
| turn|sharp >=0.15 | 368 | 0.2802 | 0.19304 | 0.17998 | 0.595 / 0.405 / 0.000 | 34.4 / 103.1 |
| turn|undef (GT crawling) | 28 | -0.0778 | -0.01158 | -0.03977 | 0.464 / 0.536 / 0.000 | 0.5 / -2.2 |
| speed|v0 <2 | 238 | 0.0203 | -0.00323 | -0.00507 | 0.479 / 0.521 / 0.000 | 7.5 / 4.8 |
| speed|v0 2-5 | 516 | 0.0853 | 0.04064 | 0.06172 | 0.568 / 0.432 / 0.000 | 30.4 / 44.0 |
| speed|v0 5-10 | 1390 | 0.0974 | 0.04238 | 0.05221 | 0.601 / 0.399 / 0.000 | 82.2 / 135.4 |
| speed|v0 10-15 | 1192 | 0.0317 | 0.03376 | 0.03713 | 0.617 / 0.383 / 0.000 | 18.6 / 37.7 |
| speed|v0 15-25 | 856 | 0.0215 | 0.01964 | 0.02224 | 0.585 / 0.415 / 0.000 | 9.5 / 18.4 |
| speed|v0 >=25 | 335 | 0.0208 | 0.02487 | 0.01984 | 0.609 / 0.391 / 0.000 | 2.1 / 7.0 |
| nav|nav=follow | 2876 | 0.0384 | 0.02257 | 0.02400 | 0.585 / 0.415 / 0.000 | 91.8 / 110.3 |
| nav|nav=left | 414 | 0.1433 | 0.04963 | 0.07269 | 0.659 / 0.341 / 0.000 | 24.5 / 59.3 |
| nav|nav=right | 1237 | 0.0629 | 0.04477 | 0.05271 | 0.589 / 0.411 / 0.000 | 58.4 / 77.8 |
**R7-15k - R7-20k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4520 | 0.0883 | 0.01978 | 0.04493 | 0.565 / 0.435 / 0.000 | 288.3 / 399.2 |
| turn|straight <0.02 | 2899 | 0.0068 | 0.01113 | 0.01833 | 0.550 / 0.450 / 0.000 | 37.1 / 19.6 |
| turn|gentle 0.02-0.06 | 722 | 0.0626 | 0.03824 | 0.04506 | 0.560 / 0.440 / 0.000 | 24.2 / 45.2 |
| turn|turn 0.06-0.15 | 513 | 0.0584 | 0.10204 | 0.12056 | 0.598 / 0.402 / 0.000 | 19.3 / 30.0 |
| turn|sharp >=0.15 | 367 | 0.8288 | 0.41628 | 0.62523 | 0.651 / 0.349 / 0.000 | 66.1 / 304.2 |
| turn|undef (GT crawling) | 19 | 0.0135 | -0.04047 | -0.04713 | 0.474 / 0.526 / 0.000 | 1.6 / 0.3 |
| speed|v0 <2 | 231 | 0.2021 | 0.01940 | 0.07834 | 0.558 / 0.442 / 0.000 | 15.7 / 46.7 |
| speed|v0 2-5 | 516 | 0.0839 | 0.09670 | 0.12099 | 0.583 / 0.417 / 0.000 | 84.7 / 43.3 |
| speed|v0 5-10 | 1390 | 0.1532 | 0.05086 | 0.08186 | 0.583 / 0.417 / 0.000 | 78.5 / 212.9 |
| speed|v0 10-15 | 1192 | 0.0796 | 0.02021 | 0.03792 | 0.580 / 0.420 / 0.000 | 32.9 / 94.8 |
| speed|v0 15-25 | 856 | -0.0017 | 0.00304 | 0.00246 | 0.513 / 0.487 / 0.000 | 12.6 / -1.5 |
| speed|v0 >=25 | 335 | 0.0089 | 0.00514 | 0.00295 | 0.546 / 0.454 / 0.000 | 4.3 / 3.0 |
| nav|nav=follow | 2871 | 0.0521 | 0.01665 | 0.03123 | 0.563 / 0.437 / 0.000 | 98.6 / 149.6 |
| nav|nav=left | 414 | 0.2328 | 0.01053 | 0.03736 | 0.519 / 0.481 / 0.000 | 55.3 / 96.4 |
| nav|nav=right | 1235 | 0.1240 | 0.04081 | 0.09305 | 0.585 / 0.415 / 0.000 | 95.3 / 153.2 |
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

**R7-15k - refcv7-50400** (sum 739.9 over 139 episodes)

* ep016: n 29, share 0.089, mean delta 2.2669, sharp-window frac 0.28, curved frac 0.34, mean v0 7.7, nav {'nav=follow': 0, 'nav=left': 29, 'nav=right': 0}
* ep067: n 34, share 0.062, mean delta 1.3404, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep028: n 34, share 0.059, mean delta 1.2862, sharp-window frac 0.21, curved frac 0.68, mean v0 5.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep061: n 34, share 0.049, mean delta 1.0613, sharp-window frac 0.62, curved frac 1.00, mean v0 9.5, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.045, mean delta 0.9865, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep124: n 34, share 0.044, mean delta 0.9502, sharp-window frac 0.32, curved frac 0.88, mean v0 9.7, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
**R7-15k - R7-30k** (sum 492.2 over 139 episodes)

* ep016: n 29, share 0.124, mean delta 2.1038, sharp-window frac 0.28, curved frac 0.34, mean v0 7.7, nav {'nav=follow': 0, 'nav=left': 29, 'nav=right': 0}
* ep124: n 34, share 0.066, mean delta 0.9596, sharp-window frac 0.32, curved frac 0.88, mean v0 9.7, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep028: n 34, share 0.061, mean delta 0.8815, sharp-window frac 0.21, curved frac 0.68, mean v0 5.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep067: n 34, share 0.060, mean delta 0.8757, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep061: n 34, share 0.055, mean delta 0.7958, sharp-window frac 0.62, curved frac 1.00, mean v0 9.5, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.043, mean delta 0.6202, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
**R7-30k - refcv7-50400** (sum 247.4 over 139 episodes)

* ep111: n 34, share 0.121, mean delta 0.8803, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep067: n 34, share 0.064, mean delta 0.4648, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep035: n 34, share 0.063, mean delta 0.4600, sharp-window frac 0.03, curved frac 0.24, mean v0 13.4, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep028: n 34, share 0.056, mean delta 0.4047, sharp-window frac 0.21, curved frac 0.68, mean v0 5.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep080: n 34, share 0.054, mean delta 0.3912, sharp-window frac 0.18, curved frac 0.32, mean v0 9.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep013: n 34, share 0.050, mean delta 0.3664, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
**R7-15k - R7-20k** (sum 399.2 over 139 episodes)

* ep016: n 30, share 0.130, mean delta 1.7233, sharp-window frac 0.27, curved frac 0.37, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 30, 'nav=right': 0}
* ep028: n 34, share 0.096, mean delta 1.1310, sharp-window frac 0.21, curved frac 0.68, mean v0 5.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep107: n 34, share 0.085, mean delta 1.0027, sharp-window frac 0.65, curved frac 0.91, mean v0 13.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep012: n 34, share 0.080, mean delta 0.9446, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep080: n 34, share 0.060, mean delta 0.7043, sharp-window frac 0.18, curved frac 0.32, mean v0 9.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep085: n 34, share 0.057, mean delta 0.6651, sharp-window frac 0.62, curved frac 1.00, mean v0 11.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
**R7-20k - refcv7-50400** (sum 338.0 over 139 episodes)

* ep067: n 34, share 0.140, mean delta 1.3928, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep041: n 34, share 0.101, mean delta 1.0037, sharp-window frac 0.18, curved frac 0.47, mean v0 9.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.088, mean delta 0.8721, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep082: n 31, share 0.076, mean delta 0.8308, sharp-window frac 0.00, curved frac 0.45, mean v0 8.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 31}
* ep124: n 34, share 0.056, mean delta 0.5595, sharp-window frac 0.32, curved frac 0.88, mean v0 9.7, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep129: n 31, share 0.053, mean delta 0.5812, sharp-window frac 0.32, curved frac 0.52, mean v0 4.7, nav {'nav=follow': 31, 'nav=left': 0, 'nav=right': 0}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**R7-15k vs R7-30k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 2211 / 0.0381 [0.0175, 0.0585] | 53 / 0.4507 [0.1467, 0.7971] |
| only new agrees | 251 / 0.0022 [-0.1417, 0.1365] | 36 / -0.0878 [-0.6998, 0.5649] |
| only earlier agrees | 323 / 0.4243 [0.2271, 0.6542] | 47 / 1.9267 [0.9766, 3.5973] |
| neither agrees | 1734 / 0.1749 [0.0651, 0.2932] | 231 / 0.9595 [0.4231, 1.6127] |
**R7-15k vs R7-20k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 2157 / 0.0448 [0.0259, 0.0638] | 64 / 0.1083 [-0.2806, 0.5673] |
| only new agrees | 309 / -0.1321 [-0.3384, 0.0262] | 25 / -0.8404 [-2.6729, 0.5002] |
| only earlier agrees | 293 / 0.2408 [0.0874, 0.3977] | 45 / 1.0298 [0.5371, 1.5106] |
| neither agrees | 1764 / 0.1696 [0.0672, 0.2822] | 233 / 1.1618 [0.5078, 1.9008] |
**R7-15k vs R7-30k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 2220 / 0.0360 [0.0158, 0.0575] | 57 / 0.4365 [0.1200, 0.8741] |
| only new agrees | 256 / 0.0790 [-0.0626, 0.2235] | 38 / 0.4569 [-0.3033, 1.3019] |
| only earlier agrees | 320 / 0.3340 [0.1467, 0.5420] | 51 / 1.4752 [0.6750, 2.7735] |
| neither agrees | 1725 / 0.1478 [0.0513, 0.2463] | 222 / 0.8607 [0.3451, 1.4651] |
**R7-15k vs R7-20k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 2167 / 0.0392 [0.0206, 0.0585] | 61 / 0.1391 [-0.2439, 0.5828] |
| only new agrees | 310 / -0.0695 [-0.2603, 0.1295] | 34 / -0.2054 [-1.9945, 1.3045] |
| only earlier agrees | 278 / 0.2246 [0.0633, 0.3838] | 41 / 1.0659 [0.3863, 1.6978] |
| neither agrees | 1768 / 0.1416 [0.0457, 0.2457] | 232 / 1.1361 [0.5293, 1.8412] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**R7-15k vs R7-30k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4510 | 0.0165 [-0.0143, 0.0507] | 0.0309 [-0.0266, 0.1012] | -0.0136 [-0.0850, 0.0428] |
| turn|straight <0.02 | 2895 | -0.0310 [-0.0482, -0.0166] | -0.0240 [-0.0506, 0.0055] | -0.0069 [-0.0383, 0.0199] |
| turn|gentle 0.02-0.06 | 721 | -0.0661 [-0.1205, -0.0179] | -0.0180 [-0.1334, 0.0773] | -0.0481 [-0.1635, 0.0733] |
| turn|turn 0.06-0.15 | 510 | -0.0382 [-0.1221, 0.0391] | 0.1191 [-0.0362, 0.3461] | -0.1579 [-0.3807, -0.0071] SEP |
| turn|sharp >=0.15 | 367 | 0.6337 [0.3031, 1.0562] | 0.4356 [0.0750, 0.8390] | 0.2063 [-0.1645, 0.5964] |
| turn|undef (GT crawling) | 17 | -0.0275 [-0.2339, 0.1080] | 0.0519 [-0.1242, 0.2388] | -0.0995 [-0.4538, 0.1304] |
| speed|v0 <2 | 221 | 0.1199 [-0.0475, 0.3007] | 0.3004 [0.1136, 0.5311] | -0.1661 [-0.3274, -0.0382] SEP |
| speed|v0 2-5 | 516 | 0.0502 [-0.1799, 0.3512] | 0.2345 [-0.1783, 0.7598] | -0.1843 [-0.7079, 0.2822] |
| speed|v0 5-10 | 1390 | 0.0120 [-0.0514, 0.0726] | -0.0151 [-0.0764, 0.0472] | 0.0270 [-0.0271, 0.0799] |
| speed|v0 10-15 | 1192 | 0.0005 [-0.0233, 0.0265] | -0.0153 [-0.0289, -0.0004] | 0.0158 [-0.0097, 0.0416] |
| speed|v0 15-25 | 856 | -0.0016 [-0.0112, 0.0088] | -0.0135 [-0.0236, -0.0044] | 0.0119 [-0.0015, 0.0259] |
| speed|v0 >=25 | 335 | 0.0117 [0.0006, 0.0224] | -0.0001 [-0.0096, 0.0089] | 0.0118 [-0.0018, 0.0271] |
**R7-15k vs R7-20k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4518 | 0.0165 [-0.0143, 0.0507] | 0.0589 [-0.0136, 0.1443] | -0.0423 [-0.1169, 0.0182] |
| turn|straight <0.02 | 2897 | -0.0310 [-0.0482, -0.0166] | -0.0207 [-0.0518, 0.0116] | -0.0104 [-0.0400, 0.0156] |
| turn|gentle 0.02-0.06 | 722 | -0.0661 [-0.1205, -0.0179] | 0.0215 [-0.0281, 0.0865] | -0.0876 [-0.1645, -0.0215] SEP |
| turn|turn 0.06-0.15 | 513 | -0.0382 [-0.1221, 0.0391] | 0.0702 [-0.1827, 0.3249] | -0.1083 [-0.3534, 0.1048] |
| turn|sharp >=0.15 | 367 | 0.6337 [0.3031, 1.0562] | 0.7306 [0.0919, 1.5744] | -0.0944 [-0.8066, 0.4300] |
| turn|undef (GT crawling) | 19 | -0.0275 [-0.2339, 0.1080] | 0.2725 [-0.0434, 0.8481] | -0.3932 [-1.1668, 0.0916] |
| speed|v0 <2 | 229 | 0.1199 [-0.0475, 0.3007] | 0.1807 [-0.0229, 0.3663] | -0.0559 [-0.2524, 0.1445] |
| speed|v0 2-5 | 516 | 0.0502 [-0.1799, 0.3512] | 0.0293 [-0.2999, 0.4019] | 0.0208 [-0.2702, 0.3207] |
| speed|v0 5-10 | 1390 | 0.0120 [-0.0514, 0.0726] | 0.1443 [-0.0058, 0.3801] | -0.1323 [-0.3491, 0.0033] |
| speed|v0 10-15 | 1192 | 0.0005 [-0.0233, 0.0265] | 0.0070 [-0.0086, 0.0222] | -0.0065 [-0.0267, 0.0189] |
| speed|v0 15-25 | 856 | -0.0016 [-0.0112, 0.0088] | -0.0014 [-0.0103, 0.0073] | -0.0002 [-0.0102, 0.0108] |
| speed|v0 >=25 | 335 | 0.0117 [0.0006, 0.0224] | 0.0033 [-0.0054, 0.0136] | 0.0084 [0.0008, 0.0149] SEP |
**R7-15k vs R7-30k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
**R7-15k vs R7-20k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* R7-15k vs R7-30k|seed0: anchor flip rate 0.244; heading delta on flipped windows 0.3213 [0.1333, 0.5080] (n 1069) vs unflipped 0.0527 [0.0301, 0.0771]; goal_dist_sel proxy new-earlier 0.2429 [-0.0601, 0.5369]; corr(window dHeading, window dGoalDist) 0.013
* R7-15k vs R7-20k|seed0: anchor flip rate 0.246; heading delta on flipped windows 0.2495 [0.0549, 0.4411] (n 1070) vs unflipped 0.0459 [0.0247, 0.0662]; goal_dist_sel proxy new-earlier 0.0433 [-0.2078, 0.2859]; corr(window dHeading, window dGoalDist) 0.038
* R7-15k vs R7-30k|seed1: anchor flip rate 0.243; heading delta on flipped windows 0.3045 [0.1469, 0.4748] (n 1068) vs unflipped 0.0396 [0.0177, 0.0625]; goal_dist_sel proxy new-earlier 0.2453 [-0.0506, 0.5338]; corr(window dHeading, window dGoalDist) 0.010
* R7-15k vs R7-20k|seed1: anchor flip rate 0.238; heading delta on flipped windows 0.2140 [0.0062, 0.4178] (n 1036) vs unflipped 0.0443 [0.0251, 0.0644]; goal_dist_sel proxy new-earlier 0.0431 [-0.2083, 0.2859]; corr(window dHeading, window dGoalDist) 0.043

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

### S6 -- R7-15k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3566 | 139 | 3.8393 | 3.2893 | 0.5500 [0.3022, 0.8206] | **SEP** |  |
| turn: straight <0.02 | 2112 | 123 | 1.6340 | 1.4638 | 0.1703 [0.1143, 0.2353] | **SEP** | 0.592 / 0.183 / -0.9314 [-1.7834, -0.0650] BONF-SEP |
| turn: gentle 0.02-0.06 | 678 | 77 | 2.7762 | 2.8833 | -0.1070 [-0.6270, 0.2487] |  | 0.190 / -0.037 / -0.8113 [-1.7698, -0.2045] BONF-SEP |
| turn: turn 0.06-0.15 | 442 | 59 | 6.9775 | 6.3377 | 0.6398 [0.0153, 1.2417] | **SEP** | 0.124 / 0.144 / 0.1025 [-0.8572, 1.0610] |
| turn: sharp >=0.15 | 312 | 33 | 16.2410 | 11.7318 | 4.5092 [2.6898, 6.5514] | **SEP** | 0.087 / 0.717 / 4.3387 [1.9625, 7.8780] BONF-SEP |
| turn: undef (GT crawling) (low n) | 22 | 13 | 9.3763 | 10.0709 | -0.6945 [-1.6720, 0.1677] |  | 0.006 / -0.008 / -1.2523 [-2.8231, -0.0823] BONF-SEP |
| speed: v0 <2 | 228 | 26 | 13.8187 | 12.5782 | 1.2405 [0.3495, 2.8822] | **SEP** | 0.064 / 0.144 / 0.7376 [-0.4697, 3.4350] |
| speed: v0 2-5 | 422 | 52 | 4.9992 | 4.1501 | 0.8490 [-0.2946, 2.0662] |  | 0.118 / 0.183 / 0.3391 [-1.5476, 2.3402] |
| speed: v0 5-10 | 1040 | 79 | 4.1750 | 3.4941 | 0.6809 [0.3172, 1.0586] | **SEP** | 0.292 / 0.361 / 0.1847 [-0.4058, 0.8490] |
| speed: v0 10-15 | 952 | 62 | 2.9686 | 2.4870 | 0.4816 [0.1433, 0.8711] | **SEP** | 0.267 / 0.234 / -0.0934 [-0.7067, 0.6017] |
| speed: v0 15-25 | 679 | 34 | 1.6750 | 1.4456 | 0.2294 [0.1145, 0.3645] | **SEP** | 0.190 / 0.079 / -0.3960 [-0.8928, 0.0924] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5110 | 0.5193 | -0.0083 [-0.0721, 0.0607] |  | 0.069 / -0.001 / -0.5995 [-1.0604, -0.2118] BONF-SEP |
| nav: nav=follow | 2248 | 88 | 2.4554 | 1.9748 | 0.4806 [0.2623, 0.7260] | **SEP** | 0.630 / 0.551 / -0.1878 [-1.1747, 0.5980] |
| nav: nav=left (low n) | 335 | 13 | 10.9818 | 9.7883 | 1.1935 [0.1320, 3.0265] | **SEP** | 0.094 / 0.204 / 0.7101 [-0.6100, 4.2706] |
| nav: nav=right | 983 | 38 | 4.5700 | 4.0805 | 0.4895 [-0.0955, 1.0138] |  | 0.276 / 0.245 / -0.0836 [-1.0731, 0.8410] |

episode concentration: 0.748 of episodes have mean delta > 0; top-5 episodes carry 0.414 of the summed delta; delta excluding the top 5 = 0.3340 (overall 0.5500).

### S6 -- R7-15k - R7-30k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3541 | 139 | 3.8477 | 3.4897 | 0.3580 [0.1800, 0.5684] | **SEP** |  |
| turn: straight <0.02 | 2094 | 122 | 1.6252 | 1.4951 | 0.1301 [0.0893, 0.1719] | **SEP** | 0.591 / 0.215 / -0.5579 [-1.3066, 0.0950] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.7712 | 2.9231 | -0.1519 [-0.5009, 0.1187] |  | 0.191 / -0.081 / -0.6301 [-1.3233, -0.1555] BONF-SEP |
| turn: turn 0.06-0.15 | 441 | 59 | 6.9815 | 6.6920 | 0.2895 [-0.3152, 0.8947] |  | 0.125 / 0.101 / -0.0783 [-1.0442, 0.9139] |
| turn: sharp >=0.15 | 312 | 33 | 16.2410 | 13.0227 | 3.2183 [1.7665, 4.9874] | **SEP** | 0.088 / 0.792 / 3.1366 [0.8449, 5.9502] BONF-SEP |
| turn: undef (GT crawling) (low n) | 19 | 10 | 10.7912 | 12.5685 | -1.7773 [-3.9136, 0.0768] |  | 0.005 / -0.027 / -2.1469 [-5.3748, 0.2175] |
| speed: v0 <2 | 203 | 26 | 15.1948 | 14.1585 | 1.0363 [0.2082, 2.6375] | **SEP** | 0.057 / 0.166 / 0.7195 [-0.3994, 3.3073] |
| speed: v0 2-5 | 422 | 52 | 4.9992 | 4.4408 | 0.5584 [-0.2548, 1.3832] |  | 0.119 / 0.186 / 0.2274 [-1.0092, 1.6119] |
| speed: v0 5-10 | 1040 | 79 | 4.1750 | 3.7918 | 0.3832 [0.0982, 0.6794] | **SEP** | 0.294 / 0.314 / 0.0355 [-0.4859, 0.5505] |
| speed: v0 10-15 | 952 | 62 | 2.9686 | 2.6243 | 0.3443 [0.0671, 0.6437] | **SEP** | 0.269 / 0.259 / -0.0188 [-0.5442, 0.5148] |
| speed: v0 15-25 | 679 | 34 | 1.6750 | 1.5277 | 0.1473 [0.0662, 0.2477] | **SEP** | 0.192 / 0.079 / -0.2608 [-0.6884, 0.1043] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5110 | 0.5291 | -0.0181 [-0.0823, 0.0379] |  | 0.069 / -0.003 / -0.4041 [-0.7778, -0.0971] BONF-SEP |
| nav: nav=follow | 2238 | 88 | 2.4629 | 2.1156 | 0.3473 [0.1797, 0.5410] | **SEP** | 0.632 / 0.613 / -0.0293 [-0.8404, 0.6161] |
| nav: nav=left (low n) | 327 | 13 | 11.1085 | 10.3337 | 0.7749 [-0.1050, 2.2359] |  | 0.092 / 0.200 / 0.4592 [-0.6093, 3.4786] |
| nav: nav=right | 976 | 38 | 4.5905 | 4.3474 | 0.2431 [-0.1799, 0.6348] |  | 0.276 / 0.187 / -0.1587 [-0.8897, 0.5654] |

episode concentration: 0.734 of episodes have mean delta > 0; top-5 episodes carry 0.448 of the summed delta; delta excluding the top 5 = 0.2051 (overall 0.3580).

### S6 -- R7-30k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3549 | 139 | 3.4846 | 3.2938 | 0.1908 [0.0588, 0.3349] | **SEP** |  |
| turn: straight <0.02 | 2100 | 122 | 1.4952 | 1.4589 | 0.0362 [-0.0110, 0.0859] |  | 0.592 / 0.112 / -0.3785 [-0.8609, 0.0334] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.9231 | 2.8900 | 0.0331 [-0.1546, 0.1783] |  | 0.190 / 0.033 / -0.1947 [-0.5714, 0.1053] |
| turn: turn 0.06-0.15 | 441 | 59 | 6.6920 | 6.3357 | 0.3563 [0.1050, 0.6272] | **SEP** | 0.124 / 0.232 / 0.1890 [-0.2408, 0.6244] |
| turn: sharp >=0.15 | 312 | 33 | 13.0227 | 11.7318 | 1.2909 [0.0867, 2.7274] | **SEP** | 0.088 / 0.595 / 1.2061 [-0.4307, 3.5327] |
| turn: undef (GT crawling) (low n) | 21 | 10 | 11.4093 | 10.5155 | 0.8938 [0.0024, 1.9559] | **SEP** | 0.006 / 0.028 / 0.7072 [-0.4215, 2.2930] |
| speed: v0 <2 | 211 | 28 | 13.6683 | 13.4029 | 0.2654 [-0.1792, 0.6725] |  | 0.059 / 0.083 / 0.0793 [-0.8546, 0.7086] |
| speed: v0 2-5 | 422 | 52 | 4.4408 | 4.1501 | 0.2907 [-0.1266, 0.7510] |  | 0.119 / 0.181 / 0.1133 [-0.5921, 0.8690] |
| speed: v0 5-10 | 1040 | 79 | 3.7918 | 3.4941 | 0.2977 [-0.0265, 0.7041] |  | 0.293 / 0.457 / 0.1512 [-0.3190, 0.8380] |
| speed: v0 10-15 | 952 | 62 | 2.6243 | 2.4870 | 0.1373 [-0.0097, 0.2911] |  | 0.268 / 0.193 / -0.0732 [-0.4397, 0.2685] |
| speed: v0 15-25 | 679 | 34 | 1.5277 | 1.4456 | 0.0821 [-0.0317, 0.2002] |  | 0.191 / 0.082 / -0.1343 [-0.4292, 0.1215] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5291 | 0.5193 | 0.0098 [-0.0491, 0.0663] |  | 0.069 / 0.004 / -0.1944 [-0.4372, -0.0034] BONF-SEP |
| nav: nav=follow | 2246 | 88 | 2.1125 | 1.9797 | 0.1327 [0.0064, 0.2914] | **SEP** | 0.633 / 0.440 / -0.1582 [-0.6592, 0.2335] |
| nav: nav=left (low n) | 327 | 13 | 10.3337 | 9.9260 | 0.4077 [-0.0314, 0.9220] |  | 0.092 / 0.197 / 0.2389 [-0.4585, 1.1807] |
| nav: nav=right | 976 | 38 | 4.3474 | 4.0957 | 0.2518 [-0.0511, 0.5890] |  | 0.275 / 0.363 / 0.0841 [-0.3758, 0.6657] |

episode concentration: 0.604 of episodes have mean delta > 0; top-5 episodes carry 0.633 of the summed delta; delta excluding the top 5 = 0.0727 (overall 0.1908).

### S6 -- R7-15k - R7-20k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3551 | 139 | 3.8679 | 3.7811 | 0.0868 [-0.0978, 0.3160] |  |  |
| turn: straight <0.02 | 2098 | 123 | 1.6337 | 1.7053 | -0.0716 [-0.1635, 0.0138] |  | 0.591 / -0.487 / -0.3873 [-1.2637, 0.1962] |
| turn: gentle 0.02-0.06 | 678 | 77 | 2.7762 | 2.9778 | -0.2016 [-0.4266, 0.0101] |  | 0.191 / -0.443 / -0.3564 [-0.8904, 0.0482] |
| turn: turn 0.06-0.15 | 442 | 59 | 6.9775 | 7.1717 | -0.1941 [-0.6395, 0.2093] |  | 0.124 / -0.278 / -0.3209 [-0.9978, 0.3614] |
| turn: sharp >=0.15 | 314 | 33 | 16.3573 | 14.1512 | 2.2061 [0.5312, 4.2684] | **SEP** | 0.088 / 2.247 / 2.3248 [-0.1534, 5.9243] |
| turn: undef (GT crawling) (low n) | 19 | 11 | 10.7892 | 11.3999 | -0.6108 [-1.3061, 0.1087] |  | 0.005 / -0.038 / -0.7014 [-1.8196, 0.2251] |
| speed: v0 <2 | 213 | 26 | 14.9989 | 14.6800 | 0.3189 [-1.0164, 2.6062] |  | 0.060 / 0.220 / 0.2469 [-1.9896, 3.4759] |
| speed: v0 2-5 | 422 | 52 | 4.9992 | 5.2980 | -0.2988 [-1.0083, 0.5732] |  | 0.119 / -0.409 / -0.4377 [-1.5020, 0.7461] |
| speed: v0 5-10 | 1040 | 79 | 4.1750 | 4.0421 | 0.1329 [-0.1578, 0.4088] |  | 0.293 / 0.448 / 0.0651 [-0.5042, 0.5328] |
| speed: v0 10-15 | 952 | 62 | 2.9686 | 2.7810 | 0.1876 [0.0036, 0.4037] | **SEP** | 0.268 / 0.579 / 0.1377 [-0.3859, 0.6106] |
| speed: v0 15-25 | 679 | 34 | 1.6750 | 1.6197 | 0.0553 [-0.0411, 0.1589] |  | 0.191 / 0.122 / -0.0390 [-0.4960, 0.3121] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5110 | 0.4613 | 0.0497 [-0.0026, 0.1031] |  | 0.069 / 0.039 / -0.0399 [-0.4177, 0.2491] |
| nav: nav=follow | 2236 | 88 | 2.4643 | 2.3270 | 0.1373 [0.0114, 0.2792] | **SEP** | 0.630 / 0.996 / 0.1364 [-0.8257, 0.7741] |
| nav: nav=left (low n) | 335 | 13 | 11.1500 | 10.4695 | 0.6805 [-0.4716, 2.6283] |  | 0.094 / 0.739 / 0.6555 [-0.7460, 4.4676] |
| nav: nav=right | 980 | 38 | 4.5811 | 4.8125 | -0.2314 [-0.5554, 0.1086] |  | 0.276 / -0.735 / -0.4395 [-1.0863, 0.1981] |

episode concentration: 0.525 of episodes have mean delta > 0; top-5 episodes carry 1.872 of the summed delta; delta excluding the top 5 = -0.0786 (overall 0.0868).

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

### S6 -- R7-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3496 | 139 | 0.9170 | 0.8149 | 0.1022 [0.0638, 0.1394] | **SEP** | 0.3045 [0.0662, 0.7287] SEP |
| seg1 | straight | 2071 | 122 | 0.6849 | 0.5435 | 0.1413 [0.0414, 0.3401] | **SEP** | 0.3754 [0.0941, 0.9549] SEP |
| seg1 | curved | 1407 | 88 | 1.1386 | 1.0929 | 0.0458 [-0.2240, 0.2248] |  | 0.2041 [0.0046, 0.4934] SEP |
| seg2 | all | 3483 | 139 | 1.6601 | 1.4614 | 0.1988 [0.0816, 0.3187] | **SEP** | 0.3599 [0.0521, 0.8084] SEP |
| seg2 | straight | 2073 | 121 | 0.9908 | 0.8390 | 0.1518 [0.0365, 0.3605] | **SEP** | 0.4137 [0.1128, 1.0026] SEP |
| seg2 | curved | 1410 | 88 | 2.6441 | 2.3764 | 0.2678 [-0.0656, 0.5807] |  | 0.2807 [-0.1114, 0.6728] |
| seg3 | all | 3465 | 139 | 2.3871 | 2.0678 | 0.3193 [0.0865, 0.5461] | **SEP** | 0.7654 [0.2995, 1.3814] SEP |
| seg3 | straight | 2061 | 122 | 1.1558 | 0.9749 | 0.1809 [0.1169, 0.2481] | **SEP** | 0.7075 [0.2995, 1.4833] SEP |
| seg3 | curved | 1404 | 88 | 4.1946 | 3.6721 | 0.5225 [-0.0499, 1.0894] |  | 0.8504 [0.1569, 1.5063] SEP |
| seg4 | all | 3439 | 139 | 3.6267 | 2.9850 | 0.6417 [0.3033, 0.9828] | **SEP** | 0.9080 [0.4299, 1.4149] SEP |
| seg4 | straight | 2050 | 122 | 1.3147 | 1.1340 | 0.1808 [-0.0505, 0.3330] |  | 0.7837 [0.4447, 1.3805] SEP |
| seg4 | curved | 1389 | 87 | 7.0390 | 5.7169 | 1.3221 [0.5397, 2.1416] | **SEP** | 1.0914 [0.0637, 2.1003] SEP |
| seg5 | all | 3412 | 139 | 5.1587 | 4.3337 | 0.8250 [0.3927, 1.2782] | **SEP** | 1.0672 [0.3385, 1.9196] SEP |
| seg5 | straight | 2035 | 122 | 1.3487 | 1.3111 | 0.0376 [-0.3929, 0.2876] |  | 0.9575 [0.3426, 2.1567] SEP |
| seg5 | curved | 1377 | 88 | 10.7894 | 8.8006 | 1.9888 [1.0115, 2.9559] | **SEP** | 1.2293 [-0.1088, 2.5020] |
| seg6 | all | 3380 | 139 | 6.9595 | 5.9834 | 0.9761 [0.4910, 1.4814] | **SEP** | 0.7965 [0.1103, 1.5188] SEP |
| seg6 | straight | 2026 | 122 | 1.7329 | 1.7347 | -0.0017 [-0.4553, 0.2679] |  | 0.6433 [0.1603, 1.4938] SEP |
| seg6 | curved | 1350 | 88 | 14.8095 | 12.3511 | 2.4583 [1.4295, 3.5040] | **SEP** | 1.0169 [-0.3976, 2.4642] |

### S6 -- R7-15k - R7-30k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3494 | 139 | 0.9175 | 0.8019 | 0.1156 [0.0380, 0.2343] | **SEP** | 0.1253 [-0.0609, 0.4487] |
| seg1 | straight | 2070 | 122 | 0.6852 | 0.5623 | 0.1229 [0.0255, 0.3209] | **SEP** | 0.2799 [0.0033, 0.8568] SEP |
| seg1 | curved | 1407 | 88 | 1.1386 | 1.0311 | 0.1075 [0.0374, 0.1786] | **SEP** | -0.0978 [-0.1899, 0.0019] |
| seg2 | all | 3480 | 139 | 1.6615 | 1.4283 | 0.2332 [0.0529, 0.4844] | **SEP** | -0.1306 [-0.2594, -0.0030] SEP |
| seg2 | straight | 2070 | 121 | 0.9921 | 0.7878 | 0.2043 [0.0095, 0.5951] | **SEP** | -0.0266 [-0.0699, 0.0194] |
| seg2 | curved | 1410 | 88 | 2.6441 | 2.3685 | 0.2757 [0.0579, 0.4853] | **SEP** | -0.2833 [-0.5765, 0.0094] |
| seg3 | all | 3454 | 139 | 2.3988 | 2.4215 | -0.0227 [-0.6069, 0.3545] |  | -0.2616 [-0.8718, 0.1663] |
| seg3 | straight | 2051 | 121 | 1.1594 | 1.3792 | -0.2198 [-0.9898, 0.1663] |  | -0.2732 [-1.0578, 0.1250] |
| seg3 | curved | 1403 | 88 | 4.2107 | 3.9452 | 0.2655 [-0.1921, 0.6974] |  | -0.2446 [-0.8693, 0.3261] |
| seg4 | all | 3421 | 139 | 3.6414 | 3.3305 | 0.3109 [-0.0995, 0.6763] |  | -0.0600 [-0.7194, 0.4529] |
| seg4 | straight | 2036 | 121 | 1.3184 | 1.3577 | -0.0394 [-0.6234, 0.2688] |  | -0.1933 [-1.1766, 0.3074] |
| seg4 | curved | 1385 | 87 | 7.0564 | 6.2306 | 0.8258 [0.2085, 1.4576] | **SEP** | 0.1360 [-0.6712, 0.9036] |
| seg5 | all | 3389 | 139 | 5.1695 | 4.6962 | 0.4732 [-0.0129, 0.9188] |  | 0.1791 [-0.3572, 0.6760] |
| seg5 | straight | 2017 | 121 | 1.3444 | 1.4081 | -0.0636 [-0.6770, 0.2533] |  | -0.0126 [-0.6245, 0.3217] |
| seg5 | curved | 1372 | 88 | 10.7927 | 9.5303 | 1.2624 [0.4858, 2.0653] | **SEP** | 0.4608 [-0.5640, 1.4603] |
| seg6 | all | 3349 | 139 | 6.9815 | 6.3514 | 0.6300 [0.2555, 1.0324] | **SEP** | 0.2837 [-0.2850, 0.8530] |
| seg6 | straight | 2002 | 121 | 1.7157 | 1.6377 | 0.0780 [-0.1628, 0.2467] |  | -0.0585 [-0.6984, 0.3083] |
| seg6 | curved | 1344 | 88 | 14.8256 | 13.3487 | 1.4769 [0.5889, 2.3545] | **SEP** | 0.7710 [-0.4077, 1.9320] |

### S6 -- R7-30k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3503 | 139 | 0.8005 | 0.8142 | -0.0137 [-0.1206, 0.0493] |  | 0.1795 [0.1104, 0.2916] SEP |
| seg1 | straight | 2074 | 122 | 0.5614 | 0.5429 | 0.0185 [0.0082, 0.0302] | **SEP** | 0.0957 [0.0798, 0.1134] SEP |
| seg1 | curved | 1410 | 88 | 1.0299 | 1.0917 | -0.0618 [-0.3267, 0.0898] |  | 0.3011 [0.1362, 0.5761] SEP |
| seg2 | all | 3487 | 139 | 1.4261 | 1.4610 | -0.0349 [-0.2550, 0.1046] |  | 0.4897 [0.2436, 0.9216] SEP |
| seg2 | straight | 2076 | 122 | 0.7862 | 0.8392 | -0.0530 [-0.2493, 0.0510] |  | 0.4392 [0.1599, 1.0155] SEP |
| seg2 | curved | 1411 | 88 | 2.3676 | 2.3758 | -0.0082 [-0.2936, 0.2017] |  | 0.5640 [0.3356, 0.8379] SEP |
| seg3 | all | 3457 | 139 | 2.4200 | 2.0736 | 0.3464 [0.0432, 0.8129] | **SEP** | 1.0200 [0.4230, 2.0968] SEP |
| seg3 | straight | 2055 | 121 | 1.3783 | 0.9797 | 0.3986 [0.0163, 1.1768] | **SEP** | 0.9783 [0.2482, 2.4994] SEP |
| seg3 | curved | 1402 | 88 | 3.9468 | 3.6768 | 0.2699 [-0.0079, 0.5586] |  | 1.0811 [0.6257, 1.6396] SEP |
| seg4 | all | 3429 | 139 | 3.3254 | 2.9935 | 0.3318 [0.0788, 0.6548] | **SEP** | 0.9641 [0.4202, 1.8581] SEP |
| seg4 | straight | 2043 | 121 | 1.3570 | 1.1396 | 0.2174 [0.0015, 0.6218] | **SEP** | 0.9700 [0.2262, 2.5038] SEP |
| seg4 | curved | 1386 | 87 | 6.2269 | 5.7263 | 0.5006 [0.0836, 0.9849] | **SEP** | 0.9553 [0.4742, 1.4362] SEP |
| seg5 | all | 3392 | 139 | 4.6946 | 4.3488 | 0.3458 [0.1019, 0.6151] | **SEP** | 0.8804 [0.2164, 1.9076] SEP |
| seg5 | straight | 2020 | 121 | 1.4102 | 1.3111 | 0.0991 [-0.0424, 0.3198] |  | 0.9737 [0.1177, 2.7351] SEP |
| seg5 | curved | 1372 | 88 | 9.5303 | 8.8212 | 0.7091 [0.2110, 1.2606] | **SEP** | 0.7430 [0.1153, 1.3508] SEP |
| seg6 | all | 3354 | 139 | 6.3528 | 6.0130 | 0.3398 [0.0689, 0.6388] | **SEP** | 0.5037 [-0.0831, 1.3207] |
| seg6 | straight | 2006 | 121 | 1.6402 | 1.7276 | -0.0874 [-0.3060, 0.0751] |  | 0.7044 [-0.0084, 2.0729] |
| seg6 | curved | 1346 | 88 | 13.3484 | 12.3837 | 0.9647 [0.3682, 1.6342] | **SEP** | 0.2176 [-0.5266, 0.9362] |

### S6 -- R7-15k - R7-20k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3493 | 139 | 0.9177 | 1.0366 | -0.1189 [-0.4354, 0.0562] |  | 0.0155 [-0.1140, 0.1060] |
| seg1 | straight | 2069 | 122 | 0.6855 | 0.6581 | 0.0274 [0.0147, 0.0418] | **SEP** | 0.0852 [0.0601, 0.1120] SEP |
| seg1 | curved | 1407 | 88 | 1.1386 | 1.4719 | -0.3332 [-1.1481, 0.0969] |  | -0.0866 [-0.3979, 0.1247] |
| seg2 | all | 3482 | 139 | 1.6617 | 1.6568 | 0.0049 [-0.2266, 0.1719] |  | 0.0070 [-0.2709, 0.2202] |
| seg2 | straight | 2071 | 121 | 0.9919 | 1.0551 | -0.0632 [-0.2623, 0.0472] |  | 0.0841 [-0.1286, 0.2147] |
| seg2 | curved | 1411 | 88 | 2.6448 | 2.5400 | 0.1048 [-0.2206, 0.3913] |  | -0.1062 [-0.5207, 0.2581] |
| seg3 | all | 3460 | 139 | 2.4023 | 2.4441 | -0.0417 [-0.5366, 0.2952] |  | 0.2440 [-0.4904, 0.7591] |
| seg3 | straight | 2053 | 122 | 1.1578 | 1.3307 | -0.1729 [-0.7671, 0.1336] |  | 0.0956 [-0.9020, 0.6177] |
| seg3 | curved | 1407 | 88 | 4.2182 | 4.0686 | 0.1496 [-0.3311, 0.6174] |  | 0.4606 [-0.1654, 1.0357] |
| seg4 | all | 3427 | 139 | 3.6266 | 3.5990 | 0.0275 [-0.3873, 0.3675] |  | 0.5886 [-0.0810, 1.1254] |
| seg4 | straight | 2039 | 122 | 1.3154 | 1.5273 | -0.2119 [-0.8128, 0.1138] |  | 0.3910 [-0.6434, 0.9550] |
| seg4 | curved | 1388 | 87 | 7.0217 | 6.6425 | 0.3793 [-0.1345, 0.9399] |  | 0.8788 [0.0915, 1.6129] SEP |
| seg5 | all | 3402 | 139 | 5.1708 | 5.2461 | -0.0753 [-0.5342, 0.3438] |  | 0.8112 [0.2873, 1.3057] SEP |
| seg5 | straight | 2024 | 122 | 1.3484 | 1.7945 | -0.4462 [-1.0821, -0.0675] | **SEP** | 0.6877 [0.0064, 1.1123] SEP |
| seg5 | curved | 1378 | 88 | 10.7852 | 10.3158 | 0.4694 [-0.1728, 1.2269] |  | 0.9925 [0.0074, 1.9018] SEP |
| seg6 | all | 3369 | 138 | 6.9764 | 7.1176 | -0.1412 [-0.5053, 0.2047] |  | 1.0040 [0.5257, 1.4890] SEP |
| seg6 | straight | 2015 | 121 | 1.7257 | 2.1695 | -0.4438 [-0.7543, -0.1862] | **SEP** | 0.9159 [0.5798, 1.2324] SEP |
| seg6 | curved | 1351 | 88 | 14.8080 | 14.4919 | 0.3161 [-0.3947, 1.0468] |  | 1.1314 [0.0568, 2.2238] SEP |

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

### S6 -- R7-15k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 378 | 40 | 5.4378 | 5.0591 | 0.3787 [0.2326, 0.5157] | **SEP** |  |
| straight <0.02 | v0 5-15 | 1050 | 84 | 0.9549 | 0.7632 | 0.1917 [0.1090, 0.2770] | **SEP** |  |
| straight <0.02 | v0 >=15 | 684 | 36 | 0.5745 | 0.5523 | 0.0221 [-0.0224, 0.0694] |  |  |
| gentle 0.02-0.06 | v0 <5 | 101 | 22 | 4.2718 | 5.4046 | -1.1328 [-4.5269, 0.8273] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 424 | 59 | 2.4937 | 2.5310 | -0.0373 [-0.2501, 0.1805] |  |  |
| gentle 0.02-0.06 | v0 >=15 (low n) | 153 | 14 | 2.5720 | 2.1952 | 0.3768 [0.1616, 0.5916] | **SEP** |  |
| turn 0.06-0.15 | v0 <5 (low n) | 71 | 20 | 10.8270 | 10.6521 | 0.1749 [-1.6550, 1.6551] |  |  |
| turn 0.06-0.15 | v0 5-15 | 293 | 43 | 6.4995 | 5.8272 | 0.6723 [-0.0951, 1.4314] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 78 | 10 | 5.2691 | 4.3281 | 0.9410 [0.3730, 1.4627] | **SEP** |  |
| sharp >=0.15 | v0 <5 (low n) | 78 | 16 | 23.0562 | 15.1685 | 7.8877 [2.3014, 15.4056] | **SEP** |  |
| sharp >=0.15 | v0 5-15 | 225 | 26 | 14.2388 | 10.7541 | 3.4847 [1.8114, 4.9842] | **SEP** |  |
| sharp >=0.15 | v0 >=15 (low n) | 9 | 3 | 7.2306 | 6.3909 | 0.8397 [0.3388, 2.0321] | **SEP** |  |
| undef (GT crawling) | v0 <5 (low n) | 22 | 13 | 9.3763 | 10.0709 | -0.6945 [-1.6720, 0.1677] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S6 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | R7-15k s0-s1 [CI] | R7-30k s0-s1 [CI] | R7-20k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|---|
| overall | 0.0385 [-0.0243, 0.1139] | 0.0185 [-0.0369, 0.0709] | 0.0032 [-0.0496, 0.0598] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | -0.0089 [-0.0410, 0.0275] | -0.0084 [-0.0264, 0.0108] | 0.0262 [-0.0207, 0.0960] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | 0.0091 [-0.0537, 0.0850] | 0.0693 [-0.0460, 0.2432] | 0.0292 [-0.0744, 0.1218] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | 0.0410 [-0.0646, 0.1414] | 0.0928 [-0.0314, 0.2301] | -0.0675 [-0.2282, 0.1022] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | 0.4196 [-0.0921, 1.0546] | 0.0475 [-0.4823, 0.4413] | -0.1077 [-0.5568, 0.3422] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | 0.0169 [-0.2314, 0.3191] | -0.9093 [-2.3766, 0.3252] | 0.0091 [-0.1871, 0.2769] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | 0.5111 [-0.0175, 1.3551] | 0.2784 [-0.0018, 0.6372] | 0.4396 [0.0260, 1.0407] SEP | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | 0.2492 [0.0195, 0.5701] SEP | -0.0736 [-0.4282, 0.2864] | -0.1181 [-0.4121, 0.0983] | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | -0.0596 [-0.1349, 0.0154] | 0.0440 [-0.0405, 0.1290] | 0.0358 [-0.1045, 0.1956] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | -0.0267 [-0.0712, 0.0117] | 0.0092 [-0.0307, 0.0556] | -0.0728 [-0.1371, -0.0117] SEP | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | 0.0013 [-0.0456, 0.0457] | -0.0311 [-0.0639, -0.0040] SEP | 0.0008 [-0.0513, 0.0624] | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | -0.0103 [-0.0282, 0.0092] | 0.0189 [-0.0164, 0.0666] | -0.0123 [-0.0357, 0.0129] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | -0.0124 [-0.0413, 0.0165] | 0.0236 [-0.0124, 0.0651] | -0.0214 [-0.0692, 0.0290] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | 0.4812 [0.0017, 1.1772] SEP | 0.1426 [-0.0193, 0.3162] | 0.0776 [0.0039, 0.1823] SEP | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | 0.0029 [-0.0995, 0.1193] | -0.0347 [-0.1898, 0.1397] | 0.0340 [-0.1064, 0.1917] | 0.0235 [-0.1175, 0.1775] |

### S6 -- R7-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.5500 | 0.5546 / 0.5416 | True / True / True | 0.0385 / 0.0128 | 14.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.1703 | 0.1769 / 0.1636 | True / True / True | 0.0089 / 0.0216 | 7.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | -0.1070 | -0.1385 / -0.0843 | False / False / False | 0.0091 / 0.0713 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.6398 | 0.6298 / 0.6499 | True / True / True | 0.0410 / 0.0611 | 10.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 4.5092 | 4.5279 / 4.4560 | True / True / True | 0.4196 / 0.2004 | 10.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | -0.6945 | 0.3228 / -1.7119 | False / False / False | 0.0169 / 1.9278 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 1.2405 | 1.3603 / 1.0532 | True / True / True | 0.5111 / 0.0328 | 2.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 2-5 | 0.8490 | 0.8732 / 0.8249 | False / False / False | 0.2492 / 0.2009 | 3.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.6809 | 0.6632 / 0.6985 | True / True / True | 0.0596 / 0.0243 | 11.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.4816 | 0.4816 / 0.4815 | True / True / True | 0.0267 / 0.0268 | 18.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.2294 | 0.2273 / 0.2315 | True / True / True | 0.0013 / 0.0055 | 42.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0083 | -0.0138 / -0.0027 | False / False / False | 0.0103 / 0.0008 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.4806 | 0.4693 / 0.4917 | True / True / True | 0.0124 / 0.0080 | 38.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 1.1935 | 1.3424 / 1.0016 | True / True / True | 0.4812 / 0.0140 | 2.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=right | 0.4895 | 0.4812 / 0.4978 | False / False / False | 0.0029 / 0.0235 | 20.9 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- R7-15k - R7-30k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.3580 | 0.3566 / 0.3601 | True / True / True | 0.0385 / 0.0185 | 9.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.1301 | 0.1249 / 0.1365 | True / True / True | 0.0089 / 0.0084 | 14.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | -0.1519 | -0.1835 / -0.1203 | False / False / False | 0.0091 / 0.0693 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.2895 | 0.2620 / 0.3170 | False / False / False | 0.0410 / 0.0928 | 3.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 3.2183 | 3.3135 / 3.1231 | True / True / True | 0.4196 / 0.0475 | 7.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | -1.7773 | -1.2663 / -2.2883 | False / False / False | 0.0169 / 0.9093 | 2.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 1.0363 | 0.9896 / 1.0867 | True / True / True | 0.5111 / 0.2784 | 2.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 2-5 | 0.5584 | 0.7198 / 0.3970 | False / False / False | 0.2492 / 0.0736 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.3832 | 0.3313 / 0.4350 | True / True / True | 0.0596 / 0.0440 | 6.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.3443 | 0.3263 / 0.3623 | True / True / True | 0.0267 / 0.0092 | 12.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.1473 | 0.1635 / 0.1311 | True / True / True | 0.0013 / 0.0311 | 4.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0181 | -0.0327 / -0.0035 | False / False / False | 0.0103 / 0.0189 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.3473 | 0.3286 / 0.3659 | True / True / True | 0.0124 / 0.0236 | 14.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.7749 | 0.8319 / 0.7229 | False / False / False | 0.4812 / 0.1426 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.2431 | 0.2617 / 0.2248 | False / False / False | 0.0029 / 0.0347 | 7.0 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- R7-30k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1908 | 0.1958 / 0.1870 | True / True / True | 0.0185 / 0.0128 | 10.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0362 | 0.0448 / 0.0298 | False / False / False | 0.0084 / 0.0216 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.0331 | 0.0316 / 0.0346 | False / False / False | 0.0693 / 0.0713 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.3563 | 0.3818 / 0.3308 | True / True / True | 0.0928 / 0.0611 | 3.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 1.2909 | 1.2144 / 1.3673 | False / True / True | 0.0475 / 0.2004 | 6.4 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|undef (GT crawling) | 0.8938 | 1.5319 / 0.2558 | True / False / True | 0.9093 / 1.9278 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.2654 | 0.4228 / 0.1281 | False / False / False | 0.2784 / 0.0328 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.2907 | 0.1534 / 0.4279 | False / False / False | 0.0736 / 0.2009 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.2977 | 0.3318 / 0.2636 | False / False / False | 0.0440 / 0.0243 | 6.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.1373 | 0.1553 / 0.1192 | False / False / False | 0.0092 / 0.0268 | 5.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0821 | 0.0638 / 0.1005 | False / False / False | 0.0311 / 0.0055 | 2.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0098 | 0.0188 / 0.0007 | False / False / False | 0.0189 / 0.0008 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.1327 | 0.1404 / 0.1251 | True / True / True | 0.0236 / 0.0080 | 5.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.4077 | 0.4797 / 0.3520 | False / False / False | 0.1426 / 0.0140 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.2518 | 0.2282 / 0.2738 | False / False / False | 0.0347 / 0.0235 | 7.3 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- R7-15k - R7-20k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0868 | 0.1036 / 0.0645 | False / False / False | 0.0385 / 0.0032 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | -0.0716 | -0.0903 / -0.0525 | False / False / False | 0.0089 / 0.0262 | 2.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | -0.2016 | -0.2133 / -0.2195 | False / False / False | 0.0091 / 0.0292 | 6.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.1941 | -0.1399 / -0.2484 | False / False / False | 0.0410 / 0.0675 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 2.2061 | 2.4697 / 1.9424 | True / True / True | 0.4196 / 0.1077 | 5.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | -0.6108 | -0.5933 / -0.6283 | False / False / False | 0.0169 / 0.0091 | 36.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.3189 | 0.3605 / 0.1816 | False / False / False | 0.5111 / 0.4396 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.2988 | -0.1152 / -0.4825 | False / False / False | 0.2492 / 0.1181 | 1.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1329 | 0.0852 / 0.1806 | False / False / False | 0.0596 / 0.0358 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.1876 | 0.2107 / 0.1646 | True / False / True | 0.0267 / 0.0728 | 2.6 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 15-25 | 0.0553 | 0.0555 / 0.0551 | False / False / False | 0.0013 / 0.0008 | 43.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0497 | 0.0507 / 0.0486 | False / True / False | 0.0103 / 0.0123 | 4.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.1373 | 0.1416 / 0.1332 | True / True / True | 0.0124 / 0.0214 | 6.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.6805 | 0.8732 / 0.4258 | False / False / False | 0.4812 / 0.0776 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.2314 | -0.2458 / -0.2161 | False / False / False | 0.0029 / 0.0340 | 6.8 | NOT PROVEN (not separated and/or within 2x floor) |

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

* seg1 (n 1407): R7-15k bias +0.153, refcv7-50400 bias +0.038, R7-30k bias +0.107, R7-20k bias +0.396; R7-15k MAE 1.139, refcv7-50400 MAE 1.093, R7-30k MAE 1.031, R7-20k MAE 1.472
* seg2 (n 1410): R7-15k bias -0.675, refcv7-50400 bias -0.243, R7-30k bias -0.399, R7-20k bias -0.049; R7-15k MAE 2.644, refcv7-50400 MAE 2.376, R7-30k MAE 2.368, R7-20k MAE 2.540
* seg3 (n 1402): R7-15k bias -2.455, refcv7-50400 bias -1.128, R7-30k bias -1.476, R7-20k bias -1.175; R7-15k MAE 4.200, refcv7-50400 MAE 3.677, R7-30k MAE 3.947, R7-20k MAE 4.066
* seg4 (n 1382): R7-15k bias -4.788, refcv7-50400 bias -2.567, R7-30k bias -3.342, R7-20k bias -3.171; R7-15k MAE 7.003, refcv7-50400 MAE 5.709, R7-30k MAE 6.175, R7-20k MAE 6.633
* seg5 (n 1370): R7-15k bias -7.438, refcv7-50400 bias -4.512, R7-30k bias -5.246, R7-20k bias -5.429; R7-15k MAE 10.736, refcv7-50400 MAE 8.809, R7-30k MAE 9.472, R7-20k MAE 10.276
* seg6 (n 1341): R7-15k bias -9.717, refcv7-50400 bias -6.970, R7-30k bias -7.323, R7-20k bias -7.590; R7-15k MAE 14.777, refcv7-50400 MAE 12.379, R7-30k MAE 13.303, R7-20k MAE 14.476
**sharp (>= 0.15)**

* seg1 (n 303): R7-15k bias -0.109, refcv7-50400 bias -0.422, R7-30k bias -0.029, R7-20k bias +0.384; R7-15k MAE 1.814, refcv7-50400 MAE 2.035, R7-30k MAE 1.545, R7-20k MAE 2.991
* seg2 (n 302): R7-15k bias -2.878, refcv7-50400 bias -0.568, R7-30k bias -1.570, R7-20k bias -1.246; R7-15k MAE 5.244, refcv7-50400 MAE 3.821, R7-30k MAE 4.098, R7-20k MAE 4.330
* seg3 (n 303): R7-15k bias -8.156, refcv7-50400 bias -2.527, R7-30k bias -4.814, R7-20k bias -4.891; R7-15k MAE 9.921, refcv7-50400 MAE 6.701, R7-30k MAE 7.733, R7-20k MAE 8.128
* seg4 (n 302): R7-15k bias -15.080, refcv7-50400 bias -6.666, R7-30k bias -9.774, R7-20k bias -10.581; R7-15k MAE 17.352, refcv7-50400 MAE 11.654, R7-30k MAE 13.513, R7-20k MAE 14.691
* seg5 (n 298): R7-15k bias -22.272, refcv7-50400 bias -11.750, R7-30k bias -15.121, R7-20k bias -16.794; R7-15k MAE 26.449, refcv7-50400 MAE 18.992, R7-30k MAE 20.919, R7-20k MAE 23.249
* seg6 (n 291): R7-15k bias -26.710, refcv7-50400 bias -17.606, R7-30k bias -20.208, R7-20k bias -21.525; R7-15k MAE 34.056, refcv7-50400 MAE 25.321, R7-30k MAE 27.595, R7-20k MAE 31.151
**curved & v0 5-10**

* seg1 (n 547): R7-15k bias +0.061, refcv7-50400 bias +0.049, R7-30k bias -0.010, R7-20k bias +0.199; R7-15k MAE 1.142, refcv7-50400 MAE 0.896, R7-30k MAE 0.993, R7-20k MAE 1.009
* seg2 (n 547): R7-15k bias -0.897, refcv7-50400 bias -0.516, R7-30k bias -0.769, R7-20k bias -0.417; R7-15k MAE 2.791, refcv7-50400 MAE 2.231, R7-30k MAE 2.434, R7-20k MAE 2.356
* seg3 (n 545): R7-15k bias -2.874, refcv7-50400 bias -1.649, R7-30k bias -2.275, R7-20k bias -1.729; R7-15k MAE 4.488, refcv7-50400 MAE 3.822, R7-30k MAE 4.235, R7-20k MAE 4.200
* seg4 (n 536): R7-15k bias -5.460, refcv7-50400 bias -3.537, R7-30k bias -4.318, R7-20k bias -3.780; R7-15k MAE 7.524, refcv7-50400 MAE 6.217, R7-30k MAE 6.881, R7-20k MAE 7.042
* seg5 (n 525): R7-15k bias -8.157, refcv7-50400 bias -5.842, R7-30k bias -6.405, R7-20k bias -6.085; R7-15k MAE 11.059, refcv7-50400 MAE 9.315, R7-30k MAE 10.127, R7-20k MAE 10.581
* seg6 (n 505): R7-15k bias -10.293, refcv7-50400 bias -8.429, R7-30k bias -8.507, R7-20k bias -8.117; R7-15k MAE 14.968, refcv7-50400 MAE 12.679, R7-30k MAE 13.699, R7-20k MAE 14.632
**sharp & v0 5-10**

* seg1 (n 139): R7-15k bias +0.172, refcv7-50400 bias +0.062, R7-30k bias +0.021, R7-20k bias +0.642; R7-15k MAE 1.664, refcv7-50400 MAE 1.251, R7-30k MAE 1.454, R7-20k MAE 1.538
* seg2 (n 139): R7-15k bias -2.425, refcv7-50400 bias -1.267, R7-30k bias -1.794, R7-20k bias -0.900; R7-15k MAE 5.063, refcv7-50400 MAE 3.564, R7-30k MAE 3.957, R7-20k MAE 3.694
* seg3 (n 139): R7-15k bias -8.469, refcv7-50400 bias -4.133, R7-30k bias -5.957, R7-20k bias -5.173; R7-15k MAE 9.896, refcv7-50400 MAE 7.029, R7-30k MAE 8.355, R7-20k MAE 7.618
* seg4 (n 139): R7-15k bias -15.938, refcv7-50400 bias -8.959, R7-30k bias -11.538, R7-20k bias -11.043; R7-15k MAE 17.479, refcv7-50400 MAE 12.455, R7-30k MAE 14.544, R7-20k MAE 14.295
* seg5 (n 137): R7-15k bias -22.393, refcv7-50400 bias -14.360, R7-30k bias -16.421, R7-20k bias -16.085; R7-15k MAE 25.708, refcv7-50400 MAE 19.770, R7-30k MAE 22.077, R7-20k MAE 22.547
* seg6 (n 133): R7-15k bias -25.152, refcv7-50400 bias -18.859, R7-30k bias -19.645, R7-20k bias -18.244; R7-15k MAE 32.666, refcv7-50400 MAE 26.056, R7-30k MAE 28.129, R7-20k MAE 29.929

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1341 / 1341 | 26.0 | R7-15k -0.3755 [-0.4661, -0.2919]; refcv7-50400 -0.2504 [-0.3245, -0.1833]; R7-30k -0.2793 [-0.3580, -0.2062]; R7-20k -0.2884 [-0.3728, -0.2151] | R7-15k -0.1251 [-0.1761, -0.0776] SEP; R7-30k -0.0289 [-0.0638, +0.0045]; R7-20k -0.0380 [-0.0851, -0.0014] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1427 / 6905 | 16.4 | R7-15k -0.3083 [-0.3907, -0.2322]; refcv7-50400 -0.1810 [-0.2473, -0.1219]; R7-30k -0.2025 [-0.2806, -0.1311]; R7-20k -0.1979 [-0.2794, -0.1211] | R7-15k -0.1273 [-0.1839, -0.0691] SEP; R7-30k -0.0216 [-0.0682, +0.0375]; R7-20k -0.0169 [-0.0727, +0.0573] |
| sharp (>= 0.15) | last segment | 291 / 291 | 62.9 | R7-15k -0.4110 [-0.5262, -0.3117]; refcv7-50400 -0.2669 [-0.3577, -0.1893]; R7-30k -0.3103 [-0.4086, -0.2253]; R7-20k -0.3197 [-0.4306, -0.2346] | R7-15k -0.1441 [-0.2093, -0.0881] SEP; R7-30k -0.0434 [-0.0888, -0.0001] SEP; R7-20k -0.0528 [-0.1173, -0.0076] SEP |
| sharp (>= 0.15) | segments 2..K | 311 / 1496 | 39.5 | R7-15k -0.3742 [-0.4885, -0.2807]; refcv7-50400 -0.1965 [-0.2838, -0.1243]; R7-30k -0.2575 [-0.3520, -0.1774]; R7-20k -0.2656 [-0.3702, -0.1828] | R7-15k -0.1776 [-0.2464, -0.1224] SEP; R7-30k -0.0610 [-0.1039, -0.0228] SEP; R7-20k -0.0691 [-0.1253, -0.0291] SEP |
| curved & v0 5-10 | last segment | 505 / 505 | 27.9 | R7-15k -0.3624 [-0.4695, -0.2640]; refcv7-50400 -0.2748 [-0.4002, -0.1735]; R7-30k -0.2937 [-0.4023, -0.2021]; R7-20k -0.2758 [-0.4029, -0.1654] | R7-15k -0.0876 [-0.1420, -0.0352] SEP; R7-30k -0.0189 [-0.0796, +0.0392]; R7-20k -0.0010 [-0.0486, +0.0380] |
| curved & v0 5-10 | segments 2..K | 547 / 2658 | 17.7 | R7-15k -0.3175 [-0.4129, -0.2340]; refcv7-50400 -0.2066 [-0.3209, -0.1203]; R7-30k -0.2428 [-0.3458, -0.1603]; R7-20k -0.2240 [-0.3313, -0.1360] | R7-15k -0.1109 [-0.1604, -0.0604] SEP; R7-30k -0.0362 [-0.0923, +0.0157]; R7-20k -0.0174 [-0.0556, +0.0159] |
| sharp & v0 5-10 | last segment | 133 / 133 | 66.9 | R7-15k -0.3734 [-0.4998, -0.2658]; refcv7-50400 -0.2729 [-0.4098, -0.1615]; R7-30k -0.3008 [-0.4236, -0.2020]; R7-20k -0.2721 [-0.4126, -0.1519] | R7-15k -0.1004 [-0.1573, -0.0553] SEP; R7-30k -0.0279 [-0.1019, +0.0430]; R7-20k +0.0008 [-0.0578, +0.0465] |
| sharp & v0 5-10 | segments 2..K | 139 / 687 | 43.7 | R7-15k -0.3414 [-0.4553, -0.2432]; refcv7-50400 -0.2114 [-0.3439, -0.1138]; R7-30k -0.2594 [-0.3802, -0.1659]; R7-20k -0.2352 [-0.3590, -0.1300] | R7-15k -0.1300 [-0.1874, -0.0811] SEP; R7-30k -0.0480 [-0.1184, +0.0180]; R7-20k -0.0239 [-0.0743, +0.0170] |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**R7-15k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3566 | 0.5500 | 0.11045 | 0.19996 | 0.623 / 0.377 / 0.000 | 1051.1 / 1961.4 |
| turn|straight <0.02 | 2112 | 0.1703 | 0.09098 | 0.12972 | 0.642 / 0.358 / 0.000 | 122.5 / 359.6 |
| turn|gentle 0.02-0.06 | 678 | -0.1070 | 0.04909 | 0.10347 | 0.534 / 0.466 / 0.000 | 39.9 / -72.6 |
| turn|turn 0.06-0.15 | 442 | 0.6398 | 0.33087 | 0.60070 | 0.633 / 0.367 / 0.000 | 69.0 / 282.8 |
| turn|sharp >=0.15 | 312 | 4.5092 | 1.68441 | 2.90109 | 0.673 / 0.327 / 0.000 | 228.2 / 1406.9 |
| turn|undef (GT crawling) | 22 | -0.6945 | 0.02016 | -0.07831 | 0.591 / 0.409 / 0.000 | 0.9 / -15.3 |
| speed|v0 <2 | 228 | 1.2405 | 0.29979 | 0.50663 | 0.706 / 0.294 / 0.000 | 111.9 / 282.8 |
| speed|v0 2-5 | 422 | 0.8490 | 0.31561 | 0.35721 | 0.673 / 0.327 / 0.000 | 239.4 / 358.3 |
| speed|v0 5-10 | 1040 | 0.6809 | 0.16305 | 0.22989 | 0.595 / 0.405 / 0.000 | 277.0 / 708.1 |
| speed|v0 10-15 | 952 | 0.4816 | 0.13341 | 0.20676 | 0.666 / 0.334 / 0.000 | 169.8 / 458.5 |
| speed|v0 15-25 | 679 | 0.2294 | 0.06110 | 0.11237 | 0.589 / 0.411 / 0.000 | 46.3 / 155.8 |
| speed|v0 >=25 | 245 | -0.0083 | 0.00484 | -0.01481 | 0.502 / 0.498 / 0.000 | 5.8 / -2.0 |
| nav|nav=follow | 2248 | 0.4806 | 0.09591 | 0.17776 | 0.624 / 0.376 / 0.000 | 467.6 / 1080.4 |
| nav|nav=left | 335 | 1.1935 | 0.20606 | 0.34476 | 0.663 / 0.337 / 0.000 | 203.7 / 399.8 |
| nav|nav=right | 983 | 0.4895 | 0.15393 | 0.21111 | 0.606 / 0.394 / 0.000 | 313.4 / 481.2 |
**R7-15k - R7-30k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3541 | 0.3580 | 0.06341 | 0.12543 | 0.576 / 0.424 / 0.000 | 833.9 / 1267.9 |
| turn|straight <0.02 | 2094 | 0.1301 | 0.05703 | 0.09336 | 0.593 / 0.407 / 0.000 | 89.9 / 272.4 |
| turn|gentle 0.02-0.06 | 675 | -0.1519 | 0.00271 | -0.00920 | 0.504 / 0.496 / 0.000 | 37.4 / -102.5 |
| turn|turn 0.06-0.15 | 441 | 0.2895 | 0.15392 | 0.33706 | 0.565 / 0.435 / 0.000 | 65.1 / 127.7 |
| turn|sharp >=0.15 | 312 | 3.2183 | 1.46961 | 2.18607 | 0.647 / 0.353 / 0.000 | 210.3 / 1004.1 |
| turn|undef (GT crawling) | 19 | -1.7773 | -0.04078 | -0.23797 | 0.368 / 0.632 / 0.000 | 0.9 / -33.8 |
| speed|v0 <2 | 203 | 1.0363 | 0.19859 | 0.34300 | 0.640 / 0.360 / 0.000 | 103.1 / 210.4 |
| speed|v0 2-5 | 422 | 0.5584 | 0.18599 | 0.23208 | 0.616 / 0.384 / 0.000 | 208.6 / 235.6 |
| speed|v0 5-10 | 1040 | 0.3832 | 0.08976 | 0.13852 | 0.583 / 0.417 / 0.000 | 196.0 / 398.5 |
| speed|v0 10-15 | 952 | 0.3443 | 0.05381 | 0.11748 | 0.575 / 0.425 / 0.000 | 141.6 / 327.8 |
| speed|v0 15-25 | 679 | 0.1473 | 0.04805 | 0.09491 | 0.566 / 0.434 / 0.000 | 42.7 / 100.0 |
| speed|v0 >=25 | 245 | -0.0181 | -0.00897 | -0.00289 | 0.461 / 0.539 / 0.000 | 4.3 / -4.4 |
| nav|nav=follow | 2238 | 0.3473 | 0.06810 | 0.13353 | 0.588 / 0.412 / 0.000 | 385.9 / 777.2 |
| nav|nav=left | 327 | 0.7749 | 0.06901 | 0.14746 | 0.581 / 0.419 / 0.000 | 180.5 / 253.4 |
| nav|nav=right | 976 | 0.2431 | 0.05164 | 0.09191 | 0.548 / 0.452 / 0.000 | 224.2 / 237.3 |
**R7-30k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3549 | 0.1908 | 0.04782 | 0.06696 | 0.581 / 0.419 / 0.000 | 624.9 / 677.1 |
| turn|straight <0.02 | 2100 | 0.0362 | 0.03351 | 0.03370 | 0.575 / 0.425 / 0.000 | 66.0 / 76.1 |
| turn|gentle 0.02-0.06 | 675 | 0.0331 | 0.07457 | 0.09536 | 0.587 / 0.413 / 0.000 | 34.5 / 22.3 |
| turn|turn 0.06-0.15 | 441 | 0.3563 | 0.15301 | 0.20016 | 0.587 / 0.413 / 0.000 | 55.0 / 157.1 |
| turn|sharp >=0.15 | 312 | 1.2909 | 0.21509 | 0.56419 | 0.593 / 0.407 / 0.000 | 134.1 / 402.8 |
| turn|undef (GT crawling) | 21 | 0.8938 | 0.07032 | 0.21360 | 0.714 / 0.286 / 0.000 | 14.6 / 18.8 |
| speed|v0 <2 | 211 | 0.2654 | 0.08221 | 0.16092 | 0.578 / 0.422 / 0.000 | 29.5 / 56.0 |
| speed|v0 2-5 | 422 | 0.2907 | 0.08953 | 0.11149 | 0.585 / 0.415 / 0.000 | 118.8 / 122.7 |
| speed|v0 5-10 | 1040 | 0.2977 | 0.07701 | 0.09054 | 0.565 / 0.435 / 0.000 | 289.3 / 309.6 |
| speed|v0 10-15 | 952 | 0.1373 | 0.06305 | 0.07700 | 0.618 / 0.382 / 0.000 | 80.6 / 130.7 |
| speed|v0 15-25 | 679 | 0.0821 | 0.02171 | 0.02008 | 0.564 / 0.436 / 0.000 | 39.1 / 55.8 |
| speed|v0 >=25 | 245 | 0.0098 | 0.01260 | 0.00239 | 0.547 / 0.453 / 0.000 | 4.4 / 2.4 |
| nav|nav=follow | 2246 | 0.1327 | 0.03370 | 0.03982 | 0.569 / 0.431 / 0.000 | 286.3 / 298.1 |
| nav|nav=left | 327 | 0.4077 | 0.09678 | 0.19025 | 0.602 / 0.398 / 0.000 | 50.4 / 133.3 |
| nav|nav=right | 976 | 0.2518 | 0.08841 | 0.10862 | 0.602 / 0.398 / 0.000 | 247.9 / 245.7 |
**R7-15k - R7-20k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3551 | 0.0868 | 0.01575 | 0.00616 | 0.519 / 0.481 / 0.000 | 802.9 / 308.3 |
| turn|straight <0.02 | 2098 | -0.0716 | 0.01339 | 0.00018 | 0.518 / 0.482 / 0.000 | 71.4 / -150.3 |
| turn|gentle 0.02-0.06 | 678 | -0.2016 | -0.01959 | -0.06344 | 0.479 / 0.521 / 0.000 | 45.0 / -136.7 |
| turn|turn 0.06-0.15 | 442 | -0.1941 | 0.04774 | 0.00096 | 0.511 / 0.489 / 0.000 | 54.8 / -85.8 |
| turn|sharp >=0.15 | 314 | 2.2061 | 0.67105 | 1.21469 | 0.624 / 0.376 / 0.000 | 203.5 / 692.7 |
| turn|undef (GT crawling) | 19 | -0.6108 | -0.00919 | -0.24970 | 0.421 / 0.579 / 0.000 | 0.5 / -11.6 |
| speed|v0 <2 | 213 | 0.3189 | -0.20767 | -0.20563 | 0.423 / 0.577 / 0.000 | 112.0 / 67.9 |
| speed|v0 2-5 | 422 | -0.2988 | -0.19249 | -0.21500 | 0.419 / 0.581 / 0.000 | 178.5 / -126.1 |
| speed|v0 5-10 | 1040 | 0.1329 | -0.00542 | 0.00298 | 0.498 / 0.502 / 0.000 | 217.9 / 138.2 |
| speed|v0 10-15 | 952 | 0.1876 | 0.05157 | 0.06212 | 0.557 / 0.443 / 0.000 | 128.6 / 178.6 |
| speed|v0 15-25 | 679 | 0.0553 | 0.04104 | 0.07295 | 0.566 / 0.434 / 0.000 | 30.0 / 37.5 |
| speed|v0 >=25 | 245 | 0.0497 | 0.02312 | 0.02128 | 0.584 / 0.416 / 0.000 | 5.9 / 12.2 |
| nav|nav=follow | 2236 | 0.1373 | 0.02756 | 0.03813 | 0.542 / 0.458 / 0.000 | 338.7 / 307.1 |
| nav|nav=left | 335 | 0.6805 | 0.06031 | 0.18067 | 0.570 / 0.430 / 0.000 | 194.3 / 228.0 |
| nav|nav=right | 980 | -0.2314 | -0.09402 | -0.16293 | 0.448 / 0.552 / 0.000 | 193.2 / -226.7 |
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

**R7-15k - refcv7-50400** (sum 1961.4 over 139 episodes)

* ep016: n 22, share 0.141, mean delta 12.5467, sharp-window frac 0.41, curved frac 0.59, mean v0 5.6, nav {'nav=follow': 0, 'nav=left': 22, 'nav=right': 0}
* ep085: n 26, share 0.091, mean delta 6.8567, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep028: n 27, share 0.069, mean delta 5.0322, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep061: n 26, share 0.060, mean delta 4.5602, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep111: n 26, share 0.053, mean delta 3.9969, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep013: n 27, share 0.051, mean delta 3.7104, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
**R7-15k - R7-30k** (sum 1267.9 over 139 episodes)

* ep016: n 22, share 0.173, mean delta 9.9759, sharp-window frac 0.41, curved frac 0.59, mean v0 5.6, nav {'nav=follow': 0, 'nav=left': 22, 'nav=right': 0}
* ep085: n 26, share 0.097, mean delta 4.7109, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep110: n 26, share 0.066, mean delta 3.2182, sharp-window frac 0.46, curved frac 0.85, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep028: n 27, share 0.057, mean delta 2.6661, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep013: n 27, share 0.055, mean delta 2.6031, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep061: n 26, share 0.055, mean delta 2.6847, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
**R7-30k - refcv7-50400** (sum 677.1 over 139 episodes)

* ep111: n 26, share 0.189, mean delta 4.9261, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.183, mean delta 4.7761, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep028: n 27, share 0.094, mean delta 2.3661, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep016: n 22, share 0.084, mean delta 2.5708, sharp-window frac 0.41, curved frac 0.59, mean v0 5.6, nav {'nav=follow': 0, 'nav=left': 22, 'nav=right': 0}
* ep085: n 26, share 0.082, mean delta 2.1458, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep061: n 26, share 0.072, mean delta 1.8755, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
**R7-15k - R7-20k** (sum 308.3 over 139 episodes)

* ep016: n 23, share 0.920, mean delta 12.3285, sharp-window frac 0.43, curved frac 0.61, mean v0 5.3, nav {'nav=follow': 0, 'nav=left': 23, 'nav=right': 0}
* ep085: n 26, share 0.258, mean delta 3.0626, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep028: n 27, share 0.255, mean delta 2.9083, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep061: n 26, share 0.224, mean delta 2.6598, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep012: n 26, share 0.215, mean delta 2.5555, sharp-window frac 0.46, curved frac 0.77, mean v0 7.0, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep107: n 26, share 0.190, mean delta 2.2495, sharp-window frac 0.62, curved frac 1.00, mean v0 13.7, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
**R7-20k - refcv7-50400** (sum 1693.4 over 139 episodes)

* ep085: n 26, share 0.058, mean delta 3.7941, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep100: n 26, share 0.056, mean delta 3.6358, sharp-window frac 0.35, curved frac 0.85, mean v0 3.6, nav {'nav=follow': 0, 'nav=left': 26, 'nav=right': 0}
* ep013: n 27, share 0.055, mean delta 3.4561, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep041: n 26, share 0.054, mean delta 3.5224, sharp-window frac 0.12, curved frac 0.38, mean v0 8.6, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.051, mean delta 3.3460, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep067: n 26, share 0.045, mean delta 2.9568, sharp-window frac 0.35, curved frac 0.96, mean v0 4.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**R7-15k vs R7-30k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1706 / 0.1320 [0.0864, 0.1828] | 34 / 1.0667 [0.3953, 1.8606] |
| only new agrees | 193 / -0.8789 [-1.6139, -0.2051] | 22 / -3.3974 [-7.7758, 1.2106] |
| only earlier agrees | 271 / 2.1133 [1.3367, 2.9846] | 40 / 8.9213 [5.2465, 13.7472] |
| neither agrees | 1371 / 0.4628 [0.0334, 0.9689] | 216 / 3.3121 [1.2272, 5.8848] |
**R7-15k vs R7-20k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1657 / 0.0249 [-0.0305, 0.0776] | 36 / 0.9211 [0.3364, 2.0568] |
| only new agrees | 248 / -1.3853 [-2.1676, -0.7531] | 20 / -6.5237 [-13.0102, -1.9276] |
| only earlier agrees | 233 / 1.3483 [0.7009, 2.0051] | 39 / 5.4041 [2.6006, 8.8158] |
| neither agrees | 1415 / 0.2518 [-0.2010, 0.8345] | 219 / 3.0230 [0.4999, 6.2501] |
**R7-15k vs R7-30k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1711 / 0.1462 [0.0993, 0.2003] | 41 / 1.2939 [0.4289, 2.2576] |
| only new agrees | 198 / -1.1437 [-1.9492, -0.5020] | 25 / -4.5614 [-9.5727, -0.8239] |
| only earlier agrees | 274 / 1.9442 [1.2307, 2.6507] | 40 / 7.1034 [4.3184, 11.0656] |
| neither agrees | 1360 / 0.5289 [0.1143, 1.0265] | 206 / 3.6469 [1.5312, 6.2855] |
**R7-15k vs R7-20k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1656 / 0.0323 [-0.0304, 0.0934] | 42 / 1.3640 [0.5789, 2.7321] |
| only new agrees | 257 / -1.7406 [-2.5080, -1.0845] | 24 / -8.8271 [-15.3738, -4.1022] |
| only earlier agrees | 231 / 1.5510 [0.9087, 2.1828] | 33 / 6.0987 [3.4072, 9.9391] |
| neither agrees | 1408 / 0.1880 [-0.1789, 0.6474] | 215 / 2.6197 [0.5614, 5.1263] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**R7-15k vs R7-30k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3527 | 0.1371 [0.0004, 0.2890] | 0.2684 [-0.0013, 0.6224] | -0.1356 [-0.4654, 0.1061] |
| turn|straight <0.02 | 2090 | -0.0618 [-0.1170, -0.0192] | -0.0116 [-0.0377, 0.0146] | -0.0414 [-0.0882, -0.0026] SEP |
| turn|gentle 0.02-0.06 | 675 | -0.1127 [-0.2438, 0.0024] | -0.3040 [-0.7972, 0.0186] | 0.1627 [-0.1103, 0.5709] |
| turn|turn 0.06-0.15 | 437 | -0.1208 [-0.4650, 0.2215] | 0.1798 [-0.2220, 0.6523] | -0.3042 [-0.7288, 0.0994] |
| turn|sharp >=0.15 | 309 | 2.4018 [1.1364, 3.9775] | 3.5454 [1.2035, 6.9651] | -1.1845 [-4.5810, 1.1762] |
| turn|undef (GT crawling) | 16 | -0.0902 [-0.6115, 0.3636] | 0.0560 [-0.2548, 0.3435] | -0.1582 [-0.8401, 0.4837] |
| speed|v0 <2 | 189 | 0.4735 [-0.0257, 1.1232] | 0.5921 [-0.0256, 1.2883] | -0.1377 [-0.5677, 0.2646] |
| speed|v0 2-5 | 422 | 0.1310 [-0.4453, 0.7842] | 1.1090 [-0.5204, 3.3488] | -0.9780 [-3.1640, 0.6901] |
| speed|v0 5-10 | 1040 | 0.3403 [-0.0556, 0.7960] | 0.3891 [0.0293, 0.8012] | -0.0488 [-0.3596, 0.2697] |
| speed|v0 10-15 | 952 | -0.0435 [-0.1644, 0.0865] | -0.0485 [-0.1090, 0.0053] | 0.0050 [-0.1069, 0.1256] |
| speed|v0 15-25 | 679 | 0.0007 [-0.0521, 0.0591] | 0.0007 [-0.0284, 0.0271] | -0.0000 [-0.0616, 0.0731] |
| speed|v0 >=25 | 245 | 0.0465 [0.0124, 0.0901] | 0.0198 [-0.0076, 0.0472] | 0.0267 [-0.0103, 0.0801] |
**R7-15k vs R7-20k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3542 | 0.1371 [0.0004, 0.2890] | 0.2188 [-0.0458, 0.5550] | -0.0717 [-0.3497, 0.1498] |
| turn|straight <0.02 | 2093 | -0.0618 [-0.1170, -0.0192] | -0.1226 [-0.2141, -0.0514] | 0.0689 [-0.0065, 0.1466] |
| turn|gentle 0.02-0.06 | 678 | -0.1127 [-0.2438, 0.0024] | -0.2909 [-0.6428, 0.0101] | 0.1816 [-0.0520, 0.4445] |
| turn|turn 0.06-0.15 | 439 | -0.1208 [-0.4650, 0.2215] | -0.1136 [-0.5918, 0.3094] | 0.0179 [-0.3840, 0.4302] |
| turn|sharp >=0.15 | 313 | 2.4018 [1.1364, 3.9775] | 4.1074 [1.6804, 7.1485] | -1.7147 [-4.2377, 0.4659] |
| turn|undef (GT crawling) | 19 | -0.0902 [-0.6115, 0.3636] | -0.3843 [-0.8422, 0.1214] | 0.4011 [-0.5769, 1.2874] |
| speed|v0 <2 | 204 | 0.4735 [-0.0257, 1.1232] | -0.0013 [-0.8575, 0.8521] | 0.7051 [0.1455, 1.5412] SEP |
| speed|v0 2-5 | 422 | 0.1310 [-0.4453, 0.7842] | 0.5277 [-0.6093, 1.9391] | -0.3967 [-1.6684, 0.6196] |
| speed|v0 5-10 | 1040 | 0.3403 [-0.0556, 0.7960] | 0.5521 [-0.0683, 1.4855] | -0.2118 [-0.8433, 0.2503] |
| speed|v0 10-15 | 952 | -0.0435 [-0.1644, 0.0865] | 0.0021 [-0.0834, 0.0813] | -0.0455 [-0.1606, 0.0673] |
| speed|v0 15-25 | 679 | 0.0007 [-0.0521, 0.0591] | -0.0380 [-0.1045, 0.0137] | 0.0387 [-0.0183, 0.1081] |
| speed|v0 >=25 | 245 | 0.0465 [0.0124, 0.0901] | 0.0175 [-0.0140, 0.0529] | 0.0291 [0.0111, 0.0497] SEP |
**R7-15k vs R7-30k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
**R7-15k vs R7-20k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* R7-15k vs R7-30k|seed0: anchor flip rate 0.249; heading delta on flipped windows 1.0882 [0.3501, 1.7985] (n 863) vs unflipped 0.1209 [0.0735, 0.1693]; goal_dist_sel proxy new-earlier 0.2924 [-0.0148, 0.5974]; corr(window dHeading, window dGoalDist) -0.024
* R7-15k vs R7-20k|seed0: anchor flip rate 0.245; heading delta on flipped windows 0.3300 [-0.4181, 1.2322] (n 857) vs unflipped 0.0316 [-0.0326, 0.0924]; goal_dist_sel proxy new-earlier 0.0975 [-0.1686, 0.3620]; corr(window dHeading, window dGoalDist) -0.010
* R7-15k vs R7-30k|seed1: anchor flip rate 0.247; heading delta on flipped windows 1.0735 [0.4021, 1.7637] (n 856) vs unflipped 0.1328 [0.0828, 0.1873]; goal_dist_sel proxy new-earlier 0.2946 [-0.0104, 0.5949]; corr(window dHeading, window dGoalDist) -0.016
* R7-15k vs R7-20k|seed1: anchor flip rate 0.241; heading delta on flipped windows 0.1746 [-0.4721, 0.9139] (n 840) vs unflipped 0.0304 [-0.0395, 0.1030]; goal_dist_sel proxy new-earlier 0.0975 [-0.1652, 0.3559]; corr(window dHeading, window dGoalDist) -0.002
