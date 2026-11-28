# refcv8 heading-drift localisation: R7-30k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-18T08:39:38; trees: `R7-30k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step30000_unc; `R7-20k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step20000_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **None**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

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

### S2 -- R7-30k - R7-20k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4517 | 139 | 1.4666 | 1.4863 | -0.0198 [-0.0637, 0.0265] |  |  |
| turn: straight <0.02 | 2895 | 132 | 1.1931 | 1.1946 | -0.0015 [-0.0231, 0.0180] |  | 0.641 / 0.048 / 0.0509 [-0.1334, 0.2215] |
| turn: gentle 0.02-0.06 | 721 | 98 | 1.0278 | 0.9847 | 0.0431 [-0.0166, 0.1235] |  | 0.160 / -0.348 / 0.0748 [-0.0416, 0.2171] |
| turn: turn 0.06-0.15 | 512 | 72 | 2.0092 | 2.2085 | -0.1993 [-0.3615, -0.0663] | **SEP** | 0.113 / 1.143 / -0.2025 [-0.4378, 0.0147] |
| turn: sharp >=0.15 | 368 | 42 | 3.2871 | 3.3254 | -0.0383 [-0.4297, 0.3994] |  | 0.081 / 0.158 / -0.0202 [-0.6233, 0.6635] |
| turn: undef (GT crawling) (low n) | 21 | 17 | 9.0979 | 9.0951 | 0.0028 [-0.2110, 0.2668] |  | 0.005 / -0.001 / 0.0227 [-0.2765, 0.4436] |
| speed: v0 <2 | 228 | 38 | 12.9709 | 13.0751 | -0.1042 [-0.3028, 0.0628] |  | 0.050 / 0.266 / -0.0889 [-0.4332, 0.1476] |
| speed: v0 2-5 | 516 | 58 | 1.9548 | 2.0874 | -0.1326 [-0.4203, 0.1521] |  | 0.114 / 0.766 / -0.1274 [-0.5773, 0.3124] |
| speed: v0 5-10 | 1390 | 84 | 1.1066 | 1.1092 | -0.0026 [-0.0724, 0.0620] |  | 0.308 / 0.040 / 0.0249 [-0.0992, 0.1508] |
| speed: v0 10-15 | 1192 | 64 | 0.6061 | 0.6040 | 0.0020 [-0.0501, 0.0499] |  | 0.264 / -0.027 / 0.0296 [-0.0833, 0.1343] |
| speed: v0 15-25 | 856 | 35 | 0.3950 | 0.3964 | -0.0014 [-0.0307, 0.0245] |  | 0.190 / 0.014 / 0.0226 [-0.0642, 0.1070] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1784 | 0.1627 | 0.0157 [-0.0086, 0.0375] |  | 0.074 / -0.059 / 0.0384 [-0.0491, 0.1138] |
| nav: nav=follow | 2869 | 88 | 0.6380 | 0.6511 | -0.0131 [-0.0535, 0.0229] |  | 0.635 / 0.422 / 0.0182 [-0.1689, 0.1748] |
| nav: nav=left (low n) | 412 | 13 | 7.3757 | 7.2617 | 0.1140 [-0.1116, 0.4150] |  | 0.091 / -0.526 / 0.1472 [-0.1907, 0.6554] |
| nav: nav=right | 1236 | 38 | 1.4202 | 1.4999 | -0.0798 [-0.1820, 0.0257] |  | 0.274 / 1.104 / -0.0826 [-0.2510, 0.0814] |

episode concentration: 0.511 of episodes have mean delta > 0; top-5 episodes carry -1.307 of the summed delta; delta excluding the top 5 = -0.0474 (overall -0.0198).

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

### S2 -- R7-30k - R7-20k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4484 | 139 | 0.5348 | 0.6685 | -0.1337 [-0.3837, -0.0014] | **SEP** | -0.0601 [-0.3109, 0.0774] |
| seg1 | straight | 2881 | 132 | 0.4621 | 0.5836 | -0.1215 [-0.4013, 0.0091] |  | -0.2066 [-0.7678, 0.0534] |
| seg1 | curved | 1588 | 101 | 0.6680 | 0.7117 | -0.0437 [-0.0786, -0.0064] | **SEP** | 0.0919 [0.0308, 0.1466] SEP |
| seg2 | all | 4477 | 139 | 0.9108 | 1.1051 | -0.1943 [-0.5341, -0.0033] | **SEP** | -0.1215 [-0.6268, 0.1615] |
| seg2 | straight | 2884 | 132 | 0.5593 | 0.8174 | -0.2581 [-0.8198, 0.0051] |  | -0.2721 [-1.1183, 0.1296] |
| seg2 | curved | 1593 | 101 | 1.5471 | 1.6259 | -0.0788 [-0.1698, 0.0262] |  | 0.1512 [-0.0038, 0.2909] |
| seg3 | all | 4463 | 139 | 1.2642 | 1.2916 | -0.0274 [-0.0865, 0.0310] |  | -0.0211 [-0.3761, 0.1959] |
| seg3 | straight | 2874 | 132 | 0.8067 | 0.8078 | -0.0012 [-0.0330, 0.0253] |  | -0.1056 [-0.6881, 0.1799] |
| seg3 | curved | 1589 | 101 | 2.0916 | 2.1665 | -0.0749 [-0.2200, 0.0783] |  | 0.1319 [-0.0565, 0.3123] |
| seg4 | all | 4447 | 139 | 1.6583 | 1.7727 | -0.1144 [-0.3943, 0.0732] |  | 0.3016 [0.1664, 0.4216] SEP |
| seg4 | straight | 2860 | 131 | 1.0151 | 1.1930 | -0.1779 [-0.5990, 0.0363] |  | 0.2397 [0.0756, 0.3476] SEP |
| seg4 | curved | 1581 | 101 | 2.8251 | 2.8250 | 0.0001 [-0.2051, 0.2273] |  | 0.4152 [0.1533, 0.6699] SEP |

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

### S2 -- R7-30k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 439 | 47 | 6.0104 | 5.9676 | 0.0428 [-0.0242, 0.1051] |  |  |
| straight <0.02 | v0 5-15 | 1519 | 93 | 0.4073 | 0.3674 | 0.0398 [0.0205, 0.0586] | **SEP** |  |
| straight <0.02 | v0 >=15 | 940 | 43 | 0.2100 | 0.1947 | 0.0153 [0.0033, 0.0272] | **SEP** |  |
| gentle 0.02-0.06 | v0 <5 | 112 | 29 | 1.6824 | 1.7630 | -0.0806 [-0.2443, 0.0378] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 461 | 74 | 1.0117 | 0.9711 | 0.0406 [0.0037, 0.0820] | **SEP** |  |
| gentle 0.02-0.06 | v0 >=15 | 148 | 20 | 0.5824 | 0.5590 | 0.0234 [-0.0286, 0.0752] |  |  |
| turn 0.06-0.15 | v0 <5 (low n) | 82 | 25 | 4.9189 | 5.0600 | -0.1411 [-0.5448, 0.2838] |  |  |
| turn 0.06-0.15 | v0 5-15 | 343 | 54 | 1.5480 | 1.4275 | 0.1205 [0.0108, 0.2617] | **SEP** |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 87 | 11 | 1.0850 | 0.9719 | 0.1131 [-0.0981, 0.3378] |  |  |
| sharp >=0.15 | v0 <5 (low n) | 93 | 16 | 5.8626 | 5.2944 | 0.5683 [0.0762, 1.0742] | **SEP** |  |
| sharp >=0.15 | v0 5-15 | 259 | 33 | 2.4887 | 2.2857 | 0.2030 [-0.1004, 0.5382] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 16 | 3 | 1.2418 | 1.3866 | -0.1448 [-0.3249, 0.2320] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 28 | 22 | 13.2368 | 13.3146 | -0.0778 [-0.2004, 0.0202] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S2 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | R7-30k s0-s1 [CI] | R7-20k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0060 [-0.0196, 0.0079] | -0.0015 [-0.0157, 0.0150] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | -0.0050 [-0.0158, 0.0047] | 0.0079 [-0.0056, 0.0266] | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | 0.0210 [-0.0180, 0.0668] | -0.0271 [-0.0663, 0.0095] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | 0.0090 [-0.0310, 0.0579] | -0.0187 [-0.0706, 0.0340] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | -0.0870 [-0.1984, 0.0173] | -0.0092 [-0.1307, 0.1113] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | -0.0060 [-0.0909, 0.0632] | 0.1339 [-0.0506, 0.3279] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | 0.0226 [-0.0617, 0.1081] | 0.0129 [-0.1193, 0.1887] | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | -0.0262 [-0.1192, 0.0801] | 0.0189 [-0.0664, 0.1076] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | -0.0190 [-0.0394, 0.0006] | -0.0115 [-0.0432, 0.0284] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | 0.0070 [-0.0025, 0.0174] | 0.0005 [-0.0131, 0.0140] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | -0.0017 [-0.0088, 0.0049] | -0.0017 [-0.0070, 0.0041] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | 0.0017 [-0.0086, 0.0125] | -0.0075 [-0.0163, -0.0008] SEP | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | 0.0003 [-0.0067, 0.0069] | -0.0062 [-0.0176, 0.0054] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | -0.0076 [-0.0615, 0.0391] | -0.0472 [-0.1021, -0.0024] SEP | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | -0.0201 [-0.0627, 0.0255] | 0.0250 [-0.0162, 0.0755] | 0.0283 [0.0035, 0.0597] SEP |

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

### S2 -- R7-30k - R7-20k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.0198 | -0.0217 / -0.0180 | False / False / False | 0.0060 / 0.0015 | 3.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | -0.0015 | -0.0074 / 0.0046 | False / False / False | 0.0050 / 0.0079 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.0431 | 0.0671 / 0.0190 | False / False / False | 0.0210 / 0.0271 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.1993 | -0.1857 / -0.2129 | True / True / True | 0.0090 / 0.0187 | 10.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | -0.0383 | -0.0772 / 0.0006 | False / False / False | 0.0870 / 0.0092 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.0028 | -0.0667 / 0.0171 | False / False / False | 0.0060 / 0.1339 | 0.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.1042 | -0.0908 / -0.1194 | False / False / False | 0.0226 / 0.0129 | 4.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.1326 | -0.1552 / -0.1101 | False / False / False | 0.0262 / 0.0189 | 5.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | -0.0026 | -0.0063 / 0.0012 | False / False / False | 0.0190 / 0.0115 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.0020 | 0.0053 / -0.0012 | False / False / False | 0.0070 / 0.0005 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0014 | -0.0014 / -0.0015 | False / False / False | 0.0017 / 0.0017 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0157 | 0.0203 / 0.0111 | False / False / False | 0.0017 / 0.0075 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | -0.0131 | -0.0099 / -0.0163 | False / False / False | 0.0003 / 0.0062 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | 0.1140 | 0.1370 / 0.0910 | False / False / False | 0.0076 / 0.0472 | 2.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.0798 | -0.1020 / -0.0584 | False / False / False | 0.0201 / 0.0250 | 3.2 | NOT PROVEN (not separated and/or within 2x floor) |

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

* seg1 (n 1588): R7-30k bias +0.046, refcv7-50400 bias +0.081, R7-20k bias +0.137; R7-30k MAE 0.668, refcv7-50400 MAE 0.623, R7-20k MAE 0.712
* seg2 (n 1592): R7-30k bias -0.190, refcv7-50400 bias -0.100, R7-20k bias -0.008; R7-30k MAE 1.548, refcv7-50400 MAE 1.449, R7-20k MAE 1.627
* seg3 (n 1589): R7-30k bias -0.808, refcv7-50400 bias -0.622, R7-20k bias -0.666; R7-30k MAE 2.092, refcv7-50400 MAE 1.986, R7-20k MAE 2.166
* seg4 (n 1581): R7-30k bias -1.144, refcv7-50400 bias -0.730, R7-20k bias -0.902; R7-30k MAE 2.825, refcv7-50400 MAE 2.691, R7-20k MAE 2.825
**sharp (>= 0.15)**

* seg1 (n 366): R7-30k bias -0.123, refcv7-50400 bias +0.010, R7-20k bias +0.284; R7-30k MAE 0.994, refcv7-50400 MAE 0.913, R7-20k MAE 1.104
* seg2 (n 366): R7-30k bias -0.822, refcv7-50400 bias -0.528, R7-20k bias +0.086; R7-30k MAE 2.645, refcv7-50400 MAE 2.413, R7-20k MAE 2.812
* seg3 (n 366): R7-30k bias -2.418, refcv7-50400 bias -1.862, R7-20k bias -1.596; R7-30k MAE 3.959, refcv7-50400 MAE 3.643, R7-20k MAE 4.029
* seg4 (n 365): R7-30k bias -3.191, refcv7-50400 bias -1.860, R7-20k bias -2.039; R7-30k MAE 5.334, refcv7-50400 MAE 4.893, R7-20k MAE 5.165
**curved & v0 5-10**

* seg1 (n 627): R7-30k bias -0.061, refcv7-50400 bias -0.038, R7-20k bias +0.030; R7-30k MAE 0.598, refcv7-50400 MAE 0.535, R7-20k MAE 0.661
* seg2 (n 627): R7-30k bias -0.325, refcv7-50400 bias -0.282, R7-20k bias -0.169; R7-30k MAE 1.525, refcv7-50400 MAE 1.363, R7-20k MAE 1.598
* seg3 (n 627): R7-30k bias -0.963, refcv7-50400 bias -0.871, R7-20k bias -0.896; R7-30k MAE 2.199, refcv7-50400 MAE 2.019, R7-20k MAE 2.199
* seg4 (n 627): R7-30k bias -1.457, refcv7-50400 bias -1.194, R7-20k bias -1.283; R7-30k MAE 2.978, refcv7-50400 MAE 2.730, R7-20k MAE 2.839
**sharp & v0 5-10**

* seg1 (n 164): R7-30k bias +0.060, refcv7-50400 bias +0.103, R7-20k bias +0.482; R7-30k MAE 0.971, refcv7-50400 MAE 0.857, R7-20k MAE 1.100
* seg2 (n 164): R7-30k bias -0.444, refcv7-50400 bias -0.422, R7-20k bias +0.462; R7-30k MAE 2.557, refcv7-50400 MAE 2.243, R7-20k MAE 2.699
* seg3 (n 164): R7-30k bias -2.198, refcv7-50400 bias -2.032, R7-20k bias -1.436; R7-30k MAE 3.740, refcv7-50400 MAE 3.465, R7-20k MAE 3.740
* seg4 (n 164): R7-30k bias -3.228, refcv7-50400 bias -2.403, R7-20k bias -2.138; R7-30k MAE 5.151, refcv7-50400 MAE 4.635, R7-20k MAE 4.837

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1581 / 1581 | 10.2 | R7-30k -0.1298 [-0.2023, -0.0675]; refcv7-50400 -0.0852 [-0.1565, -0.0286]; R7-20k -0.1002 [-0.1827, -0.0386] | R7-30k -0.0445 [-0.0652, -0.0226] SEP; R7-20k -0.0149 [-0.0488, +0.0181] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1600 / 4762 | 7.2 | R7-30k -0.1808 [-0.3003, -0.0847]; refcv7-50400 -0.1471 [-0.2660, -0.0526]; R7-20k -0.1531 [-0.2796, -0.0500] | R7-30k -0.0337 [-0.0539, -0.0136] SEP; R7-20k -0.0060 [-0.0374, +0.0261] |
| sharp (>= 0.15) | last segment | 365 / 365 | 25.3 | R7-30k -0.1438 [-0.2292, -0.0710]; refcv7-50400 -0.0926 [-0.1766, -0.0262]; R7-20k -0.1064 [-0.2043, -0.0353] | R7-30k -0.0512 [-0.0759, -0.0268] SEP; R7-20k -0.0138 [-0.0530, +0.0249] |
| sharp (>= 0.15) | segments 2..K | 368 / 1097 | 17.9 | R7-30k -0.1457 [-0.2337, -0.0702]; refcv7-50400 -0.1044 [-0.1876, -0.0369]; R7-20k -0.1064 [-0.2063, -0.0321] | R7-30k -0.0413 [-0.0649, -0.0171] SEP; R7-20k -0.0020 [-0.0421, +0.0386] |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | R7-30k -0.1170 [-0.1981, -0.0539]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; R7-20k -0.0925 [-0.1693, -0.0394] | R7-30k -0.0245 [-0.0621, +0.0105]; R7-20k -0.0000 [-0.0233, +0.0247] |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | R7-30k -0.1147 [-0.1962, -0.0510]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; R7-20k -0.0932 [-0.1722, -0.0382] | R7-30k -0.0154 [-0.0515, +0.0188]; R7-20k +0.0061 [-0.0160, +0.0286] |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | R7-30k -0.1195 [-0.2080, -0.0502]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; R7-20k -0.0853 [-0.1691, -0.0256] | R7-30k -0.0282 [-0.0696, +0.0083]; R7-20k +0.0059 [-0.0173, +0.0316] |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | R7-30k -0.1131 [-0.2023, -0.0424]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; R7-20k -0.0795 [-0.1657, -0.0181] | R7-30k -0.0176 [-0.0578, +0.0180]; R7-20k +0.0160 [-0.0068, +0.0412] |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

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
**R7-30k - R7-20k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4517 | -0.0198 | 0.00515 | -0.00264 | 0.512 / 0.488 / 0.000 | 218.9 / -89.3 |
| turn|straight <0.02 | 2895 | -0.0015 | 0.01315 | 0.00920 | 0.540 / 0.460 / 0.000 | 50.3 / -4.3 |
| turn|gentle 0.02-0.06 | 721 | 0.0431 | -0.00512 | 0.00513 | 0.488 / 0.512 / 0.000 | 36.8 / 31.0 |
| turn|turn 0.06-0.15 | 512 | -0.1993 | -0.05035 | -0.11505 | 0.414 / 0.586 / 0.000 | 29.1 / -102.0 |
| turn|sharp >=0.15 | 368 | -0.0383 | -0.07019 | -0.03816 | 0.478 / 0.522 / 0.000 | 34.4 / -14.1 |
| turn|undef (GT crawling) | 21 | 0.0028 | -0.08051 | -0.07325 | 0.381 / 0.619 / 0.000 | 1.6 / 0.1 |
| speed|v0 <2 | 228 | -0.1042 | -0.01810 | -0.04899 | 0.434 / 0.566 / 0.000 | 7.6 / -23.8 |
| speed|v0 2-5 | 516 | -0.1326 | -0.03466 | -0.04853 | 0.446 / 0.554 / 0.000 | 48.1 / -68.4 |
| speed|v0 5-10 | 1390 | -0.0026 | 0.00048 | -0.00118 | 0.500 / 0.500 / 0.000 | 72.7 / -3.6 |
| speed|v0 10-15 | 1192 | 0.0020 | 0.01323 | 0.00570 | 0.541 / 0.459 / 0.000 | 26.8 / 2.4 |
| speed|v0 15-25 | 856 | -0.0014 | 0.00997 | 0.00595 | 0.526 / 0.474 / 0.000 | 8.0 / -1.2 |
| speed|v0 >=25 | 335 | 0.0157 | 0.01969 | 0.01115 | 0.576 / 0.424 / 0.000 | 3.5 / 5.3 |
| nav|nav=follow | 2869 | -0.0131 | 0.00320 | -0.00297 | 0.509 / 0.491 / 0.000 | 86.4 / -37.7 |
| nav|nav=left | 412 | 0.1140 | 0.03047 | 0.03825 | 0.580 / 0.420 / 0.000 | 35.8 / 47.0 |
| nav|nav=right | 1236 | -0.0798 | -0.00335 | -0.02289 | 0.497 / 0.503 / 0.000 | 63.9 / -98.6 |
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

**R7-30k - refcv7-50400** (sum 247.4 over 139 episodes)

* ep111: n 34, share 0.121, mean delta 0.8803, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep067: n 34, share 0.064, mean delta 0.4648, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep035: n 34, share 0.063, mean delta 0.4600, sharp-window frac 0.03, curved frac 0.24, mean v0 13.4, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep028: n 34, share 0.056, mean delta 0.4047, sharp-window frac 0.21, curved frac 0.68, mean v0 5.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep080: n 34, share 0.054, mean delta 0.3912, sharp-window frac 0.18, curved frac 0.32, mean v0 9.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep013: n 34, share 0.050, mean delta 0.3664, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
**R7-30k - R7-20k** (sum -89.3 over 139 episodes)

* ep104: n 35, share -0.593, mean delta 1.5123, sharp-window frac 0.29, curved frac 0.31, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 35, 'nav=right': 0}
* ep078: n 34, share -0.264, mean delta 0.6925, sharp-window frac 0.00, curved frac 0.29, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep111: n 34, share -0.162, mean delta 0.4266, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep107: n 34, share -0.148, mean delta 0.3883, sharp-window frac 0.65, curved frac 0.91, mean v0 13.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep012: n 34, share -0.141, mean delta 0.3696, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep137: n 32, share -0.120, mean delta 0.3343, sharp-window frac 0.41, curved frac 0.88, mean v0 3.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 32}
**R7-20k - refcv7-50400** (sum 338.0 over 139 episodes)

* ep067: n 34, share 0.140, mean delta 1.3928, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep041: n 34, share 0.101, mean delta 1.0037, sharp-window frac 0.18, curved frac 0.47, mean v0 9.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.088, mean delta 0.8721, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep082: n 31, share 0.076, mean delta 0.8308, sharp-window frac 0.00, curved frac 0.45, mean v0 8.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 31}
* ep124: n 34, share 0.056, mean delta 0.5595, sharp-window frac 0.32, curved frac 0.88, mean v0 9.7, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep129: n 31, share 0.053, mean delta 0.5812, sharp-window frac 0.32, curved frac 0.52, mean v0 4.7, nav {'nav=follow': 31, 'nav=left': 0, 'nav=right': 0}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**R7-30k vs R7-20k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 2224 / 0.0080 [-0.0111, 0.0240] | 66 / -0.0479 [-0.4175, 0.2960] |
| only new agrees | 312 / -0.3890 [-0.6545, -0.1569] | 34 / -1.6776 [-3.8627, -0.2944] |
| only earlier agrees | 228 / 0.1112 [-0.0353, 0.2554] | 43 / 0.0670 [-0.5220, 0.6824] |
| neither agrees | 1757 / -0.0113 [-0.0994, 0.0899] | 225 / 0.1286 [-0.3301, 0.7060] |
**R7-30k vs R7-20k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 2236 / 0.0015 [-0.0184, 0.0191] | 73 / -0.1915 [-0.5771, 0.1748] |
| only new agrees | 304 / -0.2961 [-0.5876, -0.0470] | 35 / -1.1325 [-3.3858, 0.4943] |
| only earlier agrees | 208 / 0.0976 [-0.0591, 0.2404] | 29 / 0.1309 [-0.7672, 0.9464] |
| neither agrees | 1770 / -0.0086 [-0.0990, 0.0955] | 231 / 0.2166 [-0.2820, 0.7727] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**R7-30k vs R7-20k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4514 | 0.0309 [-0.0266, 0.1012] | 0.0589 [-0.0136, 0.1443] | -0.0277 [-0.1025, 0.0350] |
| turn|straight <0.02 | 2896 | -0.0240 [-0.0506, 0.0055] | -0.0207 [-0.0518, 0.0116] | -0.0043 [-0.0215, 0.0180] |
| turn|gentle 0.02-0.06 | 721 | -0.0180 [-0.1334, 0.0773] | 0.0215 [-0.0281, 0.0865] | -0.0399 [-0.1359, 0.0330] |
| turn|turn 0.06-0.15 | 510 | 0.1191 [-0.0362, 0.3461] | 0.0702 [-0.1827, 0.3249] | 0.0514 [-0.0976, 0.2554] |
| turn|sharp >=0.15 | 368 | 0.4356 [0.0750, 0.8390] | 0.7306 [0.0919, 1.5744] | -0.2949 [-1.1138, 0.3112] |
| turn|undef (GT crawling) | 19 | 0.0519 [-0.1242, 0.2388] | 0.2725 [-0.0434, 0.8481] | -0.0639 [-0.2989, 0.1234] |
| speed|v0 <2 | 225 | 0.3004 [0.1136, 0.5311] | 0.1807 [-0.0229, 0.3663] | 0.1284 [-0.0705, 0.4119] |
| speed|v0 2-5 | 516 | 0.2345 [-0.1783, 0.7598] | 0.0293 [-0.2999, 0.4019] | 0.2051 [-0.1071, 0.5416] |
| speed|v0 5-10 | 1390 | -0.0151 [-0.0764, 0.0472] | 0.1443 [-0.0058, 0.3801] | -0.1594 [-0.3975, -0.0221] SEP |
| speed|v0 10-15 | 1192 | -0.0153 [-0.0289, -0.0004] | 0.0070 [-0.0086, 0.0222] | -0.0223 [-0.0407, -0.0020] SEP |
| speed|v0 15-25 | 856 | -0.0135 [-0.0236, -0.0044] | -0.0014 [-0.0103, 0.0073] | -0.0121 [-0.0244, -0.0011] SEP |
| speed|v0 >=25 | 335 | -0.0001 [-0.0096, 0.0089] | 0.0033 [-0.0054, 0.0136] | -0.0034 [-0.0191, 0.0088] |
**R7-30k vs R7-20k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* R7-30k vs R7-20k|seed0: anchor flip rate 0.215; heading delta on flipped windows -0.0772 [-0.2413, 0.0949] (n 933) vs unflipped -0.0072 [-0.0309, 0.0140]; goal_dist_sel proxy new-earlier -0.1996 [-0.4489, 0.0453]; corr(window dHeading, window dGoalDist) -0.001
* R7-30k vs R7-20k|seed1: anchor flip rate 0.218; heading delta on flipped windows -0.0544 [-0.2296, 0.1474] (n 946) vs unflipped -0.0084 [-0.0311, 0.0122]; goal_dist_sel proxy new-earlier -0.2022 [-0.4507, 0.0398]; corr(window dHeading, window dGoalDist) 0.000

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

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

### S6 -- R7-30k - R7-20k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3539 | 139 | 3.4905 | 3.7481 | -0.2575 [-0.3952, -0.1221] | **SEP** |  |
| turn: straight <0.02 | 2091 | 122 | 1.4953 | 1.6649 | -0.1696 [-0.2543, -0.0977] | **SEP** | 0.591 / 0.389 / 0.2150 [-0.2548, 0.7507] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.9231 | 2.9577 | -0.0346 [-0.2322, 0.1845] |  | 0.191 / 0.026 / 0.2755 [-0.0810, 0.6636] |
| turn: turn 0.06-0.15 | 441 | 59 | 6.6920 | 7.1601 | -0.4681 [-0.9786, 0.0137] |  | 0.125 / 0.226 / -0.2405 [-1.0283, 0.5057] |
| turn: sharp >=0.15 | 312 | 33 | 13.0227 | 14.1420 | -1.1193 [-2.2781, 0.0389] |  | 0.088 / 0.383 / -0.9451 [-3.3898, 0.8187] |
| turn: undef (GT crawling) (low n) | 20 | 10 | 11.9548 | 10.8460 | 1.1088 [-0.1036, 2.5159] |  | 0.006 / -0.024 / 1.3741 [-0.0496, 3.4911] |
| speed: v0 <2 | 201 | 27 | 14.2799 | 14.7495 | -0.4696 [-0.7436, -0.1717] | **SEP** | 0.057 / 0.104 / -0.2248 [-0.7484, 0.3244] |
| speed: v0 2-5 | 422 | 52 | 4.4408 | 5.2980 | -0.8572 [-1.5268, -0.2436] | **SEP** | 0.119 / 0.397 / -0.6808 [-2.0149, 0.2127] |
| speed: v0 5-10 | 1040 | 79 | 3.7918 | 4.0421 | -0.2503 [-0.5548, 0.0561] |  | 0.294 / 0.286 / 0.0103 [-0.5054, 0.4847] |
| speed: v0 10-15 | 952 | 62 | 2.6243 | 2.7810 | -0.1567 [-0.3095, 0.0052] |  | 0.269 / 0.164 / 0.1380 [-0.1992, 0.5049] |
| speed: v0 15-25 | 679 | 34 | 1.5277 | 1.6197 | -0.0920 [-0.2074, 0.0096] |  | 0.192 / 0.069 / 0.2049 [-0.1234, 0.4940] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5291 | 0.4613 | 0.0677 [0.0031, 0.1593] | **SEP** | 0.069 / -0.018 / 0.3495 [0.1074, 0.6386] BONF-SEP |
| nav: nav=follow | 2237 | 88 | 2.1167 | 2.3267 | -0.2100 [-0.3526, -0.0778] | **SEP** | 0.632 / 0.516 / 0.1291 [-0.3401, 0.5972] |
| nav: nav=left (low n) | 326 | 13 | 10.3524 | 10.3154 | 0.0370 [-0.5354, 0.5665] |  | 0.092 / -0.013 / 0.3245 [-0.7255, 1.1798] |
| nav: nav=right | 976 | 38 | 4.3474 | 4.8123 | -0.4648 [-0.7775, -0.1302] | **SEP** | 0.276 / 0.498 / -0.2862 [-0.7798, 0.2283] |

episode concentration: 0.353 of episodes have mean delta > 0; top-5 episodes carry -0.252 of the summed delta; delta excluding the top 5 = -0.3340 (overall -0.2575).

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

### S6 -- R7-30k - R7-20k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3500 | 139 | 0.8009 | 1.0352 | -0.2343 [-0.6579, -0.0073] | **SEP** | -0.1094 [-0.5391, 0.1289] |
| seg1 | straight | 2072 | 122 | 0.5616 | 0.6572 | -0.0956 [-0.2889, -0.0002] | **SEP** | -0.1943 [-0.7688, 0.0849] |
| seg1 | curved | 1410 | 88 | 1.0299 | 1.4700 | -0.4401 [-1.2624, -0.0102] | **SEP** | 0.0103 [-0.2824, 0.2085] |
| seg2 | all | 3483 | 139 | 1.4276 | 1.6564 | -0.2288 [-0.6680, 0.0422] |  | 0.1396 [-0.1135, 0.3115] |
| seg2 | straight | 2072 | 122 | 0.7875 | 1.0554 | -0.2679 [-0.8432, 0.0090] |  | 0.1109 [-0.0962, 0.2344] |
| seg2 | curved | 1411 | 88 | 2.3676 | 2.5389 | -0.1714 [-0.5213, 0.1169] |  | 0.1818 [-0.1609, 0.4637] |
| seg3 | all | 3453 | 139 | 2.4222 | 2.4424 | -0.0202 [-0.1772, 0.1400] |  | 0.5073 [0.3194, 0.6825] SEP |
| seg3 | straight | 2050 | 121 | 1.3798 | 1.3326 | 0.0472 [-0.0816, 0.2624] |  | 0.3680 [0.1457, 0.5190] SEP |
| seg3 | curved | 1403 | 88 | 3.9452 | 4.0639 | -0.1187 [-0.4247, 0.1799] |  | 0.7110 [0.3765, 1.0264] SEP |
| seg4 | all | 3418 | 139 | 3.3055 | 3.5894 | -0.2839 [-0.4601, -0.1105] | **SEP** | 0.6485 [0.4371, 0.8562] SEP |
| seg4 | straight | 2036 | 121 | 1.3575 | 1.5237 | -0.1662 [-0.2680, -0.0819] | **SEP** | 0.5777 [0.4516, 0.7282] SEP |
| seg4 | curved | 1382 | 87 | 6.1753 | 6.6325 | -0.4573 [-0.8723, -0.0352] | **SEP** | 0.7528 [0.2723, 1.2261] SEP |
| seg5 | all | 3382 | 139 | 4.6742 | 5.2092 | -0.5350 [-0.7819, -0.2938] | **SEP** | 0.6120 [0.3107, 0.9081] SEP |
| seg5 | straight | 2012 | 121 | 1.4070 | 1.7594 | -0.3523 [-0.5134, -0.2221] | **SEP** | 0.6590 [0.4587, 0.8865] SEP |
| seg5 | curved | 1370 | 88 | 9.4723 | 10.2757 | -0.8033 [-1.3534, -0.2566] | **SEP** | 0.5430 [-0.1452, 1.1907] |
| seg6 | all | 3344 | 138 | 6.3357 | 7.0986 | -0.7629 [-1.0789, -0.4742] | **SEP** | 0.7019 [0.2448, 1.1818] SEP |
| seg6 | straight | 1997 | 120 | 1.6343 | 2.1192 | -0.4849 [-0.7070, -0.2938] | **SEP** | 0.9193 [0.5124, 1.4520] SEP |
| seg6 | curved | 1344 | 88 | 13.2970 | 14.4915 | -1.1944 [-1.8926, -0.5247] | **SEP** | 0.3972 [-0.5256, 1.2916] |

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

### S6 -- R7-30k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 366 | 39 | 5.2610 | 5.1490 | 0.1120 [-0.0126, 0.2394] |  |  |
| straight <0.02 | v0 5-15 | 1050 | 84 | 0.8068 | 0.7632 | 0.0436 [-0.0374, 0.1291] |  |  |
| straight <0.02 | v0 >=15 | 684 | 36 | 0.5368 | 0.5523 | -0.0156 [-0.0461, 0.0162] |  |  |
| gentle 0.02-0.06 | v0 <5 (low n) | 98 | 22 | 5.2094 | 5.5283 | -0.3188 [-1.4942, 0.4756] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 424 | 59 | 2.5827 | 2.5310 | 0.0517 [-0.0606, 0.1629] |  |  |
| gentle 0.02-0.06 | v0 >=15 (low n) | 153 | 14 | 2.4023 | 2.1952 | 0.2070 [0.0298, 0.4069] | **SEP** |  |
| turn 0.06-0.15 | v0 <5 (low n) | 70 | 20 | 10.7392 | 10.7014 | 0.0378 [-0.4908, 0.8969] |  |  |
| turn 0.06-0.15 | v0 5-15 | 293 | 43 | 6.1963 | 5.8272 | 0.3691 [0.0770, 0.6700] | **SEP** |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 78 | 10 | 4.9219 | 4.3281 | 0.5938 [0.0038, 1.1002] | **SEP** |  |
| sharp >=0.15 | v0 <5 (low n) | 78 | 16 | 17.0597 | 15.1685 | 1.8912 [0.1431, 4.2756] | **SEP** |  |
| sharp >=0.15 | v0 5-15 | 225 | 26 | 11.9292 | 10.7541 | 1.1751 [-0.3729, 2.9432] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 9 | 3 | 5.3728 | 6.3909 | -1.0180 [-2.0342, 1.8384] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 21 | 10 | 11.4093 | 10.5155 | 0.8938 [0.0024, 1.9559] | **SEP** |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S6 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | R7-30k s0-s1 [CI] | R7-20k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | 0.0185 [-0.0369, 0.0709] | 0.0032 [-0.0496, 0.0598] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | -0.0084 [-0.0264, 0.0108] | 0.0262 [-0.0207, 0.0960] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | 0.0693 [-0.0460, 0.2432] | 0.0292 [-0.0744, 0.1218] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | 0.0928 [-0.0314, 0.2301] | -0.0675 [-0.2282, 0.1022] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | 0.0475 [-0.4823, 0.4413] | -0.1077 [-0.5568, 0.3422] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | -0.9093 [-2.3766, 0.3252] | 0.0091 [-0.1871, 0.2769] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | 0.2784 [-0.0018, 0.6372] | 0.4396 [0.0260, 1.0407] SEP | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | -0.0736 [-0.4282, 0.2864] | -0.1181 [-0.4121, 0.0983] | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | 0.0440 [-0.0405, 0.1290] | 0.0358 [-0.1045, 0.1956] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | 0.0092 [-0.0307, 0.0556] | -0.0728 [-0.1371, -0.0117] SEP | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | -0.0311 [-0.0639, -0.0040] SEP | 0.0008 [-0.0513, 0.0624] | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | 0.0189 [-0.0164, 0.0666] | -0.0123 [-0.0357, 0.0129] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | 0.0236 [-0.0124, 0.0651] | -0.0214 [-0.0692, 0.0290] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | 0.1426 [-0.0193, 0.3162] | 0.0776 [0.0039, 0.1823] SEP | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | -0.0347 [-0.1898, 0.1397] | 0.0340 [-0.1064, 0.1917] | 0.0235 [-0.1175, 0.1775] |

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

### S6 -- R7-30k - R7-20k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.2575 | -0.2443 / -0.2716 | True / True / True | 0.0185 / 0.0032 | 13.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | -0.1696 | -0.1803 / -0.1604 | True / True / True | 0.0084 / 0.0262 | 6.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | -0.0346 | -0.0198 / -0.0494 | False / False / False | 0.0693 / 0.0292 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.4681 | -0.3631 / -0.5731 | False / True / False | 0.0928 / 0.0675 | 5.0 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | -1.1193 | -1.0465 / -1.1921 | False / False / False | 0.0475 / 0.1077 | 10.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 1.1088 | 0.6105 / 1.6071 | False / False / False | 0.9093 / 0.0091 | 1.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.4696 | -0.4620 / -0.4885 | True / True / True | 0.2784 / 0.4396 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.8572 | -0.8350 / -0.8794 | True / True / True | 0.0736 / 0.1181 | 7.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 5-10 | -0.2503 | -0.2462 / -0.2544 | False / False / False | 0.0440 / 0.0358 | 5.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.1567 | -0.1157 / -0.1977 | False / True / False | 0.0092 / 0.0728 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0920 | -0.1079 / -0.0760 | False / False / False | 0.0311 / 0.0008 | 3.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0677 | 0.0834 / 0.0521 | True / True / True | 0.0189 / 0.0123 | 3.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | -0.2100 | -0.1870 / -0.2335 | True / True / True | 0.0236 / 0.0214 | 8.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.0370 | 0.0883 / -0.0142 | False / False / False | 0.1426 / 0.0776 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.4648 | -0.4870 / -0.4448 | True / True / True | 0.0347 / 0.0340 | 13.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

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

* seg1 (n 1410): R7-30k bias +0.106, refcv7-50400 bias +0.037, R7-20k bias +0.395; R7-30k MAE 1.030, refcv7-50400 MAE 1.092, R7-20k MAE 1.470
* seg2 (n 1411): R7-30k bias -0.397, refcv7-50400 bias -0.242, R7-20k bias -0.047; R7-30k MAE 2.368, refcv7-50400 MAE 2.376, R7-20k MAE 2.539
* seg3 (n 1402): R7-30k bias -1.476, refcv7-50400 bias -1.128, R7-20k bias -1.175; R7-30k MAE 3.947, refcv7-50400 MAE 3.677, R7-20k MAE 4.066
* seg4 (n 1382): R7-30k bias -3.342, refcv7-50400 bias -2.567, R7-20k bias -3.171; R7-30k MAE 6.175, refcv7-50400 MAE 5.709, R7-20k MAE 6.633
* seg5 (n 1370): R7-30k bias -5.246, refcv7-50400 bias -4.512, R7-20k bias -5.429; R7-30k MAE 9.472, refcv7-50400 MAE 8.809, R7-20k MAE 10.276
* seg6 (n 1343): R7-30k bias -7.331, refcv7-50400 bias -6.971, R7-20k bias -7.620; R7-30k MAE 13.302, refcv7-50400 MAE 12.373, R7-20k MAE 14.496
**sharp (>= 0.15)**

* seg1 (n 305): R7-30k bias -0.026, refcv7-50400 bias -0.421, R7-20k bias +0.382; R7-30k MAE 1.537, refcv7-50400 MAE 2.024, R7-20k MAE 2.973
* seg2 (n 302): R7-30k bias -1.570, refcv7-50400 bias -0.568, R7-20k bias -1.246; R7-30k MAE 4.098, refcv7-50400 MAE 3.821, R7-20k MAE 4.330
* seg3 (n 303): R7-30k bias -4.814, refcv7-50400 bias -2.527, R7-20k bias -4.891; R7-30k MAE 7.733, refcv7-50400 MAE 6.701, R7-20k MAE 8.128
* seg4 (n 302): R7-30k bias -9.774, refcv7-50400 bias -6.666, R7-20k bias -10.581; R7-30k MAE 13.513, refcv7-50400 MAE 11.654, R7-20k MAE 14.691
* seg5 (n 298): R7-30k bias -15.121, refcv7-50400 bias -11.750, R7-20k bias -16.794; R7-30k MAE 20.919, refcv7-50400 MAE 18.992, R7-20k MAE 23.249
* seg6 (n 292): R7-30k bias -20.205, refcv7-50400 bias -17.582, R7-20k bias -21.616; R7-30k MAE 27.567, refcv7-50400 MAE 25.270, R7-20k MAE 31.209
**curved & v0 5-10**

* seg1 (n 547): R7-30k bias -0.010, refcv7-50400 bias +0.049, R7-20k bias +0.199; R7-30k MAE 0.993, refcv7-50400 MAE 0.896, R7-20k MAE 1.009
* seg2 (n 547): R7-30k bias -0.769, refcv7-50400 bias -0.516, R7-20k bias -0.417; R7-30k MAE 2.434, refcv7-50400 MAE 2.231, R7-20k MAE 2.356
* seg3 (n 545): R7-30k bias -2.275, refcv7-50400 bias -1.649, R7-20k bias -1.729; R7-30k MAE 4.235, refcv7-50400 MAE 3.822, R7-20k MAE 4.200
* seg4 (n 536): R7-30k bias -4.318, refcv7-50400 bias -3.537, R7-20k bias -3.780; R7-30k MAE 6.881, refcv7-50400 MAE 6.217, R7-20k MAE 7.042
* seg5 (n 525): R7-30k bias -6.405, refcv7-50400 bias -5.842, R7-20k bias -6.085; R7-30k MAE 10.127, refcv7-50400 MAE 9.315, R7-20k MAE 10.581
* seg6 (n 505): R7-30k bias -8.507, refcv7-50400 bias -8.429, R7-20k bias -8.117; R7-30k MAE 13.699, refcv7-50400 MAE 12.679, R7-20k MAE 14.632
**sharp & v0 5-10**

* seg1 (n 139): R7-30k bias +0.021, refcv7-50400 bias +0.062, R7-20k bias +0.642; R7-30k MAE 1.454, refcv7-50400 MAE 1.251, R7-20k MAE 1.538
* seg2 (n 139): R7-30k bias -1.794, refcv7-50400 bias -1.267, R7-20k bias -0.900; R7-30k MAE 3.957, refcv7-50400 MAE 3.564, R7-20k MAE 3.694
* seg3 (n 139): R7-30k bias -5.957, refcv7-50400 bias -4.133, R7-20k bias -5.173; R7-30k MAE 8.355, refcv7-50400 MAE 7.029, R7-20k MAE 7.618
* seg4 (n 139): R7-30k bias -11.538, refcv7-50400 bias -8.959, R7-20k bias -11.043; R7-30k MAE 14.544, refcv7-50400 MAE 12.455, R7-20k MAE 14.295
* seg5 (n 137): R7-30k bias -16.421, refcv7-50400 bias -14.360, R7-20k bias -16.085; R7-30k MAE 22.077, refcv7-50400 MAE 19.770, R7-20k MAE 22.547
* seg6 (n 133): R7-30k bias -19.645, refcv7-50400 bias -18.859, R7-20k bias -18.244; R7-30k MAE 28.129, refcv7-50400 MAE 26.056, R7-20k MAE 29.929

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1343 / 1343 | 26.0 | R7-30k -0.2795 [-0.3580, -0.2065]; refcv7-50400 -0.2503 [-0.3241, -0.1834]; R7-20k -0.2894 [-0.3743, -0.2173] | R7-30k -0.0291 [-0.0638, +0.0043]; R7-20k -0.0390 [-0.0863, -0.0022] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1427 / 6908 | 16.4 | R7-30k -0.2026 [-0.2806, -0.1312]; refcv7-50400 -0.1810 [-0.2473, -0.1219]; R7-20k -0.1983 [-0.2801, -0.1217] | R7-30k -0.0216 [-0.0682, +0.0373]; R7-20k -0.0173 [-0.0731, +0.0567] |
| sharp (>= 0.15) | last segment | 292 / 292 | 62.8 | R7-30k -0.3105 [-0.4086, -0.2262]; refcv7-50400 -0.2668 [-0.3574, -0.1894]; R7-20k -0.3209 [-0.4318, -0.2357] | R7-30k -0.0436 [-0.0889, -0.0003] SEP; R7-20k -0.0541 [-0.1183, -0.0090] SEP |
| sharp (>= 0.15) | segments 2..K | 311 / 1497 | 39.5 | R7-30k -0.2576 [-0.3520, -0.1779]; refcv7-50400 -0.1966 [-0.2837, -0.1244]; R7-20k -0.2661 [-0.3703, -0.1836] | R7-30k -0.0611 [-0.1040, -0.0230] SEP; R7-20k -0.0696 [-0.1255, -0.0293] SEP |
| curved & v0 5-10 | last segment | 505 / 505 | 27.9 | R7-30k -0.2937 [-0.4023, -0.2021]; refcv7-50400 -0.2748 [-0.4002, -0.1735]; R7-20k -0.2758 [-0.4029, -0.1654] | R7-30k -0.0189 [-0.0796, +0.0392]; R7-20k -0.0010 [-0.0486, +0.0380] |
| curved & v0 5-10 | segments 2..K | 547 / 2658 | 17.7 | R7-30k -0.2428 [-0.3458, -0.1603]; refcv7-50400 -0.2066 [-0.3209, -0.1203]; R7-20k -0.2240 [-0.3313, -0.1360] | R7-30k -0.0362 [-0.0923, +0.0157]; R7-20k -0.0174 [-0.0556, +0.0159] |
| sharp & v0 5-10 | last segment | 133 / 133 | 66.9 | R7-30k -0.3008 [-0.4236, -0.2020]; refcv7-50400 -0.2729 [-0.4098, -0.1615]; R7-20k -0.2721 [-0.4126, -0.1519] | R7-30k -0.0279 [-0.1019, +0.0430]; R7-20k +0.0008 [-0.0578, +0.0465] |
| sharp & v0 5-10 | segments 2..K | 139 / 687 | 43.7 | R7-30k -0.2594 [-0.3802, -0.1659]; refcv7-50400 -0.2114 [-0.3439, -0.1138]; R7-20k -0.2352 [-0.3590, -0.1300] | R7-30k -0.0480 [-0.1184, +0.0180]; R7-20k -0.0239 [-0.0743, +0.0170] |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

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
**R7-30k - R7-20k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3539 | -0.2575 | -0.05058 | -0.13276 | 0.441 / 0.559 / 0.000 | 549.1 / -911.5 |
| turn|straight <0.02 | 2091 | -0.1696 | -0.03333 | -0.08953 | 0.438 / 0.562 / 0.000 | 48.8 / -354.6 |
| turn|gentle 0.02-0.06 | 675 | -0.0346 | -0.03744 | -0.04950 | 0.481 / 0.519 / 0.000 | 85.7 / -23.4 |
| turn|turn 0.06-0.15 | 441 | -0.4681 | -0.18801 | -0.36033 | 0.420 / 0.580 / 0.000 | 92.0 / -206.4 |
| turn|sharp >=0.15 | 312 | -1.1193 | -0.40514 | -0.77737 | 0.394 / 0.606 / 0.000 | 111.4 / -349.2 |
| turn|undef (GT crawling) | 20 | 1.1088 | 0.00664 | 0.05898 | 0.550 / 0.450 / 0.000 | 18.2 / 22.2 |
| speed|v0 <2 | 201 | -0.4696 | -0.33496 | -0.51798 | 0.343 / 0.657 / 0.000 | 30.7 / -94.4 |
| speed|v0 2-5 | 422 | -0.8572 | -0.35980 | -0.50620 | 0.310 / 0.690 / 0.000 | 92.2 / -361.7 |
| speed|v0 5-10 | 1040 | -0.2503 | -0.10026 | -0.15489 | 0.445 / 0.555 / 0.000 | 210.8 / -260.3 |
| speed|v0 10-15 | 952 | -0.1567 | -0.03161 | -0.07548 | 0.450 / 0.550 / 0.000 | 76.9 / -149.2 |
| speed|v0 15-25 | 679 | -0.0920 | -0.01174 | -0.02336 | 0.476 / 0.524 / 0.000 | 22.0 / -62.4 |
| speed|v0 >=25 | 245 | 0.0677 | 0.03046 | 0.02748 | 0.596 / 0.404 / 0.000 | 6.0 / 16.6 |
| nav|nav=follow | 2237 | -0.2100 | -0.03953 | -0.09549 | 0.442 / 0.558 / 0.000 | 216.3 / -469.9 |
| nav|nav=left | 326 | 0.0370 | 0.00925 | 0.03046 | 0.521 / 0.479 / 0.000 | 61.5 / 12.1 |
| nav|nav=right | 976 | -0.4648 | -0.14715 | -0.30752 | 0.411 / 0.589 / 0.000 | 196.6 / -453.7 |
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

**R7-30k - refcv7-50400** (sum 677.1 over 139 episodes)

* ep111: n 26, share 0.189, mean delta 4.9261, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.183, mean delta 4.7761, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep028: n 27, share 0.094, mean delta 2.3661, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep016: n 22, share 0.084, mean delta 2.5708, sharp-window frac 0.41, curved frac 0.59, mean v0 5.6, nav {'nav=follow': 0, 'nav=left': 22, 'nav=right': 0}
* ep085: n 26, share 0.082, mean delta 2.1458, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep061: n 26, share 0.072, mean delta 1.8755, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
**R7-30k - R7-20k** (sum -911.5 over 139 episodes)

* ep111: n 26, share -0.064, mean delta 2.2350, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep078: n 26, share -0.051, mean delta 1.7950, sharp-window frac 0.00, curved frac 0.54, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep100: n 18, share -0.049, mean delta 2.4894, sharp-window frac 0.50, curved frac 1.00, mean v0 5.2, nav {'nav=follow': 0, 'nav=left': 18, 'nav=right': 0}
* ep029: n 26, share -0.047, mean delta 1.6608, sharp-window frac 0.00, curved frac 1.00, mean v0 12.1, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share -0.041, mean delta 1.4301, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep012: n 26, share -0.039, mean delta 1.3594, sharp-window frac 0.46, curved frac 0.77, mean v0 7.0, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
**R7-20k - refcv7-50400** (sum 1693.4 over 139 episodes)

* ep085: n 26, share 0.058, mean delta 3.7941, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep100: n 26, share 0.056, mean delta 3.6358, sharp-window frac 0.35, curved frac 0.85, mean v0 3.6, nav {'nav=follow': 0, 'nav=left': 26, 'nav=right': 0}
* ep013: n 27, share 0.055, mean delta 3.4561, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep041: n 26, share 0.054, mean delta 3.5224, sharp-window frac 0.12, curved frac 0.38, mean v0 8.6, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.051, mean delta 3.3460, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep067: n 26, share 0.045, mean delta 2.9568, sharp-window frac 0.35, curved frac 0.96, mean v0 4.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**R7-30k vs R7-20k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1721 / -0.1255 [-0.1854, -0.0728] | 47 / -0.3353 [-0.8352, 0.0913] |
| only new agrees | 257 / -2.0590 [-3.1022, -1.2128] | 27 / -10.1848 [-18.5531, -4.6938] |
| only earlier agrees | 174 / 0.3295 [-0.2938, 0.9362] | 28 / 0.6738 [-2.1847, 3.3303] |
| neither agrees | 1389 / -0.1276 [-0.3997, 0.1402] | 210 / -0.2601 [-1.5618, 1.1266] |
**R7-30k vs R7-20k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1724 / -0.1237 [-0.1853, -0.0673] | 56 / 0.0481 [-0.5374, 0.7119] |
| only new agrees | 262 / -1.7671 [-2.7072, -1.0372] | 25 / -10.7607 [-19.7100, -4.7498] |
| only earlier agrees | 165 / 0.4095 [0.0092, 0.8753] | 19 / 1.4107 [-0.9274, 3.9053] |
| neither agrees | 1389 / -0.2540 [-0.5214, 0.0299] | 212 / -0.6246 [-2.0058, 0.8716] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**R7-30k vs R7-20k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3528 | 0.2684 [-0.0013, 0.6224] | 0.2188 [-0.0458, 0.5550] | 0.0485 [-0.2073, 0.2742] |
| turn|straight <0.02 | 2089 | -0.0116 [-0.0377, 0.0146] | -0.1226 [-0.2141, -0.0514] | 0.0915 [0.0283, 0.1759] SEP |
| turn|gentle 0.02-0.06 | 675 | -0.3040 [-0.7972, 0.0186] | -0.2909 [-0.6428, 0.0101] | -0.0315 [-0.2834, 0.1908] |
| turn|turn 0.06-0.15 | 437 | 0.1798 [-0.2220, 0.6523] | -0.1136 [-0.5918, 0.3094] | 0.3106 [-0.0424, 0.7663] |
| turn|sharp >=0.15 | 310 | 3.5454 [1.2035, 6.9651] | 4.1074 [1.6804, 7.1485] | -0.4437 [-3.0421, 1.8553] |
| turn|undef (GT crawling) | 17 | 0.0560 [-0.2548, 0.3435] | -0.3843 [-0.8422, 0.1214] | 0.1780 [-0.0870, 0.4625] |
| speed|v0 <2 | 190 | 0.5921 [-0.0256, 1.2883] | -0.0013 [-0.8575, 0.8521] | 0.6135 [0.1909, 1.1762] SEP |
| speed|v0 2-5 | 422 | 1.1090 [-0.5204, 3.3488] | 0.5277 [-0.6093, 1.9391] | 0.5813 [-0.2665, 1.5495] |
| speed|v0 5-10 | 1040 | 0.3891 [0.0293, 0.8012] | 0.5521 [-0.0683, 1.4855] | -0.1630 [-0.9334, 0.3500] |
| speed|v0 10-15 | 952 | -0.0485 [-0.1090, 0.0053] | 0.0021 [-0.0834, 0.0813] | -0.0506 [-0.1408, 0.0258] |
| speed|v0 15-25 | 679 | 0.0007 [-0.0284, 0.0271] | -0.0380 [-0.1045, 0.0137] | 0.0387 [-0.0166, 0.1018] |
| speed|v0 >=25 | 245 | 0.0198 [-0.0076, 0.0472] | 0.0175 [-0.0140, 0.0529] | 0.0024 [-0.0388, 0.0298] |
**R7-30k vs R7-20k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* R7-30k vs R7-20k|seed0: anchor flip rate 0.216; heading delta on flipped windows -0.6483 [-1.2404, -0.0189] (n 751) vs unflipped -0.1356 [-0.1881, -0.0851]; goal_dist_sel proxy new-earlier -0.1949 [-0.4713, 0.0746]; corr(window dHeading, window dGoalDist) -0.005
* R7-30k vs R7-20k|seed1: anchor flip rate 0.216; heading delta on flipped windows -0.7586 [-1.3539, -0.2000] (n 752) vs unflipped -0.1402 [-0.1974, -0.0870]; goal_dist_sel proxy new-earlier -0.1971 [-0.4736, 0.0735]; corr(window dHeading, window dGoalDist) -0.012
