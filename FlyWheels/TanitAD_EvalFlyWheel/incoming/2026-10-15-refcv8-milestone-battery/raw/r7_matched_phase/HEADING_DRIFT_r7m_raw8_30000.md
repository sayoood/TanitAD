# refcv8 heading-drift localisation: RAW8-30k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-18T23:08:58; trees: `RAW8-30k` = C:/Users/Admin/refcv8_eval_archive/battery/step30000_unc; `R7-30k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step30000_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **True**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

### S2 -- RAW8-30k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4506 | 139 | 1.4381 | 1.3277 | 0.1104 [0.0554, 0.1723] | **SEP** |  |
| turn: straight <0.02 | 2898 | 132 | 1.2027 | 1.0970 | 0.1057 [0.0525, 0.1683] | **SEP** | 0.643 / 0.616 / -0.0132 [-0.2182, 0.1539] |
| turn: gentle 0.02-0.06 | 720 | 98 | 1.0561 | 1.0094 | 0.0466 [-0.0261, 0.1229] |  | 0.160 / 0.068 / -0.0759 [-0.2049, 0.0519] |
| turn: turn 0.06-0.15 | 505 | 72 | 1.9597 | 1.9068 | 0.0529 [-0.0892, 0.2138] |  | 0.112 / 0.054 / -0.0648 [-0.2699, 0.2000] |
| turn: sharp >=0.15 | 367 | 42 | 3.3717 | 3.0107 | 0.3609 [-0.0289, 0.7522] |  | 0.081 / 0.266 / 0.2728 [-0.2284, 0.9752] |
| turn: undef (GT crawling) (low n) | 16 | 12 | 0.4397 | 0.5410 | -0.1014 [-0.3747, 0.0984] |  | 0.004 / -0.003 / -0.2125 [-0.7213, 0.0834] |
| speed: v0 <2 | 217 | 37 | 11.9188 | 11.7879 | 0.1309 [-0.0338, 0.3635] |  | 0.048 / 0.057 / 0.0215 [-0.2156, 0.5207] |
| speed: v0 2-5 | 516 | 58 | 2.0778 | 1.8694 | 0.2083 [-0.1121, 0.5733] |  | 0.115 / 0.216 / 0.1106 [-0.2975, 0.6369] |
| speed: v0 5-10 | 1390 | 84 | 1.2431 | 1.0092 | 0.2340 [0.1135, 0.3519] | **SEP** | 0.308 / 0.654 / 0.1787 [-0.0203, 0.3648] |
| speed: v0 10-15 | 1192 | 64 | 0.6058 | 0.5744 | 0.0314 [-0.0168, 0.0726] |  | 0.265 / 0.075 / -0.1074 [-0.2361, 0.0021] |
| speed: v0 15-25 | 856 | 35 | 0.3607 | 0.3734 | -0.0127 [-0.0396, 0.0136] |  | 0.190 / -0.022 / -0.1519 [-0.2658, -0.0534] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1867 | 0.1576 | 0.0291 [0.0110, 0.0500] | **SEP** | 0.074 / 0.020 / -0.0879 [-0.1989, 0.0015] |
| nav: nav=follow | 2881 | 88 | 0.7200 | 0.5980 | 0.1220 [0.0605, 0.1952] | **SEP** | 0.639 / 0.707 / 0.0322 [-0.1678, 0.2587] |
| nav: nav=left (low n) | 400 | 13 | 6.4723 | 6.4981 | -0.0258 [-0.2254, 0.1748] |  | 0.089 / -0.021 / -0.1494 [-0.5516, 0.2261] |
| nav: nav=right | 1225 | 38 | 1.4831 | 1.3555 | 0.1276 [-0.0053, 0.2661] |  | 0.272 / 0.314 / 0.0236 [-0.1898, 0.2764] |

episode concentration: 0.576 of episodes have mean delta > 0; top-5 episodes carry 0.401 of the summed delta; delta excluding the top 5 = 0.0687 (overall 0.1104).

### S2 -- RAW8-30k - R7-30k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4497 | 139 | 1.4405 | 1.3851 | 0.0554 [0.0048, 0.1090] | **SEP** |  |
| turn: straight <0.02 | 2892 | 132 | 1.2047 | 1.1307 | 0.0741 [0.0218, 0.1360] | **SEP** | 0.643 / 0.859 / 0.0522 [-0.1053, 0.2203] |
| turn: gentle 0.02-0.06 | 720 | 98 | 1.0561 | 1.0281 | 0.0280 [-0.0425, 0.1040] |  | 0.160 / 0.081 / -0.0327 [-0.1662, 0.0955] |
| turn: turn 0.06-0.15 | 505 | 72 | 1.9597 | 1.9874 | -0.0277 [-0.1812, 0.1339] |  | 0.112 / -0.056 / -0.0936 [-0.3315, 0.1847] |
| turn: sharp >=0.15 | 367 | 42 | 3.3717 | 3.2952 | 0.0765 [-0.2606, 0.3876] |  | 0.082 / 0.113 / 0.0229 [-0.4544, 0.5740] |
| turn: undef (GT crawling) (low n) | 13 | 11 | 0.5116 | 0.4477 | 0.0639 [-0.1371, 0.2434] |  | 0.003 / 0.003 / 0.0085 [-0.3628, 0.3385] |
| speed: v0 <2 | 208 | 37 | 12.4257 | 12.3161 | 0.1096 [-0.0979, 0.3217] |  | 0.046 / 0.091 / 0.0568 [-0.2763, 0.4097] |
| speed: v0 2-5 | 516 | 58 | 2.0778 | 1.9548 | 0.1230 [-0.1665, 0.4677] |  | 0.115 / 0.255 / 0.0763 [-0.3317, 0.5799] |
| speed: v0 5-10 | 1390 | 84 | 1.2431 | 1.1066 | 0.1366 [0.0252, 0.2471] | **SEP** | 0.309 / 0.762 / 0.1174 [-0.0806, 0.3052] |
| speed: v0 10-15 | 1192 | 64 | 0.6058 | 0.6061 | -0.0003 [-0.0351, 0.0339] |  | 0.265 / -0.001 / -0.0758 [-0.1935, 0.0335] |
| speed: v0 15-25 | 856 | 35 | 0.3607 | 0.3950 | -0.0342 [-0.0736, 0.0010] |  | 0.190 / -0.118 / -0.1107 [-0.2265, -0.0056] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1867 | 0.1784 | 0.0082 [-0.0140, 0.0340] |  | 0.074 / 0.011 / -0.0510 [-0.1478, 0.0513] |
| nav: nav=follow | 2873 | 88 | 0.7213 | 0.6374 | 0.0839 [0.0300, 0.1507] | **SEP** | 0.639 / 0.967 / 0.0789 [-0.0975, 0.2811] |
| nav: nav=left (low n) | 400 | 13 | 6.4723 | 6.6474 | -0.1751 [-0.4049, 0.0396] |  | 0.089 / -0.281 / -0.2530 [-0.6912, 0.1274] |
| nav: nav=right | 1224 | 38 | 1.4843 | 1.4204 | 0.0638 [-0.0545, 0.1803] |  | 0.272 / 0.314 / 0.0116 [-0.1841, 0.2265] |

episode concentration: 0.504 of episodes have mean delta > 0; top-5 episodes carry 0.680 of the summed delta; delta excluding the top 5 = 0.0184 (overall 0.0554).

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

### S2 -- RAW8-30k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4472 | 139 | 0.7612 | 0.5875 | 0.1737 [0.0007, 0.5078] | **SEP** | 0.0072 [-0.0174, 0.0319] |
| seg1 | straight | 2878 | 132 | 0.8462 | 0.5699 | 0.2763 [0.0130, 0.8403] | **SEP** | -0.0142 [-0.0350, 0.0039] |
| seg1 | curved | 1584 | 101 | 0.6095 | 0.6218 | -0.0123 [-0.0406, 0.0170] |  | 0.0460 [-0.0108, 0.0992] |
| seg2 | all | 4461 | 139 | 1.1691 | 0.9373 | 0.2319 [0.0363, 0.5768] | **SEP** | 0.0159 [-0.0582, 0.0856] |
| seg2 | straight | 2877 | 132 | 1.0080 | 0.6572 | 0.3508 [0.0629, 0.9254] | **SEP** | -0.0159 [-0.0748, 0.0382] |
| seg2 | curved | 1584 | 101 | 1.4619 | 1.4460 | 0.0159 [-0.0702, 0.1059] |  | 0.0736 [-0.0905, 0.2243] |
| seg3 | all | 4449 | 139 | 1.5068 | 1.2015 | 0.3053 [0.0868, 0.6638] | **SEP** | -0.0084 [-0.1245, 0.0967] |
| seg3 | straight | 2870 | 132 | 1.1646 | 0.7736 | 0.3910 [0.0888, 0.9752] | **SEP** | -0.0184 [-0.1090, 0.0593] |
| seg3 | curved | 1579 | 101 | 2.1288 | 1.9792 | 0.1496 [0.0025, 0.3086] | **SEP** | 0.0099 [-0.2491, 0.2491] |
| seg4 | all | 4430 | 139 | 1.8189 | 1.4478 | 0.3711 [0.0937, 0.8223] | **SEP** | 0.1682 [-0.1428, 0.6310] |
| seg4 | straight | 2852 | 131 | 1.1972 | 0.7731 | 0.4241 [0.0634, 1.1277] | **SEP** | 0.2531 [-0.1195, 0.9649] |
| seg4 | curved | 1572 | 101 | 2.9515 | 2.6734 | 0.2781 [0.0523, 0.5227] | **SEP** | 0.0136 [-0.3933, 0.3900] |

### S2 -- RAW8-30k - R7-30k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4470 | 139 | 0.7614 | 0.5346 | 0.2268 [-0.0268, 0.7233] |  | -0.1470 [-0.3156, -0.0502] SEP |
| seg1 | straight | 2877 | 132 | 0.8462 | 0.4625 | 0.3837 [-0.0040, 1.2146] |  | -0.1971 [-0.4848, -0.0557] SEP |
| seg1 | curved | 1584 | 101 | 0.6095 | 0.6672 | -0.0577 [-0.0899, -0.0255] | **SEP** | -0.0563 [-0.1096, -0.0034] SEP |
| seg2 | all | 4460 | 139 | 1.1693 | 0.9098 | 0.2596 [-0.0148, 0.7636] |  | -0.0753 [-0.2142, 0.1218] |
| seg2 | straight | 2876 | 132 | 1.0082 | 0.5583 | 0.4499 [0.0382, 1.2779] | **SEP** | -0.0178 [-0.1921, 0.2809] |
| seg2 | curved | 1584 | 101 | 1.4619 | 1.5479 | -0.0860 [-0.1655, -0.0024] | **SEP** | -0.1798 [-0.3438, -0.0260] SEP |
| seg3 | all | 4446 | 139 | 1.5077 | 1.2621 | 0.2457 [0.0355, 0.5931] | **SEP** | -0.1321 [-0.2996, 0.0810] |
| seg3 | straight | 2867 | 132 | 1.1657 | 0.8063 | 0.3593 [0.0565, 0.9386] | **SEP** | -0.0211 [-0.2076, 0.2840] |
| seg3 | curved | 1579 | 101 | 2.1288 | 2.0895 | 0.0393 [-0.0865, 0.1732] |  | -0.3335 [-0.6017, -0.0914] SEP |
| seg4 | all | 4424 | 139 | 1.8211 | 1.6521 | 0.1691 [0.0238, 0.3652] | **SEP** | -0.3787 [-0.5337, -0.2308] SEP |
| seg4 | straight | 2848 | 131 | 1.1987 | 1.0123 | 0.1864 [0.0141, 0.4769] | **SEP** | -0.2767 [-0.3658, -0.1924] SEP |
| seg4 | curved | 1572 | 101 | 2.9515 | 2.8130 | 0.1385 [-0.0573, 0.3447] |  | -0.5652 [-0.9535, -0.2127] SEP |

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

### S2 -- RAW8-30k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 439 | 47 | 5.7034 | 5.5537 | 0.1497 [-0.0956, 0.4338] |  |  |
| straight <0.02 | v0 5-15 | 1519 | 93 | 0.5176 | 0.3674 | 0.1501 [0.0894, 0.2210] | **SEP** |  |
| straight <0.02 | v0 >=15 | 940 | 43 | 0.2079 | 0.1947 | 0.0133 [-0.0013, 0.0296] |  |  |
| gentle 0.02-0.06 | v0 <5 | 111 | 29 | 1.8737 | 1.7693 | 0.1044 [-0.1790, 0.4799] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 461 | 74 | 1.0341 | 0.9711 | 0.0630 [-0.0372, 0.1664] |  |  |
| gentle 0.02-0.06 | v0 >=15 | 148 | 20 | 0.5114 | 0.5590 | -0.0475 [-0.1232, 0.0126] |  |  |
| turn 0.06-0.15 | v0 <5 (low n) | 75 | 25 | 5.4195 | 5.1833 | 0.2361 [-0.1467, 0.6797] |  |  |
| turn 0.06-0.15 | v0 5-15 | 343 | 54 | 1.4715 | 1.4275 | 0.0440 [-0.1370, 0.2481] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 87 | 11 | 0.9018 | 0.9719 | -0.0702 [-0.1817, 0.0332] |  |  |
| sharp >=0.15 | v0 <5 (low n) | 92 | 16 | 5.7961 | 5.3342 | 0.4618 [-0.4010, 1.7298] |  |  |
| sharp >=0.15 | v0 5-15 | 259 | 33 | 2.6349 | 2.2857 | 0.3492 [-0.0961, 0.7332] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 16 | 3 | 1.3578 | 1.3866 | -0.0288 [-0.1840, 0.3029] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 16 | 12 | 0.4397 | 0.5410 | -0.1014 [-0.3747, 0.0984] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S2 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | RAW8-30k s0-s1 [CI] | R7-30k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0043 [-0.0210, 0.0133] | -0.0060 [-0.0196, 0.0079] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | -0.0009 [-0.0162, 0.0139] | -0.0050 [-0.0158, 0.0047] | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | -0.0028 [-0.0421, 0.0329] | 0.0210 [-0.0180, 0.0668] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | -0.0056 [-0.0810, 0.0625] | 0.0090 [-0.0310, 0.0579] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | -0.0330 [-0.1730, 0.0886] | -0.0870 [-0.1984, 0.0173] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | -0.0001 [-0.1696, 0.2010] | -0.0060 [-0.0909, 0.0632] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | 0.0101 [-0.0705, 0.0834] | 0.0226 [-0.0617, 0.1081] | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | -0.0280 [-0.1453, 0.0907] | -0.0262 [-0.1192, 0.0801] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | 0.0006 [-0.0330, 0.0341] | -0.0190 [-0.0394, 0.0006] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | -0.0015 [-0.0185, 0.0155] | 0.0070 [-0.0025, 0.0174] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | -0.0048 [-0.0142, 0.0050] | -0.0017 [-0.0088, 0.0049] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | -0.0073 [-0.0265, 0.0079] | 0.0017 [-0.0086, 0.0125] | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | -0.0097 [-0.0222, 0.0031] | 0.0003 [-0.0067, 0.0069] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | -0.0520 [-0.1654, 0.0355] | -0.0076 [-0.0615, 0.0391] | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | 0.0238 [-0.0156, 0.0677] | -0.0201 [-0.0627, 0.0255] | 0.0283 [0.0035, 0.0597] SEP |

### S2 -- RAW8-30k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1104 | 0.1027 / 0.1176 | True / True / True | 0.0043 / 0.0110 | 10.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.1057 | 0.1048 / 0.1065 | True / True / True | 0.0009 / 0.0019 | 54.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0466 | 0.0531 / 0.0411 | False / False / False | 0.0028 / 0.0161 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.0529 | 0.0263 / 0.0687 | False / False / False | 0.0056 / 0.0331 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.3609 | 0.2986 / 0.4340 | False / True / False | 0.0330 / 0.1080 | 3.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | -0.1014 | -0.0902 / -0.1278 | False / False / False | 0.0001 / 0.0237 | 4.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.1309 | 0.1044 / 0.1476 | False / False / False | 0.0101 / 0.0573 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.2083 | 0.1584 / 0.2583 | False / False / False | 0.0280 / 0.0719 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.2340 | 0.2340 / 0.2340 | True / True / True | 0.0006 / 0.0006 | 360.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0314 | 0.0307 / 0.0321 | False / False / False | 0.0015 / 0.0001 | 21.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0127 | -0.0139 / -0.0114 | False / False / False | 0.0048 / 0.0023 | 2.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0291 | 0.0257 / 0.0324 | True / True / True | 0.0073 / 0.0005 | 4.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.1220 | 0.1189 / 0.1250 | True / True / True | 0.0097 / 0.0036 | 12.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -0.0258 | -0.0684 / 0.0268 | False / False / False | 0.0520 / 0.0608 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.1276 | 0.1206 / 0.1299 | False / False / False | 0.0238 / 0.0283 | 4.5 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- RAW8-30k - R7-30k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0554 | 0.0570 / 0.0539 | True / True / True | 0.0043 / 0.0060 | 9.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0741 | 0.0763 / 0.0711 | True / True / True | 0.0009 / 0.0050 | 14.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0280 | 0.0159 / 0.0414 | False / False / False | 0.0028 / 0.0210 | 1.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.0277 | -0.0371 / -0.0233 | False / False / False | 0.0056 / 0.0090 | 3.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.0765 | 0.1159 / 0.0495 | False / False / False | 0.0330 / 0.0870 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.0639 | 0.0510 / 0.0401 | False / False / False | 0.0001 / 0.0060 | 10.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.1096 | 0.1167 / 0.1004 | False / False / False | 0.0101 / 0.0226 | 4.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.1230 | 0.1221 / 0.1239 | False / False / False | 0.0280 / 0.0262 | 4.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1366 | 0.1464 / 0.1267 | True / True / True | 0.0006 / 0.0190 | 7.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | -0.0003 | -0.0045 / 0.0040 | False / False / False | 0.0015 / 0.0070 | 0.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0342 | -0.0358 / -0.0327 | False / False / False | 0.0048 / 0.0017 | 7.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0082 | 0.0038 / 0.0127 | False / False / False | 0.0073 / 0.0017 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0839 | 0.0790 / 0.0889 | True / True / True | 0.0097 / 0.0003 | 8.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -0.1751 | -0.1856 / -0.1575 | False / False / False | 0.0520 / 0.0076 | 3.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0638 | 0.0849 / 0.0410 | False / False / False | 0.0238 / 0.0201 | 2.7 | NOT PROVEN (not separated and/or within 2x floor) |

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

### S2 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1584): RAW8-30k bias -0.049, refcv7-50400 bias +0.084, R7-30k bias +0.048; RAW8-30k MAE 0.609, refcv7-50400 MAE 0.622, R7-30k MAE 0.667
* seg2 (n 1584): RAW8-30k bias -0.360, refcv7-50400 bias -0.088, R7-30k bias -0.180; RAW8-30k MAE 1.462, refcv7-50400 MAE 1.446, R7-30k MAE 1.548
* seg3 (n 1579): RAW8-30k bias -0.890, refcv7-50400 bias -0.609, R7-30k bias -0.801; RAW8-30k MAE 2.129, refcv7-50400 MAE 1.979, R7-30k MAE 2.089
* seg4 (n 1572): RAW8-30k bias -1.206, refcv7-50400 bias -0.704, R7-30k bias -1.120; RAW8-30k MAE 2.951, refcv7-50400 MAE 2.673, R7-30k MAE 2.813
**sharp (>= 0.15)**

* seg1 (n 366): RAW8-30k bias -0.354, refcv7-50400 bias +0.010, R7-30k bias -0.123; RAW8-30k MAE 0.914, refcv7-50400 MAE 0.913, R7-30k MAE 0.994
* seg2 (n 366): RAW8-30k bias -1.268, refcv7-50400 bias -0.528, R7-30k bias -0.822; RAW8-30k MAE 2.514, refcv7-50400 MAE 2.413, R7-30k MAE 2.645
* seg3 (n 365): RAW8-30k bias -2.663, refcv7-50400 bias -1.863, R7-30k bias -2.425; RAW8-30k MAE 4.104, refcv7-50400 MAE 3.649, R7-30k MAE 3.969
* seg4 (n 364): RAW8-30k bias -3.409, refcv7-50400 bias -1.860, R7-30k bias -3.199; RAW8-30k MAE 5.720, refcv7-50400 MAE 4.902, R7-30k MAE 5.347
**curved & v0 5-10**

* seg1 (n 627): RAW8-30k bias -0.164, refcv7-50400 bias -0.038, R7-30k bias -0.061; RAW8-30k MAE 0.532, refcv7-50400 MAE 0.535, R7-30k MAE 0.598
* seg2 (n 627): RAW8-30k bias -0.566, refcv7-50400 bias -0.282, R7-30k bias -0.325; RAW8-30k MAE 1.436, refcv7-50400 MAE 1.363, R7-30k MAE 1.525
* seg3 (n 627): RAW8-30k bias -1.185, refcv7-50400 bias -0.871, R7-30k bias -0.963; RAW8-30k MAE 2.293, refcv7-50400 MAE 2.019, R7-30k MAE 2.199
* seg4 (n 627): RAW8-30k bias -1.670, refcv7-50400 bias -1.194, R7-30k bias -1.457; RAW8-30k MAE 3.178, refcv7-50400 MAE 2.730, R7-30k MAE 2.978
**sharp & v0 5-10**

* seg1 (n 164): RAW8-30k bias -0.314, refcv7-50400 bias +0.103, R7-30k bias +0.060; RAW8-30k MAE 0.846, refcv7-50400 MAE 0.857, R7-30k MAE 0.971
* seg2 (n 164): RAW8-30k bias -1.345, refcv7-50400 bias -0.422, R7-30k bias -0.444; RAW8-30k MAE 2.381, refcv7-50400 MAE 2.243, R7-30k MAE 2.557
* seg3 (n 164): RAW8-30k bias -3.089, refcv7-50400 bias -2.032, R7-30k bias -2.198; RAW8-30k MAE 4.108, refcv7-50400 MAE 3.465, R7-30k MAE 3.740
* seg4 (n 164): RAW8-30k bias -4.237, refcv7-50400 bias -2.403, R7-30k bias -3.228; RAW8-30k MAE 5.932, refcv7-50400 MAE 4.635, R7-30k MAE 5.151

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1572 / 1572 | 10.2 | RAW8-30k -0.1449 [-0.2332, -0.0793]; refcv7-50400 -0.0838 [-0.1552, -0.0273]; R7-30k -0.1286 [-0.2016, -0.0655] | RAW8-30k -0.0610 [-0.0966, -0.0259] SEP; R7-30k -0.0448 [-0.0656, -0.0231] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1588 / 4735 | 7.2 | RAW8-30k -0.1968 [-0.3192, -0.0967]; refcv7-50400 -0.1460 [-0.2642, -0.0522]; R7-30k -0.1800 [-0.2997, -0.0841] | RAW8-30k -0.0508 [-0.0826, -0.0188] SEP; R7-30k -0.0340 [-0.0539, -0.0137] SEP |
| sharp (>= 0.15) | last segment | 364 / 364 | 25.4 | RAW8-30k -0.1594 [-0.2619, -0.0822]; refcv7-50400 -0.0926 [-0.1766, -0.0261]; R7-30k -0.1439 [-0.2292, -0.0711] | RAW8-30k -0.0669 [-0.1076, -0.0291] SEP; R7-30k -0.0513 [-0.0760, -0.0269] SEP |
| sharp (>= 0.15) | segments 2..K | 367 / 1095 | 17.9 | RAW8-30k -0.1653 [-0.2672, -0.0862]; refcv7-50400 -0.1043 [-0.1876, -0.0368]; R7-30k -0.1457 [-0.2338, -0.0702] | RAW8-30k -0.0610 [-0.0986, -0.0249] SEP; R7-30k -0.0414 [-0.0650, -0.0173] SEP |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | RAW8-30k -0.1571 [-0.2195, -0.1034]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; R7-30k -0.1170 [-0.1981, -0.0539] | RAW8-30k -0.0646 [-0.1122, -0.0116] SEP; R7-30k -0.0245 [-0.0621, +0.0105] |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | RAW8-30k -0.1568 [-0.2226, -0.1030]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; R7-30k -0.1147 [-0.1962, -0.0510] | RAW8-30k -0.0576 [-0.0997, -0.0095] SEP; R7-30k -0.0154 [-0.0515, +0.0188] |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | RAW8-30k -0.1645 [-0.2361, -0.1056]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; R7-30k -0.1195 [-0.2080, -0.0502] | RAW8-30k -0.0733 [-0.1247, -0.0181] SEP; R7-30k -0.0282 [-0.0696, +0.0083] |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | RAW8-30k -0.1633 [-0.2388, -0.1026]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; R7-30k -0.1131 [-0.2023, -0.0424] | RAW8-30k -0.0678 [-0.1149, -0.0177] SEP; R7-30k -0.0176 [-0.0578, +0.0180] |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**RAW8-30k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4506 | 0.1104 | 0.01489 | 0.02701 | 0.546 / 0.454 / 0.000 | 366.4 / 497.4 |
| turn|straight <0.02 | 2898 | 0.1057 | 0.01847 | 0.03184 | 0.579 / 0.421 / 0.000 | 190.0 / 306.3 |
| turn|gentle 0.02-0.06 | 720 | 0.0466 | -0.02333 | -0.01729 | 0.457 / 0.543 / 0.000 | 44.3 / 33.6 |
| turn|turn 0.06-0.15 | 505 | 0.0529 | -0.00783 | -0.01502 | 0.495 / 0.505 / 0.000 | 45.4 / 26.7 |
| turn|sharp >=0.15 | 367 | 0.3609 | 0.04484 | 0.15273 | 0.531 / 0.469 / 0.000 | 59.7 / 132.5 |
| turn|undef (GT crawling) | 16 | -0.1014 | 0.02550 | 0.01527 | 0.562 / 0.438 / 0.000 | 0.3 / -1.6 |
| speed|v0 <2 | 217 | 0.1309 | 0.05266 | 0.06691 | 0.571 / 0.429 / 0.000 | 11.4 / 28.4 |
| speed|v0 2-5 | 516 | 0.2083 | 0.02536 | 0.03907 | 0.527 / 0.473 / 0.000 | 79.2 / 107.5 |
| speed|v0 5-10 | 1390 | 0.2340 | 0.02509 | 0.06377 | 0.545 / 0.455 / 0.000 | 117.9 / 325.2 |
| speed|v0 10-15 | 1192 | 0.0314 | 0.01400 | 0.02502 | 0.566 / 0.434 / 0.000 | 32.5 / 37.4 |
| speed|v0 15-25 | 856 | -0.0127 | -0.00183 | -0.00371 | 0.495 / 0.505 / 0.000 | 8.3 / -10.8 |
| speed|v0 >=25 | 335 | 0.0291 | 0.01932 | 0.02107 | 0.618 / 0.382 / 0.000 | 2.4 / 9.7 |
| nav|nav=follow | 2881 | 0.1220 | 0.01292 | 0.02042 | 0.547 / 0.453 / 0.000 | 218.5 / 351.5 |
| nav|nav=left | 400 | -0.0258 | 0.01412 | -0.00470 | 0.527 / 0.472 / 0.000 | 26.1 / -10.3 |
| nav|nav=right | 1225 | 0.1276 | 0.02774 | 0.05225 | 0.549 / 0.451 / 0.000 | 115.1 / 156.3 |
**RAW8-30k - R7-30k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4497 | 0.0554 | -0.01312 | -0.00658 | 0.471 / 0.529 / 0.000 | 336.4 / 249.2 |
| turn|straight <0.02 | 2892 | 0.0741 | -0.00639 | 0.00108 | 0.481 / 0.519 / 0.000 | 188.6 / 214.2 |
| turn|gentle 0.02-0.06 | 720 | 0.0280 | -0.04391 | -0.03671 | 0.439 / 0.561 / 0.000 | 43.3 / 20.1 |
| turn|turn 0.06-0.15 | 505 | -0.0277 | -0.06379 | -0.06172 | 0.444 / 0.556 / 0.000 | 48.9 / -14.0 |
| turn|sharp >=0.15 | 367 | 0.0765 | -0.02729 | 0.00880 | 0.477 / 0.523 / 0.000 | 43.5 / 28.1 |
| turn|undef (GT crawling) | 13 | 0.0639 | 0.07999 | 0.10611 | 0.769 / 0.231 / 0.000 | 0.6 / 0.8 |
| speed|v0 <2 | 208 | 0.1096 | 0.05610 | 0.07092 | 0.606 / 0.394 / 0.000 | 10.4 / 22.8 |
| speed|v0 2-5 | 516 | 0.1230 | -0.02919 | 0.00065 | 0.455 / 0.545 / 0.000 | 74.4 / 63.5 |
| speed|v0 5-10 | 1390 | 0.1366 | -0.01553 | 0.01119 | 0.481 / 0.519 / 0.000 | 107.8 / 189.8 |
| speed|v0 10-15 | 1192 | -0.0003 | -0.01010 | -0.01029 | 0.466 / 0.534 / 0.000 | 29.7 / -0.3 |
| speed|v0 15-25 | 856 | -0.0342 | -0.01845 | -0.02714 | 0.437 / 0.563 / 0.000 | 9.4 / -29.3 |
| speed|v0 >=25 | 335 | 0.0082 | -0.00788 | -0.00076 | 0.472 / 0.528 / 0.000 | 2.8 / 2.8 |
| nav|nav=follow | 2873 | 0.0839 | -0.00657 | -0.00066 | 0.483 / 0.517 / 0.000 | 206.8 / 241.1 |
| nav|nav=left | 400 | -0.1751 | -0.06854 | -0.08714 | 0.410 / 0.590 / 0.000 | 25.3 / -70.0 |
| nav|nav=right | 1224 | 0.0638 | -0.02107 | 0.00085 | 0.461 / 0.539 / 0.000 | 103.7 / 78.2 |
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

### S2 -- the episodes carrying the summed delta (panel file index only)

**RAW8-30k - refcv7-50400** (sum 497.4 over 139 episodes)

* ep024: n 35, share 0.109, mean delta 1.5542, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.090, mean delta 1.3169, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep085: n 34, share 0.086, mean delta 1.2600, sharp-window frac 0.62, curved frac 1.00, mean v0 11.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep012: n 34, share 0.060, mean delta 0.8727, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep121: n 35, share 0.056, mean delta 0.7990, sharp-window frac 0.20, curved frac 0.74, mean v0 6.1, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep048: n 34, share 0.053, mean delta 0.7689, sharp-window frac 0.00, curved frac 0.53, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 34, 'nav=right': 0}
**RAW8-30k - R7-30k** (sum 249.2 over 139 episodes)

* ep024: n 35, share 0.216, mean delta 1.5368, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep085: n 34, share 0.137, mean delta 1.0076, sharp-window frac 0.62, curved frac 1.00, mean v0 11.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.130, mean delta 0.9505, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep121: n 35, share 0.102, mean delta 0.7251, sharp-window frac 0.20, curved frac 0.74, mean v0 6.1, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep012: n 34, share 0.096, mean delta 0.7008, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep048: n 34, share 0.091, mean delta 0.6664, sharp-window frac 0.00, curved frac 0.53, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 34, 'nav=right': 0}
**R7-30k - refcv7-50400** (sum 247.4 over 139 episodes)

* ep111: n 34, share 0.121, mean delta 0.8803, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep067: n 34, share 0.064, mean delta 0.4648, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep035: n 34, share 0.063, mean delta 0.4600, sharp-window frac 0.03, curved frac 0.24, mean v0 13.4, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep028: n 34, share 0.056, mean delta 0.4047, sharp-window frac 0.21, curved frac 0.68, mean v0 5.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep080: n 34, share 0.054, mean delta 0.3912, sharp-window frac 0.18, curved frac 0.32, mean v0 9.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep013: n 34, share 0.050, mean delta 0.3664, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**RAW8-30k vs R7-30k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1550 / -0.0254 [-0.0438, -0.0082] | 41 / -0.2129 [-0.5204, 0.0130] |
| only new agrees | 311 / -0.1702 [-0.3036, -0.0591] | 30 / -0.3470 [-1.3790, 0.5591] |
| only earlier agrees | 986 / 0.2083 [0.0993, 0.3382] | 59 / 0.9307 [-0.1657, 2.1939] |
| neither agrees | 1656 / 0.0867 [-0.0236, 0.2084] | 238 / 0.0290 [-0.3525, 0.3558] |
**RAW8-30k vs R7-30k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1562 / -0.0302 [-0.0505, -0.0118] | 46 / -0.1315 [-0.4124, 0.1395] |
| only new agrees | 313 / -0.2178 [-0.3990, -0.0627] | 27 / -1.0106 [-2.5484, 0.0530] |
| only earlier agrees | 978 / 0.2024 [0.1008, 0.3246] | 62 / 0.8846 [-0.0625, 2.0636] |
| neither agrees | 1647 / 0.0971 [-0.0152, 0.2160] | 232 / -0.0145 [-0.3531, 0.3074] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**RAW8-30k vs R7-30k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4472 | 0.1317 [0.0503, 0.2216] | 0.0309 [-0.0266, 0.1012] | 0.1051 [0.0409, 0.1736] SEP |
| turn|straight <0.02 | 2881 | 0.1094 [0.0347, 0.1926] | -0.0240 [-0.0506, 0.0055] | 0.1337 [0.0604, 0.2208] SEP |
| turn|gentle 0.02-0.06 | 719 | 0.0321 [-0.1263, 0.1782] | -0.0180 [-0.1334, 0.0773] | 0.0526 [-0.0614, 0.1751] |
| turn|turn 0.06-0.15 | 499 | 0.1412 [-0.0060, 0.3251] | 0.1191 [-0.0362, 0.3461] | 0.0274 [-0.1160, 0.1775] |
| turn|sharp >=0.15 | 366 | 0.4862 [0.0504, 1.0363] | 0.4356 [0.0750, 0.8390] | 0.0902 [-0.2251, 0.4555] |
| turn|undef (GT crawling) | 7 | 0.3052 [-0.0394, 0.6850] | 0.0519 [-0.1242, 0.2388] | 0.0413 [-0.4088, 0.3833] |
| speed|v0 <2 | 183 | 0.1248 [-0.1304, 0.5065] | 0.3004 [0.1136, 0.5311] | -0.1418 [-0.5504, 0.3147] |
| speed|v0 2-5 | 516 | 0.8156 [0.3166, 1.3105] | 0.2345 [-0.1783, 0.7598] | 0.5811 [0.1814, 0.9885] SEP |
| speed|v0 5-10 | 1390 | 0.1007 [-0.0253, 0.2475] | -0.0151 [-0.0764, 0.0472] | 0.1157 [0.0011, 0.2500] SEP |
| speed|v0 10-15 | 1192 | -0.0023 [-0.0254, 0.0189] | -0.0153 [-0.0289, -0.0004] | 0.0131 [-0.0100, 0.0355] |
| speed|v0 15-25 | 856 | 0.0067 [-0.0068, 0.0214] | -0.0135 [-0.0236, -0.0044] | 0.0201 [0.0053, 0.0351] SEP |
| speed|v0 >=25 | 335 | 0.0074 [-0.0037, 0.0216] | -0.0001 [-0.0096, 0.0089] | 0.0075 [-0.0083, 0.0275] |
**RAW8-30k vs R7-30k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* RAW8-30k vs R7-30k|seed0: anchor flip rate 0.478; heading delta on flipped windows 0.1501 [0.0467, 0.2713] (n 2187) vs unflipped -0.0309 [-0.0527, -0.0099]; goal_dist_sel proxy new-earlier -0.2448 [-0.4586, -0.0378]; corr(window dHeading, window dGoalDist) -0.028
* RAW8-30k vs R7-30k|seed1: anchor flip rate 0.475; heading delta on flipped windows 0.1554 [0.0373, 0.2690] (n 2165) vs unflipped -0.0403 [-0.0611, -0.0203]; goal_dist_sel proxy new-earlier -0.2376 [-0.4540, -0.0294]; corr(window dHeading, window dGoalDist) -0.031

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

### S6 -- RAW8-30k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3531 | 139 | 3.6605 | 3.1980 | 0.4625 [0.2405, 0.6893] | **SEP** |  |
| turn: straight <0.02 | 2104 | 123 | 1.8240 | 1.3821 | 0.4418 [0.2935, 0.6381] | **SEP** | 0.596 / 0.569 / -0.0511 [-0.8689, 0.7214] |
| turn: gentle 0.02-0.06 | 676 | 77 | 3.5739 | 2.8878 | 0.6861 [0.3480, 1.0604] | **SEP** | 0.191 / 0.284 / 0.2766 [-0.3408, 0.9348] |
| turn: turn 0.06-0.15 | 430 | 59 | 6.1508 | 6.3579 | -0.2071 [-0.6264, 0.1953] |  | 0.122 / -0.055 / -0.7624 [-1.4465, -0.1381] BONF-SEP |
| turn: sharp >=0.15 | 307 | 33 | 12.5126 | 11.4442 | 1.0684 [-0.8978, 3.1652] |  | 0.087 / 0.201 / 0.6636 [-1.9429, 3.9307] |
| turn: undef (GT crawling) (low n) | 14 | 9 | 13.2375 | 13.1910 | 0.0464 [-0.2517, 0.3555] |  | 0.004 / 0.000 / -0.4177 [-1.0303, 0.2209] |
| speed: v0 <2 | 193 | 24 | 13.6366 | 12.5929 | 1.0438 [-0.1203, 2.6228] |  | 0.055 / 0.123 / 0.6149 [-1.1028, 3.5062] |
| speed: v0 2-5 | 422 | 52 | 5.1179 | 4.1501 | 0.9677 [0.0911, 1.9561] | **SEP** | 0.120 / 0.250 / 0.5738 [-0.6701, 2.1801] |
| speed: v0 5-10 | 1040 | 79 | 4.1809 | 3.4941 | 0.6868 [0.1610, 1.2264] | **SEP** | 0.295 / 0.437 / 0.3179 [-0.5860, 1.2394] |
| speed: v0 10-15 | 952 | 62 | 2.6652 | 2.4870 | 0.1782 [0.0205, 0.3503] | **SEP** | 0.270 / 0.104 / -0.3892 [-0.9315, 0.0647] |
| speed: v0 15-25 | 679 | 34 | 1.5823 | 1.4456 | 0.1367 [-0.0063, 0.3058] |  | 0.192 / 0.057 / -0.4033 [-0.8814, 0.0244] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.7091 | 0.5193 | 0.1898 [0.0851, 0.2933] | **SEP** | 0.069 / 0.028 / -0.2930 [-0.7140, 0.0583] |
| nav: nav=follow | 2252 | 88 | 2.3219 | 1.9773 | 0.3446 [0.1702, 0.5478] | **SEP** | 0.638 / 0.475 / -0.3254 [-1.2071, 0.5253] |
| nav: nav=left (low n) | 323 | 13 | 9.3040 | 9.1827 | 0.1213 [-1.1689, 1.6356] |  | 0.091 / 0.024 / -0.3755 [-2.4408, 2.0752] |
| nav: nav=right | 956 | 38 | 4.9069 | 4.0515 | 0.8554 [0.3515, 1.4145] | **SEP** | 0.271 / 0.501 / 0.5388 [-0.3076, 1.4999] |

episode concentration: 0.712 of episodes have mean delta > 0; top-5 episodes carry 0.421 of the summed delta; delta excluding the top 5 = 0.2774 (overall 0.4625).

### S6 -- RAW8-30k - R7-30k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3514 | 139 | 3.6574 | 3.3901 | 0.2673 [0.0215, 0.5176] | **SEP** |  |
| turn: straight <0.02 | 2088 | 122 | 1.8040 | 1.4111 | 0.3929 [0.2498, 0.5731] | **SEP** | 0.594 / 0.874 / 0.3097 [-0.4308, 1.2085] |
| turn: gentle 0.02-0.06 | 675 | 77 | 3.5760 | 2.9231 | 0.6529 [0.2609, 1.1018] | **SEP** | 0.192 / 0.469 / 0.4773 [-0.2015, 1.3161] |
| turn: turn 0.06-0.15 | 430 | 59 | 6.1508 | 6.7468 | -0.5960 [-1.1345, -0.1108] | **SEP** | 0.122 / -0.273 / -0.9837 [-1.7597, -0.2271] BONF-SEP |
| turn: sharp >=0.15 | 307 | 33 | 12.5126 | 12.7222 | -0.2096 [-2.3157, 1.9103] |  | 0.087 / -0.069 / -0.5226 [-3.7551, 2.6412] |
| turn: undef (GT crawling) (low n) | 14 | 9 | 13.2375 | 13.3314 | -0.0940 [-0.3710, 0.1259] |  | 0.004 / -0.001 / -0.3627 [-0.9923, 0.1430] |
| speed: v0 <2 | 176 | 24 | 14.5387 | 13.8075 | 0.7312 [-0.8600, 2.4761] |  | 0.050 / 0.137 / 0.4884 [-1.7441, 3.0722] |
| speed: v0 2-5 | 422 | 52 | 5.1179 | 4.4408 | 0.6771 [-0.2865, 1.6336] |  | 0.120 / 0.304 / 0.4657 [-0.8221, 1.9023] |
| speed: v0 5-10 | 1040 | 79 | 4.1809 | 3.7918 | 0.3891 [-0.1747, 0.9297] |  | 0.296 / 0.431 / 0.1730 [-0.7409, 0.9912] |
| speed: v0 10-15 | 952 | 62 | 2.6652 | 2.6243 | 0.0409 [-0.1412, 0.2123] |  | 0.271 / 0.041 / -0.3104 [-0.8802, 0.2918] |
| speed: v0 15-25 | 679 | 34 | 1.5823 | 1.5277 | 0.0546 [-0.0918, 0.2222] |  | 0.193 / 0.039 / -0.2636 [-0.7489, 0.2516] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.7091 | 0.5291 | 0.1800 [0.0621, 0.2913] | **SEP** | 0.070 / 0.047 / -0.0938 [-0.5331, 0.3468] |
| nav: nav=follow | 2241 | 88 | 2.3276 | 2.1153 | 0.2123 [0.0830, 0.3571] | **SEP** | 0.638 / 0.507 / -0.1517 [-1.0904, 0.9107] |
| nav: nav=left (low n) | 320 | 13 | 9.2054 | 9.5693 | -0.3639 [-1.9226, 1.1959] |  | 0.091 / -0.124 / -0.6944 [-3.3802, 1.7808] |
| nav: nav=right | 953 | 38 | 4.9214 | 4.3130 | 0.6084 [-0.0436, 1.3134] |  | 0.271 / 0.617 / 0.4681 [-0.6812, 1.5121] |

episode concentration: 0.705 of episodes have mean delta > 0; top-5 episodes carry 0.662 of the summed delta; delta excluding the top 5 = 0.0936 (overall 0.2673).

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

### S6 -- RAW8-30k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3486 | 139 | 1.1103 | 0.7626 | 0.3477 [0.0131, 0.9886] | **SEP** | 0.0133 [-0.0391, 0.0660] |
| seg1 | straight | 2070 | 122 | 0.8688 | 0.5437 | 0.3251 [0.0364, 0.9085] | **SEP** | -0.2741 [-0.8545, 0.0189] |
| seg1 | curved | 1402 | 88 | 1.3461 | 0.9621 | 0.3840 [-0.0514, 1.2382] |  | 0.4372 [-0.0370, 1.2662] |
| seg2 | all | 3473 | 139 | 1.7262 | 1.4011 | 0.3251 [0.0989, 0.6794] | **SEP** | 0.1333 [-0.1272, 0.4944] |
| seg2 | straight | 2073 | 122 | 1.0619 | 0.7521 | 0.3098 [0.0804, 0.7216] | **SEP** | -0.0689 [-0.1670, 0.0184] |
| seg2 | curved | 1400 | 88 | 2.7099 | 2.3620 | 0.3479 [0.0794, 0.7264] | **SEP** | 0.4326 [-0.2147, 1.2826] |
| seg3 | all | 3438 | 139 | 2.4819 | 1.9978 | 0.4842 [0.1436, 0.9844] | **SEP** | 0.3162 [-0.3899, 1.4334] |
| seg3 | straight | 2052 | 121 | 1.4062 | 0.8905 | 0.5156 [0.0924, 1.3185] | **SEP** | 0.4451 [-0.3477, 1.9796] |
| seg3 | curved | 1386 | 88 | 4.0747 | 3.6370 | 0.4376 [0.0376, 0.8958] | **SEP** | 0.1254 [-0.6271, 0.8513] |
| seg4 | all | 3411 | 139 | 3.5340 | 2.9010 | 0.6329 [0.2578, 1.0936] | **SEP** | -0.1591 [-0.8040, 0.6707] |
| seg4 | straight | 2041 | 121 | 1.6807 | 1.0529 | 0.6278 [0.2428, 1.2715] | **SEP** | 0.0592 [-0.7078, 1.4394] |
| seg4 | curved | 1370 | 87 | 6.2949 | 5.6543 | 0.6406 [0.0475, 1.2769] | **SEP** | -0.4842 [-1.3569, 0.3545] |
| seg5 | all | 3369 | 139 | 4.9731 | 4.1921 | 0.7810 [0.3529, 1.2180] | **SEP** | -0.6593 [-1.3688, 0.1545] |
| seg5 | straight | 2017 | 122 | 1.9966 | 1.1391 | 0.8575 [0.4853, 1.3829] | **SEP** | -0.3864 [-1.1507, 0.9363] |
| seg5 | curved | 1352 | 87 | 9.4136 | 8.7467 | 0.6669 [-0.2035, 1.5296] |  | -1.0666 [-2.2157, 0.0641] |
| seg6 | all | 3331 | 139 | 6.6406 | 5.8505 | 0.7901 [0.3370, 1.2637] | **SEP** | -1.2761 [-2.0265, -0.5271] SEP |
| seg6 | straight | 2011 | 121 | 2.5410 | 1.5628 | 0.9782 [0.6321, 1.3990] | **SEP** | -0.9033 [-1.6131, 0.0371] |
| seg6 | curved | 1320 | 88 | 12.8861 | 12.3826 | 0.5035 [-0.5905, 1.5278] |  | -1.8441 [-3.2908, -0.4666] SEP |

### S6 -- RAW8-30k - R7-30k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3486 | 139 | 1.1103 | 0.8007 | 0.3095 [-0.0231, 0.9447] |  | -0.1149 [-0.1699, -0.0661] SEP |
| seg1 | straight | 2070 | 122 | 0.8687 | 0.5620 | 0.3067 [0.0190, 0.8918] | **SEP** | -0.3701 [-0.9530, -0.0768] SEP |
| seg1 | curved | 1402 | 88 | 1.3461 | 1.0292 | 0.3169 [-0.1088, 1.1495] |  | 0.2614 [-0.2116, 1.1053] |
| seg2 | all | 3468 | 139 | 1.7286 | 1.4208 | 0.3077 [0.0400, 0.7509] | **SEP** | -0.3072 [-0.4470, -0.1766] SEP |
| seg2 | straight | 2068 | 122 | 1.0642 | 0.7864 | 0.2778 [0.0511, 0.6891] | **SEP** | -0.4229 [-0.8386, -0.1813] SEP |
| seg2 | curved | 1400 | 88 | 2.7099 | 2.3579 | 0.3520 [-0.0190, 0.9478] |  | -0.1365 [-0.6452, 0.4988] |
| seg3 | all | 3427 | 139 | 2.4896 | 2.3550 | 0.1346 [-0.0451, 0.3190] |  | -0.6157 [-0.9171, -0.3152] SEP |
| seg3 | straight | 2040 | 121 | 1.4132 | 1.2944 | 0.1188 [0.0191, 0.2470] | **SEP** | -0.3631 [-0.6425, 0.0555] |
| seg3 | curved | 1387 | 88 | 4.0727 | 3.9148 | 0.1578 [-0.2361, 0.5805] |  | -0.9871 [-1.5931, -0.4857] SEP |
| seg4 | all | 3394 | 139 | 3.5484 | 3.2479 | 0.3005 [-0.0001, 0.6108] |  | -1.0425 [-1.4459, -0.6656] SEP |
| seg4 | straight | 2026 | 121 | 1.6893 | 1.2722 | 0.4171 [0.2102, 0.6957] | **SEP** | -0.7429 [-0.9975, -0.4714] SEP |
| seg4 | curved | 1368 | 87 | 6.3017 | 6.1739 | 0.1278 [-0.5064, 0.7467] |  | -1.4862 [-2.3800, -0.6332] SEP |
| seg5 | all | 3351 | 139 | 4.9759 | 4.5446 | 0.4313 [-0.0102, 0.8636] |  | -1.3711 [-1.9225, -0.8366] SEP |
| seg5 | straight | 2001 | 121 | 1.9849 | 1.2324 | 0.7526 [0.4770, 1.1065] | **SEP** | -1.0245 [-1.3720, -0.6743] SEP |
| seg5 | curved | 1350 | 87 | 9.4093 | 9.4542 | -0.0449 [-1.0083, 0.9127] |  | -1.8849 [-3.1371, -0.7256] SEP |
| seg6 | all | 3306 | 139 | 6.6458 | 6.1947 | 0.4511 [-0.1130, 0.9919] |  | -1.6052 [-2.3219, -0.8916] SEP |
| seg6 | straight | 1990 | 120 | 2.5105 | 1.4543 | 1.0562 [0.6668, 1.5181] | **SEP** | -1.2693 [-1.7698, -0.8282] SEP |
| seg6 | curved | 1316 | 88 | 12.8991 | 13.3631 | -0.4640 [-1.6681, 0.7056] |  | -2.1131 [-3.6997, -0.6270] SEP |

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

### S6 -- RAW8-30k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 370 | 40 | 5.3710 | 4.6724 | 0.6986 [0.3127, 1.2018] | **SEP** |  |
| straight <0.02 | v0 5-15 | 1050 | 84 | 1.3089 | 0.7632 | 0.5457 [0.3268, 0.8077] | **SEP** |  |
| straight <0.02 | v0 >=15 | 684 | 36 | 0.6959 | 0.5523 | 0.1436 [0.0833, 0.2102] | **SEP** |  |
| gentle 0.02-0.06 | v0 <5 (low n) | 99 | 22 | 6.9753 | 5.4867 | 1.4887 [0.6828, 2.5809] | **SEP** |  |
| gentle 0.02-0.06 | v0 5-15 | 424 | 59 | 3.1700 | 2.5310 | 0.6390 [0.1880, 1.2084] | **SEP** |  |
| gentle 0.02-0.06 | v0 >=15 (low n) | 153 | 14 | 2.4926 | 2.1952 | 0.2973 [-0.2326, 0.9254] |  |  |
| turn 0.06-0.15 | v0 <5 (low n) | 59 | 20 | 9.9291 | 11.6767 | -1.7476 [-3.1522, -0.6972] | **SEP** |  |
| turn 0.06-0.15 | v0 5-15 | 293 | 43 | 5.8447 | 5.8272 | 0.0176 [-0.4459, 0.4982] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 78 | 10 | 4.4425 | 4.3281 | 0.1144 [-0.2278, 0.5515] |  |  |
| sharp >=0.15 | v0 <5 (low n) | 73 | 16 | 18.3920 | 14.1944 | 4.1976 [-0.8348, 9.7323] |  |  |
| sharp >=0.15 | v0 5-15 | 225 | 26 | 10.9089 | 10.7541 | 0.1549 [-2.0820, 1.9288] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 9 | 3 | 4.9156 | 6.3909 | -1.4753 [-2.2934, 0.1722] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 14 | 9 | 13.2375 | 13.1910 | 0.0464 [-0.2517, 0.3555] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S6 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | RAW8-30k s0-s1 [CI] | R7-30k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0017 [-0.0982, 0.0822] | 0.0185 [-0.0369, 0.0709] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | -0.0190 [-0.1011, 0.0536] | -0.0084 [-0.0264, 0.0108] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | 0.0472 [-0.0969, 0.1891] | 0.0693 [-0.0460, 0.2432] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | 0.1530 [-0.0411, 0.3572] | 0.0928 [-0.0314, 0.2301] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | -0.2071 [-0.9426, 0.4273] | 0.0475 [-0.4823, 0.4413] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | 0.0099 [-0.2210, 0.2320] | -0.9093 [-2.3766, 0.3252] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | -0.6409 [-1.8946, 0.0554] | 0.2784 [-0.0018, 0.6372] | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | -0.1626 [-0.4747, 0.1466] | -0.0736 [-0.4282, 0.2864] | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | 0.1320 [-0.0303, 0.2896] | 0.0440 [-0.0405, 0.1290] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | 0.0516 [-0.0223, 0.1221] | 0.0092 [-0.0307, 0.0556] | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | 0.0088 [-0.0227, 0.0392] | -0.0311 [-0.0639, -0.0040] SEP | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | -0.0197 [-0.0886, 0.0327] | 0.0189 [-0.0164, 0.0666] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | 0.0088 [-0.0476, 0.0626] | 0.0236 [-0.0124, 0.0651] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | -0.1908 [-1.0990, 0.3917] | 0.1426 [-0.0193, 0.3162] | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | 0.0376 [-0.1343, 0.2238] | -0.0347 [-0.1898, 0.1397] | 0.0235 [-0.1175, 0.1775] |

### S6 -- RAW8-30k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.4625 | 0.4507 / 0.4842 | True / True / True | 0.0017 / 0.0128 | 36.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.4418 | 0.4425 / 0.4412 | True / True / True | 0.0190 / 0.0216 | 20.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.6861 | 0.6732 / 0.6990 | True / True / True | 0.0472 / 0.0713 | 9.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | -0.2071 | -0.1352 / -0.2643 | False / False / False | 0.1530 / 0.0611 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 1.0684 | 0.8605 / 1.3738 | False / False / False | 0.2071 / 0.2004 | 5.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.0464 | -0.0025 / 0.1367 | False / False / False | 0.0099 / 1.9278 | 0.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 1.0438 | 0.6181 / 1.6277 | False / True / False | 0.6409 / 0.0328 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.9677 | 0.7860 / 1.1494 | True / True / True | 0.1626 / 0.2009 | 4.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 5-10 | 0.6868 | 0.7649 / 0.6086 | True / True / True | 0.1320 / 0.0243 | 5.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.1782 | 0.2174 / 0.1390 | True / False / True | 0.0516 / 0.0268 | 3.5 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 15-25 | 0.1367 | 0.1384 / 0.1350 | True / False / False | 0.0088 / 0.0055 | 15.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.1898 | 0.1795 / 0.2001 | True / True / True | 0.0197 / 0.0008 | 9.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.3446 | 0.3449 / 0.3446 | True / True / True | 0.0088 / 0.0080 | 39.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.1213 | 0.0062 / 0.3323 | False / False / False | 0.1908 / 0.0140 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.8554 | 0.8499 / 0.8640 | True / True / True | 0.0376 / 0.0235 | 22.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S6 -- RAW8-30k - R7-30k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.2673 | 0.2564 / 0.2801 | True / True / True | 0.0017 / 0.0185 | 14.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.3929 | 0.3870 / 0.3989 | True / True / True | 0.0190 / 0.0084 | 20.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.6529 | 0.6419 / 0.6639 | True / True / True | 0.0472 / 0.0693 | 9.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | -0.5960 | -0.5412 / -0.6290 | True / True / True | 0.1530 / 0.0928 | 3.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | -0.2096 | -0.3392 / -0.0801 | False / False / False | 0.2071 / 0.0475 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | -0.0940 | -0.2045 / 0.0166 | False / False / False | 0.0099 / 0.9093 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.7312 | 0.1958 / 1.2940 | False / False / False | 0.6409 / 0.2784 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.6771 | 0.6326 / 0.7215 | False / False / False | 0.1626 / 0.0736 | 4.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.3891 | 0.4331 / 0.3450 | False / False / False | 0.1320 / 0.0440 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.0409 | 0.0621 / 0.0198 | False / False / False | 0.0516 / 0.0092 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0546 | 0.0745 / 0.0346 | False / False / False | 0.0088 / 0.0311 | 1.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.1800 | 0.1607 / 0.1993 | True / True / True | 0.0197 / 0.0189 | 9.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.2123 | 0.2047 / 0.2200 | True / True / True | 0.0088 / 0.0236 | 9.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -0.3639 | -0.5321 / -0.1957 | False / False / False | 0.1908 / 0.1426 | 1.9 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.6084 | 0.6422 / 0.5806 | True / False / False | 0.0376 / 0.0347 | 16.2 | NOT PROVEN (not separated and/or within 2x floor) |

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

### S6 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1402): RAW8-30k bias +0.369, refcv7-50400 bias +0.173, R7-30k bias +0.114; RAW8-30k MAE 1.346, refcv7-50400 MAE 0.962, R7-30k MAE 1.029
* seg2 (n 1400): RAW8-30k bias -0.233, refcv7-50400 bias -0.215, R7-30k bias -0.372; RAW8-30k MAE 2.710, refcv7-50400 MAE 2.362, R7-30k MAE 2.358
* seg3 (n 1386): RAW8-30k bias -1.282, refcv7-50400 bias -1.069, R7-30k bias -1.427; RAW8-30k MAE 4.075, refcv7-50400 MAE 3.637, R7-30k MAE 3.916
* seg4 (n 1368): RAW8-30k bias -2.660, refcv7-50400 bias -2.454, R7-30k bias -3.310; RAW8-30k MAE 6.302, refcv7-50400 MAE 5.661, R7-30k MAE 6.174
* seg5 (n 1350): RAW8-30k bias -4.124, refcv7-50400 bias -4.409, R7-30k bias -5.223; RAW8-30k MAE 9.409, refcv7-50400 MAE 8.753, R7-30k MAE 9.454
* seg6 (n 1316): RAW8-30k bias -5.908, refcv7-50400 bias -6.878, R7-30k bias -7.296; RAW8-30k MAE 12.899, refcv7-50400 MAE 12.407, R7-30k MAE 13.363
**sharp (>= 0.15)**

* seg1 (n 303): RAW8-30k bias +0.123, refcv7-50400 bias +0.175, R7-30k bias -0.022; RAW8-30k MAE 2.090, refcv7-50400 MAE 1.438, R7-30k MAE 1.542
* seg2 (n 298): RAW8-30k bias -2.114, refcv7-50400 bias -0.532, R7-30k bias -1.535; RAW8-30k MAE 4.641, refcv7-50400 MAE 3.819, R7-30k MAE 4.097
* seg3 (n 298): RAW8-30k bias -4.775, refcv7-50400 bias -2.397, R7-30k bias -4.711; RAW8-30k MAE 8.327, refcv7-50400 MAE 6.641, R7-30k MAE 7.679
* seg4 (n 297): RAW8-30k bias -8.468, refcv7-50400 bias -6.315, R7-30k bias -9.695; RAW8-30k MAE 13.357, refcv7-50400 MAE 11.555, R7-30k MAE 13.573
* seg5 (n 292): RAW8-30k bias -12.349, refcv7-50400 bias -11.391, R7-30k bias -15.025; RAW8-30k MAE 19.377, refcv7-50400 MAE 18.744, R7-30k MAE 20.883
* seg6 (n 287): RAW8-30k bias -16.953, refcv7-50400 bias -17.106, R7-30k bias -19.971; RAW8-30k MAE 25.195, refcv7-50400 MAE 25.140, R7-30k MAE 27.590
**curved & v0 5-10**

* seg1 (n 547): RAW8-30k bias -0.167, refcv7-50400 bias +0.049, R7-30k bias -0.010; RAW8-30k MAE 0.951, refcv7-50400 MAE 0.896, R7-30k MAE 0.993
* seg2 (n 547): RAW8-30k bias -0.978, refcv7-50400 bias -0.516, R7-30k bias -0.769; RAW8-30k MAE 2.693, refcv7-50400 MAE 2.231, R7-30k MAE 2.434
* seg3 (n 543): RAW8-30k bias -2.176, refcv7-50400 bias -1.641, R7-30k bias -2.282; RAW8-30k MAE 4.439, refcv7-50400 MAE 3.822, R7-30k MAE 4.226
* seg4 (n 533): RAW8-30k bias -3.758, refcv7-50400 bias -3.572, R7-30k bias -4.377; RAW8-30k MAE 6.698, refcv7-50400 MAE 6.228, R7-30k MAE 6.893
* seg5 (n 519): RAW8-30k bias -5.586, refcv7-50400 bias -5.848, R7-30k bias -6.448; RAW8-30k MAE 9.822, refcv7-50400 MAE 9.351, R7-30k MAE 10.164
* seg6 (n 489): RAW8-30k bias -7.548, refcv7-50400 bias -8.544, R7-30k bias -8.596; RAW8-30k MAE 13.296, refcv7-50400 MAE 12.908, R7-30k MAE 13.941
**sharp & v0 5-10**

* seg1 (n 139): RAW8-30k bias -0.643, refcv7-50400 bias +0.062, R7-30k bias +0.021; RAW8-30k MAE 1.403, refcv7-50400 MAE 1.251, R7-30k MAE 1.454
* seg2 (n 139): RAW8-30k bias -2.807, refcv7-50400 bias -1.267, R7-30k bias -1.794; RAW8-30k MAE 4.668, refcv7-50400 MAE 3.564, R7-30k MAE 3.957
* seg3 (n 139): RAW8-30k bias -6.117, refcv7-50400 bias -4.133, R7-30k bias -5.957; RAW8-30k MAE 8.696, refcv7-50400 MAE 7.029, R7-30k MAE 8.355
* seg4 (n 138): RAW8-30k bias -10.559, refcv7-50400 bias -9.099, R7-30k bias -11.691; RAW8-30k MAE 13.483, refcv7-50400 MAE 12.470, R7-30k MAE 14.579
* seg5 (n 136): RAW8-30k bias -14.415, refcv7-50400 bias -14.484, R7-30k bias -16.561; RAW8-30k MAE 19.392, refcv7-50400 MAE 19.898, R7-30k MAE 22.220
* seg6 (n 131): RAW8-30k bias -17.747, refcv7-50400 bias -19.043, R7-30k bias -19.872; RAW8-30k MAE 25.028, refcv7-50400 MAE 26.350, R7-30k MAE 28.485

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1316 / 1316 | 26.2 | RAW8-30k -0.2284 [-0.3029, -0.1487]; refcv7-50400 -0.2460 [-0.3210, -0.1783]; R7-30k -0.2780 [-0.3598, -0.2025] | RAW8-30k +0.0175 [-0.0394, +0.0818]; R7-30k -0.0321 [-0.0676, +0.0030] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1411 / 6820 | 16.4 | RAW8-30k -0.1598 [-0.2488, -0.0592]; refcv7-50400 -0.1765 [-0.2427, -0.1178]; R7-30k -0.2014 [-0.2837, -0.1247] | RAW8-30k +0.0167 [-0.0606, +0.1202]; R7-30k -0.0249 [-0.0726, +0.0354] |
| sharp (>= 0.15) | last segment | 287 / 287 | 63.1 | RAW8-30k -0.2680 [-0.3585, -0.1793]; refcv7-50400 -0.2613 [-0.3519, -0.1850]; R7-30k -0.3088 [-0.4083, -0.2234] | RAW8-30k -0.0067 [-0.0788, +0.0677]; R7-30k -0.0475 [-0.0938, -0.0027] SEP |
| sharp (>= 0.15) | segments 2..K | 306 / 1472 | 39.8 | RAW8-30k -0.2341 [-0.3276, -0.1503]; refcv7-50400 -0.1910 [-0.2760, -0.1198]; R7-30k -0.2564 [-0.3531, -0.1772] | RAW8-30k -0.0431 [-0.1144, +0.0293]; R7-30k -0.0654 [-0.1100, -0.0241] SEP |
| curved & v0 5-10 | last segment | 489 / 489 | 28.3 | RAW8-30k -0.2550 [-0.3165, -0.1852]; refcv7-50400 -0.2755 [-0.4018, -0.1730]; R7-30k -0.2947 [-0.4036, -0.2026] | RAW8-30k +0.0205 [-0.0764, +0.1413]; R7-30k -0.0192 [-0.0805, +0.0398] |
| curved & v0 5-10 | segments 2..K | 547 / 2631 | 17.8 | RAW8-30k -0.2194 [-0.2729, -0.1612]; refcv7-50400 -0.2068 [-0.3222, -0.1203]; R7-30k -0.2433 [-0.3463, -0.1605] | RAW8-30k -0.0126 [-0.1002, +0.1032]; R7-30k -0.0365 [-0.0924, +0.0154] |
| sharp & v0 5-10 | last segment | 131 / 131 | 67.0 | RAW8-30k -0.2717 [-0.3371, -0.2025]; refcv7-50400 -0.2746 [-0.4120, -0.1619]; R7-30k -0.3031 [-0.4255, -0.2035] | RAW8-30k +0.0029 [-0.0978, +0.1351]; R7-30k -0.0285 [-0.1038, +0.0432] |
| sharp & v0 5-10 | segments 2..K | 139 / 683 | 43.7 | RAW8-30k -0.2403 [-0.3002, -0.1812]; refcv7-50400 -0.2122 [-0.3465, -0.1139]; R7-30k -0.2606 [-0.3821, -0.1663] | RAW8-30k -0.0281 [-0.1279, +0.1009]; R7-30k -0.0484 [-0.1188, +0.0177] |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**RAW8-30k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3531 | 0.4625 | 0.10445 | 0.18147 | 0.606 / 0.394 / 0.000 | 1123.3 / 1633.0 |
| turn|straight <0.02 | 2104 | 0.4418 | 0.13081 | 0.20197 | 0.673 / 0.327 / 0.000 | 359.0 / 929.6 |
| turn|gentle 0.02-0.06 | 676 | 0.6861 | 0.08561 | 0.20459 | 0.564 / 0.436 / 0.000 | 165.3 / 463.8 |
| turn|turn 0.06-0.15 | 430 | -0.2071 | -0.06025 | -0.12357 | 0.472 / 0.528 / 0.000 | 55.7 / -89.0 |
| turn|sharp >=0.15 | 307 | 1.0684 | -0.54157 | -0.12608 | 0.430 / 0.570 / 0.000 | 226.8 / 328.0 |
| turn|undef (GT crawling) | 14 | 0.0464 | 0.05435 | 0.09085 | 0.500 / 0.500 / 0.000 | 1.2 / 0.7 |
| speed|v0 <2 | 193 | 1.0438 | 0.16508 | 0.29324 | 0.601 / 0.399 / 0.000 | 87.5 / 201.5 |
| speed|v0 2-5 | 422 | 0.9677 | 0.23210 | 0.43841 | 0.645 / 0.355 / 0.000 | 192.9 / 408.4 |
| speed|v0 5-10 | 1040 | 0.6868 | 0.11776 | 0.24067 | 0.583 / 0.417 / 0.000 | 400.2 / 714.2 |
| speed|v0 10-15 | 952 | 0.1782 | 0.10320 | 0.16124 | 0.620 / 0.380 / 0.000 | 111.0 / 169.6 |
| speed|v0 15-25 | 679 | 0.1367 | 0.05960 | 0.06718 | 0.568 / 0.432 / 0.000 | 58.7 / 92.8 |
| speed|v0 >=25 | 245 | 0.1898 | 0.08719 | 0.13547 | 0.690 / 0.310 / 0.000 | 8.2 / 46.5 |
| nav|nav=follow | 2252 | 0.3446 | 0.06304 | 0.08928 | 0.586 / 0.414 / 0.000 | 517.9 / 776.1 |
| nav|nav=left | 323 | 0.1213 | 0.12905 | 0.11736 | 0.579 / 0.421 / 0.000 | 150.5 / 39.2 |
| nav|nav=right | 956 | 0.8554 | 0.45051 | 0.43687 | 0.661 / 0.339 / 0.000 | 361.3 / 817.7 |
**RAW8-30k - R7-30k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3514 | 0.2673 | 0.06188 | 0.11230 | 0.553 / 0.447 / 0.000 | 1101.2 / 939.1 |
| turn|straight <0.02 | 2088 | 0.3929 | 0.10134 | 0.17198 | 0.623 / 0.377 / 0.000 | 327.8 / 820.4 |
| turn|gentle 0.02-0.06 | 675 | 0.6529 | 0.01046 | 0.12539 | 0.508 / 0.492 / 0.000 | 169.5 / 440.7 |
| turn|turn 0.06-0.15 | 430 | -0.5960 | -0.27278 | -0.39424 | 0.381 / 0.619 / 0.000 | 56.8 / -256.3 |
| turn|sharp >=0.15 | 307 | -0.2096 | -0.82990 | -0.78866 | 0.414 / 0.586 / 0.000 | 219.5 / -64.4 |
| turn|undef (GT crawling) | 14 | -0.0940 | 0.04548 | 0.00274 | 0.571 / 0.429 / 0.000 | 0.5 / -1.3 |
| speed|v0 <2 | 176 | 0.7312 | 0.12219 | 0.13820 | 0.602 / 0.398 / 0.000 | 84.9 / 128.7 |
| speed|v0 2-5 | 422 | 0.6771 | 0.15987 | 0.34890 | 0.602 / 0.398 / 0.000 | 171.2 / 285.7 |
| speed|v0 5-10 | 1040 | 0.3891 | 0.06901 | 0.14722 | 0.540 / 0.460 / 0.000 | 381.8 / 404.6 |
| speed|v0 10-15 | 952 | 0.0409 | 0.03821 | 0.09302 | 0.538 / 0.462 / 0.000 | 114.7 / 39.0 |
| speed|v0 15-25 | 679 | 0.0546 | 0.00558 | 0.02215 | 0.511 / 0.489 / 0.000 | 57.2 / 37.0 |
| speed|v0 >=25 | 245 | 0.1800 | 0.12082 | 0.14554 | 0.657 / 0.343 / 0.000 | 8.3 / 44.1 |
| nav|nav=follow | 2241 | 0.2123 | 0.02700 | 0.04820 | 0.531 / 0.469 / 0.000 | 488.8 / 475.8 |
| nav|nav=left | 320 | -0.3639 | 0.04707 | -0.03036 | 0.528 / 0.472 / 0.000 | 148.9 / -116.4 |
| nav|nav=right | 953 | 0.6084 | 0.34361 | 0.32883 | 0.611 / 0.389 / 0.000 | 350.0 / 579.8 |
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

### S6 -- the episodes carrying the summed delta (panel file index only)

**RAW8-30k - refcv7-50400** (sum 1633.0 over 139 episodes)

* ep013: n 27, share 0.093, mean delta 5.6095, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep136: n 20, share 0.091, mean delta 7.3947, sharp-window frac 0.35, curved frac 0.55, mean v0 1.3, nav {'nav=follow': 0, 'nav=left': 20, 'nav=right': 0}
* ep059: n 27, share 0.080, mean delta 4.8484, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep137: n 25, share 0.080, mean delta 5.2343, sharp-window frac 0.44, curved frac 1.00, mean v0 4.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 25}
* ep085: n 26, share 0.078, mean delta 4.8819, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep111: n 26, share 0.066, mean delta 4.1190, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
**RAW8-30k - R7-30k** (sum 939.1 over 139 episodes)

* ep137: n 25, share 0.145, mean delta 5.4317, sharp-window frac 0.44, curved frac 1.00, mean v0 4.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 25}
* ep136: n 20, share 0.144, mean delta 6.7496, sharp-window frac 0.35, curved frac 0.55, mean v0 1.3, nav {'nav=follow': 0, 'nav=left': 20, 'nav=right': 0}
* ep059: n 27, share 0.133, mean delta 4.6225, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep013: n 27, share 0.129, mean delta 4.5022, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep078: n 26, share 0.112, mean delta 4.0292, sharp-window frac 0.00, curved frac 0.54, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep048: n 27, share 0.095, mean delta 3.3098, sharp-window frac 0.00, curved frac 0.67, mean v0 6.9, nav {'nav=follow': 0, 'nav=left': 27, 'nav=right': 0}
**R7-30k - refcv7-50400** (sum 677.1 over 139 episodes)

* ep111: n 26, share 0.189, mean delta 4.9261, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.183, mean delta 4.7761, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep028: n 27, share 0.094, mean delta 2.3661, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep016: n 22, share 0.084, mean delta 2.5708, sharp-window frac 0.41, curved frac 0.59, mean v0 5.6, nav {'nav=follow': 0, 'nav=left': 22, 'nav=right': 0}
* ep085: n 26, share 0.082, mean delta 2.1458, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep061: n 26, share 0.072, mean delta 1.8755, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**RAW8-30k vs R7-30k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1248 / 0.0691 [0.0128, 0.1266] | 38 / 0.0582 [-0.6850, 0.7203] |
| only new agrees | 262 / -1.1069 [-1.9743, -0.4109] | 24 / -8.9129 [-15.6651, -2.8353] |
| only earlier agrees | 728 / 1.2865 [0.8135, 1.8786] | 36 / 8.1901 [1.8380, 18.5031] |
| neither agrees | 1277 / 0.1318 [-0.3527, 0.5922] | 209 / -0.8961 [-3.2314, 1.2985] |
**RAW8-30k vs R7-30k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1246 / 0.0800 [0.0281, 0.1343] | 39 / -0.1005 [-0.8652, 0.6299] |
| only new agrees | 270 / -1.2604 [-2.1425, -0.5541] | 25 / -9.6920 [-16.4966, -4.4244] |
| only earlier agrees | 738 / 1.1369 [0.7224, 1.6998] | 42 / 6.7325 [1.1163, 14.5494] |
| neither agrees | 1262 / 0.3062 [-0.2623, 0.9101] | 201 / -0.3041 [-3.1822, 2.4222] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**RAW8-30k vs R7-30k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3480 | 0.1673 [-0.1848, 0.5417] | 0.2684 [-0.0013, 0.6224] | -0.0666 [-0.2888, 0.1544] |
| turn|straight <0.02 | 2072 | -0.1885 [-0.3786, -0.0310] | -0.0116 [-0.0377, 0.0146] | -0.1489 [-0.3247, 0.0092] |
| turn|gentle 0.02-0.06 | 674 | -0.5792 [-1.4105, 0.0784] | -0.3040 [-0.7972, 0.0186] | -0.2736 [-0.6953, 0.1252] |
| turn|turn 0.06-0.15 | 427 | 0.2374 [-0.2171, 0.7609] | 0.1798 [-0.2220, 0.6523] | 0.1501 [-0.2151, 0.5710] |
| turn|sharp >=0.15 | 300 | 4.1922 [1.3227, 7.4611] | 3.5454 [1.2035, 6.9651] | 0.6417 [-1.4066, 2.9202] |
| turn|undef (GT crawling) | 7 | 0.7119 [0.1396, 1.3784] | 0.0560 [-0.2548, 0.3435] | 0.6633 [-0.0591, 1.6874] |
| speed|v0 <2 | 142 | 0.7792 [-0.8007, 3.4526] | 0.5921 [-0.0256, 1.2883] | 0.9662 [-0.4911, 3.1047] |
| speed|v0 2-5 | 422 | 0.7467 [-0.8361, 2.5321] | 1.1090 [-0.5204, 3.3488] | -0.3623 [-1.1966, 0.3994] |
| speed|v0 5-10 | 1040 | 0.3952 [-0.2579, 1.1073] | 0.3891 [0.0293, 0.8012] | 0.0061 [-0.5589, 0.6371] |
| speed|v0 10-15 | 952 | -0.2802 [-0.4593, -0.1415] | -0.0485 [-0.1090, 0.0053] | -0.2317 [-0.3837, -0.1078] SEP |
| speed|v0 15-25 | 679 | -0.0080 [-0.0790, 0.0632] | 0.0007 [-0.0284, 0.0271] | -0.0087 [-0.0857, 0.0683] |
| speed|v0 >=25 | 245 | 0.0367 [-0.0098, 0.1058] | 0.0198 [-0.0076, 0.0472] | 0.0169 [-0.0433, 0.0998] |
**RAW8-30k vs R7-30k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* RAW8-30k vs R7-30k|seed0: anchor flip rate 0.469; heading delta on flipped windows 0.5377 [0.0436, 1.0353] (n 1656) vs unflipped 0.0058 [-0.0493, 0.0575]; goal_dist_sel proxy new-earlier -0.2132 [-0.4418, 0.0108]; corr(window dHeading, window dGoalDist) -0.024
* RAW8-30k vs R7-30k|seed1: anchor flip rate 0.468; heading delta on flipped windows 0.5718 [0.0002, 1.1732] (n 1651) vs unflipped 0.0219 [-0.0366, 0.0838]; goal_dist_sel proxy new-earlier -0.2084 [-0.4374, 0.0171]; corr(window dHeading, window dGoalDist) -0.026
