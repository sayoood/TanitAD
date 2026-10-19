# refcv8 heading-drift localisation: R7-20k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-18T06:01:00; trees: `R7-20k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step20000_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **None**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

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

### S2 -- R7-20k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 439 | 47 | 6.0720 | 5.9668 | 0.1052 [-0.0117, 0.2485] |  |  |
| straight <0.02 | v0 5-15 | 1519 | 93 | 0.3996 | 0.3674 | 0.0322 [0.0087, 0.0552] | **SEP** |  |
| straight <0.02 | v0 >=15 | 940 | 43 | 0.1980 | 0.1947 | 0.0033 [-0.0067, 0.0134] |  |  |
| gentle 0.02-0.06 | v0 <5 | 112 | 29 | 1.5431 | 1.7630 | -0.2199 [-0.7762, 0.1547] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 461 | 74 | 0.9881 | 0.9711 | 0.0170 [-0.0278, 0.0593] |  |  |
| gentle 0.02-0.06 | v0 >=15 | 148 | 20 | 0.5514 | 0.5590 | -0.0075 [-0.0655, 0.0572] |  |  |
| turn 0.06-0.15 | v0 <5 (low n) | 82 | 25 | 5.8373 | 5.0600 | 0.7773 [0.2170, 1.4170] | **SEP** |  |
| turn 0.06-0.15 | v0 5-15 | 343 | 54 | 1.6057 | 1.4275 | 0.1782 [0.0537, 0.3182] | **SEP** |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 87 | 11 | 1.1648 | 0.9719 | 0.1929 [0.0440, 0.3978] | **SEP** |  |
| sharp >=0.15 | v0 <5 (low n) | 93 | 16 | 5.9205 | 5.2944 | 0.6261 [-0.6006, 1.9644] |  |  |
| sharp >=0.15 | v0 5-15 | 259 | 33 | 2.5035 | 2.2857 | 0.2178 [0.0200, 0.4402] | **SEP** |  |
| sharp >=0.15 | v0 >=15 (low n) | 16 | 3 | 1.5470 | 1.3866 | 0.1605 [0.1088, 0.3137] | **SEP** |  |
| undef (GT crawling) | v0 <5 (low n) | 21 | 17 | 9.0951 | 9.1504 | -0.0553 [-0.3755, 0.1854] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S2 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | R7-20k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|
| overall | -0.0015 [-0.0157, 0.0150] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | 0.0079 [-0.0056, 0.0266] | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | -0.0271 [-0.0663, 0.0095] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | -0.0187 [-0.0706, 0.0340] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | -0.0092 [-0.1307, 0.1113] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | 0.1339 [-0.0506, 0.3279] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | 0.0129 [-0.1193, 0.1887] | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | 0.0189 [-0.0664, 0.1076] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | -0.0115 [-0.0432, 0.0284] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | 0.0005 [-0.0131, 0.0140] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | -0.0017 [-0.0070, 0.0041] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | -0.0075 [-0.0163, -0.0008] SEP | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | -0.0062 [-0.0176, 0.0054] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | -0.0472 [-0.1021, -0.0024] SEP | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | 0.0250 [-0.0162, 0.0755] | 0.0283 [0.0035, 0.0597] SEP |

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

* seg1 (n 1588): R7-20k bias +0.137, refcv7-50400 bias +0.081; R7-20k MAE 0.712, refcv7-50400 MAE 0.623
* seg2 (n 1592): R7-20k bias -0.008, refcv7-50400 bias -0.100; R7-20k MAE 1.627, refcv7-50400 MAE 1.449
* seg3 (n 1589): R7-20k bias -0.666, refcv7-50400 bias -0.622; R7-20k MAE 2.166, refcv7-50400 MAE 1.986
* seg4 (n 1581): R7-20k bias -0.902, refcv7-50400 bias -0.730; R7-20k MAE 2.825, refcv7-50400 MAE 2.691
**sharp (>= 0.15)**

* seg1 (n 366): R7-20k bias +0.284, refcv7-50400 bias +0.010; R7-20k MAE 1.104, refcv7-50400 MAE 0.913
* seg2 (n 366): R7-20k bias +0.086, refcv7-50400 bias -0.528; R7-20k MAE 2.812, refcv7-50400 MAE 2.413
* seg3 (n 366): R7-20k bias -1.596, refcv7-50400 bias -1.862; R7-20k MAE 4.029, refcv7-50400 MAE 3.643
* seg4 (n 365): R7-20k bias -2.039, refcv7-50400 bias -1.860; R7-20k MAE 5.165, refcv7-50400 MAE 4.893
**curved & v0 5-10**

* seg1 (n 627): R7-20k bias +0.030, refcv7-50400 bias -0.038; R7-20k MAE 0.661, refcv7-50400 MAE 0.535
* seg2 (n 627): R7-20k bias -0.169, refcv7-50400 bias -0.282; R7-20k MAE 1.598, refcv7-50400 MAE 1.363
* seg3 (n 627): R7-20k bias -0.896, refcv7-50400 bias -0.871; R7-20k MAE 2.199, refcv7-50400 MAE 2.019
* seg4 (n 627): R7-20k bias -1.283, refcv7-50400 bias -1.194; R7-20k MAE 2.839, refcv7-50400 MAE 2.730
**sharp & v0 5-10**

* seg1 (n 164): R7-20k bias +0.482, refcv7-50400 bias +0.103; R7-20k MAE 1.100, refcv7-50400 MAE 0.857
* seg2 (n 164): R7-20k bias +0.462, refcv7-50400 bias -0.422; R7-20k MAE 2.699, refcv7-50400 MAE 2.243
* seg3 (n 164): R7-20k bias -1.436, refcv7-50400 bias -2.032; R7-20k MAE 3.740, refcv7-50400 MAE 3.465
* seg4 (n 164): R7-20k bias -2.138, refcv7-50400 bias -2.403; R7-20k MAE 4.837, refcv7-50400 MAE 4.635

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1581 / 1581 | 10.2 | R7-20k -0.1002 [-0.1827, -0.0386]; refcv7-50400 -0.0852 [-0.1565, -0.0286] | R7-20k -0.0149 [-0.0488, +0.0181] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1600 / 4762 | 7.2 | R7-20k -0.1531 [-0.2796, -0.0500]; refcv7-50400 -0.1471 [-0.2660, -0.0526] | R7-20k -0.0060 [-0.0374, +0.0261] |
| sharp (>= 0.15) | last segment | 365 / 365 | 25.3 | R7-20k -0.1064 [-0.2043, -0.0353]; refcv7-50400 -0.0926 [-0.1766, -0.0262] | R7-20k -0.0138 [-0.0530, +0.0249] |
| sharp (>= 0.15) | segments 2..K | 368 / 1097 | 17.9 | R7-20k -0.1064 [-0.2063, -0.0321]; refcv7-50400 -0.1044 [-0.1876, -0.0369] | R7-20k -0.0020 [-0.0421, +0.0386] |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | R7-20k -0.0925 [-0.1693, -0.0394]; refcv7-50400 -0.0925 [-0.1767, -0.0360] | R7-20k -0.0000 [-0.0233, +0.0247] |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | R7-20k -0.0932 [-0.1722, -0.0382]; refcv7-50400 -0.0993 [-0.1849, -0.0421] | R7-20k +0.0061 [-0.0160, +0.0286] |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | R7-20k -0.0853 [-0.1691, -0.0256]; refcv7-50400 -0.0912 [-0.1852, -0.0297] | R7-20k +0.0059 [-0.0173, +0.0316] |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | R7-20k -0.0795 [-0.1657, -0.0181]; refcv7-50400 -0.0955 [-0.1896, -0.0316] | R7-20k +0.0160 [-0.0068, +0.0412] |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)


### S2 -- the episodes carrying the summed delta (panel file index only)


### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed


### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)


### S2 -- selection / goal proxy (new vs earlier, per seed)


## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

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

### S6 -- R7-20k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 365 | 40 | 6.0576 | 5.2135 | 0.8441 [0.5003, 1.3412] | **SEP** |  |
| straight <0.02 | v0 5-15 | 1050 | 84 | 0.9622 | 0.7632 | 0.1990 [0.1047, 0.3084] | **SEP** |  |
| straight <0.02 | v0 >=15 | 684 | 36 | 0.5264 | 0.5523 | -0.0260 [-0.0646, 0.0109] |  |  |
| gentle 0.02-0.06 | v0 <5 | 102 | 22 | 5.3033 | 5.4491 | -0.1458 [-2.4385, 1.4733] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 424 | 59 | 2.6546 | 2.5310 | 0.1236 [-0.0341, 0.2751] |  |  |
| gentle 0.02-0.06 | v0 >=15 (low n) | 153 | 14 | 2.4562 | 2.1952 | 0.2610 [0.0349, 0.4970] | **SEP** |  |
| turn 0.06-0.15 | v0 <5 (low n) | 71 | 20 | 11.5557 | 10.6521 | 0.9036 [-0.1598, 1.9680] |  |  |
| turn 0.06-0.15 | v0 5-15 | 293 | 43 | 6.6423 | 5.8272 | 0.8152 [0.1394, 1.5259] | **SEP** |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 78 | 10 | 5.1693 | 4.3281 | 0.8413 [0.3980, 1.2768] | **SEP** |  |
| sharp >=0.15 | v0 <5 (low n) | 78 | 16 | 20.1197 | 15.1685 | 4.9512 [2.0097, 9.1518] | **SEP** |  |
| sharp >=0.15 | v0 5-15 | 225 | 26 | 12.3076 | 10.7541 | 1.5536 [0.4153, 2.6616] | **SEP** |  |
| sharp >=0.15 | v0 >=15 (low n) | 9 | 3 | 8.1948 | 6.3909 | 1.8039 [0.9857, 3.4599] | **SEP** |  |
| undef (GT crawling) | v0 <5 (low n) | 21 | 11 | 10.3328 | 10.5237 | -0.1909 [-0.5355, 0.1873] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S6 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | R7-20k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|
| overall | 0.0032 [-0.0496, 0.0598] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | 0.0262 [-0.0207, 0.0960] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | 0.0292 [-0.0744, 0.1218] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | -0.0675 [-0.2282, 0.1022] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | -0.1077 [-0.5568, 0.3422] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | 0.0091 [-0.1871, 0.2769] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | 0.4396 [0.0260, 1.0407] SEP | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | -0.1181 [-0.4121, 0.0983] | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | 0.0358 [-0.1045, 0.1956] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | -0.0728 [-0.1371, -0.0117] SEP | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | 0.0008 [-0.0513, 0.0624] | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | -0.0123 [-0.0357, 0.0129] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | -0.0214 [-0.0692, 0.0290] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | 0.0776 [0.0039, 0.1823] SEP | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | 0.0340 [-0.1064, 0.1917] | 0.0235 [-0.1175, 0.1775] |

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

* seg1 (n 1410): R7-20k bias +0.395, refcv7-50400 bias +0.037; R7-20k MAE 1.470, refcv7-50400 MAE 1.092
* seg2 (n 1411): R7-20k bias -0.047, refcv7-50400 bias -0.242; R7-20k MAE 2.539, refcv7-50400 MAE 2.376
* seg3 (n 1405): R7-20k bias -1.165, refcv7-50400 bias -1.126; R7-20k MAE 4.073, refcv7-50400 MAE 3.671
* seg4 (n 1387): R7-20k bias -3.141, refcv7-50400 bias -2.557; R7-20k MAE 6.638, refcv7-50400 MAE 5.694
* seg5 (n 1376): R7-20k bias -5.408, refcv7-50400 bias -4.502; R7-20k MAE 10.310, refcv7-50400 MAE 8.790
* seg6 (n 1350): R7-20k bias -7.583, refcv7-50400 bias -6.962; R7-20k MAE 14.523, refcv7-50400 MAE 12.343
**sharp (>= 0.15)**

* seg1 (n 305): R7-20k bias +0.382, refcv7-50400 bias -0.421; R7-20k MAE 2.973, refcv7-50400 MAE 2.024
* seg2 (n 302): R7-20k bias -1.246, refcv7-50400 bias -0.568; R7-20k MAE 4.330, refcv7-50400 MAE 3.821
* seg3 (n 303): R7-20k bias -4.891, refcv7-50400 bias -2.527; R7-20k MAE 8.128, refcv7-50400 MAE 6.701
* seg4 (n 302): R7-20k bias -10.581, refcv7-50400 bias -6.666; R7-20k MAE 14.691, refcv7-50400 MAE 11.654
* seg5 (n 299): R7-20k bias -16.892, refcv7-50400 bias -11.739; R7-20k MAE 23.325, refcv7-50400 MAE 18.957
* seg6 (n 293): R7-20k bias -21.695, refcv7-50400 bias -17.555; R7-20k MAE 31.256, refcv7-50400 MAE 25.217
**curved & v0 5-10**

* seg1 (n 547): R7-20k bias +0.199, refcv7-50400 bias +0.049; R7-20k MAE 1.009, refcv7-50400 MAE 0.896
* seg2 (n 547): R7-20k bias -0.417, refcv7-50400 bias -0.516; R7-20k MAE 2.356, refcv7-50400 MAE 2.231
* seg3 (n 545): R7-20k bias -1.729, refcv7-50400 bias -1.649; R7-20k MAE 4.200, refcv7-50400 MAE 3.822
* seg4 (n 536): R7-20k bias -3.780, refcv7-50400 bias -3.537; R7-20k MAE 7.042, refcv7-50400 MAE 6.217
* seg5 (n 526): R7-20k bias -6.161, refcv7-50400 bias -5.847; R7-20k MAE 10.649, refcv7-50400 MAE 9.313
* seg6 (n 507): R7-20k bias -8.184, refcv7-50400 bias -8.421; R7-20k MAE 14.674, refcv7-50400 MAE 12.654
**sharp & v0 5-10**

* seg1 (n 139): R7-20k bias +0.642, refcv7-50400 bias +0.062; R7-20k MAE 1.538, refcv7-50400 MAE 1.251
* seg2 (n 139): R7-20k bias -0.900, refcv7-50400 bias -1.267; R7-20k MAE 3.694, refcv7-50400 MAE 3.564
* seg3 (n 139): R7-20k bias -5.173, refcv7-50400 bias -4.133; R7-20k MAE 7.618, refcv7-50400 MAE 7.029
* seg4 (n 139): R7-20k bias -11.043, refcv7-50400 bias -8.959; R7-20k MAE 14.295, refcv7-50400 MAE 12.455
* seg5 (n 138): R7-20k bias -16.303, refcv7-50400 bias -14.317; R7-20k MAE 22.718, refcv7-50400 MAE 19.688
* seg6 (n 134): R7-20k bias -18.444, refcv7-50400 bias -18.791; R7-20k MAE 30.042, refcv7-50400 MAE 25.934

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1350 / 1350 | 26.0 | R7-20k -0.2901 [-0.3746, -0.2172]; refcv7-50400 -0.2496 [-0.3225, -0.1834] | R7-20k -0.0405 [-0.0866, -0.0045] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1432 / 6929 | 16.3 | R7-20k -0.1992 [-0.2808, -0.1220]; refcv7-50400 -0.1807 [-0.2466, -0.1219] | R7-20k -0.0185 [-0.0741, +0.0549] |
| sharp (>= 0.15) | last segment | 293 / 293 | 62.9 | R7-20k -0.3221 [-0.4325, -0.2366]; refcv7-50400 -0.2657 [-0.3542, -0.1889] | R7-20k -0.0563 [-0.1207, -0.0116] SEP |
| sharp (>= 0.15) | segments 2..K | 311 / 1499 | 39.6 | R7-20k -0.2673 [-0.3715, -0.1847]; refcv7-50400 -0.1961 [-0.2823, -0.1242] | R7-20k -0.0712 [-0.1269, -0.0303] SEP |
| curved & v0 5-10 | last segment | 507 / 507 | 28.0 | R7-20k -0.2781 [-0.4036, -0.1688]; refcv7-50400 -0.2729 [-0.3978, -0.1735] | R7-20k -0.0052 [-0.0540, +0.0349] |
| curved & v0 5-10 | segments 2..K | 547 / 2661 | 17.8 | R7-20k -0.2261 [-0.3318, -0.1363]; refcv7-50400 -0.2058 [-0.3196, -0.1201] | R7-20k -0.0203 [-0.0591, +0.0147] |
| sharp & v0 5-10 | last segment | 134 / 134 | 67.1 | R7-20k -0.2749 [-0.4142, -0.1557]; refcv7-50400 -0.2707 [-0.4071, -0.1612] | R7-20k -0.0042 [-0.0645, +0.0431] |
| sharp & v0 5-10 | segments 2..K | 139 / 689 | 43.8 | R7-20k -0.2377 [-0.3601, -0.1311]; refcv7-50400 -0.2104 [-0.3435, -0.1137] | R7-20k -0.0272 [-0.0792, +0.0153] |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)


### S6 -- the episodes carrying the summed delta (panel file index only)


### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed


### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)


### S6 -- selection / goal proxy (new vs earlier, per seed)

