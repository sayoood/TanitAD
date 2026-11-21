#### T1-kin DAG - FLAT, seed 0

| family | row | delta [lo, hi] | sep | verdict | arm mean | comparator mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.5957 [-0.7378, -0.4568] | yes | BETTER | 2.2777 | 2.8734 | 2399/141 |
| LONGITUDINAL | speed MAE | -0.2816 [-0.3844, -0.1730] | yes | BETTER | 1.3115 | 1.5931 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.3693 [-0.5082, -0.2327] | yes | BETTER | 1.8299 | 2.1993 | 2399/141 |
| LONGITUDINAL | accel MAE | +0.2652 [-0.0439, +0.5894] | no | ns | 2.8970 | 2.6318 | 2399/141 |
| LONGITUDINAL | DK-VIOL | +0.0487 [+0.0187, +0.0858] | yes | **WORSE** | - | - | 595/70 |
| LONGITUDINAL | DK-HW | +0.1830 [-0.1940, +0.5084] | no | ns | - | - | 700/72 |
| LONGITUDINAL | DK-TG | +0.0824 [+0.0043, +0.1799] | yes | **WORSE** | - | - | 653/72 |
| LATERAL | heading MAE | -5.1913 [-6.8264, -3.7970] | yes | BETTER | 6.1932 | 11.3844 | 2373/141 |
| LATERAL | yaw-rate MAE | -10.6960 [-14.6827, -7.3158] | yes | BETTER | 19.8327 | 30.5287 | 2365/141 |
| LATERAL | curvature MAE | -0.0807 [-0.1112, -0.0526] | yes | BETTER | 0.1336 | 0.2143 | 2365/141 |
| LATERAL | cross-track MAE | -0.4146 [-0.5184, -0.3170] | yes | BETTER | 0.9402 | 1.3549 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -1.6683 [-2.4296, -0.9665] | yes | BETTER | 1.9749 | 3.6432 | 2365/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0544 [-0.0899, -0.0262] | yes | BETTER | 0.0454 | 0.0997 | 2105/135 |
| TACTICAL | TAC-LAT | +0.1747 [+0.1321, +0.2186] | yes | BETTER | - | - | 2399/141 |
| TACTICAL | TAC-LON | +0.1046 [+0.0599, +0.1521] | yes | BETTER | - | - | 2399/141 |
| crawl | M1 crawl share | -0.0368 [-0.0515, -0.0231] | yes | BETTER | 0.0695 | 0.1064 | 2105/135 |

#### T1-kin DAG - FLAT, seed 1

| family | row | delta [lo, hi] | sep | verdict | arm mean | comparator mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.3505 [-0.4695, -0.2293] | yes | BETTER | 2.4565 | 2.8070 | 2399/141 |
| LONGITUDINAL | speed MAE | +0.0899 [-0.0156, +0.1943] | no | ns | 1.5860 | 1.4961 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.0802 [-0.1958, +0.0371] | no | ns | 2.0010 | 2.0812 | 2399/141 |
| LONGITUDINAL | accel MAE | +0.9470 [+0.6198, +1.3196] | yes | **WORSE** | 3.5335 | 2.5865 | 2399/141 |
| LONGITUDINAL | DK-VIOL | +0.0202 [+0.0000, +0.0457] | no | ns | - | - | 595/70 |
| LONGITUDINAL | DK-HW | +0.0452 [-0.2250, +0.2827] | no | ns | - | - | 707/72 |
| LONGITUDINAL | DK-TG | -0.0086 [-0.0444, +0.0237] | no | ns | - | - | 651/72 |
| LATERAL | heading MAE | -3.5617 [-4.4483, -2.7479] | yes | BETTER | 7.9534 | 11.5152 | 2373/141 |
| LATERAL | yaw-rate MAE | -2.6142 [-4.9736, -0.3422] | yes | BETTER | 28.5328 | 31.1471 | 2365/141 |
| LATERAL | curvature MAE | -0.0257 [-0.0463, -0.0060] | yes | BETTER | 0.1857 | 0.2114 | 2365/141 |
| LATERAL | cross-track MAE | -0.4281 [-0.5309, -0.3362] | yes | BETTER | 0.9927 | 1.4207 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -0.3808 [-0.9681, +0.2251] | no | ns | 3.2711 | 3.6518 | 2365/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0121 [-0.0348, +0.0070] | no | ns | 0.0766 | 0.0886 | 2105/135 |
| TACTICAL | TAC-LAT | +0.1701 [+0.1280, +0.2113] | yes | BETTER | - | - | 2399/141 |
| TACTICAL | TAC-LON | -0.0046 [-0.0455, +0.0363] | no | ns | - | - | 2399/141 |
| crawl | M1 crawl share | -0.0152 [-0.0269, -0.0045] | yes | BETTER | 0.0840 | 0.0991 | 2105/135 |

#### T1-kin DAG-0 - FLAT (beside), seed 0

| family | row | delta [lo, hi] | sep | verdict | arm mean | comparator mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.0522 [-0.2432, +0.1675] | no | ns | 2.8212 | 2.8734 | 2399/141 |
| LONGITUDINAL | speed MAE | +0.1842 [+0.0212, +0.3649] | yes | **WORSE** | 1.7773 | 1.5931 | 2399/141 |
| LONGITUDINAL | along-track MAE | +0.0323 [-0.1800, +0.2768] | no | ns | 2.2316 | 2.1993 | 2399/141 |
| LONGITUDINAL | accel MAE | +0.2846 [+0.0524, +0.5217] | yes | **WORSE** | 2.9163 | 2.6318 | 2399/141 |
| LONGITUDINAL | DK-VIOL | +0.0067 [-0.0083, +0.0225] | no | ns | - | - | 595/70 |
| LONGITUDINAL | DK-HW | -0.0310 [-0.2968, +0.2088] | no | ns | - | - | 695/72 |
| LONGITUDINAL | DK-TG | -0.0013 [-0.0360, +0.0323] | no | ns | - | - | 653/72 |
| LATERAL | heading MAE | -1.6774 [-2.3982, -1.0546] | yes | BETTER | 9.7071 | 11.3844 | 2373/141 |
| LATERAL | yaw-rate MAE | -4.2347 [-7.2604, -1.3822] | yes | BETTER | 26.2940 | 30.5287 | 2365/141 |
| LATERAL | curvature MAE | -0.0736 [-0.1074, -0.0431] | yes | BETTER | 0.1407 | 0.2143 | 2365/141 |
| LATERAL | cross-track MAE | -0.1192 [-0.2053, -0.0282] | yes | BETTER | 1.2357 | 1.3549 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | +0.0008 [-0.7415, +0.7797] | no | ns | 3.6440 | 3.6432 | 2365/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0507 [-0.0872, -0.0211] | yes | BETTER | 0.0491 | 0.0997 | 2105/135 |
| TACTICAL | TAC-LAT | +0.0746 [+0.0346, +0.1180] | yes | BETTER | - | - | 2399/141 |
| TACTICAL | TAC-LON | +0.0504 [+0.0133, +0.0891] | yes | BETTER | - | - | 2399/141 |
| crawl | M1 crawl share | -0.0253 [-0.0387, -0.0130] | yes | BETTER | 0.0811 | 0.1064 | 2105/135 |

#### T1-kin DAG-0 - FLAT (beside), seed 1

| family | row | delta [lo, hi] | sep | verdict | arm mean | comparator mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | +0.0541 [-0.1083, +0.2198] | no | ns | 2.8611 | 2.8070 | 2399/141 |
| LONGITUDINAL | speed MAE | +0.3715 [+0.2320, +0.5126] | yes | **WORSE** | 1.8676 | 1.4961 | 2399/141 |
| LONGITUDINAL | along-track MAE | +0.1688 [+0.0064, +0.3444] | yes | **WORSE** | 2.2500 | 2.0812 | 2399/141 |
| LONGITUDINAL | accel MAE | +0.4980 [+0.2979, +0.7073] | yes | **WORSE** | 3.0845 | 2.5865 | 2399/141 |
| LONGITUDINAL | DK-VIOL | -0.0101 [-0.0238, +0.0031] | no | ns | - | - | 595/70 |
| LONGITUDINAL | DK-HW | -0.0634 [-0.2371, +0.0870] | no | ns | - | - | 699/72 |
| LONGITUDINAL | DK-TG | -0.0223 [-0.0530, +0.0046] | no | ns | - | - | 652/72 |
| LATERAL | heading MAE | -1.7971 [-2.5253, -1.1087] | yes | BETTER | 9.7180 | 11.5152 | 2373/141 |
| LATERAL | yaw-rate MAE | -0.9175 [-3.7889, +2.1937] | no | ns | 30.2295 | 31.1471 | 2365/141 |
| LATERAL | curvature MAE | -0.0425 [-0.0671, -0.0213] | yes | BETTER | 0.1689 | 0.2114 | 2365/141 |
| LATERAL | cross-track MAE | -0.1212 [-0.2093, -0.0434] | yes | BETTER | 1.2996 | 1.4207 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | +0.3334 [-0.4027, +1.2276] | no | ns | 3.9853 | 3.6518 | 2365/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0249 [-0.0512, -0.0028] | yes | BETTER | 0.0638 | 0.0886 | 2105/135 |
| TACTICAL | TAC-LAT | -0.0217 [-0.0634, +0.0200] | no | ns | - | - | 2399/141 |
| TACTICAL | TAC-LON | -0.0763 [-0.1222, -0.0312] | yes | **WORSE** | - | - | 2399/141 |
| crawl | M1 crawl share | -0.0070 [-0.0179, +0.0028] | no | ns | 0.0921 | 0.0991 | 2105/135 |

#### T1-kin FLAT - P10 (scale, beside), seed 0

| family | row | delta [lo, hi] | sep | verdict | arm mean | comparator mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.5699 [-0.7725, -0.3740] | yes | BETTER | 2.8734 | 3.4433 | 2399/141 |
| LONGITUDINAL | speed MAE | -0.2949 [-0.4704, -0.1320] | yes | BETTER | 1.5931 | 1.8880 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.2589 [-0.4636, -0.0636] | yes | BETTER | 2.1993 | 2.4581 | 2399/141 |
| LONGITUDINAL | accel MAE | +0.5419 [+0.2435, +0.8197] | yes | **WORSE** | 2.6318 | 2.0899 | 2399/141 |
| LONGITUDINAL | DK-VIOL | -0.0134 [-0.0318, +0.0033] | no | ns | - | - | 595/70 |
| LONGITUDINAL | DK-HW | -0.2735 [-0.5584, -0.0035] | yes | BETTER | - | - | 700/72 |
| LONGITUDINAL | DK-TG | -0.0864 [-0.1454, -0.0380] | yes | BETTER | - | - | 653/72 |
| LATERAL | heading MAE | +1.8319 [+0.7601, +3.0431] | yes | **WORSE** | 11.3844 | 9.5525 | 2373/141 |
| LATERAL | yaw-rate MAE | +10.6518 [+6.9322, +14.6214] | yes | **WORSE** | 30.5624 | 19.9106 | 2366/141 |
| LATERAL | curvature MAE | +0.1420 [+0.1005, +0.1870] | yes | **WORSE** | 0.2156 | 0.0736 | 2366/141 |
| LATERAL | cross-track MAE | -0.5646 [-0.6872, -0.4572] | yes | BETTER | 1.3549 | 1.9195 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -0.7636 [-1.8276, +0.1193] | no | ns | 3.6421 | 4.4057 | 2366/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0832 [+0.0482, +0.1279] | yes | **WORSE** | 0.0997 | 0.0165 | 2105/135 |
| TACTICAL | TAC-LAT | +0.0179 [-0.0271, +0.0633] | no | ns | - | - | 2399/141 |
| TACTICAL | TAC-LON | +0.0242 [-0.0397, +0.0917] | no | ns | - | - | 2399/141 |
| crawl | M1 crawl share | +0.0834 [+0.0581, +0.1121] | yes | **WORSE** | 0.1064 | 0.0230 | 2105/135 |

#### T1-kin FLAT - P10 (scale, beside), seed 1

| family | row | delta [lo, hi] | sep | verdict | arm mean | comparator mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.6363 [-0.8227, -0.4580] | yes | BETTER | 2.8070 | 3.4433 | 2399/141 |
| LONGITUDINAL | speed MAE | -0.3919 [-0.5841, -0.2246] | yes | BETTER | 1.4961 | 1.8880 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.3769 [-0.5617, -0.1987] | yes | BETTER | 2.0812 | 2.4581 | 2399/141 |
| LONGITUDINAL | accel MAE | +0.4966 [+0.1415, +0.8179] | yes | **WORSE** | 2.5865 | 2.0899 | 2399/141 |
| LONGITUDINAL | DK-VIOL | -0.0151 [-0.0374, +0.0037] | no | ns | - | - | 595/70 |
| LONGITUDINAL | DK-HW | -0.3296 [-0.6298, -0.0542] | yes | BETTER | - | - | 707/72 |
| LONGITUDINAL | DK-TG | -0.0795 [-0.1444, -0.0195] | yes | BETTER | - | - | 650/72 |
| LATERAL | heading MAE | +1.9626 [+0.9781, +2.9533] | yes | **WORSE** | 11.5152 | 9.5525 | 2373/141 |
| LATERAL | yaw-rate MAE | +11.2668 [+8.3226, +14.5788] | yes | **WORSE** | 31.1774 | 19.9106 | 2366/141 |
| LATERAL | curvature MAE | +0.1390 [+0.1005, +0.1853] | yes | **WORSE** | 0.2126 | 0.0736 | 2366/141 |
| LATERAL | cross-track MAE | -0.4988 [-0.6234, -0.3801] | yes | BETTER | 1.4207 | 1.9195 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -0.7550 [-1.8966, +0.2146] | no | ns | 3.6508 | 4.4057 | 2366/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0721 [+0.0443, +0.1069] | yes | **WORSE** | 0.0886 | 0.0165 | 2105/135 |
| TACTICAL | TAC-LAT | +0.0150 [-0.0321, +0.0605] | no | ns | - | - | 2399/141 |
| TACTICAL | TAC-LON | +0.1188 [+0.0601, +0.1797] | yes | BETTER | - | - | 2399/141 |
| crawl | M1 crawl share | +0.0762 [+0.0523, +0.1038] | yes | **WORSE** | 0.0991 | 0.0230 | 2105/135 |
