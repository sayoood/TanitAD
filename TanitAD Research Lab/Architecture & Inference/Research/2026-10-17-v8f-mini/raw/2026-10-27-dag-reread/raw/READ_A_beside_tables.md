#### DAG-0 - P10 (beside), seed 0

| family | row | arm - P10 [lo, hi] | sep | verdict | arm mean | P10 mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.6221 [-0.8551, -0.3711] | yes | BETTER | 2.8212 | 3.4433 | 2399/141 |
| LONGITUDINAL | speed MAE | -0.1107 [-0.3036, +0.0902] | no | ns | 1.7773 | 1.8880 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.2265 [-0.4531, +0.0446] | no | ns | 2.2316 | 2.4581 | 2399/141 |
| LONGITUDINAL | accel MAE | +0.8265 [+0.4972, +1.1391] | yes | **WORSE** | 2.9163 | 2.0899 | 2399/141 |
| LONGITUDINAL | DK-VIOL | -0.0067 [-0.0286, +0.0141] | no | ns | 0.0403 | 0.0471 | 595/70 |
| LONGITUDINAL | DK-HW | -0.2894 [-0.6484, +0.0186] | no | ns | 2.0731 | 2.3625 | 702/72 |
| LONGITUDINAL | DK-TG | -0.0872 [-0.1483, -0.0394] | yes | BETTER | 0.3139 | 0.4012 | 652/72 |
| LATERAL | heading MAE | +0.1546 [-0.6445, +1.0125] | no | ns | 9.7071 | 9.5525 | 2373/141 |
| LATERAL | yaw-rate MAE | +6.4213 [+4.2450, +8.6676] | yes | **WORSE** | 26.2940 | 19.8727 | 2365/141 |
| LATERAL | curvature MAE | +0.0683 [+0.0477, +0.0907] | yes | **WORSE** | 0.1407 | 0.0724 | 2365/141 |
| LATERAL | cross-track MAE | -0.6838 [-0.8103, -0.5642] | yes | BETTER | 1.2357 | 1.9195 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -0.7631 [-1.4492, -0.1401] | yes | BETTER | 3.6440 | 4.4071 | 2365/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0325 [+0.0204, +0.0479] | yes | **WORSE** | 0.0491 | 0.0165 | 2106/135 |
| TACTICAL | TAC-LAT | +0.0925 [+0.0563, +0.1326] | yes | BETTER | 0.6857 | 0.5932 | 2399/141 |
| TACTICAL | TAC-LON | +0.0746 [+0.0037, +0.1430] | yes | BETTER | 0.4460 | 0.3714 | 2399/141 |
| STRATEGIC | S-POS@30 | +0.1178 [-0.2696, +0.4935] | no | ns | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@50 | -0.9084 [-1.9668, +0.0696] | no | ns | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@80 | -3.8923 [-6.0888, -1.8480] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| crawl | M1 crawl share | +0.0586 [+0.0405, +0.0792] | yes | **WORSE** | 0.0815 | 0.0230 | 2106/135 |

#### DAG-0 - P10 (beside), seed 1

| family | row | arm - P10 [lo, hi] | sep | verdict | arm mean | P10 mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.5822 [-0.8003, -0.3496] | yes | BETTER | 2.8611 | 3.4433 | 2399/141 |
| LONGITUDINAL | speed MAE | -0.0204 [-0.2118, +0.1689] | no | ns | 1.8676 | 1.8880 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.2081 [-0.4195, +0.0215] | no | ns | 2.2500 | 2.4581 | 2399/141 |
| LONGITUDINAL | accel MAE | +0.9946 [+0.6733, +1.3122] | yes | **WORSE** | 3.0845 | 2.0899 | 2399/141 |
| LONGITUDINAL | DK-VIOL | -0.0252 [-0.0454, -0.0092] | yes | BETTER | 0.0218 | 0.0471 | 595/70 |
| LONGITUDINAL | DK-HW | -0.3886 [-0.7316, -0.1095] | yes | BETTER | 1.9849 | 2.3735 | 703/72 |
| LONGITUDINAL | DK-TG | -0.1030 [-0.1718, -0.0418] | yes | BETTER | 0.3200 | 0.4229 | 654/72 |
| LATERAL | heading MAE | +0.1655 [-0.6384, +0.9069] | no | ns | 9.7180 | 9.5525 | 2373/141 |
| LATERAL | yaw-rate MAE | +10.3568 [+7.1634, +13.7231] | yes | **WORSE** | 30.2295 | 19.8727 | 2365/141 |
| LATERAL | curvature MAE | +0.0965 [+0.0672, +0.1291] | yes | **WORSE** | 0.1689 | 0.0724 | 2365/141 |
| LATERAL | cross-track MAE | -0.6199 [-0.7510, -0.5014] | yes | BETTER | 1.2996 | 1.9195 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -0.4218 [-1.0970, +0.1628] | no | ns | 3.9853 | 4.4071 | 2365/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0473 [+0.0287, +0.0702] | yes | **WORSE** | 0.0639 | 0.0165 | 2106/135 |
| TACTICAL | TAC-LAT | -0.0067 [-0.0555, +0.0429] | no | ns | 0.5865 | 0.5932 | 2399/141 |
| TACTICAL | TAC-LON | +0.0425 [-0.0271, +0.1093] | no | ns | 0.4139 | 0.3714 | 2399/141 |
| STRATEGIC | S-POS@30 | +0.8148 [+0.3481, +1.2917] | yes | **WORSE** | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@50 | +0.3335 [-0.8136, +1.4573] | no | ns | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@80 | -1.9495 [-4.3742, +0.4859] | no | ns | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| crawl | M1 crawl share | +0.0696 [+0.0479, +0.0946] | yes | **WORSE** | 0.0926 | 0.0230 | 2106/135 |

#### DAG - P10 STRICT, scene frozen + limit causal (beside), seed 0

| family | row | arm - P10 [lo, hi] | sep | verdict | arm mean | P10 mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -1.0603 [-1.2273, -0.9014] | yes | BETTER | 2.3830 | 3.4433 | 2399/141 |
| LONGITUDINAL | speed MAE | -0.5488 [-0.7485, -0.3787] | yes | BETTER | 1.3393 | 1.8880 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.5499 [-0.6843, -0.4213] | yes | BETTER | 1.9083 | 2.4581 | 2399/141 |
| LONGITUDINAL | accel MAE | +0.7828 [+0.3065, +1.2481] | yes | **WORSE** | 2.8726 | 2.0899 | 2399/141 |
| LONGITUDINAL | DK-VIOL | +0.0269 [-0.0031, +0.0643] | no | ns | - | - | 595/70 |
| LONGITUDINAL | DK-HW | -0.1249 [-0.5101, +0.2210] | no | ns | - | - | 714/72 |
| LONGITUDINAL | DK-TG | -0.0234 [-0.0919, +0.0489] | no | ns | - | - | 653/72 |
| LATERAL | heading MAE | -2.7217 [-3.4250, -2.0901] | yes | BETTER | 6.8016 | 9.5233 | 2372/141 |
| LATERAL | yaw-rate MAE | +0.2554 [-2.1650, +2.5958] | no | ns | 20.0048 | 19.7494 | 2364/141 |
| LATERAL | curvature MAE | +0.0629 [+0.0430, +0.0861] | yes | **WORSE** | 0.1351 | 0.0723 | 2364/141 |
| LATERAL | cross-track MAE | -0.9280 [-1.0697, -0.8000] | yes | BETTER | 0.9916 | 1.9195 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -2.3150 [-3.6920, -1.1802] | yes | BETTER | 2.0482 | 4.3632 | 2364/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0286 [+0.0169, +0.0425] | yes | **WORSE** | 0.0452 | 0.0165 | 2106/135 |
| TACTICAL | TAC-LAT | +0.1663 [+0.1283, +0.2060] | yes | BETTER | - | - | 2399/141 |
| TACTICAL | TAC-LON | +0.1134 [+0.0504, +0.1767] | yes | BETTER | - | - | 2399/141 |
| STRATEGIC | S-POS@30 | NOT COMPUTED | | | | | |
| STRATEGIC | S-POS@50 | NOT COMPUTED | | | | | |
| STRATEGIC | S-POS@80 | NOT COMPUTED | | | | | |
| crawl | M1 crawl share | +0.0487 [+0.0308, +0.0694] | yes | **WORSE** | 0.0717 | 0.0230 | 2106/135 |

#### DAG - P10 STRICT, scene frozen + limit causal (beside), seed 1

| family | row | arm - P10 [lo, hi] | sep | verdict | arm mean | P10 mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.8535 [-1.0280, -0.6727] | yes | BETTER | 2.5898 | 3.4433 | 2399/141 |
| LONGITUDINAL | speed MAE | -0.2833 [-0.4653, -0.1148] | yes | BETTER | 1.6047 | 1.8880 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.3904 [-0.5482, -0.2237] | yes | BETTER | 2.0678 | 2.4581 | 2399/141 |
| LONGITUDINAL | accel MAE | +1.4033 [+0.9111, +1.9358] | yes | **WORSE** | 3.4931 | 2.0899 | 2399/141 |
| LONGITUDINAL | DK-VIOL | +0.0000 [-0.0268, +0.0309] | no | ns | - | - | 595/70 |
| LONGITUDINAL | DK-HW | -0.2804 [-0.7247, +0.0842] | no | ns | - | - | 711/72 |
| LONGITUDINAL | DK-TG | -0.0771 [-0.1458, -0.0156] | yes | BETTER | - | - | 654/72 |
| LATERAL | heading MAE | -0.8571 [-1.4577, -0.2218] | yes | BETTER | 8.6955 | 9.5525 | 2373/141 |
| LATERAL | yaw-rate MAE | +9.3010 [+6.4773, +12.2903] | yes | **WORSE** | 29.1738 | 19.8727 | 2365/141 |
| LATERAL | curvature MAE | +0.1169 [+0.0809, +0.1600] | yes | **WORSE** | 0.1893 | 0.0724 | 2365/141 |
| LATERAL | cross-track MAE | -0.8238 [-0.9626, -0.6950] | yes | BETTER | 1.0958 | 1.9195 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -1.0725 [-2.0595, -0.2122] | yes | BETTER | 3.3345 | 4.4071 | 2365/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0594 [+0.0376, +0.0853] | yes | **WORSE** | 0.0760 | 0.0165 | 2105/135 |
| TACTICAL | TAC-LAT | +0.1396 [+0.1020, +0.1802] | yes | BETTER | - | - | 2399/141 |
| TACTICAL | TAC-LON | +0.0975 [+0.0283, +0.1623] | yes | BETTER | - | - | 2399/141 |
| STRATEGIC | S-POS@30 | NOT COMPUTED | | | | | |
| STRATEGIC | S-POS@50 | NOT COMPUTED | | | | | |
| STRATEGIC | S-POS@80 | NOT COMPUTED | | | | | |
| crawl | M1 crawl share | +0.0642 [+0.0437, +0.0881] | yes | **WORSE** | 0.0872 | 0.0230 | 2105/135 |

#### DAG - P10 limit logged (beside), seed 0

| family | row | arm - P10 [lo, hi] | sep | verdict | arm mean | P10 mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -1.2130 [-1.3965, -1.0428] | yes | BETTER | 2.2303 | 3.4433 | 2399/141 |
| LONGITUDINAL | speed MAE | -0.5691 [-0.7642, -0.3933] | yes | BETTER | 1.3190 | 1.8880 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.6797 [-0.8150, -0.5498] | yes | BETTER | 1.7784 | 2.4581 | 2399/141 |
| LONGITUDINAL | accel MAE | +1.0438 [+0.5006, +1.5505] | yes | **WORSE** | 3.1336 | 2.0899 | 2399/141 |
| LONGITUDINAL | DK-VIOL | +0.0286 [+0.0017, +0.0592] | yes | **WORSE** | - | - | 595/70 |
| LONGITUDINAL | DK-HW | -0.0416 [-0.4533, +0.3025] | no | ns | - | - | 712/72 |
| LONGITUDINAL | DK-TG | -0.0008 [-0.0617, +0.0604] | no | ns | - | - | 653/72 |
| LATERAL | heading MAE | -3.2907 [-4.0839, -2.5711] | yes | BETTER | 6.2618 | 9.5525 | 2373/141 |
| LATERAL | yaw-rate MAE | +1.4403 [-1.0393, +3.8759] | no | ns | 21.3130 | 19.8727 | 2365/141 |
| LATERAL | curvature MAE | +0.0684 [+0.0473, +0.0917] | yes | **WORSE** | 0.1408 | 0.0724 | 2365/141 |
| LATERAL | cross-track MAE | -0.9877 [-1.1351, -0.8516] | yes | BETTER | 0.9319 | 1.9195 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -2.3180 [-3.6922, -1.1289] | yes | BETTER | 2.0890 | 4.4071 | 2365/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0328 [+0.0201, +0.0472] | yes | **WORSE** | 0.0493 | 0.0165 | 2106/135 |
| TACTICAL | TAC-LAT | +0.1813 [+0.1405, +0.2232] | yes | BETTER | - | - | 2399/141 |
| TACTICAL | TAC-LON | +0.1638 [+0.1004, +0.2246] | yes | BETTER | - | - | 2399/141 |
| STRATEGIC | S-POS@30 | -1.7175 [-2.2707, -1.2803] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@50 | -4.4832 [-5.6271, -3.4998] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@80 | -9.6871 [-11.8435, -7.5669] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| crawl | M1 crawl share | +0.0500 [+0.0332, +0.0699] | yes | **WORSE** | 0.0729 | 0.0230 | 2106/135 |

#### DAG - P10 limit logged (beside), seed 1

| family | row | arm - P10 [lo, hi] | sep | verdict | arm mean | P10 mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -1.0089 [-1.2128, -0.8172] | yes | BETTER | 2.4344 | 3.4433 | 2399/141 |
| LONGITUDINAL | speed MAE | -0.3009 [-0.4809, -0.1326] | yes | BETTER | 1.5872 | 1.8880 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.4689 [-0.6398, -0.3016] | yes | BETTER | 1.9893 | 2.4581 | 2399/141 |
| LONGITUDINAL | accel MAE | +1.5105 [+0.9806, +2.0721] | yes | **WORSE** | 3.6004 | 2.0899 | 2399/141 |
| LONGITUDINAL | DK-VIOL | +0.0034 [-0.0245, +0.0355] | no | ns | - | - | 595/70 |
| LONGITUDINAL | DK-HW | -0.1539 [-0.6059, +0.2274] | no | ns | - | - | 714/72 |
| LONGITUDINAL | DK-TG | -0.0917 [-0.1577, -0.0367] | yes | BETTER | - | - | 654/72 |
| LATERAL | heading MAE | -1.6132 [-2.4224, -0.7503] | yes | BETTER | 7.9393 | 9.5525 | 2373/141 |
| LATERAL | yaw-rate MAE | +9.5606 [+6.2108, +13.1182] | yes | **WORSE** | 29.4333 | 19.8727 | 2365/141 |
| LATERAL | curvature MAE | +0.1292 [+0.0925, +0.1713] | yes | **WORSE** | 0.2016 | 0.0724 | 2365/141 |
| LATERAL | cross-track MAE | -0.9498 [-1.1041, -0.8145] | yes | BETTER | 0.9698 | 1.9195 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -1.1663 [-2.2431, -0.2513] | yes | BETTER | 3.2408 | 4.4071 | 2365/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0620 [+0.0361, +0.0941] | yes | **WORSE** | 0.0786 | 0.0165 | 2106/135 |
| TACTICAL | TAC-LAT | +0.1776 [+0.1359, +0.2191] | yes | BETTER | - | - | 2399/141 |
| TACTICAL | TAC-LON | +0.1455 [+0.0776, +0.2090] | yes | BETTER | - | - | 2399/141 |
| STRATEGIC | S-POS@30 | -1.8734 [-2.3640, -1.4387] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@50 | -4.9034 [-5.9596, -3.9235] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@80 | -10.3254 [-12.4401, -8.3492] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| crawl | M1 crawl share | +0.0690 [+0.0474, +0.0939] | yes | **WORSE** | 0.0920 | 0.0230 | 2106/135 |
