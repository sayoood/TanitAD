#### DAG-HACC - P10, STRICT (scene frozen + limit causal), BESIDE, seed 0

| family | row | DAG-HACC - P10 [lo, hi] | sep | verdict | DAG-HACC mean | P10 mean | n (win/ep) | plain DAG - P10 (reread) |
|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.9849 [-1.1304, -0.8522] | yes | BETTER | 2.4584 | 3.4433 | 2399/141 | - |
| LONGITUDINAL | speed MAE | -0.6075 [-0.7418, -0.4906] | yes | BETTER | 1.2806 | 1.8880 | 2399/141 | - |
| LONGITUDINAL | along-track MAE | -0.6105 [-0.7217, -0.5032] | yes | BETTER | 1.8477 | 2.4581 | 2399/141 | - |
| LONGITUDINAL | accel MAE | -0.6395 [-0.9153, -0.4124] | yes | BETTER | 1.4504 | 2.0899 | 2399/141 | - |
| LONGITUDINAL | DK-VIOL | +0.0336 [-0.0034, +0.0872] | no | ns | - | - | 595/70 | - |
| LONGITUDINAL | DK-HW | -0.1778 [-0.6212, +0.2230] | no | ns | - | - | 715/72 | - |
| LONGITUDINAL | DK-TG | -0.0415 [-0.1161, +0.0410] | no | ns | - | - | 654/72 | - |
| LATERAL | heading MAE | -2.3179 [-2.8689, -1.8472] | yes | BETTER | 7.2346 | 9.5525 | 2373/141 | - |
| LATERAL | yaw-rate MAE | -0.7655 [-2.6141, +0.8596] | no | ns | 19.1451 | 19.9106 | 2366/141 | - |
| LATERAL | curvature MAE | +0.0190 [+0.0103, +0.0285] | yes | **WORSE** | 0.0926 | 0.0736 | 2366/141 | - |
| LATERAL | cross-track MAE | -0.7186 [-0.8503, -0.6036] | yes | BETTER | 1.2009 | 1.9195 | 2399/141 | - |
| LATERAL | L1 lateral-accel MAE | -1.4277 [-2.4389, -0.5785] | yes | BETTER | 2.9780 | 4.4057 | 2366/141 | - |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0104 [+0.0048, +0.0169] | yes | **WORSE** | 0.0269 | 0.0165 | 2106/135 | - |
| TACTICAL | TAC-LAT | +0.1842 [+0.1483, +0.2222] | yes | BETTER | - | - | 2399/141 | - |
| TACTICAL | TAC-LON | +0.1663 [+0.1163, +0.2166] | yes | BETTER | - | - | 2399/141 | - |
| STRATEGIC | S-POS@30 | -1.6223 [-2.2049, -1.1199] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | - |
| STRATEGIC | S-POS@50 | -4.1446 [-5.2677, -3.1338] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | - |
| STRATEGIC | S-POS@80 | -8.4011 [-10.4398, -6.5610] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | - |
| crawl | M1 crawl share | +0.0253 [+0.0135, +0.0390] | yes | **WORSE** | 0.0482 | 0.0230 | 2106/135 | - |

#### DAG-HACC - P10, STRICT (scene frozen + limit causal), BESIDE, seed 1

| family | row | DAG-HACC - P10 [lo, hi] | sep | verdict | DAG-HACC mean | P10 mean | n (win/ep) | plain DAG - P10 (reread) |
|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.9906 [-1.1457, -0.8428] | yes | BETTER | 2.4527 | 3.4433 | 2399/141 | - |
| LONGITUDINAL | speed MAE | -0.7338 [-0.8994, -0.5830] | yes | BETTER | 1.1543 | 1.8880 | 2399/141 | - |
| LONGITUDINAL | along-track MAE | -0.6262 [-0.7582, -0.4955] | yes | BETTER | 1.8319 | 2.4581 | 2399/141 | - |
| LONGITUDINAL | accel MAE | -0.8406 [-1.1526, -0.5765] | yes | BETTER | 1.2493 | 2.0899 | 2399/141 | - |
| LONGITUDINAL | DK-VIOL | -0.0034 [-0.0310, +0.0286] | no | ns | - | - | 595/70 | - |
| LONGITUDINAL | DK-HW | -0.3405 [-0.6367, -0.0763] | yes | BETTER | - | - | 713/72 | - |
| LONGITUDINAL | DK-TG | -0.0636 [-0.1372, +0.0303] | no | ns | - | - | 654/72 | - |
| LATERAL | heading MAE | -1.5668 [-1.9955, -1.1697] | yes | BETTER | 7.9857 | 9.5525 | 2373/141 | - |
| LATERAL | yaw-rate MAE | +1.1912 [-0.4967, +2.9300] | no | ns | 21.1018 | 19.9106 | 2366/141 | - |
| LATERAL | curvature MAE | +0.0465 [+0.0276, +0.0683] | yes | **WORSE** | 0.1201 | 0.0736 | 2366/141 | - |
| LATERAL | cross-track MAE | -0.6893 [-0.8118, -0.5719] | yes | BETTER | 1.2302 | 1.9195 | 2399/141 | - |
| LATERAL | L1 lateral-accel MAE | -2.0805 [-3.2022, -1.1176] | yes | BETTER | 2.3252 | 4.4057 | 2366/141 | - |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0327 [+0.0143, +0.0577] | yes | **WORSE** | 0.0492 | 0.0165 | 2106/135 | - |
| TACTICAL | TAC-LAT | +0.1150 [+0.0783, +0.1544] | yes | BETTER | - | - | 2399/141 | - |
| TACTICAL | TAC-LON | +0.1738 [+0.1197, +0.2275] | yes | BETTER | - | - | 2399/141 | - |
| STRATEGIC | S-POS@30 | -1.4169 [-1.9444, -0.9708] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | - |
| STRATEGIC | S-POS@50 | -4.0053 [-5.0610, -3.0714] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | - |
| STRATEGIC | S-POS@80 | -9.1739 [-11.2660, -7.3517] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | - |
| crawl | M1 crawl share | +0.0415 [+0.0247, +0.0611] | yes | **WORSE** | 0.0644 | 0.0230 | 2106/135 | - |

#### DAG-HACC - P10, limit logged, BESIDE, seed 0

| family | row | DAG-HACC - P10 [lo, hi] | sep | verdict | DAG-HACC mean | P10 mean | n (win/ep) | plain DAG - P10 (reread) |
|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -1.1085 [-1.2680, -0.9611] | yes | BETTER | 2.3348 | 3.4433 | 2399/141 | - |
| LONGITUDINAL | speed MAE | -0.6560 [-0.7990, -0.5297] | yes | BETTER | 1.2321 | 1.8880 | 2399/141 | - |
| LONGITUDINAL | along-track MAE | -0.6993 [-0.8191, -0.5907] | yes | BETTER | 1.7588 | 2.4581 | 2399/141 | - |
| LONGITUDINAL | accel MAE | -0.6257 [-0.9148, -0.3845] | yes | BETTER | 1.4642 | 2.0899 | 2399/141 | - |
| LONGITUDINAL | DK-VIOL | +0.0101 [-0.0155, +0.0348] | no | ns | - | - | 595/70 | - |
| LONGITUDINAL | DK-HW | -0.1562 [-0.4513, +0.1208] | no | ns | - | - | 716/72 | - |
| LONGITUDINAL | DK-TG | -0.0348 [-0.0921, +0.0195] | no | ns | - | - | 654/72 | - |
| LATERAL | heading MAE | -2.6774 [-3.2378, -2.1534] | yes | BETTER | 6.8751 | 9.5525 | 2373/141 | - |
| LATERAL | yaw-rate MAE | -0.5208 [-2.4175, +1.2123] | no | ns | 19.3898 | 19.9106 | 2366/141 | - |
| LATERAL | curvature MAE | +0.0217 [+0.0126, +0.0320] | yes | **WORSE** | 0.0953 | 0.0736 | 2366/141 | - |
| LATERAL | cross-track MAE | -0.7820 [-0.9209, -0.6564] | yes | BETTER | 1.1376 | 1.9195 | 2399/141 | - |
| LATERAL | L1 lateral-accel MAE | -1.4158 [-2.4936, -0.5261] | yes | BETTER | 2.9900 | 4.4057 | 2366/141 | - |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0121 [+0.0053, +0.0200] | yes | **WORSE** | 0.0286 | 0.0165 | 2106/135 | - |
| TACTICAL | TAC-LAT | +0.2005 [+0.1635, +0.2385] | yes | BETTER | - | - | 2399/141 | - |
| TACTICAL | TAC-LON | +0.2105 [+0.1634, +0.2584] | yes | BETTER | - | - | 2399/141 | - |
| STRATEGIC | S-POS@30 | NOT COMPUTED (no S-POS block for this run) | | | | | | |
| STRATEGIC | S-POS@50 | NOT COMPUTED (no S-POS block for this run) | | | | | | |
| STRATEGIC | S-POS@80 | NOT COMPUTED (no S-POS block for this run) | | | | | | |
| crawl | M1 crawl share | +0.0236 [+0.0128, +0.0368] | yes | **WORSE** | 0.0465 | 0.0230 | 2106/135 | - |

#### DAG-HACC - P10, limit logged, BESIDE, seed 1

| family | row | DAG-HACC - P10 [lo, hi] | sep | verdict | DAG-HACC mean | P10 mean | n (win/ep) | plain DAG - P10 (reread) |
|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -1.1541 [-1.3259, -0.9946] | yes | BETTER | 2.2892 | 3.4433 | 2399/141 | - |
| LONGITUDINAL | speed MAE | -0.7795 [-0.9504, -0.6232] | yes | BETTER | 1.1085 | 1.8880 | 2399/141 | - |
| LONGITUDINAL | along-track MAE | -0.7133 [-0.8491, -0.5855] | yes | BETTER | 1.7448 | 2.4581 | 2399/141 | - |
| LONGITUDINAL | accel MAE | -0.8467 [-1.1628, -0.5715] | yes | BETTER | 1.2431 | 2.0899 | 2399/141 | - |
| LONGITUDINAL | DK-VIOL | -0.0101 [-0.0339, +0.0128] | no | ns | - | - | 595/70 | - |
| LONGITUDINAL | DK-HW | -0.3712 [-0.6799, -0.0990] | yes | BETTER | - | - | 715/72 | - |
| LONGITUDINAL | DK-TG | -0.1158 [-0.1816, -0.0627] | yes | BETTER | - | - | 653/72 | - |
| LATERAL | heading MAE | -1.8970 [-2.4447, -1.3797] | yes | BETTER | 7.6555 | 9.5525 | 2373/141 | - |
| LATERAL | yaw-rate MAE | +2.3542 [+0.4668, +4.2557] | yes | **WORSE** | 22.2648 | 19.9106 | 2366/141 | - |
| LATERAL | curvature MAE | +0.0564 [+0.0354, +0.0794] | yes | **WORSE** | 0.1300 | 0.0736 | 2366/141 | - |
| LATERAL | cross-track MAE | -0.7968 [-0.9402, -0.6652] | yes | BETTER | 1.1227 | 1.9195 | 2399/141 | - |
| LATERAL | L1 lateral-accel MAE | -1.9526 [-3.0786, -0.9822] | yes | BETTER | 2.4532 | 4.4057 | 2366/141 | - |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0304 [+0.0145, +0.0514] | yes | **WORSE** | 0.0469 | 0.0165 | 2106/135 | - |
| TACTICAL | TAC-LAT | +0.1584 [+0.1205, +0.1988] | yes | BETTER | - | - | 2399/141 | - |
| TACTICAL | TAC-LON | +0.2218 [+0.1700, +0.2718] | yes | BETTER | - | - | 2399/141 | - |
| STRATEGIC | S-POS@30 | NOT COMPUTED (no S-POS block for this run) | | | | | | |
| STRATEGIC | S-POS@50 | NOT COMPUTED (no S-POS block for this run) | | | | | | |
| STRATEGIC | S-POS@80 | NOT COMPUTED (no S-POS block for this run) | | | | | | |
| crawl | M1 crawl share | +0.0417 [+0.0245, +0.0619] | yes | **WORSE** | 0.0647 | 0.0230 | 2106/135 | - |
