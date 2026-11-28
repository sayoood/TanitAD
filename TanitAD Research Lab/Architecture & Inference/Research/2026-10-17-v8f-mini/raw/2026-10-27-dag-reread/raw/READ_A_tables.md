#### Read A, seed 0: DAG - P10, T1-kin decisive (m=10, causal bucket)

| family | row | arm - P10 [lo, hi] | sep | verdict | arm mean | P10 mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -1.1656 [-1.3437, -0.9963] | yes | BETTER | 2.2777 | 3.4433 | 2399/141 |
| LONGITUDINAL | speed MAE | -0.5765 [-0.7791, -0.4040] | yes | BETTER | 1.3115 | 1.8880 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.6282 [-0.7671, -0.4916] | yes | BETTER | 1.8299 | 2.4581 | 2399/141 |
| LONGITUDINAL | accel MAE | +0.8072 [+0.3253, +1.2728] | yes | **WORSE** | 2.8970 | 2.0899 | 2399/141 |
| LONGITUDINAL | DK-VIOL | +0.0353 [+0.0046, +0.0726] | yes | **WORSE** | 0.0824 | 0.0471 | 595/70 |
| LONGITUDINAL | DK-HW | -0.0491 [-0.4481, +0.2980] | no | ns | 2.3197 | 2.3688 | 713/72 |
| LONGITUDINAL | DK-TG | -0.0035 [-0.0682, +0.0611] | no | ns | 0.4127 | 0.4162 | 653/72 |
| LATERAL | heading MAE | -3.3593 [-4.1043, -2.7104] | yes | BETTER | 6.1932 | 9.5525 | 2373/141 |
| LATERAL | yaw-rate MAE | -0.0400 [-2.3623, +2.1885] | no | ns | 19.8327 | 19.8727 | 2365/141 |
| LATERAL | curvature MAE | +0.0612 [+0.0415, +0.0839] | yes | **WORSE** | 0.1336 | 0.0724 | 2365/141 |
| LATERAL | cross-track MAE | -0.9793 [-1.1231, -0.8460] | yes | BETTER | 0.9402 | 1.9195 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -2.4322 [-3.8282, -1.2766] | yes | BETTER | 1.9749 | 4.4071 | 2365/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0288 [+0.0165, +0.0433] | yes | **WORSE** | 0.0454 | 0.0165 | 2106/135 |
| TACTICAL | TAC-LAT | +0.1926 [+0.1546, +0.2323] | yes | BETTER | 0.7857 | 0.5932 | 2399/141 |
| TACTICAL | TAC-LON | +0.1288 [+0.0651, +0.1917] | yes | BETTER | 0.5002 | 0.3714 | 2399/141 |
| STRATEGIC | S-POS@30 | -1.7175 [-2.2707, -1.2803] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@50 | -4.4832 [-5.6271, -3.4998] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@80 | -9.6871 [-11.8435, -7.5669] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| crawl | M1 crawl share | +0.0470 [+0.0296, +0.0673] | yes | **WORSE** | 0.0700 | 0.0230 | 2106/135 |

#### Read A, seed 1: DAG - P10, T1-kin decisive (m=10, causal bucket)

| family | row | arm - P10 [lo, hi] | sep | verdict | arm mean | P10 mean | n (win/ep) |
|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.9868 [-1.1778, -0.8060] | yes | BETTER | 2.4565 | 3.4433 | 2399/141 |
| LONGITUDINAL | speed MAE | -0.3020 [-0.4813, -0.1312] | yes | BETTER | 1.5860 | 1.8880 | 2399/141 |
| LONGITUDINAL | along-track MAE | -0.4571 [-0.6243, -0.2880] | yes | BETTER | 2.0010 | 2.4581 | 2399/141 |
| LONGITUDINAL | accel MAE | +1.4436 [+0.9463, +1.9592] | yes | **WORSE** | 3.5335 | 2.0899 | 2399/141 |
| LONGITUDINAL | DK-VIOL | +0.0050 [-0.0196, +0.0345] | no | ns | 0.0521 | 0.0471 | 595/70 |
| LONGITUDINAL | DK-HW | -0.2065 [-0.6444, +0.1422] | no | ns | 2.1712 | 2.3777 | 712/72 |
| LONGITUDINAL | DK-TG | -0.0934 [-0.1585, -0.0402] | yes | BETTER | 0.3295 | 0.4229 | 654/72 |
| LATERAL | heading MAE | -1.5991 [-2.3295, -0.8482] | yes | BETTER | 7.9534 | 9.5525 | 2373/141 |
| LATERAL | yaw-rate MAE | +8.6601 [+5.7957, +11.8486] | yes | **WORSE** | 28.5328 | 19.8727 | 2365/141 |
| LATERAL | curvature MAE | +0.1134 [+0.0787, +0.1525] | yes | **WORSE** | 0.1857 | 0.0724 | 2365/141 |
| LATERAL | cross-track MAE | -0.9269 [-1.0757, -0.7954] | yes | BETTER | 0.9927 | 1.9195 | 2399/141 |
| LATERAL | L1 lateral-accel MAE | -1.1360 [-2.1320, -0.2847] | yes | BETTER | 3.2711 | 4.4071 | 2365/141 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0604 [+0.0356, +0.0930] | yes | **WORSE** | 0.0769 | 0.0165 | 2106/135 |
| TACTICAL | TAC-LAT | +0.1851 [+0.1461, +0.2255] | yes | BETTER | 0.7782 | 0.5932 | 2399/141 |
| TACTICAL | TAC-LON | +0.1142 [+0.0442, +0.1792] | yes | BETTER | 0.4856 | 0.3714 | 2399/141 |
| STRATEGIC | S-POS@30 | -1.8734 [-2.3640, -1.4387] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@50 | -4.9034 [-5.9596, -3.9235] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| STRATEGIC | S-POS@80 | -10.3254 [-12.4401, -8.3492] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV (DAG_READ.json) |
| crawl | M1 crawl share | +0.0614 [+0.0409, +0.0859] | yes | **WORSE** | 0.0844 | 0.0230 | 2106/135 |
