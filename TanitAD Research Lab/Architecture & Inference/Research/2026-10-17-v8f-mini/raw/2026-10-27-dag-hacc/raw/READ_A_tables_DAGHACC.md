#### Read A, seed 0: DAG-HACC - P10, T1-kin decisive (m = 10, causal bucket, dead-reckoned nav, logged scene)

| family | row | DAG-HACC - P10 [lo, hi] | sep | verdict | DAG-HACC mean | P10 mean | n (win/ep) | plain DAG - P10 (reread) |
|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -1.0392 [-1.1941, -0.9008] | yes | BETTER | 2.4041 | 3.4433 | 2399/141 | -1.1656 [-1.3437, -0.9963] better |
| LONGITUDINAL | speed MAE | -0.6126 [-0.7482, -0.4957] | yes | BETTER | 1.2754 | 1.8880 | 2399/141 | -0.5765 [-0.7791, -0.4040] better |
| LONGITUDINAL | along-track MAE | -0.6262 [-0.7417, -0.5177] | yes | BETTER | 1.8320 | 2.4581 | 2399/141 | -0.6282 [-0.7671, -0.4916] better |
| LONGITUDINAL | accel MAE | -0.6444 [-0.9256, -0.4115] | yes | BETTER | 1.4455 | 2.0899 | 2399/141 | +0.8072 [+0.3253, +1.2728] WORSE |
| LONGITUDINAL | DK-VIOL | +0.0269 [-0.0064, +0.0760] | no | ns | 0.0740 | 0.0471 | 595/70 | +0.0353 [+0.0046, +0.0726] WORSE |
| LONGITUDINAL | DK-HW | -0.0812 [-0.4369, +0.2675] | no | ns | 2.3335 | 2.3869 | 716/72 | -0.0491 [-0.4481, +0.2980] |
| LONGITUDINAL | DK-TG | -0.0393 [-0.0980, +0.0149] | no | ns | 0.3916 | 0.4229 | 654/72 | -0.0035 [-0.0682, +0.0611] |
| LATERAL | heading MAE | -2.7563 [-3.2938, -2.2564] | yes | BETTER | 6.7963 | 9.5525 | 2373/141 | -3.3593 [-4.1043, -2.7104] better |
| LATERAL | yaw-rate MAE | -1.1388 [-3.0578, +0.5428] | no | ns | 18.7718 | 19.9106 | 2366/141 | -0.0400 [-2.3623, +2.1885] |
| LATERAL | curvature MAE | +0.0174 [+0.0091, +0.0270] | yes | **WORSE** | 0.0910 | 0.0736 | 2366/141 | +0.0612 [+0.0415, +0.0839] WORSE |
| LATERAL | cross-track MAE | -0.7736 [-0.9141, -0.6471] | yes | BETTER | 1.1460 | 1.9195 | 2399/141 | -0.9793 [-1.1231, -0.8460] better |
| LATERAL | L1 lateral-accel MAE | -1.4898 [-2.5369, -0.6077] | yes | BETTER | 2.9159 | 4.4057 | 2366/141 | -2.4322 [-3.8282, -1.2766] better |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0112 [+0.0050, +0.0190] | yes | **WORSE** | 0.0277 | 0.0165 | 2106/135 | +0.0288 [+0.0165, +0.0433] WORSE |
| TACTICAL | TAC-LAT | +0.2018 [+0.1651, +0.2397] | yes | BETTER | 0.7949 | 0.5932 | 2399/141 | +0.1926 [+0.1546, +0.2323] better |
| TACTICAL | TAC-LON | +0.1647 [+0.1171, +0.2113] | yes | BETTER | 0.5361 | 0.3714 | 2399/141 | +0.1288 [+0.0651, +0.1917] better |
| STRATEGIC | S-POS@30 | -1.9915 [-2.6029, -1.4745] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -1.7175 [-2.2707, -1.2803] better |
| STRATEGIC | S-POS@50 | -5.1059 [-6.3201, -4.0103] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -4.4832 [-5.6271, -3.4998] better |
| STRATEGIC | S-POS@80 | -10.7722 [-13.1090, -8.6474] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -9.6871 [-11.8435, -7.5669] better |
| crawl | M1 crawl share | +0.0242 [+0.0126, +0.0381] | yes | **WORSE** | 0.0471 | 0.0230 | 2106/135 | +0.0470 [+0.0296, +0.0673] WORSE |

#### Read A, seed 1: DAG-HACC - P10, T1-kin decisive (m = 10, causal bucket, dead-reckoned nav, logged scene)

| family | row | DAG-HACC - P10 [lo, hi] | sep | verdict | DAG-HACC mean | P10 mean | n (win/ep) | plain DAG - P10 (reread) |
|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -1.0916 [-1.2535, -0.9322] | yes | BETTER | 2.3517 | 3.4433 | 2399/141 | -0.9868 [-1.1778, -0.8060] better |
| LONGITUDINAL | speed MAE | -0.7449 [-0.9160, -0.5928] | yes | BETTER | 1.1432 | 1.8880 | 2399/141 | -0.3020 [-0.4813, -0.1312] better |
| LONGITUDINAL | along-track MAE | -0.6472 [-0.7803, -0.5163] | yes | BETTER | 1.8109 | 2.4581 | 2399/141 | -0.4571 [-0.6243, -0.2880] better |
| LONGITUDINAL | accel MAE | -0.8469 [-1.1600, -0.5836] | yes | BETTER | 1.2430 | 2.0899 | 2399/141 | +1.4436 [+0.9463, +1.9592] WORSE |
| LONGITUDINAL | DK-VIOL | +0.0034 [-0.0248, +0.0372] | no | ns | 0.0504 | 0.0471 | 595/70 | +0.0050 [-0.0196, +0.0345] |
| LONGITUDINAL | DK-HW | -0.3167 [-0.6545, -0.0282] | yes | BETTER | 2.1495 | 2.3869 | 713/72 | -0.2065 [-0.6444, +0.1422] |
| LONGITUDINAL | DK-TG | -0.1139 [-0.1803, -0.0586] | yes | BETTER | 0.2913 | 0.4229 | 653/72 | -0.0934 [-0.1585, -0.0402] better |
| LATERAL | heading MAE | -2.1159 [-2.6448, -1.6318] | yes | BETTER | 7.4366 | 9.5525 | 2373/141 | -1.5991 [-2.3295, -0.8482] better |
| LATERAL | yaw-rate MAE | +1.2187 [-0.4700, +3.0036] | no | ns | 21.1293 | 19.9106 | 2366/141 | +8.6601 [+5.7957, +11.8486] WORSE |
| LATERAL | curvature MAE | +0.0471 [+0.0284, +0.0683] | yes | **WORSE** | 0.1207 | 0.0736 | 2366/141 | +0.1134 [+0.0787, +0.1525] WORSE |
| LATERAL | cross-track MAE | -0.7943 [-0.9345, -0.6665] | yes | BETTER | 1.1253 | 1.9195 | 2399/141 | -0.9269 [-1.0757, -0.7954] better |
| LATERAL | L1 lateral-accel MAE | -2.0790 [-3.1986, -1.1025] | yes | BETTER | 2.3267 | 4.4057 | 2366/141 | -1.1360 [-2.1320, -0.2847] better |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0280 [+0.0127, +0.0488] | yes | **WORSE** | 0.0445 | 0.0165 | 2106/135 | +0.0604 [+0.0356, +0.0930] WORSE |
| TACTICAL | TAC-LAT | +0.1651 [+0.1279, +0.2043] | yes | BETTER | 0.7582 | 0.5932 | 2399/141 | +0.1851 [+0.1461, +0.2255] better |
| TACTICAL | TAC-LON | +0.1792 [+0.1247, +0.2323] | yes | BETTER | 0.5506 | 0.3714 | 2399/141 | +0.1142 [+0.0442, +0.1792] better |
| STRATEGIC | S-POS@30 | -2.0186 [-2.6019, -1.5203] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -1.8734 [-2.3640, -1.4387] better |
| STRATEGIC | S-POS@50 | -5.4377 [-6.7005, -4.2945] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -4.9034 [-5.9596, -3.9235] better |
| STRATEGIC | S-POS@80 | -12.0676 [-14.5739, -9.7827] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -10.3254 [-12.4401, -8.3492] better |
| crawl | M1 crawl share | +0.0390 [+0.0221, +0.0587] | yes | **WORSE** | 0.0619 | 0.0230 | 2106/135 | +0.0614 [+0.0409, +0.0859] WORSE |
