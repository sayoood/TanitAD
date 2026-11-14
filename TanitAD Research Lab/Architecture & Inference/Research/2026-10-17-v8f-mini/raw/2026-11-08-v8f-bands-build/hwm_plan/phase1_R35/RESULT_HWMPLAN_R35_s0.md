# HWM-PLAN read, every row stamped with its tier: T0 oracle-goal rows are diagnostics, T1-kin rows score the plan integrated by the kinematic model (994 windows, 71 episodes; estimator paired episode-cluster bootstrap, taniteval/ci.py)
STRATEGIC: PAUSED (PI item 25, 2026-09-19): not a pass, not omitted

* LATERAL.cross_track_vs_lane: n = 0: no lane geometry; xtrack here is the lateral offset of the plan end from the logged-action trajectory end
* TACTICAL.goal_setting: n = 0: the planner exposes no goal or anchor selection; TACTICAL here is the manoeuvre implied by the planned trajectory, not a vocabulary output

* headway scope: headway / time gap / TTC use taniteval.lead_metrics.distance_keeping over the b1 eval lead blocks (taniteval/tools/build_lead_block_b1.py): the 0-2 s band reads the banked K = 10 block (2 s lead track); the 2-6 s and full 6 s bands read the K = 30 block hwm_plan/data/b1_eval_lead_block_k30.npz (same sources, sha256 equal, 6 s lead track). A window needs a lead track and labelled span over the whole of the block's horizon, so the K = 30 block has fewer LEAD rows than the K = 10 block. In a band that starts after NOW the time-gap denominator is each path's own speed at the band start
* tactical scope: plan-derived decision, NOT a vocabulary output: the planned and the logged trajectory (both the kinematic integration of the action sequence from the same v0) are classified by ONE rule, tanitad.refs.refc_tactical.factor_from_kinematics through taniteval.four_families.maneuver_kinematics, on the path of each band re-expressed in the frame of the pose at the band start. The rule reads path geometry only (net path tangent, chord speeds); the goal that conditioned the plan is not an input of the rule. Thresholds are the programme's 2 s thresholds applied unchanged to each band (not rescaled to 4 s or 6 s)
* lead-state counts of the join: {'LEAD': 219, 'NO_LEAD': 273, 'NOT_STRAIGHT': 417, 'NO_LABEL': 85, 'NO_ROW': 0}

| kind | family | band | metric | row | tier | n | delta or mean | 95 % interval | label |
|---|---|---|---|---|---|---|---|---|---|
| arm | LONGITUDINAL | 0-2s | acc_mae | hwm_oracle | T0 | 994 | 1.2877 | [1.2160, 1.3582] |  |
| arm | LONGITUDINAL | 0-2s | acc_mae | flat_oracle | T0 | 994 | 1.3798 | [1.3044, 1.4579] |  |
| arm | LONGITUDINAL | 0-2s | acc_mae | hold | T1-kin | 994 | 0.2970 | [0.2571, 0.3372] |  |
| arm | LONGITUDINAL | 0-2s | acc_mae | prior | T1-kin | 994 | 0.4736 | [0.4017, 0.5504] |  |
| contrast | LONGITUDINAL | 0-2s | acc_mae | hwm_oracle - hold | T0 | 994 | 0.9907 | [0.9249, 1.0566] | worse |
| contrast | LONGITUDINAL | 0-2s | acc_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.0921 | [-0.1642, -0.0203] | better |
| contrast | LONGITUDINAL | 0-2s | acc_mae | flat_oracle - hold | T0 | 994 | 1.0827 | [1.0088, 1.1564] | worse |
| arm | LONGITUDINAL | 0-2s | speed_err | hwm_oracle | T0 | 994 | 1.4876 | [1.3490, 1.6287] |  |
| arm | LONGITUDINAL | 0-2s | speed_err | flat_oracle | T0 | 994 | 2.0804 | [1.9035, 2.2573] |  |
| arm | LONGITUDINAL | 0-2s | speed_err | hold | T1-kin | 994 | 0.5146 | [0.4421, 0.5901] |  |
| arm | LONGITUDINAL | 0-2s | speed_err | prior | T1-kin | 994 | 0.8602 | [0.7205, 1.0114] |  |
| contrast | LONGITUDINAL | 0-2s | speed_err | hwm_oracle - hold | T0 | 994 | 0.9730 | [0.8502, 1.0996] | worse |
| contrast | LONGITUDINAL | 0-2s | speed_err | hwm_oracle - flat_oracle | T0 | 994 | -0.5928 | [-0.7411, -0.4379] | better |
| contrast | LONGITUDINAL | 0-2s | speed_err | flat_oracle - hold | T0 | 994 | 1.5658 | [1.3836, 1.7423] | worse |
| arm | LONGITUDINAL | 0-2s | hw_err | hwm_oracle | T0 | 214 | 1.3381 | [0.7487, 2.1520] |  |
| arm | LONGITUDINAL | 0-2s | hw_err | flat_oracle | T0 | 214 | 1.5119 | [0.9095, 2.2196] |  |
| arm | LONGITUDINAL | 0-2s | hw_err | hold | T1-kin | 214 | 0.2887 | [0.1250, 0.5513] |  |
| arm | LONGITUDINAL | 0-2s | hw_err | prior | T1-kin | 214 | 0.8682 | [0.4108, 1.5049] |  |
| contrast | LONGITUDINAL | 0-2s | hw_err | hwm_oracle - hold | T0 | 214 | 1.0494 | [0.5340, 1.7641] | worse |
| contrast | LONGITUDINAL | 0-2s | hw_err | hwm_oracle - flat_oracle | T0 | 214 | -0.1737 | [-0.5813, 0.2723] | none |
| contrast | LONGITUDINAL | 0-2s | hw_err | flat_oracle - hold | T0 | 214 | 1.2232 | [0.7122, 1.8647] | worse |
| arm | LONGITUDINAL | 0-2s | tg_err | hwm_oracle | T0 | 182 | 0.2129 | [0.1242, 0.3319] |  |
| arm | LONGITUDINAL | 0-2s | tg_err | flat_oracle | T0 | 182 | 0.2878 | [0.1477, 0.4724] |  |
| arm | LONGITUDINAL | 0-2s | tg_err | hold | T1-kin | 182 | 0.0718 | [0.0287, 0.1329] |  |
| arm | LONGITUDINAL | 0-2s | tg_err | prior | T1-kin | 182 | 0.1621 | [0.0812, 0.2854] |  |
| contrast | LONGITUDINAL | 0-2s | tg_err | hwm_oracle - hold | T0 | 182 | 0.1410 | [0.0779, 0.2300] | worse |
| contrast | LONGITUDINAL | 0-2s | tg_err | hwm_oracle - flat_oracle | T0 | 182 | -0.0749 | [-0.2044, 0.0152] | none |
| contrast | LONGITUDINAL | 0-2s | tg_err | flat_oracle - hold | T0 | 182 | 0.2160 | [0.1041, 0.3665] | worse |
| arm | LONGITUDINAL | 0-2s | ttc_err | hwm_oracle | T0 | 214 | 4.0436 | [2.5833, 5.6746] |  |
| arm | LONGITUDINAL | 0-2s | ttc_err | flat_oracle | T0 | 214 | 4.9441 | [3.2365, 6.9025] |  |
| arm | LONGITUDINAL | 0-2s | ttc_err | hold | T1-kin | 214 | 0.7046 | [0.3033, 1.1573] |  |
| arm | LONGITUDINAL | 0-2s | ttc_err | prior | T1-kin | 214 | 2.3186 | [1.2131, 3.7213] |  |
| contrast | LONGITUDINAL | 0-2s | ttc_err | hwm_oracle - hold | T0 | 214 | 3.3390 | [1.8321, 5.0741] | worse |
| contrast | LONGITUDINAL | 0-2s | ttc_err | hwm_oracle - flat_oracle | T0 | 214 | -0.9005 | [-2.2089, 0.3755] | none |
| contrast | LONGITUDINAL | 0-2s | ttc_err | flat_oracle - hold | T0 | 214 | 4.2395 | [2.5979, 6.1440] | worse |
| arm | LONGITUDINAL | 0-2s | hw_d | hwm_oracle | T0 | 214 | 0.1362 | [-0.5131, 0.6905] |  |
| arm | LONGITUDINAL | 0-2s | hw_d | flat_oracle | T0 | 214 | 0.3755 | [-0.2819, 1.0717] |  |
| arm | LONGITUDINAL | 0-2s | hw_d | hold | T1-kin | 214 | -0.1389 | [-0.4049, 0.0202] |  |
| arm | LONGITUDINAL | 0-2s | hw_d | prior | T1-kin | 214 | -0.3289 | [-0.7469, 0.0059] |  |
| arm | LONGITUDINAL | 0-2s | tg_d | hwm_oracle | T0 | 182 | 0.0387 | [-0.0330, 0.1158] |  |
| arm | LONGITUDINAL | 0-2s | tg_d | flat_oracle | T0 | 182 | -0.0411 | [-0.2342, 0.0965] |  |
| arm | LONGITUDINAL | 0-2s | tg_d | hold | T1-kin | 182 | -0.0415 | [-0.0964, 0.0009] |  |
| arm | LONGITUDINAL | 0-2s | tg_d | prior | T1-kin | 182 | -0.0635 | [-0.1463, -0.0029] |  |
| arm | LONGITUDINAL | 0-2s | ttc_d | hwm_oracle | T0 | 214 | -0.5802 | [-2.4791, 1.0517] |  |
| arm | LONGITUDINAL | 0-2s | ttc_d | flat_oracle | T0 | 214 | -1.0823 | [-3.4357, 0.9785] |  |
| arm | LONGITUDINAL | 0-2s | ttc_d | hold | T1-kin | 214 | 0.0112 | [-0.2654, 0.2760] |  |
| arm | LONGITUDINAL | 0-2s | ttc_d | prior | T1-kin | 214 | -0.5008 | [-1.7772, 0.5194] |  |
| arm | LONGITUDINAL | 2-6s | acc_mae | hwm_oracle | T0 | 994 | 0.9582 | [0.9156, 1.0070] |  |
| arm | LONGITUDINAL | 2-6s | acc_mae | flat_oracle | T0 | 994 | 1.2785 | [1.2031, 1.3500] |  |
| arm | LONGITUDINAL | 2-6s | acc_mae | hold | T1-kin | 994 | 0.6360 | [0.5385, 0.7374] |  |
| arm | LONGITUDINAL | 2-6s | acc_mae | prior | T1-kin | 994 | 0.4716 | [0.3989, 0.5502] |  |
| contrast | LONGITUDINAL | 2-6s | acc_mae | hwm_oracle - hold | T0 | 994 | 0.3222 | [0.2544, 0.3897] | worse |
| contrast | LONGITUDINAL | 2-6s | acc_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.3203 | [-0.3892, -0.2515] | better |
| contrast | LONGITUDINAL | 2-6s | acc_mae | flat_oracle - hold | T0 | 994 | 0.6425 | [0.5377, 0.7471] | worse |
| arm | LONGITUDINAL | 2-6s | speed_err | hwm_oracle | T0 | 994 | 2.4376 | [2.2049, 2.7049] |  |
| arm | LONGITUDINAL | 2-6s | speed_err | flat_oracle | T0 | 994 | 3.0200 | [2.6012, 3.5007] |  |
| arm | LONGITUDINAL | 2-6s | speed_err | hold | T1-kin | 994 | 2.3983 | [2.0456, 2.7712] |  |
| arm | LONGITUDINAL | 2-6s | speed_err | prior | T1-kin | 994 | 2.0577 | [1.7073, 2.4384] |  |
| contrast | LONGITUDINAL | 2-6s | speed_err | hwm_oracle - hold | T0 | 994 | 0.0393 | [-0.2650, 0.3581] | none |
| contrast | LONGITUDINAL | 2-6s | speed_err | hwm_oracle - flat_oracle | T0 | 994 | -0.5824 | [-1.0111, -0.1895] | better |
| contrast | LONGITUDINAL | 2-6s | speed_err | flat_oracle - hold | T0 | 994 | 0.6218 | [0.1219, 1.1779] | worse |
| arm | LONGITUDINAL | 2-6s | hw_err | hwm_oracle | T0 | 115 | 4.3986 | [3.2382, 6.0505] |  |
| arm | LONGITUDINAL | 2-6s | hw_err | flat_oracle | T0 | 78 | 6.7381 | [5.2820, 9.1853] |  |
| arm | LONGITUDINAL | 2-6s | hw_err | hold | T1-kin | 156 | 2.2384 | [1.3739, 3.3350] |  |
| arm | LONGITUDINAL | 2-6s | hw_err | prior | T1-kin | 138 | 2.2450 | [1.3181, 3.5752] |  |
| contrast | LONGITUDINAL | 2-6s | hw_err | hwm_oracle - hold | T0 | 102 | 1.8797 | [1.0495, 2.8491] | worse |
| contrast | LONGITUDINAL | 2-6s | hw_err | hwm_oracle - flat_oracle | T0 | 60 | -3.0453 | [-5.1463, -1.3013] | better |
| contrast | LONGITUDINAL | 2-6s | hw_err | flat_oracle - hold | T0 | 70 | 4.1726 | [2.3943, 6.5388] | worse |
| arm | LONGITUDINAL | 2-6s | tg_err | hwm_oracle | T0 | 89 | 3.0529 | [1.5519, 5.1167] |  |
| arm | LONGITUDINAL | 2-6s | tg_err | flat_oracle | T0 | 60 | 4.9062 | [2.2184, 8.6949] |  |
| arm | LONGITUDINAL | 2-6s | tg_err | hold | T1-kin | 130 | 0.9990 | [0.4512, 1.6895] |  |
| arm | LONGITUDINAL | 2-6s | tg_err | prior | T1-kin | 109 | 1.4163 | [0.6151, 2.4912] |  |
| contrast | LONGITUDINAL | 2-6s | tg_err | hwm_oracle - hold | T0 | 72 | 1.4882 | [0.2610, 3.4765] | worse |
| contrast | LONGITUDINAL | 2-6s | tg_err | hwm_oracle - flat_oracle | T0 | 38 | -0.9591 | [-4.8066, 2.6874] | none |
| contrast | LONGITUDINAL | 2-6s | tg_err | flat_oracle - hold | T0 | 45 | 1.4825 | [0.1986, 2.9700] | worse |
| arm | LONGITUDINAL | 2-6s | ttc_err | hwm_oracle | T0 | 115 | 8.7355 | [5.8478, 12.0769] |  |
| arm | LONGITUDINAL | 2-6s | ttc_err | flat_oracle | T0 | 78 | 10.2332 | [6.6274, 13.3647] |  |
| arm | LONGITUDINAL | 2-6s | ttc_err | hold | T1-kin | 156 | 4.5309 | [2.6365, 6.7093] |  |
| arm | LONGITUDINAL | 2-6s | ttc_err | prior | T1-kin | 138 | 4.0464 | [1.9436, 6.7985] |  |
| contrast | LONGITUDINAL | 2-6s | ttc_err | hwm_oracle - hold | T0 | 102 | 3.6798 | [0.8608, 6.5206] | worse |
| contrast | LONGITUDINAL | 2-6s | ttc_err | hwm_oracle - flat_oracle | T0 | 60 | -1.0900 | [-7.1804, 6.6941] | none |
| contrast | LONGITUDINAL | 2-6s | ttc_err | flat_oracle - hold | T0 | 70 | 5.4744 | [-0.4701, 10.2656] | none |
| arm | LONGITUDINAL | 2-6s | hw_d | hwm_oracle | T0 | 115 | 1.5144 | [-0.3098, 3.2112] |  |
| arm | LONGITUDINAL | 2-6s | hw_d | flat_oracle | T0 | 78 | 0.2185 | [-2.1248, 3.7488] |  |
| arm | LONGITUDINAL | 2-6s | hw_d | hold | T1-kin | 156 | -0.6653 | [-1.6122, 0.2102] |  |
| arm | LONGITUDINAL | 2-6s | hw_d | prior | T1-kin | 138 | 0.6935 | [-0.0669, 1.4335] |  |
| arm | LONGITUDINAL | 2-6s | tg_d | hwm_oracle | T0 | 89 | 2.0894 | [0.4534, 4.1789] |  |
| arm | LONGITUDINAL | 2-6s | tg_d | flat_oracle | T0 | 60 | 2.4388 | [0.1034, 6.6053] |  |
| arm | LONGITUDINAL | 2-6s | tg_d | hold | T1-kin | 130 | 0.0848 | [-0.4009, 0.5753] |  |
| arm | LONGITUDINAL | 2-6s | tg_d | prior | T1-kin | 109 | 0.7769 | [-0.1402, 1.8298] |  |
| arm | LONGITUDINAL | 2-6s | ttc_d | hwm_oracle | T0 | 115 | 1.2595 | [-2.8712, 5.5455] |  |
| arm | LONGITUDINAL | 2-6s | ttc_d | flat_oracle | T0 | 78 | -3.9825 | [-8.8255, 2.1589] |  |
| arm | LONGITUDINAL | 2-6s | ttc_d | hold | T1-kin | 156 | -0.1845 | [-1.9394, 1.4295] |  |
| arm | LONGITUDINAL | 2-6s | ttc_d | prior | T1-kin | 138 | 0.9250 | [-0.8270, 2.8234] |  |
| arm | LONGITUDINAL | full6s | acc_mae | hwm_oracle | T0 | 994 | 1.0681 | [1.0240, 1.1164] |  |
| arm | LONGITUDINAL | full6s | acc_mae | flat_oracle | T0 | 994 | 1.3123 | [1.2450, 1.3794] |  |
| arm | LONGITUDINAL | full6s | acc_mae | hold | T1-kin | 994 | 0.5230 | [0.4449, 0.6017] |  |
| arm | LONGITUDINAL | full6s | acc_mae | prior | T1-kin | 994 | 0.4722 | [0.4021, 0.5482] |  |
| contrast | LONGITUDINAL | full6s | acc_mae | hwm_oracle - hold | T0 | 994 | 0.5450 | [0.4907, 0.5977] | worse |
| contrast | LONGITUDINAL | full6s | acc_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.2442 | [-0.3002, -0.1895] | better |
| contrast | LONGITUDINAL | full6s | acc_mae | flat_oracle - hold | T0 | 994 | 0.7892 | [0.7117, 0.8677] | worse |
| arm | LONGITUDINAL | full6s | speed_err | hwm_oracle | T0 | 994 | 2.4376 | [2.2049, 2.7049] |  |
| arm | LONGITUDINAL | full6s | speed_err | flat_oracle | T0 | 994 | 3.0200 | [2.6012, 3.5007] |  |
| arm | LONGITUDINAL | full6s | speed_err | hold | T1-kin | 994 | 2.3983 | [2.0456, 2.7712] |  |
| arm | LONGITUDINAL | full6s | speed_err | prior | T1-kin | 994 | 2.0577 | [1.7073, 2.4384] |  |
| contrast | LONGITUDINAL | full6s | speed_err | hwm_oracle - hold | T0 | 994 | 0.0393 | [-0.2650, 0.3581] | none |
| contrast | LONGITUDINAL | full6s | speed_err | hwm_oracle - flat_oracle | T0 | 994 | -0.5824 | [-1.0111, -0.1895] | better |
| contrast | LONGITUDINAL | full6s | speed_err | flat_oracle - hold | T0 | 994 | 0.6218 | [0.1219, 1.1779] | worse |
| arm | LONGITUDINAL | full6s | hw_err | hwm_oracle | T0 | 214 | 3.4399 | [1.9923, 5.2535] |  |
| arm | LONGITUDINAL | full6s | hw_err | flat_oracle | T0 | 214 | 3.7521 | [2.2915, 5.6812] |  |
| arm | LONGITUDINAL | full6s | hw_err | hold | T1-kin | 214 | 1.9745 | [1.0280, 3.0946] |  |
| arm | LONGITUDINAL | full6s | hw_err | prior | T1-kin | 214 | 2.3291 | [1.2186, 3.7639] |  |
| contrast | LONGITUDINAL | full6s | hw_err | hwm_oracle - hold | T0 | 214 | 1.4653 | [0.5422, 2.6507] | worse |
| contrast | LONGITUDINAL | full6s | hw_err | hwm_oracle - flat_oracle | T0 | 214 | -0.3123 | [-1.0108, 0.4292] | none |
| contrast | LONGITUDINAL | full6s | hw_err | flat_oracle - hold | T0 | 214 | 1.7776 | [0.6543, 3.1602] | worse |
| arm | LONGITUDINAL | full6s | tg_err | hwm_oracle | T0 | 182 | 0.5818 | [0.3353, 0.9141] |  |
| arm | LONGITUDINAL | full6s | tg_err | flat_oracle | T0 | 182 | 0.6623 | [0.3461, 1.0713] |  |
| arm | LONGITUDINAL | full6s | tg_err | hold | T1-kin | 182 | 0.3965 | [0.1939, 0.7000] |  |
| arm | LONGITUDINAL | full6s | tg_err | prior | T1-kin | 182 | 0.4195 | [0.2360, 0.6652] |  |
| contrast | LONGITUDINAL | full6s | tg_err | hwm_oracle - hold | T0 | 182 | 0.1853 | [0.0521, 0.3347] | worse |
| contrast | LONGITUDINAL | full6s | tg_err | hwm_oracle - flat_oracle | T0 | 182 | -0.0805 | [-0.3480, 0.1209] | none |
| contrast | LONGITUDINAL | full6s | tg_err | flat_oracle - hold | T0 | 182 | 0.2658 | [-0.0018, 0.6017] | none |
| arm | LONGITUDINAL | full6s | ttc_err | hwm_oracle | T0 | 214 | 6.5614 | [5.1058, 8.1832] |  |
| arm | LONGITUDINAL | full6s | ttc_err | flat_oracle | T0 | 214 | 6.0172 | [4.1896, 8.1411] |  |
| arm | LONGITUDINAL | full6s | ttc_err | hold | T1-kin | 214 | 2.6508 | [1.4524, 3.9708] |  |
| arm | LONGITUDINAL | full6s | ttc_err | prior | T1-kin | 214 | 3.9406 | [2.3891, 5.7889] |  |
| contrast | LONGITUDINAL | full6s | ttc_err | hwm_oracle - hold | T0 | 214 | 3.9106 | [2.4260, 5.6169] | worse |
| contrast | LONGITUDINAL | full6s | ttc_err | hwm_oracle - flat_oracle | T0 | 214 | 0.5443 | [-1.3525, 2.4382] | none |
| contrast | LONGITUDINAL | full6s | ttc_err | flat_oracle - hold | T0 | 214 | 3.3663 | [1.2540, 5.8156] | worse |
| arm | LONGITUDINAL | full6s | hw_d | hwm_oracle | T0 | 214 | 0.7678 | [-0.8066, 2.5358] |  |
| arm | LONGITUDINAL | full6s | hw_d | flat_oracle | T0 | 214 | 0.7693 | [-0.7299, 2.5122] |  |
| arm | LONGITUDINAL | full6s | hw_d | hold | T1-kin | 214 | -0.8009 | [-1.7679, 0.1023] |  |
| arm | LONGITUDINAL | full6s | hw_d | prior | T1-kin | 214 | 0.1250 | [-0.8603, 0.9903] |  |
| arm | LONGITUDINAL | full6s | tg_d | hwm_oracle | T0 | 182 | 0.0678 | [-0.1523, 0.3024] |  |
| arm | LONGITUDINAL | full6s | tg_d | flat_oracle | T0 | 182 | -0.0263 | [-0.4380, 0.3102] |  |
| arm | LONGITUDINAL | full6s | tg_d | hold | T1-kin | 182 | -0.1795 | [-0.4014, 0.0175] |  |
| arm | LONGITUDINAL | full6s | tg_d | prior | T1-kin | 182 | -0.0930 | [-0.3082, 0.0799] |  |
| arm | LONGITUDINAL | full6s | ttc_d | hwm_oracle | T0 | 214 | 0.4007 | [-2.2789, 2.8233] |  |
| arm | LONGITUDINAL | full6s | ttc_d | flat_oracle | T0 | 214 | -0.5714 | [-3.5376, 1.9510] |  |
| arm | LONGITUDINAL | full6s | ttc_d | hold | T1-kin | 214 | 0.1568 | [-0.9165, 1.3114] |  |
| arm | LONGITUDINAL | full6s | ttc_d | prior | T1-kin | 214 | 0.8048 | [-1.0604, 2.7069] |  |
| arm | LATERAL | 0-2s | yaw_mae | hwm_oracle | T0 | 994 | 0.1221 | [0.1115, 0.1329] |  |
| arm | LATERAL | 0-2s | yaw_mae | flat_oracle | T0 | 994 | 0.1256 | [0.1142, 0.1380] |  |
| arm | LATERAL | 0-2s | yaw_mae | hold | T1-kin | 994 | 0.0225 | [0.0174, 0.0280] |  |
| arm | LATERAL | 0-2s | yaw_mae | prior | T1-kin | 994 | 0.0416 | [0.0315, 0.0530] |  |
| contrast | LATERAL | 0-2s | yaw_mae | hwm_oracle - hold | T0 | 994 | 0.0996 | [0.0913, 0.1077] | worse |
| contrast | LATERAL | 0-2s | yaw_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.0034 | [-0.0143, 0.0074] | none |
| contrast | LATERAL | 0-2s | yaw_mae | flat_oracle - hold | T0 | 994 | 0.1031 | [0.0935, 0.1135] | worse |
| arm | LATERAL | 0-2s | head_err | hwm_oracle | T0 | 994 | 0.1589 | [0.1395, 0.1801] |  |
| arm | LATERAL | 0-2s | head_err | flat_oracle | T0 | 994 | 0.1944 | [0.1737, 0.2171] |  |
| arm | LATERAL | 0-2s | head_err | hold | T1-kin | 994 | 0.0423 | [0.0325, 0.0530] |  |
| arm | LATERAL | 0-2s | head_err | prior | T1-kin | 994 | 0.0792 | [0.0592, 0.1015] |  |
| contrast | LATERAL | 0-2s | head_err | hwm_oracle - hold | T0 | 994 | 0.1166 | [0.1026, 0.1309] | worse |
| contrast | LATERAL | 0-2s | head_err | hwm_oracle - flat_oracle | T0 | 994 | -0.0355 | [-0.0552, -0.0159] | better |
| contrast | LATERAL | 0-2s | head_err | flat_oracle - hold | T0 | 994 | 0.1521 | [0.1341, 0.1708] | worse |
| arm | LATERAL | 0-2s | curv_err | hwm_oracle | T0 | 950 | 0.0208 | [0.0155, 0.0270] |  |
| arm | LATERAL | 0-2s | curv_err | flat_oracle | T0 | 950 | 0.0205 | [0.0156, 0.0264] |  |
| arm | LATERAL | 0-2s | curv_err | hold | T1-kin | 950 | 0.0040 | [0.0028, 0.0054] |  |
| arm | LATERAL | 0-2s | curv_err | prior | T1-kin | 950 | 0.0076 | [0.0050, 0.0107] |  |
| contrast | LATERAL | 0-2s | curv_err | hwm_oracle - hold | T0 | 950 | 0.0168 | [0.0123, 0.0221] | worse |
| contrast | LATERAL | 0-2s | curv_err | hwm_oracle - flat_oracle | T0 | 950 | 0.0003 | [-0.0025, 0.0035] | none |
| contrast | LATERAL | 0-2s | curv_err | flat_oracle - hold | T0 | 950 | 0.0165 | [0.0125, 0.0215] | worse |
| arm | LATERAL | 0-2s | xtrack | hwm_oracle | T0 | 994 | 1.6358 | [1.3884, 1.9219] |  |
| arm | LATERAL | 0-2s | xtrack | flat_oracle | T0 | 994 | 2.1650 | [1.8093, 2.5439] |  |
| arm | LATERAL | 0-2s | xtrack | hold | T1-kin | 994 | 0.2255 | [0.1803, 0.2757] |  |
| arm | LATERAL | 0-2s | xtrack | prior | T1-kin | 994 | 0.6216 | [0.4594, 0.8038] |  |
| contrast | LATERAL | 0-2s | xtrack | hwm_oracle - hold | T0 | 994 | 1.4103 | [1.1696, 1.6686] | worse |
| contrast | LATERAL | 0-2s | xtrack | hwm_oracle - flat_oracle | T0 | 994 | -0.5292 | [-0.8079, -0.2662] | better |
| contrast | LATERAL | 0-2s | xtrack | flat_oracle - hold | T0 | 994 | 1.9394 | [1.5939, 2.3132] | worse |
| arm | LATERAL | 0-2s | ade | hwm_oracle | T0 | 994 | 0.9737 | [0.8812, 1.0754] |  |
| arm | LATERAL | 0-2s | ade | flat_oracle | T0 | 994 | 1.2833 | [1.1656, 1.4154] |  |
| arm | LATERAL | 0-2s | ade | hold | T1-kin | 994 | 0.1370 | [0.1213, 0.1542] |  |
| arm | LATERAL | 0-2s | ade | prior | T1-kin | 994 | 0.4497 | [0.3824, 0.5207] |  |
| contrast | LATERAL | 0-2s | ade | hwm_oracle - hold | T0 | 994 | 0.8368 | [0.7453, 0.9350] | worse |
| contrast | LATERAL | 0-2s | ade | hwm_oracle - flat_oracle | T0 | 994 | -0.3096 | [-0.4068, -0.2182] | better |
| contrast | LATERAL | 0-2s | ade | flat_oracle - hold | T0 | 994 | 1.1464 | [1.0261, 1.2811] | worse |
| arm | LATERAL | 2-6s | yaw_mae | hwm_oracle | T0 | 994 | 0.1031 | [0.0961, 0.1108] |  |
| arm | LATERAL | 2-6s | yaw_mae | flat_oracle | T0 | 994 | 0.1114 | [0.1018, 0.1207] |  |
| arm | LATERAL | 2-6s | yaw_mae | hold | T1-kin | 994 | 0.0523 | [0.0397, 0.0660] |  |
| arm | LATERAL | 2-6s | yaw_mae | prior | T1-kin | 994 | 0.0423 | [0.0312, 0.0544] |  |
| contrast | LATERAL | 2-6s | yaw_mae | hwm_oracle - hold | T0 | 994 | 0.0508 | [0.0424, 0.0586] | worse |
| contrast | LATERAL | 2-6s | yaw_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.0083 | [-0.0155, -0.0013] | better |
| contrast | LATERAL | 2-6s | yaw_mae | flat_oracle - hold | T0 | 994 | 0.0591 | [0.0491, 0.0689] | worse |
| arm | LATERAL | 2-6s | head_err | hwm_oracle | T0 | 994 | 0.2743 | [0.2324, 0.3210] |  |
| arm | LATERAL | 2-6s | head_err | flat_oracle | T0 | 994 | 0.3953 | [0.3484, 0.4471] |  |
| arm | LATERAL | 2-6s | head_err | hold | T1-kin | 994 | 0.2187 | [0.1655, 0.2784] |  |
| arm | LATERAL | 2-6s | head_err | prior | T1-kin | 994 | 0.2070 | [0.1509, 0.2669] |  |
| contrast | LATERAL | 2-6s | head_err | hwm_oracle - hold | T0 | 994 | 0.0556 | [0.0273, 0.0833] | worse |
| contrast | LATERAL | 2-6s | head_err | hwm_oracle - flat_oracle | T0 | 994 | -0.1210 | [-0.1570, -0.0846] | better |
| contrast | LATERAL | 2-6s | head_err | flat_oracle - hold | T0 | 994 | 0.1766 | [0.1298, 0.2236] | worse |
| arm | LATERAL | 2-6s | curv_err | hwm_oracle | T0 | 967 | 0.0184 | [0.0143, 0.0228] |  |
| arm | LATERAL | 2-6s | curv_err | flat_oracle | T0 | 967 | 0.0218 | [0.0158, 0.0286] |  |
| arm | LATERAL | 2-6s | curv_err | hold | T1-kin | 967 | 0.0102 | [0.0069, 0.0141] |  |
| arm | LATERAL | 2-6s | curv_err | prior | T1-kin | 967 | 0.0074 | [0.0049, 0.0104] |  |
| contrast | LATERAL | 2-6s | curv_err | hwm_oracle - hold | T0 | 967 | 0.0082 | [0.0054, 0.0113] | worse |
| contrast | LATERAL | 2-6s | curv_err | hwm_oracle - flat_oracle | T0 | 967 | -0.0033 | [-0.0067, -0.0003] | better |
| contrast | LATERAL | 2-6s | curv_err | flat_oracle - hold | T0 | 967 | 0.0116 | [0.0068, 0.0169] | worse |
| arm | LATERAL | 2-6s | xtrack | hwm_oracle | T0 | 994 | 9.5673 | [8.1639, 11.1465] |  |
| arm | LATERAL | 2-6s | xtrack | flat_oracle | T0 | 994 | 14.7407 | [12.4162, 17.2322] |  |
| arm | LATERAL | 2-6s | xtrack | hold | T1-kin | 994 | 4.1409 | [3.2343, 5.1397] |  |
| arm | LATERAL | 2-6s | xtrack | prior | T1-kin | 994 | 5.2076 | [3.9008, 6.6983] |  |
| contrast | LATERAL | 2-6s | xtrack | hwm_oracle - hold | T0 | 994 | 5.4264 | [4.1037, 6.7488] | worse |
| contrast | LATERAL | 2-6s | xtrack | hwm_oracle - flat_oracle | T0 | 994 | -5.1734 | [-7.1202, -3.3526] | better |
| contrast | LATERAL | 2-6s | xtrack | flat_oracle - hold | T0 | 994 | 10.5999 | [8.3757, 13.1202] | worse |
| arm | LATERAL | 2-6s | ade | hwm_oracle | T0 | 994 | 8.6355 | [7.7789, 9.5640] |  |
| arm | LATERAL | 2-6s | ade | flat_oracle | T0 | 994 | 12.4210 | [11.1151, 13.8115] |  |
| arm | LATERAL | 2-6s | ade | hold | T1-kin | 994 | 3.7452 | [3.2261, 4.3240] |  |
| arm | LATERAL | 2-6s | ade | prior | T1-kin | 994 | 5.3660 | [4.5195, 6.2639] |  |
| contrast | LATERAL | 2-6s | ade | hwm_oracle - hold | T0 | 994 | 4.8903 | [4.1098, 5.7169] | worse |
| contrast | LATERAL | 2-6s | ade | hwm_oracle - flat_oracle | T0 | 994 | -3.7854 | [-4.9104, -2.7465] | better |
| contrast | LATERAL | 2-6s | ade | flat_oracle - hold | T0 | 994 | 8.6758 | [7.3517, 10.1000] | worse |
| arm | LATERAL | full6s | yaw_mae | hwm_oracle | T0 | 994 | 0.1094 | [0.1021, 0.1173] |  |
| arm | LATERAL | full6s | yaw_mae | flat_oracle | T0 | 994 | 0.1161 | [0.1069, 0.1258] |  |
| arm | LATERAL | full6s | yaw_mae | hold | T1-kin | 994 | 0.0424 | [0.0325, 0.0532] |  |
| arm | LATERAL | full6s | yaw_mae | prior | T1-kin | 994 | 0.0420 | [0.0315, 0.0537] |  |
| contrast | LATERAL | full6s | yaw_mae | hwm_oracle - hold | T0 | 994 | 0.0671 | [0.0615, 0.0724] | worse |
| contrast | LATERAL | full6s | yaw_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.0067 | [-0.0137, 0.0001] | none |
| contrast | LATERAL | full6s | yaw_mae | flat_oracle - hold | T0 | 994 | 0.0737 | [0.0659, 0.0817] | worse |
| arm | LATERAL | full6s | head_err | hwm_oracle | T0 | 994 | 0.2743 | [0.2324, 0.3210] |  |
| arm | LATERAL | full6s | head_err | flat_oracle | T0 | 994 | 0.3953 | [0.3484, 0.4471] |  |
| arm | LATERAL | full6s | head_err | hold | T1-kin | 994 | 0.2187 | [0.1655, 0.2784] |  |
| arm | LATERAL | full6s | head_err | prior | T1-kin | 994 | 0.2070 | [0.1509, 0.2669] |  |
| contrast | LATERAL | full6s | head_err | hwm_oracle - hold | T0 | 994 | 0.0556 | [0.0273, 0.0833] | worse |
| contrast | LATERAL | full6s | head_err | hwm_oracle - flat_oracle | T0 | 994 | -0.1210 | [-0.1570, -0.0846] | better |
| contrast | LATERAL | full6s | head_err | flat_oracle - hold | T0 | 994 | 0.1766 | [0.1298, 0.2236] | worse |
| arm | LATERAL | full6s | curv_err | hwm_oracle | T0 | 974 | 0.0198 | [0.0151, 0.0252] |  |
| arm | LATERAL | full6s | curv_err | flat_oracle | T0 | 974 | 0.0217 | [0.0161, 0.0280] |  |
| arm | LATERAL | full6s | curv_err | hold | T1-kin | 974 | 0.0081 | [0.0056, 0.0109] |  |
| arm | LATERAL | full6s | curv_err | prior | T1-kin | 974 | 0.0077 | [0.0050, 0.0110] |  |
| contrast | LATERAL | full6s | curv_err | hwm_oracle - hold | T0 | 974 | 0.0118 | [0.0083, 0.0157] | worse |
| contrast | LATERAL | full6s | curv_err | hwm_oracle - flat_oracle | T0 | 974 | -0.0019 | [-0.0045, 0.0006] | none |
| contrast | LATERAL | full6s | curv_err | flat_oracle - hold | T0 | 974 | 0.0137 | [0.0093, 0.0188] | worse |
| arm | LATERAL | full6s | xtrack | hwm_oracle | T0 | 994 | 9.5673 | [8.1639, 11.1465] |  |
| arm | LATERAL | full6s | xtrack | flat_oracle | T0 | 994 | 14.7407 | [12.4162, 17.2322] |  |
| arm | LATERAL | full6s | xtrack | hold | T1-kin | 994 | 4.1409 | [3.2343, 5.1397] |  |
| arm | LATERAL | full6s | xtrack | prior | T1-kin | 994 | 5.2076 | [3.9008, 6.6983] |  |
| contrast | LATERAL | full6s | xtrack | hwm_oracle - hold | T0 | 994 | 5.4264 | [4.1037, 6.7488] | worse |
| contrast | LATERAL | full6s | xtrack | hwm_oracle - flat_oracle | T0 | 994 | -5.1734 | [-7.1202, -3.3526] | better |
| contrast | LATERAL | full6s | xtrack | flat_oracle - hold | T0 | 994 | 10.5999 | [8.3757, 13.1202] | worse |
| arm | LATERAL | full6s | ade | hwm_oracle | T0 | 994 | 6.0816 | [5.4809, 6.7324] |  |
| arm | LATERAL | full6s | ade | flat_oracle | T0 | 994 | 8.7084 | [7.8028, 9.6753] |  |
| arm | LATERAL | full6s | ade | hold | T1-kin | 994 | 2.5424 | [2.1925, 2.9329] |  |
| arm | LATERAL | full6s | ade | prior | T1-kin | 994 | 3.7272 | [3.1424, 4.3483] |  |
| contrast | LATERAL | full6s | ade | hwm_oracle - hold | T0 | 994 | 3.5392 | [2.9903, 4.1254] | worse |
| contrast | LATERAL | full6s | ade | hwm_oracle - flat_oracle | T0 | 994 | -2.6268 | [-3.3978, -1.9033] | better |
| contrast | LATERAL | full6s | ade | flat_oracle - hold | T0 | 994 | 6.1660 | [5.2497, 7.1475] | worse |
| arm | TACTICAL | 0-2s | tac_man_dis | hwm_oracle | T0 | 994 | 0.6559 | [0.6107, 0.6942] |  |
| arm | TACTICAL | 0-2s | tac_man_dis | flat_oracle | T0 | 994 | 0.7686 | [0.7254, 0.8119] |  |
| arm | TACTICAL | 0-2s | tac_man_dis | hold | T1-kin | 994 | 0.1781 | [0.1439, 0.2143] |  |
| arm | TACTICAL | 0-2s | tac_man_dis | prior | T1-kin | 994 | 0.3531 | [0.2857, 0.4256] |  |
| contrast | TACTICAL | 0-2s | tac_man_dis | hwm_oracle - hold | T0 | 994 | 0.4779 | [0.4276, 0.5252] | worse |
| contrast | TACTICAL | 0-2s | tac_man_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.1127 | [-0.1680, -0.0593] | better |
| contrast | TACTICAL | 0-2s | tac_man_dis | flat_oracle - hold | T0 | 994 | 0.5905 | [0.5372, 0.6449] | worse |
| arm | TACTICAL | 0-2s | tac_lat_dis | hwm_oracle | T0 | 994 | 0.3581 | [0.3078, 0.4075] |  |
| arm | TACTICAL | 0-2s | tac_lat_dis | flat_oracle | T0 | 994 | 0.4467 | [0.3873, 0.5070] |  |
| arm | TACTICAL | 0-2s | tac_lat_dis | hold | T1-kin | 994 | 0.0694 | [0.0493, 0.0926] |  |
| arm | TACTICAL | 0-2s | tac_lat_dis | prior | T1-kin | 994 | 0.1419 | [0.0926, 0.1962] |  |
| contrast | TACTICAL | 0-2s | tac_lat_dis | hwm_oracle - hold | T0 | 994 | 0.2887 | [0.2455, 0.3330] | worse |
| contrast | TACTICAL | 0-2s | tac_lat_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.0885 | [-0.1569, -0.0181] | better |
| contrast | TACTICAL | 0-2s | tac_lat_dis | flat_oracle - hold | T0 | 994 | 0.3773 | [0.3199, 0.4366] | worse |
| arm | TACTICAL | 0-2s | tac_lon_dis | hwm_oracle | T0 | 994 | 0.5262 | [0.4769, 0.5734] |  |
| arm | TACTICAL | 0-2s | tac_lon_dis | flat_oracle | T0 | 994 | 0.6881 | [0.6419, 0.7314] |  |
| arm | TACTICAL | 0-2s | tac_lon_dis | hold | T1-kin | 994 | 0.1419 | [0.1117, 0.1740] |  |
| arm | TACTICAL | 0-2s | tac_lon_dis | prior | T1-kin | 994 | 0.2757 | [0.2143, 0.3421] |  |
| contrast | TACTICAL | 0-2s | tac_lon_dis | hwm_oracle - hold | T0 | 994 | 0.3843 | [0.3330, 0.4356] | worse |
| contrast | TACTICAL | 0-2s | tac_lon_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.1620 | [-0.2153, -0.1056] | better |
| contrast | TACTICAL | 0-2s | tac_lon_dis | flat_oracle - hold | T0 | 994 | 0.5463 | [0.4899, 0.5976] | worse |
| arm | TACTICAL | 2-6s | tac_man_dis | hwm_oracle | T0 | 994 | 0.6690 | [0.6237, 0.7143] |  |
| arm | TACTICAL | 2-6s | tac_man_dis | flat_oracle | T0 | 994 | 0.7545 | [0.7062, 0.8008] |  |
| arm | TACTICAL | 2-6s | tac_man_dis | hold | T1-kin | 994 | 0.5181 | [0.4537, 0.5845] |  |
| arm | TACTICAL | 2-6s | tac_man_dis | prior | T1-kin | 994 | 0.5734 | [0.4889, 0.6559] |  |
| contrast | TACTICAL | 2-6s | tac_man_dis | hwm_oracle - hold | T0 | 994 | 0.1509 | [0.0875, 0.2143] | worse |
| contrast | TACTICAL | 2-6s | tac_man_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.0855 | [-0.1449, -0.0241] | better |
| contrast | TACTICAL | 2-6s | tac_man_dis | flat_oracle - hold | T0 | 994 | 0.2364 | [0.1559, 0.3189] | worse |
| arm | TACTICAL | 2-6s | tac_lat_dis | hwm_oracle | T0 | 994 | 0.3652 | [0.3159, 0.4185] |  |
| arm | TACTICAL | 2-6s | tac_lat_dis | flat_oracle | T0 | 994 | 0.5644 | [0.5070, 0.6227] |  |
| arm | TACTICAL | 2-6s | tac_lat_dis | hold | T1-kin | 994 | 0.2364 | [0.1781, 0.2988] |  |
| arm | TACTICAL | 2-6s | tac_lat_dis | prior | T1-kin | 994 | 0.2495 | [0.1751, 0.3230] |  |
| contrast | TACTICAL | 2-6s | tac_lat_dis | hwm_oracle - hold | T0 | 994 | 0.1288 | [0.0835, 0.1730] | worse |
| contrast | TACTICAL | 2-6s | tac_lat_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.1992 | [-0.2726, -0.1247] | better |
| contrast | TACTICAL | 2-6s | tac_lat_dis | flat_oracle - hold | T0 | 994 | 0.3280 | [0.2505, 0.4064] | worse |
| arm | TACTICAL | 2-6s | tac_lon_dis | hwm_oracle | T0 | 994 | 0.5463 | [0.5000, 0.5936] |  |
| arm | TACTICAL | 2-6s | tac_lon_dis | flat_oracle | T0 | 994 | 0.5915 | [0.5443, 0.6368] |  |
| arm | TACTICAL | 2-6s | tac_lon_dis | hold | T1-kin | 994 | 0.4507 | [0.3893, 0.5121] |  |
| arm | TACTICAL | 2-6s | tac_lon_dis | prior | T1-kin | 994 | 0.4588 | [0.3803, 0.5383] |  |
| contrast | TACTICAL | 2-6s | tac_lon_dis | hwm_oracle - hold | T0 | 994 | 0.0956 | [0.0261, 0.1640] | worse |
| contrast | TACTICAL | 2-6s | tac_lon_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.0453 | [-0.1016, 0.0121] | none |
| contrast | TACTICAL | 2-6s | tac_lon_dis | flat_oracle - hold | T0 | 994 | 0.1408 | [0.0563, 0.2304] | worse |
| arm | TACTICAL | full6s | tac_man_dis | hwm_oracle | T0 | 994 | 0.7193 | [0.6771, 0.7616] |  |
| arm | TACTICAL | full6s | tac_man_dis | flat_oracle | T0 | 994 | 0.7596 | [0.7103, 0.8078] |  |
| arm | TACTICAL | full6s | tac_man_dis | hold | T1-kin | 994 | 0.4769 | [0.4225, 0.5312] |  |
| arm | TACTICAL | full6s | tac_man_dis | prior | T1-kin | 994 | 0.6539 | [0.5704, 0.7324] |  |
| contrast | TACTICAL | full6s | tac_man_dis | hwm_oracle - hold | T0 | 994 | 0.2425 | [0.1770, 0.3079] | worse |
| contrast | TACTICAL | full6s | tac_man_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.0402 | [-0.0885, 0.0141] | none |
| contrast | TACTICAL | full6s | tac_man_dis | flat_oracle - hold | T0 | 994 | 0.2827 | [0.2113, 0.3531] | worse |
| arm | TACTICAL | full6s | tac_lat_dis | hwm_oracle | T0 | 994 | 0.4980 | [0.4557, 0.5382] |  |
| arm | TACTICAL | full6s | tac_lat_dis | flat_oracle | T0 | 994 | 0.6137 | [0.5624, 0.6670] |  |
| arm | TACTICAL | full6s | tac_lat_dis | hold | T1-kin | 994 | 0.2364 | [0.1891, 0.2857] |  |
| arm | TACTICAL | full6s | tac_lat_dis | prior | T1-kin | 994 | 0.2968 | [0.2173, 0.3803] |  |
| contrast | TACTICAL | full6s | tac_lat_dis | hwm_oracle - hold | T0 | 994 | 0.2616 | [0.2082, 0.3129] | worse |
| contrast | TACTICAL | full6s | tac_lat_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.1157 | [-0.1690, -0.0594] | better |
| contrast | TACTICAL | full6s | tac_lat_dis | flat_oracle - hold | T0 | 994 | 0.3773 | [0.3089, 0.4467] | worse |
| arm | TACTICAL | full6s | tac_lon_dis | hwm_oracle | T0 | 994 | 0.5674 | [0.5181, 0.6147] |  |
| arm | TACTICAL | full6s | tac_lon_dis | flat_oracle | T0 | 994 | 0.5875 | [0.5261, 0.6469] |  |
| arm | TACTICAL | full6s | tac_lon_dis | hold | T1-kin | 994 | 0.4215 | [0.3712, 0.4718] |  |
| arm | TACTICAL | full6s | tac_lon_dis | prior | T1-kin | 994 | 0.5463 | [0.4658, 0.6258] |  |
| contrast | TACTICAL | full6s | tac_lon_dis | hwm_oracle - hold | T0 | 994 | 0.1459 | [0.0744, 0.2143] | worse |
| contrast | TACTICAL | full6s | tac_lon_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.0201 | [-0.0795, 0.0423] | none |
| contrast | TACTICAL | full6s | tac_lon_dis | flat_oracle - hold | T0 | 994 | 0.1660 | [0.0865, 0.2555] | worse |
| inference_seed_replicate | LONGITUDINAL | 0-2s | acc_mae | hwm_oracle | T0 | 994 | -0.0013 | [-0.0205, 0.0171] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | speed_err | hwm_oracle | T0 | 994 | -0.0390 | [-0.0895, 0.0067] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | hw_err | hwm_oracle | T0 | 214 | 0.1614 | [-0.0582, 0.4956] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | tg_err | hwm_oracle | T0 | 182 | 0.0272 | [-0.0113, 0.0795] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | ttc_err | hwm_oracle | T0 | 214 | 0.2716 | [-0.1101, 0.7352] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | acc_mae | hwm_oracle | T0 | 994 | -0.0158 | [-0.0319, -0.0010] | better |
| inference_seed_replicate | LONGITUDINAL | 2-6s | speed_err | hwm_oracle | T0 | 994 | -0.0137 | [-0.1113, 0.0936] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | hw_err | hwm_oracle | T0 | 87 | -0.6972 | [-1.4882, 0.3745] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | tg_err | hwm_oracle | T0 | 61 | -1.0633 | [-2.3269, 0.0335] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | ttc_err | hwm_oracle | T0 | 87 | -1.9813 | [-3.5907, -0.6370] | better |
| inference_seed_replicate | LONGITUDINAL | full6s | acc_mae | hwm_oracle | T0 | 994 | -0.0109 | [-0.0212, -0.0005] | better |
| inference_seed_replicate | LONGITUDINAL | full6s | speed_err | hwm_oracle | T0 | 994 | -0.0137 | [-0.1113, 0.0936] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | hw_err | hwm_oracle | T0 | 214 | 0.4640 | [-0.0948, 1.3300] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | tg_err | hwm_oracle | T0 | 182 | 0.0859 | [-0.0068, 0.2263] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | ttc_err | hwm_oracle | T0 | 214 | -0.7409 | [-1.6114, 0.0549] | none |
| inference_seed_replicate | LATERAL | 0-2s | yaw_mae | hwm_oracle | T0 | 994 | -0.0002 | [-0.0023, 0.0020] | none |
| inference_seed_replicate | LATERAL | 0-2s | head_err | hwm_oracle | T0 | 994 | 0.0014 | [-0.0037, 0.0065] | none |
| inference_seed_replicate | LATERAL | 0-2s | curv_err | hwm_oracle | T0 | 950 | -0.0004 | [-0.0012, 0.0004] | none |
| inference_seed_replicate | LATERAL | 0-2s | xtrack | hwm_oracle | T0 | 994 | -0.0093 | [-0.0676, 0.0489] | none |
| inference_seed_replicate | LATERAL | 2-6s | yaw_mae | hwm_oracle | T0 | 994 | 0.0002 | [-0.0013, 0.0018] | none |
| inference_seed_replicate | LATERAL | 2-6s | head_err | hwm_oracle | T0 | 994 | 0.0008 | [-0.0078, 0.0098] | none |
| inference_seed_replicate | LATERAL | 2-6s | curv_err | hwm_oracle | T0 | 967 | -0.0003 | [-0.0011, 0.0005] | none |
| inference_seed_replicate | LATERAL | 2-6s | xtrack | hwm_oracle | T0 | 994 | -0.0144 | [-0.3555, 0.3467] | none |
| inference_seed_replicate | LATERAL | full6s | yaw_mae | hwm_oracle | T0 | 994 | 0.0001 | [-0.0010, 0.0013] | none |
| inference_seed_replicate | LATERAL | full6s | head_err | hwm_oracle | T0 | 994 | 0.0008 | [-0.0078, 0.0098] | none |
| inference_seed_replicate | LATERAL | full6s | curv_err | hwm_oracle | T0 | 974 | -0.0005 | [-0.0011, 0.0001] | none |
| inference_seed_replicate | LATERAL | full6s | xtrack | hwm_oracle | T0 | 994 | -0.0144 | [-0.3555, 0.3467] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_man_dis | hwm_oracle | T0 | 994 | 0.0020 | [-0.0211, 0.0272] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_lat_dis | hwm_oracle | T0 | 994 | -0.0070 | [-0.0312, 0.0181] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_lon_dis | hwm_oracle | T0 | 994 | 0.0141 | [-0.0121, 0.0423] | none |
| inference_seed_replicate | TACTICAL | 2-6s | tac_man_dis | hwm_oracle | T0 | 994 | 0.0080 | [-0.0282, 0.0423] | none |
| inference_seed_replicate | TACTICAL | 2-6s | tac_lat_dis | hwm_oracle | T0 | 994 | -0.0372 | [-0.0704, -0.0040] | better |
| inference_seed_replicate | TACTICAL | 2-6s | tac_lon_dis | hwm_oracle | T0 | 994 | 0.0272 | [-0.0101, 0.0634] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_man_dis | hwm_oracle | T0 | 994 | 0.0070 | [-0.0191, 0.0342] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_lat_dis | hwm_oracle | T0 | 994 | 0.0091 | [-0.0262, 0.0443] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_lon_dis | hwm_oracle | T0 | 994 | -0.0151 | [-0.0483, 0.0191] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | acc_mae | flat_oracle | T0 | 994 | -0.0031 | [-0.0137, 0.0078] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | speed_err | flat_oracle | T0 | 994 | 0.0060 | [-0.0269, 0.0398] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | hw_err | flat_oracle | T0 | 214 | -0.0679 | [-0.2696, 0.0563] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | tg_err | flat_oracle | T0 | 182 | -0.0002 | [-0.0198, 0.0147] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | ttc_err | flat_oracle | T0 | 214 | -0.1832 | [-0.6716, 0.1306] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | acc_mae | flat_oracle | T0 | 994 | -0.0012 | [-0.0074, 0.0046] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | speed_err | flat_oracle | T0 | 994 | 0.0092 | [-0.0309, 0.0507] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | hw_err | flat_oracle | T0 | 76 | 0.2203 | [-0.2331, 0.8368] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | tg_err | flat_oracle | T0 | 57 | -0.0307 | [-0.7811, 0.7725] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | ttc_err | flat_oracle | T0 | 76 | -0.3626 | [-1.4025, 0.2355] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | acc_mae | flat_oracle | T0 | 994 | -0.0018 | [-0.0068, 0.0033] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | speed_err | flat_oracle | T0 | 994 | 0.0092 | [-0.0309, 0.0507] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | hw_err | flat_oracle | T0 | 214 | 0.0556 | [-0.2920, 0.4531] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | tg_err | flat_oracle | T0 | 182 | 0.0288 | [-0.0308, 0.1070] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | ttc_err | flat_oracle | T0 | 214 | -0.2206 | [-0.7476, 0.1659] | none |
| inference_seed_replicate | LATERAL | 0-2s | yaw_mae | flat_oracle | T0 | 994 | 0.0003 | [-0.0008, 0.0014] | none |
| inference_seed_replicate | LATERAL | 0-2s | head_err | flat_oracle | T0 | 994 | 0.0010 | [-0.0016, 0.0037] | none |
| inference_seed_replicate | LATERAL | 0-2s | curv_err | flat_oracle | T0 | 950 | 0.0002 | [-0.0000, 0.0006] | none |
| inference_seed_replicate | LATERAL | 0-2s | xtrack | flat_oracle | T0 | 994 | -0.0248 | [-0.0561, 0.0101] | none |
| inference_seed_replicate | LATERAL | 2-6s | yaw_mae | flat_oracle | T0 | 994 | 0.0001 | [-0.0006, 0.0008] | none |
| inference_seed_replicate | LATERAL | 2-6s | head_err | flat_oracle | T0 | 994 | -0.0003 | [-0.0045, 0.0040] | none |
| inference_seed_replicate | LATERAL | 2-6s | curv_err | flat_oracle | T0 | 967 | 0.0001 | [-0.0001, 0.0004] | none |
| inference_seed_replicate | LATERAL | 2-6s | xtrack | flat_oracle | T0 | 994 | 0.0400 | [-0.1781, 0.2285] | none |
| inference_seed_replicate | LATERAL | full6s | yaw_mae | flat_oracle | T0 | 994 | 0.0002 | [-0.0004, 0.0009] | none |
| inference_seed_replicate | LATERAL | full6s | head_err | flat_oracle | T0 | 994 | -0.0003 | [-0.0045, 0.0040] | none |
| inference_seed_replicate | LATERAL | full6s | curv_err | flat_oracle | T0 | 974 | 0.0002 | [-0.0000, 0.0005] | none |
| inference_seed_replicate | LATERAL | full6s | xtrack | flat_oracle | T0 | 994 | 0.0400 | [-0.1781, 0.2285] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_man_dis | flat_oracle | T0 | 994 | 0.0020 | [-0.0131, 0.0171] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_lat_dis | flat_oracle | T0 | 994 | 0.0000 | [-0.0141, 0.0151] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_lon_dis | flat_oracle | T0 | 994 | -0.0020 | [-0.0201, 0.0151] | none |
| inference_seed_replicate | TACTICAL | 2-6s | tac_man_dis | flat_oracle | T0 | 994 | 0.0050 | [-0.0091, 0.0181] | none |
| inference_seed_replicate | TACTICAL | 2-6s | tac_lat_dis | flat_oracle | T0 | 994 | 0.0040 | [-0.0111, 0.0201] | none |
| inference_seed_replicate | TACTICAL | 2-6s | tac_lon_dis | flat_oracle | T0 | 994 | 0.0091 | [-0.0060, 0.0241] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_man_dis | flat_oracle | T0 | 994 | -0.0091 | [-0.0191, 0.0010] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_lat_dis | flat_oracle | T0 | 994 | -0.0050 | [-0.0161, 0.0060] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_lon_dis | flat_oracle | T0 | 994 | -0.0040 | [-0.0181, 0.0101] | none |
