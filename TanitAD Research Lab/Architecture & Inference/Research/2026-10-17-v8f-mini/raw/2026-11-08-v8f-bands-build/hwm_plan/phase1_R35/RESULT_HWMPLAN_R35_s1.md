# HWM-PLAN read, every row stamped with its tier: T0 oracle-goal rows are diagnostics, T1-kin rows score the plan integrated by the kinematic model (994 windows, 71 episodes; estimator paired episode-cluster bootstrap, taniteval/ci.py)
STRATEGIC: PAUSED (PI item 25, 2026-09-19): not a pass, not omitted

* LATERAL.cross_track_vs_lane: n = 0: no lane geometry; xtrack here is the lateral offset of the plan end from the logged-action trajectory end
* TACTICAL.goal_setting: n = 0: the planner exposes no goal or anchor selection; TACTICAL here is the manoeuvre implied by the planned trajectory, not a vocabulary output

* headway scope: headway / time gap / TTC use taniteval.lead_metrics.distance_keeping over the b1 eval lead blocks (taniteval/tools/build_lead_block_b1.py): the 0-2 s band reads the banked K = 10 block (2 s lead track); the 2-6 s and full 6 s bands read the K = 30 block hwm_plan/data/b1_eval_lead_block_k30.npz (same sources, sha256 equal, 6 s lead track). A window needs a lead track and labelled span over the whole of the block's horizon, so the K = 30 block has fewer LEAD rows than the K = 10 block. In a band that starts after NOW the time-gap denominator is each path's own speed at the band start
* tactical scope: plan-derived decision, NOT a vocabulary output: the planned and the logged trajectory (both the kinematic integration of the action sequence from the same v0) are classified by ONE rule, tanitad.refs.refc_tactical.factor_from_kinematics through taniteval.four_families.maneuver_kinematics, on the path of each band re-expressed in the frame of the pose at the band start. The rule reads path geometry only (net path tangent, chord speeds); the goal that conditioned the plan is not an input of the rule. Thresholds are the programme's 2 s thresholds applied unchanged to each band (not rescaled to 4 s or 6 s)
* lead-state counts of the join: {'LEAD': 219, 'NO_LEAD': 273, 'NOT_STRAIGHT': 417, 'NO_LABEL': 85, 'NO_ROW': 0}

| kind | family | band | metric | row | tier | n | delta or mean | 95 % interval | label |
|---|---|---|---|---|---|---|---|---|---|
| arm | LONGITUDINAL | 0-2s | acc_mae | hwm_oracle | T0 | 994 | 1.1888 | [1.1194, 1.2560] |  |
| arm | LONGITUDINAL | 0-2s | acc_mae | flat_oracle | T0 | 994 | 1.3605 | [1.2850, 1.4347] |  |
| arm | LONGITUDINAL | 0-2s | acc_mae | hold | T1-kin | 994 | 0.2970 | [0.2571, 0.3372] |  |
| arm | LONGITUDINAL | 0-2s | acc_mae | prior | T1-kin | 994 | 0.4736 | [0.4017, 0.5504] |  |
| contrast | LONGITUDINAL | 0-2s | acc_mae | hwm_oracle - hold | T0 | 994 | 0.8917 | [0.8178, 0.9634] | worse |
| contrast | LONGITUDINAL | 0-2s | acc_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.1717 | [-0.2510, -0.0947] | better |
| contrast | LONGITUDINAL | 0-2s | acc_mae | flat_oracle - hold | T0 | 994 | 1.0635 | [0.9849, 1.1413] | worse |
| arm | LONGITUDINAL | 0-2s | speed_err | hwm_oracle | T0 | 994 | 1.4779 | [1.3503, 1.6264] |  |
| arm | LONGITUDINAL | 0-2s | speed_err | flat_oracle | T0 | 994 | 2.0597 | [1.9016, 2.2258] |  |
| arm | LONGITUDINAL | 0-2s | speed_err | hold | T1-kin | 994 | 0.5146 | [0.4421, 0.5901] |  |
| arm | LONGITUDINAL | 0-2s | speed_err | prior | T1-kin | 994 | 0.8602 | [0.7205, 1.0114] |  |
| contrast | LONGITUDINAL | 0-2s | speed_err | hwm_oracle - hold | T0 | 994 | 0.9633 | [0.8360, 1.0986] | worse |
| contrast | LONGITUDINAL | 0-2s | speed_err | hwm_oracle - flat_oracle | T0 | 994 | -0.5818 | [-0.7488, -0.4146] | better |
| contrast | LONGITUDINAL | 0-2s | speed_err | flat_oracle - hold | T0 | 994 | 1.5451 | [1.3854, 1.7025] | worse |
| arm | LONGITUDINAL | 0-2s | hw_err | hwm_oracle | T0 | 214 | 1.5260 | [0.8360, 2.5069] |  |
| arm | LONGITUDINAL | 0-2s | hw_err | flat_oracle | T0 | 214 | 1.1788 | [0.6346, 1.8269] |  |
| arm | LONGITUDINAL | 0-2s | hw_err | hold | T1-kin | 214 | 0.2887 | [0.1250, 0.5513] |  |
| arm | LONGITUDINAL | 0-2s | hw_err | prior | T1-kin | 214 | 0.8682 | [0.4108, 1.5049] |  |
| contrast | LONGITUDINAL | 0-2s | hw_err | hwm_oracle - hold | T0 | 214 | 1.2373 | [0.6100, 2.1635] | worse |
| contrast | LONGITUDINAL | 0-2s | hw_err | hwm_oracle - flat_oracle | T0 | 214 | 0.3472 | [-0.1303, 1.1110] | none |
| contrast | LONGITUDINAL | 0-2s | hw_err | flat_oracle - hold | T0 | 214 | 0.8901 | [0.4420, 1.4123] | worse |
| arm | LONGITUDINAL | 0-2s | tg_err | hwm_oracle | T0 | 182 | 0.2286 | [0.1281, 0.3766] |  |
| arm | LONGITUDINAL | 0-2s | tg_err | flat_oracle | T0 | 182 | 0.1952 | [0.1089, 0.3102] |  |
| arm | LONGITUDINAL | 0-2s | tg_err | hold | T1-kin | 182 | 0.0718 | [0.0287, 0.1329] |  |
| arm | LONGITUDINAL | 0-2s | tg_err | prior | T1-kin | 182 | 0.1621 | [0.0812, 0.2854] |  |
| contrast | LONGITUDINAL | 0-2s | tg_err | hwm_oracle - hold | T0 | 182 | 0.1568 | [0.0752, 0.2825] | worse |
| contrast | LONGITUDINAL | 0-2s | tg_err | hwm_oracle - flat_oracle | T0 | 182 | 0.0334 | [-0.0369, 0.1374] | none |
| contrast | LONGITUDINAL | 0-2s | tg_err | flat_oracle - hold | T0 | 182 | 0.1234 | [0.0603, 0.2033] | worse |
| arm | LONGITUDINAL | 0-2s | ttc_err | hwm_oracle | T0 | 214 | 4.2502 | [2.8783, 5.8674] |  |
| arm | LONGITUDINAL | 0-2s | ttc_err | flat_oracle | T0 | 214 | 4.7114 | [3.2312, 6.3827] |  |
| arm | LONGITUDINAL | 0-2s | ttc_err | hold | T1-kin | 214 | 0.7046 | [0.3033, 1.1573] |  |
| arm | LONGITUDINAL | 0-2s | ttc_err | prior | T1-kin | 214 | 2.3186 | [1.2131, 3.7213] |  |
| contrast | LONGITUDINAL | 0-2s | ttc_err | hwm_oracle - hold | T0 | 214 | 3.5456 | [2.2773, 5.1455] | worse |
| contrast | LONGITUDINAL | 0-2s | ttc_err | hwm_oracle - flat_oracle | T0 | 214 | -0.4612 | [-1.5572, 0.4908] | none |
| contrast | LONGITUDINAL | 0-2s | ttc_err | flat_oracle - hold | T0 | 214 | 4.0069 | [2.5149, 5.7015] | worse |
| arm | LONGITUDINAL | 0-2s | hw_d | hwm_oracle | T0 | 214 | 0.1662 | [-0.6031, 0.9133] |  |
| arm | LONGITUDINAL | 0-2s | hw_d | flat_oracle | T0 | 214 | 0.5613 | [0.0373, 1.1611] |  |
| arm | LONGITUDINAL | 0-2s | hw_d | hold | T1-kin | 214 | -0.1389 | [-0.4049, 0.0202] |  |
| arm | LONGITUDINAL | 0-2s | hw_d | prior | T1-kin | 214 | -0.3289 | [-0.7469, 0.0059] |  |
| arm | LONGITUDINAL | 0-2s | tg_d | hwm_oracle | T0 | 182 | 0.0398 | [-0.0616, 0.1585] |  |
| arm | LONGITUDINAL | 0-2s | tg_d | flat_oracle | T0 | 182 | 0.0678 | [-0.0014, 0.1501] |  |
| arm | LONGITUDINAL | 0-2s | tg_d | hold | T1-kin | 182 | -0.0415 | [-0.0964, 0.0009] |  |
| arm | LONGITUDINAL | 0-2s | tg_d | prior | T1-kin | 182 | -0.0635 | [-0.1463, -0.0029] |  |
| arm | LONGITUDINAL | 0-2s | ttc_d | hwm_oracle | T0 | 214 | -1.1516 | [-3.0611, 0.5903] |  |
| arm | LONGITUDINAL | 0-2s | ttc_d | flat_oracle | T0 | 214 | -0.2093 | [-2.1957, 1.8226] |  |
| arm | LONGITUDINAL | 0-2s | ttc_d | hold | T1-kin | 214 | 0.0112 | [-0.2654, 0.2760] |  |
| arm | LONGITUDINAL | 0-2s | ttc_d | prior | T1-kin | 214 | -0.5008 | [-1.7772, 0.5194] |  |
| arm | LONGITUDINAL | 2-6s | acc_mae | hwm_oracle | T0 | 994 | 0.8713 | [0.8265, 0.9175] |  |
| arm | LONGITUDINAL | 2-6s | acc_mae | flat_oracle | T0 | 994 | 1.1774 | [1.1051, 1.2513] |  |
| arm | LONGITUDINAL | 2-6s | acc_mae | hold | T1-kin | 994 | 0.6360 | [0.5385, 0.7374] |  |
| arm | LONGITUDINAL | 2-6s | acc_mae | prior | T1-kin | 994 | 0.4716 | [0.3989, 0.5502] |  |
| contrast | LONGITUDINAL | 2-6s | acc_mae | hwm_oracle - hold | T0 | 994 | 0.2353 | [0.1642, 0.3046] | worse |
| contrast | LONGITUDINAL | 2-6s | acc_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.3061 | [-0.3778, -0.2367] | better |
| contrast | LONGITUDINAL | 2-6s | acc_mae | flat_oracle - hold | T0 | 994 | 0.5414 | [0.4292, 0.6542] | worse |
| arm | LONGITUDINAL | 2-6s | speed_err | hwm_oracle | T0 | 994 | 2.3747 | [2.1747, 2.6157] |  |
| arm | LONGITUDINAL | 2-6s | speed_err | flat_oracle | T0 | 994 | 3.0840 | [2.7168, 3.4616] |  |
| arm | LONGITUDINAL | 2-6s | speed_err | hold | T1-kin | 994 | 2.3983 | [2.0456, 2.7712] |  |
| arm | LONGITUDINAL | 2-6s | speed_err | prior | T1-kin | 994 | 2.0577 | [1.7073, 2.4384] |  |
| contrast | LONGITUDINAL | 2-6s | speed_err | hwm_oracle - hold | T0 | 994 | -0.0236 | [-0.3563, 0.2892] | none |
| contrast | LONGITUDINAL | 2-6s | speed_err | hwm_oracle - flat_oracle | T0 | 994 | -0.7093 | [-1.0818, -0.3449] | better |
| contrast | LONGITUDINAL | 2-6s | speed_err | flat_oracle - hold | T0 | 994 | 0.6857 | [0.2010, 1.1782] | worse |
| arm | LONGITUDINAL | 2-6s | hw_err | hwm_oracle | T0 | 120 | 3.4201 | [2.4218, 4.7556] |  |
| arm | LONGITUDINAL | 2-6s | hw_err | flat_oracle | T0 | 87 | 4.7635 | [3.5907, 6.3156] |  |
| arm | LONGITUDINAL | 2-6s | hw_err | hold | T1-kin | 156 | 2.2384 | [1.3739, 3.3350] |  |
| arm | LONGITUDINAL | 2-6s | hw_err | prior | T1-kin | 138 | 2.2450 | [1.3181, 3.5752] |  |
| contrast | LONGITUDINAL | 2-6s | hw_err | hwm_oracle - hold | T0 | 110 | 1.5849 | [0.7575, 2.5694] | worse |
| contrast | LONGITUDINAL | 2-6s | hw_err | hwm_oracle - flat_oracle | T0 | 66 | -1.5443 | [-2.8716, -0.3421] | better |
| contrast | LONGITUDINAL | 2-6s | hw_err | flat_oracle - hold | T0 | 81 | 3.0605 | [1.4929, 4.4450] | worse |
| arm | LONGITUDINAL | 2-6s | tg_err | hwm_oracle | T0 | 88 | 3.1736 | [1.2181, 6.0457] |  |
| arm | LONGITUDINAL | 2-6s | tg_err | flat_oracle | T0 | 55 | 2.3499 | [1.1705, 4.4696] |  |
| arm | LONGITUDINAL | 2-6s | tg_err | hold | T1-kin | 130 | 0.9990 | [0.4512, 1.6895] |  |
| arm | LONGITUDINAL | 2-6s | tg_err | prior | T1-kin | 109 | 1.4163 | [0.6151, 2.4912] |  |
| contrast | LONGITUDINAL | 2-6s | tg_err | hwm_oracle - hold | T0 | 74 | 2.6680 | [0.5913, 6.0081] | worse |
| contrast | LONGITUDINAL | 2-6s | tg_err | hwm_oracle - flat_oracle | T0 | 31 | -1.0010 | [-2.4020, -0.0682] | better |
| contrast | LONGITUDINAL | 2-6s | tg_err | flat_oracle - hold | T0 | 49 | 1.5184 | [0.3741, 3.4300] | worse |
| arm | LONGITUDINAL | 2-6s | ttc_err | hwm_oracle | T0 | 120 | 5.6693 | [3.5321, 7.7632] |  |
| arm | LONGITUDINAL | 2-6s | ttc_err | flat_oracle | T0 | 87 | 7.5709 | [4.5159, 11.3134] |  |
| arm | LONGITUDINAL | 2-6s | ttc_err | hold | T1-kin | 156 | 4.5309 | [2.6365, 6.7093] |  |
| arm | LONGITUDINAL | 2-6s | ttc_err | prior | T1-kin | 138 | 4.0464 | [1.9436, 6.7985] |  |
| contrast | LONGITUDINAL | 2-6s | ttc_err | hwm_oracle - hold | T0 | 110 | 0.7920 | [-2.1814, 3.7637] | none |
| contrast | LONGITUDINAL | 2-6s | ttc_err | hwm_oracle - flat_oracle | T0 | 66 | -3.0985 | [-8.2446, 1.7097] | none |
| contrast | LONGITUDINAL | 2-6s | ttc_err | flat_oracle - hold | T0 | 81 | 4.0307 | [1.8698, 5.9358] | worse |
| arm | LONGITUDINAL | 2-6s | hw_d | hwm_oracle | T0 | 120 | 0.9923 | [-0.3665, 2.0251] |  |
| arm | LONGITUDINAL | 2-6s | hw_d | flat_oracle | T0 | 87 | 1.8902 | [-0.3233, 3.5434] |  |
| arm | LONGITUDINAL | 2-6s | hw_d | hold | T1-kin | 156 | -0.6653 | [-1.6122, 0.2102] |  |
| arm | LONGITUDINAL | 2-6s | hw_d | prior | T1-kin | 138 | 0.6935 | [-0.0669, 1.4335] |  |
| arm | LONGITUDINAL | 2-6s | tg_d | hwm_oracle | T0 | 88 | 2.2774 | [0.2858, 5.1151] |  |
| arm | LONGITUDINAL | 2-6s | tg_d | flat_oracle | T0 | 55 | 1.3808 | [0.1028, 3.4300] |  |
| arm | LONGITUDINAL | 2-6s | tg_d | hold | T1-kin | 130 | 0.0848 | [-0.4009, 0.5753] |  |
| arm | LONGITUDINAL | 2-6s | tg_d | prior | T1-kin | 109 | 0.7769 | [-0.1402, 1.8298] |  |
| arm | LONGITUDINAL | 2-6s | ttc_d | hwm_oracle | T0 | 120 | 0.0945 | [-3.2117, 3.5849] |  |
| arm | LONGITUDINAL | 2-6s | ttc_d | flat_oracle | T0 | 87 | 0.2809 | [-4.5077, 6.3381] |  |
| arm | LONGITUDINAL | 2-6s | ttc_d | hold | T1-kin | 156 | -0.1845 | [-1.9394, 1.4295] |  |
| arm | LONGITUDINAL | 2-6s | ttc_d | prior | T1-kin | 138 | 0.9250 | [-0.8270, 2.8234] |  |
| arm | LONGITUDINAL | full6s | acc_mae | hwm_oracle | T0 | 994 | 0.9771 | [0.9326, 1.0229] |  |
| arm | LONGITUDINAL | full6s | acc_mae | flat_oracle | T0 | 994 | 1.2384 | [1.1717, 1.3076] |  |
| arm | LONGITUDINAL | full6s | acc_mae | hold | T1-kin | 994 | 0.5230 | [0.4449, 0.6017] |  |
| arm | LONGITUDINAL | full6s | acc_mae | prior | T1-kin | 994 | 0.4722 | [0.4021, 0.5482] |  |
| contrast | LONGITUDINAL | full6s | acc_mae | hwm_oracle - hold | T0 | 994 | 0.4541 | [0.3914, 0.5138] | worse |
| contrast | LONGITUDINAL | full6s | acc_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.2613 | [-0.3244, -0.1987] | better |
| contrast | LONGITUDINAL | full6s | acc_mae | flat_oracle - hold | T0 | 994 | 0.7154 | [0.6244, 0.8101] | worse |
| arm | LONGITUDINAL | full6s | speed_err | hwm_oracle | T0 | 994 | 2.3747 | [2.1747, 2.6157] |  |
| arm | LONGITUDINAL | full6s | speed_err | flat_oracle | T0 | 994 | 3.0840 | [2.7168, 3.4616] |  |
| arm | LONGITUDINAL | full6s | speed_err | hold | T1-kin | 994 | 2.3983 | [2.0456, 2.7712] |  |
| arm | LONGITUDINAL | full6s | speed_err | prior | T1-kin | 994 | 2.0577 | [1.7073, 2.4384] |  |
| contrast | LONGITUDINAL | full6s | speed_err | hwm_oracle - hold | T0 | 994 | -0.0236 | [-0.3563, 0.2892] | none |
| contrast | LONGITUDINAL | full6s | speed_err | hwm_oracle - flat_oracle | T0 | 994 | -0.7093 | [-1.0818, -0.3449] | better |
| contrast | LONGITUDINAL | full6s | speed_err | flat_oracle - hold | T0 | 994 | 0.6857 | [0.2010, 1.1782] | worse |
| arm | LONGITUDINAL | full6s | hw_err | hwm_oracle | T0 | 214 | 3.5321 | [1.8277, 5.7733] |  |
| arm | LONGITUDINAL | full6s | hw_err | flat_oracle | T0 | 214 | 3.2096 | [1.6614, 5.2734] |  |
| arm | LONGITUDINAL | full6s | hw_err | hold | T1-kin | 214 | 1.9745 | [1.0280, 3.0946] |  |
| arm | LONGITUDINAL | full6s | hw_err | prior | T1-kin | 214 | 2.3291 | [1.2186, 3.7639] |  |
| contrast | LONGITUDINAL | full6s | hw_err | hwm_oracle - hold | T0 | 214 | 1.5576 | [0.4377, 3.0917] | worse |
| contrast | LONGITUDINAL | full6s | hw_err | hwm_oracle - flat_oracle | T0 | 214 | 0.3225 | [-0.7443, 1.7699] | none |
| contrast | LONGITUDINAL | full6s | hw_err | flat_oracle - hold | T0 | 214 | 1.2350 | [-0.1023, 2.7273] | none |
| arm | LONGITUDINAL | full6s | tg_err | hwm_oracle | T0 | 182 | 0.5995 | [0.3080, 0.9993] |  |
| arm | LONGITUDINAL | full6s | tg_err | flat_oracle | T0 | 182 | 0.4641 | [0.2754, 0.6912] |  |
| arm | LONGITUDINAL | full6s | tg_err | hold | T1-kin | 182 | 0.3965 | [0.1939, 0.7000] |  |
| arm | LONGITUDINAL | full6s | tg_err | prior | T1-kin | 182 | 0.4195 | [0.2360, 0.6652] |  |
| contrast | LONGITUDINAL | full6s | tg_err | hwm_oracle - hold | T0 | 182 | 0.2030 | [0.0250, 0.4382] | worse |
| contrast | LONGITUDINAL | full6s | tg_err | hwm_oracle - flat_oracle | T0 | 182 | 0.1354 | [-0.0587, 0.4163] | none |
| contrast | LONGITUDINAL | full6s | tg_err | flat_oracle - hold | T0 | 182 | 0.0676 | [-0.1591, 0.2512] | none |
| arm | LONGITUDINAL | full6s | ttc_err | hwm_oracle | T0 | 214 | 5.6173 | [4.0412, 7.4739] |  |
| arm | LONGITUDINAL | full6s | ttc_err | flat_oracle | T0 | 214 | 6.9386 | [5.1783, 8.7998] |  |
| arm | LONGITUDINAL | full6s | ttc_err | hold | T1-kin | 214 | 2.6508 | [1.4524, 3.9708] |  |
| arm | LONGITUDINAL | full6s | ttc_err | prior | T1-kin | 214 | 3.9406 | [2.3891, 5.7889] |  |
| contrast | LONGITUDINAL | full6s | ttc_err | hwm_oracle - hold | T0 | 214 | 2.9664 | [1.2459, 5.0502] | worse |
| contrast | LONGITUDINAL | full6s | ttc_err | hwm_oracle - flat_oracle | T0 | 214 | -1.3213 | [-2.5012, -0.1549] | better |
| contrast | LONGITUDINAL | full6s | ttc_err | flat_oracle - hold | T0 | 214 | 4.2878 | [2.6057, 6.1571] | worse |
| arm | LONGITUDINAL | full6s | hw_d | hwm_oracle | T0 | 214 | 0.6172 | [-1.2388, 2.6219] |  |
| arm | LONGITUDINAL | full6s | hw_d | flat_oracle | T0 | 214 | 1.5585 | [0.0067, 3.6966] |  |
| arm | LONGITUDINAL | full6s | hw_d | hold | T1-kin | 214 | -0.8009 | [-1.7679, 0.1023] |  |
| arm | LONGITUDINAL | full6s | hw_d | prior | T1-kin | 214 | 0.1250 | [-0.8603, 0.9903] |  |
| arm | LONGITUDINAL | full6s | tg_d | hwm_oracle | T0 | 182 | 0.0362 | [-0.2514, 0.3566] |  |
| arm | LONGITUDINAL | full6s | tg_d | flat_oracle | T0 | 182 | 0.1775 | [-0.0381, 0.4152] |  |
| arm | LONGITUDINAL | full6s | tg_d | hold | T1-kin | 182 | -0.1795 | [-0.4014, 0.0175] |  |
| arm | LONGITUDINAL | full6s | tg_d | prior | T1-kin | 182 | -0.0930 | [-0.3082, 0.0799] |  |
| arm | LONGITUDINAL | full6s | ttc_d | hwm_oracle | T0 | 214 | 0.0368 | [-2.5171, 2.6427] |  |
| arm | LONGITUDINAL | full6s | ttc_d | flat_oracle | T0 | 214 | 0.8488 | [-2.1454, 3.8695] |  |
| arm | LONGITUDINAL | full6s | ttc_d | hold | T1-kin | 214 | 0.1568 | [-0.9165, 1.3114] |  |
| arm | LONGITUDINAL | full6s | ttc_d | prior | T1-kin | 214 | 0.8048 | [-1.0604, 2.7069] |  |
| arm | LATERAL | 0-2s | yaw_mae | hwm_oracle | T0 | 994 | 0.1120 | [0.1013, 0.1231] |  |
| arm | LATERAL | 0-2s | yaw_mae | flat_oracle | T0 | 994 | 0.1276 | [0.1160, 0.1390] |  |
| arm | LATERAL | 0-2s | yaw_mae | hold | T1-kin | 994 | 0.0225 | [0.0174, 0.0280] |  |
| arm | LATERAL | 0-2s | yaw_mae | prior | T1-kin | 994 | 0.0416 | [0.0315, 0.0530] |  |
| contrast | LATERAL | 0-2s | yaw_mae | hwm_oracle - hold | T0 | 994 | 0.0895 | [0.0811, 0.0973] | worse |
| contrast | LATERAL | 0-2s | yaw_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.0156 | [-0.0274, -0.0046] | better |
| contrast | LATERAL | 0-2s | yaw_mae | flat_oracle - hold | T0 | 994 | 0.1051 | [0.0949, 0.1151] | worse |
| arm | LATERAL | 0-2s | head_err | hwm_oracle | T0 | 994 | 0.1461 | [0.1265, 0.1665] |  |
| arm | LATERAL | 0-2s | head_err | flat_oracle | T0 | 994 | 0.1993 | [0.1797, 0.2205] |  |
| arm | LATERAL | 0-2s | head_err | hold | T1-kin | 994 | 0.0423 | [0.0325, 0.0530] |  |
| arm | LATERAL | 0-2s | head_err | prior | T1-kin | 994 | 0.0792 | [0.0592, 0.1015] |  |
| contrast | LATERAL | 0-2s | head_err | hwm_oracle - hold | T0 | 994 | 0.1038 | [0.0904, 0.1180] | worse |
| contrast | LATERAL | 0-2s | head_err | hwm_oracle - flat_oracle | T0 | 994 | -0.0532 | [-0.0713, -0.0359] | better |
| contrast | LATERAL | 0-2s | head_err | flat_oracle - hold | T0 | 994 | 0.1570 | [0.1399, 0.1743] | worse |
| arm | LATERAL | 0-2s | curv_err | hwm_oracle | T0 | 950 | 0.0187 | [0.0143, 0.0239] |  |
| arm | LATERAL | 0-2s | curv_err | flat_oracle | T0 | 950 | 0.0223 | [0.0168, 0.0287] |  |
| arm | LATERAL | 0-2s | curv_err | hold | T1-kin | 950 | 0.0040 | [0.0028, 0.0054] |  |
| arm | LATERAL | 0-2s | curv_err | prior | T1-kin | 950 | 0.0076 | [0.0050, 0.0107] |  |
| contrast | LATERAL | 0-2s | curv_err | hwm_oracle - hold | T0 | 950 | 0.0147 | [0.0111, 0.0189] | worse |
| contrast | LATERAL | 0-2s | curv_err | hwm_oracle - flat_oracle | T0 | 950 | -0.0037 | [-0.0069, -0.0009] | better |
| contrast | LATERAL | 0-2s | curv_err | flat_oracle - hold | T0 | 950 | 0.0183 | [0.0136, 0.0241] | worse |
| arm | LATERAL | 0-2s | xtrack | hwm_oracle | T0 | 994 | 1.4847 | [1.2299, 1.7576] |  |
| arm | LATERAL | 0-2s | xtrack | flat_oracle | T0 | 994 | 2.1086 | [1.7978, 2.4622] |  |
| arm | LATERAL | 0-2s | xtrack | hold | T1-kin | 994 | 0.2255 | [0.1803, 0.2757] |  |
| arm | LATERAL | 0-2s | xtrack | prior | T1-kin | 994 | 0.6216 | [0.4594, 0.8038] |  |
| contrast | LATERAL | 0-2s | xtrack | hwm_oracle - hold | T0 | 994 | 1.2591 | [1.0265, 1.5050] | worse |
| contrast | LATERAL | 0-2s | xtrack | hwm_oracle - flat_oracle | T0 | 994 | -0.6240 | [-0.8401, -0.4200] | better |
| contrast | LATERAL | 0-2s | xtrack | flat_oracle - hold | T0 | 994 | 1.8831 | [1.5847, 2.2263] | worse |
| arm | LATERAL | 0-2s | ade | hwm_oracle | T0 | 994 | 0.9452 | [0.8601, 1.0345] |  |
| arm | LATERAL | 0-2s | ade | flat_oracle | T0 | 994 | 1.2597 | [1.1495, 1.3721] |  |
| arm | LATERAL | 0-2s | ade | hold | T1-kin | 994 | 0.1370 | [0.1213, 0.1542] |  |
| arm | LATERAL | 0-2s | ade | prior | T1-kin | 994 | 0.4497 | [0.3824, 0.5207] |  |
| contrast | LATERAL | 0-2s | ade | hwm_oracle - hold | T0 | 994 | 0.8083 | [0.7282, 0.8947] | worse |
| contrast | LATERAL | 0-2s | ade | hwm_oracle - flat_oracle | T0 | 994 | -0.3145 | [-0.3950, -0.2361] | better |
| contrast | LATERAL | 0-2s | ade | flat_oracle - hold | T0 | 994 | 1.1227 | [1.0161, 1.2374] | worse |
| arm | LATERAL | 2-6s | yaw_mae | hwm_oracle | T0 | 994 | 0.1027 | [0.0960, 0.1103] |  |
| arm | LATERAL | 2-6s | yaw_mae | flat_oracle | T0 | 994 | 0.1105 | [0.1009, 0.1206] |  |
| arm | LATERAL | 2-6s | yaw_mae | hold | T1-kin | 994 | 0.0523 | [0.0397, 0.0660] |  |
| arm | LATERAL | 2-6s | yaw_mae | prior | T1-kin | 994 | 0.0423 | [0.0312, 0.0544] |  |
| contrast | LATERAL | 2-6s | yaw_mae | hwm_oracle - hold | T0 | 994 | 0.0504 | [0.0421, 0.0580] | worse |
| contrast | LATERAL | 2-6s | yaw_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.0078 | [-0.0160, -0.0001] | better |
| contrast | LATERAL | 2-6s | yaw_mae | flat_oracle - hold | T0 | 994 | 0.0583 | [0.0472, 0.0692] | worse |
| arm | LATERAL | 2-6s | head_err | hwm_oracle | T0 | 994 | 0.2679 | [0.2260, 0.3139] |  |
| arm | LATERAL | 2-6s | head_err | flat_oracle | T0 | 994 | 0.3794 | [0.3364, 0.4260] |  |
| arm | LATERAL | 2-6s | head_err | hold | T1-kin | 994 | 0.2187 | [0.1655, 0.2784] |  |
| arm | LATERAL | 2-6s | head_err | prior | T1-kin | 994 | 0.2070 | [0.1509, 0.2669] |  |
| contrast | LATERAL | 2-6s | head_err | hwm_oracle - hold | T0 | 994 | 0.0492 | [0.0182, 0.0800] | worse |
| contrast | LATERAL | 2-6s | head_err | hwm_oracle - flat_oracle | T0 | 994 | -0.1116 | [-0.1542, -0.0713] | better |
| contrast | LATERAL | 2-6s | head_err | flat_oracle - hold | T0 | 994 | 0.1607 | [0.1132, 0.2093] | worse |
| arm | LATERAL | 2-6s | curv_err | hwm_oracle | T0 | 967 | 0.0192 | [0.0150, 0.0238] |  |
| arm | LATERAL | 2-6s | curv_err | flat_oracle | T0 | 967 | 0.0215 | [0.0161, 0.0275] |  |
| arm | LATERAL | 2-6s | curv_err | hold | T1-kin | 967 | 0.0102 | [0.0069, 0.0141] |  |
| arm | LATERAL | 2-6s | curv_err | prior | T1-kin | 967 | 0.0074 | [0.0049, 0.0104] |  |
| contrast | LATERAL | 2-6s | curv_err | hwm_oracle - hold | T0 | 967 | 0.0090 | [0.0058, 0.0124] | worse |
| contrast | LATERAL | 2-6s | curv_err | hwm_oracle - flat_oracle | T0 | 967 | -0.0023 | [-0.0052, 0.0003] | none |
| contrast | LATERAL | 2-6s | curv_err | flat_oracle - hold | T0 | 967 | 0.0113 | [0.0067, 0.0164] | worse |
| arm | LATERAL | 2-6s | xtrack | hwm_oracle | T0 | 994 | 8.8892 | [7.3626, 10.5060] |  |
| arm | LATERAL | 2-6s | xtrack | flat_oracle | T0 | 994 | 13.8603 | [11.9094, 16.0214] |  |
| arm | LATERAL | 2-6s | xtrack | hold | T1-kin | 994 | 4.1409 | [3.2343, 5.1397] |  |
| arm | LATERAL | 2-6s | xtrack | prior | T1-kin | 994 | 5.2076 | [3.9008, 6.6983] |  |
| contrast | LATERAL | 2-6s | xtrack | hwm_oracle - hold | T0 | 994 | 4.7484 | [3.5128, 6.0488] | worse |
| contrast | LATERAL | 2-6s | xtrack | hwm_oracle - flat_oracle | T0 | 994 | -4.9711 | [-6.5354, -3.4568] | better |
| contrast | LATERAL | 2-6s | xtrack | flat_oracle - hold | T0 | 994 | 9.7195 | [7.5861, 11.9449] | worse |
| arm | LATERAL | 2-6s | ade | hwm_oracle | T0 | 994 | 8.3363 | [7.5292, 9.1816] |  |
| arm | LATERAL | 2-6s | ade | flat_oracle | T0 | 994 | 11.9185 | [10.9447, 12.9498] |  |
| arm | LATERAL | 2-6s | ade | hold | T1-kin | 994 | 3.7452 | [3.2261, 4.3240] |  |
| arm | LATERAL | 2-6s | ade | prior | T1-kin | 994 | 5.3660 | [4.5195, 6.2639] |  |
| contrast | LATERAL | 2-6s | ade | hwm_oracle - hold | T0 | 994 | 4.5911 | [3.8753, 5.3600] | worse |
| contrast | LATERAL | 2-6s | ade | hwm_oracle - flat_oracle | T0 | 994 | -3.5822 | [-4.4049, -2.7632] | better |
| contrast | LATERAL | 2-6s | ade | flat_oracle - hold | T0 | 994 | 8.1733 | [7.0852, 9.3709] | worse |
| arm | LATERAL | full6s | yaw_mae | hwm_oracle | T0 | 994 | 0.1058 | [0.0988, 0.1134] |  |
| arm | LATERAL | full6s | yaw_mae | flat_oracle | T0 | 994 | 0.1162 | [0.1067, 0.1259] |  |
| arm | LATERAL | full6s | yaw_mae | hold | T1-kin | 994 | 0.0424 | [0.0325, 0.0532] |  |
| arm | LATERAL | full6s | yaw_mae | prior | T1-kin | 994 | 0.0420 | [0.0315, 0.0537] |  |
| contrast | LATERAL | full6s | yaw_mae | hwm_oracle - hold | T0 | 994 | 0.0634 | [0.0583, 0.0686] | worse |
| contrast | LATERAL | full6s | yaw_mae | hwm_oracle - flat_oracle | T0 | 994 | -0.0104 | [-0.0185, -0.0029] | better |
| contrast | LATERAL | full6s | yaw_mae | flat_oracle - hold | T0 | 994 | 0.0739 | [0.0644, 0.0833] | worse |
| arm | LATERAL | full6s | head_err | hwm_oracle | T0 | 994 | 0.2679 | [0.2260, 0.3139] |  |
| arm | LATERAL | full6s | head_err | flat_oracle | T0 | 994 | 0.3794 | [0.3364, 0.4260] |  |
| arm | LATERAL | full6s | head_err | hold | T1-kin | 994 | 0.2187 | [0.1655, 0.2784] |  |
| arm | LATERAL | full6s | head_err | prior | T1-kin | 994 | 0.2070 | [0.1509, 0.2669] |  |
| contrast | LATERAL | full6s | head_err | hwm_oracle - hold | T0 | 994 | 0.0492 | [0.0182, 0.0800] | worse |
| contrast | LATERAL | full6s | head_err | hwm_oracle - flat_oracle | T0 | 994 | -0.1116 | [-0.1542, -0.0713] | better |
| contrast | LATERAL | full6s | head_err | flat_oracle - hold | T0 | 994 | 0.1607 | [0.1132, 0.2093] | worse |
| arm | LATERAL | full6s | curv_err | hwm_oracle | T0 | 974 | 0.0195 | [0.0151, 0.0245] |  |
| arm | LATERAL | full6s | curv_err | flat_oracle | T0 | 974 | 0.0220 | [0.0167, 0.0281] |  |
| arm | LATERAL | full6s | curv_err | hold | T1-kin | 974 | 0.0081 | [0.0056, 0.0109] |  |
| arm | LATERAL | full6s | curv_err | prior | T1-kin | 974 | 0.0077 | [0.0050, 0.0110] |  |
| contrast | LATERAL | full6s | curv_err | hwm_oracle - hold | T0 | 974 | 0.0115 | [0.0083, 0.0151] | worse |
| contrast | LATERAL | full6s | curv_err | hwm_oracle - flat_oracle | T0 | 974 | -0.0025 | [-0.0052, -0.0003] | better |
| contrast | LATERAL | full6s | curv_err | flat_oracle - hold | T0 | 974 | 0.0140 | [0.0094, 0.0192] | worse |
| arm | LATERAL | full6s | xtrack | hwm_oracle | T0 | 994 | 8.8892 | [7.3626, 10.5060] |  |
| arm | LATERAL | full6s | xtrack | flat_oracle | T0 | 994 | 13.8603 | [11.9094, 16.0214] |  |
| arm | LATERAL | full6s | xtrack | hold | T1-kin | 994 | 4.1409 | [3.2343, 5.1397] |  |
| arm | LATERAL | full6s | xtrack | prior | T1-kin | 994 | 5.2076 | [3.9008, 6.6983] |  |
| contrast | LATERAL | full6s | xtrack | hwm_oracle - hold | T0 | 994 | 4.7484 | [3.5128, 6.0488] | worse |
| contrast | LATERAL | full6s | xtrack | hwm_oracle - flat_oracle | T0 | 994 | -4.9711 | [-6.5354, -3.4568] | better |
| contrast | LATERAL | full6s | xtrack | flat_oracle - hold | T0 | 994 | 9.7195 | [7.5861, 11.9449] | worse |
| arm | LATERAL | full6s | ade | hwm_oracle | T0 | 994 | 5.8726 | [5.3056, 6.4651] |  |
| arm | LATERAL | full6s | ade | flat_oracle | T0 | 994 | 8.3656 | [7.6864, 9.0877] |  |
| arm | LATERAL | full6s | ade | hold | T1-kin | 994 | 2.5424 | [2.1925, 2.9329] |  |
| arm | LATERAL | full6s | ade | prior | T1-kin | 994 | 3.7272 | [3.1424, 4.3483] |  |
| contrast | LATERAL | full6s | ade | hwm_oracle - hold | T0 | 994 | 3.3302 | [2.8352, 3.8646] | worse |
| contrast | LATERAL | full6s | ade | hwm_oracle - flat_oracle | T0 | 994 | -2.4929 | [-3.0665, -1.9214] | better |
| contrast | LATERAL | full6s | ade | flat_oracle - hold | T0 | 994 | 5.8231 | [5.0656, 6.6545] | worse |
| arm | TACTICAL | 0-2s | tac_man_dis | hwm_oracle | T0 | 994 | 0.6600 | [0.6087, 0.7123] |  |
| arm | TACTICAL | 0-2s | tac_man_dis | flat_oracle | T0 | 994 | 0.7726 | [0.7183, 0.8249] |  |
| arm | TACTICAL | 0-2s | tac_man_dis | hold | T1-kin | 994 | 0.1781 | [0.1439, 0.2143] |  |
| arm | TACTICAL | 0-2s | tac_man_dis | prior | T1-kin | 994 | 0.3531 | [0.2857, 0.4256] |  |
| contrast | TACTICAL | 0-2s | tac_man_dis | hwm_oracle - hold | T0 | 994 | 0.4819 | [0.4245, 0.5362] | worse |
| contrast | TACTICAL | 0-2s | tac_man_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.1127 | [-0.1690, -0.0563] | better |
| contrast | TACTICAL | 0-2s | tac_man_dis | flat_oracle - hold | T0 | 994 | 0.5946 | [0.5292, 0.6579] | worse |
| arm | TACTICAL | 0-2s | tac_lat_dis | hwm_oracle | T0 | 994 | 0.3109 | [0.2575, 0.3702] |  |
| arm | TACTICAL | 0-2s | tac_lat_dis | flat_oracle | T0 | 994 | 0.4909 | [0.4336, 0.5503] |  |
| arm | TACTICAL | 0-2s | tac_lat_dis | hold | T1-kin | 994 | 0.0694 | [0.0493, 0.0926] |  |
| arm | TACTICAL | 0-2s | tac_lat_dis | prior | T1-kin | 994 | 0.1419 | [0.0926, 0.1962] |  |
| contrast | TACTICAL | 0-2s | tac_lat_dis | hwm_oracle - hold | T0 | 994 | 0.2414 | [0.1931, 0.2938] | worse |
| contrast | TACTICAL | 0-2s | tac_lat_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.1801 | [-0.2445, -0.1177] | better |
| contrast | TACTICAL | 0-2s | tac_lat_dis | flat_oracle - hold | T0 | 994 | 0.4215 | [0.3612, 0.4829] | worse |
| arm | TACTICAL | 0-2s | tac_lon_dis | hwm_oracle | T0 | 994 | 0.5503 | [0.5000, 0.6026] |  |
| arm | TACTICAL | 0-2s | tac_lon_dis | flat_oracle | T0 | 994 | 0.6901 | [0.6388, 0.7425] |  |
| arm | TACTICAL | 0-2s | tac_lon_dis | hold | T1-kin | 994 | 0.1419 | [0.1117, 0.1740] |  |
| arm | TACTICAL | 0-2s | tac_lon_dis | prior | T1-kin | 994 | 0.2757 | [0.2143, 0.3421] |  |
| contrast | TACTICAL | 0-2s | tac_lon_dis | hwm_oracle - hold | T0 | 994 | 0.4085 | [0.3541, 0.4628] | worse |
| contrast | TACTICAL | 0-2s | tac_lon_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.1398 | [-0.1952, -0.0805] | better |
| contrast | TACTICAL | 0-2s | tac_lon_dis | flat_oracle - hold | T0 | 994 | 0.5483 | [0.4889, 0.6056] | worse |
| arm | TACTICAL | 2-6s | tac_man_dis | hwm_oracle | T0 | 994 | 0.6519 | [0.6096, 0.6952] |  |
| arm | TACTICAL | 2-6s | tac_man_dis | flat_oracle | T0 | 994 | 0.7515 | [0.6982, 0.8028] |  |
| arm | TACTICAL | 2-6s | tac_man_dis | hold | T1-kin | 994 | 0.5181 | [0.4537, 0.5845] |  |
| arm | TACTICAL | 2-6s | tac_man_dis | prior | T1-kin | 994 | 0.5734 | [0.4889, 0.6559] |  |
| contrast | TACTICAL | 2-6s | tac_man_dis | hwm_oracle - hold | T0 | 994 | 0.1338 | [0.0684, 0.1952] | worse |
| contrast | TACTICAL | 2-6s | tac_man_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.0996 | [-0.1630, -0.0332] | better |
| contrast | TACTICAL | 2-6s | tac_man_dis | flat_oracle - hold | T0 | 994 | 0.2334 | [0.1439, 0.3249] | worse |
| arm | TACTICAL | 2-6s | tac_lat_dis | hwm_oracle | T0 | 994 | 0.3541 | [0.3018, 0.4105] |  |
| arm | TACTICAL | 2-6s | tac_lat_dis | flat_oracle | T0 | 994 | 0.5634 | [0.5050, 0.6197] |  |
| arm | TACTICAL | 2-6s | tac_lat_dis | hold | T1-kin | 994 | 0.2364 | [0.1781, 0.2988] |  |
| arm | TACTICAL | 2-6s | tac_lat_dis | prior | T1-kin | 994 | 0.2495 | [0.1751, 0.3230] |  |
| contrast | TACTICAL | 2-6s | tac_lat_dis | hwm_oracle - hold | T0 | 994 | 0.1177 | [0.0714, 0.1640] | worse |
| contrast | TACTICAL | 2-6s | tac_lat_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.2093 | [-0.2837, -0.1308] | better |
| contrast | TACTICAL | 2-6s | tac_lat_dis | flat_oracle - hold | T0 | 994 | 0.3270 | [0.2425, 0.4074] | worse |
| arm | TACTICAL | 2-6s | tac_lon_dis | hwm_oracle | T0 | 994 | 0.5292 | [0.4889, 0.5704] |  |
| arm | TACTICAL | 2-6s | tac_lon_dis | flat_oracle | T0 | 994 | 0.5805 | [0.5211, 0.6389] |  |
| arm | TACTICAL | 2-6s | tac_lon_dis | hold | T1-kin | 994 | 0.4507 | [0.3893, 0.5121] |  |
| arm | TACTICAL | 2-6s | tac_lon_dis | prior | T1-kin | 994 | 0.4588 | [0.3803, 0.5383] |  |
| contrast | TACTICAL | 2-6s | tac_lon_dis | hwm_oracle - hold | T0 | 994 | 0.0785 | [0.0231, 0.1358] | worse |
| contrast | TACTICAL | 2-6s | tac_lon_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.0513 | [-0.1157, 0.0131] | none |
| contrast | TACTICAL | 2-6s | tac_lon_dis | flat_oracle - hold | T0 | 994 | 0.1298 | [0.0382, 0.2254] | worse |
| arm | TACTICAL | full6s | tac_man_dis | hwm_oracle | T0 | 994 | 0.7042 | [0.6579, 0.7485] |  |
| arm | TACTICAL | full6s | tac_man_dis | flat_oracle | T0 | 994 | 0.7394 | [0.6761, 0.8028] |  |
| arm | TACTICAL | full6s | tac_man_dis | hold | T1-kin | 994 | 0.4769 | [0.4225, 0.5312] |  |
| arm | TACTICAL | full6s | tac_man_dis | prior | T1-kin | 994 | 0.6539 | [0.5704, 0.7324] |  |
| contrast | TACTICAL | full6s | tac_man_dis | hwm_oracle - hold | T0 | 994 | 0.2274 | [0.1579, 0.2968] | worse |
| contrast | TACTICAL | full6s | tac_man_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.0352 | [-0.0946, 0.0272] | none |
| contrast | TACTICAL | full6s | tac_man_dis | flat_oracle - hold | T0 | 994 | 0.2626 | [0.1891, 0.3400] | worse |
| arm | TACTICAL | full6s | tac_lat_dis | hwm_oracle | T0 | 994 | 0.4849 | [0.4366, 0.5362] |  |
| arm | TACTICAL | full6s | tac_lat_dis | flat_oracle | T0 | 994 | 0.6006 | [0.5433, 0.6590] |  |
| arm | TACTICAL | full6s | tac_lat_dis | hold | T1-kin | 994 | 0.2364 | [0.1891, 0.2857] |  |
| arm | TACTICAL | full6s | tac_lat_dis | prior | T1-kin | 994 | 0.2968 | [0.2173, 0.3803] |  |
| contrast | TACTICAL | full6s | tac_lat_dis | hwm_oracle - hold | T0 | 994 | 0.2485 | [0.1911, 0.3028] | worse |
| contrast | TACTICAL | full6s | tac_lat_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.1157 | [-0.1912, -0.0452] | better |
| contrast | TACTICAL | full6s | tac_lat_dis | flat_oracle - hold | T0 | 994 | 0.3642 | [0.2827, 0.4467] | worse |
| arm | TACTICAL | full6s | tac_lon_dis | hwm_oracle | T0 | 994 | 0.5483 | [0.4970, 0.5986] |  |
| arm | TACTICAL | full6s | tac_lon_dis | flat_oracle | T0 | 994 | 0.5624 | [0.4950, 0.6247] |  |
| arm | TACTICAL | full6s | tac_lon_dis | hold | T1-kin | 994 | 0.4215 | [0.3712, 0.4718] |  |
| arm | TACTICAL | full6s | tac_lon_dis | prior | T1-kin | 994 | 0.5463 | [0.4658, 0.6258] |  |
| contrast | TACTICAL | full6s | tac_lon_dis | hwm_oracle - hold | T0 | 994 | 0.1268 | [0.0594, 0.1962] | worse |
| contrast | TACTICAL | full6s | tac_lon_dis | hwm_oracle - flat_oracle | T0 | 994 | -0.0141 | [-0.0664, 0.0433] | none |
| contrast | TACTICAL | full6s | tac_lon_dis | flat_oracle - hold | T0 | 994 | 0.1408 | [0.0664, 0.2173] | worse |
| inference_seed_replicate | LONGITUDINAL | 0-2s | acc_mae | hwm_oracle | T0 | 994 | -0.0136 | [-0.0313, 0.0050] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | speed_err | hwm_oracle | T0 | 994 | -0.0005 | [-0.0462, 0.0428] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | hw_err | hwm_oracle | T0 | 214 | 0.0433 | [-0.1448, 0.2175] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | tg_err | hwm_oracle | T0 | 182 | 0.0125 | [-0.0237, 0.0509] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | ttc_err | hwm_oracle | T0 | 214 | -0.0518 | [-0.3295, 0.3163] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | acc_mae | hwm_oracle | T0 | 994 | -0.0041 | [-0.0204, 0.0104] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | speed_err | hwm_oracle | T0 | 994 | -0.0226 | [-0.1108, 0.0586] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | hw_err | hwm_oracle | T0 | 98 | 0.4544 | [-0.3396, 1.9457] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | tg_err | hwm_oracle | T0 | 65 | -0.8039 | [-2.5723, 0.3166] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | ttc_err | hwm_oracle | T0 | 98 | 0.5103 | [-0.7258, 1.7344] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | acc_mae | hwm_oracle | T0 | 994 | -0.0073 | [-0.0208, 0.0053] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | speed_err | hwm_oracle | T0 | 994 | -0.0226 | [-0.1108, 0.0586] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | hw_err | hwm_oracle | T0 | 214 | 0.5306 | [0.0013, 1.0960] | worse |
| inference_seed_replicate | LONGITUDINAL | full6s | tg_err | hwm_oracle | T0 | 182 | 0.1089 | [-0.0203, 0.2769] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | ttc_err | hwm_oracle | T0 | 214 | 0.3587 | [-0.1952, 0.9557] | none |
| inference_seed_replicate | LATERAL | 0-2s | yaw_mae | hwm_oracle | T0 | 994 | -0.0002 | [-0.0025, 0.0021] | none |
| inference_seed_replicate | LATERAL | 0-2s | head_err | hwm_oracle | T0 | 994 | -0.0008 | [-0.0056, 0.0041] | none |
| inference_seed_replicate | LATERAL | 0-2s | curv_err | hwm_oracle | T0 | 950 | -0.0005 | [-0.0014, 0.0004] | none |
| inference_seed_replicate | LATERAL | 0-2s | xtrack | hwm_oracle | T0 | 994 | -0.0114 | [-0.0729, 0.0473] | none |
| inference_seed_replicate | LATERAL | 2-6s | yaw_mae | hwm_oracle | T0 | 994 | 0.0014 | [-0.0000, 0.0027] | none |
| inference_seed_replicate | LATERAL | 2-6s | head_err | hwm_oracle | T0 | 994 | -0.0032 | [-0.0119, 0.0063] | none |
| inference_seed_replicate | LATERAL | 2-6s | curv_err | hwm_oracle | T0 | 967 | -0.0000 | [-0.0009, 0.0009] | none |
| inference_seed_replicate | LATERAL | 2-6s | xtrack | hwm_oracle | T0 | 994 | -0.1050 | [-0.4657, 0.2319] | none |
| inference_seed_replicate | LATERAL | full6s | yaw_mae | hwm_oracle | T0 | 994 | 0.0008 | [-0.0005, 0.0021] | none |
| inference_seed_replicate | LATERAL | full6s | head_err | hwm_oracle | T0 | 994 | -0.0032 | [-0.0119, 0.0063] | none |
| inference_seed_replicate | LATERAL | full6s | curv_err | hwm_oracle | T0 | 974 | -0.0003 | [-0.0008, 0.0002] | none |
| inference_seed_replicate | LATERAL | full6s | xtrack | hwm_oracle | T0 | 994 | -0.1050 | [-0.4657, 0.2319] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_man_dis | hwm_oracle | T0 | 994 | 0.0101 | [-0.0121, 0.0322] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_lat_dis | hwm_oracle | T0 | 994 | 0.0101 | [-0.0111, 0.0302] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_lon_dis | hwm_oracle | T0 | 994 | -0.0040 | [-0.0322, 0.0221] | none |
| inference_seed_replicate | TACTICAL | 2-6s | tac_man_dis | hwm_oracle | T0 | 994 | 0.0060 | [-0.0302, 0.0412] | none |
| inference_seed_replicate | TACTICAL | 2-6s | tac_lat_dis | hwm_oracle | T0 | 994 | 0.0211 | [-0.0101, 0.0533] | none |
| inference_seed_replicate | TACTICAL | 2-6s | tac_lon_dis | hwm_oracle | T0 | 994 | -0.0171 | [-0.0533, 0.0171] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_man_dis | hwm_oracle | T0 | 994 | 0.0000 | [-0.0312, 0.0312] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_lat_dis | hwm_oracle | T0 | 994 | -0.0151 | [-0.0463, 0.0161] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_lon_dis | hwm_oracle | T0 | 994 | 0.0131 | [-0.0221, 0.0453] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | acc_mae | flat_oracle | T0 | 994 | -0.0010 | [-0.0101, 0.0088] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | speed_err | flat_oracle | T0 | 994 | -0.0154 | [-0.0414, 0.0122] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | hw_err | flat_oracle | T0 | 214 | 0.0257 | [-0.0664, 0.1365] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | tg_err | flat_oracle | T0 | 182 | -0.0029 | [-0.0331, 0.0208] | none |
| inference_seed_replicate | LONGITUDINAL | 0-2s | ttc_err | flat_oracle | T0 | 214 | 0.3479 | [-0.0427, 0.9226] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | acc_mae | flat_oracle | T0 | 994 | 0.0018 | [-0.0045, 0.0081] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | speed_err | flat_oracle | T0 | 994 | 0.0045 | [-0.0190, 0.0280] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | hw_err | flat_oracle | T0 | 80 | 0.0290 | [-0.3174, 0.4933] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | tg_err | flat_oracle | T0 | 50 | -0.5582 | [-2.0537, 0.0561] | none |
| inference_seed_replicate | LONGITUDINAL | 2-6s | ttc_err | flat_oracle | T0 | 80 | -0.5639 | [-1.6941, 0.5158] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | acc_mae | flat_oracle | T0 | 994 | 0.0009 | [-0.0047, 0.0062] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | speed_err | flat_oracle | T0 | 994 | 0.0045 | [-0.0190, 0.0280] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | hw_err | flat_oracle | T0 | 214 | 0.1730 | [0.0340, 0.3258] | worse |
| inference_seed_replicate | LONGITUDINAL | full6s | tg_err | flat_oracle | T0 | 182 | 0.0200 | [-0.0019, 0.0418] | none |
| inference_seed_replicate | LONGITUDINAL | full6s | ttc_err | flat_oracle | T0 | 214 | -0.3765 | [-0.8637, -0.0163] | better |
| inference_seed_replicate | LATERAL | 0-2s | yaw_mae | flat_oracle | T0 | 994 | 0.0000 | [-0.0012, 0.0011] | none |
| inference_seed_replicate | LATERAL | 0-2s | head_err | flat_oracle | T0 | 994 | -0.0026 | [-0.0056, 0.0001] | none |
| inference_seed_replicate | LATERAL | 0-2s | curv_err | flat_oracle | T0 | 950 | 0.0001 | [-0.0002, 0.0004] | none |
| inference_seed_replicate | LATERAL | 0-2s | xtrack | flat_oracle | T0 | 994 | -0.0297 | [-0.0605, 0.0015] | none |
| inference_seed_replicate | LATERAL | 2-6s | yaw_mae | flat_oracle | T0 | 994 | 0.0003 | [-0.0003, 0.0009] | none |
| inference_seed_replicate | LATERAL | 2-6s | head_err | flat_oracle | T0 | 994 | 0.0004 | [-0.0049, 0.0054] | none |
| inference_seed_replicate | LATERAL | 2-6s | curv_err | flat_oracle | T0 | 967 | -0.0002 | [-0.0005, 0.0001] | none |
| inference_seed_replicate | LATERAL | 2-6s | xtrack | flat_oracle | T0 | 994 | -0.0875 | [-0.2536, 0.0849] | none |
| inference_seed_replicate | LATERAL | full6s | yaw_mae | flat_oracle | T0 | 994 | 0.0002 | [-0.0005, 0.0008] | none |
| inference_seed_replicate | LATERAL | full6s | head_err | flat_oracle | T0 | 994 | 0.0004 | [-0.0049, 0.0054] | none |
| inference_seed_replicate | LATERAL | full6s | curv_err | flat_oracle | T0 | 974 | -0.0001 | [-0.0004, 0.0001] | none |
| inference_seed_replicate | LATERAL | full6s | xtrack | flat_oracle | T0 | 994 | -0.0875 | [-0.2536, 0.0849] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_man_dis | flat_oracle | T0 | 994 | -0.0060 | [-0.0191, 0.0070] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_lat_dis | flat_oracle | T0 | 994 | -0.0121 | [-0.0282, 0.0020] | none |
| inference_seed_replicate | TACTICAL | 0-2s | tac_lon_dis | flat_oracle | T0 | 994 | 0.0050 | [-0.0080, 0.0181] | none |
| inference_seed_replicate | TACTICAL | 2-6s | tac_man_dis | flat_oracle | T0 | 994 | 0.0080 | [-0.0070, 0.0231] | none |
| inference_seed_replicate | TACTICAL | 2-6s | tac_lat_dis | flat_oracle | T0 | 994 | 0.0050 | [-0.0141, 0.0221] | none |
| inference_seed_replicate | TACTICAL | 2-6s | tac_lon_dis | flat_oracle | T0 | 994 | 0.0010 | [-0.0101, 0.0121] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_man_dis | flat_oracle | T0 | 994 | 0.0040 | [-0.0050, 0.0141] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_lat_dis | flat_oracle | T0 | 994 | -0.0050 | [-0.0171, 0.0070] | none |
| inference_seed_replicate | TACTICAL | full6s | tac_lon_dis | flat_oracle | T0 | 994 | -0.0030 | [-0.0121, 0.0060] | none |
