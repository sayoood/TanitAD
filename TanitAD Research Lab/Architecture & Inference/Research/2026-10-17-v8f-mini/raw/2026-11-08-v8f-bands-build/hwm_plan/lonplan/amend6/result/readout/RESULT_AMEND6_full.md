# RESULT_AMEND5 (generated from RESULT_AMEND5.json; the MM writes the reading)

Tier T0 (oracle goal): a world-model diagnostic, never driving performance. Estimator: paired episode-cluster bootstrap, B 2,000 seed 0 (another draw of EPISODES only); F_I answers another INFERENCE run.

Base solver: A0 (B3). Committed arm: RAL1. CEIL-ORACLE: False (per chassis {'s0': False, 's1': False}).

## Labels (section 6)

| arm | s0 | s1 | label | effective | qualifier | READ-LIMITED | HOLD-collapsed | abs dH < 2 F_I |
|---|---|---|---|---|---|---|---|---|
| RAL1 | NARROWS | NARROWS | RAL1-NARROWS | RAL1-NARROWS | MEAN-ONLY | True | {'s0': False, 's1': False} | {'s0': False, 's1': False} |

Multiplicity guard: {'two_arms': False, 'ceiling_arms': [], 'guard_applies': False, 'needs_null_replicates': False}

Next step (RAL1): READ-LIMITED: (1) R35's response along RA's direction (ACTRSP Option B / action-sensitivity training; blocked on box 2 and the PI, Q2b); (2) RA's precision (more TRAIN clips' level-1 latents from box 2; blocked, PI P3)

* RAL1 s0: dH +0.4547 [+0.3976, +0.5159]; dB -0.5105 [-0.5701, -0.4491]; dM -0.0071; guard xtrack -0.0228; F_I 0.0073; D 0.885
* RAL1 s1: dH +0.4886 [+0.4285, +0.5482]; dB -0.3799 [-0.4390, -0.3220]; dM +0.0034; guard xtrack +0.0024; F_I 0.0104; D 0.938

## Chassis R35_s0: 994 windows, 71 episodes (windows sha256 371c753bcc3f)

### RAL1_BASE (c_RAL1_BASE_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.5105 | [-0.5701, -0.4491] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0704 | [-0.1351, -0.0098] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.0550 | [-0.1003, -0.0081] | better | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0032 | [-0.0061, +0.0168] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.3857 | [-1.2209, +0.2168] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0713 | [-0.1645, +0.0139] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.1729 | [-0.4153, +0.0444] | none | 102 | 23 |
| LONGITUDINAL | 2-6s | tg_err | +0.0530 | [-0.2715, +0.5435] | none | 77 | 21 |
| LONGITUDINAL | 2-6s | ttc_err | -0.6426 | [-1.2481, -0.1948] | better | 102 | 23 |
| LONGITUDINAL | full6s | acc_mae | -0.1702 | [-0.1900, -0.1497] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0713 | [-0.1645, +0.0139] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.0332 | [-0.1391, +0.0947] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0038 | [-0.0166, +0.0107] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.1933 | [-0.6262, +0.2105] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0003 | [-0.0007, +0.0000] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.0228 | [-0.0380, -0.0075] | better | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0674 | [-0.0885, -0.0480] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0001 | [-0.0001, +0.0002] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.1320 | [-0.2304, -0.0386] | better | 994 | 71 |
| LATERAL | 2-6s | ade | -0.2310 | [-0.3712, -0.1062] | better | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0000 | [-0.0002, +0.0001] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.1320 | [-0.2304, -0.0386] | better | 994 | 71 |
| LATERAL | full6s | ade | -0.1764 | [-0.2739, -0.0881] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0312 | [-0.0624, -0.0010] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0000 | [-0.0030, +0.0030] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0453 | [-0.0795, -0.0121] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0010 | [-0.0030, +0.0060] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0020 | [+0.0000, +0.0050] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0010 | [-0.0050, +0.0020] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0181 | [-0.0433, +0.0040] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0030 | [-0.0020, +0.0091] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | -0.0322 | [-0.0604, -0.0070] | better | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | -0.1279 | [-0.4955, +0.1935] | none | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 856 | 69 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.4547 | [+0.3976, +0.5159] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.8438 | [+0.7270, +0.9704] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.0484 | [+0.5264, +1.7809] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1510 | [+0.0804, +0.2482] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +2.9049 | [+1.8805, +4.1216] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.2580 | [+0.1939, +0.3218] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | -0.2177 | [-0.5274, +0.0762] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +1.9839 | [+0.9003, +3.4671] | worse | 94 |
| LONGITUDINAL | 2-6s | tg_err | +0.4876 | [-0.5877, +1.8218] | none | 65 |
| LONGITUDINAL | 2-6s | ttc_err | +3.5877 | [+0.8302, +6.2958] | worse | 94 |
| LONGITUDINAL | full6s | acc_mae | +0.3235 | [+0.2769, +0.3710] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | -0.2177 | [-0.5274, +0.0762] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.7278 | [+0.5682, +3.2289] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1828 | [+0.0050, +0.3624] | worse | 182 |
| LONGITUDINAL | full6s | ttc_err | +3.1248 | [+1.5309, +4.9267] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0937 | [+0.0861, +0.1016] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.1082 | [+0.0969, +0.1195] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0154 | [+0.0115, +0.0200] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.2923 | [+1.0682, +1.5339] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7314 | [+0.6439, +0.8223] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0465 | [+0.0389, +0.0537] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | 2-6s | curv_err | +0.0074 | [+0.0049, +0.0101] | worse | 967 |
| LATERAL | 2-6s | xtrack | +5.0636 | [+3.6933, +6.5348] | worse | 994 |
| LATERAL | 2-6s | ade | +4.2220 | [+3.4094, +5.1241] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0622 | [+0.0569, +0.0673] | worse | 994 |
| LATERAL | full6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | full6s | curv_err | +0.0107 | [+0.0075, +0.0144] | worse | 974 |
| LATERAL | full6s | xtrack | +5.0636 | [+3.6933, +6.5348] | worse | 994 |
| LATERAL | full6s | ade | +3.0585 | [+2.4961, +3.6851] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4235 | [+0.3642, +0.4819] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2706 | [+0.2233, +0.3209] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3330 | [+0.2807, +0.3873] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.1087 | [+0.0453, +0.1751] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0845 | [+0.0443, +0.1247] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0815 | [+0.0231, +0.1439] | worse | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1620 | [+0.0956, +0.2294] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2183 | [+0.1539, +0.2777] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.0493 | [-0.0131, +0.1127] | none | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.6250 | [-1.0000, -0.2500] | worse | 8 |
| STRATEGIC | full6s | s_adh_deg | -52.8092 | [-90.0344, -15.5841] | worse | 8 |
| STRATEGIC | full6s | s_false | -0.0844 | [-0.1332, -0.0393] | lower | 853 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### RAL1_MEANL1 (c_RAL1_MEANL1_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.0071 | [-0.0352, +0.0205] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0696 | [-0.1353, -0.0087] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.0366 | [-0.1137, +0.0083] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0029 | [-0.0081, +0.0011] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.0444 | [-0.1754, +0.3716] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0727 | [-0.1661, +0.0132] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.0288 | [-0.1573, +0.1713] | none | 103 | 23 |
| LONGITUDINAL | 2-6s | tg_err | -0.3210 | [-0.5928, -0.0830] | better | 76 | 21 |
| LONGITUDINAL | 2-6s | ttc_err | -0.3588 | [-0.9888, +0.0195] | none | 103 | 23 |
| LONGITUDINAL | full6s | acc_mae | -0.0024 | [-0.0117, +0.0068] | none | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0727 | [-0.1661, +0.0132] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.0529 | [-0.1506, +0.0254] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0026 | [-0.0112, +0.0057] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.0336 | [-0.2826, +0.2509] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0001 | [-0.0003, +0.0000] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.0158 | [-0.0284, -0.0051] | better | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0093 | [-0.0218, +0.0035] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0000 | [-0.0002, +0.0001] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.1193 | [-0.2141, -0.0301] | better | 994 | 71 |
| LATERAL | 2-6s | ade | -0.1888 | [-0.3265, -0.0636] | better | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0001 | [-0.0002, +0.0001] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.1193 | [-0.2141, -0.0301] | better | 994 | 71 |
| LATERAL | full6s | ade | -0.1289 | [-0.2234, -0.0421] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0352 | [-0.0634, -0.0070] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0020 | [+0.0000, +0.0050] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0553 | [-0.0855, -0.0252] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0030 | [+0.0000, +0.0080] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0010 | [+0.0000, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0010 | [-0.0020, +0.0040] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0181 | [-0.0423, +0.0030] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0040 | [+0.0000, +0.0091] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | -0.0332 | [-0.0604, -0.0070] | better | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | -0.1038 | [-0.8820, +0.2427] | none | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 856 | 69 |

### RAL1_A1 (c_RAL1_A1_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.4481 | [+0.3905, +0.5082] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.8622 | [+0.7467, +0.9871] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.9150 | [+0.4439, +1.5875] | worse | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.1409 | [+0.0737, +0.2354] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +2.4138 | [+1.3750, +3.5860] | worse | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0407 | [-0.1125, +0.1879] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +1.8771 | [+0.7743, +3.3985] | worse | 93 | 22 |
| LONGITUDINAL | 2-6s | tg_err | -0.1373 | [-2.8156, +1.5831] | none | 66 | 20 |
| LONGITUDINAL | 2-6s | ttc_err | +2.4843 | [-0.0748, +5.9361] | none | 93 | 22 |
| LONGITUDINAL | full6s | acc_mae | +0.1494 | [+0.1302, +0.1694] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0407 | [-0.1125, +0.1879] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +1.5183 | [+0.4602, +2.9612] | worse | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0927 | [-0.1090, +0.2783] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +2.6176 | [+1.2526, +4.1155] | worse | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0911 | [+0.0834, +0.0988] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.1046 | [+0.0935, +0.1157] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0149 | [+0.0110, +0.0194] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.2476 | [+1.0293, +1.4894] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.6991 | [+0.6103, +0.7900] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0490 | [+0.0374, +0.0613] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0006 | [-0.0001, +0.0013] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +4.4729 | [+3.3583, +5.7331] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +4.1292 | [+3.4164, +4.9015] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0304 | [+0.0278, +0.0329] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0490 | [+0.0374, +0.0613] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0055 | [+0.0036, +0.0079] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | +4.4729 | [+3.3583, +5.7331] | worse | 994 | 71 |
| LATERAL | full6s | ade | +2.9858 | [+2.4848, +3.5299] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.4296 | [+0.3732, +0.4869] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2757 | [+0.2294, +0.3249] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3360 | [+0.2857, +0.3893] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0030 | [-0.0050, +0.0131] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0030 | [-0.0060, +0.0141] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0040 | [-0.0020, +0.0121] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0845 | [+0.0352, +0.1328] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.1278 | [+0.0774, +0.1761] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0101 | [-0.0332, +0.0543] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.2500 | [-0.5000, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | -11.6355 | [-16.8984, -2.6145] | worse | 12 | 4 |
| STRATEGIC | full6s | s_false | -0.0258 | [-0.0631, +0.0081] | none | 853 | 69 |

### SHUF_RAL1 (c_SHUF_RAL1_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.1144 | [+0.0673, +0.1585] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.0045 | [-0.0828, +0.0912] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0538 | [-0.0084, +0.1173] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0107 | [-0.0586, +0.0144] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.2464 | [-0.7794, +0.1810] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.1353 | [+0.0295, +0.2410] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.1411 | [-0.7777, +0.4239] | none | 100 | 23 |
| LONGITUDINAL | 2-6s | tg_err | +0.0501 | [-0.3936, +0.4387] | none | 77 | 21 |
| LONGITUDINAL | 2-6s | ttc_err | +0.6360 | [-0.0268, +1.6647] | none | 100 | 23 |
| LONGITUDINAL | full6s | acc_mae | +0.0381 | [+0.0224, +0.0528] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.1353 | [+0.0295, +0.2410] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0514 | [-0.0518, +0.1779] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0033 | [-0.0364, +0.0240] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.0752 | [-0.5922, +0.4059] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0002 | [-0.0007, +0.0001] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0153 | [-0.0094, +0.0393] | none | 994 | 71 |
| LATERAL | 0-2s | ade | +0.0205 | [-0.0041, +0.0438] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0002 | [-0.0007, +0.0003] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.0081 | [-0.1859, +0.1648] | none | 994 | 71 |
| LATERAL | 2-6s | ade | +0.1147 | [-0.0790, +0.3000] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0002 | [-0.0006, +0.0002] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.0081 | [-0.1859, +0.1648] | none | 994 | 71 |
| LATERAL | full6s | ade | +0.0833 | [-0.0528, +0.2117] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0131 | [-0.0262, +0.0533] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0010 | [-0.0020, +0.0050] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0161 | [-0.0312, +0.0644] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0020 | [-0.0050, +0.0000] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0020 | [-0.0050, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0211 | [+0.0000, +0.0423] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0342 | [+0.0020, +0.0644] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 11 | 4 |
| STRATEGIC | full6s | s_adh_deg | +0.7102 | [-0.0437, +2.2821] | none | 11 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 855 | 69 |

### SHUF_BASE (c_SHUF_BASE_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.3961 | [-0.4523, -0.3411] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0659 | [-0.1393, +0.0047] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.0013 | [-0.0584, +0.0603] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0075 | [-0.0489, +0.0179] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.6321 | [-1.6662, +0.0741] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0640 | [-0.0142, +0.1456] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.3079 | [-0.9176, +0.2498] | none | 101 | 23 |
| LONGITUDINAL | 2-6s | tg_err | +0.1020 | [-0.3610, +0.5964] | none | 78 | 21 |
| LONGITUDINAL | 2-6s | ttc_err | -0.0192 | [-0.7294, +0.8844] | none | 101 | 23 |
| LONGITUDINAL | full6s | acc_mae | -0.1320 | [-0.1508, -0.1137] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0640 | [-0.0142, +0.1456] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0182 | [-0.1105, +0.1753] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0072 | [-0.0481, +0.0256] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.2685 | [-0.8316, +0.2401] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0005 | [-0.0011, -0.0001] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.0075 | [-0.0294, +0.0140] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0469 | [-0.0730, -0.0224] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0001 | [-0.0007, +0.0004] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.1402 | [-0.2858, +0.0098] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.1163 | [-0.2792, +0.0418] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0002 | [-0.0008, +0.0002] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.1402 | [-0.2858, +0.0098] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.0931 | [-0.2079, +0.0182] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0181 | [-0.0523, +0.0151] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0010 | [-0.0020, +0.0040] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0292 | [-0.0714, +0.0111] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0010 | [-0.0070, +0.0050] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0020 | [+0.0000, +0.0050] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0030 | [-0.0091, +0.0020] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0030 | [-0.0111, +0.0181] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0030 | [-0.0020, +0.0091] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0020 | [-0.0241, +0.0282] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 11 | 4 |
| STRATEGIC | full6s | s_adh_deg | +0.5868 | [+0.0000, +2.4320] | none | 11 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 855 | 69 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.5691 | [+0.5120, +0.6247] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.8483 | [+0.7306, +0.9608] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.1022 | [+0.5478, +1.8577] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1404 | [+0.0765, +0.2271] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +2.6585 | [+1.5530, +3.9152] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.2580 | [+0.1939, +0.3218] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | -0.0824 | [-0.3881, +0.2178] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +2.0430 | [+0.9524, +3.6130] | worse | 93 |
| LONGITUDINAL | 2-6s | tg_err | +0.6459 | [-0.2949, +1.8370] | none | 67 |
| LONGITUDINAL | 2-6s | ttc_err | +3.8699 | [+1.4228, +6.4391] | worse | 93 |
| LONGITUDINAL | full6s | acc_mae | +0.3617 | [+0.3127, +0.4067] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | -0.0824 | [-0.3881, +0.2178] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.7792 | [+0.5713, +3.3234] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1795 | [-0.0056, +0.3656] | none | 182 |
| LONGITUDINAL | full6s | ttc_err | +3.0495 | [+1.5669, +4.7603] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0937 | [+0.0861, +0.1016] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.1082 | [+0.0969, +0.1195] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0151 | [+0.0114, +0.0197] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.3076 | [+1.0871, +1.5529] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7519 | [+0.6625, +0.8430] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0465 | [+0.0389, +0.0537] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | 2-6s | curv_err | +0.0072 | [+0.0046, +0.0100] | worse | 967 |
| LATERAL | 2-6s | xtrack | +5.0554 | [+3.6390, +6.5575] | worse | 994 |
| LATERAL | 2-6s | ade | +4.3367 | [+3.5153, +5.2069] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0622 | [+0.0569, +0.0673] | worse | 994 |
| LATERAL | full6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | full6s | curv_err | +0.0105 | [+0.0075, +0.0142] | worse | 974 |
| LATERAL | full6s | xtrack | +5.0554 | [+3.6390, +6.5575] | worse | 994 |
| LATERAL | full6s | ade | +3.1418 | [+2.5654, +3.7473] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4366 | [+0.3783, +0.4909] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2716 | [+0.2264, +0.3209] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3491 | [+0.2948, +0.4034] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.1066 | [+0.0433, +0.1730] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0845 | [+0.0443, +0.1247] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0795 | [+0.0201, +0.1419] | worse | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1831 | [+0.1137, +0.2556] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2183 | [+0.1539, +0.2777] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.0835 | [+0.0221, +0.1469] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.6250 | [-1.0000, -0.2857] | worse | 8 |
| STRATEGIC | full6s | s_adh_deg | -58.2465 | [-104.4438, -25.2427] | worse | 8 |
| STRATEGIC | full6s | s_false | -0.0843 | [-0.1356, -0.0386] | lower | 854 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### NEG_RAL1 (c_NEG_RAL1_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.0238 | [-0.0456, +0.0896] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.0688 | [-0.0848, +0.2156] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.1273 | [+0.0307, +0.2576] | worse | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0317 | [+0.0069, +0.0668] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.2155 | [-1.0893, +0.5538] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.5741 | [+0.3270, +0.8095] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.2901 | [-1.2763, +0.4910] | none | 97 | 23 |
| LONGITUDINAL | 2-6s | tg_err | -0.3993 | [-1.6869, +0.5413] | none | 72 | 20 |
| LONGITUDINAL | 2-6s | ttc_err | +0.5374 | [-0.6198, +1.7875] | none | 97 | 23 |
| LONGITUDINAL | full6s | acc_mae | +0.0079 | [-0.0152, +0.0299] | none | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.5741 | [+0.3270, +0.8095] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.1637 | [-0.1007, +0.3908] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.1004 | [+0.0067, +0.2207] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.3790 | [-1.5425, +0.8205] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0014 | [-0.0027, -0.0003] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0844 | [+0.0369, +0.1332] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0200 | [-0.0610, +0.0174] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0002 | [-0.0013, +0.0008] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.6250 | [+0.1889, +1.0777] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +0.4446 | [+0.0039, +0.8554] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0006 | [-0.0016, +0.0004] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.6250 | [+0.1889, +1.0777] | worse | 994 | 71 |
| LATERAL | full6s | ade | +0.2897 | [-0.0151, +0.5755] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0171 | [-0.0282, +0.0614] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0030 | [-0.0040, +0.0131] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0181 | [-0.0805, +0.0392] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0070 | [-0.0191, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0050 | [-0.0161, +0.0040] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0080 | [-0.0161, -0.0010] | better | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0231 | [-0.0070, +0.0533] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | -0.0070 | [-0.0151, -0.0010] | better | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0845 | [+0.0262, +0.1388] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 7 | 3 |
| STRATEGIC | full6s | s_adh_deg | +0.9887 | [-1.5169, +3.8110] | none | 7 | 3 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 852 | 69 |

### ORACLE_BASE (c_ORACLE_BASE_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.3988 | [-0.4819, -0.3162] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.0966 | [-0.0566, +0.2439] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0594 | [-0.0543, +0.2270] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0102 | [-0.0097, +0.0310] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.0282 | [-1.7516, +1.7754] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.1914 | [+0.0126, +0.3669] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.4617 | [-0.8177, -0.1273] | better | 99 | 23 |
| LONGITUDINAL | 2-6s | tg_err | -0.5097 | [-1.1452, -0.0208] | better | 76 | 21 |
| LONGITUDINAL | 2-6s | ttc_err | -0.8602 | [-2.2362, +0.4984] | none | 99 | 23 |
| LONGITUDINAL | full6s | acc_mae | -0.1329 | [-0.1606, -0.1054] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.1914 | [+0.0126, +0.3669] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.1679 | [-0.2980, -0.0327] | better | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0228 | [-0.0739, +0.0229] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.5967 | [-2.0054, +0.7906] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0013 | [-0.0029, -0.0002] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0172 | [-0.0258, +0.0619] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0130 | [-0.0580, +0.0307] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0008 | [-0.0020, +0.0001] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.0400 | [-0.3695, +0.4395] | none | 994 | 71 |
| LATERAL | 2-6s | ade | +0.2206 | [-0.1578, +0.5802] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0009 | [-0.0022, -0.0001] | better | 974 | 71 |
| LATERAL | full6s | xtrack | +0.0400 | [-0.3695, +0.4395] | none | 994 | 71 |
| LATERAL | full6s | ade | +0.1427 | [-0.1210, +0.3944] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0302 | [-0.0070, +0.0734] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0040 | [-0.0020, +0.0131] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0292 | [-0.0212, +0.0815] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0020 | [-0.0080, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0010 | [-0.0060, +0.0070] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0050 | [-0.0111, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0020 | [-0.0312, +0.0302] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | -0.0020 | [-0.0070, +0.0030] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0262 | [-0.0252, +0.0795] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 9 | 4 |
| STRATEGIC | full6s | s_adh_deg | -0.2884 | [-2.6668, +2.7489] | none | 9 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 854 | 69 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.5663 | [+0.4859, +0.6489] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +1.0108 | [+0.8632, +1.1621] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.1628 | [+0.5723, +1.9413] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1581 | [+0.0898, +0.2516] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +3.2625 | [+1.7243, +5.2938] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.2580 | [+0.1939, +0.3218] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | +0.0450 | [-0.2841, +0.3571] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +1.8471 | [+0.7037, +3.4099] | worse | 90 |
| LONGITUDINAL | 2-6s | tg_err | +0.0609 | [-0.9079, +0.8055] | none | 65 |
| LONGITUDINAL | 2-6s | ttc_err | +2.7998 | [-0.0730, +5.4482] | none | 90 |
| LONGITUDINAL | full6s | acc_mae | +0.3607 | [+0.3001, +0.4197] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | +0.0450 | [-0.2841, +0.3571] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.5931 | [+0.5027, +3.0046] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1638 | [+0.0060, +0.3325] | worse | 182 |
| LONGITUDINAL | full6s | ttc_err | +2.7213 | [+0.8388, +4.7882] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0937 | [+0.0861, +0.1016] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.1082 | [+0.0969, +0.1195] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0143 | [+0.0108, +0.0188] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.3323 | [+1.1062, +1.5755] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7857 | [+0.6901, +0.8825] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0465 | [+0.0389, +0.0537] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | 2-6s | curv_err | +0.0065 | [+0.0042, +0.0091] | worse | 967 |
| LATERAL | 2-6s | xtrack | +5.2356 | [+3.8499, +6.7113] | worse | 994 |
| LATERAL | 2-6s | ade | +4.6735 | [+3.7594, +5.6595] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0622 | [+0.0569, +0.0673] | worse | 994 |
| LATERAL | full6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | full6s | curv_err | +0.0098 | [+0.0070, +0.0132] | worse | 974 |
| LATERAL | full6s | xtrack | +5.2356 | [+3.8499, +6.7113] | worse | 994 |
| LATERAL | full6s | ade | +3.3776 | [+2.7436, +4.0615] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4849 | [+0.4235, +0.5493] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2746 | [+0.2294, +0.3239] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.4074 | [+0.3481, +0.4668] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.1056 | [+0.0433, +0.1720] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0835 | [+0.0442, +0.1217] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0775 | [+0.0201, +0.1408] | worse | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1781 | [+0.1096, +0.2485] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2133 | [+0.1489, +0.2736] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.1076 | [+0.0432, +0.1751] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.8333 | [-1.0000, -0.5000] | worse | 6 |
| STRATEGIC | full6s | s_adh_deg | -72.2764 | [-93.1902, -32.8892] | worse | 6 |
| STRATEGIC | full6s | s_false | -0.0843 | [-0.1356, -0.0387] | lower | 854 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### Mechanism rows (beside, no bar)

| arm | DEP mean abs 2 s departure (base) | HIT median final cost | HIT share at B1 |
|---|---|---|---|
| RAL1 | 0.7369 (0.7727) | 8.404e-10 | 0.115 |
| NEG | 0.6122 (0.7727) | 4.378e-06 | 0.342 |
| SHUF | 0.7103 (0.7727) | 1.233e-11 | 0.181 |
| ORACLE | 0.8207 (0.7727) | 1.569e-07 | 0.257 |

### DIR-ACC (beside)

| arm | r | CI | n windows | n episodes |
|---|---|---|---|---|
| BASE | +0.3795 | [+0.3088, +0.4475] | 994 | 71 |
| RAL1 | +0.3947 | [+0.3168, +0.4664] | 994 | 71 |
| MEANL1 | +0.3795 | [+0.3088, +0.4475] | 994 | 71 |
| NEG | -0.1452 | [-0.2143, -0.0786] | 994 | 71 |
| SHUF | +0.3383 | [+0.2715, +0.4044] | 994 | 71 |
| ORACLE | +0.3770 | [+0.3063, +0.4417] | 994 | 71 |

C-SHUF on SCORE: {'delta': 0.1144, 'lo': 0.0673, 'hi': 0.1585, 'separated': True, 'ok': True}; C-OFF5 on SCORE: {'ok': True, 'plan_raw': True, 'a1_raw': True, 'cost1': True}

## Chassis R35_s1: 994 windows, 71 episodes (windows sha256 8833e2923c17)

### RAL1_BASE (c_RAL1_BASE_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.3799 | [-0.4390, -0.3220] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0516 | [-0.1092, -0.0004] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.0011 | [-0.0631, +0.0455] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0020 | [-0.0138, +0.0066] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.6921 | [-1.8854, +0.0037] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0227 | [-0.0736, +0.0307] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.0497 | [-0.2974, +0.2853] | none | 117 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -0.4565 | [-1.3026, +0.0986] | none | 83 | 23 |
| LONGITUDINAL | 2-6s | ttc_err | -0.0860 | [-0.5536, +0.3455] | none | 117 | 25 |
| LONGITUDINAL | full6s | acc_mae | -0.1266 | [-0.1463, -0.1073] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0227 | [-0.0736, +0.0307] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0204 | [-0.0634, +0.1215] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0002 | [-0.0269, +0.0253] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.5524 | [-1.2250, -0.0472] | better | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0002 | [-0.0004, -0.0000] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0024 | [-0.0109, +0.0151] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0489 | [-0.0666, -0.0317] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0000 | [-0.0002, +0.0001] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.0698 | [-0.1457, +0.0054] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.1287 | [-0.2432, -0.0160] | better | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0001 | [-0.0002, +0.0001] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.0698 | [-0.1457, +0.0054] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.1021 | [-0.1831, -0.0227] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0201 | [-0.0423, +0.0030] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0332 | [-0.0634, -0.0050] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0000 | [-0.0030, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0010 | [+0.0000, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0000 | [-0.0030, +0.0030] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0040 | [-0.0211, +0.0131] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | -0.0080 | [-0.0342, +0.0161] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.1250 | [-0.3750, +0.0000] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | -1.6069 | [-2.7340, -0.5293] | worse | 8 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 858 | 70 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.4886 | [+0.4285, +0.5482] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.8700 | [+0.7499, +0.9916] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.2096 | [+0.5731, +2.1647] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1555 | [+0.0691, +0.2761] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +2.9236 | [+1.7157, +4.4884] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1794 | [+0.1204, +0.2353] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | -0.2493 | [-0.5648, +0.0577] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +2.1953 | [+0.9499, +3.7488] | worse | 105 |
| LONGITUDINAL | 2-6s | tg_err | +2.6379 | [+0.7480, +6.3893] | worse | 72 |
| LONGITUDINAL | 2-6s | ttc_err | -0.1479 | [-2.6263, +2.1014] | none | 105 |
| LONGITUDINAL | full6s | acc_mae | +0.2825 | [+0.2349, +0.3280] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | -0.2493 | [-0.5648, +0.0577] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.6274 | [+0.3470, +3.1772] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1535 | [-0.0960, +0.3986] | none | 182 |
| LONGITUDINAL | full6s | ttc_err | +2.3080 | [+0.5706, +4.2771] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0851 | [+0.0772, +0.0926] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.0972 | [+0.0842, +0.1107] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0134 | [+0.0100, +0.0176] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.1988 | [+0.9918, +1.4179] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7173 | [+0.6425, +0.7903] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0449 | [+0.0367, +0.0523] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | 2-6s | curv_err | +0.0077 | [+0.0049, +0.0109] | worse | 967 |
| LATERAL | 2-6s | xtrack | +4.0403 | [+2.8292, +5.2859] | worse | 994 |
| LATERAL | 2-6s | ade | +3.8788 | [+3.2021, +4.6128] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0583 | [+0.0531, +0.0633] | worse | 994 |
| LATERAL | full6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | full6s | curv_err | +0.0102 | [+0.0072, +0.0137] | worse | 974 |
| LATERAL | full6s | xtrack | +4.0403 | [+2.8292, +5.2859] | worse | 994 |
| LATERAL | full6s | ade | +2.8250 | [+2.3576, +3.3385] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4447 | [+0.3873, +0.5010] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2364 | [+0.1871, +0.2887] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3370 | [+0.2857, +0.3904] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.0875 | [+0.0302, +0.1449] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0976 | [+0.0533, +0.1378] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0352 | [-0.0171, +0.0896] | none | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1670 | [+0.1056, +0.2294] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.1972 | [+0.1469, +0.2475] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.0785 | [+0.0181, +0.1408] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.5000 | [-0.8571, -0.2000] | worse | 8 |
| STRATEGIC | full6s | s_adh_deg | -43.8281 | [-77.7902, -11.6106] | worse | 8 |
| STRATEGIC | full6s | s_false | -0.0889 | [-0.1394, -0.0421] | lower | 855 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### RAL1_MEANL1 (c_RAL1_MEANL1_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.0034 | [-0.0223, +0.0304] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0554 | [-0.1132, -0.0029] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0522 | [-0.0103, +0.1384] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0033 | [-0.0172, +0.0065] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.0379 | [-0.1744, +0.2890] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0273 | [-0.0778, +0.0260] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.0273 | [-0.2889, +0.4288] | none | 117 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -0.4359 | [-1.2580, +0.1100] | none | 83 | 23 |
| LONGITUDINAL | 2-6s | ttc_err | -0.1012 | [-0.6065, +0.3333] | none | 117 | 25 |
| LONGITUDINAL | full6s | acc_mae | +0.0011 | [-0.0074, +0.0101] | none | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0273 | [-0.0778, +0.0260] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0871 | [-0.0151, +0.2172] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0128 | [+0.0006, +0.0291] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.1477 | [-0.3942, +0.0711] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0002 | [-0.0004, -0.0001] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.0008 | [-0.0087, +0.0072] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0109 | [-0.0242, +0.0035] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0001 | [-0.0003, +0.0001] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.0722 | [-0.1419, -0.0043] | better | 994 | 71 |
| LATERAL | 2-6s | ade | -0.1075 | [-0.2159, +0.0067] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0002 | [-0.0003, +0.0000] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.0722 | [-0.1419, -0.0043] | better | 994 | 71 |
| LATERAL | full6s | ade | -0.0753 | [-0.1519, +0.0061] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0211 | [-0.0443, +0.0000] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0332 | [-0.0614, -0.0060] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0010 | [-0.0020, +0.0040] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0010 | [+0.0000, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0010 | [-0.0020, +0.0040] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0060 | [-0.0231, +0.0111] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | -0.0010 | [-0.0030, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | -0.0091 | [-0.0352, +0.0161] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.1250 | [-0.3750, +0.0000] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | -1.0502 | [-2.1478, +0.0498] | none | 8 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 858 | 70 |

### RAL1_A1 (c_RAL1_A1_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.4583 | [+0.4025, +0.5136] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.8551 | [+0.7416, +0.9720] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +1.0566 | [+0.5073, +1.8711] | worse | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.1326 | [+0.0640, +0.2280] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +2.5237 | [+1.4178, +3.8819] | worse | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0873 | [-0.0568, +0.2327] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +1.9174 | [+0.9736, +3.1543] | worse | 108 | 23 |
| LONGITUDINAL | 2-6s | tg_err | +1.3172 | [-2.8317, +5.2881] | none | 77 | 22 |
| LONGITUDINAL | 2-6s | ttc_err | +0.7139 | [-0.7744, +1.9568] | none | 108 | 23 |
| LONGITUDINAL | full6s | acc_mae | +0.1528 | [+0.1342, +0.1712] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0873 | [-0.0568, +0.2327] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +1.5448 | [+0.6209, +2.7302] | worse | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.2022 | [+0.0638, +0.3766] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +2.3320 | [+1.1481, +3.6848] | worse | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0809 | [+0.0733, +0.0883] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0914 | [+0.0791, +0.1042] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0125 | [+0.0094, +0.0165] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.1276 | [+0.9297, +1.3331] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.6564 | [+0.5811, +0.7264] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0305 | [+0.0172, +0.0442] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0005 | [-0.0002, +0.0013] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +3.5018 | [+2.5420, +4.5059] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +3.7520 | [+3.1502, +4.3702] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0270 | [+0.0244, +0.0294] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0305 | [+0.0172, +0.0442] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0045 | [+0.0032, +0.0060] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | +3.5018 | [+2.5420, +4.5059] | worse | 994 | 71 |
| LATERAL | full6s | ade | +2.7201 | [+2.2983, +3.1516] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.4316 | [+0.3742, +0.4889] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2294 | [+0.1791, +0.2817] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3179 | [+0.2656, +0.3692] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0020 | [-0.0060, +0.0020] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0020 | [-0.0020, +0.0060] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0020 | [-0.0070, +0.0020] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0855 | [+0.0533, +0.1187] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.1016 | [+0.0604, +0.1439] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0423 | [+0.0080, +0.0765] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.3750 | [-0.8571, +0.0000] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | -10.3116 | [-16.8994, -6.3589] | worse | 8 | 4 |
| STRATEGIC | full6s | s_false | -0.0327 | [-0.0668, -0.0024] | lower | 855 | 70 |

### SHUF_RAL1 (c_SHUF_RAL1_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.0687 | [+0.0250, +0.1135] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0255 | [-0.1121, +0.0724] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.0877 | [-0.1765, -0.0288] | better | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0061 | [-0.0191, +0.0075] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.0797 | [-0.5149, +0.3329] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0322 | [-0.0574, +0.1250] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.1887 | [-0.7264, +0.3239] | none | 117 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -0.4098 | [-2.4237, +0.9430] | none | 83 | 22 |
| LONGITUDINAL | 2-6s | ttc_err | -0.0589 | [-0.7101, +0.6682] | none | 117 | 25 |
| LONGITUDINAL | full6s | acc_mae | +0.0229 | [+0.0083, +0.0378] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0322 | [-0.0574, +0.1250] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.1078 | [-0.3001, +0.0612] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0149 | [-0.0580, +0.0209] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.0058 | [-0.3624, +0.3463] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0005 | [-0.0012, -0.0000] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0229 | [+0.0079, +0.0386] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.0004 | [-0.0225, +0.0232] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0002 | [-0.0009, +0.0004] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.1854 | [+0.0573, +0.3246] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +0.0292 | [-0.1690, +0.2407] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0003 | [-0.0009, +0.0003] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.1854 | [+0.0573, +0.3246] | worse | 994 | 71 |
| LATERAL | full6s | ade | +0.0196 | [-0.1174, +0.1679] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0101 | [-0.0433, +0.0212] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0030 | [-0.0342, +0.0402] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0030 | [+0.0000, +0.0070] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0010 | [-0.0060, +0.0050] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0050 | [-0.0020, +0.0151] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0060 | [-0.0231, +0.0121] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0030 | [+0.0000, +0.0080] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0010 | [-0.0292, +0.0302] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.1250 | [+0.0000, +0.3750] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | +2.1959 | [+0.8359, +5.4792] | better | 8 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 857 | 70 |

### SHUF_BASE (c_SHUF_BASE_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.3113 | [-0.3688, -0.2508] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0771 | [-0.1500, +0.0041] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.0887 | [-0.1942, -0.0259] | better | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0081 | [-0.0248, +0.0066] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.7718 | [-1.6791, -0.1462] | better | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0095 | [-0.0710, +0.0916] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.2385 | [-0.6542, +0.1696] | none | 117 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -0.8558 | [-2.6903, +0.3271] | none | 84 | 22 |
| LONGITUDINAL | 2-6s | ttc_err | -0.1449 | [-0.5840, +0.3082] | none | 117 | 25 |
| LONGITUDINAL | full6s | acc_mae | -0.1038 | [-0.1229, -0.0836] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0095 | [-0.0710, +0.0916] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.0875 | [-0.2620, +0.0687] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0147 | [-0.0499, +0.0139] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.5582 | [-1.2226, -0.0098] | better | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0007 | [-0.0015, -0.0001] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0253 | [+0.0067, +0.0443] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0485 | [-0.0735, -0.0240] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0003 | [-0.0010, +0.0004] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.1156 | [-0.0154, +0.2502] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.0995 | [-0.2858, +0.0982] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0004 | [-0.0011, +0.0003] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.1156 | [-0.0154, +0.2502] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.0825 | [-0.2127, +0.0545] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0302 | [-0.0624, +0.0020] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0302 | [-0.0664, +0.0060] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0030 | [-0.0010, +0.0080] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0000 | [-0.0040, +0.0050] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0050 | [-0.0030, +0.0151] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0101 | [-0.0262, +0.0111] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0030 | [+0.0000, +0.0080] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | -0.0070 | [-0.0352, +0.0221] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 8 | 5 |
| STRATEGIC | full6s | s_adh_deg | +0.9785 | [-0.2935, +3.3968] | none | 8 | 5 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 858 | 70 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.5572 | [+0.4972, +0.6228] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.8445 | [+0.7200, +0.9852] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.1219 | [+0.5299, +1.9972] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1494 | [+0.0659, +0.2679] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +2.8439 | [+1.7092, +4.3500] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1794 | [+0.1204, +0.2353] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | -0.2171 | [-0.5064, +0.0678] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +2.0513 | [+0.9139, +3.4958] | worse | 105 |
| LONGITUDINAL | 2-6s | tg_err | +1.7025 | [+0.5292, +3.3626] | worse | 77 |
| LONGITUDINAL | 2-6s | ttc_err | -0.2135 | [-2.3045, +1.8789] | none | 105 |
| LONGITUDINAL | full6s | acc_mae | +0.3054 | [+0.2575, +0.3521] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | -0.2171 | [-0.5064, +0.0678] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.5196 | [+0.2756, +3.0180] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1386 | [-0.1113, +0.3659] | none | 182 |
| LONGITUDINAL | full6s | ttc_err | +2.3022 | [+0.6983, +4.2503] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0851 | [+0.0772, +0.0926] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.0972 | [+0.0842, +0.1107] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0129 | [+0.0099, +0.0165] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.2217 | [+1.0143, +1.4408] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7177 | [+0.6424, +0.7962] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0449 | [+0.0367, +0.0523] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | 2-6s | curv_err | +0.0075 | [+0.0049, +0.0102] | worse | 967 |
| LATERAL | 2-6s | xtrack | +4.2257 | [+3.0202, +5.4440] | worse | 994 |
| LATERAL | 2-6s | ade | +3.9080 | [+3.1853, +4.6763] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0583 | [+0.0531, +0.0633] | worse | 994 |
| LATERAL | full6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | full6s | curv_err | +0.0099 | [+0.0072, +0.0130] | worse | 974 |
| LATERAL | full6s | xtrack | +4.2257 | [+3.0202, +5.4440] | worse | 994 |
| LATERAL | full6s | ade | +2.8445 | [+2.3434, +3.3773] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4346 | [+0.3803, +0.4909] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2364 | [+0.1871, +0.2887] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3400 | [+0.2907, +0.3913] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.0905 | [+0.0332, +0.1479] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0966 | [+0.0533, +0.1358] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0402 | [-0.0131, +0.0966] | none | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1610 | [+0.1006, +0.2243] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2002 | [+0.1499, +0.2505] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.0795 | [+0.0211, +0.1419] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.4000 | [-0.8571, -0.1000] | worse | 10 |
| STRATEGIC | full6s | s_adh_deg | -46.0662 | [-93.6424, -14.4040] | worse | 10 |
| STRATEGIC | full6s | s_false | -0.0888 | [-0.1392, -0.0422] | lower | 856 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### NEG_RAL1 (c_NEG_RAL1_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.0137 | [-0.0752, +0.0469] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.0104 | [-0.1128, +0.1310] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0697 | [-0.0396, +0.1793] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0415 | [+0.0052, +0.0922] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.5880 | [-0.4517, +1.8221] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.4188 | [+0.1632, +0.6557] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.3355 | [-0.2290, +0.9081] | none | 113 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -1.7008 | [-4.0830, -0.2296] | better | 79 | 22 |
| LONGITUDINAL | 2-6s | ttc_err | +1.5698 | [-0.0453, +3.1120] | none | 113 | 25 |
| LONGITUDINAL | full6s | acc_mae | -0.0046 | [-0.0251, +0.0156] | none | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.4188 | [+0.1632, +0.6557] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.4758 | [+0.1023, +0.9001] | worse | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.1914 | [+0.0316, +0.4482] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +0.8272 | [-0.0691, +1.8197] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0016 | [-0.0036, -0.0002] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0760 | [+0.0295, +0.1220] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0449 | [-0.0788, -0.0122] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0010 | [-0.0028, +0.0006] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.6158 | [+0.2286, +1.0167] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +0.2622 | [-0.0903, +0.6013] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0012 | [-0.0029, +0.0003] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.6158 | [+0.2286, +1.0167] | worse | 994 | 71 |
| LATERAL | full6s | ade | +0.1599 | [-0.0831, +0.3931] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0070 | [-0.0533, +0.0372] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0070 | [+0.0010, +0.0151] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0262 | [-0.0865, +0.0302] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0030 | [-0.0020, +0.0091] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0040 | [-0.0111, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0010 | [-0.0080, +0.0111] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0121 | [-0.0412, +0.0151] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0040 | [-0.0030, +0.0131] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0040 | [-0.0403, +0.0473] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.1429 | [+0.0000, +0.5000] | none | 7 | 3 |
| STRATEGIC | full6s | s_adh_deg | -0.8760 | [-5.8757, +1.1356] | none | 7 | 3 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 853 | 70 |

### ORACLE_BASE (c_ORACLE_BASE_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.2259 | [-0.3137, -0.1354] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.2522 | [+0.0847, +0.4259] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.1357 | [-0.0414, +0.3018] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0193 | [-0.0018, +0.0379] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.0381 | [-1.4991, +1.2854] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.3562 | [+0.1975, +0.5187] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.0407 | [-0.4242, +0.5939] | none | 116 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -0.9768 | [-3.2128, +1.1361] | none | 82 | 22 |
| LONGITUDINAL | 2-6s | ttc_err | +0.4107 | [-0.2911, +1.3522] | none | 116 | 25 |
| LONGITUDINAL | full6s | acc_mae | -0.0753 | [-0.1046, -0.0451] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.3562 | [+0.1975, +0.5187] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.0872 | [-0.3409, +0.1684] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0132 | [-0.0668, +0.0367] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +0.0753 | [-0.8790, +1.0310] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0002 | [-0.0007, +0.0001] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0289 | [-0.0157, +0.0737] | none | 994 | 71 |
| LATERAL | 0-2s | ade | +0.0073 | [-0.0381, +0.0539] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0001 | [-0.0006, +0.0005] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.0833 | [-0.3242, +0.5042] | none | 994 | 71 |
| LATERAL | 2-6s | ade | +0.6221 | [+0.2309, +1.0243] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0001 | [-0.0005, +0.0004] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.0833 | [-0.3242, +0.5042] | none | 994 | 71 |
| LATERAL | full6s | ade | +0.4171 | [+0.1438, +0.7004] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0392 | [-0.0121, +0.0936] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0010 | [+0.0000, +0.0030] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0563 | [-0.0060, +0.1197] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0000 | [-0.0040, +0.0040] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0000 | [-0.0050, +0.0050] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0272 | [+0.0050, +0.0503] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0010 | [-0.0020, +0.0050] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0543 | [+0.0141, +0.0926] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.1429 | [-0.3750, +0.0000] | none | 7 | 4 |
| STRATEGIC | full6s | s_adh_deg | -2.3661 | [-5.0098, +1.1589] | none | 7 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 855 | 70 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.6426 | [+0.5643, +0.7217] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +1.1738 | [+1.0102, +1.3375] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.3464 | [+0.6714, +2.2645] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1768 | [+0.0883, +0.3006] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +3.5775 | [+1.9771, +5.5630] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1794 | [+0.1204, +0.2353] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | +0.1296 | [-0.2031, +0.4675] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +2.2945 | [+1.0222, +4.0537] | worse | 106 |
| LONGITUDINAL | 2-6s | tg_err | +1.9678 | [+0.8123, +3.2930] | worse | 76 |
| LONGITUDINAL | 2-6s | ttc_err | -0.1668 | [-2.4144, +1.8064] | none | 106 |
| LONGITUDINAL | full6s | acc_mae | +0.3338 | [+0.2754, +0.3922] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | +0.1296 | [-0.2031, +0.4675] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.5198 | [+0.3119, +2.9506] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1401 | [-0.0903, +0.3530] | none | 182 |
| LONGITUDINAL | full6s | ttc_err | +2.9357 | [+1.0968, +5.0070] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0851 | [+0.0772, +0.0926] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.0972 | [+0.0842, +0.1107] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0134 | [+0.0100, +0.0175] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.2253 | [+1.0164, +1.4450] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7734 | [+0.6943, +0.8548] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0449 | [+0.0367, +0.0523] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | 2-6s | curv_err | +0.0077 | [+0.0049, +0.0108] | worse | 967 |
| LATERAL | 2-6s | xtrack | +4.1934 | [+2.9579, +5.4673] | worse | 994 |
| LATERAL | 2-6s | ade | +4.6296 | [+3.8209, +5.4730] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0583 | [+0.0531, +0.0633] | worse | 994 |
| LATERAL | full6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | full6s | curv_err | +0.0102 | [+0.0072, +0.0137] | worse | 974 |
| LATERAL | full6s | xtrack | +4.1934 | [+2.9579, +5.4673] | worse | 994 |
| LATERAL | full6s | ade | +3.3442 | [+2.7804, +3.9261] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.5040 | [+0.4416, +0.5654] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2374 | [+0.1881, +0.2897] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.4266 | [+0.3602, +0.4900] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.0875 | [+0.0292, +0.1449] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0966 | [+0.0533, +0.1368] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0352 | [-0.0181, +0.0905] | none | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1982 | [+0.1358, +0.2636] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.1982 | [+0.1479, +0.2485] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.1408 | [+0.0755, +0.2113] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.5000 | [-0.8571, +0.0000] | none | 6 |
| STRATEGIC | full6s | s_adh_deg | -54.7617 | [-81.4778, -4.7813] | worse | 6 |
| STRATEGIC | full6s | s_false | -0.0893 | [-0.1387, -0.0450] | lower | 851 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### Mechanism rows (beside, no bar)

| arm | DEP mean abs 2 s departure (base) | HIT median final cost | HIT share at B1 |
|---|---|---|---|
| RAL1 | 0.7731 (0.8112) | 6.119e-10 | 0.120 |
| NEG | 0.5899 (0.8112) | 1.936e-06 | 0.358 |
| SHUF | 0.7218 (0.8112) | 8.618e-11 | 0.211 |
| ORACLE | 0.9057 (0.8112) | 2.394e-07 | 0.287 |

### DIR-ACC (beside)

| arm | r | CI | n windows | n episodes |
|---|---|---|---|---|
| BASE | +0.4109 | [+0.3502, +0.4682] | 994 | 71 |
| RAL1 | +0.4245 | [+0.3662, +0.4812] | 994 | 71 |
| MEANL1 | +0.4109 | [+0.3502, +0.4682] | 994 | 71 |
| NEG | -0.1787 | [-0.2469, -0.1149] | 994 | 71 |
| SHUF | +0.3403 | [+0.2722, +0.4058] | 994 | 71 |
| ORACLE | +0.3065 | [+0.2395, +0.3693] | 994 | 71 |

C-SHUF on SCORE: {'delta': 0.0687, 'lo': 0.025, 'hi': 0.1135, 'separated': True, 'ok': True}; C-OFF5 on SCORE: {'ok': True, 'plan_raw': True, 'a1_raw': True, 'cost1': True}

