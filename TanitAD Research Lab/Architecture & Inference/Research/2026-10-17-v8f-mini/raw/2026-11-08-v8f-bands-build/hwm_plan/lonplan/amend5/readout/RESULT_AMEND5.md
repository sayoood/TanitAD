# RESULT_AMEND5 (generated from RESULT_AMEND5.json; the MM writes the reading)

Tier T0 (oracle goal): a world-model diagnostic, never driving performance. Estimator: paired episode-cluster bootstrap, B 2,000 seed 0 (another draw of EPISODES only); F_I answers another INFERENCE run.

Base solver: A0 (B3). Committed arm: RAL1. CEIL-ORACLE: False (per chassis {'s0': False, 's1': False}).

## Labels (section 6)

| arm | s0 | s1 | label | effective | qualifier | READ-LIMITED | HOLD-collapsed | abs dH < 2 F_I |
|---|---|---|---|---|---|---|---|---|
| RAL1 | NARROWS | NARROWS | RAL1-NARROWS | RAL1-NARROWS | MEAN-ONLY | True | {'s0': False, 's1': False} | {'s0': False, 's1': False} |

Multiplicity guard: {'two_arms': False, 'ceiling_arms': [], 'guard_applies': False, 'needs_null_replicates': False}

Next step (RAL1): READ-LIMITED: (1) R35's response along RA's direction (ACTRSP Option B / action-sensitivity training; blocked on box 2 and the PI, Q2b); (2) RA's precision (more TRAIN clips' level-1 latents from box 2; blocked, PI P3)

* RAL1 s0: dH +0.4481 [+0.3929, +0.5079]; dB -0.5170 [-0.5771, -0.4557]; dM -0.0136; guard xtrack -0.0207; F_I 0.0073; D 0.877
* RAL1 s1: dH +0.4937 [+0.4328, +0.5544]; dB -0.3748 [-0.4339, -0.3174]; dM +0.0085; guard xtrack +0.0027; F_I 0.0069; D 0.939

## Chassis R35_s0: 994 windows, 71 episodes (windows sha256 371c753bcc3f)

### RAL1_BASE (c_RAL1_BASE_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.5170 | [-0.5771, -0.4557] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0806 | [-0.1463, -0.0193] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.0578 | [-0.1210, +0.0022] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0032 | [-0.0065, +0.0165] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.3569 | [-1.2669, +0.2868] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0688 | [-0.1597, +0.0191] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.1785 | [-0.3980, +0.0420] | none | 103 | 23 |
| LONGITUDINAL | 2-6s | tg_err | +0.0373 | [-0.2094, +0.4232] | none | 77 | 21 |
| LONGITUDINAL | 2-6s | ttc_err | -0.2803 | [-0.6099, +0.0407] | none | 103 | 23 |
| LONGITUDINAL | full6s | acc_mae | -0.1723 | [-0.1924, -0.1519] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0688 | [-0.1597, +0.0191] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0060 | [-0.0963, +0.1403] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0005 | [-0.0157, +0.0148] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.1629 | [-0.6330, +0.2579] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0003 | [-0.0006, +0.0000] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.0207 | [-0.0368, -0.0049] | better | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0719 | [-0.0933, -0.0517] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0001 | [-0.0000, +0.0002] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.1256 | [-0.2181, -0.0331] | better | 994 | 71 |
| LATERAL | 2-6s | ade | -0.2437 | [-0.3935, -0.1138] | better | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0000 | [-0.0002, +0.0002] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.1256 | [-0.2181, -0.0331] | better | 994 | 71 |
| LATERAL | full6s | ade | -0.1864 | [-0.2924, -0.0935] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0282 | [-0.0573, -0.0010] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0000 | [-0.0030, +0.0030] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0453 | [-0.0765, -0.0151] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0010 | [-0.0030, +0.0060] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0020 | [+0.0000, +0.0050] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0010 | [-0.0050, +0.0020] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0131 | [-0.0382, +0.0080] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0030 | [-0.0020, +0.0091] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | -0.0241 | [-0.0533, +0.0020] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | -0.2734 | [-0.9392, +0.2288] | none | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 856 | 69 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.4481 | [+0.3929, +0.5079] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.8336 | [+0.7179, +0.9598] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.0457 | [+0.5231, +1.7893] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1510 | [+0.0800, +0.2485] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +2.9337 | [+1.8740, +4.1743] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.2580 | [+0.1939, +0.3218] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | -0.2151 | [-0.5227, +0.0713] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +2.0293 | [+0.9608, +3.5126] | worse | 95 |
| LONGITUDINAL | 2-6s | tg_err | +0.1966 | [-0.5661, +0.9505] | none | 65 |
| LONGITUDINAL | 2-6s | ttc_err | +3.8343 | [+1.1491, +6.4996] | worse | 95 |
| LONGITUDINAL | full6s | acc_mae | +0.3213 | [+0.2751, +0.3686] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | -0.2151 | [-0.5227, +0.0713] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.7670 | [+0.5897, +3.2824] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1862 | [+0.0075, +0.3672] | worse | 182 |
| LONGITUDINAL | full6s | ttc_err | +3.1551 | [+1.5151, +4.9008] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0937 | [+0.0861, +0.1016] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.1082 | [+0.0969, +0.1195] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0154 | [+0.0115, +0.0201] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.2945 | [+1.0694, +1.5355] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7269 | [+0.6405, +0.8179] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0465 | [+0.0389, +0.0537] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | 2-6s | curv_err | +0.0074 | [+0.0049, +0.0102] | worse | 967 |
| LATERAL | 2-6s | xtrack | +5.0700 | [+3.7051, +6.5348] | worse | 994 |
| LATERAL | 2-6s | ade | +4.2092 | [+3.4023, +5.0908] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0622 | [+0.0569, +0.0673] | worse | 994 |
| LATERAL | full6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | full6s | curv_err | +0.0107 | [+0.0075, +0.0145] | worse | 974 |
| LATERAL | full6s | xtrack | +5.0700 | [+3.7051, +6.5348] | worse | 994 |
| LATERAL | full6s | ade | +3.0484 | [+2.4931, +3.6577] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4266 | [+0.3702, +0.4849] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2706 | [+0.2233, +0.3209] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3330 | [+0.2827, +0.3863] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.1087 | [+0.0453, +0.1751] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0845 | [+0.0443, +0.1247] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0815 | [+0.0231, +0.1439] | worse | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1670 | [+0.1016, +0.2354] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2183 | [+0.1539, +0.2777] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.0573 | [-0.0040, +0.1217] | none | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.6250 | [-1.0000, -0.2500] | worse | 8 |
| STRATEGIC | full6s | s_adh_deg | -53.0275 | [-89.9902, -16.0648] | worse | 8 |
| STRATEGIC | full6s | s_false | -0.0844 | [-0.1332, -0.0393] | lower | 853 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### RAL1_MEANL1 (c_RAL1_MEANL1_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.0136 | [-0.0409, +0.0132] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0798 | [-0.1452, -0.0195] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.0394 | [-0.1245, +0.0148] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0029 | [-0.0090, +0.0021] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.0732 | [-0.1516, +0.3485] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0701 | [-0.1596, +0.0184] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.0126 | [-0.1210, +0.1215] | none | 104 | 23 |
| LONGITUDINAL | 2-6s | tg_err | -0.2295 | [-0.4609, -0.0206] | better | 77 | 21 |
| LONGITUDINAL | 2-6s | ttc_err | -0.0028 | [-0.2092, +0.1599] | none | 104 | 23 |
| LONGITUDINAL | full6s | acc_mae | -0.0045 | [-0.0136, +0.0044] | none | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0701 | [-0.1596, +0.0184] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.0137 | [-0.1191, +0.0580] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0007 | [-0.0075, +0.0079] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.0032 | [-0.2314, +0.2289] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0001 | [-0.0003, +0.0001] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.0137 | [-0.0261, -0.0020] | better | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0138 | [-0.0260, -0.0013] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0000 | [-0.0001, +0.0001] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.1129 | [-0.2039, -0.0242] | better | 994 | 71 |
| LATERAL | 2-6s | ade | -0.2016 | [-0.3432, -0.0714] | better | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0000 | [-0.0001, +0.0001] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.1129 | [-0.2039, -0.0242] | better | 994 | 71 |
| LATERAL | full6s | ade | -0.1390 | [-0.2366, -0.0488] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0322 | [-0.0594, -0.0060] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0020 | [+0.0000, +0.0050] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0553 | [-0.0835, -0.0282] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0030 | [+0.0000, +0.0080] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0010 | [+0.0000, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0010 | [-0.0020, +0.0040] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0131 | [-0.0372, +0.0080] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0040 | [+0.0000, +0.0091] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | -0.0252 | [-0.0533, +0.0020] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | -0.2494 | [-0.9382, +0.2427] | none | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 856 | 69 |

### RAL1_A1 (c_RAL1_A1_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.4416 | [+0.3866, +0.4998] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.8520 | [+0.7371, +0.9770] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.9123 | [+0.4405, +1.5900] | worse | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.1409 | [+0.0729, +0.2361] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +2.4427 | [+1.3675, +3.6713] | worse | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0433 | [-0.1062, +0.1864] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +1.9067 | [+0.8330, +3.3522] | worse | 94 | 22 |
| LONGITUDINAL | 2-6s | tg_err | -0.4178 | [-2.9554, +0.8084] | none | 66 | 20 |
| LONGITUDINAL | 2-6s | ttc_err | +2.6369 | [-0.1330, +6.3422] | none | 94 | 22 |
| LONGITUDINAL | full6s | acc_mae | +0.1472 | [+0.1289, +0.1666] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0433 | [-0.1062, +0.1864] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +1.5575 | [+0.4755, +3.0179] | worse | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0961 | [-0.1067, +0.2807] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +2.6479 | [+1.2229, +4.2486] | worse | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0911 | [+0.0834, +0.0988] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.1046 | [+0.0935, +0.1157] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0149 | [+0.0110, +0.0195] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.2497 | [+1.0320, +1.4897] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.6946 | [+0.6065, +0.7867] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0490 | [+0.0374, +0.0613] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0006 | [-0.0001, +0.0014] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +4.4794 | [+3.3760, +5.7456] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +4.1164 | [+3.4055, +4.8829] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0304 | [+0.0278, +0.0329] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0490 | [+0.0374, +0.0613] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0055 | [+0.0036, +0.0079] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | +4.4794 | [+3.3760, +5.7456] | worse | 994 | 71 |
| LATERAL | full6s | ade | +2.9758 | [+2.4787, +3.5146] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.4326 | [+0.3772, +0.4889] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2757 | [+0.2294, +0.3249] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3360 | [+0.2877, +0.3873] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0030 | [-0.0050, +0.0131] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0030 | [-0.0060, +0.0141] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0040 | [-0.0020, +0.0121] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0895 | [+0.0412, +0.1368] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.1278 | [+0.0774, +0.1761] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0181 | [-0.0231, +0.0604] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.2500 | [-0.5000, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | -11.7810 | [-16.8689, -3.0953] | worse | 12 | 4 |
| STRATEGIC | full6s | s_false | -0.0258 | [-0.0631, +0.0081] | none | 853 | 69 |

### SHUF_RAL1 (c_SHUF_RAL1_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.0937 | [+0.0464, +0.1425] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0616 | [-0.1628, +0.0453] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0644 | [-0.0035, +0.1412] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0063 | [-0.0076, +0.0200] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.1373 | [-0.4940, +1.0891] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.1377 | [+0.0312, +0.2551] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.2324 | [-0.5885, +0.0998] | none | 101 | 22 |
| LONGITUDINAL | 2-6s | tg_err | -0.2515 | [-0.6062, -0.0517] | better | 74 | 20 |
| LONGITUDINAL | 2-6s | ttc_err | -0.0157 | [-0.9494, +0.9648] | none | 101 | 22 |
| LONGITUDINAL | full6s | acc_mae | +0.0312 | [+0.0155, +0.0475] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.1377 | [+0.0312, +0.2551] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0468 | [-0.1167, +0.2461] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0206 | [-0.0156, +0.0892] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.2022 | [-0.7483, +0.4345] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0004 | [-0.0008, -0.0000] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0206 | [+0.0015, +0.0380] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.0041 | [-0.0198, +0.0273] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0005 | [-0.0009, -0.0001] | better | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.2839 | [+0.1042, +0.4896] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +0.0402 | [-0.1956, +0.2835] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0004 | [-0.0008, -0.0001] | better | 974 | 71 |
| LATERAL | full6s | xtrack | +0.2839 | [+0.1042, +0.4896] | worse | 994 | 71 |
| LATERAL | full6s | ade | +0.0282 | [-0.1332, +0.1964] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0050 | [-0.0362, +0.0443] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0010 | [+0.0000, +0.0030] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0010 | [-0.0513, +0.0473] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0020 | [-0.0060, +0.0020] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0030 | [-0.0070, +0.0000] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0030 | [-0.0080, +0.0010] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0141 | [-0.0060, +0.0372] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | -0.0020 | [-0.0050, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0272 | [-0.0040, +0.0584] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | +1.7042 | [+0.3487, +3.9886] | better | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 853 | 69 |

### SHUF_BASE (c_SHUF_BASE_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.4233 | [-0.4811, -0.3620] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.1422 | [-0.2313, -0.0494] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0066 | [-0.0673, +0.0752] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0095 | [-0.0053, +0.0291] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.2196 | [-0.5999, +0.1445] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0689 | [-0.0203, +0.1630] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.4046 | [-0.8130, -0.0374] | better | 101 | 22 |
| LONGITUDINAL | 2-6s | tg_err | -0.3575 | [-0.6023, -0.1787] | better | 75 | 20 |
| LONGITUDINAL | 2-6s | ttc_err | -0.3016 | [-1.3971, +0.6709] | none | 101 | 22 |
| LONGITUDINAL | full6s | acc_mae | -0.1411 | [-0.1604, -0.1207] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0689 | [-0.0203, +0.1630] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0528 | [-0.1290, +0.2649] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0201 | [-0.0216, +0.0887] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.3652 | [-0.9579, +0.1533] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0006 | [-0.0012, -0.0002] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.0001 | [-0.0133, +0.0122] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0677 | [-0.0916, -0.0444] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0004 | [-0.0008, +0.0000] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.1583 | [+0.0046, +0.3408] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | -0.2035 | [-0.4102, +0.0013] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0004 | [-0.0008, -0.0001] | better | 974 | 71 |
| LATERAL | full6s | xtrack | +0.1583 | [+0.0046, +0.3408] | worse | 994 | 71 |
| LATERAL | full6s | ade | -0.1583 | [-0.3027, -0.0160] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0231 | [-0.0604, +0.0131] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0010 | [+0.0000, +0.0030] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0463 | [-0.0936, -0.0020] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0010 | [-0.0050, +0.0020] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0010 | [-0.0050, +0.0020] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0040 | [-0.0091, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0010 | [-0.0191, +0.0211] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0010 | [-0.0020, +0.0040] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0030 | [-0.0272, +0.0332] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | +1.4308 | [+0.3530, +3.0267] | better | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 853 | 69 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.5418 | [+0.4885, +0.5963] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.7720 | [+0.6582, +0.8921] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.1100 | [+0.5620, +1.8728] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1574 | [+0.0863, +0.2527] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +3.0710 | [+1.6631, +4.8699] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.2580 | [+0.1939, +0.3218] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | -0.0774 | [-0.3753, +0.2054] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +1.9615 | [+0.8898, +3.3754] | worse | 93 |
| LONGITUDINAL | 2-6s | tg_err | +0.1088 | [-0.5664, +0.7472] | none | 65 |
| LONGITUDINAL | 2-6s | ttc_err | +4.0741 | [+0.7686, +7.5231] | worse | 93 |
| LONGITUDINAL | full6s | acc_mae | +0.3526 | [+0.3082, +0.3960] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | -0.0774 | [-0.3753, +0.2054] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.8138 | [+0.7218, +3.2848] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.2068 | [+0.0661, +0.3715] | worse | 182 |
| LONGITUDINAL | full6s | ttc_err | +2.9529 | [+1.1101, +5.0988] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0937 | [+0.0861, +0.1016] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.1082 | [+0.0969, +0.1195] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0150 | [+0.0113, +0.0195] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.3150 | [+1.0860, +1.5609] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7310 | [+0.6452, +0.8217] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0465 | [+0.0389, +0.0537] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | 2-6s | curv_err | +0.0069 | [+0.0046, +0.0095] | worse | 967 |
| LATERAL | 2-6s | xtrack | +5.3539 | [+3.9530, +6.8253] | worse | 994 |
| LATERAL | 2-6s | ade | +4.2494 | [+3.4313, +5.1341] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0622 | [+0.0569, +0.0673] | worse | 994 |
| LATERAL | full6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | full6s | curv_err | +0.0103 | [+0.0074, +0.0138] | worse | 974 |
| LATERAL | full6s | xtrack | +5.3539 | [+3.9530, +6.8253] | worse | 994 |
| LATERAL | full6s | ade | +3.0766 | [+2.5107, +3.6929] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4316 | [+0.3783, +0.4849] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2716 | [+0.2254, +0.3220] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3320 | [+0.2847, +0.3793] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.1066 | [+0.0423, +0.1720] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0815 | [+0.0412, +0.1207] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0785 | [+0.0191, +0.1408] | worse | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1811 | [+0.1147, +0.2505] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2163 | [+0.1529, +0.2746] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.0845 | [+0.0262, +0.1449] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.6667 | [-1.0000, -0.3077] | worse | 9 |
| STRATEGIC | full6s | s_adh_deg | -59.1289 | [-96.9161, -22.7404] | worse | 9 |
| STRATEGIC | full6s | s_false | -0.0844 | [-0.1332, -0.0393] | lower | 853 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### NEG_RAL1 (c_NEG_RAL1_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.0351 | [-0.0331, +0.0991] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.0902 | [-0.0624, +0.2324] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.1429 | [+0.0449, +0.2742] | worse | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0333 | [+0.0075, +0.0699] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.1622 | [-0.9438, +0.5409] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.5798 | [+0.3378, +0.8050] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.2059 | [-1.1813, +0.5218] | none | 98 | 23 |
| LONGITUDINAL | 2-6s | tg_err | -0.1394 | [-1.0866, +0.5953] | none | 72 | 20 |
| LONGITUDINAL | 2-6s | ttc_err | +0.2648 | [-0.9695, +1.4372] | none | 98 | 23 |
| LONGITUDINAL | full6s | acc_mae | +0.0117 | [-0.0110, +0.0330] | none | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.5798 | [+0.3378, +0.8050] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.1540 | [-0.1204, +0.3994] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.1009 | [+0.0031, +0.2283] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.3454 | [-1.5499, +0.8738] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0015 | [-0.0029, -0.0003] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0847 | [+0.0371, +0.1338] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0117 | [-0.0522, +0.0252] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0003 | [-0.0015, +0.0007] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.6249 | [+0.1955, +1.0740] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +0.5040 | [+0.0680, +0.9122] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0007 | [-0.0018, +0.0003] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.6249 | [+0.1955, +1.0740] | worse | 994 | 71 |
| LATERAL | full6s | ade | +0.3321 | [+0.0284, +0.6186] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0070 | [-0.0322, +0.0463] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0030 | [-0.0040, +0.0131] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0241 | [-0.0795, +0.0272] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0070 | [-0.0191, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0050 | [-0.0161, +0.0040] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0080 | [-0.0161, -0.0010] | better | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0131 | [-0.0161, +0.0402] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | -0.0070 | [-0.0151, -0.0010] | better | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0624 | [+0.0070, +0.1147] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 7 | 3 |
| STRATEGIC | full6s | s_adh_deg | +1.6901 | [-1.6053, +5.6618] | none | 7 | 3 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 852 | 69 |

### ORACLE_BASE (c_ORACLE_BASE_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.3966 | [-0.4842, -0.3078] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.1003 | [-0.0576, +0.2605] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0523 | [-0.0766, +0.2235] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0016 | [-0.0365, +0.0297] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.0912 | [-1.8654, +1.7220] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.1984 | [+0.0159, +0.3851] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.6470 | [-0.9822, -0.3193] | better | 99 | 23 |
| LONGITUDINAL | 2-6s | tg_err | -0.5234 | [-1.4663, +0.1366] | none | 76 | 21 |
| LONGITUDINAL | 2-6s | ttc_err | -0.7998 | [-2.0083, +0.2719] | none | 99 | 23 |
| LONGITUDINAL | full6s | acc_mae | -0.1322 | [-0.1614, -0.1026] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.1984 | [+0.0159, +0.3851] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.2057 | [-0.3782, -0.0241] | better | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0395 | [-0.0990, +0.0094] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.6287 | [-2.0486, +0.7240] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0014 | [-0.0029, -0.0003] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0157 | [-0.0244, +0.0578] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0134 | [-0.0582, +0.0301] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0008 | [-0.0019, -0.0000] | better | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.0073 | [-0.4007, +0.3625] | none | 994 | 71 |
| LATERAL | 2-6s | ade | +0.2430 | [-0.1427, +0.6313] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0009 | [-0.0020, -0.0001] | better | 974 | 71 |
| LATERAL | full6s | xtrack | -0.0073 | [-0.4007, +0.3625] | none | 994 | 71 |
| LATERAL | full6s | ade | +0.1575 | [-0.1106, +0.4278] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0272 | [-0.0151, +0.0734] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0070 | [-0.0020, +0.0221] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0151 | [-0.0412, +0.0744] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0010 | [-0.0070, +0.0060] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0040 | [-0.0010, +0.0101] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0040 | [-0.0101, +0.0010] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0000 | [-0.0312, +0.0332] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | -0.0010 | [-0.0060, +0.0040] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0141 | [-0.0352, +0.0674] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 9 | 4 |
| STRATEGIC | full6s | s_adh_deg | -0.4564 | [-1.3577, +1.3462] | none | 9 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 854 | 69 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.5686 | [+0.4918, +0.6518] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +1.0145 | [+0.8630, +1.1728] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.1557 | [+0.5703, +1.9324] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1494 | [+0.0854, +0.2377] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +3.1994 | [+1.6381, +5.2442] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.2580 | [+0.1939, +0.3218] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | +0.0521 | [-0.2648, +0.3692] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +1.6588 | [+0.6158, +3.1335] | worse | 90 |
| LONGITUDINAL | 2-6s | tg_err | +0.1601 | [-0.5865, +0.7747] | none | 66 |
| LONGITUDINAL | 2-6s | ttc_err | +3.1214 | [+0.3636, +5.6537] | worse | 90 |
| LONGITUDINAL | full6s | acc_mae | +0.3615 | [+0.3029, +0.4198] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | +0.0521 | [-0.2648, +0.3692] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.5554 | [+0.4761, +2.9190] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1471 | [-0.0116, +0.3089] | none | 182 |
| LONGITUDINAL | full6s | ttc_err | +2.6894 | [+0.7896, +4.7591] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0937 | [+0.0861, +0.1016] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.1082 | [+0.0969, +0.1195] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0143 | [+0.0108, +0.0186] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.3308 | [+1.1056, +1.5762] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7853 | [+0.6943, +0.8817] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0465 | [+0.0389, +0.0537] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | 2-6s | curv_err | +0.0065 | [+0.0043, +0.0091] | worse | 967 |
| LATERAL | 2-6s | xtrack | +5.1883 | [+3.8119, +6.6629] | worse | 994 |
| LATERAL | 2-6s | ade | +4.6960 | [+3.8294, +5.6381] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0622 | [+0.0569, +0.0673] | worse | 994 |
| LATERAL | full6s | head_err | +0.0384 | [+0.0096, +0.0673] | worse | 994 |
| LATERAL | full6s | curv_err | +0.0098 | [+0.0070, +0.0133] | worse | 974 |
| LATERAL | full6s | xtrack | +5.1883 | [+3.8119, +6.6629] | worse | 994 |
| LATERAL | full6s | ade | +3.3924 | [+2.7871, +4.0547] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4819 | [+0.4155, +0.5473] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2777 | [+0.2324, +0.3249] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3934 | [+0.3280, +0.4618] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.1066 | [+0.0433, +0.1731] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0865 | [+0.0463, +0.1268] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0785 | [+0.0201, +0.1419] | worse | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1801 | [+0.1127, +0.2485] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2143 | [+0.1499, +0.2747] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.0956 | [+0.0332, +0.1590] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.8571 | [-1.0000, -0.6250] | worse | 7 |
| STRATEGIC | full6s | s_adh_deg | -78.7772 | [-106.4119, -48.2404] | worse | 7 |
| STRATEGIC | full6s | s_false | -0.0842 | [-0.1356, -0.0386] | lower | 855 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### RAL2_P_BASEP (c_RAL2_P_BASEP_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.4888 | [-0.5586, -0.4182] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0433 | [-0.1589, +0.0829] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0335 | [-0.0292, +0.1012] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0149 | [-0.0092, +0.0464] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.5400 | [-2.0423, +0.3897] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.1656 | [+0.0142, +0.3510] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.2351 | [-0.6391, +1.1793] | none | 126 | 23 |
| LONGITUDINAL | 2-6s | tg_err | +0.3172 | [+0.0236, +0.6241] | worse | 102 | 22 |
| LONGITUDINAL | 2-6s | ttc_err | +0.6908 | [-0.2244, +1.5762] | none | 126 | 23 |
| LONGITUDINAL | full6s | acc_mae | -0.1629 | [-0.1862, -0.1394] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.1656 | [+0.0142, +0.3510] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.2806 | [-0.0831, +0.7022] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0459 | [-0.0099, +0.1117] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.1608 | [-1.1873, +0.7204] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0008 | [-0.0015, -0.0002] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0242 | [-0.0057, +0.0574] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0514 | [-0.0865, -0.0156] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0004 | [-0.0012, +0.0004] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.2174 | [-0.0196, +0.4851] | none | 994 | 71 |
| LATERAL | 2-6s | ade | +0.1228 | [-0.1712, +0.4679] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0006 | [-0.0013, +0.0001] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.2174 | [-0.0196, +0.4851] | none | 994 | 71 |
| LATERAL | full6s | ade | +0.0647 | [-0.1407, +0.3038] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0091 | [-0.0503, +0.0342] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0060 | [-0.0020, +0.0181] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0211 | [-0.0724, +0.0322] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0050 | [-0.0131, +0.0010] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0030 | [-0.0070, +0.0000] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0010 | [-0.0070, +0.0080] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0020 | [-0.0181, +0.0211] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | -0.0060 | [-0.0161, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0191 | [-0.0121, +0.0513] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0833 | [+0.0000, +0.2500] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | +1.3162 | [-0.3937, +3.7100] | none | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 854 | 69 |

### Mechanism rows (beside, no bar)

| arm | DEP mean abs 2 s departure (base) | HIT median final cost | HIT share at B1 |
|---|---|---|---|
| RAL1 | 0.7279 (0.7727) | 6.752e-10 | 0.106 |
| NEG | 0.6171 (0.7727) | 3.076e-06 | 0.339 |
| SHUF | 0.6291 (0.7727) | 3.407e-12 | 0.157 |
| ORACLE | 0.8315 (0.7727) | 1.881e-07 | 0.249 |

### DIR-ACC (beside)

| arm | r | CI | n windows | n episodes |
|---|---|---|---|---|
| BASE | +0.3795 | [+0.3088, +0.4475] | 994 | 71 |
| RAL1 | +0.3923 | [+0.3150, +0.4629] | 994 | 71 |
| MEANL1 | +0.3795 | [+0.3088, +0.4475] | 994 | 71 |
| NEG | -0.1433 | [-0.2110, -0.0768] | 994 | 71 |
| SHUF | +0.2720 | [+0.1887, +0.3506] | 994 | 71 |
| ORACLE | +0.3724 | [+0.2952, +0.4442] | 994 | 71 |

C-SHUF on SCORE: {'delta': 0.0937, 'lo': 0.0464, 'hi': 0.1425, 'separated': True, 'ok': True}; C-OFF5 on SCORE: {'ok': True, 'plan_raw': True, 'a1_raw': True, 'cost1': True}

## Chassis R35_s1: 994 windows, 71 episodes (windows sha256 8833e2923c17)

### RAL1_BASE (c_RAL1_BASE_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.3748 | [-0.4339, -0.3174] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0663 | [-0.1136, -0.0160] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0072 | [-0.0610, +0.0635] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0001 | [-0.0112, +0.0080] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.6458 | [-1.8622, +0.0724] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0142 | [-0.0573, +0.0328] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.0785 | [-0.3284, +0.2626] | none | 117 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -0.4254 | [-1.2292, +0.1237] | none | 83 | 23 |
| LONGITUDINAL | 2-6s | ttc_err | -0.0697 | [-0.5237, +0.3607] | none | 117 | 25 |
| LONGITUDINAL | full6s | acc_mae | -0.1249 | [-0.1446, -0.1058] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0142 | [-0.0573, +0.0328] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.0021 | [-0.0797, +0.0948] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0019 | [-0.0314, +0.0231] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.4310 | [-1.1233, +0.1050] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0002 | [-0.0003, -0.0000] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0027 | [-0.0107, +0.0152] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0503 | [-0.0660, -0.0354] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0000 | [-0.0002, +0.0001] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.0656 | [-0.1394, +0.0026] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.1368 | [-0.2288, -0.0325] | better | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0000 | [-0.0002, +0.0001] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.0656 | [-0.1394, +0.0026] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.1080 | [-0.1741, -0.0344] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0342 | [-0.0604, -0.0091] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | -0.0010 | [-0.0030, +0.0000] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0402 | [-0.0704, -0.0101] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0000 | [-0.0030, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0000 | [-0.0030, +0.0030] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0030 | [-0.0171, +0.0111] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | -0.0080 | [-0.0292, +0.0131] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.1250 | [-0.3750, +0.0000] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | -1.3850 | [-2.7624, -0.0075] | worse | 8 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 858 | 70 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.4937 | [+0.4328, +0.5544] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.8552 | [+0.7339, +0.9828] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.2178 | [+0.5763, +2.1703] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1574 | [+0.0705, +0.2802] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +2.9699 | [+1.7293, +4.5662] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1794 | [+0.1204, +0.2353] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | -0.2407 | [-0.5486, +0.0587] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +2.1320 | [+0.9057, +3.7105] | worse | 105 |
| LONGITUDINAL | 2-6s | tg_err | +2.6936 | [+0.7438, +6.5921] | worse | 72 |
| LONGITUDINAL | 2-6s | ttc_err | -0.1298 | [-2.5900, +2.1177] | none | 105 |
| LONGITUDINAL | full6s | acc_mae | +0.2842 | [+0.2359, +0.3299] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | -0.2407 | [-0.5486, +0.0587] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.6050 | [+0.3224, +3.1474] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1514 | [-0.0974, +0.3979] | none | 182 |
| LONGITUDINAL | full6s | ttc_err | +2.4294 | [+0.6493, +4.4053] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0851 | [+0.0772, +0.0926] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.0972 | [+0.0842, +0.1107] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0134 | [+0.0100, +0.0177] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.1991 | [+0.9902, +1.4190] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7158 | [+0.6416, +0.7887] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0449 | [+0.0367, +0.0523] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | 2-6s | curv_err | +0.0077 | [+0.0049, +0.0109] | worse | 967 |
| LATERAL | 2-6s | xtrack | +4.0445 | [+2.8288, +5.2833] | worse | 994 |
| LATERAL | 2-6s | ade | +3.8707 | [+3.1908, +4.6148] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0583 | [+0.0531, +0.0633] | worse | 994 |
| LATERAL | full6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | full6s | curv_err | +0.0102 | [+0.0072, +0.0137] | worse | 974 |
| LATERAL | full6s | xtrack | +4.0445 | [+2.8288, +5.2833] | worse | 994 |
| LATERAL | full6s | ade | +2.8191 | [+2.3441, +3.3367] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4306 | [+0.3742, +0.4859] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2354 | [+0.1851, +0.2877] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3300 | [+0.2807, +0.3823] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.0875 | [+0.0302, +0.1449] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0966 | [+0.0533, +0.1368] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0352 | [-0.0171, +0.0896] | none | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1680 | [+0.1066, +0.2314] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.1972 | [+0.1469, +0.2475] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.0785 | [+0.0181, +0.1398] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.5000 | [-0.8571, -0.2000] | worse | 8 |
| STRATEGIC | full6s | s_adh_deg | -43.6210 | [-77.3034, -11.5063] | worse | 8 |
| STRATEGIC | full6s | s_false | -0.0889 | [-0.1394, -0.0421] | lower | 855 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### RAL1_MEANL1 (c_RAL1_MEANL1_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.0085 | [-0.0182, +0.0348] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0701 | [-0.1184, -0.0186] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0604 | [-0.0080, +0.1503] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0014 | [-0.0133, +0.0079] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.0842 | [-0.1145, +0.3759] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0187 | [-0.0615, +0.0289] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.0014 | [-0.3083, +0.4261] | none | 117 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -0.4048 | [-1.2064, +0.1364] | none | 83 | 23 |
| LONGITUDINAL | 2-6s | ttc_err | -0.0850 | [-0.5720, +0.3352] | none | 117 | 25 |
| LONGITUDINAL | full6s | acc_mae | +0.0028 | [-0.0061, +0.0116] | none | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0187 | [-0.0615, +0.0289] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0646 | [-0.0337, +0.1932] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0107 | [-0.0014, +0.0272] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.0263 | [-0.2970, +0.2569] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0002 | [-0.0004, -0.0001] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.0004 | [-0.0077, +0.0070] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0123 | [-0.0253, +0.0014] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0001 | [-0.0003, +0.0001] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.0680 | [-0.1324, -0.0055] | better | 994 | 71 |
| LATERAL | 2-6s | ade | -0.1156 | [-0.2149, -0.0051] | better | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0001 | [-0.0003, +0.0000] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.0680 | [-0.1324, -0.0055] | better | 994 | 71 |
| LATERAL | full6s | ade | -0.0812 | [-0.1508, -0.0028] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0352 | [-0.0634, -0.0100] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | -0.0010 | [-0.0030, +0.0000] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0402 | [-0.0724, -0.0111] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0010 | [-0.0020, +0.0040] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0010 | [-0.0020, +0.0040] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0050 | [-0.0191, +0.0091] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | -0.0010 | [-0.0030, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | -0.0091 | [-0.0302, +0.0121] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.1250 | [-0.3750, +0.0000] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | -0.8282 | [-2.2257, +0.5692] | none | 8 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 858 | 70 |

### RAL1_A1 (c_RAL1_A1_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.4635 | [+0.4049, +0.5199] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.8403 | [+0.7305, +0.9564] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +1.0649 | [+0.5075, +1.8754] | worse | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.1345 | [+0.0661, +0.2299] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +2.5700 | [+1.4340, +3.9313] | worse | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0959 | [-0.0423, +0.2370] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +1.8902 | [+0.9561, +3.1549] | worse | 108 | 23 |
| LONGITUDINAL | 2-6s | tg_err | +1.3849 | [-2.7721, +5.4919] | none | 77 | 22 |
| LONGITUDINAL | 2-6s | ttc_err | +0.7315 | [-0.7558, +1.9860] | none | 108 | 23 |
| LONGITUDINAL | full6s | acc_mae | +0.1545 | [+0.1350, +0.1733] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0959 | [-0.0423, +0.2370] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +1.5223 | [+0.6004, +2.7335] | worse | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.2000 | [+0.0605, +0.3729] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +2.4534 | [+1.2167, +3.8469] | worse | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0809 | [+0.0733, +0.0883] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0914 | [+0.0791, +0.1042] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0126 | [+0.0094, +0.0165] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.1279 | [+0.9295, +1.3356] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.6549 | [+0.5791, +0.7270] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0305 | [+0.0172, +0.0442] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0006 | [-0.0002, +0.0013] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +3.5060 | [+2.5396, +4.5135] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +3.7439 | [+3.1371, +4.3711] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0270 | [+0.0244, +0.0294] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0305 | [+0.0172, +0.0442] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0045 | [+0.0032, +0.0061] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | +3.5060 | [+2.5396, +4.5135] | worse | 994 | 71 |
| LATERAL | full6s | ade | +2.7143 | [+2.2887, +3.1544] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.4175 | [+0.3612, +0.4728] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2284 | [+0.1790, +0.2807] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3109 | [+0.2606, +0.3632] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0020 | [-0.0060, +0.0020] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0010 | [-0.0020, +0.0050] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0020 | [-0.0070, +0.0020] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0865 | [+0.0553, +0.1197] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.1016 | [+0.0604, +0.1439] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0423 | [+0.0120, +0.0724] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.3750 | [-0.8571, +0.0000] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | -10.1045 | [-16.5900, -6.2132] | worse | 8 | 4 |
| STRATEGIC | full6s | s_false | -0.0327 | [-0.0668, -0.0024] | lower | 855 | 70 |

### SHUF_RAL1 (c_SHUF_RAL1_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.0503 | [+0.0064, +0.0947] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0838 | [-0.1750, +0.0094] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0012 | [-0.0582, +0.0670] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0152 | [-0.0022, +0.0421] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.0760 | [-0.3268, +0.1583] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0625 | [-0.0301, +0.1642] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.0078 | [-0.5743, +0.6341] | none | 116 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -0.7169 | [-2.1616, +0.3371] | none | 82 | 23 |
| LONGITUDINAL | 2-6s | ttc_err | +0.3665 | [-0.1788, +1.0896] | none | 116 | 25 |
| LONGITUDINAL | full6s | acc_mae | +0.0168 | [+0.0021, +0.0316] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0625 | [-0.0301, +0.1642] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0610 | [-0.0808, +0.2174] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0016 | [-0.0493, +0.0475] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +0.1934 | [-0.2654, +0.6744] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0003 | [-0.0008, +0.0001] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0017 | [-0.0169, +0.0196] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0025 | [-0.0254, +0.0208] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0004 | [-0.0009, +0.0003] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.0393 | [-0.1063, +0.1864] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.1084 | [-0.3215, +0.1043] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0003 | [-0.0008, +0.0002] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.0393 | [-0.1063, +0.1864] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.0731 | [-0.2196, +0.0741] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0101 | [-0.0241, +0.0483] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0010 | [+0.0000, +0.0030] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0030 | [-0.0453, +0.0402] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0010 | [-0.0050, +0.0020] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0010 | [-0.0050, +0.0020] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0010 | [-0.0030, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0030 | [-0.0141, +0.0201] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0161 | [-0.0101, +0.0443] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.1667 | [+0.0000, +0.6000] | none | 6 | 4 |
| STRATEGIC | full6s | s_adh_deg | +0.9308 | [-0.1927, +3.1780] | none | 6 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 856 | 70 |

### SHUF_BASE (c_SHUF_BASE_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.3245 | [-0.3772, -0.2700] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.1502 | [-0.2254, -0.0765] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0084 | [-0.0611, +0.0764] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0151 | [-0.0006, +0.0377] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.7217 | [-1.9219, +0.0180] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0484 | [-0.0393, +0.1466] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.0863 | [-0.4110, +0.4200] | none | 116 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -0.8708 | [-1.9273, -0.1166] | better | 83 | 23 |
| LONGITUDINAL | 2-6s | ttc_err | +0.2961 | [-0.0733, +0.7426] | none | 116 | 25 |
| LONGITUDINAL | full6s | acc_mae | -0.1082 | [-0.1257, -0.0900] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0484 | [-0.0393, +0.1466] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0589 | [-0.0777, +0.2199] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0004 | [-0.0338, +0.0277] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.2376 | [-0.9555, +0.3646] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0005 | [-0.0010, -0.0000] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0044 | [-0.0183, +0.0247] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0528 | [-0.0780, -0.0269] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0004 | [-0.0010, +0.0003] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.0263 | [-0.1833, +0.1265] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.2451 | [-0.4460, -0.0493] | better | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0003 | [-0.0009, +0.0002] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.0263 | [-0.1833, +0.1265] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.1810 | [-0.3200, -0.0439] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0241 | [-0.0513, +0.0030] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0433 | [-0.0775, -0.0101] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0010 | [-0.0050, +0.0020] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0010 | [-0.0050, +0.0020] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0010 | [-0.0040, +0.0020] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0000 | [-0.0161, +0.0191] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0080 | [-0.0171, +0.0342] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 7 | 5 |
| STRATEGIC | full6s | s_adh_deg | -0.2261 | [-0.4689, -0.0142] | worse | 7 | 5 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 856 | 70 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.5441 | [+0.4869, +0.6041] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.7714 | [+0.6510, +0.9005] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.2190 | [+0.5596, +2.1824] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1726 | [+0.0771, +0.3142] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +2.8939 | [+1.6704, +4.4883] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1794 | [+0.1204, +0.2353] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | -0.1782 | [-0.4618, +0.1032] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +2.1626 | [+1.1430, +3.5702] | worse | 104 |
| LONGITUDINAL | 2-6s | tg_err | +1.6747 | [+0.4710, +3.9419] | worse | 75 |
| LONGITUDINAL | 2-6s | ttc_err | +0.3692 | [-1.7434, +2.4480] | none | 104 |
| LONGITUDINAL | full6s | acc_mae | +0.3010 | [+0.2507, +0.3481] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | -0.1782 | [-0.4618, +0.1032] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.6660 | [+0.3698, +3.2112] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1529 | [-0.0940, +0.3968] | none | 182 |
| LONGITUDINAL | full6s | ttc_err | +2.6228 | [+0.9695, +4.5601] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0851 | [+0.0772, +0.0926] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.0972 | [+0.0842, +0.1107] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0132 | [+0.0099, +0.0170] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.2008 | [+0.9934, +1.4163] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7133 | [+0.6371, +0.7917] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0449 | [+0.0367, +0.0523] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | 2-6s | curv_err | +0.0074 | [+0.0047, +0.0104] | worse | 967 |
| LATERAL | 2-6s | xtrack | +4.0838 | [+2.8482, +5.3301] | worse | 994 |
| LATERAL | 2-6s | ade | +3.7624 | [+3.0567, +4.4984] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0583 | [+0.0531, +0.0633] | worse | 994 |
| LATERAL | full6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | full6s | curv_err | +0.0099 | [+0.0070, +0.0133] | worse | 974 |
| LATERAL | full6s | xtrack | +4.0838 | [+2.8482, +5.3301] | worse | 994 |
| LATERAL | full6s | ade | +2.7460 | [+2.2540, +3.2565] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4406 | [+0.3853, +0.4970] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2364 | [+0.1871, +0.2887] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3270 | [+0.2716, +0.3813] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.0865 | [+0.0292, +0.1439] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0956 | [+0.0513, +0.1358] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0342 | [-0.0181, +0.0885] | none | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1710 | [+0.1097, +0.2354] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.1972 | [+0.1469, +0.2475] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.0946 | [+0.0332, +0.1610] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.5714 | [-1.0000, -0.2500] | worse | 7 |
| STRATEGIC | full6s | s_adh_deg | -61.0897 | [-117.0095, -26.3545] | worse | 7 |
| STRATEGIC | full6s | s_false | -0.0901 | [-0.1398, -0.0442] | lower | 855 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### NEG_RAL1 (c_NEG_RAL1_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.0120 | [-0.0745, +0.0490] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.0311 | [-0.0980, +0.1577] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0745 | [-0.0319, +0.1811] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0466 | [+0.0054, +0.1054] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.6044 | [-0.4779, +1.8950] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.3870 | [+0.1297, +0.6324] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.3849 | [-0.2256, +0.9623] | none | 113 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -1.7613 | [-4.2309, -0.2420] | better | 79 | 22 |
| LONGITUDINAL | 2-6s | ttc_err | +1.5339 | [-0.0462, +3.0722] | none | 113 | 25 |
| LONGITUDINAL | full6s | acc_mae | -0.0040 | [-0.0248, +0.0163] | none | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.3870 | [+0.1297, +0.6324] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.5169 | [+0.1385, +0.9434] | worse | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.2183 | [+0.0407, +0.5079] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +0.6985 | [-0.2151, +1.7123] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0017 | [-0.0037, -0.0002] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0768 | [+0.0292, +0.1238] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0371 | [-0.0720, -0.0042] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0011 | [-0.0029, +0.0005] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.6207 | [+0.2239, +1.0326] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +0.2731 | [-0.0758, +0.6164] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0013 | [-0.0031, +0.0003] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.6207 | [+0.2239, +1.0326] | worse | 994 | 71 |
| LATERAL | full6s | ade | +0.1697 | [-0.0713, +0.4048] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0141 | [-0.0282, +0.0543] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0091 | [+0.0020, +0.0181] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0080 | [-0.0624, +0.0433] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0030 | [-0.0020, +0.0091] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0020 | [-0.0080, +0.0040] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0010 | [-0.0080, +0.0111] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0181 | [-0.0463, +0.0080] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0040 | [-0.0030, +0.0131] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0000 | [-0.0443, +0.0423] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.1429 | [+0.0000, +0.5000] | none | 7 | 3 |
| STRATEGIC | full6s | s_adh_deg | -1.1448 | [-6.4955, +1.1064] | none | 7 | 3 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 853 | 70 |

### ORACLE_BASE (c_ORACLE_BASE_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.1945 | [-0.2828, -0.0994] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.2970 | [+0.1306, +0.4744] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.1469 | [-0.0151, +0.3307] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0183 | [-0.0048, +0.0397] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.1825 | [-1.5585, +0.8683] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.3709 | [+0.1794, +0.5545] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.3631 | [-0.3199, +1.1016] | none | 115 | 25 |
| LONGITUDINAL | 2-6s | tg_err | -1.0418 | [-3.7672, +1.2523] | none | 80 | 22 |
| LONGITUDINAL | 2-6s | ttc_err | +0.3270 | [-0.5968, +1.4643] | none | 115 | 25 |
| LONGITUDINAL | full6s | acc_mae | -0.0648 | [-0.0943, -0.0331] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.3709 | [+0.1794, +0.5545] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.1795 | [-0.0301, +0.4418] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0219 | [-0.0406, +0.0868] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.0978 | [-0.9932, +0.8036] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0002 | [-0.0007, +0.0001] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0300 | [-0.0171, +0.0787] | none | 994 | 71 |
| LATERAL | 0-2s | ade | +0.0252 | [-0.0175, +0.0717] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0001 | [-0.0007, +0.0005] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.0990 | [-0.3175, +0.5287] | none | 994 | 71 |
| LATERAL | 2-6s | ade | +0.7166 | [+0.3152, +1.1448] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0001 | [-0.0006, +0.0005] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.0990 | [-0.3175, +0.5287] | none | 994 | 71 |
| LATERAL | full6s | ade | +0.4861 | [+0.2051, +0.7859] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0513 | [+0.0030, +0.1036] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0000 | [-0.0030, +0.0030] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0734 | [+0.0141, +0.1348] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0010 | [-0.0050, +0.0020] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0000 | [-0.0030, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0000 | [-0.0050, +0.0050] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0241 | [+0.0010, +0.0483] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0030 | [+0.0000, +0.0070] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0453 | [+0.0060, +0.0875] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.1250 | [-0.3750, +0.0000] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | -1.9540 | [-5.1742, +1.2662] | none | 8 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 857 | 70 |

a - HOLD:

| family | band | metric | a - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.6740 | [+0.5923, +0.7576] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +1.2186 | [+1.0467, +1.3928] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.3575 | [+0.6725, +2.3050] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1758 | [+0.0886, +0.2975] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +3.4331 | [+1.9845, +5.2050] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1794 | [+0.1204, +0.2353] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | +0.1443 | [-0.1921, +0.4830] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +2.5734 | [+1.1787, +4.3598] | worse | 105 |
| LONGITUDINAL | 2-6s | tg_err | +2.0594 | [+0.6730, +4.0538] | worse | 74 |
| LONGITUDINAL | 2-6s | ttc_err | -0.3757 | [-2.5901, +1.5829] | none | 105 |
| LONGITUDINAL | full6s | acc_mae | +0.3443 | [+0.2848, +0.4029] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | +0.1443 | [-0.1921, +0.4830] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.7865 | [+0.4704, +3.3320] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1752 | [-0.0561, +0.4024] | none | 182 |
| LONGITUDINAL | full6s | ttc_err | +2.7626 | [+0.9511, +4.7714] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0851 | [+0.0772, +0.0926] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.0972 | [+0.0842, +0.1107] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0134 | [+0.0100, +0.0175] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.2265 | [+1.0172, +1.4456] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7913 | [+0.7072, +0.8748] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0449 | [+0.0367, +0.0523] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | 2-6s | curv_err | +0.0076 | [+0.0049, +0.0107] | worse | 967 |
| LATERAL | 2-6s | xtrack | +4.2091 | [+2.9805, +5.4628] | worse | 994 |
| LATERAL | 2-6s | ade | +4.7241 | [+3.8927, +5.5738] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0583 | [+0.0531, +0.0633] | worse | 994 |
| LATERAL | full6s | head_err | +0.0296 | [-0.0018, +0.0588] | none | 994 |
| LATERAL | full6s | curv_err | +0.0102 | [+0.0072, +0.0137] | worse | 974 |
| LATERAL | full6s | xtrack | +4.2091 | [+2.9805, +5.4628] | worse | 994 |
| LATERAL | full6s | ade | +3.4132 | [+2.8310, +4.0079] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.5161 | [+0.4577, +0.5744] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2364 | [+0.1871, +0.2887] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.4437 | [+0.3803, +0.5020] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.0865 | [+0.0282, +0.1439] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0966 | [+0.0533, +0.1358] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0352 | [-0.0181, +0.0905] | none | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1952 | [+0.1318, +0.2606] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2002 | [+0.1509, +0.2505] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.1318 | [+0.0654, +0.2012] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.4286 | [-0.8571, +0.0000] | none | 7 |
| STRATEGIC | full6s | s_adh_deg | -44.3404 | [-80.1760, +3.0942] | none | 7 |
| STRATEGIC | full6s | s_false | -0.0891 | [-0.1396, -0.0422] | lower | 853 |

STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {"state": "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself", "tier_stamp": "T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan

### RAL2_P_BASEP (c_RAL2_P_BASEP_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.4883 | [-0.5588, -0.4184] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.1365 | [-0.2477, -0.0305] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0404 | [-0.0402, +0.1262] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0183 | [-0.0007, +0.0498] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.0471 | [-0.9230, +0.6558] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0602 | [-0.1869, +0.0663] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.5007 | [-0.9436, -0.0701] | better | 113 | 23 |
| LONGITUDINAL | 2-6s | tg_err | -0.1316 | [-2.4726, +2.8050] | none | 80 | 22 |
| LONGITUDINAL | 2-6s | ttc_err | +0.1120 | [-0.8087, +1.0702] | none | 113 | 23 |
| LONGITUDINAL | full6s | acc_mae | -0.1628 | [-0.1863, -0.1395] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0602 | [-0.1869, +0.0663] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0828 | [-0.0716, +0.2490] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0252 | [-0.0000, +0.0566] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.1360 | [-0.6573, +0.3990] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0007 | [-0.0014, -0.0002] | better | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0239 | [+0.0048, +0.0431] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0882 | [-0.1206, -0.0575] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0009 | [-0.0019, +0.0000] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.0795 | [-0.0892, +0.2511] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.3043 | [-0.5819, -0.0325] | better | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0008 | [-0.0016, -0.0001] | better | 974 | 71 |
| LATERAL | full6s | xtrack | +0.0795 | [-0.0892, +0.2511] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.2323 | [-0.4284, -0.0464] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0382 | [-0.0775, +0.0000] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0000 | [-0.0030, +0.0030] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0503 | [-0.0956, -0.0060] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0030 | [-0.0020, +0.0111] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0010 | [-0.0020, +0.0040] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0020 | [-0.0040, +0.0101] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0040 | [-0.0231, +0.0141] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0010 | [-0.0020, +0.0040] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | -0.0050 | [-0.0342, +0.0221] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | +0.3880 | [-2.1034, +4.6309] | none | 8 | 4 |
| STRATEGIC | full6s | s_false | +0.0000 | [+0.0000, +0.0000] | none | 855 | 69 |

### Mechanism rows (beside, no bar)

| arm | DEP mean abs 2 s departure (base) | HIT median final cost | HIT share at B1 |
|---|---|---|---|
| RAL1 | 0.7616 (0.8112) | 6.084e-10 | 0.125 |
| NEG | 0.5975 (0.8112) | 2.614e-06 | 0.351 |
| SHUF | 0.6832 (0.8112) | 3.017e-11 | 0.209 |
| ORACLE | 0.9235 (0.8112) | 3.739e-07 | 0.299 |

### DIR-ACC (beside)

| arm | r | CI | n windows | n episodes |
|---|---|---|---|---|
| BASE | +0.4109 | [+0.3502, +0.4682] | 994 | 71 |
| RAL1 | +0.4127 | [+0.3514, +0.4688] | 994 | 71 |
| MEANL1 | +0.4109 | [+0.3502, +0.4682] | 994 | 71 |
| NEG | -0.1581 | [-0.2237, -0.0951] | 994 | 71 |
| SHUF | +0.3506 | [+0.2940, +0.4045] | 994 | 71 |
| ORACLE | +0.2747 | [+0.2017, +0.3429] | 994 | 71 |

C-SHUF on SCORE: {'delta': 0.0503, 'lo': 0.0064, 'hi': 0.0947, 'separated': True, 'ok': True}; C-OFF5 on SCORE: {'ok': True, 'plan_raw': True, 'a1_raw': True, 'cost1': True}

