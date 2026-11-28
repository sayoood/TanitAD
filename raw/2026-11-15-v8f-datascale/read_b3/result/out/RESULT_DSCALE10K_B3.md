# DSCALE10K block 3: the H-DATA-SCALE-1 sub-read (UE_s0, BL_s0 against B_s0, U_s0, B_rep), stage A, ONE SEED (seed 0), MEASURED

Outcome: **['NEITHER']**. Controls: problems none.  141 episodes / 2,399 windows; paired episode-cluster bootstrap B 2000 seed 0; tiers: ADE6 and T's latcons T0, LONGITUDINAL / LATERAL / TACTICAL T1-kin, STRATEGIC T0.

## H-DATA-SCALE-1 label (card line 195-197: ADE6 against 2F_A; lower ADE is better)

* F_A = 0.0828 m (|ADE6(B_rep) - ADE6(B_s0)|), 2F_A = 0.1656 m.  One replicate pair: a weak floor.
* Delta_data U_s0 - B_s0 = -0.1280 m; Delta_ep BL_s0 - B_s0 = -0.1190 m.
* U_s0 - BL_s0 (literal 'U - BL') = -0.0090 m -> **NEITHER**; UE_s0 - BL_s0 = -0.0958 m -> **NEITHER**.
* Label: **NEITHER**.  Intervals beside: {"U-BL.A": {"point": -0.008977866237384724, "lo": -0.1055, "hi": 0.0897, "separated": false}, "UE-BL.A": {"point": -0.09575654595833671, "lo": -0.183, "hi": -0.0053, "separated": true}}

T's latcons (T0, beside; T-DATA-LIMITED rule of the card, per pair): U-B_s0.A: NOT T-DATA-LIMITED; BL-B_s0.A: NOT T-DATA-LIMITED; UE-BL.A: NOT T-DATA-LIMITED; U-BL.A: NOT T-DATA-LIMITED; UE-B_s0.A: NOT T-DATA-LIMITED; UE-U_s0.A: NOT T-DATA-LIMITED

Family status at stage A per pair: {"U-B_s0.A": {"LONGITUDINAL": "gain", "LATERAL": "none", "TACTICAL": "cost", "STRATEGIC": "none"}, "BL-B_s0.A": {"LONGITUDINAL": "gain", "LATERAL": "gain", "TACTICAL": "gain", "STRATEGIC": "cost"}, "UE-BL.A": {"LONGITUDINAL": "none", "LATERAL": "none", "TACTICAL": "cost", "STRATEGIC": "gain"}, "U-BL.A": {"LONGITUDINAL": "none", "LATERAL": "none", "TACTICAL": "cost", "STRATEGIC": "none"}}

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): U-B_s0.A = U_s0 - B_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | -0.0997 [-0.1271, -0.0725] sep=True F=0.0496 GAIN | 2399 |
| operative_0_2s:along_mae_m | -0.0331 [-0.0435, -0.0232] sep=True F=0.0154 GAIN | 2399 |
| operative_0_2s:accel_mae_mps2 | -0.4557 [-0.5885, -0.3250] sep=True F=0.2024 GAIN | 2399 |
| operative_0_2s:dk_hw_abs_err_m | -0.0459 [-0.1105, +0.0223] sep=False F=0.0466 . | 668 |
| operative_0_2s:dk_tg_abs_err_s | -0.0231 [-0.0558, +0.0009] sep=False F=0.0166 . | 645 |
| tactical_2_6s:speed_mae_mps | -0.3887 [-0.5334, -0.2443] sep=True F=0.0156 GAIN | 2399 |
| tactical_2_6s:along_mae_m | -0.3688 [-0.5554, -0.1878] sep=True F=0.1837 GAIN | 2399 |
| tactical_2_6s:accel_mae_mps2 | -0.9355 [-1.2461, -0.6469] sep=True F=0.0971 GAIN | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | -0.0859 [-0.2321, +0.0625] sep=False F=0.0054 . | 531 |
| tactical_2_6s:dk_tg_abs_err_s | -0.0187 [-0.0528, +0.0158] sep=False F=0.0068 . | 510 |

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): BL-B_s0.A = BL_s0 - B_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | -0.0862 [-0.1110, -0.0623] sep=True F=0.0496 GAIN | 2399 |
| operative_0_2s:along_mae_m | -0.0278 [-0.0374, -0.0178] sep=True F=0.0154 GAIN | 2399 |
| operative_0_2s:accel_mae_mps2 | -0.4306 [-0.5487, -0.3220] sep=True F=0.2024 GAIN | 2399 |
| operative_0_2s:dk_hw_abs_err_m | -0.0777 [-0.1520, -0.0125] sep=True F=0.0466 GAIN | 672 |
| operative_0_2s:dk_tg_abs_err_s | -0.0117 [-0.0255, +0.0013] sep=False F=0.0166 . | 648 |
| tactical_2_6s:speed_mae_mps | -0.2393 [-0.4059, -0.0844] sep=True F=0.0156 GAIN | 2399 |
| tactical_2_6s:along_mae_m | -0.3600 [-0.5521, -0.1716] sep=True F=0.1837 GAIN | 2399 |
| tactical_2_6s:accel_mae_mps2 | -0.5129 [-0.8395, -0.1842] sep=True F=0.0971 GAIN | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | -0.0540 [-0.3244, +0.2018] sep=False F=0.0054 . | 533 |
| tactical_2_6s:dk_tg_abs_err_s | -0.0361 [-0.0919, +0.0102] sep=False F=0.0068 . | 517 |

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): U-BL.A = U_s0 - BL_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | -0.0135 [-0.0361, +0.0091] sep=False F=0.0496 . | 2399 |
| operative_0_2s:along_mae_m | -0.0052 [-0.0164, +0.0048] sep=False F=0.0154 . | 2399 |
| operative_0_2s:accel_mae_mps2 | -0.0252 [-0.1248, +0.0753] sep=False F=0.2024 . | 2399 |
| operative_0_2s:dk_hw_abs_err_m | +0.0315 [-0.0310, +0.1037] sep=False F=0.0466 . | 676 |
| operative_0_2s:dk_tg_abs_err_s | -0.0116 [-0.0404, +0.0080] sep=False F=0.0166 . | 646 |
| tactical_2_6s:speed_mae_mps | -0.1495 [-0.3219, +0.0105] sep=False F=0.0156 . | 2399 |
| tactical_2_6s:along_mae_m | -0.0088 [-0.1801, +0.1680] sep=False F=0.1837 . | 2399 |
| tactical_2_6s:accel_mae_mps2 | -0.4226 [-0.7516, -0.1069] sep=True F=0.0971 GAIN | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | -0.0475 [-0.3277, +0.2319] sep=False F=0.0054 . | 545 |
| tactical_2_6s:dk_tg_abs_err_s | +0.0284 [-0.0299, +0.1009] sep=False F=0.0068 . | 512 |

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UE-BL.A = UE_s0 - BL_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | -0.0225 [-0.0428, -0.0043] sep=True F=0.0496 . | 2399 |
| operative_0_2s:along_mae_m | -0.0024 [-0.0137, +0.0065] sep=False F=0.0154 . | 2399 |
| operative_0_2s:accel_mae_mps2 | -0.0916 [-0.1820, -0.0046] sep=True F=0.2024 . | 2399 |
| operative_0_2s:dk_hw_abs_err_m | +0.0247 [-0.0556, +0.1142] sep=False F=0.0466 . | 674 |
| operative_0_2s:dk_tg_abs_err_s | -0.0118 [-0.0611, +0.0191] sep=False F=0.0166 . | 649 |
| tactical_2_6s:speed_mae_mps | -0.1878 [-0.3317, -0.0356] sep=True F=0.0156 GAIN | 2399 |
| tactical_2_6s:along_mae_m | -0.1959 [-0.3494, -0.0352] sep=True F=0.1837 GAIN | 2399 |
| tactical_2_6s:accel_mae_mps2 | -0.2847 [-0.6111, +0.0699] sep=False F=0.0971 . | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | +0.1088 [-0.1574, +0.4278] sep=False F=0.0054 . | 548 |
| tactical_2_6s:dk_tg_abs_err_s | +0.0068 [-0.0411, +0.0576] sep=False F=0.0068 . | 515 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): U-B_s0.A = U_s0 - B_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | +0.0020 [-0.0010, +0.0051] sep=False F=0.0025 . | 2399 |
| operative_0_2s:heading_mae_deg | -0.0173 [-0.0594, +0.0236] sep=False F=0.082 . | 2313 |
| operative_0_2s:yaw_rate_mae_degps | -0.1800 [-0.3395, -0.0111] sep=True F=0.3804 . | 2304 |
| operative_0_2s:curvature_mae_1pm | -0.0032 [-0.0070, +0.0006] sep=False F=0.0077 . | 2304 |
| tactical_2_6s:cross_mae_m | +0.1785 [+0.0658, +0.3062] sep=True F=0.1868 . | 2399 |
| tactical_2_6s:heading_mae_deg | -2.4626 [-3.8783, -1.1983] sep=True F=0.6075 GAIN | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | -15.6435 [-21.6621, -10.3488] sep=True F=1.2069 GAIN | 2338 |
| tactical_2_6s:curvature_mae_1pm | -0.1003 [-0.1538, -0.0529] sep=True F=0.0183 GAIN | 2338 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): BL-B_s0.A = BL_s0 - B_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | -0.0112 [-0.0155, -0.0072] sep=True F=0.0025 GAIN | 2399 |
| operative_0_2s:heading_mae_deg | -0.1764 [-0.2297, -0.1326] sep=True F=0.082 GAIN | 2312 |
| operative_0_2s:yaw_rate_mae_degps | -0.4932 [-0.6946, -0.3158] sep=True F=0.3804 GAIN | 2303 |
| operative_0_2s:curvature_mae_1pm | -0.0091 [-0.0151, -0.0040] sep=True F=0.0077 GAIN | 2303 |
| tactical_2_6s:cross_mae_m | -0.1758 [-0.2780, -0.0824] sep=True F=0.1868 . | 2399 |
| tactical_2_6s:heading_mae_deg | -5.1805 [-6.6879, -3.8247] sep=True F=0.6075 GAIN | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | -17.6030 [-23.6252, -12.1535] sep=True F=1.2069 GAIN | 2338 |
| tactical_2_6s:curvature_mae_1pm | -0.1386 [-0.1949, -0.0866] sep=True F=0.0183 GAIN | 2338 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): U-BL.A = U_s0 - BL_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | +0.0132 [+0.0089, +0.0185] sep=True F=0.0025 COST | 2399 |
| operative_0_2s:heading_mae_deg | +0.1639 [+0.1174, +0.2206] sep=True F=0.082 COST | 2312 |
| operative_0_2s:yaw_rate_mae_degps | +0.3149 [+0.1093, +0.5403] sep=True F=0.3804 . | 2303 |
| operative_0_2s:curvature_mae_1pm | +0.0060 [+0.0007, +0.0122] sep=True F=0.0077 . | 2303 |
| tactical_2_6s:cross_mae_m | +0.3543 [+0.2151, +0.5058] sep=True F=0.1868 COST | 2399 |
| tactical_2_6s:heading_mae_deg | +2.7179 [+1.8639, +3.6052] sep=True F=0.6075 COST | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | +1.9330 [-1.6237, +5.4715] sep=False F=1.2069 . | 2339 |
| tactical_2_6s:curvature_mae_1pm | +0.0380 [-0.0029, +0.0817] sep=False F=0.0183 . | 2339 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UE-BL.A = UE_s0 - BL_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | +0.0056 [+0.0019, +0.0098] sep=True F=0.0025 COST | 2399 |
| operative_0_2s:heading_mae_deg | +0.0546 [+0.0105, +0.1047] sep=True F=0.082 . | 2313 |
| operative_0_2s:yaw_rate_mae_degps | +0.0988 [-0.0937, +0.2652] sep=False F=0.3804 . | 2304 |
| operative_0_2s:curvature_mae_1pm | +0.0009 [-0.0035, +0.0049] sep=False F=0.0077 . | 2304 |
| tactical_2_6s:cross_mae_m | +0.2193 [+0.1146, +0.3285] sep=True F=0.1868 COST | 2399 |
| tactical_2_6s:heading_mae_deg | +0.0362 [-0.8934, +0.9161] sep=False F=0.6075 . | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | -0.4208 [-3.3299, +2.3986] sep=False F=1.2069 . | 2340 |
| tactical_2_6s:curvature_mae_1pm | -0.0138 [-0.0387, +0.0094] sep=False F=0.0183 . | 2340 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): U-B_s0.A = U_s0 - B_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | -0.0871 [-0.1297, -0.0446] sep=True F=0.0767 COST | 2399 |
| TAC-LON | -0.0013 [-0.0284, +0.0254] sep=False F=0.0179 . | 2399 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): BL-B_s0.A = BL_s0 - B_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | +0.0817 [+0.0467, +0.1175] sep=True F=0.0767 GAIN | 2399 |
| TAC-LON | +0.0550 [+0.0229, +0.0875] sep=True F=0.0179 GAIN | 2399 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): U-BL.A = U_s0 - BL_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | -0.1688 [-0.2109, -0.1242] sep=True F=0.0767 COST | 2399 |
| TAC-LON | -0.0563 [-0.0967, -0.0154] sep=True F=0.0179 COST | 2399 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UE-BL.A = UE_s0 - BL_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | -0.0855 [-0.1216, -0.0492] sep=True F=0.0767 COST | 2399 |
| TAC-LON | +0.0042 [-0.0342, +0.0433] sep=False F=0.0179 . | 2399 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): U-B_s0.A = U_s0 - B_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| S1:pos_err_m@80 | +0.2273 [-0.2508, +0.7152] sep=False F=0.207 . | 2294 |
| S2:skill@80 | -0.0205 [-0.0747, +0.0361] sep=False F=0.0164 . | 244 |
| RC-A:pos_err_m@30 | +0.0997 [-0.0056, +0.2225] sep=False F=0.0141 . | 2322 |
| RC-A:pos_err_m@50 | +0.2233 [+0.0019, +0.4872] sep=True F=0.0643 COST | 2319 |
| RC-A:head_err_deg@30 | +0.2950 [-0.0519, +0.6969] sep=False F=0.119 . | 2322 |
| RC-A:head_err_deg@50 | +0.1969 [-0.3329, +0.7526] sep=False F=0.1346 . | 2319 |
| RC-A:head_err_deg@80 | +0.2248 [-0.6926, +1.2857] sep=False F=0.0613 . | 2294 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): BL-B_s0.A = BL_s0 - B_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| S1:pos_err_m@80 | +0.4853 [-0.0269, +1.0474] sep=False F=0.207 . | 2294 |
| S2:skill@80 | -0.0861 [-0.1840, -0.0078] sep=True F=0.0164 COST | 244 |
| RC-A:pos_err_m@30 | +0.0023 [-0.0853, +0.1147] sep=False F=0.0141 . | 2322 |
| RC-A:pos_err_m@50 | +0.0805 [-0.0954, +0.2878] sep=False F=0.0643 . | 2319 |
| RC-A:head_err_deg@30 | +0.0152 [-0.2371, +0.2807] sep=False F=0.119 . | 2322 |
| RC-A:head_err_deg@50 | +0.3354 [-0.1894, +0.9389] sep=False F=0.1346 . | 2319 |
| RC-A:head_err_deg@80 | +1.2817 [+0.0768, +2.5911] sep=True F=0.0613 COST | 2294 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): U-BL.A = U_s0 - BL_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| S1:pos_err_m@80 | -0.2581 [-0.9336, +0.3839] sep=False F=0.207 . | 2294 |
| S2:skill@80 | +0.0656 [-0.0078, +0.1478] sep=False F=0.0164 . | 244 |
| RC-A:pos_err_m@30 | +0.0974 [-0.0024, +0.2109] sep=False F=0.0141 . | 2322 |
| RC-A:pos_err_m@50 | +0.1427 [-0.1084, +0.4060] sep=False F=0.0643 . | 2319 |
| RC-A:head_err_deg@30 | +0.2798 [-0.1003, +0.6751] sep=False F=0.119 . | 2322 |
| RC-A:head_err_deg@50 | -0.1385 [-0.7795, +0.4938] sep=False F=0.1346 . | 2319 |
| RC-A:head_err_deg@80 | -1.0569 [-2.4085, +0.2074] sep=False F=0.0613 . | 2294 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): UE-BL.A = UE_s0 - BL_s0, n = 2399 windows / 141 episodes

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| S1:pos_err_m@80 | -0.7246 [-1.3748, -0.1470] sep=True F=0.207 GAIN | 2294 |
| S2:skill@80 | +0.1189 [+0.0513, +0.2016] sep=True F=0.0164 GAIN | 244 |
| RC-A:pos_err_m@30 | +0.0282 [-0.0479, +0.1092] sep=False F=0.0141 . | 2322 |
| RC-A:pos_err_m@50 | -0.1041 [-0.3057, +0.0953] sep=False F=0.0643 . | 2319 |
| RC-A:head_err_deg@30 | -0.0932 [-0.4113, +0.2294] sep=False F=0.119 . | 2322 |
| RC-A:head_err_deg@50 | -0.5046 [-1.1680, +0.0560] sep=False F=0.1346 . | 2319 |
| RC-A:head_err_deg@80 | -1.3199 [-2.7912, -0.1006] sep=True F=0.0613 GAIN | 2294 |

## STRATEGIC members not in this card

{"detected_information": {"status": "n/a", "reason": "the recipe of record has no ctx head: no output"}, "derived_goals": {"status": "not in this arm"}, "strategic_intent_6_30s": {"status": "not computable", "reason": "no output beyond 80 m of arc"}}