# DSCALE mixture arm: UM_s0 against UE_s0 and BL_s0 (deciding), B_s0 and U_s0 (beside), stage A, ONE SEED (seed 0), MEASURED

Outcome: **['DILUTION-PARTIAL [read under UM-A1 (PI-accepted 2026-11-18); K-TACSRC as committed: tac_src_empty_batches 10 > 5, BLOCKED as committed]']**

Controls: problems none; blocked none; unread none.  141 episodes / 2,399 windows; paired episode-cluster bootstrap B 2000 seed 0; tiers: LONGITUDINAL / LATERAL / TACTICAL T1-kin, STRATEGIC T0, ADE6 / latcons / goal F1 T0 beside.  ONE seed: not quotable as a lever effect before the replicate.

## K-TACSRC

* As committed: **tac_src_empty_batches 10 > 5, BLOCKED as committed**.
* read under UM-A1 (PI-accepted 2026-11-18): exact equality with the replay (empty batches, rows listed, rows other) and every other committed field: **PASS** {"tac_src_w": 0.0, "tac_src_list_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1", "tac_src_weight_share_listed": 1.0, "tac_src_rows_listed": 1223523, "tac_src_rows_other": 2708437, "tac_src_empty_batches": 10, "replay": {"empty_batches_replayed": 10, "rows_listed_replayed": 1223523, "rows_other_replayed": 2708437}}

## Deciding label (TAC-LAT, T1-kin, stage A; F_cell = 0.0767)

* UM-UE.A: +0.0692 [+0.0283, +0.1092] sep=True F=0.0767 .
* UM-BL.A: -0.0163 [-0.0433, +0.0121] sep=False F=0.0767 .

Label: **DILUTION-PARTIAL [read under UM-A1 (PI-accepted 2026-11-18); K-TACSRC as committed: tac_src_empty_batches 10 > 5, BLOCKED as committed]**

Beside labels: {"STRAT-KEPT": false, "TAC-LON-PRICE": false, "other_family_costs_in_UM_minus_UE": {}} (read under UM-A1 (PI-accepted 2026-11-18))

T's latcons (T0, beside; T-DATA-LIMITED rule, per pair): UM-UE.A: NOT T-DATA-LIMITED; UM-BL.A: NOT T-DATA-LIMITED; UM-B_s0.A: NOT T-DATA-LIMITED; UM-U_s0.A: NOT T-DATA-LIMITED; UE-BL.A: NOT T-DATA-LIMITED; U-B_s0.A: NOT T-DATA-LIMITED; BL-B_s0.A: NOT T-DATA-LIMITED

Family status per pair: {"UM-UE.A": {"LONGITUDINAL": "none", "LATERAL": "none", "TACTICAL": "none", "STRATEGIC": "none"}, "UM-BL.A": {"LONGITUDINAL": "none", "LATERAL": "none", "TACTICAL": "none", "STRATEGIC": "none"}, "UM-B_s0.A": {"LONGITUDINAL": "gain", "LATERAL": "none", "TACTICAL": "gain", "STRATEGIC": "none"}, "UM-U_s0.A": {"LONGITUDINAL": "none", "LATERAL": "none", "TACTICAL": "gain", "STRATEGIC": "none"}}

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-UE.A = UM_s0 - UE_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | +0.0077 [-0.0046, +0.0198] sep=False F=0.0496 . | 2399 |
| operative_0_2s:along_mae_m | -0.0004 [-0.0073, +0.0062] sep=False F=0.0154 . | 2399 |
| operative_0_2s:accel_mae_mps2 | +0.0709 [+0.0149, +0.1237] sep=True F=0.2024 . | 2399 |
| operative_0_2s:dk_hw_abs_err_m | +0.0059 [-0.0294, +0.0437] sep=False F=0.0466 . | 671 |
| operative_0_2s:dk_tg_abs_err_s | +0.0067 [-0.0136, +0.0357] sep=False F=0.0166 . | 648 |
| tactical_2_6s:speed_mae_mps | +0.1072 [-0.0284, +0.2283] sep=False F=0.0156 . | 2399 |
| tactical_2_6s:along_mae_m | -0.0125 [-0.1822, +0.1352] sep=False F=0.1837 . | 2399 |
| tactical_2_6s:accel_mae_mps2 | +0.2344 [+0.0233, +0.4352] sep=True F=0.0971 COST | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | +0.2321 [+0.0272, +0.4571] sep=True F=0.0054 COST | 553 |
| tactical_2_6s:dk_tg_abs_err_s | +0.0433 [-0.0039, +0.1061] sep=False F=0.0068 . | 522 |

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-BL.A = UM_s0 - BL_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | -0.0148 [-0.0354, +0.0034] sep=False F=0.0496 . | 2399 |
| operative_0_2s:along_mae_m | -0.0028 [-0.0156, +0.0069] sep=False F=0.0154 . | 2399 |
| operative_0_2s:accel_mae_mps2 | -0.0207 [-0.1130, +0.0616] sep=False F=0.2024 . | 2399 |
| operative_0_2s:dk_hw_abs_err_m | +0.0416 [-0.0385, +0.1365] sep=False F=0.0466 . | 672 |
| operative_0_2s:dk_tg_abs_err_s | +0.0035 [-0.0087, +0.0177] sep=False F=0.0166 . | 648 |
| tactical_2_6s:speed_mae_mps | -0.0806 [-0.2293, +0.0785] sep=False F=0.0156 . | 2399 |
| tactical_2_6s:along_mae_m | -0.2084 [-0.3794, -0.0336] sep=True F=0.1837 GAIN | 2399 |
| tactical_2_6s:accel_mae_mps2 | -0.0503 [-0.3427, +0.2477] sep=False F=0.0971 . | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | +0.2302 [-0.0407, +0.5394] sep=False F=0.0054 . | 542 |
| tactical_2_6s:dk_tg_abs_err_s | +0.0383 [-0.0198, +0.1106] sep=False F=0.0068 . | 517 |

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-B_s0.A = UM_s0 - B_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | -0.1010 [-0.1225, -0.0800] sep=True F=0.0496 GAIN | 2399 |
| operative_0_2s:along_mae_m | -0.0306 [-0.0410, -0.0213] sep=True F=0.0154 GAIN | 2399 |
| operative_0_2s:accel_mae_mps2 | -0.4513 [-0.5460, -0.3577] sep=True F=0.2024 GAIN | 2399 |
| operative_0_2s:dk_hw_abs_err_m | -0.0361 [-0.1070, +0.0330] sep=False F=0.0466 . | 671 |
| operative_0_2s:dk_tg_abs_err_s | -0.0082 [-0.0214, +0.0047] sep=False F=0.0166 . | 647 |
| tactical_2_6s:speed_mae_mps | -0.3199 [-0.4868, -0.1598] sep=True F=0.0156 GAIN | 2399 |
| tactical_2_6s:along_mae_m | -0.5684 [-0.7651, -0.3744] sep=True F=0.1837 GAIN | 2399 |
| tactical_2_6s:accel_mae_mps2 | -0.5632 [-0.8781, -0.2535] sep=True F=0.0971 GAIN | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | +0.1729 [-0.0283, +0.3893] sep=False F=0.0054 . | 532 |
| tactical_2_6s:dk_tg_abs_err_s | +0.0258 [-0.0248, +0.0829] sep=False F=0.0068 . | 516 |

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-U_s0.A = UM_s0 - U_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | -0.0013 [-0.0185, +0.0163] sep=False F=0.0496 . | 2399 |
| operative_0_2s:along_mae_m | +0.0024 [-0.0054, +0.0105] sep=False F=0.0154 . | 2399 |
| operative_0_2s:accel_mae_mps2 | +0.0045 [-0.0828, +0.0925] sep=False F=0.2024 . | 2399 |
| operative_0_2s:dk_hw_abs_err_m | +0.0381 [-0.0144, +0.0983] sep=False F=0.0466 . | 670 |
| operative_0_2s:dk_tg_abs_err_s | +0.0172 [-0.0002, +0.0460] sep=False F=0.0166 . | 646 |
| tactical_2_6s:speed_mae_mps | +0.0689 [-0.0935, +0.2306] sep=False F=0.0156 . | 2399 |
| tactical_2_6s:along_mae_m | -0.1996 [-0.3601, -0.0419] sep=True F=0.1837 GAIN | 2399 |
| tactical_2_6s:accel_mae_mps2 | +0.3723 [+0.1137, +0.6348] sep=True F=0.0971 COST | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | +0.2863 [+0.0732, +0.5120] sep=True F=0.0054 COST | 541 |
| tactical_2_6s:dk_tg_abs_err_s | +0.0406 [-0.0123, +0.1014] sep=False F=0.0068 . | 516 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-UE.A = UM_s0 - UE_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | +0.0009 [-0.0015, +0.0031] sep=False F=0.0025 . | 2399 |
| operative_0_2s:heading_mae_deg | +0.0185 [-0.0202, +0.0567] sep=False F=0.082 . | 2313 |
| operative_0_2s:yaw_rate_mae_degps | +0.0735 [-0.0618, +0.2168] sep=False F=0.3804 . | 2306 |
| operative_0_2s:curvature_mae_1pm | +0.0028 [-0.0010, +0.0067] sep=False F=0.0077 . | 2306 |
| tactical_2_6s:cross_mae_m | -0.0742 [-0.1471, -0.0023] sep=True F=0.1868 . | 2399 |
| tactical_2_6s:heading_mae_deg | +1.0144 [+0.1091, +2.0890] sep=True F=0.6075 COST | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | +3.3913 [+0.6367, +6.3063] sep=True F=1.2069 COST | 2339 |
| tactical_2_6s:curvature_mae_1pm | +0.0263 [+0.0043, +0.0511] sep=True F=0.0183 COST | 2339 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-BL.A = UM_s0 - BL_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | +0.0064 [+0.0028, +0.0103] sep=True F=0.0025 COST | 2399 |
| operative_0_2s:heading_mae_deg | +0.0804 [+0.0285, +0.1338] sep=True F=0.082 . | 2312 |
| operative_0_2s:yaw_rate_mae_degps | +0.1934 [+0.0486, +0.3599] sep=True F=0.3804 . | 2304 |
| operative_0_2s:curvature_mae_1pm | +0.0045 [+0.0007, +0.0091] sep=True F=0.0077 . | 2304 |
| tactical_2_6s:cross_mae_m | +0.1452 [+0.0392, +0.2535] sep=True F=0.1868 . | 2399 |
| tactical_2_6s:heading_mae_deg | +1.0506 [+0.2792, +1.8632] sep=True F=0.6075 COST | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | +2.9977 [-0.7908, +7.1081] sep=False F=1.2069 . | 2339 |
| tactical_2_6s:curvature_mae_1pm | +0.0124 [-0.0184, +0.0451] sep=False F=0.0183 . | 2339 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-B_s0.A = UM_s0 - B_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | -0.0048 [-0.0085, -0.0014] sep=True F=0.0025 GAIN | 2399 |
| operative_0_2s:heading_mae_deg | -0.1036 [-0.1605, -0.0506] sep=True F=0.082 GAIN | 2313 |
| operative_0_2s:yaw_rate_mae_degps | -0.3099 [-0.4522, -0.1780] sep=True F=0.3804 . | 2304 |
| operative_0_2s:curvature_mae_1pm | -0.0049 [-0.0085, -0.0017] sep=True F=0.0077 . | 2304 |
| tactical_2_6s:cross_mae_m | -0.0307 [-0.1433, +0.0798] sep=False F=0.1868 . | 2399 |
| tactical_2_6s:heading_mae_deg | -4.1298 [-5.5983, -2.7707] sep=True F=0.6075 GAIN | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | -14.5880 [-19.9924, -9.2715] sep=True F=1.2069 GAIN | 2337 |
| tactical_2_6s:curvature_mae_1pm | -0.1280 [-0.1709, -0.0882] sep=True F=0.0183 GAIN | 2337 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-U_s0.A = UM_s0 - U_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | -0.0068 [-0.0101, -0.0040] sep=True F=0.0025 GAIN | 2399 |
| operative_0_2s:heading_mae_deg | -0.0862 [-0.1334, -0.0421] sep=True F=0.082 GAIN | 2313 |
| operative_0_2s:yaw_rate_mae_degps | -0.1524 [-0.2826, -0.0227] sep=True F=0.3804 . | 2305 |
| operative_0_2s:curvature_mae_1pm | -0.0026 [-0.0062, +0.0005] sep=False F=0.0077 . | 2305 |
| tactical_2_6s:cross_mae_m | -0.2091 [-0.3289, -0.0956] sep=True F=0.1868 GAIN | 2399 |
| tactical_2_6s:heading_mae_deg | -1.6673 [-2.5875, -0.6922] sep=True F=0.6075 GAIN | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | +1.0969 [-2.5429, +5.2003] sep=False F=1.2069 . | 2338 |
| tactical_2_6s:curvature_mae_1pm | -0.0261 [-0.0603, +0.0050] sep=False F=0.0183 . | 2338 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-UE.A = UM_s0 - UE_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | +0.0692 [+0.0283, +0.1092] sep=True F=0.0767 . | 2399 |
| TAC-LON | -0.0179 [-0.0521, +0.0158] sep=False F=0.0179 . | 2399 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-BL.A = UM_s0 - BL_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | -0.0163 [-0.0433, +0.0121] sep=False F=0.0767 . | 2399 |
| TAC-LON | -0.0138 [-0.0509, +0.0254] sep=False F=0.0179 . | 2399 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-B_s0.A = UM_s0 - B_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | +0.0654 [+0.0329, +0.0983] sep=True F=0.0767 . | 2399 |
| TAC-LON | +0.0413 [+0.0075, +0.0776] sep=True F=0.0179 GAIN | 2399 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): UM-U_s0.A = UM_s0 - U_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | +0.1526 [+0.1067, +0.1992] sep=True F=0.0767 GAIN | 2399 |
| TAC-LON | +0.0425 [+0.0058, +0.0808] sep=True F=0.0179 GAIN | 2399 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): UM-UE.A = UM_s0 - UE_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| S1:pos_err_m@80 | +0.0915 [-0.2207, +0.4370] sep=False F=0.207 . | 2294 |
| S2:skill@80 | -0.0533 [-0.1440, +0.0305] sep=False F=0.0164 . | 244 |
| RC-A:pos_err_m@30 | +0.0080 [-0.0487, +0.0644] sep=False F=0.0141 . | 2322 |
| RC-A:pos_err_m@50 | +0.0490 [-0.0853, +0.1898] sep=False F=0.0643 . | 2319 |
| RC-A:head_err_deg@30 | +0.0494 [-0.1847, +0.2840] sep=False F=0.119 . | 2322 |
| RC-A:head_err_deg@50 | +0.0477 [-0.3656, +0.4931] sep=False F=0.1346 . | 2319 |
| RC-A:head_err_deg@80 | -0.3334 [-1.0458, +0.3302] sep=False F=0.0613 . | 2294 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): UM-BL.A = UM_s0 - BL_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| S1:pos_err_m@80 | -0.6331 [-1.3560, +0.0080] sep=False F=0.207 . | 2294 |
| S2:skill@80 | +0.0656 [-0.0453, +0.1702] sep=False F=0.0164 . | 244 |
| RC-A:pos_err_m@30 | +0.0362 [-0.0431, +0.1201] sep=False F=0.0141 . | 2322 |
| RC-A:pos_err_m@50 | -0.0551 [-0.2948, +0.1806] sep=False F=0.0643 . | 2319 |
| RC-A:head_err_deg@30 | -0.0438 [-0.3641, +0.2856] sep=False F=0.119 . | 2322 |
| RC-A:head_err_deg@50 | -0.4569 [-1.2871, +0.2421] sep=False F=0.1346 . | 2319 |
| RC-A:head_err_deg@80 | -1.6533 [-3.3965, -0.1426] sep=True F=0.0613 GAIN | 2294 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): UM-B_s0.A = UM_s0 - B_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| S1:pos_err_m@80 | -0.1478 [-0.5416, +0.2257] sep=False F=0.207 . | 2294 |
| S2:skill@80 | -0.0205 [-0.0886, +0.0402] sep=False F=0.0164 . | 244 |
| RC-A:pos_err_m@30 | +0.0385 [-0.0299, +0.1232] sep=False F=0.0141 . | 2322 |
| RC-A:pos_err_m@50 | +0.0255 [-0.1153, +0.1824] sep=False F=0.0643 . | 2319 |
| RC-A:head_err_deg@30 | -0.0286 [-0.2228, +0.1833] sep=False F=0.119 . | 2322 |
| RC-A:head_err_deg@50 | -0.1215 [-0.5548, +0.2712] sep=False F=0.1346 . | 2319 |
| RC-A:head_err_deg@80 | -0.3716 [-1.2411, +0.4591] sep=False F=0.0613 . | 2294 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): UM-U_s0.A = UM_s0 - U_s0, n = 2399 windows / 141 episodes (read under UM-A1 (PI-accepted 2026-11-18))

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| S1:pos_err_m@80 | -0.3751 [-0.9680, +0.1389] sep=False F=0.207 . | 2294 |
| S2:skill@80 | +0.0000 [-0.0837, +0.0661] sep=False F=0.0164 . | 244 |
| RC-A:pos_err_m@30 | -0.0612 [-0.1552, +0.0262] sep=False F=0.0141 . | 2322 |
| RC-A:pos_err_m@50 | -0.1978 [-0.4078, +0.0007] sep=False F=0.0643 . | 2319 |
| RC-A:head_err_deg@30 | -0.3236 [-0.6930, -0.0270] sep=True F=0.119 GAIN | 2322 |
| RC-A:head_err_deg@50 | -0.3184 [-1.0604, +0.3185] sep=False F=0.1346 . | 2319 |
| RC-A:head_err_deg@80 | -0.5964 [-1.8852, +0.3695] sep=False F=0.0613 . | 2294 |

## STRATEGIC members not in this card

{"detected_information": {"status": "n/a", "reason": "no ctx head in the recipe of record: no output"}, "derived_goals": {"status": "not in this arm"}, "strategic_intent_6_30s": {"status": "not computable", "reason": "no output beyond 80 m of arc"}}