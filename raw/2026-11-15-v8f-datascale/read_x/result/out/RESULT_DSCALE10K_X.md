# DSCALE10K X_s0 read (the 10,000 clips alone against B_s0 and U_s0), stages A and B, ONE SEED (seed 0), MEASURED

Outcome: **['X-NO-COST (one seed)']**.  Controls: problems none.  141 episodes / 2,399 windows; paired episode-cluster bootstrap B 2000 seed 0; ruling file sha256 e6e3f24cc79f.

## Deciding cell TAC-LAT (T1-kin, higher is better), X_s0 - B_s0

| stage | Delta [95 % interval] | separated | F_cell | label |
|---|---|---|---|---|
| A (decides) | +0.0317 [-0.0192, +0.0796] | False | 0.0767 | X-NO-COST |
| B (beside) | +0.0071 [-0.0133, +0.0305] | False | 0.0300 | X-NO-COST |

Label: **X-NO-COST (one seed)**.  X_s0 - U_s0 beside (no bar): stage A +0.1188 [+0.0796, +0.1583] sep=True F=0.0767 GAIN; stage B +0.0575 [+0.0304, +0.0851] sep=True F=0.03 GAIN

Next: X-NO-COST: The 10k clips alone teach TAC-LAT as well as the corpus.  The U cost appears only in the mixture, so the mixture arm UM_s0 is the discriminating lever (it waits for the PI's go).

R5: ONE seed; one-pair floor.  X has 10,000 clips against B's 4,512 at the same steps, so X sees each window about 1.8 times (19,164 x 64 / 678,482 bank windows, ESTIMATED) against B's 4.0: the same confound direction as U against B, and block 3 showed that passes do not explain U's TAC-LAT cost.  A CONTENT label is read with that caveat and a NO-COST is conservative.  The geometry goal classes and the STRATEGIC cells are read but do not decide.

K-VLM-ZERO (stage A): X_s0 VLM macro-F1 0.0953, B_s0 0.1257, block-1 floor 0.0274, bars 0.0984 / 0.05: **PASS**.  GEO macro-F1 beside: X_s0 0.5936, B_s0 0.5928.

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-B_s0.A = X_s0 - B_s0, n = 2399 windows / 141 episodes, status gain

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | -0.0650 [-0.0883, -0.0411] sep=True F=0.0496 GAIN | 2399 |
| operative_0_2s:along_mae_m | -0.0191 [-0.0299, -0.0083] sep=True F=0.0154 GAIN | 2399 |
| operative_0_2s:accel_mae_mps2 | -0.2798 [-0.3891, -0.1657] sep=True F=0.2024 GAIN | 2399 |
| operative_0_2s:dk_hw_abs_err_m | -0.0347 [-0.1235, +0.0512] sep=False F=0.0466 . | 669 |
| operative_0_2s:dk_tg_abs_err_s | -0.0255 [-0.0596, -0.0016] sep=True F=0.0166 GAIN | 647 |
| tactical_2_6s:speed_mae_mps | -0.3038 [-0.4579, -0.1523] sep=True F=0.0156 GAIN | 2399 |
| tactical_2_6s:along_mae_m | -0.3133 [-0.4884, -0.1360] sep=True F=0.1837 GAIN | 2399 |
| tactical_2_6s:accel_mae_mps2 | -0.8438 [-1.1567, -0.5404] sep=True F=0.0971 GAIN | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | -0.1454 [-0.3648, +0.0316] sep=False F=0.0054 . | 535 |
| tactical_2_6s:dk_tg_abs_err_s | -0.0508 [-0.1072, -0.0064] sep=True F=0.0068 GAIN | 515 |

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-U_s0.A = X_s0 - U_s0, n = 2399 windows / 141 episodes, status none

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | +0.0347 [+0.0187, +0.0516] sep=True F=0.0496 . | 2399 |
| operative_0_2s:along_mae_m | +0.0140 [+0.0070, +0.0213] sep=True F=0.0154 . | 2399 |
| operative_0_2s:accel_mae_mps2 | +0.1759 [+0.1001, +0.2610] sep=True F=0.2024 . | 2399 |
| operative_0_2s:dk_hw_abs_err_m | +0.0310 [-0.0317, +0.0898] sep=False F=0.0466 . | 666 |
| operative_0_2s:dk_tg_abs_err_s | -0.0006 [-0.0113, +0.0083] sep=False F=0.0166 . | 644 |
| tactical_2_6s:speed_mae_mps | +0.0849 [-0.0520, +0.2284] sep=False F=0.0156 . | 2399 |
| tactical_2_6s:along_mae_m | +0.0554 [-0.0931, +0.2242] sep=False F=0.1837 . | 2399 |
| tactical_2_6s:accel_mae_mps2 | +0.0917 [-0.1819, +0.3629] sep=False F=0.0971 . | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | +0.0485 [-0.2012, +0.2990] sep=False F=0.0054 . | 546 |
| tactical_2_6s:dk_tg_abs_err_s | -0.0208 [-0.0597, +0.0115] sep=False F=0.0068 . | 512 |

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-B_s0.B = X_s0 - B_s0, n = 2399 windows / 141 episodes, status none

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | -0.0012 [-0.0098, +0.0053] sep=False F=0.0026 . | 2399 |
| operative_0_2s:along_mae_m | -0.0012 [-0.0078, +0.0041] sep=False F=0.0036 . | 2399 |
| operative_0_2s:accel_mae_mps2 | +0.0090 [-0.0033, +0.0223] sep=False F=0.0204 . | 2399 |
| operative_0_2s:dk_hw_abs_err_m | +0.0000 [-0.0457, +0.0376] sep=False F=0.0217 . | 698 |
| operative_0_2s:dk_tg_abs_err_s | +0.0021 [-0.0059, +0.0101] sep=False F=0.0073 . | 646 |
| tactical_2_6s:speed_mae_mps | -0.0613 [-0.1627, +0.0301] sep=False F=0.021 . | 2399 |
| tactical_2_6s:along_mae_m | +0.0527 [-0.0488, +0.1543] sep=False F=0.0475 . | 2399 |
| tactical_2_6s:accel_mae_mps2 | -0.3026 [-0.5478, -0.1049] sep=True F=0.0697 GAIN | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | +0.2644 [-0.0736, +0.6107] sep=False F=0.0775 . | 564 |
| tactical_2_6s:dk_tg_abs_err_s | +0.0613 [-0.0098, +0.1636] sep=False F=0.0098 . | 519 |

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-U_s0.B = X_s0 - U_s0, n = 2399 windows / 141 episodes, status gain

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:speed_mae_mps | -0.0068 [-0.0127, -0.0008] sep=True F=0.0026 GAIN | 2399 |
| operative_0_2s:along_mae_m | -0.0001 [-0.0040, +0.0041] sep=False F=0.0036 . | 2399 |
| operative_0_2s:accel_mae_mps2 | -0.0253 [-0.0371, -0.0133] sep=True F=0.0204 GAIN | 2399 |
| operative_0_2s:dk_hw_abs_err_m | -0.0068 [-0.1187, +0.1128] sep=False F=0.0217 . | 695 |
| operative_0_2s:dk_tg_abs_err_s | +0.0091 [-0.0148, +0.0409] sep=False F=0.0073 . | 645 |
| tactical_2_6s:speed_mae_mps | -0.2640 [-0.3753, -0.1600] sep=True F=0.021 GAIN | 2399 |
| tactical_2_6s:along_mae_m | -0.2059 [-0.3511, -0.0525] sep=True F=0.0475 GAIN | 2399 |
| tactical_2_6s:accel_mae_mps2 | -0.8634 [-1.2332, -0.5556] sep=True F=0.0697 GAIN | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | +0.1370 [-0.3359, +0.5788] sep=False F=0.0775 . | 548 |
| tactical_2_6s:dk_tg_abs_err_s | +0.0204 [-0.0925, +0.1790] sep=False F=0.0098 . | 507 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-B_s0.A = X_s0 - B_s0, n = 2399 windows / 141 episodes, status none

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | +0.0026 [-0.0006, +0.0058] sep=False F=0.0025 . | 2399 |
| operative_0_2s:heading_mae_deg | -0.0165 [-0.0609, +0.0245] sep=False F=0.082 . | 2313 |
| operative_0_2s:yaw_rate_mae_degps | -0.0549 [-0.2596, +0.1418] sep=False F=0.3804 . | 2304 |
| operative_0_2s:curvature_mae_1pm | -0.0027 [-0.0082, +0.0024] sep=False F=0.0077 . | 2304 |
| tactical_2_6s:cross_mae_m | +0.0324 [-0.0789, +0.1464] sep=False F=0.1868 . | 2399 |
| tactical_2_6s:heading_mae_deg | -4.0589 [-5.5969, -2.7034] sep=True F=0.6075 GAIN | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | -7.4782 [-11.6862, -3.3545] sep=True F=1.2069 GAIN | 2338 |
| tactical_2_6s:curvature_mae_1pm | -0.0489 [-0.0927, -0.0050] sep=True F=0.0183 GAIN | 2338 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-U_s0.A = X_s0 - U_s0, n = 2399 windows / 141 episodes, status none

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | +0.0006 [-0.0021, +0.0035] sep=False F=0.0025 . | 2399 |
| operative_0_2s:heading_mae_deg | +0.0008 [-0.0454, +0.0435] sep=False F=0.082 . | 2313 |
| operative_0_2s:yaw_rate_mae_degps | +0.1251 [-0.0244, +0.2619] sep=False F=0.3804 . | 2304 |
| operative_0_2s:curvature_mae_1pm | +0.0005 [-0.0037, +0.0045] sep=False F=0.0077 . | 2304 |
| tactical_2_6s:cross_mae_m | -0.1461 [-0.2236, -0.0740] sep=True F=0.1868 . | 2399 |
| tactical_2_6s:heading_mae_deg | -1.5963 [-2.4613, -0.8173] sep=True F=0.6075 GAIN | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | +8.1653 [+3.5827, +13.6814] sep=True F=1.2069 COST | 2338 |
| tactical_2_6s:curvature_mae_1pm | +0.0514 [+0.0064, +0.1045] sep=True F=0.0183 COST | 2338 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-B_s0.B = X_s0 - B_s0, n = 2399 windows / 141 episodes, status cost

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | +0.0024 [-0.0012, +0.0062] sep=False F=0.003 . | 2399 |
| operative_0_2s:heading_mae_deg | +0.0492 [+0.0113, +0.0914] sep=True F=0.0529 . | 2313 |
| operative_0_2s:yaw_rate_mae_degps | +0.2112 [+0.0237, +0.4631] sep=True F=0.0493 COST | 2304 |
| operative_0_2s:curvature_mae_1pm | +0.0034 [-0.0012, +0.0090] sep=False F=0.0001 . | 2304 |
| tactical_2_6s:cross_mae_m | -0.0106 [-0.1131, +0.0888] sep=False F=0.0924 . | 2399 |
| tactical_2_6s:heading_mae_deg | -0.7460 [-1.4560, -0.0965] sep=True F=1.5636 . | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | -3.6048 [-7.7830, -0.0488] sep=True F=3.0812 GAIN | 2340 |
| tactical_2_6s:curvature_mae_1pm | -0.0165 [-0.0430, +0.0089] sep=False F=0.0301 . | 2340 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-U_s0.B = X_s0 - U_s0, n = 2399 windows / 141 episodes, status none

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| operative_0_2s:cross_mae_m | -0.0090 [-0.0140, -0.0045] sep=True F=0.003 GAIN | 2399 |
| operative_0_2s:heading_mae_deg | -0.0798 [-0.1352, -0.0296] sep=True F=0.0529 GAIN | 2313 |
| operative_0_2s:yaw_rate_mae_degps | -0.0392 [-0.1225, +0.0467] sep=False F=0.0493 . | 2303 |
| operative_0_2s:curvature_mae_1pm | -0.0002 [-0.0023, +0.0019] sep=False F=0.0001 . | 2303 |
| tactical_2_6s:cross_mae_m | -0.2706 [-0.3638, -0.1813] sep=True F=0.0924 GAIN | 2399 |
| tactical_2_6s:heading_mae_deg | -2.9591 [-3.8124, -2.1766] sep=True F=1.5636 GAIN | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | -13.9552 [-18.5356, -9.5538] sep=True F=3.0812 GAIN | 2340 |
| tactical_2_6s:curvature_mae_1pm | -0.0756 [-0.1060, -0.0487] sep=True F=0.0301 GAIN | 2340 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-B_s0.A = X_s0 - B_s0, n = 2399 windows / 141 episodes, status none

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | +0.0317 [-0.0192, +0.0796] sep=False F=0.0767 . | 2399 |
| TAC-LON | +0.0063 [-0.0221, +0.0330] sep=False F=0.0179 . | 2399 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-U_s0.A = X_s0 - U_s0, n = 2399 windows / 141 episodes, status gain

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | +0.1188 [+0.0796, +0.1583] sep=True F=0.0767 GAIN | 2399 |
| TAC-LON | +0.0075 [-0.0200, +0.0354] sep=False F=0.0179 . | 2399 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-B_s0.B = X_s0 - B_s0, n = 2399 windows / 141 episodes, status gain

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | +0.0071 [-0.0133, +0.0305] sep=False F=0.03 . | 2399 |
| TAC-LON | +0.0563 [+0.0167, +0.0971] sep=True F=0.0204 GAIN | 2399 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)): X-U_s0.B = X_s0 - U_s0, n = 2399 windows / 141 episodes, status gain

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| TAC-LAT | +0.0575 [+0.0304, +0.0851] sep=True F=0.03 GAIN | 2399 |
| TAC-LON | +0.0938 [+0.0488, +0.1371] sep=True F=0.0204 GAIN | 2399 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): X-B_s0.A = X_s0 - B_s0, n = 2399 windows / 141 episodes, status cost

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| RC-A:pos_err_m@30 | +0.1181 [+0.0256, +0.2301] sep=True F=0.0141 COST | 2322 |
| RC-A:pos_err_m@50 | +0.2911 [+0.0667, +0.5563] sep=True F=0.0643 COST | 2319 |
| S1:pos_err_m@80 | +0.8395 [+0.2224, +1.5332] sep=True F=0.207 COST | 2294 |
| RC-A:head_err_deg@30 | +0.1598 [-0.1429, +0.4852] sep=False F=0.119 . | 2322 |
| RC-A:head_err_deg@50 | +0.7229 [+0.0975, +1.4086] sep=True F=0.1346 COST | 2319 |
| RC-A:head_err_deg@80 | +1.6389 [+0.2273, +3.1890] sep=True F=0.0613 COST | 2294 |
| S2:skill@80 | +0.0000 [-0.0837, +0.0799] sep=False F=0.0164 . | 244 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): X-U_s0.A = X_s0 - U_s0, n = 2399 windows / 141 episodes, status cost

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| RC-A:pos_err_m@30 | +0.0184 [-0.0467, +0.0923] sep=False F=0.0141 . | 2322 |
| RC-A:pos_err_m@50 | +0.0678 [-0.0762, +0.2245] sep=False F=0.0643 . | 2319 |
| S1:pos_err_m@80 | +0.6123 [+0.2086, +1.0296] sep=True F=0.207 COST | 2294 |
| RC-A:head_err_deg@30 | -0.1352 [-0.4516, +0.1514] sep=False F=0.119 . | 2322 |
| RC-A:head_err_deg@50 | +0.5260 [+0.1302, +1.0077] sep=True F=0.1346 COST | 2319 |
| RC-A:head_err_deg@80 | +1.4142 [+0.1561, +2.6774] sep=True F=0.0613 COST | 2294 |
| S2:skill@80 | +0.0205 [-0.0340, +0.0790] sep=False F=0.0164 . | 244 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): X-B_s0.B = X_s0 - B_s0, n = 2399 windows / 141 episodes, status gain

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| RC-A:pos_err_m@30 | +0.0821 [+0.0105, +0.1622] sep=True F=0.0273 COST | 2322 |
| RC-A:pos_err_m@50 | +0.0843 [-0.0704, +0.2531] sep=False F=0.0121 . | 2319 |
| S1:pos_err_m@80 | +0.1658 [-0.2253, +0.5542] sep=False F=0.0451 . | 2294 |
| RC-A:head_err_deg@30 | +0.0459 [-0.2418, +0.3418] sep=False F=0.0517 . | 2322 |
| RC-A:head_err_deg@50 | -0.0233 [-0.4839, +0.4136] sep=False F=0.1747 . | 2319 |
| RC-A:head_err_deg@80 | +0.0751 [-1.0639, +1.1106] sep=False F=0.2306 . | 2294 |
| S2:skill@80 | +0.0902 [+0.0283, +0.1563] sep=True F=0.0041 GAIN | 244 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)): X-U_s0.B = X_s0 - U_s0, n = 2399 windows / 141 episodes, status none

| cell | Delta [95 % interval] sep F verdict | n windows |
|---|---|---|
| RC-A:pos_err_m@30 | -0.0368 [-0.1234, +0.0492] sep=False F=0.0273 . | 2322 |
| RC-A:pos_err_m@50 | -0.0780 [-0.2736, +0.0993] sep=False F=0.0121 . | 2319 |
| S1:pos_err_m@80 | +0.0915 [-0.3541, +0.6054] sep=False F=0.0451 . | 2294 |
| RC-A:head_err_deg@30 | -0.2383 [-0.5832, +0.0808] sep=False F=0.0517 . | 2322 |
| RC-A:head_err_deg@50 | -0.0348 [-0.5847, +0.5483] sep=False F=0.1747 . | 2319 |
| RC-A:head_err_deg@80 | +0.5005 [-0.4772, +2.0813] sep=False F=0.2306 . | 2294 |
| S2:skill@80 | +0.0287 [-0.0300, +0.1008] sep=False F=0.0041 . | 244 |

## STRATEGIC members not in this card

{"detected_information": {"status": "n/a", "reason": "the recipe of record has no ctx head: no output"}, "derived_goals": {"status": "not in this arm"}, "strategic_intent_6_30s": {"status": "not computable", "reason": "no output beyond 80 m of arc"}}