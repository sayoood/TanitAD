# DSCALE10K block 2: the deciding read (B_s0 / U_s0 and B_s1 / U_s1), MEASURED; 141 episodes / 2,399 windows; paired episode-cluster bootstrap B 2,000 seed 0

Outcome: **['DATA-HURTS']**. Controls: problems none. Reach (R_s0, T0): REACHED. T mechanism (T0, beside) seed 0: NOT T-DATA-LIMITED; seed 1: NOT T-DATA-LIMITED.
Seed 0: reused from RESULT_DSCALE10K_B1.json (K-RE B1: every recomputed field equals the banked row exactly). Seed 1: computed here. Floor F: the seed-0 replicate pair (B_rep), at both seeds. Evidence class MEASURED; tiers: LONGITUDINAL / LATERAL / TACTICAL T1-kin, STRATEGIC T0.

Family status at the deciding stage B: {"seed_0": {"LONGITUDINAL": "none", "LATERAL": "cost", "TACTICAL": "cost", "STRATEGIC": "none"}, "seed_1": {"LONGITUDINAL": "none", "LATERAL": "none", "TACTICAL": "cost", "STRATEGIC": "gain"}}

## LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)), n = 2399 windows / 141 episodes

| cell | seed 0 (U_s0 - B_s0): Delta [95 % interval] sep F verdict | seed 1 (U_s1 - B_s1) | n windows |
|---|---|---|---|
| operative_0_2s:speed_mae_mps | +0.0056 [-0.0051, +0.0140] sep=False F=0.0026 . | +0.0025 [-0.0048, +0.0099] sep=False F=0.0026 . | 2399 |
| operative_0_2s:along_mae_m | -0.0011 [-0.0088, +0.0049] sep=False F=0.0036 . | -0.0074 [-0.0130, -0.0021] sep=True F=0.0036 GAIN | 2399 |
| operative_0_2s:accel_mae_mps2 | +0.0343 [+0.0206, +0.0483] sep=True F=0.0204 COST | +0.0256 [+0.0128, +0.0390] sep=True F=0.0204 COST | 2399 |
| operative_0_2s:dk_hw_abs_err_m | +0.0070 [-0.0953, +0.0987] sep=False F=0.0217 . | +0.0317 [-0.0146, +0.1009] sep=False F=0.0217 . | 698 |
| operative_0_2s:dk_tg_abs_err_s | -0.0067 [-0.0353, +0.0165] sep=False F=0.0073 . | +0.0117 [+0.0011, +0.0265] sep=True F=0.0073 COST | 646 |
| tactical_2_6s:speed_mae_mps | +0.2027 [+0.0954, +0.3039] sep=True F=0.021 COST | +0.1191 [+0.0137, +0.2184] sep=True F=0.021 COST | 2399 |
| tactical_2_6s:along_mae_m | +0.2586 [+0.0881, +0.4153] sep=True F=0.0475 COST | +0.1647 [+0.0192, +0.3025] sep=True F=0.0475 COST | 2399 |
| tactical_2_6s:accel_mae_mps2 | +0.5608 [+0.3166, +0.8501] sep=True F=0.0697 COST | -0.0755 [-0.2915, +0.1261] sep=False F=0.0697 . | 2399 |
| tactical_2_6s:dk_hw_abs_err_m | +0.0420 [-0.2114, +0.3017] sep=False F=0.0775 . | -0.0427 [-0.3651, +0.2159] sep=False F=0.0775 . | 547 |
| tactical_2_6s:dk_tg_abs_err_s | +0.0460 [-0.0236, +0.1205] sep=False F=0.0098 . | +0.0027 [-0.0638, +0.0580] sep=False F=0.0098 . | 514 |

## LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)), n = 2399 windows / 141 episodes

| cell | seed 0 (U_s0 - B_s0): Delta [95 % interval] sep F verdict | seed 1 (U_s1 - B_s1) | n windows |
|---|---|---|---|
| operative_0_2s:cross_mae_m | +0.0114 [+0.0059, +0.0175] sep=True F=0.003 COST | +0.0017 [-0.0021, +0.0051] sep=False F=0.003 . | 2399 |
| operative_0_2s:heading_mae_deg | +0.1290 [+0.0741, +0.1880] sep=True F=0.0529 COST | +0.0528 [+0.0178, +0.0895] sep=True F=0.0529 . | 2313 |
| operative_0_2s:yaw_rate_mae_degps | +0.2533 [+0.0608, +0.4900] sep=True F=0.0493 COST | +0.0468 [-0.0349, +0.1187] sep=False F=0.0493 . | 2305 |
| operative_0_2s:curvature_mae_1pm | +0.0037 [-0.0008, +0.0093] sep=False F=0.0001 . | -0.0001 [-0.0018, +0.0015] sep=False F=0.0001 . | 2305 |
| tactical_2_6s:cross_mae_m | +0.2600 [+0.1527, +0.3667] sep=True F=0.0924 COST | -0.0617 [-0.1677, +0.0330] sep=False F=0.0924 . | 2399 |
| tactical_2_6s:heading_mae_deg | +2.2131 [+1.5327, +2.9653] sep=True F=1.5636 COST | -0.8022 [-1.6399, -0.0399] sep=True F=1.5636 . | 2348 |
| tactical_2_6s:yaw_rate_mae_degps | +10.3505 [+6.7997, +14.3443] sep=True F=3.0812 COST | -1.5852 [-4.4947, +1.1252] sep=False F=3.0812 . | 2340 |
| tactical_2_6s:curvature_mae_1pm | +0.0591 [+0.0277, +0.0937] sep=True F=0.0301 COST | +0.0030 [-0.0265, +0.0292] sep=False F=0.0301 . | 2340 |

## TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)), n = 2399 windows / 141 episodes

| cell | seed 0 (U_s0 - B_s0): Delta [95 % interval] sep F verdict | seed 1 (U_s1 - B_s1) | n windows |
|---|---|---|---|
| TAC-LAT | -0.0504 [-0.0738, -0.0267] sep=True F=0.03 COST | -0.0292 [-0.0488, -0.0108] sep=True F=0.03 . | 2399 |
| TAC-LON | -0.0375 [-0.0755, +0.0017] sep=False F=0.0204 . | -0.0959 [-0.1379, -0.0567] sep=True F=0.0204 COST | 2399 |

## STRATEGIC (T0 at the window's first tick (--ages fresh)), n = 2399 windows / 141 episodes

| cell | seed 0 (U_s0 - B_s0): Delta [95 % interval] sep F verdict | seed 1 (U_s1 - B_s1) | n windows |
|---|---|---|---|
| RC-A:pos_err_m@30 | +0.1189 [+0.0429, +0.2004] sep=True F=0.0273 COST | +0.0294 [-0.0446, +0.1003] sep=False F=0.0273 . | 2322 |
| RC-A:pos_err_m@50 | +0.1623 [-0.0202, +0.3647] sep=False F=0.0121 . | -0.0297 [-0.2204, +0.1423] sep=False F=0.0121 . | 2319 |
| S1:pos_err_m@80 | +0.0744 [-0.4443, +0.6198] sep=False F=0.0451 . | -0.2211 [-0.7209, +0.1786] sep=False F=0.0451 . | 2294 |
| RC-A:head_err_deg@30 | +0.2842 [-0.0055, +0.5788] sep=False F=0.0517 . | -0.0622 [-0.5356, +0.2972] sep=False F=0.0517 . | 2322 |
| RC-A:head_err_deg@50 | +0.0115 [-0.5840, +0.6308] sep=False F=0.1747 . | -0.4452 [-1.1552, +0.1358] sep=False F=0.1747 . | 2319 |
| RC-A:head_err_deg@80 | -0.4254 [-1.8057, +0.8336] sep=False F=0.2306 . | -1.0166 [-2.6268, +0.1085] sep=False F=0.2306 . | 2294 |
| S2:skill@80 | +0.0615 [-0.0082, +0.1271] sep=False F=0.0041 . | +0.1025 [+0.0458, +0.1615] sep=True F=0.0041 GAIN | 244 |

## STRATEGIC members not in this card

{"detected_information": {"status": "n/a", "reason": "the recipe of record has no ctx head: no output"}, "derived_goals": {"status": "not in this arm"}, "strategic_intent_6_30s": {"status": "not computable", "reason": "no output beyond 80 m of arc"}}

## Cross-venue beside (seed 1, OBD-CTL_s1)

{}