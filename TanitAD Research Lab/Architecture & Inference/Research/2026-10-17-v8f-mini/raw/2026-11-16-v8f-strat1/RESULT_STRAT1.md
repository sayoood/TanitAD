# RESULT_STRAT1: the strategic layer in v8f-mini (route text in, detected information out)

Card: `PREREG_V8F_STRAT1.md` (frozen 2026-11-16, sha256 02c16b83...). Read: `F:/TanitAD-scratch/strat1/read/out/RESULT_STRAT1.json`, `STRAT1_STRATEGIC.json`, `STRAT1_CTX.json`, `strat1_read.log` (dev-box CPU, 2026-11-16, zero training). Evidence class: MEASURED here unless marked. 7 arms read: C0_s0, N1_s0, S1_s0, C0_s1, N1_s1, S1_s1, C0_rep (the floor replicate).

## Headline

* **Driving outcome: SEEDS-DISAGREE** (as written by the reader). No cell is a cost at both seeds in S1 - C0. Four cells are a cost at exactly one seed (seed 0 only): TAC-LAT, `operative_0_2s:dk_hw_abs_err_m`, `operative_0_2s:dk_tg_abs_err_s`, `tactical_2_6s:accel_mae_mps2`. No gain cell in S1 - C0 at either seed. Route setting: `route_cost` false (no ROUTE-COST).
* **STRATEGIC route following: S-PARTIAL** (as written): real - donor separated at both seeds, not above the best constant at 80 m. Detected information: country DETECTS + DETECTS-BEYOND-RIG, surface DETECTS, illumination DETECTS, period_of_day DETECTS, **weather NOT-DETECTED** (the card rule needs both seeds).
* **K9 (T1 tick 0 vs T0): FAILS as written on all 7 arms; EXPLAINED** on the smoke grid (declared T1 limit mode plus a float32 batch-layout floor). Full-grid confirmation K9b: PASS (section 3.1). The read wrote the T1 per-tick STRATEGIC row "not quoted" (its K9 ran as written); with K9b passing on the full grid the row is quotable with the limit-mode note. It is not extracted in this file.
* Estimator for every interval: `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0, over the eval-141 episodes. It answers "another draw of EPISODES" only. Floor F = |C0_rep - C0_s0| per cell: one replicate pair, so F is a weak floor (tiny on several cells, e.g. ADE 0.0003 m). Inference variance does not apply (deterministic planner).

## Committed next steps (from the card outcome table; NOT launched)

1. **SEEDS-DISAGREE: C0 and S1 at seeds 10 (init 0) and 11 (init 1)** = 4 pod arms, about 3 h (card estimate, ESTIMATED). A cell is a cost at >= 3 of 4 seeds with one per init. Awaits the PI go on pod time: the MM freeze ruling read "Build the arm" as covering the 7 arms only.
2. **S-PARTIAL: the 120 m checkpoint card** (6-30 s band). Needs the PI go. Card reason: inside 80 m the scene already shows the turn, so the nav value lies beyond the vision horizon.
3. **Weather NOT-DETECTED: zero-training probe on S1 frozen adapter** (attention pooling over all 40 tokens and 4 history frames, fit on train once, scored once on eval); if it detects, the next arm is that head, otherwise the representation lever (8x20 grid or 416x1024 encode). Seed 1 alone separates (+0.0717 [+0.0254,+0.1204]); seed 0 does not (+0.0463 [-0.0108,+0.1014]). Reported as written: NOT-DETECTED.
4. Work items from the criteria check (section 4).

Beside rows (no bar): N1 - C0 costs exist at seed 1 on speed_mae and along_mae (operative 0-2 s) and TAC-LAT; at seed 0 on dk_hw, dk_tg, tactical accel_mae and operative yaw_rate (lower CI bound +2e-06), with one gain, TAC-LON. These do not fire COSTS-NAV (no cell costs at both seeds). The N1 cost set differs from the S1 cost set, so the one-seed costs cannot yet be assigned to the heads or to the nav text.

## 1. Driving families (24 cells per seed; S1 - C0 and N1 - C0)

Tier: T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene), 2,399 windows / 141 episodes, per seed. Cell format: delta [CI lo, hi] flag. COST / gain = separated on the worse / better side AND |delta| > F; "sep<F" = separated but below the floor (neither gain nor cost); neutral otherwise. Error cells: a positive delta is worse. Accuracy cells (TAC-LAT, TAC-LON): a negative delta is worse. Seed 0 = M2b_s0 init, seed 1 = M2b_s1 init.

### PRIMARY (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene); n 2399 windows / 141 episodes)

| cell | F | S1-C0 seed 0 | S1-C0 seed 1 | N1-C0 seed 0 | N1-C0 seed 1 |
|---|---|---|---|---|---|
| operative_0_2s:ade_m | 0.000343 | +0.000169 [-0.0048,+0.0054] neutral | +0.003615 [-0.002,+0.0086] neutral | -0.0006027 [-0.0054,+0.0045] neutral | +0.004699 [-0.001,+0.0098] neutral |
| operative_0_2s:fde_m | 0.00271 | +0.0004701 [-0.0148,+0.0161] neutral | +0.0149 [-0.0026,+0.0319] neutral | -0.0007692 [-0.0156,+0.0147] neutral | +0.01727 [-0.0012,+0.0341] neutral |
| tactical_2_6s:ade_m | 0.0688 | +0.04269 [-0.0989,+0.1781] neutral | +0.07351 [-0.0769,+0.2053] neutral | +0.06772 [-0.0728,+0.2002] neutral | +0.1137 [-0.0417,+0.2456] neutral |
| tactical_2_6s:fde_m | 0.262 | +0.1525 [-0.1936,+0.4775] neutral | +0.1653 [-0.1955,+0.4728] neutral | +0.2254 [-0.1156,+0.5423] neutral | +0.251 [-0.1183,+0.5616] neutral |

### LONGITUDINAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene); n 2399 windows / 141 episodes)

| cell | F | S1-C0 seed 0 | S1-C0 seed 1 | N1-C0 seed 0 | N1-C0 seed 1 |
|---|---|---|---|---|---|
| operative_0_2s:speed_mae_mps | 0.00306 | -0.0004393 [-0.007,+0.0069] neutral | +0.005571 [-0.0009,+0.0126] neutral | -0.0006484 [-0.0072,+0.0062] neutral | +0.006956 [+0.0002,+0.0141] COST |
| operative_0_2s:along_mae_m | 0.000528 | -0.0005362 [-0.0052,+0.0046] neutral | +0.004666 [-0.0001,+0.0091] neutral | -0.0007689 [-0.0056,+0.0045] neutral | +0.005268 [+0.0001,+0.0098] COST |
| operative_0_2s:accel_mae_mps2 | 0.0225 | +0.007886 [-0.0019,+0.0198] neutral | +0.002211 [-0.006,+0.0117] neutral | +0.006401 [-0.0029,+0.0177] neutral | +0.007762 [-0.0011,+0.0175] neutral |
| operative_0_2s:dk_hw_abs_err_m | 0.00387 | +0.04238 [+0.0063,+0.0962] COST | +0.01972 [-0.0447,+0.0868] neutral | +0.03751 [+0.0054,+0.0834] COST | +0.008405 [-0.0548,+0.0727] neutral |
| operative_0_2s:dk_tg_abs_err_s | 0.015 | +0.01784 [+0.0018,+0.0477] COST | +0.003922 [-0.0038,+0.0116] neutral | +0.01739 [+0.0016,+0.0471] COST | -0.01005 [-0.0373,+0.0071] neutral |
| tactical_2_6s:speed_mae_mps | 0.0813 | +0.04428 [-0.0283,+0.1207] neutral | +0.01355 [-0.0598,+0.0828] neutral | +0.06273 [-0.0104,+0.1395] neutral | +0.006932 [-0.066,+0.0812] neutral |
| tactical_2_6s:along_mae_m | 0.122 | +0.01297 [-0.1046,+0.1269] neutral | +0.06698 [-0.0665,+0.1955] neutral | +0.02849 [-0.0864,+0.1428] neutral | +0.09035 [-0.0536,+0.2208] neutral |
| tactical_2_6s:accel_mae_mps2 | 0.178 | +0.2251 [+0.0507,+0.4276] COST | -0.04722 [-0.243,+0.1471] neutral | +0.183 [+0.043,+0.3333] COST | -0.06661 [-0.2578,+0.1144] neutral |
| tactical_2_6s:dk_hw_abs_err_m | 0.0962 | -0.07273 [-0.2798,+0.134] neutral | -0.0828 [-0.3039,+0.1095] neutral | -0.0327 [-0.218,+0.1439] neutral | -0.1222 [-0.2844,+0.0353] neutral |
| tactical_2_6s:dk_tg_abs_err_s | 0.0219 | -0.02108 [-0.056,+0.0104] neutral | -0.01433 [-0.0528,+0.022] neutral | -0.03484 [-0.095,+0.0108] neutral | -0.002018 [-0.0459,+0.0376] neutral |

### LATERAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene); n 2399 windows / 141 episodes)

| cell | F | S1-C0 seed 0 | S1-C0 seed 1 | N1-C0 seed 0 | N1-C0 seed 1 |
|---|---|---|---|---|---|
| operative_0_2s:cross_mae_m | 0.00036 | +0.001283 [-0.0007,+0.0035] neutral | -0.0007801 [-0.0039,+0.0023] neutral | +0.0005825 [-0.0013,+0.0026] neutral | +0.0001664 [-0.0029,+0.0034] neutral |
| operative_0_2s:heading_mae_deg | 0.017 | +0.02264 [-0.0066,+0.0558] neutral | -0.03197 [-0.0792,+0.0104] neutral | +0.02971 [-0.0019,+0.0637] neutral | -0.01472 [-0.0606,+0.0302] neutral |
| operative_0_2s:yaw_rate_mae_degps | 0.0385 | -0.006208 [-0.0718,+0.0679] neutral | -0.005283 [-0.1246,+0.1036] neutral | +0.07647 [+2e-06,+0.1577] COST | +0.07013 [-0.0371,+0.1996] neutral |
| operative_0_2s:curvature_mae_1pm | 0.00144 | -0.0004579 [-0.0031,+0.0021] neutral | +0.002078 [-0.0001,+0.0046] neutral | +0.001265 [-0.0004,+0.0032] neutral | +0.002358 [-0.0002,+0.0054] neutral |
| tactical_2_6s:cross_mae_m | 0.0547 | +0.03845 [-0.0424,+0.1214] neutral | -0.01573 [-0.0891,+0.0526] neutral | +0.04874 [-0.0355,+0.1295] neutral | +0.01372 [-0.0623,+0.0808] neutral |
| tactical_2_6s:heading_mae_deg | 1.57 | +0.8294 [+0.1618,+1.523] sep<F | +0.6039 [+0.0928,+1.182] sep<F | +0.7761 [+0.0904,+1.455] sep<F | +0.5412 [-0.0404,+1.179] neutral |
| tactical_2_6s:yaw_rate_mae_degps | 8.13 | +2.832 [+0.5433,+5.342] sep<F | +0.03547 [-1.975,+2.234] neutral | +1.767 [-0.1533,+3.885] neutral | +0.5285 [-2.596,+4.002] neutral |
| tactical_2_6s:curvature_mae_1pm | 0.0747 | +0.00815 [-0.007,+0.0238] neutral | +0.002094 [-0.0242,+0.0303] neutral | +0.0106 [-0.0049,+0.028] neutral | +0.01185 [-0.0229,+0.0503] neutral |

### TACTICAL (T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene); n 2399 windows / 141 episodes)

| cell | F | S1-C0 seed 0 | S1-C0 seed 1 | N1-C0 seed 0 | N1-C0 seed 1 |
|---|---|---|---|---|---|
| TAC-LAT | 0.00667 | -0.02251 [-0.0438,-0.0025] COST | -0.01501 [-0.0358,+0.0063] neutral | -0.01584 [-0.0329,+0.0012] neutral | -0.02334 [-0.045,-0.0037] COST |
| TAC-LON | 0.015 | +0.02334 [-0.0013,+0.0487] neutral | +0.006669 [-0.0204,+0.0371] neutral | +0.02751 [+0.0042,+0.053] gain | -0.006669 [-0.0312,+0.02] neutral |

### Per-family reading
* LONGITUDINAL (incl. distance-keeping): 10 cells. S1 - C0: 3 cost cells at seed 0 (dk_hw_abs_err_m and dk_tg_abs_err_s at 0-2 s, accel_mae at 2-6 s), none at seed 1. N1 - C0 costs the same three cells at seed 0, so the text nav, not the heads, is the common factor there at that seed.
* LATERAL (incl. curvature and yaw-rate): 8 cells. S1 - C0: no cost or gain at either seed. Two 2-6 s cells (heading, yaw-rate) are separated at seed 0 but below F, so they are not costs under the card rule.
* TACTICAL: TAC-LAT is a cost at seed 0 for S1 - C0 (-0.0225 [-0.0438,-0.0025]) and at seed 1 for N1 - C0.
* PRIMARY (ADE/FDE): 4 cells, all neutral in all four pairs. ADE is one row of four families here, not the result.

## 2. STRATEGIC

### 2.1 Route setting vs RC-A (T0 real nav, fresh ages; 6 cells per seed)
S1 - C0 position and heading error at 30/50/80 m, seed 0: every cell is above F and none is separated (e.g. pos_err@80 +1.4643 m [-0.163,+3.2973], n 2,294 windows / 136 episodes). No cost under the card rule, so no ROUTE-COST. Card caveat: the C0 side is echo-prone (v9 nav shares the ego future with RC-A). Seed-1 rows are in `STRAT1_STRATEGIC.json` (`route_setting_RC-A.rows`); UNVERIFIED here (not extracted into this report).

### 2.2 Route following (S_CKPT, 80 m deciding arc; T0 at the first tick, COMPLETE_M 20, theta 25 deg)
Outcome **S-PARTIAL**. Level at 80 m, best constant (majority side) 0.766, n_scorable 244 windows / 25 episodes:

| arm | real | donor | zero | real - donor [CI] | real - best constant [CI] |
|---|---|---|---|---|---|
| C0_s0 | 0.734 | 0.205 | 0.180 | +0.529 [0.377,0.652] sep | -0.033 [-0.256,0.206] no |
| N1_s0 | 0.848 | 0.213 | 0.193 | +0.635 [0.539,0.725] sep | +0.082 [-0.104,0.301] no |
| S1_s0 | 0.844 | 0.197 | 0.180 | +0.648 [0.552,0.734] sep | +0.078 [-0.110,0.298] no |
| C0_s1 | 0.750 | 0.156 | 0.143 | +0.594 [0.442,0.714] sep | -0.016 [-0.238,0.208] no |
| N1_s1 | 0.857 | 0.205 | 0.189 | +0.652 [0.546,0.738] sep | +0.090 [-0.085,0.292] no |
| S1_s1 | 0.857 | 0.205 | 0.197 | +0.652 [0.545,0.737] sep | +0.090 [-0.083,0.291] no |

S1 - C0 real-nav level: +0.111 [0.014,0.244] separated at seed 0; +0.107 [0.000,0.249] not separated at seed 1. The arm follows the text (real - donor large at every arm), but at 80 m the always-majority-side answer is within CI. The reader n at 80 m (244 / 25) differs from the card INHERITED 214 / 24 and best constant 0.794; the reader value is the MEASURED one. Controls K7, K8 in section 3.

### 2.3 Detected information (S1, T0 at the window NOW frame; 2,399 windows / 141 episodes)
Constants are on the eval-window basis (slightly different from the card step basis 0.846 / 0.785 / 0.595).

| variable | S1_s0 acc | model - constant, s0 [CI] | S1_s1 acc | model - constant, s1 [CI] | label |
|---|---|---|---|---|---|
| country | 0.494 | +0.402 [0.340,0.471] | 0.537 | +0.445 [0.383,0.513] | DETECTS + BEYOND-RIG (vs rig 0.078: +0.416 [0.342,0.491] / +0.459 [0.383,0.538]) |
| weather | 0.904 | +0.046 [-0.011,0.101] not sep | 0.929 | +0.072 [0.025,0.120] sep | NOT-DETECTED (needs both seeds) |
| surface | 0.922 | +0.125 [0.063,0.187] | 0.921 | +0.124 [0.067,0.179] | DETECTS |
| illumination | 0.966 | +0.360 [0.290,0.435] | 0.963 | +0.357 [0.284,0.436] | DETECTS (n 2,359 windows, twilight excluded) |
| period_of_day | 0.479 | +0.238 [0.142,0.330] | 0.496 | +0.255 [0.178,0.337] | DETECTS |

Country within platform stratum, s0: platform 0 acc 0.475 vs constant 0.134; platform 1 0.511 vs 0.095. Per-class recall and confusions are in `STRAT1_CTX.json`. Weather balanced accuracy 0.708 / 0.671: minority classes remain weak. Label ceiling (card, INHERITED): weather 36/40, surface 32/40, illumination 30/39 against the PI frames; no number here is accuracy against the world.
Smoke findings carried and confirmed in the full read: twilight is 240 steps in 5 clips (the card text said 1 clip) and is excluded from illumination; illumination baseline 0.6049 masked (step basis) vs 0.5947 (report basis), 0.6062 on windows. C0 and N1: n/a, reason "no output".

### 2.4 Not in this arm (reported, not omitted)
Derived goals: not in this arm (later card; goal-input rule). Strategic intent 6-30 s: not computable (no output beyond 80 m of arc; would-have n 32 at 6-12 s, 69 at 12-30 s, 20 above 30 s, INHERITED). Visibility: not built (label 25/40 below the constant 31/40). No strategic decision-quality row beyond route following and route setting.

## 3. Controls K1-K11, pass/fail as written

| id | control | result |
|---|---|---|
| K1 | self-pair C0_s0 | PASS: delta 0.0, CI [0,0] |
| K2 | constant predictor, detected-info reader | PASS on all 5 variables |
| K3 | planted perfect predictor | PASS: 1.0 and DETECTS on all 5 |
| K4 | C-RAND (S1 ctx_head re-initialised) | PASS: NOT DETECTS on all 5 (accuracy 0.07-0.19), seeds 0 and 1 |
| K5 | clip-shuffled labels | PASS: NOT DETECTS on all 5 |
| K6 | rig-only country | PASS: reads 0.0779 = rig baseline; NOT DETECTS-BEYOND-RIG |
| K7 | S_CKPT hold / x40 deg / mirrored | PASS: 0.0 / 1.0 / 0.0 |
| K8 | donor on a copy-the-nav synthetic arm | PASS: real 1.0, donor 0.1066 = expected 0.1066 |
| K9 | T1 tick-0 S output vs T0, atol 1e-6 | **FAIL as written on all 7 arms** (max abs diff, m: C0_s0 21.04, N1_s0 101.79, S1_s0 101.74, C0_s1 11.67, N1_s1 75.70, S1_s1 93.78, C0_rep 21.96). EXPLAINED, see 3.1 |
| K10 | trainer mutations | reader-side: `MUTATIONS.json`, each named test went RED (MEASURED). Trainer-side mutations (BUILD_SPEC section 9) belong to the build, not re-run here: UNVERIFIED |
| K11 | G-CLOCK-NAV, sign flipped | belongs to the build, not run here: UNVERIFIED |

### 3.1 K9 diagnosis (smoke grid, 6 episodes / 48 windows: a code check, never a result)
* Mechanism: T1 dumps run `--limit causal` at batch 32; T0 feeds the logged limit at batch 64. The strategic brain `run_s` takes a speed-limit token (`v8f_mini.py:978` at the read tree). `k9diag.json`: `--limit causal` max 10.58 m (S1_s0), 4.40 m (C0_s0); `--limit logged` max 3.81e-06 m on both (7/48 and 10/48 windows over 1e-6).
* The `k9diag.py` criterion read the logged residual against 1e-6, so its own verdict line says "NOT EXPLAINED". Its follow-up `k9diag2.py` (criteria committed before running) tested batch layout: T0 at batch 32 vs batch 64 on the same windows and checkpoint = 3.81e-06 m (S1_s0 and C0_s0), the same size as the residual; T1 `--limit logged --batch 64` vs T0 batch 64 = exactly 0.0 on 48/48 windows at both arms. Verdict: **EXPLAINED: declared limit mode plus a float32 batch-layout floor.**
* K9 still FAILS as written. The T1 per-tick STRATEGIC row is not quoted until K9b passes.
* **K9b full grid (2,399 windows, S1_s0 and S1_s1, `--limit logged --batch 64` vs the read T0 arrays, bar <= 1e-6 m): PASS** (`k9diag/k9full.py`, filled by the MM). K9B_RESULT: PASS: S1_s0 max abs diff 0.0 m over 2399 windows; S1_s1 max abs diff 0.0 m over 2399 windows (MEASURED, `k9diag/k9full.json`). The T1 per-tick STRATEGIC row of the causal dumps is quotable with the limit-mode note; the read wrote it "not quoted" because its K9 ran as written.

## 4. Criteria check (TanitAD_BenchmarkCriteria) and work items
`criteria_check.py` returns **UNKNOWN_SCOPE** for this artifact (registry 2.10.3 has no marker for the `families.<F>.cells` schema): it is not scored and not counted compliant. Report and manual diff: `BENCHMARK_CHECK_STRAT1.md`. Work items:
1. Registry: add an in_scope marker and per-cell mapping for the STRAT1 schema, with a deliberate-regression arm in `tools/tests/test_criteria_check.py` (EvalFlyWheel).
2. Absent from this read, to implement or refuse with reason and n: long.progress; TTC and the DK-VIOL cell (card 4.1 lists it); tac.confusion; tac.goal_selection (goal macro-F1, anchor selection); T own decision F1 and top-3 reads; the 4.3 nav-echo split (real, donor, zero nav by announced and unannounced windows); TACTICAL nav-compliance and its shuffle and zero controls (the S_CKPT donor and zero rows cover STRATEGIC only). UNVERIFIED whether the dumps hold them.
3. Tier stamp is the descriptive string "T1-kin" per family; the registry tier paths expect T0/T1/T2 values.
4. Seed-1 route-setting rows not extracted here.
5. Not-built rows: derived goals, strategic intent 6-30 s, visibility.

## 5. Provenance
Reader code and pins: `FILES_TO_LAND.txt`. Checkpoint and input sha256: `RESULT_STRAT1.json` `provenance`. Reader tests: 20 passed (`strat1_read.log`). Dumps 2026-11-16 20:02-20:14.
