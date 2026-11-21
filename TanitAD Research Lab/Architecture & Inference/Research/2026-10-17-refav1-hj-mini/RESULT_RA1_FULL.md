# RESULT — RA-1 on the full corpus (`PREREG_RA1_FULL.md`)

**Status:** FINAL -- outcome **K_PASS**.  
**Tier:** T1-class open-loop trajectory (eval-141 = 2,399 windows / 141 episodes; holdout = 60 clips, 4,076 windows).  
**Estimator:** paired episode-cluster bootstrap (`taniteval.ci`), B = 2,000 — it answers *another draw of EPISODES*; training-seed variance is answered only by the floor F = |s0 − s1| (two seeds, so F is itself one draw); the readout is deterministic (no inference seed).  
**Evidence class:** MEASURED — box 2, code at tip `d568e539f03f72de0fabe2893eeb6774743f0dd8`; train list sha256 `37f4f9ee468e…`, holdout `f6a1dee02af0…`, holdout P10-yaw-4512 ADE@6s 3.0289.  
**Generated** by `code/ra1_full_report.py` from `raw/ra1_full/` — every reading is copied from `ra1_full_verdict.py`'s JSON, never re-decided here.

## The readings (pre-registered order)

* **Selection (holdout only):** S*_R0 = **3000**, S*_A1 = **3000**; edge flags {'R0': None, 'A1': None}; K arm = **A1** (tie ≤ 0.01 m → R0). `full1.json` reading: S*_A1 = 3000, S*_R0 = 3000; K arm = A1. Phase 2: A1x seed 0 at S*_A1, then the gate and K.
* **Phase 2 — reading 2, the A1x gate at S*_A1:** A1x − A1 eval-141 ADE@6s +0.0376 [+0.0119, +0.0643] **sep** → `OK: A1x loses to A1, separated`.
* **Phase 2 — reading 3, K:** bar: K-arm - P10-yaw-4512 ADE@6s < 0, separated at both seeds, |delta| > 2x |A1_s0 - A1_s1|. K arm **A1** → **PASS**; training-seed floor (A1) 0.010622104297989488.

  | arm | seed | ADE@6s | P10-yaw-4512 | arm − P10 [95 % CI] |
  |---|---|---|---|---|
  | K: A1 | 0 | 2.2005 | 2.4870 | -0.2865 [-0.3936, -0.1790] **sep** |
  | K: A1 | 1 | 2.2111 | 2.4870 | -0.2758 [-0.3756, -0.1799] **sep** |
  | other (descriptive) | 0 | 2.2843 | 2.4870 | -0.2027 [-0.3258, -0.0809] **sep** |
  | other (descriptive) | 1 | 2.3302 | 2.4870 | -0.1568 [-0.2744, -0.0413] **sep** |

* **Phase 2 outcome `K_PASS`:** READING 4 PASS: the gate holds at S*_A1 = 3000 and A1 beats P10-yaw-4512 at both seeds. Data scale is the lever. Next: A2 and SENS (with A0) at S* on the full corpus, under PREREG_HJ_STAGE0's other bars.

## The author's reading — written AFTER the data, by hand; NOT generated and NOT a pre-registered reading

**Verdict (MEASURED, pre-registered, T1-class open loop): K PASSES.**
* A1 (H-JEPA + IDM, b0.08g50) at 3,000 steps on 4,512 clips beats **P10-yaw-4512**, the prior refit on the same data, at 6 s:
  * seed 0: **−0.2865 m** [−0.3936, −0.1790];
  * seed 1: **−0.2758 m** [−0.3756, −0.1799].
* That is **11.5 % / 11.1 %** of P10's 2.4870 m. Both are separated, at **13.5× / 13.0×** the 2F bar (2F = 0.0212 m).
* The gate (A1x loses to A1) holds on eval-141, as pre-registered.
* This reverses the 540-clip result. There, every compliant arm sat **above** P10-yaw-540 (A1 +0.9476 / +0.7988) and the gate failed (`D-RA1-YAW-VERDICT`).

**Data scale is the lever: measured on the same windows, not inferred.**
* Same arm, budget and seed, full − 540-clip, eval-141, paired:
  * A1 @ 1000: −0.2690 / −0.2705;
  * R0 @ 1000: −0.5596 / −0.6161.
* All four are separated and beyond 2F (table below).
* Memorisation moved later with the data, as H-RA1-FULL-1 predicted:
  * at 540 clips, R0's best budget was 250 steps;
  * here it is 3,000, and at 6,000 both arms lose their margin over P10 (holdout, not separated).

**What this result does NOT yet show. Each point is disclosed; none moves a bar.**
1. **The time-order share of A1's gain rests on ONE A1x seed and ONE window set.**
   * On eval-141:
     * A1x − A1 = **+0.0376** [+0.0119, +0.0643] (the gate);
     * A1x − R0 = −0.0461 [−0.1193, +0.0257], not separated.
   * So roughly half of A1's edge over R0 needs true time order, and the rest comes from the auxiliary training as such.
   * On the 60-clip holdout, A1x − A1 = **−0.0029** [−0.0534, +0.0526]: the holdout neither confirms nor refutes the +0.0376.
   * The bootstrap answers *another draw of episodes*, never *another training run* (`H-ESTIM-SEED-1`). ⇒ The cheapest discriminating run is **A1x seed 1**.
2. **A1 − R0 is beyond 2F at one seed, not both.**
   * eval-141: −0.0837 (s0) and −0.1190 (s1), both separated.
   * R0's seed floor at 3,000 steps is 0.0459, so 2F = 0.0918, and s0 sits inside it.
   * ⇒ "The world-model objective improves the readout" is supported at seed 1 only, under the 2F rule.
3. **R0 alone also passes K's bar** (−0.2027 / −0.1568, separated, beyond its own 2F).
   * With 4,512 clips, the frozen pooled DINOv3 readout beats the refit prior even without the world model.
   * The world model's contribution is the A1 − R0 margin above, not the whole 0.28 m.
4. **The holdout P10 here is 3.0289, on 4,076 windows (stride 1).**
   * The register's 3.0682 is on PREREG_RA1_FULL's 540-window holdout table.
   * Different window sets, not a discrepancy.
5. **The four families are descriptive, with single-arm intervals only** (no paired family delta exists yet; work item).
   * A1's clearest margins over P10 are **LATERAL**:
     * yaw-rate MAE 3.42 / 3.41 vs 5.23 °/s;
     * curvature 0.0172 / 0.0166 vs 0.0280 1/m;
     * cross-track 0.91 / 0.91 vs 1.19 m.
   * **TACTICAL:**
     * lateral-decision κ 0.558 / 0.571 vs 0.463;
     * 5-way κ 0.415 / 0.426 vs 0.342;
     * the paired turn stratum is −0.73 / −0.80 m, separated at both seeds;
     * stop and brake are separated at seed 0 only.
   * **LONGITUDINAL** margins are small (speed MAE 0.941 / 0.940 vs 0.980 m/s). They are **inadmissible as capability claims**, because the v0 anti-echo control is undischarged.
   * **STRATEGIC** is PAUSED (PI ruling 2026-09-19). Distance-keeping is NOT COMPUTABLE (no lead track in this cache).
6. **COST (not a PREREG_RA1_FULL reading):** L2 − L1 is negative and separated at every A1 budget (3,000: −0.1056 / −0.1004). PREREG_HJ_STAGE0's COST bar (L2 − L1 > 0) would read **FAIL**.
7. **SENS is NOT READ.**
   * A1's level-1 yaw-rate sensitivity (k = 10) is **0.759 / 0.772**, 7.6× the 0.10 level, at both seeds.
   * But "separated above A0" needs A0, which this design did not run at full scale.

**Next lever (Rule Zero).**
* PREREG_HJ_STAGE0's branch rule makes **SENS the decider** now that K passes:
  * SENS PASS + K PASS ⇒ frozen + adapter suffices ⇒ build refav2 (RA-2);
  * SENS FAIL ⇒ raise γ to 200, then IDM at level 2 only.
* The run set is **A0 ×2** (SENS), **A2 ×2** (K's best-of includes the 3-level arm), and **A1x seed 1** (the gate's training-seed replicate), at 3,000 steps on the same cache. That is about 4.2 h on box 2.
* It is pre-registered as `PREREG_RA1_FULL` Amendment 1. Box 2 is queued with v8F-mini and then rung β, so the slot after β is a **PI decision**.


## Reading 1 (descriptive) and the selection grid — holdout selects, eval-141 never does

| arm | steps | holdout ADE@6s s0 / s1 | seed mean | mean − P10-4512 | F_hold | holdout Δ vs P10 s0 | s1 | eval-141 Δ vs P10 s0 (NON-DECIDING) | s1 | F_141 |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 500 | 2.8657 / 2.8761 | 2.8709 | -0.1580 | 0.0104 | -0.1632 [-0.2472, -0.0876] **sep** | -0.1528 [-0.2442, -0.0729] **sep** | -0.1049 [-0.1418, -0.0693] **sep** | -0.0894 [-0.1244, -0.0551] **sep** | 0.0155 |
| A1 | 1000 | 2.7731 / 2.7852 | 2.7792 | -0.2498 | 0.0121 | -0.2559 [-0.3581, -0.1597] **sep** | -0.2437 [-0.3402, -0.1528] **sep** | -0.1843 [-0.2311, -0.1384] **sep** | -0.1775 [-0.2292, -0.1264] **sep** | 0.0069 |
| A1 | 3000 ⭐S* | 2.6531 / 2.6463 | 2.6497 | -0.3792 | 0.0068 | -0.3758 [-0.5587, -0.2003] **sep** | -0.3826 [-0.5331, -0.2306] **sep** | -0.2865 [-0.3936, -0.1790] **sep** | -0.2758 [-0.3756, -0.1799] **sep** | 0.0106 |
| A1 | 6000 | 2.9511 / 2.8763 | 2.9137 | -0.1152 | 0.0748 | -0.0779 [-0.3947, +0.2524] | -0.1526 [-0.3964, +0.1091] | +0.0420 [-0.1405, +0.2413] | +0.0472 [-0.1079, +0.2061] | 0.0052 |
| R0 | 500 | 2.8782 / 2.8656 | 2.8719 | -0.1570 | 0.0126 | -0.1508 [-0.2394, -0.0773] **sep** | -0.1633 [-0.2652, -0.0778] **sep** | -0.0818 [-0.1288, -0.0379] **sep** | -0.0588 [-0.1082, -0.0092] **sep** | 0.0230 |
| R0 | 1000 | 2.8029 / 2.8357 | 2.8193 | -0.2096 | 0.0328 | -0.2260 [-0.3420, -0.1152] **sep** | -0.1932 [-0.3047, -0.0802] **sep** | -0.1640 [-0.2237, -0.1083] **sep** | -0.1616 [-0.2207, -0.1057] **sep** | 0.0024 |
| R0 | 3000 ⭐S* | 2.7301 / 2.7190 | 2.7245 | -0.3044 | 0.0111 | -0.2988 [-0.5027, -0.0872] **sep** | -0.3100 [-0.5009, -0.1001] **sep** | -0.2027 [-0.3258, -0.0809] **sep** | -0.1568 [-0.2744, -0.0413] **sep** | 0.0459 |
| R0 | 6000 | 2.9719 / 2.9693 | 2.9706 | -0.0583 | 0.0026 | -0.0570 [-0.3203, +0.2364] | -0.0596 [-0.3299, +0.2206] | +0.0215 [-0.1368, +0.1752] | +0.0150 [-0.1213, +0.1546] | 0.0065 |

F = |s0 − s1| (training-seed floor, one draw). `sep` = the paired episode CI excludes 0. A negative Δ = the arm beats its prior.

## Where A1's gain over R0 comes from (descriptive) — R0 → A1x → A1 at 3000 steps

A1x is A1 with its prediction/IDM targets TIME-SHUFFLED (the gate's control). So A1x − R0 is what the auxiliary training buys without true time order, and A1x − A1 is what the true order adds (the gate's own quantity; the gate reads it on eval-141 only). A1x has ONE training seed: its rows answer *another draw of episodes*, not *another training run*. Paired over episodes on the same windows (asserted).

| difference | seeds | eval-141 ADE@6s | holdout ADE@6s |
|---|---|---|---|
| A1 − R0 | A1 s0 − R0 s0 | -0.0837 [-0.1587, -0.0135] **sep** | -0.0770 [-0.2080, +0.0384] |
| A1 − R0 | A1 s1 − R0 s1 | -0.1190 [-0.1858, -0.0590] **sep** | -0.0727 [-0.2129, +0.0551] |
| A1x − R0 | A1x s0 − R0 s0 | -0.0461 [-0.1193, +0.0257] | -0.0799 [-0.2140, +0.0375] |
| A1x − A1 (the gate's quantity) | A1x s0 − A1 s0 | +0.0376 [+0.0119, +0.0643] **sep** | -0.0029 [-0.0534, +0.0526] |

## The data-scale effect (descriptive, PREREG "Reported, NOT read") — full (4,512 clips) − 540-clip run

Same arm, budget and training seed; same windows (asserted); paired over episodes. A cross-scale Δ is also a cross-RUN Δ, so it is flagged **> 2F** only when |Δ| exceeds twice the larger training-seed floor of the two scales (eval-141 floors).

| arm | steps | seed | eval-141 ADE@6s full − 540 | holdout ADE@6s full − 540 | 2F (eval-141) | beyond 2F (eval-141) |
|---|---|---|---|---|---|---|
| R0 | 500 | 0 | -0.2403 [-0.3088, -0.1767] **sep** | -0.1877 [-0.3169, -0.0576] **sep** | 0.1072 | yes |
| R0 | 500 | 1 | -0.2708 [-0.3523, -0.1940] **sep** | -0.2430 [-0.3885, -0.1179] **sep** | 0.1072 | yes |
| A1 | 500 | 0 | -0.0951 [-0.1517, -0.0433] **sep** | -0.1027 [-0.1733, -0.0371] **sep** | 0.0310 | yes |
| A1 | 500 | 1 | -0.0891 [-0.1417, -0.0403] **sep** | -0.0806 [-0.1610, -0.0123] **sep** | 0.0310 | yes |
| R0 | 1000 | 0 | -0.5596 [-0.6789, -0.4445] **sep** | -0.4443 [-0.6238, -0.2794] **sep** | 0.1176 | yes |
| R0 | 1000 | 1 | -0.6161 [-0.7524, -0.4847] **sep** | -0.5675 [-0.7945, -0.3675] **sep** | 0.1176 | yes |
| A1 | 1000 | 0 | -0.2690 [-0.3483, -0.1912] **sep** | -0.2860 [-0.4152, -0.1748] **sep** | 0.0168 | yes |
| A1 | 1000 | 1 | -0.2705 [-0.3443, -0.1977] **sep** | -0.1948 [-0.3229, -0.0916] **sep** | 0.0168 | yes |

Budgets on one scale only (no matched row): [250, 2000, 3000, 6000].

## P10-yaw-540 deltas (descriptive) — full run − the 540-clip prior, same windows

P10-yaw-540 read from `st250/R0_s0` (identical across all 14 banked 540-clip runs).

| arm | steps | seed | eval-141 arm − P10-yaw-540 | holdout arm − P10-yaw-540 |
|---|---|---|---|---|
| A1 | 500 | 0 | -0.1806 [-0.2407, -0.1239] **sep** | -0.1587 [-0.2212, -0.0985] **sep** |
| A1 | 500 | 1 | -0.1652 [-0.2215, -0.1105] **sep** | -0.1484 [-0.2152, -0.0867] **sep** |
| A1 | 1000 | 0 | -0.2601 [-0.3278, -0.1925] **sep** | -0.2514 [-0.3299, -0.1759] **sep** |
| A1 | 1000 | 1 | -0.2532 [-0.3251, -0.1863] **sep** | -0.2393 [-0.3211, -0.1621] **sep** |
| A1 | 3000 | 0 | -0.3622 [-0.4840, -0.2382] **sep** | -0.3714 [-0.5580, -0.2027] **sep** |
| A1 | 3000 | 1 | -0.3516 [-0.4698, -0.2351] **sep** | -0.3782 [-0.5295, -0.2301] **sep** |
| A1 | 6000 | 0 | -0.0338 [-0.2270, +0.1719] | -0.0734 [-0.3988, +0.2634] |
| A1 | 6000 | 1 | -0.0286 [-0.1971, +0.1431] | -0.1482 [-0.3918, +0.1185] |
| R0 | 500 | 0 | -0.1576 [-0.2183, -0.0971] **sep** | -0.1463 [-0.2349, -0.0610] **sep** |
| R0 | 500 | 1 | -0.1345 [-0.1975, -0.0705] **sep** | -0.1589 [-0.2429, -0.0823] **sep** |
| R0 | 1000 | 0 | -0.2397 [-0.3266, -0.1587] **sep** | -0.2216 [-0.3325, -0.1080] **sep** |
| R0 | 1000 | 1 | -0.2374 [-0.3207, -0.1568] **sep** | -0.1888 [-0.3019, -0.0800] **sep** |
| R0 | 3000 | 0 | -0.2785 [-0.4218, -0.1419] **sep** | -0.2944 [-0.5066, -0.0674] **sep** |
| R0 | 3000 | 1 | -0.2326 [-0.3626, -0.1008] **sep** | -0.3055 [-0.5165, -0.0898] **sep** |
| R0 | 6000 | 0 | -0.0542 [-0.2267, +0.1192] | -0.0526 [-0.3236, +0.2442] |
| R0 | 6000 | 1 | -0.0607 [-0.2130, +0.0995] | -0.0552 [-0.3293, +0.2417] |

## The four metric families (reported, NOT read) — T1-class open loop, eval-141

Arm columns and the P10-yaw-4512 column carry **single-arm** episode-cluster intervals: the eval publishes no paired per-family delta (`perwindow.npz` holds ADE only), so **no arm − P10 family difference is claimed here** — pairing them is a WORK ITEM (the eval must save per-window family components). The strata rows ARE paired.

### A1 @ 3000 steps (the K arm)

| family | metric | A1 s0 | A1 s1 | P10-yaw-4512 |
|---|---|---|---|---|
| LONGITUDINAL | speed_mae_mps | 0.9407 [0.8598, 1.0229] | 0.9402 [0.8575, 1.0237] | 0.9802 [0.8992, 1.0602] |
| LONGITUDINAL | speed_bias_mps | -0.0243 [-0.1319, 0.0814] | 0.0490 [-0.0541, 0.1527] | -0.0508 [-0.1513, 0.0474] |
| LONGITUDINAL | speed_rmse_mps | 1.5964 [1.4550, 1.7347] | 1.5994 [1.4565, 1.7388] | 1.5982 [1.4677, 1.7248] |
| LONGITUDINAL | target_speed_acc.within_0.5_mps | 0.5230 [0.4964, 0.5512] | 0.5237 [0.4983, 0.5515] | 0.4906 [0.4674, 0.5165] |
| LONGITUDINAL | target_speed_acc.within_1.0_mps | 0.6962 [0.6713, 0.7226] | 0.6982 [0.6730, 0.7238] | 0.6751 [0.6494, 0.7028] |
| LONGITUDINAL | target_speed_acc.within_2.0_mps | 0.8572 [0.8377, 0.8771] | 0.8596 [0.8399, 0.8790] | 0.8507 [0.8316, 0.8712] |
| LONGITUDINAL | along_mae_m | 1.7272 [1.5807, 1.8787] | 1.7442 [1.5922, 1.8991] | 1.8217 [1.6691, 1.9729] |
| LONGITUDINAL | accel_mae_mps2 | 0.5252 [0.4770, 0.5758] | 0.5242 [0.4755, 0.5761] | 0.6376 [0.5888, 0.6897] |
| LONGITUDINAL | ego_progress.progress_ratio_mean | 1.0364 [0.9964, 1.0846] | 1.0631 [1.0212, 1.1130] | 1.0915 [1.0416, 1.1523] |
| LONGITUDINAL | ego_progress.under_progress_rate | 0.5391 [0.4960, 0.5803] | 0.5188 [0.4787, 0.5582] | 0.6003 [0.5628, 0.6382] |
| LATERAL | heading_mae_deg | 4.5386 [3.4997, 5.8750] | 4.5642 [3.4822, 5.9864] | 5.4747 [4.3139, 6.9261] |
| LATERAL | yaw_rate_mae_degps | 3.4237 [2.8611, 4.0755] | 3.4079 [2.8458, 4.0546] | 5.2304 [4.4367, 6.1219] |
| LATERAL | curvature_mae_1pm | 0.0172 [0.0128, 0.0219] | 0.0166 [0.0127, 0.0212] | 0.0280 [0.0211, 0.0360] |
| LATERAL | cross_mae_m | 0.9149 [0.7749, 1.0679] | 0.9132 [0.7777, 1.0597] | 1.1858 [0.9970, 1.3895] |
| LATERAL | cross_final_mae_m | 2.9058 [2.4327, 3.4280] | 2.8927 [2.4297, 3.3886] | 3.7299 [3.0992, 4.4113] |
| TACTICAL | lateral_decision.accuracy | 0.8016 [0.7656, 0.8341] | 0.8074 [0.7730, 0.8399] | 0.7728 [0.7326, 0.8098] |
| TACTICAL | lateral_decision.kappa | 0.5577 [0.4846, 0.6240] | 0.5714 [0.5018, 0.6369] | 0.4632 [0.3940, 0.5303] |
| TACTICAL | longitudinal_decision.accuracy | 0.5723 [0.5377, 0.6078] | 0.5761 [0.5433, 0.6118] | 0.5336 [0.4987, 0.5683] |
| TACTICAL | longitudinal_decision.kappa | 0.3436 [0.2899, 0.3950] | 0.3486 [0.2977, 0.3993] | 0.2911 [0.2357, 0.3425] |
| TACTICAL | maneuver_5way_collapsed.accuracy | 0.5506 [0.5144, 0.5876] | 0.5590 [0.5250, 0.5968] | 0.4915 [0.4539, 0.5292] |
| TACTICAL | maneuver_5way_collapsed.kappa | 0.4146 [0.3663, 0.4615] | 0.4257 [0.3819, 0.4724] | 0.3415 [0.2935, 0.3868] |
| TACTICAL | goal_setting.goal_bearing_mae_deg (point) | 4.8847 | 4.5750 | 5.0976 |
| TACTICAL | goal_setting.goal_range_ratio (point) | 1.0736 | 1.1069 | 1.1307 |
| TACTICAL | goal_setting.goal_long_bias_m (point) | 0.0202 | 0.4310 | 0.3175 |
| TACTICAL | goal_setting.goal_lat_bias_m (point) | -0.0899 | 0.0644 | 0.1690 |
| TACTICAL stratum | turn: ADE@6s arm − P10 (PAIRED; n 443 win / 51 ep) | -0.7315 [-1.0719, -0.3695] **sep** | -0.8030 [-1.1198, -0.4589] **sep** | — |
| TACTICAL stratum | stop: ADE@6s arm − P10 (PAIRED; n 101 win / 22 ep) | -0.4852 [-0.8883, -0.0659] **sep** | +0.0001 [-0.4322, +0.4583] | — |
| TACTICAL stratum | brake: ADE@6s arm − P10 (PAIRED; n 201 win / 45 ep) | -0.3512 [-0.6717, -0.0063] **sep** | -0.0128 [-0.3319, +0.3301] | — |
| STRATEGIC | — | **PAUSED** | **PAUSED** | — |
| LONGITUDINAL | distance_keeping | **UNAVAILABLE** (n 0) | same | — |

* **STRATEGIC:** PAUSED — PI 2026-09-19, PI_DECISION_QUEUE.md ITEM 25: the strategic layer is switched off temporarily until tactical and operative drive with high quality. Reported as PAUSED -- not a pass, not omitted. Underlying: UNAVAILABLE (strategic decisions not present in the scored pass (missing ['route_pred', 'route_gt']). A world-model FIDELITY pass does not traverse the hierarchy — run_one p…). Never a pass, never omitted.
* **Distance-keeping / lead braking:** UNAVAILABLE, n = 0 — NOT COMPUTABLE from the pooled cache: it carries no lead-agent track (needs the v9 join lead_gap); n = 2399 windows would have been scored
* **Longitudinal admissibility:** `_longitudinal_claim_admissible` = False. The v0 anti-echo control is UNDISCHARGED (no v0 in the window dict), so every LONGITUDINAL number above is a fidelity diagnostic, **not** a longitudinal capability result.

## COST and SENS (not PREREG_RA1_FULL readings)

* **SENS:** not in this design (it needs A0).
* **COST** (T0; bar from PREREG_HJ_STAGE0: L2 − L1 > 0), A1 runs, reported only:

| steps | seed | L2 − L1 native | L2 − L1 matched |
|---|---|---|---|
| 500 | 0 | -0.0817 [-0.0899, -0.0737] **sep** | -0.0443 [-0.0508, -0.0378] **sep** |
| 500 | 1 | -0.0809 [-0.0899, -0.0715] **sep** | -0.0434 [-0.0519, -0.0350] **sep** |
| 1000 | 0 | -0.0902 [-0.1003, -0.0810] **sep** | -0.0526 [-0.0611, -0.0439] **sep** |
| 1000 | 1 | -0.0786 [-0.0875, -0.0699] **sep** | -0.0417 [-0.0503, -0.0335] **sep** |
| 3000 | 0 | -0.1056 [-0.1154, -0.0957] **sep** | -0.0656 [-0.0745, -0.0565] **sep** |
| 3000 | 1 | -0.1004 [-0.1112, -0.0896] **sep** | -0.0622 [-0.0724, -0.0515] **sep** |
| 6000 | 0 | -0.1282 [-0.1397, -0.1159] **sep** | -0.0918 [-0.1035, -0.0791] **sep** |
| 6000 | 1 | -0.1323 [-0.1459, -0.1190] **sep** | -0.0961 [-0.1096, -0.0826] **sep** |

## Instruments, per run

| run | instrument_ok | P10 − P10 | GT ADE max | P10 vs banked refit (m) | untrained params | n_train / overlap | cfg | eval s |
|---|---|---|---|---|---|---|---|---|
| st500/R0_s0 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | bd90eadf4f3f | 125 |
| st500/R0_s1 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | bd90eadf4f3f | 133 |
| st500/A1_b0.08g50_s0 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | 0ff8b3024143 | 132 |
| st500/A1_b0.08g50_s1 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | 0ff8b3024143 | 141 |
| st1000/R0_s0 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | bd90eadf4f3f | 123 |
| st1000/R0_s1 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | bd90eadf4f3f | 124 |
| st1000/A1_b0.08g50_s0 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | 0ff8b3024143 | 134 |
| st1000/A1_b0.08g50_s1 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | 0ff8b3024143 | 141 |
| st3000/R0_s0 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | bd90eadf4f3f | 123 |
| st3000/R0_s1 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | bd90eadf4f3f | 133 |
| st3000/A1_b0.08g50_s0 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | 0ff8b3024143 | 133 |
| st3000/A1_b0.08g50_s1 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | 0ff8b3024143 | 139 |
| st3000/A1x_b0.08g50_s0 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | 92e0f66366a2 | 132 |
| st6000/R0_s0 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | bd90eadf4f3f | 122 |
| st6000/R0_s1 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | bd90eadf4f3f | 131 |
| st6000/A1_b0.08g50_s0 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | 0ff8b3024143 | 134 |
| st6000/A1_b0.08g50_s1 | True | True | 0.0 | 1.52587890625e-05 | 0 | 4572 / 0 | 0ff8b3024143 | 133 |

