# LONHEAD_C phase 2 -- FAIL_F (head better, F1 and F2 FAIL)

**Tier:** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  
**Estimator:** paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, bootstrap seed 0, alpha 0.05, over EPISODES only (models fixed): answers 'another draw of EPISODES' only -- blind to training variance (ONE refcv8 training seed), to head-fit variance (ONE LH fit, ONE DER fit) and to inference variance within a seed; inference seeds 0 and 1 are reported separately, never pooled.  
**Prereg:** PREREG_LONHEAD_C.md 6c81d4b0 + AMENDMENT1 715b9df1 (FIT rows = the 292 fresh clips with obstacle labels); LH = logistic lambda 100 (C 0.01, the sealed grid's strongest regularisation, grid edge, disclosed).  
**Committed outcome (prereg s5):** head accuracy is not the binding constraint for the factored composition: the progress the sharpened class implies is (next: a class-conditioned progress head on the fresh split, pre-registered first).

## K-HEAD (T0, go/no-go): LH decel AUC 0.7305 [0.6749, 0.7855] vs reference 0.685 (native reproduced 0.6851); DER 0.4267; LH - native {'point': 0.0454, 'ci95': [-0.0125, 0.1054]} (episode-resampling bootstrap, n 2495 union windows) -> GO

## Bars (S6 compliant)  PASS_s0 = False  A_s0 = False  REPLICATED_s1 = False  controls_ALL_PASS = True

### inference seed 0: F1 False | F2 False | G True (worse rows: none) | A1 False | A2 True

| arm | MD | FD | n decel | n cruise |
|---|---|---|---|---|
| FACTORED_DEPLOY8 | 0.3897 | 0.1503 | 639 | 1856 |
| H2CN_DEPLOY8 | 0.5336 | 0.0603 | 639 | 1856 |
| BASE_OS | 0.4695 | 0.1395 | 639 | 1856 |
| RETIME_DEPLOY8 | 0.4695 | 0.1369 | 639 | 1856 |
| LHS_FACT8 | 0.3912 | 0.1638 | 639 | 1856 |
| DER_FACT8 | 0.4757 | 0.1886 | 639 | 1856 |

* F1_FD_LHS_vs_FX: +0.0135 [-0.0134, +0.0383] (0.1638 vs 0.1503, n=1856/135 ep)
* F1_MD_LHS_vs_FX: +0.0016 [-0.0303, +0.0335] (0.3912 vs 0.3897, n=639/79 ep)
* F2_MD_LHS_vs_H8: -0.1424 [-0.2000, -0.0849] SEP (0.3912 vs 0.5336, n=639/79 ep)
* F2_FD_LHS_vs_H8: +0.1034 [+0.0709, +0.1385] SEP (0.1638 vs 0.0603, n=1856/135 ep)
* A1_FD_LHS_vs_DER: -0.0248 [-0.0721, +0.0184] (0.1638 vs 0.1886, n=1856/135 ep)
* A2_MD_LHS_vs_DER: -0.0845 [-0.1380, -0.0361] SEP (0.3912 vs 0.4757, n=639/79 ep)
* beside_DER_F1_FD_vs_FX: +0.0383 [-0.0080, +0.0882] (0.1886 vs 0.1503, n=1856/135 ep)
* beside_DER_F1_MD_vs_FX: +0.0861 [+0.0359, +0.1429] SEP (0.4757 vs 0.3897, n=639/79 ep)
* beside_DER_F2_MD_vs_H8: -0.0579 [-0.1272, +0.0061] (0.4757 vs 0.5336, n=639/79 ep)
* beside_DER_F2_FD_vs_H8: +0.1282 [+0.0780, +0.1831] SEP (0.1886 vs 0.0603, n=1856/135 ep)

### inference seed 1: F1 False | F2 False | G True (worse rows: none)

| arm | MD | FD | n decel | n cruise |
|---|---|---|---|---|
| FACTORED_DEPLOY8 | 0.3897 | 0.1525 | 639 | 1856 |
| H2CN_DEPLOY8 | 0.5321 | 0.0603 | 639 | 1856 |
| BASE_OS | 0.4710 | 0.1374 | 639 | 1856 |
| RETIME_DEPLOY8 | 0.4710 | 0.1374 | 639 | 1856 |
| LHS_FACT8 | 0.3834 | 0.1616 | 639 | 1856 |

* F1_FD_LHS_vs_FX: +0.0092 [-0.0164, +0.0312] (0.1616 vs 0.1525, n=1856/135 ep)
* F1_MD_LHS_vs_FX: -0.0063 [-0.0385, +0.0269] (0.3834 vs 0.3897, n=639/79 ep)
* F2_MD_LHS_vs_H8: -0.1487 [-0.2060, -0.0927] SEP (0.3834 vs 0.5321, n=639/79 ep)
* F2_FD_LHS_vs_H8: +0.1013 [+0.0688, +0.1363] SEP (0.1616 vs 0.0603, n=1856/135 ep)

## Beside rows
* SHIP-R8 Amendment-5 machinery on LHS_FACT8: overall PASS (forward ALL_PASS True)
* LHS_FACT8 E1: s0: -0.0307 [-0.0421, -0.0195] SEP (0.2552 vs 0.2858, n=4159/138 ep) | s1: -0.0297 [-0.0410, -0.0188] SEP (0.2561 vs 0.2858, n=4159/138 ep)
* LHS_FACT8 E3: s0: -0.9104 [-1.2266, -0.6011] SEP (2.4993 vs 3.4098, n=3251/138 ep) | s1: -0.9097 [-1.2212, -0.6054] SEP (2.5000 vs 3.4098, n=3251/138 ep)
* FACTORED_DEPLOY8_landed E1: s0: -0.0331 [-0.0453, -0.0214] SEP (0.2527 vs 0.2858, n=4159/138 ep) | s1: -0.0309 [-0.0429, -0.0199] SEP (0.2549 vs 0.2858, n=4159/138 ep)
* FACTORED_DEPLOY8_landed E3: s0: -0.9883 [-1.2959, -0.6863] SEP (2.4215 vs 3.4098, n=3251/138 ep) | s1: -0.9816 [-1.2920, -0.6865] SEP (2.4282 vs 3.4098, n=3251/138 ep)
* H2CN_DEPLOY8_landed E1: s0: -0.0471 [-0.0585, -0.0363] SEP (0.2387 vs 0.2858, n=4159/138 ep) | s1: -0.0464 [-0.0576, -0.0360] SEP (0.2394 vs 0.2858, n=4159/138 ep)
* H2CN_DEPLOY8_landed E3: s0: -1.1975 [-1.4942, -0.9168] SEP (2.2123 vs 3.4098, n=3251/138 ep) | s1: -1.1909 [-1.4800, -0.9163] SEP (2.2189 vs 3.4098, n=3251/138 ep)
* compliance (LHS_FACT8): S2|s0: {"exceed_rate_strict": 0.005771, "n_exceed_strict": 24} | S6|s0: {"exceed_rate_strict": 0.004306, "n_exceed_strict": 14} | S2|s1: {"exceed_rate_strict": 0.007213, "n_exceed_strict": 30} | S6|s1: {"exceed_rate_strict": 0.004614, "n_exceed_strict": 15}
* compliance (FACTORED_DEPLOY8_landed): S2|s0: {"exceed_rate_strict": 0.006011, "n_exceed_strict": 25} | S6|s0: {"exceed_rate_strict": 0.003999, "n_exceed_strict": 13} | S2|s1: {"exceed_rate_strict": 0.006973, "n_exceed_strict": 29} | S6|s1: {"exceed_rate_strict": 0.004614, "n_exceed_strict": 15}
* compliance (OSONLY_DEPLOY8): S2|s0: {"exceed_rate_strict": 0.004088, "n_exceed_strict": 17} | S6|s0: {"exceed_rate_strict": 0.003384, "n_exceed_strict": 11} | S2|s1: {"exceed_rate_strict": 0.006011, "n_exceed_strict": 25} | S6|s1: {"exceed_rate_strict": 0.005537, "n_exceed_strict": 18}
* compliance (RETIME_DEPLOY8): S2|s0: {"exceed_rate_strict": 0.00553, "n_exceed_strict": 23} | S6|s0: {"exceed_rate_strict": 0.004922, "n_exceed_strict": 16} | S2|s1: {"exceed_rate_strict": 0.006011, "n_exceed_strict": 25} | S6|s1: {"exceed_rate_strict": 0.005537, "n_exceed_strict": 18}
* LH confusion on eval (T0, S6 compliant, n 2999): LH acc 0.5832 vs native 0.6279 vs constant 0.3591; LH recall {"0": 0.622, "1": 0.756, "3": 0.455, "4": 0.033, "5": 0.194, "7": 0.423}; native recall {"0": 0.798, "1": 0.72, "3": 0.458, "4": 0.067, "5": 0.452, "7": 0.509}
* four families: per_seed.<s>.families in LONHEAD_VERDICT.json (ADE / LONGITUDINAL / LATERAL / TACTICAL vs FACTORED_DEPLOY8, H2CN_DEPLOY8, BASE-os); STRATEGIC and distance-keeping n/a with reasons (families_not_computed)

## Beside: LHS_FACT8 vs FACTORED_DEPLOY8 on the four families (s0, S6 compliant; not a bar)
ADE +0.0779 m [+0.0228, +0.1389] SEP worse; FDE +0.2226 SEP worse; LONGITUDINAL speed MAE +0.0419 / along MAE +0.0854 / accel MAE +0.0182 / along |final| +0.2441 all SEP worse; LATERAL all unseparated; TACTICAL traj_lon_correct -0.0228 [-0.0395, -0.0061] SEP worse, traj_lat_correct unseparated; vs BASE-os no row separated worse (G PASS). STRATEGIC and distance-keeping: n/a (no route label / no lead track in the banked dumps; n = 0).

## Post-hoc DESCRIPTIVE (not pre-registered, not a bar) -- why a better-ranking head did not move the plan
LH's argmax IS a better decel DECISION at the label level (GT-decel windows forced to a decel class 0.4961 vs native 0.4163; GT-cruise windows forced to decel 0.1288 vs 0.1358; n 639 / 1,856, seed 0), yet the composed plan does not move (MD 0.3912 vs 0.3897, FD 0.1638 vs 0.1503). The label -> plan transfer is leaky both ways: on GT-decel windows whose label is NOT decel the plan still decelerates 0.357 (LHS) / 0.450 (FACTORED), and on GT-cruise windows labelled decel it decelerates 0.577 / 0.456. The windows LH adds to 'decel' are ones the os-only plan already decelerates on more often (base-os 0.763 vs 0.692), so the marginal planner gain is small, and its decel labels on cruise convert to braking more often. => the binding constraint is the progress the forced class implies, not the class (prereg s5's committed branch).

## Next lever
class-conditioned progress head on the fresh split (prereg s5 committed branch for 'head better, F1 and F2 FAIL') -- BLOCKED on: a Master-Mind-sealed pre-registration (s5: 'pre-registered first'); 0 GPU until sealed.

## Deviations and disclosures
* D1 one label file per head serves both LHS inference seeds: eval z_tac and the fed v0 are bit-identical across seeds 0/1 (asserted, max |diff| 0.0 on all 4,754 keys), as SELFSHARP's s1 roll reused its s0 labels.
* D2 LHS_FACT8 was produced TWICE and asserted bit-equal on S6 at both seeds: (a) the full SHIP-R8 Amendment-5 forward (code/sub_fx_lhs = ship_r8_sub_fx with a 3-line diff: the trace source; outputs in fwd_lhs/, nothing written into ship_r8_sub_fx) and (b) the light composer (SF.compose + RS.retime_plan, registered deploy table) that reproduces the LANDED FACTORED_DEPLOY8 bit for bit from the SELFSHARP traces at both seeds (plan8 / pos8 / S6 pos; COMPOSER_VALIDATION.json, with a mutation that must and does differ).
* D3 DER_FACT8 (seed 0 only, as the prereg rolls it) comes from the light composer; its compliance row and E1 / E3 are NOT computed because the SHIP-R8 forward reads BOTH inference seeds.
* D4 G's BASE-os = the LANDED OSONLY_DEPLOY8 S6 positions, asserted == the os-only plan staged (lane AND l2_common.apply_stage) and reproducing the K_NONE literals (0.46948, 0.13955) s0 / (0.47105, 0.13739) s1.
* D5 A when both F-bars pass would need a reading of 'the frontier bar that passes'; coded as (F1 and A1) or (F2 and A2). Moot: neither F-bar passed at either seed.
* D6 the shared gpu_lock returned ok:true while its own gate field read ok:false (free RAM 5.92 GB < its 8.0 GB rule; no other python compute app, GPU 3.5 GB used by desktop apps). Rolls ran one at a time under the lock and completed 4,754/4,754 each (2,091 / 2,022 / 1,999 s).
* D7 the roll logs (logs/roll_*.log) are written by the reused upstream roll code and carry 12-char clip-id prefixes: they are EXCLUDED from the manifest and must not be landed.
* D8 K-HEAD's v0 = the z_tac bank's fed v0_in (== extras v0 exactly, max |diff| 0.0).

ZZLHVERDICTENDZZ
