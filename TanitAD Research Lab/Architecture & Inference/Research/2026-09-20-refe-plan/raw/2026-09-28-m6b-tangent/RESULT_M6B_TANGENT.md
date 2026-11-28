# RESULT: M6b tangent-consistency term -- **REFUTED** (PRIMARY failed for ALL T arms; seed-mean PDMS CI upper bound < -1.0)

Pre-registration `eval/PREREG_M6B_TANGENT.md`, blob at analysis `83e45c8390f28c9e894c9e5aea0d51133939743c`. Every number below is read from `result_m6b.json`.

| gate | result |
|---|---|
| G1 | PASS |
| G2 | PASS |
| G3 | PASS |
| G4 | PASS |
| G5 | PASS |

| T arm | (a) winner median | (b) beyond-pi % | (c) all-slot vs own tangent | pass |
|---|---|---|---|---|
| T0 | 0.1035 | 1.7809 | 0.1004051488895461 | FAIL |
| T1 | 0.1130 | 1.7768 | 0.1560192238622986 | FAIL |
| T2 | 0.1303 | 1.7002 | 0.13325411841668666 | FAIL |

Seed-mean D = **-3.596 [-4.716, -2.496]**; per seed: 0: -3.038 [-5.181, -0.934]; 1: -3.796 [-6.016, -1.812]; 2: -3.955 [-5.719, -2.139]. Between-seed SD of D 0.490 (reported).

<!-- m6b_report: everything below is appended by eval/m6b_report.py -->

## Manipulation check (each wrapped arm) and position guard (each seed)

| arm | median winner step-19 heading error (rad) | raw step-19 \|h\| > pi (%) | stays >= 0.50 |
|---|---|---|---|
| Wt0 | 0.8073 | 59.9510 | PASS |
| Wt1 | 0.8854 | 61.2659 | PASS |
| Wt2 | 0.8232 | 60.6648 | PASS |

| seed | T - Wt, winner ADE over the 8 NAVSIM poses (m) | <= +0.10 |
|---|---|---|
| 0 | +0.0158 | PASS |
| 1 | -0.0398 | PASS |
| 2 | -0.0236 | PASS |

## PDMS: seams, sub-scores, between-seed spread

Estimator: paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927; 1123 tokens. Between-seed SD (reported, never gating): of D 0.490, of Wt PDMS 1.480, of T PDMS 1.141.

| seam | PDMS x100 | no_at_fault_collisions | drivable_area_compliance | ego_progress | time_to_collision_within_bound | comfort | driving_direction_compliance |
|---|---|---|---|---|---|---|---|
| refe_m6_Wt0_on | 78.063 | 96.26 | 91.54 | 68.82 | 90.29 | 97.86 | 95.24 |
| refe_m6_Wt1_on | 81.023 | 95.64 | 93.32 | 74.23 | 89.40 | 98.13 | 95.10 |
| refe_m6_Wt2_on | 79.566 | 95.50 | 92.88 | 71.88 | 89.40 | 98.04 | 95.24 |
| refe_m6_T0_off | 75.025 | 93.10 | 89.58 | 69.87 | 84.68 | 96.71 | 93.59 |
| refe_m6_T1_off | 77.227 | 95.59 | 91.36 | 67.56 | 89.67 | 97.68 | 95.01 |
| refe_m6_T2_off | 75.610 | 94.84 | 90.03 | 66.52 | 88.25 | 96.44 | 95.99 |

## Four families (families6 on the six gating seams; binding 2026-08-02)

Tier: T1-family (stage-1 loop OPEN; the plan is the model's single query), NAVSIM navtest 1,123 tokens. Intervals: families6's episode-cluster bootstrap (2,000 resamples, 136 episodes).

### LONGITUDINAL

| seam | speed_mae_mps | speed_bias_mps | along_mae_m | along_final_bias_m | accel_mae_mps2 | target speed within 0.5 / 1.0 / 2.0 m/s | progress ratio (mean) |
|---|---|---|---|---|---|---|---|
| refe_m6_Wt0_on | 1.1731 | -0.4307 | 1.8695 | -1.7087 | 0.7716 | 0.4344 / 0.6203 / 0.8203 | 0.9711 |
| refe_m6_Wt1_on | 1.0641 | -0.0882 | 1.6429 | -0.3204 | 0.7507 | 0.4677 / 0.6596 / 0.8540 | 1.0620 |
| refe_m6_Wt2_on | 1.2209 | -0.3340 | 1.8590 | -1.3075 | 0.8626 | 0.4383 / 0.6240 / 0.8069 | 1.0303 |
| refe_m6_T0_off | 1.1452 | 0.0223 | 1.7890 | 0.1924 | 0.7749 | 0.4484 / 0.6395 / 0.8326 | 1.0784 |
| refe_m6_T1_off | 1.2495 | -0.6027 | 1.9219 | -2.3093 | 0.8662 | 0.4324 / 0.6000 / 0.7981 | 0.9457 |
| refe_m6_T2_off | 1.3657 | -0.6573 | 2.0921 | -2.6173 | 0.9647 | 0.4102 / 0.5789 / 0.7687 | 0.9296 |

### LATERAL

| seam | heading_mae_deg | yaw_rate_mae_degps | curvature_mae_1pm | cross_mae_m | cross_final_mae_m |
|---|---|---|---|---|---|
| refe_m6_Wt0_on | 3.0467 | 4.5653 | 0.0368 | 0.4154 | 0.9838 |
| refe_m6_Wt1_on | 3.1195 | 4.4249 | 0.0304 | 0.4694 | 1.1053 |
| refe_m6_Wt2_on | 3.1627 | 4.7288 | 0.0337 | 0.4819 | 1.1564 |
| refe_m6_T0_off | 3.0362 | 4.0632 | 0.0278 | 0.4938 | 1.1610 |
| refe_m6_T1_off | 3.1334 | 5.0628 | 0.0395 | 0.4584 | 1.0968 |
| refe_m6_T2_off | 3.4120 | 5.4947 | 0.0440 | 0.4807 | 1.1575 |

### TACTICAL (trajectory-derived decisions; accuracy / kappa -- kappa is the readable number)

| seam | lateral_decision | longitudinal_decision | maneuver_5way_collapsed | goal point error (m) | goal bearing MAE (deg) |
|---|---|---|---|---|---|
| refe_m6_Wt0_on | 0.9065 / 0.7819 | 0.6616 / 0.4647 | 0.7231 / 0.6295 | 4.7094 | 2.4929 |
| refe_m6_Wt1_on | 0.9172 / 0.8028 | 0.6883 / 0.5099 | 0.7489 / 0.6661 | 4.2301 | 2.6869 |
| refe_m6_Wt2_on | 0.8949 / 0.7552 | 0.6687 / 0.4734 | 0.7159 / 0.6208 | 4.8053 | 2.6392 |
| refe_m6_T0_off | 0.9029 / 0.7676 | 0.6901 / 0.5161 | 0.7355 / 0.6514 | 4.5660 | 2.7160 |
| refe_m6_T1_off | 0.8807 / 0.7241 | 0.6652 / 0.4673 | 0.7070 / 0.6092 | 4.9613 | 2.6722 |
| refe_m6_T2_off | 0.8940 / 0.7528 | 0.6643 / 0.4624 | 0.7088 / 0.6087 | 5.4141 | 2.8356 |

### STRATEGIC: **UNAVAILABLE** (n = 1123) -- NAVSIM (nuPlan/OpenScene) DOES carry a map, a lane graph and a route — this is NOT the PhysicalAI-AV corpus gap. STRATEGIC is n/a here for two different reasons: (1) the BENCHMARK never scores it — `driving_command` is an agent INPUT and PDMS/EPDMS have no strategic term (NAVSIM_PROTOCOL.md §8, coverage row strat.decision / strat.route_goal: ABSENT); (2) this harness has no NAVSIM strategic label 

### Adverse separations, T_s vs Wt_s (a T interval entirely on the worse side of Wt's; named, NOT gating)

- seed 0: 0 of 23 components -- none
- seed 1: 2 of 23 components -- longitudinal.ego_progress.progress_ratio_median (Wt [0.9133, 0.9806], T [0.7979, 0.8553]), longitudinal.ego_progress.under_progress_rate (Wt [0.5382, 0.6354], T [0.6833, 0.7712])
- seed 2: 1 of 23 components -- longitudinal.ego_progress.progress_ratio_median (Wt [0.864, 0.9355], T [0.7655, 0.8411])

## Heading profile (reported, not gating)

- Wt0: winner heading error, median by NAVSIM pose 0.5..4.0 s: [0.0161, 0.0242, 0.0217, 0.0397, 0.0239, 0.0264, 0.0335, 0.8073]; raw |h| > pi (%) by native step 0..19: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 59.951]; all-slot step-19 vs own tangent (median rad): 1.0542
- Wt1: winner heading error, median by NAVSIM pose 0.5..4.0 s: [0.0148, 0.0208, 0.0222, 0.0249, 0.0246, 0.0314, 0.0314, 0.8854]; raw |h| > pi (%) by native step 0..19: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 61.266]; all-slot step-19 vs own tangent (median rad): 1.0940
- Wt2: winner heading error, median by NAVSIM pose 0.5..4.0 s: [0.0183, 0.0245, 0.0228, 0.0314, 0.028, 0.0326, 0.0335, 0.8232]; raw |h| > pi (%) by native step 0..19: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60.665]; all-slot step-19 vs own tangent (median rad): 0.9992
- T0: winner heading error, median by NAVSIM pose 0.5..4.0 s: [0.0162, 0.0211, 0.0243, 0.0266, 0.03, 0.0281, 0.0282, 0.1035]; raw |h| > pi (%) by native step 0..19: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.781]; all-slot step-19 vs own tangent (median rad): 0.1004
- T1: winner heading error, median by NAVSIM pose 0.5..4.0 s: [0.0181, 0.0206, 0.0334, 0.032, 0.0271, 0.0336, 0.031, 0.113]; raw |h| > pi (%) by native step 0..19: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.777]; all-slot step-19 vs own tangent (median rad): 0.1560
- T2: winner heading error, median by NAVSIM pose 0.5..4.0 s: [0.017, 0.0238, 0.0267, 0.0284, 0.0238, 0.0471, 0.0342, 0.1303]; raw |h| > pi (%) by native step 0..19: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.078, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.7]; all-slot step-19 vs own tangent (median rad): 0.1333

## Declared implementation choices and the run record (M6b)

1. **Micro-batch 64 x accum 4 = effective batch 256**, as M6 (proof: `raw/2026-09-27-m6-proxy/microbatch_invariance.json`). The scorer loss on covered samples only, the visual context on labelled frames only, the live calibration table, the harness label prefix: all as M6 (its RESULT section 'Pre-data implementation deviations').
2. **All six arms read ONE train-cache path**: ['C:\\Users\\Admin\\refe_proxy_fast\\cache_train_ep015'] (G5 compares argv).
3. **The NVMe copy**: the epoch-2 copy was KEPT as M6b's one authorised copy (coordinator 00:4x: floor >= 30 GiB with it present, emergency < 15 GiB, delete after analyze-m6b) and RE-HASHED file by file against the source manifest before any arm read it: 80 shard files, verify rc 0, C: free 41.56 GiB; 80 shard files, verify rc 0, C: free 41.09 GiB.
4. **G3 re-run under the M6b trainer bytes** (`arms/G3_zero_lr`, M6's argv) and **the untouched-snapshot decode re-run under the final proxy_eval bytes** (`dumps/base.json`) -- every stage ran the SAME code (the prereg's reason for re-running the wrapped arms).
5. **Code identity**: the chain recorded the sha256 of every module an arm or a decode imports at start (`code_baseline.json`, 10 modules) and asserted it unchanged before and after every GPU stage; each arm's own `config.json` code_sha256 was asserted equal to it.
6. **Selftests on the final bytes**: selftest_measures 84 checks, failed []; selftest_proxy_train 20, failed []; selftest_proxy_eval 45, failed []. Measures and proxy_train ran in a sibling copy of the package holding the SAME bytes (tested sha256 == the package's, asserted by the chain) while the epoch-2 analysis still held the package; selftest_measures (~25 min) ran BESIDE the first arms and the analysis waited for its PASS on the chain's exact measures.py.
7. **Chain v1 -> v2 at 00:51**: v1 admitted two stages on ONE VRAM reading (harmless then: 2,525 + 800 + 2,500 = 5,825 MiB <= 6,960); v2 serialises the gate and counts its own ramping stages. v1's two running stages (G3, the base decode) were adopted, not repeated.

