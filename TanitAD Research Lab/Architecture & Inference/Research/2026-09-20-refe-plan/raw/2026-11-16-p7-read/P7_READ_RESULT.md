# P7 navtest read: result (full navtest, T1, one look, one training seed)

*Written 2026-11-16 by the REFe land agent from the banked artifacts in this folder. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Paths below are relative to `PK/raw/2026-11-16-p7-read/` unless stated; `chain/...` is the read's work dir, `stage2_p7_full/...` the stage-2 output. Every number cites its file. Evidence class MEASURED (ours) unless stated. The MM's verdict entry (dagland `LOOP_STATE_P6_2026-11-06.md`, "2026-11-16 22:10Z MM: P7 READ COMPLETE") was read as a pointer only; each of its numbers below was re-read from the artifact.*

## 0. Headline

**P7 = 91.0724 PDMS x100 on the full navtest. Against P6 it is -0.12 [-0.51, +0.28]: NOT PROVEN (the interval contains 0). Against the deployed system it is +6.62 [+5.76, +7.50]. BELOW the 94.55 goal.** Tier T1. One training seed. Determinism PASS.

| | |
|---|---|
| Tier | **T1** = NAVSIM v1.1 PDMS, an ego pseudo-simulation of an open-loop plan against logged agents (self-action open loop, not closed loop). The command is an input. `stage3/r2_result.json` (`tier`) |
| Population | 12,146 tokens, 136 logs, harness `PASS`, 12,146 successful, 0 failed, `C1_max_abs_delta` 0.0. `stage2_p7_full/score_status.json` |
| Checkpoint | `model_final.pt` sha256 `2613fc17b9e5d63342c9f42120c5612a6ec7061fda72f83dac46ddd2e70fd3fe`, 1,298,292,047 bytes. `stage3/r2_result.json` (`checkpoint`). Training: 9,075 steps, HF `Sayood/tanitad-refe-p7-reserve-head` (INHERITED from the MM's restart entry in dagland `LOOP_STATE_P6_2026-11-06.md`, line "P7 DONE"; not re-verified here). Parameter count of P7: **UNVERIFIED** (no config in this read set) |
| Yaw input | the navtest yaw table sha256 `f5ebe24fd05a8067215c22f08f873808a7020a97eca6940fdcc568f8b39e29a5` was fed (12,146/12,146 tokens), as `SPEC_P7.md` Amendment YAW requires. `stage2_p7_full/result.json` (`ego_yaw`), `chain/markers/ident.json` |
| Estimator | paired log-cluster bootstrap (cluster unit `log_name`, 136 logs), 10,000 resamples, 95 % percentile, seed 20260927 (`proxy_eval.boot`). **It answers "another draw of LOGS" and nothing else.** Training variance (one seed per arm, `H-ESTIM-SEED-1`) is unmeasured; inference is deterministic (section 3). `stage3/r2_result.json` (`estimator`) |

## 1. Endpoint: PDMS and sub-scores (x100), paired differences P7 minus comparator

Source for every cell: `stage3/r2_result.json`, blocks `R2` (vs P6, C1), `R2b` (vs P2, C2), and `stage2_p7_full/result.json` (`primary`, vs deployed). n = 12,146 tokens, 136 logs everywhere. "sep" = 95 % interval excludes 0.

**PDMS level.** P7 91.0724 (`stage2_p7_full/score_status.json`). Marginal interval **[90.20, 91.93]**: an AGENT RECOMPUTE, not part of the read: log-cluster bootstrap, 10,000 draws, numpy `default_rng(20260927)` (not the pinned `proxy_eval.boot` stream, so the 2nd decimal may differ); script and output `marginal_p7_recompute.py` / `marginal_p7_recompute.json`. For reference P6 is 91.20 [90.41, 91.98] (`MODEL_REGISTRY.md` 14.3, `raw/2026-11-09-p6-navtest-read/level_ci.json`).

**PDMS, paired.**

| comparator | comparator PDMS | P7 minus comparator | better / worse / tied | zero-PDMS tokens (cmp -> P7) | read |
|---|---|---|---|---|---|
| **P6** (C1, `refe_p6final`) | 91.1965 | **-0.1241 [-0.5139, +0.2786]** | 3,530 / 4,560 / 4,056 | 384 -> 393 | not separated |
| P2 (C2) | 88.8184 | **+2.2540 [+1.6462, +2.8726]** | 6,238 / 2,631 / 3,277 | 509 -> 393 | sep, positive |
| deployed (`refe_navtest_final_routefix`) | 84.4561 | **+6.6163 [+5.7604, +7.4990]** | 7,112 / 2,135 / 2,899 | 837 -> 393 | sep, positive |

**Sub-scores x100.** P7 level 98.8721 / 97.8594 / 86.6816 / 95.2906 / 99.9918 / 96.5009 (NC / DAC / EP / TTC / C / DDC), `stage2_p7_full/score_status.json`. Paired differences, comparator level in brackets:

| sub-score | vs P6 | vs P2 |
|---|---|---|
| NC | +0.0247 [-0.2377, +0.2918] (98.8474) | +0.0576 [-0.2294, +0.3413] (98.8144) |
| DAC | -0.0906 [-0.3940, +0.1877] (97.9499) | **+0.9880 [+0.4587, +1.5250]** (96.8714) |
| EP | -0.3528 [-0.7539, +0.0166] (87.0344) | **+4.3421 [+3.7212, +4.9785]** (82.3396) |
| TTC | +0.2058 [-0.3111, +0.7420] (95.0848) | -0.0247 [-0.7240, +0.6829] (95.3153) |
| C | -0.0082 [-0.0279, 0.0000] (100.0000) | +0.0165 [0.0000, +0.0425] (99.9753) |
| DDC | +0.0165 [-0.2494, +0.2794] (96.4844) | **-1.0044 [-1.4589, -0.5673]** (97.5054), flagged `adverse` |

Against the deployed system only the PDMS difference is in the artifact; deployed sub-score levels (`MODEL_REGISTRY.md` 14.2): NC 98.13, DAC 94.71, EP 76.70, TTC 93.69, C 99.93, DDC 97.61. Paired sub-score intervals vs deployed were **not computed** by this read.

**vs P6, no sub-score is adverse** (`R2.adverse_subscores = []`). **vs P2, DDC is adverse** (`R2b.adverse_subscores = ["DDC"]`). The C difference vs P6 is one token (1 worse, 0 better).

## 2. SPEC_P7 section 5 verdict and goal status

* **Rule (`SPEC_P7.md` section 5):** ADOPT iff the lower bound of the difference vs P6 is above 0 and no longitudinal or lateral family component is adverse; lever claim only above F = 0.33; REFUTED iff the upper bound is below 0; else NOT PROVEN. Result: lower bound -0.5139 (not above 0), upper bound +0.2786 (not below 0) => **NOT PROVEN**. `stage3/r2_result.json`: `R2.outcome = "NO-GO (NOT PROVEN)"`, `why = "interval contains 0"`. Not ADOPT and not REFUTED, so the family-adverse clause cannot change the word.
* **The registered families check did not run inside stage 3.** `R2.families_vs_base = {"missing": true, "note": "a families JSON is absent: NOT COMPUTED (never a pass)"}` (same for `R2b`). Cause: `stage3/p7_full_families.log` is a traceback (`numpy ArrayMemoryError` allocating 1.11 MiB in `families6.py`), and `stage2_p7_full/p7_full_families.log` is the same. The families come from the later `fam_ci` route (section 4). So **do not quote "stage 3 says no family is adverse": stage 3 never read the families.** `adverse_subscores = []` concerns the six PDMS sub-scores only.
* **Bundle claim.** P7 minus P6 bundles four changes: the drivable-area head, the 9.84 % smaller training set (session reserve), the yaw input, and noise (`SPEC_P7.md` section 4 and Amendment YAW). No lever is attributable.
* **Goal 94.55 (PUBLISHED, DriveZero A13): BELOW GOAL.** Point 91.07; marginal upper bound 91.93 (agent recompute above) is below 94.55.
* `stage2_p7_full/result.json` carries `"verdict": "NOT PROVEN"` for P7 vs the **deployed** system with a stage-2 rule; stage 3's own `WARNING` says that word is R1/R2b style and not the P6 verdict. The P7 verdict is the R2 block (vs P6). Both read NOT PROVEN, from different comparators.

## 3. Determinism: PASS

* `chain/markers/rep.DONE`: `tokens_equal: true`, `poses_bit_identical: true`, `n: 12146`, `rc: 1`.
* **Re-verified by this agent from the seams on disk** (`E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7_full.npz` vs `stage2_p7_full_rep.npz`): `token`, `fingerprint`, `poses` (12,146 x 8 x 3 float32) and `sampling` are byte-equal; only the `arm` name string differs. Both seam npz stay on the dev box (not copied; banked on HF by the MM).
* `rc 1` is the **expected** frame-control rc, not a determinism failure: both runs end `ZZSEAM_FAIL 12146 6.1806` (`stage2_p7_full/seam.log`, `stage2_p7_full_rep/seam.log`), because `frame_control.max_m` is 6.18 m against a 0.001 m bar while the median is 9.2e-16 (`stage2_p7_full/result.json`, `gates.frame_control_reported`). Stage 3 shows the same frame control as P6's (`frame_control_equal_to_base: true`, `seam_settings_equal_to_base: true`). `stage2_p7_full_rep/result.json` therefore reads `"verdict": "INCOMPLETE (seam failed)"`: **do not read that as a determinism failure.**
* The chain's `seam1` also ended rc 1 (`chain/CHAIN.log`), after two earlier failed attempts (`chain/markers/seam1.FAILED.attempt_1`, `.attempt_2`: seam artifacts absent). The quoted read is from the third attempt; the `score` stage then ran rc 0 in 2,041 s.

## 4. The four metric families (P7 minus P6), `chain/fam_ci_P6_vs_P7.json`

* **Frame:** `executed_tracked` is PRIMARY (`frames.executed_tracked.role = "PRIMARY"`). `executed_transparent` and `reference` are diagnostics (counts in 4.5).
* **Estimator:** paired log-cluster bootstrap, clusters = logs, B = 10,000, seed 20260927, 95 % percentile. **Answers another draw of LOGS only (not another training run, not another inference run)** (`estimator`, `answers`). Direction = B minus A = P7 minus P6. "Separated" = `separated_from_0`. Cell n is per cell.
* **Per family, never pooled.** Units are each cell's own. Not-separated cells are counted and named; their values are in the JSON.

### 4.1 LONGITUDINAL: 20 cells computed, 8 separated, 12 not separated (2 more cells are `HUMAN-ONLY`, undefined for a plan)

| cell | n tokens / logs | P6 | P7 | P7 minus P6 [95 % CI] |
|---|---|---|---|---|
| `speed_bias_mps` | 12,146 / 136 | 0.29096 | 0.2461 | **-0.04486 [-0.06001, -0.03167]** |
| `along_bias_m` | 12,146 / 136 | 0.50328 | 0.42972 | **-0.07356 [-0.09562, -0.05439]** |
| `along_final_bias_m` | 12,146 / 136 | 1.1383 | 0.93535 | **-0.20296 [-0.25718, -0.15420]** |
| `ego_progress.progress_ratio_mean` | 12,053 / 136 | 1.1111 | 1.0995 | **-0.011635 [-0.015917, -0.0077705]** |
| `ego_progress.progress_ratio_median` | 12,053 / 136 | 1.0757 | 1.0689 | **-0.0068085 [-0.010992, -0.0027767]** |
| `ego_progress.progress_error_mean` | 12,053 / 136 | 0.22813 | 0.22207 | **-0.0060612 [-0.0089251, -0.003227]** |
| `lead_progress_ratio` | 5,331 / 132 | 1.1714 | 1.1502 | **-0.021167 [-0.028433, -0.014413]** |
| `ego_progress.under_progress_rate` | 12,053 / 136 | 0.35402 | 0.36705 | **+0.013026 [+0.0050081, +0.021683]** |

Direction: the bias and over-progress cells fall (P6 over-progresses, ratio 1.11); the **under-progress rate rises, the one separated longitudinal move against P7**. Not separated (12): `headway_min_m`, `time_gap_min_s`, `min_ttc_s`, `lead_viol`, `lead_at_risk`, `speed_mae_mps`, `speed_rmse_mps`, `target_speed_acc` at 0.5 / 1.0 / 2.0 m/s, `along_mae_m`, `accel_mae_mps2`.

### 4.2 LATERAL: 8 cells computed, 5 separated, 3 not separated

| cell | n tokens / logs | P6 | P7 | P7 minus P6 [95 % CI] |
|---|---|---|---|---|
| `heading_mae_deg` | 12,102 / 136 | 3.2097 | 3.0861 | **-0.1236 [-0.18073, -0.067218]** |
| `yaw_rate_mae_degps` | 11,965 / 136 | 3.0367 | 2.9893 | **-0.047424 [-0.093107, -0.0018071]** (upper edge near 0) |
| `curvature_mae_1pm` | 11,965 / 136 | 0.013551 | 0.013308 | **-0.00024278 [-0.00041506, -0.00005855]** |
| `cross_mae_m` | 12,146 / 136 | 0.46442 | 0.43842 | **-0.026002 [-0.035733, -0.016058]** |
| `cross_final_mae_m` | 12,146 / 136 | 1.0843 | 1.0234 | **-0.060858 [-0.088014, -0.032326]** |

All five separated lateral differences are error reductions. Not separated (3): `curvature_bias_1pm`, `cross_bias_m`, `last_pose_heading_err_deg`.

### 4.3 TACTICAL: 33 cells computed, 12 separated (11 distinct values), 21 not separated (1 more cell `NO-PER-TOKEN-DATA`)

| cell | n tokens / logs | P6 | P7 | P7 minus P6 [95 % CI] |
|---|---|---|---|---|
| `lateral_decision.per_class.turn_left.recall` | 2,811 / 102 | 0.90537 | 0.91782 | **+0.012451 [+0.00089751, +0.024861]** |
| `maneuver_5way_collapsed.per_class.turn_left.recall` | 2,811 / 102 | 0.90537 | 0.91782 | same cell as the row above (identical n and values), counted twice in the 12 |
| `longitudinal_decision.per_class.brake_stop.recall` | 4,285 / 127 | 0.76663 | 0.7867 | **+0.02007 [+0.0089703, +0.031574]** |
| `longitudinal_decision.per_class.accelerate.recall` | 3,861 / 133 | 0.87205 | 0.88345 | **+0.011396 [+0.0011921, +0.021075]** |
| `longitudinal_decision.per_class.accelerate.precision` | 5,903 / 134 | 0.6124 | 0.62029 | **+0.0078901 [+0.000047361, +0.015921]** (lower edge near 0) |
| `maneuver_5way_collapsed.accuracy` | 12,146 / 136 | 0.74856 | 0.75597 | **+0.0074098 [+0.000079934, +0.014472]** (lower edge near 0) |
| `maneuver_5way_collapsed.per_class.accelerate.precision` | 3,014 / 133 | 0.55865 | 0.57772 | **+0.019067 [+0.0024188, +0.036344]** |
| `maneuver_5way_collapsed.per_class.brake_stop.recall` | 3,726 / 121 | 0.75845 | 0.78744 | **+0.028986 [+0.014755, +0.042531]** |
| `goal_setting.goal_point_error_m` | 12,146 / 136 | 3.2718 | 3.2229 | **-0.048981 [-0.09375, -0.0012753]** (upper edge near 0) |
| `goal_setting.goal_long_bias_m` | 12,146 / 136 | 1.1383 | 0.93535 | **-0.20296 [-0.25718, -0.1542]**: identical to `along_final_bias_m` in 4.1 (one quantity in two families) |
| `goal_setting.goal_bearing_mae_deg` | 12,053 / 136 | 2.7461 | 2.5575 | **-0.18852 [-0.26098, -0.11105]** |
| `goal_setting.goal_range_ratio` | 12,053 / 136 | 1.1154 | 1.1029 | **-0.012533 [-0.016814, -0.0087239]** |

Not separated (21), among them `lateral_decision.accuracy` (+0.00041 [-0.00568, +0.00623]), `lateral_decision.kappa`, `longitudinal_decision.accuracy`, `longitudinal_decision.kappa`, `maneuver_5way_collapsed.kappa` and `goal_setting.goal_lat_bias_m`.

**Command adherence** (the registered tactical cell, `chain/command_adherence_p7_full.json`, same estimator, turn-active subset n = 4,076 tokens / 113 logs): P7 accuracy **0.8805 [0.8555, 0.9025]**, kappa 0.7739 [0.7311, 0.8120]. P6 0.8869 [0.8634, 0.9084]; P2 0.8827 [0.8586, 0.9053]. P7 minus P6: accuracy -0.0064 [-0.0155, +0.0033], kappa -0.0112 [-0.0268, +0.0053] (not separated); minus P2: accuracy -0.0022 [-0.0198, +0.0141] (not separated). Random-proposal floor (mean over the 64 proposals): 0.8841 [0.8645, 0.9020], so **P7's executed plan is not above its own random-proposal floor on the turn-active subset** (floor above the point). The logged human future scores 0.8810 on the same labeller. Authority NONE: reported, not gating.

### 4.4 STRATEGIC: not computed: work item

**Not computed: the instrument still stamps PAUSED (`chain/fam_ci_P6_vs_P7.json`, `frames.*.families.strategic`: `status "PAUSED"`, "PI ruling 2026-09-19 (PI_DECISION_QUEUE item 25)"), but the PI lifted the pause 2026-11-16 => work item.** There is no strategic-family instrument for a REFe read (route and goal-setting quality). It is not reported as paused and it is not a pass. Backlog item: a STRATEGIC-family instrument for REFe reads.

### 4.5 Diagnostic frames (counts only, `chain/fam_ci_P6_vs_P7.json`)

Separated cells in `executed_transparent`: longitudinal 9, lateral 5, tactical 9. In `reference`: longitudinal 9, lateral 5, tactical 9. Not analysed cell by cell here.

### 4.6 Reading the families

25 of 61 computed primary-frame cells are separated (8 + 5 + 12; 23 distinct values after the two double counts above). Lateral errors and the longitudinal bias and over-progress cells improve; one longitudinal cell moves the other way. PDMS does not move. Cells within a family are correlated and no multiplicity correction is applied: at a nominal 5 %, about 3 of 61 would separate by chance (ESTIMATED, arithmetic only). On the tiny v7 rig an identical-flags replicate separated 9.5 % of family cells (`CLAUDE.md`, `H-ESTIM-SEED-1`; MEASURED on that rig, not this one). **A separated cell here is necessary, not sufficient, for a lever claim.**

## 5. Caveats (binding wording)

* **"Another draw of logs" only.** Every interval here (paired and marginal) resamples logs with the models held fixed. One training seed; training variance and the P7-minus-P6 bundle (head + reserve + yaw + noise) are unmeasured. A replicate arm is needed before any lever claim.
* Single look, full navtest, no tuning on it. T1, not closed loop.
* The deployed comparator's sub-score intervals were not computed; the marginal P7 interval is an agent recompute, flagged as such.
* Frame control: max 6.18 m on at least one token (median ~1e-15 m), identical to P6's; the seam stage's rc 1 is this gate and is expected.
* Stage-3 families are missing (traceback); the families come from `fam_ci`.

## 6. Mechanism reads still open (SPEC_P7 section 5, NOT PROVEN branch)

DAC-zero share vs P6, head IoU [0, 40) m on navtest and the reserve (bar 0.80), head-vs-scorer DAC AUC: owned by the P7MECH agent (`scorer_gym_20261107/p7_next/BRIEF_P7MECH.md`). Not in this land set.
