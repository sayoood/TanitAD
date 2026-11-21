# Open-loop binding-KPI suite — `refav1-21109-openloop`

**Instrument:** `taniteval/tools/openloop_suite.py` · **generated:** 2026-09-04T07:04:26Z · **schema:** `taniteval.openloop_suite/1`
**Regime:** **OPEN LOOP** — every arm below. PI, 2026-09-02: a predictor consuming the planner's own output is STILL open loop, because the model's trajectory does not affect the ego data it is next fed. The other regime means the trajectory controls a vehicle in a simulator (AlpaSim) or a real test vehicle.

## 1. Headline — does the model beat its trivial control?

The question is **does `cl` beat the trivial control `ha0`, per family, on the same windows, paired?** — not *what is the ADE*. ⛔ A paired interval that excludes zero while favouring the FLOOR means the trivial baseline WON. Render that as LOST, never as a tie.

| family | verdict vs `ha0` (oracle nav) | verdict vs `ha0` (deployment, nav withheld) |
|---|---|---|
| **STRATEGIC** | REFUSED | covered in-family by the `nav_zero` conditioning row, when present |

`ha0` is constant velocity at the measured `v0`. It is the strongest trivial baseline and the only arm that is bit-comparable across models. A margin over it is the admissible cross-model statistic; a level is not.

### 1.1 ⭐ The two margins carry EQUAL billing

The right-hand column is not a footnote. The nav/route token this model consumes is an **ORACLE** (provenance `ego-future`) and **will not exist at deployment**, so:

- **`os − ha0`** — the oracle-nav margin. The fair like-for-like against another arm that is also fed nav.
- **`os_navzero − ha0`** — the **deployment** margin, nav withheld. This is the one that answers *does this drive*.

⛔ **Quoting only the oracle-nav margin OVERSTATES the system.** Both are computed, both are admissible, and both are printed side by side throughout this report.

> ⛔ **NON-PARITY — read before any cross-arm comparison.** ⛔ THIS RUN IS NON-PARITY unless `parity_status` says otherwise. refcv3's own config.json carries `v2_parity.parity false`, `checked false` and `corpus_key null`, and the trainer prints a non-parity warning on line 2 of its train.log. ⇒ refcv3 is NOT cross-arm comparable with refc-base or refc-xl, and a reader must not assume the programme's usual parity. The ONLY admissible cross-model statistic here is each arm's MARGIN over the shared `ha0` floor, which is bit-identically defined everywhere — never a level against another model's level.

## 2. Void gates — what SHAPE is the arm, before any family row

⛔ If either profile is degenerate the read is VOID, not negative. Void means the instrument saw the baseline (or one constant anchor), not the model.

| arm | n | straight | const-speed | trivial (CV plan) |
|---|---|---|---|---|

> Trivial profile: no arm besides the floor is degenerate. (`ha0` is expected here — it *is* the constant-velocity plan.)

**Selection profile:** not available

## 3. The four families — reported separately, never pooled

Sayed 2026-08-02, binding: LONGITUDINAL + LATERAL + TACTICAL + STRATEGIC in ADDITION to ADE, per family, NEVER pooled. This suite emits no composite score and refuses to invent one. A family reported REFUSED/UNAVAILABLE is a WORK ITEM, not a pass.

Estimator: **paired_episode_cluster_bootstrap — two arms on the SAME windows, resampled together so the shared per-window difficulty cancels inside each draw**, n_boot 2000, seed 0, cluster unit **the episode (windows inside one clip are dependent)**. overlapping_holdout_se is NOT used anywhere in this suite. It is not a jackknife, it is anti-conservative, and it BIASES THE POINT ESTIMATE bidirectionally — measured up to a sign flip on a paired delta. Named here only to disavow it.

### 3.1 ADE

*No paired block for this family in the record.*

### 3.2 LONGITUDINAL

*No paired block for this family in the record.*

### 3.3 LATERAL

*No paired block for this family in the record.*

### 3.4 TACTICAL

*No paired block for this family in the record.*

### 3.5 STRATEGIC

**REFUSED** · contrast `accuracy vs the MAJORITY-CLASS rate (the no-information value for a classifier), plus the nav controls` · reported on its own; STRATEGIC has no paired block against `ha0` because the constant-velocity floor has no route head

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|

### 3.6 Per-family absolute readings (headline arm)

**LONGITUDINAL** — n_windows 282, tier `T1`

| component | value | 95 % CI (episode-cluster bootstrap) |
|---|---|---|
| speed_mae_mps | 0.5412 | [0.4741, 0.6105] |
| speed_bias_mps | -0.114 | [-0.1955, -0.0257] |
| speed_rmse_mps | 0.8888 | [0.7774, 0.9963] |
| along_mae_m | 0.4241 | [0.3741, 0.4744] |
| along_bias_m | -0.0881 | [-0.155, -0.0189] |
| along_final_bias_m | -0.1513 | [-0.3221, 0.0252] |
| accel_mae_mps2 | 0.5499 | [0.4805, 0.6165] |
| ego_progress | OK | see the JSON record for this sub-block |
| distance_keeping | OK | see the JSON record for this sub-block |
| anti_echo | OK | see the JSON record for this sub-block |
| target_speed_acc.within_0.5_mps | 0.6713 | [0.6305, 0.7135] |
| target_speed_acc.within_1.0_mps | 0.8298 | [0.7975, 0.8631] |
| target_speed_acc.within_2.0_mps | 0.9475 | [0.9309, 0.9628] |
| ego_progress.progress_ratio_mean | 0.9713 | [0.9443, 0.9953] |
| ego_progress.progress_ratio_median | 0.9912 | [0.987, 0.9938] |
| ego_progress.progress_error_mean | 0.1029 | [0.0789, 0.132] |
| ego_progress.under_progress_rate | 0.6314 | [0.573, 0.6884] |
| ego_progress.gt_progress_mean_m | 23.7972 | [21.4135, 26.287] |
| ego_progress.t0_axis_gt_self_ratio | 0.9822 | [0.9595, 0.9975] |
| distance_keeping.mean_headway_min_m | 26.2149 | [22.2808, 30.6606] |
| distance_keeping.mean_min_ttc_s | 22.9028 | [20.7193, 24.9668] |
| distance_keeping.mean_time_gap_min_s | 4.6913 | [3.2815, 6.67] |

**LATERAL** — n_windows 282, tier `T1`

| component | value | 95 % CI (episode-cluster bootstrap) |
|---|---|---|
| heading_mae_deg | 2.462 | [1.6256, 3.5707] |
| yaw_rate_mae_degps | 2.1993 | [1.7042, 2.6873] |
| curvature_mae_1pm | 0.007205 | [0.004627, 0.010071] |
| curvature_bias_1pm | 0.001327 | [-0.001331, 0.004412] |
| cross_mae_m | 0.2257 | [0.1772, 0.2788] |
| cross_bias_m | 0.0629 | [0.0071, 0.1221] |
| cross_final_mae_m | 0.5888 | [0.459, 0.732] |
| n_steps_heading | 2699 | no interval emitted for this component |
| n_steps_curvature | 2427 | no interval emitted for this component |
| n_steps_yaw_rate | 2427 | no interval emitted for this component |
| excluded_below_min_ds | 121 | no interval emitted for this component |
| min_ds_m | 0.1 | no interval emitted for this component |
| min_ds_mps | 0.5 | no interval emitted for this component |

## 4. Controls — including the one that must read a known value

⛔ A model number without its control is unreadable. MEASURED 2026-08-23: a T1 arm read 9.3697 m while its own hold-action control read 0.4246 m on the SAME windows — the trivial baseline beat the model 22x, and an artifact reporting only the model's value looked like a result.

### 4.1 `const0` — the constant-only control  ·  **OK**

*Definition:* a predictor that emits the SAME output for every input: identically zero ego-frame displacement at every slot. *Evidence class:* MEASURED (ours; dump C:\Users\Admin\refav1_eval_slice\thor\full). n = 282 windows / 141 episodes.

| check | expected (known value) | measured | tolerance | pass |
|---|---|---|---|---|
| const0 paired against ITSELF, episode-cluster bootstrap | `delta 0.0, [0.0, 0.0], separated False` | `delta 0.0, [0.0, 0.0], separated False` | **NONE — this one is bit-exact** | **PASS** |
| ade_m — mean over (window, slot) of ‖GT‖, float64 numpy | `12.732511` | `12.732511` | rel 0.0001 (|Δ| 4.99e-07) | **PASS** |
| LON_speed_mae_mps — mean over (window, slot) of the GT speed — a zero path has speed 0 everywhere, so its MAE IS the GT speed | `11.569022` | `11.569022` | rel 0.0001 (|Δ| 0) | **PASS** |

> a control that must read a KNOWN value is the only thing that catches a metric pipeline which is confidently wrong. If this block does not pass, THE HARNESS IS WRONG, NOT THE MODEL.

### 4.2 The trivial and nav controls

| control | definition | tier | why it is here |
|---|---|---|---|
| `ha0` | constant velocity at the MEASURED v0 — a = 0, kappa = 0, integrated on the corpus's native 0.1 s tick and index-selected onto the scoring grid | `T1` | the strongest trivial baseline, and the ONLY arm that is bit-comparable across models — identically defined, identical windows, identical integrator, and exactly zero in either action unit |
| `ha` | hold the last OBSERVED (a, steer) | `T1` | ⭐ the reason ha0 exists: holding a NOISY observed steer drifts, so an arm that does nothing can beat `ha` and read as lateral skill. A win over `ha` alone is not skill. |
| `os_navshuf` | withholds the PAIRING between a window and its nav token; the nav MARGINAL is preserved exactly | `T1` | is the model using *this* window's nav? |
| `os_navzero` | withholds the SIGNAL — nav_cmd=None, the model's own documented no-nav path. This is the DEPLOYMENT condition, because the nav token is an ORACLE (provenance ego-future) that will not exist there. | `T1` | what is the model worth WITHOUT the oracle nav — i.e. at deployment |

> ⛔ a shuffle cannot stand in for a zero. MEASURED elsewhere: a tactical head ranked turns at AUC 0.873 under true nav and collapsed to 0.520 (chance) under nav_zero.

## 5. Arms — tier and evidence class, every one

| arm | tier | regime | tier status | evidence class | meaning |
|---|---|---|---|---|---|
| `cl` | `T1` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | — |
| `ha` | `T1` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | — |
| `ha0` | `T1` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | — |
| `ol` | `T0` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | — |
| `const0` | `T1` | OPEN LOOP | VACUOUS — consumes no input at all | MEASURED (ours; dump C:\Users\Admin\refav1_eval_slice\thor\full) | a predictor that emits the SAME output for every input: identically zero ego-frame displacement at every slot |

## 6. Protocol, parity and the leak guards

| field | value |
|---|---|
| `tier` | T1 |
| `loop_class` | OPEN LOOP |
| `inference_inputs` | vision (the OBSERVED window only), the clip's nav/route token, and the measured v0 at t0. ⛔ Nothing after the window origin reaches the forward pass; future poses are read ONLY to build targets. |
| `vision_only` | PARTIAL BY DESIGN, and stated rather than claimed: the forward pass also consumes a route token and the measured v0. v0 at its own cycle time is admissible (PI ruling 2026-09-02); the route token is admissible as a goal input (PI 2026-08-03) but is an ORACLE here, which is why the nav-zero arm is reported beside it as the deployment condition. |
| `goal_source` | the clip's v7.2 nav_command token (provenance: ego-future, oracle=True). ⛔ It is NOT computed from any situation classifier's output, in any form — the goal path and the situation path are information-disjoint at inference (PI 2026-08-03, binding). |
| `goal_situation_disjoint` | True |
| `labels_may_use_ego` | yes — label derivation may use ego and privileged channels (PI 2026-08-03). The restriction is on INFERENCE only. |

| parity field | value |
|---|---|
| `parity_status` | NON-PARITY (assume non-parity unless the run's config.json v2_parity says otherwise) |
| `parity_key` | None |
| `corpus` | UNDECLARED — pass --corpus |
| `n_episodes_scored` | 141 |
| `n_windows_scored` | 282 |
| `window_stride` | 1 |
| `grid` | 2s |
| `labels_md5` | None |
| `train_eval_disjoint` | UNVERIFIED BY THIS TOOL |

> ⛔ THIS RUN IS NON-PARITY unless `parity_status` says otherwise. refcv3's own config.json carries `v2_parity.parity false`, `checked false` and `corpus_key null`, and the trainer prints a non-parity warning on line 2 of its train.log. ⇒ refcv3 is NOT cross-arm comparable with refc-base or refc-xl, and a reader must not assume the programme's usual parity. The ONLY admissible cross-model statistic here is each arm's MARGIN over the shared `ha0` floor, which is bit-identically defined everywhere — never a level against another model's level.

> ⛔ train ∩ eval = ∅ is a property of the RUN, not of this tool, and this tool cannot check it. Assert it before quoting anything (GATE 3).

## 7. Honest gaps — KPIs this run could NOT measure

a KPI that could not be measured is NAMED with its reason and its n. An eval that silently omits one reads at a glance exactly like an eval that has no gap.

| criterion | family | state | reason | work item |
|---|---|---|---|---|
| `strat.decision / strat.route_goal` | STRATEGIC | **REFUSED** | strategic decisions not present in the scored pass (missing ['route_pred', 'route_gt']). A world-model FIDELITY pass does not traverse the hierarchy — run_one prints this explicitly. Producing this family needs a hierarc | the route head's sidecar block, or an external corpus — PhysicalAI-AV carries no map, lane graph, junction label or route signal |

## 8. Criteria-completeness check

Registry `products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.5.0 · scope **IN_SCOPE** (matched key_present='four_families')

**VIOLATIONS (silently absent, required): 0** · **WORK ITEMS (refused with a reason, or partial): 3**

**LONGITUDINAL** — 4/4 present

| criterion | state | detail |
|---|---|---|
| target-speed accuracy | **PRESENT** | four_families.longitudinal.speed_mae_mps |
| along-track displacement error | **PRESENT** | four_families.longitudinal.along_mae_m |
| ego progress | **PRESENT** | four_families.longitudinal.ego_progress: status=OK |
| distance-keeping (headway / time-gap / TTC to the lead agent) | **PRESENT** | four_families.longitudinal.distance_keeping: status=OK |

**LATERAL** — 4/4 present

| criterion | state | detail |
|---|---|---|
| cross-track error | **PRESENT** | four_families.lateral.cross_mae_m |
| heading error | **PRESENT** | four_families.lateral.heading_mae_deg |
| curvature error | **PRESENT** | four_families.lateral.curvature_mae_1pm |
| yaw-rate error | **PRESENT** | four_families.lateral.yaw_rate_mae_degps |

**TACTICAL** — 3/3 present

| criterion | state | detail |
|---|---|---|
| manoeuvre-decision quality (selected vs executed) | **PRESENT** | four_families.tactical.lateral_decision: status=OK |
| confusion over the manoeuvre classes | **PRESENT** | four_families.tactical.lateral_decision.confusion_gt_rows_pred_cols |
| tactical goal / anchor selection quality | **PRESENT** | four_families.tactical.goal_setting: status=OK |

**STRATEGIC** — 0/2 present

| criterion | state | detail |
|---|---|---|
| strategic decision quality | **REFUSED** | four_families.strategic: strategic decisions not present in the scored pass (missing ['route_pred', 'route_gt']). A world-model FIDELITY pass does not traverse the hierarchy — run_one prints this expl |
| strategic route / goal-setting quality | **REFUSED** | four_families.strategic: strategic decisions not present in the scored pass (missing ['route_pred', 'route_gt']). A world-model FIDELITY pass does not traverse the hierarchy — run_one prints this expl |

**ARTIFACT HYGIENE**

| criterion | state | detail |
|---|---|---|
| tier stamp present | **PRESENT** | T1 (T1) |
| estimator named | **PRESENT** | estimator.interval |
| n printed (windows and episodes) | **PRESENT** | n_windows |
| overlapping_holdout_se not used as a decision-grade interval | **PRESENT** | named only to disavow or deprecate it (estimator.forbidden_estimator) |
| corpus identity recorded (parity key + skip-hash) | **PRESENT** | parity |
| headline metrics reported against their trivial control(s) | **PRESENT** | floors |

**LEAK GUARDS**

| criterion | state | detail |
|---|---|---|
| Inference uses VISION ONLY. Labels may use ego/privileged data (PI, 2026-08-03, binding). | **PRESENT** | protocol.inference_inputs |
| The goal path and the situation path stay information-disjoint at inference; a goal input must not carry the situation classifier's output in any form (PI, 2026-08-03, binding). | **PRESENT** | protocol.goal_source |
| A route/goal head scoring near 1.0 must publish its echo test against its own input. | **REFUSED** | strategic.echo_test: the arm record carries no route-head conditionings, so no echo index against the fed nav token can be formed. ⛔ A route/goal score is INADMISSIBLE without it — WORK ITEM, not a pa |

## 9. Vocabulary — the regime label, and where the repo is still stale

T1 is the doctrine's machine-readable stamp and the criteria registry keys on it. The LETTER is kept; the doctrine's PROSE for that row is superseded by the ruling above and is reported, not edited, in stale_loop_labels.

| location | symbol | what it still says | action |
|---|---|---|---|
| `taniteval/tools/t1_eval.py` | `_TIER_NOTE / _tier_doctrine` | describes T1 with the superseded regime phrase | REPORTED, not edited — separate stream (PI ruling 2026-09-02) |
| `products/P7-TanitEval/CRITERIA_REGISTRY.json` | `tiers.values.T1.note` | describes T1 with the superseded regime phrase, as the PRIMARY driving tier | REPORTED, not edited — the registry is the EvalFlyWheel's |
| `Project Steering/EVAL_DOCTRINE.md` | `the T1 row` | defines T1 with the superseded regime phrase | REPORTED, not edited — doctrine change is a PI/Master-Mind call |

### 9.1 A second, different mismatch — checker wording vs emitter wording

- **`taniteval/four_families.py` · `longitudinal.anti_echo.holdv0_baseline.estimator`** — the checker's exoneration list carries 'not used' / 'never used' / 'is not' / 'must not' / 'deprecated' / 'forbidden' / 'banned', and none of them matches the emitter's own 'NOT <estimator>' phrasing — so a correct, explicit disavowal is scored as a USE
  - *Consequence:* hyg.no_forbidden_estimator reads MISSING on an artifact whose estimator discipline is exemplary
  - *Action:* REPORTED, not edited. This suite republishes the string with the programme's canonical disavowal clause appended (recorded in four_families._disavowal_annotations) rather than touching either the emitter or the checker.

Sites annotated on republication: **2**. these strings already disavowed the forbidden estimator in the EMITTER's wording; the criteria checker's exoneration list does not carry that phrasing, so a correct disavowal scored as a use. The clause is appended on republication, the sites are listed here, and the mismatch is reported as a finding — neither the emitter nor the checker was edited.

## 10. Provenance

```json
{
 "source": "BANKED RECORD (another stream)",
 "path": "C:\\Users\\Admin\\refav1_eval_slice\\thor\\refav1_t1_full_21109.json",
 "sha256": "4b23ef0d69992bd5bf1956405a7ab4cb209e11e3965fcf86d97b38d3738d7b4a",
 "gpu_used": "none — the record was read, not recomputed",
 "stack_pin": null,
 "resolved_trees": {
  "tanitad": "C:\\Users\\Admin\\tanitad-wt\\stack\\tanitad\\__init__.py",
  "tanitad_version": "0.0.1",
  "taniteval": "C:\\Users\\Admin\\tanitad-wt\\taniteval\\taniteval\\__init__.py",
  "taniteval_version": "0.1.0",
  "torch": "C:\\Users\\Admin\\venvs\\tanitad\\Lib\\site-packages\\torch\\__init__.py",
  "torch_version": "2.11.0+cu128",
  "numpy": "C:\\Users\\Admin\\venvs\\tanitad\\Lib\\site-packages\\numpy\\__init__.py",
  "numpy_version": "2.5.1"
 }
}
```

*Every number above is `MEASURED (ours)` through `taniteval/tools/refcv3_arm.py` → `taniteval/tools/t1_eval.py::analyze` → `taniteval/four_families.py` / `taniteval/ci.py`. This tool computes no geometry of its own.*
