# Open-loop binding-KPI suite — `suite-refav1-21109-cos-ext`

**Instrument:** `taniteval/tools/openloop_suite.py` · **generated:** 2026-09-05T02:36:13Z · **schema:** `taniteval.openloop_suite/1`
**Regime:** **OPEN LOOP** — every arm below. PI, 2026-09-02: a predictor consuming the planner's own output is STILL open loop, because the model's trajectory does not affect the ego data it is next fed. The other regime means the trajectory controls a vehicle in a simulator (AlpaSim) or a real test vehicle.

## 1. Headline — does the model beat its trivial control?

The question is **does `cl` beat the trivial control `ha0`, per family, on the same windows, paired?** — not *what is the ADE*. ⛔ A paired interval that excludes zero while favouring the FLOOR means the trivial baseline WON. Render that as LOST, never as a tie.

| family | verdict vs `ha0` (oracle nav) | verdict vs `ha0` (deployment, nav withheld) |
|---|---|---|
| **ADE** | LOST: 1/2 LOST, 1/2 TIED — the trivial control won where the arm LOST | — |
| **LONGITUDINAL** | LOST: 2/3 LOST, 1/3 TIED — the trivial control won where the arm LOST | — |
| **LATERAL** | TIED: 3/3 TIED — no separation from the control | — |
| **TACTICAL** | TIED: 2/2 TIED — no separation from the control | — |
| **STRATEGIC** | LOST: 2/4 WON, 1/4 LOST, 1/4 TIED — the trivial control won where the arm LOST | covered in-family by the `nav_zero` conditioning row above |

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
| `cl` | 282 | 1.0 | 0.9574 | 0.9574 |
| `ha` | 282 | 0.0248 | 0.0177 | 0.0177 |
| `ha0` | 282 | 1.0 | 1.0 | 1.0 |
| `ol` | 282 | 0.0248 | 0.0177 | 0.0177 |
| `ha0_ext` | 282 | 0.0319 | 0.0177 | 0.0177 |

> ⛔ **VOID-RISK:** `['cl']` are the constant-velocity plan on more than half the windows. A read whose arm IS the baseline is VOID, never 'no difference'. ⇒ read every LATERAL row below as a CONTROL, never as planning skill.

**Selection profile:** not available

## 3. The four families — reported separately, never pooled

Sayed 2026-08-02, binding: LONGITUDINAL + LATERAL + TACTICAL + STRATEGIC in ADDITION to ADE, per family, NEVER pooled. This suite emits no composite score and refuses to invent one. A family reported REFUSED/UNAVAILABLE is a WORK ITEM, not a pass.

Estimator: **paired_episode_cluster_bootstrap — two arms on the SAME windows, resampled together so the shared per-window difficulty cancels inside each draw**, n_boot 2000, seed 0, cluster unit **the episode (windows inside one clip are dependent)**. overlapping_holdout_se is NOT used anywhere in this suite. It is not a jackknife, it is anti-conservative, and it BIASES THE POINT ESTIMATE bidirectionally — measured up to a sign flip on a paired delta. Named here only to disavow it.

### 3.1 ADE

**LOST: 1/2 LOST, 1/2 TIED — the trivial control won where the arm LOST** · contrast `cl - ha0` · this family is reported on its own; no composite over families is computed anywhere in this suite

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| ADE (mean L2 over the grid) | `+0.0158  [0.0007, 0.0315]` | 282 | 141 | **LOST** |
| FDE (final displacement) | `+0.0354  [-0.0044, 0.0769]` | 282 | 141 | **TIED** |

### 3.2 LONGITUDINAL

**LOST: 2/3 LOST, 1/3 TIED — the trivial control won where the arm LOST** · contrast `cl - ha0` · this family is reported on its own; no composite over families is computed anywhere in this suite

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| target-speed MAE (m/s) | `+0.0308  [0.0049, 0.0585]` | 282 | 141 | **LOST** |
| along-track MAE (m) | `+0.0258  [0.0061, 0.0467]` | 282 | 141 | **LOST** |
| acceleration MAE (m/s²) | `+0.0238  [-0.0023, 0.0518]` | 282 | 141 | **TIED** |

### 3.3 LATERAL

**TIED: 3/3 TIED — no separation from the control** · contrast `cl - ha0` · this family is reported on its own; no composite over families is computed anywhere in this suite

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| cross-track MAE (m) | `+0.0000  [0.0000, 0.0000]` | 282 | 141 | **TIED** |
| heading MAE (deg) | `+0.0000  [0.0000, 0.0000]` | 272 | 140 | **TIED** |
| yaw-rate MAE (rad/s) | `+0.0000  [0.0000, 0.0000]` | 282 | 141 | **TIED** |

### 3.4 TACTICAL

**TIED: 2/2 TIED — no separation from the control** · contrast `cl - ha0` · this family is reported on its own; no composite over families is computed anywhere in this suite

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| lateral manoeuvre agreement (fraction) | `+0.0000  [0.0000, 0.0000]` | 282 | 141 | **TIED** |
| longitudinal manoeuvre agreement (fraction) | `-0.0035  [-0.0248, 0.0213]` | 282 | 141 | **TIED** |

### 3.5 STRATEGIC

**LOST: 2/4 WON, 1/4 LOST, 1/4 TIED — the trivial control won where the arm LOST** · contrast `accuracy vs the MAJORITY-CLASS rate (the no-information value for a classifier), plus the nav controls` · reported on its own; STRATEGIC has no paired block against `ha0` because the constant-velocity floor has no route head

| metric | value or Δ, with 95 % CI (see each row) | n windows | n eps | verdict |
|---|---|---|---|---|
| route accuracy — nav_true | `1.0  [1.0, 1.0]  vs no-info 0.6383` | 141 | 141 | **WON** |
| route accuracy — nav_shuffled | `0.4184  [0.3404, 0.5035]  vs no-info 0.6383` | 141 | 141 | **LOST** |
| route accuracy — nav_zero | `0.6383  [0.5603, 0.7163]  vs no-info 0.6383` | 141 | 141 | **TIED** |
| nav dependence — true minus shuffled accuracy (paired) | `+0.5816  [0.4965, 0.6596]` | 141 | 141 | **WON** |

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

*Definition:* a predictor that emits the SAME output for every input: identically zero ego-frame displacement at every slot. *Evidence class:* MEASURED (ours; dump C:\Users\Admin\ccos_eval\thor\dumps\dump_cos_ext). n = 282 windows / 141 episodes.

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
| `cl` | `T1` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | T1 — plan() at t0, TRUE nav; predictor consumes the planner's own actions; trajectory = unicycle(controls, measured v0) |
| `ha` | `T1` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | T1 — hold the last OBSERVED (a, kappa) (closes at t0) for K steps; consumes no recorded future |
| `ha0` | `T1` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | T1 — CONSTANT VELOCITY: a = 0, kappa = 0 at the measured v0, i.e. a straight line at constant speed. The STRONGEST TRIVIAL BASELINE and the echo test's real bar |
| `ol` | `T0` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | T0 — the RECORDED future (a, kappa) integrated from v0: the kinematic-contract control (must reproduce GT), NOT a WM diagnostic |
| `ha0_ext` | `T1` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | T1 — THE ECHO CONTROL (stack/tanitad/eval/echo_gate.py::ha0_ext, refav1 form): constant (a0, kappa0) from the MEASURED t0 state held for K steps. a0 = (v[2t] -  |
| `const0` | `T1` | OPEN LOOP | VACUOUS — consumes no input at all | MEASURED (ours; dump C:\Users\Admin\ccos_eval\thor\dumps\dump_cos_ext) | a predictor that emits the SAME output for every input: identically zero ego-frame displacement at every slot |

## 6. Protocol, parity and the leak guards

| field | value |
|---|---|
| `tier` | T1 |
| `loop_class` | OPEN LOOP |
| `inference_inputs` | cached frozen DINOv3 patch features of the OBSERVED window (vision); measured v0 at t0 (PI ruling 2026-09-02); the v7.2 nav token (goal input, PI 2026-08-03); NOTHING recorded after frame 2t on the T1 / hold-action arms |
| `vision_only` | vision + v0(t0) + nav token — both additions admitted by the two PI rulings above; no ego state beyond v0, no future |
| `goal_source` | per arm — cl: tactical_imagined 100.0 % [space tactical_query_field]. tactical_imagined = plan() imagined the goal from vision + nav + measured v0 (refa_v1.plan, e609a98); supplied = the TRUE future field (cl_oraclegoal, T0); none = goal-free cost (floor baseline by construction) |
| `goal_situation_disjoint` | True by construction: nav is the labels blob's nav_command token (allow_oracle_nav-stamped, Alpamayo-CoT+ego derivation), not a situation classifier output |
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

*No registry criterion was refused on this run.*

## 8. Criteria-completeness check

Registry `products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.5.0 · scope **IN_SCOPE** (matched key_present='four_families')

**VIOLATIONS (silently absent, required): 0** · **WORK ITEMS (refused with a reason, or partial): 0**

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

**STRATEGIC** — 2/2 present

| criterion | state | detail |
|---|---|---|
| strategic decision quality | **PRESENT** | four_families.strategic: status=OK |
| strategic route / goal-setting quality | **PRESENT** | four_families.strategic: status=OK |

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
| A route/goal head scoring near 1.0 must publish its echo test against its own input. | **PRESENT** | strategic.echo_test: status=OK |

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
 "path": "C:\\Users\\Admin\\ccos_eval\\thor\\rec_cos_ext.json",
 "sha256": "fba6a623067419c0768077e11d7451aa1ffdfeb269381fc7121b43cd1bfd7cdc",
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
