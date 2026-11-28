# Open-loop binding-KPI suite — `refcv3-40284-openloop`

**Instrument:** `taniteval/tools/openloop_suite.py` · **generated:** 2026-09-04T04:10:38Z · **schema:** `taniteval.openloop_suite/1`
**Regime:** **OPEN LOOP** — every arm below. PI, 2026-09-02: a predictor consuming the planner's own output is STILL open loop, because the model's trajectory does not affect the ego data it is next fed. The other regime means the trajectory controls a vehicle in a simulator (AlpaSim) or a real test vehicle.

## 1. Headline — does the model beat its trivial control?

The question is **does `os` beat the trivial control `ha0`, per family, on the same windows, paired?** — not *what is the ADE*. ⛔ A paired interval that excludes zero while favouring the FLOOR means the trivial baseline WON. Render that as LOST, never as a tie.

| family | verdict vs `ha0` (oracle nav) | verdict vs `ha0` (deployment, nav withheld) |
|---|---|---|
| **ADE** | WON: 2/2 WON | WON: 2/2 WON |
| **LONGITUDINAL** | LOST: 2/3 WON, 1/3 LOST — the trivial control won where the arm LOST | LOST: 1/3 WON, 1/3 LOST, 1/3 TIED — the trivial control won where the arm LOST |
| **LATERAL** | LOST: 2/3 WON, 1/3 LOST — the trivial control won where the arm LOST | LOST: 2/3 WON, 1/3 LOST — the trivial control won where the arm LOST |
| **TACTICAL** | MIXED: 1/2 WON, 1/2 TIED | MIXED: 1/2 WON, 1/2 TIED |
| **STRATEGIC** | MIXED: 3/4 WON, 1/4 TIED | covered in-family by the `nav_zero` conditioning row above |

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
| `os` | 4,823 | 0.0 | 0.0 | 0.0 |
| `ha` | 4,823 | 0.0402 | 0.0319 | 0.0311 |
| `ha0` | 4,823 | 1.0 | 1.0 | 1.0 |
| `os_navshuf` | 4,823 | 0.0 | 0.0 | 0.0 |
| `os_navzero` | 4,823 | 0.0 | 0.0 | 0.0 |
| `oracle_sel` | 4,823 | 0.0 | 0.0 | 0.0 |

> Trivial profile: no arm besides the floor is degenerate. (`ha0` is expected here — it *is* the constant-velocity plan.)

**Selection profile** — distinct anchors selected: `50`, modal anchor `57` at `0.1482`, entropy `2.8425`.

> ⭐ The trivial profile is **blind** to this model's characteristic degeneracy: a constant anchor is neither straight nor constant-speed, so `trivial_frac` reads 0 while the selection carries zero scene information. Both gates are required.

## 3. The four families — reported separately, never pooled

Sayed 2026-08-02, binding: LONGITUDINAL + LATERAL + TACTICAL + STRATEGIC in ADDITION to ADE, per family, NEVER pooled. This suite emits no composite score and refuses to invent one. A family reported REFUSED/UNAVAILABLE is a WORK ITEM, not a pass.

Estimator: **paired_episode_cluster_bootstrap — two arms on the SAME windows, resampled together so the shared per-window difficulty cancels inside each draw**, n_boot 2000, seed 0, cluster unit **the episode (windows inside one clip are dependent)**. overlapping_holdout_se is NOT used anywhere in this suite. It is not a jackknife, it is anti-conservative, and it BIASES THE POINT ESTIMATE bidirectionally — measured up to a sign flip on a paired delta. Named here only to disavow it.

### 3.1 ADE

**WON: 2/2 WON** · contrast `os - ha0` · this family is reported on its own; no composite over families is computed anywhere in this suite

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| ADE (mean L2 over the grid) | `-0.2304  [-0.2881, -0.1781]` | 4,823 | 141 | **WON** |
| FDE (final displacement) | `-0.4741  [-0.5997, -0.3600]` | 4,823 | 141 | **WON** |

*Deployment condition (nav withheld):* **WON: 2/2 WON**

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| ADE (mean L2 over the grid) | `-0.2064  [-0.2673, -0.1479]` | 4,823 | 141 | **WON** |
| FDE (final displacement) | `-0.4373  [-0.5667, -0.3141]` | 4,823 | 141 | **WON** |

### 3.2 LONGITUDINAL

**LOST: 2/3 WON, 1/3 LOST — the trivial control won where the arm LOST** · contrast `os - ha0` · this family is reported on its own; no composite over families is computed anywhere in this suite

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| target-speed MAE (m/s) | `-0.0364  [-0.0636, -0.0079]` | 4,823 | 141 | **WON** |
| along-track MAE (m) | `-0.0674  [-0.0923, -0.0415]` | 4,823 | 141 | **WON** |
| acceleration MAE (m/s²) | `+0.2020  [0.1713, 0.2342]` | 4,823 | 141 | **LOST** |

*Deployment condition (nav withheld):* **LOST: 1/3 WON, 1/3 LOST, 1/3 TIED — the trivial control won where the arm LOST**

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| target-speed MAE (m/s) | `-0.0134  [-0.0437, 0.0179]` | 4,823 | 141 | **TIED** |
| along-track MAE (m) | `-0.0429  [-0.0715, -0.0117]` | 4,823 | 141 | **WON** |
| acceleration MAE (m/s²) | `+0.2226  [0.1893, 0.2561]` | 4,823 | 141 | **LOST** |

### 3.3 LATERAL

**LOST: 2/3 WON, 1/3 LOST — the trivial control won where the arm LOST** · contrast `os - ha0` · this family is reported on its own; no composite over families is computed anywhere in this suite

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| cross-track MAE (m) | `-0.2048  [-0.2686, -0.1500]` | 4,823 | 141 | **WON** |
| heading MAE (deg) | `-1.3741  [-1.7569, -1.0418]` | 4,548 | 141 | **WON** |
| yaw-rate MAE (rad/s) | `+0.1700  [0.1006, 0.2534]` | 4,823 | 141 | **LOST** |

*Deployment condition (nav withheld):* **LOST: 2/3 WON, 1/3 LOST — the trivial control won where the arm LOST**

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| cross-track MAE (m) | `-0.2034  [-0.2669, -0.1489]` | 4,823 | 141 | **WON** |
| heading MAE (deg) | `-1.3562  [-1.7376, -1.0191]` | 4,550 | 141 | **WON** |
| yaw-rate MAE (rad/s) | `+0.1957  [0.1186, 0.2864]` | 4,823 | 141 | **LOST** |

### 3.4 TACTICAL

**MIXED: 1/2 WON, 1/2 TIED** · contrast `os - ha0` · this family is reported on its own; no composite over families is computed anywhere in this suite

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| lateral manoeuvre agreement (fraction) | `+0.0881  [0.0580, 0.1233]` | 4,823 | 141 | **WON** |
| longitudinal manoeuvre agreement (fraction) | `-0.0100  [-0.0356, 0.0162]` | 4,823 | 141 | **TIED** |

*Deployment condition (nav withheld):* **MIXED: 1/2 WON, 1/2 TIED**

| metric | Δ (arm − floor) with 95 % CI | n windows | n eps | verdict |
|---|---|---|---|---|
| lateral manoeuvre agreement (fraction) | `+0.0875  [0.0576, 0.1212]` | 4,823 | 141 | **WON** |
| longitudinal manoeuvre agreement (fraction) | `-0.0108  [-0.0367, 0.0156]` | 4,823 | 141 | **TIED** |

### 3.5 STRATEGIC

**MIXED: 3/4 WON, 1/4 TIED** · contrast `accuracy vs the MAJORITY-CLASS rate (the no-information value for a classifier), plus the nav controls` · reported on its own; STRATEGIC has no paired block against `ha0` because the constant-velocity floor has no route head

| metric | value or Δ, with 95 % CI (see each row) | n windows | n eps | verdict |
|---|---|---|---|---|
| route accuracy — nav_true | `0.7667  [0.7097, 0.8224]  vs no-info 0.6742` | 3,622 | 128 | **WON** |
| route accuracy — nav_shuffled | `0.7667  [0.7097, 0.8224]  vs no-info 0.6742` | 3,622 | 128 | **WON** |
| route accuracy — nav_zero | `0.7667  [0.7097, 0.8224]  vs no-info 0.6742` | 3,622 | 128 | **WON** |
| nav dependence — true minus shuffled accuracy (paired) | `+0.0000  [0.0000, 0.0000]` | 3,622 | 128 | **TIED** |

### 3.6 Per-family absolute readings (headline arm)

**LONGITUDINAL** — n_windows 4,823, tier `T1`

| component | value | 95 % CI (episode-cluster bootstrap) |
|---|---|---|
| speed_mae_mps | 0.4516 | [0.4197, 0.4828] |
| speed_bias_mps | 0.0327 | [-0.0106, 0.0744] |
| speed_rmse_mps | 0.7384 | [0.6836, 0.7917] |
| along_mae_m | 0.403 | [0.3722, 0.4332] |
| along_bias_m | 0.0372 | [-0.0047, 0.0799] |
| along_final_bias_m | 0.0706 | [-0.0187, 0.1603] |
| accel_mae_mps2 | 0.6806 | [0.6464, 0.7156] |
| ego_progress | OK | see the JSON record for this sub-block |
| distance_keeping | OK | see the JSON record for this sub-block |
| anti_echo | OK | see the JSON record for this sub-block |
| target_speed_acc.within_0.5_mps | 0.7135 | [0.6916, 0.7357] |
| target_speed_acc.within_1.0_mps | 0.8784 | [0.8632, 0.8936] |
| target_speed_acc.within_2.0_mps | 0.9713 | [0.9647, 0.9776] |
| ego_progress.progress_ratio_mean | 1.0031 | [0.9886, 1.0144] |
| ego_progress.progress_ratio_median | 1.0007 | [0.9976, 1.0035] |
| ego_progress.progress_error_mean | 0.0807 | [0.0656, 0.0993] |
| ego_progress.under_progress_rate | 0.4915 | [0.4545, 0.5275] |
| ego_progress.gt_progress_mean_m | 23.8025 | [21.4717, 26.342] |
| ego_progress.t0_axis_gt_self_ratio | 0.9892 | [0.9738, 0.997] |
| distance_keeping.mean_headway_min_m | 28.0645 | [24.0239, 32.3241] |
| distance_keeping.mean_time_gap_min_s | 3.9744 | [3.2122, 4.8629] |
| distance_keeping.mean_min_ttc_s | 24.9877 | [23.5602, 26.316] |

**LATERAL** — n_windows 4,823, tier `T1`

| component | value | 95 % CI (episode-cluster bootstrap) |
|---|---|---|
| heading_mae_deg | 1.3109 | [0.8087, 2.2671] |
| yaw_rate_mae_degps | 1.794 | [1.5761, 2.0309] |
| curvature_mae_1pm | 0.008815 | [0.006313, 0.012054] |
| curvature_bias_1pm | -0.0002 | [-0.001206, 0.000669] |
| cross_mae_m | 0.1084 | [0.0958, 0.1222] |
| cross_bias_m | -0.0027 | [-0.0154, 0.0099] |
| cross_final_mae_m | 0.2337 | [0.205, 0.2648] |
| n_steps_heading | 18115 | no interval emitted for this component |
| n_steps_curvature | 13538 | no interval emitted for this component |
| n_steps_yaw_rate | 13538 | no interval emitted for this component |
| excluded_below_min_ds | 1177 | no interval emitted for this component |
| min_ds_m | 0.25 | no interval emitted for this component |
| min_ds_mps | 0.5 | no interval emitted for this component |

## 4. Controls — including the one that must read a known value

⛔ A model number without its control is unreadable. MEASURED 2026-08-23: a T1 arm read 9.3697 m while its own hold-action control read 0.4246 m on the SAME windows — the trivial baseline beat the model 22x, and an artifact reporting only the model's value looked like a result.

### 4.1 `const0` — the constant-only control  ·  **OK**

*Definition:* a predictor that emits the SAME output for every input: identically zero ego-frame displacement at every slot. *Evidence class:* MEASURED (ours; dump /workspace/eval/refcv3_40284_dump). n = 4,823 windows / 141 episodes.

| check | expected (known value) | measured | tolerance | pass |
|---|---|---|---|---|
| const0 paired against ITSELF, episode-cluster bootstrap | `delta 0.0, [0.0, 0.0], separated False` | `delta 0.0, [0.0, 0.0], separated False` | **NONE — this one is bit-exact** | **PASS** |
| ade_m — mean over (window, slot) of ‖GT‖, float64 numpy | `14.248286` | `14.248286` | rel 0.0001 (|Δ| 1.07e-07) | **PASS** |
| LON_speed_mae_mps — mean over (window, slot) of the GT speed — a zero path has speed 0 everywhere, so its MAE IS the GT speed | `11.395674` | `11.395674` | rel 0.0001 (|Δ| 0) | **PASS** |

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
| `os` | `T1` | OPEN LOOP | UNRULED — see tier_ruling | MEASURED (ours; the banked dump + this record) | T1 (RULING OPEN) — ONE forward pass at t0: observed frames + the clip's v7.2 nav token + the MEASURED v0, path = the model's OWN sel_score_v3 selection out['tra |
| `ha` | `T1` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | T1 — HOLD-ACTION control: the (a, steer) that CLOSES at t0 held for the horizon. Consumes no recorded future. NOT the floor — a held noisy steer drifts, so this |
| `ha0` | `T1` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | T1 — ⭐ CONSTANT VELOCITY: a = 0, kappa = 0 at the measured v0, i.e. a straight line at constant speed. The STRONGEST TRIVIAL BASELINE, the echo test's real bar, |
| `os_navshuf` | `T1` | OPEN LOOP | UNRULED — see tier_ruling | MEASURED (ours; the banked dump + this record) | T1 (RULING OPEN) — as os with nav_cmd PERMUTED across the eval windows: breaks the PAIRING while preserving the nav marginal. Answers 'is the model using THIS w |
| `os_navzero` | `T1` | OPEN LOOP | UNRULED — see tier_ruling | MEASURED (ours; the banked dump + this record) | T1 (RULING OPEN) — ⭐ as os with nav WITHHELD (nav_cmd=None): the E13 injection into the tactical and strategic layers is SKIPPED ENTIRELY and the core collapses |
| `oracle_sel` | `T0` | OPEN LOOP | stamped | MEASURED (ours; the banked dump + this record) | T0 — the a_star-selected anchor's refinement: a_star is the GT-NEAREST anchor (refc_v3_train.py:460), so this is the CEILING, not a driveable arm. Never compare |
| `const0` | `T1` | OPEN LOOP | VACUOUS — consumes no input at all | MEASURED (ours; dump /workspace/eval/refcv3_40284_dump) | a predictor that emits the SAME output for every input: identically zero ego-frame displacement at every slot |

> ⚠️ **THE TIER RULING IS OPEN — decided by PI / Master Mind — NOT a FlyWheel, not by this instrument.**
>
> *The question:* does EVAL_DOCTRINE admit as T1 a model that consumes NO actions at all? Its T1 row reads 'the predictor consumes the decoder/planner's own actions', which does not literally cover refcv3.
>
> *Benchmarks' recommendation, flagged as a recommendation:* ADMIT IT, flagged as a recommendation: the doctrine's PURPOSE is to keep future information out of inference, and a correctly-gated refcv3 forward pass (frames <= t0, nav token, measured v0, selection by sel_score_v3 and never by a_star) admits none — but keep the DISTINCT arm name `os` so no reader believes two `cl` columns describe the same procedure.
>
> ⭐ **The margin framing is what makes this survivable either way.** `os − ha0` is a difference of two arms measured on the same windows with the same instrument; it stays meaningful whatever tier label the ruling attaches.

**ABSENT arms** — not missing, not skipped, not a work item: structurally non-existent for this model.

| arm | tier if it existed | why it does not exist | use instead |
|---|---|---|---|
| `ol` | `T0` | `ol` is 'the RECORDED future (a, steer) integrated from v0'. refcv3 CONSUMES NO ACTIONS (refc_v3.py:480 — the forward signature has no action argument), so integrating the recorded actions is not a rollout OF THIS MODEL: it is a property of the corpus and of the unicycle, identical for every refcv3 checkpoint ever trained. Emitting it under this model's name would put the same name on two differen | use `ha0` — the ONLY bit-comparable arm across refav1 and refcv3 — as the shared floor; the kinematic-contract control that `ol` provides lives in refav1_arm.py |

## 6. Protocol, parity and the leak guards

| field | value |
|---|---|
| `tier` | T1 |
| `loop_class` | OPEN LOOP |
| `inference_inputs` | the OBSERVED window's frames (vision); the MEASURED v0 at t0 (PI ruling 2026-09-02); the clip's v7.2 nav token (goal input, PI 2026-08-03); NOTHING recorded after the window origin on any arm here |
| `vision_only` | vision + v0(t0) + nav token — both additions admitted by the two PI rulings above; no ego state beyond v0, no future |
| `goal_source` | the model's own tactical goal head (E4/E9), decoded from vision + nav + v0 inside the forward pass; the selection it grafts is banked per window (sel_score_v3) |
| `goal_situation_disjoint` | True by construction: nav is the v7.2 labels blob's nav_command token, not a situation classifier output |
| `labels_may_use_ego` | yes — label derivation may use ego and privileged channels (PI 2026-08-03). The restriction is on INFERENCE only. |

| parity field | value |
|---|---|
| `parity_status` | NON-PARITY - MEASURED from the run's own config.json: v2_parity.parity false, checked false, corpus_key null; the trainer prints its own non-parity warning on line 2 of train.log |
| `parity_key` | eval slice; the run did NOT assert corpus parity |
| `corpus` | PhysicalAI B1 v7.2 EVAL - 141 clips with pixels of 147 |
| `n_episodes_scored` | 141 |
| `n_windows_scored` | 4823 |
| `window_stride` | 5 |
| `grid` | 2s |
| `labels_md5` | aa12c948f062181c3297265b51526ec5 |
| `train_eval_disjoint` | ARITHMETIC EVIDENCE, not a check: config.json train cache /root/data/train holds 4,572 clips = 4,713 - 141 eval clips. v2_parity.checked is FALSE, so the trainer verified nothing - assert by episode id before quoting (GATE 3). |

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
 "source": "ROLLED OUT by taniteval/tools/refcv3_arm.py::run_dump",
 "dump_dir": "/workspace/eval/refcv3_40284_dump",
 "manifest_episodes": 141,
 "gpu_used": "cuda",
 "stack_pin": null,
 "resolved_trees": {
  "tanitad": "/workspace/TanitAD/stack/tanitad/__init__.py",
  "tanitad_version": "0.0.1",
  "taniteval": "/workspace/TanitAD/taniteval/taniteval/__init__.py",
  "taniteval_version": "0.1.0",
  "torch": "/usr/local/lib/python3.12/dist-packages/torch/__init__.py",
  "torch_version": "2.8.0+cu128",
  "numpy": "/usr/local/lib/python3.12/dist-packages/numpy/__init__.py",
  "numpy_version": "2.1.2"
 }
}
```

*Every number above is `MEASURED (ours)` through `taniteval/tools/refcv3_arm.py` → `taniteval/tools/t1_eval.py::analyze` → `taniteval/four_families.py` / `taniteval/ci.py`. This tool computes no geometry of its own.*
