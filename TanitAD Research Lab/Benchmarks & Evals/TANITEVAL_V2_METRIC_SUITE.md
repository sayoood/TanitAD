# TanitEval v2 — the driving-capability metric suite

*Live design doc · v1 drafted 2026-07-21 · commissioned by Sayed · agent run, code + doc STAGED not pushed*
*Corpus for every "MEASURED (this doc)" number: `taniteval/results/windows_flagship-30k.pt` — 881 windows / 40 val episodes, `physicalai-val-0c5f7dac3b11`. Reproduce with `python taniteval/v2_tier0_probe.py --arm flagship-30k` (CPU, ~30 s, no pod).*

**Label legend, used on every number in this document.**
**MEASURED (ours)** — from `Project Steering/MODEL_REGISTRY.md`, a raw eval JSON, or this doc's own probe, always cited.
**PUBLISHED** — from an external paper/docs, always cited.
**PROPOSED** — a design choice or threshold with no measurement behind it yet. *Any threshold not carrying one of these three marks is a bug in this document.*

---

## TL;DR

1. **ADE's win over the floor is not the win we thought it was.** Paired episode-cluster bootstrap on flagship v1's own 881 eval windows: ADE beats CV by **+0.4106 m, CI [+0.2050, +0.6240], separated**; the **cross-track** half beats it by **+0.7720 [+0.4166, +1.1914], separated**; the **along-track** half by **+0.2543 [−0.0278, +0.5304] — NOT separated**; and **speed MAE by −0.0032 [−0.1285, +0.1182] — the model does not track speed better than constant velocity at all.** MEASURED (this doc). One scalar was hiding a competency the arm does not have.
2. **On cruise it is worse than doing nothing.** Split the corpus by realised longitudinal regime: on the 639 steady windows the model's speed MAE is **0.4231 m/s vs hold-v0's 0.2109** — paired Δ **−0.2122 [−0.2778, −0.1443], separated against us**. On the 95 braking and 147 accelerating windows it wins by +0.6433 and +0.5716, both separated. **Cruise quality and transient response point in opposite directions and ADE averages them away.** MEASURED (this doc).
3. **Most of the suite already exists.** `pathspeed.py` implements the Frenet long/lat split and the speed-decoupled path geometry; `refc_rescorer.py` implements the propose-vs-rank decomposition *including the chance floor and the span-normalised ranking skill* the community usually omits; `closedloop.py` implements the closed-loop drift panel; `efficiency.py` implements the deployment axis. **The work is promotion, normalisation and honest gating — not new formulas.**
4. **A tier-0 set runs today on CPU from committed artifacts.** The probe reproduces MODEL_REGISTRY §1.2 to 4 decimals (ADE 0.4271, CV 0.8377, long/lat RMSE 1.042/0.360, top-decile speed bias +0.659). Cost: seconds. Blocker: **only 2 of 10 registry arms have a window dump in the repo** (§7 E1).
5. **Four things on Sayed's list should not be built.** Headway/distance-keeping (no lead-agent state exists anywhere), any VTARGET-referenced 2 s target-speed metric (already refuted), intersection/roundabout capability at a 2 s horizon (the events are 5–20 s), and naive curvature MAE at the persisted resolution (measured to be 24× the signal). §6.
6. **The metric that most changes what we learn is HCS** — the hierarchical contribution score measured *on the scored decode path* with a **within-episode permutation control**. Our headline number is computed on a code path where zero conditioning seams participate (verified in source), so today we cannot settle the question this program exists to answer. §3.6.

---

## 0. Reading rules (binding for every metric below)

**R0 — every metric names its estimator.** The decision-grade interval is the **episode-cluster bootstrap** over the 40 val episodes (`taniteval/ci.py::episode_cluster_bootstrap`, B=2000); for two arms or an arm-vs-floor on the same windows it is the **paired** form. The legacy `heldout ± ci95` is `overlapping_holdout_se`, measured **1.28–2.06× too narrow** across 10 arms (MEASURED, `Project Steering/CI_RECOMPUTE_2026-07-20.json`). **No metric in this suite may depend on it.** Three modules still use it internally — `hierarchy.py::_jack`, `closedloop.py::_jack`, `bench.py::_agg` — and must be migrated (§7 E3).

**R1 — every metric names its floor, and the floor is the strongest trivial thing available.** Not chance, not zero.
| metric family | floor | availability |
|---|---|---|
| trajectory (ADE/FDE/Frenet) | best-of-3 kinematic (CV / go-straight / CTRV) | CV persisted; **CTRV + go-straight are computed in `bench.collect_full` and discarded by `save_windows`** (§7 E2) |
| longitudinal / speed | **hold-v0** (go straight at the observed entry speed) | constructible from the persisted `speed` column |
| lateral / geometry | CV and hold-v0 (both are straight lines → they are the honest "no lateral skill" bar) | persisted |
| classification (maneuver / route) | **majority-class base rate**, never 1/3 | needs the label join (§7 E4) |
| selection / ranking | **chance** = expected ADE of a uniformly random candidate | already implemented, `refc_rescorer.py:572` |
| *lane-centre* | **NOT AVAILABLE** — no lane geometry exists in our data (§2.3) | — |

**R2 — an unseparated win is a tie.** `bench.run` already enforces this for ADE-vs-CV (`beats_cv_separated`). Extend to every headline.

**R3 — report `n` per stratum, and mark strata below 30 windows low-confidence.** `pathspeed.run` already does (`low_confidence`, `min_n=30`).

**R4 — open-loop numbers are weak claims.** Every open-loop row carries `claim_strength: "open-loop / weak"` (already in `bench.run`'s protocol block). Our own evidence: open-loop **0.452 → closed-loop 1.685**, divergence >5 m on **22.2 %** of windows (MEASURED, MODEL_REGISTRY §1.2).

**R5 — mean is not always the right reducer.** MEASURED (this doc): heading MAE@2s has a bootstrap CI of **[2.34, 12.02]** around a mean of 6.61° — a handful of windows dominate. Heading, curvature and jerk metrics must be reported as **median + exceedance rate**, not mean. `ci.py`'s `REDUCERS` currently offers only `mean` and `rms`; a `statistic=` callable path is needed (§7 E3).

---

## 1. Why ADE alone failed — five reasons, each verified

| # | Failure | Evidence | Status |
|---|---|---|---|
| 1 | **ADE hides which competency is missing.** | **89.33 %** of flagship v1's 2 s squared error is along-track; long-RMSE 1.042 m vs lat-RMSE 0.360 m. At the top speed decile: speed over-prediction **+0.659 m/s**, long-RMSE **1.379 m**. MEASURED (this doc; reproduces MODEL_REGISTRY §1.2's "89 % / 1.04 / 0.36 / +0.66 / 1.38"). | **confirmed, and worse than stated** — §3.1 shows the longitudinal half is not CI-separated from CV at all |
| 2 | **ADE does not predict closed-loop.** | open-loop 0.4522 → closed-loop 1.685 ± 0.098; divergence >5 m on 22.2 %. MEASURED (MODEL_REGISTRY §1.2). | confirmed |
| 3 | **ADE hides propose-vs-rank.** | REF-C proposes ~2× better than flagship v1.5 (oracle 0.164 vs 0.338) while v1.5 mis-ranks half as often (0.235 vs 0.454) at near-identical ADE. MEASURED (MODEL_REGISTRY §4.1). | confirmed |
| 4 | **ADE is structurally blind to the hierarchy.** | `metric_dynamics.rollout_decode` calls `predictor(win_s, win_a)` with **no `intent` argument** — verified at `stack/tanitad/models/metric_dynamics.py:236`. `hierarchy.py` states the same: `deployed_operative_is_intent_free = True`. So 0.4522 is produced on a path where **none of the three conditioning seams participate**. | **verified in source this session** |
| 5 | **Accuracy alone cannot rank models.** | Ranks 1–2 tie (paired Δ +0.0443, CI [−0.0544, +0.1465], not separated) while latency differs **2.3–4.0×**. MEASURED (MODEL_REGISTRY §6). | confirmed |

A sixth, discovered while writing this: **6 — ADE cannot see plan instability.** Our eval protocol uses **stride 8** (windows 0.8 s apart), so we have never compared two *consecutive* plans. NAVSIM v2's Extended Comfort exists precisely for this and is fully computable for us (§4). MEASURED (protocol: `rollout.collect(stride=8)`).

---

## 2. What already exists — read this before proposing anything

### 2.1 Implemented and reusable (do **not** rebuild)

| capability | where | what it already gives |
|---|---|---|
| Frenet along/cross split, per horizon, signed + RMSE + `long_frac_of_sqerr` | `taniteval/taniteval/pathspeed.py::frenet_residual`, `metric_block` | the whole longitudinal/lateral decomposition, on the dense 20-step path, with speed and curvature strata |
| Speed-decoupled path geometry (arc-length resampled cross-track) | `pathspeed.py::path_geometry_crosstrack` | lateral geometry with the speed profile factored out |
| Planned-speed / accel profiles, heading, arc-length progress | `pathspeed.py::step_speed`, `step_accel`, `heading_deg`, `arclength` | the longitudinal primitives |
| Best-of-3 kinematic floor, ego-status ceiling (AD-MLP repro), per-stratum `skill_score` | `bench.py::kinematic_floor`, `_best_ridge_ceiling`, `diagnostic` | floors and ceilings in leaderboard units |
| Episode-cluster bootstrap + paired form | `taniteval/taniteval/ci.py` | the only admissible interval |
| Closed-loop drift, compounding delta, divergence rate, comfort, speed strata | `taniteval/taniteval/closedloop.py` | the entire closed-loop panel, with an honest limitations block |
| **Propose-vs-rank decomposition incl. chance floors and span-normalised skill** | `stack/tanitad/models/refc_rescorer.py:560-593`, `stack/scripts/refc_v12_eval.py:183-186` | `sel/base/oracle/oracle_k/chance/chance_k/sel_gap/sel_gap_k/sel_2x/rank_acc/gap_recovered/**span_k**` |
| Multi-candidate fan viz + per-frame oracle table | `taniteval/taniteval/plan_fan.py` | `ade`, `oracle_ade`, `vocab_ade`, entropy, `n_modes`, per frame |
| Hierarchy seam ablation, consistency/kappa, H18 | `taniteval/taniteval/hierarchy.py` | seam deltas vs mean/zero/none, agreement + Cohen's κ, grounded-vs-ungrounded |
| Vision-ablation / anticipation / CTRV-divergence stratification | `taniteval/taniteval/generalization.py` | the causal "does it read the scene" panel |
| Inference efficiency (latency p50/p95/p99, stage breakdown, FLOPs, VRAM, throughput, 10 Hz headroom, contamination check) | `taniteval/taniteval/efficiency.py`, report panel **04b** | the deployment axis — **do not touch** |
| Scenario-metric library (LAL, TMS, OKRI, LOPS, CNCE) | `stack/tanitad/eval/metrics.py` | ready, but needs `ScenarioTelemetry` (hazard LoS, blind-spot distance, occlusion flags, collisions) — **tier-2** |
| LAL-v2 decel-onset anticipation | `stack/tanitad/eval/metrics.py:202-251` (`decel_onset_index`, `compute_lal_v2`) | ✅ **MERGED 2026-07-09 in `3784e34`**, `stack/tests/test_lal_v2.py` green. *(This row previously read "implemented but unmerged" **while citing the merged path** — corrected 2026-07-21, `RETRACTION_LOG` class C4+C2.)* **L5 is blocked by ONE line — `taniteval/taniteval/rollout.py:94` keeps 4 of the 20 steps it computes** — not by a merge. |

**One naming decision, made here:** `refc_rescorer.py`'s `span_k` — `(chance_k_ade − sel_ade) / (chance_k_ade − oracle_k_ade)` — is exactly the normalisation the propose/rank family needs, and it already exists. This document adopts it as **RANK-SKILL** and generalises it beyond REF-C. It is **not** a new metric; it is a promotion.

⚠️ **RANK-SKILL is K-sensitive and must always be quoted with its K.** MEASURED (`.../incoming/2026-07-20-refc-v12/`): at **K=8** the unmodified REF-C selector scores `span_k` **0.6765** (`chance_k` 1.0336, `oracle_k` 0.2026) — informative. On the **full 256-anchor fan** the same selector scores **0.9777** and v1.2's learned rescorer **0.97835** — a Δ of +0.0006, because uniform selection over 256 anchors is catastrophic (`chance_ade` **13.96 m**), so the denominator swamps everything. **Report RANK-SKILL at K=8 (or a declared K ≈ the deployable shortlist); the full-fan form saturates and is not admissible as a headline.**

### 2.2 What is persisted, and what is thrown away

`rollout.save_windows` writes `pred/gt/cv [N,4,2] · eid · speed · head_deg · wp_steps`. **`rollout.collect` computes the dense `wp_full [b,20,2]` and discards 16 of the 20 steps at `index_select` time** (`taniteval/taniteval/rollout.py:94`). That single line is the tier-0/tier-1 boundary for the entire suite.

Verified: the persisted dump reproduces MODEL_REGISTRY exactly — full-set `ade_0_2s` **0.4271**, CV **0.8377**, long/lat RMSE **1.042 / 0.360**, top-decile speed bias **+0.659**, top-decile long-RMSE **1.3789**. MEASURED (this doc, `v2_tier0_probe.py --arm flagship-30k`, sanity block all-green).

### 2.3 What does **not** exist in our data (the hard constraint on everything below)

Stored per episode: `frames uint8 [T,9,256,256]`, `actions [T,2] = (steer_road_rad, accel_mps2)`, `poses [T,4] = (x,y,yaw,v)`. That is all. MEASURED (`stack/tanitad/data/_contract.py:59-78`, `stack/tanitad/lake/schema.py:121-125`).

**No HD map, no lane geometry, no drivable-area polygons, no other-agent boxes, no tracks, no traffic-light state, no per-frame timestamps, no lat/lon.** Lead state is a shape-fixed stub — `vlm_pending_lead_state()` returns `{"present": None, "gap_m": None, "closing_speed_ms": None, "ttc_s": None, "_pending": True}` (`stack/tanitad/lake/enrich.py:61-65`). `LANE_HALF_M = 1.75` in `goal_labels.py` is a **hard-coded assumption standing in for a measurement we do not have.**

**Goal-vocabulary coverage.** 18 slots / 114 tokens, frozen (`V3_GOAL_VOCABULARY_V1.md`, code mirror `stack/tanitad/lake/vocab.py`). **6 mint kinematically today** — tactical `VTARGET, VSOURCE, LONMODE, LATMANEUVER, DYN` + strategic `ROUTE` — one is conditional (`SIGNAL`, only where CAN carries the blinker) and **11 are `unknown`, awaiting the deferred Cosmos-Reason2 pass** (`stack/tanitad/lake/goal_labels.py:200-228`; only the 6 are promoted to queryable columns in `enrichment_arrow_schema`).

> ⚠️ **Conflict with the commissioning brief, reported per rule 4.** The brief states the frozen vocabulary has `road_geometry` and `scenario_tag` enums. It does not. Those two enums live in `stack/scripts/vlm_route_labels.py:64-80` as `ENUMS_SCENARIO` — **VLM prompt vocabularies for a deferred pass, with no corresponding stored dataset column**. `road_geometry` ∈ {straight, curve_left, curve_right, junction, roundabout, merge, fork, unknown}; `scenario_tag` ∈ {cut_in, lead_brake, pedestrian_crossing, cyclist, parked_vehicle_pass, construction, emergency_vehicle, blocked_lane, oncoming_encroach, unprotected_turn, traffic_light_stop, yield_merge, none}. **Scenario identification therefore has no ground truth today** (§3.5).

**Route-label coverage.** v2 needed 15 s of future and looked 25 s ahead on ~20 s clips; **74 % of windows fell through the guard and the guard returned `ROUTE_STRAIGHT`** — "cannot judge" and "goes straight" were the same emitted class. v2.1 fixes it: coverage **26.0 % → 81.9 %**, unlearnable genuine turns **63.1 % → 8.9 %**. MEASURED (`Benchmarks & Eval/Research/2026-07-20-route-label-fix-and-vlm-crossval.md`; the "70 %" in `refb_labels.py:483-491` is the same defect measured on a different sample). **No VLM can supply direction**: conditional left/right hit rate **57.1 %**, episode-cluster CI **[0.400, 0.745]** — contains chance (MEASURED, `Data Engineering/Research/2026-07-20-cosmos-reason1-vs-reason2-headtohead.md:104-105`).

---

## 3. The suite

Each metric carries: **definition + units · inputs · estimator · floor · failure mode exposed · OL/CL · tier**.

### 3.1 Longitudinal

> ⚠️ Load-bearing prior, honoured. **VTARGET at a 2 s reference is refuted, with numbers.** It sits **+1.42 m/s above v0**, is a 10–20 s free-flow aspiration, and used as a 2 s reference is **worse than holding v0 (MAE 1.65 vs 0.475)** and makes braking windows **+0.51 m worse**; a GT-perfect speed-matcher scores **1.1236 vs baseline 0.4714**. MEASURED (MODEL_REGISTRY §4.1; `stack/tanitad/models/refc_rescorer.py:42-47`). **No metric below references VTARGET at 2 s.** The reference is the *realised* future speed — which is simply the ground truth we already have.

#### L1 · CRUISE-QUALITY (CQ) — "does it hold a steady speed steadily?"
- **Definition.** On **steady windows** (`|mean realised longitudinal accel over the 2 s window| < 0.5 m/s²`, PROPOSED threshold), the mean absolute error of the planned speed profile against the realised speed profile. Units m/s. Report alongside the **fraction of windows classified steady**.
- **Inputs.** `pred`, `gt` waypoints + `speed` (v0). **All persisted.**
- **Estimator.** Episode-cluster bootstrap; **paired vs hold-v0** for the verdict.
- **Floor.** hold-v0. (Not CV — on a steady window CV *is* nearly the right answer, which is the point.)
- **Failure exposed.** Injected speed noise on free-flow cruise: the model disturbing something that needed no disturbing.
- **OL + CL.**  **Tier 0.**
- **MEASURED (this doc).** n=639 steady windows. Model speed MAE **0.4231 m/s**, hold-v0 **0.2109 m/s**, paired Δ **−0.2122, CI [−0.2778, −0.1443], separated — against the model.** The deployed flagship is **2.0× worse than doing nothing** on cruise. ADE on the same windows still looks fine (0.3834 vs hold-v0 0.5430) because the geometry carries it.

#### L2 · TRANSIENT-RESPONSE (TR) — "does it respond when the speed must change?"
- **Definition.** Same quantity as CQ, on **braking** (`accel ≤ −0.5`) and **accelerating** (`≥ +0.5`) windows, reported separately. Units m/s, plus the along-track error in m.
- **Inputs / estimator / floor.** As CQ.
- **Failure exposed.** Mean-regression: a model that predicts "keep going" scores well on the 73 % steady majority and fails exactly where driving is decided.
- **OL + CL.**  **Tier 0.**
- **MEASURED (this doc).** brake n=95: speed MAE **0.6104** vs hold-v0 **1.2536**, Δ **+0.6433 [+0.4466, +0.8276] separated**. accel n=147: **0.5889** vs **1.1605**, Δ **+0.5716 [+0.4336, +0.7076] separated**. **CQ and TR point in opposite directions for the same checkpoint.** This is the decomposition ADE destroys.

#### L3 · ALONG-TRACK ERROR + BIAS (Frenet) — the headline longitudinal number
- **Definition.** Project the per-horizon residual `pred−gt` onto the **GT path tangent**: `along` (+ = ahead). Report `|along|@2s` (m), `along@2s` signed bias (m), and `long_frac_of_sqerr` = `E[along²]/(E[along²]+E[cross²])`. Already `pathspeed.frenet_residual`.
- **Inputs.** persisted. **Estimator.** episode-cluster, paired vs floor. **Floor.** CV and hold-v0.
- **Failure exposed.** Over- or under-shoot along the road — invisible in BEV, invisible in ADE.
- **OL + CL.**  **Tier 0.**
- **MEASURED (this doc).** `|along|@2s` **0.8412 [0.7293, 0.9591]**; signed **+0.3498 [+0.1366, +0.5486]** (overshoot). Paired **vs CV: Δ +0.2543 [−0.0278, +0.5304] — NOT separated.** Paired vs hold-v0: Δ +0.2628 [+0.0217, +0.4973], separated but marginal. `long_frac_of_sqerr@2s` **0.8933**.

#### L4 · PROGRESS ERROR — the honest, map-free stand-in for route progress
- **Definition.** `arclength(pred) − arclength(gt)` over the horizon: absolute (m) and signed (m). This is progress along the *realised* path, **not** NAVSIM EP (which needs a route centerline — §4).
- **Inputs.** persisted. **Estimator.** episode-cluster. **Floor.** CV, hold-v0.
- **Failure exposed.** Systematic under/over-driving; pairs with L1/L2 to separate "wrong speed" from "wrong distance".
- **OL + CL.**  **Tier 0.**
- **MEASURED (this doc).** `|Δprogress|` **0.8370 [0.7248, 0.9523]** m; signed **+0.3821 [+0.1718, +0.5815]** m over 2 s.

#### L5 · DECEL-ONSET LEAD TIME (anticipation) — **tier 1**
- **Definition.** On braking windows, `t(GT decel onset) − t(planned decel onset)`, seconds; positive = the plan commits before the log does. Onset = first index where longitudinal jerk crosses `JERK_BRAKE_THRESHOLD = −1.5 m/s³` (**already implemented**: `stack/tanitad/eval/metrics.py::decel_onset_index`, `compute_lal_v2`).
- **Inputs.** the **dense 20-step path** (10 Hz) — the 0.5 s waypoint surface cannot resolve an onset. **Tier 1** (§5).
- **Estimator.** episode-cluster with a *median* reducer (onsets are heavy-tailed) + the **no-reaction rate** (sentinel `LAL_NO_REACTION`).
- **Floor.** hold-v0 never brakes → its lead time is undefined; the honest floor is the **CTRV/CV onset**, i.e. zero anticipation. PROPOSED.
- **Failure exposed.** Reactive-vs-anticipatory braking. Pairs with `generalization.py`'s vision-ablation to prove the anticipation is *read from the scene*.
- **OL + CL.**
- ✅ **MERGED on 2026-07-09, the day of the intake** (`3784e34` → `stack/tanitad/eval/metrics.py:202-251`, tests in `stack/tests/test_lal_v2.py`). *The earlier "still unmerged" note was wrong and had propagated into `V4_FLAGSHIP_DESIGN.md` as a 12-day-idle escalation; retracted 2026-07-21, class C4+C2.* **What actually blocks L5 is `rollout.py:94` discarding 16 of 20 computed steps** — and flagship v4's dense emitted plan is the first arm that makes fixing it worthwhile.

#### L6 · HEADWAY / DISTANCE-KEEPING — **REFUSED.** See §6.1.

#### L7 · VTARGET consistency — **only at its proven timescale.** Not built at 2 s. See §6.2.

---

### 3.2 Lateral

#### T1 · CROSS-TRACK ERROR + BIAS (Frenet) — the headline lateral number
- **Definition.** The normal component of the same residual: `cross` (+ = left). `|cross|@2s`, signed bias, and the squared share (= 1 − `long_frac`).
- **Inputs / estimator / floor.** persisted / episode-cluster paired / CV + hold-v0. **OL + CL. Tier 0.**
- **Failure exposed.** Path-geometry error, cleanly separated from speed error.
- **MEASURED (this doc).** `|cross|@2s` **0.2369 [0.1820, 0.2960]**; signed **+0.0221 [−0.0577, +0.0950]** (no left/right bias). Paired vs CV **Δ +0.7720 [+0.4166, +1.1914], separated**. **This is where flagship v1's entire CI-separated advantage lives.**

#### T2 · SPEED-DECOUPLED PATH GEOMETRY (PG)
- **Definition.** Resample `pred` and `gt` at the **same arc lengths** `d_j = j/m · min(L_pred, L_gt)`, take the RMS perpendicular deviation. Two paths tracing the same geometry at different speeds score ~0. Already `pathspeed.path_geometry_crosstrack`.
- **Inputs / estimator / floor.** persisted / episode-cluster paired / CV + hold-v0. **OL + CL. Tier 0.**
- **Failure exposed.** Whether a lateral error is *geometry* or merely *speed leaking into the cross-track term*.
- **MEASURED (this doc).** model **0.1110 [0.0860, 0.1402]** m vs CV **0.5217**, hold-v0 **0.4652**; paired vs CV **Δ +0.4107 [+0.2250, +0.6293], separated.** (Reproduces the registry's 20-step figure of 0.10 m.) **Flagship v1's path geometry is excellent; its speed is not.**

#### T3 · HEADING ERROR, stratified by GT curvature
- **Definition.** `|wrap(heading(pred) − heading(gt))|` at each horizon, degrees, reported **per curvature bucket** (`straight <5°`, `gentle 5–15°`, `sharp ≥15°` — the existing `driving_diagnostic.curvature_bucket` split) and as **median + exceedance rate**, per R5.
- **Inputs / estimator / floor.** persisted / episode-cluster with a median/quantile reducer (§7 E3) / CV + hold-v0. **OL + CL. Tier 0.**
- **Failure exposed.** Heading jitter — a comfort and controllability failure that costs almost nothing in L2 at short range.
- **MEASURED (this doc).** Corpus mean **6.61°** with CI **[2.34, 12.02]** — useless as a mean (R5). Per bucket: **straight (n=634) model 7.98° vs CV 1.399°** — **the model is 5.7× worse than a straight line at going straight**; gentle (n=103) 2.06° vs 7.852°; sharp (n=144) 3.811° vs 28.743°. **A new, unreported failure mode**, and precisely what nuPlan's OLS weights 2× (§4).

#### T4 · CURVATURE — sign agreement now, profile error later
- **Definition (tier 0).** **CURVATURE-SIGN AGREEMENT**: fraction of interior knots where `sign(κ_pred) = sign(κ_gt)`, counting windows with `|κ_gt| < 1e-3 m⁻¹` as free (PROPOSED threshold). Menger curvature on the polyline.
- **Definition (tier 1).** **CURVATURE-PROFILE ERROR**: `|κ_pred(s) − κ_gt(s)|` resampled at fixed **arc length** on the dense 20-step path, reported as median + p90.
- **Inputs.** tier-0: persisted. tier-1: dense path.
- **Estimator / floor.** episode-cluster / CV + hold-v0.
- **Failure exposed.** Whether the model turns the right way and by the right amount.
- ⚠️ **Naive curvature MAE at the persisted resolution is refuted.** MEASURED (this doc): at the low-speed tercile the knot spacing is ~1.93 m, `|κ_gt|` is 0.0495 m⁻¹ and the model's curvature MAE is **1.2015 m⁻¹ — 24× the signal.** It measures knot jitter, not curvature. Sign agreement survives and discriminates: **straight 0.9469 (CV 0.7639) · gentle 0.9320 (CV 0.2330) · sharp 0.9977 (CV 0.2037)**; corpus **0.9535 [0.9364, 0.9693]** vs CV 0.6103.

#### T5 · LANE-CENTRE DEVIATION — **REFUSED.** No lane geometry exists (§2.3, §6.4).

---

### 3.3 Navigation

The honest position: **we have no route graph and no map, and our clips are ~20 s.** NAVSIM EP and nuPlan `ego_progress_along_expert_route` are therefore not computable (§4). What *is* honest:

#### N1 · ROUTE ACCURACY ON THE VALID SUBSET (+ coverage, always reported together)
- **Definition.** Accuracy of the strategic route head against `route_from_future_v21`, computed **only on windows where `route_valid` is true**, reported next to **coverage** = fraction valid. Never average over `unknown`; never let the sentinel become a class.
- **Inputs.** `refb_labels.route_from_future_v21` per window — **not currently in the window dump** (§7 E4). **Tier 1.**
- **Estimator.** episode-cluster on the 0/1 correctness vector; paired vs the base rate.
- **Floor.** **majority-class (straight) base rate on the valid subset**, not 1/3. `hierarchy.py` already computes `majority_straight_rate` and the `vision_route_beats_majority` predicate — reuse verbatim.
- **Failure exposed.** A head that echoes the command, or that just says "straight". MEASURED precedent: `route_skill_vs_chance` **0.0** for flagship v1 — a pure command echo (MODEL_REGISTRY, D-033).
- **OL.** (No route to follow closed-loop over 2 s.)
- ⚠️ **Ground-truth honesty clause.** The only admissible route ground truth is **kinematic v2.1** (coverage 81.9 %). **A VLM may be used to DETECT that a route event occurred (turn-detection recall ~78 %) and never to supply its direction (57.1 %, CI [0.400, 0.745] — chance).** MEASURED (direction §2.3; detection `Data Engineering/Research/2026-07-21-cosmos-reason2-production-semantic-labeling.md` §1).
  > ⚠️ **CORRECTED 2026-07-21 — the "89.3 % agreement" this clause used to quote is retracted, twice over.** **(1) It does not reproduce.** Three independent measurements of Cosmos-Reason2-8B against the kinematic v2.1 label, on three different window sets, give **76.8 %** (eval pod, 200 win) · **80.6 %** (pod3 banked re-score, 400 win) · **78.6 %** (enum-order probe, 200 held-out win) — they cluster at **77–81 %** and none approaches 89.3 %. **(2) It was *agreement*, not *recall*.** Agreement counts straight–straight matches, so on a **~74 %-straight** corpus a model answering "straight" every time scores ~74 % while detecting **zero** turns — it was never evidence of detector quality. **The clause's qualitative content is unchanged: the VLM is a competent event detector and an incompetent direction reader.** Source: `Data Engineering/Research/2026-07-21-cosmos-reason2-production-semantic-labeling.md` §1.

#### N2 · ROUTE↔TRAJECTORY COHERENCE
- **Definition.** Agreement + **Cohen's κ** between the commanded route class and the direction of the rolled-out trajectory's net heading, reported **overall and on the turn-active subset** (raw agreement is inflated by a straight-dominated corpus: 653 of 881 windows are `<5°`). Already `hierarchy.py::consistency::commanded_route_vs_trajectory`.
- **Inputs.** route labels + rollout. **Tier 1** (labels).
- **Estimator.** episode-cluster with a **κ statistic** reducer (§7 E3). **Floor.** κ = 0.
- **Failure exposed.** A strategic layer that is consistent with nothing it drives.
- **OL + CL.**

#### N3 · ROUTE-FOLLOWING EFFICIENCY — **partially refused.** The 2 s-horizon surrogate is **L4 progress error**, and it must be named that, not "route efficiency". A true efficiency metric requires a route graph we do not have. §6.5.

---

### 3.4 Tactical maneuver planning

#### M1 · MANEUVER CORRECTNESS — balanced, not raw
- **Definition.** Against `refb_labels.classify_maneuver`'s 5 classes (`lane_keep, turn_left, turn_right, accelerate, brake_stop`, 2 s horizon): **macro-F1 and Cohen's κ**, plus the per-class confusion. **Raw accuracy is inadmissible** on this corpus.
- **Inputs.** maneuver targets per window (**not in the dump** — §7 E4) + the tactical head. **Tier 1.**
- **Estimator.** episode-cluster with a macro-F1 / κ statistic reducer. **Floor.** majority-class base rate.
- **Failure exposed.** A head that emits `lane_keep` everywhere and still posts a respectable raw accuracy, because the corpus is lane-keep-dominated.
- ⚠️ `hierarchy.py` currently reports **raw** `man_corr_real` only. That is the change.
- **OL + CL.**

#### M2 · MANEUVER-ONSET TIMING — **tier 1.** Same machinery as L5, applied to the maneuver-transition windows. Requires **stride 1** near transitions (§5).

#### M3 · PROPOSE-vs-RANK — the family, generalised
For any arm producing N scored candidates (REF-C's 256 anchors, P2's CEM population, any future MPC fan):

| term | definition | already implemented |
|---|---|---|
| `sel_ade` | ADE of the candidate the model picked | ✅ `refc_rescorer.py:560` |
| `oracle_ade` / `oracle_k_ade` | min ADE over all N / over the top-K | ✅ |
| `chance_ade` / `chance_k_ade` | **expected ADE of a skill-free ranker** over all N / top-K | ✅ `:572` |
| `vocab_ade` | min ADE over the **raw, pre-refinement** candidates → the coverage floor | ✅ `plan_fan.py` |
| `sel_gap`, `sel_gap_k` | `sel − oracle` (the gap; **not headroom** — see below) | ✅ |
| `frac_sel_2x_worse` | fraction of windows where the pick is >2× the oracle | ✅ |
| `rank_acc` | fraction where the pick *is* the argmin | ✅ |
| **RANK-SKILL** (`span_k`) | **`(chance_k − sel) / (chance_k − oracle_k)` ∈ [0,1]** — the fraction of the achievable ranking range captured, **at a declared K** | ✅ `:588-593` |

- **Why RANK-SKILL is the headline and `sel_gap` is not.** `oracle_ade` is a **minimum over N candidates scored against ONE realised future**; its downward bias grows with N, so `sel_gap` is not comparable across fan sizes and is not available headroom. Our own evidence: a learned re-scorer recovers at most **8.4 %** of it across 47 trained arms, and REF-C v1.2's honest delta is **+2.9 %, not significant** (paired Δ +0.00893, CI [−0.0062, +0.0250]). MEASURED (MODEL_REGISTRY §4.1). RANK-SKILL normalises the aleatoric floor away, so REF-C and P2 become comparable **at a matched K**. **MEASURED: the unmodified REF-C selector scores `span_k` = 0.6765 at K=8** — the incumbent already captures two-thirds of the achievable ranking range, which reads very differently from "0.3075 m of headroom". ⚠️ **Always declare K** (see §2.1): on the full 256-anchor fan the same selector reads 0.9777 because `chance_ade` is 13.96 m, and the metric stops discriminating.
- **Add (PROPOSED):** the **oracle-vs-N curve** (`oracle_ade` at N = 1,2,4,…,256, free from the same fan) so the min-statistic bias is visible on the page rather than argued about.
- **Inputs.** the per-window fan + logits — **not persisted** (`plan_fan.py` writes scalars only and drops the `[N,S,2]` tensors). **Tier 1** (§7 E2).
- **Estimator.** episode-cluster; paired for two selectors on the same fan. **Floor.** `chance_k_ade`.
- **Failure exposed.** Coverage failure vs ranking failure — the two are opposite fixes.
- **OL + CL.**

---

### 3.5 Scenario capability — intersections, roundabouts, merges

**The blocking constraint is not the metric. It is the horizon.** An intersection traverse is ~5–10 s; a roundabout 8–20 s; a merge 5–15 s. Our planning horizon is **2 s** and our clips are **~20 s (199 frames)**. A 2 s window inside a roundabout is kinematically indistinguishable from a constant-radius curve. **No scoring function fixes that.** §6.3.

What is buildable:

#### S1 · KINEMATIC SCENARIO STRATA (tier 0–1) — honest naming required
Stratify every metric above by signatures derivable from `poses` alone. **These are kinematic signatures and must never be labelled "intersection" or "roundabout" in any output.**

| stratum | rule (all PROPOSED except where noted) | tier |
|---|---|---|
| `straight` / `gentle` / `sharp` | `driving_diagnostic.curvature_bucket` on net heading change | 0 ✅ exists |
| speed terciles + top decile | `speed` quantiles | 0 ✅ exists |
| `brake` / `steady` / `accel` | realised mean accel, ±0.5 m/s² | 0 (this doc) |
| `launch_from_stop` | `v0 < 0.3 m/s` (`refb_labels.STOP_V_MS`, MEASURED constant) and future `v > 1.0` (`MOVING_V_MS`) | 0 |
| `stop_approach` | future `v` reaches `< 0.3 m/s` | 0 |
| `sustained_turn` | same-sign yaw rate over the whole window and `|net heading| ≥ 15°` | 0 |
| `lane_change_signature` | `LATMANEUVER` mint: `\|net yaw\| < 0.20 rad` **and** lateral offset ≥ `LANE_HALF_M = 1.75 m` (`goal_labels.py:170-194`) — ⚠️ the 1.75 m is an **assumed** lane half-width, not a measurement | 1 |
| `episode-level turn/roundabout signature` | cumulative heading over the **whole clip**, not the window | 1 (needs episode-level aggregation) |

#### S2 · SCENARIO-TAGGED CAPABILITY (tier 2) — needs the deferred VLM enrichment
`road_geometry` / `scenario_tag` enums exist as *prompt vocabularies* only (§2.3). Realising them means running the Cosmos-Reason2 pass over the corpus and merging into the lake. **Constraint carried forward:** the VLM is a competent event **detector** and an incompetent **direction reader**; scenario tags must be validated the same way route labels were (kinematic cross-val + episode-cluster CI) before any capability number is published against them. Pilot coverage is **51–53 % on 595 records — not coverage** (MEASURED, `V35_DESIGN.md` C5).

---

### 3.6 Hierarchical / strategic–tactical capability — **the commissioning question**

**The problem, stated precisely.** Our headline metric is computed by `rollout_decode`, which calls `predictor(win_s, win_a)` with **no intent** (`stack/tanitad/models/metric_dynamics.py:236`, verified). `hierarchy.py` therefore measures the seams in **latent space** (`lat_cos`, `lat_rel`) and openly records `deployed_operative_is_intent_free = True` and that threading intent into the grounded rollout is **out-of-regime** (the step-readout is calibrated intent-free). So today: **the number we deploy cannot move when the hierarchy changes, and the number that moves is not the number we deploy.** That is why H26 has produced a verdict — 1 of 3 seams load-bearing, `intent→operative` **harmful at cos −0.238** with `‖intent_proj‖ 31.4` vs `‖act_emb‖ 28.3` (MEASURED, `ARCHITECTURE_WIRING_COMPARISON.md` F3) — that nobody can act on with confidence.

Three metrics fix this. **All three are the deliverable that settles Sayed's arc.**

#### H1 · HCS — Hierarchical Contribution Score (**the metric that most changes what we learn**)
- **Definition.** For each seam `u → d`, the **paired** change in the *task* metric of layer `d`, measured **on the decode path that is scored**, when `u`'s per-window signal is replaced by a control:
  `HCS(u→d; control) = mean( M_control − M_real )` in the helps-positive orientation, with a **paired episode-cluster bootstrap** over the 40 episodes.
- **Three controls, and the third is the new one.**
  1. `mean` — the on-distribution average signal. Removes content, preserves the mean. (Exists.)
  2. `zero` / `none` — full removal. Removes content **and** magnitude, so it confounds. (Exists.)
  3. **`perm` — within-episode random permutation of the upstream signal across windows.** *(PROPOSED, new.)* Preserves the exact marginal distribution, the norm, and the scale; destroys only the **window-to-window alignment**. This is the control that separates "the seam carries information about *this* window" from "the seam contributes a useful constant offset the downstream layer co-adapted to". `hierarchy.py`'s existing `cond_norms` diagnostic exists precisely because magnitude is suspected of driving the result — permutation settles it, mean-replacement cannot.
- **Verdict rule (PROPOSED).** A seam is **LOAD-BEARING** iff `HCS(u→d; perm)` is **CI-separated from 0 in the helping direction** *and* exceeds the minimum practical effect (`MIN_ADE_M = 0.05 m`, `MIN_ACC = 0.02`, `MIN_COS = 0.01` — MEASURED constants already in `hierarchy.py:75-77`). `mean` and `zero` remain as reported context, never as the verdict.
- **Inputs.** an arm with trained tactical + strategic policies; the ablation harness already exists (`hierarchy.py::run`, phase-2 cache). Adding `perm` is ~15 lines.
- **Estimator.** **paired** episode-cluster bootstrap — replacing `hierarchy.py::_jack`, which is the deprecated `overlapping_holdout_se`. **This alone may flip existing verdicts, in either direction.**
- **Floor.** zero effect. **Failure exposed.** Decorative conditioning; co-adapted constant offsets masquerading as information.
- **OL + CL.** **Tier 1.**
- ⚠️ **Regime honesty.** HCS is only meaningful on a path where the seam is actually used. Two admissible readings, both must be published: **(a) in-regime** — the intent-conditioned multi-horizon latent prediction (what `hierarchy.py` measures today); **(b) deployed-path** — requires the scored rollout to *be* intent-conditioned, i.e. an **intent-conditioned step-readout calibration** (a training/architecture change, not a metric change). Until (b) exists, every hierarchy claim must carry the sentence *"measured in latent space; the deployed trajectory is intent-free."*

#### H2 · DECISION-WINDOW HCS
- **Definition.** H1 restricted to **decision windows** — maneuver-transition, braking-onset, and valid non-straight route windows — reported next to the corpus-wide value.
- **Rationale.** **634 of 881** windows are `<5°` straight and **639 of 881** are longitudinally steady (MEASURED, this doc). A hierarchy that helps uniformly across a lane-keep-dominated corpus is suspicious; one that helps *where a decision exists* is real. A corpus average dilutes a decision-window effect by ~4× and can turn a real seam into a null.
- **Inputs.** the label join (§7 E4). **Tier 1.** **Floor.** zero. **Estimator.** paired episode-cluster on the subset.

#### H3 · COHERENCE-AS-PREDICTOR (turn a consistency number into a capability)
- **Definition.** Do the layers *disagree where the model is wrong*? Report `E[error | layers disagree] − E[error | layers agree]` with a paired episode-cluster interval, and the **AUC** of layer disagreement as a per-window error detector.
- **Rationale.** `hierarchy.py` already measures agreement + κ. Agreement is cheap and a straight-dominated corpus inflates it. **A hierarchy whose internal disagreement predicts its own failure is load-bearing for deployment even if the conditioning deltas are small** — it is a free runtime monitor. That is a genuinely different, and operationally valuable, claim.
- **Inputs.** existing hierarchy outputs + per-window error. **Estimator.** episode-cluster with an AUC statistic reducer (§7 E3). **Floor.** AUC 0.5. **Failure exposed.** Coherence that is decorative rather than diagnostic. **OL + CL. Tier 1.**

#### H4 · SEAM NORM-PARITY (keep — it has already caught two bugs)
`‖contribution(u)‖ / ‖signal it is added to‖`, per seam. **Not a capability metric — a wiring monitor.** It fired twice: v1's `intent_proj` 31.4 vs `act_emb` 28.3 (→ harmful seam), and v1.5's ROUTE seam at **2.80× the measurement norm despite `rt_gate` 0.10** (MEASURED, `V35_DESIGN.md` P15). **Report it beside every HCS; a seam with a norm ratio > 1 has a mis-scaled implementation, and its HCS is measuring a bug, not an architecture.**

---

### 3.7 Inference efficiency — integrate, do not redesign

`taniteval/taniteval/efficiency.py` + report panel **04b** stay exactly as they are: latency mean/p50/p95/p99, stage breakdown, analytic FLOPs, peak VRAM, params, batched throughput, 10 Hz headroom, an enforced fairness contract (precision applied identically and recorded), and a `contamination_check` that quarantines a dirty run.

**How it enters the verdict — a gate plus a frontier, not a scalar.**

1. **ADMISSIBILITY (a hard gate, both must pass).**
   (a) meets the **10 Hz budget at p99** in the declared deploy precision; (b) beats **every** trivial floor on the headline capability metric with a **CI-separated paired** bootstrap.
   MEASURED: flagship v1 eager **fails** (a) in all three precisions (p99 146.60 / 102.71 / 113.13 ms); with the composed L4 lever it **passes** at **18.76 ms p99 = 53.3 Hz** (MODEL_REGISTRY §1.2, R12). REF-C **passes** (a) in all three.
2. **PARETO RANK among admissible arms** over (capability, p99 latency). Report the frontier; do not collapse it.
3. **NO SCALAR COMPOSITE.** *(PROPOSED, and this is a refusal — §6.6.)* Any single number trading metres against milliseconds embeds an exchange rate nobody has measured, and our own arms already rank *oppositely* on the two axes: REF-C wins latency 2.3–4.0×, the flagship wins batched throughput 34.8 vs 29.9 windows/s. `efficiency.py`'s existing `ms_per_metre_of_ade_beaten_vs_cv` is fine as a **diagnostic ratio**; it must not become the leaderboard sort key.
4. **Every efficiency figure carries its tick definition, hardware, checkpoint and corpus.** This is not pedantry — the bare "11.16 ms / 89.6 Hz" propagated into three documents before being corrected; it differed from the scored tick in **five** dimensions at once (MODEL_REGISTRY §1.2).

---

## 4. What is established as standard — adopt, adapt, or refuse

Our constraint: single front-wide camera + ego `(x, y, yaw, v)` @ 10 Hz. **No HD map, no agent boxes, no simulator, no renderer** (AlpaSim's NuRec is unrunnable on the eval pod — unprivileged container, seccomp blocks user namespaces).

| standard | needs | verdict for TanitAD |
|---|---|---|
| **NAVSIM PDMS** = `Π{NC, DAC} × Σ{EP·5, TTC·5, C·2}/12`, planner emits one **4 s** trajectory propagated by a bicycle+LQR at 10 Hz (PUBLISHED, arXiv:2406.15349) | NC → agent + static boxes; DAC → drivable-area polygons; TTC → agent boxes; EP → route centerline + the PDM-Closed reference planner | **REFUSE the aggregate.** Two of its multipliers are structurally uncomputable and a zero there zeroes everything. **Adopt `C` only, under its own name** (below). Never publish a partial PDMS. |
| **NAVSIM v2 / EPDMS** adds DDC, TLC, LK, HC, **EC**, with a false-positive filter that neutralises a sub-metric when the *human* log also fails it (PUBLISHED, arXiv:2506.04218) | DDC/TLC/LK → map + signals | **REFUSE the aggregate. ADOPT two ideas.** (i) **EC (Extended Comfort)** — see below, our biggest import. (ii) **the human-log filter**: if the logged human trajectory itself violates a bound, the model is not charged. That is an honesty device we should copy into every bounded metric we report. |
| **nuPlan CLS-NR / CLS-R** (PUBLISHED, arXiv:2106.11810 + devkit) | the nuPlan simulator, HD map, background agents (log-replay vs IDM) | **REFUSE.** Not approximable. |
| **nuPlan OLS** = `miss_rate_within_bound` (multiplier) × weighted mean of avg-L2 **(w 1)**, avg-heading-error **(w 2)**, final-L2 **(w 1)**, final-heading-error **(w 2)**; 1 Hz, horizons [3,5,8] s, bounds 8 m / 8 m / 0.8 rad, miss thresholds [6,8,16] m (PUBLISHED, `open_loop_boxes_weighted_average.yaml`) | **ego trajectory + logged expert only** | ✅ **ADOPT the form, re-declare the horizons.** This is the one fully-specified, fully ego-computable published aggregate available to us. Our horizons are 0.5/1/1.5/2 s, so the bounds do not transfer and must be re-derived — state that, never quote a nuPlan-comparable OLS. **Its 2× weight on heading error is what makes our T3 result matter**: on straights we are 5.7× worse than a straight line. |
| **CARLA Driving Score** = route completion × infraction penalty; LB ≤2.0 multiplicative `Π p_j^{n_j}`, **LB 2.1 linear `1/(1+Σ c_j n_j)`** (PUBLISHED, leaderboard.carla.org) | the CARLA simulator, a route graph, simulator-generated infractions | **REFUSE** until CARLA-on-pod exists (already tracked as G2, blocked). |
| **nuScenes open-loop L2 / collision rate**; conventions **TemAvg** (ST-P3 arXiv:2207.07601, VAD arXiv:2303.12077) vs **NoAvg** (UniAD arXiv:2212.10156) (PUBLISHED, naming per arXiv:2406.17680) | L2 → ego only; collision → agent boxes | ✅ **L2 ADOPTED and already implemented in both conventions** (`bench._l2_conventions`, "cumulative = mean over waypoints ≤T (ST-P3); endpoint = mean of per-horizon final errors"). ❌ **collision rate REFUSED** — no boxes. ❌ BEV-Planner's road-boundary intersection rate REFUSED — no boundaries. |
| **The nuScenes critique line** — AD-MLP (arXiv:2305.10430), BEV-Planner (arXiv:2312.03031), SparseDrive's collision-code defects (arXiv:2405.19620) | — | ✅ **ALREADY ADOPTED as a falsifier**: `bench.diagnostic`'s ego-status ridge ceiling is an AD-MLP repro, MEASURED at **0.5735** on our val. **Keep it as a first-class suite member, not a side panel** — see the warning below. |
| **WOMD minADE/minFDE-k, miss rate** with speed-scaled **lateral/longitudinal** thresholds (1.0/2.0 m @3 s, 1.8/3.6 @5 s, 3.0/6.0 @8 s; scale γ(v) interpolated between 1.4 and 11.0 m/s), mAP over 8 behaviour buckets (PUBLISHED, arXiv:2104.10133 + `motion_metrics.proto`) | ego trajectory + yaw + v | ✅ **ADOPT the lateral/longitudinal miss decomposition and the speed scaling** — it is the same Frenet idea we already use, with published thresholds, and it directly replaces our unscaled `miss_rate@2m`. Re-derive thresholds for a 2 s horizon (PROPOSED). **minADE-k applies only to multi-candidate arms** (REF-C, P2) — for them it is the natural partner of RANK-SKILL. ❌ overlap rate REFUSED (needs other agents). |
| **WOSAC** composite realism (PUBLISHED, arXiv:2305.12032 + 2025 config) | 0.70 of the weight is interactive/map/traffic-light | ❌ **REFUSE the composite.** ✅ **ADOPT the 0.20-weight kinematic block as an idea**: likelihood of our rollouts' linear speed / linear accel / angular speed / angular accel under the logged distributions. This is a **distribution-realism** axis we have never measured and it needs no map. *(PROPOSED for tier-1; requires the dense path.)* |
| **Comfort thresholds** — nuPlan/NAVSIM `ego_is_comfortable`, binary, all bounds must hold: `MIN_LON_ACCEL −4.05`, `MAX_LON_ACCEL +2.40`, `MAX_ABS_LAT_ACCEL 4.89` m/s², `MAX_ABS_LON_JERK 4.13`, `MAX_ABS_MAG_JERK 8.37` m/s³, `MAX_ABS_YAW_RATE 0.95` rad/s, `MAX_ABS_YAW_ACCEL 1.93` rad/s² (PUBLISHED, `navsim/.../pdm_comfort_metrics.py`; nuPlan devkit) | ego kinematics only | ✅ **ADOPT VERBATIM — and replace ours.** ⚠️ `closedloop.py` currently uses `A_LON_COMFORT 2.0`, `A_LAT_COMFORT 3.0`, `JERK_COMFORT 2.0`, self-labelled "comfort (not safety), documented" but **unsourced and substantially tighter than the published standard**. Our violation rates therefore overstate discomfort and are not comparable to any published number. §7 E6. |
| **NAVSIM v2 Extended Comfort (EC)** — deltas between **two consecutive frames' planned trajectories**: accel 0.7 m/s², jerk 0.5 m/s³, yaw rate 0.1 rad/s, yaw accel 0.1 rad/s² (PUBLISHED, `ego_is_two_frame_extended_comfort()`) | consecutive plans; **no map, no agents** | ✅✅ **ADOPT — the single most valuable import in this table.** It measures **plan stability**, an axis we have never measured and cannot see: our protocol strides 8 windows (0.8 s), so consecutive plans are never compared. It is exactly the pathology T3's straight-line heading jitter (7.98° vs CV 1.399°) is hinting at. **Blocker: needs stride-1 evaluation** (§5, tier 1). |

> ⚠️ **The warning that has to be in this document.** The set of metrics computable from ego logs alone is *precisely* the set the critique literature showed is gameable by an ego-status MLP with no perception (arXiv:2305.10430, arXiv:2312.03031); NAVSIM's own Ego-Status-MLP scores **PDMS 65.6** vs 83+ for sensor agents (PUBLISHED, arXiv:2406.15349). **A map-free suite built only from the adoptable rows cannot, on its own, discriminate perception quality.** Two of our existing panels are the antidote and must be promoted to first-class members of TanitEval v2, not optional extras: `bench.diagnostic`'s **ego-status ceiling** (MEASURED 0.5735 — flagship v1's 0.4522 clears it) and `generalization.py`'s **vision-ablation on high-divergence windows** (MEASURED: vision effect **+1.325 m, CI [+1.04, +1.64]**, CI-separated).

---

## 5. Tiers, with cost

### Tier 0 — computable **today**, CPU-only, from committed artifacts. Cost: **seconds. Zero GPU.**
Everything on the persisted 4-waypoint surface: **L1 CQ · L2 TR · L3 along-track · L4 progress · T1 cross-track · T2 path geometry · T3 heading (stratified) · T4 curvature sign · S1 kinematic strata · ADE/FDE/miss** — each with an episode-cluster bootstrap and a **paired** test against CV and hold-v0.
Reference implementation staged: **`taniteval/v2_tier0_probe.py`** (sanity-pinned against MODEL_REGISTRY; refuses to publish if the pin fails).
🚧 **Blocker, not a cost:** only **2 of the 10** registry arms have a `windows_<key>.pt` in the repo. The other 8 live on `tanitad-eval:/root/taniteval/results` (~96 KB each). §7 E1.

### Tier 1 — modest new logging, **no new data, no training**. Cost: **one existing eval re-run per arm** (`python3 -m taniteval.runner run --model <key> --episodes 40`), single-GPU minutes each; engineering **1–3 days total**.
| # | change | unlocks |
|---|---|---|
| **P1** | **persist the dense 20-step path** — `rollout.collect` already computes `wp_full [b,20,2]` and discards 16 steps at `rollout.py:94`. Save `pred_full/gt_full/ctrv/gs` (~1 MB/arm). | 10 Hz speed/accel/**jerk**, adopted **comfort** bounds, **curvature-profile** error (T4 tier-1), TMS on the real path, **L5 decel-onset lead time**, WOSAC kinematic realism |
| **P2** | **persist the per-window fan** (candidates + logits) for multi-candidate arms | **M3 propose-vs-rank / RANK-SKILL for every arm**, minADE-k, oracle-vs-N curve |
| **P3** | **join the per-window labels** into the dump: `route_v21 + valid`, maneuver target, regime, `vt_v2 + valid` (all already minted — `v15_prep.py:249-275` shows the exact cache) | **N1, N2, M1, M2, H2**, and every stratified read |
| **P4** | **a stride-1 evaluation pass** (or stride-1 on a subset) | **EC / plan stability**, maneuver-onset timing |
| **P5** | **add `perm` control + paired episode-cluster to `hierarchy.py`** (~15 lines + the estimator swap) | **H1, H2, H3** — the commissioning question |
| **P6** | **`ci.py`: a `statistic=` callable path + quantile reducer** | median/p90 metrics (R5), κ, macro-F1, AUC |

### Tier 2 — needs a simulator or new data. Cost: **weeks, external dependency.**
- **Lead-state / headway / TTC / cut-in / follow-lead** → a detector + monodepth or the Cosmos-Reason2 enrichment pass, validated to coverage (pilot is 51–53 % on 595 records).
- **Collisions, drivable-area, NAVSIM PDMS/EPDMS, nuPlan CLS, CARLA DS** → HD map + agent boxes + a simulator.
- **Intersection / roundabout / merge capability** → a **longer horizon** (5–20 s) *and* scenario labels. Horizon first; labels second.
- **LAL / OKRI / LOPS / CNCE** → `ScenarioTelemetry` (hazard LoS, blind-spot distance, occlusion flags, collision counts). The metric code is written and tested; the telemetry does not exist.
- **A photoreal external closed loop** (NuRec / HUGSIM) — the only cure for `closedloop.py`'s self-referentiality: the world model is currently both planner-state-estimator and simulator, so failures it cannot imagine are invisible.

---

## 6. Refusals — what I recommend **not** building, and why

**6.1 · Headway / distance-keeping / TTC. REFUSE.** There is no lead-agent state anywhere in the stored data — `lead_state` is a shape-fixed `None` stub with `_pending: True`, no boxes, no tracks, no depth (§2.3). Building it means inventing the lead vehicle from the ego's own deceleration, which would make the metric a re-description of L2. Three independent primary sources already state the same conclusion for the planner cost, the closed loop and the P2 spec ("no gap/TTC/collision term — our data has no lead-agent boxes or HD map"). **Revisit when lead-state labels exist at coverage, not at 51–53 %.**

**6.2 · Any VTARGET-referenced 2 s target-speed metric. REFUSE.** Already refuted with numbers (§3.1 prior). It is the right quantity at the wrong timescale, and at 2 s it loses to holding v0. **Additionally: a 10–20 s VTARGET metric is not measurable at a 2 s planning horizon either** — so the honest answer is that TanitEval v2 has **no VTARGET metric at all** until the horizon changes. Note the mint's own two measured defects (`VT_LOOK_LO` never enforced; jitter-driven asymmetric free-flow gate biasing the percentile upward — `stack/tanitad/lake/vtarget.py:12-30`) as a further reason not to build a metric on top of it.

**6.3 · Intersection / roundabout capability at the 2 s horizon. REFUSE.** The events are 5–20 s; the horizon is 2 s; the clips are ~20 s. A metric cannot manufacture the horizon. Building one would produce a number that looks like scenario capability and measures curve-following. **The correct response to Sayed's ask is a horizon and data decision, not a metric** — and it should be escalated as such. Interim: S1 kinematic strata, named honestly.

**6.4 · Lane-centre deviation / lane-keeping. REFUSE.** No lane geometry exists. The only lane number in the codebase is the hard-coded `LANE_HALF_M = 1.75` used as a proxy inside the `LATMANEUVER` mint. A lane-keeping metric built on an assumed constant measures the constant.

**6.5 · "Route-following efficiency" as a named metric. REFUSE the name, keep the quantity.** With no route graph, the only computable thing is **progress along the realised path** (L4). Calling it route efficiency would import NAVSIM EP's meaning without its inputs. Report L4 under its own name.

**6.6 · A scalar capability×efficiency composite. REFUSE.** §3.7 item 3.

**6.7 · `sel_gap` as a headline. REFUSE (demote).** It is a min-over-N statistic against one realised future; ~92 % of it is irreducible (MEASURED across 47 arms). Report **RANK-SKILL** as the headline and `sel_gap` as a diagnostic beside it.

**6.8 · Naive curvature MAE at the persisted resolution. REFUSE.** 24× the signal (§3.2 T4). Sign agreement at tier 0; arc-length curvature profile at tier 1.

---

## 7. Escalations — integration this suite needs (per rule 3: raised here, not buried)

| # | issue | why it blocks | who/where |
|---|---|---|---|
| **E1** | ✅ **CLOSED 2026-07-21.** All dumps pulled — `taniteval/results/` now holds **24** `windows_*.pt`, and tier-0 has been **promoted from this probe into `taniteval/taniteval/driving.py`** and run across every one of them (`python -m taniteval.runner driving-all`, CPU-only, ~1 min total). Artifacts: `taniteval/results/driving_<key>.json` + inline `driving` block in every `results/<key>.json`; dashboard panel **04c**; `Benchmarks & Eval/LEADERBOARD.md` §2 rewritten from them (closing registry gap **R5** as a side effect). | — | done |
| **E9** | **The curvature buckets disagree between two live panels.** This suite and `v2_tier0_probe.py` use straight <5° / gentle 5–15° / sharp ≥15° (all §3.2/Appendix A numbers, n = 634/103/144). `driving_diagnostic.curvature_bucket` — which `bench.py`'s own `by_curvature` panel uses — has `CURV_GENTLE_DEG = **20.0**`. Both boundaries are now recorded in every `driving_<key>.json` and pinned by `test_driving.py` so neither can drift silently, but the two panels are **not interchangeable** until one is chosen. | A reader comparing panel 01b's `by_curvature` with panel 04c's is comparing different strata under identical labels. | TanitEval owner |
| **E10** | **Registry §6's latency figures and the committed `eff_*.json` disagree for the flagship.** §6 reading 3 quotes 103.42 / 93.76 / 104.49 ms (fp32/tf32/amp16); `eff_flagship-30k.json` says 97.32 / 97.70 / 123.83, its own replicate says 97.13 fp32, and `eff_repeatability.json` (5 clean reps) says 99.03–100.05 p50. REF-C's fp32/tf32 agree across both; its amp16 does not (26.12 vs 21.00). | The conclusion is unchanged in every version, but two primary sources currently disagree and the leaderboard has to pick one. It quotes the artifact and flags the conflict. | eval-pod owner |
| **E2** | **`rollout.save_windows` persists 4 of 20 rollout steps** (`rollout.py:94`) and no CTRV / go-straight, and `plan_fan.py` writes fan *scalars* while dropping the `[N,S,2]` tensors. | Every tier-1 metric traces to these two lines. Cheapest high-leverage change in the whole design. | TanitEval owner |
| **E3** | **Three modules still compute intervals with the deprecated `overlapping_holdout_se`**: `hierarchy.py::_jack`, `closedloop.py::_jack`, `bench.py::_agg`. `ci.py` has no `statistic=` callable path or quantile reducer. | Every hierarchy and closed-loop verdict currently rests on an estimator measured **1.28–2.06× too narrow**. Migrating may flip published verdicts in either direction — which is the point. | TanitEval owner |
| **E4** | **Per-window labels are minted but never joined into the eval dump.** | N1, N2, M1, M2 and H2 are blocked on a join, not on science. `stack/scripts/v15_prep.py:249-275` already shows the exact cache format. | Data Eng ↔ TanitEval |
| ~~**E5**~~ | 🟥 **WITHDRAWN 2026-07-21 — LAL-v2 was merged on 2026-07-09** (`3784e34`), the day of the intake. The "12 days idle" framing was **false** and had been escalated onward into `V4_FLAGSHIP_DESIGN.md` (P5e + escalation 1b). `RETRACTION_LOG`, class **C4+C2**: inherited from this table without a `git ls-files` or a grep of the named module. | **The replacement escalation, which is real and one line:** `taniteval/taniteval/rollout.py:94` computes the dense `wp_full [b,20,2]` and persists only 4 of 20 steps, which is what actually blocks L5, jerk, the comfort bounds and a real curvature profile. Flagship v4 emits a dense plan and is the first arm that can use them. | TanitEval owner |
| **E6** | **`closedloop.py`'s comfort bounds are unsourced and tighter than the published nuPlan/NAVSIM standard** (2.0/3.0/2.0 vs −4.05/+2.40/4.89/4.13/8.37/0.95/1.93). | Our comfort violation rates overstate discomfort and are not comparable to any published figure. | TanitEval owner |
| **E7** | **`hier_<key>.json` exists nowhere in the repo** — the hierarchy panel's outputs live only on the eval pod. | Our only evidence on the program's central claim is unbacked-up and un-auditable offline. | eval-pod owner → repo |
| **E8** | **No scenario ground truth exists** (§2.3 conflict note). Sayed's scenario ask needs a horizon decision **and** the deferred VLM enrichment. | Should go back to the PI as a scope question, not be silently narrowed. | PI decision |

---

## 8. Open gaps / UNVERIFIED

- ✅ **RESOLVED 2026-07-21 — cross-arm generality.** Every "MEASURED (this doc)" number was from **flagship-30k only**; tier-0 has now run on all **24** committed dumps (`Benchmarks & Eval/LEADERBOARD.md` §2–§4). Verdict: **the CQ/TR inversion is universal** — all 14 leaderboard arms are CI-separated *against* hold-v0 on the 639 steady windows, from −0.054 (refc-base) to −7.28 (flagship v2) — while **the straight-line heading result is arm-specific**: `refb` (1.854°) and `refb-10k` (2.181°) sit close to CV's 1.399° and the REF-C arms roughly halve the flagship's error (3.863°), so 7.980° is a flagship property, not a corpus artifact. Two further cross-arm findings the single-arm doc could not see: (i) among the rank-1= three-way ADE tie, **only the two REF-C arms separate on along-track** — flagship v1 does not; (ii) `flagship-v16-ab-ft` is the **only arm in the program with a CI-separated speed-MAE win over CV** (+0.0785 [+0.0066, +0.1516]) and it bought that by *losing* lateral geometry (cross 0.423 vs 0.237, path-geometry 0.204 vs 0.111, κ-sign 0.865 vs 0.954) at an ADE that is a statistical tie.
- **Heading, restated under its own R5 rule.** The doc's "5.7× worse than a straight line" is a ratio of **means** (7.980 / 1.399). On the **median** — the reducer R5 mandates for heading — it is **2.45×** (1.105 / 0.451). Both are now emitted (`heading_med_2s_deg`); the mean is the tail evidence, the median is the headline.
- **UNVERIFIED — regime thresholds.** The ±0.5 m/s² brake/steady/accel split and the `|κ_gt| < 1e-3` straight gate are PROPOSED. The measured effect is large (Δ −0.212 m/s on cruise) and unlikely to be threshold-driven, but a sensitivity sweep is owed.
- **UNVERIFIED — 4-waypoint speed is a 0.5 s box-average**, not an instantaneous speed. The CQ/TR values are therefore *conservative* (smoothing hides jitter). Tier-1 P1 will make them stricter, not looser.
- **UNVERIFIED — WOMD miss thresholds re-derived for 2 s.** The published values start at 3 s; the 2 s scaling is PROPOSED and needs deriving, not guessing.
- **UNVERIFIED — WOMD `overlap rate` top-1-only rule and the mAP duplicate-FP rule** could not be quoted verbatim from primary sources (both are REFUSED for us anyway).
- ✅ **ADDRESSED 2026-07-21 — how TanitEval v2 rows enter `LEADERBOARD.md`.** `Benchmarks & Eval/LEADERBOARD.md` was rewritten from `MODEL_REGISTRY.md` §6 with units labelled on every column (metric-BEV `ade_0_2s` in m, not the old camera-frame ADE@1s), the tier-0 tables as §2–§4, and the camera-frame gate ladder demoted to a clearly-labelled historical section — which closes registry gap **R5**. §2's table is regenerable: `python -m taniteval.driving --leaderboard`. **The gate protocol is still not addressed** — `Project Steering/GATE_PROTOCOL.md` decides on ADE-family metrics only, and nothing in it yet consumes the along/cross split or the cruise-quality result. That is the remaining follow-up.

---

## Appendix A — flagship v1 (`flagship-30k`) under TanitEval v2, tier-0

*MEASURED (this doc). n = 881 windows / 40 episodes. Intervals: episode-cluster bootstrap, B = 2000. Deltas: paired episode-cluster bootstrap. Sanity pin vs MODEL_REGISTRY: all green.*

| metric | model | CI95 | CV floor | hold-v0 floor | paired vs CV | separated |
|---|---|---|---|---|---|:--:|
| ADE 0–2 s (m) | **0.4271** | [0.3675, 0.4871] | 0.8377 | 0.7876 | +0.4106 [+0.2050, +0.6240] | ✅ |
| FDE @2 s (m) | 0.9075 | [0.7851, 1.0306] | — | — | — | — |
| miss @2 m | 0.0454 | [0.0239, 0.0681] | — | — | — | — |
| **\|along\| @2 s (m)** | **0.8412** | [0.7293, 0.9591] | 1.0955 | 1.1040 | **+0.2543 [−0.0278, +0.5304]** | **❌** |
| along bias @2 s (m) | +0.3498 | [+0.1366, +0.5486] | — | — | — | — |
| **speed MAE (m/s)** | **0.4710** | [0.4156, 0.5276] | 0.4678 | 0.4818 | **−0.0032 [−0.1285, +0.1182]** | **❌** |
| speed bias (m/s) | +0.1911 | [+0.0859, +0.2907] | — | — | — | — |
| \|Δprogress\| (m) | 0.8370 | [0.7248, 0.9523] | — | — | — | — |
| **\|cross\| @2 s (m)** | **0.2369** | [0.1820, 0.2960] | 1.0089 | 0.9137 | **+0.7720 [+0.4166, +1.1914]** | **✅** |
| path geometry (m) | 0.1110 | [0.0860, 0.1402] | 0.5217 | 0.4652 | +0.4107 [+0.2250, +0.6293] | ✅ |
| heading MAE @2 s (°) | 6.61 | [2.34, 12.02] ⚠️ | 6.623 | 6.344 | +0.0168 [−5.70, +4.30] | ❌ |
| curvature sign agree | 0.9535 | [0.9364, 0.9693] | 0.6103 | 0.5119 | — | — |
| long_frac of 2 s sq-err | 0.8933 | — | 0.4134 | 0.4431 | — | — |

**Longitudinal regime** (realised mean accel, ±0.5 m/s²)

| regime | n | model ADE | hold-v0 ADE | model speed MAE | hold-v0 speed MAE | paired Δ (hold-v0 − model) | separated |
|---|---:|---:|---:|---:|---:|---|:--:|
| brake | 95 | 0.5196 | 1.2569 | 0.6104 | 1.2536 | **+0.6433 [+0.4466, +0.8276]** | ✅ |
| **steady** | **639** | 0.3834 | 0.5430 | **0.4231** | **0.2109** | **−0.2122 [−0.2778, −0.1443]** | ✅ **against the model** |
| accel | 147 | 0.5575 | 1.5475 | 0.5889 | 1.1605 | **+0.5716 [+0.4336, +0.7076]** | ✅ |

**Speed strata** (thresholds 7.067 / 13.325 / 27.488 m/s)

| stratum | n | model ADE | CV ADE | long-RMSE | lat-RMSE | long_frac | speed bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 294 | 0.3594 | 0.9322 | 1.0319 | 0.1547 | 0.978 | −0.127 |
| med | 293 | 0.3704 | 0.9345 | 0.9590 | 0.3212 | 0.899 | +0.279 |
| high | 294 | 0.5513 | 0.6468 | 1.1278 | 0.5118 | 0.829 | +0.422 |
| **top 10 %** | **89** | **0.7159** | **0.3003** | 1.3789 | 0.6312 | 0.827 | **+0.659** |

⚠️ **At the top speed decile the model LOSES to constant velocity, 0.7159 vs 0.3003.** Consistent with the registry's skill-vs-floor **1.785** on that stratum; the tier-0 surface shows *why* — long-RMSE 1.38 m against a +0.659 m/s speed over-prediction.

**Curvature buckets**

| bucket | n | model heading MAE | CV heading MAE | model κ-sign | CV κ-sign |
|---|---:|---:|---:|---:|---:|
| **straight (<5°)** | **634** | **7.98°** | **1.399°** | 0.9469 | 0.7639 |
| gentle (5–15°) | 103 | 2.06° | 7.852° | 0.9320 | 0.2330 |
| sharp (≥15°) | 144 | 3.811° | 28.743° | 0.9977 | 0.2037 |

---

## Appendix B — one-line summary of the whole design

**Keep ADE as a component; stop using it as the verdict.** Score capability as **(longitudinal, lateral, navigation, tactical, hierarchical)** — each against the strongest trivial floor, each with a paired episode-cluster interval, each stratified by regime — then gate on **admissibility** (10 Hz at p99 + CI-separated over every floor) and rank on a **Pareto frontier**, never a scalar. Where the data cannot support a metric, say so in the suite rather than shipping a number that looks like the thing it cannot measure.
