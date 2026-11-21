# Data strategy and data building for the hierarchy proof

**Date:** 2026-07-26 (Europe/Berlin) · **Author:** chief-architect, 4-brain dominance stream
**Part §2 of** `4BRAIN_DOMINANCE_PROGRAM.md` · **Companion:** `STRATEGIC_TACTICAL_PROBLEM_SPEC.md`
**Compute used:** dev box only. Zero GPU, zero pod SSH, nothing staged, nothing committed.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (our doc,
not re-verified here) · `ESTIMATED` (arithmetic shown) · `HYPOTHESIS` · `PLANNED` (to be built).

---

## 0. The verdict, in one table

**The question a data strategy must answer:** *which corpus can carry a decision problem — i.e. supply
an option set, a non-circular target, and enough episode-clusters to power a two-arm test?*

| Corpus | Option set (map/lane graph) | Reactive agents | Non-circular strategic target | Horizon | Power today | Publishable | **Role** |
|---|:--:|:--:|---|---|---|:--:|---|
| **PhysicalAI-AV parity** (2,376 eps, 13.13 h) | ❌ **none — card says so outright** | ❌ replay | 🟡 kinematic arc only (v2.1/v3, 80.4 %) | 20 s hard ceiling (**K ≤ 190**) | decision stratum **n ≈ 13** | ❌ `gated-confidential` | **TRAIN + operative/O1 + HP-1/5/6** |
| **PhysicalAI v2-balanced** (9,000 clips, 49.74 h) | ❌ none | ❌ replay | 🟡 same labels, **2× turn density** | same | junction clips ~5,500 | ❌ same gate | **TRAIN (the ladder's corpus)** |
| ⭐ **AlpaSim NuRec scenes** | ✅ **`trajdata.VectorMap`, 130–472 lane polygons + 130–393 road edges + wait-lines per scene** | ✅ **SMART+CAT-K in-tree, Apache-2.0, NEVER ENABLED** | ✅ map ∩ realized path; **and `Y_outcome` needs no label at all** | 10 s rollouts, extensible | **n=37 balanced today; per-category n=6–8** | ❌ gated + no-derivative renderer + ~12-mo term | ⭐ **PROVE — the only instrument with both halves** |
| **Cosmos-Drive-Dreams** | 🟡 HD map shipped; lane-graph adapter unbuilt | ❌ replay | 🟡 map available, untested | 10 s clips | 5,843 labelled clips; **roundabout count UNKNOWN** | ✅ **CC-BY-4.0, `owned-safe`, commercial-OK, already loaded** | ⭐ **PUBLISH — the plausible twin** |
| **nuScenes** | ✅ 11 map layers, I+E, instance tracks + visibility | ❌ replay | ✅ turn-key | 20 s | 1,000 scenes; roundabouts sparse | ❌ `nc-research` (likely **SA**) | 🟡 external check only |
| **L2D** | 🟡 OSM topology per frame; **no intrinsics, no 3D agents** | ❌ replay | ✅ NL instruction names the exit; **96.74 % carry a metric distance** | ✅ **drives reconstruct to 35.4 min** | **3,532 roundabout eps**, 6,728 right-of-way, 2,080 unprotected | ✅ Apache-2.0 → `ship` | 🟡 the only EU-roundabout **shippable** path; expensive |

> **The one-line verdict.** ***No single corpus can carry the proof.*** The program needs a
> **three-corpus composition**: **train** on PhysicalAI (parity/v2), **prove** on AlpaSim (the only
> asset with an option set *and* reactive agents), **publish** on Cosmos-Drive-Dreams (the only
> `ship`-tier multi-camera asset with an HD map that we already own). Everything below prices that.

---

## 1. PhysicalAI-AV — honest assessment

### 1.1 What it has

`MEASURED` (`CORPUS_PROFILE.md`, poses+labels on pod3, read-only):

| quantity | parity `physicalai-train-e438721ae894` | v2-balanced `physicalai-v2bal-4b7eeeac222d` |
|---|---:|---:|
| clips / hours | **2,376 / 13.13 h** (472,627 frames, 406,099 windows) | **9,000 / 49.74 h** |
| clips containing a junction | 37.8 % | **61.4 %** |
| clips containing a turn | 42.6 % | **76.5 %** |
| step-weighted turn_left+turn_right | **14.2 %** | **28.0 %** |
| clips with **no** turn at all | **57.4 %** | — |
| net heading > 90° (roundabout-ish) | **10.4 %** | 13.9 % |
| epochs at 30 k steps | **4.73** | — |

`MEASURED` (H2 substrate audit, via `LOOP_STATE`): **intersections 846 eps** and **lane-changes 1,172
eps** are *powered*; **roundabouts 19 strict ⇒ descoped**.

### 1.2 What it structurally cannot have

`MEASURED` + `PUBLISHED` (`DATA_STRATEGY_FOR_HIERARCHY.md` §1, three independent legs):

- **No geo, and it was never there.** NVIDIA's card: *"we do not include open maps data."* Keyword
  sweep of the full 31,935-byte card: `GPS` 0 · `latitude` 0 · `HD map` 0 · `lane` 0 · `traffic light`
  0. Both pose products are **clip-local**, starting at the origin. `metadata/data_collection.parquet`
  has no location field finer than **country**.
- ⇒ **Map-matching (GPS → OSM) is inapplicable and cannot be made applicable.** The thread is closed.
- ⇒ **The option set does not exist**, so S1/S2/S4 have **no ground truth** on this corpus. `MEASURED`
  corroboration from our own labels: **4 of 9 v3 ROUTE tokens are never minted** — `straight`,
  `exit_left`, `exit_right`, `merge` — each because it *"asserts a junction exists = a MAP fact"*.
- **The 20 s ceiling is hard.** Clips are 188–205 frames; `corridor.horizon_ceiling(T) = T−W−1` ⇒
  **K ≤ 190 (19.0 s)**. K=200 is structurally impossible. `MEASURED`.
- ⚠️ **`clips` are not drives** — over all 3,146 chunks the largest `(country, month, hour, platform)`
  group inside a chunk is a median **5.1 %**; the archive is shuffled at ingest, so temporally adjacent
  clips cannot be re-chained. **There is no path to a longer horizon on this corpus.**

### 1.3 The one unexploited asset — and the caution attached to it

`obstacle.offline`: **3D agent tracks on 96.90 % of our corpus** (track id, class, 3D box, orientation,
**already in the ego rig frame**); median **134 tracks per 20 s clip**, **38.9 agents/frame**;
corpus-wide lead presence **38.51 % of windows / 66.1 % of clips**. Our ingest reads **2 of 36
features**. `MEASURED`.

> ⚠️ **The retraction that governs this.** `RETRACTION_LOG` 07-21 (C3): *"`obstacle.offline` unblocks
> agent-relational tactics ⇒ ingest all 197 chunks this week"* was **falsified by its own pre-registered
> gate** — adding lead state to an ego-state regressor moves the 2 s longitudinal target by
> **+1.16 % [−0.92, +3.19]**, inside the FAIL band, ≤ +1.83 % out to **6 s**. **Cost of finding out:
> 1.1 GB, 0 GPU-h.**
>
> **What this plan does differently.** That gate tested tracks as an **input feature for trajectory
> regression**. This plan uses them for **`Y_expert` target and option-set construction** on a different
> surface (T1–T3) — a *different claim*, which therefore gets **its own pre-registered gate on one
> chunk before any 12.4 GB ingest** (`PROGRAM` §3, D-G6). **Do not spend the ingest on the old
> argument, and do not read the old null as covering the new one.**

### 1.4 Verdict

**PhysicalAI is the training corpus and the operative/horizon instrument. It is not, and cannot be
made into, a strategic-decision corpus.** It carries **HP-1** (horizon growth), **HP-5** (data
efficiency), **HP-6** (recovery), **S3** (manoeuvre timing) and **O1**. It cannot carry **S1, S2, S4,
T1–T4, or HP-4**.

---

## 2. ⭐ AlpaSim — what changes, and what it costs

### 2.1 The two findings that make strategic and tactical problems possible

`MEASURED` (`ALPASIM_STATE.md` §7 + `gate0_prereq_probe.json`, 5 junction scenes):

1. **A full `trajdata.VectorMap` per scene USDZ**, loadable at inference via `ArtifactSceneProvider`:
   **130–472 lane polygons**, 130–393 road edges, plus wait-lines. Our PhysicalAI corpus has none of
   this.
2. **An unused reactive-agent model in-tree**: `src/trafficsim/alpasim_trafficsim/catk/smart/` —
   SMART trajectory-tokenisation + **CAT-K** closed-loop fine-tuning, **Apache-2.0**, present on the
   pod, **disabled in every run we have ever made**.

A strategic problem needs a topology to choose *over*; a tactical problem needs an agent to choose
*against*. **We have owned both for weeks and switched on neither.**

### 2.2 ⚠️ The gating unknown — connectivity, not polygons

`gate0_prereq_probe.json` measured **counts**, and its trajectory read *errored*
(`AttributeError("'utils_rs.Trajectory' object has no attribute 'poses'")`,
`IndexError` on the closest road edge). **It never read lane successors.**

> 🟥 **If `VectorMap` carries polygons but not connectivity, `succ(lane)` must be reconstructed
> geometrically (endpoint proximity + heading continuity), which is a real sub-project with its own
> error rate — and S1/S2/S4's option sets inherit that error as label noise.**
>
> **This is the single highest-leverage $0 probe in the entire program.** It gates four of the nine
> problems. Two probes required (CLAUDE.md rule 2): the `trajdata` API on a loaded scene **and** raw
> USDZ prim inspection. Runnable today on the free eval pod.

### 2.3 The measured numbers to use — and the one to stop using

`MEASURED` (`scenario_stratified_scaled_results.json`, n=37 balanced, paired):

| category | n | flag pass | refc pass | ΔScore (flag−refc) | read |
|---|---:|---|---|---|---|
| roundabout | 8 | 2/8 | 2/8 | **+0.002** | **dead heat** |
| highway | 8 | 2/8 | 3/8 | −0.074 [−0.294, +0.144] | **tie**, CI ∋ 0 |
| intersection | 7 | **0/7** | 1/7 | −0.063 | **both collapse**; flagship offroad 6/7 |
| traffic_light | 6 | 2/6 | 4/6 | −0.224 | REF-C wins |
| straight_other | 8 | 3/8 | 5/8 | −0.272 | REF-C's best category |
| **OVERALL** | **37** | 9/37 | 15/37 | **−0.1228 [−0.2079, −0.0412]** | REF-C wins **modestly**, geometry-dependently |

⛔ **Do not quote the n=12 suite's Δ −0.43.** That suite was **8/12 straight-or-urban — REF-C's single
best category** — and the balanced number is **~3.5× smaller**.

### 2.4 The binding limits — carry all four or the number is wrong

| # | Limit | Consequence |
|---|---|---|
| 1 | **Reconstruction OOD 3.21×** (`MEASURED`, REF-C open-loop ADE **1.5157** in-sim vs **0.4728** real, 4 scenes / 288 preds) | **Absolute rates are model × reconstruction-fidelity and are inadmissible.** ✅ **Paired arm-vs-arm deltas ARE admissible — both arms see the same OOD input, so the reconstruction term differences out.** This is exactly what the proof needs, and it is the reason AlpaSim works for us at all. |
| 2 | **Licence** — renderer NRE is NGC-gated, closed-binary, **forbids derivative works**, and names *"autonomous vehicle applications"* a **Critical Application** it is *"not tested or certified"* for. Scenes are gated-confidential with a **~12-month term**. Simulator source is Apache-2.0. | **AlpaSim is an INTERNAL instrument, unforked.** Any public claim needs a `ship`-tier twin (§4). *"Safety-grade closed-loop"* is a phrase to retire for this stack. |
| 3 | **Per-category n=6–8 is directional, not powered.** Only the overall −0.1228 has a CI excluding zero. | Scaling the suite is a **prerequisite**, not a nicety (§5). |
| 4 | **Real-time factor 0.75–0.98× @480×854; 0.29× native.** Renderer-bound. | Eval is cheap in GPU-days but real in wall-clock; budget in §5.3. |

### 2.5 ⭐ The unlock nobody has priced: `Y_outcome` needs no label

With `trafficsim` ON, T1–T4's ground truth is **the simulated consequence**, not an annotation:
`at_fault_collision`, `conflict_clearance_margin`, `induced_deceleration_on_follower`, `deadlock`,
`progress`. **A consequence cannot be circular with an input by construction** — it is the strongest
admissibility position available anywhere in this program, and it costs **~1–3 eng-days** to switch on.

---

## 3. External corpora — the publishable-twin decision

Scope from `H2_EXTERNAL_DATA_SURVEY.md` (2026-07-25). Ranked *for the hierarchy proof*, which is a
different criterion than that survey's (which ranked for multi-camera H2).

| # | Corpus | Why it ranks here | Cost | Blocking unknown |
|:--:|---|---|---|---|
| **1** | ⭐ **Cosmos-Drive-Dreams** | The **only** fully-specified asset that is `owned-safe` **CC-BY-4.0, commercial-OK, already loaded** (registry `cosmos_dd`, since D-014), with **7 cameras, pinhole intrinsics + 30 fps poses, 4D object tracking with IDs and movement state, and an HD map**. Zero new licence risk, zero new registry entry. | **2–3 eng-days** (cheapest by 2×; the 120° f-theta → canonical crop path is the proven ZOD-class drop-in) | **Roundabout / multi-option-junction count is UNKNOWN.** ⭐ **Answerable for $0 from already-cached metadata — do this before ranking it.** |
| **2** | **nuScenes** | The one place the label mechanism is turn-key: I+E per camera, 23-class 3D boxes with **instance tracks and a visibility attribute**, `ego_pose`, 11 map layers. ~60 GB keyframes-only. | 4–6 eng-days + registration (⚠️ **a human must accept the terms — never a subagent**) | `nc-research`, and ⚠️ **possibly ShareAlike** — `schema.py` registers `share_alike=False` against nuscenes.org's CC-BY-NC-**SA**-4.0. **Escalate; do not fix in passing.** Also `fx ≈ 1266` on 1600×900 ⇒ `f_eff ≈ 360` vs canonical 266 — the PandaSet-class geometry wall; the fix (`calib_r1.pinhole_rectify`, validated 9/9) is **still not folded into `stack/tanitad/data/calib.py`**. |
| **3** | **L2D** | The **only** source that is simultaneously a real 6-camera EU rig, **commercially clean (Apache-2.0)**, **roundabout-rich** (3,532 roundabout eps, 6,728 right-of-way, 2,080 unprotected turns, 96.74 % of instructions carry a metric distance), and **long-horizon** (drives reconstruct to **35.4 min**; ≥120 s on 78.6 % of episodes). | **10–15 eng-days** | **No intrinsics ship at all** (only `extrinsic_RDF.yaml`) → GeoCalib-class estimation + the geometry falsifier is *mandatory*; **no 3D agents at all** → the tactical target must be pseudo-labelled. ⚠️ **90.8 % of consecutive episodes overlap (median stride 13.76 s, shared frames byte-identical) — de-duplicate by timestamp and split on reconstructed DRIVES, never episodes**, or the split leaks exactly as REF-A's I-JEPA val did. |
| — | Argoverse 2 / nuPlan / ONCE / KITTI-360 | dominated: AV2 by nuScenes; nuPlan by **16 TB for 10 % of logs**; KITTI-360 by **73.7 km total**, too small for 40 clusters | — | — |
| ⛔ | **Waymo Open / Waymax** | **`refuse`** — terms follow the **trained weights**; `assemble_lake_record` raises. Not proposed. | — | — |
| ⛔ | A2D2 | double kill: 3D boxes only inside the **front** camera's FOV; **CC-BY-ND** forbids derivatives | — | — |

**The publishability rule that decides this.** PhysicalAI-AV is `gated-confidential`; AlpaSim's renderer
forbids derivative works and its scenes carry a term. **Neither can carry a public claim.** Per
`TANITDATASET_TIER_INTEGRATION` §4 a derivative inherits the strictest input tier, so the hierarchy
result itself inherits the gate. ⇒ **If the 4-brain dominance result is ever to be a paper or a
product USP, a `ship`-tier twin is required regardless of the counts** — and Cosmos-DD is the only
candidate that is already ours, already loaded, and already commercial-OK.

---

## 4. What must be BUILT — the data-building work package

Six items. Each states the artifact, the non-circularity argument, and the falsifier.

### D-B1 — The decision-point miner (`PLANNED`)

**Input:** a scene's `trajdata.VectorMap` + the ego's realized pose track.
**Output:** one `DecisionPoint` record per multi-option junction approach.

```jsonc
{
  "scene_id": "clipgt-…", "t_dp_us": 1234567,        // last frame with |succ| >= 2 and entry in [15,60] m
  "decision_type": "branch" | "lane" | "roundabout_exit" | "merge",
  "ego_lane_id": "…",
  "option_set": [ {"lane_id": "…", "centerline": [[x,y],…], "heading_at_entry": 1.23,
                   "reachable_goal_ids": ["…"]}, … ],   // arity 2..K, FROM THE MAP
  "chosen_index": 1,                                     // FROM THE REALIZED PATH (polygon containment)
  "goal_easy":  {"polyline_ego": [[x,y]×6]},             // route truncated at 30 m
  "goal_hard":  {"point_ego": [x,y]},                    // single point at 150-200 m, names no direction
  "horizon_s": 20.0,
  "provenance": {"map": "trajdata.VectorMap@usdz", "path": "realized", "labeler": "dpm_v1"}
}
```

**Non-circularity:** `option_set` ⟸ map alone. `chosen_index` ⟸ map ∩ realized future. `goal_*` ⟸ map ∩
destination. **No field is a function of another field that the model also receives.**
**Falsifier:** `blind_conditioning_baseline(goal_hard → chosen_index)` reaches ≥ 0.98 × ceiling ⇒ the
goal encoding leaks the answer ⇒ **push the goal further out and re-run.** Report `acc_blind` always.

### D-B2 — Option-set construction from lane polygons (`PLANNED`, gated on §2.2)

Two implementations, and **which one ships depends on the connectivity probe**:

| case | method | error model |
|---|---|---|
| **connectivity present** | `succ(lane)` read directly | exact |
| **polygons only** | geometric successor: endpoint within `ε` **and** heading continuity within `θ` | **must be validated** against ≥50 hand-checked junctions; report precision/recall; the option-set error becomes label noise and **must be quoted with every S1/S2/S4 number** |

**Falsifier:** geometric-successor recall < 0.90 on the hand-checked set ⇒ S1/S2/S4 are not buildable
on AlpaSim and the strategic half of the proof moves to Cosmos-DD/nuScenes.

### D-B3 — Counterfactual ground truth for HP-3 (`PLANNED`)

For each `DecisionPoint`, emit a **goal-swap family**: the same observation window paired with one
goal per option (`goal_k` = a point reachable only via option `k`). Then:

- **divergence:** `‖traj(goal_i) − traj(goal_j)‖` — headline `cross_track_2s_m` and its **p90**
  (`strategic_probes.py` already computes this shape).
- **correctness:** does the executed path enter option `k` when given `goal_k`?
  → **`counterfactual_route_correctness`**, chance = `1/|options|`, and the CI must clear chance.

**This is the discriminating experiment that needs zero training and that our current data cannot
express at all** — because a counterfactual requires *the branch not taken*, and only a map has it.
`strategic_probes.py` (500 lines, 17 tests, `MEASURED` on synthetic fixtures: the FLAT arm scores
**exactly 0** divergence while echoing the command **1.0**) is the harness; the option sets are what it
lacks.

### D-B4 — The tactical-conflict record (`PLANNED`, gated on §7 item 2)

```jsonc
{ "scene_id":"…", "t_dp_us":…, "conflict_type":"unprotected_left"|"merge"|"uncontrolled_xing"|"overtake",
  "ego_branch_id":"…", "agents":[{"track_id":…,"class":"automobile","ttc_s":…,"gap_m":…,
                                  "conflict_point":[x,y],"is_reactive":true}],
  "Y_expert": {"ego_first": true},                 // replay-safe, WEAK (one sample of a bimodal choice)
  "Y_outcome": {"at_fault_collision":false,"clearance_margin_s":1.9,
                "induced_decel_follower_ms2":0.4,"progress_frac":0.87,"deadlock":false} }
```

⚠️ `Y_expert` and `Y_outcome` are **never pooled**. `Y_outcome` requires `is_reactive: true` — on a
replay it is not a counterfactual, it is a description.

### D-B5 — Route-compliance and decision metrics into `taniteval/` (`PLANNED`)

Already landed and reusable: `corridor.py` (474 L — `corridor_departure` at **arbitrary K**, junction
stratum, `paired_stratum_delta`, `horizon_ceiling`), `lateral.py` (555 L — lat/lon split, tails,
`paired_cross_track` with `reduce="p90"`), `strategic_probes.py` (500 L — HP-3), `hierarchy.py`
(migrated to `episode_cluster_bootstrap`, emits `PC1_route_input_works`). `MEASURED`: **286 tests pass**.

**To build:** `route_compliance_rate`, `lane_feasibility_rate`, `exit_ordinal_accuracy`,
`between_branch_rate` (HP-7), `branch_flip_rate` (HP-8), `counterfactual_route_correctness`,
`frozen_robot_rate`. All consume `pred_dense`/`gt_dense`, which T3-11 now persists.

🟠 **Blocker inherited from HPP-1 §5:** *no committed `results/windows_*.pt` has `pred_dense`* ⇒
**`corridor.from_windows` cannot run on any archived arm.** One `rollout.collect` re-run per arm
(GPU, minutes). And `driving.tier0` does not yet call `lateral.block` / `corridor.from_windows` —
**two lines, both guarded, and it needs an owner or it becomes the next 10-day orphan.**

### D-B6 — The circularity firewall (`PLANNED`, CPU-minutes)

`blind_conditioning_baseline` (§0.1 of the problem spec) implemented in `taniteval/`, plus a regression
test that a synthetic echo label is **REFUSED**. Run against **every existing label** —
`route_target`, `route_target_v21`, `route_from_future_v3`, `ttm`, `dist_target`, the v4 goal scalars —
and publish the `acc_blind` table. **This would have caught `route_target = _NAV_TO_ROUTE[nav_cmd]`
for CPU-minutes instead of months.**

---

## 5. Coverage targets and the power calculation

### 5.1 The bars

| claim shape | resampling unit | bar | source |
|---|---|---:|---|
| single-arm proportion ("X % of junctions") | val episode / scene | **n ≥ 40 per stratum** | program standing bar = `taniteval/ci.py`'s unit |
| **two-arm paired comparison** | the same | **n ≥ 200 per stratum** | brief; corroborated by the arithmetic below |

**The arithmetic behind 200** (`ESTIMATED`, reasoning shown). For a paired binary outcome under the
normal approximation, `n ≈ (z_{α/2}+z_β)²·(p01+p10)/(p01−p10)²`. At 80 % power / α=0.05 two-sided, with
a discordance mass of 0.30 and a **10 pp** paired difference: `n ≈ 7.84 × 0.30 / 0.01 ≈ 235`. At **15 pp**
with discordance 0.35: `n ≈ 122`. ⇒ **200 clusters detects ~10 pp; 40 clusters detects only ~25 pp.**

For a **continuous** paired metric, `n ≈ 7.84/d²`: `d=0.50 → n≈32` · `d=0.30 → n≈87` · `d=0.20 → n≈196`.
Calibrating against our own measured effects:
- **E1b junction corridor-departure** Δ **−0.4270 [−0.6838, −0.1648]** at n=44 held-out episodes ⇒
  `d ≈ 0.48` ⇒ **40 clusters is sufficient for closed-loop departure effects of that size.**
- **AlpaSim balanced suite** Δscore **−0.1228 [−0.2079, −0.0412]** at n=37 ⇒ `d ≈ 0.475`, *just*
  separated. **Detecting half that effect needs n ≈ 136.**

### 5.2 Where we stand against the bars

| stratum | today | bar (2-arm) | gap |
|---|---:|---:|---|
| PhysicalAI **route decisions** (canonical val) | **n ≈ 13** | 200 | **15×** — and the corpus cannot supply an option set at any n |
| PhysicalAI **junction windows** @K=185 (E1a held-out) | n = 6 junction episodes | 200 | **33×** ⚠️ the E1a junction number rests on **6 episodes** |
| AlpaSim **overall** | **n = 37** | 200 | **5.4×** |
| AlpaSim **per category** | n = 6–8 | 40 (single-arm) / 200 | **5–33×** |

### 5.3 The build plan to close it

**A. PhysicalAI route-decision set (`route_eval_v1`) — cheap, and it triples what we have.**
Build on the **v2.1/v3 adaptive-arc labels (80.43 % coverage)** instead of the v1 fixed-25 s labeler
(27.2 %), drawn from the **v2-balanced corpus** (junction 61.4 %, turns 28.0 % step-weighted).
Parity-safe: v2 selection re-derives labels on the same episodes and never re-selects
(`corpus_key_match: true`, `skip_hash f09e44db` unchanged). Stratify by **route token × `dist_target`
band** so "approaching a decision" separates from "executing a turn" from "cruise".
`ESTIMATED` yield: 9,000 clips × 61.4 % junction ≈ **5,500 junction clips** ⇒ ≥200 per stratum is
comfortably met **for the strata that do not need a map** (S3, HP-1, HP-2, HP-5, HP-6). **0 GPU.**

**B. AlpaSim suite scale-up — the dominant data cost, and it is affordable.**
The pipeline is **committed and proven**: `kf_download.sh → kf_batch.py → select_suite.py →
scaled_wizard_gen.sh → scaled_master.sh → scaled_aggregate2.py`, with **356 screened keyframes'
worth of labelling already banked** (`keyframes/` 12, `scaled_keyframes/` 38,
`scaled_roundabout_verify/` 16 auditable). Source pool: **1,606 scenes (`public_2604`)**.

| resource | arithmetic | figure |
|---|---|---|
| **rollout wall-clock** | 50 steps @5 Hz ≈ 10 s sim ÷ RTF 0.75–0.98 ≈ **10–13 s render**, + scene load/teardown ⇒ **~3 min per (scene, arm)** `ESTIMATED` | 200 scenes × 3 arms × 3 min ≈ **30 h ≈ 1.25 pod-days** |
| **scene bytes** | 1.5–1.7 GB each | **~320 GB for 200 scenes** |
| ⚠️ **quota** | `/workspace` MooseFS quota **~466 GB**, which `df` does **not** show | ⇒ **stream-and-delete per scene**; verify capacity with a real `dd`, never `df` |

⇒ **AlpaSim evaluation is CHEAP relative to training.** The expensive thing in this program is arms,
not scenes. That inverts the usual sequencing intuition and it is the key scheduling fact of §4 of the
main plan.

**C. ⚠️ The scarcity risk that must be measured before the scale-up is scheduled.**
The balanced suite reached **8 roundabout scenes** from **356 screened keyframes**, and the notes record
that it *"turned 0 roundabout scenes into 8"* — i.e. roundabouts were the scarce category.
`ESTIMATED`, and **explicitly not decision-grade** (this is a scalar off a small sample — class C5):
naive extrapolation of 8/356 to the 1,606-scene pool gives **~36 roundabout scenes**, which is **below
even the 40-cluster single-arm bar**.

> ⭐ **$0 GATE, run it first:** count the category frequencies over the **356 already-banked screened
> labels** and project onto the pool. **If the pool cannot supply ≥40 roundabout scenes, S4 is not
> powerable on AlpaSim and moves to L2D (3,532 roundabout episodes) or is descoped with that stated.**
> This costs arithmetic and it prevents scheduling a scale-up that cannot reach its bar.

**D. Cosmos-Drive-Dreams $0 metadata count.** Same shape: count junction/roundabout/multi-option
content from already-cached clip metadata **before** any download. Decides whether a publishable twin
exists at all.

---

## 6. The composition, and what each corpus is licensed to say

| Corpus | Tier | Can train | Can produce an internal number | Can produce a **published** number |
|---|---|:--:|:--:|:--:|
| PhysicalAI-AV | `gated-confidential` / firewalled | ✅ | ✅ | ❌ |
| AlpaSim (NuRec scenes + NRE renderer) | gated, no-derivative renderer, ~12-mo term | ❌ (eval only) | ✅ **paired deltas only** | ❌ |
| Cosmos-Drive-Dreams | **`owned-safe` CC-BY-4.0** | ✅ | ✅ | ✅ **attribution only** |
| L2D | **`owned-safe` Apache-2.0** | ✅ | ✅ | ✅ ⚠️ GDPR: never re-host frames without a face/plate check |
| nuScenes | `nc-research` (SA?) | ✅ internal | ✅ | ❌ + derivative inherits `nc` |

> **The derivative rule bites hardest here.** *A hierarchy result trained on PhysicalAI and proven on
> AlpaSim is `gated-confidential` twice over.* If the PI wants this to be a paper, the **twin run on
> Cosmos-DD is not optional and it must be scheduled from the start**, not bolted on after the internal
> result exists. That is the difference between a 3-week and a 3-month path to publication.

---

## 7. Risks to the data plan itself, each with its falsifier

| # | Risk | Falsifier / mitigation |
|:--:|---|---|
| 1 | **VectorMap has no connectivity** ⇒ option sets must be reconstructed geometrically | §2.2 probe (2 probes). If recall < 0.90 on 50 hand-checked junctions ⇒ strategic half moves off AlpaSim |
| 2 | **`trafficsim` does not actually work** — never once enabled | run one scene; verify non-ego agents **deviate from their logged tracks**. If they don't, T1–T4 lose `Y_outcome` and fall back to weak `Y_expert` |
| 3 | **Roundabout scarcity** in the 1,606-scene pool | §5.3-C $0 gate before scheduling the scale-up |
| 4 | **3.21× OOD swamps a small hierarchy effect** even in a paired design | the paired design differences the *mean* OOD term, not its *variance*. ⇒ **re-run the OOD control at native 1080** (~0.06 pod-day, already ranked #1 in `LOOP_STATE`) and report the ratio beside every AlpaSim delta |
| 5 | **`obstacle.offline` ingest repeats the 07-21 C3 error** | one-chunk pre-registered gate before the 12.4 GB ingest (§1.3) |
| 6 | **A new label turns out circular** | D-B6 firewall is a **pre-flight**, not a review |
| 7 | **L2D split leakage** (90.8 % episode overlap, byte-identical shared frames) | de-dup by unix timestamp; split on **reconstructed drives** — this is exactly the failure that made REF-A's I-JEPA val unusable |
| 8 | **The scenes expire** (~12-month term) | diary it; and it is a further argument for the Cosmos-DD twin |
| 9 | **The NRE image-pull procedure exists only as prose** in `LOOP_STATE.md` | 🔴 **the one real stranding.** `BUILD_AND_USE.md` §3 transcribes it but the script is **NOT re-run**. A pod reset costs the hardest step. ~30 min to verify, no GPU |

---

## 8. Sequenced data plan

| When | Item | GPU | Gates |
|---|---|:--:|---|
| **TODAY** | VectorMap **connectivity probe** (×2) · `trafficsim` one-scene probe · Cosmos-DD $0 count · AlpaSim category-frequency arithmetic on the 356 banked labels · `blind_conditioning_baseline` implemented | **0** | S1/S2/S4 buildability; T1–T4 buildability; twin existence; S4 powerability |
| **+2 d** | `route_eval_v1` on v2-balanced with v3 labels · junction/multi-option stratum in `driving.py` · wire `lateral.block` + `corridor.from_windows` into `driving.tier0` | **0** | HP-1/2/5/6 instrument |
| **+3–5 d** | Decision-point miner (D-B1/D-B2) on the 37 banked scenes · goal-swap families (D-B3) | **0** | HP-3 with **real** option sets |
| **+5–7 d** | `trafficsim` enabled + validated · tactical-conflict records (D-B4) | eval pod | T1–T4 |
| **+1–2 wk** | AlpaSim scale-up to **n ≥ 200** (≥40/category), stream-and-delete | eval pod ~1.25 pod-days | the powered proof surface |
| **parallel** | one `rollout.collect` re-run per archived arm to emit `pred_dense` | minutes/arm | unblocks corridor+lateral on history |
| **decision** | Cosmos-DD twin adapter (2–3 eng-days) **iff** the $0 count clears | 0 | publishability |

---

## 9. Deliverable manifest

| Artifact | Where |
|---|---|
| **This document** | `…/incoming/2026-07-26-4brain-dominance-program/DATA_STRATEGY.md` (working tree, **NOT staged**) |
| Companions | `4BRAIN_DOMINANCE_PROGRAM.md`, `STRATEGIC_TACTICAL_PROBLEM_SPEC.md` |
| Primary sources | `ALPASIM_STATE.md` · `TANITSIM_FORK_RECOMMENDATION.md` · `gate0_prereq_probe.json` · `scenario_stratified_scaled_results.json` (via `ALPASIM_STATE` §4.1) · `CORPUS_PROFILE.md` · `V2_CORPUS_QA.md` (via HPP-0 §4.2) · `labels_train_v4_provenance.json` (via HPP-0 §4.3) · `DATA_STRATEGY_FOR_HIERARCHY.md` · `H2_EXTERNAL_DATA_SURVEY.md` · `HPP0_CONFOUND_AUDIT.md` · `HPP1_UNBLOCK_REPORT.md` · `RETRACTION_LOG.md` · `MODEL_REGISTRY.md` §6 |
| Data downloaded / code changed / pods touched | **none** |
