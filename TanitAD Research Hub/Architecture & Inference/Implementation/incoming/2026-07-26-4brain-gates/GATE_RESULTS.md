# The two gates of the 4-Brain Dominance Program — results

**Date:** 2026-07-26 (Europe/Berlin) · **Author:** 4-brain critical-path agent
**Gates:** ① AlpaSim `trajdata.VectorMap` connectivity · ② `trafficsim` reactive agents
**Pods touched:** `tanitad-eval` **only**. pod1 / pod2 / pod3 never contacted.
**GPU used:** none (Gate 1 is pure CPU; Gate 2 runs CATK on CPU — see §2.3).
**Nothing staged, committed or pushed.**

**Evidence classes** (CLAUDE.md operating standard 1): `MEASURED` (ours + artifact path) ·
`PUBLISHED` (cited) · `INHERITED` (another doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.

> **Eval-pod occupancy, checked before use.** The brief flagged the eval pod as busy with a
> wheelbase measurement. `MEASURED` 2026-07-26 04:0x UTC, **three probes**: `ps -eo … | grep -i
> wheelbase` → empty · `nvidia-smi --query-compute-apps` → **no running processes, 0 MiB / 46068 MiB**
> · `find /workspace /root -iname '*wheelbase*'` → empty. The last write was `v4eval_15k.log`,
> finished ~30 min earlier. **The pod was idle and was used.** Four orphaned `alpasim`
> `multiprocessing-fork` workers (~2 GB RSS, 3 days old, 0 % CPU) were observed and **left alone**.

---

## 0. TL;DR — the two verdicts

| Gate | Question | Verdict |
|---|---|---|
| **① VectorMap connectivity** | is `succ(lane)` readable; can a junction's branches be enumerated; is `target_branch = the branch the ego actually took` computable? | ✅ **PASS, decisively.** Connectivity readable on **51/51 scenes**, two independent probes agreeing **51/51 exactly**. |
| **② reactive agents** | do non-ego agents DEVIATE from their logged tracks in response to the ego? | ❌ **FAIL.** They **do** deviate from the logs (**not replay** — 8.37–78.17 m mean; ≥95.5 % of poses differ from the logged pose), but the deviation is **not a response to the ego**: bounded at **[−0.21, +0.14] m** in the best-powered scene against a **4.5 m** sampling-noise floor, and **null in the near-ego stratum of 4/4 scenes** — while the ego itself differed between arms by up to **149 m**. |

**The load-bearing Gate-1 numbers** (`MEASURED`, `gate1_connectivity.json`, 51 scenes):
**11,877 lanes · 12,030 successor edges · 0 dangling · 790 lanes with |succ| ≥ 2 across 42/51 scenes ·
6,981 lanes (58.8 %) with a real parallel neighbour across 46/51 scenes · ego-to-lane match rate
mean 0.9827.**

⚠️ **And the binding limitation, stated up front:** the maps are richer than the program assumed, but
**the 51 scenes we hold yield only 20 scene-clusters carrying an S1 decision point** — **2× short of the
≥40 single-arm bar and 10× short of the ≥200 two-arm bar.** Gate 1 passing means the *instrument*
exists. It does **not** mean the *corpus* exists. §3.

**What the split verdict means for the programme.** `STRATEGIC_TACTICAL_PROBLEM_SPEC.md` states that
**7 of the 9 decision problems** are gated here. The gates do not resolve together:

- **S1 · S2 · S4 · HP-4 (the strategic half) are UNBLOCKED** — the map carries everything they need,
  and the non-circular target is computable today. They are now blocked only on **corpus size**, which
  is a download decision (§3).
- **T1 · T2 · T3 · T4 (the tactical half) are NOT unblocked.** Their shared admissibility argument is
  `Y_outcome` — *a simulated consequence, therefore non-circular by construction* — and that argument
  requires the consequence to be a function of the policy's choice. **It is not measurably one.**
  Per the brief this is not worked around: **it is a PI-level decision** (§2.6).

⭐ **The single most useful negative here:** the program has for weeks described `trafficsim` as an
owned-but-unswitched-on asset that *would* supply tactical ground truth. It is now switched on for the
first time — weights fetched, extensions built, service serving, **CATK genuinely simulating and
provably not replaying** — and the property the tactical problems actually depend on **does not
reproduce**. That is worth more than the assumption it replaces.

---

## 1. GATE 1 — VectorMap connectivity ✅ PASS

### 1.1 What was probed, and why twice

CLAUDE.md rule 2: *absence found at ONE location is not absence.* The converse binds equally — a
**presence** found at one location, through one library, is one code path. Two independent probes:

| | Probe **A** | Probe **B** |
|---|---|---|
| Path | AlpaSim `ArtifactSceneProvider` → `trajdata.VectorMap` object model | **raw USDZ archive**, `zipfile` + `pyarrow`, reading `clipgt/association.parquet` directly |
| Reads | `RoadLane.next_lanes / prev_lanes / adj_lanes_{left,right}` | `key.kind ∈ {NEXT_LANE, PREVIOUS_LANE, LEFT_LANE, RIGHT_LANE}` rows |
| Independence | uses `trajdata.dataset_specific.mads.populate_vector_map` | **`trajdata` is never imported on this path** |
| Scenes OK | **51/51** | **51/51** |

**The USDZ is a zip**, and its map payload is **MADS parquet**, not USD prims — so the honest
"raw read" is the archive's own parquet, which is strictly closer to the source than a prim walk
would be. `MEASURED`: 36 archive entries, 20 `.parquet`, plus `map.xodr`, `mesh_ground.ply`,
`volume.nurec`.

### 1.2 The five questions, answered

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | **Is `succ(lane)` / connectivity readable?** | ✅ **YES.** 12,030 successor edges over 11,877 lanes, **0 dangling ids**. Predecessors likewise (11,048 `PREVIOUS_LANE` rows). | `MEASURED`, both probes |
| 2 | **Junction + outgoing branches (S1 option set)?** | ✅ **YES.** **790 lanes with \|succ\| ≥ 2**, on **42/51 scenes**. Arity 2–4. Independently corroborated **twice more**: `map.xodr` `<junction>` elements > 0 on **51/51**, and `intersection_area.parquet` rows > 0 on **51/51**. | `MEASURED` |
| 3 | **Parallel lanes on an approach (S2 option set)?** | ✅ **YES**, with a correction — see §1.3. **6,981 / 11,877 lanes (58.8 %)** have a real left/right neighbour, on **46/51 scenes**. | `MEASURED` |
| 4 | **Roundabout entries/exits, and ORDINAL?** | ✅ **YES**, recovered as directed cycles in the lane graph. **5/51 scenes** carry an annular ring with ≥3 exits; the ego actually traverses one in **2**. Example: 8 ring lanes, radius **11.0 m**, radial CV **0.034**, **4 exits / 4 entries** — the exits are ordered around the ring, so the answer is an ordinal, exactly as S4 requires. | `MEASURED` |
| 5 | **Lane ↔ ego realised path ⇒ `target_branch` computable?** | ✅ **YES.** Ego-to-lane match rate **mean 0.9827**, ≥0.9 on **48/51** scenes, median distance to the matched centreline **0.274 m**. **30 of 43** mined decision points resolve to a branch the ego demonstrably entered. | `MEASURED` |

**Question 5 is the one that matters most**, because it is the non-circular target: a **map fact
intersected with a realised trajectory**, touching no model input. It is computable.

### 1.3 ⚠️ Two corrections to my own first-pass probe — both found and fixed in-session

Logged because the root-cause *class* is the reusable part.

**(a) `adj_lanes_left/right = {'-1'}` is a SENTINEL, not a neighbour.**
The first pass reported "every one of 385 lanes has exactly 1 left and 1 right neighbour" — which
would have been a *false positive* for S2. Probe B settled it: `association.parquet` carries
`RIGHT_LANE → objects: ['-1']` alongside `LEFT_LANE → objects: ['173938708']`. `-1` means *no such
neighbour*. Corrected count: **58.8 %** of lanes have a real parallel neighbour, not 100 %.
**Class: reading a library's object model without checking the source encoding.**

**(b) The decision-point miner's "junction within 15–60 m" was measured to the end of the ego's
CURRENT lane.** MADS lane segments are short — **median 20.65 m** (`MEASURED`) — so the window was
rarely satisfiable and the miner returned **13** decision points. The correct measure walks **forward
along the lane graph** through unique-successor chains, accumulating arclength, until a lane with
|succ| ≥ 2 is reached. Corrected: **43** decision points, **30** with a resolved target.
**Class: a threshold applied to the wrong geometric unit — 3.3× under-count.**

Neither error survived into a reported number; both are recorded because the second one would have
produced a *false negative* on the whole strategic programme ("AlpaSim has almost no decision points").

### 1.4 What else the archive turned out to carry — unasked-for, and material

`MEASURED`, probe B, `association.parquet` relation kinds summed over 51 scenes (37 distinct kinds,
143,078 rows, **0 unparseable**):

| Relation | rows | why it matters |
|---|---:|---|
| `BRANCH_SIBLING_LANE` / `MERGE_SIBLING_LANE` | 1,888 / 1,887 | the map **labels branch vs merge directly** — S1's option set need not be inferred from `|succ|≥2` alone |
| `ROAD_SEGMENT_SIBLING_LANE` | 11,877 | the proper **S2 option set** (all parallel lanes of a road segment), better than pairwise L/R |
| `OPPOSITE_ROAD_SEGMENT_SIBLING_LANE` | 4,366 | **oncoming** lanes — the T1 unprotected-conflict geometry |
| `INTERSECTION_AREA_TO_LANE` | 252 | explicit **junction membership** |
| `LIGHT_TO_LANE` / `WAIT_LINE_TO_LANE` / `SIGN_TO_LANE` | 810 / 637 / 1,229 | signal + stop-line + sign association per lane |
| `CROSSWALK_TO_LANE` | 867 | |

Plus, per scene: `traffic_light.parquet`, `obstacle.parquet` (agent tracks), `road_island.parquet`,
`buffer_zone.parquet`, and a full **`map.xodr` (OpenDRIVE)** — a *third* independent road-network
representation with native junction/connection semantics.

> **This materially widens the plan.** `STRATEGIC_TACTICAL_PROBLEM_SPEC.md` §3 records `SIGNAL` and
> `LIGHTSTATE` as *"unmintable from every real source we hold"*. That statement was about
> nuScenes/AV2/Waymo/ZOD and is not contradicted for those — but **AlpaSim scenes carry
> `traffic_light.parquet` and `LIGHT_TO_LANE` on 51/51 scenes**, so the T4/traffic-light
> option set is not as blocked as the spec assumed. `MEASURED`; not yet decoded to states.

### 1.5 Gate-1 per-scene coverage

Full per-scene records: **`gate1_connectivity.json`** (51 scenes, both probes). Roll-up:

| quantity | value |
|---|---|
| scenes held (4 scenesets, deduped) | **51** |
| probe A ok / probe B ok | **51 / 51** |
| A-vs-B exact agreement, `n_lanes` | **51 / 51** |
| A-vs-B exact agreement, `n_lanes_succ_ge2` | **51 / 51** |
| scenes with ≥1 branching lane | **42 / 51** |
| scenes with parallel lanes | **46 / 51** |
| scenes with an S1 decision point | **23 / 51** |
| scenes with a RESOLVED `target_branch` | **20 / 51** |
| scenes with an annular roundabout ring (≥3 exits) | **5 / 51** (ego traverses **2**) |
| median lanes / scene | **193** |
| median scene duration · ego path | **20.0 s · 207.6 m** |

---

## 2. GATE 2 — do reactive agents actually react?

### 2.1 What had to be unblocked first (all `MEASURED`, all this session)

`trafficsim` had **never once been enabled** in this program. Three blockers, none of them documented
before now:

1. **The CATK weights were never on the pod.** `data/trafficsim-models/catk_v120/latest.ckpt` was a
   **133-byte Git-LFS pointer** — because the clone recipe uses `GIT_LFS_SKIP_SMUDGE=1`
   (`BUILD_AND_USE.md` §2.1). Same for `config.yaml` and both token vocabularies.
   **Fixed:** `lfs_fetch.py` pulls them via the LFS **batch API over plain HTTPS** (no `git-lfs`
   binary, which this pod does not have), sha256-verified against the pointer OIDs.
   **`latest.ckpt` = 69,960,427 bytes, sha256 `7c5a89bc…` verified; loads to 811 tensors.**
   Public Apache-2.0 repo — **no NGC/HF gate involved.**
2. **The model paths in `server.yaml` are container paths** (`/mnt/trafficsim-models/…`) — the same
   trap as `/mnt/nre-data` (`ALPASIM_STATE.md` trap #2). Overridden on the CLI to host paths.
3. **PyG compiled extensions absent.** `torch_cluster` / `torch_scatter` / `torch_sparse` all missing,
   and PyG 2.8's native fallback needs `pyg-lib≥0.6.0`, which is **not on PyPI**. `torch-cluster`
   supplies `radius` / `radius_graph`, used at 3 call sites in `catk/smart/modules/`.
   **Fixed:** built **from source, CPU-only** (`FORCE_ONLY_CPU=1`) — this pod has g++ 13.3 but **no
   `nvcc`**, so a CUDA build is impossible. Consequence: CATK runs on **CPU** (§2.3).

### 2.2 The test design — and the control that makes it admissible

The harness (`gate2_reactivity.py`) drives the **`trafficsim` gRPC service directly** — no renderer,
no physics, no driver, no rendering GPU. That isolates precisely the question asked.

- **PROCEED** — ego follows its logged ground-truth trajectory.
- **YIELD** — ego stops dead at the handover pose and stays there.
  *Same scene, same seed, same session construction; the ego trajectory is the only difference.*
- **PROCEED2** — a third arm, identical in construction to PROCEED.
  ⭐ **This is the control that makes the result admissible.** It measures the model's own
  **stochastic floor**. A YIELD-vs-PROCEED divergence is evidence of *reaction* only if it exceeds
  the PROCEED-vs-PROCEED2 floor. Without it, a sampling model's own noise reads as "reaction" — the
  same class of error as attributing an AlpaSim closed-loop failure to the model before running the
  open-loop control (`RETRACTION_LOG.md:52`, class C6).

Divergence is measured on **non-ego agents only** (the ego is excluded — we set its trajectory, so it
differs trivially). Handover at t₀+2 s; queries at 1 s steps to the end of the 20 s scene; poses
compared at the queried timestamp.

### 2.3 ⚠️ Standing caveat on any Gate-2 number

CATK runs **on CPU**, because there is no `nvcc` on `tanitad-eval` and the PyG extensions therefore
had to be built CPU-only. This is a **throughput** limitation, not a fidelity one — the same weights
and the same code path — but it means the timing here says nothing about closed-loop cost, and a
production tactical suite will want a CUDA build of `torch-cluster` on a pod that has `nvcc`.

### 2.4 Result — ❌ **GATE 2 FAILS**

The gate asks one question with two clauses. They separate cleanly, and that separation *is* the result.

| clause | verdict |
|---|---|
| (a) do agents **deviate from their logged tracks**? | ✅ **YES, decisively.** `MEASURED`: returned positions differ from the agents' own logged tracks by **8.37–78.17 m mean**, and the fraction of returned poses lying within 0.1 m of the logged pose is **0.0000–0.0448** across the four scenes (i.e. ≥95.5 % of poses are *not* the logged pose, in every scene). **This is NOT replay-with-extra-steps.** CATK genuinely simulates. |
| (b) do they deviate **in response to the ego**? | ❌ **NOT DEMONSTRATED.** The ego-induced change is **not separable from the traffic model's own run-to-run sampling noise.** |

**The measurement** (`MEASURED`, `gate2_reactivity.json`, `gate2_conflict_*.json`): 5 repeats per arm;
`within` = mean pairwise distance among same-arm repeats, `between` = mean pairwise distance across
arms — **like-for-like, equal under the null**; paired episode-cluster bootstrap, **unit = agent**,
B=2000. Statistic is `between − within`; reaction ⇒ **positive and separated**.

| scene | T1 conflict? | stratum | agents | between m | within m | Δ = between−within | 95 % CI | separated |
|---|---|---|---:|---:|---:|---:|---|---|
| `00169207` | no | all | 6 | 5.79 | 6.12 | **−0.331** | [−0.960, +0.150] | no |
| `00169207` | no | ≤50 m | 4 | 9.54 | 10.69 | **−1.154** | [−1.552, −0.010] | yes — **negative** |
| `6dcd2117` | yes | all | 23 | 24.81 | 23.17 | **+1.644** | [+0.012, +3.731] | yes (marginal) |
| `6dcd2117` | yes | ≤50 m | 9 | 10.75 | 10.77 | **−0.023** | [−0.454, +0.572] | no |
| ⭐ `59cb0598` | yes | all | **59** | 4.45 | 4.47 | **−0.026** | **[−0.207, +0.139]** | no |
| ⭐ `59cb0598` | yes | ≤50 m | **36** | 4.25 | 4.23 | **+0.018** | **[−0.149, +0.213]** | no |
| `780ece49` | yes | all | 6 | 15.53 | 10.46 | **+5.068** | [+0.002, +14.721] | yes (marginal) |
| `780ece49` | yes | ≤50 m | 4 | 0.249 | 0.240 | **+0.009** | [0.000, +0.021] | no |

**Why this reads as a null and not as a weak positive:**

1. ⭐ **The near-ego stratum is null in 4 / 4 scenes.** If agents responded to the ego, the effect must
   be *concentrated* in the agents near it. It is absent there in every scene — and in the one scene
   where it separates, it separates **negative**.
2. ⭐ **The best-powered test is a clean null with the tightest interval.** `59cb0598` carries **59
   dynamic agents** and bounds the effect at **[−0.21, +0.14] m** (all) and **[−0.15, +0.21] m**
   (near-ego). That excludes any ego-induced effect larger than ~0.2 m — against a **4.5 m noise floor**.
3. **The two nominal positives are far-field only, and marginal.** CI lower bounds **+0.012** and
   **+0.002** — i.e. sitting on the α boundary — across **8 tests**. That is what multiplicity produces
   on its own, and both vanish in the near-ego stratum of the same scene.

**Three controls that make this null admissible rather than an artefact:**

- ⭐ **The intervention definitely reached the model.** `MEASURED`: the PROCEED and YIELD ego inputs
  differ after handover by **mean 19.94 m / max 60.99 m** (`59cb0598`) and **mean 42.06 m / max
  149.14 m** (`6dcd2117`). The ego drives away or stands still for the rest of the scene; CATK receives
  that difference in its ego-future conditioning (`populate_ego_future_from_trajectory`, logged
  per request). **A car standing dead in the road for 18 s moved nearby agents by less than 0.2 m.**
- **CATK really ran.** `alpasim_trafficsim.catk.model_adapter:create_model_input` logs real inference
  per request (~590–600 proxy triplets, history window t∈[0,16]). The metadata string
  `"simple-traffic-service"` is a **default `service_version` parameter of the CATK servicer**
  (`grpc/servicer.py:89`), not a different service. `MEASURED`, two probes.
- **The output is physically plausible**, so the null is not "the model emits garbage": simulated agent
  speeds are **mean 1.33 · p50 0.51 · p90 3.33 · max 12.03 m/s**, with **0.0000** of steps above
  40 m/s (`gate2_plausibility.json`).

> ### ⚠️ The one thing that could overturn this, named honestly
> This drives the `trafficsim` service **directly over its own gRPC contract**, with the session built
> after `runtime/services/traffic_service.py` — **faithful, but not byte-identical to a full runtime
> integration** (handover seeding and per-step history could differ). The decisive follow-up is **one
> full closed-loop rollout with `trafficsim=catk`**, which now costs only renderer time because every
> other blocker in §2.1 is removed. Until that is run, the correct statement is **"FAILS as measured
> through the service's documented contract"**, not "CAT-K cannot react".

### 2.5 A second `MEASURED` fact with a real cost consequence

**The traffic model is strongly nondeterministic run-to-run even with a fixed `random_seed`:** mean
pairwise spread between *identical* repeats is **4.2 m (59cb0598) to 23.2 m (6dcd2117)** over a 17.5 s
rollout. Whatever the reactivity verdict, this prices `Y_outcome`: a tactical outcome metric must
average **many rollouts per scene**, and a single-rollout `Y_outcome` — which is what
`STRATEGIC_TACTICAL_PROBLEM_SPEC.md` §3 T1 implicitly assumes — would be **dominated by traffic-model
noise rather than by the policy's decision.** That is a design consequence for T1–T4 independent of §2.4.

### 2.6 Consequence — and the stop

`Y_outcome` for **T1, T2, T3, T4** is *"roll the policy with `trafficsim` ON and score the simulated
consequence"*, and its entire admissibility argument is *"it cannot be circular with any input by
construction"*. That argument requires the simulated consequence to **be a function of the policy's
choice**. On this evidence it is not measurably one.

⇒ **Per the brief, this gate is not worked around.** T1's `Y_outcome` is not built, and the T1 slice is
delivered as `Y_expert` + coverage/power only (`S1_T1_SLICE.md` §3). **This is a PI-level decision
point**, and §3 of that document shows the corpus blocks T1 independently anyway.

---

## 3. What Gate 1 passing does and does not license

**Licensed** (`MEASURED`): S1's option set, S2's option set, S4's ordinal exit set, and the
non-circular target `target_branch = map ∩ realised path` are all **constructible today** on scenes we
already hold. Four of the nine problems stop being blocked on "no map exists".

**NOT licensed — the corpus bar is not met.** Resampling unit = **AlpaSim scene**:

| | available | bar | shortfall |
|---|---:|---:|---:|
| S1 clusters (scenes with a resolved target) | **20** | 40 (single-arm) | **2.0×** |
| S1 clusters | **20** | 200 (two-arm) | **10.0×** |
| S4 clusters (ego traverses a roundabout) | **2** | 40 | **20×** |

**The fix is scene acquisition, not method.** Yield `MEASURED` at **0.39 resolved-target clusters per
scene** (20 / 51). The NuRec `public_2604` pool holds **1,606 scenes** (`INHERITED`,
`ALPASIM_STATE.md` §4.1), and the balanced-suite builder that selects from it is already committed.

- ≥40 clusters (single-arm) ⇒ **~103 scenes** ⇒ **ESTIMATED ~155 GB** at 1.5 GB/scene
- ≥200 clusters (two-arm) ⇒ **~513 scenes** ⇒ **ESTIMATED ~770 GB**

⚠️ The second figure is **larger than the pod's ~466 GB MooseFS quota** (`INHERITED`, memory
`physicalai-workspace-mfs-quota`), so a 200-cluster S1 suite needs streaming/eviction or a second
volume. **That is a PI-level cost decision and is flagged, not assumed.**

---

## 4. Deliverable manifest

| Artifact | Where it lives |
|---|---|
| **This document** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-4brain-gates/GATE_RESULTS.md` (working tree, **NOT staged**) |
| `gate1_connectivity_probe.py` | same dir — the two-probe sweep (runs on `tanitad-eval`) |
| `gate1_connectivity.json` | same dir — 51 scenes, both probes, full per-scene records |
| `s1_decision_points.json` | same dir — 43 mined decision points, 30 with a resolved target |
| `s1_augment_option_geometry.py` | same dir — adds per-option centreline geometry (needed to make the blind attack strong) |
| `gate2_reactivity.py` | same dir — v1 harness (session construction, arm definitions, stochastic-floor control) |
| **`gate2_reactivity_v2.py`** | same dir — **the harness the verdict rests on**: R repeats/arm, like-for-like pairwise statistic, dynamic + near-ego strata, replay control |
| `gate2_reactivity.json` · `gate2_conflict_clipgt-{6dcd2117,59cb0598,780ece49}.json` | same dir — the four scenes |
| `gate2_plausibility.json` · `gate2_conflict_scenes.log` | same dir — physical-plausibility control + run log |
| `lfs_fetch.py` | same dir — CATK weight fetch via the LFS batch API, sha256-verified |
| `blind_conditioning_baseline.py` | same dir — the §0.1 circularity firewall + self-test |
| `s1_slice.py`, `S1_RESULTS.json`, `S1_T1_SLICE.md` | same dir — §S1 |
| `t1_conflict_miner.py`, `t1_conflict_points.json`, `t1_sensitivity_sweep.txt` | same dir — §T1 |
| On the pod (regenerable) | `/workspace/{gate1_v2,s1_decision_points_aug,gate2_reactivity_v3,t1_conflict_points,t1_raw_crossings}.json`, `/workspace/{trafficsim,g2_build,g2_conflict,t1_miner}.log`, **CATK weights at `alpasim/data/trafficsim-models/`** (fetched this session; sha256-verified) |

**Nothing staged. Nothing committed. Nothing pushed.** pod1/pod2/pod3 untouched.
