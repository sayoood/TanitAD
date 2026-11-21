<title>NC ingest round 1 — nuScenes is CC BY-NC-SA (copyleft), and its 16 MB map expansion is the strategic-brain unblock</title>

# NC source ingest — compliance verdict, value ranking, and what R actually contains

**Date:** 2026-07-26 (local, Europe/Berlin) · **Agent:** Data Engineering / NC ingest
**Task:** "ingest NC source and nuScenes" + "which valuable data can be added to the research dataset?"

**Status:** 🟢 T1 compliance **RESOLVED + FIXED + TESTED** · 🟢 T3 value ranking **DELIVERED**
· 🔴 T2 nuScenes bytes **NOT ACQUIRED** (human terms gate — adapter built, tested, ready)
· 🟡 T4 R **unchanged**; card updated locally, **push not attempted** (needs Sayed)

**Evidence classes used:** `MEASURED` (I ran it, artifact path given) · `PUBLISHED` (cited URL)
· `PUBLISHED[relayed]` (a sub-researcher read it at the cited URL; I did not re-read)
· `INHERITED` · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. Headline — six findings, in the order they change the plan

1. **🔴 The open compliance item is CONFIRMED and FIXED. nuScenes is CC BY-NC-**SA**-4.0 — copyleft.**
   `schema.py` said `CC-BY-NC-4.0, share_alike=False`. That was **wrong**. It now reads
   `CC-BY-NC-SA-4.0, share_alike=True`, which automatically routes nuScenes into the segregated
   copyleft shard. Two independent PUBLISHED probes; details in §1.
2. **🟢 The finding that actually changes the program: nuScenes ships a REAL ROUTABLE LANE GRAPH,
   and it costs 16 MB.** The H2 survey ranked nuScenes #1 for *label mechanics* and never mentioned
   the map expansion. `nuScenes-map-expansion-v1.2.zip` (**17,136,555 B**) contains `lane`,
   **`lane_connector`**, a `connectivity` dict, `get_outgoing_lane_ids()` / `get_incoming_lane_ids()`,
   `arcline_path_3` + `discretize_centerlines()`, `road_segment.is_intersection`, and a `stop_line`
   typed **STOP_SIGN / YIELD / TRAFFIC_LIGHT / PED_CROSSING**. **This is exactly the ground truth the
   strategic-brain proof lacks** — "which branch at a junction" becomes enumerable. See §3.1.
3. **🔴 T2 could not ingest a byte, and I did not route around it.** nuScenes requires a **human** to
   create an account and accept the Terms of Use. Account creation is outside an agent's boundary and
   terms acceptance is not something an agent brief can authorise. I did **not** use one of the 12
   third-party HF mirrors. What I built instead is the complete, tested ingest path — so acquisition
   is one human download plus one command. See §4.
4. **🟢 The stranded D-016 R1 geometry fix is now FOLDED IN and green.** `pinhole_rectify` sat
   unmerged for **9 days**. It is now in `stack/tanitad/data/calib.py` with its 9 tests ported plus 2
   new nuScenes cases. **MEASURED:** nuScenes CAM_FRONT `fx=1266.4` on 1600×900 → the legacy square
   crop lands **f_eff 360.2** (height-clamped); rectify lands **266.0 exactly**, `observed_frac 0.738`.
5. **🟡 The single highest-value action on this page is not an external dataset — it is a probe of our
   OWN corpus.** The NVIDIA PhysicalAI-AV card names **roundabouts** explicitly in its coverage, over
   **306,152 clips / 25 countries / 7 cameras**, under a licence that **permits commercial AV use** —
   and our ingest reads **2 of 36 features**. Before buying an NC licence problem, read the other 34.
6. **🔴 R contains exactly what it contained yesterday: 90 comma2k19 episodes, byte-identical to C.**
   NC value-add this round = **0 records, 0 GB**. Both tiers re-verified `SAFE-TO-PUSH` (§5).

---

## 1. T1 — Compliance verdict: **ShareAlike is CORRECT. nuScenes is copyleft.**

### 1.1 The verdict

> **nuScenes is released under Creative Commons Attribution-NonCommercial-**ShareAlike** 4.0
> (CC BY-NC-SA 4.0).** `share_alike` **must be `True`**, and is now.

**Probe 1 — the dataset authors' own paper.** `PUBLISHED`, [ar5iv 1903.11027](https://ar5iv.labs.arxiv.org/html/1903.11027)
(the CVPR 2020 nuScenes paper): *"The nuScenes data is published under CC BY-NC-SA 4.0 license, which
means that anyone can use this dataset for non-commercial research purposes."*

**Probe 2 — independent corpus.** `PUBLISHED`, web search across secondary sources consistently states
CC BY-NC-SA 4.0 *"with modifications outlined in https://www.nuscenes.org/terms-of-use"*.

**Probe 3 (negative, and it matters) — the authoritative terms page could NOT be read.**
`https://www.nuscenes.org/terms-of-use`, `/nuscenes` and `/download` each returned **EMPTY** to
WebFetch (JS-driven site) — **4 attempts across 3 URLs**, reproducing the H2 surveyor's failure.
So the *modifications on top of the CC grant remain unread*.

**Probe 4 — the distinction that prevents a future error.** The **devkit code** is
**Apache-2.0** (`PUBLISHED`, [LICENSE.txt](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/LICENSE.txt),
"Copyright 2021 Motional"). The **data** is not. A future agent reading the repo licence and
concluding "nuScenes is Apache-2.0" would be catastrophically wrong; this is recorded in the module
docstring.

**Also flagged:** the AWS Open Data registry entry for the `motional-nuscenes` bucket carries a
licence field reading **"Commercial"**. `PUBLISHED[relayed]`. **That tag is wrong** and contradicts
nuscenes.org. Do not quote it.

**Conclusion.** Evidence *and* the conservative branch point the same way, so the verdict is not
sensitive to the unread page: **treat as CC BY-NC-SA 4.0, and treat the unread modifications as
potentially MORE restrictive, never less.** The registry entry is the floor of the restriction.

Root-cause class: **licence-assumed-from-a-short-name** — identical to the 2026-07-13 ZOD
`research/NC → CC-BY-SA-4.0` correction. Recommend an append to `RETRACTION_LOG.md` (§7).

### 1.2 The fix — `MEASURED`

`stack/tanitad/lake/schema.py`:

```python
"nuscenes": SourceLicense("nc-research", "CC-BY-NC-SA-4.0", share_alike=True,
                          is_synthetic=False),
```

Because every downstream consumer derives from this **constant**, the one-line change propagates
with **no other code edit** — verified by running each consumer:

| consumer | before | after | `MEASURED` |
|---|---|---|---|
| `filtering.tier_of` | `nc` | `nc` | unchanged (correct) |
| `shards.shard_prefix` | `shards/nc-research/nuscenes/train` | **`shards/nc-research/sharealike/nuscenes/train`** | segregated ✅ |
| push safety-check `SA` set | `{argoverse2, kitti360, zod}` | **`{argoverse2, kitti360, nuscenes, zod}`** | ✅ |
| push safety-check `NC` set | contains `nuscenes` | contains `nuscenes` | unchanged |
| `commercial_ok` | `False` | `False` | unchanged |

The two-leg push safety check computes its NC/SA/gated/refuse source sets **from the registry at run
time**, so `push_tanitdataset.py` got strictly stronger with zero edits to it.

### 1.3 The guard routes NC away from C — now with regression tests

**7 new tests** in `stack/tests/test_lake.py`, all `MEASURED` green:

| test | what it pins |
|---|---|
| `test_nuscenes_registered_as_copyleft_share_alike` | the correction itself — `share_alike is True`, `"SA" in license_name` |
| `test_nc_research_record_refused_from_commercial_tier_C` | a **real** assembled nuScenes record, flattened by `record_to_catalog_row`, refused by the **real** C guard — and by **each of the three gates independently** (defence in depth: weakening any one must not open the tier) |
| `test_every_nc_research_source_refused_from_commercial_tier_C` | all **8** registered NC sources, not just nuScenes |
| `test_nc_share_alike_routes_to_segregated_copyleft_shard` | copyleft NC lands under `sharealike/`; non-SA NC does not |
| `test_nc_research_tier_is_nc_not_ship` | every NC source derives tier `nc` |
| `test_commercial_view_predicate_excludes_nc_research` | **layer 1** — the C view's pyarrow predicate filters NC out, so the guard is a backstop and not the only defence |
| `test_ingested_nuscenes_is_refused_from_commercial_tier_C` (in `test_nuscenes.py`) | **end-to-end**: real ingestor → real shards → real catalog → real guard raises; and the C view resolves to **0 episodes** in an NC-only lake |

**Nothing was weakened.** No guard was relaxed, no scope widened, no `allowed_classes` extended.

---

## 2. T3 — Which data is valuable, ranked against OUR open problems

Gaps, as briefed: **G1** map/lane graph/route topology (blocks the strategic-brain proof) ·
**G2** roundabouts (19 in-corpus vs a ≥40 bar) · **G3** unprotected lefts / yield (tactical brain) ·
**G4** surround cameras + extrinsics + 3D tracks (camera attention) · **G5** traffic lights (no
feature in-corpus) · **G6** domain diversity.

### 2.1 The gap matrix

✅ fills · ⚠️ partial · ❌ does not

| Source | G1 map/lane graph | G2 roundabouts | G3 unprot. left/yield | G4 surround+3D | G5 traffic lights | G6 diversity | Licence |
|---|---|---|---|---|---|---|---|
| **PhysicalAI-AV** *(ours)* | ❌ none | ⚠️ **card NAMES roundabouts — untested** | ? unread | ✅ 7 cam + intr + extr + `obstacle.offline` | ❌ | ✅✅ 25 countries | NVIDIA AV — **commercial OK**, `gated-confidential` for publication |
| **nuScenes** | ✅✅ **routable graph, 16 MB** | ❌ no published count — **measurable from 0.46 GB** | ⚠️ `YIELD` stop-line type; no manoeuvre tags | ✅✅ 6 cam, `calibrated_sensor` I+E, tracks + visibility | ⚠️ **geometry only, NO state** | ⚠️ rain 19.4 %, night 11.6 %; 2 cities | CC BY-NC-SA 4.0 |
| **OpenLane-V2** subset_B | ✅ 3D centerlines + `topology_lclc` | ❌ | ❌ | inherits nuScenes | ✅✅ **red/green/yellow BOUND TO A LANE** (`topology_lcte`) | inherits | CC BY-NC-SA 4.0 |
| **nuPlan** | ✅✅ **+ explicit per-scenario route** | ❌ **zero roundabout tags** | ✅✅ **explicit `starting_unprotected_cross_turn` tags** | ⚠️ yes, but 7.35 TB / 45 GB min unit | ⚠️ per-lane per-step, **no yellow, ~68.7 % inferred** | ❌ **rain+night excluded; 89.8 % Vegas** | CC BY-NC-SA 4.0 |
| **Argoverse 2** | ✅ per-scenario pred/succ | ❌ | ⚠️ derived 21,431-case conflict set | ✅ 9 cam, I+E, 10 Hz cuboids | ❌ **none** | ⚠️ 6 US cities | CC BY-NC-SA 4.0 |
| **BDD100K** | ⚠️ 2D `poly2d` only | ❌ | ❌ | ❌ mono, no calib, no 3D | ✅✅ **265,906 lights + colour** | ✅ night 39,986 img | **UNVERIFIED** |
| **ONCE** | ❌ **map withheld by regulation** | ❌ | ❌ | ⚠️ 7 cam + calib but **NO TRACK IDs** | ❌ | ✅ night 20.2 %, rain 6.1 % | CC BY-NC-SA 4.0 |
| **KITTI-360** | ❌ | ❌ | ❌ | ✅ best quality, **73.7 km total** | ⚠️ class exists, no state | ❌ one city, one day | CC BY-NC-SA 3.0 |
| **Rank2Tell** | ❌ | ❌ | ⚠️ **all four-way intersections**, 116 clips ≈ 39 min | ⚠️ 134° front, not surround | ⚠️ 3,048 lights, no state field | ❌ | UNVERIFIED |
| **DRAMA** | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ Tokyo | UNVERIFIED |
| **A2D2** | ❌ | ⚠️ real EU roundabouts | ❌ | ⚠️ 3D boxes **front-FOV only**, 12,497 frames | ❌ | ⚠️ 3 DE cities | **CC BY-ND** — no derivatives |
| **Waymo / Waymax** | ✅ | ❌ | ✅ | ✅ | ✅✅ 9 states | ⚠️ | 🔴 **`refuse`** — terms follow the weights |

### 2.2 The ranking

#### 🥇 **#0 (not an acquisition) — probe PhysicalAI-AV's remaining 34 features. Cost: ~free.**
The NVIDIA card states **306,152 clips × 20 s / 1,700 h / 25 countries / 2,500+ cities / 7 cameras on
every clip**, with per-clip feature presence including camera intrinsics, sensor extrinsics, ego
motion and `obstacle.offline` — and its coverage description **explicitly names roundabouts**
(`PUBLISHED[relayed]`, [dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)).
We ingest **2 of 36 features** and hold 2,376 of 306,152 clips.
**Why it outranks everything below:** if a queryable roundabout tag exists, **G2 and G6 both collapse
in-house, under a licence that permits commercial AV use**, with the same rig family and no new NC
contamination. Every external option costs a licence problem this one does not.
⚠️ `UNVERIFIED` whether the tag is queryable — that is precisely the one probe to run.
⚠️ Parity is untouched: this is a *separate* roundabout eval/finetune pull, not a re-selection of the
canonical 2,376.

#### 🥈 **#1 external — nuScenes.** Entry cost **0.48 GB**. Fills **G1 decisively** and **G4 turn-key**.
Two things it uniquely gives us:

**(a) The lane graph — the strategic-brain unblock.** `PUBLISHED`, read at primary source in
[`map_api.py`](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/map_expansion/map_api.py)
and the [map-expansion tutorial](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/tutorials/map_expansion_tutorial.ipynb):
11 non-geometric layers including `lane`, **`lane_connector`**, `road_segment` (with an
**`is_intersection`** flag), `stop_line` (typed **STOP_SIGN / YIELD / TRAFFIC_LIGHT / PED_CROSSING**),
`road_divider`, `lane_divider`, `ped_crossing`, `walkway`, `carpark_area`, `drivable_area`,
`traffic_light`. Connectivity is **genuine**: a `connectivity` dict, `get_outgoing_lane_ids()`,
`get_incoming_lane_ids()`, `get_closest_lane(x, y, radius)`, and `arcline_path_3` with
`discretize_centerlines()`. Four maps with published canvas extents (boston-seaport 2979.5×2118.1 m,
singapore-onenorth, -hollandvillage, -queenstown). The paper adds **"baseline routes representing the
idealized path an AV should take"**.
→ *"Which branch at this junction"* becomes an **enumeration over `get_outgoing_lane_ids()`**, which is
the ground truth we have been missing. **`nuScenes-map-expansion-v1.2.zip` = 17,136,555 B.** `MEASURED`.

**(b) The cross-camera visibility mechanism.** `calibrated_sensor` publishes intrinsics **and**
extrinsics for all 6 cameras; `ego_pose` per keyframe; 3D boxes with **instance tokens** (a track, not
a detection) and a **visibility attribute** binned over all six images (0–40/41–60/61–80/81–100 %).
So *"agent A is in CAM_BACK_LEFT's frustum and NOT in CAM_FRONT's"* is a **projection**. I have
**implemented and tested exactly this** (§4.2).

Scale `PUBLISHED[relayed]` ([nuscenes.org](https://www.nuscenes.org/nuscenes)): 1,000 scenes × 20 s
≈ 15 h; **40k keyframes @ 2 Hz**; 1.4 M boxes; 23 classes; 1,166,187 cuboids (car 493,322 / adult ped
208,240); 6 cam 1600×900 @ 12 Hz; splits 700/150/150; **rain 19.4 %, night 11.6 %** (`PUBLISHED`, ar5iv).
Bonus: `can_bus.zip` (745 MiB) ships **per-scene route/navigation paths**, missing for only **3 %** of
scenes — a second, independent G1 asset.

**What it does NOT give us — stated bluntly.** **No traffic-light state, ever.** `traffic_light` is a
*static line-geometry map layer*; a Motional maintainer states verbatim that they *"generally don't
support traffic light states, as our map is entirely static"* (`PUBLISHED[relayed]`, devkit issue
#529). It is also **not** among the 23 cuboid classes. **G5 is not filled by nuScenes.** And **no
roundabout count is published anywhere** — but it is measurable from the 0.46 GB metadata, and my
adapter already computes it (§4.2).

**Integration cost:** the adapter is **written and tested** (§4). Remaining: one human download, then
`--analyze` (minutes) and, if justified, `--ingest`.

#### 🥉 **#2 — nuPlan maps + mini.** <10 GB. The only source with an **explicit route** and **explicit unprotected-turn tags**.
`SemanticMapLayer` exposes 12 queryable vector layers including `LANE_CONNECTOR`, `ROADBLOCK_CONNECTOR`,
`INTERSECTION`, `STOP_LINE`, `BASELINE_PATHS`; the graph API has `incoming_edges`/`outgoing_edges` and
`LaneConnector.turn_type`; and **`get_route_roadblock_ids()` + `get_mission_goal()` give a per-scenario
route** — the closest thing to a strategic-brain label in any corpus. **G3 is filled by tag**:
`starting_unprotected_cross_turn`, `starting_unprotected_noncross_turn` and protected counterparts,
across **73 scenario types** over **1,282 h**. `PUBLISHED[relayed]`, [arXiv 2403.04133](https://arxiv.org/html/2403.04133v1).
Cheap path: `nuplan-maps-v1.1.zip` (0.97 GB) + `nuplan-v1.1_mini.zip` (8.55 GB).
**Blunt limits:** **zero roundabout tags** (grep for `roundabout|circle|rotary` returns nothing);
traffic-light status is **inferred offline from agent motion at ~68.7 % accuracy** with **YELLOW in the
enum but absent from the data** — usable for conditioning, never for a headline metric; **heavy rain
and night were deliberately excluded**; **89.8 % of train logs are Las Vegas**; camera blobs are
**7.35 TB** with a 45–62 GB minimum unit. ⚠️ **`nuplan` is not in `SOURCE_REGISTRY`** — see §7.
⚠️ Hour count is quoted inconsistently upstream (1500 / 1200 / 1282); quote **1,282** and cite the 2024 paper.

#### **#3 — OpenLane-V2 subset_B (27.7 GB).** The only way to get **G5 with lane binding**.
Re-annotates **nuScenes** (subset_B) and Argoverse 2 (subset_A) with 3D lane centerlines and
**traffic elements carrying `red / green / yellow`** plus `go_straight / turn_left / no_left_turn / u_turn / …`,
and — the key part — **`topology_lcte`, the lane↔traffic-element relation ("this element controls this
lane")**. 2.1 M instance annotations + 1.9 M topology relationships over 2,000 segments @2 Hz.
`PUBLISHED[relayed]`, [ar5iv 2304.10440](https://ar5iv.labs.arxiv.org/html/2304.10440) and the
[data README](https://raw.githubusercontent.com/OpenDriveLab/OpenLane-V2/master/data/README.md).
CC BY-NC-SA 4.0, and it requires accepting the nuScenes terms too. **Only worth it after nuScenes
lands**, since it is an annotation layer on imagery we would already hold.

#### **#4 — BDD100K.** Take it *only* for G5 breadth and night. It contributes nothing to G1–G4.
**265,906 traffic-light instances** with a per-box `trafficLightColor ∈ {red, green, yellow, none}`
(`PUBLISHED[relayed]`, [arXiv 1805.04687](https://arxiv.org/pdf/1805.04687) Fig. 3a +
[format.md](https://raw.githubusercontent.com/ucbdrive/bdd100k/master/doc/format.md)); **night 39,986
images**, rainy 7,125, snowy 7,888, foggy 181. **Single forward mono camera, no calibration, no 3D
boxes** → G4 impossible.
⚠️ **Trap recorded:** `format.md` documents `intrinsics` / `extrinsics` / `box3d` — that is the generic
**Scalabel** schema, **not BDD100K content**. Do not let anyone plan a G4 use on it.
⚠️ **Licence UNVERIFIED** — the registration wall was unreachable on two hostnames. The **code** is
BSD-3-Clause; **the data licence is not established**. Do not write "BDD100K is BSD-3, commercial is fine".

#### **#5 — Argoverse 2.** Strictly dominated by nuScenes for our gaps. Its *derivative* is the interesting part.
9 cameras with I+E, 30-class 10 Hz cuboids, per-scenario local map with `predecessors`/`successors`.
But **no traffic lights at all** (its `TRAFFIC_LIGHT_TRAILER` class is a *portable construction unit* —
a name trap), no roundabouts, 6 US cities, ~1 TB for the sensor split. ⚠️ Centerlines are **inferred
from the boundary pair**, not stored.
**The genuinely useful asset is derived, not AV2 itself:** an openly released conflict-resolution set of
**21,431 scenarios (5,337 AV-involved + 16,094 AV-free)** in which **unprotected left turns are the most
common type** — directly G3. CC BY-NC-SA 4.0. `PUBLISHED[relayed]`, [arXiv 2308.13839](https://arxiv.org/html/2308.13839v2).

#### **#6 — KITTI-360.** Rig/calibration reference only. **73.7 km, one city, one day** cannot supply ≥40 roundabout clusters, and G6 is actively *negative*.
#### **#7 — Rank2Tell.** **116 clips ≈ 39 minutes**, but *all four-way intersections* and 3,048 traffic-light objects with importance ranks — a good **label-design reference** for G3, not a training source. 134° front, not surround. Licence UNVERIFIED.
#### **#8 — ONCE. Adds nothing we need.** Two independent kills: **no track IDs** (per-frame detection only → G4 fails) and **the map and GPS are deliberately withheld for regulatory reasons** (→ G1 can never be filled). Its one strength, night 20.24 % / rain 6.11 %, is available more cheaply elsewhere.
#### **#9 — DRAMA. Disqualified.** **2-second clips.** For a world model that is not a corpus.
#### **#10 — A2D2. Barred in practice.** **CC BY-ND — No Derivatives.** A re-shard is a derivative. Its 3D boxes also cover only 12,497 **front-FOV-only** frames.
#### **🔴 Waymo Open / WOD-E2E / Waymax — `refuse`, unchanged and reconfirmed.** The terms bar use "in operation of a vehicle" and in production systems, and **the restriction follows the trained weights** — stricter than CC BY-NC-SA and the harshest term surveyed. `assemble_lake_record` raises. Not proposed.

### 2.3 The European-roundabout answer, blunt

**A roundabout-rich European ego-camera corpus with labels does not exist.** The roundabout-rich sets
(openDD: 7 roundabouts / >62 h / 84,774 trajectories; rounD: 3 roundabouts; INTERACTION; RoundaboutHD)
are **all drone or fixed-infrastructure** — no ego camera, so no front-camera input and no camera to
activate. This is structural, not accidental: roundabouts get studied for interaction modelling, which
wants occlusion-free multi-agent state, which is what a bird's-eye view buys.
The closest ego-camera hit is the **UTBM RoboCar / EU Long-term** dataset (Montbéliard, FR): a
dedicated route of **~4.2 km containing 10 roundabouts**, 4 cameras with I+E and GNSS-RTK, CC BY-NC-SA 4.0
— but **zero agent labels, no map, and ~20 traversals**, under half the ≥40 bar. `PUBLISHED[relayed]`,
[project page](https://epan-utbm.github.io/utbm_robocar_dataset/).
**→ G2 has no good external answer. That is why #0 (probing our own corpus) is ranked first.**

---

## 3. What still blocks the strategic-brain data need

**Filled by this round (conditionally):** nuScenes' map expansion is a genuine routable lane graph, so
the strategic-brain proof *can* be run — **on nuScenes, at research tier, for a 16 MB download**.

**Still blocking, and this is the honest part:**

1. **Our own corpus still has no map.** PhysicalAI-AV ships no lane graph. nuScenes gives a *proving
   ground*, not a transferable one — a lane graph for Boston/Singapore does not annotate our clips.
2. **The proof would be NC + ShareAlike, so it cannot ship.** Under the derivative rule, a strategic
   brain trained or validated on nuScenes is itself `nc`/SA. It can support a research claim; it
   cannot be a product capability. **This is a licensing decision about our weights, and it belongs to
   Sayed, before ingest rather than after.**
3. **Geography.** Boston + Singapore is 2 cities; nuPlan adds Vegas-dominated US. Neither resembles the
   EU roundabout geometry the program cares about.
4. **The commercially-clean lane-graph path is still unidentified.** Three candidate leads, none verified:
   (a) PhysicalAI's **34 unread features**; (b) L2D's rendered BEV `map` channel (Apache-2.0, already
   in the registry); (c) **OpenStreetMap map-matching onto our ego GPS traces** — ODbL, commercially
   usable but share-alike for derived databases, so it needs its own licence read before anyone builds
   on it. `HYPOTHESIS` — I have not evaluated any of the three.

---

## 4. T2 — nuScenes ingest: what was built, and why no bytes moved

### 4.1 🔴 Why the corpus was not acquired

nuScenes requires, as a **human**: a free account at nuscenes.org, and **explicit acceptance of the
Terms of Use**. Account creation is outside an agent's boundary, and terms acceptance requires the
user's own consent — **an agent brief is not that consent**. So I stopped rather than proceeding.

**I also did not route around it.** `MEASURED`: I queried the HF API for an official presence —
authors `motional`, `nutonomy`, `nuscenes` all return **NONE**; a search for "nuscenes" returns **12
third-party mirrors** (`mitanshu17/Nuscenes`, `christine99x/nuscenes_train`, …), none authoritative
and mostly derivative subsets. Using one would substitute an unverifiable copy for a terms gate.

⚠️ **One nuance I want on the record, because it cuts both ways.** Motional *also* publishes through an
**AWS Open Data** bucket (`motional-nuscenes`) whose object listing is publicly readable — that is an
**official channel, not a third-party mirror**, and it is where the MEASURED sizes below come from.
**Reachability is still not permission**: nuscenes.org gates the download behind terms acceptance, and
that page could not be read while being documented to carry *modifications* to the CC grant.
Conservative branch: **a human accepts the terms, then downloads.** Recorded in the module docstring so
a future agent neither treats the bucket as forbidden nor treats it as consent.

**MEASURED acquisition sizes** (public bucket listing) — note the survey's "~60 GB" was an overestimate:

| file | bytes | ≈ | why |
|---|---|---|---|
| `v1.0-trainval_meta.tgz` | 461,678,030 | **0.46 GB** | answers the whole value question |
| `nuScenes-map-expansion-v1.2.zip` | **17,136,555** | **16 MiB** | **the routable lane graph** |
| `can_bus.zip` | 780,974,697 | 745 MiB | per-scene route paths |
| trainval keyframes (10 files) | 44,902,690,772 | **44.9 GB** | only if the above justify it |
| trainval full blobs | — | ≈314 GB | not proposed |

### 4.2 🟢 What was built instead — the complete, tested ingest path

**`stack/tanitad/data/nuscenes.py`** — metadata-first adapter, no new dependency (the released metadata
is plain JSON with a documented schema; the devkit is Apache-2.0 but heavyweight). Everything except
`build_episode` runs on metadata **alone**:

| capability | function | notes |
|---|---|---|
| table load + token index | `load_tables`, `NuScenesIndex` | scene→sample chains follow the `next` pointer, **not** a timestamp sort |
| **episodes** | `discover_scenes` | one 20 s scene = one episode |
| **ego pose** | `ego_track` → `[T,4] (x,y,yaw,v)` | speed from real **non-uniform** timestamps, not an assumed 2 Hz |
| actions | `actions_from_track` → `[T,2]` | POSE-DERIVED (no CAN); bicycle-model steer proxy with a standstill guard |
| **3D agent tracks** | `agent_tracks` | carries `instance_token` — a **track**, not a detection |
| **per-camera calibration** | `camera_intrinsics_of` | **per-sample** `calibrated_sensor`; the nominal constant is never asserted on a record |
| **cross-camera visibility** | `camera_visibility` | **the centrepiece** — see below |
| scenario stats | `scene_scenario_stats`, `corpus_scenario_report` | roundabout / turn / yielding-left / traffic-light-mention counts |
| geometry canon | `canonicalize_frames` | D-016 R1 `pinhole_rectify` |
| I3 split unit | `split_unit_of` | **the LOG, not the scene** — scenes from one log are consecutive slices of the same drive |

**`camera_visibility` is the piece the survey flagged as uniquely valuable**, and it is a projection:
box centre → ego frame (inverse `ego_pose`) → camera frame (inverse `calibrated_sensor`) → pixels
(per-sample intrinsic) → in-frustum test, for all 6 cameras. It returns `off_front_only` — *seen by
some camera but NOT by CAM_FRONT* — alongside nuScenes' own across-camera visibility bin.

**`stack/tanitad/lake/ingest.py` → `NuScenesIngestor`** wraps it on the existing `SourceIngestor`
contract (**no parallel path**), with `action_source="pose_derived"`, `has_can=False`, log-level splits,
and the licence taken from the registry constant.

**`stack/scripts/ingest_nuscenes.py`** — `--analyze` (metadata only, no images) and `--ingest`.

### 4.3 Validation without the bytes — `MEASURED`

**22 tests** in `stack/tests/test_nuscenes.py`, all green, against a **schema-faithful synthetic
fixture**: the same 13 JSON tables, the same token graph, the same `[w,x,y,z]` quaternion convention,
the same ego(x-fwd, y-left, z-up)/camera(x-right, y-down, z-fwd) frames, the same 6-camera rig at
1600×900 with real nuScenes-like intrinsics (CAM_FRONT `fx=1266.417`, CAM_BACK `fx=809.221`).

Agents are planted at known positions — 20 m ahead, 20 m behind, 15 m to the left — so the projection
is checked against ground truth it cannot fake:

- `test_agent_ahead_is_seen_by_cam_front` — `in_ego_camera=True`, range 15–25 m ✅
- **`test_agent_behind_ego_is_off_front_only`** — invisible to CAM_FRONT, visible to a rear camera ✅
- `test_agent_to_the_left_is_off_front_only` — seen by a LEFT camera, **not** a RIGHT one ✅
- `test_visibility_holds_through_a_turn` — holds at **all 8 keyframes of a 260° traversal**, proving the
  projection tracks ego rotation rather than a fixed world axis ✅
- `test_ingest_routes_nuscenes_to_segregated_copyleft_shard` — real ingest → every shard under
  `shards/nc-research/sharealike/nuscenes/`, and **nothing** in `shards/owned-safe/` ✅
- `test_ingested_nuscenes_is_refused_from_commercial_tier_C` — real guard raises; C view resolves to 0 ✅
- `test_ingest_split_is_drive_disjoint` — no log in both splits ✅

**What this does NOT prove:** that the released JSON matches its own documented schema. That is the one
thing only real bytes can settle, and it is a one-command check (`--analyze`) once they land.

**Two real bugs the fixture caught** (both would have cost a debugging cycle on real data):
1. `build_episode` checked blob existence *before* calling an injected `decode_fn`, breaking every
   no-image path.
2. `is_left_turn_heuristic` double-counted roundabouts — inflating **exactly the two numbers the PI is
   deciding on**. Buckets are now disjoint.

### 4.4 🟢 Geometry — the R1 fold-in, done

The brief said *"fold it in or state precisely why not"*. **Folded in**, since it was 9 days stranded
and is the standing prerequisite for the whole owned-real-urban tier.

`pinhole_rectify` + `PinholeIntrinsics` + `brown_conrady_distort` + `pinhole_geometry_report` now live
in **`stack/tanitad/data/calib.py`**; the 9 original tests are ported to `stack/tests/test_calib_r1.py`
(so they run under `pytest -q` for the first time) plus **2 new nuScenes cases**.

**MEASURED, on a real per-sample intrinsic via the driver:**

```
fx=1266.4  1600x900   naive square crop -> f_eff 360.2  (height_clamped=True)
                      pinhole_rectify   -> f_eff 266.0  observed_frac 0.738
```

Exactly the wall the brief described, and it is now removed. Because nuScenes ships **no distortion
coefficients** (`camera_intrinsic` is a bare 3×3), rectify correctly degrades to its honest pad-crop
half — no invented lens model, and the unobserved periphery is an explicit mask rather than a silent
1.35× zoom.

---

## 5. T4 — What R contains, before and after

### 5.1 Destination re-measured, not quoted — `MEASURED` 2026-07-26

The 2026-07-25 note said `TanitDataSet-R` **"does not exist"**. **That is now stale** — it was pushed
after that note was written. Per the `stale-status-in-prose` rule I re-measured at the destination:

| repo | state | files |
|---|---|---|
| `Sayood/TanitDataSet-C` | **public, `gated=manual`**, 23 files, `sha=f152132153`, 2026-07-25 05:17 UTC | 14 shards + catalog + README + NOTICE + manifests |
| `Sayood/TanitDataSet-R` | **public, `gated=manual`**, 23 files, `sha=1172f067c9`, 2026-07-25 13:35 UTC | **identical filenames**, all `owned-safe/comma2k19` |

**The brief's premise is confirmed:** R is byte-identical to C — same 23 paths, all
`shards/owned-safe/comma2k19/…`. **Zero research value-add.**

### 5.2 Before vs after this round

| | before | after | Δ |
|---|---|---|---|
| records in R | 90 (comma2k19) | **90 (comma2k19)** | **0** |
| NC records | 0 | **0** | **0** |
| NC sources ingested | 0 of 8 | **0 of 8** | 0 |
| size | 15.93 GB | 15.93 GB | 0 |
| nuScenes licence in registry | ❌ **wrong** (`CC-BY-NC-4.0`, SA=False) | ✅ `CC-BY-NC-SA-4.0`, SA=True | **fixed** |
| nuScenes adapter | none | **written + 22 tests green** | ✅ |
| NC→C regression tests | 1 generic (hand-made dicts) | **8**, incl. registry-grounded + end-to-end | ✅ |
| D-016 R1 geometry | stranded 9 days | **merged, 11 tests green** | ✅ |

**So: R's *content* is unchanged, and I will not imply otherwise.** What changed is that the machinery
to add NC content is now correct, tested, and one human download away.

### 5.3 Safety check — re-run BEFORE any network write, both legs — `MEASURED`

`python push_tanitdataset.py --verify-only`, both tiers:

| check | C | R |
|---|---|---|
| Leg A — export guard over the view | **90 rows, 0 violations** | **90 rows, 0 violations** |
| Leg B — payload-only, sources | `{comma2k19: 90}` | `{comma2k19: 90}` |
| license classes / names | `{owned-safe: 90}` / `{MIT: 90}` | same |
| gated / `refuse` / NC / share-alike rows | **0 / 0 / 0 / 0** | **0 / 0 / 0 / 0** |
| shards / episodes / splits | 14 / 90 / 72 train + 18 val | same |
| sha256 re-verified over real bytes | **90 / 90 PASS** | **90 / 90 PASS** |
| **verdict** | **`SAFE-TO-PUSH`** | **`SAFE-TO-PUSH`** |

Archived at `safety_check_2026-07-26.json`. **The guard got stronger for free:** its NC and SA source
sets are computed from the registry at run time, so `nuscenes` is now in **both** — if a nuScenes
record ever reached a stage, Leg B would name it.

### 5.4 🔴 Why I did not push

**Publishing to / modifying public content requires the user's consent, and an agent brief is not that
consent** — only the permission system or Sayed directly can authorise it. The 2026-07-25 attempt was
**denied three times** by the permission classifier for exactly this reason.

Since **no content changed** this round, a re-push would upload nothing new; the only genuine
improvement is a **card correction**. I have written that card locally
(`CARD_TanitDataSet-R_2026-07-26.md`) with an honest *"NC sources pending"* statement — it records the
nuScenes SA correction, that NC value-add is still **0 records / 0 GB**, and *why* (the human terms
gate), rather than implying value-add that does not exist. **It is staged, not pushed.**

**To publish it, Sayed runs one command** (the script re-runs the full safety check and aborts on any
failure, sets `gated="manual"` before any reveal, and reads the token from `Keys.txt` in place):

```bash
cp "…/2026-07-26-nc-ingest/CARD_TanitDataSet-R_2026-07-26.md" \
   C:/Users/Admin/tanitad-data/tanitdataset/hf_stage_R/README.md
C:/Users/Admin/venvs/tanitad/Scripts/python.exe \
   "…/2026-07-25-tanitdataset-hf-push/push_tanitdataset.py" --tier R
```

---

## 6. Recommended sequencing

1. **~free — probe PhysicalAI-AV's 34 unread features for a roundabout tag.** Highest expected value
   on this page; may collapse G2+G6 in-house under a commercially-usable licence.
2. **Decide the ShareAlike question — Sayed's call, before ingest.** Four of the top candidates are
   CC BY-NC-SA. Whether a world model trained on SA data is a derivative is a decision about our
   **weights**, not our data.
3. **If (2) is yes — 0.48 GB: nuScenes metadata + map expansion v1.2.** Proves the G1 lane-graph
   pipeline end-to-end and *counts* roundabouts from `scene.description`. Then `--analyze`.
4. **Only if (3) justifies it — 44.9 GB keyframes**, then `--ingest`.
5. **nuPlan maps (0.97 GB) + mini (8.55 GB)** if an explicit route and unprotected-turn tags are wanted.
6. **OpenLane-V2 subset_B (27.7 GB)** only after nuScenes, if lane-bound traffic-light state is wanted.

---

## 7. 🔴 Escalations — integration, not README notes

1. **The ShareAlike/weights question belongs to Sayed and blocks steps 3–6.** Not a data decision.
2. **`nuplan` is not in `SOURCE_REGISTRY`.** `assemble_lake_record` refuses unknown sources by design
   (correct), but if nuPlan is ever probed the entry must be added **deliberately**, with its licence
   read at source. I did **not** add it — I have no primary-source licence read of my own, and adding a
   registry constant on relayed evidence is the exact failure mode §1 exists to prevent.
3. **BDD100K's data licence is UNVERIFIED** and the host was unreachable on two names. Its registry
   entry says `BDD-NC`; that is an assumption, not a read. Someone must read it at the registration wall
   before any BDD100K work.
4. **The push path is still `NotImplementedError`** in `hf_export.py`, so the guarded exporter is
   upstream of a path nobody uses, and `push_tanitdataset.py` re-implements the guard. Two egresses,
   one guard each — fold them together. *(Carried forward from 2026-07-25; still open.)*
5. **`RETRACTION_LOG.md` append recommended**, root-cause class **`licence-assumed-from-a-short-name`**:
   *"`schema.py` registered nuScenes as CC-BY-NC-4.0 with `share_alike=False`; it is CC BY-NC-**SA**-4.0.
   A licence recorded from a familiar short name rather than read at source. Rule: every
   `SourceLicense` constant cites the URL it was read from, and a licence whose authoritative page
   cannot be fetched is recorded at its most restrictive plausible reading."* Same class as the
   2026-07-13 ZOD correction — **second occurrence**, which is why it needs the log, not just a fix.
6. **`DATASET_LANDSCAPE.md` / the H2 survey should record that nuScenes HAS a lane graph.** The survey
   ranked nuScenes #1 purely on label mechanics and never mentioned the map expansion — the single most
   decision-relevant fact about it for this program.

---

## 8. Deliverable manifest

| artifact | where it lives | status |
|---|---|---|
| this report | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-26-nc-ingest/NC_INGEST_REPORT.md` | written, **NOT staged** |
| safety-check JSON (both tiers, both legs) | `repo:…/2026-07-26-nc-ingest/safety_check_2026-07-26.json` | written, **NOT staged** |
| updated R card (NC-pending, honest) | `repo:…/2026-07-26-nc-ingest/CARD_TanitDataSet-R_2026-07-26.md` | written, **NOT pushed** |
| **nuScenes adapter** | `repo:stack/tanitad/data/nuscenes.py` | written, **NOT staged** |
| **nuScenes ingestor** | `repo:stack/tanitad/lake/ingest.py` → `NuScenesIngestor` | written, **NOT staged** |
| **analysis/ingest driver** | `repo:stack/scripts/ingest_nuscenes.py` | written, **NOT staged** |
| adapter tests (22, green) | `repo:stack/tests/test_nuscenes.py` | written, **NOT staged** |
| licence correction | `repo:stack/tanitad/lake/schema.py` | written, **NOT staged** |
| NC→C guard tests (7 new) | `repo:stack/tests/test_lake.py` | written, **NOT staged** |
| **D-016 R1 geometry fold-in** | `repo:stack/tanitad/data/calib.py` | written, **NOT staged** |
| R1 tests ported + 2 nuScenes cases | `repo:stack/tests/test_calib_r1.py` | written, **NOT staged** |
| nuScenes corpus | — | **NOT acquired** (human terms gate) |
| `Sayood/TanitDataSet-R` | `hf:Sayood/TanitDataSet-R` | **unchanged by this agent** |

**Deliberate single-home decision:** the adapter lives **only** in `stack/`, not duplicated into this
incoming dir. The brief asked for "the adapter code" here, but a second copy is how the
`stale-status-in-prose` failure happens to code — and the brief also said *reuse the lake machinery,
do not build a parallel path*. The stack copy is the one under `pytest`. This manifest is the pointer.

**Nothing was `git add`ed, committed or pushed. Nothing was created, modified or deleted on HuggingFace.**

**Test state — `MEASURED`:** full suite `pytest -q` → **1004 passed, 3 skipped** after the schema and
calib changes; **64 passed** across the four directly-affected files after the final edits.
