# H2 external data survey — accessible MULTI-CAMERA sources for intersections & roundabouts

**Date:** 2026-07-25 (local, Europe/Berlin) · **Author:** data-strategy researcher (subagent)
**Scope:** everything EXCEPT the NVIDIA PhysicalAI-AV parity corpus (audited separately by a sibling agent).
**Status:** SURVEY ONLY. Nothing downloaded, no pod, no GPU, no `git add`. Acquisition is a PI decision.

**Workstream:** H2 — *Attention-Based Modality Steering* (`Project Steering/Mission Plan.md` L56;
`ROADMAP.md` row L2; gate **G0.7** = quality-vs-FLOPs Pareto on multi-view clips beating a fixed-camera
baseline at matched FLOPs). The model sees **only the front camera** and must decide when an **additional**
camera is worth its FLOPs. The other cameras are needed twice over: to **build** the ground-truth
"was that camera necessary" label, and to **be** the thing that gets activated.

> **Minimum viable spec for this workstream:** ≥2 cameras with non-front coverage · published
> **intrinsics + extrinsics** (frustum membership is the label mechanism) · **3D agent boxes/tracks** ·
> **ego pose**. Miss any one and the dataset is at best a scenario-statistics reference, not a training source.

**Evidence classes used throughout:** `PUBLISHED` (paper / dataset card / license page, cited) ·
`MEASURED` (I queried it — I queried **nothing**; no MEASURED claims appear below) ·
`INHERITED` (another TanitAD agent/doc, not re-verified by me) · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. Headline — five findings, in order of how much they change the plan

1. **ZOD is DISQUALIFIED and the brief's premise about it is wrong.** ZOD was listed in the task as a
   "surround-camera" Nordic set. It is not: **one** 120° front camera, 3848×2168 @10.1 Hz. Two independent
   probes agree ([zod.zenseact.com](https://zod.zenseact.com/) sensor spec; the paper's sensor summary).
   ZOD stays our best EU *front-camera* corpus (CC-BY-SA-4.0, 14 countries, real CAN + OxTS, loader shipped
   and geometry-falsifier PASS — `2026-07-18-zod-loader-and-geometry-falsifier.md`) and is **useless for H2**.
   *(This is exactly root-cause class "absence/presence assumed from a name, not probed".)*
2. **The EU-roundabout × surround-camera × 3D-boxes intersection is nearly empty.** Roundabouts are dense in
   European data; surround rigs with 3D tracks are dense in US/CN data; the overlap is essentially
   **L2D** (surround + EU, *no* 3D boxes, *no* intrinsics) and **A2D2** (surround + DE, but 3D boxes only
   inside the **front** camera's FOV, and a **no-derivatives** license). That is the structural finding.
3. **We already own a fully-specified 7-camera asset with 4D object tracks and a commercial license:
   Cosmos-Drive-Dreams** (`nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams`, **CC-BY-4.0**,
   registry `cosmos_dd` = `owned-safe`, already **loaded** since D-014). 7 views, pinhole intrinsics +
   30 fps camera/ego poses, 4D object tracking with IDs and movement state, HD map. Zero new license risk,
   zero new registry entry, and by far the cheapest integration. It is a *different repo* from the gated
   PhysicalAI-AV corpus the sibling is auditing.
4. **nuScenes is the only candidate where the H2 label mechanism is turn-key** — 6-camera 360° @12 Hz, full
   `calibrated_sensor` intrinsics+extrinsics, 3D boxes over 23 classes with **instance tracks and a
   per-annotation visibility attribute**, `ego_pose`, and a **~60 GB** keyframe-only download a small team can
   actually pull. Its cost is licensing (`nc-research`) and geography (US/SG → thin roundabouts).
5. **A license discrepancy to escalate, not to fix here.** `stack/tanitad/lake/schema.py` registers
   `nuscenes = SourceLicense("nc-research", "CC-BY-NC-4.0", share_alike=False)`. nuscenes.org states the
   dataset is released under **CC BY-NC-SA 4.0** — i.e. **ShareAlike**. If confirmed, `share_alike` must flip
   to `True`, which moves nuScenes into the segregated copyleft shard logic. See §6.

---

## 1. Comparison table — all candidates against the H2 criteria

Legend: **Rig** = cameras / coverage · **Calib** = published intrinsics **I** + extrinsics **E** ·
**3D** = 3D agent boxes/tracks and their coverage · **Pose** = ego pose/odometry ·
**Scenario** = roundabout / intersection content · **Class** = license class **in OUR taxonomy** ·
**Comm** = commercial use permitted · **SA** = share-alike.

| Dataset | Rig | Calib | 3D agents | Pose | ⭐ Scenario content | Access / size | Class · Comm · SA | H2 verdict |
|---|---|---|---|---|---|---|---|---|
| **nuScenes** | **6 cam, full 360°**, 1600×900 @12 Hz `[P]` | **I+E** (`calibrated_sensor`) `[P]` | **23 classes, 2 Hz keyframes, 1.4 M boxes / 40 k keyframes, instance tracks + visibility attribute** `[P]` | `ego_pose` `[P]` | 1000×20 s scenes, Boston Seaport + SG One North/Queenstown/Holland Village; **no published roundabout count**; rotaries/roundabouts present but sparse `[E]` | free registration; **~400 GB full / ~60 GB keyframes-only** `[P]` | `nc-research` · ✗ · **likely SA** (see §6) | ✅ **#1 — label mechanism turn-key** |
| **Cosmos-Drive-Dreams** | **7 cam** (front_wide 120°, cross L/R 120°, rear L/R 70°, rear tele 30°, front tele 30°) `[P]` | **I+E** (pinhole intrinsics per view; camera poses from vehicle pose @30 fps) `[P]` | **4D object tracking** — position, dims, movement state, track IDs `[P]` | vehicle pose @30 fps (FLU) `[P]` | 5,843 labelled 10 s clips + 81,802 synthetic; 7 weathers, intersections + VRUs `[I, DATASET_LANDSCAPE]`; **roundabout count unknown**, provenance is NVIDIA RDS-HQ → US-heavy `[E]` | HF, ungated; **~3 TB** total, shard-streamable `[P]` | **`owned-safe`** (CC-BY-4.0) · **✓** · ✗ | ✅ **#2 — already ours, already loaded** |
| **L2D** (`yaak-ai/L2D`) | **6 cam, 360°**, 1080×1920 @10 Hz + rendered BEV `map` `[P]` | ⚠️ **E only** — `extrinsic_RDF.yaml`; **NO intrinsics ship anywhere** `[I, l2d note, verified on real tree]` | ❌ **none** — no perception labels at all | GPS + IMU + CAN state; waypoints; 1,103 drives `[I]` | **30 German cities**, 60 EVs, 3 yrs; **roundabout ROUTE slot on 3,532 episodes** `[I, TIER_INTEGRATION]`; NL instructions incl. roundabout-exit phrasing `[P]` | HF, **ungated**, `gated=False`; 100 k eps / 735 h / **~90 TB** full, but sliceable per-episode via `from/to_timestamp` `[P/I]` | **`owned-safe`** (Apache-2.0) · **✓** · ✗ | ✅ **#3 — only commercial + roundabout-rich surround set** |
| **Argoverse 2 (Sensor)** | **7 ring cam 360°**, 2048×1550 @20 Hz + 2 stereo `[P]` | **I+E** for all 9 cams + 6-DOF map-aligned pose `[P]` | 3D tracks, 30 classes, 10 Hz `[P]` | 6-DOF pose `[P]` | 1000 × 15 s scenes, 6 **US** cities (Austin, Detroit, Miami, Pittsburgh, Palo Alto, DC); rich intersections, **roundabouts rare** `[E]` | free registration; **~1 TB** sensor split `[E]` | `nc-research` (CC-BY-NC-SA-4.0) · ✗ · **✓ SA** | ⚠️ viable but strictly dominated by nuScenes (bigger, no roundabouts, SA) |
| **nuPlan** | **8 surround cam @10 Hz** `[P]` | I+E `[P]` | exhaustive tracks, 6 agent classes `[P]` | full logs, 1300+ h `[P]` | Las Vegas / Boston / Pittsburgh / Singapore; **US-style mega-intersections**, roundabouts rare `[E]` | camera subset only: **~128 h / ~16 TB = 10 % of logs** `[P]` | `nc-research` (not yet in registry) · ✗ · verify | ⚠️ 16 TB for the camera subset — accessibility fails for a small team |
| **ONCE** | **7 cam, 360°** `[P]` | I+E `[P]` | 3D boxes, 5 classes, **16 k annotated scenes / 417 k boxes** (of 1 M) `[P]` | ego `[I, registry]` | China; lidar-centric; **roundabouts rare** in CN urban `[E]` | registration; TB-scale | `nc-research` (ONCE-NC) · ✗ · ✗ | ⚠️ no scenario advantage over nuScenes |
| **KITTI-360** | **4 cam** — front 90° perspective **stereo** + **2× 180° side fisheye** `[P]` | I+E (MEI fisheye model) `[P]` | 3D bounding primitives, static+dynamic; 150 k annotated images `[P]` | full SLAM/GNSS pose `[P]` | **Karlsruhe DE suburbs, 73.7 km total** — tiny; some EU roundabouts `[E]` | free registration; ~300 GB | `nc-research` (CC-BY-NC-SA-3.0) · ✗ · ✓ SA | ⚠️ **no rear camera**; 73.7 km is far too small for ≥40 clusters |
| **PandaSet** | **6 cam, 360°** `[P]` | I+E `[P]` | 3D boxes @10 Hz, 103 seq × 8 s `[P]` | GPS/IMU `[P]` | **San Francisco / El Camino only** — day, no roundabouts `[I, DATASET_LANDSCAPE]` | pandaset.org sign-up (+ HF mirror `georghess/pandaset`); ~44.5 GB `[I]` | **`owned-safe`** (CC-BY-4.0) · ✓ · ✗ | ⚠️ loader exists but **geometry-blocked** (front fx=1970 → f_eff 467≠266); ~14 min of data |
| **A2D2** | **6 cam, 360°** `[P]` | I+E `[P]` | ⛔ **12,497 frames, and only for objects inside the FRONT camera's FOV** `[P]` | bus data `[P]` | 3 south-German cities (Gaimersheim / Ingolstadt / Munich) → **real EU roundabouts** `[E]` | AWS Open Data + a2d2.audi; ~2.3 TB | `nc-research` (**CC-BY-ND-4.0**) · nominally ✓ · ✗ | ❌ **DISQUALIFIED twice over** — front-FOV-only 3D boxes break the label; **ND = no derivatives** |
| **OmniHD-Scenes** | **6 cam** + 128-beam lidar + 6× 4D radar `[P]` | not stated in abstract `[?]` | **200 annotated clips / 514 k 3D boxes** (of 1501 clips × 30 s) `[P]` | not stated `[?]` | China; roundabouts rare `[E]` | 2077ai.com; license **not published in the abstract** `[?]` | **UNVERIFIED — treat as refuse-adjacent until a license file is read** | ⚠️ watch-list only; unverified ≠ permissive |
| **Lyft L5 / Woven Planet** | 7 cam + 3D boxes `[P]` | I+E `[P]` | 3D boxes `[P]` | ego `[P]` | Palo Alto only; roundabouts rare `[E]` | ⚠️ **current availability could not be confirmed** — no live download path verified `[?]` | not in registry; verify | ❌ accessibility unproven; superseded by AV2 |
| **BDD100K** | ⛔ **single front camera**, 720p @30 fps `[P]` | ✗ | 2D only | GPS/IMU | very high US urban diversity | ~1.8 TB | `nc-research` · ✗ · ✗ | ❌ **DISQUALIFIED — single camera** |
| **Zenseact ZOD** | ⛔ **ONE** 120° front camera, 3848×2168 @10.1 Hz `[P ×2]` | I (KB) `[P]` | 2D/3D boxes, front FOV `[P]` | OxTS 100 Hz `[P]` | **14 EU countries** — the best EU scenario mix we know of | signed agreement + SDK | `owned-safe` (CC-BY-SA-4.0) · ✓ · ✓ SA | ❌ **DISQUALIFIED — single camera** |
| **Cityscapes** | ⛔ front **stereo pair** only; 30-frame snippets `[P]` | I+E (stereo) | 3D boxes in Cityscapes-3D, front only | vehicle odom | 50 DE cities — EU roundabouts | ~60 GB | `nc-research`-class · ✗ · ✗ | ❌ **DISQUALIFIED — no non-front coverage** |
| **comma2k19** | ⛔ single front cam | — | ✗ | CAN+GNSS | highway commute | ~100 GB | `owned-safe` (MIT) · ✓ | ❌ **DISQUALIFIED — single camera** (named in brief) |
| **PhysicalAI-WorldModel-Synthetic** | 7 cam @24 fps `[P]` | ? | ❌ none | ⛔ **POSE-LESS** `[I, measured 2026-07-15 on real bytes]` | 264 k clips, targeted long-tail incl. lanechange `[P]` | HF ungated, 8.3 TB | `owned-safe` (OpenMDW-1.1) · ✓ · ✗ | ❌ no ego pose, no 3D boxes → cannot build the label |
| **rounD** | ⛔ **DRONE / BEV** | n/a | trajectories | n/a | **3 roundabouts, 6 h, 13,746 road users** `[P]` | RWTH request form | NC | ❌ **DISQUALIFIED — no ego camera exists** |
| **inD** | ⛔ **DRONE / BEV** | n/a | trajectories | n/a | **4 unsignalised intersections, 50 k+ frames, 8,200 veh + 5,300 VRU** `[P]` | RWTH | NC | ❌ **DISQUALIFIED — drone** |
| **exiD / highD** | ⛔ **DRONE / BEV** | n/a | trajectories | n/a | exiD: 7 highway sections, 16 h, 69,430 users, merges/lane-changes `[P]` | RWTH | NC | ❌ **DISQUALIFIED — drone** |
| **openDD** | ⛔ **DRONE / BEV**, 4K | n/a | trajectories | n/a | roundabout-focused, 62 h `[P]` | request | NC | ❌ **DISQUALIFIED — drone** |
| **INTERACTION** | ⛔ **DRONE + fixed infrastructure cameras** | n/a | trajectories | n/a | roundabouts, merges, unsignalised intersections, multi-country | request | NC | ❌ **DISQUALIFIED — no ego-vehicle camera** |
| **Waymo Open / WOD-E2E / Waymax** | (5 cam) | — | — | — | — | — | 🔴 **`refuse`** — terms follow the **trained weights** | ❌ **NOT PROPOSED.** Excluded by `schema.py`; `assemble_lake_record` raises |

`[P]` PUBLISHED · `[I]` INHERITED · `[E]` ESTIMATED · `[?]` unverified.

### ⚠️ On the drone datasets, explicitly
rounD / inD / exiD / highD / openDD / INTERACTION are **bird's-eye drone or fixed-infrastructure**
recordings. They contain **no ego-vehicle camera of any kind**, therefore no front-camera input and no
alternative camera to activate. Their roundabout counts (rounD: 3 roundabouts, 13,746 road users) are the
richest numbers in this survey and are **exactly the kind of number that must not be allowed to
masquerade as a fit.** Their only admissible use for H2 is (a) *label-design* reference — what does a
yielding-conflict geometry look like at a roundabout entry — and (b) *scenario-statistics priors*
(how often does the binding agent approach from outside a ±60° front cone). Neither is a training source.

---

## 2. Ranked top-3 for H2

### 🥇 #1 — **nuScenes** · license class `nc-research` (CC-BY-NC-**SA**? see §6) · commercial ✗ · redistribution ✗
**Why it wins:** it is the only candidate where **the H2 ground-truth label falls out of the released
annotations with no new perception model.** nuScenes publishes, per keyframe: 3D boxes for 23 classes with
**instance tokens** (so an agent is a track, not a detection), a **visibility attribute** derived across all
six cameras, `calibrated_sensor` intrinsics+extrinsics for every camera, and `ego_pose`. The label
*"agent A was inside CAM_BACK_LEFT's frustum at t and NOT visible from CAM_FRONT"* is a direct projection
computation — the definition of the mechanism, not an approximation of it. Everything else on this list
requires either pseudo-labelling (L2D), front-FOV-only boxes (A2D2), or a bigger download for no extra
label fidelity (AV2, nuPlan).

Second reason: **it is the only surround set a small team can obtain today at a sane size** —
~60 GB keyframes-only, or ~400 GB full including the ~12 Hz non-keyframe sweeps we would want for a
10 Hz world model.

**What binds us:** `nc-research`. Under our tier machinery this is tier `nc` — internal training and
research claims only, **never** in TanitDataSet-C, never in a commercial shipment, and (per
`TANITDATASET_TIER_INTEGRATION` §4) **any derivative inherits the strictest input tier**, so an H2 gating
policy trained on nuScenes is itself `nc`. Redistribution of bytes is out of the question; we ship pointers
plus a build recipe, as we already do. If the ShareAlike finding in §6 confirms, nuScenes rows must also go
to the **segregated copyleft shard**, never co-mingled.

**Acquisition steps (concrete):**
1. Register at nuscenes.org (free, non-commercial declaration) → accept the terms **as a human**; a subagent
   must not accept terms on the PI's behalf.
2. Pull `v1.0-trainval` **metadata** first (~0.4 GB) — this alone answers the scenario question offline:
   parse `scene.description`, `map` layers, and `ego_pose` curvature to count roundabout and unprotected-left
   traversals **before** committing to any image download. This is the cheap discriminating step.
3. Only then pull image blobs — keyframes (~60 GB) for label construction, sweeps only if 2 Hz proves too
   coarse for the trigger.
4. Adapter: extend `stack/tanitad/data/` with a `nuscenes.py` mirroring the PandaSet/ZOD loaders; add
   `nuscenes` handling to `LakeRecord` (already registered in `SOURCE_REGISTRY`, so the license axis is free).

**Integration cost `[ESTIMATED]`:** **4–6 engineer-days.** Breakdown: adapter + drive/scene-disjoint split
(1.5 d) · multi-camera record extension — our `LakeRecord.frames` is `[T,C,S,S]` single-view and H2 needs a
**camera axis**, which is a schema change (1.5–2 d, and it is shared work with any other candidate) ·
frustum-membership label builder from boxes+calib (1 d) · geometry (0.5–1 d, below).

**Geometry `[ESTIMATED, must re-verify]`:** nuScenes CAM_FRONT `fx ≈ 1266 px` on 1600×900. Our square-crop
canon is height-bound at 900 → `f_eff ≈ 1266 × 256/900 ≈ 360 px` vs the canonical **266** — a ~1.35×
mismatch, i.e. **the same PandaSet-class wall** (the standing rule is *fx > 1122 px on a 1080-tall frame is
not square-croppable to 266*; scaled to 900-tall the threshold is ~935 px). The fix already exists:
`calib_r1.pinhole_rectify` (D-016 R1), validated 9/9 but **still not folded into
`stack/tanitad/data/calib.py`** `[INHERITED, 2026-07-17 note]`. So nuScenes does **not** need new geometry
research — it needs the R1 integration that is already the standing prerequisite for the whole owned
real-urban tier. GeoCalib is **not** needed: nuScenes publishes intrinsics.

### 🥈 #2 — **Cosmos-Drive-Dreams** · license class **`owned-safe`** (CC-BY-4.0) · commercial **✓** · SA ✗ · redistribution: attribution-only, so *technically* redistributable
**Why:** it is the **only** fully-specified multi-camera candidate whose license lets an H2 result be
shipped, and we already hold it. 7 views spanning front-wide 120°, **cross left/right 120°** (exactly the
roundabout-entry geometry), rear left/right 70°, and two teles; pinhole intrinsics per view; camera poses
derived from a 30 fps vehicle pose; **4D object tracking** with IDs, dimensions and movement state; plus HD
map polylines. Registry entry `cosmos_dd` exists, tier `ship`, `commercial_ok=True`, and the loader was
shipped at D-014 — so the marginal work is *extending an existing loader to six more camera keys and
reading `all_object_info`*, not building an ingest.

**What binds us:** CC-BY-4.0 requires attribution in the shard NOTICE (we already have `attribution_id`).
It is **`is_synthetic=True`** in our registry — the 81,802 generated videos are renders, and per
`TANITDATASET_TIER_INTEGRATION` §4 a derivative inherits the strictest input, so anything conditioned on
gated PhysicalAI clips is **still gated** regardless of how synthetic it looks. Only the CC-BY-4.0 RDS-HQ
lineage is `ship`. Also: a synthetic-only H2 demo invites the "does it transfer" objection at G0.7.

**Integration cost `[ESTIMATED]`:** **2–3 engineer-days** — the cheapest on this list by a factor of two,
because the geometry (120° f-theta → canonical crop) is the ZOD-class path already proven drop-in, and the
license axis needs no new registry entry.

**Open risk `[HYPOTHESIS]`:** its roundabout content is unknown and its provenance (NVIDIA RDS-HQ) is
US-weighted, so it may simply not contain enough roundabout entries. This is answerable for **$0** from the
already-cached clip metadata — do that before ranking it above nuScenes.

### 🥉 #3 — **L2D** (`yaak-ai/L2D`) · license class **`owned-safe`** (Apache-2.0) · commercial **✓** · SA ✗
**Why:** the only source in the survey that is simultaneously (a) a genuine **6-camera 360° ego rig**,
(b) **commercially clean**, and (c) **roundabout-rich in the right geography** — 60 EVs across **30 German
cities**, with the roundabout ROUTE slot already minted on **3,532 episodes** `[INHERITED, TIER_INTEGRATION
2026-07-21]` and NL instructions that name roundabout exits explicitly. Driving-school data is
*disproportionately* intersection- and roundabout-dense by construction: that is what learner routes are for.
The adapter is already staged (`stack/tanitad/data/l2d.py`, 9 tests green, one drive ingested end-to-end).

**Why only #3 — the two gaps are precisely H2's two needs:**
- **No intrinsics ship** — only `extrinsic_RDF.yaml`. Frustum membership needs a focal. This is the one
  candidate where a **GeoCalib-style intrinsics estimation step is mandatory**, cross-checked against ego
  speed and the known extrinsics, then run through the geometry falsifier. Until it passes, L2D camera
  frames carry `intrinsics_native={"estimated": true}` and must never assert `f_eff=266`.
- **No 3D agent annotations at all.** The label must be **pseudo-generated**: a detector per view plus
  monocular depth, or a multi-view 3D detector, then temporal association. This is a real perception
  sub-project, and its error becomes label noise in the H2 target.
- **GDPR:** real German driving-school footage with **no anonymisation statement in the card**
  `[INHERITED, l2d note]`. Apache-2.0 grants copyright, not the data-protection right. A face/plate check is
  required before **re-hosting any frame** — this does not block internal training, it blocks publication.

**Integration cost `[ESTIMATED]`:** **10–15 engineer-days** — adapter multi-camera extension (2 d) ·
intrinsics estimation + falsifier (3–4 d) · pseudo-3D label pipeline and its validation (5–8 d). Expensive,
and the *only* path to a commercially-shippable H2 capability on European roundabouts.

---

## 3. DISQUALIFIED — and the specific reason

| Dataset | Disqualifying reason |
|---|---|
| **Zenseact ZOD** | **Single front camera** (1× 120°, 3848×2168). Verified twice. No camera to activate. |
| **BDD100K** | **Single front camera**, 720p. Confirmed in the CVPR paper. |
| **comma2k19** | **Single front camera** (our own `owned-safe` anchor). |
| **Cityscapes** | Front **stereo pair** only — no left/right/rear coverage; no video-rate 3D tracks. |
| **A2D2** | Two independent kills: 3D boxes exist on only **12,497 frames and only for objects in the FRONT camera's FOV** — the label needs agents that the front camera *cannot* see, which A2D2 never annotates; and **CC-BY-ND-4.0 forbids derivatives**, which is what a re-shard is. |
| **rounD / inD / exiD / highD / openDD** | **Drone / bird's-eye view.** No ego camera → no front-camera input, no camera to switch on. Scenario-statistics reference only. |
| **INTERACTION** | **Drone + fixed infrastructure cameras.** Same kill. |
| **Waymo Open / WOD-E2E / Waymax** | 🔴 **`refuse`** — the only licenses surveyed whose terms follow the **trained weights** into the product (Waymax §2.e bars foundation-model training). `assemble_lake_record` raises on them. **Not proposed, per the brief and per `schema.py`.** |
| **PhysicalAI-WorldModel-Synthetic-Scenarios** | 7-cam rig, but **pose-less** (measured on real bytes 2026-07-15) and no 3D boxes → the label cannot be constructed. |
| **Lyft L5 / Woven Planet** | **Accessibility unproven** — no live download path confirmed in this survey; strictly dominated by Argoverse 2 anyway. |
| **OmniHD-Scenes** | **License not published** in the material I could reach. *Unverified ≠ permissive* — same treatment as TLD / LISA. Watch-list until a license file is read. |
| **nuPlan** | Not disqualified on capability (8 cams, tracks) but on **accessibility**: the camera subset alone is ~16 TB for 10 % of logs. |
| **KITTI-360** | Not disqualified on capability, but **73.7 km total** cannot supply ≥40 roundabout episode-clusters, and there is **no rear camera**. |

---

## 4. Do we need external data at all? — the conditional recommendation

I do not have the sibling's PhysicalAI counts, so this is framed as a decision rule. Note the standing
in-corpus reserve: **500 multi-view clips (front+L+R+rear) were reserved for the G0.7 modality demo**
(`Project Steering/Phase 0 Plan.md` §2.2 row B2) `[INHERITED]` — the open question is what *fraction* of
those 500 are intersection/roundabout entries, and whether the side/rear bytes are actually materialised
rather than merely reserved.

**Two thresholds, because H2 has two distinct claims:**

### N_mech = 40 multi-camera episode-clusters containing intersection or roundabout entries
This is the bar for the **descriptive claim** — *"in X % of intersection frames, a decision-relevant agent is
visible only off-front (95 % CI ±y)"* — which is the falsifier that decides whether H2 has a signal at all.
40 is the program's standing power bar: our decision-grade interval is the **episode-cluster bootstrap over
40 val episodes** (`taniteval/ci.py`), and this claim is a single-arm proportion, not a two-arm comparison,
so 40 clusters is exactly sufficient and not more.
- **If PhysicalAI's multi-view reserve yields ≥40 such clusters → run the mechanism study in-corpus. Acquire nothing.**
- **If <40 → acquire nuScenes** (free, ~60 GB, and its metadata alone answers the count for $0 before any
  image download).

### N_train = 200 drive-disjoint roundabout/intersection episode-clusters
This is the bar for the **G0.7 claim** — a *trained* gating policy beating a fixed-camera baseline at matched
FLOPs. That is a two-arm comparison, so the 40-cluster bar applies to the **held-out val set alone**; at a
conventional 80/20 drive-disjoint split that implies ~200 total, and 200 is already thin for learning a
policy on a rare trigger. **Below 200 → acquire before training, not after.**

### The third condition, which is not about counts at all
**PhysicalAI-AV is `gated-confidential`.** Even if it clears both thresholds, an H2 result trained on it can
never be published, never shipped, and — per the derivative rule — neither can anything conditioned on it.
So:

> **If the H2 result is ever intended to be a public USP or a product capability
> (`Master Plan.md` names ABMS as a core USP), a `ship`-tier corpus is required regardless of the counts.**
> That means **Cosmos-Drive-Dreams** (cheapest, already ours, synthetic) and/or **L2D** (expensive, real, EU
> roundabouts). This condition is independent of N and should be decided separately.

### Recommended sequencing
1. **$0, this week:** sibling reports PhysicalAI roundabout/multi-view counts. In parallel, count
   Cosmos-Drive-Dreams roundabout clips from already-cached metadata. Neither costs a download.
2. **If N_mech unmet, or the mechanism study wants a clean external check:** register for nuScenes, pull
   **metadata only** (~0.4 GB), count roundabout + unprotected-left traversals from `scene.description` and
   map layers, *then* decide on the 60 GB image pull.
3. **Only if a shippable H2 capability is wanted:** open the L2D multi-camera track (intrinsics estimation
   first — it is the cheap falsifier; if the focal cannot be pinned to the falsifier's satisfaction, the
   whole L2D camera path is dead and 10 days are saved).

---

## 5. Cross-cutting cost that no single dataset choice avoids

**Our record schema is single-view.** `LakeRecord.frames` is `uint8 [T, C, S, S]` with `image_size=256` and
one asserted `f_eff=266`. H2 needs a **camera axis** (`[T, K, C, S, S]`) plus per-camera intrinsics,
extrinsics and an activation mask, and the license/tier machinery must carry through unchanged. That is
**1.5–2 engineer-days** and it is a **prerequisite for every candidate**, including staying entirely
in-corpus on PhysicalAI. It should be scoped before the acquisition decision, not after — otherwise the
data arrives and sits, which is the failure mode operating-standard rule 3 exists to prevent.

---

## 6. ⚠️ Escalations (not fixed here — I did not edit code or the registry)

1. **`nuscenes` license may be mis-registered.** `schema.py` has
   `SourceLicense("nc-research", "CC-BY-NC-4.0", share_alike=False)`. nuscenes.org states the dataset is
   released under **CC BY-NC-SA 4.0** — ShareAlike. If confirmed against the terms-of-use page (my fetch of
   that page returned empty; this rests on the dataset page and secondary sources, so **treat as
   PROVISIONAL and re-verify at the source before acting**), `share_alike` must flip to `True`, which routes
   nuScenes rows into the segregated copyleft shard. Owner: Data Engineering. This is the same root-cause
   class as the ZOD `research/NC → CC-BY-SA-4.0` correction of 2026-07-13.
2. **`nuplan` and `pandaset`-adjacent gaps:** nuPlan is not in `SOURCE_REGISTRY` at all. If it is ever
   probed, it needs an explicit entry first — `assemble_lake_record` refuses unknown sources by design, which
   is correct, but the entry should be added deliberately rather than in a hurry.
3. **`DATASET_LANDSCAPE.md` is stale on ZOD's rig.** It lists ZOD's sensors as "cam (KB fisheye …)" without
   stating **one** camera, which is what let ZOD be briefed to me as a surround-camera candidate. Recommend
   adding an explicit **camera-count column** to that table so this class of error cannot recur.
4. **Integration escalation (not a README note):** the multi-camera `LakeRecord` extension in §5 blocks H2
   for every candidate. It needs an owner and a slot, not a line in this document.

---

## 7. Deliverable manifest

| Artifact | Location | Status |
|---|---|---|
| This survey | `TanitAD Research Hub/Data Engineering/Research/2026-07-25-h2-multicam-data-survey/H2_EXTERNAL_DATA_SURVEY.md` (repo working tree) | written, **NOT staged** (per brief: do not `git add`) |
| Datasets downloaded | none | none — survey only, per brief |
| Code changed | none | none — no registry or loader edits made |

**Sources:**
[zod.zenseact.com](https://zod.zenseact.com/) ·
[ZOD paper (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Alibeigi_Zenseact_Open_Dataset_A_Large-Scale_and_Diverse_Multimodal_Dataset_for_ICCV_2023_paper.pdf) ·
[nuScenes](https://www.nuscenes.org/nuscenes) ·
[nuScenes paper](https://arxiv.org/pdf/1903.11027) ·
[Argoverse 2 user guide — Sensor](https://argoverse.github.io/user-guide/datasets/sensor.html) ·
[Argoverse 2 paper](https://arxiv.org/abs/2301.00493) ·
[Cosmos-Drive-Dreams dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams) ·
[L2D — LeRobot goes to driving school](https://huggingface.co/blog/lerobot-goes-to-driving-school) ·
[A2D2](https://www.a2d2.audi/en/) ·
[A2D2 on AWS Open Data](https://registry.opendata.aws/aev-a2d2/) ·
[ONCE](https://openreview.net/pdf?id=KBbxt3JGn0Y) ·
[KITTI-360](https://arxiv.org/pdf/2109.13410) ·
[BDD100K](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.pdf) ·
[PandaSet](https://pandaset.org/) ·
[nuPlan](https://arxiv.org/html/2403.04133v1) ·
[OmniHD-Scenes](https://arxiv.org/abs/2412.10734) ·
[inD](https://arxiv.org/pdf/1911.07602) ·
[rounD (ITSC 2020)](https://dl.acm.org/doi/10.1109/ITSC45102.2020.9294728) ·
[drone-dataset-tools (RWTH)](https://github.com/ika-rwth-aachen/drone-dataset-tools)
