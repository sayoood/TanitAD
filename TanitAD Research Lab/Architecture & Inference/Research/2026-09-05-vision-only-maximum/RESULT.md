# What is maximally achievable with cameras alone — the v5a (pure-vision) study

**status: IN PROGRESS — done: §0–§4 (inventory, mechanisms, the ten-rung v5a ladder, the v5a/v5b decision) / next: §5 manifest + the D-V5A-* / H-V5A-* register rows in GOALS_AND_CLAIMS.md**

- **Agent:** TanitAD Architecture & Inference FlyWheel · **date:** 2026-09-05 · **branch:** `agent/arch-inf-20260803`
- **GPU spent by this document:** **0** (two network probes on the dev box, no CUDA; `nvidia-smi` showed the 4060 at 52 % / 4,610 MiB from sibling streams and was not touched)
- **Trigger (PI, verbatim, 2026-09-05):** *"i prefer to do the environment extensions in two versions/steps, let start by pure vision and then add lidar. So check, what we can do maximally with vision, bev, und was else?"*
- **Scope:** the **v5a** release = everything reachable from **cameras alone**. **v5b** (LiDAR) is out of scope here and is priced only as a delta in §4.
- **Sibling stream boundary:** `Project Steering/REFCV5_DESIGN_PLAN.md` §7–§10 is being written concurrently on the older single-track assumption. ⛔ This document does **not** edit it. Where the two disagree, §5 lists the reconciliation items for the Master Mind.

---

## §0 Reading guide

Every number carries its evidence class — `MEASURED (ours + artifact path)` · `PUBLISHED (cited)` ·
`INHERITED (another doc, NOT re-verified)` · `ESTIMATED` · `HYPOTHESIS` — and, where it is an eval
number, its tier (T0 diagnostic / T1 self-action open loop). Nothing here is a driving-performance
claim; this document decides what to **build and measure**, not what the model can do.

**The one-line answer.** Vision-only is not a fallback: **PhysicalAI-AV is a 7-camera 360°-covered
corpus and we read exactly one of the seven, which is 33.4 % of the azimuth circle** (MEASURED
today). Buying the other six for our training corpus costs **429 GB** — **27 % of what the LiDAR
would cost** — and takes the observable azimuth to **100 %**. Two of DiffusionDrive's three grounded
attentions (waypoint-indexed spatial, agent cross-attention) are reachable from cameras alone with
**no new sensor at all**; the third (ego-query) we deliberately refuse. LiDAR's remaining, genuine
contribution is **metric depth without a learned scale prior**, and §4 states honestly how much of
that a maximised v5a already recovers.

---

## §1 The camera inventory, settled by evidence

### 1.1 The question, and why an impression was not good enough

Every prior programme document says "vision-only" and then quietly means *"the front-wide camera"*.
Those are not the same statement, and the difference is the largest single lever in this study. The
census below is therefore built from **three independent probes**, two of which were run today:

| probe | mechanism | independent of | evidence class |
|---|---|---|---|
| **P-1 — the pinned read-set** | `stack/tests/test_physicalai_feature_readset.py` asserts the three read-set layers **and the exact feature names** against source (`stack/tanitad/data/physicalai.py:232-235`, `stack/scripts/physicalai_r0.py:36-38`) | any prose count — this test exists *because* the prose count rotted four times | **MEASURED** (source, this repo) |
| **P-2 — the dataset's own manifest + the 36-feature census** | `features.csv` has exactly 36 rows; the DataFlyWheel decoded them 2026-07-26 (`…/Data Engineering/Implementation/incoming/2026-07-26-physicalai-feature-probe/`) | our code entirely | **MEASURED** by that stream (INHERITED here, and **corrected** in §1.4) |
| **P-3 — a live HF blob listing + the rig extrinsics** | `code/probe_camera_rig.py`, run today: `repo_info(files_metadata=True)` over **every** blob (not one chunk × a multiplier), plus the 63 KB `calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet` decoded into per-sensor mount poses and boresight azimuths | both of the above; it is the only probe that can say **where each camera points** | **MEASURED** today (`raw/camera_rig_probe.json`) |

⚠️ P-1 and P-2 answer *"which features exist and which do we read"*. **Neither can answer the
coverage question**, because a feature name does not say where the sensor points — `camera_rear_left_70fov`
could be anywhere. That is why P-3 was run, and it is the probe the §2 ranking rests on.

### 1.2 How many cameras, and where they point (MEASURED today, P-3)

`calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet` is MultiIndexed by
`(clip_id, sensor_name)` with columns `qx, qy, qz, qw, x, y, z`; the probe read **1,700 rows =
100 clips × 17 sensors** (7 cameras + 1 LiDAR + 9 radar — this chunk is a `radar_config = low`
rig). Poses are the per-sensor median over the 100 clips; the p95−p05 spread is banked per axis in
the JSON.

**Convention control.** The rig frame is x forward, y left, z up (MEASURED by the parked-car
experiment, `stack/tanitad/data/bev_raster.py`). A camera's optical axis is its own +z. The probe
computed the azimuth of **both** the rotated +z and the rotated +x so a convention error would be
visible rather than assumed — and the control read its known value: `camera_front_wide_120fov`'s
+z axis lands at **−0.58° azimuth, −0.54° elevation**, i.e. straight ahead and level. The +x axis
lands at −90.58°, which is the image-right direction, exactly as expected. **+z is the boresight.**

| sensor | mount (x fwd, y left, z up) m | boresight az | HFOV (name) | azimuth span | do we read it? |
|---|---|---:|---:|---|:--:|
| `camera_front_wide_120fov` | (+1.754, −0.001, **+1.430**) | **−0.58°** | 120° | [−60.6°, +59.4°] | ✅ **the only one** |
| `camera_front_tele_30fov` | (+1.715, +0.093, +1.445) | −0.23° | 30° | [−15.2°, +14.8°] | ❌ |
| `camera_cross_left_120fov` | (+2.519, +0.937, +0.900) | +67.33° | 120° | [+7.3°, +127.3°] | ❌ |
| `camera_cross_right_120fov` | (+2.522, −0.930, +0.904) | −66.47° | 120° | [−126.5°, −6.5°] | ❌ |
| `camera_rear_left_70fov` | (+1.996, +1.051, +0.902) | +151.80° | 70° | [+116.8°, +186.8°] | ❌ |
| `camera_rear_right_70fov` | (+1.966, −1.006, +0.902) | −151.25° | 70° | [−186.3°, −116.3°] | ❌ |
| `camera_rear_tele_30fov` | (+0.430, +0.306, +1.456) | +179.83° | 30° | [+164.8°, +194.8°] | ❌ |
| *(`lidar_top_360fov`)* | (+1.189, +0.000, +1.864) | 360° | — | full circle | ❌ (v5b) |

**⭐ The headline number — azimuth coverage, computed as the union of the intervals on the circle
(3,600 bins, `union_coverage()`):**

| read-set | azimuth coverage | what it means |
|---|---:|---|
| **`camera_front_wide_120fov` alone — TODAY** | **33.36 %** | two thirds of the circle around the car is unobserved at inference |
| + `cross_left` + `cross_right` (the three 120° cameras) | **70.53 %** | continuous [−126.5°, +127.3°]: both flanks, both adjacent lanes, cross traffic |
| **all seven** | **100.00 %** | full surround, with two 30° tele overlays (front and rear) |

**⚠️ Coverage is azimuth, not capability.** Two qualifications the number does not carry:
1. **Angular resolution differs 4× between lenses.** Native video is 1080p30 (PUBLISHED, card
   §"Camera Data"), so the wide gives 120°/1920 px = **0.0625°/px** and the tele 30°/1920 px =
   **0.0156°/px** — 4× finer. **Our canonical frame is coarser still**: 640 px over 120° cylindrical
   = **0.1875°/px** (`stack/tanitad/data/calib.py`, `f_ref` 305.577; ⛔ the pinhole formula gives
   92.6° on this projection and is wrong — CLAUDE.md). A 1.8 m car at 100 m is **≈ 5.5 px** wide in
   our frame and **≈ 66 px** in the native tele. That is the long-range lever, and it is a
   *different* lever from coverage.
2. **The trunk is coarser than the frame.** The 8 × 20 PV map is **6.0° per column**; at 30 m one
   column is ≈ 3.1 m of lateral extent — wider than a lane. Any waypoint-indexed sampling (§2.1)
   reads the trunk, so its useful resolution is set here, not by the camera.

### 1.3 What it costs — MEASURED bytes, not an extrapolation (P-3)

Full-repo blob listing, summed per feature directory (`raw/camera_rig_probe.json`,
`probes.P_A_blob_listing`; 3,146 files per feature, matching the chunk count):

| feature | corpus bytes | TB | MB per clip (÷ 306,152) | B1 corpus (4,713 clips) |
|---|---:|---:|---:|---:|
| `camera_front_wide_120fov` ✅ | 4,086,374,039,221 | **4.086** | **13.35** | 62.9 GB *(already held as the 161 GB `*.v2ep.pt` cache)* |
| `camera_cross_left_120fov` | 4,610,385,483,349 | 4.610 | 15.06 | **71.0 GB** |
| `camera_cross_right_120fov` | 5,068,459,870,882 | 5.068 | 16.56 | **78.0 GB** |
| `camera_front_tele_30fov` | 3,639,866,288,965 | 3.640 | 11.89 | **56.0 GB** |
| `camera_rear_left_70fov` | 5,029,361,682,098 | 5.029 | 16.43 | **77.4 GB** |
| `camera_rear_right_70fov` | 5,406,859,900,018 | 5.407 | 17.66 | **83.2 GB** |
| `camera_rear_tele_30fov` | 4,099,604,818,752 | 4.100 | 13.39 | **63.1 GB** |
| **all 7 cameras** | 31,940,912,083,285 | **31.94** | 104.33 | 491.7 GB |
| **the 6 we do not read** | 27,854,538,044,064 | **27.85** | **90.98** | **428.8 GB** |
| *(`lidar_top_360fov`, for the §4 comparison)* | 99,601,712,048,740 | **99.60** | 333.9 *(÷298,326 covered clips)* | **1,574 GB** |
| `labels/obstacle.offline` (already read, pod-side) | 158,019,064,340 | 0.158 | 0.52 | 2.4 GB |
| `calibration/sensor_extrinsics` (already read) | 130,504,437 | 0.00013 | 0.0004 | 2.0 MB |

**⭐ The decision this table makes:** **all six unread cameras for the whole B1 training corpus cost
429 GB — 27.2 % of what the LiDAR costs (1,574 GB), and they take coverage from 33.4 % to 100 %.**
The cheapest useful increment, the **cross pair** (the two that close the ±60–127° flanks), is
**149 GB** — under 10 % of the LiDAR bill.

**Coverage of the cameras themselves is not a risk.** All seven cameras are present on **100.00 %**
of the corpus (P-2, `feature_presence.parquet` over 306,152 × 36) and on **500/500** of our own r0
selection (`…/2026-07-26-physicalai-feature-probe/our_corpus_profile.json`,
`our_feature_availability`, every camera `frac: 1.0`) — whereas LiDAR and every `.offline` feature
sit at **97.44 % / 0.99**. **Adding cameras costs no episodes; adding LiDAR costs 2.56 % of them.**

**The rig transform for all seven is already on disk.** `physicalai.py` downloads
`calibration/sensor_extrinsics` per clip and then **filters to the front-wide row**
(`_extrinsics_from_parquet` → `df[name_col] == _FRONT_WIDE_CAM`, `physicalai.py:317-324`). The other
six rows — and the LiDAR row — are in the same 41 KB parquet we already pay for. **Marginal
calibration cost of going multi-camera: zero bytes and one changed filter.**

### 1.4 ⚠️ A correction to a banked number that this decision depends on

The 2026-07-26 census priced every feature as **`chunk_0000` size × 3,146**. For LiDAR that is
almost exact (101.74 TB extrapolated vs **99.60 TB** measured, ×1.02) because a LiDAR spin is a
near-constant number of bytes. **For cameras it is not**, because video bitrate is scene-dependent
and `chunk_0000` is a single ~97-clip draw:

| figure | as banked 2026-07-26 | **MEASURED 2026-09-05** | factor |
|---|---:|---:|---:|
| `camera_front_wide_120fov`, corpus | 6.45 TB | **4.09 TB** | ×1.58 high |
| the 6 unread cameras, corpus | **40.7 TB** | **27.85 TB** | ×1.46 high |
| per clip per camera | **21.1 MB** | **13.35 MB** (front-wide) / **14.90 MB** (7-camera mean) | ×1.58 / ×1.42 high |
| 2,376 clips × 6 cameras | 301 GB | **216 GB** | ×1.39 high |

**Root-cause class:** *a true measurement quoted outside its scope* — the same family as the `df`
/ Thor `free` / cgroup `usage_in_bytes` / `step_s` traps in CLAUDE.md, here as **a one-draw
extrapolation presented as a corpus figure**. It did not change that document's recommendation
(it recommended *against* the cameras either way), but it inflates by ~46 % exactly the number a
"can we afford surround cameras?" decision turns on. **The correction makes the answer easier, not
harder**, which is why it must be stated rather than quietly used.
⇒ **Retraction-log candidate; the Master Mind decides whether it rises to an entry.** The old
figures should not be re-quoted; `raw/camera_rig_probe.json` is the artifact.

### 1.5 What the read-set change actually is

Adding cameras is **a declared change to a pinned test, not a silent edit**. `physicalai.py`'s
read-set is asserted by `stack/tests/test_physicalai_feature_readset.py` at three layers (r0 **2**,
episode build **5**, program-wide **6**), *and* a drift detector fails when a new HF feature path
appears in `physicalai.py` that the test does not declare. A v5a multi-camera ingest therefore
ships with:
* the new feature names added to the test's declared set and the counts updated (the failure
  message already names the documents to fix — that machinery exists precisely because this count
  rotted four times), and
* a named layer, e.g. *"the surround-camera episode build"*, so nobody again writes a bare
  "our ingest reads N".

### 1.7 Where z = 0 is — the metric-scale anchor, MEASURED with a control (P-4)

Monocular depth is scale-free, and the textbook fix is a **known camera height above the road**.
`sensor_extrinsics` puts the front-wide camera at **z = +1.430 m** in the rig frame — but that is
only a *height above the road* if the rig's z = 0 **is** the road surface, and nothing in this repo
had ever asserted it. `code/probe_rig_ground_plane.py` settled it today from one 63.7 MB
`obstacle.offline` chunk (12 clips, **87,481 cuboids**, `reference_frame == "rig"` on 87,481/87,481
rows), using a control whose answer is known in advance: **a ground-standing object's bottom face
must land at z ≈ 0.**

| class | n | median `size_z` | median `center_z` | **median bottom face `center_z − size_z/2`** |
|---|---:|---:|---:|---:|
| `automobile` | 76,574 | 1.587 m | 0.695 m | **−0.125 m** |
| `person` | 7,394 | 1.700 m | 0.807 m | **−0.050 m** |
| `heavy_truck` | 1,426 | 3.480 m | 1.533 m | **−0.122 m** |
| `bus` | 492 | 3.369 m | 1.613 m | **−0.093 m** |
| `other_vehicle` | 184 | 2.202 m | 1.080 m | **−0.055 m** |
| `protruding_object` *(the negative control)* | 176 | 0.543 m | 2.091 m | **+1.676 m** |

**Verdict: z = 0 in the rig frame IS the road plane, to within 0.05–0.13 m** (MEASURED,
`raw/rig_ground_plane_probe.json`). Every class that stands on the road reads ≈ 0; the one class
that must *not* — `protruding_object`, i.e. overhanging signs and branches — reads **+1.68 m**. Two
controls, in opposite directions, both reading their known values. (The p25/p75 spread on cars,
−0.73 / +0.35 m, is road slope and range, not a frame error.)

⇒ **The front-wide camera sits ≈ 1.43–1.56 m above the road** (1.430 m plus a 0.05–0.13 m
ground-plane offset). That is a **metric-scale prior good to ≈ 3.5–9 %** with no LiDAR at all, and
§2.4 shows how `egomotion` speed removes even that residual. The 10-class enum is confirmed
**all-dynamic** (no static/map semantics), consistent with CLAUDE.md.

### 1.6 The three things §1 settles for the rest of this document

1. **We are not at the vision limit; we are at the one-camera limit.** 33.4 % of 100 %.
2. **The surround upgrade is affordable** — 429 GB for the whole B1 corpus, 149 GB for the cross
   pair — and it costs **no episodes** (100 % coverage) and **no new calibration**.
3. **Two of the three DiffusionDrive grounded attentions do not need any of this.** Waypoint-indexed
   sampling (§2.1) and agent tokens (§2.5) run **today, on one camera, at zero new bytes**. The
   ladder in §3 is ordered accordingly.

---

## §2 What each vision-only mechanism buys, priced and ranked

### 2.0 The three budgets every row is priced against

| budget | the anchor | class |
|---|---|---|
| **Thor tact** | **60.3 ms p50 / 63.1 ms p95** end-to-end on Jetson Thor with real step-29999 weights (bf16, dynamic 1..9 engine, batched fan, 60 windows / 12 eps), against a **100 ms** budget | **MEASURED** (`PROGRAM_OVERVIEW.md` §5.0.2) |
| **…of which the encoder** | **≈ 90 % of a REF-C tick** (`refc.py:2238`) ⇒ **≈ 54 ms encoder, ≈ 6 ms everything else** | INHERITED (source comment), and it is the single most consequential number in this section |
| **train** | one v7-tiny rig arm **≈ 29 min on the dev-box 4060 / ≈ 17 min on Thor**; one full-scale arm = one refcv4b-class A40 slot | INHERITED (`REFCV5_DESIGN_PLAN.md` §7) |
| **data** | §1.3: 429 GB for all six extra cameras on B1; 149 GB for the cross pair; 0 GB for everything that runs on the camera we already hold | **MEASURED** today |

⭐ **The encoder share is why the ranking below is what it is.** Anything that adds work to the
**decoder** costs a few percent of 6 ms and is effectively free on Thor. Anything that adds a
**second pass through the vision trunk** costs ~54 ms per extra view and **blows the 100 ms budget at
three cameras** (≈ 172 ms) — unless the extra views are batched into one forward, which on Thor's
20 SMs *may* be nearly free because **throughput is flat from batch 1 to batch 8**. That is a
**HYPOTHESIS**, not a fact: the saturation figure was measured on a *training* step of a *different*
trunk (CLAUDE.md), and quoting it here would be the `step_s` scope error. **WP-V5A-0 measures it**
(§3) — one Thor forward at batch 1 vs 3, ~20 minutes, and it decides whether surround cameras are an
*input* or only a *teacher*.

### 2.1 Waypoint-indexed sampling in perspective view — `H-DDA-1` / arm **E-DDA-1**

| | |
|---|---|
| **What the planner gets** | DiffusionDrive's first grounded attention, without any BEV. Today the 117 anchor queries attend 160 ungrounded PV tokens through a content MHA; the query knows *what* the candidate is only via `traj_proj(x_est)` and the attention **never looks where the candidate goes** (MEASURED, audit §1.3 #19). Projecting each candidate's 8 waypoints into the image and bilinear-sampling the 8 × 20 map makes "is there something on my path?" a **read**, not an inference. |
| **Data cost** | **0 bytes.** `camera_intrinsics` and `sensor_extrinsics` are already in the episode build (`physicalai.py:232-235`), the fan is already in the rig frame, and `bev_raster.readout_column_index(projection="cylindrical")` already maps azimuth → column. |
| **Param cost** | ≈ **0.3 M / decoder layer** (value + output projection at d = 384 plus a point-mixing softmax), **≈ 1.2 M** over 4 layers — **0.5 % of a 263 M arm**. Zero-init gate ⇒ bit-identical at step 0. |
| **Train cost** | 1 tiny-rig arm + its control ≈ **1 h on the 4060**. |
| **Thor cost** | decoder-side ⇒ **≈ 0.2 ms** on the ~6 ms non-encoder share. ESTIMATED. |
| ⚠️ **The projection is not the standard one** | Our frame is **cylindrical**: `φ = atan2(y_cam, z_cam)`, `col = f_ref·φ + W/2` with `f_ref` 305.577; the vertical is tan-based (`calib.py:88-90`, `:151-157`). ⛔ Using a pinhole `K` here is silently wrong — it reads 92.6° for a 120° camera. |
| ⚠️ **The resolution ceiling** | The sampled map is **6.0° per column** (§1.2). At 30 m one column ≈ 3.1 m — **wider than a lane**. Waypoints 1.5 m apart laterally at 30 m land in the *same* column. ⇒ the mechanism grounds **longitudinal** structure well and **lateral** structure poorly at range, and the honest expectation is that it moves LONGITUDINAL first. Sampling the trunk's stride-8 stage (32 × 80, 1.5°/column) is the obvious follow-on and is a **trunk change**, so it is priced separately (§3, WP-V5A-4). |
| **What LiDAR would give that this cannot** | Range. A PV sample tells you *what is along that bearing*, not *how far*. Occlusion is invisible: a candidate passing "through" a parked van samples the van's pixels either way. |
| **Cheapest deciding experiment** | Already pre-registered (`H-DDA-1`). Control = **the same module with random sampling locations at equal parameter count** — it must read the no-information value. Readouts: oracle-in-fan, selection gap, and the LONGITUDINAL family (headway / time-gap / TTC against the `obstacle.offline` replay). |

### 2.2 Surround cameras — the coverage lever, and the reason it is not automatically an input

| | |
|---|---|
| **What the planner gets** | Azimuth coverage **33.4 % → 70.5 % (cross pair) → 100 % (all six)**, MEASURED §1.2. Concretely: an adjacent-lane vehicle at ±90° is **currently unobservable**, and so is anything overtaking from behind. That is not a modelling gap, it is a **sensing** gap, and no amount of architecture recovers it. |
| **Which families** | **LATERAL and TACTICAL, primarily.** A lane change is a decision about a space we cannot see; a merge/yield is a decision about a vehicle in the blind flank. The TACTICAL family's "selected vs executed manoeuvre" confusion is where a blind-zone error shows up, and the STRATEGIC family inherits it. LONGITUDINAL gains least — the lead vehicle is already in the front-wide frustum. |
| **Data cost** | **149 GB** (cross pair) / **429 GB** (all six) for B1, MEASURED §1.3; **no episode loss** (100 % coverage). ⚠️ Whole-chunk pulls would be far worse — B1's clips are spread over an unbanked number of the 3,146 chunks — so the build must use **HTTP range reads of zip members** (`HfFileSystem` + `zipfile`), the same mechanism the LiDAR plan needs. **Validating that on cameras is 3–4× cheaper than validating it on LiDAR, and it de-risks v5b for free.** |
| **Cache cost** | A second `*.v2ep.pt`-shaped cache per camera. Measured anchor: **34.0 MB/episode** for the front-wide PNG cache (CLAUDE.md) ⇒ **≈ 160 GB per extra camera** for B1, ESTIMATED — and the codec must stay **PNG**, because `v2_dataset.py:325` and `slice_v2_cache.py` **refuse to sub-frame a lossy cache**. |
| **Param cost** | **0** if the trunk is shared across views (the standard surround recipe: one trunk, per-view forward, a learned view embedding added to the tokens). +1 small embedding table. |
| **Thor cost** | ⛔ **This is the binding constraint, and it is UNMEASURED.** Three views through the trunk is **≈ 163 ms of encoder** if the cost is linear (⇒ 172 ms tick, **over budget**), or **≈ 54 ms** if Thor's 20 SMs absorb batch 3 the way they absorb batch 8 in training. **WP-V5A-0 settles it in ~20 minutes.** ⛔ Never quote a Thor latency from the A40. |
| ⭐ **The escape hatch if the answer is "too slow"** | **Use the extra cameras as a TRAINING-TIME TEACHER, not an inference input** — exactly the pattern the LiDAR plan proposes (`REFCV5_DESIGN_PLAN.md` §3.4, E-BEV-1). Surround views supervise a **360° occupancy / agent-presence target** that the front-only trunk must predict from its 120° window; at inference the extra cameras are absent and the tick is unchanged. This is strictly weaker than seeing the flank — a model cannot invent an occluded car — but it *is* how the model learns that the flank exists and is uncertain, and it costs **0 ms**. |
| **What LiDAR would give that this cannot** | Nothing about coverage — LiDAR's 360° is matched by the seven cameras. LiDAR adds **range**, not **azimuth**. |
| **Cheapest deciding experiment** | **E-CAM-1**, and it is cheap because it needs **one chunk, not the corpus**: pull the cross pair for ~100 clips (≈ 3 GB), build the sidecar caches, train a tiny-rig arm with a shared trunk + view embedding against the front-only control on the *same windows*, read the four families with the paired bootstrap. Both outcomes committed: separation on LATERAL/TACTICAL ⇒ buy the 149 GB; no separation on 100 clips ⇒ report as underpowered (state `n`), and decide between a bigger pilot and the teacher route. |

### 2.3 Camera→BEV lift (LSS / BEVDet / BEVFormer / BEVDepth / SimpleBEV) — what one 120° camera can and cannot produce

| | |
|---|---|
| **What the planner gets** | A **metric ego-frame feature grid** the fan can index *without any projection* — both are in the rig frame, so waypoint-indexed sampling becomes a direct `grid_sample`, DD's mechanism verbatim. It also gives the trunk a **geometric supervision surface** (occupancy) it has never had. |
| ⛔ **The honest limit of a one-camera lift** | A lift from a single 120° camera produces BEV **only inside that camera's frustum** — i.e. **33.4 % of the circle** (§1.2), and within it only where the scene is actually imaged. **Everything else is UNOBSERVED, and must be represented as unobserved, not as empty.** ⛔ A BEV whose out-of-frustum cells read "free" is the zero-memmap trap in a new costume (CLAUDE.md): a floor arm full of zeros makes every competitor look good. ⇒ **any v5a BEV ships with a `fov_mask` channel and every loss is masked by it.** `bev_raster.py` already computes exactly such a mask (`fov_mask`, `fov_census`) for the existing 60 m × ±16 m @ 0.5 m → [120, 64] agent raster — **that machinery exists and should be reused, not rebuilt.** |
| **vs a surround rig** | The published lifts (LSS, BEVDet, BEVFormer, BEVDepth, SimpleBEV) are all **nuScenes 6-camera surround** methods; their headline numbers are 360° numbers and **are not achievable from one camera** — quoting them for a one-camera arm would be the classic scope error. From one camera the lift is a *frustum* lift. ⇒ **§2.2 and §2.3 are complements, not alternatives**: the lift is worth much more once the cross pair is in. |
| ⚠️ **What our CYLINDRICAL projection changes** | Every one of those implementations assumes a **pinhole** camera and unprojects with `K⁻¹[u, v, 1]`. On our frame that is wrong. The correct unprojection is explicit: for column `u`, `φ = (u − W/2)/f_ref`, ray direction `(cos φ, sin φ, ·)` with the vertical from the tan-based row map. **This is ~10 lines, but it fails silently if the library is used as-is.** ⭐ And cylindrical is *easier* here, not harder: because the column is **linear in azimuth**, each feature column is exactly one constant-width BEV ray fan (0.1875°/px in the frame, 6.0° per trunk column), so a **polar** BEV is a reshape and only the polar→Cartesian resample needs interpolation. A polar-native BEV is the recommended v5a form. |
| **Which lift** | **LSS-style depth-distribution splat** is the right first arm: it is the cheapest, it needs no BEV queries, and its depth distribution is directly supervisable by §2.4's depth or by LiDAR later. **BEVFormer-style deformable BEV queries** need a projection function BEV→image, which we have (`readout_column_index`), but it is more machinery for the same first question. |
| **Param cost** | LSS head over the stride-8 stage (32 × 80) with D = 48 depth bins and C = 64 lifted channels ≈ **0.1–0.5 M**; a small BEV encoder (ResNet-18-class) over the polar grid ≈ **3–5 M**. ESTIMATED. |
| **Thor cost** | The splat is a scatter-add over 32 × 80 × 48 ≈ 123 k points ⇒ ~1–3 ms; the BEV encoder is a **second convolutional trunk** and therefore the risk item — priced only after WP-V5A-0. ESTIMATED. |
| **What LiDAR would give that this cannot** | **Metric depth without a learned prior.** A camera lift *infers* range; the depth distribution is where all the error lives, and BEVDepth's whole contribution is that explicit depth supervision (from LiDAR) is what makes the lift work. Without a depth teacher, a one-camera lift is a learned prior over scene layout — useful, but it will be confidently wrong on unusual geometry, and it has no way to know it. |
| **Cheapest deciding experiment** | **E-BEVA-1**, and it must not start before E-DDA-1 reports. The prerequisite readout is exactly §2.1: **if grounding the attention in PV does not separate, the geometry is not what is missing** and a lift is premature. If it does separate, the lift's own control is a **BEV of the agent raster only** (`bev_raster.py`, zero new data, train-time label) as the target — which measures whether a *metric frame* helps, before spending anything on *depth*. |

### 2.4 Self-supervised monocular depth → pseudo-LiDAR occupancy

| | |
|---|---|
| **What the planner gets** | A per-pixel range map, hence a point cloud, hence exactly the input TransFuser's LiDAR branch consumes — **from the camera we already read**. It also gives §2.3's lift the depth supervision it otherwise lacks. |
| ⭐ **The metric-scale problem, and why our corpus solves it twice over** | Photometric self-supervision (SfMLearner / MonoDepth2 lineage) recovers depth **up to an unknown scale**, and the field's standard workaround — median-scaling against LiDAR at eval — is exactly what a vision-only arm may not do. We have **two independent metric anchors**: (1) **known camera height.** §1.7 MEASURED that rig `z = 0` **is** the road plane to within 0.05–0.13 m, and the front-wide camera sits at **z = +1.430 m** ⇒ an effective height of **1.43–1.56 m**, a **3.5–9 % scale prior** from geometry alone. (2) **measured ego speed.** `egomotion` gives the between-frame translation; the pose network's predicted translation is scaled to match it. This is PackNet-SfM's *velocity supervision* loss, and it is **exact** here rather than approximate, because `v0` at cycle time is a measured, PI-sanctioned input (`velocity-at-cycle-time` ruling, 2026-09-02). **Two anchors that must agree is also the control**: if the height-derived and speed-derived scales disagree by more than a few percent, the depth is wrong and says so. |
| **Data cost** | **0 bytes.** Self-supervision trains on the frames and poses we already hold. |
| **Param cost** | A depth decoder on the existing trunk ≈ **2–4 M**; the pose network is **not needed at inference** (`egomotion` supplies it at train time and the deployed arm reads only `v0`). ESTIMATED. |
| **Thor cost** | Depth decoder only ⇒ **a few ms** if it shares the trunk; **0 ms** if depth is used purely as a *training* signal for the lift (the recommended v5a form). |
| ⛔ **The trap to avoid** | "Pseudo-LiDAR" invites treating an inferred point cloud as a measurement. It is a **learned prior rendered as geometry**, and it inherits every failure of the depth net while *looking* like a sensor. ⇒ v5a uses depth as a **supervision signal for the lift**, and if it is ever consumed as points, the arm carries an explicit uncertainty channel and a **raw-input floor** (the un-lifted PV features) that it must beat (CLAUDE.md probe rule). |
| **What LiDAR would give that this cannot** | Absolute range on **untextured, unusual and dynamic** geometry, at night and in rain, with **no prior at all**. Monocular depth on a moving object is ill-posed even in principle — the photometric loss cannot separate object motion from depth — and that is precisely the lead-vehicle case our LONGITUDINAL family lives on. |
| **Cheapest deciding experiment** | **E-DEPTH-0, and it needs no depth network at all.** `obstacle.offline` gives 3D boxes on 97.44 % of clips. Train a monocular depth head, then score its predicted range **at the box centres** against the label range, and report the error as a function of range — with the two scale anchors as the control (they must agree). If range error at typical lead distances is worse than the headway resolution the planner needs, the mechanism is refused before anything is wired into the decoder. |

### 2.5 Monocular 3D detection → agent tokens — ⭐ the highest-leverage vision-only mechanism

**This is admissible and it is the one that closes DiffusionDrive's second grounded attention.** The
binding rule is *labels may use privileged signals; INFERENCE is vision-only* (PI, 2026-08-03).
`obstacle.offline` carries **3D tracked cuboids on 97.44 % of the corpus** — 10 all-dynamic classes,
`reference_frame == "rig"` on 87,481/87,481 rows MEASURED today (§1.7), ~39 tracks/clip. Those are
**labels**. A detector trained on them and reading **only pixels** at inference produces the 30 agent
queries DD's `cross_agent_attention` consumes, with **no LiDAR and no new bytes**.

| | |
|---|---|
| **What the planner gets** | The attention DD has and we have **zero of**: candidate trajectories attending **objects**, not texture. This is where the LONGITUDINAL family's headway / time-gap / TTC lives, and 92.2 % of refcv3's own deficit is along-track (`D-REFCV3-AXIS1`). It is also the honest input to the TACTICAL family: a yield/merge decision is about *an agent*, not a feature map. |
| **Data cost** | **0 new bytes** — `obstacle.offline` is already read program-wide by the pod-side join (`stack/scripts/build_obstacle_join.py`); the corpus slice for B1 is **2.4 GB** (§1.3) and much of it is already pulled. |
| **Param cost** | DETR-style head: K = 30 queries × d = 384, 2 decoder layers, box + class heads ≈ **2–4 M** (0.8–1.5 % of a 263 M arm), plus one `cross_agent` MHA per decoder layer (**≈ 1.2 M** over 4 layers), zero-init gated. ESTIMATED. |
| **Train cost** | The detector shares the trunk and trains as an **auxiliary head** — no separate training run. Hungarian matching on ≤ 39 boxes/frame is CPU-cheap. |
| **Thor cost** | Decoder-side + a small head ⇒ **≈ 1–2 ms** ESTIMATED, on the ~6 ms non-encoder share. No second trunk. |
| ⛔ **Leak check (I3)** | At inference the tokens come from the **trunk**; `obstacle.offline` is read only at train time. The admissibility question from CLAUDE.md — *"could this input have been computed from the thing we are measuring?"* — is answered no: the labels supervise a head, they are not fed. ⚠️ But the **eval** must not read them either as a planner input; they stay where they already are, in the *reward/metric* path (the lead-vehicle replay), and the `FORBIDDEN_REWARD_INPUTS` audit already covers that surface. |
| **What LiDAR would give that this cannot** | Range accuracy on distant and partially-occluded agents, and detection at night/in rain. Published monocular 3D detection is roughly **half** the accuracy of LiDAR detection on the same benchmarks; the gap is almost entirely **depth**, not classification or azimuth — and azimuth is what a 0.1875°/px cylindrical frame is *good* at. |
| ⭐ **"What accuracy is needed?" — the question is answerable without a detector** | **E-AGT-0, the oracle-ceiling arm, and it must run first.** Wire `cross_agent` and feed it the **ground-truth `obstacle.offline` boxes at inference** — deliberately inadmissible, run **once**, purely as the ceiling. Two controls, both of which must read their known values: (a) **shuffled tokens** (boxes from a random other frame) must read the no-information value; (b) **no-token** = today's arm. Then: **if the ORACLE arm does not separate on the LONGITUDINAL and TACTICAL families, no detector accuracy can help and the mechanism is refused for 0 GPU-days.** If it does separate, the oracle's margin **is** the budget, and the accuracy bar becomes a *derived* quantity: degrade the oracle boxes with calibrated noise (range σ swept 0.5 / 1 / 2 / 4 m, plus a miss rate) and read where the separation dies. **That sweep — not a literature number — is the accuracy specification for the detector**, and it costs a handful of tiny-rig arms. |
| **Deliberate-regression arm** | Feed boxes for the *wrong* frame (temporal shuffle). It must **lose**. If a shuffled agent set helps, the head is reading capacity, not agents. |

### 2.6 Occupancy / freespace prediction, and what else the corpus actually supports

| | |
|---|---|
| **What we can build today, at zero new bytes** | **Agent occupancy** — `stack/tanitad/data/bev_raster.py` already rasterises `obstacle.offline` cuboids into an ego-frame grid (60 m forward × ±16 m at 0.5 m → **[120, 64]**) with a `fov_mask`, a `readout_column_index` and an FOV census. That is a ready-made **BEV occupancy target** for §2.3's lift and for a freespace head, and it needs **no LiDAR and no new ingest**. |
| ⛔ **What we cannot build, and must stop re-asking** | **General occupancy** in the SurroundOcc / TPVFormer / Occ3D sense needs *static* structure — buildings, kerbs, poles, drivable surface. Our only 3D label source is `obstacle.offline`, whose enum is **10 classes, all dynamic agents** (re-confirmed MEASURED today over 87,481 cuboids, §1.7), and PhysicalAI-AV ships **no map, lane graph, junction annotation, traffic-light feature or route/goal signal** — the card says verbatim that open maps data is not included, and `egomotion` carries **no lat/lon**, so OSM map-matching is impossible (CLAUDE.md, five independent probes). ⇒ **in v5a, "occupancy" means agent occupancy plus an explicitly-unobserved mask.** True static occupancy needs either LiDAR (v5b) or an external corpus / AlpaSim's `map.xodr`. |
| **Free 2D evidence nobody has used** | Every camera zip ships a **`blurred_boxes.parquet`** — the anonymisation boxes for faces and plates, at 100 % coverage, **already inside files we download**. Plates are a near-perfect *vehicle-presence* proxy and faces a *pedestrian* proxy; as a weak 2D auxiliary target they cost **0 bytes and 0 extra requests**. Low expected value, but the price is zero and it is a genuinely independent signal. Flagged, not scheduled. |
| **What LiDAR would give** | Static occupancy and drivable-area geometry — the DAC half of a PDMS-style score, which v5a simply cannot compute. §4 states this as a hard boundary. |

### 2.7 The ranking

Ordered by **(value to the four families) ÷ (cost + risk)**, with the binding constraint named:

| # | mechanism | new bytes | Thor risk | starts | expected family | verdict |
|---|---|---:|---|---|---|---|
| **1** | **§2.5 agent tokens**, beginning with the **E-AGT-0 oracle ceiling** | **0** | none (decoder-side) | **today, 0 GPU beyond the rig** | LONGITUDINAL, TACTICAL | ⭐ **do first** — it is the missing DD attention, it is refusable for 0 GPU-days if the oracle does not separate, and it needs no new sensor |
| **2** | **§2.1 waypoint-indexed PV sampling** (`H-DDA-1`, already pre-registered) | **0** | none | **today** | LONGITUDINAL | **do** — grounding without geometry; also the gate on §2.3 |
| **3** | **§2.4 monocular depth**, measured against `obstacle.offline` ranges (**E-DEPTH-0**) | **0** | none (train-time) | **today** | prerequisite for §2.3 | **do** — and the two scale anchors are its own control |
| **4** | **§2.2 surround cameras**, pilot first (**E-CAM-1**, ~3 GB) | 149 GB → 429 GB | ⛔ **UNMEASURED — WP-V5A-0 decides input vs teacher** | after WP-V5A-0 | LATERAL, TACTICAL | **buy the pilot now**; the corpus pull waits on the pilot separating |
| **5** | **§2.3 camera→BEV lift**, polar-native, `fov_mask` mandatory | 0 | second trunk — real | **gated on #2 reporting** | LONGITUDINAL, TACTICAL | **do only if #2 separates**; agent-raster target before any depth target |
| **6** | **§2.6 agent-occupancy head** | 0 | none | with #5 | TACTICAL | bundle with #5 as its supervision target |
| — | general/static occupancy, maps, drivable area | — | — | **never in v5a** | — | ⛔ not supported by this corpus at all (§2.6) |

**Which of DiffusionDrive's three grounded attentions v5a recovers:** **waypoint-indexed spatial —
yes** (#2 in PV now, #5 in BEV later); **agent cross-attention — yes** (#1, from monocular
detection on `obstacle.offline` labels); **ego-query — deliberately refused**, because our FiLM +
`ego_dropout 0.5` + X15 design is the one lever MEASURED to make an arm read the scene (`H-ECHO-8`)
and DD's ego query is an unguarded echo channel. ⇒ **v5a recovers two of three and refuses the
third on evidence. None of them needs LiDAR.**


## §3 The v5a ladder

Every rung is a **one-lever, pre-registered** work package in the `TanitAD_ValidateAIDesign` form:
one variable, everything else held constant; a **deliberate-regression arm**; **controls that must
read known values**; both outcomes committed **before** the run; hyper-parameters fitted on the FIT
split only; four families with the paired episode-cluster bootstrap; T-tier stamped. A rung that
does not separate is a **refusal that saves the next rung's compute** — that is the point of the
ordering, not a consolation.

**Rig cost anchor:** one v7-tiny arm ≈ **29 min on the dev-box 4060 / ≈ 17 min on Thor** (INHERITED,
`REFCV5_DESIGN_PLAN.md` §7). ⚠️ Both are contended by three sibling streams; the ladder is
**serial**, and no arm in this document has been launched.

### 3.0 The ladder at a glance

| WP | arm | one variable | gate on | rig GPU | starts |
|---|---|---|---|---:|---|
| **WP-V5A-0** | `E-THOR-MV` | views batched through the trunk (1 / 2 / 3 / 6) | — | ~0 (inference only, ≈ 20 min Thor) | ⭐ **now, zero training GPU** |
| **WP-V5A-1** | `E-AGT-0` | `cross_agent` present, fed **oracle** boxes | — | 4 arms ≈ **2.0 h** | ⭐ **now** |
| **WP-V5A-2** | `E-AGT-1` | oracle-box **noise σ** (0 / 0.5 / 1 / 2 / 4 m + miss rate) | WP-V5A-1 separates | 6 arms ≈ **2.9 h** | after 1 |
| **WP-V5A-3** | `E-AGT-2` | tokens from a **learned** monocular 3D head vs oracle | WP-V5A-2 says the bar is reachable | 3 arms ≈ **3.0 h** | after 2 |
| **WP-V5A-4** | `E-DDA-1` (= `H-DDA-1`, already pre-registered) | waypoint-indexed sampling on the 8 × 20 PV map | — | 2 arms ≈ **1.0 h** | ⭐ **now** |
| **WP-V5A-5** | `E-DDA-1b` | **which feature stage** is sampled (8 × 20 vs stride-8 32 × 80) | WP-V5A-4 separates | 2 arms ≈ **1.0 h** | after 4 |
| **WP-V5A-6** | `E-DEPTH-0` | monocular depth head present (measured against label ranges) | — | 2 arms ≈ **1.0 h** | ⭐ **now** |
| **WP-V5A-7** | `E-CAM-1` | the **cross pair** present as trunk input (100-clip pilot, ≈ 3 GB) | WP-V5A-0 | 2 arms ≈ **1.0 h** | after 0 |
| **WP-V5A-8** | `E-BEVA-1` | a polar camera→BEV lift with `fov_mask`, agent-raster target | WP-V5A-4 separates | 3 arms ≈ **1.5 h** | after 4 |
| **WP-V5A-9** | corpus pull + cache build (149 GB or 429 GB) | — (engineering, not an arm) | WP-V5A-7 separates | 0 | after 7 |
| **WP-V5A-F** | the **refcv5a full-scale run** — only levers that passed their rung | — | ⛔ **refcv4b's final eval frees the A40** | 1 refcv4b-class slot | after refcv4b |

**Total ladder ≈ 13.4 rig-GPU-hours**, serial, on hardware we already have. ⭐ **Four rungs
(WP-V5A-0, -1, -4, -6) start today at zero pod GPU and zero new bytes**, and between them they
decide the two highest-value mechanisms in §2.

### 3.1 WP-V5A-0 · `E-THOR-MV` — does a second camera cost 54 ms or nothing?

```yaml
hypothesis: H-V5A-THOR          # register row below
one_variable: n_views batched through the vision trunk in a single forward (1, 2, 3, 6)
held_constant: [weights, precision(bf16), input HxW, warmup, batch of the fan, engine]
success: "p50 trunk latency at 3 views <= 1.30x the 1-view p50 => surround cameras are an
          affordable INFERENCE INPUT within the 100 ms tact"
failure: "p50 at 3 views >= 2.0x the 1-view p50 => surround is a TRAINING-TIME TEACHER only;
          WP-V5A-7 changes from an input arm to a distillation arm"
controls:
  - known_value: "1-view end-to-end tick must reproduce 60.3 ms p50 / 63.1 ms p95 within noise
                  (MEASURED, PROGRAM_OVERVIEW 5.0.2); if it does not, the rig is wrong and the
                  sweep is inadmissible"
  - deliberate_regression: "6 views -- must be visibly slower than 3; a flat 1..6 curve means the
                            timer is not measuring the trunk"
```
* **Why it is first:** it is the only *cheap* measurement that changes a *design*, and it is the
  single UNMEASURED number §2.2 hangs on.
* **Instrument rules:** ⛔ on Thor only in-process `torch.cuda.max_memory_allocated()` is admissible
  for memory; latency is wall-clock p50/p95 over ≥ 200 warmed iterations. ⛔ Never quote an A40
  number for this. ⚠️ Check Thor is free of sibling load first.
* **Files:** a standalone probe under `…/2026-09-05-vision-only-maximum/code/`; **no stack change.**

### 3.2 WP-V5A-1 · `E-AGT-0` — the oracle agent-token ceiling ⭐

```yaml
hypothesis: H-V5A-AGT-1
one_variable: cross_agent attention present in every decoder layer, fed K=30 oracle boxes
held_constant: [corpus, seed, steps, batch, window, vocabulary, selection flags, nav wiring]
success: "oracle-token arm beats the no-token control on the LONGITUDINAL family
          (headway / time-gap / TTC vs the obstacle.offline replay) AND on TACTICAL
          (manoeuvre-decision quality), paired episode-cluster bootstrap CI excluding 0"
failure: "no separation on either family => NO detector accuracy can help; the whole agent-token
          mechanism (WP-V5A-2, -3) is REFUSED for zero further GPU-days"
controls:
  - constant_only:  "a fixed learned token set, identical every frame -- must read the
                     no-information value; it measures added capacity, not agents"
  - no_token:       "today's arm, byte-identical (zero-init gate at step 0)"
  - deliberate_regression: "TEMPORALLY SHUFFLED boxes (from a random other frame of the same
                     clip). It must LOSE. If shuffled agents help, the head is reading capacity."
splits: {fit: v7-tiny train, val: carved from FIT, test: the 141 EVAL clips, scored never tuned}
```
* **The design point:** the oracle arm is **deliberately inadmissible as a capability claim** (it
  feeds a label at inference, violating I3) and is run **once, as a ceiling**. That is what makes
  the mechanism refusable **before** a detector exists. It also answers the PI-relevant question
  *"what accuracy do the agent tokens need?"* — see WP-V5A-2, where the bar becomes a **derived**
  number instead of a borrowed literature figure.
* **Files / parameter delta:**
  * `stack/tanitad/refs/refc.py` — `CrossAttnLayer` gains `cross_agent(q, a, a)` behind
    `DecoderConfig.agent_tokens: bool = False`, **zero-init output projection** ⇒ the arm is
    bit-identical to refcv4b at step 0 (pin it with a parity test in the shape of
    `tests/test_refc_v4.py::test_v3_parity`). **+≈ 0.3 M / layer, ≈ +1.2 M over 4** (+0.5 %).
  * `stack/tanitad/refs/refc_v3_train.py` — `V3Dataset.__getitem__` emits
    `item["agent_boxes"] [K, 8]` (centre xyz, size xyz, sin/cos yaw) + `item["agent_mask"] [K]`
    from the existing pod-side join (`stack/scripts/build_obstacle_join.py`).
  * ⛔ **Leak guard:** the loader path that supplies oracle boxes is behind an explicit
    `AGENT_ORACLE=1` env flag that the deployable config can never set, and the flag is asserted
    OFF by a test. `obstacle.offline` stays a train-time label everywhere else.
* **Tiny-rig gate:** `G-DRIVE` only (this is a planner change, not a representation change) — but
  the run still prints `n` and the per-family CIs, and a family with no lead agent in frame is
  reported **per family with its reason and its n**, never silently dropped.

### 3.3 WP-V5A-2 · `E-AGT-1` — the accuracy budget, derived not borrowed

```yaml
hypothesis: H-V5A-AGT-2
one_variable: the range-noise sigma applied to the oracle boxes (0, 0.5, 1, 2, 4 m), plus a
              separate miss-rate sweep (0, 10, 25, 50 %) run as its own one-variable panel
success: "the separation from WP-V5A-1 survives to at least sigma = 1 m => a monocular detector
          in the published accuracy band is enough, and WP-V5A-3 is worth building"
failure: "separation dies below sigma = 0.5 m => the mechanism needs LiDAR-grade range and is a
          v5b item, not a v5a item -- a clean, quantitative v5a/v5b boundary"
controls: [same as WP-V5A-1, plus sigma=0 which must reproduce WP-V5A-1 exactly]
```
⭐ **This rung is the reason §4 can be written honestly**: it converts *"how much would LiDAR
help?"* from an opinion into **the σ at which our own separation dies**, measured on our own corpus,
for the cost of six tiny arms and zero new bytes.

### 3.4 WP-V5A-3 · `E-AGT-2` — the monocular 3D detector head

One variable: **the token source** (learned head vs the WP-V5A-1 oracle vs no-token). DETR-style,
K = 30 queries, Hungarian matching, box + class heads on `obstacle.offline`'s 10 dynamic classes,
trained as an **auxiliary head on the shared trunk** (no separate run). **+2–4 M params.**
Readouts: the four families **and** the detector's own range error vs distance (which feeds §4).
**Deliberate regression:** the detector head trained with **shuffled** labels — its tokens must not
help. **Bar:** the learned arm must land inside the σ budget WP-V5A-2 derived, or the shortfall is
reported **as the v5b case**, not as a failure to be re-tuned.

### 3.5 WP-V5A-4 / -5 · `E-DDA-1`, `E-DDA-1b` — grounded spatial attention

`E-DDA-1` is **already pre-registered as `H-DDA-1`** and is not re-litigated here; §2.1 adds two
implementation facts it must carry: the **cylindrical** projection (`col = f_ref·φ + W/2`, ⛔ never
a pinhole `K`) and the **6.0°/column** resolution ceiling. `E-DDA-1b` is the **separate** arm that
changes only **which feature stage is sampled** (8 × 20 → the stride-8 32 × 80 stage, 1.5°/column);
⛔ it is a distinct arm precisely because bundling it into `E-DDA-1` would make the result
non-attributable — the `--v2` conflation failure again.

### 3.6 WP-V5A-6 · `E-DEPTH-0` — depth measured before it is used

One variable: a self-supervised monocular depth head on the shared trunk. **Its readout is not a
driving metric** — it is **range error at the `obstacle.offline` box centres, as a function of
distance**, which is the number §4 needs. **Two controls, both of which must agree**: the
**camera-height** scale (1.43–1.56 m, §1.7) and the **ego-speed** scale (PackNet-SfM velocity
supervision, exact here because `v0` is measured). **A disagreement of more than a few percent
means the depth is wrong and says so.** ⚠️ A raw-input floor is mandatory: a depth head that does
not beat "predict the corpus-median range for that image row" has added nothing.

### 3.7 WP-V5A-7 / -9 · `E-CAM-1` — the surround pilot, then the corpus

One variable: **the cross pair present as trunk input** (shared trunk + view embedding), on a
**100-clip pilot (≈ 3 GB)** fetched by **HTTP range reads of zip members**, not whole chunks.
⭐ Validating the range-read fetcher on cameras is **3–4× cheaper than on LiDAR and de-risks v5b for
free**. If WP-V5A-0 returned the *linear* answer, this rung inverts into a **distillation** arm: the
cross pair supervises a 360° agent-presence target that the **front-only** trunk must predict, and
the tick is unchanged. **Both variants are pre-registered; WP-V5A-0 selects which one runs.**
⚠️ `n = 100` clips is small — the failure branch reports **underpowered with its n**, never "no
effect" (CLAUDE.md: `n ≪ d` is underpowered *by construction*).

### 3.8 WP-V5A-8 · `E-BEVA-1` — the lift, gated and masked

One variable: an **LSS-style polar-native lift** from the stride-8 stage into an ego-frame grid,
with the fan indexing it directly (no projection — both are rig-frame). ⛔ **Non-negotiable:** an
explicit `fov_mask` channel and **every loss masked by it**, because an out-of-frustum cell is
**UNOBSERVED, not free** — reuse `bev_raster.py`'s existing `fov_mask` / `fov_census`. **Target for
the first arm = the agent raster only** (zero new data), so the arm measures whether a *metric
frame* helps before anything is spent on *depth*. **Deliberate regression:** the same lift with the
`fov_mask` removed (out-of-frustum = free) — it must be **caught**, i.e. score better on a naive
occupancy metric and worse on the families. If the masked and unmasked arms are indistinguishable,
the occupancy readout is not measuring occupancy.

### 3.9 What v5a recovers, and what genuinely needs v5b

| DiffusionDrive's grounded attention | v5a? | how |
|---|---|---|
| **waypoint-indexed spatial** (`GridSampleCrossBEVAttention`) | ✅ **recovered** | WP-V5A-4 in PV today; WP-V5A-8 in a camera-lifted BEV. The *mechanism* is recovered; a LiDAR BEV would improve the *map it samples*, not the attention |
| **agent cross-attention** (30 detection queries) | ✅ **recovered** | WP-V5A-1 → -3: `obstacle.offline` as **labels**, a monocular detector at inference. This is the one the audit called MISSING and it needs **no new sensor** |
| **ego-query attention** | ⛔ **deliberately refused** | our FiLM + `ego_dropout 0.5` + X15 is the one design MEASURED to make an arm read the scene (`H-ECHO-8`); DD's ego query is an unguarded echo channel |
| *(their LiDAR BEV encoder branch)* | ❌ | **v5b** — and §4 prices what it actually buys |

---

## §4 The v5a/v5b boundary, stated as a decision

### 4.1 What LiDAR adds **on top of a maximised v5a** — four things, and only four

| # | what v5b adds | how big, honestly | class |
|---|---|---|---|
| **1** | **Metric range without a learned prior.** Everything else in this table is downstream of this one | ⭐ **This is the whole case**, and §3 makes it a *measured* quantity rather than an argument: `E-DEPTH-0` reports our monocular range error vs distance, and `E-AGT-1` reports the σ at which our own separation dies. **v5a produces the number that prices v5b.** | the decision |
| **2** | **The unknown-object class.** LiDAR returns points regardless of semantics; a detector trained on 10 dynamic classes cannot flag an object it has no class for | ⭐ **The strongest safety argument for v5b**, and it is *structural* — no amount of v5a training removes it. It is also the hardest to quantify on this corpus, because the labels themselves are the 10 classes | safety |
| **3** | **Night and adverse weather.** MEASURED: **15.41 % of the corpus is night** (14.4 % of our own r0 selection) — `our_corpus_profile.json` | a real, bounded slice; the right readout is a **night-stratified** four-family table on any v5a arm, which costs nothing extra | bounded |
| **4** | **Static geometry** (kerbs, poles, structures) | ⚠️ **Much less than it sounds.** LiDAR gives *occupied volume*, **not lane semantics**: PhysicalAI-AV has **no map, lane graph, junction annotation or traffic-light feature**, and `egomotion` carries no lat/lon, so even *with* LiDAR we cannot compute a NAVSIM-style **DAC**. ⛔ **The PI should not expect drivable-area compliance from v5b** | over-claimed elsewhere |

### 4.2 What v5a already closes — the honest expectation

* **Coverage: fully.** 7 cameras reach **100 % azimuth** (MEASURED §1.2). LiDAR adds **no azimuth**.
* **The two missing DD attentions: fully.** The audit's finding was *"we have 0 of the 3 grounded
  attentions"* (`D-REFC-DDAUDIT-4`); v5a recovers **2** and refuses the 3rd on evidence (§3.9).
  **None of that recovery needs LiDAR.**
* **The lead vehicle — probably most of it.** Our measured deficit is **92.2 % along-track**
  (`D-REFCV3-AXIS1`), and the LONGITUDINAL family is dominated by the **lead vehicle**, which is
  **close, large and centred** — the regime where monocular range is *most* accurate. ⇒ the honest
  expectation is that **v5b's marginal value concentrates on far, lateral, occluded and night
  agents**, not on the headway signal we most need. ⚠️ **HYPOTHESIS, and `E-DEPTH-0` refutes or
  supports it for the price of two tiny arms.**
* **Occlusion: not at all, for either release.** LiDAR does not see through a van. A frequent
  over-estimate of the LiDAR gain comes from crediting it with occlusion reasoning it does not have.

### 4.3 What v5a can **never** do

1. Give **range on an object it has no prior for** — the monocular failure mode is *confident* and
   *silent*, which is worse than a gap.
2. Produce a **measurement** rather than an **inference**. There is no physically-grounded
   uncertainty; a v5a depth or box carries a learned confidence, and a learned confidence is exactly
   what a distribution shift breaks.
3. See in **darkness beyond the headlights**, or through heavy rain/fog, at the same accuracy.
4. Provide **any** static/structural occupancy — no LiDAR, no map, no static labels (§2.6).
5. Detect the **unclassed obstacle** (4.1 #2).

### 4.4 The decision, stated so the PI can take it

> **Run v5a to completion first — and let v5a produce the number that prices v5b.**

| | v5a (pure vision) | v5b (LiDAR on top) |
|---|---|---|
| **new bytes** | **0** for four of the six mechanisms; **149 GB** (cross pair) or **429 GB** (all six) if `E-CAM-1` separates — **MEASURED** | **≈ 1.57 TB** transit for B1 (MEASURED-derived from the 99.60 TB corpus total ÷ 298,326 covered clips), plus a **Draco decode + rasterise + sidecar** pipeline, plus **14–38 GB** of PNG sidecar (ESTIMATED, `REFCV5_DESIGN_PLAN.md` §3.3) |
| **episodes lost** | **0** — all 7 cameras are at **100.00 %** coverage | **2.56 %** — LiDAR is 97.44 % |
| **new engineering** | a `cross_agent` MHA, a grid-sample module, an LSS lift, a view embedding — all inside `refc.py` / the loader | a new codec path, a new cache format, a streamed corpus build on a pod whose quota `df` cannot see, **and a doctrine ruling** on whether LiDAR may be an *inference* input at all (`D-REFCV5-PLAN-7`, unresolved) |
| **rig GPU** | **≈ 13.4 h**, serial, on hardware we own | on top of v5a's, not instead of it |
| **buys** | 2 of DD's 3 grounded attentions · 33.4 % → 100 % azimuth · a metric BEV · agent tokens | **range accuracy at distance/night** and **the unclassed object** |
| **risk** | the Thor tick (WP-V5A-0 settles it in 20 min) | 1.57 TB moved before the first arm can say whether it was worth it |

**The rule that makes this a decision rather than a preference:** every v5b argument reduces to
*"how much range accuracy are we missing?"*, and **two v5a rungs measure exactly that at zero new
bytes** — `E-DEPTH-0` (range error vs distance against the label boxes) and `E-AGT-1` (the σ at
which our own separation dies). ⇒ **v5b is scheduled if and only if `E-AGT-1`'s σ bar lands below
what `E-DEPTH-0`/`E-AGT-2` achieve.** If they land above it, LiDAR is a *safety* investment
(4.1 #2) to be argued on its own terms, not a *performance* one — and that is a much better
conversation to have with a number in hand.

⚠️ **One thing v5a should do for v5b regardless of the verdict:** WP-V5A-7 validates the
**HTTP-range zip-member fetcher** on cameras. That mechanism is a HYPOTHESIS in the LiDAR plan
(`REFCV5_DESIGN_PLAN.md` §3.1) and it is on v5b's critical path. Proving it on 3 GB of camera is
strictly cheaper than discovering it fails on 1.57 TB of LiDAR.


## §5 Deliverable manifest, primaries to bank, integration escalations

*(to be filled)*
