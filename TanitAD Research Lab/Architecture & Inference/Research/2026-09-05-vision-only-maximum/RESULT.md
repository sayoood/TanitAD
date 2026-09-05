# What is maximally achievable with cameras alone — the v5a (pure-vision) study

**status: IN PROGRESS — done: §0 reading guide, §1 camera inventory (three probes, two of them run today) / next: §2 mechanism ladder, §3 v5a work packages, §4 the v5a/v5b boundary, §5 manifest + register rows**

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

*(to be filled)*

## §3 The v5a ladder

*(to be filled)*

## §4 The v5a/v5b boundary, stated as a decision

*(to be filled)*

## §5 Deliverable manifest, primaries to bank, integration escalations

*(to be filled)*
