# Qwen-Drive usage review: the model was fine. Our input packing and my renderer were not.

**Master Mind, 2026-09-13.** Triggered by the PI's rejection of the 2026-09-11 validation video:
*"the video content is not correct, the occupancy map has no details, the boxes not correct, the map
small and noisy, review the complete usage of qwen drive including their github repo"*.

⛔⛔ **PERCEPTION TEACHER ONLY — never Qwen-Drive's trajectories** (PI directive 2026-09-11; the
planner weights are not even downloaded).
⚠️ **Tier:** none of the numbers below is a driving tier (T0/T1/T2). They are **perception-teacher
validation** readings against packed ground truth, on small frame sets, with no interval claimed.

---

## 0. Headline

1. ⭐ **Qwen-Drive-1.0-4B works on Thor.** Its own demo, through its own runner and its own
   visualizer, finds **54 / 57** GT boxes over the 4 nuPlan demo frames, map mIoU **0.799**,
   occupancy geometric IoU **0.389** (MEASURED, `raw/demo_control_metrics.json`, arm `A0`). The CUDA
   kernels compiled (`qwen_drive_ms_deform_attn_bf16.so`, `qwen_drive_voxel_pool.so` present), and
   the kernel loader has **no fallback path** — it compiles or raises (`src/qwen_drive_perception/ops/__init__.py`).
2. ⛔ **Our 2026-09-11 packing put every camera outside the geometry the weights were trained on.**
   Upstream says so in two places (§2). The decisive lever is **image scale (focal length)**, proven
   on the demo, where ground truth exists: changing **only** the focal collapses detection from
   **54/57 to 0/57** (A2) and to **2/57** with no black pixels at all (A6) (§3).
3. ⭐ **v2 packing** — a virtual nuPlan 8-camera rig synthesised from the PhysicalAI f-theta cameras by
   exact reprojection — **more than doubles recall on OUR data at identical instants**: **0.182 → 0.404**,
   precision **0.250 → 0.429**, front-sector recall **0.048 → 0.444** (MEASURED, 14 frames, 225 GT
   boxes ≤ 50 m, `raw/ours_metrics.json`) (§6).
4. ⚠️ **Honest attribution of the three complaints** (§1): *"occupancy has no details"* and *"map
   small and noisy"* were **mostly my renderer**, not the model — decoded by the upstream visualizer,
   even the v1 input shows road structure. *"Boxes not correct"* was **both**: my first renderer drew
   every box rotated 90°, **and** the v1 packing really does miss most boxes.
5. ⚠️ **v2 is a large fix, not a finished teacher.** Recall 0.40 on our GT is far below the demo's 0.95.
   Whether that is acceptable for augmentation is a **PI decision** (§9), and what would raise it is
   named there.

---

## 1. What the PI saw, and what actually caused it

| PI complaint | root cause | evidence | class |
|---|---|---|---|
| *"the occupancy map has no details"* | ⛔ **my renderer.** Occupancy is a **200×200×16 semantic** grid over 10 classes with `empty` = label 9. I drew `occ.max(axis=2)` through a continuous colormap, i.e. a near-uniform field of 9s. Upstream draws non-empty voxels in 3D, coloured by class, hiding background voxels over the road. | `docs/perception.md:8`; `configuration_perception.py:38` (`OCC_CLASS_NAMES`, empty is index 9); `visualize.py:285-319`; `media/v1_vs_v2_4fbd97b6a4b7_f0.jpg` left half: the **v1** input, decoded upstream, shows a road-shaped occupancy | output decoded from a guessed schema |
| *"the map small and noisy"* | ⚠️ **small is by design, noisy was my renderer.** The map is **60 m × 30 m at 0.15 m** (400×200 cells), a stored **(Y, X)** raster. I drew it un-transposed (sideways) through `magma`. Upstream transposes and flips so ego-forward is up, with a 6-class palette. | `docs/perception.md:9`; `configuration_perception.py:95-96`; `visualize.py:322-325` | output decoded from a guessed schema |
| *"the boxes not correct"* | ⛔ **both.** (a) my first renderer passed column 3 as width — upstream: *"`w` along the heading direction"* — drawing every box across the road (fixed 2026-09-12); (b) the v1 packing, decoded correctly, scores **recall 0.182 / precision 0.250** against our GT, vs **0.404 / 0.429** for v2 on the same instants. | `docs/perception.md:96-97`; `geometry.py:106-114`; `raw/ours_metrics.json` | teacher judged on inputs outside its trained geometry |

⭐ **The upstream visualizer existed in the repo the whole time** (`scripts/visualize_perception.py`,
`src/qwen_drive_perception/visualize.py`, 17,442 B). Writing my own decoder instead of running
theirs is the root cause of two of the three complaints.

---

## 2. What the upstream repo binds (PUBLISHED, `https://github.com/QwenLM/Qwen-Drive-1.0` cloned on Thor, HEAD `28091c1`, committed 2026-09-03)

| fact | source |
|---|---|
| *"trained on nuScenes (6 cameras) and OpenScene/nuPlan (8 cameras) at a fixed 896x512 input resolution with fixed camera configurations. A different camera layout or resolution is not covered by the released weights and will likely degrade results."* | `docs/perception.md:110-112`; restated `configuration_perception.py:17-20` |
| detection: ≤ 300 boxes, 7 classes, **lidar frame**, `[x, y, z, w, l, h, yaw, vx, vy]`, z at box **bottom**, `w` along heading; packed GT boxes are in the **ego** frame | `docs/perception.md:6, 96-104`; `geometry.py:24-25, 106-114` |
| occupancy: 200×200×16 semantic, 10 classes, ego frame, axes `[X, Y, Z]`; nuPlan range x,y ±50 m, z −4…+4 m at 0.5 m | `docs/perception.md:8, 100-101`; `configuration_perception.py:92-93` |
| map: 6 classes, x −30…+30, y −15…+15 at 0.15 m → stored (200, 400) | `configuration_perception.py:41-48, 95-96` |
| prompt: ordered `(view tag, image)` pairs + `"Analyze the current driving scene."`; nuPlan tags `<FRONT VIEW>` … `<FRONT LEFT VIEW>` in `cam_order` `F0,R0,R1,R2,B0,L2,L1,L0` | `data/demo/perception/*/frame.json`; `dataset.py:161-200` |
| `lidar.npy` is for **visualization only** — the model never reads it | `dataset.py:24, 80` |
| ⚠️ **PhysicalAI appears upstream only in PLANNING**: three RAW f-theta cameras (`camera_front_wide_120fov`, `camera_cross_left_120fov`, `camera_cross_right_120fov`), every video frame decoded to JPEG with no rectification. ⇒ the VLM has seen raw PhysicalAI imagery, but the **perception head was never trained or evaluated on PhysicalAI** and upstream ships **no PhysicalAI perception recipe** — any packing for our corpus is ours to design and to validate | `docs/data.md:121-142, 189-191`; `docs/perception.md:110-112` |
| runner and visualizer to use: `scripts/run_perception.py` (default `flash_attention_2`; `sdpa` is a supported choice), `scripts/visualize_perception.py` (threshold 0.25) | the scripts themselves |

**The nuPlan rig the weights know** (MEASURED from the demo's own `calib.npz`, `raw/demo_rig_probe.txt`):
all 8 cameras of all 4 nuPlan frames carry **K = [[1545, 0, 960], [0, 1545, 560]] at 1920×1080**
⇒ at model resolution **fx 721.0 / fy 732.4, principal point (448, 265.5), HFOV 63.7°**; yaws
**0, ±55, ±111, ±141, 180**; pitch 0–2.4° down; mounting height 1.37–1.54 m above the ego origin;
`lidar2ego` = identity. The demo LiDAR's ground mode sits at **z = −0.325 m in 4/4 nuPlan frames**
(`raw/demo_lidar_ground_height.txt`), i.e. **nuPlan's ego origin is 0.325 m above the road**.

**What we fed on 2026-09-11** (MEASURED, `raw/v1_packing_rig_probe.txt`):

| | nuPlan rig (weights) | v1 PhysicalAI packing |
|---|---|---|
| cameras | 8 | **6** (R1, L1 absent) |
| fx @896 | 721.0, all views | **313.7** (F0, R0, L0) / **664.2** (R2, L2) / **1732.3** (B0) |
| HFOV | 63.7°, all views | **110° / 68° / 29°** |
| principal point v0 @512 | 265.5 | up to **383.7** |
| yaws | 0, ±55, ±111, ±141, 180 | 0, ±67, ±152, 179 |
| ego origin | 0.325 m above road | on the road |

---

## 3. ⭐ The discriminating control — degrade THEIR demo one property at a time

Four nuPlan demo frames, 57 GT boxes ≤ 51.2 m. Every variant rewrites `calib.npz` to match its
resampled images, so geometry stays exact and any drop is the model's sensitivity, not a projection
bug. Detection: class-agnostic BEV centre match ≤ 2 m, score ≥ 0.25 (MEASURED, `raw/demo_control_metrics.json`).

| arm | the one change | P | R | TP/FP/FN | map mIoU | occ geo IoU | occ mIoU |
|---|---|---|---|---|---|---|---|
| **A0** | demo as shipped | 0.730 | **0.947** | 54/20/3 | **0.799** | 0.389 | 0.398 |
| A0r | ⭐ **no-op control**: resampled through our remap, same K | 0.711 | 0.947 | 54/22/3 | 0.796 | 0.382 | 0.373 |
| A1 | CAM_R1 + CAM_L1 dropped (6 cameras, as v1) | 0.740 | 0.947 | 54/19/3 | 0.791 | 0.376 | 0.377 |
| A2 | focal → 313.7 px (v1's 110°) — ⚠️ also 81 % black | 0.000 | **0.000** | 0/18/57 | 0.169 | 0.0006 | 0.0008 |
| A3 | A2 + principal point → v0 383.7 | 0.045 | 0.018 | 1/21/56 | 0.253 | 0.006 | 0.002 |
| A4 | ⭐ **81 % black at the CORRECT scale** (separates blackness) | 0.613 | 0.667 | 38/24/19 | 0.480 | 0.290 | 0.132 |
| A5 | only CAM_B0 81 % black (**the v2 design**) | 0.684 | 0.912 | 52/24/5 | 0.782 | 0.368 | 0.331 |
| A6 | ⭐ **zoom ×2.40 (v1's B0 focal), NO black pixels** (separates scale) | 0.041 | **0.035** | 2/47/55 | 0.163 | 0.092 | 0.025 |

Reading, in the order the arms were designed:

* **A0r ≈ A0** ⇒ the resampling path is faithful; the collapses below are the model.
* **A1 ≈ A0** ⇒ the missing two cameras were **not** the cause.
* ⚠️ **A2 alone did not prove scale** — a 63.7° source cannot fill a 110° canvas, so A2 was also
  81 % black, a side effect the v1 packing never had. *Caught before reporting; same family as the
  2026-09-11 rule "match the intervention's SIDE EFFECT".* A4 and A6 were added to separate the two.
* **A6 collapses with zero black pixels; A4 survives at 0.667 recall** ⇒ ⭐ **image SCALE is the
  dominant lever.** Blackness costs something (A4) but does not collapse the model.
* **A5 costs 2/57 recall and −0.067 occupancy mIoU** ⇒ a mostly-black B0 is an acceptable price for
  a view no PhysicalAI camera can fill (§4).
* The 2 nuScenes demo frames score 26/14/5 and 30/21/3 at A0, map 0.839 / 0.843 — the other
  trained rig also works.

⚠️ n = 4 frames. The effects this panel decides are 54 → 0 and 54 → 2 of 57; nothing smaller is read
off it.

---

## 4. v2 packing — a virtual nuPlan rig (`code/build_frames_v2.py`)

Every output view is a pinhole camera with **nuPlan's exact K at 1920×1080**, a nuPlan-like yaw,
1° down-pitch and zero roll, synthesised from **one** PhysicalAI f-theta camera **through that camera's
own optical centre**. A rotation about the optical centre is an exact reprojection at every depth — no
parallax, no seams — and the calibration written is the virtual camera's, so `lidar2img` is exact.
`lidar` frame := PhysicalAI rig; `lidar2ego` = translate(0, 0, −0.325) so the road sits where nuPlan's
does.

| view | source | yaw used (nuPlan) | observed |
|---|---|---|---|
| F0 | front_wide_120 | 0 (0) | 1.000 |
| R0 / L0 | cross_right / cross_left_120 | ∓55 (∓55) | 0.999–1.000 |
| R1 / L1 | cross_right / cross_left_120 | **∓95** (∓111) | 0.970–0.985 |
| R2 / L2 | rear_right / rear_left_70 | **∓149** (∓141) | 0.973–1.000 |
| B0 | rear_tele_30 | 180 (180) | **0.181–0.183** (rest black, recorded) |

Yaw is chosen per clip **from calibration only**: nuPlan's own yaw if ≥ 97 % of its pixels are real,
otherwise rotated toward the source until they are. B0 is never filled from another camera: the two
rear-side cameras are **2.0 m** apart, which would put a parallax seam straight behind the car. A5 (§3)
prices the black B0 on the demo.

⛔ **What v2 cannot fix:** mounting **height** (cross cameras at ~0.81 m vs nuPlan ~1.5 m — needs
depth), the **domain** (nuPlan cities vs PhysicalAI's 25 countries), and B0's coverage.

### 4.1 Geometry verified BEFORE any GPU (`code/gt_projection_check.py`)
Our `obstacle.offline` boxes, projected through **Qwen-Drive's own** `build_lidar2img` /
`box_corners` / `project_to_image`, sit tightly on the white van beside the car (R2, B0), the dark
sedan (L1, L0), and the roofs of cars parked behind a wall (R1). Boxes come from the label parquet,
pixels from the video, and only the builder's `calib.npz` joins them — so intrinsics, extrinsics, the
virtual rotation, the z-shift and time alignment are checked at once. The LiDAR's own ground mode reads
**z = +0.025 m in the rig frame** (`meta.json`), confirming the rig origin is on the road.

---

## 5. Where the numbers on our data come from

* GT = PhysicalAI `obstacle.offline` cuboids, per-track nearest within 60 ms, mapped to the 7-class
  taxonomy (`CLASS_MAP` in the builder); only boxes ≤ 50 m are scored.
* ⚠️ obstacle.offline includes **occluded** tracks no camera can see. The **LiDAR-supported** subset
  (≥ 3 points of the frame's 100 k-point subsample inside the box) is reported beside the full set
  where LiDAR exists (4 of 6 clips).
* **No semantic occupancy or map GT exists for PhysicalAI.** The "occupancy ground truth" panel in
  our renders is a **LiDAR pseudo-occupancy** (ground drawn as driveable, agents from the boxes,
  everything else background) — a visual reference, **never scored as GT**. The map GT panel is empty.

---

## 6. Our data — v1 vs v2 at identical instants, and full-clip sequences

**Legacy set** — the exact 14 instants of the rejected 2026-09-11 sample (6 clips), both packings
scored against the **identical** GT set (MEASURED, `raw/ours_metrics.json`):

| set | frames | GT ≤ 50 m | P | R @2 m | R @4 m | R vehicle | R VRU | R front / side / rear | R 0–20 / 20–35 / 35–50 m | R LiDAR-supported | occupancy vs LiDAR, BEV IoU any / obstacle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v1 packing** (rejected) | 14 | 225 | 0.250 | **0.182** | 0.267 | 0.218 | 0.061 | 0.048 / 0.106 / 0.412 | 0.292 / 0.215 / 0.102 | 0.136 | 0.370 / 0.119 |
| **v2 packing** | 14 | 225 | 0.429 | **0.404** | 0.507 | 0.454 | 0.245 | 0.444 / 0.277 / 0.544 | 0.646 / 0.430 / 0.265 | 0.413 | 0.477 / 0.150 |

⚠️ The occupancy column compares against a **single LiDAR sweep** labelled ground/background/boxes — a geometric agreement, not an occupancy score; BEV columns only, so v2's 0.325 m z-shift cannot be scored as a difference.

**Sequences** — 5 Hz over whole clips, v2 packing (MEASURED, `raw/ours_metrics.json`):

| set | frames | GT ≤ 50 m | P | R @2 m | R @4 m | R vehicle | R VRU | R front / side / rear | R 0–20 / 20–35 / 35–50 m | R LiDAR-supported | occupancy vs LiDAR, BEV IoU any / obstacle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `4fbd97b6a4b7` dense junction, daylight | 96 | 2236 | 0.516 | **0.382** | 0.453 | 0.371 | 0.865 | 0.628 / 0.268 / 0.511 | 0.538 / 0.255 / 0.455 | 0.391 | 0.590 / 0.109 |
| `73495082f98b` VRU-dense, night | 96 | 3141 | 0.482 | **0.477** | 0.608 | 0.554 | 0.283 | 0.518 / 0.441 / 0.472 | 0.671 / 0.568 / 0.305 | 0.509 | 0.560 / 0.189 |
| `0d90d20036a3` night turn onto a ramp, VRUs | 96 | 1318 | 0.507 | **0.403** | 0.508 | 0.458 | 0.331 | 0.345 / 0.453 / 0.403 | 0.748 / 0.540 / 0.112 | 0.415 | 0.514 / 0.168 |
| `6924358fafe0` near-empty mountain road (control) | 96 | 53 | 0.589 | **0.811** | 0.981 | 0.811 | 0.000 | 1.000 / 1.000 / 0.444 | 0.952 / 0.333 / 0.200 | 0.811 | 0.157 / 0.124 |

⚠️ The control clip was chosen for having one GT track over the whole clip (INHERITED, the 2026-09-11 `sample_plan.json` note); its 53 in-range boxes are that track's frames within 50 m. its precision **0.589** is the hallucination rate that matters most (≈ 0.3 false boxes per frame on an almost empty road). Its occupancy agreement is low (0.157) because LiDAR returns dense roadside vegetation the camera head leaves empty.


### 6.1 Score threshold — selected on FIT, reported on held-out clips (`raw/pr_sweep.json`)

FIT = frames_legacy, seq_4fbd97b6a4b7, seq_73495082f98b (206 frames); SCORED = seq_0d90d20036a3, seq_6924358fafe0 (192 frames).

| threshold | FIT P | FIT R | FIT F1 | SCORED P | SCORED R | SCORED F1 |
|---|---|---|---|---|---|---|
| 0.05 | 0.091 | 0.725 | 0.162 | 0.053 | 0.794 | 0.099 |
| 0.10 | 0.127 | 0.674 | 0.213 | 0.124 | 0.657 | 0.208 |
| 0.15 | 0.231 | 0.575 | 0.329 | 0.228 | 0.551 | 0.322 |
| 0.20 | 0.350 | 0.500 | 0.412 | 0.359 | 0.478 | 0.410 |
| 0.25 (upstream) | 0.491 | 0.436 | 0.462 | 0.512 | 0.419 | 0.461 |
| 0.30 ⭐ selected | 0.633 | 0.375 | 0.471 | 0.657 | 0.375 | 0.477 |
| 0.35 | 0.723 | 0.309 | 0.433 | 0.749 | 0.339 | 0.467 |
| 0.40 | 0.762 | 0.277 | 0.406 | 0.779 | 0.306 | 0.439 |
| 0.50 | 0.799 | 0.237 | 0.365 | 0.813 | 0.270 | 0.405 |

### 6.2 Temporal consistency — selected on FIT, reported on held-out clips (`raw/temporal_filter.json`)

Rule selected on FIT: score ≥ **0.25**, confirmed by **1** neighbouring frame(s) within **1.5 m** after ego-motion compensation and propagation by the predicted velocity.

| held-out clips | P | R | F1 | TP / predictions / GT |
|---|---|---|---|---|
| upstream threshold 0.25, single frame | 0.512 | 0.419 | 0.461 | 574 / 1121 / 1371 |
| **selected temporal rule** | 0.575 | 0.393 | 0.467 | 539 / 937 / 1371 |



### 6.3 Verdict on the two 0-GPU levers

* ⛔ **Neither moves the teacher materially.** On the held-out clips the selected threshold (0.30) changes F1
  **0.461 → 0.477** and the temporal rule **0.461 → 0.467** — trade-offs between precision and recall, not
  gains in both, on **2 clips with no replicate and no interval**. Both are reported as measured; neither is
  claimed as an improvement.
* ⭐ **What the sweep does settle:** the misses are **not** a thresholding artefact. Recall reaches **0.794**
  only at score 0.05, where precision is **0.053**. And recall at a 4 m match (**0.45–0.61** on the three
  real clips) against 2 m (**0.38–0.48**) says a large share of the "misses" are **localisation (depth)
  errors of 2–4 m**, not absent detections.
* ⇒ **The next detection lever is not a filter on the teacher's output but a change to the teacher**: a
  head fine-tune on PhysicalAI against `obstacle.offline` + LiDAR, which this corpus uniquely allows. That
  is a compute decision, and — because `obstacle.offline` already supplies better boxes than the teacher —
  it only pays if the **map / occupancy** layers need it too. ⇒ **measure those first** (§9, lever 4).

---

## 7. Media

All renders are produced by **Qwen-Drive's own `visualize.render_frame`**, unmodified; our script only
adds a banner naming each panel and paints the frame token out (clip ids are gated-confidential — only
`sha256(clip_id)[:12]` appears).

| file | what it shows |
|---|---|
| `media/demo_A0_as_shipped_vs_A2_focal_changed.jpg` | Qwen's demo frame as shipped (54/57 over 4 frames) beside the same frame with only the focal changed (0/57) |
| `media/v1_vs_v2_4fbd97b6a4b7_f0.jpg` | the rejected v1 packing and v2 at the **same instant**, both decoded upstream |
| `media/gt_projection_check_4fbd97b6a4b7_f0.jpg` | our obstacle.offline boxes projected into the 8 v2 views through upstream geometry — the pre-GPU check |
| `media/v2_6924358fafe0_f0.jpg` | the near-empty control clip: the passing van is found; one pedestrian hallucination on the hillside |
| `media/v2_2bb37d62b419_f1.jpg` | a night/snow instant |
| `media/v2_seq_4fbd97b6a4b7_t12.0s.jpg` | a still from the dense-junction sequence |
| ⚠️ `seq_*.mp4` (4 clips, 96 frames each at 5 Hz) | **not in git** (`*.mp4` is git-ignored): `devbox: C:\Users\Admin\AppData\Local\Temp\claude\…\scratchpad\v2\` and copied to this package's `media/` on D: (untracked); delivered to the PI in-session |

---

## 8. What this does NOT establish

* ⛔ **Not a benchmark.** 14 + ~380 frames from 6 clips of one chunk; no interval is claimed.
* ⛔ **Occupancy and map quality on PhysicalAI are unscored** — there is no semantic GT. v1 and v2
  maps genuinely differ (at instant `4fbd97b6a4b7_f0`: crosswalk cells **355 vs 3,052**, centroids
  16.4 m ahead vs 2.9 m behind — MEASURED from the two `.npz`), but **which is right is not
  measurable here**, and no v1→v2 improvement is claimed for occupancy or map. ⚠️ *A draft of this
  bullet said "v1's map has no crosswalk"; the count refuted it before it was published.*
* ⛔ **Nothing here says Qwen-Drive labels will improve a TanitAD model.** That needs its own
  pre-registered arm.

---

## 9. Next levers (RULE ZERO) and the decisions that are the PI's

**Done when:** the PI validates the v2 output — that gate is his, by his own instruction (*"Show the
visualization of a small sample to validate the output"*). Everything below is ready and waits only on it.

**Decisions (PI_DECISION_QUEUE item 14):** (a) does v2 pass visual validation; (b) augmentation scope —
**7.13 GB of camera data per 100 clips** and **~4.1 s/frame on Thor** (both MEASURED), so B1-EVAL at 2 Hz is
≈ 9.9 GB + 6.3 h and parity ≈ 170 GB + 108 h (ESTIMATED by scaling); (c) which layers to keep.

**Levers that would raise the teacher, ranked by measured or expected effect, cheapest first:**

1. ⭐ **Score threshold on our domain** (0 GPU): the upstream 0.25 was set on nuPlan/nuScenes. Swept with the
   threshold **selected on FIT sets only and reported on held-out clips** — result in §6.
2. **Temporal consistency** (0 GPU): the model is single-frame; at 5 Hz a detection that survives ≥ 2 of 3
   ego-motion-compensated frames is far likelier real. Needs the egomotion join for these clips.
3. **LiDAR-agreement filter** for any label that will train a model: keep teacher boxes and occupancy only
   where the sweep supports them. It converts the teacher's weakest property (precision ~0.5) into a gate.
4. **Map validation without map GT** (0 GPU): predicted `driveable_surface` against LiDAR ground returns,
   `road_edge` against curb height steps. Required before the map layer trains anything.
5. ⚠️ **Side views** (recall weakest): R1/L1 come from bumper-height cross cameras at ±95°. Height cannot be
   changed without depth; a B0/side composite A/B on the demo (where GT exists) would price a seam
   before any is built.

⛔ **Not levers:** feeding more cameras (A1 shows the count does not matter), or any change to the
upstream decoder (the renders are theirs precisely so the decoding cannot be doubted again).

---

## 10. Retraction classes this earned

Logged in `Project Steering/RETRACTION_LOG.md` (2026-09-13):

* ⛔ **CLASS C — an output decoded from a guessed schema when the producer ships its own decoder.**
  Two of the PI's three complaints. Rule: run the producer's decoder first; write your own only to add
  something, and diff it against theirs on the same bytes.
* ⛔ **CLASS D — a pretrained model judged on inputs outside its published operating geometry.** The v1
  builder's docstring called its output *"in the model's training distribution"* because the camera
  COUNT matched — the one axis A1 shows does not matter. Rule: reproduce the demo on our hardware against
  its GT first, then tabulate our input against every axis the docs pin.
* ⚠️ Caught before publishing: A2's black-canvas side effect (the 2026-09-11 "match the side effect"
  rule, recurring in a new stream two days later); a "has / has not" map claim that a count refuted; a
  staged LiDAR artifact carrying 4 plaintext clip ids under a document that says it carries none.

---

## 11. Deliverable manifest

Landed in one commit on `agent/arch-inf-20260803` (plumbing route, CAS, push verified by `ls-remote`
and a per-path `cat-file -e`). Clip ids are replaced by sha12 in every landed file (scan: 0 hits, with a
same-breath control that the scanner matches a known id).

| artifact | where it lives |
|---|---|
| this result | `repo: TanitAD Research Lab/Data Engineering/Research/2026-09-13-qwen-drive-usage-review/RESULT.md` |
| v2 frame builder (virtual nuPlan rig) | `repo: …/code/build_frames_v2.py` |
| pre-GPU geometry check | `repo: …/code/gt_projection_check.py` |
| demo control, pass 1 + 2 (A0–A6) | `repo: …/code/ctrl_make_variants.py`, `ctrl_make_variants_b.py`, `ctrl_metrics.py`, `ctrl_go.sh`, `ctrl_go_b.sh` |
| our-data scorer + threshold sweep | `repo: …/code/ours_metrics.py`, `pr_sweep.py` |
| Thor runner (upstream scripts, unmodified) | `repo: …/code/v2_go.sh` |
| v1 outputs through the upstream visualizer | `repo: …/code/v1_render_theirvis.py` |
| banner + redaction + video | `repo: …/code/annotate_and_video.py`, `make_media.py` |
| rig probe (demo + v1) | `repo: …/code/qd_probe_calib.py`, `…/raw/demo_rig_probe.txt`, `…/raw/v1_packing_rig_probe.txt` |
| control metrics | `repo: …/raw/demo_control_metrics.json`, `…/raw/demo_lidar_ground_height.txt` |
| our-data metrics + sweep | `repo: …/raw/ours_metrics.json`, `…/raw/pr_sweep.json` |
| media (PNG/JPG) | `repo: …/media/` |
| ⚠️ 4 sequence MP4s | `devbox scratchpad v2/` + `D: …/media/` (untracked, git-ignored) — ONE machine |
| ⚠️ v2 frame dirs (14 + 384 frames, ~2 GB, gated PhysicalAI pixels) | `devbox: C:\Users\Admin\qwenvis\v2\` and `thor: /home/nvidia/qwendrive/v2/` — deliberately not in git; rebuildable by `build_frames_v2.py` |
| ⚠️ Qwen-Drive predictions (`.npz`) + upstream renders | `thor: /home/nvidia/qwendrive/v2/out_*` — rebuildable in ~30 min of Thor |
| stranded 2026-09-11 packages, landed with this commit | `repo: TanitAD Research Lab/Architecture & Inference/Research/2026-09-11-lidar-bev-gt/` (16 files, `raw/lidar_probe.json` redacted) and `repo: TanitAD Research Lab/Data Engineering/Research/2026-09-11-qwen-drive-perception-sample/code/` (SUPERSEDED banner, `sample_plan.json` redacted) |

---

## 12. Map and occupancy quality on OUR data — measured without map ground truth (second pass, same day)

⭐ **The method.** PhysicalAI ships no map or semantic occupancy, so every number below is agreement with
**independent** evidence — the LiDAR sweep nearest the frame, the GT agent boxes, and the ego's own driven
path — computed by one code path (`code/mapq_core.py`) on **Qwen-Drive's nuPlan demo first, where the TRUE
map exists**. A metric is admissible only if the true raster separates from a shuffled and a mirrored raster
there. MEASURED, `raw/demo_calibration.json` (4 nuPlan frames) and `raw/mapq_ours.json` (384 frames, 4 clips).

⚠️ **Three defects caught while building it, each fixed before any number here was read:**
(1) the map raster's orientation was **measured**, not assumed — under the right convention the demo's true map
puts **0.2 %** of static obstacles on driveable, under the three mirrorings **11–25 %**;
(2) a sparse **32-beam nuScenes** sweep broke the local ground estimate (9,870 of 13,030 "obstacles" were
ground) — calibration therefore uses the nuPlan rig and a sparse cell falls back to a 6 m ground cell;
(3) PhysicalAI labels no cones/barriers/bollards and a raw sweep can hold self-returns, so the obstacle metrics
use **tall** obstacles (1.2–3 m) with the ego footprint excluded, recalibrated on the demo (true map **0.03 %**).
⛔ The LiDAR **paint** proxy (bright ground returns) is **not admissible**: gravel/grass verges return brightly
(visible in `media/mapq_*`), and it cannot be calibrated on the demo (no intensity there).

| metric (better) | TRUE map, nuPlan demo | Qwen on nuPlan (in-distribution) | shuffled, demo | **Qwen v2 on OUR data** (pooled) | ours, per-clip medians | ours fused ±2 s | shuffled, ours | mirrored, ours | prior, ours |
|---|---|---|---|---|---|---|---|---|---|
| tall static obstacles (walls, poles, trees) on predicted **driveable** (lower) | 0.03 % | 0.03 % | 60.8 % | **8.3 %** | 1–30 % | 7.6 % | 42.1 % | 21.5 % | 56.0 % |
| the ego's own driven path on predicted **road** (higher) | — | — | — | **93.5 %** | 94–100 % | 96.4 % | 77.8 % | 70.9 % | 82.1 % |
| vehicle GT-box centres on predicted **road** (higher) | 100.0 % | 100.0 % | 33.3 % | **90.4 %** | 100–100 % | 92.7 % | 60.6 % | 47.6 % | 57.4 % |
| predicted **road edges** on a real curb step or obstacle edge (higher) | 74.7 % | 72.8 % | 31.3 % | **50.8 %** | 38–65 % | 55.3 % | 39.7 % | 38.0 % | 32.9 % |
| tall static obstacles covered by predicted **occupancy** (higher) | 99.9 % | 96.2 % | 39.6 % | **52.6 %** | 32–70 % | 52.6 % | 39.8 % | 36.0 % | — |
| LiDAR points on agents covered by predicted **object** occupancy (higher) | 100.0 % | 99.1 % | 1.0 % | **56.0 %** | 54–99 % | 56.0 % | 6.2 % | 4.7 % | — |

Per clip (median of frames, v2):

| clip | MAP_A2 | MAP_B | MAP_C | MAP_E | OCC_A2 | OCC_B |
|---|---|---|---|---|---|---|
| `4fbd97b6a4b7` dense junction | 15.2 % | 99.4 % | 100.0 % | 38.4 % | 31.8 % | 53.8 % |
| `73495082f98b` VRU-dense night | 30.2 % | 100.0 % | 100.0 % | 37.8 % | 47.3 % | 54.6 % |
| `0d90d20036a3` night ramp turn | 15.4 % | 100.0 % | 100.0 % | 65.0 % | 70.3 % | 68.6 % |
| `6924358fafe0` forest road (control) | 0.9 % | 94.5 % | 100.0 % | 55.2 % | 46.4 % | 99.3 % |

Same 10 legacy instants with LiDAR, v1 vs v2 packing:

| metric | v1 packing | v2 packing |
|---|---|---|
| tall static obstacles (walls, poles, trees) on predicted **driveable** | 23.4 % | 9.5 % |
| the ego's own driven path on predicted **road** | 94.8 % | 97.8 % |
| vehicle GT-box centres on predicted **road** | 83.8 % | 89.2 % |
| predicted **road edges** on a real curb step or obstacle edge | 49.5 % | 49.7 % |
| tall static obstacles covered by predicted **occupancy** | 38.5 % | 51.1 % |
| LiDAR points on agents covered by predicted **object** occupancy | 18.6 % | 59.7 % |

### 12.1 Verdict

* ⭐ **The map layout is real, but coarse.** On every calibrated map metric Qwen's v2 map beats the shuffled,
  mirrored and prior controls — it knows where the road is. It falls well short of a true map where geometry
  must be exact: road edges sit on a real curb or obstacle edge about half the time (true map ~75 %), and in the
  urban clips the driveable class spills onto walls, poles and trees.
* ⛔ **The occupancy is weak on our data.** It covers roughly half of the static obstacles and agent points
  that the in-distribution model covers almost entirely on its own demo. ⇒ **for occupancy, the LiDAR
  (+ obstacle.offline) is the better teacher**; Qwen's value is the semantic **map**, which nothing else supplies.
* ⭐ **Temporal fusion (the cheapest lever) helps the area/edge classes** — see the fused column — but plain
  majority voting **erases thin road lines** (`media/mapq_*`), so any production fusion must be class-aware.
* ⚠️ No interval; 4 clips from one chunk; the demo calibration is 4 frames. These rank levers; they are not a benchmark.

