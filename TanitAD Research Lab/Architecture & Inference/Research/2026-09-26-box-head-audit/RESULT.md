# RESULT: the refcv6 3-D box and agent heads. Why they "detect boxes where there is nothing", what the GT and the overlay really show, and what refcv7 must change

**Date:** 2026-09-26/27 (Europe/Berlin).
**Agent:** box-head audit (Architecture & Inference), reporting to the Master Mind.
**Code:** tip `a756e81` of `agent/arch-inf-20260803`. The box path files are byte-identical at the launch `284393c`, the resume `82c2331` and the tip (`git diff --stat` is empty for all 9 of them).
**Model:** refcv6-r101-s0, step 38,000.
- ckpt md5 `5a2e7222a9f5f8c7aa7bf38ef4698d8a`, verified by content on Thor.
- config md5 `a3193a4685ce0d07a6ae6b89fdc994a8`.
- metrics.jsonl md5 `1f195e3b1b897e4255bd1b8f3984514d`.
**Compute:** Thor CPU only, in a fresh dir `/home/nvidia/bha_2325/`, 4 threads, `CUDA_VISIBLE_DEVICES=""`. The dev box was used for file handling only, because its RAM stayed below the 7.5 GB start floor all night.
**Evidence:** MEASURED (ours) unless a line says otherwise. **Tier:** none. These are diagnostics of one checkpoint, not a T0/T1 result. Clip ids appear as sha12 only.

**The PI, 2026-09-26, verbatim:** *"the detected objects bounding boxes are completelly messy, the model is detecting boxes where is nothing. We need to understand what happening, and review training, architecture, wiring, training signal flow and validate gt."*

Later, after the GT-only stills: *"some of the problems are comining definetly from the gt boxes or their interpretation … In some caes the gt boxes ionclude occluded boxes which can not be detect by the camera"*.

---

## 0. Headline

**Why refcv6 draws boxes where there is nothing.** The causes below are ranked by MEASURED size. The base is the 3-D box head at the model's own gate 0.5, on a representative eval sample: 139 eval clips × 4 labelled windows = 556 windows, 2,908 trainer targets (`raw/analysis/analyze_clipgrid.json`).

1. **Decision rule. The presence head was trained to over-report, and then read at the wrong threshold.**
   - The presence loss weights an empty slot 0.1. The head therefore learned exactly p = π / (π + 0.1(1 − π)), which is ×10 odds on its real belief π. The fit is exact: the ECE of σ(logit − ln 10) against "this slot gets matched" is **0.0042**, against 0.187 for the raw σ.
   - The 0.5 gate therefore fires at π ≥ 0.091. Result: **9,885 confident boxes for 2,908 targets (3.4×)**, precision 0.162, recall 0.551.
   - At the precision = recall gate (0.719), **70.6 % of those boxes and 75.3 % of the false positives disappear.** The gate measured on TRAIN is 0.713, within 0.006, so it transfers. Applied to eval it gives 1.05 boxes per target, precision 0.289, recall 0.304.
2. **Localisation: the box is near a real object but more than 2 m off.**
   - This is 45.1 % of the false positives at 0.5, and 54.9 % at the P = R gate.
   - Presence carries NO localisation quality: Spearman(presence, centre error) = **−0.016** on Hungarian pairs.
   - The median matched centre error is 1.05 m, and 20.8 % of pairs are more than 2 m off.
3. **Hallucination: no GT box of any kind within 5 m.** This is 34.4 % of the false positives at 0.5 and 19.2 % at P = R.
4. **Duplicates: a second confident slot on an object already claimed.** 18.0 % of the false positives at 0.5, 24.7 % at P = R.
5. **The GT asks for the impossible: many targets cannot be seen.**
   - An exact z-buffer of the GT cuboids in the clip's own camera shows 25.9 % of representative targets under 10 % visible (19.1 % fully hidden). On the PI's 5 clips the figure is **56.2 %** (46.2 % fully hidden).
   - 20.7 % of the head's own hits are on such hidden targets.
   - The camera cannot confirm them, the head was trained to produce them, and under camera-visibility scoring they are false positives. Re-scored under VIS-1, refcv6's precision at 0.5 falls from 0.162 to 0.109.

**These causes are not additive: #1 multiplies #2-#4.** The head is not broken:
- presence AUROC is **0.855 [0.824, 0.886]** (clip-cluster bootstrap), and within a window 0.73;
- the matched box geometry is decent: 0.68-1.26 m centre error, 0.06-0.08 m width error, 8-14° yaw error, 0.05-0.18 m z error by range band;
- Σ calibrated presence per window tracks the true count (K 3-5: 4.48 for 3.82 targets; K 6-10: 8.05 for 7.50).

But its best operating point is **F1 0.30 / AP@2 m 0.21** on eval, and **F1 0.33 / AP 0.23 on its own TRAINING windows**. It UNDERFITS; it does not overfit. **No box quality signal was ever logged**, so 38,000 steps of a falling presence loss (0.103 → 0.065) looked like progress.

**The overlay is geometrically CORRECT (§1).** It was verified against the raw mp4 pixels, against LiDAR through the LiDAR's own extrinsic, and against the SAM3 pipeline's camera model. Its only defect is the DRAWING: edges crossing the border are dropped, which makes the "one surface" faces. That is patched, tested, and banked.

**The GT is geometrically correct for what it labels (§1.4).**
- LiDAR returns lie inside 92-100 % of targets out to 60 m.
- Cars, persons and riders are ground-standing.
- ⛔ **CORRECTION:** "heavy_truck is sunk" was retracted at 02:0x. It is range drift at 90-150 m, beyond the decode box (§1.5).

**But the GT is not camera-visible (§2).**

**⚠️ Integration and decisions for the Master Mind (not done by me):**
- (a) land the renderer patch, which someone has already copied into `taniteval/tools/`, see §1.6;
- (b) register VIS-1 as a SPEC_REFCV7 amendment BEFORE any refcv7 box number;
- (c) register `raw/PREREG_G_BOX_OVERFIT.md`;
- (d) adopt the presence-objective fix, the G-LIVE checks and `raw/LOGGING_SPEC_BOX.md` in the refcv7 build and launch gate;
- (e) re-render the PI's video with the train-derived gate 0.713 if the PI wants refcv6's boxes at a sane operating point;
- (f) the literature package landed at 02:02 and is reconciled in §7.3. The two AGREE; the measurements here re-order its R1-R9. Decision rule and metrics first; then focal presence + per-layer loss + VIS-1-with-IGNORE; then geometry and denoising on the ladder.

---

## 1. Task 0: independent audit of the 3-D camera overlay. VERDICT: CORRECT

**The problem with the renderer's own control.** Its control ("GT drawn through the predicted path lands on itself") is self-consistency. It cannot see an error that hits GT and predictions alike. Every check below instead uses a reference that shares NO code or assumption with the renderer. Code: `code/ov_audit.py`, `code/ov_p2p5.py`. Artifacts: `raw/overlay_audit/`.

**The clip.** 73495082f98b is the only B1 clip with raw LiDAR on Thor. It is a TRAIN clip, at night, rig B (cy 753.65). Its 95 frames all have a LiDAR spin within 49 ms of the frame time.

| # | check (independent reference) | result |
|---|---|---|
| P1 | renderer (`cuboid_corners` + tanitad `RigCamera`, built from the renderer's extrinsics table) vs my numpy projection from the RAW `sensor_extrinsics` parquet row (quaternion to R by hand; u = 511.5 + f·atan2(x,z), v = 207.5 + f·y/hypot(x,z)) | **max \|Δu\| 2.3e-13 px, max \|Δv\| 1.1e-13 px**, over 25,375 corners of 3,272 GT boxes. The table entry equals the raw parquet row exactly (max abs 0.0). ⚠️ This is AGREEMENT only; P2-P5 test the assumptions. |
| P2 | the cached 416×1024 frame vs **my own re-render of the RAW front-wide mp4** through the clip's f-theta intrinsics (`camera_intrinsics` parquet: fw_poly, cx, cy) | On 6 frames, the mean \|RGB\| error at the declared geometry is **0.73-0.87 /255**. Shifting the grid by ±1 px gives 2.17-4.09, by ±2 px 3.76-6.58; scaling f by ±0.5 % gives 2.80-4.20. **Best shift (0,0) and best f-scale 1.000 on 6/6 frames.** The mp4 frame = searchsorted(t_cam, linspace(t0, t1, int(span·10))) with offset 0, on 6/6. ⇒ The cylinder is centred on the clip's principal point, at f_ref 488.924, in the camera's own optical frame (no roll or pitch rectification), exactly as `RigCamera`/`CylProjector` assume. |
| P3 | LiDAR (`lidar_top_360fov`) sent **through its own extrinsic** into the rig, then through the camera extrinsic, onto the cached frame | Building facades, window rows, lamp posts, a crossing silver car and a van all land on their image structure. The GT cuboids enclose the objects' LiDAR returns, with bottoms at the wheels. 6 panels: `raw/overlay_audit/ov_73495082f98b_f{003,033,063,093,124,154}.png`. ⚠️ **Provenance.** The sweeps are the SAM3 pipeline's pre-decoded ones (`/home/nvidia/sam3map/data/sweeps/`: the LiDAR `sensor_extrinsics` row applied, then z − 0.325, per `mapq_ours.Clip.sweep_ego`). The 0.325 shift is undone here. The sweeps are NOT deskewed, so points can sit up to about ±0.5 m off at the frame time; the inside-box test uses a 0.3 m margin. |
| P5 | the SAM3 pipeline's camera model (`/home/nvidia/sam3map/camera_model.py`, native f-theta; the PI validated its map reprojection videos by eye) with the pipeline's OWN `calib.npz` | Its calib equals our parquet (R 2.8e-16, t 0.0, poly 0.0, cx/cy 0.0). Despite the `sensor2lidar` name it is camera→RIG. On 17,761 points, its native pixel vs ours (cylinder → ray → f-theta) differs by at most **6.8e-13 / 3.4e-13 px**. Control: the same points fed in the LiDAR frame land 239/614 px off, so the check can fail. |

**Camera axes from the quaternion.**
- boresight (0.9996, −0.0045, −0.0288), i.e. **1.65° pitch-down**;
- right (−0.004, −1.000, 0.006);
- down (−0.029, −0.006, −0.9996).

This is RDF-optical to FLU-rig, as every consumer assumes.

**Timing.** The image drawn is the newest raw frame of the stack (raw f + 2). That is the episode pose f's instant (`v2_dataset.py:35-38`, `_decode_stacked` `:109-149`), the same instant at which the join places its boxes.

### 1.4 GT geometry (P4)

**LiDAR evidence.** Trainer TARGETS with at least 3 LiDAR returns inside the cuboid (grown 0.3 m, at least 0.25 m above the local ground). The same box mirrored (y → −y) is the control.

| range | targets with LiDAR | mirrored control |
|---|---|---|
| 0-10 m | 68/68 | 61.9 % |
| 10-20 m | 226/233 = 97.0 % | 63.9 % |
| 20-30 m | 240/242 = 99.2 % | 68.5 % |
| 30-45 m | 243/243 = 100 % | 49.5 % |
| 45-60 m | 297/324 = 91.7 % | 20.5 % |

⚠️ The mirror control is weak near the ego: on a dense street a mirrored footprint often lands on another object. The near-range evidence therefore rests on the P3 panels as much as on the ratio.

**Height.** Box bottom (join3d cz − h/2) minus the local LiDAR ground (10th percentile of a 1-3 m ring around the box):
- automobile **−0.076 m** (n 5,228), person **+0.021 m** (n 1,242), rider **+0.036 m** (n 139). They are ground-standing.
- The LiDAR road plane within 20 m sits at rig z **+0.017 m** (n 685). This is consistent, within 5 cm, with the 09-11 LiDAR package's −0.035 m.
- The earlier memory claim "obstacle.offline z is ~1 m under the LiDAR ground" does NOT hold on our join for cars, persons or riders (MEASURED). Its likely source is INFERRED, not re-measured: the qwendrive-v2 frame the SAM3 checks used, whose points carry z − 0.325 (`mapq_ours.py`, `Z = 0.325`), and/or far trucks (§1.5).

**No stage rewrites z.** join3d equals the raw obstacle.offline parquet: max |Δcz| **5.0e-5 m** and max |Δh| **5.0e-4 m** (n 1,564). `targets_from_join`, `zh_targets` and the renderer pass cz/h through unchanged.

### 1.5 heavy_truck: a range effect, not a label defect (CORRECTION, dated 2026-09-27 ~02:05)

I first reported "heavy_truck is sunk, a LABEL defect" from 3 rows of one truck on the LiDAR clip, and from the video agent's pooled −0.63 m. **That is RETRACTED.** The coordinator was told at 02:0x.

The measurement uses d = bottom minus the median bottom of the same frame's cars, persons and riders within 15 m. Slope and pitch move a box and its neighbours together, so d isolates a CLASS offset from the terrain.

**On the GT-validation clips themselves** (`code/gt_zcensus_gtval.py`, `raw/analysis/gt_zcensus_gtval.json`):
- The 639 heavy_truck rows sit at median range **99.8 / 90.8 / 147.8 m** (0191487845ef / 22d47ae7e052 / 9f8bedcfb9de). They are all beyond the 60 m decode box, so they are NOT trainer targets.
- Their absolute bottoms (−0.97 / −0.76 / −0.32 m) are terrain at 90-150 m. A 0.5° pitch alone gives 0.8-1.3 m there.
- Relative to their neighbours, d = **−0.13 m** (n 140; p10 −0.25, p90 +0.18). For comparison: bus −0.10 (n 652), automobile +0.002 (n 22,927), person +0.002 (n 41,829).

**On the representative samples** (`raw/analysis/gt_zcensus.json`):

| class | eval (139 clips) | train (64 clips) |
|---|---|---|
| heavy_truck | **+0.021 m** (n 90) | **+0.030 m** (n 29) |
| bus | −0.082 (n 82) | – |
| trailer | −0.027 (n 30) | – |
| automobile | −0.006 (n 10,183) | – |
| person | +0.020 (n 2,754) | – |

- protruding_object: +1.17 m (eval), +1.51 m (train). It is airborne by definition, so this is correct semantics.

**For refcv7: no truck-specific target correction.**
- Re-seating trucks on the rig ground (z = 0) would be wrong: it would lift far trucks 0.6-1.0 m off the slope their neighbours share.
- Dropping the truck z-term is not warranted either: inside the decode box the offset is +0.02-0.03 m.

### 1.6 The one real overlay defect: the DRAWING. Patched and banked

**The defect.** `render_refcv6_map_video.draw_cuboid_cam` (`:625-640` at md5 `0becac57…`) drew an edge only if BOTH corners were in the frame, and drew each edge as the corner-to-corner chord.
- A cuboid crossing the ±60° border therefore lost every border-crossing edge and read as ONE FACE. That is the PI's bus #39 in 9f8bedcfb9de w104, which is also fully hidden (§2).
- The cylinder bends a horizontal 3-D edge, and a chord does not.
- The GT-validation tool has the same rule (`draw_gt_camera`).

**Edge census over the 22,432 GT-val targets** (`raw/visibility/vis_panels.json`):
- the old rule drew fewer edges than have an in-frame part for **943 (4.2 %)** of targets;
- 4 edges or fewer for **569 (2.5 %)**;
- NO edge for **56 (0.25 %)**.

These are the near, large, border-crossing boxes, the most visible ones.

**The patch** (`code/patch_draw_cuboid/`):
- `cuboid_edge_runs`: every edge is projected as a SAMPLED 3-D segment (32 samples), clipped at the frame border by bisection (24 steps). `draw_cuboid_cam` draws the in-frame runs.
- The GT-val tool reuses the same function by path (one implementation), uses a phase-continuous dashed polyline, and skips a cuboid only when NO edge part is in frame. The old rule skipped it when no CORNER was in frame, which lost a bus alongside.
- Files:
  - `render_refcv6_map_video.py` (md5 `e5890efbe4caf7f582584240f5bf4f75`);
  - `render_refcv6_gt_validation.py` (md5 `62f6d2a0ba46118b92a14e786a120499`);
  - `draw_cuboid_clipped.diff`, against the verified originals `0becac57…` and `30215c0a…`;
  - `test_draw_cuboid_clipped.py`: **5/5 pass on Thor**. It covers a 12-edge car with exact corner ends; a border-crossing bus drawing more edges than the old rule, with every non-corner end on the border; a > 3 px bend on a long edge; nothing drawn behind the camera; and a RED arm that re-implements the old rule and must fail the border property.
- ⚠️ **Integration state (observed, not done by me).** At 01:13:40 `taniteval/tools/render_refcv6_map_video.py` in the D: working tree became byte-identical to my patched file. At 01:14 the GT-val tool changed to md5 `394ea228…`, which is my patch plus someone's further edits. The Master Mind should confirm which version lands.

---

## 2. Task 0b: GT camera visibility

### 2.1 Schema: no visibility, occlusion or point-count field exists anywhere

- **obstacle.offline** has 16 columns: `timestamp_us, source, track_id, center_x/y/z, size_x/y/z, orientation_x/y/z/w, label_class, reference_frame, reference_frame_timestamp_us`.
  - `source` = `scene:obstacles:autolabels:v2` on 251,069 / 251,069 rows (40 train clips).
- **The 2-D join agent** carries cx, cy, yaw, l, w, occ, track_id, cls (`build_obstacle_join.py:9-27`).
  - Its `occ` is only |atan2(cy,cx)| > 60° (`visibility_occ`, `:285-307`). It means out-of-field, NOT occluded.
- **join3d** adds cz and h (`agent_cuboid_gt.py:298-311`).
- **`targets_from_join`** sets `valid = True` for every row (`agent_slots.py:1063-1064`).
- **`visible_target_filter`** is field + decode box only (`refc_agents.py:397-404`). Its docstring says *"necessary, not sufficient: horizontal only, with no vertical, hood, or inter-agent occlusion"* (`:385-389`).

### 2.2 Occlusion, measured

**Method** (`code/vis_zbuf.py`): an exact per-pixel ray/oriented-cuboid z-buffer.
- The camera is the clip's own 416×1024 cylinder, with the per-clip extrinsics (the renderer's table) and the clip's f-theta observed mask.
- The occluders are EVERY GT cuboid of the frame.
- `vis_frac` = pixels where the box is nearest AND inside the observed frame, over its full silhouette on an extended cylinder.
- ⚠️ This is an UPPER bound on visibility: walls, trees, poles, the hood and unlabelled objects are absent.

| sample | windows | targets | median vis_frac | < 10 % | < 30 % | < 50 % | fully hidden |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT-val (the PI's 5 clips; = the video agent's 22,432, a control) | 855 | 22,432 | 0.024 | **56.2 %** | 65.5 % | 74.2 % | 46.2 % |
| representative eval (139 clips × 4) | 556 | 2,908 | 0.510 | **25.9 %** | 38.9 % | 49.6 % | 19.1 % |
| trainer's in-run eval windows | 121 | 509 | 0.660 | 17.7 % | 32.6 % | 44.4 % | 12.6 % |
| TRAIN (64 clips × 4) | 256 | 1,133 | 0.598 | 25.0 % | 39.2 % | 46.1 % | 15.0 % |
| LiDAR clip 73495082f98b | 95 | 1,125 | 0.707 | 17.2 % | 26.9 % | 38.9 % | 13.5 % |

**GT-val, share under 10 % visible:**

| by clip | < 10 % | by class | < 10 % | by range | < 10 % |
|---|---|---|---|---|---|
| 0191487845ef | 62.2 % | person | 65.0 % (n 14,765) | 0-10 m | 25.4 % |
| 9f8bedcfb9de | 62.1 % | automobile | 40.6 % (n 6,902) | 45-60 m | 73.5 % |
| 34765c024267 | 37.4 % | bus | 49.2 % | | |
| 22d47ae7e052 | 20.3 % | rider | 31.7 % | | |
| 59d98b34ae8b | 5.7 % | | | | |

**Representative eval, share under 10 % visible:** person 42.0 % (n 774), automobile 20.7 % (n 1,954).

**LiDAR cross-check.** Hidden targets are REAL objects that the camera cannot see; the roof LiDAR sees over the cars.

| vis_frac band | targets with ≥ 3 LiDAR returns | median returns |
|---|---|---|
| < 0.1 | 85.6 % (n 194) | 77 |
| 0.1-0.3 | 95.4 % | 152 |
| 0.3-0.5 | 99.3 % | 1,148 |
| ≥ 0.5 | 99.7 % | 628 |

### 2.3 The PI's cases, explained with numbers

Panels are in `raw/visibility/vis_<sha12>_w<win>.png`: targets are coloured green (≥ 50 % visible), yellow (10-50 %) or red (< 10 %), with the z-buffer owner map below.

- **9f8bedcfb9de w104.**
  - The "automobile 11 m #100" behind the van has **vis 0.000**, and so does the "bus 13 m #39".
  - The van itself is at 0.498; half of it is outside the frame.
  - **16 of 28 targets there are under 10 % visible.**
  - The bus's "one surface" is the drawing defect of §1.6 on a box that is also fully hidden.
- **0191487845ef w064.** The persons at 5.6, 10.3, 14.6 and 15.8 m, including "person 10 m #58", are all at 0.000, hidden behind parked cars and other people.
- **"Not consistent to the image"** has three parts:
  - occlusion, above;
  - the edge defect;
  - range drift of far boxes. That is slope or pitch in the BODY frame, which labels and camera share. It projects CONSISTENTLY and is not an overlay error. The truck case is §1.5.

### 2.4 Proposed refcv7 rule VIS-1 (for registration as a SPEC_REFCV7 amendment before any box number)

One function (`code/vis_rules.py`) is applied IDENTICALLY to TRAINING targets and to EVAL scoring, after `visible_target_filter`:

| label | condition |
|---|---|
| POSITIVE | vis_frac ≥ **0.30** AND n_vis ≥ **100 px** |
| IGNORE | **0.05** ≤ vis_frac < 0.30, OR vis_frac ≥ 0.30 with n_vis < 100 px |
| DROPPED | vis_frac < 0.05 |

- **IGNORE rows:**
  - excluded from Hungarian matching;
  - an unmatched slot whose centre lies within **2.0 m** (BEV) of an IGNORE row gets presence weight 0;
  - at eval, a confident detection that the greedy 2 m matcher pairs with an IGNORE row is removed from the PR count (KITTI DontCare).
- **DROPPED rows** are background for the loss and absent for the metric.

**Effect on targets per window** (`raw/visibility/vis_rules.json`):

| sample | now | POSITIVE | IGNORE | DROPPED |
|---|---:|---:|---:|---:|
| GT-val | **26.24** | **8.49** | 3.86 | 13.89 |
| representative eval | **5.23** | **3.11** | 0.95 | 1.17 |
| LiDAR clip | 11.84 | 8.19 | 1.84 | 1.81 |

Sensitivity of POSITIVES per window, representative eval:

| threshold | 50 px | 100 px | 200 px |
|---|---:|---:|---:|
| vis ≥ 0.2 | 3.48 | 3.37 | 3.01 |
| vis ≥ 0.3 | 3.18 | **3.11** | 2.82 |
| vis ≥ 0.4 | 2.90 | 2.85 | 2.63 |
| vis ≥ 0.5 | 2.64 | 2.61 | 2.42 |

**What it does to refcv6's numbers** (`raw/analysis/tp_vis_clipgrid.json`). Re-scored under VIS-1, the box3d head reads:
- precision at 0.5 **0.162 → 0.109**, recall 0.551 → 0.578;
- AP 0.210 → 0.151;
- P = R 0.296 → 0.235.

**20.7 %** of its hits are on targets under 10 % visible (misses: 32.3 %). refcv6 fires on hidden objects because it was trained to. The rule therefore does not flatter refcv6; it exposes it.

**Test (to build with the rule; all literal):**
- a car fully behind a van → DROPPED;
- a car half behind a van → POSITIVE, with its expected vis within ±0.02 of the analytic 0.50;
- a 60 px car at 50 m → IGNORE;
- GT written as slots → P = R = 1 under the IGNORE accounting.

**RED arms:**
- the depth order reversed (farthest wins), and the z-buffer skipped (filter-only). Each must fail "the hidden car is not a POSITIVE";
- IGNORE handling removed. That must fail "a detection on an IGNORE row is neither TP nor FP".

**Interactions:**
- **The presence fix.** Fewer positives per slot lowers π everywhere, so every gate, band and G-LIVE number must be computed on VIS-1 POSITIVES. IGNORE needs its presence mask, or the loss pushes partly visible objects to "empty".
- **G-BOX-OVERFIT.** It scores VIS-1 POSITIVES and IGNORE rows identically. It is NOT the visibility test: 16 frames can be memorised with hidden targets too.
- **Scope.** The planner's agent tokens may legitimately want OCCLUDED agents (P4). That would be a separate target and metric, not this camera detector's.

---

## 3. Task 1: static trace of the box path, end to end (file:line at tip `a756e81`)

⚠️ in the last column marks a hop where the "empty slot" signal is weakened or lost.

| # | hop | file:line | what happens | empty-slot signal |
|---|---|---|---|---|
| G1 | obstacle.offline → per-track arrays | `stack/scripts/build_obstacle_join.py:364-403` (`clip_tracks`) | raw cuboids (autolabels v2), rig frame at their own timestamp; the class is the track's first sample | – |
| G2 | per-frame sample | `:406-435` (`world_agents_at`) | per track, the nearest sample within **0.06 s**; rig@sample → world through egomotion (2-D: x, y, yaw) | – |
| G3 | the frame's label | `:438-496` (`join_clip`) | LABELLED iff the frame time is inside the obstacle span ± tol (`:466-469`); world → ego@frame (`:471`); `occ` = azimuth > 60° (`:472`, `:285-307`). **Every track in range, in every direction, visible or not.** | ⚠️ no visibility, no range cap, no point count |
| G4 | join reader | `stack/scripts/train_p8_occupancy.py:240-486` (`JoinFileReader`), `bev_raster.agents_to_array:141-165` | `[A,6] = (cx, cy, yaw, l, w, occ)`, cls, rates, track ids; an absent line = NO_LABEL | – |
| G5 | dataset block | `stack/scripts/refc_v3_train.py:2856-2956` (`V3Dataset._agent_item`), `enable_agent_join:2687-2798` | RAW targets at NOW = t + W − 1 (**no filter here**, `:2865-2868`); pad 397; join3d cz/h BY TRACK ID at raw frame f + n_stack − 1 (`:2933-2955`) | – |
| G6 | targets | `stack/tanitad/models/agent_slots.py:1014-1077` (`targets_from_join`) | `valid = True` for every row (`:1064`); an out-of-vocabulary class → −1, so it is excluded from the class term only (`:1065-1069`) | ⚠️ hidden objects become positives |
| G7 | field cut | `stack/tanitad/refs/refc_agents.py:410-433` → `:336-407` | keep = valid ∧ \|atan2(\|cy\|, cx)\| ≤ 60° ∧ 0 ≤ cx ≤ 60 ∧ \|cy\| ≤ 16 (`:397-404`); applied BEFORE matching in both heads (`box3d_head.py:376-398`, `refc_agents.py:848-851`). It removes **83.7 %** of the raw rows (the run's own train rows: 387,498 → 63,230) | ⚠️ horizontal only (§2) |
| D1 | box memory | `stack/tanitad/models/box3d_head.py:132-221` (`Box3DMemory`) | stride-16 trunk map of the LAST frame (26×64 = 1,664 tokens, 1,024 ch → 256) + BEV features average-pooled to 30×16 = 480 tokens; +2 source embeddings | – |
| D2 | box decoder | `:227-284` (`Box3DSlotDecoder`) → `agent_slots.py:546-684` | 100 learned queries, 3-layer pre-norm TransformerDecoder (d 256, 8 heads), one linear head of 23 channels | – |
| D3 | presence | `agent_slots.py:168`, prior `:573`, `:615-617` | one logit per slot; bias init logit(0.05) = −2.94 | – |
| D4 | decode | `agent_slots.py:656-684`, `box3d_head.py:265-284` | cx = r·60, cy = r·16; l, w = softplus; yaw = atan2 of the normalised (sin, cos); cz = r·4; h = softplus·2 (bias at 1.594 m); rates; occ | – |
| D5 | agent head (planner tokens) | `stack/tanitad/refs/refc.py:3653-3670` (build), `:4366-4383` (forward on `fmap`, stride 32, not detached), `refc_agents.py:238-255` | the same `AgentSlotDecoder` on the trunk's stride-32 map | – |
| D6 | agent tokens | `refc_agents.py:313-333` | tokens × **sigmoid(presence)** (SOFT gate, `:332`); `presence_hard` off (`:203`); gate 0.5 (`:199`) | ⚠️ the planner/tactical losses reach the presence logit with NO object label (the size is UNMEASURED; the tilted calibration survives, ECE 0.0036) |
| M1 | matcher | `agent_slots.py:788-823` (`match_slots`), cost `:767-785`, weights `:216-221` | Hungarian over ALL 100 slots; cost = 1.0·L1 centre (m) + 0.5·L1 size (m) + 1.0·(−p[true class]) + **1.0·(−σ(presence))**; with more than 100 targets the farthest are dropped and counted | ⚠️ **every target gets a slot however far away it is**, so presence learns "P(assigned)", not "P(an object here)" |
| L1 | presence loss | `agent_slots.py:864-873` | BCE; matched → 1 (weight 1), **every other slot → 0 with weight `NO_OBJECT_W` 0.1** (`:231-233`); `reduction="mean"` over B·100 (not Σw) | ⚠️⚠️ **the ×10 odds tilt (§4); the gradient is ~7× weaker than DETR's Σw normalisation at K = 5** |
| L2 | class | `:897-918` | CE on matched slots only, with the class-weight vector `--agent-cls-weight b1` (digest `c9dbc340acf7d27e`, span 1,298:1), normalised by Σw | ⚠️ the same tilt for classes: rare classes are over-predicted (§5.4) |
| L3 | box terms | `:884-896`, `:919-930`; z/h `box3d_head.py:419-439` | L1 centre (m), L1 size (m), 1 − cos yaw, L1 rates (masked), BCE occ; z, h L1 under `zh_mask`. Weights `SLOT_LOSS_W:226-229` (1, 1, 1, 1, 1, 0.5, 0.5), `BOX3D_LOSS_W:126` (1, 1) | none of them teaches presence where the box is |
| T1 | trainer | `refc_v3_train.py:4742-4800` (box3d, labelled rows `:4743-4784`, `loss += w_box3d(1.0)·total` `:4793`), `:4290-4384` (agent, `loss += w_agent(1.0)·total` `:4369`) | NO_LABEL rows are removed before the loss; LABELLED-CLEAR rows stay (all 100 slots negative) | – |
| R1 | gradient reach | `refcv6_perception_branch.py:38-47`, `:462-464`; `refc.py:4067-4133` | box3d → decoder → memory → **trunk (stride 16, attached)** and **the BEV lift + encoder** (`use_bev` = w_map > 0, `:198-200`); agent → head → trunk (stride 32); planner → presence (D6) | ⚠️ **never logged**: 0 `ga_*` keys in 4,621 rows |
| O1 | what was logged | `refc_v3_train.py:4370-4384` (agent: presence, cls, centre, size, yaw + counts; **rates and occ not logged**), `:4794-4797` (box3d: every term + counts + the filter flag), eval `:8647-8650` (mean over 8 batches) | loss terms and counts only. **No precision, recall, AP, confident count or calibration, for either head.** `n_matched == n_target` by construction | ⚠️⚠️ the defect was invisible |

---

## 4. Task 2: the analytic presence optimum, and its measured confirmation

**The formula.** The map audit's partial (`…/2026-09-26-map-signal-audit/raw/box_presence_partial.md`) is built on, not redone. Under matched → 1 (weight 1) and unmatched → 0 (weight w0 = 0.1), a slot that the model believes is matched with probability π minimises its expected loss at

**p\* = π / (π + 0.1(1 − π)), i.e. logit(p\*) = logit(π) + ln 10.**

- The gate 0.5 is therefore crossed at **π = 1/11 = 0.091**.
- A 50 % belief needs p ≥ 10/11 = 0.909.
- With exchangeable slots, all 100 are confident once K ≥ 10 targets per 100 slots.

**Additions:**
1. **The reduction does not move the optimum, but it moves the learning speed.** `binary_cross_entropy_with_logits(..., weight=w)` with `reduction="mean"` divides by B·100, not by Σw. At K = 5 the ratio is 100 / (5 + 0.1·95) = **6.9**, so the presence term's gradient is ~7× weaker than DETR's weighted-mean class loss.
   - In the run, presence is **1.3 %** of the box3d total (0.065 of 5.20 at 34.5-38.3k).
   - The box3d total is 27-34 % of the training loss (`raw/analysis/traj.json`).
2. **MEASURED: the head sits AT the tilted optimum.** The 10-bin ECE of σ(logit − ln 10) against the Hungarian-matched indicator:

   | head | clipgrid | inrun | train | raw σ |
   |---|---|---|---|---|
   | box3d | **0.0042** | 0.0043 | 0.0045 | 0.187 |
   | agent | 0.0036 | – | – | 0.200 |

   The model learned exactly what the loss asked. **The over-reporting is the objective working as written.**
3. **Hungarian + tilt ⇒ all-100 only when crowded.** Every target takes one slot, so π per slot concentrates on a few specialised slots. Windows with ALL 100 slots confident (clipgrid box3d):

   | K band | windows with all 100 confident |
   |---|---|
   | K ≤ 10 | 0 |
   | K 11-20 | 1 of 58 |
   | K 21-40 | 1 of 20 |
   | K ≥ 41 | 4 of 5 |

   This is consistent with the exchangeable threshold K ≥ 10 plus partial specialisation. The PI's video clips carry 21-80 targets per window, so "all 100 confident" there is the tail of this curve.
4. **Σ calibrated presence ≈ the true count.**

   | K band | Σp' per window | targets per window |
   |---|---:|---:|
   | K 1-2 | 2.05 | 1.39 |
   | K 3-5 | 4.48 | 3.82 |
   | K 6-10 | 8.05 | 7.50 |
   | K 11-20 | 12.19 | 14.66 |
   | K 21-40 | 18.16 | 23.95 |

   Presence knows HOW MANY objects there are, and it is over-confident at the 0.5 gate by construction.

---

## 5. Task 3: MEASURED on real windows (per-slot dumps rolled on Thor)

**Dumps** (`code/bha_dump.py`, the trainer's own forward interrupted after `model(...)`; `raw/dumps/`):

| set | rule | windows | clips | targets | per window |
|---|---|---:|---:|---:|---:|
| **inrun** | the trainer's own in-run eval subset (`train():7893-7895`, seed 12345) | 128 (121 labelled) | 76 | 509 | 4.21 |
| **clipgrid** | every eval clip × 4 windows evenly over its labelled windows | 556 | 139 | 2,908 | 5.23 |
| **train** | every 68th train clip by sha12 × 4 | 256 | 64 | 1,133 | 4.43 |

Speed: 5.0-5.5 s per window on 4 ARM cores.

**Scoring** (`code/bha_analyze.py`): every match, loss and filter is the tip tree's own function (`visible_target_filter`, `match_slots`, `box3d_match_rows` at 2 m BEV, `box3d_set_loss`/`slot_set_loss`, the decoders' `decode`).

### 5.1 Controls (all read their known values)

- **In-run control PASS.** Recomputing the run's own `eval_*` row at step 38,000 batch by batch (8 × 16, labelled rows, mean over batches), CPU vs the run's GPU:

  | term | recomputed | run's row | Δ |
  |---|---|---|---|
  | box3d presence | 0.06245 | 0.06254 | 0.14 % |
  | box3d cls | 1.2986 | 1.3004 | – |
  | box3d centre | 1.5336 | 1.5353 | – |
  | box3d z | 0.17018 | 0.17027 | – |
  | box3d n_target | 63.625 | 63.625 | 0 |
  | agent presence | 0.06169 | 0.06175 | – |
  | agent centre | 1.7163 | 1.7017 | 0.86 %, the worst term |
  | agent n_target | 63.625 | 63.625 | 0 |

  The build reproduces the run.
- The re-decode of the stored raw head outputs equals the model's own decode: max abs diff **0.0**.
- The agent join census reproduces config.json exactly: 22,663 / 23,772 eval windows labelled; 776,801 prefilter boxes.
- **Constant presence → AUROC 0.5 exactly**, on every set and head.
- **GT written into the slot format → AUROC 1.0, P = R = 1.0.** On clipgrid, recall is 0.9986 because 4 targets are dropped by the 100-query budget.
- Presence permuted WITHIN each window → AUROC **0.784** (clipgrid). This is the between-window component: the model knows which scenes are crowded.

### 5.2 Presence: informative ranking, tilted calibration

| | box3d clipgrid | agent clipgrid | box3d inrun | box3d TRAIN |
|---|---:|---:|---:|---:|
| presence, matched slots (median / p95) | 0.603 / 0.873 | 0.638 / 0.905 | 0.573 / 0.810 | – |
| presence, unmatched slots (median / p95) | 0.134 / 0.684 | 0.135 / 0.713 | 0.145 / 0.645 | – |
| AUROC matched vs unmatched (pooled) [95 % clip bootstrap] | **0.855 [0.824, 0.886]** | 0.860 [0.831, 0.890] | 0.833 [0.789, 0.871] | 0.871 [0.832, 0.905] |
| AUROC objectness (a target within 2 m) | 0.864 [0.847, 0.881] | 0.883 | 0.859 | 0.873 |
| AUROC matched, within window (median) | 0.727 (n 400) | 0.754 | 0.756 | 0.745 |

**The deciding question** was whether the presence RANKING is informative. **Yes**: AUROC 0.83-0.87 pooled and 0.73-0.76 within window, well above 0.5. So a calibrated gate or a re-weighting can fix precision up to the ranking's limit. That limit is low, and §5.3 shows why.

### 5.3 Gates, and why a gate alone is not enough (box3d, clipgrid; [ ] = 95 % clip-cluster bootstrap; TRAIN values in parentheses)

| gate | confident slots | precision | recall |
|---|---:|---:|---:|
| 0.5, the model's (refcv6's rule) | 9,885 (3.40× targets) | 0.162 [0.143, 0.186] | 0.551 [0.499, 0.604] |
| 10/11, the tilt-corrected 50 % belief | 74 | 0.568 | **0.014** |
| **P = R: 0.719 (TRAIN: 0.713)** | 2,908 | **0.296** (TRAIN 0.315) | 0.296 |
| best F1: 0.687 | 3,688 | 0.269 | 0.341 (F1 0.300; TRAIN 0.326) |

- **AP@2 m BEV:** 0.210 eval, 0.230 train.
- **The calibrated 50 % gate kills recall.** The matched-slot presence rarely reaches 0.909, because each object's assignment mass is SPLIT across several slots (duplicates). The same mechanism leaves 25 % of the P = R false positives as duplicates.
- **The train-derived P = R gate (0.713) transfers to eval unchanged** (eval's own is 0.719; the bootstrap CI of the eval P = R gate is [0.69, 0.90]). The G-LIVE arm table (§7.2) reads it on every set.

**False-positive composition (box3d, clipgrid):**

| category | at 0.5 (8,283 FPs) | at P = R (2,046 FPs) |
|---|---:|---:|
| mislocalised: nearest visible target 2-5 m away | **45.1 %** | **54.9 %** |
| hallucination: no GT row of any kind within 5 m | **34.4 %** | 19.2 % |
| duplicate: the nearest target within 2 m is already taken | **18.0 %** | 24.7 % |
| on a row the field cut removed (≤ 5 m) | 2.0 % | 1.2 % |
| predicted outside the field | 0.5 % | 0.0 % |

- **False negatives at 0.5 (1,306):**
  - 792 have a slot within 2 m but below the gate (a presence miss);
  - 388 have no slot within 2 m (a localisation miss);
  - 126 have a slot that was claimed by another target.
- **Localisation vs presence:** Spearman(presence, centre error) = **−0.016** over 2,904 Hungarian pairs. 20.8 % of pairs are more than 2 m off, and 13.3 % are both more than 2 m off AND confident.

**Crowdedness (box3d, clipgrid), confident slots per window at 0.5 against targets:**

| K band | windows | confident at 0.5 | targets | precision / recall |
|---|---:|---:|---:|---|
| K 0 | 155 | 0.24 | 0 | – |
| K 1-2 | 116 | 4.0 | 1.39 | – |
| K 3-5 | 124 | 12.5 | 3.82 | 0.12 / 0.38 |
| K 6-10 | 78 | 34.4 | 7.50 | – |
| K 11-20 | 58 | 54.4 | 14.7 | – |
| K 21-40 | 20 | 76.9 | 24.0 | 0.23 / 0.72 |

On empty scenes the head is nearly silent. It over-reports in proportion to the crowd.

### 5.4 Per-class and per-range box errors (box3d, clipgrid)

**Class accuracy on Hungarian pairs within 2 m:**

| class | accuracy | n |
|---|---:|---:|
| automobile | 0.70 | 1,722 |
| person | 0.51 | 424 |
| rider | 0.62 | 78 |
| heavy_truck | 0.74 | 27 |
| bus | 0.63 | 8 |
| trailer | 0.50 | 8 |
| other_vehicle | 0.36 | 11 |
| protruding_object | 0.22 | 18 |

**The class tilt.** Predicted class among the 2,904 matched slots vs GT counts:

| class | predicted | GT |
|---|---:|---:|
| protruding_object | **420** | 19 |
| rider | 271 | 96 |
| trailer | 70 | 9 |
| stroller | 59 | 3 |
| automobile | 1,437 | 1,954 |
| person | 468 | 774 |

This is the class-level twin of the presence tilt: a class-weighted CE learns q_c ∝ w_c·P(c). The b1 weights this run trained with (digest `c9dbc340acf7d27e`, from the dump record) span **1,298:1**:

| class | weight | class | weight |
|---|---:|---|---:|
| automobile | 0.004 | protruding_object | 0.522 |
| person | 0.012 | stroller | 1.245 |
| rider | 0.097 | other_vehicle | 1.532 |
| heavy_truck | 0.214 | animal | 5.554 |
| trailer | 0.34 | | |
| bus | 0.48 | | |

It is the map audit's prior-correction rule, one head over.

**Recall at 0.5 by class:** automobile 0.63, person 0.38, rider 0.31, heavy_truck 0.55, bus 0.36, trailer 0.00 (n 9).

**Box error by range, Hungarian pairs, medians:**

| range | n | centre L2 | \|range err\| | lateral | \|Δl\| | \|Δw\| | yaw (mod 180°) | \|Δz\| | \|Δh\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-10 m | 227 | 0.68 | 0.45 | 0.36 | 0.22 | 0.06 | 8.4° | 0.05 | 0.04 |
| 10-20 m | 613 | 0.88 | 0.61 | 0.47 | 0.19 | 0.06 | 13.9° | 0.08 | 0.06 |
| 20-30 m | 580 | 1.01 | 0.61 | 0.53 | 0.18 | 0.06 | 14.0° | 0.10 | 0.06 |
| 30-45 m | 788 | 1.13 | 0.66 | 0.61 | 0.22 | 0.08 | 13.0° | 0.13 | 0.06 |
| 45-60 m | 680 | 1.26 | 0.70 | 0.70 | 0.24 | 0.08 | 12.1° | 0.18 | 0.08 |

The signed range error is ≈ 0 in every band (−0.09 to +0.09 m), so there is no range bias. Size, z and h are good. The failure is set-level, meaning which slot fires where, not box geometry.

**The agent head (planner tokens).** Clipgrid, the same picture:
- AUROC 0.860; P = R 0.263 at 0.744; AP 0.171;
- false positives at 0.5: mislocalised 45.7 %, hallucination 41.2 %, duplicates 11.3 %;
- Hungarian median error 1.33 m, 27.4 % of pairs more than 2 m off.

---

## 6. Task 4: training trajectory (`code/traj.py` → `raw/analysis/traj.json`)

Medians per step band, over the run's own rows. "Exchangeable" is the analytic loss if all 100 slots shared p\* at the observed K.

| steps | box3d presence | exchangeable at K | prior init (p = 0.05) at K | K per labelled window | box3d centre (m) | box3d cls | box3d total / loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-200 | 0.185 | 0.101 | 0.177 | 5.75 | 6.05 | 2.40 | 0.31 |
| 200-1k | **0.101** | **0.099** | 0.171 | 5.53 | 3.16 | 2.23 | 0.27 |
| 1k-5k | 0.088 | 0.093 | 0.153 | 4.94 | 3.00 | 1.99 | 0.34 |
| 5k-20k | 0.078 | 0.097 | 0.164 | 5.31 | 2.25 | 1.44 | 0.33 |
| 20k-34.5k | 0.069 | 0.093 | 0.155 | 5.00 | 1.67 | 0.97 | 0.32 |
| 34.5k-38.3k | 0.065 | 0.091 | 0.147 | 4.74 | 1.46 | 0.87 | 0.27 |

- **Presence did not collapse.** It rose above the prior level for the first ~200 steps (0.196 at step 100), reached the **exchangeable optimum within ~1,000 steps** (every slot the same tilted p), and then specialised slowly, to 28 % below exchangeable.
- In-run eval presence fell monotonically, 0.0838 (step 500) → 0.0619 (36,500). Every regression term kept falling to the end.
- **No box quality proxy was ever logged.** The only "conf" and "prec" keys are other heads'. **0 `ga_*` keys.** The field cut per the run's own counts: 387,498 prefilter → 63,230 visible (**83.7 %** removed).
- **Why the log could not show the defect.** A presence loss sitting 28 % below the exchangeable optimum reads as learning. It is fully compatible with 3.4× confident-over-targets at the 0.5 gate, because the tilt is part of the optimum. Only a count, a precision or a calibration key could have shown it (`raw/LOGGING_SPEC_BOX.md`).

---

## 7. Task 5: refcv7 recommendation (the fix, then the checks). Each check has an arm that MUST fail

### 7.1 The fix, ranked by the measured size of what it addresses

1. **Presence objective and decision rule** (addresses #1, and through the ranking #2 and #4):
   - **(a) Remove the tilt, or declare it.** Either
     - `NO_OBJECT_W = 1.0` with a positive-count or Σw normalisation, or
     - the DETR-family standard: a sigmoid focal loss (α 0.25, γ 2) normalised by the number of matched targets.

     Then σ(logit) is the calibrated belief. If any w0 ≠ 1 stays, the config must DECLARE the decision rule `logit − ln(1/w0)`, and every consumer (the gate, the agent-token soft gate, the eval, the video) must read it.
   - **(b) Make presence quality-aware:** the matched target becomes a localisation quality, e.g. q = exp(−d / 1 m) of the pair, in the style of Varifocal / IoU-aware classification.
     - MEASURED motivation: Spearman(presence, error) = −0.016, and 45-55 % of false positives are mislocalised 2-5 m. A quality target makes the ranking prefer well-placed slots, which directly attacks both the mislocalised and the duplicate categories.
   - **(c) Normalise presence like DETR** (Σw or #positives, not B·100). That gives a ~7× stronger presence gradient at K = 5, for the de-duplication that the set loss is supposed to provide.
   - **(d) Operating gate:** derive it on TRAIN windows by the P = R rule, stamp it in config.json, and apply it unchanged to eval and the video. refcv6's train-derived gate is 0.713; eval's own is 0.719.
     - **For the PI's refcv6 video today:** render at 0.713 instead of 0.5. MEASURED on clipgrid (box3d), that is 1.05 confident boxes per target instead of 3.40: **69 % fewer boxes**, precision 0.162 → **0.289**, recall 0.551 → 0.304 (`raw/analysis/glive_arms.json`).
2. **Target definition: VIS-1** (§2.4), addressing #5.
3. **Class decision rule:** the prior-corrected argmax (z_c − log w_c) under the class-weighted CE, or cap the weights. This addresses the rare-class over-prediction (protruding_object 420 predicted vs 19 GT). Same mechanism as the map audit.
4. **Logging:** `raw/LOGGING_SPEC_BOX.md` (count, precision/recall/AP at the gate, calibration, per class, grad reach), with mutation tests.
5. **No z correction for trucks** (§1.5). Keep protruding_object's airborne z as labelled.
6. **Renderer:** land the drawing patch (§1.6).

### 7.2 Checks (literal bands; each with its must-fail arm, measured on refcv6 where possible)

**G-LIVE-GATE (every in-run eval row, and the end-of-smoke eval).**
- Check: **conf_ratio = Σ(slots at or above the declared gate) / Σ(VIS-1 POSITIVES) ∈ [0.5, 1.5]**, pooled over the eval windows.
  - The upper end 1.5 is the literature package's bar (§7.3); this audit's first draft read 2.0.
  - The table below reads both bands.
- Red arm, measured on refcv6 38k at its gate 0.5 against the trainer's targets: **3.40 (box3d), 3.75 (agent)** on clipgrid. It must, and does, FAIL.

**G-LIVE-COUNT (smoke-able from step 1,000, because the exchangeable optimum is reached by ~1k steps).**
- Check: **Σ p̂ / Σ POSITIVES ∈ [0.7, 1.4]**, where p̂ is the declared calibrated probability.
- Red arm: the SAME checkpoint read with the tilt IGNORED (raw σ).

The measured readings for both checks are in `raw/analysis/glive_arms.json` and the table below:

| refcv6 38k reading (`code/glive_arms.py`) | box3d clipgrid | agent clipgrid | box3d inrun | box3d TRAIN | band | verdict |
|---|---:|---:|---:|---:|---|---|
| GATE: conf_ratio at **0.5** vs the trainer's targets (**RED arm**) | **3.40** | **3.75** | 3.73 | 3.56 | [0.5, 1.5] (or 2.0) | **FAILS both bands, as required** |
| GATE: conf_ratio at the **train-derived P = R gate** (0.713 box3d / 0.753 agent) vs targets | 1.05 | 0.92 | 0.86 | 1.00 | [0.5, 1.5] | pass |
| GATE: the same gate vs VIS-1 POSITIVES | 1.77 | 1.55 | 1.30 | 1.68 | [0.5, 1.5] | **fails** (except inrun): refcv6 fires on hidden objects; each passes the looser 2.0 |
| COUNT: Σ σ(logit − ln 10) / targets (correct map) | 0.94 | 1.03 | 1.05 | 0.95 | [0.7, 1.4] | pass |
| COUNT: Σ σ(logit − ln 10) / VIS-1 POSITIVES | **1.57** | **1.74** | **1.58** | **1.60** | [0.7, 1.4] | **fails**: refcv6 counts the hidden objects it was trained on |
| COUNT: Σ raw σ(logit) / targets (**RED arm**: the tilt ignored) | **4.58** | **4.81** | 5.55 | 4.76 | [0.7, 1.4] | **FAILS, as required** |

- P / R at the train-derived gate (clipgrid): 0.289 / 0.304 for box3d and 0.268 / 0.247 for the agent head. The gate transfers from TRAIN to eval with no retuning.
- The refcv7 launch reads both checks on **VIS-1 POSITIVES**. refcv6 fails G-LIVE-COUNT on that definition (1.57-1.74), which is the intended sensitivity: a head trained on hidden targets over-counts what the camera can see.

**G-BOX-OVERFIT.** `raw/PREREG_G_BOX_OVERFIT.md`, registered before any run.
- Frame set: 16 TRAIN frames from 16 clips, 113 VIS-1 POSITIVES, md5 `b291404c…`, selected from labels only.
- PASS: AP ≥ 0.90; P and R ≥ 0.90 at the declared gate; count within ±10 %; median centre error ≤ 0.30 m; class accuracy ≥ 0.90.
- Must fail: `memory_zeros` and `presence_w0`.
- Controls: GT-as-slots = 1.0, constant presence = 0.5, the VIS-1 recount equals the file.

**VIS-1 unit test + red arms:** §2.4. **Logging mutation tests:** `raw/LOGGING_SPEC_BOX.md`. **Renderer patch test:** 5/5 on Thor, including its red arm.

### 7.3 Literature reconciliation (read `…/2026-09-27-box-head-literature/RESULT.md` at 02:3x, after it landed)

**The two packages AGREE.** Its §6 said the FP decomposition, the AUROC and the visibility result would re-order its R1-R9. This audit's measurements do the re-ordering as follows.

**Its §6 decision rule.** "High AUROC with a saturated gate → mostly the decision rule; AUROC near 0.5 → the learning signal." MEASURED: AUROC **0.855** (0.73 within window) with a saturated gate. So its **R1(d) + P0** (declared decision rule + detection metrics) come FIRST.
- But the best achievable operating point is F1 0.30 / AP 0.21. It is the same on TRAIN (0.33 / 0.23), so the head underfits. **Its R1(a)-(c) (focal presence, focal matching cost, prior 0.01) and R2 (per-layer loss: −11.5 AP without it at 3 layers, S27) are therefore also needed.** A gate alone caps at F1 0.30.

**Duplicates** measure 18 % of FPs at 0.5 and 25 % at P = R. That supports its **R1(b)**: the geometry-dominated matching cost, where the confidence term buys 1.6 m vs 14.9 m in the mmdet3d recipe (its `matching_cost_balance.json`). It also supports **R6** (DN).

**Mislocalised 2-5 m** measures 45-55 % of FPs, with Spearman(presence, error) = −0.016. That supports its **R5** (geometry-grounded feature access) and adds one item it did not list: a quality-aware presence target (this audit's §7.1 1(b); in the same family as Sparse4D v3's centerness/yawness).

**Filtered-GT FPs** (detections on objects the field cut deleted, at 5 m or less) measure only **2.0 %** at 0.5. The "visible-beyond-the-box becomes a negative" half of its **R3** is therefore minor on our data. The hidden-target half is major: 26 % of representative targets and 56 % on the PI's clips are under 10 % visible. **R3 = VIS-1 with IGNORE semantics is the same rule** (§2.4). Our thresholds: 0.30, 100 px, 0.05.

**Its R4 (query count)** is consistent. Budget drops on representative eval: 4 of 2,908 targets. Under VIS-1 the crowded pool falls from 26.2 to 8.5 positives per window. **Re-rule M17 AFTER VIS-1, on TRAIN**, as it says.

**Its R8 (class weights)** is confirmed by measurement. The 1,298:1 weights over-predict rare classes: protruding_object 420 predicted vs 19 GT among matched slots. Either its no-weight / √-inverse, or this audit's prior-corrected argmax, fixes the decision; the loss form is its call.

**Its R9 (planner → presence knockout arm)** is consistent with hop D6 (§3). That effect is UNMEASURED; the agent head's tilted calibration is intact (ECE 0.0036).

**Its "never both" rule is adopted.** Either keep BCE 0.1 and DECLARE the ln 10 correction everywhere, OR adopt focal with a calibration-split gate. Stacking the two double-corrects. The tilt-corrected gate / P = R gate of §7.1 1(d) is the zero-retrain readout for refcv6@38k ONLY.

**Its §5 bar** ("confident / visible ≤ 1.5 on train-density windows") and this audit's first-draft G-LIVE-GATE band [0.5, 2.0] were the same check at two widths. **Adopted in §7.2: [0.5, 1.5]** (the literature's tighter upper end), read against VIS-1 POSITIVES. The refcv6 red arm reads 3.40, so it fails either band.

---

## 8. Task 6: the GT validation (the video agent's package), folded in

From `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-gt-validation/RESULT.md`, not redone:
- 72,687 GT boxes, 69.1 % removed by the field filter, 22,432 targets (26.2 per window) on 855 windows of 5 clips;
- sizes plausible (62 outliers, 0.09 %);
- 145 duplicate pairs, mostly neighbouring persons;
- bottom median −0.07 m within 20 m.

Two items are extended here:
- **Visibility** (§2): their §6 hypothesis, now MEASURED. My z-buffer target count equals theirs (22,432), which is a control.
- **heavy_truck −0.63 m** (§1.5): a range effect of trucks 90-150 m away. It is not a label defect and not a target.

**Ranking the GT side against the loss side, by measured size.** On representative eval, the loss-side decision rule accounts for ~71 % of the confident boxes. GT invisibility accounts for 3.4 % of the confident boxes as hidden-target "hits" (20.7 % of hits). But GT invisibility dominates the PI's crowded clips (56 % of targets under 10 % visible) and corrupts the training signal everywhere (26 % of targets).

---

## 9. Not done, uncertain, or departures

- **The planner → presence gradient (D6)** was traced, not measured. The agent head's tilted calibration is intact (ECE 0.0036), so its effect is at most small. **UNVERIFIED.**
- **LiDAR checks** ran on ONE clip (73495082f98b). Raw LiDAR for other B1 clips is not on Thor, and the dev box, which has DracoPy, stayed below its RAM floor. The 09-13 package's corpus result (the LiDAR BEV registered to the box join at 3.23× mirror / 4.29× marginal over 134 eval clips) is consistent with, but not a re-measurement of, §1.4.
- **The z-buffer is an UPPER bound on visibility** (§2.2). A static-occluder mask, e.g. SAM3 building/vegetation masks or LiDAR first-hit, would lower it.
- **Departures.**
  - The dumps were rolled on Thor with the battery's `refcv6_loader.py`, copied into `code/` (md5 `5f43cd81…`, `remap={}`: the run's own Thor paths). `--trunk-compile` was dropped (the loader's rule: the module tree is unchanged).
  - The TRAIN dump pointed `--eval-cache/--eval-labels/--speed-max-sidecar-v6-eval` at the train cache (the A6 train-roll override).
  - The banked `ov_audit.py`, `ov_p2p5.py`, `ov_probe.py` and `vis_zbuf.py` differ from the Thor-run copies only in removing a literal 8-hex clip prefix from their argument defaults. The prefix-floor rule of `tools/clipid_scan.py` now makes it a required runtime argument.
- **The literature reconciliation** is done (§7.3), against the package as landed at 02:02. Its published numbers were not re-read here: they are its PUBLISHED / ANALYTIC classes, cited, not re-verified.

---

## 10. Deliverable manifest

See `LANDING_READY.txt` for the landing list.

| artifact | where | only one place? |
|---|---|---|
| `RESULT.md` | repo (D: working tree): `TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit/RESULT.md` | D: only, until landed |
| `LANDING_READY.txt` | D: package | D: only |
| `raw/PREREG_G_BOX_OVERFIT.md`, `raw/LOGGING_SPEC_BOX.md` | D: package | D: only |
| `raw/overlay_audit/` (6 PNG, `ov_audit_*.json`, `ov_boxes_*.json`, `ov_p2_*.json`, `ov_p5_*.json`) | D: package; also Thor `/home/nvidia/bha_2325/ov/out/` | two places |
| `raw/visibility/` (the vis_* summaries and per-box JSONs, `vis_rules.json`, 3 PI-frame panels, `vis_panels.json`, `gbo_frameset.json`) | D: package; also Thor `…/ov/out/` | two places |
| `raw/analysis/` (`analyze_{inrun,clipgrid,train}.json`, `traj.json`, `gt_zcensus*.json`, `tp_vis_*.json`, `glive_arms.json`) | D: package; also Thor `/home/nvidia/bha_2325/out/` | two places |
| `raw/dumps/` (31 `*_chunk*.pt` ≈ 25 MB of per-slot dumps: smoke 1, inrun 4, clipgrid 18, train 8; plus 4 `*_record.json`) | D: package; also Thor `…/out/` | two places. ⚠️ Large binaries: land or keep on D: at the Master Mind's discretion |
| `raw/logs/` | D: package; also Thor | two places |
| `code/` (`bha_dump`, `bha_analyze`, `ov_*`, `vis_*`, `gt_zcensus*`, `gbo_select`, `tp_visibility`, `glive_arms`, `traj`, `refcv6_loader` copy) | D: package; the Thor run copies in `/home/nvidia/bha_2325/{code,ov}` | two places |
| `code/patch_draw_cuboid/` (2 patched tools, the diff, the test) | D: package; Thor `/home/nvidia/bha_2325/patch/`; the map-video tool is also already in `taniteval/tools/` (someone else's copy) | several places |
| Thor work dir `/home/nvidia/bha_2325/` (the tree, dumps, logs) | Thor | everything banked above; the dir can be deleted after landing |
