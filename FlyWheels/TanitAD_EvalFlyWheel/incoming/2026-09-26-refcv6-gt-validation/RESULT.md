# RESULT — refcv6 agent GROUND TRUTH, validated: a GT-only reel and sanity stats

EvalFlyWheel · 2026-09-26/27. This answers the PI after the step-38,000 boxes video, verbatim: *"the
detected objects bounding boxes are completelly messy, the model is detecting boxes where is nothing.
We need to understand what happening, and review training, architecture, wiring, training signal flow
and validate gt. let begin by the last one and have a second video image in our video with onyl the gt
boxes amd their classes"*.

## Headline

**The GT is plausible in class, size and placement near the car. The three clips the PI watched are
the extreme tail of GT density in the eval set.**

MEASURED, over 855 windows on 5 clips.

**Density.** The typical eval clip has 2.65 trainer targets per labelled window (the median over all 139
eval clips; p90 13.6). The three boxes-video clips have 21.0, 24.5 and 80.3.

**What the GT is.** 72,687 GT boxes, all classes known, all 3-D labelled, and none truncated by the
397-row pad.
- Sizes: 62 boxes (0.09 %) fall outside literal class norms, all of them marginal.
- Duplicates: 145 same-frame pairs with BEV IoU > 0.5, mostly person-person in crowds.
- Bottom z: no systematic sink near the ego. At 0-20 m the median bottom is **−0.07 m** (n 18,830).

**What the trainer removes.** The trainer's own filter removes **69.1 %** of the GT, and 43.5 % of all
GT is behind the ego. The model is trained on the remaining 30.9 %.

**Clip `0191487845ef`.** 90 % of its 80 targets per window are **pedestrians**. The crowd the model
paints there is largely real and LiDAR-labelled. What the GT alone cannot settle is **visibility**: the
trainer's filter is horizontal-FOV only, so a pedestrian hidden behind a nearer one is still a target.
That is the next lever (§6).

**The files.** The videos are on the D: working tree only, because `*.mp4` is gitignored. Both decode
back clean.

| file | size | format |
|---|---|---|
| `D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-gt-validation/raw/refcv6_gt_validation_step38000.mp4` | 61,881,657 B (59.0 MiB) | 1920 × 1200, 980 frames, 98.0 s at 10 fps; **re-rendered 2026-09-27 01:38 with the box-head audit's drawing patch** (§1b) |
| `.../raw/refcv6_gt_validation_step38000_compact.mp4` (the PI copy, 8 MB or less) | 6,930,078 B (6.61 MiB) | 1280 wide, CRF 34 (the loop tried 30 / 32 / 34 until it was 8 MB or less), same 980 frames, patched |
| `.../raw/stills/` | 4 PNG | 1920 × 1200 |

The stills:
- `still_9f8bedcfb9de_w104.png`, the PI's frame
- `still_0191487845ef_w064.png`
- `still_34765c024267_w108.png`
- `still_22d47ae7e052_w086.png`, an ordinary clip

**Tool.** `taniteval/tools/render_refcv6_gt_validation.py` (md5 at the patched run `e88bf54a…`: the audit's patch plus two caption lines; the first, pre-patch run was `4866cb6c…`), with 11 literal
unit tests in `taniteval/tests/test_render_refcv6_gt_validation.py`.

**No model forward.** No network is built, loaded or run. The only model-derived object is the camera
bank, `refc_v3_train._build_rig_camera(cfg, args)`, which is built from the run's config with no weights.

## 0. Pre-registered BEFORE the run (written 2026-09-26 23:27 Berlin, before any extra clip was seen)

**Clips.** The three clips of the boxes video are fixed: `0191487845ef`, `9f8bedcfb9de` and
`34765c024267`. Two LESS crowded eval clips are added by this rule:

- **Eligible:** eval clips with at least 1 agent-labelled window and a per-clip extrinsics row.
- **Score:** the mean number of TRAINER TARGETS per labelled window. A trainer target is the 2-D GT at
  each eval window's NOW frame after `refc_agents.visible_target_filter`.
- **Take** the 2 eligible clips (not among the three fixed) whose score is CLOSEST to the median score
  over all eligible clips. Ties go by sha12.

**Result of the rule** (`raw/clip_selection_gt.json`). All 139 eval clips are eligible. The median
score is **2.649**, and the quantiles are:

| p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|
| 0.056 | 0.743 | 2.649 | 6.205 | 13.599 |

The extras are `22d47ae7e052` (2.649) and `59d98b34ae8b` (2.754).

**Stills.**
- Clip `9f8bedcfb9de`, window 104. This is the frame of the PI's screenshot.
- Clip `0191487845ef`, window 64.
- Clip `34765c024267`, window 108.
- The middle window of the first extra clip.

**Literal class norms** are in the tool (`NORMS`), authored independently and not fit to this data. The
PI's two examples are pinned exactly: a `person` longer than 2 m, and an `automobile` taller than 3 m.

## 1. What each frame shows

| panel | content | depends on the camera model? |
|---|---|---|
| **camera A** (top) | The step-38,000 boxes video's camera image, cropped from its SAVED frame: model boxes in class colours, GT targets in green. For the 2 extra clips, which that video never rendered, it is the RAW camera, labelled so. | yes |
| **camera B** (below) | **GT ONLY.** Projected cuboids in class colours, each labelled class · range · track #. Trainer targets are solid. GT the trainer's filter REMOVES is dashed and labelled `filtered: <reason>`. A row without a 3-D label is drawn as its ground footprint; there are 0 such rows here. | yes |
| **BEV A** | The boxes video's detection panel (model + GT), cropped. | no |
| **BEV B** | **GT ONLY**, wide field x −20…80 m, y ±32 m, with the trainer's target region outlined in green. Class colours and a 3-letter tag. **Judge GT placement here.** | **no** |
| text | The frame's GT census: boxes, targets, filtered by first reason, per class, and the sanity flags. | no |

### 1b. The drawing patch (box-head audit, 2026-09-27): the reel is RE-RENDERED with it

**The audit's verdict.** Evidence class: INHERITED from the box-head audit, not re-verified here. Source:
its patch docstring in
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit/code/patch_draw_cuboid/`.
- **The projection was right.** It was verified independently by a raw-mp4 re-render, by LiDAR through
  its own extrinsic, and by the SAM3 camera model.
- **The PI's "one surface" boxes were a DRAWING defect.** An edge was drawn only if BOTH its corners
  were in the frame, so a cuboid crossing ±60° lost its border edges. Example: bus #39 in
  `9f8bedcfb9de`, window 104. Edges were also drawn as straight chords, although the cylinder bends
  them.
- The audit also finds **heavy_truck GT boxes are sunk: a label defect.** §3's heavy_truck median bottom
  of −0.63 m is that defect as seen from the stats.

**What was done.**
- The audit's two patched files were installed **verbatim** into `taniteval/tools/`.
  - `render_refcv6_map_video.py`, md5 `e5890efb…`: adds `cuboid_edge_runs`, which projects each edge as
    32 sampled 3-D points and bisects its end onto the frame border.
  - `render_refcv6_gt_validation.py`, md5 `62f6d2a0…`: camera B draws through it, with a
    phase-continuous dashed polyline for filtered GT.
  - Before installing, each was checked to be a pure SUPERSET of my current file: 53 diff lines, every
    one inside the drawing code.
- On top of the patch I changed two text lines only:
  - the camera caption now reads *"3-D overlay verified independently against LiDAR; heavy_truck GT boxes
    are sunk (label defect)"*, as the Master Mind directed;
  - camera A's label now says it was **drawn with the PRE-PATCH edge rule (GPU re-render pending)**.
    Camera A is a crop of the boxes video's saved frames, and redrawing its model cuboids needs the
    model forward. A6 holds the GPU.
- The unit tests still pass: 11 GT-tool tests and 24 across both renderers.
- **Superseded, pre-patch outputs** (kept for before/after): `raw/superseded_pre_patch/`, holding the 4
  stills, the compact mp4, `render_record_pre_patch.json` and `gtval_pre_patch.log`. The pre-patch full
  mp4 is on C: only, in `C:/Users/Admin/qland/work/gtval/superseded_pre_patch/`.
- **The GT numbers do not change.** The drawing never fed a statistic. Every `gt_stats_*.json` is
  regenerated by the same code on the same data.

## 2. GT sanity per clip (MEASURED; `raw/gt_stats_<sha12>.json`, pooled `raw/gt_stats_pooled.json`)

| clip | role | windows (NO_LABEL) | GT per window mean [max] | targets per window mean [max] | removed by the filter | persons among targets |
|---|---|---|---|---|---|---|
| `0191487845ef` | boxes video | 171 (0) | 224.8 [395] | 80.3 [120] | 64.3 % | 12,396 / 13,726 (90 %) |
| `9f8bedcfb9de` | boxes video, the PI's frame | 171 (0) | 87.8 [127] | 24.5 [32] | 72.1 % | 1,835 / 4,192 |
| `34765c024267` | boxes video, turn 75° | 171 (0) | 76.1 [107] | 21.0 [26] | 72.4 % | 226 / 3,590 |
| `22d47ae7e052` | median rule | 171 (0) | 28.8 [41] | 2.65 [9] | 90.8 % | 192 / 453 |
| `59d98b34ae8b` | median rule | 171 (0) | 7.6 [12] | 2.75 [8] | 63.5 % | 116 / 471 |
| **pooled** | | **855 (0)** | **85.0 [395]** | **26.2 [120]** | **69.1 %** | 14,765 / 22,432 |

**Filter reasons, pooled.** Each box is counted once, by its first reason, as a fraction of all GT:

| behind the ego (x < 0) | outside the 120° field | beyond 60 m ahead | \|y\| > 16 m |
|---|---|---|---|
| 43.5 % | 9.3 % | 9.4 % | 7.1 % |

The rule is `refc_agents.visible_target_filter` (`refc_agents.py:410-433`). The decomposition reproduces
the trainer's own mask exactly (§4).

**Classes, pooled GT:**

| person | automobile | rider | bus | heavy_truck | protruding_object | trailer | animal | other_vehicle | unknown |
|---|---|---|---|---|---|---|---|---|---|
| 43,231 | 25,803 | 1,837 | 755 | 639 | 191 | 171 | 35 | 25 | **0** |

**Sizes, per-class medians in metres** (`sizes_per_class`, L × W × H):
- automobile 4.05-4.13 × 1.92 × 1.58-1.60
- person 0.60 × 0.66-0.69 × 1.66
- bus 11.8-12.1 × 3.0-3.1 × 3.3-3.4
- heavy_truck 8.2-10.5 × 2.9-3.2 × 3.2-3.9

62 of 72,687 boxes are outside the literal norms (0.09 %). Examples: bus width 3.26 m against 3.2 m, and
heavy_truck width 3.22 m. LiDAR cuboids include the mirrors.

**Duplicates.** 145 same-frame GT pairs have rotated BEV IoU > 0.5, and 18 of them have both boxes as
trainer targets. They are mostly person-person on crowded sidewalks (`0191487845ef`: 68 pairs), so they
may be real neighbours rather than double labels. There are 0 duplicate track ids within a frame.

**Range of the trainer targets, pooled:**

| 0-10 m | 10-20 m | 20-30 m | 30-40 m | 40-60 m | 60-80 m |
|---|---|---|---|---|---|
| 1,275 | 4,921 | 5,149 | 4,401 | 6,547 | 139 |

The 139 at 60-80 m sit at the decode-box corner, where the range from the rig origin exceeds 60 m while
x is still 60 m or less.

## 3. Box-bottom height against the rig's road plane (the Master Mind's request)

**Definition.** bottom = join3d `cz − h/2` (`agent_cuboid_gt.base_from_centre`, `:430`), in the rig
frame of `obstacle.offline` (+z up). The road plane is z = 0 only where the ground is flat and level
with the rig.

| range from the rig | n | median | p10 | p90 |
|---|---|---|---|---|
| 0-20 m | 18,830 | **−0.069 m** | −0.373 | +0.307 |
| 20-40 m | 22,494 | −0.116 m | −0.634 | +0.425 |
| 40 m and beyond | 31,363 | −0.238 m | −1.364 | +0.640 |

- **There is no systematic −1 m sink near the ego.** The per-clip medians at 0-20 m are −0.105,
  −0.075, −0.072, +0.154 and +0.249 m.
- **With range, the median moves per clip and in BOTH directions.** `22d47ae7e052` reads
  +0.15 / −0.37 / −0.87 m and `59d98b34ae8b` reads +0.25 / +0.34 / +0.36 m. That is the signature of
  road slope or vehicle pitch in the rig frame, not a constant label offset. A 0.5° pitch alone moves
  z by 0.35-0.52 m at 40-60 m.
- **30.6 %** of all GT have \|bottom\| > 0.5 m, mostly at 40 m and beyond.
- **Flag for the geometric review, since CONFIRMED by the box-head audit as a label defect (INHERITED):**
  `heavy_truck` has a pooled median bottom of **−0.63 m** (n 639, from 3 clips: −0.97, −0.32 and
  −0.76 m).
- `protruding_object` has a median bottom of +1.17 m, which is airborne by definition.

**Is `cz` a centre or a base?** The raw join3d `cz` medians are banked too (`raw_join3d_cz_per_class`).
For automobiles they are 0.61-0.81 m. That reads as a CENTRE: bottom ≈ 0 near the car. Read as a base,
it would float cars 0.6-0.8 m above the road.

Full per-class distributions, in 0.25-0.5 m bins, are in `box_bottom_cz_minus_h_over_2.per_class_hist`
and `.pooled_hist`.

## 4. Controls (each must read a known value; all passed)

| control | expected | measured | n |
|---|---|---|---|
| reason decomposition vs the trainer's `visible_target_filter` mask | 0 mismatches | 0 | 855 windows |
| model `RigCamera` vs an independent implementation of `render_refcv3_video.CylProjector`'s formula (3-D lift), on every drawn corner | ≤ 0.01 px, refuse otherwise | 1.3e-13 px | 264,506 corners |
| BEV B pixel → metres round trip | ≤ 0.5 px | 3.1e-13 px | 72,687 boxes |
| `bev_iou(box, box)` | 1 | 1.0 | every window with GT |
| join subsets reproduce the boxes render's full-file load of the 3 fixed clips | 600 2-D records; 600 3-D lines; 78,336 3-D agents | 600; 600; 78,336 | 3 clips |
| eval-only 3-D subset size | = the eval join's 26,394 lines (`AgentJoin3D` docstring) | 26,394 lines over 139 clips | |
| unit tests (literal expectations) | pass | 11 / 11 (GT tool); 24 / 24 across both renderers | |

⚠️ **The projection control shows AGREEMENT, not CORRECTNESS.** Both sides read the same per-clip
extrinsics table with the same cylindrical formula. The CORRECTNESS evidence is the box-head audit's
independent LiDAR, raw-mp4 and SAM3 check (§1b, INHERITED).

## 5. Departures and run history (stated, not hidden)

1. **The joins are eval-only LINE SUBSETS** of the kit's train+eval files:
   - 2-D: `b1_train_plus_eval_agents.jsonl.xz`, md5 `0c31a3a6…`
   - 3-D: `b1_train_plus_eval_agents_3d.jsonl.xz`, md5 `401c9c31…`
   - Every kept line is byte-identical and selected by its own `clip_id` against the eval cache's clip
     set: 26,394 of 875,657 lines.
   - Why: a full xz parse did not finish in 7 min under tonight's contention; it took 77 s / 103 s on a
     quiet box.
   - Evidence: `raw/logs/extract_record_{2d,3d}.json` and `code/extract_eval_joins.py`. The tool refuses
     a subset whose extract record did not pass its known-value control.
2. **The RAM floor for THIS render was the Master Mind's**: start at 5.5 GB or more on 3 samples 30 s
   apart, with A6 SUSPENDED by the Master Mind for the duration, and a chain-wide 4.0 GB abort watchdog
   (`code/ram_watchdog.py`).
   - The first full run started with 8.08 / 7.47 / 7.16 GB free and saw a minimum of 4.76 GB.
   - The BANKED, patched run (01:32-01:38) started with 6.97 / 6.76 / 6.68 GB and saw a minimum of 5.48 GB.
3. **Four runs besides the banked one**, all in `raw/logs/attempts_aborted_and_failed.txt`:
   - Attempt 0 was ABORTED by the 4.0 GB watchdog while parsing the full join, because A6 had grown.
   - Attempt 1 was stopped by me when the floor changed; it never passed its gate.
   - Attempt 2 FAILED on a code defect: `lookup_track_ids` returns an array, so `x or []` is ambiguous.
     It is fixed.
   - The first complete run (00:31-00:37) is SUPERSEDED by the patched run. Its outputs are in
     `raw/superseded_pre_patch/`.
   - Patched run 2 was stopped by me after its stills, to label camera A as pre-patch.

   Apart from the superseded folder, none of them wrote a frame or a number that was banked.
4. **Camera A for the 2 extra clips is the RAW camera.** The boxes video never rendered those clips, and
   the GPU was held by A6.

## 6. What this does not show, and the next lever (Rule Zero)

- **Visibility.** Whether each target can actually be SEEN by the camera is not measured here. The
  trainer's filter is horizontal-FOV only, and its own docstring says *"no vertical, hood, or
  inter-agent occlusion"*.
  - The Master Mind assigned this to the **box-head audit (its task 0b)**, and this package's pass was
    DROPPED to avoid a duplicate.
  - The tool keeps an unrun `--visibility-only` mode: an angle-space z-buffer from the camera centre,
    independent of the lens model. It is covered by 3 literal unit tests, and no number from it is
    reported.
- **The model side is not in this package.** The boxes video measured that the model's own 0.5 gate
  lets 71-99 of 100 slots through per window (`../2026-09-26-refcv6-boxes-video/RESULT.md`). That is
  the training-signal review.
- **No interval.** Five clips are five episodes, and the windows are correlated.
