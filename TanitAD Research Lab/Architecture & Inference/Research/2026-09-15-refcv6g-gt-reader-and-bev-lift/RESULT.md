# refcv6-grounded, the first two zero-GPU pieces: a SAM3 map GT reader and the BEV lift geometry. Both are built and tested, and the lift is registered to real pixels. Nothing is wired in.

**Date:** 2026-09-15 (Europe/Berlin)
**Agent:** Architecture & Inference implementation agent
**Worktree:** `C:/Users/Admin/tanitad-wt-refcv6g`, detached at `d4e8bc0` (tip of `agent/arch-inf-20260803`). Files are staged, not committed.
**Design:** `Project Steering/REFCV6_DESIGN_GROUNDED.md` §2.2–2.3, §4, §7 (C7, C8). It awaits PI approval, so the trainer (`stack/scripts/refc_v3_train.py`) and the model (`stack/tanitad/refs/refc.py`) were **not touched**.
**Compute:** dev-box CPU only.
**Tier:** none. These are a label reader and a geometry; no planner metric family applies.

Evidence classes: **MEASURED** means ours, with the artifact under `raw/`. **INHERITED** means cited from its primary source. **ESTIMATED** is marked as such. Clip ids appear only as sha12 = sha256(clip_id)[:12].

## 0. Headline

| piece | outcome |
|---|---|
| `stack/tanitad/data/semantic_map_gt.py` | Opens a clip's `tanitad.sam3_map_gt/2` file. Returns `cart` [N,9,120,64] float in [0,1] and `seen` (share ≥ 0.5). **Refuses** a wrong schema, grid, channel order or array, a clip-identity mismatch, an out-of-range, negative or non-integer frame index, and time misalignment > 1 ms. `coverage()` reports readable / absent / **inconclusive**; an unreadable file or root is never counted as absent. `check_pose_alignment()` verifies the frame axis **without timestamps**. |
| `stack/tanitad/models/bev_lift.py` | Per-clip `grid_sample` grids for 120×64 cell centres × heights {0, 0.5, 1.5, 2.5} m, through the clip's extrinsics into the 256×640 cylindrical cache frame, then onto the stride-16 map. The validity mask is: in front ∧ inside 120° ∧ inside rows, with an optional observed-pixel term. `BEVLift`: sample → concat heights → 1×1 conv → + learned "unobserved" embedding. **180,480 parameters** (MEASURED, `python -m tanitad.models.bev_lift`). |
| tests | **35/35 pass, 0 skipped**, real data included (`raw/pytest_new_tests.txt`). **8/8 re-introduced defects go RED** and the control stays green (`raw/mutation_runs.json`). |
| (d) validity vs LiDAR `cart_cam_vis` | **134 clips** (62 rig A, 72 rig B): IoU min **0.99677**, median **0.99912**. **922/922** disagreeing cells lie within 1 cell of the field boundary. The lift **re-emulates `cam_vis` exactly on 134/134** (details in §3). ⚠️ This reference is **not independent**, and its IoU is blind to a mirror or a 2° error (§3). |
| independent registration (SAM3 camera model) | On real frames of 3 clips (all rig B), lane-line AUC peaks at column shift **−1 / 0 / −2 px** and row shift **0 / 0 / +1 px**. The mirror is lower on **3/3** (0.585 vs 0.680, 0.415 vs 0.533, 0.370 vs 0.597) (`raw/pixel_registration.json`). |
| (e) real alignment on v2ep episodes | 3/3 clips with GT + v2ep + camera timestamps: `t_img` reproduced **exactly (0 µs)**. Pose residual after the `v·(t_img−t_query)` correction is **≤ 2.3 mm**, and a one-frame shift is ≥ 0.57 m. Every shift is refused (`raw/real_alignment.json`). |

### ⛔ Escalations for the Master Mind (integration and decisions; nothing here is wired)

1. **Trainer hook (model change, awaits approval).**
   - `ResNetEncoder.forward` returns only the stride-32 map (`refc.py:1273-1277`). The lift needs the stage-3 (stride-16) output, so `refc.py` must expose it.
   - Label frame = stacked row + `n_stack − 1` (`semantic_map_gt.raw_frame_index`).
   - Map loss only on `seen`.
   - Build geometry per batch (2.3 ms/clip CPU, MEASURED) rather than precomputing all clips (276 KB/clip ⇒ ~1.3 GB for 4,719, ESTIMATED).
2. **Gate C7 wiring.**
   - v2ep payloads carry **no timestamps**. C7 must use either `check_pose_alignment(payload["poses"])` (no timestamps, MEASURED on 3/3) or the camera `timestamps.parquet` on the training box (availability UNVERIFIED).
   - Coverage must be read at **window** level, with verdict COMPLETE.
   - ⛔ **Eval exclusion must come from `stack/tanitad/data/v72_eval_clip_digests.json`, never from the GT's `source.split` field** (§6.1).
3. **Gate C8 wiring.** The pixel-registration instrument (`code/refcv6g_checks.py pixreg`) and the lift-based `cam_vis` re-emulation exist. C8's thresholds and its 20-frame sample must be pre-registered before use.
4. **Decisions.**
   - (a) Observed-pixel term on or off, and its source: intrinsics mask vs frame content. They differ on 26/72 rig-B clips by up to 8,916 px (§6.3).
   - (b) Keep the REF-C feature-centre convention (offset 0, §1.3). Any trunk swap must re-derive it.
   - (c) Soft-CE target normalisation: channel sums are 253–257 / 255.
5. **SAM3 GT coverage today is 3 / 4,719 clips**: the local copy of `SEMANTIC_MAPS_MANIFEST.json`, generated 2026-09-15 08:01+0200, shows 4,716 PENDING. **C7 would refuse now.** HF was not queried for later publishes.
6. **Doc conflict.** The brief calls the LiDAR `cam_vis` grids an "independent geometry reference ... a different author's camera model". They were projected through the **same** `rig_projection.RigCamera.from_extrinsics(FrontWideExtrinsics)` the lift reuses (`…/2026-09-13-bev-lidar-corpus-and-head/code/p2_build_corpus.py:197-199,215,240`).

## 1. What was built

### 1.1 Reader: `stack/tanitad/data/semantic_map_gt.py`

| API | contract |
|---|---|
| `open_clip(root, clip_id)` | `<root>/semantic_maps/gt/<clip_id>.sam3mapgt.npz`, validated tier only. It reads the `.npy` headers without decompressing, then checks `meta_json`: schema, frame `rig`, grid 120×64 @ 0.5 m, the exact 9-channel order, `fraction_scale` 255 and `source.clip_sha12`. `t_img_us` must be non-decreasing. `allow_pickle=False`. |
| `ClipMapGT.read(frame_idx, frame_t_us=None, tol_us=1000)` | RAW v2ep indices. Returns `MapFrames(cart [N,9,120,64] f32, seen [N,120,64] bool, t_img_us, frame_idx)`. `seen` is computed in integers as `2·(255 − not_seen) ≥ 255`. |
| `check_times` / `check_pose_alignment` | A 1 ms check against the caller's timestamps. A pose-content check against the v2ep payload's raw `poses` (verdict ALIGNED or INCONCLUSIVE; raises on misalignment). |
| `coverage(clip_ids, root)` | Readable / absent / inconclusive. Absence needs two probes: the GT dir is listable **and** the file stat is ENOENT. Reports lower and upper fractions and a verdict, keyed by sha12. |
| `episode_frame_times_us(t_cam)` | A restatement of `stack/scripts/v2_compressed.py:115-121`, the `_resampled` grid. That function decodes video, so it cannot be called for times alone. Refuses non-µs clocks. |

### 1.2 Lift: `stack/tanitad/models/bev_lift.py`: camera-model source

No projection is re-derived. Every crossing reuses the code that built the pixels:

| step | source (file:line) |
|---|---|
| quaternion → `R_cam_to_rig` | `stack/tanitad/data/physicalai.py:258-265` (`FrontWideExtrinsics.rotation_cam_to_vehicle`) via `stack/tanitad/data/rig_projection.py:218-232` (`RigCamera.from_extrinsics`) |
| rig → camera (+x right, +y down, +z boresight) | `rig_projection.py:254-260`: `p_cam = Rᵀ(p − t)` |
| camera → cache pixel | `rig_projection.py:272-314`: `col = (W−1)/2 + f_ref·atan2(x,z)`, `row = (H−1)/2 + f_ref·y/hypot(x,z)`. This is the exact inverse of `stack/tanitad/data/calib.py:906-924` (`cylindrical_rays`), the ray fan `calib.py:937-998` (`cylindrical_grid` / `cylindrical_rectify`) used to build the cache, called from `stack/scripts/v2_compressed.py:45-50`. Index-coordinate `(W−1)/2` is the design's continuous `W/2`. |
| cell centres | `stack/tanitad/data/bev_raster.py:114-120,310-320`. Row 0 is x ∈ [0, 0.5) m; col 0 is y = −16 m (RIGHT). This is the SAM3 GT's own grid. |
| frame | `calib.py:1247-1248` `PHYSICALAI_WIDE120_256x640` (f_ref 305.5774907364391). This matches the v2ep payload's `frame` dict on all 3 clips (MEASURED, `raw/real_alignment.json`). |

**Validity:** z_cam > 0; col ∈ [−0.5, W−0.5], i.e. |azimuth| ≤ 60.000°; row ∈ [−0.5, H−0.5]. Optionally, the nearest pixel must be observed. The per-clip cy needs no term: the cylindrical frame is centred on each clip's boresight by construction. The per-clip pitch comes from the extrinsic.

### 1.3 Feature-map coordinates: REF-C's convention, derived and pinned

REF-C's `ResNetEncoder` (`refc.py:1204-1213,1240-1248`) pads every downsampling layer by (k−1)/2. So feature index `k` at stride `s` is centred on image pixel index `s·k`, **offset 0**, and not on `s·k + (s−1)/2` as a symmetric tiling would assume.

- The difference is 7.5 px = 1.41° at stride 16, about 0.5 m lateral at 20 m (one BEV cell).
- `receptive_field_centre()` recomputes (16, 0.0) and (32, 0.0) from the module objects, on both the main and the shortcut paths.
- A forward impulse through the real class confirms it: the response is symmetric about column 7 for an impulse at pixel 112, and tied between columns 7 and 8 for pixel 120.
- The stride-16 map shape 16×40 is re-MEASURED by a forward pass.
- 352 channels = 4 × base_width 88 (INHERITED: `Project Steering/MODEL_REGISTRY.md:2317,2549`; `…/2026-09-13-bev-lidar-corpus-and-head/RESULT.md:220-223`).
- Grids are normalised for `grid_sample(align_corners=False)`.

## 2. Tests: MEASURED (`raw/pytest_new_tests.txt`, `raw/mutation_runs.json`)

| file | tests | what |
|---|---:|---|
| `stack/tests/test_semantic_map_gt.py` | **27 pass** (24 synthetic, 3 real) | (a) every refusal: 5 schema variants, 5 array variants, non-monotone time axis, corrupt archive, 6 index variants, time ±1 ms / one-frame shift / length / NaN. **Mutation: a file renamed to another clip id is refused** (synthetic, and real clip 0's file under clip 1's name). Pose check, coverage 3 states, episode-grid restatement. (e) real alignment. |
| `stack/tests/test_bev_lift.py` | **8 pass** (7 analytic/synthetic, 1 real) | (b) literals: ahead → col 319.5 / row 150.41831180523292; LEFT (y = +3) → col 274.00258341360995 < 320; behind → invalid; far ground rows 229.359 … 127.958 → 127.5 level and 162.482 … 116.834 → 116.829 with 2° pitch, monotone. (c) **mirrored lift fails (b)**, at point and grid level. RF centre. The lift pixel reproduces the builder's `cylindrical_grid` native pixel (< 0.05 native px; mirror > 20 px). `BEVLift` samples exactly the index it claims (< 1e-3), refuses a wrong stride / channel / height count, and has exactly 180,480 params. (d) on 13 real clips. |

Mutation runs: each defect is re-introduced by monkeypatch, and both files are run.

| mutation | tests RED |
|---|---|
| M0 none (control) | 0 (35 pass) |
| M1 identity check removed | 3: renamed-file, coverage, real renamed |
| M2 time check disabled | 2: synthetic, real (e) |
| M3 negative index wraps | 1 |
| M4 seen threshold off by one | 1. Synthetic only: real data never sits at 127/128 (§6.2). |
| M5 pose check disabled | 2 |
| M6 lift mirrored (y → −y) | 5: (b) ×2, (c), builder-consistency, (d) via exact emulation |
| M7 symmetric-tiling feature offset | 3 |
| M8 validity = rows only | 4 |

Also ran the repo-wide audits touching new files: `test_no_frame_shadowing_repo_wide`, `test_ref_offset_repo_wide`, `test_text_encoding_is_explicit`, `test_build_parity_guard`. **No finding names a new file.** Two pre-existing failures are unrelated:
- `test_text_encoding_is_explicit` (1 test) lists 15 call sites in 11 existing files.
- `test_build_parity_guard` has 15 failing tests: 14 because banked evidence lies outside this sparse checkout, and 1 that names 2 existing scripts.

## 3. Cross-check against the LiDAR BEV GT `cart_cam_vis`: MEASURED (`raw/cam_vis_crosscheck.json`)

Compared: lift valid at ≥ 2 of 4 heights vs `cart_cam_vis ≥ 128`. All 134 B1-eval LiDAR GT clips were compared. Extrinsics come from the HF `calibration/sensor_extrinsics.parquet`, which is **equal to the per-chunk file the LiDAR builder read on 134/134**.

| set | n | IoU min | IoU median | IoU mean |
|---|---:|---:|---:|---:|
| all, ≥ 2 of 4 heights | 134 | **0.99677** | **0.99912** | 0.99899 |
| rig A | 62 | 0.99781 | 0.99919 | 0.99911 |
| rig B | 72 | 0.99677 | 0.99897 | 0.99889 |
| any height valid | 134 | 0.99590 | 0.99707 | 0.99701 |
| all 4 heights valid | 134 | 0.99197 | 0.99649 | 0.99604 |
| ≥ 2 of 4, + intrinsics observed-pixel term | 134 | 0.99706 | 0.99897 | 0.99887 |

**Where they disagree:**
- 2–22 cells per clip (median 6) out of 6,790–6,852 in-field cells.
- **922/922 lie within 1 cell** of the `cam_vis` in/out boundary (max distance 1; the grid frame is not counted as a boundary).
- 617 are at the lateral ±60° edge and 305 at the near-field bottom-row edge.

**Which model is wrong? Neither.** The difference is definitional: heights {0, .5, 1.5, 2.5} at the cell centre vs {.3, 1, 2, 3} at 4×4 sub-samples; bounds [−0.5, W−0.5] vs [0, W−1]; no observed-pixel term. Proof: the builder's recipe, run **through the lift's cell centres and `project_rig_points`**, reproduces `cart_cam_vis` **byte-exactly on 134/134**:
- on 108 clips with the intrinsics observed mask;
- on the other 26 (all rig B) with the builder's own frame-content mask (max over 6 decoded frames > 0, `p2_build_corpus.py:206-213`).

**⚠️ What this cross-check can and cannot see.** The reference shares the projection code, so agreement validates composition (grid orientation, heights, bounds, extrinsics file), not the projection formula. The IoU is also nearly blind to real errors. Perturbation arms:

| arm | IoU (≥ 2/4) median / min | clips outside the 1-cell band | exact-emulation cells changed (12 rig-A clips) |
|---|---|---:|---|
| mirror y → −y | 0.9960 / 0.9912 | 0 of 20 | **102–168** |
| yaw +2° / +0.5° | 0.9924 / 0.9912 (+2°) | 0 of 20 | **82–89** (+0.5°) |
| pitch +2° / +0.5° | 0.9983 / 0.9972 (+2°) | 0 of 20 | **57–85** (+0.5°) |
| height +0.30 m / +0.05 m | 0.9988 / 0.9975 (+0.30 m) | 0 of 20 | **48–63** (+0.05 m) |
| nominal camera, no extrinsics | 0.9968 / 0.9941 | 0 of 20 | not run |

⇒ Validity-mask IoU ≥ 0.99 is **not evidence of correct geometry**. The exact re-emulation is a strong regression guard, now part of test (d). The independent evidence is §4, plus the analytic literals and builder-consistency tests (§2).

## 4. Independent registration against the SAM3 GT on real pixels: MEASURED (`raw/pixel_registration.json`)

The SAM3 map's camera model is independent: Thor-side `sam3_paint.lift` with the qwendrive camera parameters (`TanitAD Research Lab/Data Engineering/Research/2026-09-13-sam3-only-road-map/code/sam3map_extract_v31.py:81-86,194-202`).

**Method:**
- On the v2ep PNG frame the label belongs to (every 10th raw frame, 19–20 frames/clip), sample luminance at the lift's z = 0 pixel of each cell.
- Compare SAM3 lane-line cells (ch2 ≥ 0.5) with plain drivable cells (ch1 ≥ 0.9). Both must be seen, with x ∈ [cam_x + 4, 30] m and |y| ≤ 12 m.
- Score with AUC, then shift columns or rows by −8…+8 px and mirror.

| sha12 | rig | pitch | lane / plain samples | AUC aligned | AUC mirror | best col shift | best row shift |
|---|---|---:|---|---:|---:|---:|---:|
| `029d8d52075c` | B | −0.02° | 672 / 12,469 | 0.680 | 0.585 | −1 px | 0 px |
| `00213e99adca` | B | 1.46° | 3,216 / 19,271 | 0.533 | 0.415 | 0 px | 0 px |
| `0191487845ef` | B | 1.70° | 544 / 17,478 | 0.597 | 0.370 | −2 px | +1 px |

⇒ The lift is registered to the pixels within about 2 px in columns (0.37°) and 1 px in rows (0.19°) on these clips, and the mirror is worse on 3/3. For scale, a 2° yaw or pitch error would move a peak by ~10.7 px. ⚠️ The AUC peaks are broad; this is a registration check, not a calibrated estimator; n = 3 clips, all rig B.

## 5. Real-data alignment (e): MEASURED (`raw/real_alignment.json`)

Sources: GT from `C:/Users/Admin/tanitad-caches/refcv6g-20260915` (sha256 = the manifest's); v2ep payloads from `D:/Projects/TanitAD-artifacts/refcv5cmp/data/eval` (PNG, `n_stack` 3, 256×640 cylindrical); camera timestamps from `physicalai-b1/r0/camera_front_wide_120fov`.

| sha12 | T GT = v2ep = grid | `t_img` / `cam_frame_idx` / `t_query` equal | worst \|Δt\| | refused: 1-frame, +1001 µs, pose shift | pose residual max | ±1-frame median | LiDAR GT `t_img_us` equal |
|---|---|---|---:|---|---:|---:|---|
| `029d8d52075c` | 200 | yes / yes / yes | 0 µs | yes / yes / yes | 2.26 mm | 1.026 m | yes |
| `00213e99adca` | 201 | yes / yes / yes | 0 µs | yes / yes / yes | 1.14 mm | 2.674 m | yes |
| `0191487845ef` | 201 | yes / yes / yes | 0 µs | yes / yes / yes | 1.00 mm | 0.570 m | yes |

A +999 µs offset passes (the tolerance is inclusive). These are the only 3 clips on the dev box with both a SAM3 GT file and a v2ep episode.

## 6. Findings along the way (MEASURED unless marked)

1. ⛔ **The GT `source.split` field is the upstream PhysicalAI split, not the TanitAD eval split.**
   - All 3 published clips read `"train"` and all 3 are in `stack/tanitad/data/v72_eval_clip_digests.json` (147 v7.2 EVAL clips). Two probes, both positive: that file, and the B1-eval LiDAR GT set.
   - In the corpus index `index/clip_to_chunk.parquet` (local copy), the 147 eval clips carry upstream split 98 train / 27 val / 22 test.
   - A trainer that filters SAM3 GT by `split == "train"` would train the map head on 98 eval clips. The manifest's `published_by_split: {"train": 3}` is misleading for the same reason.
2. **Channel sums** are 253–257 / 255, with 99.19–99.34% of cells exactly 255, across 3 clips. **No cell has `not_seen` of 127 or 128**, because 25 sub-samples quantise it. So the ≥ 0.5 seen threshold is unambiguous on real data, and only the synthetic fixture can catch an off-by-one (M4).
3. **Observed-pixel masks disagree on rig B.** On 26/72 rig-B clips, the LiDAR builder's frame-content mask marks 17–8,916 px black that the intrinsics place inside the sensor (0 px the other way). The large cases (4 clips, 2,289–8,916 px) are a band in rows 168–238, within 54 px of the sensor edge, on day-lit clips (mean luminance 55–64). It is black video content near the bottom edge; the cause (ego mask or encoding) is UNVERIFIED. Effect on `cam_vis` ≤ 64/255 on affected cells.
4. With the intrinsics mask, spec-valid lift samples that land on black pixels:
   - rig B: median 29.5, max 66 of 30,720 per clip; at most 5 cells lose every height;
   - rig A: ≤ 10 samples.
5. **Partially valid cells** (0 < valid heights < 4): 36–78 per clip (median 42) of 7,680. Where a sample is zeroed, BEVLift cannot tell "zero feature" from "invalid height"; this affects < 1% of cells.
6. **Cache builder quirk, not corrected.** `calib.cylindrical_grid` normalises by (w−1) while `cylindrical_rectify` samples with `align_corners=False`. So native sampling is stretched by w/(w−1) about the image centre: ≤ 0.70 native px, ≤ 0.18 cache px in columns and ≤ 0.19 in rows over 40 clips (0.007 native px at the frame centre). This is below the lift's resolution; noted for the calib owner.

## 7. UNVERIFIED

- The **trained** refcv5-v2 trunk's effective receptive-field centre. Offset 0 is analytic and confirmed with synthetic symmetric weights only.
- The lift on real trunk features. No feature-level test, and no map head trained.
- Pixel registration on **rig A** (all 3 SAM3 clips are rig B) and on more than 3 clips. Time and pose alignment on **train** clips (no train clip on the dev box has both GT and v2ep).
- SAM3 GT coverage beyond the manifest snapshot (HF not queried).
- Camera `timestamps.parquet` availability on the training box (Thor) for the timestamp route of C7.
- The pose route assumes v2ep poses and GT `T_world_rig` share the egomotion world frame. True on 3/3; not verified corpus-wide.
- Flat road plane at rig z = 0. INHERITED as MEASURED over 87,481 cuboids (`rig_projection.py:40-47`); slopes are not modelled.
- The cause of the rig-B black band (§6.3).
- The Simple-BEV primary is still not banked (owed by the design, §11).

## 8. Reproduce

```
set PYTHONPATH=C:/Users/Admin/tanitad-wt-refcv6g/stack
python -m pytest tests/test_semantic_map_gt.py tests/test_bev_lift.py -v -s     # from stack/
python "…/2026-09-15-refcv6g-gt-reader-and-bev-lift/code/refcv6g_checks.py" camvis pixreg align
python "…/2026-09-15-refcv6g-gt-reader-and-bev-lift/code/mutation_runs.py"
python -m tanitad.models.bev_lift                                               # prints 180,480
```

Data paths are overridable by environment variable: `TANITAD_SAM3_GT_ROOT`, `TANITAD_V2EP_EVAL_DIR`, `TANITAD_CAM_TS_DIR`, `TANITAD_LIDAR_BEV_GT_DIR`, `TANITAD_CALIB_DIR`, `TANITAD_B1_ROOT`. Off the dev box, the real-data tests skip and name what is missing.
