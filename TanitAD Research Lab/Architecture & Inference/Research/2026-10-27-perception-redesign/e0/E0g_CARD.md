# E0g card -- box-derived depth-target builder (REFCV9_DESIGN s2.3 B1, s4.1 E0g, s6b D4): B1 CAN BE BUILT

**Question.** Can `obstacle.offline` cuboids be ray-cast through the CYLINDRICAL 416 x 1024 model into LID soft depth targets, with an analytic control that a known range reads back and a pinhole substitution fails?
**Answer.** Yes. `e0g_depth_targets.py` (numpy; torch only imported lazily inside the geometry helpers), 26 tests PASS, real-data content assertion PASS on 300 TRAIN-split frames.

**What was reused (no second projection formula in this file).** Pixel -> ray: `tanitad.data.rig_projection.frame_to_cam_ray` (keyed on `frame.projection`; the inverse of `project_cam_to_frame`; same ray as `calib.cylindrical_rays`). Frame: `CanonicalFrame.from_hfov(120, 416, 1024, "cylindrical")`, f_ref 488.92398517830253 == `vis1_zbuffer.F` (asserted). Box layout `[cx, cy, cz, l, w, h, yaw]` and camera `(R_cam_to_rig, t)` are `vis1_zbuffer.zbuffer`'s arguments. Cuboid reader (content run): `agent_cuboid_gt.cuboids_at_time`; extrinsics/intrinsics: `build_rig_extrinsics_table._front_wide_rows` + `vis1_zbuffer.clip_camera`. **New:** the depth VALUE (the vendored z-buffer keeps its depth local and returns ids only), LID, the s8/s16 aggregation.

## Controls (`test_e0g_depth.py`)
| id | control | result |
|---|---|---|
| C1 | cuboid at D = 5 / 20 / 50 / 99 m returns D at the 4 pixels around the centre | PASS, tol 1e-3 m (analytic deviation 1.0e-6 D) |
| C2 | OFF-CENTRE, cols 895 / 130 / 700 (azimuth +44.9 / -44.7 / +22.1 deg): `rho` == D to 1e-9; `range` == D sqrt(1+v^2) to 1e-9; across the whole face `rho` = D / cos(dphi) (column linear in azimuth) | PASS |
| C3 | no box / only invisible boxes: 0 fg pixels, 0 fg cells, 0 target mass, all weights = alpha_bg | PASS |
| C4 | nearer of two overlapping cuboids wins; identical depth buffer in either input order | PASS |
| C5 | behind the camera, at 75 deg, wholly beyond the 60 deg edge: none; straddling the edge: cast to the frame edge only | PASS |
| **RED** | pinhole rays (same grid, same f_ref) in place of cylindrical: centre still passes; off-centre narrow box is MISSED (38 vs 45 deg), a wide wall returns a finite WRONG rho (> 0.15 m off at 30 m) | the known-range control FAILS, as required. Whole-pipeline mutation (pinhole inside `pixel_rays`): 12 of 26 tests go RED |
| X1 | prefilter == brute force, bit-for-bit, on 3 random 40-box scenes incl. a car at azimuth 180 deg | PASS |
| X2 | nearest-box id buffer vs `vis1_zbuffer.zbuffer` (independent implementation), 3 random scenes: 1.000000 pixel agreement (460,805 object pixels) | PASS |
| X3 | hit points project back to their own pixel through `project_cam_to_frame` (< 1e-6 px) and lie on the named box's surface (< 1e-9) | PASS |
| X4 | LID literals (edge_i = bs i(i+1)/2, bs = 200/(80*81)), tau round trip 1e-9, two-hot sums to 1 and preserves E[tau] | PASS |
Other mutations caught: yaw sign (M1), prefilter shrink (M2), last-wins instead of nearest (M3), LID i*i (M4). Import in a fresh interpreter leaves `torch` out of `sys.modules` (pinned).

## Content on real data (`e0g_content.py`, `raw/E0G_RESULT.json`; MEASURED, dev box, CPU, 0.05 s/frame)
60 TRAIN-split clips (`clip_index.split == train`, sha256 order, no re-selection) x 5 times = **300 frames**; all cuboids cast; f-theta observed mask applied (60/60 clips).
| quantity | mean | p5 | p50 | p95 |
|---|---|---|---|---|
| foreground pixel share of the frame | **0.0583** | 0.0 | 0.0233 | 0.2349 |
| fg cell share, s8 (52 x 128) | 0.0587 | 0.0 | 0.0237 | 0.2361 |
| fg cell share, s16 (26 x 64) | 0.0579 | 0.0 | 0.0228 | 0.2368 |
Cuboids per frame mean 40.7 (p95 152); with foreground pixels mean 9.5. **41 of 300 frames have zero foreground**; Assertion A (frames with a cuboid centre in view within 95 m, n = 256, all have fg > 0): 0 violations; Assertion B (every zero-fg frame has no cuboid corner/centre in the field within 100 m): 0 violations. So the share is non-zero on every frame that has an object in view; 13.7 % of frames have none. Run vs `vis1_zbuffer` on 30 real frames: id agreement 1.0 over 638,106 object pixels.
Foreground-pixel depth histogram (m): 0-5 **36.8 %**, 5-10 **36.6 %**, 10-20 14.7 %, 20-30 5.5 %, 30-40 2.9 %, 40-60 2.4 %, 60-80 0.9 %, 80-100 0.4 %. 21,598 pixels fall beyond 100 m (ignored); 2.2 % of cuboid pixels sit on sensor-unobserved pixels (masked).

## Three decisions C3 must take (the design does not pin them) -- escalated, not buried
1. **Depth definition.** Default `range` (Euclidean along the ray). `rho` = hypot(x_cam, z_cam) is the cylinder's analogue of pinhole z-depth; differs by 1/sqrt(1+v^2), <= 9 % at the frame's top/bottom. Nearest-surface ordering is identical. Pick one for the head and the eval.
2. **alpha_bg has nothing to weight.** B1 is box-only (D4 dropped LiDAR): a background cell carries NO depth label (in CaDDN both classes have LiDAR depth), so alpha_fg 3.25 is a constant loss scale and alpha_bg 0.25 is inert unless C3 adds a background label (the free road-plane ray intersection is the candidate; NOT in the pinned design, and D4 says dense ground depth is forgone). The weight map is emitted in the CaDDN shape; no background label is invented.
3. **Per-pixel CE is a near-field loss.** 73 % of foreground pixels are < 10 m and 3.6 % are >= 40 m, while the design's range finding is a box-AP collapse with range (0.539 -> 0.092 over 0-60 m, R5). A pixel-mean CE will mostly teach the near field; per-cuboid or range-balanced normalisation is a C3 choice (HYPOTHESIS that it matters; not tested).
Unpinned choices made here, all keyword arguments: D = 80 LID bins (free: the CaDDN text, banked 2103.01100 sha12 3970e26ca935, fixes the LID formula (eq. 2, matches ours) and alpha 3.25 / 0.25 (focal-loss alpha, s3.5) but not D), a two-hot soft target in the bin-centre coordinate (the design says "soft"; CaDDN supervises with a one-hot, reproduced by `soft=False`), FG cell iff >= 50 % foreground pixels, partial cells ignored, target = mean of the cell's pixel targets.

## Limits (stated, not hidden)
* Occluders are labelled cuboids only (vis1's own caveat): a box hidden behind an UNLABELLED wall/tree still labels its pixels. Magnitude not measured; VIS-1's `vis_frac` could down-weight such boxes (HYPOTHESIS).
* Cuboid faces are looser than the car surface (depth bias, ESTIMATED, not measured). The yaw / (l, w) convention is the programme's (shared with vis1 and the cuboid reader); agreement with vis1 shows the two implementations match, not that the labels sit on the image -- no image overlay was made (UNVERIFIED).
* Cuboids are taken nearest-in-time per track within `DEFAULT_TOL_S` of the frame time, as the join does; the real camera frame timestamps were not used.
* No LiDAR depth check exists (D4). Membership of the 2,376-episode parity corpus was NOT checked for the 60 sampled clips (they are TRAIN-split clips of `obstacle_offline_b1train`).
* The module lives under `e0/`, not `stack/`: C3 must move it (e.g. `stack/tanitad/data/box_depth_targets.py`, no code change) before the trainer can import it.

**Evidence class.** MEASURED (ours: `raw/E0G_RESULT.json`, `test_e0g_depth.py`). Tier: n/a (a label-construction instrument).
