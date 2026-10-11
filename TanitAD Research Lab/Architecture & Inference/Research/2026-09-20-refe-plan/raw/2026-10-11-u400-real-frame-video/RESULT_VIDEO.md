# RESULT_VIDEO -- REFe SFT-4 B u400 on REAL camera frames (no optical-flow interpolation)

**Deliverable:** `E:/Projects/TanitAD/data/videos/u400_realframes.mp4` -- 904 MB, 559 s = 9.32 min, 2560x1440, h264 (High), CRF 18 preset slow, 12.9 Mbit/s, 10 fps, 5590 frames. (`*.mp4` is git-ignored: this file is the only copy.)

**Fidelity gate: PASS.** The re-run u400 plans reproduce the banked full-navtest plans on all 1090 video tokens with max |delta| = **0** (xy 0 m, heading 0 rad; tolerance 1e-3 m; 1090/1090 tokens bit-identical), and argmax of the scorer rule equals the recorded pick on every token.

## What was wrong, what changed

The previous video (`raw/2026-10-01-final-video/render_sync_video.py`) motion-interpolated 2 Hz OpenScene frames to 10 fps with DIS optical flow (`interp_frame`, `Flow`), producing doubled trees, ghosted sign outlines and smeared structure. This renderer contains **no frame synthesis**: every camera image is one decoded sensor JPEG shown unmodified (front native 1920x1080; side/rear area-downscaled to 480x270). The only blends in the code are alpha blends of *synthetic overlays* (`cv2.addWeighted` on the overlay layers); no `remap`, no flow, no warping.

## Frame sources per camera (MEASURED, `manifests/clip*.csv` = one row per output frame)

| camera | source | rate shown | frames |
|---|---|---|---|
| CAM_F0 (front) | raw nuPlan `nuplan-camera/mini/<log>_CAM_F0.zip` (all 10 navtest logs), DB `image.filename_jpg` | native 10 Hz, each frame at its OWN timestamp | 5450 (all) |
| CAM_L0 (left) | raw `nuplan-camera/scenarios4/<log>_CAM_L0.zip` for 1 navtest log (`2021.05.25.14.16.10_veh-35_01690_02183`); the OpenScene 2 Hz frame (the planner's own input frame) for the other 9, HELD for the 0.5 s token window and labelled `2 Hz (held)` on screen | real 10 Hz: 565 frames; held 2 Hz: 4885 frames | 5450 |
| CAM_R0 (right) | raw `nuplan-camera/scenarios4/<log>_CAM_R0.zip` for 1 navtest log (`2021.05.25.14.16.10_veh-35_01690_02183`); the OpenScene 2 Hz frame (the planner's own input frame) for the other 9, HELD for the 0.5 s token window and labelled `2 Hz (held)` on screen | real 10 Hz: 565 frames; held 2 Hz: 4885 frames | 5450 |
| CAM_B0 (rear) | raw `nuplan-camera/scenarios4/<log>_CAM_B0.zip` for 1 navtest log (`2021.05.25.14.16.10_veh-35_01690_02183`); the OpenScene 2 Hz frame (the planner's own input frame) for the other 9, HELD for the 0.5 s token window and labelled `2 Hz (held)` on screen | real 10 Hz: 565 frames; held 2 Hz: 4885 frames | 5450 |

Share of output frames with ALL FOUR cameras real: **10.4 %**. Only one navtest log has raw L0/R0/B0 zips locally (`scenarios4/` holds B0/L0/R0 for 7 logs; only `01690_02183` is navtest). No download was made (permission required); the other navtest logs' side/rear frames are held, never warped.

## Measured rates (never assumed)

```
log: 2021.05.25.14.16.10_veh-35_01690_02183
CAM_F0: n=4930 span=492.900 s rate=9.999997 Hz  dt ms min/median/max/std = 99.890 / 100.009 / 100.457 / 0.069   n(dt>150 ms)=0
CAM_B0: n=4930 span=492.900 s rate=9.999996 Hz  dt ms min/median/max/std = 99.890 / 100.009 / 100.484 / 0.068   n(dt>150 ms)=0
CAM_L0: n=4930 span=492.900 s rate=9.999996 Hz  dt ms min/median/max/std = 99.890 / 100.008 / 100.475 / 0.069   n(dt>150 ms)=0
CAM_R0: n=4930 span=492.900 s rate=9.999996 Hz  dt ms min/median/max/std = 99.890 / 100.008 / 100.475 / 0.069   n(dt>150 ms)=0
lidar_pc: n=9860 rate=20.000006 Hz  dt ms min/median/max = 49.933 / 50.007 / 50.044  (lidar_box rows attach to these)
ego_pose: n=49500 rate=100.000 Hz
image.timestamp - its own ego_pose.timestamp (CAM_F0), us: min -930 max 1253
CAM_F0: DB filenames present in raw zip: 4930/4930  (mini)
CAM_L0: DB filenames present in raw zip: 4930/4930  (scenarios4)
CAM_R0: DB filenames present in raw zip: 4930/4930  (scenarios4)
CAM_B0: DB filenames present in raw zip: 4930/4930  (scenarios4)
```

Across ALL 5450 output front frames the consecutive front-frame spacing within a clip is min 99.89 / median 100.009 / max 100.45 ms (9.999 Hz); 0 spacings fall outside 95-105 ms. Maximum |image timestamp - its ego_pose timestamp| = 25018 us. Real side/rear frames are within 25.02 ms of the front frame; held frames are within 22.3 ms of the token time. DB ego speed vs the token's export speed at token frames: median |diff| 0.007 m/s, max 0.039 m/s.

## How the overlays are registered (no pose interpolation, no image warping)

* The plan of token k (64 hypotheses + selected) is expressed in the rear-axle frame at the token's `lidar_pc` pose. For EACH front frame it is transformed into that frame's body frame using the high-rate `ego_pose` row the DB itself links to the image (`image.ego_pose_token`; full 3-D pose incl. roll/pitch/z, so a braking pitch is respected), then projected with the DB camera calibration (`calib_table.read_db`: K, extrinsics) **including the Caltech lens distortion (k1 = -0.356 ...) applied to the projected points** (the image is never undistorted/warped). Plan points lie on the ground plane at the rear-axle height of token k.
* Calibration check: `stills/check_calibration_logged_agent_boxes.png` draws the DB `lidar_box` footprints (20 Hz) on the same pipeline; they sit on the cars/cones in the image.
* BEV: metric (12 px/m, scale bar, 10 m rings), nuPlan map (gpkg reprojected exactly as the devkit does) + logged agents from DB `lidar_box` at the nearest 20 Hz lidar frame to each camera frame (no interpolation), ego, 64 hypotheses, selected plan (green), human future (white).
* Drivable area (camera + BEV) = NAVSIM's DAC polygon layers (lane groups + intersections + car parks), labelled "map: rendering aid, not a model input".

## Provenance QA (is every displayed front frame ONE real JPEG?)

36 random MOVING (>3 m/s) output frames were decoded from the final mp4 and compared on the overlay-free front region (front y 110-400, sky/buildings) with (a) their own source JPEG and (b) the adjacent source JPEG (+-0.1 s): PSNR vs own source min 29.9 / median 38.3 dB; vs the adjacent frame median 21.6 dB (max 28.1). Own beats adjacent on every sampled frame: **True** (min gap 8.8 dB). A motion-interpolated blend would not match its own source this closely. Frame accounting: expected 5590 frames, decoded 5590 (match).

## Content

94 clips = every run of >= 6 consecutive navtest tokens (0.5 s apart) in the 10 navtest logs with raw front frames: 1090 tokens = 545 s of driving + title/summary cards. Hard cuts between clips (a "NEW CLIP i/94" banner marks each). Full table: `RUNS.md`. Coverage: turns, stops, lane keeping at speed, lane changes (heuristic tag), and **71 failure tokens (true PDMS < 0.5) are included and shown in red** -- nothing was filtered by score.

True per-token score of these 1090 tokens (u400 full-navtest harness CSV `refe_stage2_sft4_B_u400_navtest_full.csv`, executed plan): mean **87.38**, median 94.0; the full 12,146-token navtest PDMS of this checkpoint is 88.2469 (result.json). This sample is NOT the benchmark.

## Known limits (read before quoting)

* Overlays are drawn on a flat ground plane and are not occlusion-aware: far plan/drivable pixels can paint over a wall or hedge that hides that ground. Road slope is not modelled.
* Drivable area = the DAC polygon set, not the full PDM scorer (which also uses footprint corners, LQR tracking, etc.).
* Predicted PDMS on screen is the u400 scorer head's estimate under the deployed rule (`navsim_v1_lane`: NC*DAC*(5EP+5TTC+2C)/12 * p_lane); the TRUE score comes from the harness CSV.
* 94 discontinuous clips: navtest tokens are sparse, so there is no continuous drive.
* Side/rear cameras are held 2 Hz for 9 of the 10 logs (see above).
* The model itself consumes F0/L0/R0/B0 at the token time; the video shows what it saw (first frame of each token window is the planner's input frame, marked on screen).
* `MODEL_REGISTRY.md` has no row for u400 (grep `u400` = 0 hits): model facts here come from the raw eval JSON (`raw/2026-10-05-sft-stage2/sft4_B_u400_navtest_full/result.json`, `seam_report.json`) and the checkpoint md5 `8e382eebb35b3e155862453617e5786e` (verified; expected prefix 8e382eeb). **Escalation: the registry lacks this checkpoint.**

## Reproduce

`select_tokens.py` -> `run_inference.py` (gate) -> `run_pipeline.py` (render/cards/concat/QA/stills) -> `write_result.py`. Renderer: `render_u400_video.py` + `u400_lib.py` (reuse `raw/2026-10-01-final-video/render_final_video.py`: text, BEV class, box geometry, turbo ramp). Per-frame provenance: `manifests/`.
