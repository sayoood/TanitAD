# RESULT_VIDEO -- REFe P6 (model_final.pt) on REAL camera frames (no optical-flow interpolation)

**Deliverable:** `F:\TanitAD_p6video\out/p6_realframes.mp4` -- 932 MB, 593 s = 9.88 min, 2560x1440, h264 (High), CRF 18 preset slow, 12.6 Mbit/s, 10 fps, 5930 frames. 1080p copy: `F:\TanitAD_p6video\out/p6_realframes_1080p.mp4` (261 MB). `*.mp4` is git-ignored and raw camera frames stay private: the video and its frames do not go to any repo.

**Fidelity gate: PASS.** No model was run for this video: the plans are sliced from P6's second full-navtest pass (`proposals_p6final.npz`, the first pass's seam command plus `--dump-proposals`). For all 1090 video tokens the selected trajectory of the dump equals the banked plan (`seams/refe_p6final.npz`): max |delta| = **0** (xy 0 m, heading 0 rad; tolerance 1e-3 m; 1090/1090 tokens bit-identical). argmax of rule `navsim_v1` over the logits differs from the recorded pick on 0 tokens. `dump.DONE`: poses_bit_identical True over all 12,146 tokens (tokens differing 0). Dump sha256 `f61b5ddece39f660...`.

**Checkpoint:** `model_final.pt` sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1`; rule `navsim_v1` = NC*DAC*(5EP+5TTC+2C)/12 over six score outputs, NO lane factor (u400's rule had one); nav_mode command.

## Measured results of this checkpoint (on the intro/outro cards; copied from `P6_RESULT.md` and `level_ci.json`)

MEASURED, full navtest, 12,146 tokens, harness PASS: **PDMS 91.20 [90.41, 91.98]** (`level_ci.json`); +2.95 [+2.19, +3.77] over u400 88.25 (`P6_RESULT.md`); goal 94.55 (DriveZero Table A13): BELOW GOAL. **PREREG_P6, read as written: NOT PROVEN against u400 and against P2. BELOW GOAL.** Sub-scores NC 98.85, DAC 97.95, EP 87.03, TTC 95.08, C 100.00, DDC 96.48 (`level_ci.json`). T1-class open loop against logged agents: the car in this video is the human's; the overlays show what P6 would have planned.

## Frame sources per camera (MEASURED, `manifests/clip*.csv` = one row per output frame)

| camera | source | rate shown | frames |
|---|---|---|---|
| CAM_F0 (front) | raw nuPlan `nuplan-camera/mini/<log>_CAM_F0.zip` (all 10 navtest logs), DB `image.filename_jpg` | native 10 Hz, each frame at its OWN timestamp | 5450 (all) |
| CAM_L0 (left) | raw `scenarios4/<log>_CAM_L0.zip` for 1 navtest log (`2021.05.25.14.16.10_veh-35_01690_02183`); the OpenScene 2 Hz frame (the planner's own input frame) for the other 9, HELD for the 0.5 s token window and labelled `2 Hz (held)` on screen | real 10 Hz: 565 frames; held 2 Hz: 4885 frames | 5450 |
| CAM_R0 (right) | raw `scenarios4/<log>_CAM_R0.zip` for 1 navtest log (`2021.05.25.14.16.10_veh-35_01690_02183`); the OpenScene 2 Hz frame (the planner's own input frame) for the other 9, HELD for the 0.5 s token window and labelled `2 Hz (held)` on screen | real 10 Hz: 565 frames; held 2 Hz: 4885 frames | 5450 |
| CAM_B0 (rear) | raw `scenarios4/<log>_CAM_B0.zip` for 1 navtest log (`2021.05.25.14.16.10_veh-35_01690_02183`); the OpenScene 2 Hz frame (the planner's own input frame) for the other 9, HELD for the 0.5 s token window and labelled `2 Hz (held)` on screen | real 10 Hz: 565 frames; held 2 Hz: 4885 frames | 5450 |

Share of output frames with ALL FOUR cameras real: **10.4 %**. No download was made; the other navtest logs' side/rear frames are held, never warped.

## Measured rates (never assumed; from the u400 package's probe, same data)

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

Across all 5450 output front frames the consecutive front-frame spacing within a clip is min 99.89 / median 100.009 / max 100.45 ms (9.999 Hz); 0 spacings fall outside 95-105 ms. Max |image timestamp - its ego_pose timestamp| = 25018 us. DB ego speed vs the token's export speed at token frames: median |diff| 0.007 m/s, max 0.039 m/s.

## How the overlays are registered (no pose interpolation, no image warping)

* The plan of token k (64 hypotheses + selected) is expressed in the rear-axle frame at the token's `lidar_pc` pose. For EACH front frame it is transformed into that frame's body frame with the high-rate `ego_pose` row the DB links to the image (full 3-D pose), then projected with the DB camera calibration including the Caltech lens distortion applied to the projected points.
* BEV: metric (12 px/m), nuPlan map + logged agents (DB `lidar_box`, 20 Hz, nearest frame), ego, 64 hypotheses, selected plan (green), human future (white).
* Drivable area (camera + BEV) = NAVSIM's DAC polygon layers, labelled "map: rendering aid, not a model input".

## Provenance QA (is every displayed front frame ONE real JPEG?)

36 random MOVING (>3 m/s) output frames decoded from the final mp4 and compared on the overlay-free front region with their own source JPEG and the adjacent one: PSNR vs own source min 29.9 / median 38.0 dB; vs the adjacent frame median 21.6 dB (max 28.1). Own beats adjacent on every sampled frame: **True** (min gap 8.8 dB). Frame accounting: expected 5930, decoded 5930 (match).

## Content

94 clips (the same as the u400 video) = every run of >= 6 consecutive navtest tokens in the 10 navtest logs with raw front frames: 1090 tokens = 545 s of driving + two result cards. Hard cuts between clips ("NEW CLIP i/94" banner). Table: `RUNS.md` (its score columns are u400's; P6's are on screen). **38 failure tokens (P6 true PDMS < 0.5) are shown in red**; nothing was filtered by score.

True per-token score of these 1090 tokens (`refe_p6final.csv`, executed plan; MEASURED): mean **91.35**, median 97.9. The full 12,146-token navtest PDMS of this checkpoint is 91.20 [90.41, 91.98]. This sample is NOT the benchmark. (u400's mean on the same tokens was 87.38.)

## Known limits (read before quoting)

* Overlays are drawn on a flat ground plane and are not occlusion-aware. Road slope is not modelled.
* Drivable area = the DAC polygon set, not the full PDM scorer.
* Predicted PDMS on screen is the P6 scorer head's estimate under `navsim_v1` (no lane factor); the TRUE score comes from the harness CSV.
* 94 discontinuous clips; side/rear cameras are held 2 Hz for 9 of the 10 logs.
* The model consumes F0/L0/R0/B0 at the token time; the first frame of each token window is the planner's input frame, marked on screen.
* One checkpoint, one training seed, deterministic planner; the level CI answers "another draw of logs" only (P6_RESULT.md).

## Reproduce

`gate_fidelity.py` (slice + gate) -> `run_all.py` (render, cards, concat, QA, 1080p, this file). Renderer `render_p6_video.py` + `p6_lib.py` (reuse `raw/2026-10-01-final-video/render_final_video.py`). Per-frame provenance: `manifests/`.
