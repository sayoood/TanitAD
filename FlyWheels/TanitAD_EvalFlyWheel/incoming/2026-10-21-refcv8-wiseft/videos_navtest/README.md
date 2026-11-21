# NavSim navtest video reels -- refcv8 WiSE blend theta_0.5 vs EMA-48,852 vs model-free prior vs human
EvalFlyWheel teammate, rendered 2026-10-21 (README finalised 2026-10-21 17:37 local, after the coordinator's title/legend correction). **STAGED ONLY -- nothing committed or pushed; the Master Mind lands it.**
Eval tier (every number below and on screen): NavSim open-loop PDM-score benchmark (T1-family), zero-shot PhysicalAI-AV -> nuPlan cameras, non-parity; NEVER closed loop. NOT leaderboard-submitted. One training seed (n=1).

## 0. Headline
RENDERED. Six clips (35.5-49.9 s each incl. a 3.5 s intro card) + one combined reel (261.8 s: title card, 6 clips, summary card, chapter markers), 1920x1080 H.264 10 fps.
Camera view with all four paths projected on the road **and** a metric BEV pane **and** per-scene PDMS/NC/DAC/TTC/EP/C text with the best arm highlighted.
**NEEDS LANDING:** the MP4s (~330 MB primary + ~330 MB alternate set) exist **only on the dev box** at `C:/Users/Admin/refcv8_pf/navsim_video/`; the code, this README and the manifest are staged in the hold.

## 1. Log SELECTION RULE (written 2026-10-21 17:22 local, BEFORE any rendering; unchanged since)
1. **Contiguity.** The scenes of one navtest log are ordered by the timestamp of their t0 (current) frame. A *contiguous run* is a maximal sequence of scenes whose consecutive t0 gaps are <= 0.6 s (MEASURED: the modal gap is 0.5 s = the 2 Hz frame spacing, 10,040 of the 12,010 gaps; the rest are 1.0 s ... 576 s jumps where scenes were filtered out).
2. **Eligibility.** A log is eligible iff it has a contiguous run of >= 40 scenes; its LONGEST run is its clip (whole run, no trimming).
3. **Ranking.** Eligible logs are ranked by the mean over the clip's scenes of (PDMS theta_0.5 - PDMS EMA arm shown), per-token PDMS from the score CSVs.
4. **Pick** the 2 highest, the 2 whose rank is closest to the median, the 2 lowest. If fewer than 7 logs are eligible the three groups cover every eligible log (stated, not hidden), and the groups are then only the TIER labels HIGH / MID / LOW on the clip cards.

**Result: only 6 of 136 logs are eligible, so the "selection" is the whole eligible set** -- HIGH/MID/LOW are tier labels, not a sample. (A looser contiguity gap would admit more logs but with visible jump cuts; I kept the strict rule.) The six clips are 276 scenes = 2.3% of navtest and are NOT a representative sample. The LOW tier shows one clip where theta_0.5 is behind EMA (-0.69 on the LEGAL arm), so losses are shown, but the worst case in the whole split may be worse than anything here.

### 1a. Primary set (EMA arm = R8_LEGAL stage; files under `navsim_video/LEGAL/`)
| rank | tier | log | scenes | span (s) | mean PDMS x100: theta_0.5 / EMA / prior | clip mean theta_0.5 - EMA | whole-log (n) |
|---|---|---|---|---|---|---|---|
| 1 | HIGH | `2021.10.06.07.26.10_veh-52_00006_00398` | 58 | 28.5 | 83.65 / 34.08 / 85.95 | **+49.58** | +14.42 (196) |
| 2 | HIGH | `2021.10.06.07.26.10_veh-52_02208_02394` | 40 | 19.5 | 66.08 / 34.67 / 41.98 | **+31.41** | +25.30 (146) |
| 3 | MID | `2021.10.06.08.16.17_veh-52_01430_01579` | 49 | 24.0 | 82.87 / 67.31 / 59.19 | **+15.57** | +19.08 (126) |
| 4 | MID | `2021.09.16.19.12.04_veh-42_00289_00398` | 45 | 22.0 | 64.77 / 53.68 / 28.24 | **+11.09** | +7.04 (67) |
| 5 | LOW | `2021.10.06.08.16.17_veh-52_01949_02501` | 43 | 21.0 | 87.48 / 85.33 / 27.52 | **+2.15** | +10.54 (341) |
| 6 | LOW | `2021.08.16.14.23.37_veh-45_00015_00132` | 41 | 20.0 | 48.97 / 49.66 / 16.68 | **-0.69** | -1.91 (77) |

### 1b. Alternate set (EMA arm = R8_RT re-timed stage; files under `navsim_video/RT_alt/`) -- same six logs, same order
| rank | tier | log | scenes | span (s) | mean PDMS x100: theta_0.5 / EMA / prior | clip mean theta_0.5 - EMA | whole-log (n) |
|---|---|---|---|---|---|---|---|
| 1 | HIGH | `2021.10.06.07.26.10_veh-52_00006_00398` | 58 | 28.5 | 83.65 / 32.76 / 85.95 | **+50.89** | +15.38 (196) |
| 2 | HIGH | `2021.10.06.07.26.10_veh-52_02208_02394` | 40 | 19.5 | 66.08 / 34.64 / 41.98 | **+31.44** | +26.36 (146) |
| 3 | MID | `2021.10.06.08.16.17_veh-52_01430_01579` | 49 | 24.0 | 82.87 / 64.14 / 59.19 | **+18.74** | +21.67 (126) |
| 4 | MID | `2021.09.16.19.12.04_veh-42_00289_00398` | 45 | 22.0 | 64.77 / 51.73 / 28.24 | **+13.04** | +8.40 (67) |
| 5 | LOW | `2021.10.06.08.16.17_veh-52_01949_02501` | 43 | 21.0 | 87.48 / 85.21 / 27.52 | **+2.28** | +12.23 (341) |
| 6 | LOW | `2021.08.16.14.23.37_veh-45_00015_00132` | 41 | 20.0 | 48.97 / 47.32 / 16.68 | **+1.65** | -0.44 (77) |

## 2. THE EMA ARM -- discrepancy in the original brief, CORRECTED by the coordinator (2026-10-21, after the first render)
The brief gave the EMA plans/scores as `seam_R8_RT.npz` + `r8nt_s48852e_R8_RT.csv` and the title card as "EMA-48,852, shipped re-timed stage: 57.84". I found they disagree: `r8nt_s48852e_R8_RT.csv` `average` = **56.67**, `r8nt_s48852e_R8_LEGAL.csv` = **57.84**. **Coordinator ruling:** 57.84 is EMA-48,852's RAW plan (R8_LEGAL); the shipped re-timed stage (R8_RT) is 56.67; keep LEGAL as the primary comparator (theta_0.5 is the WiSE blend of the LEGAL stage, so LEGAL is like-for-like).
- Primary set `navsim_video/LEGAL/`: EMA arm = **R8_LEGAL, legend "refcv8 EMA-48,852 (raw plan)"**. Title card text (exact): "refcv8 on NavSim navtest (12,146 scenes, open loop, one training seed). WiSE blend theta_0.5 (half refcv7, half refcv8 weights, no training): PDMS 72.78. refcv8 as trained (EMA-48,852, raw plan): 57.84; its shipped re-timed stage: 56.67. refcv7: 70.20. Model-free prior: 58.66."
- Alternate set `navsim_video/RT_alt/`: EMA arm = **R8_RT, legend "refcv8 EMA-48,852 (shipped, re-timed)"**. Same title card text plus the line "In this set the refcv8 arm on screen is the shipped re-timed stage (56.67)."
Ranking and tiers are identical under both arms (same six logs, same order). Revision history: the first render (17:29) carried the old wording on the title card/legend; all clips, cards and both reels were re-rendered at 17:3x with the corrected labels (the legend is burnt into every frame).

## 3. Playback timing
Scenes of a clip are 0.5 s apart in real time (MEASURED from `timestamps_us`). Each scene is **held 0.8 s (8 frames at 10 fps) = 1.6x slow motion**, stated on every frame; the camera image is the scene's t0 frame, updated every 0.5 s of real time -- a scene-by-scene review, not a smooth dash-cam. Clip content = 40-58 scenes -> 32-46.4 s each (+3.5 s intro card).

## 4. What is on each frame
- Camera pane (1280x520): the banked 416x1024 **cylindrical** t0 frame; human future = green ribbon (1.8 m) + line; theta_0.5 orange, refcv8 EMA-48,852 magenta (legend names the stage), prior grey dashed; dots every 0.5 s (4.0 s end ringed); dashed white = the horizon row of the camera.
- BEV pane (640x520): ego frame (rear axle), x forward up, metres; per-clip fixed scale (6/8/10/12 px/m from the clip's 98th-percentile forward extent), 10 or 20 m grid, ego footprint, +-60 deg field-of-view wedge. **No agent boxes**: `navtest_inputs.json.gz` carries none.
- Text: log, token, scene_token, nav command, ego speed/accel, per-arm PDMS + NC DAC TTC EP C (gold box = best PDMS arm, "BEST (tie)" on ties, none when all arms score 0), the human row marked "reference; not scored per token", the PDMS ribbon of the clip (3 arms x N scenes, cursor), clip means and "so far" means, per-scene win/loss counts vs EMA.
- Every number comes from the per-token score CSVs (`navsim_bars8.read_scores_csv`; paths in `code/nv_select.py`); spot-checked against a raw `grep` of the CSV rows for 2 tokens (match to the printed digits).

## 5. Camera projection and its POSITIVE CHECKS
Method (reused, not re-derived): the bridge's own virtual camera -- `frames416.rig_record` (CAM_F0 mount in the NavSim lidar/ego frame + the stitch's rotation M, from each log's own pickle) through `r8v_geom.CylCam` (copied beside the code; md5 `3fb05763df4fd92725445c2ea2cadac7`), road plane z = -0.3557 m (`road_plane_navhard_warmup_logs.json` summary.median, the value the model's BEV lift used). Cylindrical: column linear in azimuth, **not** pinhole. Camera ~ (1.66, -0.01) m from the rear axle, 1.87 m above the road.
1. **Visual check (the one the brief asks for), PASS:** `checks/projcheck_clip0_scene10.png` (straight, 25 km/h: the path runs in the lane between the double-yellow and the dashed white), `checks/projcheck_clip3_scene20.png` (straight, 36 km/h: path centred on the carriageway), `checks/projcheck_clip5_scene20.png` (LEFT bend, 35 km/h: the path bends left with the kerb). Viewed with the image reader; the ribbon sits on the carriageway ahead in all three and follows the curvature.
2. **rig_key control:** the rig key recomputed from the log pickles equals the frame-bank index `rig_key` for all 276 tokens (0 mismatches; `code/nv_rig.py`).
3. **Frame-bank integrity + t0 identity:** all 276 banked 4-frame blocks re-hash to the index sha256[:16] (on Thor; `data/extract_report.json`); scp copy md5-identical on both ends (`data/dev_md5.txt`); for every consecutive scene pair (267/267) the later token's second-to-last row is the earlier token's last row -- rows are time-ordered and the last row is the scene's t0 frame.
4. **Plan <-> score pairing control** (`data/pairing_check.json`, `code/nv_pairing_check.py`): Spearman(plan path length, scorer `progress_raw_m[1]`) over all 12,146 tokens = theta_0.5 0.991, EMA-LEGAL 0.990, EMA-RT 0.990, prior 0.976, against a permuted (mispaired) control of -0.011 / -0.003 / 0.019 / 0.017. The dev-box theta_0.5 seam is therefore the array that was scored. (Plan progress vs the normalised EP column is NOT a clean control -- rho 0.07-0.10, prior -0.10 -- recorded beside, not used.)
Not checked: lidar frame == rear-axle frame in x/y is assumed from the bridge (it uses the same frame for its lift); the three images agree to the eye, which is not a sub-metre measurement.

## 6. Reproduce
```
python -I code/nv_select.py --select                       # selection.json (both EMA arms)
ssh -n tanitad-thor: python3 thor_extract_frames.py wanted.json   # -> work/frames (scp, md5-verified)
PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval python code/nv_rig.py   # -> work/rigs.json
python -I code/nv_driver.py --ema-arm LEGAL --out C:/Users/Admin/refcv8_pf/navsim_video/LEGAL   # 6 clips + cards + reel (~20 s)
python -I code/nv_driver.py --ema-arm RT    --out C:/Users/Admin/refcv8_pf/navsim_video/RT_alt
python -I code/nv_manifest.py                              # MANIFEST_videos_navtest.json
```
CPU only throughout (dev-box GPU untouched, `CUDA_VISIBLE_DEVICES=` set). Thor: read-only numpy mmap of the frame bank under `nice -n 10`; wrote only `/home/nvidia/r8ns/navsim_video_tmp/` (6 frame npy + meta + the script, 344 MB; safe to delete -- the frames are also on the dev box in `navsim_video/work/frames/`). Work dir on the dev box (not staged, regenerable): `C:/Users/Admin/refcv8_pf/navsim_video/work/`.

## 7. Honest limits
- Open-loop PDMS: the plans are not driving; the recorded camera does not react to them.
- 276 scenes, one training seed. theta_0.5 beats EMA-LEGAL in the HIGH/MID clips and is about even in LOW. No population statement can be read off these clips (the full-split numbers are on the title card).
- Site/time correlation: 4 of the 6 clips are from the same site (sg-one-north, the two `2021.10.06.*` sessions, vehicle veh-52); the other two are us-pa-pittsburgh-hazelwood.
- No agent boxes are drawn (not in the inputs). The human row has no per-token PDMS (the CSVs hold model arms only).
- The frames are the bridge's stitched 3-camera cylinder; the black bands at the top/bottom/sides are the unobserved region, not rendering errors.
- refcv7's per-token plans/scores are not in the inputs given, so refcv7 appears only on the title card (70.20), not in the per-frame panels.
