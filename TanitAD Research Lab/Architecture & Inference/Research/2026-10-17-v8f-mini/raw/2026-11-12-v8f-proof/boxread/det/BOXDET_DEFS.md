# BOXDET: detector-pass kit of card ADD127 (BOXREAD-T0), definitions and commands

Dev-box folder `F:/TanitAD-scratch/v8proof/boxread/det/`; suggested pod copy `/workspace/v8proof/boxread/kit/det/`. Imports, never edits: `cropread/enc/enc_common.py` (ADD99 kit) and `telegap/enc/tele_encode.py` (ADD122 kit). Tested on the dev box with a stand-in detector and patched decode and render only; nothing has run on real video, real weights or the pod. Evidence class of every number: ESTIMATED unless it says otherwise.

## 0. OPEN POINTS (read first; none decided silently)

1. **Frame count: DECIDED by the lead.** The card's cost line of about 17,000 frames per camera means the frames with a labelled lead, not all ~101 cache frames per clip. The tool therefore has `--box-dir DIR --only-lead-frames`: per clip it reads `DIR/<clip>.npz` key `has_lead` (bool [n_frames]; a missing file or a length different from the cache frame count is refused) and runs the detector (and decodes) ONLY on frames where it is true. Without the flag every cache frame runs. The smoke projection uses the selected counts of the full lists (all listed clips, both cameras; the tele projection is therefore an upper bound).
2. **Detector input size.** torchvision's detector resizes every image internally to a shorter side of 800 px (default `min_size=800`, `max_size=1333`). The wide window (768 px) is therefore scaled up 1.04 times and the tele window (1024 px) is scaled DOWN to 800 px, i.e. 50 instead of 64 pixels per degree. The card says "used as it is", so the defaults are kept and no argument changes them. Boxes are reported in the coordinates of the rendered window (768 or 1024 px), not of the resized image: torchvision maps them back.
3. **Source resolution.** The wide window of 32 degrees at 768 px (24 px/deg) is sampled from a 1920 px frame of 120 degrees (about 16 px/deg at the centre): it is an up-sampling by about 1.5, with no new information. The tele camera (30 degree field, 1920 px) gives about 64 px/deg native, so 1024 px for 16 degrees is native.
4. **Elevation zero is the optical axis, not the horizon** (ENC_DEFS 0.3): the card's "6 degrees down" is 6 degrees below the wide optical axis, which itself points 0.07 to 1.7 degrees below the vehicle horizontal. The tele window is at (0, 0) of the TELE camera's own frame as the card words it, NOT the wide-frame direction carried through the extrinsics that note N1 of card ADD122 used for T8 and T4. The tele window argument is therefore `--az-left-deg 0 --el-up-deg 0`.
5. **Score bar.** The card says score at least 0.5; the tool keeps 0.3 and up (`--score-keep 0.3`, as asked), so the reader applies the 0.5 of the card (and the file carries what lies between 0.3 and 0.5 for information). The bars of the controls are set without knowledge of those scores.
6. **Tele frame match** follows ADD122 exactly (`tele_encode.pilot_times` = the time of the wide video frame decoded for cache frame j; `match_frames` at most 17000 us, ties to the earlier frame). At 30 Hz this nearly always matches (TELE_ENC_DEFS 0.3); a frame with no match has `frame_ok` False and no detector run. A clip with no staged tele video gets no file. The card's "where the clip has a tele video" is read as that.
7. The pod state file `FORK_STATE_FILE` (default `/workspace/FORK_STATE.txt`, appended, the last line counts): the queue writes exactly `BUSY box-read detector passes of the v8proof fork <UTC time>` at the start of every GPU stage and `IDLE <UTC time>` when the queue ends or stops; `FORK_STATE_FILE=` (empty) switches it off.

## 1. Files

| file | job |
|---|---|
| `box_detect.py` | the detector pass, one camera per call; also `--project` (cost projection) |
| `pod_boxread_queue.sh` | smoke (3 clips per camera + projection + stop rule), wide, tele; markers |
| `test_box_detect.py` | 24 local tests (stand-in detector, patched decode and render, stand-in programs for the queue) |
| `helper_pytest.out` | output of the last full pytest run |

## 2. Output format

`<out-dir>/<camera>/<clip>.npz` (one per clip, `camera` is `wide` or `tele`; written under `.part` and renamed once; never overwritten; a clip whose file exists is skipped). Keys, exactly:

| key | dtype, shape | meaning |
|---|---|---|
| `n_frames` | int64 scalar | number of cache frames of the clip (`ceil(n_target / 2)`, ENC_DEFS 2) |
| `frame_ok` | bool [n_frames] | False where the frame could not be decoded or, for tele, has no matched tele frame. A frame the lead selection left out was not tried and stays True (unless tele-unmatched) |
| `frame_run` | bool [n_frames] | True where the detector was run: selected (`--only-lead-frames`), decoded and, for tele, matched. A frame with `frame_run` False carries no detection |
| `det_frame` | int32 [M] | cache index j of each kept detection (ascending) |
| `det_box` | float32 [M, 4] | x1 y1 x2 y2 in pixels of the rendered window, origin at the top left of the window, x to the right, y down |
| `det_score` | float32 [M] | detector score, all at least `--score-keep` |
| `det_label` | int16 [M] | COCO id of the weights: 3 car, 4 motorcycle, 6 bus, 8 truck (others dropped) |
| `win` | float64 [4] | size_deg, px, az_left_deg, el_up_deg of the render |
| `camera` | str | `wide` or `tele` (read with `str(z["camera"])`) |
| `weights_sha256` | str | full sha256 of the weights file |

Pixel convention and geometry for the reader: the window is the ADD99 window (`window_frame(size_deg, px)`): square, `px` by `px`, cylindrical, horizontal angular size exactly `size_deg` with azimuth linear in the column (f_ref = (px/2) / radians(size_deg/2)), the vertical axis the per-column pinhole (row offset = f_ref * tan(elevation)). Pixel (x, y) has azimuth left-positive = (px/2 - x) / f_ref radians relative to the window centre, and elevation up-positive = atan((px/2 - y) / f_ref), both added to the window centre (az_left_deg, el_up_deg) by the rotation `R = Ry(-az) Rx(el)` of `enc_common.rotation_az_el`. Use that routine for the size formula. Pixel x of a column centre is x + 0.5 in these terms; the tool states the box as the detector gives it.

Channel order: `decode_selected` converts to rgb24, `window_sample` keeps the channel order, so the window is **uint8 RGB, channels first**; the detector gets float32 in 0..1 in the same RGB order (what torchvision detectors expect; `detect_batch` divides by 255 and applies no other normalisation: the detector's own transform normalises).

`<out-dir>/<camera>/BOXDET_RECORD.json` (the last run) and `BOXDET_RUNS.jsonl` (one line per run): arguments, `clips_done`, `frames_selected`, `frames_run`, `box_dir`, `only_lead_frames`, `clips_skipped_existing`, `clips_without_video` (+ first 8 characters of each id), `frames`, `frames_not_ok`, `frames_tele_unmatched`, `frames_detected`, `detections_kept`, `decode_errors`, `failures`, `detector_seconds`, `gpu_seconds` (equal to the detector seconds on a cuda device, 0 on cpu), `detector_first_call_seconds`/`_frames` (warm-up), `prep_seconds_sum`, `wall_seconds`, `tool_sha256`, `weights_sha256`, `versions` (torch, torchvision). Counts are for the run; a resumed run counts the skipped clips separately.

## 3. How cache index j maps to video frames

* Cache frame j = episode frame 2j; wide video frame `frame_idx[2j]` with `frame_idx = searchsorted(t_frames, linspace(t_frames[0], t_frames[-1], n_target))` (`enc_common.frame_plan`, `plan["cache_idx"][j]`), decoded from the wide stage root `<root>/r0/camera/<clip>.mp4` (PyAV, as the cache builder).
* Wide read: the window is cut from the raw 1080 x 1920 wide frame of video frame `cache_idx[j]`, with the wide intrinsics of the wide stage root.
* Tele read: the time of the wide video frame is `t_frames[cache_idx[j]]` (`tele_encode.pilot_times`); the tele frame is the one nearest in time in the tele timestamps table `<tele-root>/ts/<clip>.timestamps.parquet` (row index = tele video frame index), matched when at most 17000 us away; it is decoded from `<tele-root>/r0/camera/<clip>.mp4` and cut with the tele intrinsics of the tele stage root (`calibration/physicalai_front_wide_intrinsics.csv`, which holds the TELE rows, as `tele_stage.py` writes it). Several cache frames can map to one tele frame; it is decoded once.

## 4. Arguments of `box_detect.py`

```
--camera wide|tele   --list FILE (repeatable)   --root WIDE_STAGE_ROOT (alias --wide-root; needed by both cameras)
--tele-root TELE_STAGE_ROOT (tele only)   --weights LOCAL.pth   --weights-sha-prefix dd69338a   --out-dir DIR
--win-size-deg  --win-px  --az-left-deg  --el-up-deg      card: wide 32 768 0 -6 ; tele 16 1024 0 0
--box-dir DIR  --only-lead-frames   (per-clip has_lead files; run the detector only on lead frames)
--score-keep 0.3  --batch 8  --limit-clips N  --device cuda  --workers 2 (at most 4)  --torch-threads 2
--project FOLDER (--project-frames N | --project-n-clips N)     (prints PROJECT_H <hours>; no detector)
--count-lead-frames --box-dir DIR --list FILE...   (prints LEAD_FRAMES <n> MISSING_BOX_FILES <m>; CPU only)
```
The weights file must exist and its sha256 must begin with the prefix, else exit 2 before anything is built. The model is `fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)` with the state dict loaded from the local file: no weight enum, nothing can be fetched. Exit codes: 0 done, 5 some clip failed or a decode error, 2 refused. `--limit-clips N` counts clips that have a video.

## 5. Commands for the pod

```
# 0. push this folder (box_detect.py, pod_boxread_queue.sh, test_box_detect.py, BOXDET_DEFS.md) to /workspace/v8proof/boxread/kit/det/
#    the ADD99 kit (/workspace/v8proof/cropread/kit/enc) and the ADD122 kit (/workspace/v8proof/telegap/kit/enc) must be in place; tests:
cd /workspace/v8proof/boxread/kit/det && BOXDET_TEST_CWD=/tmp /workspace/venv/bin/python -m pytest test_box_detect.py -q -p no:cacheprovider
# 1. dry run: prints CMD lines, starts nothing
DRY=1 OUT=/workspace/v8proof/boxread/out bash /workspace/v8proof/boxread/kit/det/pod_boxread_queue.sh
# 2. full queue, detached (smoke, stop rule on the projection, wide, tele); keep 28 GB of container memory free
nohup setsid env OUT=/workspace/v8proof/boxread/out WORKERS=2 bash /workspace/v8proof/boxread/kit/det/pod_boxread_queue.sh > /dev/null 2>&1 < /dev/null &
# 3. wait:  until [ -e $OUT/BOXREAD_DONE.txt ] || [ -e $OUT/BOXREAD_STOP.txt ]; do sleep 20; done
# resume after a stop: RESUME=1 (every pass skips clips whose file exists). Single stage: STAGES="wide" (or smoke, tele).
# single camera by hand (the smoke form):
/workspace/venv/bin/python box_detect.py --camera tele --list LIST --root /workspace/v8proof/cropread/stage --tele-root /workspace/v8proof/telegap/stage \
  --weights /workspace/v8proof/boxread/fetch/fasterrcnn_resnet50_fpn_v2_coco.pth --weights-sha-prefix dd69338a --out-dir OUT \
  --win-size-deg 16 --win-px 1024 --az-left-deg=0 --el-up-deg=0 --limit-clips 3
```
Queue settings (all environment, `SET-ME` marks the guesses of the pod layout): `PY KIT DET PROJ OUT WEIGHTS SHA_PREFIX ROOT_WIDE ROOT_TELE BOX_DIR LIST_EVAL LIST_TRAIN WIDE_SIZE WIDE_PX WIDE_AZ WIDE_EL TELE_SIZE TELE_PX TELE_AZ TELE_EL SCORE_KEEP BATCH DEVICE WORKERS SMOKE_CLIPS MAX_H (1.0) MIN_FREE_GB (28) CG_DIR STAGES DRY RESUME FORK_STATE_FILE`. Threads: OMP/MKL 4, one decode thread per worker, at most 4 workers. The queue never removes, moves or stops anything; it refuses under the memory floor (`memory.max` minus the `anon` line of `memory.stat`), when a START marker exists without `RESUME=1`, and writes `BOXREAD_STOP.txt` with exit 4 when the smoke projection (seconds per frame without the warm-up call, times the selected lead frames of all listed clips counted from the box files, both cameras summed) is above `MAX_H` or cannot be measured. The last line printed is `ZZ BOXREAD-QUEUE <stage> DONE rc <rc> <time>`.

Output sizes: one npz per clip of a few kilobytes; no pixel is written. Memory: a prepared clip is `n x 3 x px x px` bytes (about 180 MB wide, 320 MB tele) with `workers + 1` clips in flight.
