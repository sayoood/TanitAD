# RESULT S4B — native 256x640 cylindrical v2ep for the 600 stage-A TRAIN clips (2026-10-17)

**Headline: the exact HF-mp4 route works with the banked builder, but it is NOT pixel-exact to the banked eval v2ep (MAE 0.76/255, max 3; cause: the PyAV 18 / swscale 9 YUV->RGB step, not torch). Poses, actions, keys, dtypes and geometry ARE bit-identical. The 600-clip build is RUNNING detached; see `HANDOFF_S4B.md`. Orchestrator decision needed on the eval side (section 3).**

## 1. What was built / is being built
| item | value | evidence |
|---|---|---|
| clips | the picker's 600 (`pull_encode_pooled.pick_clips`, train, n=600; sha12 order), disjoint from eval-141 | `raw/s4b/clips600.txt`; overlap 0 asserted in `stage_report.json` |
| builder | banked `stack/scripts/pod_build_b1_epcache.py` -> `v2_compressed.build_compressed`, UNCHANGED (png, 256x640, hfov 120, cylindrical, n_stack 3) | launched with `--clips clips600.txt --workers 6`, parity gate: kept 600 / dropped 0 |
| output | `C:/Users/Admin/tanitad-data/refav1-train600-256/<clip>.v2ep.pt` | MEASURED avg 35.5 MB/clip on the first 12 -> ~21.3 GB projected |
| hardware | CPU only: `CUDA_VISIBLE_DEVICES=-1`, `assert not torch.cuda.is_available()` in every script (Python 3.13.5, torch 2.11.0+cu128, av 18.0.0, libswscale 9.5.102). No Thor contact. | `validation_native256.json` `cuda_available: false` |

## 2. Inputs: every one proven to be the HF corpus bytes (MEASURED, `raw/s4b/stage_report.json`)
* mp4: the local `tanitad-data/physicalai/camera/camera_front_wide_120fov/<clip>.mp4` ARE the HF `camera/<clip>.mp4` files: sha256 of all 600 == HF `camera/camera_sha256.json` (0 mismatch, 0 missing, 19 s). So **no mp4 was downloaded** (brief said to pull from HF; 7 GB saved). They are pre-existing files of another dataset dir, so they were NOT deleted; the stage holds hardlinks only.
* timestamps: 600 from HF `timestamps/timestamps.tar`; the local parquets are byte-identical (0 differ).
* egomotion: 600 members from HF `egomotion/egomotion_alpamayo.tar` (the pod build's route), repacked into a STORED zip (`<clip>.egomotion.parquet`, 248 MB); the local `labels/egomotion_alpamayo/*.parquet` are byte-identical (0 differ).
* intrinsics: local per-clip CSV vs HF `calibration/camera_intrinsics.parquet` (front_wide rows, 4,719): 0 mismatch, max |diff| 1.1e-13. (per-clip cy is load-bearing: rig B is the majority.)
* The two HF tars (2.0 GB) were scratch and are deleted; token read in place, never printed.

## 3. Priority 1 — does the builder reproduce the native format? (MEASURED, `raw/s4b/validation_native256.json`, `feat_residue.json`)
No banked native copy of any of the 600 TRAIN clips exists locally (only the 141 eval eps), so the bit-for-bit test was done on **2 EVAL clips** (01be5919, 060531e1) rebuilt with the same builder from the same sources and compared with `refav1-eval141/eps`:
| field | result |
|---|---|
| key set, dtypes, shapes, `frame` (f_ref 305.5774907364391), image_h/w 256/640, codec png, projection cylindrical, n_stack 3, quality, episode_id, clip_id | identical |
| `poses` [201,4] f32, `actions` [201,2] f32 | **bit-identical** (both clips) |
| `jpeg_buf` / `jpeg_len` (PNG frames) | **NOT bit-identical.** Decoded pixels: MAE 0.768 / 0.748 of 255, max 3, ~72 % of pixels differ (13 sampled frames per clip). File size 35,344,591 vs 35,901,199 and 46,590,095 vs 47,799,439 (-1.5 % / -2.5 %). |
Two TRAIN clips (8f387977, bf5ec76e) built from the real stage have the same keys as an eval file, n=201, 256x640 png cylindrical, finite poses/actions, non-zero frames.

**Cause (attributed by experiment, `raw/s4b/exp_*.py`):** not torch — float32 vs float64 `grid_sample` differ by <=1 level on 1.0 % of pixels. It is the **decode**: PyAV 18's libswscale 9 converts yuv420p->RGB differently from whatever produced the banked eval frames. Variants on frame 0 of 01be5919 (MAE vs banked): swscale default 0.764 (interpolation flags and `src_color_range=MPEG` change nothing; ITU709 1.04; JPEG range 6.93); my own float BT.601 limited-range with nearest chroma + floor **0.18-0.25, max 2**; BT.709 0.70-0.76. So the banked frames look like old-swscale chroma replication + BT.601. **UNVERIFIED:** the pod's actual PyAV/ffmpeg version (not recorded anywhere I found). Installing an older `av` was not done (outside the brief).

**Does it matter? (MEASURED, fp32 CPU DINOv3-L, 2 clips x 5 frames; `feat_residue.json`)** rebuilt-vs-banked frames: full 640-token field cos 0.9988-0.9990, rel-L2 4.5-5.2 %; pooled 4x10 cos 0.9996-0.9997, **rel-L2 2.5-2.8 %**. With the nearest-chroma emulation: pooled rel-L2 2.0 %, pixel MAE 0.2. For scale, the HANDOFF's V3b reads pooled-4x10 rel-L2 4.6 % for the 416 route and 3.0 % even for the native route vs the shipped cache (that baseline already contains bf16/fp8 and a GPU-vs-CPU difference, so the numbers are NOT like-for-like). Reading: the residue is of the same order as the cache-to-cache noise already present, small but not zero, and DINOv3 is visibly sensitive to +-1..3 grey levels. I do NOT claim the rebuilt train frames are better matched to the banked eval cache than the 416 route; n is 2 clips.

**Decision for the orchestrator (not taken here):** (a) accept (train frames from swscale 9, eval from the banked frames: the residue is a train/eval preprocessing mismatch of ~2.5 % pooled); (b) **rebuild eval-141 through the same builder** so train and eval share one decode path (+5.3 GB, ~15 min at 6 workers; would take total output past the brief's 25 GB cap, hence not done); (c) patch the decode to the BT.601-nearest emulation (deviates from the banked builder, residue only drops 2.5 -> 2.0 %, not worth it).

## 4. Priority 2 — dry-run preflight (MEASURED)
`pull_encode_pooled.py --source local --src-dir C:/Users/Admin/tanitad-data/refav1-train600-256 --src-geom 256 --split train --n 600 --grids 4x10,8x20 --dry-run` (run on the still-empty dir, before the build):
`ZZSPLIT-OKZZ split=train pick=600 listing=4713 eval=141 train_all=4572 train==labels=True train&eval=False` · `[parity] n_in 600, n_disqualifying 0, n_kept 600` (21 in physicalai-train-e438721ae894, 0 of 40 deployed-val) · `[source] kind=local256 ... remote=False resample416=False` · `[plan] 600 to build`. Rerun after completion (HANDOFF step 2).

## 5. Timings and incident
* Staging 3 min (19 s of that is the 600 sha256). Single builder process: 57-60 s/clip (+~16 s Python/torch start) on this CPU. 6 workers: 10 clips per 127 s measured between 16:34 and 16:36 => ~12.7 s/clip, ETA ~2 h from 16:31 (about 18:30-18:45).
* **Launch incident:** the first detached launch died within seconds: `forrtl: error (200): program aborting due to window-CLOSE event` (plain `Win32_Process.Create` child that imports torch; reproduced with a 3-line probe). `CreateFlags` CREATE_NO_WINDOW|CREATE_NEW_PROCESS_GROUP (0x08000200) survives (probe ran 25 s to `survived`); DETACHED_PROCESS (0x8) did not log, CREATE_NO_WINDOW alone returned rc 21. The launcher script carries the fix and a comment. This refines the memory note "launch via WMI": WMI alone is not enough for a torch process here.
* RAM is the constraint (free 3.8 GB with 6 workers at ~1.3 GB each plus ~5 GB held by other sessions).

## 6. Completion (filled by S4c, 2026-10-17 ~18:00; MEASURED, artifacts in `raw/s4c/`)
* build.log final line: `[build] DONE ok=600 fail=0 20.67GB in 82.3min (437eps/h)` (started 16:31, finished 17:54).
* `check_train600_native256.py --workers 3` -> **`ZZCHECK-OKZZ`** in 47 s: C1 set == picker's 600 (0 missing, 0 extra), C2 eval-141 overlap 0, C3 0 bad of 600 (content verify + same keys/dtypes/geometry as an eval native file; per-clip n in {191,199,200,201,202,207,209}, poses/actions/jpeg_len consistent), C4 no `_failures.json` and DONE line `ok=600 fail=0`. `_manifest_sha256.jsonl` written in the output dir. (`raw/s4c/check_report_train600.json`)
* post-build dry-run (`--source local --src-dir ...refav1-train600-256 --src-geom 256 --split train --n 600 --dry-run`): `train==labels=True train&eval=False`, `[parity] n_in 600, n_disqualifying 0, n_kept 600`, `[source] kind=local256 ... remote=False resample416=False`, `[plan] 600 in pick, 0 already done+verified, 600 to build`. (`raw/s4c/dry_run_train_post.out`)
* determinism spot check: sha256 of `8f387977-...` (4d229504...8d3c) and `bf5ec76e-...` (28ca420f...b57d) in the output are **byte-identical** to the copies built earlier in a different process (`_valtrain`). The builder is deterministic on this box.
* final size on disk: **22,192,351,528 bytes = 22.19 GB (20.67 GiB)**, 600 files + manifests (projected 21.3).

## 7. Deliverable manifest
| artifact | where | copies |
|---|---|---|
| 600 native v2ep (in progress) | `C:/Users/Admin/tanitad-data/refav1-train600-256/` (dev box C:) | **ONE place only** (rebuildable from the staged inputs in ~2 h; sha256 manifest follows the check) |
| staged inputs (hardlinks, egomotion zip, CSV) | `C:/Users/Admin/tanitad-data/refav1-train600-256_stage/` | one place, small, rebuildable |
| code | `code/stage_train600_native256.py`, `validate_native256.py`, `launch_build_train600.ps1`, `check_train600_native256.py` in this package | repo working tree, **not staged/committed (per session amendment: orchestrator lands them)** |
| evidence | `raw/s4b/` (stage_report, validation, feat_residue, exp_decode/_decode2/_feat/_gs.py + logs, clips600.txt) | repo working tree, not staged |
| handoff / this file | `HANDOFF_S4B.md`, `RESULT_S4B.md` in this package | repo working tree, not staged |
