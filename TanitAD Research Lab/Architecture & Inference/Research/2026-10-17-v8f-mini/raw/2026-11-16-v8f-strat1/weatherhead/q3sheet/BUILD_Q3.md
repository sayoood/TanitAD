# BUILD_Q3: the weather label audit sheet (PI's Q3, approved 2026-11-17)

Evidence class of the build facts: MEASURED (ours, files below). All CPU, `CUDA_VISIBLE_DEVICES=-1`, one python job at a time, `wait_commit.py 6 120` passed (14.66 GB free), TMP/TEMP under `q3sheet/tmp`.

## What was reused
* The instrument of `H-CTX-LABELS-V1` (register row on branch `agent/arch-inf-20260803`, not on `main` and not in the E: working tree; probed the mirror's `origin/agent/arch-inf-20260803`, tip `93c5406c`). Its folder `raw/2026-11-09-context-labels/` gave: the single-file local HTML page pattern with JPEG data URIs (`build_pilot_audit.py`), the frame convention (label frame k, video frame 6k at 30 fps, `strat_common.py`), the camera folder `C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov`, ffmpeg frame extraction, and the PI's answers (`pilot_audit_answers_PI_60.json`: 60 frames, 60 clips, 40 of them the random stratum). Copies used by the build are in `q3sheet/ref/`.
* Label source: `F:/TanitAD-scratch/strat1/labels/strat_labels_table.npz` (sha256 `8055a321...cf184eff`), field `train_weather_id`, the label the weather head trained on. Train windows follow `wprobe_fit.py` (8 NOW steps per clip). The re-count reproduces clear/fog/rain/snow 30,988 / 831 / 1,647 / 2,585 = 36,051.
* The head's train cache (`speedread/cache/train`) holds features only; it was used for the window rule, not for images.

## The draw
Pre-registered in `PREREG_Q3.md` (sha256 `4d17a3dd31913374e6f16375994111aad5ed2bef3e36ec41e13075cebcd0db21`, in `PREREG_Q3.sha256`, written before `build_q3.py` ran). Seed `q3-weather-v1`, order permutation seed 20261117.
* 40 frames from 40 distinct clips: fog 11, rain 11, snow 11, clear 7 (`draw_report.json`).
* Eligible clips after the exclusion: clear 4,074, fog 252, rain 362, snow 443.
* Excluded: the 60 clips of the PI's earlier answers (none drawn: checked, 0).
* 35 of 40 frames also show the frame 1 s earlier (the other 5 have k < 5).
* The pre-registered bar: per-class agreement >= 0.80 sound, 0.70 to 0.80 borderline, < 0.70 relabel candidate, n answered < 8 not auditable.

## Files and sizes
`Q3_SHEET.html` 1,142,528 bytes (about 1.1 MB, under the 8 MB limit, no external request); `key.json` 7,238 bytes; `score_q3.py` 3,475; `build_q3.py` 8,570; `draw_report.json`; `PREREG_Q3.md`.

## Blindness checks
The sheet contains 0 of the 40 clip ids and none of the label-file words; a frame shows only its opaque id (8 hex) in the JSON the PI copies. `key.json` (id -> clip, step, frame index, label) is never read by the sheet. Do not open `key.json` before answering.

## How the PI uses it
1. Open `F:/TanitAD-scratch/strat1/whead/q3sheet/Q3_SHEET.html` in a browser (works offline, about 20 to 30 minutes).
2. For each of the 40 frames pick clear, fog, rain, snow or can't tell; add a note if wanted. The page keeps a draft in the browser.
3. Press "Copy answers" and paste the JSON into a file, for example `answers_q3.json`.
4. Score: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe q3sheet/score_q3.py answers_q3.json --out scores_q3.json`. It prints per-class agreement with Wilson 95 % intervals, the pooled and prevalence-weighted overall, the 4 x 4 confusion (with a can't tell column) and the bar reading per class.

## Scorer test (planted answers, `tmp/`, deleted below)
All-match answers: agreement 1.0 for clear, fog, rain, snow and pooled 1.0. One planted rain-to-fog answer: exactly two confusion cells changed (rain/rain 11 to 10, rain/fog 0 to 1), the rain agreement read 10/11 = 0.909, the other three classes stayed 1.0.

## Deliverable manifest
| file (`F:/TanitAD-scratch/strat1/whead/q3sheet/`) | role | to git? |
|---|---|---|
| `PREREG_Q3.md`, `PREREG_Q3.sha256` | pre-registration and its hash | yes |
| `score_q3.py` | scorer | yes |
| `key.json` | answer key, labels and clip ids only, no images (land it after the PI has answered, so the key is not at hand while answering) | yes, after the answers |
| `build_q3.py`, `draw_report.json`, `BUILD_Q3.md` | build, draw record, this note | yes |
| `Q3_SHEET.html` | the sheet, holds 75 embedded camera frames | NO (frames never go to git) |
| `ref/` | copies of the earlier instrument and the PI's earlier answers (from branch `agent/arch-inf-20260803`, re-creatable by `git show`) | no, source stays in the repo |
Single-copy items: `Q3_SHEET.html` (re-creatable by running `build_q3.py`, deterministic); all others re-creatable or in this folder only. Nothing staged in git (the MM lands). Nothing sent to any external service.

## Deletions (literal paths)
`F:/TanitAD-scratch/strat1/whead/q3sheet/tmp/` (planted answers, scorer outputs, one decoded test frame): 28 KB on disk by `du`, deleted after this note was written; re-creatable.
