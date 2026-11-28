# BUILD_Q3RAIN: second blind rain draw and fog provenance (2026-11-17)

Evidence class of the build facts: MEASURED (files below). CPU only, `CUDA_VISIBLE_DEVICES=-1`, one python job at a time, TMP/TEMP under `q3rain/tmp`. Nothing committed, nothing sent to any external service.

## What was built
* `PREREG_Q3RAIN.md`, sha256 `8c9d95aacd6319501a7b6b9178edc9f84992ed2df9c43d80c69310d01727b0e9` (`PREREG_Q3RAIN.sha256`), written before the draw. Pooled bar from Q3's bar: >= 0.80 sound (25 of 31 or more), < 0.70 relabel candidate (21 of 31 or fewer), 0.70 to 0.80 borderline with the reason stated (interval about 0.17 wide at n = 31); fewer than 25 answered is not auditable.
* `build_q3rain.py` imports `build_q3.py` from `q3sheet/` (its `draw()`, `jpeg()`, `TPL`, `H`) and changes only seed (`q3rain-weather-v1`), quota (rain 20), exclusion list (the PI's 60 clips plus the 40 Q3 clips = 100) and order seed (20261118). No logic is copied. Output: `Q3RAIN_SHEET.html`, `key_q3rain.json`, `draw_report_q3rain.json`.
* The draw: 20 rain-labelled frames from 20 distinct train clips (349 eligible rain clips after exclusion); 16 also show the frame 1 s earlier. Overlap with the 100 excluded clips: 0 (asserted, and the Q3 key separately). The sheet holds none of the 20 clip ids (grep count 0).
* `score_q3rain.py` reads both answer files and both keys, and reports the pooled 31-frame rain agreement with Wilson 95 %, the answer distribution, the Q3 11 alone and the second draw alone (reference only).
* Scorer test on planted answers (all passed): all rain gives 31/31 = 1.0; one planted mismatch gives 30/31 and the second-draw count 19; 7 can't tell gives n = 24 and NOT AUDITABLE; 9 mismatches gives 22/31 = 0.7097 BORDERLINE; 10 mismatches gives 21/31 = 0.6774 RELABEL CANDIDATE. With the real Q3 answers and an empty second file the pooled rain count reads 8 of 11 (rain 8, clear 1, snow 2) and NOT AUDITABLE, as it should.
* `FOG_PROVENANCE.md`: the fog label is Qwen2.5-VL-3B letter answers (1,000 clips, 1 Hz) plus a SigLIP 2 logistic probe on the other frames, smoothed; no chain-of-thought text. 831 fog windows in 260 train clips (92 Qwen, 739 probe); 9 of the PI's 10 answered fog frames are probe frames. Options for a stronger labeller listed, nothing built.

## Sizes
`Q3RAIN_SHEET.html` 665,030 bytes (about 0.65 MB, offline, no external request); `key_q3rain.json` 3,618; `draw_report_q3rain.json` 588; scripts about 2.6 KB each; `FOG_PROVENANCE.md` 8.4 KB.

## How the PI uses it
1. Open `F:/TanitAD-scratch/strat1/whead/q3rain/Q3RAIN_SHEET.html` in a browser (offline, about 10 minutes for 20 frames; the draft is kept in the browser).
2. Per frame pick clear, fog, rain, snow or can't tell (the label is hidden, so any answer is possible); add a note if wanted.
3. "Copy answers", paste into a file, for example `answers_q3rain.json` (same `q3-answers-v1` format).
4. Do not open `key_q3rain.json` first. Score: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe q3rain/score_q3rain.py q3sheet/answers_q3.json q3rain/answers_q3rain.json --out q3rain/scores_q3rain.json`.

## Manifest (all in `F:/TanitAD-scratch/strat1/whead/q3rain/`; one place only unless noted)
| file | role | to git? |
|---|---|---|
| `PREREG_Q3RAIN.md`, `PREREG_Q3RAIN.sha256` | pre-registration and hash | yes |
| `build_q3rain.py`, `score_q3rain.py` | build and scorer | yes |
| `draw_report_q3rain.json`, `BUILD_Q3RAIN.md`, `FOG_PROVENANCE.md` | draw record, this note, provenance | yes |
| `key_q3rain.json` | id to clip and label, no images | yes, after the PI has answered |
| `Q3RAIN_SHEET.html` | sheet with up to 36 embedded camera frames | **NO** (raw frames never go to git; re-creatable by `build_q3rain.py`, deterministic) |
Integration: the MM lands the git-eligible files after the answers; the scorer imports `q3sheet/score_q3.py` and `build_q3.py`, so those two must be landed together with it (landing path layout is the MM's call).

## Deletions
`F:/TanitAD-scratch/strat1/whead/q3rain/tmp/` (planted answers, exclusion list, provenance scratch scripts and tables): 49 KB by `du`, deleted at the end of this job; all re-creatable (the table and numbers are in `FOG_PROVENANCE.md`). The two background shell jobs I started for a recursive search of `F:/TanitAD-scratch/strat1` were stopped; no files were deleted by them.
