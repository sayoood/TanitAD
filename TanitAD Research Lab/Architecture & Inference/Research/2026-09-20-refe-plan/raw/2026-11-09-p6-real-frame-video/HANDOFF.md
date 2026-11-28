# HANDOFF -- P6 real-frame video (written 2026-11-09 12:45 local, dev box)

**State: built, tested on real data, launched. Nothing needs a live agent until `pipeline_done.json` or `pipeline_FAILED.json` exists.**

Dir `F:/TanitAD_p6video/` (= D). Python `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`, cwd D, CUDA_VISIBLE_DEVICES=-1, no GPU.
**PIDs:** launcher 27492 (venv trampoline) and its child `run_all.py` **48264** (started 12:43:04). Log `D/run_all.log`.

## What runs
`run_all.py`: waits for `F:/TanitAD_p6read_work/markers/dump.DONE|dump.FAILED` (pass 2 was at 110/136 logs, ETA ~12:50Z) ->
`gate_fidelity.py` (slices the 1,090 tokens of `video_tokens.json` from `pass2/proposals_p6final.npz` into `p6_video_proposals.npz`; gate: dump selected pose == banked `seams/refe_p6final.npz` within 1e-3 m, argmax(navsim_v1) == pick, dump.DONE poses_bit_identical true) ->
render 94 clips (3 workers, resumable, `segments/`) -> intro/outro result cards -> concat -> QA + stills -> 1080p -> `write_result.py` -> `pipeline_done.json` -> deletes `segments/` (logged in `DELETIONS.json`).
Waits while free RAM < 6 GB before each stage and before each clip. Finished stages are in `stages_done.json`.
ETA after dump.DONE: gate seconds, render ~25-40 min, rest ~10 min.

## Outputs
`D/out/p6_realframes.mp4` (2560x1440, ~0.9 GB), `D/out/p6_realframes_1080p.mp4`, `D/RESULT_VIDEO.md`, `D/qa_report.json`, `D/stills/*.png`, `D/manifests/clip*.csv`, `D/FIDELITY_GATE.json`, `D/p6_video_proposals.npz`.

## Check
`tail -5 D/run_all.log` ; `ls D/pipeline_done.json D/pipeline_FAILED.json D/GATE_FAIL.json`.
* `GATE_FAIL.json` or `pipeline_FAILED.json` naming "dump": dump.FAILED, or poses_bit_identical false, or gate delta > 1e-3 m. NO render happens. STOP and report to the PI; do not bypass.
* Other failure: read the `why` in `pipeline_FAILED.json` and the log, fix, rerun the launcher (`python launch_detached.py`; it removes stale pipeline_* files, resumes clips and stages). Never start a second copy while PID 48264 is alive.

## Exact next step once `pipeline_done.json` has `"ok": true`
1. Read 2-3 PNGs in `D/stills/` (turn with 4 real cams, lowest-P6-score token, fastest clip): no overlay/HUD overlap, `2 Hz (held)` labels, red FAILURE on bad tokens.
2. Read `D/RESULT_VIDEO.md`; check the cards' numbers against `P6_RESULT.md` (the code asserts the levels against `level_ci.json`).
3. Master Mind lands the small files (`*.py`, `RESULT_VIDEO.md`, `qa_report.json`, `FIDELITY_GATE.json`, `video_*.json`, `manifests/`, `stills/`, `RUNS.md`, `probe_db_rates.txt`, `DELETIONS.json`; `p6_video_proposals.npz` ~1 MB). The mp4s and frames stay private: no repo, git-ignored. Only copy of the mp4s: `D/out/` on F:.

## Verified on a real test clip (before the dump existed)
Stand-in from `run/dump/dump_p6.npz` (same 64 proposals/logits for the 117 video tokens it covers; `props8` == pass-2 `proposals` format: props8[pick] == banked pose, 0 difference). Rendered 7 consecutive tokens of clip 91 (35 frames) end to end, cards, concat, QA (frames 515/515, provenance own > adjacent), 1080p, cleanup logic, dump.FAILED path. Gate: PASS on the stand-in, and FAIL (rc 2, GATE_FAIL.json) when one pose was shifted by 5 mm. Stills: `D/stills_test/`.

## Deletions so far (my own scratch)
`F:/TanitAD_p6video/test/` (62.7 MB: stand-in npz x2 8.2 MB each, test mp4s, test stills, patch scripts), `F:/TanitAD_p6video/__pycache__`. The pipeline deletes `segments/` (~0.9 GB) itself on success and writes DELETIONS.json.
Not touched: PKG (u400 package), the GPU lock, any other process.
