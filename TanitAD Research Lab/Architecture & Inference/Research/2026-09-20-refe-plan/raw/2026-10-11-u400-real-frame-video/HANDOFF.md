# HANDOFF -- u400 real-frame video (written 2026-10-11 19:05 local, dev box)

**State: everything is built and tested; a fully autonomous background pipeline is running. Nothing needs a live agent until `pipeline_done.json` exists.**

Package: `D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-11-u400-real-frame-video/` (call it `PKG`)
Python for everything: `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`, cwd = PKG.

## What is running (started ~18:13 / 19:00)
1. `run_inference.py` (log `inference_run.log`): u400 re-run on the 1,090 video tokens, one log at a time, each log gated against the banked full-navtest plans.
   **6 of 10 logs done, every gate PASS with max |delta| = 0.000 (bit-identical)**. Remaining 4 logs ~8 min each (ETA ~19:35). Holds the dev-box GPU lock.
2. `run_pipeline.py` (log `pipeline.log`): waits for `inference/FIDELITY_GATE.json`, then renders 94 clips (4 workers, CRF 18 slow, segments in `segments/`, resumable),
   intro/outro cards, concat -> `E:/Projects/TanitAD/data/videos/u400_realframes.mp4`, then `qa_video.py --stills` -> `qa_report.json`, `stills/`. ETA ~20:00-20:15.
   Writes `pipeline_done.json` and prints `ZZPIPELINE_DONE`.

## Check
`tail -3 inference_run.log pipeline.log` ; `ls pipeline_done.json GATE_FAIL.json`.
* `GATE_FAIL.json` exists => fidelity gate failed: STOP and report it (a video of a different decision than the scored one is not this model).
* pipeline step failed (`pipeline_done.json` has `failed`) => read `pipeline.log`, fix, rerun `python -u run_pipeline.py` (resumable; `run_inference.py` skips finished logs).

## Exact next steps once `pipeline_done.json` has `"ok": true` (about 10 min of work)
1. Look at `stills/*.png` (3 report stills + `check_calibration_logged_agent_boxes.png`) and one mid-video frame; confirm no overlay/HUD overlap and `2 Hz (held)` labels on the held clips.
2. `python write_result.py` -> `RESULT_VIDEO.md` (generated from FIDELITY_GATE.json, qa_report.json, probe_db_rates.txt; read it, fix any wording).
3. Cleanup: `rm -rf tests __pycache__`; `segments/` holds git-ignored mp4s (leave them or delete after the final mp4 is verified).
4. Stage by EXACT path (never `git add <dir>`; verify with `git status --short -- <PKG>` -- a new directory can silently no-op):
   `*.py`, `*.md`, `*.json` (video_runs, video_tokens, video_meta, qa_report), `RUNS.md`, `tokens_by_log/`, `manifests/`, `stills/`, `probe_db_rates.txt`,
   `inference/FIDELITY_GATE.json`, `inference/*.gate.json`, `inference/*.seam.report.json`, `u400_video_proposals.npz` (merged proposals+logits, ~7 MB),
   `inference_run.log`, `pipeline.log`. Do NOT stage `inference/*.seam.log` (thousands of warning lines), per-log `*.dump.npz`/`*.seam.npz` (duplicates of the merged npz), `segments/`, `tests/`.
5. Report <= 300 words: mp4 path + size (`ls -l`), duration, fidelity max |delta|, provenance PSNR, share of all-four-real frames, failure tokens shown, known limits, and the **escalation: MODEL_REGISTRY.md has no u400 row**.

## Facts you will need (all MEASURED, details in RESULT_VIDEO.md after step 2)
* Checkpoint `E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt` md5 `8e382eebb35b3e155862453617e5786e` (verified). Banked dump arm `REFe_sft4_B_u400_navtest_full`, rule `navsim_v1_lane`.
* Camera rate 9.999997 Hz (dt 99.89/100.009/100.457 ms), lidar 20 Hz, ego_pose 100 Hz (`probe_db_rates.txt`).
* Only ONE navtest log has raw L0/R0/B0 (`2021.05.25.14.16.10_veh-35_01690_02183`, clips 1-10); the other 9 logs hold the 2 Hz side/rear frame, labelled on screen. No downloads were made.
* Design decisions: front frame native 1920x1080 at its own timestamp; plan projected with the frame's own DB ego_pose (3-D) + DB calibration incl. Caltech distortion (points, not image); drivable area = NAVSIM DAC polygons (lane groups + intersections + car parks), labelled as a rendering aid.
* E: has only ~29 GB free; the final mp4 should be ~1 GB.
