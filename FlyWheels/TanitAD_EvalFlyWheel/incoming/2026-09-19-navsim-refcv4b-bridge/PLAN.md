# PLAN — E2 TanitAD → NavSim bridge (refcv4b, warmup_two_stage)

Priority order (a PRIORITY ORDER, not a dependency chain). Status as of the report.

| # | step | where | depends on |
|---|---|---|---|
| 1 | Read the scorer, the loaders and refcv4b's input contract FROM SOURCE | devkit @0a380a9, `refc_v3*.py`, `v2_dataset.py`, `refcv3_arm.py` | — |
| 2 | **Measure what is on disk** before designing: stage-1 frames (0/192), bank coverage (204 stage-2), token keys (`initial_token` ≠ `scene_token`) | `code/export_agent_inputs.py` | 1 |
| 3 | SPEC.md pre-registered and staged (blob hash in `raw/SPEC_PREREG_HASH.txt`) BEFORE any score | `SPEC.md` | 2 |
| 4 | NavSim-side export of the devkit's own AgentInput per token (+ CV poses, human futures, log windows) | `code/export_agent_inputs.py` → `raw/navsim_agent_inputs.json`, `raw/log_windows_human_future.npz` | 2 |
| 5 | TanitAD-side bridge: declare → frames (sha-checked, ST/NT/BLIND) → refcv4b CPU → spline → seam + manifest per arm | `code/tanitad_navsim_bridge.py`, `code/run_bridge.py` → `raw/seam_*.npz`, `raw/seam_*.manifest.json` | 4 |
| 6 | Controls that need no scorer: K1 (in 5), K2/K3/K9/KF/KC, K5/K6, K7 | `tests/*.py` → `raw/K*.json` | 4, 5 |
| 7 | **Poll E1's `CACHE_DONE.json`** (never build a second cache) | `C:/Users/Admin/navsim/exp/metric_cache_warmup_two_stage` | E1 |
| 8 | Official scorer per arm (+ official CV, + K4 seam-CV) with count guards | `code/score_arm.py` → `raw/score_*.{csv,log,counts.json,calls.jsonl}` | 5, 7 |
| 9 | Pre-registered statistics, paired deltas, BAR-E2-1 | `code/parse_scores.py` → `raw/scores_summary.json` | 8 |
| 10 | TanitEval artifacts via the adapter + `tools/criteria_check.py` | `code/build_artifacts.py` → `raw/artifact_*.json`, `raw/criteria_check_*.txt` | 9 |
| 11 | RESULT.md / COMMS.md, stage exact paths, re-verify at end of turn | this folder | all |
| 12 | navhard_two_stage: the bridge takes the split as a parameter; frames for navhard must be BUILT (same `build_map`/`sample` geometry, bit-exactly reproducing the warmup bank as the control) | named next action | PI-authorised download + E1's navhard cache |
