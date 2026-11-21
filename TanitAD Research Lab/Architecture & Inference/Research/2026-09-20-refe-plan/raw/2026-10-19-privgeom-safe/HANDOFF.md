# SUPERSEDED 2026-10-19 ~12:00: PRIVGEOM-SAFE is FINISHED. Read RESULT.md (DAC INFO-UPSTREAM primary + companion; NC INCONCLUSIVE). The text below is the pre-result handoff, kept as provenance.

# HANDOFF (PRIVGEOM-SAFE), written 2026-10-19 ~11:00 by the Research Lab agent. Read PREREG_PRIVGEOM_SAFE.md + AMENDMENT1 first.

## State
- Step 1 DONE: cost measured (`run/cost/cost.json`, sha256 `92529fc5...`): 5.31 ms/prop NAVSIM + 2.94 ms/prop scalars, 1.36 s/token, 0.53 s/set, peak RSS 0.43 GB/process. Subset = ALL 41 logs with a local nuPlan DB (4,941 of 15,808 f10 sets; 2,782 tokens; projected 1.78 CPU-h). The other 76 logs have DBs only on the pod (named blocker for widening).
- Step 2 DONE: PREREG sealed 10:39:42, sha256 `c6d6e1fb228f7e4806d5b0316699f8fef05855684d0c535bd246248217bd7084`; AMENDMENT1 (fit/analysis code pins) sealed 10:54:07, sha256 `7382a2d7c345db05afaed9b1e2084975d943c6d605d568ce5d78fc7e1ed306c9`. Both are in `.sha256` files; every stage refuses on a mismatch. DO NOT EDIT any pinned file (the gate token binds them); a needed change = a sealed Amendment 2 BEFORE the gate runs.
- Step 3 RUNNING detached: the label-free extraction (`code/pg_extract.py work`), worker 1 = cmd 34192 -> python 71604 (since 10:40); worker 2 = cmd 70460 -> python 70444 (waits at its scripted start gate for >= 4 GB available RAM; the box is busy). Finish watcher = cmd 50588 -> bash 27776/59368 (`code/finish.sh`, log `run/finish.log`): waits for 41 `.npz` and no locks, then `join` -> `gate` -> `fit` (2 shards) -> `analyze`.
- Markers (poll with a script, not turns): `run/FINISH_DONE.txt` = success; `run/FINISH_FAILED.txt` = a stage failed (reason inside; see `run/finish.log`, `run/fit0.log`, `run/fit1.log`, `run/gate_report.json`).
- Expected: extraction ~1 to 1.7 h wall, gate ~8 min (labels first opened here), fits ~20 min, analyze ~1 min.

## Exact next step (a fresh agent)
1. `until [ -f run/FINISH_DONE.txt ] || [ -f run/FINISH_FAILED.txt ]; do sleep 120; done` as a background Monitor, never a turn loop.
2. If FAILED:
   - extraction stall / stale `run/scalars/*.lock`: first confirm by `Get-CimInstance Win32_Process` that no `pg_extract.py` python is alive, kill by explicit PID only, delete the stale lock(s), relaunch a worker (pattern below) and the watcher.
   - gate FAIL: read `run/gate_report.json` (C-REACH values, suite tails). Do NOT retune lr or margins; report the failure and what it blocks. (A C-AGREE failure does not stop the fits; it VOIDs the row.)
3. If DONE: read `run/results/RESULT_tables.md` and `results.json`. Check the CONTROLS before the outcome: `gate_report.json` (C-AGREE >= 99 %, label-identity rates, C-REACH, suites), each row's null fire list and weak-plant power, `join_report.json` lost sets. Then the calls per row (`final_call`), and the companion variants (prereg 3.4: DAC-nosign, NC-floored). ⚠️ Read an INFO-UPSTREAM beside its companion: the primary G contains the label's own test (prereg 3.4).
4. Write `RESULT.md`: headline per row (final call + companion + the committed next arm of prereg section 9), the per-row per-seed table, controls, the pick-level read (descriptive), limits (41 of 117 logs, SELECTION tier, SEEN, P2 in-sample, intervals answer "another draw of LOGS" only), the 76-log blocker, and a PROPOSED register text (do not edit landed files).
5. `python code/build_package.py` (stages `out/<repo path>/` + `out/paths.json` + MANIFEST; re-runnable), then hand back <= 250 words + manifest. Stage never push; no git add/commit.

## Relaunch patterns (PowerShell tool, WMI; `code/launch.ps1`)
- worker: `powershell -File code/launch.ps1 -Cmd 'set CUDA_VISIBLE_DEVICES=-1&& set OMP_NUM_THREADS=1&& C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe -B -W ignore C:\Users\Admin\par18\privgeom\code\pg_extract.py work --subset C:\Users\Admin\par18\privgeom\run\subset.json --out C:\Users\Admin\par18\privgeom\run\scalars' -Log C:\Users\Admin\par18\privgeom\run\extract_w3.log`
- watcher: `-Cmd 'set PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%&& C:\PROGRA~1\Git\bin\bash.exe C:/Users/Admin/par18/privgeom/code/finish.sh' -Log C:\Users\Admin\par18\privgeom\run\finish_launch.log`
- Interpreter for every stage: `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -B -W ignore`, `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=8` (extraction 1).

## Facts a successor needs
- NAVSIM is the PINNED `D:/Archive/devbox-C/navsim/navsim-3e8291bf.../` (NOT `C:/Users/Admin/navsim/devkit`, whose `MetricCacheProcessor` signature differs and fails). The landed labeller path is reused unchanged; the geometry reads `PDMScorer._states` / `_ego_coords` after `navsim_pdm_targets.score_plans`.
- `run_smoke/` is a plumbing smoke (labels := recomputed verdicts, never the stored labels); NOT a result; do not stage it.
- C: had 0.02 GB free earlier (MM note); all outputs here are small (scalars ~27 MB total); verified the pinned files re-hash to their recorded sha256 after the event.

## Hard limits (unchanged)
CPU only, <= 2 Python processes (extraction counts), psutil available >= 4 GB to start and >= 3 GB while running (scripted), never the full stack suite, do not touch `srttc_s0/`, `par18/{probe,l1run,r0eb,hnfp,fastneg,pickrule,ncfastcal}` (read only), locks, GPU, pod. Kill by explicit PID only. LF files.
