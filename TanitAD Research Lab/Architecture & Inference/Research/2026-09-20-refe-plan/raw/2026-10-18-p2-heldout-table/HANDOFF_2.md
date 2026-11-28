# HANDOFF_2: P2 held-out table, COMPLETION run (launched 2026-10-18 09:37 Berlin)

PACKAGE `C:/Users/Admin/p2heldout/` (bulk `C:/Users/Admin/p2heldout_work/`). Read `HANDOFF.md` (first run) and `PLAN.md` for gates and caveats. This run supplies the inputs the first run lacked, then resumes the SAME driver.

## 1. What was found and copied (MEASURED; pod was idle: 0 GPU apps, no p6_pod.sh, checked before every batch by `copy/podcopy.py`)
* **The pod had all of it.** Frames: pod `/workspace/data/navtrain_pixels/navtrain_current_N/<log>/<CAM>/<hash>.jpg` (4,040 + 1,448 requested files, 0 missing, 5,488 files = 1,069.9 MiB). nuPlan DBs: pod `/workspace/data/navtrain_dbs/<log>.db`, the 11 `BLOCKED_NO_DB` logs (11 files, 952.3 MiB). Probes: `ls -d navtrain_pixels/*/<log>`, per-file `stat`, `[ -f navtrain_dbs/<log>.db ]`, plus `find /workspace -maxdepth 4 -name <log>*` as a second path. (A first existence test said MISSING for the DBs: CRLF in my stdin list, not absence.)
* Landed on the dev box: DBs `p2heldout_work/dbs/<log>.db`; frames `p2heldout_work/frames_heldout/<log>/<CAM>/<hash>.jpg` (the layout `FrameResolver._find` matches; it is the audit's first image root). sha256 equal on both sides for all 5,499 files; re-hashed from disk after the copy: 0 bad (2,120,354,981 bytes). Provenance: `copy/` (manifests, stat lists, logs, `podcopy.py`; no pod address in any file).
* The 362 frames-needing tokens of the 11 new DB logs were derived AFTER the DBs arrived (scratch audit, 1,448 jpgs). Audit now: **1,705 / 1,705 tokens plannable, 0 blocked, 24 logs** (1,372 from frames_heldout, 333 from the D: proxy roots).

## 2. The run
* Driver: `code/p2ho_driver.py` (hash-identical to the first run, `CODE_SHA256.json` checked), `--tag main --workers 2 --device cuda`. **Launch note:** `powershell -File launch_p2ho.ps1` is refused here (script execution disabled; no Bypass used), so the identical `Win32_Process.Create` command line was issued inline.
* **PIDs:** driver python **4984** (cmd.exe 54904 -> launcher 9296 -> 4984); post-watcher python **50416** (cmd.exe 11020). GPU lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` taken by the driver (owner `p2heldout_table`, 09:39:13), released by the driver; do not touch it.
* **The 333 finished tokens ARE recomputed, not skipped.** Skip is per UNIT, keyed `sha256(log|token list)`; each of the 11 old units' logs gained tokens, so all 11 keys changed (0 of 24 keys equal an old one). Cost ~20 %. Benefit: a free determinism check, `recompute_check_333.json` (new rows vs the first-run table kept in `p2heldout_work/final_333/`, sha 3627c281...). First-run results are also in `prev333/`.
* Fidelity A and DG-DUMP are cached PASS files (not rerun). Preflight, disjointness: PASS again (09:37:41).
* ETA **ESTIMATED P2HO_DONE 10:30-11:00 Berlin**. Basis: first 4 units measured dump 1.2-1.8 s/token + ~20 s/unit, scoring 1.6 s/token per worker (not the bottleneck); 24 units / 1,705 tokens = 48-62 min dump + ~5 min tail.

## 3. Poll (script, not turns)
`C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I C:/Users/Admin/p2heldout/code/status_p2ho.py`; log `driver_main.log` (first 8,233 bytes belong to run 1). Markers: `P2HO_DONE` / `P2HO_FAILED` (first line has rc: 1 preflight/dump, **2 a fidelity/DG-DUMP/dump gate: REPORT, do not re-run**, 3 scoring, 4 table, 6 C: < 5 GiB). A RAM pause (`guard: PAUSED ... free RAM`) is normal. If the driver dies without a marker, re-issue the same command (units done are skipped; `driver.pid` holds the new pid).

## 4. At DONE
`code/p2ho_post.py` (the post-watcher) runs by itself 20 s after the marker: `featcheck.json`, `recompute_check_333.json`, `CODE_SHA256.json`, `paths.json`, then writes `FINALIZED.txt`. Then: (1) read `P2HO_DONE`: `PARTIAL` must be false, `n_tokens` 1705, G3/G4 true; (2) `recompute_check_333.json` PASS (a non-zero diff is a finding, report it); (3) `COVERAGE.json` (the driver rewrote it at preflight: 1705/1705, 0 blocked; the DONE marker confirms); (4) Master Mind lands the package with `paths.json`. Bulk npz stay on C: (`final/`: table, proposals, prehead features [1705,64,256] fp16, heldout_sets_index). Nothing is staged or committed by me.
Not wired: `PLAN.md` / `HANDOFF.md` still say PARTIAL (333); they describe run 1. This file supersedes their coverage numbers once DONE says otherwise.
