# HANDOFF YP7 (2026-11-06 ~23:30Z): the yaw-lever P7 package; one detached pipeline still running (~2-2.5 h)

State: all edits are DONE and tested (below). A detached pipeline (WMI pid 40728; bash pid in `pipeline.status`) is running the long CPU jobs. Nothing staged/committed/pushed; the pod and the dev-box GPU lock untouched.

## Work tree and files
- Tree: `C:/Users/Admin/pyp7` (= raw-blob export of tip `b17b5492`'s PK + YT's 7 files + YP7 edits). DO NOT TOUCH IT while the pipeline runs (the pre-run binds its tree digest before and after).
- Final mirror (written by the pipeline's stage D): `E:/Projects/TanitAD-artifacts/yawrate_wp/YP7/tree/<PK-relative path>` for the 13 changed/new files; `CHANGES.md` (tip blob = land base guard, final blob); `RESULT.txt`.
- Narrative of every change and decision: `CHANGES_head.md` (becomes the top of `CHANGES.md`).

## Pipeline (`run_pipeline.sh`, log `pipeline.status`, `pipeline.out`)
A sanity `suite-prerun` (`suite_prerun_PRE.*`) -> B `p7_ship_list.py derive --trace` (fresh trace, `ship/derive.log`, `ship/p7_trace.json`) + `check` (must read ZZP7SHIP_OK, `ship/check.log`) + yaw files present in the list -> C FINAL `suite-prerun` on the exact final bytes (`suite_prerun_FINAL.json/.out`, must read ZZSUITE-PRERUN-PASSZZ with 0 skips) -> D `finalize.py` (mirror, CHANGES.md, RESULT.txt) -> E full local `test_ego_yaw.py` (SLOW block + mutation harness, `logs/test_ego_yaw_full.log`, non-gating).
Why derive BEFORE the final pre-run: `refe/p7_ship_list.json` is inside the gate's tree digest (`.json` is a tree suffix).
Done = the last line of `pipeline.status` is `ZZYP7_PIPELINE_DONE`; a stop reads `ZZYP7_PIPELINE_STOPPED <stage>` (then read that stage's log; a failing script is named per line in `suite_prerun_*.out` or `ZZP7SHIP_TRACE` lines in `ship/derive.log`).
Check: `cat E:/Projects/TanitAD-artifacts/yawrate_wp/YP7/pipeline.status`; if the process died without a final line, re-run `launch.cmd` via WMI (`Invoke-CimMethod Win32_Process Create`, `cmd /c ...\launch.cmd`); every stage is rerunnable.

## MEASURED so far (CPU, driverl-eval, `logs/`)
`test_p7_chain.py` ZZP7CHAIN_PASS 110 (was 81); `test_p7_integration.py` ZZP7INT_PASS 225 (was 221) and `--mutations` ZZP7INT_MUTATIONS_OK (M26-M28 new, all RED); `test_p7_head_wiring.py --static` ZZP7HEAD_PASS 182; `test_ego_yaw.py --fast` ZZEGOYAW_PASS 160, 0 `ZZTEST_SKIP` lines. The full pre-run and the ship-list check are PENDING (pipeline).

## Next step for whoever reads this
1. Read `RESULT.txt` (or `pipeline.status`). 2. If DONE and green: MM lands the 13 files from `tree/` with the base guards in `CHANGES.md`, regenerates `SHIP_MANIFEST.json` for the landed commit, re-runs `suite-prerun` if the pod gate needs evidence bound to the landed commit (the pre-run is bound to the OLD manifest's commit `22b40687`), re-ships (ship-list `check` must still read ZZP7SHIP_OK at the landed bytes), replaces the waiting chain. 3. If stopped: fix the named stage; do not edit the tree without re-running from stage A.
Open: the pod's table-build duration and the 100 % coverage of the 1,192 DBs are UNVERIFIED without a pod (chain refuses on any shortfall); `ego_yaw_table.py control` is not in the chain (dev-box only, YT ran it).

Early safety copy (bytes final except `refe/p7_ship_list.json`, which the pipeline rewrites): `YP7/tree_pre_derive/<PK path>` (12 files). Tip blobs for the base guards: see the table `CHANGES.md` gets at stage D; until then `git rev-parse b17b5492:"<PK>/<path>"` in `C:/Users/Admin/tanitad-push`.
