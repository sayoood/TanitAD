# HANDOFF_RBDSTRAT (2026-11-18, build agent RBDSTRAT)

State: the NAVRBD chain is LAUNCHED (WMI, 23:21:13 local; the UM read had already ended with DONE, so gate0 passed at once and step 1 started).
Chain: `F:/TanitAD-scratch/strat1/navrbd/read/navrbd_chain.sh` (job_navrbd_chain.cmd, launch_wmi.ps1). Log: `read/out/navrbd_read_chain.log`. Sentinel (last line): `ZZ NAVRBD READ DONE ZZ` or `ZZ NAVRBD READ FAIL ZZ <step and why>`.
Re-runnable: each finished step leaves `out/steps/<name>.ok`. Result: `read/out/RESULT_NAVRBD.json` + `.md`. Nothing staged, committed or pushed. Pod: reads and pulls only.

## Steps and checks
0 gate: UM log last line DONE/FAIL (done), F: >= 5 GB, commit >= 5.5 GB before every step.
1 `navrbd_pod.py pull-ckpts` -> `read/ckpts/N1D_s{0,1}/` (sha256 == pod sha256sum; BANK_MANIFEST.json written, commit 0c3b374f).
2 `navrbd_export.py 0c3b374f navrbd/tree` (stack/scripts, stack/tanitad, taniteval/taniteval, taniteval/tools; blob-hash verified, repo paths <= 200); `navrbd_dumps.py` 4 tags -> `read/dumps141/`; `navrbd_t0.py` real/donor/zero x 2 -> `read/out/t0/`.
3 `navrbd_view.py` (N1D shown as N1, banked C0 beside), frozen `strat1_strategic.py`, `strat1_ctx_read.py`, `navrb2_ctx_nofog.py`, then `navrbd_strategic.py` -> `out/NAVRBD_STRATEGIC_contrast.json` (`route_cost['N1D-N1R']`, NAVRB2 rule unchanged).
5 `navrbd_pod.py wait-end`: polls every 5 min, cap 6 h, for `ZZ RBDPOD ALL END ZZ main: ...; zero dumps DONE|FAIL` (per the MM ruling; the line is printed into the chain log). Then `pull-dumps` -> `read/dumps/` (every dump_NAVRBD-HO-* in out/t1 and out/t1_zero, per-file sha256).
6 both pytest files (whole), then `navrbd_read.py --out out/RESULT_NAVRBD.json --contrast ...`.

## Verified (MEASURED)
- Tests `test_navrbd_chain.py` 19 green; 11 mutation arms all RED (`out/MUTATIONS_NAVRBD_CHAIN.json`); `test_navrbd_read.py` 24 green.
- `navrbd_dumps.py` smoked against the real `v8f_t1_eval_rbd.py` (12 windows, N1D_s0 follow and zero): worked unchanged. The only fix needed was the EXPORT: the T1 analysis loads `taniteval/tools/t1_eval.py`, so `taniteval/tools` is in the export.
- `navrbd_t0.py`: all 6 real passes ran (22-80 s each).
- Dry run of the whole chain on stand-ins (real N1D T0 passes, N1R ticks under N1D names, N1R held-out dumps aliased): ended `ZZ NAVRBD READ DONE ZZ` (`out/navrbd_read_chain_standin.log`).

## Donor
At T0, donor = `--nav-shuffle` (base nav swapped for a donor clip's) with `--nav-rb-variant zero`: the RB channel is held at ZERO beside the donor nav. This is exactly what NAVRB2 did for N1R (READING r4), so the contrast is like for like. The wrapper refuses shuffle with `follow`; no new variant was invented.

## NOT-COMPUTABLE / NOT-YET (with reasons)
- Held-out Z-EVAL mask (step 4): NOT-COMPUTABLE from the B2 tables (they hold the 4,572 train clips and the 141 eval clips; the held-out episodes, ids up to 9996, are not in them). No mask file; that cell reads NOT-YET. Z-EVAL on eval-141 is still read.
- Frozen STRATEGIC view and ctx reads: no S1D arm exists, so `strat1_strategic.py` prints S1 NOT-YET and its route-following label NOT-YET ("needs both seeds"), and `strat1_ctx_read.py` / no-fog ctx print all five variables NOT-YET (the ctx head belongs to the S1 family). Only the N1D - N1R contrast (ROUTE-COST) is computed, which is what the read needs. Fog unscored.
- No K4 pass (no S1D arm).
- N1Dx_s0 dump: if absent, the read prints NOT-YET for it.

## Temp deleted
`C:/Users/Admin/rbdstrat` 273,952,437 bytes (tree export, 2 smoke-pulled N1D checkpoints [source on the pod], smoke dumps, stand-in passes, dry-run out), by literal path, logged in `out/DELETIONS_RBDSTRAT.log`. `F:/.../navrbd/read/tmp` none created. The chain deletes its own view and pytest basetemp.
Kept on F: `navrbd/tree`, `read/ckpts`, `read/dumps141`, `read/dumps`, `read/out/t0` (evidence for the read).

## MM next
Read the last line of `read/out/navrbd_read_chain.log`. On FAIL the step is named; fix and relaunch `launch_wmi.ps1` (finished steps are skipped). Bank ckpts, tree sha, dumps and RESULT_NAVRBD.
