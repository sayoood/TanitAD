# BRIEF: PRIVGEOM-CLEAN operational Amendment 1, the worker-completion bug (Master Mind, 2026-10-20 ~01:20)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/pgclean/` (this package).
- CPU only. Never commit or push. C: has about 4 GB free: write only small files.

## What happened (MEASURED by the MM, 2026-10-20 ~01:15)
The sealed read (`PREREG_PRIVGEOM_CLEAN.md`, sha256 `148ebf0f…`) stopped at 01:09 with `ZZPGC_STOP extraction worker crashed 4 times` (`run_clean/read.log`, `run_clean/CHAIN_STOP.txt`). The extraction itself COMPLETED:
- `run_clean/scalars/` holds 76 / 76 `.npz`;
- `worker1.log` ends `WORKER_DONE pid 68968 logs_done 42` and `worker2.log` ends `WORKER_DONE pid 65328 logs_done 34`, with every log `fail 0`.

The cause, MM hypothesis (VERIFY it, do not assume it):
- `code/pgc_read_chain.py` `spawn()` (:109-115) Popens the venv `python.exe`.
- On Windows that is a LAUNCHER whose child interpreter has a DIFFERENT pid. `pg_extract.py` writes `WORKER_DONE_<its own pid>.txt`.
- The chain checks `WORKER_DONE_{p.pid}.txt` with the launcher's pid (:161-163), so every clean exit counts as a crash and is respawned. The respawns find no work, exit, and after `MAX_RESPAWN` = 3 the chain stops.
- Any relaunch would stop the same way.

## Deliverables, in priority order
1. **Verify the cause** from the logs and files: the pids in the `WORKER_DONE_*.txt` names vs the `worker{k} started pid …` lines in `read.log`.
2. **The smallest fix in `pgc_read_chain.py`.** A worker that exits rc 0 counts as finished when its `WORKER_DONE` record exists under ANY pid written after that worker's start, OR when all N_LOGS are extracted and no lock is held. A nonzero rc, or rc 0 with logs still missing and no done record, is still a crash.
   - Tests: RED on the current bytes (a launcher-pid mismatch with clean completion counts as a crash), GREEN on the fix, plus a RED arm that the fix still treats an rc-0 exit with logs missing as a crash, plus a mutation arm. Use planted processes or files, not the real extraction.
3. **The amendment**, `PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md` (DRAFT; the MM seals), stating:
   - it is OPERATIONAL ONLY: no rule, bar, estimator, gate, stage order or analysis code changes;
   - `pg_extract.py`'s sha is unchanged and the 76 extracted outputs are KEPT (list their sha256 in a manifest);
   - the old and new sha of `pgc_read_chain.py`;
   - that no stage after extraction (join, gate, fit, C-POWER, C-NULL, call) has run, and no label has been read by any analysis stage.
   - Find how the chain's seal check and `code/PINS.json` / `pgc_pins.py` / `pgc_seal.py` bind code shas. Prepare the re-pin as the package's own tooling expects for an amendment (an amendment file with its own sha that the seal check accepts alongside the prereg). ⛔ NEVER edit the sealed prereg text, and never hand-write a seal.
4. **Resume behaviour:** confirm that a relaunch of the fixed chain finds 76 / 76 extracted, spawns no needless work (or harmless work), and proceeds to join → gate → fit. Do NOT launch it; give the exact launch command (`code/launch.ps1` is blocked by execution policy, so use the WMI `Invoke-CimMethod Win32_Process Create` form with `cmd /c "… < NUL >> log 2>&1"`).
5. Write `HANDOFF_OPS1.md` (1.5k tokens or fewer).

⛔ Do not run any stage after extraction; do not open any label or analysis output; do not touch `run_clean/scalars/` (read their names and hashes only).

## Hand-back (200 words or fewer)
The cause verified or refuted, the diff, the tests, what the MM must seal, and the launch command.
