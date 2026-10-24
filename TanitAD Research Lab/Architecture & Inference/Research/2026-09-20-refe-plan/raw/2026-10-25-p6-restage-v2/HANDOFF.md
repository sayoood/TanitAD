# HANDOFF -- restage_v2 (2026-10-25). Nothing here holds a pod id/ip/port/token; the token file is `<TOKEN_FILE>`.

## State
Built and dev-box-tested, NOT run on any pod (there is none). Files, all in `C:/Users/Admin/par18/restage_v2/` (ONE place, not in the repo, not staged):
`restage_v2.sh` (driver, 5c69e3b3f08bfaeb...), `restage_helpers.py` (stdlib: `place` / `verify` / `guard`, 6a1be646a52ee276...), `ship_restage_v2.txt` (made by `make_ship.sh`: base64 + sha check, no auto-start), `README.md`, `layout_check.txt`, `dry_run_output.txt`, `evidence/` (the probes behind layout_check).
Tested: `bash -n`; `RESTAGE_DRY=1` default and with `P6_RESUME_REV=fc2adb00b59e RS_TWIN_BANK=3c`; `zzv` key matching with pass + fail controls; `fetch_file`/`curl -f` against a local server (404 saves no body, the old bug does); `restage_helpers.py place` end to end on real HF files (verify, idempotent re-place, tampered ckpt refused), `guard` clean + dirty; offline mock of `p2r_sync.recover` (restores when ckpt absent, 0 fetches when pre-placed).

## Exact next steps
1. PI decision: resume from main (step 4,321, default) or ROLLBACK to `fc2adb00b59e` (step 2,603, `export P6_RESUME_REV=fc2adb00b59e`). Rollback pushes go to a NEW HF repo `...-rb2603` unless `P6_RESUME_PUSH_REPO=same` (then main's head is tagged first).
2. New pod: place `<TOKEN_FILE>` (pod-migration-checklist step 3; MM does it). Add the new pod's address to `land.py`'s secret-scan PAT before landing anything.
3. Ship: `bash C:/Users/Admin/par18/podx.sh <hold-s> < ship_restage_v2.txt`, require `ZZSHIP_OK` x2 and `ZZSYNTAX_OK`; then start detached (command is the last, commented line of the ship file).
4. Monitor by script, not turns: `grep -a ZZRS_ /workspace/ops/restage_v2/driver.log`; failure = `/workspace/ops/restage_v2/FAILED`; alerts = `ALERT_COVERAGE.txt`, `ALERT_ARGS.txt` there. Re-run the same command after any stop (resumable).
5. After "training": confirm `ZZRS_ARGS_OK`, `ZZGUARD_OK` in `guard.log`, first `step` lines in `$RUN/metrics.jsonl`.

## Open risks (UNVERIFIED, all fail loud)
* D9 file count 107,860 is INHERITED from the packer's log (`D9_FILES_EXPECT` overrides); a nonzero `built=` in the prebuild's no-op cache stage STOPS (`RS_ALLOW_D9_BUILT=1` accepts).
* Hard links on the new volume (rehearsed on the old MooseFS); `--how link` STOPS otherwise: then decide copy (~9 GB) with the MM.
* `train_grow.tar.zst` member list (7 files, calib_table.json) not enumerated; relies on the chain's own restore.
* DB fetch tries system python3 first (`--plan` smoke), falls back to the teacher env. Teacher-env build time, DB/pixel bandwidth: no measured rate.
* `RS_TWIN_BANK=3c` (kept-only twin build, overlapped) never run; default is the rehearsed full build.
* The HF tag call (`same` mode) is warn-only and untested.
* Rollback metrics.jsonl: its last line IS the ckpt event 2603 (read 2026-10-25), so no stale step lines follow it.
