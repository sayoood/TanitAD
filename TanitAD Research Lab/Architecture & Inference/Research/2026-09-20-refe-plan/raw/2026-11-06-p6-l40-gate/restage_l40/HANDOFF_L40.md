# HANDOFF_L40 -- P6 restage on the fresh L40 pod (written 2026-11-06 09:45Z). No pod id/ip/port/token in this file; token file = `<TOKEN_FILE>`, pod = `<POD>`.

## What is running (MEASURED at 09:40Z)
`restage_v2.sh` (pod copy, edited, see below), launched detached by `setsid -f nohup`, driver pid **671**, started 09:37:51Z.
Done: preflight (599 GB free, 20 GiB write ok), code (tip tree `b0fad37d9704`, 272 files, d7fix 6/6, pod_helpers `2d26ea95...`), bank3c (rank1 `2a63acc2`, rank0 `daa23b96`), quota (110 GiB write ok, 78 s; hard links verified on this overlay).
Running tracks (pids in `$O/track_pids`: env 1487, dbs 1488, pix 1490, d9 1494, restore 1498): teacher_env at "venv" step, dbs 10/1192 at ~35 MB/s (ramp-up; watch it), d9 tar 3.8 GB in the first minute, pix shards 1-2, restore waits for teacher_env. Then prebuild -> twin_parts (8 workers) -> twin_link (+D7 K=1,2,3) -> final chain. ESTIMATED 2.5-3.3 h from start (training ~12:15-13:00Z).

## Paths (on the pod)
`O=/workspace/ops/restage_v2`: `driver.log` (grep `ZZRS_`), `driver.out`, `FAILED` (exists only on failure), `ALERT_COVERAGE.txt`, `ALERT_ARGS.txt`, `guard.log`, `dbs.log`, `pix.log`, `track_*.out`. Chain: `/workspace/data/refe_p6/chain.log`, `p6_stdout.log`, `teacher_env.log`, `restore.log`. Run dir `/workspace/data/refe_runs/p6_paper_faithful`. Driver files: `/root/tad/restage_v2/` (`restage_v2.sh`, `restage_v2.sh.orig`, `restage_helpers.py`, `pod_helpers.local.py`).

## Settings I changed / pinned
1. **Expected resume step 4321 -> 6266**: ONE place, `restage_v2.sh` line 51 on the pod (`EXPECT_STEP="${P6_EXPECT_RESUME_STEP:-6266}"`). It feeds `verify_resume` (after the prebuild and right before the final chain) which reads `ckpt_last.pt` + last `ckpt` event of `metrics.jsonl` from disk. `restage_helpers.py` has no hardcoded step. `P6_RESUME_REV` is unset (resume from HF main). Diff: `pod_l40.diff`.
2. **Label workers 6**: the driver already pins `RS_LABEL_WORKERS` default 6 and writes `P6_LABEL_WORKERS=6` into `p6_start.env`; the launch also passed `RS_LABEL_WORKERS=6` explicitly. The cgroup here is 15 cores, so the chain would have derived 13. STATIC check (recomputed from p6_pod.sh at the tip + the pod paths): LW=6 -> ARGS sha `570619873b1ebaa0` (= the run's), LW=13 -> `f74aee47cf0b708f`. The real value comes from `ZZRS_ARGS_OK` in driver.log.
3. Unchanged on purpose: `P6_D7_FALLBACK=0`, `P6_CBANK=/workspace/data/refe_navtrain/train_grow_cmd_3c`, twin link + D7 check before the chain, coverage guard, quota step (110 GiB on the 600 GB overlay passed; no edit).
Only the launch path differs from the README (`/root/tad/restage_v2/` instead of `/workspace/ops/restage_v2/`; state still in `$O`).

## Exact next checks
1. Run the watcher (it polls every 5 min, one ssh call each): `bash C:/Users/Admin/par18/restage_v2/watch_l40.sh` (background/Monitor). Exit lines: `WATCH_DRIVER_FAILED|DRIVER_DEAD|CHAIN_EXITED|ALERT_ARGS|ALERT_COVERAGE|GUARD_TIMEOUT|POD_UNREACHABLE|CHAIN_LAUNCHED|GUARD_OK|MAX_POLLS`. After `WATCH_CHAIN_LAUNCHED` re-run with `WATCH_PHASE=guard` for the guard verdict.
2. On `CHAIN_LAUNCHED`: `grep -a ZZRS_ARGS_OK $O/driver.log` must show `ARGS sha 570619873b1ebaa0`; the chain's launch gate token must be PASS (no hand-written token); first `step` lines in `metrics.jsonl` must continue from 6266.
3. Guard: `ZZGUARD_OK` (first 5 labeller events no_parts == 0). `ALERT_COVERAGE.txt` does not kill training; investigate the twin link.
4. On FAILED: read `$O/FAILED` and the tail of `driver.log`; fix the cause; re-run the same launch command (resumable, `$O/<step>.done`): `cd /root/tad/restage_v2 && RS_LABEL_WORKERS=6 setsid -f nohup bash /root/tad/restage_v2/restage_v2.sh > /workspace/ops/restage_v2/driver.out 2>&1 < /dev/null`.
Kill only by explicit PID (never `pkill -f`/`pgrep -f`).

## Open risks (UNVERIFIED)
DB fetch rate (35 MB/s early) and teacher-env time; HF main moving past 6266 (verify_resume would STOP, loud); D9 file count 107,860 is inherited from the packer log; G-LIVE-LAB's 6.735 sets/s bar with 6 workers has not been measured on this pod (15 cores, so more headroom than the 8.5-CPU pod it was measured on).
