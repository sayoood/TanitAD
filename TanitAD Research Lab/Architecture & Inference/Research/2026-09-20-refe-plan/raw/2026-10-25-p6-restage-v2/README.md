# restage_v2 -- fresh RunPod pod -> "P6 training resumed" (2026-10-25)

`restage_v2.sh` replaces the 2026-10-22 `rebuild_driver.sh` (~1.5 days: it rebuilt the D9 cache 5.2 CPU-h and the 3-C bank 23 h). Now both are FETCHED from `Sayood/tanitad-refe-navtrain-banks` (size + sha256 + member layout checked, `layout_check.txt`). Resumable (`$O/<step>.done`), any mismatch STOPS (`ZZRS_FAIL`, `$O/FAILED`). Token only from `<TOKEN_FILE>` into child envs / a mode-600 curl header file deleted before the chain. `RESTAGE_DRY=1` prints every step + check, no network, no writes (`dry_run_output.txt`). Ship: `ship_restage_v2.txt` (base64 + sha check); start only after the token is placed.

| step (track) | est. wall | what the checks prove |
|---|---|---|
| preflight | 1 min | token file, tools (zstd via apt), cgroup cores, local disk df>=360 GB **and** a real 20 GiB write, GPU idle |
| code | 3-6 | treeless clone; tip tree SHIP_MANIFEST = 272 files / tree `b0fad37d9704` (the identity P6 trained under); G-SUITE extra files, d7fix 6/6, bank3c code == commit blobs; `pod_helpers.py` sha |
| teacher_env (A) | 15-25 | venv + maps (the script's own CUDA conv2d / import checks) |
| dbs (B) | 30-60 | 1,192 DBs from nuPlan S3 to **local** `/root/tad`, symlink back; the tool CRC-verifies |
| pix (C) | 30-60, overlapped | 32 shards, authenticated `curl -f`, local disk + symlink, `.done_N` |
| quota (E) | 3-5 | real 110 GiB write on /workspace (df cannot see the quota) |
| d9 (D) | 10-18 | `curl -f`; size 36,282,746,880; sha256 == HF LFS oid; first member `metric_cache/`; 107,860 files; **0 lane-rank files** |
| bank3c (D) | 1-2 | size/sha; `targets_rank0` daa23b96, `rank1` 2a63acc2 (the bank P6 trained on) |
| restore (A') | ~4 (measured) | the chain's own `p2r_sync restore` (train_grow + calib, md5-pinned) |
| prebuild | 25-35 | bcc04b5 chain, `STOP_AFTER=cache`: mints `commands.ok`/`cache.ok`, rebuilds pre-3C `train_grow_cmd` (shas daa23b96 / 2f3e4613), cache stage must build **0** parts, resume point verified from disk |
| twin_parts | 31 (PI figure) | old `build_twin_parts_3c.sh` + the 2026-10-22 PYTHONPATH fix (its own D7 gate K=1,2,3) |
| **twin_link** | 3-6 | `pod_helpers parts plan`: plan_ok, missing_src=0, conflict=0, distinct=21,821; `apply --how link`: ok, errors=0, copied=0, dst_missing_after=0; then `p6_prep.py lane-distinct` `ZZD7_DISTINCT_OK` for K=1,2,3 **before** the chain |
| final_chain | ~45 (measured parts) | tip chain, `P6_D7_FALLBACK=0` (BAD refuses), `P6_CBANK` 3-C; guard: first 5 labeller events after launch must have `no_parts==0` else `ALERT_COVERAGE.txt` (training NOT killed); `ARGS sha 570619873b1ebaa0` else `ALERT_ARGS.txt` |

**Wall estimate (ESTIMATED, not measured end to end): ~2.5-3.3 h to training, vs ~36 h before.** Critical path: DB fetch (bandwidth unknown) -> prebuild -> twin parts -> link -> chain (lab 2.7 + mem 8 + gate 24 + preflight 1.6 min, from the 2026-10-22 chain.log). `RS_TWIN_BANK=3c` builds only the 21,821 kept twins, overlapped with the prebuild: ~2-2.7 h, UNVERIFIED (default is the rehearsed path).

**Why `train_grow_cmd` is rebuilt, not fetched:** the pre-3C bank is on no HF archive (`refe_train_grow` = P2 bank before commands; `_banks_rest` = aug/r0/sc dirs; `_3c`, `_3c_p7` are derived). The tip chain *refuses* the commands stage once `P6_CBANK` is set and needs `commands.ok`, and the twin script enumerates twins from it; the commands stage is 19 min measured, so it is regenerated and sha-checked.

## Rollback (`P6_RESUME_REV=fc2adb00b59e`, step 2,603; unset = HF main, step 4,321)
Full sha `fc2adb00b59ef5363fe1f7175e2923ce6e265417`. Before any chain runs, `restage_helpers.py place` puts that revision's `run/{ckpt_last.pt, config.json, metrics.jsonl, snap_epoch001..006.pt}` into the run dir after: sizes, sha256 vs HF LFS oid, config byte-equal to main's, and **the downloaded metrics.jsonl's last `"event": "ckpt"` step == 2603** (`P6_EXPECT_RESUME_STEP`; required for any other rev). **Recover cannot overwrite:** `p2r_sync.recover` (f2f604f) restores run files only `if not (run/"ckpt_last.pt").is_file()`; the placed checkpoint makes it a no-op (`ckpt=no`; proven by an offline mock with a restoring control), and its stage marker is written by the prebuild chain, so the final chain skips it. The disk state is re-verified (sha of ckpt_last == placed, last ckpt step) after the prebuild and right before the final chain. **Nothing on HF is discarded:** the watcher would otherwise push the rolled-back run over main's `run/*` within 30 min, so rollback pushes go to a NEW repo `<repo>-rb2603` (`P6_RESUME_PUSH_REPO`; `same` pushes to the original and tags main's head `pre-rollback-<utc>`; HF history keeps the 4,321 checkpoint either way). Later restages must then read from the new repo (`P6_REPO`).

Not verified: any pod-side run (hard links on the new volume, DB/pixel rates, teacher-env time, tag API, `built=0` wording of a no-op cache build).
