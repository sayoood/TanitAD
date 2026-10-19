# HANDOFF: pod-stage (SH5 FULL frames + 76 nuPlan DBs), 2026-10-19 15:50 local

STATE: NO job is running (the chain stopped at the DISK GUARD, not at an error; the partial SH5 plan finished). Nothing was started on the pod beyond `tail` (P6 guard), `stat`, `tar -c` and `sha256sum`, all at `nice -n 19 ionice -c3`, one stream, <= 120 MB batches; no GPU process, no git, no pkill.

## Result so far (MEASURED; `MANIFEST.md`, `work/*.json`)
- **Frames: 5,776/7,864 jpg staged and verified (1.204 GB, 0 failed; size = pod stat, sha256 disk = stream = pod `sha256sum`, JPEG magic).** Folder `D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/frames_full/`. The remaining 2,088 files are 0.42 GB of data but ~2.04 GiB allocated on D: (exFAT, 1 MiB clusters, MEASURED).
- **Why it stopped:** D: free fell 35.3 -> 20.2 GiB in ten minutes (this copy added ~5.6 GiB; ~9 GiB came from OTHER writers). The copier refuses a batch that would leave < 20 GiB free (the S0 stager's floor, kept).
- **Partial SH5 plan** (CPU, same command as READY.md Q1, `--images-root` = frames_full): `D:/.../sh5_ctx_2026-10-19/plan_partial_5776/` = **tokens 2,260/2,782, sets 4,018/4,941, logs 40/41** (before: 816 / 1,444 / 38). It is NOT the FULL variant (FULL = all 2,782 tokens needs the remaining 2,088 frames = 522 tokens).
- **DBs: 76 wanted, all 76 on the pod, total 11,324,555,264 B = 11.32 GB = 10.55 GiB (<= 20 GB, so allowed), 5 MiB - 1,607 MiB each; 0 transferred** (D: would be left with ~9.6 GiB). Target = `D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/` (the flat folder PRIVGEOM's `index_dbs()` reads, on D:).

## Exact next step (needs ONE Master Mind decision on D: space)
1. Free space on D: (or name another drive, or lower the guard). Needed: frames ~2.1 GiB above the 20 GiB floor, DBs ~10.6 GiB more.
2. Frames: `cd C:/Users/Admin/par18/podstage; python -B code/podstage.py frames` (resumes: skips the 5,776 verified, copies the 2,088, verifies all 7,864; `ZZFRAMES_VERIFIED 7864/7864`), then `python -B code/podstage.py plan` (writes `plan_full`, marker `ZZPLAN_FULL_OK`). If the PM lowers the guard instead: `PODSTAGE_MIN_FREE_GB=<n>` in front.
3. DBs: `python -B code/podstage.py dbs` (ascending size, resumable, `ZZDBS_VERIFIED 76/76`).
   For a run > 4 min launch detached: `powershell -File code/launch.ps1 -Cmd "C:\Python314\python.exe -B C:\Users\Admin\par18\podstage\code\podstage.py all"` (frames -> plan -> dbs -> package; log `logs/podstage.log`; markers `ZZFRAMES_VERIFIED`, `ZZPLAN_FULL_OK`, `ZZDBS_VERIFIED`, `ZZPODSTAGE_DONE`; failure `ZZPODSTAGE_FAIL ... rc` 4 mismatch / 5 disk / 6 ssh / 7 verify / 8 DB > 20 GB).
4. `python -B code/build_package.py` rebuilds `MANIFEST.md` + `out/paths.json` from the artifacts; then grep MANIFEST/log for the pod's IP and `ssh` tokens (must be 0) and land `out/paths.json` into repo dir `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-pod-stage/`.
5. The Master Mind launches the SH5 GPU stages (READY.md Q2 with `--plan .../plan_full/plan.json --out-dir DD/ctx_full`); this package launches no GPU job.

## Paths
scripts/logs: `C:/Users/Admin/par18/podstage/{code,logs,work}` (`work/manifest_frames.tsv`, `manifest_dbs.tsv` hold local + pod sha); sha256 lists on D: `D:/Projects/TanitAD-artifacts/pod_stage_2026-10-19/`. The local pointer file `podstage/pod_cfg.local` (names the file with the ssh prefix) is NOT staged and must not be.
