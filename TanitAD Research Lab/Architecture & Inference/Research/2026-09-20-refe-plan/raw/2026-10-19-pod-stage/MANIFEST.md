# MANIFEST: pod reads under P6's overlap rule (2026-10-19)

Two pod reads, PI-approved 2026-10-19 under the P6 OVERLAP RULE: every pod-side command at `nice -n 19 ionice -c3`, ONE stream, batches of <= 120 MB, a `P6_GUARD` line in P6's chain log at the start of each pod command pauses (it is checked in the SAME remote shell as the read), no GPU process, no pod-side python, no `git` on the pod, never `pkill -f`/`pgrep -f`, P6's processes and files untouched. Pod-side work per batch: `tail` (guard), `tar -c`, `sha256sum` (the PI's approval for this task). The pod's address, port, id and token files appear in NO staged file (the ssh alias lives in a local file outside the repo; every logged string is redacted).

## READ 1: SH5 FULL frames (priority 1)

- **Frames staged: 5776/7864 jpg, 1.204 GB (1.121 GiB); every staged file verified: YES; COMPLETE: NO** (MEASURED, `raw/verify_frames.json`).
- ⛔ **STOPPED BY THE DISK GUARD, NOT BY AN ERROR.** D: (exFAT, 1 MiB clusters: MEASURED 200 files of 200 KiB = 200 MiB) fell from 35.3 GiB free (15:34) to 20.2 GiB while this copy added only ~5.6 GiB allocated: other writers on D: took ~9 GiB meanwhile. The copier refuses a batch that would leave < 20 GiB free (the S0 guard, kept). The remaining 2088 files are 0.42 GB of data but ~2.0 GiB ALLOCATED, which would leave ~18.2 GiB free. Resume = one command after space is freed (or the guard is lowered by the Master Mind): `PODSTAGE_MIN_FREE_GB=<n> python code/podstage.py frames`.
- Source list: `needed_frames.json` of the SH5 plan = 1,966 tokens x 4 cameras; 7864 distinct names; on the pod 7864 found, 0 missing; 23 pod shards; pod-side `stat` total 1646644900 B = 1.647 GB (READY.md estimated ~1.5 GB).
- Checks on EVERY staged file: exists; size = pod `stat` = manifest; non-zero; JPEG magic `FF D8 FF`; sha256 of the bytes on disk = sha256 of the received stream = pod-side `sha256sum` (computed at nice/ionice). 5776 of 5776 end with `FF D9`. The checker was mutation-tested in scratch (flipped byte, file truncated by 1 byte, zero-size file, broken magic: all 4 caught; clean control 6/6, 0 failed).
- Where: `D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/frames_full` (layout `<log>/<CAM>/<hash>.jpg`, exactly the names the DBs give). sha256 list (lives on D:, not in the repo): `D:/Projects/TanitAD-artifacts/pod_stage_2026-10-19/frames_full_sha256.txt`.

### SH5 plan re-run for FULL (CPU, `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`; READY.md Q1 verbatim)

- **`plan_full` NOT produced: the frames are incomplete (disk guard).** The same command was re-run on the 5776 staged frames into `D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/plan_partial_5776` (CPU, `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`, ~24 s), so it is a PARTIAL plan, not the FULL variant.
- **PARTIAL coverage: tokens 2260/2782, sets 4018/4941, logs 40** (before any staging: 816/2,782 tokens, 1,444/4,941 sets, 38/41 logs). Sets kept by fold [887, 615, 1071, 111, 1334] of [1030, 661, 1368, 179, 1703]. The staged 5,776 frames are exactly 1,444 complete tokens (4 cameras each); the other 522 tokens need the 2,088 frames that did not fit.
- Output: `D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/plan_partial_5776` (`plan.json`, `coverage.json`, ...); `coverage.json` also staged as `raw/plan_partial_coverage.json`. Do NOT use the partial plan as the FULL variant (the prereg's FULL is all 2,782 tokens).

## READ 2: nuPlan DBs of the 76 f10 logs PRIVGEOM-SAFE could not read (priority 2)

- The 117 f10 logs = the `*.npz` of `C:/Users/Admin/srttc_s0/work/sets` (PRIVGEOM's `SETS_DIR`); 41 have a DB in the local folder `D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits` (PRIVGEOM's `NS.index_dbs()` = `navtrain_scenarios.DEFAULT_DB_ROOT`, on D:); the other **76** are needed. All 76 exist on the pod (`navtrain_dbs`), 0 missing.
- **MEASURED TOTAL SIZE (pod-side `stat`, nice/ionice): 11324555264 B = 11.325 GB = 10.547 GiB** (min 5.3 MiB, max 1607 MiB). <= 20 GB, so the transfer was allowed.
- ⛔ **DBs transferred: 0/76 (NOT transferred, deliberately).** The size passes the 20 GB rule (11.32 GB), but the staging drive does not have the room: D: has 20.2 GiB free now, 1 MiB clusters make the 10.55 GiB occupy ~10.60 GiB, so the transfer would leave ~9.6 GiB, under the 20 GiB floor the S0 stager keeps (and other jobs are writing to D: right now: ~9 GiB vanished in ten minutes during the frames copy). Priority 1 (frames) is also still incomplete for the same reason, so nothing was started for priority 2.
- What would unblock it: free ~10 GiB on D: (DBs only) or the Master Mind lowers the guard (`PODSTAGE_MIN_FREE_GB`) or names another drive; then `python code/podstage.py dbs` (resumable, ascending size, verifies by sha256 pod = stream = disk and by a read-only sqlite open). The pod-side list is already made (`work/copy_list_dbs.tsv`).

## Run facts

- Batches: 19 frame batches + 0 DB batches logged; P6_GUARD waits: 0. D: free after the run: 20.2 GiB (the copier refuses a batch that would leave < 20 GiB).
- Markers in the log (`logs/podstage.log`):

```
[2026-10-19T15:43:11+0200] ZZPODSTAGE_FAIL frames copy rc=5 (4 = size/sha mismatch, 5 = disk guard)
[2026-10-19T15:43:11+0200] ZZPODSTAGE_FAIL phase frames rc=5
[2026-10-19T15:48:29+0200] ZZFRAMES_PARTIAL_DISK the disk guard (20.0 GiB free) stopped the frames copy; what is in is verified below; resume after space is freed or with PODSTAGE_MIN_FREE_GB lowered by the PM/PI
[2026-10-19T15:48:32+0200] ZZFRAMES_VERIFIED 5776/7864 files, 1.204 GB, failed 0
```

## Deliverable manifest

| artifact | where |
|---|---|
| staged frames (5776 jpg) | `D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/frames_full` (D: ONLY: re-fetchable from the pod, which is training P6 and is not storage) |
| sha256 lists (frames / DBs) | `D:/Projects/TanitAD-artifacts/pod_stage_2026-10-19` (D: only; counts above) |
| SH5 FULL plan | `D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/plan_full` (D: only; coverage.json also in the repo) |
| 76 nuPlan DBs | `D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits` (D: only; re-fetchable from the pod) |
| scripts: `podcopy2.py`, `podstage.py`, `build_package.py`, `launch.ps1` | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-pod-stage/code/` (staged by the lander from `out/paths.json`); also `C:/Users/Admin/par18/podstage/code/` |
| `MANIFEST.md`, `HANDOFF.md`, `raw/*.json`, `raw/podstage.log` | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-pod-stage/`; also `C:/Users/Admin/par18/podstage/` |

Never staged for the repo: the frames, the DBs, the sha lists, anything with the pod's address.
