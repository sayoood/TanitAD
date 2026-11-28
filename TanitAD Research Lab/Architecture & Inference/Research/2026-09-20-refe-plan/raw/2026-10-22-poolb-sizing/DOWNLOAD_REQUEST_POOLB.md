# DOWNLOAD_REQUEST_POOLB (2026-10-22). NOTHING WAS DOWNLOADED. PI approval is asked per item.

Sizes are exact bytes from listings (HF tree at the pinned revisions below, S3 central directories cached locally) unless marked E. Evidence and reasoning: `RESULT.md`. Machine-readable lists, one row per file:
`raw/MANIFEST_mirror_poolB.csv` (URL, bytes, sha256), `raw/MANIFEST_openscene_archives_poolB.csv`, `raw/MANIFEST_db_range_poolB.csv` (zip URL, local-header offset, sizes, CRC32).
Pins: mirror `kesai-labs/nuplan@484d9d14e18fdf712dbc70962997ba2d53617bce`; OpenScene `OpenDriveLab/OpenScene@a76f840b65e972bc45e56c2adced897498e9a026`; DBs from `https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/`.
No credentials are needed: HF repos are `gated: false`, the S3 bucket is public (anonymous HEAD 200 measured).
Licence of everything below: nuPlan Terms of Use, non-commercial (RESULT.md section 5); research use, derived frames kept private.

## The decision in one table

| option | contents | bytes | what it buys |
|---|---|---|---|
| **A** | P0 + G0 | 0.545 GB + 1.7 MB | decides whether the mirror is trustworthy and whether row-level range reads work (which would cut B's frames from 12.8 GB to ~2.5 GB, E). **Approve this first** |
| **B (recommended minimum)** | A + T1-F + T1-D | **14.01 GB** (14,009,683,706 B) | the 20 logs P6 can actually read: 2,624 anchors, ~290-660 token-equivalents (E), 95 % STRAIGHT |
| B+ | B + X1 | 16.98 GB (X1 is E) | adds a first-party byte check on three pool-B logs |
| C | B + T2 | 51.84 GB | adds the 36 routed UNKNOWN-command logs: usable by **P2 (goal mode) only**, not by P6 |
| D | C + T3 | 56.87 GB | adds 4 route-less logs: **unscorable, not recommended** |
| first-party only | K20 whole OpenScene archives + T1-D | 113.24 GB (E prefix cut: 68.4 GB) | same 20 logs without the third-party mirror; 2x the bytes needed are other cameras |

All 60 logs, 4 cameras, mirror: 53,638,377,856 B. The 56 routed ones: 48,608,372,608 B. First-party whole archives: 292,933,138,217 B (53 archives) / 279,920,352,270 B (51).
**Disk gate:** D: has 8.4 GB free, C: 9.7 GB. Every item must stream per file (largest K20 file 357,429,650 B) into `<log>_<CAM>.zip` stored zips, deleting the Arrow file after its gates pass. K20 output 2.17 GB (E). C and D cannot land (frames for 10,474 anchors = 8.67 GB E) without freeing disk first.

## P0. Probes (approve with A; total about 1.7 MB)

| id | URL | source | bytes | purpose | integrity check |
|---|---|---|---|---|---|
| P0-a | the 4 gate-log files of G0 below, **HTTP Range** on the last 262,144 B and the first 65,536 B of each (8 requests) | mirror (HF -> CDN, `Accept-Ranges: bytes` measured) | ~1.3 MB | Arrow IPC file vs stream, schema, timestamp unit, record-batch size, rows per file. Fixes `read_arrow` in `code/verify_frames.py`; decides whether row-level range reads are possible | row count x 100 ms ~= log duration (77.5 s, from the pkl); timestamps strictly increasing |
| P0-b | `https://huggingface.co/datasets/kesai-labs/nuplan/resolve/484d9d14e18fdf712dbc70962997ba2d53617bce/README.md` | mirror | 3,605 | the mirror's own card: provenance, conversion, licence statement (RESULT.md section 5 quotes only card metadata) | git blob id must equal `cf953ef22422b96063c81453301718d7ae3afb9d` (listing) |
| P0-c | `https://huggingface.co/datasets/kesai-labs/nuplan/resolve/484d9d14e18fdf712dbc70962997ba2d53617bce/index.parquet` | mirror | 358,834 | per-log frames / duration / split for all 60 logs, compared with the pkl before any bulk fetch | sha256 `b3ed411295f8e327b416941caafdaa69754174a4ab2743792a71fd8ad475721f` |

## G0. Integrity gate log (approve with A): mirror versus first-party frames we already hold

Log `2021.09.16.14.39.34_veh-42_01609_01687` (navtest, 66 tokens, 264 frames x cameras already local from the OpenScene test archives; test DB local).
Base URL `https://huggingface.co/datasets/kesai-labs/nuplan/resolve/484d9d14e18fdf712dbc70962997ba2d53617bce/logs/nuplan_test/2021.09.16.14.39.34_veh-42_01609_01687/`

| file | bytes | sha256 (listing) |
|---|---|---|
| `camera.pcam_f0.arrow` | 133,161,298 | `5a2842320e87e03a30f2da94a8dfebadb52609d9428f5ab9048833fa32388f9f` |
| `camera.pcam_l0.arrow` | 137,709,282 | `089f7351f164d7ee3db65ef478fb6c5a3ea18999541e5e3bcd9bcfc45c6ce6e3` |
| `camera.pcam_r0.arrow` | 151,458,906 | `15594a6061ce91c614b815effbc9714456098357d25422fc0dd54ddb2d9e33ae` |
| `camera.pcam_b0.arrow` | 122,568,122 | `97c07e7b89cfcb586f73360b1a449fd992a229dff00a05612faaaac3d6129049` |
| total | **544,897,608** | |

Purpose: prove that the mirror's `pcam_*` files are nuPlan's CAM_F0/L0/R0/B0 bytes, un-re-encoded, and that the planner's nearest-row rule picks the same frame as OpenScene. Check: G1 sha256, then **264 / 264 sha256 equalities** (mirror JPEG vs local OpenScene-extracted frame by DB timestamp). Controls built and passing in `code/test_verify_frames.py`: zero control, L0<->R0 swap, byte flip, dropped row. **If G0 fails, nothing below is requested.**

## T1-F. K20 frames (the 20 P6-readable logs), mirror, 4 cameras

80 files, **12,788,718,912 B**. URL template `https://huggingface.co/datasets/kesai-labs/nuplan/resolve/484d9d14e18fdf712dbc70962997ba2d53617bce/logs/nuplan_{train|val}/<log>/camera.pcam_{f0,l0,r0,b0}.arrow`; every URL, size and sha256 is in `raw/MANIFEST_mirror_poolB.csv` (rows `tier=K20`); per-log bytes in `raw/K20_table.json`.
Source: third-party mirror, converted by the 123D framework (arXiv:2605.08084). 15 sessions, 7 vehicles, 18 Las Vegas + 1 Boston + 1 Pittsburgh log. Anchors: STRAIGHT 2,488, LEFT 133, RIGHT 3.
Purpose: P6 (command mode) read on every pool-B log it was trained to read; also P2 on the same logs.
Integrity: G1 sha256 == listing for all 80 files; G2 (nearest-row residual <= 50.5 ms for all 2,624 x 4 pairs, 10 Hz grid, 4 cameras present, 1920x1080 decode); G4 exact timestamps on `..veh-38_04275_04425` and `..veh-45_00440_00501`.

## T1-D. K20 nuPlan DBs

18 DBs (2 more are local), **675,704,747 B compressed** (1,078,562,816 B inflated). Range-fetch one DEFLATE member each from the S3 train zips (`nuplan-v1.1_train_{boston, vegas_1..6}.zip`), offsets and CRC32 in `raw/MANIFEST_db_range_poolB.csv` (rows `tier=K20`, `already_local=no`); tool pattern `fetch_navtrain_dbs.py` (local header 30 bytes + member). Source: nuPlan public bucket, first party.
Purpose: the P2/P6 seam reads each log's scenario, route, frame lookup and camera rig from its DB. The NAVSIM metric cache does not need them.
Integrity: G5 (CRC32 == central directory, `PRAGMA integrity_check`, all pkl tokens in `lidar_pc`, DB pairing == pkl pairing). The same CRC check reproduced the 7 local DBs exactly.

## X1. Optional first-party byte check on pool-B logs themselves (about 2.97 GB, E)

Prefix-stream, stop after the target log's `CAM_B0` members (cut about 1.25 x the estimate), from `https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/a76f840b65e972bc45e56c2adced897498e9a026/openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{N}.tgz`:

| archive N | whole bytes | HF sha256 (whole file; a prefix cannot be hashed against it) | target log (K20) | position | prefix E |
|---|---|---|---|---|---|
| 167 | 4,046,465,984 | `764d9a85f140b426981081893dfbc0df14f814ed36345b5e360855a47c364084` | `2021.06.14.18.42.45_veh-12_04534_04609` | 1 of 6 | 0.65 GB |
| 19 | 4,397,351,699 | `361c29e61232737f3124457cb278f95ccab131d2c43d96dd334f081c00a621db` | `2021.06.14.16.32.09_veh-35_04370_04488` | 1 of 7 | 0.92 GB |
| 55 | 8,486,966,802 | `ebd58ae2d28cecfd38c700a99cc2c061733c36a91752b0fb1e384ec292cd4520` | `2021.07.16.21.17.55_veh-26_05156_05225` | 3 of 7 | 1.39 GB |

Purpose: G3, mirror frame == first-party frame for ~2,100 (frame, camera) pairs of three pool-B logs and three vehicles. Integrity: members are `<log>/<CAM>/<hash>.jpg`; the hash names must equal the pkl `data_path` list exactly; all 8 cameras x n_frames members present before the cut; then sha256 equality with the mirror's nearest-row frame.

## T2. The 36 routed UNKNOWN-command logs (P2 goal mode only; ask only if a P2 read on pool B is wanted)

Mirror: 144 files, **35,819,653,696 B** (`tier=R36`). DBs: 35, **2,011,216,836 B** (`tier=R36`, `already_local=no`). Anchors 7,634, all UNKNOWN. P6 must **not** be read on these (untrained command embedding). Same gates as T1.

## T3. The 4 route-less logs: not requested

5,030,005,248 B (`tier=N4`). No `roadblock_ids`, constant UNKNOWN command: cannot be PDMS-scored by NAVSIM.

## Alternative first-party route (no third-party mirror), K20

20 whole OpenScene archives, **112,564,990,613 B** (ids 19, 26, 34, 44, 54, 55, 59, 77, 82, 83, 92, 94, 102, 110, 117, 151, 163, 167, 188, 196; E prefix cut 67.76 GB), URLs and sha256 in `raw/MANIFEST_openscene_archives_poolB.csv`. Integrity: archive sha256 == listing oid (whole archives only), name set == pkl `data_path`. R56 would be 51 archives / 279,920,352,270 B; A60 53 archives / 292,933,138,217 B. Member-level range fetch is **not** possible (gzip tar). nuPlan's own camera archives are tars without index and 56-157 GB each: not an option.

## Not requested but needed (CPU work, no download)
Mirror-to-zip extractor (names = pkl `data_path`), pool-B W3 export + token file + NAVSIM metric cache + harness re-point, then gate G6 (reproduce the banked navtest PDMS of the 66 gate-log tokens through the new path).
