# DOWNLOAD_REQUEST (UNSEEN-POOL-2, 2026-10-22). NOTHING WAS DOWNLOADED; everything below is OPTIONAL

The verdict in `../RESULT.md` was reached from LOCAL files plus metadata listings only (S3 XML listing, HF tree/API listings; every request is in `requests_log.jsonl`, 0 object bodies read).
**No item below is needed to support the verdict.** They exist so the PI can (A) tighten two UNVERIFIED name-level claims for under half a megabyte, or (B) buy a pool the verdict recommends AGAINST buying.
Sizes are from the listings (exact bytes), not from a download. `<rev>` pins are the commit shas the listings returned on 2026-10-22.

## A. Verification only (total 0.43 MB; answers "are the nuPlan public sensor sets EXACTLY OpenScene's three releases, by name?")
Current evidence: byte-exact size prediction (T4 in `test_run.out`) plus a third-party mirror's per-split log directories (T5): both say yes. These files would make it a first-party name-for-name comparison.

| # | URL | bytes | source | question answered |
|---|---|---|---|---|
| A1 | `https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/train_set/public_set_train_sensor.txt` | 42,950 | nuPlan public bucket | does the raw nuPlan TRAIN sensor set list exactly the 1,085 logs of OpenScene trainval minus val? |
| A2 | `https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/test_set/public_set_test_sensor.txt` | 5,903 | same | is the raw TEST set exactly the 147 OpenScene test logs? |
| A3 | `https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan_mini_sensor.txt` | 2,622 | same | is mini exactly the 64 mini logs? |
| A4 | `https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/a76f840b65e972bc45e56c2adced897498e9a026/openscene-v1.1/openscene_sensor_test_0-31.json` | 7,906 | OpenScene HF | which of the 32 test camera archives holds the 11 test logs outside navtest (the archives are already local, so this saves a 120 GB scan) |
| A5 | `https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/a76f840b65e972bc45e56c2adced897498e9a026/openscene-v1.1/openscene_sensor_mini_0-31.json` | 4,088 | same | mini log-to-archive map |
| A6 | `https://huggingface.co/datasets/kesai-labs/nuplan/resolve/484d9d14e18fdf712dbc70962997ba2d53617bce/index.parquet` | 358,834 | third-party mirror | its own per-log index (frames, duration, split) |
| A7 | `https://huggingface.co/datasets/kesai-labs/nuplan/resolve/484d9d14e18fdf712dbc70962997ba2d53617bce/README.md` | 3,605 | same | provenance, format and licence statement of the mirror (tag says `license:other`) |

## B. Optional: the private test sets (unscorable, listed only because the brief asked for "any private split with public frames")
`https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/a76f840b65e972bc45e56c2adced897498e9a026/openscene-v1.1/` + `openscene_metadata_private_test_e2e.tgz` (4,157,677 B), `openscene_metadata_private_test_wm.tgz` (7,316,160 B), `openscene_metadata_private_test_hard.tar.gz` (179,080 B). Total 11.65 MB.
Answers only: which logs and how many frames they hold (their frames are 23.59 GB / 14.99 GB / 0.64 GB). Their ground truth is hidden, so no answer can make them a fit set. Not recommended.

## C. Frames for a route-less pool (ONLY if the PI authorises route synthesis; the verdict recommends NOT doing so)
The 58 logs outside navtrain's sessions are listed with sizes in `mirror_log_sizes_u58.json` and `pool_lists.json`. 56 of them have no route, so they cannot be PDMS-scored as they stand.

| route | what | size | note |
|---|---|---|---|
| C1 (cheapest) | `kesai-labs/nuplan@484d9d14...`, per log `logs/nuplan_{train,val}/<log>/camera.pcam_f0.arrow` for the 56 route-less + 2 routed logs | **39.7 GB** (CAM_F0 only; 38.9 GB for the 56) | third-party mirror, **integrity UNVERIFIED**, 10 Hz native rate (5x what OpenScene's 2 Hz needs), Arrow format, needs timestamp alignment to the OpenScene tokens |
| C2 | same, 4 cameras (F0, L0, R0, B0) | 161.9 GB | |
| C3 | OpenScene `openscene_sensor_trainval_camera_{N}.tgz`, 49 archives (ids in `openscene_trainval_extras_archives.json`) | 329.1 GB whole archives; ~165 GB if a gzip prefix scan is enough (ESTIMATED, uniform-position assumption) | first-party; gzip stream, no random access; all 8 cameras ride along |
| C4 | `WeiXiCZ/navsim-trainval-full-front@a08a0aa3...` (14 parts of 10 GiB) | 148.3 GB | third-party; whole trainval front camera, 1,310 logs; plus `selection/files.list0` 64.3 MB and `selection/files.tsv` 72.5 MB |

Frames for the 2 routed unseen-session logs (the only PDMS-scorable pair): `2021.08.24.13.12.55_veh-45_00386_00472` is ALREADY LOCAL (CAM_F0, 191 MB, `D:/Projects/TanitAD/data/nuplan-camera/mini/`); `..._00507_00867` needs 0.61 GB (mirror CAM_F0) or 4 cameras 2.6 GB. A ~137-token pool does not justify either.

Metadata (OpenScene pkl, ground truth, routes) and nuPlan maps for every pool above are ALREADY LOCAL. No metadata download is needed for any option.
