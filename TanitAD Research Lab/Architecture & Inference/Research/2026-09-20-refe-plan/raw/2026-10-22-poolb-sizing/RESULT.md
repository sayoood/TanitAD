# POOLB-SIZING: what it costs to read P2 / P6 on pool B, and what pool B actually is (2026-10-22, CPU, 0 GPU, 0 downloads)

Evidence classes: **M** MEASURED (ours, artifact named) · **P** PUBLISHED · **I** INHERITED (another agent's file, not re-verified) · **E** ESTIMATED · **U** UNVERIFIED.
Paths under `raw/` and `code/` are relative to this package. Code was read at mirror tip `323a13d` (copies in `raw/*.323a13d`). Pool B = the 60 OpenScene-trainval logs that are not in navtrain but whose sessions are (`raw/poolB_logs.json`, from the prior probe's `pool_lists.json`; its 10,474 route anchors and 12,501 frames are re-derived here from the pkl, M).

## HEADLINE (needs an MM / PI decision, items 1 and 2)

1. **The "56 routed logs, ~1,656 tokens" figure does not describe what P6 can read.** Of the 56 routed logs only **20 logs (15 sessions, 7 vehicles, 2,624 anchors) carry a NAVSIM driving command of LEFT / STRAIGHT / RIGHT**. The other **36 logs carry UNKNOWN on every anchor** (7,634 anchors; 216 more sit inside one mixed K20 log). P6 is a command-mode model trained on L / S / R only (my re-count of the navtrain token table = 25,904 / 65,338 / 11,795 / UNKNOWN 0, equal to the training table quoted in the P6 dry run; navtrain tokens contain UNKNOWN **0 of 103,037** times, while 5.7 % of navtrain frames are UNKNOWN). Reading P6 on the 36 logs would feed it an untrained command embedding. The readable set is also **95 % STRAIGHT** (2,488 STRAIGHT, 133 LEFT, 3 RIGHT). **NAVSIM-token-equivalent count: ~290-660 (E), point 440 (E)**, not 1,656. NAVSIM's own token list holds **0** tokens in pool B (the logs are not in `navtrain.yaml` `log_names`).
2. **Cheapest sufficient request (tier K20, the 20 readable logs): 12.79 GB frames + 0.68 GB DBs + 0.55 GB integrity gate = 14.0 GB** (all from listings, exact), plus an optional 2.97 GB (E) first-party cross-check. All 60 logs / the 56 routed logs, 4 cameras: **53.64 GB / 48.61 GB** from the third-party mirror, or 292.9 GB / 279.9 GB of first-party OpenScene archives (whole; 188.7 / 171.9 GB if a prefix cut works, E). Section 2.
3. **The NAVSIM metric cache needs no nuPlan DB; the P2/P6 planner leg does.** The pkl + maps are enough for the cache (devkit code read, M). The seam builds its scenario, route, frame lookup and per-log camera rig from each log's nuPlan DB. **7 of 60 pool-B DBs are local; 53 are not** (all 53 inside the nuPlan train city zips). They are range-fetchable exactly: **2.687 GB** (K20 only: 18 DBs, 0.676 GB), central directories already cached locally. Section 4.
4. **Disk is the binding constraint today: D: has 8.4 GB free, C: 9.7 GB** (`df`, 2026-10-22). Nothing above can be landed as raw files; it must stream per file (largest K20 file 0.36 GB) into per-log camera zips. Frames for all 10,474 routed anchors would need 8.67 GB (E) and do not fit either.
5. **Integration items for the Master Mind:** (a) correct the pool-B line of `UNSEEN-POOL-2` (`RESULT.md` section 2 row B and section 5(d): tokens ~1,656 E -> ~290-660 E, readable logs 56 -> 20); (b) instrument work that does not exist yet and is not a download: the mirror-to-zip extractor, the pool-B W3 export + metric cache + harness re-point (`export_navtest_inputs.py` hard-codes `navsim_logs/test`), `verify_frames.py` is delivered and tested (section 3).

## 1. Which cameras, at what rate, do P2 and P6 consume? (Q1; code read, M unless marked)

| item | answer | evidence |
|---|---|---|
| cameras | **4: CAM_F0, CAM_L0, CAM_R0, CAM_B0**, in this order | `model.py:87-88` (`n_cameras=4`, `cameras=(...)`); `planner.py:54` `CAMERAS`; `train.py:43`; P6 checkpoint `meta.cameras` and 4 frames resolved on 48/48 tokens (P6 dry run Q2, **I**). P2: same `REFeConfig` and the same seam; `REFe.forward` raises on a camera-count mismatch (`model.py:733`) and P2's navtest read scored 12,146 / 12,146 with 0 seam misses (`RESULT_P2.md`, **I**), so P2 consumed 4 cameras. P2's checkpoint `meta` itself was not opened (**U**, low risk) |
| frames per scored sample | **ONE per camera, at the token's own time. No video, no history frames.** `ego_dim 7` is a single-time vector; images are never stacked over time | `planner.py:585-622` (`_image_for`: one path per camera for `ego.time_point.time_us`), `train.py:810-832` |
| pairing | per camera, the DB `image` row nearest to the token's lidar timestamp, refused beyond 120 ms; the worst camera decides | `planner.py:119-134`. Baseline residual on 8,192 pool-B pairs: mean 14.9 ms, **max 49.9 ms** (`raw/residual_baseline.json`) |
| source cadence | nuPlan camera rows **10 Hz** (100.0 ms spacing, 40,932 diffs, 7 DB-local logs); OpenScene archives keep **one jpg per 2 Hz pkl frame per camera** (1,012 per camera in a local test archive) | `raw/residual_baseline.json`, `raw/members_test_camera_30.json` |
| decode | native **1920x1080 JPEG** -> `cv2.resize((960, 512))` -> RGB -> /255; per-log rig (K, distortion, extrinsics) from the log's DB `camera` table, `undistort True` | `planner.py:615`, `calib_table.py:43-66`. A re-encoded or pre-resized source is **not equivalent**: fetch native JPEG bytes |
| what that means for bytes | unique frames = the scored anchors' **current frame** x 4 cameras. **827,390 B per frame (4 cameras, mean over 12,146 navtest frames, exact)**: F0 201,123 / L0 211,504 / R0 214,797 / B0 199,966 | `raw/jpg_sizes_local.json` |
| navigation input (decides which logs are usable) | **P6: `nav_mode command`** (LEFT/STRAIGHT/RIGHT/UNKNOWN one-hot, UNKNOWN never trained). **P2: goal mode** (route polyline -> goal points, `pdm_route` fix), so it needs a route, not a command | `p6_pod.sh:421`, `nav_command.py`, `PREREG_P6.md` (C2 row) |

## 2. Exact bytes to fetch (Q2). EXACT = sum of listing sizes (HF tree at pinned revisions, S3 central directory); E = estimate.

Subsets: **A60** all pool-B logs; **R56** the routed ones; **K20** the 20 routed logs with a known command (P6-readable); **R36** = R56 minus K20; **N4** = the 4 route-less logs (unscorable; never requested). Per-log table of K20: `raw/K20_table.json`.

| subset | logs / sessions | 2 Hz frames | route anchors | known-command anchors | **R-M mirror, 4 cameras, whole files (EXACT)** | R-M F0 only | **R-O OpenScene archives, whole (EXACT)** | R-O prefix cut (E) | frames-only floor, anchors x 4 cam (E) |
|---|---|---|---|---|---|---|---|---|---|
| A60 | 60 / 30 | 12,501 | 10,474 | 2,624 | **53,638,377,856 B** (53.64 GB) | 13.47 GB | 53 archives, **292,933,138,217 B** | 188.74 GB | 8.67 GB |
| R56 | 56 / 27 | 11,202 | 10,474 | 2,624 | **48,608,372,608 B** (48.61 GB) | 12.22 GB | 51 archives, **279,920,352,270 B** | 171.92 GB | 8.67 GB |
| K20 | 20 / 15 | 3,100 | 2,840 | 2,624 | **12,788,718,912 B** (12.79 GB) | 3.14 GB | 20 archives, **112,564,990,613 B** | 67.76 GB | 2.17 GB |
| R36 | 36 / 17 | 8,102 | 7,634 | 0 | 35,819,653,696 B | | | | |
| N4 | 4 / 3 | 1,299 | 0 | 0 | 5,030,005,248 B | | | | |

(R36 and N4 rows: differences of the rows above; `raw/cost_matrix.json`.) Anchors are NAVSIM geometry (4 history + current + 10 future frames, 2 Hz, non-empty `roadblock_ids`), re-derived from the pkl (M, `raw/poolB_pkl_summary.json`).

### Routes found

| route | what it gives | verdict for pool B |
|---|---|---|
| **R-M** third-party mirror `kesai-labs/nuplan@484d9d14` (HF, 123D Arrow). Per log and camera `logs/nuplan_{train,val}/<log>/camera.pcam_{f0,l0,r0,b0}.arrow`; all 60 logs present with all 8 cameras (assert in `code/mirror_poolb_sizes.py`) | **10 Hz, whole log**, native JPEG bytes probably verbatim (E: 208-220 KB per implied 10 Hz row vs 200-215 KB per OpenScene jpg). HEAD: `Accept-Ranges: bytes`, `x-linked-etag` = the listing's sha256 (M, `raw/head_probes.json`). Exact bytes above; `raw/MANIFEST_mirror_poolB.csv` has every URL, size, sha256 | **Cheapest. 5x redundant** (10 Hz, only 2 Hz token frames used). Row-level range reads could cut K20 toward 2.2-2.6 GB (E) **if** the Arrow file is the random-access IPC file format with small record batches (**U**; probe P0 settles it for ~1 MB) |
| **R-O** OpenScene first-party `OpenDriveLab/OpenScene@a76f840b`, `openscene_sensor_trainval_camera_{N}.tgz` (200 archives, 4.6-9 GB, 8 cameras, 2 Hz) | exact URLs and HF sha256 in `raw/MANIFEST_openscene_archives_poolB.csv`. Members are `<log>/<CAM>/<hash>.jpg`, the **same hash names the planner looks up** and **byte-identical** to nuPlan's own jpgs (1,761 / 1,761 common jpgs equal in size and CRC32 across OpenScene test archives and nuPlan first-party mini zips, M, `raw/openscene_vs_firstparty_crc.json`) | **Whole archives only.** Gzip tar, no index: no member-level range fetch (`Accept-Ranges` is present, M, so a **prefix** cut is possible; TanitAstra's own inventory states the interior is not random access, **I**). In-archive order is log by log, cameras F0, L0, L1, L2, R0, R1, R2, B0 (M, local archive); first member = first log of the map in **200 / 200** archives (M). Only 48.5 % of an archive's bytes are the 4 cameras we need (M, `raw/members_test_camera_30.json`) |
| R-N nuPlan raw S3 `sensor_blobs/{train_set,val_set}/nuplan-v1.1_*_camera_N.zip` | first-party, anonymous (HEAD 200, `Accept-Ranges: bytes`). 43 train archives = 5.389 TB (avg 116.7 GiB), 12 val archives = 867 GB | **Not usable.** Only mini is a real ZIP; train/val are tars named `.zip` with no central directory (RETRACTION_LOG R14, project-measured 2026-09-22), so each K20 log would cost a whole 56-157 GB archive; the log-to-archive map is unknown without reading them (**U**) |
| R-W `WeiXiCZ/navsim-trainval-full-front` (148.3 GB, 14 parts, zstd tar) and R-S `richardyann/navsim-select` `sensor_blobs/trainval.zip` (148.2 GB) | CAM_F0 only (its own card: front camera only; the second by size arithmetic, **U**) | **Cannot supply 4 cameras.** Usable only as an independent F0 cross-check |
| `OpenDriveLab/OpenScene` `navsim/navtrain_{current,history}_*.tgz` | NAVSIM's navtrain sensor tars | pool B logs are outside navtrain `log_names`, so they are not in them (E; not opened) |
| other HF repos | 50 datasets from 16 searches and 25 follow-up listings (`raw/hf_search_routes.json`, `raw/hf_other_routes*.json`): navtrain caches, maps, QA, occupancy, mini samples | none holds non-navtrain trainval frames. ModelScope / Baidu mirrors not probed (**U**) |

### nuPlan DBs (needed by the P2/P6 seam, not by the metric cache)
`raw/db_sizes_poolB.json`, `raw/MANIFEST_db_range_poolB.csv`: for each log the S3 zip URL, local-header offset, compressed / uncompressed bytes, CRC32, from the **local central-directory cache** (`_cdcache`, zero bytes fetched). Cross-check: for the 7 local DBs the central-directory size and CRC32 equal the local file (M), which also validates the cache-key mapping.

| | DBs | to fetch | compressed (EXACT) | uncompressed |
|---|---|---|---|---|
| all 60 | 60 (7 local, all nuPlan *val* logs) | 53 | **2,686,921,583 B** | 4,276,269,056 B |
| K20 | 20 | 18 (2 local) | **675,704,747 B** | 1,078,562,816 B |
| R36 | 36 | 35 | 2,011,216,836 B | |

The 53 sit in 7 train zips (boston 2, vegas_1 2, vegas_2 16, vegas_3 26, vegas_4 1, vegas_5 3, vegas_6 3). A search of C: and D: (3.6 M files; E: is the same volume as D:) found **only the 7**; no other copy and no archive named after a pool-B log (`raw/find_poolb_dbs.json`). Method as `fetch_navtrain_dbs.py` (range-fetch one DEFLATE member, inflate, verify size and CRC32).

## 3. Integrity (Q3): how we know the frames are the right ones

What was **already executed** (all local, no download): (a) OpenScene jpgs == nuPlan first-party jpgs, **1,761 / 1,761** (size + CRC32 from zip directories; 10 logs F0, 1 log B0/L0/R0); (b) pkl pairing == DB rule on **48,584 / 48,584** navtest pairs and **8,192 / 8,192** pool-B pairs (7 DB-local logs), pkl K + distortion == DB camera table on 28 / 28 camera rigs; (c) baseline residual distribution (above); (d) `verify_frames.py` run against a synthetic Arrow fixture built from first-party frames, with a zero control and four deliberate-regression arms (`raw/test_verify_frames.out`: **ALL_AS_EXPECTED true**). One honest finding from (d): a clock offset below 50 ms is **invisible to the residual gate** (a 10 Hz grid absorbs it by picking a neighbouring row; M2 read 48.1 ms), which is why G3/G4 exist.

| gate | what it proves | how | cost |
|---|---|---|---|
| **G1** transfer | bytes = the repository's bytes | `sha256(file) == lfs oid` from the tree listing (the HEAD `x-linked-etag` equals it, M). OpenScene: archive sha256 == listing oid (the prior probe did this for the metadata tgz). DB: size + CRC32 == central directory | CPU only; ~12.8 GB hashed for K20 |
| **G0** mirror == first-party, on a log whose frames we hold | camera naming (F0/L0/R0/B0 map), no re-encoding, nearest-row pairing picks the same frame | fetch the 4 mirror files of navtest log `2021.09.16.14.39.34_veh-42_01609_01687` (**544,897,608 B**), pair its 66 navtest tokens by the DB rule, compare sha256 of the mirror JPEG against the local OpenScene-extracted frame: **264 comparisons, PASS = 264 / 264**. Control: reference vs itself; mutation: L0<->R0 swap must read 0 / 66 on both | 0.545 GB (E: ~17 MB if P0 shows row-level range reads work) |
| **G2** per log, DB-free | frame set is complete and on the 10 Hz grid | every (anchor, camera): nearest mirror row <= **50.5 ms** (baseline max 49.92 ms, M); row spacing 100 ms +-1; 4 cameras present; every K20 JPEG decodes at 1920x1080 | CPU, minutes |
| **G3** pool-B-specific first-party anchor (optional) | the mirror's frames for pool-B logs themselves equal first-party | prefix-stream OpenScene archives **167, 19, 55** (K20 logs `..veh-12_04534_04609`, `..veh-35_04370_04488`, `..veh-26_05156_05225`, positions 1, 1, 3; three vehicles), name each member by pkl `data_path`, compare sha256 to the mirror's nearest-row frame for all their 2 Hz frames x 4 cameras (~2,100 comparisons) | **2.97 GB (E)** |
| **G4** DB-local exactness | the picked row is the DB's row | for the 2 K20 logs with local DBs (`..veh-38_04275_04425`, `..veh-45_00440_00501`) the picked row timestamp must equal the DB `image.timestamp` exactly, all pairs | free |
| **G5** DB fetch | DBs right and complete | CRC32 == central directory; `PRAGMA integrity_check`; every pkl token of the log is in `lidar_pc`; DB-derived pairing == pkl pairing for all anchors | CPU |
| **G6** end to end | planner leg is wired | the PDMS read of P2 / P6 on the gate log's 66 navtest tokens through the new pool-B code path must reproduce the already banked navtest PDMS for those tokens (a known value; the path differs only in log source) | 66 tokens x ~1.4 s = ~2 min of RTX 4060 (P6 dry run rate, **I**; E) |

What none of these can show: that the mirror is right on every one of its 4 x 20 files beyond G1-G2 (G0 and G3 sample 1 + 3 logs). I judge that sufficient because all files come from one converter; it is an inference, not a measurement.

## 4. Everything else a PDMS read on pool B needs (Q4)

| need | status |
|---|---|
| **metric cache** | NAVSIM v1.1 `caching.py` builds from `SceneLoader` (pkl) + `NavSimScenario` (`map_root`) with `SensorConfig.build_no_sensors()`: **no nuPlan DB, no images** (code read at `navsim-3e8291b`, M; build not run on pool B, **U**). Pkl: local (60 read from the sha-verified trainval tgz, M). Maps: all four locations local (`nuplan-maps-v1.0/{us-nv-las-vegas-strip, us-ma-boston, us-pa-pittsburgh-hazelwood, sg-one-north}`). The brief's premise "the cache needs agents from the nuPlan DB" holds for TanitAD's training-time D9 parts (`build_metric_cache.py`), not for the evaluation cache |
| **routes** | local, pkl `roadblock_ids`: 10,474 route anchors in 56 logs (M, equals the prior probe) |
| **commands** | local, pkl `driving_command`: K20 anchors STRAIGHT 2,488 / LEFT 133 / RIGHT 3 (M) |
| **P2/P6 planner leg** | needs the DB (scenario, route polyline, frame lookup, rig). 7 of 60 local; 53 to fetch (section 2). Frame names alone do not need it: pkl `data_path` gives the exact JPEG (item b above) |
| **harness instruments** | `export_navtest_inputs.py` reads `navsim_logs/test` only; `stage2_sft.py` hard-codes the navtest export, tokens and baselines. A pool-B export, token file, metric-cache dir and harness paths are CPU work, not downloads (not written here) |

### Expected token count after NAVSIM's own filters (ESTIMATED)

| definition | n | class |
|---|---|---|
| NAVSIM-official navtrain tokens inside pool B | **0** | by construction (not in `log_names`); 1,191 of 1,192 navtrain logs hold >= 1 token (M) |
| anchors passing `SceneFilter` (4+10 window, `has_route`) | 10,474 (56 logs) | M |
| ... with command in {L,S,R} (readable by P6) | **2,624 (20 logs)** | M |
| ... and ego speed >= 0.5 m/s | 1,100 (13 logs) | **I** (TanitAstra `compact_metadata`; its 2,624 equals the pkl count) |
| token-equivalent, command-known filter, navtrain ratio 0.1668 | **438** (bootstrap CI 422-456; per-session p10-p90 354-660) | E, `raw/token_estimate.json` |
| token-equivalent, command-known and moving, ratio 0.2612 | **287** (233-370) | E |
| prior probe: all route anchors x 0.1581 | 1,656 (1,245-2,628) | E, **superseded**: includes 7,850 UNKNOWN anchors that no navtrain token resembles |

Held-out check of the ratio method: leave-one-session-out over 162 navtrain sessions, predicted / actual tokens p10-p90 = 0.66-1.24, total 103,097 vs 103,037 (M). It does not cover pool B's selection effect: these logs were left out of navtrain, so they are probably **less** interesting than the logs the ratio was fitted on (K20: 42 % moving anchors vs 64 % in navtrain, M/I), which favours the lower half of the range. NAVSIM's tokens are contiguous runs, not a thinning: 83.5 % of consecutive token pairs are one frame apart (M, `raw/token_signature.json`). Why the filter drops these logs is **U** (OpenScene's goal-to-command rule is unread, `nav_command.py` header).

Statistical weight (E, assumes pool-B per-log variance equals navtest's): the R2 interval on 136 logs was +-0.58 PDMS; 20 logs scale it by sqrt(136/20) = 2.6 to **about +-1.5 PDMS**, on a pool that is 95 % straight with 3 right turns. It cannot decide a lever smaller than that.

## 5. Licence and provenance (Q5; fields are the cards' own metadata, prose is paraphrased)

| source | card / terms | provenance |
|---|---|---|
| nuPlan raw S3 (DB zips, camera zips) | the nuScenes / nuPlan / nuImages / nuReality **Terms of Use**, last updated 2021-11-16 (local copy `D:/Projects/TanitAD/data/nuplan/LICENSE`, 25,319 B, sha256 `1a218286e733...`): non-commercial use; data under CC BY-NC-SA 4.0 plus additional terms that prevail; terms also cover data derived from the datasets | first party (Motional's public bucket; anonymous HEAD 200) |
| OpenScene HF `OpenDriveLab/OpenScene` | `license: cc-by-nc-sa-4.0` (card metadata); repo README is 32 bytes | first party for OpenScene; the underlying images are nuPlan's, so the nuPlan terms apply as well |
| mirror HF `kesai-labs/nuplan` | `license: other`, `license_name: nuplan`, `license_link: https://www.nuscenes.org/terms-of-use`; card summary: nuPlan v1.1 in the 123D Arrow format. Its `LICENSE` file (25,319 B) has git blob id `30525e69840147c98e6451b6d8069848766ce08b`, **equal to `git hash-object` of our local nuPlan Terms of Use** (M): the same text | converted by the 123D framework, arXiv:2605.08084 (Dauner, Chitta, Geiger et al., the NAVSIM authors; code URL `github.com/kesai-labs/py123d` in the abstract, P). HF org `kesai-labs` is not marked verified (M, 4 members). Created 2026-09-01; unmodified-bytes claim is **U** until G0 |
| WeiXiCZ / richardyann | no licence field in the card metadata | unknown uploaders; F0 only |

For this programme: non-commercial research use is what TanitAD does; derived frames published privately fit the ShareAlike and derived-data clauses. The mirror's README (3,605 B) was **not** read (it is a file body); request item P0-b.

## 6. Time and disk (E)
Dev-box HF throughput 6-12 MB/s (P6 dry run, **I**): K20 mirror 12.8 GB = 18-36 min; R36 35.8 GB = 50-100 min; first-party K20 whole archives 112.6 GB = 2.6-5.2 h. Disk: stream per file (<= 0.36 GB transient) into `<log>_<CAM>.zip` stored zips named by the pkl hash (the layout `planner.FrameResolver` reads; loose files cost 1 MiB each on D: exFAT); K20 output 2.17 GB.

## 7. Limits and UNVERIFIED
- **U:** Arrow schema, IPC-file vs stream, record-batch size, timestamp unit (P0); mirror byte-identity to nuPlan beyond the sampled logs (G0, G3); in-archive log order beyond the first member of each archive (assumed from the map order); the cause of the UNKNOWN commands; the pool-B metric-cache build and harness run; mirror README; P2 checkpoint meta; other mirror sites.
- **E:** every token count, every prefix cut, the frames-only floor, the +-1.5 PDMS width, all times.
- **I:** compact_metadata speeds (1,100 moving anchors), P6 dry-run throughput and training command counts, RESULT_P2 facts.
- The prior probe's pool membership and the 10,474 / 12,501 totals were re-derived from the pkl and agree; its mirror and archive listings reproduce byte for byte (the OpenScene tree at the pinned revision is identical to the earlier listing, M).
- Nothing was downloaded: 133 requests, all metadata (94 HF tree listings, 16 searches, 5 S3 XML listings, 6 API / arXiv metadata, 12 HEADs), 2.69 MB of listing payload, 0 object bodies (`raw/requests_log.jsonl`). Local reads only; pickles through a numpy-only whitelist (DB camera blobs through an inert-stub shim, `code/dbpickle.py`).

## 8. Reproduce (CPU; local inputs plus the listings in `raw/`)
```
python -I code/mirror_poolb_sizes.py ; python -I code/db_sizes_from_cdcache.py ; python -I code/openscene_archives_poolb.py
/d/venvs/opencv-ref-454/Scripts/python.exe -I code/extract_poolb_frames.py    # ~5 min, streams the 7 GB metadata tgz
/d/venvs/opencv-ref-454/Scripts/python.exe -I code/poolb_from_pkl.py ; ... code/token_signature.py ; ... code/token_estimate.py ; ... code/residual_baseline.py ; ... code/pairing_pkl_vs_db.py
python -I code/openscene_vs_firstparty_crc.py ; python -I code/jpg_sizes_local.py ; python -I code/cost_matrix.py ; python -I code/make_manifests.py
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I code/test_verify_frames.py   # ALL_AS_EXPECTED true
```

## Deliverable manifest (nothing staged, committed or pushed, per the local override; every artifact exists in ONE place only)
| artifact | where | other copy |
|---|---|---|
| `RESULT.md`, `DOWNLOAD_REQUEST_POOLB.md` | `C:/Users/Admin/par18/poolb/` | **none** |
| `code/` (23 scripts: listings, cost matrix, manifests, pkl and DB checks, token estimate, `verify_frames.py`, `test_verify_frames.py`) | `.../poolb/code/` | none |
| `raw/MANIFEST_mirror_poolB.csv`, `MANIFEST_openscene_archives_poolB.csv`, `MANIFEST_db_range_poolB.csv` (every URL, size, hash) | `.../poolb/raw/` | none |
| `raw/` listings and evidence: `mirror_poolB_sizes.json`, `hf_tree_openscene_trainval_camera_pinned.json`, `s3_list_*_poolb.json`, `db_sizes_poolB.json`, `head_probes.json`, `cards_api.json`, `hf_search_routes.json`, `hf_other_routes*.json`, `requests_log.jsonl`, `find_poolb_dbs.json` | same | none |
| `raw/` derived: `poolB_frames.json.gz` (60 pkl, 4-camera jpg names, commands), `poolB_pkl_summary.json`, `cost_matrix.json`, `token_estimate.json`, `token_signature.json`, `pairing_pkl_vs_db.json`, `residual_baseline.json`, `jpg_sizes_local.json`, `openscene_vs_firstparty_crc.json`, `test_verify_frames.out`, `K20_table.json`, `members_test_camera_30.json` | same | none (inputs are TanitAstra / dev-box files on D:, read-only) |
| `raw/*.323a13d` (read-only copies of the code read at mirror tip `323a13d`) | same | the repo tip |
| `raw/SHA256SUMS.txt` | same | none |
Suggested repo path: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-22-poolb-sizing/`. The Master Mind lands it.
