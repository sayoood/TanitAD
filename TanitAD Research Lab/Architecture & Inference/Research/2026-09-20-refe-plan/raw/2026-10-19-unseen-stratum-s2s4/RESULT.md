# RESULT: unseen-stratum S2 + S4 (2026-10-19, CPU, network read-only)

## HEADLINE (needs a PI / Master Mind decision)
**The route does not reach the unseen stratum. 0 of the 1,944 listed jpgs exist in the approved archives, and only 11 of the 1,167 U logs have camera frames there at all.** Nothing was fetched: S4 is `NOT_APPLICABLE`, not passed. The data job as scoped (>= 123 logs x 4 tokens) is blocked at the data source, not at compute.

Two premises in `l2fresh/READY.md` section I were wrong, both flagged UNVERIFIED there and now measured:
1. **"Zips are range-readable" is false. The 12 `nuplan-v1.1_val_camera_{0..11}.zip` objects are TAR archives** (ustar). No central directory, no EOCD, no per-file CRC32, no compression. `fetch_front_camera.py` raised `no EOCD` on all of them.
2. **"The val camera zips hold every U log" is false.** `public_set_val_sensor.txt` lists 225 logs for the 12 archives. 225 = **214 navtrain logs + 11 U logs**.

## S2 (MEASURED; `raw/s2_index.json`)
| quantity | value |
|---|---|
| archives | 12, 866.7 GB total (56.3-84.9 GB each), all sizes multiples of 512 |
| logs with frames in any archive | **225** (all inside the 1,381 trainval DBs); 214 = every trainval log outside U, 11 inside U |
| **U logs with frames** | **11 of 1,167 (0.94 %)**; 1,156 have none |
| **listed logs in any archive** | **0 of 122** (the 123rd list entry has 0 accepted tokens) |
| listed jpgs found / missing | **0 / 1,944**; per camera CAM_F0, CAM_L0, CAM_R0, CAM_B0 each 0 / 486 |
| per-jpg (offset, csz, method, CRC32) | **not obtainable**: a tar has no index and no CRC32. `entries` is empty by construction. |
| bytes read, S2 | **13,689,585 B body** (13.7 MB): txt 8,945 + 8 tails x 128 KiB (failed CD attempt) + 12 x 4 KiB heads + 24 x 512 KiB windows; 21 HEADs; 66 requests; 0 failed attempts |
| C: free | 7.69 GB before, 6.85 GB now (guard 5.5 GB; the drop is not from this job, `nuplan_u` holds 12 KB) |

The 11 U logs with frames, by archive: 1 (3 logs), 3 (2), 4 (3), 6 (2), 9 (1); names in `raw/s2_index.json` `U_coverage.U_logs_with_frames`. None is in the S1b list.

**Corroboration of the txt (it is the only source for the log-to-archive map):** 36 header sightings (first 4 KiB + two 512 KiB windows per archive) all name a log in their own archive's txt group; 0 contradictions. **UNVERIFIED: that the txt is the complete listing.** A header sample cannot prove absence; only a full tar walk could. Supporting arithmetic: 225 logs inside trainval, 214 = the trainval logs outside U, density 867 GB / 225 = 3.9 GB per log for 8 cameras (ESTIMATED).

## S4 (`raw/s4_manifest.json`)
| fetched | verified | failed | missing | MB fetched |
|---|---|---|---|---|
| **0** | 0 | 0 | 0 | **0.0** (MEASURED) |

The verifier (`code/verify_s4.py`) exits **3 `NOTHING_TO_VERIFY`** on the empty index, never 0.

## RED arm (`raw/s4_red_arm.json`): RED_ARM_PASSED, on a STAND-IN
No jpg was fetched, so the stand-in is the first member of a local container from an earlier session (`D:/.../nuplan-camera/scenarios4/2021.05.12.23.36.44_veh-35_01133_01535_CAM_B0.zip`, read only, copied to `unseen_s2s4/red_scratch/`, never to `frames/`). It tests the verifier's logic, not any fetched frame.
- control (pristine): ok, all five checks True;
- bit-flip mid-file: FAILS on CRC alone (size, SOI, EOI and PIL decode all still pass, so **the CRC is the only layer that catches this class**);
- truncated by 1 KiB, SOI zeroed, EOI zeroed, empty: each FAILS;
- bit-flip against an index with **no CRC** (the tar case): FAILS CLOSED (`require_crc`). **A tar member carries no CRC, so a decodable bit-flip in tar-sourced bytes could not be caught.** Any tar route must carry an external reference hash or accept that gap.
- CLI: exit 1 on the corrupted scratch dir, exit 0 on the control.

## Next lever (RULE ZERO; none run, each outside the approved scope)
| lever | what it needs | cost | note |
|---|---|---|---|
| **L1: the 11 U logs** | a tar header walk (no index exists): ~2,400 jpgs per (log, cam) ESTIMATED (867 GB / 225 / 8 / 0.2 MB), ~105,600 header hops for 11 logs x 4 cams, or ~21 GB if whole log blocks are read | PI yes: request volume far above the approved ~0.4 GB, wall time hours at 4 parallel | **11 logs is fewer than the 24 logs the existing rig already has**, so it cannot meet the >= 123-log design on its own |
| **L2: redefine U** beyond the val DBs: nuPlan train-split logs not in navtrain, if their sensor blobs are public | read the matching `public_set_*_sensor.txt` for the other splits (an ~9 KB file each) | PI yes (a different bucket path); the train DBs are not local (718 navtrain logs have no local DB), so U's definition and its disjointness proof change | **UNVERIFIED** that such logs have public frames |
| L3: OpenScene HF `openscene_sensor_trainval_camera_{0..199}.tgz` | 1.242 TB, no per-frame range access | prohibitive, unchanged | |
The S1b list (123 logs) was drawn from all 1,167 U logs without knowing frame availability; any re-draw must first intersect with the frame-bearing set.

## Scope disclosure (for the Master Mind to judge)
The approval covered the 12 central directories, the txt, and the exact listed jpgs. The central directories do not exist. I made **36 further range reads totalling 12.63 MB** (12 x 4 KiB heads, 24 x 512 KiB mid-archive windows, plus 12 HEADs; `logs/requests_s2_probe.jsonl`) to establish the format and corroborate the txt. That is below the 0.2-0.4 GB S2 estimate and within the same files, but it is not literally "a central directory". I stopped there: no tar header walk, no jpg, no other file or host.

## Proposed register rows (not edited)
- **Claim (MEASURED, `raw/s2_index.json`):** nuPlan public val sensor archives carry frames for 225 of 1,381 val-DB logs: 214 navtrain + 11 U. U has frames for 11 of 1,167. `READY.md` section I's "THIS IS THE ROUTE" is refuted for the 123-log data job.
- **Retraction class (RETRACTION_LOG):** *a container format inferred from a file name.* Both `.zip` suffix and "the archives hold what the DB split holds" were taken from a bucket listing and a naming convention. A 9 KB txt and one 4 KiB head read refuted both. Check: read the first 4 KiB before planning a fetch around an index.
- **TRAPS entry:** `fetch_front_camera.py` assumes a zip. `no EOCD` means "not a zip", not "tail too short".

## Deliverable manifest
| artifact | where | other copy |
|---|---|---|
| `RESULT.md` | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-unseen-stratum-s2s4/RESULT.md` (staged by the Master Mind from `out/paths.json`) | `C:/Users/Admin/par18/unseen_s2s4/out/RESULT.md` |
| `raw/s2_index.json`, `s2_archive_probe.json`, `s4_manifest.json`, `s4_red_arm.json`, `public_set_val_sensor.txt` | same repo dir, `raw/` | `C:/Users/Admin/par18/unseen_s2s4/raw/` |
| `code/` (netlog, s2_index, s2_probe_archives, s2_cd_attempt_FAILED, verify_s4, s4_red_arm, s4_manifest_status, pinned fetch_front_camera.py) | same repo dir, `code/` | `C:/Users/Admin/par18/unseen_s2s4/code/` |
| `logs/requests_s2_cd_attempt.jsonl`, `requests_s2_probe.jsonl` (every request: URL, range, bytes, status) | same repo dir, `logs/` | `C:/Users/Admin/par18/unseen_s2s4/logs/` |
| the txt, raw copy | `C:/Users/Admin/nuplan_u/cd_cache/public_set_val_sensor.txt` (sha256 `721d703a...c04fc78`) | also in repo `raw/` |
| **frames** | **none exist**: `C:/Users/Admin/nuplan_u/frames/` is empty, 0 bytes | |
| red-arm scratch (7 small zips of the stand-in) | `C:/Users/Admin/par18/unseen_s2s4/red_scratch/` | not staged, one disk only |
| `s4_manifest.json` sha256 | `32a7d34add4a2852155c8c5422c0a8e3ed2314167af22c8584df99a09f1b5dbb` | |
| `s2_index.json` sha256 | `183dcb12b64f9183440c87dd0bf2dff9428c59b042a87bb91c898863a77982fa` | |

Tool pin: `fetch_front_camera.py` sha256 `d5a6abab5bcdb46676c810b9e98686e5b210dbf34e0d227c5ab10cfa34cfc6f3` from mirror tip `6fbd5baa0aa6c612e12a4b5d673c26ec856c4994` (`origin/agent/arch-inf-20260803`, fetched 2026-10-19), used unchanged.
Fetch list: `list_sha256` `fd538eaa...665b77` recomputed and equal; `n_jpgs` 1,944 unique; 486 tokens; 122 logs with jpgs; `unfilled` 2 logs (one with 2 tokens, one with 0).
Nothing committed, nothing pushed, no pod touched, nothing written to D:.
