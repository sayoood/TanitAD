# UNSEEN-POOL-2: is there a public sensor-bearing log pool outside navtrain U navtest? (2026-10-22, CPU, 0 GPU, 0 downloads)

Evidence classes: **M** MEASURED (ours, artifact named) · **P** PUBLISHED · **I** INHERITED (not re-verified) · **E** ESTIMATED · **U** UNVERIFIED.
`raw/` paths are relative to this package. Definitions are those of `draw_reserve.py` (session = `log.rsplit('_',2)[0]`; vehicle-day = date + vehicle); `code/test_pool.py` asserts they equal the reference on all 1,457 names.

## HEADLINE (needs an MM / PI decision, items 1 and 2)
**A pool exists, but it is not a usable fit set. Of 1,457 public sensor-bearing logs, 129 lie outside navtrain U navtest (11 sessions, 0 vehicle-days). Only ONE of those sessions can be PDMS-scored, and it holds about 137 tokens.**
1. **Correct two register claims.** `D-DATA-U-FRAMES-1` ("public sensor set is essentially navtrain U navtest") reached the right conclusion from the wrong object: it read the nuPlan-raw VAL index (225 logs, 11 outside), not the 1,457-log universe (129 outside). `FEASIBILITY.md` §7 ("no session-unseen pool outside navtest") is false as worded (9 sessions exist outside navtrain) and true in effect (1 scorable session). Section 6.
2. **The P7 session reserve (carved from navtrain) stays the only workable fit set for P7.2.** No external pool rescues it. A route-synthesised alternative (58 logs, ~6,200 tokens E) would be NAVSIM-unscorable as it stands, still vehicle-day-seen, and costs 39.7-161.9 GB from a third-party mirror. Recommendation: do not buy it. Section 5.
3. The 118 trainval logs outside navtrain are not random leftovers. **60 of the 118 (and 69 of all 129) have no route at all** (`roadblock_ids` empty on every frame, `driving_command` constant `(0,0,0,1)`), which is exactly what NAVSIM's `has_route: true` filter removes. Navtrain has 0 such logs of 1,192 (M).

## 1. The sensor-bearing universe (Q1; all M unless marked)
| release | logs | sessions | vehicle-days | in navtrain | in navtest | **outside both** | source |
|---|---|---|---|---|---|---|---|
| OpenScene **trainval** | 1,310 | 171 | 46 | 1,192 | 0 | **118** | `raw/members_trainval_tgz.json` (tar listing of the local 7.05 GB tgz, sha256 = HF LFS oid, `raw/sha_check.json`); equals the camera-archive map `openscene_sensor_trainval_0-199.json` (200 archives, 1,310 distinct logs) and TanitAstra's inventory receipt |
| OpenScene **test** | 147 | 46 | 20 | 0 | 136 | **11** | `raw/members_test_tgz.json` (local tgz, sha verified) = the 147 extracted pkl in `D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test` |
| OpenScene **mini** | 64 | 52 | 35 | 52 | 10 | **2** (both also in the trainval 118) | `raw/members_mini_tgz.json` |
| **union** | **1,457** | **217** | **66** | 1,192 | 136 | **129** | `raw/pool_lists.json` |
| navtrain yaml / navtest yaml | 1,192 / 136 | 162 / 44 | 46 / 20 | | | | sha `5f3c406a...` / `46a60ff7...`, equal to the repo's `SOURCES.json` |

* **Sensors exist for all of them.** Trainval: 200 camera archives, 1,242.2 GB, plus 200 lidar archives. Test: 32 camera archives, 127.9 GB, **already local** at `D:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera/`. Mini: 32 camera archives, 90 GB (none found locally; the nuPlan-mini CAM_F0 zips for all 64 logs, 52 GB, ARE local at `D:/Projects/TanitAD/data/nuplan-camera/mini/`). Sizes: HF tree at main = pinned commit `a76f840b` (M, `raw/hf_tree_main_*.json`).
* **The raw nuPlan public sensor sets are the same three releases; no log hides there** (this is the object the prior probe never looked at). S3 listing: train_set 43 camera archives (5.39 TB), test_set 12 (644 GB), val_set 12 (867 GB), mini_set 9 (451 GB). The three `public_set_*_sensor.txt` sizes (42,950 / 5,903 / 2,622 bytes) are reproduced **to the byte** by 1,085 (= trainval minus val) / 147 / 64 names (T4, with the val file as a control at 8,945). The third-party mirror `kesai-labs/nuplan` lists exactly 1,085 + 225 + 147 per-log directories whose names equal the OpenScene lists (T5). The `.txt` files themselves are a download and were not read (`raw/DOWNLOAD_REQUEST.md` A1-A3).
* **Warmup and navhard add nothing.** `warmup_two_stage`: 7 logs (local), all in navtest. `navhard_two_stage.yaml`: 76 logs, all in navtest. Their extra frames are synthetic views of navtest scenes.
* **Private test sets** (`private_test_e2e` 23.59 GB, `_wm` 14.99 GB, `_hard` 0.64 GB; navsim-v2 `private_test_hard_two_stage` 10.54 GB) have public frames and hidden ground truth: unscorable by us. Their log membership is **U** (metadata not local, 11.65 MB, request B).

## 2. How unseen is the outside pool (Q2)
Counts are from the OpenScene pkl of each log (restricted unpickler, 158 trainval + 147 test logs profiled; 40 navtrain control logs reproduce TanitAstra's independent parse exactly, T7). Anchors use NAVSIM's navtrain geometry: 4 history frames incl. current, 10 future frames, 2 Hz, and a non-empty route at the current frame. **Tokens (E)** = route anchors x 0.1581 (navtrain tokens / route anchors over its 1,192 logs: 103,037 / 651,526); range = the p10-p90 of per-session ratios (0.119-0.251, n = 162 sessions).

| pool | logs | sessions | vehicle-days | cities | hours | route anchors | est. tokens (p10-p90) | session-unseen vs navtrain | vehicle-day-unseen |
|---|---|---|---|---|---|---|---|---|---|
| **A. trainval, session-unseen** (outside navtrain's 162 sessions) | 58 | **9** | 9 | 3 (LV 3, Pittsburgh 3, Boston 3); 7 vehicles | 5.68 | **866** | **137 (103-217)** | yes | **no: 9 of 9 have 1-7 navtrain sibling sessions the same day** |
| A-scorable (has route) | **2** | **1** (`2021.08.24.13.12.55_veh-45`) | 1 | Pittsburgh | 0.12 | 866 | 137 | yes | no (7 siblings) |
| A-route-less | 56 | 8 | 8 | 3 | 5.55 | 0 (39,310 geometric) | 0 as scored; ~6,217 (4,671-9,862) only if a route is synthesised | yes | no |
| **B. trainval, log-unseen but session-seen** | 60 | 30 | 20 | 4 (incl. Singapore) | 1.73 | 10,474 (56 routed logs) | ~1,656 (1,245-2,628) | **no** (every session is in navtrain) | no |
| **C. test, outside navtest** | 11 | 4 (2 are navtest sessions, 2 new) | 4, all navtest's | Boston | 0.51 | 306 (2 routed logs) | ~48 (36-77) | yes vs navtrain; **navtest-adjacent** | no (all 4 are navtest vehicle-days) |
| D. mini, outside both | 2 | 2 | | | | 159 (1 routed log) | ~25 | = 2 of pool A | no |

* **Session-unseen: 11 sessions in total** (9 in A + 2 new in C) against 217. **Vehicle-day-unseen: 0 of 66** (outside_vehicle_days = 0): every vehicle-day of the whole public universe is already in navtrain or navtest. So the weakness the MM measured on the R-A reserve (every session keeps a same-vehicle same-date sibling in training) is **not curable from outside**; navtest's own unseenness (0 of 20 vehicle-days shared) cannot be reproduced.
* Pool A's 9 sessions: `2021.06.08.12.10.22_veh-38`, `2021.06.23.20.43.31_veh-16`, `2021.07.16.23.22.27_veh-14`, `2021.08.17.15.02.08_veh-45`, `2021.08.17.18.43.12_veh-43`, `2021.08.24.13.12.55_veh-45`, `2021.09.15.13.12.49_veh-39`, `2021.09.15.16.17.26_veh-28`, `2021.10.01.14.20.36_veh-28` (per-session detail: `raw/pool_summary.json`, `raw/pool_lists.json`).
* Where they sit: 54 of pool A's logs come from the raw nuPlan TRAIN sensor set and 4 from VAL (the prior probe's 11 val-outside logs are 4 of A plus 7 of B). So the prior lever **L2** ("train-split logs not in navtrain with public frames") is answered: they exist, 107 of the 118, and they are the same route-less / session-seen material.
* Not outside the pool, not counted: navtrain's own ~548,000 anchors that are not navtrain tokens (log-seen), and navtest's 57,000 non-token anchors in the same logs as the judge (using them to fit would leak the final judge's logs).

## 3. Can it be PDMS-scored by our harness? (Q3)
* **What PDMS needs (M, read from `D:/Archive/devbox-C/navsim/devkit/navsim`).** `NavSimScenario` is built from the OpenScene pkl (ego states, GT agent boxes, traffic lights, `roadblock_ids`) plus the nuPlan maps; `metric_cache_processor.py` imports no nuPlan DB builder. `dataloader.py:53` drops every scene whose current frame has empty `roadblock_ids` (`has_route`), and the cache's planner needs the route roadblocks. **The nuPlan DB is not required; the route is.** Maps are local (`D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.1.zip`). Whether TanitAD's own harness has any extra DB dependency is **U**; for the record only 4 of the 58 pool-A logs have a local nuPlan DB (`raw/u58_db_status.json`; the rest sit in the nuPlan train zips, 1.02 TB, not local).
* **All 58 pool-A pkl carry GT** (`anns.gt_boxes` etc., `traffic_lights`, 8 cameras, `ego_dynamic_state`, `driving_command`; identical key set to the controls). **But 56 of 58 have no route** (and a constant unknown nav command), so NAVSIM-standard PDMS cannot be computed on them. Same for 9 of the 11 test extras (probe `raw/route_probe_test.json`: 0 roadblock ids on every frame of the 2 probed route-less logs; the 2 routed logs probed carry 15-37).
* Scorable and session-unseen: 2 logs (866 anchors). **Nav is an input at training and inference (CLAUDE.md), so a route-less log cannot even be fed in the deployed regime**; synthesising a route from the ego's own future is the "optimistic by construction" supplier the CLAUDE.md warns about and would need a PI ruling.
* Private test splits: hidden GT, unscorable.

## 4. Fetch cost (Q4; sizes M from listings, savings E)
Metadata (pkl, GT, routes) and maps for everything above are already local. Frames:
| need | cheapest route | size |
|---|---|---|
| frames for the 2 scorable logs | 1 log is local (`.../nuplan-camera/mini/2021.08.24.13.12.55_veh-45_00386_00472_CAM_F0.zip`, 191 MB, 860 jpg); the other: mirror CAM_F0 0.61 GB | 0.61 GB |
| route-less 56 + 2 routed, CAM_F0 | third-party mirror per-log arrow files (integrity **U**, 10 Hz, needs timestamp alignment) | **39.7 GB**; 4 cameras 161.9 GB |
| same, first-party | 49 of 200 OpenScene trainval camera archives (gzip stream, all 8 cameras) | 329.1 GB whole; ~165 GB prefix-scan **E** |
| test extras (11) | already local inside the 32 test archives; locating them needs request A4 (7.9 KB) or a 120 GB scan | 0 download |
Itemised URLs, sizes and the question each answers: `raw/DOWNLOAD_REQUEST.md` (nothing in it is needed for this verdict).

## 5. Verdict (Q4)
| use | does a pool exist? | size vs need | judgement |
|---|---|---|---|
| **(a) selection / calibration on P6's scorer** (P6 trained on all of navtrain) | session-unseen and scorable: **1 session, 2 logs, 866 anchors, ~137 tokens E** | the RL-val selection tier alone was 3,000 sets / 75 logs (I, `GOALS_AND_CLAIMS.md` D-REFE-SELRULE-FIT); the P7 reserve is 16 sessions / ~10,100 scenes | **No.** Two orders of magnitude too small; a CI on it would be wider than any lever. Not vehicle-day-unseen either |
| **(b) P7.2 fit set** (scorer ensemble, pessimism, calibration) | same pool; P7's reserve cannot be replaced | as above | **No.** Keep the navtrain-internal session reserve (R-A or R-B, MM ruling pending). External data cannot supply navtest-like vehicle-day-unseenness: 0 of 66 vehicle-days are outside |
| (c) a route-synthesised pool (A-route-less) | 56 logs, 8 sessions, ~6,200 tokens E | 39.7-161.9 GB, third-party | **Do not buy.** Unscorable under NAVSIM as is; synthesised route is oracle-optimistic and changes the regime (no nav command in the data); 8 of 8 sessions have navtrain siblings; value is below the reserve already in hand |
| (d) B (log-unseen, session-seen) | 56 routed logs, ~1,656 tokens E | | **No.** Session-seen: its sessions' neighbours were trained on |
Cheapest remaining lever if the reserve is judged too small (RULE ZERO): enlarge the reserve inside navtrain (the MM's open question #2: 24-32 sessions), a 0-GPU decision, instead of buying external data.

## 6. The two prior claims, challenged
* **`D-DATA-U-FRAMES-1`** (val index `public_set_val_sensor.txt`: 225 = 214 + 11; "essentially navtrain U navtest"). *Right in effect, wrong in object and number.* The val release is 225 of 1,457 sensor-bearing logs (15 %). Across the universe 129 logs are outside (8.9 %), 118 in trainval and 11 in test. By vehicle-day the statement is exact (66 of 66 inside); by session it is not (11 of 217 outside); by usable route-bearing session-unseen it is nearly exact (1). Its lever L2 and L3 are now answered by name lists, not by assumption.
* **`FEASIBILITY.md` §7** ("no session-unseen pool outside navtest"). *False as written* (9 trainval sessions are absent from navtrain; 2 more in test); *true in effect* (1 scorable session, ~137 tokens). Suggested wording: "no usable session-unseen pool exists outside navtest: the public universe holds 11 sessions outside navtrain U navtest, 1 of them PDMS-scorable (866 anchors); all 66 vehicle-days are inside."

## 7. Limits, UNVERIFIED items, and what I did not do
* **U:** name-for-name identity of the raw nuPlan `public_set_*_sensor.txt` with OpenScene's lists (size-exact and mirror-corroborated, not read); the log membership of the private test sets; that the third-party mirror's bytes match nuPlan's; whether TanitAD's own harness needs anything beyond pkl + maps for PDMS; the exact per-log byte positions inside gzip archives (cost 165 GB is a uniform-position **E**).
* **E:** all token counts (a ratio fitted on navtrain's filter; the filter is not a uniform rate, hence the p10-p90 range). A pool we would select ourselves could hold up to the full anchor count (866 / 39,310 / 10,474), but those overlapping windows are not independent scenes.
* **Scope note.** "Public sensor-bearing driving-log pool" was read as the nuPlan / OpenScene / NAVSIM universe, the only one PDMS can score. Other corpora (PhysicalAI-AV, nuScenes, Waymo) are domain-unseen, not session-unseen, and have no NAVSIM metric cache path; not probed. nuPlan's sensorless logs (mirror `logs_sensorless`: 121 + 12 + 13 tars, ~462 GB) carry no camera frames and are not sensor-bearing.
* A third-party mirror was used for listings only. I read no file body from it and no header of any remote archive; locally, pickles were read only through a restricted whitelist (numpy array globals only; anything else raises).
* Nothing was downloaded; 0 object bytes read from any remote. Remote requests: 111, all metadata listings (6 S3 prefix listings, 26 HF tree listings, 73 HF per-log directory listings of the mirror, 3 HF repo-info, 3 HF search), 2.97 MB of listing payload in total, 0 errors (`raw/requests_log.jsonl`, URL + status + bytes each).

## 8. Reproduce (CPU, local inputs only)
```
python -I code/test_pool.py                      # 23 checks incl. controls and a mutation arm -> raw/test_run.out (PASS 23)
/d/venvs/opencv-ref-454/Scripts/python.exe -I code/profile_logs.py <trainval.tgz> raw/wanted_profile.json raw/profile_logs.json   # numpy needed; ~6 min
python -I code/analyze_extras.py ; python -I code/build_lists.py     # -> raw/pool_summary.json, raw/pool_lists.json
```
`code/draw_reserve_REF.py` is a verbatim copy of the reference rule (from mirror tip `f1f2926`), used only by the test to pin definitions.

## Deliverable manifest (nothing staged, committed or pushed, per the local override; every artifact exists in ONE place only)
| artifact | where | other copy |
|---|---|---|
| `RESULT.md` | `C:/Users/Admin/par18/unseenpool/RESULT.md` | **none** |
| `raw/DOWNLOAD_REQUEST.md` | `.../unseenpool/raw/` | none |
| `raw/pool_lists.json` (129 outside logs with session, vehicle-day, route, anchors, archive id, mirror bytes; 43 session rows), `raw/pool_summary.json`, `raw/profile_logs.json` (158 trainval logs), `raw/profile_test_logs.json` (147 test logs), `raw/openscene_trainval_extras_archives.json` | `.../unseenpool/raw/` | none (inputs are TanitAstra / devbox files on D:, read-only) |
| `raw/members_{trainval,test,mini}_tgz.json`, `raw/test_release_logs.json`, `raw/mini_logs.json`, `raw/sha_check.json` | same | none |
| listings: `raw/s3_list_*.json`, `raw/hf_tree_main_*.json`, `raw/hf_other/*`, `raw/mirror_log_sizes_*.json`, `raw/hf_search_results.json`, `raw/requests_log.jsonl` | same | none |
| `raw/src/` (navtrain / navtest lists, val sensor index, token table copies from mirror tip `f1f2926`) | same | the repo (these four are copies) |
| `raw/test_run.out`, `raw/route_probe_test.json`, `raw/navtrain_density.json`, `raw/SHA256SUMS.txt` | same | none |
| `code/` (common, test_pool, profile_logs, profile_dir, route_probe, analyze_extras, build_lists, navtrain_density, mirror_log_sizes, s3_list, hf_tree, list_tgz_members, sha_check, draw_reserve_REF) | `.../unseenpool/code/` | none |
Suggested repo path: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-22-unseen-pool-2/`. The Master Mind lands it; the two register corrections in section 6 are the integration items.
