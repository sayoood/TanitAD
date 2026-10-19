# UNSEEN-STRATUM DATA JOB, step S1: a DB-only token rule (2026-10-19; CPU, read-only; no network, no label, no GPU)

Package `C:/Users/Admin/par18/unseen_s1/` (staged copy `out/repo/...`, list `out/paths.json`). Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `CUDA_VISIBLE_DEVICES=-1`. Nothing committed or pushed. Plan: `C:/Users/Admin/par18/l2fresh/READY.md` section I, S1.

## 0. Headline (MEASURED unless marked)
| item | value | artifact |
|---|---|---|
| identity, 214 navtrain logs in the val split (the FIT set) | **recall 18,122 / 18,122 = 1.0000** (all 214 logs = 1.0); precision 18,122 / 93,034 = **0.1948** (per log: p5 0.056, p25 0.131, **median 0.189**, p75 0.246, p95 0.423, max 0.574) | `raw/identity_214.json` |
| identity, 248 top-level navtrain logs (OUT OF SAMPLE: rule frozen first) | **recall 19,307 / 19,810 = 0.9746** (241 of 248 logs = 1.0, 7 logs = 0); precision 19,307 / 130,190 = **0.1483** (median 0.150) | `raw/identity_toplevel248.json` |
| RED, off-by-one cadence (phase +1, -1, +5) | recall **0.0000** on both sets (0 of 214 and 0 of 248 logs reach 0.5) | `red_arms.cadence_phase_*` |
| RED, wrong history / future window | count shifts by exactly **+/-1 per log per frame** (history 3: +214, 5: -214, 6: -428, 8: -856; future 9: +200, 11: -201, 8: +400; the brief's 2 s / 4 s = (5, 8): +186), recall falls only when the window is stricter than navtrain's | `red_arms.window_*` |
| U (1,167 logs): candidate tokens | **548,812 tokens in 1,129 logs** (38 logs carry 0: every window fails the route clause); per log median **292**, p5 112, p25 172, p75 515, p95 1,431, max 7,238 | `raw/U_candidates_per_log.json`, `raw/U_candidate_tokens.json.gz` |
| fetch list (seeded, PROVISIONAL) | **1,968 jpgs** (123 logs x 4 tokens x 4 cameras, 1,968 unique, 0 tokens skipped, max camera dt 47.2 ms); **ESTIMATED 404 MB** (1,968 x 0.2055 MB, mean of 3,000 local navtrain jpgs, MEASURED sizes) | `raw/fetch_list_s2_s4.{json,txt}`, `raw/jpg_size_sample.json` |
| frozen module | `code/unseen_token_rule.py`, LF sha256 `9fd2917f67f4d06c2ddeefb8acd74b0f3bd475f7cc07e6258cf360a2581c374f` (`raw/rule_freeze.json`); test: `10 passed` | `code/test_unseen_token_rule.py`, `logs/pytest_test_unseen_token_rule.out` |

**The rule names a SUPERSET of navtrain** (about 5 candidates per navtrain token). Navtrain's token list is a PDM-score filter on top of the window rule (section 2, C4); the DB cannot reproduce it. The downstream sampler therefore needs MORE than candidate validity if its tokens are to resemble navtrain's (section 5).

## 1. The sources (every adopted clause)
`devkit` = `C:/Users/Admin/navsim-crun/devkit` (the NAVSIM checkout on this box; no git metadata, so the pinned commit is UNVERIFIED).

| clause | rule | source (file:line) |
|---|---|---|
| window | a scene = 4 history + 10 future = 14 consecutive OpenScene frames, stride `frame_interval` = 1; the scene TOKEN is frame index `num_history_frames - 1` = 3; a window shorter than 14 is dropped | `devkit/navsim/common/dataloader.py:24-26` (`split_list`), `:47`, `:49-50`, `:57`; `TanitAD Research Lab/.../2026-09-20-refe-plan/splits/navtrain.yaml:3-6` (`num_history_frames: 4`, `num_future_frames: 10`, `frame_interval: 1`, `has_route: true`), byte-identical in `devkit/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml:3-6`; defaults `devkit/navsim/common/dataclasses.py:760-763` |
| route | a window whose CURRENT frame has empty `roadblock_ids` is dropped; in the DB the route is `scene.roadblock_ids` | `devkit/navsim/common/dataloader.py:52-54` |
| cadence | OpenScene frames are 2 Hz = every 10th nuPlan `lidar_pc` (20 Hz: median dt 0.050 s) | `devkit/docs/cache.md:3` ("sensor data at 2Hz"); stride and phase are MEASURED (below), because the OpenScene metadata builder is not on this box |
| NOT in the rule | navtrain's `tokens:` list (103,288 tokens = the token table, set-equal, MEASURED) is a PDMS filter: CV agent PDMS > 0.8 removed, human PDMS < 0.8 removed | Dauner et al. 2024 sec. 3.1, quoted in `products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md:386-391` |

## 2. What was MEASURED and had to be corrected (no clause is taken on faith)
Tokens are `lidar_pc` tokens: 18,122 of 18,122 known navtrain tokens in the 214 logs are found in their DB (hex of the BLOB).
1. **Phase.** Per log, every known token sits on ONE residue mod 10 of the timestamp-sorted `lidar_pc` index (distinct residues per log: 1 in all 60 logs first probed; later 214/214). The residue is the first `lidar_pc` with a synchronised camera image: a lidar_pc is "paired" when its nearest image is within 25 ms (`|dt|` is bimodal: 12-14 ms and 36-38 ms over 191,903 lidar_pc in 40 DBs).
2. **First version (CAM_F0 phase, devkit window k >= 3): recall 0.98758 on the 214, with TWO defects found by looking at the misses.** (a) **0 of 18,122 known tokens sat at the third frame (k = 3) while 37 sat at the mirrored end position**; the OpenScene list starts one grid step after the first paired lidar_pc (`DROP_FIRST_FRAME = 1`, equivalently k >= 4 in the undropped grid; known tokens by undropped-grid position: 0..3 -> 0, 4 -> 37, 5 -> 55, 6 -> 47). (b) **6 logs, all of drive `2021.08.24.20.03.01_veh-45`, had recall 0**: CAM_F0 pairs at lidar index 0, but the first CAM_L0 image arrives 80 ms later. The max of the CAM_F0 and CAM_L0 first-paired indices fixes all 214 (searched: all 255 camera subsets, F0+L0 = 214/214; CAM_L0 alone 213).
3. **Out of sample (rule frozen, 248 top-level navtrain logs): 241 of 248 logs recall 1.0; 7 logs recall 0** (all vehicle veh-26, `2021.06.14.20.14.09`, `2021.06.23.14.06.20`, `2021.06.23.18.23.38`, `2021.07.09.01.37.16`): the true grid is 2 lidar_pc later than the rule's phase. In them CAM_F0 and CAM_L0 first images arrive 10-22 ms AFTER lidar_pc 0 while R0, B0, L1, L2, R1, R2 pair at lidar_pc 1. **No rule tried recovers them** (`code/explore/explore8.py`, `explore9.py`, `explore12.py`: signed windows, "image available at or before the lidar_pc", all-8-cameras, subset x window scans: best 455 of 462 logs). **KNOWN LIMIT, not hidden:** 7 of 462 logs (1.5 %; 2.8 % of the out-of-sample 248; 0 of the 214 fit logs, which were fitted) have candidates on a grid 100 ms off OpenScene's; the rate on U is UNVERIFIED (the failures sit in particular camera-timing signatures, so U may carry more or fewer). Those tokens are still valid frames (F0 and L0 images within 25 ms), so a sampler that needs VALID frames is unaffected; a check against the OpenScene metadata (7.05 GB, not downloaded) would be.
4. **Edges (frozen rule, both sets):** known tokens at k = 3: 37 (214) / 29 (248); at the last allowed position k = M - 11: 37 / 37; none beyond either edge. Future 10 and history 3 are therefore each attained, and F = 11 / H = 6 lose recall (RED).
5. **Superset size:** the rule's candidates per log: median 341 (214) / 318.5 (248); navtrain keeps 19.5 % / 14.8 % of them; navtrain density per log spans 0.021 to 0.574.

## 3. RED arms (all on the frozen module; full table in `raw/identity_*.json`, `red_arms`)
| arm | 214 logs: recall / n candidates | 248 logs: recall / n candidates | reads |
|---|---|---|---|
| frozen rule | **1.0000** / 93,034 | **0.9746** / 130,190 | |
| phase +1 / -1 / +5 (off-by-one, half a frame) | **0.0000** (all three) | **0.0000** | cadence collapses |
| phase fixed at 0 (camera pairing ignored) | 0.2068 | 0.2746 | pairing matters |
| stride 9 / 11 | 0.1093 / 0.0902 | 0.1101 / 0.0900 | stride matters |
| stride 20 (1 Hz) / 5 (4 Hz) | 0.4838 / 1.0000 (precision 0.0959) | 0.4749 / 0.9746 | 4 Hz is a superset |
| pairing tolerance 40 ms / 5 ms | 0.2222 / 0.0040 | 0.3168 / 0.0006 | tolerance is load-bearing |
| phase from CAM_F0 only (first version) / CAM_L0 only | 0.9876 / 0.9931 | 0.8352 / 0.9743 | corrections earned |
| first grid frame NOT dropped | recall 1.0000, **+214 candidates** | 0.9746, +248 | NOT identifiable by recall; identified by the zero count at that position (section 2.2a) |
| history 3 / 5 / 6 / 8 | count +214 / -214 / -428 / -856; recall 1.0 / 0.9980 / 0.9949 / 0.9893 | +248 / -248 / -496 / -992 | count shifts measurably |
| future 8 / 9 / 11 | +400 / +200 / -201; recall 1.0 / 1.0 / 0.9980 | +472 / +236 / -236 | |
| the brief's 2 s history, 4 s future = (5, 8) | +186; recall 0.9980 | +224; 0.9731 | a looser rule than navtrain's |
| route filter off | +620 candidates, recall 1.0 | +169 | the clause bites on 0.7 % / 0.1 % of windows here, on 2.7 % of U windows |

## 4. U (1,167 logs; STRUCTURAL reads only: lidar_pc, CAM_F0 + CAM_L0 image timestamps, scene.roadblock_ids emptiness; no ego_pose, no scenario_tag, no label)
* The U run was repeated on the frozen (LF-normalised) file: `U_candidates_per_log.json` and `fetch_list_s2_s4.txt` are byte-identical to the first run (sha256 `bb2b4179...` / `5e7b4b44...`).
* U is disjoint from the navtrain yaml's 1,192 `log_names` (0 shared; the yaml names 1 log more than the token table's 1,191).
* **548,812 candidate tokens**, 1,129 logs with at least 20, **38 logs with 0** (all their windows have no route: logs of 121-1,171 grid frames; whole logs with empty `roadblock_ids`). Per-log distribution above. Phase residues: 0 -> 387 logs, 1 -> 552, 2 -> 228. 15,289 windows (2.7 %) were removed by the route clause. Per log counts: `raw/U_candidates_per_log.json`; the token list (gzip, 5.6 MB): `raw/U_candidate_tokens.json.gz`.

## 5. Does the sampler need only candidate validity? NO, for comparability with navtrain. The sampler proposal.
**Validity is necessary and the frozen rule provides it. It is not sufficient for a like-for-like A1.** MEASURED on the 214 logs, features that need NO label (ego pose, scenario tags; navtrain logs only, `raw/bias_214.json`, `raw/sampler_check.json`), navtrain tokens vs ALL candidates, pooled:
| feature | navtrain | all candidates |
|---|---|---|
| stationary (ego speed < 0.5 m/s) | **6.3 %** | **30.8 %** |
| 4-s progress < 2 m of the logged ego | 2.3 % | 29.3 % |
| turning (> 15 deg heading change in 4 s) | **36.3 %** | **13.9 %** |
| nuPlan tag "stationary" at the token | 4.4 % | 28.2 % |
| speed median / p25 (m/s) | 5.42 / 2.93 | 4.53 / 0.00 |
A uniform draw over the candidates therefore hands A1 a population with 5 times navtrain's stationary share and 0.4 times its turning share. Label-free features separate navtrain from the other candidates with AUC **0.879** (speed + 4-s progress + 4-s yaw + acceleration) and 0.883 (+ scenario tags), 5-fold by log; speed alone 0.733. So the filter is partly, not fully, reproducible without scores.

**Proposed sampler (S3 to adopt or amend; NOT frozen here):** candidates of the frozen rule; a seeded weighted draw WITHOUT replacement within each log (Efraimidis-Spirakis keys `u^(1/w)`), 4 tokens per log, clusters = logs; weights `w` by ego speed at t0 (an INPUT at inference, not a label), 13 bins with edges 0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12 m/s: `0.168, 1.600, 1.653, 1.341, 1.349, 1.247, 1.249, 1.186, 1.045, 0.917, 0.632, 0.354, 0.259`, fitted by 60 rounds of iterative proportional fitting so that the DESIGN-weighted (equal log weight) speed histogram equals navtrain's on the 214 logs (`raw/sampler_check.json`).
| design-weighted share (equal log weight) | navtrain | uniform | **speed-weighted** | gap closed |
|---|---|---|---|---|
| 214 (fit): stationary < 0.5 m/s | 0.077 | 0.273 | **0.077** | 100 % |
| 214: tag "stationary" (not in the weights) | 0.050 | 0.248 | 0.078 | 86 % |
| 214: mean abs longitudinal acc (not in the weights) | 0.595 | 0.383 | 0.550 | 79 % |
| 214: progress 4 s < 2 m (RESIDUAL, logged future) | 0.026 | 0.257 | 0.111 | 63 % |
| 214: turning > 15 deg (RESIDUAL, logged future) | 0.321 | 0.143 | 0.206 | **35 %** |
| **248 (OUT OF SAMPLE, weights frozen)**: stationary | 0.096 | 0.345 | **0.105** | 96 % |
| 248: tag "stationary" | 0.094 | 0.324 | 0.135 | 82 % |
| 248: progress 4 s < 2 m | 0.034 | 0.331 | 0.148 | 62 % |
| 248: turning > 15 deg | 0.298 | 0.108 | 0.169 | **32 %** |
A simulated seeded 4-per-log draw (200 reps) reproduces the analytic stationary share: 0.078 (214) and 0.107 (248).

**Why it may still bias A1, stated plainly:** the weights remove the stationary excess and close 60-80 % of the gap on features they do not use, but **close only a third of the turning gap**, and navtrain's remaining edge comes from the PDMS filter itself (CV agent fails, human passes), which depends on agents and the map. The DB cannot reproduce that. **Whether A1 depends on these covariates is NOT measured here (it needs labels, which S1 may not read).** The weights use only t0 ego speed. Matching the turning / progress shares needs the logged expert FUTURE, which is the planning ground truth: label-adjacent, so NOT proposed in S1 and left to the MM / PI. Recommended for the S3 prereg: (a) the speed-weighted draw above; (b) report A1 within label-free strata (moving / stationary, turning / straight from the logged ego) and the navtrain-composition-reweighted A1 beside the plain one, both committed in advance; (c) stamp the result "OpenScene-unfiltered candidates, t0-speed-matched to navtrain, NOT PDMS-filtered", never as like-for-like with Step A's navtrain number; (d) both outcomes committed before any U label is read.

## 6. The fetch list (S2 / S4; nothing fetched, no network)
`raw/fetch_list_s2_s4.txt` (one `<log>/<CAM>/<hash>.jpg` per line, 1,968 lines, the layout of `readtwo_fetch_list.txt`) and `raw/fetch_list_s2_s4.json` (per token: log, lidar_pc token, timestamp, per-camera path and dt). Camera set = Step A's bundle: **CAM_F0, CAM_L0, CAM_R0, CAM_B0** (`refe/planner.py:54` `CAMERAS`; `make_stepA_bundle.py:82` builds `FrameResolver` over them; nearest image per camera, refused beyond 120 ms, `planner.py:71`); the bundle reads ONE timestamp per token, so 4 jpgs per token, no history frames. Sample: seed 20261020 (**PROVISIONAL**: S3's sealed prereg fixes the real seed and sampler), 123 logs drawn from the 1,129 logs with at least 4 candidates, 4 tokens per log drawn UNIFORMLY from the rule's candidates (the speed-weighted draw would need `ego_pose` of U, which S1 does not read; a re-draw under S3's sampler is one command and the same 1,968 jpgs in count). Size: 1,968 x 0.2055 MB = **404 MB** (mean of 3,000 local navtrain jpgs: F0 0.195-0.200, L0 0.201-0.213, R0 0.216, B0 0.198-0.205 MB; `raw/jpg_size_sample.json`). S2 (archive lookup) needs only the 123 log names.

## 7. Findings that change the plan
1. The token frame is now fixed and checked: no navtrain list is needed to name candidates in U.
2. 38 of 1,167 U logs have no route anywhere; the usable pool is 1,129 logs (the design calc needed >= 123).
3. The 2.8 % phase-miss logs are a known limit; if exact OpenScene alignment matters later, download the OpenScene metadata for U and compare (not needed for validity).
4. UNVERIFIED and named: that the val camera zips hold every U log (S2); the NAVSIM devkit commit behind `devkit/`; whether A1 is sensitive to the covariates of section 5.

## 8. Deliverable manifest
| artifact | where |
|---|---|
| frozen rule `code/unseen_token_rule.py`, test `code/test_unseen_token_rule.py` (10 passed) | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-unseen-stratum-s1/code/` (STAGED copy `C:/Users/Admin/par18/unseen_s1/out/repo/...`; originals `C:/Users/Admin/par18/unseen_s1/code/`) |
| `identity_check.py`, `bias_measure.py`, `sampler_check.py`, `apply_U_and_fetch.py`, `make_stage.py`, `code/explore/*` | same |
| `raw/identity_214.json`, `identity_toplevel248.json` (+ `_per_log`), `bias_214.json`, `sampler_check.json`, `rule_freeze.json`, `jpg_size_sample.json`, `U_candidates_per_log.json`, `U_candidate_tokens.json.gz`, `fetch_list_s2_s4.{json,txt}`, `logs/*` | same (`raw/`) |
| the package list | `C:/Users/Admin/par18/unseen_s1/out/paths.json` |
Single-location artifacts: none beyond the staged copy; large files: none (the token list is 5.6 MB). D: untouched (read only). Integration: the rule is ready for the S3 prereg; a decision is needed on section 5 (sampler weights; whether to match the turning share with logged-future information).
