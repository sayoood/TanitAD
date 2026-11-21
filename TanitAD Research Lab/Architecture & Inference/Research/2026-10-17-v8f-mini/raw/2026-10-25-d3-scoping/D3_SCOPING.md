# D3 SCOPING: what would "+4,512 clips" (2x unique) and "+40,000 clips" cost, and what is missing before the day the DSCALE read lands

Written 2026-10-25 on tip `343f366d` (export). Read-only: no download, no GPU, no box 2, nothing deleted. Probes: `probe_holdout_shift.py`, `probe_pool_gates.py`, `d3_cost_model.py` (each with its `.out` beside it).
Evidence classes: MEASURED (ours, artifact named) · INHERITED (another doc, not re-run) · ESTIMATED (arithmetic on the constants, or a guess, said so) · UNKNOWN.

## HEADLINE

1. **D3 at 2x (+4,512 clips) is feasible on the hardware we own; what gates it is PI approvals (download, disk, box-2 hours) and code that does not exist yet.** ESTIMATED wall, pipelined: 11-17 h to a finished cache (download 3.6-15.6 h, dev-box CPU build 10.3 h, box-2 encode 6.3 h), then 4.4 h of box-2 training per seed at fixed steps. All times are arithmetic on MEASURED constants (`d3_cost_model.out`).
2. **+40,000 is not a D3 rung.** The programme's own extrapolation rule (<= 2x beyond the fitted range) caps D3 at +4,512 until that rung reads. As a capacity number: 0.56-0.90 TB download, 31-136 h of WAN, a 174 GB cache that does not fit box 2 (last read 85.9 GB free), 185 box-2 hours on the RA-1 recipe. It needs a disk decision and its own PREREG.
3. **PI decisions (all five block only the download/GPU steps; steps A-D below start now):**
   * **bandwidth:** 65.1 GB (B1-like clip sizes) to 102.8 GB (corpus-mean sizes) from `nvidia/PhysicalAI-Autonomous-Vehicles` on the dev box, preceded by a ~0.4 GB / 20-clip pilot to measure the rate (nobody has a recent number);
   * **disk:** the dev box has **C: 64.6 GB, E: 2.1 GB (exFAT, 100 % full), F: 787 GB (Transcend, NTFS; it holds unrelated personal archives, so using it is the PI's call), G: 61.4 GB** free (MEASURED today). A 400-clip chunk needs 20-23 GB transient. C: can do it with no other growth; F: is the only roomy drive. Box 2 `D:` was **85.9 GB free at 2026-10-19 06:37 (INHERITED, not re-probed)**: it holds +19.7 GB of 4x10-only cache, NOT +94 GB of both grids;
   * **box-2 hours:** 6.3 h encode (variant V3: dev builds, box 2 only encodes) versus 20.9 h (V1: the RA-1 FULL recipe), then 2 x 4.4 h training;
   * **pool policy:** uniform over all 301,433 non-B1 clips (default) versus NVIDIA's train split only versus also dropping the parity-train clips (see 4);
   * **dev-box 4060 for encoding (V2):** the 2026-10-17 orchestrator decision said "not on the dev-box 4060".
4. **Three traps would have invalidated D3 silently; one is MEASURED here by simulation, two are established from source `file:line`.**
   * **The RA-1 holdout is excluded by POSITION (`--exclude 540:600`, `v8f_mini_train.py:328`). Merge new clips into the manifest and 60/60 of the original holdout clips are trained on, and 60 other clips are held out. MEASURED** (`probe_holdout_shift.out`). G-DVB checks the COUNT (60 = 60) and `excluded_in_train == 0`, not the identity (`v8f_mini_train.py:235-239`); neither `v8f_mini_train.py` nor `launch_gate.py` asserts the `f6a1dee0` pin (grep: the trainer only records `excluded_sha256`, `:342`, `:515`). It is asserted in box-2 chain scripts (`chain_v8f_grid.sh:47,179`) and RA-1 readers, so the trainer's own gate would pass and only a chain that carries that assertion would catch it.
   * **The encoder REFUSES any clip outside B1** (`pull_encode_pooled.py:172`, `:537`: `train_all == tr_lab`, `--only` must lie in B1-minus-eval). `ra1_full_box2.py`, `verify_cache.py` and `stage_train_all_native256.py` are hard-bound to the same 4,572. D3 needs a code change, with a reachability proof and a G-DVB entry.
   * **Constraint standardisation moves with the data:** `M.cons_stats(J)` (`v8f_mini_train.py:418`) takes mean/sd over the whole join, so an extended join changes T's targets: a second lever unless pinned to B's values. DSCALE's Q arm does not have this problem (its join is unchanged).

## 1. The pipeline for ONE new clip, end to end

`v8f_mini_train.py` reads (build_data, `:301-362`): `<cache>/train/manifest.jsonl` + `<cache>/train/clips/<clip>.pt` (`v8f_mini_data.py:103`, `:408`), the v9 release npz + sibling manifest (`:318`), `lead_train*.npz` and the speed sidecar jsonl (`:320`, joined by `build_label_join` `v8f_mini_data.py:149`), `native_tracks.pt` (`:351`, `v8f_mini_data.py:301`), and the frozen `p10_prior.pt` (`:92`). Walking backwards:

| # | stage | script (file:line) | inputs | output / size per clip | runs on |
|---|---|---|---|---|---|
| 0 | pick clips, gate | `pull_encode_pooled.clip_sets` `:150`, `parity_gate` `:194`; `parity.guard_corpus_build` `parity.py:2278` | `clip_index.parquet` (306,152 clips, 3,146 chunks, MEASURED), digests `deployed_val40 / parity_train / v72_eval_clip_digests.json` | a sha256-pinned list | dev, CPU |
| 1 | download camera, timestamps | HTTP-range zip reader: `Data Engineering/.../2026-08-24-label-extraction-overnight/code/pull_camera.py` (+ `pull_egomotion_range.py`); NOT `physicalai_r0.stage_fetch_camera` `:198` (whole 2.05 GB chunks) | `nvidia/PhysicalAI-Autonomous-Vehicles` `camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_NNNN.zip`: members `<clip>.*.mp4` + `.timestamps.parquet` | mp4 13.06 MB (B1 mean, MEASURED; corpus-mean ~21.4 ESTIMATED), ts 9.4 KB | dev, WAN |
| 2 | download ego log | same recipe | `labels/egomotion/egomotion.chunk_NNNN.zip` -> `<clip>.egomotion.parquet` | 415 KB | dev, WAN |
| 3 | intrinsics (per clip) | `physicalai.intrinsics_for_clip` `physicalai.py:198`; `pull_intrinsics.py` | `calibration/camera_intrinsics/camera_intrinsics.chunk_NNNN.parquet` -> rows appended to `physicalai_front_wide_intrinsics.csv` | ~40 KB per chunk | dev |
| 4 | lead agents | `taniteval/tools/build_lead_block_b1.py:407-560` (range pull), `build_b1_agent_join.py` (the 2026-09-06 train join, per `Benchmarks & Evals/Research/2026-09-06-b1-train-join/README.md`), `build_v9_labels.py cmd_lead_pass :1111` | `labels/obstacle.offline/obstacle.offline.chunk_NNNN.zip` | 517 KB + 262 KB per chunk (MEASURED); 97.1 % of clips have a member | dev |
| 5 | decode + rectify + PNG | `pod_build_b1_epcache.py:155` (`--clips`, parity gate `:186-192`) -> `v2_compressed.build_compressed` `:129` | mp4, ts, `egomotion_all.zip` (a STORED zip of `<clip>.egomotion.parquet`), the intrinsics csv | `<clip>.v2ep.pt` **36.99 MB** (256x640 cylindrical, png) | dev CPU or box 2 CPU |
| 6 | DINOv3-L tokens + pooling | `pull_encode_pooled.py` `encode_clip :302` -> `dinov3_fp8_encode_ship.encode_episode :108`; `pool_field :254`; `kinematics :269` | v2ep; `facebook/dinov3-vitl16-pretrain-lvd1689m` (local HF cache, 1.2 GB, present on the dev box) | `<cache>/train/clips/<clip>.pt`: `pool4x10` fp8 [T,40,1024] (+ `pool8x20` fp8 [T,160,1024] for GRID), `mean` fp16, `v,a,steer,yaw_rate,pose_grid,kin3,p10`. **4.356 MB (4x10), 20.905 MB (both)** | GPU: box 2 or dev |
| 7 | native pose tracks | `build_native_tracks.py:build` (reads `_v2manifest.pt` poses; row i = raw frame i+2) | the v2ep poses (read BEFORE the v2ep is deleted) | ~3.5 KB (`native_tracks.pt` 16.59 MB / 4,713 clips) | dev CPU |
| 8 | label records (no CoT) | `s2_geom_emit_v7.py` `emit_one` (`--clip-index`, no `--alpamayo`) | ego log | one record; the speed band `g_tac.goals.SPEED_BAND.v_hi_ms` | dev CPU, 4.5 ms (MEASURED, B1 records) |
| 9 | speed sidecar | `build_refcv6_speed_max_window.py:70` | step-8 records | `refcv6_speed_max_v8_train.jsonl` row (per CLIP) | dev CPU |
| 10 | clip clock | `build_clip_clock_sidecar.py` | v2 manifest poses + ego log | one jsonl row; without it `clock_src = pose_dt`, grid start 0.0 instead of ~+0.113 s (`build_v9_labels.py:494`) | dev CPU |
| 11 | v9 labels | `build_v9_labels.py cmd_build :1179` (`--manifest --sidecar --labels --lead --ego`) | v2 manifest, clock sidecar, step-8 records, lead pass, ego log | `v9_labels_train.npz` **58 KB/clip**; **0.228 s/clip** (MEASURED) | dev CPU |
| 12 | trainer join | `build_label_join` `v8f_mini_data.py:149`, saved `--join-train` | v9 npz, lead npz, speed jsonl, cache manifest | `join_train.npz` (minutes, once) | box 2 / dev |

Notes that are easy to miss:
* **Labels extend, but not all of them.** MEASURED from source (`build_v9_labels.py:1005`, run): on a clip with no CoT record `has_cot = False`, so `goal_w = 0` for the **13 VLM tokens** (YIELD, CORRIDOR_OFFSET, EVADE_IN_CORRIDOR, OVERTAKE_VEHICLE, MERGE, GAP_TARGET, REACT_ON_ONCOMING, TAKE_EXIT_L/R and **the four TRAFFIC_LIGHT_REACT[_RED/_YELLOW/_GREEN] tokens**) and for LANE_CHANGE_L/R. The 7 geometry goals, lateral/longitudinal classes, the 26 constraint cells, RC checkpoints and nav extend at ego-log cost. So D3 adds no information to the TL and VLM-goal heads (only B1's 4,572 records carry them), and a D3 read of "tactical goal F1" for those tokens is B1-bound.
* **Two label-policy asymmetries.** (a) B1 turn labels carry the PI's CoT-contested-turn suppression (68 clips, `s2_geom_emit_v7.py:872-900`); new clips get geometry-only turns. (b) Lead cells need steps 4 and 11's lead pass; a clip with no `obstacle.offline` member (2.9 % in B1) is NO_LABEL, `lead_valid = 0`.
* **Integrity.** NVIDIA ships no per-file ledger (B1's `camera_sha256.json` is the programme's own, Sayood). `zipfile.read` checks the member CRC32, `pull_camera.py` checks the `ftyp` box; record a sha256 at pull time and run a known-value control first (re-pull 2 B1 clips by range; their sha256 must equal the local copy and the Sayood ledger).

## 2. Per-clip cost

| quantity | value | class and artifact |
|---|---|---|
| download, all four feature files + obstacle | **14.0 MB** (B1 sizes) to **22.3 MB** (corpus-mean) + 0.79 MB per chunk touched | mp4 mean 13.057 MB over 4,719 files (61.617 GB) MEASURED on this box; 2.05 GB per chunk (docstring, INHERITED) / 95.8 clips per chunk MEASURED = 21.4 MB ESTIMATED; ego 415 KB, ts 9.4 KB, obstacle 517,098 B + 262,146 B per chunk MEASURED (`2026-09-06-b1-train-join/README.md`) |
| v2ep intermediate | 36.99 MB | MEASURED, `raw/s4c/check_report_train600.json` (22,192,351,528 B / 600) |
| **cache per clip** | **4.356 MB** (`pool4x10`+`mean`+kinematics), **20.905 MB** (both grids), 4.149 MB without `mean` | MEASURED three ways: (a) a real cache clip re-saved to 4x10-only in scratch; (b) 38 real clips, mean 20.894 MB, `C:/Users/Admin/refav1_hj_cache/test_runscript/train/clips` (416-route test, not adopted); (c) box 2's eval manifest, 141 clips, mean 20.899 MB (`raw/2026-10-22-grid/run/manifest_eval.jsonl`). The real train cache (`D:\tanitad\cache_full`, 4,572 clips, ~95.6 GB) is on box 2 and was not touched; I found no second copy on the dev box or E: and no record of one (UNKNOWN whether it is banked on HF) |
| DINOv3-L extraction | **5.01 s/clip** box 2 SAFE MODE (423 clips; 4.98 on 64); 4.20 mean / 3.62 median / 9.2 max on the 141-clip eval manifest; dev-box 4060 4.86 median, 5.21 mean, contended (38 clips; the incident file says do not quote it as a rate) | MEASURED: `box2_ship/logs/encode_A2s.log`, `encode_B2s.log`, `manifest_eval.jsonl` `enc_s`, `raw/INCIDENT_gpu_touch.md` |
| v2ep build | **8.23 s/clip** dev box, 6 workers (600 clips in 82.3 min; eval 5.62); box 2 all-in (build+encode+verify+delete) **16.66 s/clip** (3,972 clips, 10-18 12:14 -> 10-19 06:37) so the box-2 build is ~11.7 s | MEASURED: `raw/s4c/build_train600.log`, `build_eval141.log`, `box2_ship/logs/chain_ra1_full.out`. The prereg ESTIMATED 22 h; it took 18.4 h |
| labels | v9 **0.228 s/clip** (997.4 s / 4,369), S2 emit 4.5 ms; obstacle pull 0.25 s + join 0.41 s; lead pass UNMEASURED (guess 0.5 s) | MEASURED: `qland/work/refcv8_hf_v9/release/v9_labels_train.manifest.json` (`wall_s`), GOALS `D-TACGT-COVERAGE-IS-A-WRITER-LIMIT`, b1-train-join README (18.5 min, 30.1 min) |
| LAN, dev -> box 2 | 14.64 MB/s | MEASURED, 51,480,904,147 B in 3,517 s (`transfer_check.json`, `ship_trainall.log`) |
| WAN, HF -> dev | **UNKNOWN for large reads.** 1.83 MB/s (B1 camera bank: 61.29 GB written over 9.31 h by file mtime, ESTIMATED), 2.41 MB/s (obstacle range pull, request-bound, MEASURED), >= 4.8 MB/s (a ~2 GB tar staged inside 420 s, ESTIMATED); 24.6 MB/s was Thor's, 2026-08-03 (INHERITED) | the 20-clip pilot replaces all three |
| M2b training | 1.217 steps/s -> 19,164 steps = **4.37 h/seed** (fixed steps); 8.75 h if S doubles with the clips | MEASURED, `RESULT_V8F_GRID.md` (B's rate) |

## 3. Feasibility (the full table is `d3_cost_model.out`)

| | **+4,512 clips (2x)** | **+40,000 clips** |
|---|---|---|
| download | 65.1 GB (B1 sizes) / 102.8 GB (corpus-mean), of which 2.0 GB is range overhead over **2,408** chunks (uniform draw, MEASURED) | 562 GB / 896 GB, over all 3,146 chunks. A whole-chunk pull would be 6.45 TB (2.05 GB x 3,146): the range recipe is mandatory |
| WAN time at 5.0 / 2.41 / 1.83 MB/s | 3.6 / 7.5 / 9.9 h (B1 sizes); 5.7 / 11.8 / 15.6 h (corpus-mean) | 31 / 65 / 85 h; 50 / 103 / 136 h |
| cache added | **+19.7 GB** (4x10 only); +94.3 GB (both grids) | +174 GB; +836 GB |
| v2ep transient | 20-23 GB per 400-clip chunk (167 GB if never deleted) | same per chunk (1.48 TB if never deleted) |
| fits? | dev C: 64.6 GB free: one chunk at a time, nothing else growing. F: 787 GB: yes. E: 2.1 GB: no. **Box 2 `D:` 85.9 GB (10-19, INHERITED): the 4x10-only cache fits, both grids do not** | box 2: no (174 GB > 85.9). Dev F: could hold it, but the trainer is input-bound (64 windows x ~0.8 MB per step) and cannot read it over the 14.6 MB/s LAN |
| V1 box 2 builds + encodes | **20.9 box-2 h** (dev box: download, stage, ship) | 185 h |
| V2 dev box builds + encodes (4060) | 16.4 h serial / 10.3 h if CPU and GPU overlap; box-2 hours 0, ship the 19.7 GB cache in 0.37 h | 145 / 91 h |
| **V3 dev builds v2ep, box 2 encodes only** | **6.3 box-2 h**, dev CPU 10.3 h, ships 167 GB of v2ep in 3.2 h, overlapped | 55.7 box-2 h, dev CPU 91 h |
| labels (v9 + S2) / lead block | 0.29 h / 1.45 h (lead pass guessed) | 2.6 h / 12.9 h |

* **Extraction, box 2 versus dev box.** Box 2: one job at a time, CPU and GPU must not overlap (the 2026-10-17 blue-screen hypothesis), SAFE MODE; the RA-1 FULL build ran 18.4 h clean at 4 workers. Dev box: i9-12900F 16C/24T, 31.8 GB RAM (8.3 GB free now; the S4b note says keep <= 6 workers), RTX 4060 8 GB with 2.3 GiB already used by another session, and a shared `devbox_gpu.lock`. The scarce resource is box-2 hours, which is why V3 (6.3 h) beats V1 (20.9 h) while the dev CPU, which is idle anyway, builds.
* **Pipelining.** In 400-clip chunks (the `ra1_full_box2.py` pattern: state file, resumable, delete only after verify), wall = max(download, dev build, box-2 encode) + tail = ~11-17 h at +4,512; serial it is 20-32 h.

## 4. Admissibility traps

1. **RA-1 holdout (60 clips, sha256 `f6a1dee0...`).** The probe reproduces the pin and the 4,512-clip train digest `37f4f9ee...` from `raw/ra1_full/clips_all.txt`, then merges candidates: **+4,512 -> 0/60 still excluded, 60/60 trained on; +40,000 -> the same.** The explicit-id control keeps 0/60 in train. **Fix: train and holdout by explicit lists pinned by sha256, asserted inside `declared_vs_built`, never by position.** `select_train_clips` (`v8f_mini_data.py:114`, nested by sha12) has the same property, so B's 4,512 and the new 4,512 must be separate named lists, not `--data-frac` prefixes of the union.
2. **Eval-141 episodes (v7.2 eval, 147 digests).** MEASURED: 0 of the 301,433 pool clips are in it; all 147 are inside B1. A pool of "everything except B1's 4,719" excludes them by construction. Keep the digest check anyway, as a belt.
3. **Deployed val40.** MEASURED: **34 pool clips are in it** (6 more are inside B1, already dropped by the 4,713 rule). Expected hits in a uniform draw: 0.51 at 4,512, 4.51 at 40,000. `pull_encode_pooled.parity_gate` uses `mode="refuse"` (aborts the whole run), `pod_build_b1_epcache` uses `exclude`: filter the list BEFORE any byte is pulled (the gate's own advice).
4. **Parity corpus (`physicalai-train-e438721ae894`, 2,376 episodes, skip-hash `f09e44db`).** D3 does not select, re-select or touch any of its episodes, so the parity rule is not breached. MEASURED: **2,199 pool clips are in the parity-train digest set** (33 expected at 4,512; role=`train` does not treat them as a hazard). All 34 val40 and all 2,199 parity-train clips lie in NVIDIA's `train` split; the `val` (89,688) and `test` (60,906) splits have 0 of either, while B1 itself ignores the NVIDIA split (2,786 / 1,240 / 693). That gives the PI a zero-cost, zero-overlap option, at the price of a B1-inconsistent rule.
5. **D3 is a new corpus id, not B1.** D-CORPUS-B1 already made B1 "a NEW domain"; corpus identity = release manifest hash + blob lineage. Name it from the union list: `b1x2-<sha8 of the 9,024-clip list>`, with the v9 release md5, the cache-manifest sha256 and the native-tracks sha256 recorded beside it. Arms stay comparable by **nesting and one lever**: D3's train set contains B's exact 4,512 (list sha `37f4f9ee...`), and everything else is B's: eval-141 (2,399 windows), holdout `f6a1dee0...`, the frozen **P10-yaw-4512** base (do not refit it: `--p10` is a derived base), 19,164 steps (`--steps`; `--steps-mult 4` would double S on a 2x manifest, `v8f_mini_train.py:423`), `--lru` and the sampler chunk (`:344`; `lru` is derived from `J["n"].max()`, the GRID F1 trap), and `cons_stats` pinned to B's join. Banked M2b (seeds 0 and 1, F = 0.1207 m) is the control.
6. **Cache-root rule.** The trainer trains on every clip in its cache manifest (`HANDOFF_RA1_FULL.md`, `PREREG_RA1_FULL.md`): never write new clips into `cache_full`. A combined root would be a 95.6 GB copy that box 2 cannot hold, so the trainer needs a second root or an explicit clip-list flag (new `FLAG_CHECKS` entry, else the launch gate refuses it).
7. **Traffic-light and VLM goals** are B1-bound (section 1 notes). Report them as unchanged by construction, not as a D3 result.
8. **Pixel route.** An earlier route change moved pooled features by ~2.5 % rel-L2 (S4b). The v2ep builder gave byte-identical files on the dev box and box 2 (C0 control, 2026-10-18), so any machine is fine, but the new pipeline must pass the same known-value control (rebuild 2 B1 clips; sha equal) before it touches a new clip.

## 5. The smallest runnable D3 plan (ESTIMATED wall; classes as in section 2)

**Starts now, no PI input, no download, no GPU:**
* **A. PREREG_V8F_D3** (the rung, the lists, steps = 19,164, frozen P10 and `cons_stats`, TL/VLM statement, the reading). Dev. ~0.5 day of Master Mind time.
* **B. `d3_select.py`**: draw (seed pinned) from the pool, apply the gates in section 4, write `clips_d3_new.txt` + B's list + holdout list with sha256s. The two probes are its prototype. Seconds.
* **C. Code:** (i) encoder: `pull_encode_pooled` with a declared extra-clip list that keeps the B1 assertions for B and gates the new list separately (or a copy: it must not weaken the original); (ii) trainer: explicit train and holdout lists (sha-asserted in `declared_vs_built`), a second cache root, `cons_stats` pin, FLAG_CHECKS entries; (iii) D3 driver derived from `ra1_full_box2.py`, keeping v2ep poses before deleting. Each with a reachability proof from the real caller, a non-zero control, a mutation that turns it RED, and a launch-gate token. Dev CPU.
* **D. Label rehearsal on clips already on disk:** run steps 8-11 on 50 B1 ego logs with their CoT records withheld; assert `has_cot = 0`, VLM `goal_w = 0`, and that the 4,512 B rows of a rebuilt union release are bit-identical to `83e30bd3` (the same identity control the 4,369 -> 4,572 superset passed). ~35 min for 9,084 clips.

**Needs approval:**
1. **Pilot (PI: ~0.4 GB):** range-pull 20 new clips + 2 B1 clips as a known-value control; measure MB/s, request rate, mp4 sizes. Dev, ~10 min. Converts the WAN row from UNKNOWN to MEASURED.
2. **Pull 4,512 clips** in 400-clip chunks (PI: 65-103 GB, 3.6-15.6 h; a resumable detached script, sha ledger, writes to the PI-approved scratch). Also ego, obstacle, intrinsics.
3. **Build v2ep** on the dev CPU, 6 workers, 8.23 s/clip = **10.3 h**, overlapped with step 2 chunk by chunk. Keep the poses; delete the mp4 and v2ep only after verify. C0 control first.
4. **Encode on box 2** (PI: 6.3 h of box-2 time, one job, SAFE MODE), `--grids 4x10`, into a new root `cache_d3` (+19.7 GB on box 2 `D:`), via the extended encoder. Ship each v2ep chunk over the LAN (14.8 GB, ~17 min).
5. **Labels and joins, dev CPU, ~1.5 h:** S2 emit -> native tracks -> clock sidecar -> speed sidecar -> lead block -> v9 on the union (identity control) -> `join_train.npz`.
6. **Launch gate, then 2 x M2b** on box 2 (PI: 8.75 h) at fixed 19,164 steps, seeds 0 and 1, the PRIMARY bar 2F at both seeds as the DSCALE table commits; eval-141 unchanged.

Totals, ESTIMATED: first cache ~11-17 h pipelined after approval; box 2 = 6.3 h + 8.75 h = **~15 h** (V3), or 20.9 + 8.75 = ~30 h (V1, no cross-machine v2ep ship).

## 6. UNKNOWN or not done

* WAN rate for large range reads; the lead-pass and clock-sidecar seconds; whether the NVIDIA gated token still works (not read, not tested).
* Box 2's free space now (ssh forbidden); box 2's filesystem for hard links (decides a combined root).
* The corpus-wide mp4 mean (13.06 MB on B1 vs ~21.4 MB implied by the chunk size): the pilot answers it.
* The no-CoT label path was read and the goal-bit census was run on the builder's tables, but no real new clip has gone through it (step D).
* The pooled train cache appears to exist on one disk (box 2 `D:`), 95.6 GB, rebuildable in ~18.4 h of box 2. Worth a banking decision before D3 adds to that root's neighbourhood.

## Manifest (all in the export, none staged anywhere else)

`PK/raw/2026-10-25-d3-scoping/`: `D3_SCOPING.md` (this file), `probe_holdout_shift.py/.out`, `probe_pool_gates.py/.out`, `d3_cost_model.py/.out`. Scratch only: a 4x10-only re-save of one cache clip was written to the session scratchpad and deleted.
