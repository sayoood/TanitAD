# Agent check (D-VDAC-6): feasibility inventory (REFe Research Lab teammate, 2026-11-14; zero GPU, no pod, no box 2, no new navtest truth read)

HOLD = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/agent_check_feas/`. Scripts and raw outputs: `scripts/`, `*.json` beside this file.
PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Tip read through the mirror `C:/Users/Admin/tanitad-push` (`origin/agent/arch-inf-20260803`), nothing checked out.
Evidence classes: MEASURED (artifact named) / PUBLISHED / INHERITED (not re-verified) / ESTIMATED (arithmetic shown) / HYPOTHESIS / UNVERIFIED.
Nothing here read navtest per-proposal truth. One navtest cache token was opened for TIMING and STRUCTURE only (scores neither printed nor stored; section 2.3).

## 0. Headline (five lines)

1. **The agent annotations exist, for navtrain and navtest, in three places, and the NAVSIM metric cache already holds the logged agent futures** the PDM scorer scores against (51 occupancy maps at 10 Hz over 5 s, 215 objects on the sample token). The "item 9" gap in `FEASIBILITY.md` is closed: no new annotation source is needed.
2. **The ceiling plan needs NO forward pass and NO pod.** `F:/tanitad_scratch/p6fit/fit_pack.npz` (sha256 `e58653c5...`, identical to the pod's `fit_pack.npz`, `pod_sha.txt`) holds P6's own 64 proposals, logits and per-proposal NAVSIM sub-scores (`nc dac ep ttc c ddc pdms`) for **8,489 navtrain sets (1,705 held-out + 6,784 fit-extra, 101 logs)**.
3. **Oracle ceiling, MEASURED on those sets:** a TRUE agent veto (NC>0 and TTC>0) lifts P6's pick **94.49 -> +1.215 PDMS**; TTC carries it (**+1.181**), NC adds only **+0.219**; TRUE DAC veto +0.586; all three **+1.833**. Held-out 1,705 sets alone: +1.666 agent, +2.352 agent+DAC.
4. **P6's own NC and TTC heads already rank well (pooled AUC 0.973 / 0.952) and fail CONFIDENTLY**: of the 2.58 % of picks with an agent failure, **98.6 % have P6's own p_NC x p_TTC above 0.5**. A check that is no better than P6's heads cannot move the pick; the synthetic conversion curve (section 4.3) says a soft entry is negative below ~0.9 effective AUC and a hard veto needs about 0.8 to pay at all.
5. **Binding risk:** P6 and any head on its `visual_ctx` see ONE time step (no history field in `config.json` meta or `model.py`), so future agent motion is not observable in the input. The first CPU gate (S0-B in the card) measures how much of the ceiling survives when agents are frozen at t0.

## 1. Inventory of per-token agent annotations (deliverable 1)

Every absence claim below was probed at two locations. "Token" = NAVSIM token = nuPlan `lidar_pc.token` (hex): MEASURED, 5 of 5 sampled fit-pack tokens found as `lidar_pc.token` in their log's DB (`scripts/tokcheck3.py`).

| # | source | where (this box) | what | navtrain | navtest |
|---|---|---|---|---|---|
| A | **nuPlan DB `lidar_box` + `track` + `category` + `ego_pose`** | `E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/` (325 DBs at top level + `trainval/` 1,381 DBs + `mini/` + `test/`; 349 GB), fully written since Sep 2026. `C:/Users/Admin/rj5db` and `F:/tanitad_scratch/rj5nt/dbB` are LIVE download staging (files appear and disappear within minutes) and were NOT used. | schema section 1.1 | **48 of the 101 fit-pack logs (3,976 of 8,489 sets; held-out 1,361 of 1,705) are on this box today** (`scripts/tokcheck2.py`). The other 53 logs are on the pod / download chains. | `splits/test/` (not queried; the navtest cache covers it) |
| B | **NAVSIM `metric_cache` pickles (v1.1)** | `F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest/` **12,146 `metric_cache.pkl`** (MEASURED count). Also v2 caches `C:/Users/Admin/navsim-crun/exp/metric_cache_navtest_v2*` (carry `future_tracked_objects` 40 steps and `current_tracked_objects`). Navhard / warmup caches under `F:/Archive/devbox-C/navsim/exp/`. | schema section 1.2 | **none on this box** (two probes: `find` for `metric_cache*` dirs under C:/Users/Admin, F:/, E:/Projects/TanitAD-artifacts; `find` for `*.pkl.xz`, `p6parts`, `metric_cache_navtrain` under E:/Projects, F:/tanitad_scratch, F:/TanitAD_p6read_work: 0 hits). The navtrain **parts** (`tanitad.p6parts/1`, `<log>/<token>.pkl.xz`, observation included) are built by `PK/refe/build_metric_cache.py` on the pod. | 12,146 tokens, 136 logs |
| C | **OpenScene / NAVSIM logs `navsim_logs`** (`anns`: `gt_boxes` ego frame, `gt_names`, `gt_velocity_3d`, `instance_tokens`, `track_tokens` per 2 Hz frame) | `F:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test/` 147 pickles (~1 GB), `_corrupt_20261106/` 4 files | section 1.3 | **none** (`navsim_logs/` holds only `test` and `_corrupt_20261106`; a second probe for a `trainval` dir found only the nuPlan one) | yes |
| D | the P6 label bank `p6base_labels` | pod `/workspace/data/p6base_labels` (`PK`-relative manifest `F:/tanitad_scratch/p6ship/labels_manifest.json`) | per-proposal PDM sub-scores of P6's own proposals, NOT agents | 60,000 train + 6,784 fit-extra + 1,705 held-out sets | none |
| E | the `fit_pack.npz` dev-box copy of D's fit part | `F:/tanitad_scratch/p6fit/` (`fit_pack.npz` 160.7 MB, `fit_pack8.npz` 82.5 MB) | section 3.1 | **8,489 sets** | none |
| F | the drivable store `aux_drivable` | `E:/Projects/TanitAD-artifacts/aux_drivable` | drivable area only; **no agents** (FEASIBILITY.md section 3, re-read) | | |

### 1.1 Source A, nuPlan DB (MEASURED on `.../splits/trainval/2021.06.07.11.59.52_veh-35_00008_00083.db`, 43.6 MB, opened `mode=ro&immutable=1`)

| item | value |
|---|---|
| `lidar_box` columns | `token, lidar_pc_token, track_token, next_token, prev_token, x, y, z, width, length, height, vx, vy, vz, yaw, confidence` (162,546 rows in this DB) |
| `track` | `token, category_token, width, length, height` (625 tracks) |
| `category` (7) | vehicle, bicycle, pedestrian, traffic_cone, barrier, czone_sign, generic_object. Rows by class here: generic_object 82,440, traffic_cone 40,600, vehicle 22,290, pedestrian 12,296, barrier 4,116, czone_sign 675, bicycle 129 |
| time base | `lidar_pc` 1,501 frames over 75 s = **20.01 Hz**; NAVSIM tokens sit at 2 Hz scenario steps |
| frame | **global map frame** (UTM-like: x 664,740, y 3,999,386 on this log), yaw global, boxes carry velocity `vx vy`; the ego pose comes from `ego_pose` via `lidar_pc.ego_pose_token` (quaternion `qw qx qy qz`, velocity, acceleration) |
| futures | YES: a token's future agents are the `lidar_box` rows of later `lidar_pc` frames of the same `track_token` (`next_token` chain). These are logged ground truth futures: **non-reactive**, which is also what v1.1 PDMS scores against |
| confidence | `confidence` is 1.0 for human-labelled boxes and below 1 (0.56 to 0.85 in the first 2,000 rows read) for auto-labelled ones. **Neither devkit filters on it**: `grep -c confidence` on `nuplan_scenario_queries.py` = 0 in `F:/Archive/devbox-C/navsim/nuplan-devkit` and no match in `C:/Users/Admin/dz/DriveZero/DriveRL/nuplan-devkit/nuplan/database/nuplan_db/` (two locations). So NAVSIM's labels include the auto-labelled boxes: **a label build must not filter on confidence if it is to equal NAVSIM's** |
| query cost | one 8 s window of boxes (19,910 rows): **0.070 s** (one process, warm file) |
| reader code | `nuplan_scenario_queries.get_tracked_objects_for_lidarpc_token_from_db` (DriveRL devkit line 691), `get_future_waypoints_for_agents_from_db` (line 733); `NuPlanScenario.get_tracked_objects_at_iteration` / `get_future_tracked_objects` (`nuplan_scenario.py:260`, `:358`) |

### 1.2 Source B, v1.1 NAVSIM `MetricCache` (MEASURED on one navtest smoke token; lzma-compressed pickle, `navsim/planning/metric_caching/metric_cache.py`)

Fields: `file_path, trajectory (PDM-Closed reference), ego_state, observation, centerline, route_lane_ids, drivable_area_map`. (v2 additionally `log_name, timepoint, scene_type, human_trajectory, past_human_trajectory, past/current/future_tracked_objects, map_parameters`: MEASURED on a v2 cache token, 109 current objects, 40 future steps.)

| item | value (v1.1 token) |
|---|---|
| `observation` | `PDMObservation`: **51 occupancy maps at 10 Hz** (`_observation_samples` 50, i.e. 5 s + the 1 s TTC extension is inside the 51), `_map_radius` **100 m**, `_unique_objects` 215 |
| object types | `Agent:VEHICLE`, `Agent:PEDESTRIAN`, `StaticObject:BARRIER / CZONE_SIGN / GENERIC_OBJECT / TRAFFIC_CONE` (bicycles map into Agent) |
| construction | `MetricCacheProcessor._interpolate_gt_observation` (`metric_cache_processor.py:91-205`): GT tracks sampled at **2 Hz over 5 s** (11 samples), linearly interpolated to 0.1 s; `observation_sample_res=1`. Static objects are held fixed. Non-reactive |
| frame | global map frame, polygons (shapely); ego state from `ego_state.rear_axle` |
| size / load | 1.25 MB raw, 0.02 to 0.04 s decompress, **1.65 s first unpickle** (imports), then cheap |
| scorer reading it | `PDMScorer._calculate_no_at_fault_collision` (`pdm_scorer.py:363`) and `_calculate_ttc` (`:492`: ego bounding box projected forward 1 s at constant speed, tested at steps 0,3,6,9 in the future against the occupancy map; `stopped_speed_threshold` 5e-3; ttc weight 5.0) |

### 1.3 Source C, `navsim_logs` (MEASURED on `test/2021.05.25.14.16.10_veh-35_00083_00485.pkl`, 804 frames)
Per frame: `anns` = `gt_boxes` [n,7] **ego frame** (x y z l w h yaw), `gt_names` (the same 7 classes), `gt_velocity_3d` [n,3], `instance_tokens`, `track_tokens`; also `token`, `timestamp` (frame spacing 0.500 s), `ego2global_*`, `can_bus`, `driving_command`, `ego_dynamic_state`, `occ_gt_final_path` / `flow_gt_final_path` (paths only, not read). 59 boxes on frame 0. Futures = the next 8 frames' boxes of the same `track_token` (0.5 s steps, 4 s). Coverage on this box: **navtest only**; navtrain `navsim_logs` would have to be downloaded (UNVERIFIED size, not attempted) and add nothing over source A.

### 1.4 Coordinate frames and alignment, for the label build
Global (A, B) -> ego rear-axle frame at t0 with `ego_state.rear_axle` (x, y, heading), the same frame as the DAC store (`x in [-8,72) m, y in [-32,32) m`, 0.5 m cells, 160 x 128; FEASIBILITY.md section 3). Time: t0 is the token's `lidar_pc` timestamp; future steps t0 + 0.1 k s. Proposal poses are NAVSIM's 8 poses at 0.5 s (`planner.executed_plans8`); the fit pack's `traj_xy [8489,64,20,2]` and `yaw [8489,64,20]` are P6's 20-pose 5 Hz originals (REFe layout).

## 2. Label design and CPU cost (deliverable 2)

### 2.1 Two labels, one is already banked
| label | definition | source | status |
|---|---|---|---|
| **L-PROP** (per proposal) | NAVSIM `no_at_fault_collisions` (0 / 0.5 / 1) and `time_to_collision_within_bound` (0 / 1) of each of P6's 64 proposals, `score_plans(parts, executed_plans8(props))` (`PK/refe/navsim_pdm_targets.py:99`), which calls NAVSIM's own `PDMScorer` | `fit_pack.npz` `pdm[..., 0]`, `[..., 3]` | **BANKED for 8,489 navtrain sets (MEASURED, `ceiling_fitpack.json`)**; not available for the 60,000 train sets on this box (pod only) |
| **L-OCC** (per token, perception) | future agent occupancy raster **[8 steps (0.5..4.0 s), 160, 128]** at 0.5 m in the DAC store frame, with a channel split dynamic (vehicle, pedestrian, bicycle) / static (barrier, cone, czone_sign, generic_object), because NC scores static contact 0.5 and dynamic at-fault contact 0 (semantics in the next paragraph). Optional later: per-cell velocity | source A (or the pod's parts, source B format) | **NOT BUILT**; build described in 2.2 |

NAVSIM semantics the label must respect (read from `pdm_scorer.py`): NC is **at-fault** (rear-ended or lateral-with-ego-stopped contacts do not count; static collisions give 0.5); TTC is evaluated from ego's constant-velocity projection over 1 s; both act on the non-reactive logged futures. A raster label alone cannot encode at-fault; so the raster supervises PERCEPTION and the per-proposal sub-score (L-PROP) stays the DECISION label.

### 2.2 Building L-OCC with NAVSIM's own code, and the cheaper path
* **Faithful path (identity control):** run `PT.metric_cache_parts(scenario)` (`navsim_pdm_targets.py:87`; NAVSIM `MetricCacheProcessor._interpolate_gt_observation`) and rasterise `observation[5(k+1)]` polygons. This is the exact observation `score_plans` uses. Cost: parts build **0.80 to 1.97 CPU-s per set (INHERITED, M-LAB, `build_metric_cache.py` docstring; not re-timed here: the scenario builder needs the full nuPlan stack and a DB per log)**.
* **Cheap path:** read `lidar_box` windows straight from the DB (0.070 s per 8 s window, MEASURED), take the 2 Hz samples, interpolate linearly to 0.1 s, transform to the ego frame, rasterise. Rasterising one token's cached agent polygons into [8,160,128] with PIL: **0.034 s** (MEASURED, 102 to 115 polygons in grid per step, 739 to 789 occupied cells per step on the sample token; `scripts/raster_time.py`). Estimated total **0.2 to 0.4 s per token** (ESTIMATED: query 0.07 + interpolation + 0.034 + overhead). That is **0.5 to 1.0 CPU-h for 8,489 sets and 6 to 12 CPU-h for the 103,139 navtrain keys of the aux store (ESTIMATED, single process)**.
* **Identity control (must exist before the cheap path is used):** the cheap raster equals the raster of the parts' `observation` on at least 64 pod tokens, cell for cell (a one-cell tolerance reported, exact where static). This is the analogue of G20 (label alignment) and D9's `verify`.

### 2.3 Time for ONE token (MEASURED)
`scripts/time1.py` on one v1.1 navtest cache token (timing and structure only; synthetic straight-line proposals; no sub-score printed or stored):
`_setup` 4.24 s once; **`score_plans` on 64 proposals 0.522 s and 0.727 s warm (1.738 s cold)**, which agrees with M-LAB's 0.34 to 0.83 s per set (INHERITED, now MEASURED on a second token). The NC and TTC sub-scores come out of the same call; there is no cheaper way to get NAVSIM's NC/TTC than this call. Therefore **recomputing L-PROP for 60,000 train sets costs ~8.7 to 12 CPU-h (ESTIMATED, 0.52 to 0.73 s x 60,000 / 3,600)** and is NOT needed for the ceiling plan (the 8,489 are banked).

## 3. Ceiling plan without a new navtest look (deliverable 3)

### 3.1 What exists (MEASURED)
`fit_pack.npz` keys: `log token step rank ckpt_step traj_xy[8489,64,20,2] yaw[8489,64,20] logits[8489,64,6] pdm[8489,64,7] pdm_names(nc,dac,ep,ttc,c,ddc,pdms) has_pdm is_heldout logit_space`. `ckpt_step` 10,075 for all rows (the label chain's checkpoint step), `rank` 0. `logit_space` = "raw head output, bf16 autocast, as P6 emits it": **the pick rule needs the sigmoid** (`read_scorer1p6.pick_logits`). ⚠️ My first version of the ceiling script skipped it and printed pick PDMS 91.18, which matched P6's navtest 91.196 by COINCIDENCE; the sigmoid version reads **94.49**, and `ceiling_fitpack.py` now carries the correction in a comment. Both runs are in the transcript only; the banked JSON is the corrected run.
Held-out = `is_heldout` 1,705 sets of the 24 SFT held-out logs (label manifest `heldout_logs`); fit-extra = 6,784 sets from 77 further logs "never in the arms' training" (manifest `log_split.rule`). **UNVERIFIED: whether P6 itself trained on the 77 fit-extra logs** (the rule is written for the arms; FEASIBILITY.md section 6 item 3 and A3 disclose that H48 sat inside P6's own training set). Read the held-out column as the clean one.

### 3.2 Oracle ceilings with TRUE agent labels (MEASURED; `ceiling_fitpack.json`; navtrain; veto = drop proposals with the sub-score at 0, fall back to P6's pick if all are dropped; gain in PDMS x 100 over P6's pick)
| set | n | P6 pick | best of 64 | O_NC | O_TTC | O_NC+TTC | O_DAC | O_agent+DAC |
|---|---|---|---|---|---|---|---|---|
| held-out | 1,705 | 94.024 | 98.416 | +0.358 | +1.610 | **+1.666** | +0.636 | +2.352 |
| fit-extra | 6,784 | 94.609 | 98.023 | +0.184 | +1.073 | +1.102 | +0.574 | +1.702 |
| **all** | 8,489 | 94.491 | 98.102 | +0.219 | +1.181 | **+1.215** | +0.586 | +1.833 |

Pick-level failure shares (all): NC 0 **0.31 %**, DAC 0 **0.72 %**, TTC 0 **2.54 %**, PDMS exactly 0 **1.01 %**. A soft entry with the TRUE factors (rule x NC, rule x [0.5 + 0.5 TTC]) gives +0.279 (NC), +1.169 (TTC), +1.207 (both). A NC-clean alternative exists for 96 % of the NC-0 picks.

### 3.3 Navtrain is not navtest: the distribution shift (MEASURED shares, ESTIMATED projection)
P6 picks 94.49 on these navtrain sets and 91.196 on navtest (`PK/raw/2026-11-09-p6-navtest-read/gap_to_human/p6_gap_to_human.json`). Failure shares differ: NC 0 at pick 0.31 % here vs **1.12 %** on navtest; TTC 0 2.54 % vs 4.92 % (TTC-0 at a non-zero pick **3.70 %**, 449 tokens, `F:/tanitad_scratch/selq2read/p6_e3_regret_split.json`, MEASURED: regret split zero-pick selection 2.81 [2.34, 3.30] + proposal 0 + TTC-0 1.58 [1.29, 1.89] + rest 2.32 = 6.70; zero picks by gate NC 136, DAC 248). That file is the banked artifact behind the LOOP_STATE `selq2` note: **MEASURED, not INHERITED**.
ESTIMATED projection of the TRUE-label ceilings to navtest by ratio of pick-failure shares: NC `0.219 x (1.12/0.31) = 0.79` (the FEASIBILITY.md `~0.83` is the same order); TTC `1.181 x (3.70/2.54 .. 4.92/2.54) = 1.72 .. 2.29`; the navtest TTC-0 regret term itself is **1.577** (MEASURED) and is the tighter upper bound for TTC. **So the perfect-label agent ceiling on navtest is about 2.3 to 2.9 PDMS (ESTIMATED, assumes alternatives exist at the navtrain rate), against a 3.355 gap and a P1 bar d > 0.4971.** The ceiling is large; the question is conversion.

### 3.4 What would still have to be computed (and where)
| step | what | where | cost (ESTIMATED unless marked) |
|---|---|---|---|
| S0-B | rasters L-OCC for the fit-pack sets whose DB is local (3,976 sets, 48 logs) + the footprint readout of section 4 with TRUE rasters, vs the banked TRUE NC/TTC | dev box, CPU, one process | 0.2 to 0.4 s per token raster, ~0.3 to 0.5 CPU-h for 3,976 sets; readout vectorised numpy, minutes. The other 4,513 sets wait for the 53 missing DBs (download chains) or run on the pod |
| S0-C (optional) | identity control of the cheap raster against pod parts on >= 64 tokens | pod CPU | minutes |
| S1 | trains an occupancy head on P6's frozen `visual_ctx` | pod GPU (needs images) | ~4.3 L40-h per arm (ESTIMATED: HC-s0's 1 m-cell head, A5 section 10; the extra time channels only widen the output) |
| S2-nt | the head's predictions on the 8,489 fit-pack sets, then the entry's selection gain | pod GPU forward (images are on the pod only) + dev-box CPU | ~0.4 L40-h forward (ESTIMATED, analogous to the 0.44 forward in A5 section 10) |
| navtest read | the one K-ledger step | MM | a keyed read, not run here |

## 4. How a per-proposal agent check enters the pick, and what it must beat

### 4.1 Readout (deterministic, no training)
Given a predicted or true occupancy raster `R[t, y, x]` and a proposal's poses, `q_hit(proposal) = max_t max_(cell in ego footprint at pose(t)) R[t]` (footprint = the nuPlan ego `car_footprint` box of NAVSIM's scorer (dimensions read from `ego_state.car_footprint`, not hard-coded); poses interpolated to 10 Hz, 4 s horizon + 1 s forward projection for TTC with the proposal's own finite-difference speed). Entries: **hard veto** (drop proposals with q_hit > tau, fall back to P6's pick), **soft multiplier** (rule x (1 - q_hit)^g). The ego state a TTC projection needs is the proposal's own speed from its poses: no ego input at inference beyond what P6 already feeds the scorer (FEASIBILITY.md section 4 "tracked footprint" caveat applies: not used).
**The readout, not the perception, may be the binding constraint**: for the DAC, the TRUE raster under a pose-only veto captured 14 % of the oracle (`PK/raw/2026-10-16-auxveto-read/RESULT.md`, via FEASIBILITY.md C2). The same structure (at-fault logic, tracked vs raw poses) applies here; S0-B measures it BEFORE any head is trained.

### 4.2 What the check must beat (MEASURED, `p6_agent_blindspot.json`, 8,489 sets)
| fact | value |
|---|---|
| P6 scorer pooled / within-set AUC for NC>0 | 0.973 / 0.952 (held-out 0.970 / 0.950) |
| TTC>0 | 0.952 / 0.926 (held-out 0.950 / 0.932) |
| DAC>0 (reference) | 0.986 / 0.981 |
| picks with an agent failure | **2.58 %** |
| P6's own p_NC x p_TTC at those failing picks | median **0.951**; **98.6 %** above 0.5, **88.1 %** above 0.8 (confident misses) |
| P6's own AUC for agent-ok among its TOP-5 proposals (411 sets with both classes) | **0.787** |
| a clean proposal exists when the pick fails | 98.2 %; median rank of the first agent-ok proposal in P6's order **3**, 90th percentile **21** |
| failure rate across logs (logs with >= 20 sets) | min 0 %, median 2.1 %, max 14.9 % (concentrated) |
The target is therefore **top-5 discrimination of agent failure**, where P6 reads 0.787. A check with a global AUC of 0.95 is not an improvement over P6; a check whose AUC on P6's top-5 is >= 0.90 would be.

### 4.3 Synthetic conversion curve (ESTIMATED; independent per-proposal noise is OPTIMISTIC because real perception errors are shared across the proposals of one scene; `auc_and_conversion.json`; agent-ok oracle label = NC>0 and TTC>0 with score = label x delta + N(0,1))
| effective AUC of the check | hard veto gain | soft multiply gain |
|---|---|---|
| 0.6 | -0.07 | -1.31 |
| 0.7 | +0.03 | -0.69 |
| 0.8 | +0.38 | -0.39 |
| 0.9 | +0.52 | -0.16 |
| 0.95 | +0.57 | +0.13 |
| 0.99 | +0.92 | +0.45 |
| 1.00 (true label) | +1.22 | +1.18 |
Reading: the hard veto pays from AUC ~0.8, the soft multiplier only above ~0.95 (it adds noise to a ranking that is already 0.95 good). Both are ESTIMATED upper bounds for independent noise. This is the number the S1 arm's predicted rasters are read against.

## 5. UNVERIFIED and named gaps
1. Whether P6 trained on the 77 fit-extra logs (3.1). The held-out column (24 logs) is the clean read.
2. 53 of the 101 fit-pack logs have no DB on this box today (3.4); the download chains may land them.
3. The cheap raster's identity against NAVSIM's `observation` (2.2) is not yet checked.
4. `score_plans` was timed with synthetic proposals on a navtest cache token; a P6-proposal timing was not taken (it would read navtest truth).
5. Whether `lidar_box` rows of the whole navtrain set carry the same confidence mix as the one DB probed (1 DB, 162,546 rows).
6. The navtrain-to-navtest ratios in 3.3 rest on pick-failure shares only (no navtrain navtest paired sets).
7. A single time step of input for P6 is read from the absence of a history field in `config.json` meta (`n_cameras 4`, `cameras` list) and a `grep -i history|frames` over `model.py` (one unrelated comment): MEASURED absence at two locations, but P6's trainer might stack frames outside `model.py`; confirm in `train.py`.
8. How the ego footprint and the tracked states relate to a pose-only readout is not measured (section 4.1, S0-B does it).
