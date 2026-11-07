# SCHEMA - `features.parquet` (one row per navtest token)

Code: `code/mine_features.py` (pinned in `CODE_PIN.json`). Evidence class of every column: **MEASURED from the OpenScene metadata pkl** (ground-truth annotations, i.e. privileged, offline label data - fine for a scenario table, never an inference input).
Tier stamp: these are **scene descriptors, not model results** (no T0/T1 number involved).

## Conventions (apply to every column)

* **Token / window.** NAVSIM `navtest.yaml` (`num_history_frames: 4`, `num_future_frames: 10`, `frame_interval: 1`, `has_route: true`). Following `navsim/common/dataloader.py:22-57`, windows are `frames[i : i+14]` and the token is `frames[i+3]["token"]`; a token is dropped if `roadblock_ids` of that frame is empty (line 49). `t0` = that frame `j = i+3`.
* **Feature horizon.** History = frames `j-3 .. j` (4 frames incl. t0, 1.5 s). Future = frames `j+1 .. j+8` (**8 frames = 4.0 s at 2 Hz**, the NAVSIM trajectory/PDMS horizon). The loader also needs frames `j+9, j+10` to exist (yaml asks for 10 future frames) but no feature reads them. "win" = frames `j .. j+8` (9 frames).
* **Ego frame.** x forward, y left, origin = rear axle (the frame `gt_boxes` are stored in; `lidar2ego` is identity). Agent boxes: `[x, y, z, length, width, height, heading]` (`navsim/common/enums.py:48-58`, `BoundingBoxIndex`). `gt_velocity_3d` is the agent's **absolute** velocity expressed in the ego axes of that frame (checked: static `generic_object` median speed 0.00 m/s; a lead at the ego's speed reads ~ ego speed, not 0).
* **Ego speed** = `hypot(ego_dynamic_state[0], ego_dynamic_state[1])`; `ego_dynamic_state = [vx, vy, ax, ay]` in the ego frame (`navsim/common/dataclasses.py:190-193`: `[:2]` velocity, `[2:]` acceleration).
* **dt** = 0.5 s (frames are ~2 Hz; see `dt_ok`).
* **Ego geometry.** Rear axle to front bumper = **4.049 m** (`nuplan.common.actor_state.vehicle_parameters.get_pacifica_parameters().front_length`, MEASURED by import 2026-11-07). A lead's **gap** is bumper to bumper: `gap = (x_center - hx) - 4.049`, `hx = 0.5*(L*|cos h| + W*|sin h|)` (half extent along ego x of the oriented box).
* **Corridor.** `|y_center| < 1.8 m` and `0 < x_center < 60 m` in the ego frame of *that* frame (fixed straight strip, NOT the lane and NOT the planned path). Only class `vehicle` forms a lead.
* **NaN** = "undefined" (no lead, no closing speed, ego too slow for a time gap, ...). Booleans are never NaN.

## Identity / meta

| column | type | definition |
|---|---|---|
| `token` | str | NAVSIM scene token (= `frames[j]["token"]`), primary key |
| `log_name` | str | pkl log name |
| `map_location` | str | `frames[j]["map_location"]` (4 cities) |
| `scene_token` | str | `frames[j]["scene_token"]` |
| `frame_idx` | int | the pkl field `frame_idx` = index of the frame inside its *scene* (restarts at 0 for every scene of a log; NOT a list index) |
| `list_idx` | int | index j of t0 in the log pkl's frame list (use this to locate the frame: `frames[list_idx]["token"] == token`) |
| `timestamp_t0_us` | int | `frames[j]["timestamp"]` (microseconds) |
| `dt_ok` | bool | all 11 consecutive timestamp gaps of frames j-3..j+8 are within 0.5 +- 0.05 s |
| `cmd0..cmd3` | int | raw `driving_command` 4-vector at t0 |
| `cmd_idx` | int | argmax of the vector if it is one-hot, else -1. Decoded as **0 = left, 1 = straight, 2 = right, 3 = unknown**; this order is the NAVSIM convention and is **checked empirically in SUMMARY.md section 5** (mean realised yaw change per index), not read from devkit source (the devkit only passes the vector through, `dataclasses.py:138,193`) |
| `n_roadblocks` | int | `len(roadblock_ids)` at t0 (route length; >0 by the `has_route` filter) |

## Ego

| column | type | definition |
|---|---|---|
| `ego_v0` | float m/s | speed at t0 |
| `ego_v_min_fut`, `ego_v_max_fut`, `ego_v_end` | float m/s | min / max / last of speed over future frames j+1..j+8 |
| `ego_v_hist_min`, `ego_v_hist_max` | float m/s | min / max speed over history j-3..j |
| `ego_ax_min_mps2`, `ego_ax_max_mps2` | float | min / max of `ego_dynamic_state[2]` (longitudinal accel, +forward) over win j..j+8. Source-reported, noisier than the speed difference |
| `ego_decel_max_ax_mps2`, `ego_accel_max_ax_mps2` | float >= 0 | `max(0, -ax_min)` and `max(0, ax_max)` |
| `ego_decel_max_fd_mps2`, `ego_accel_max_fd_mps2` | float >= 0 | finite difference of speed over the 8 steps of win (0.5 s): `max(0, -min dv/dt)` and `max(0, max dv/dt)`. **Primary** braking/accel measure (the ax series can differ from the speed series) |
| `ego_stopped_t0` | bool | `ego_v0 < 0.5` |
| `ego_stops_in_future` | bool | `ego_v0 >= 0.5` and `ego_v_min_fut < 0.5` (comes to a stop within 4 s) |
| `ego_stationary_window` | bool | speed < 0.5 m/s for all frames j-3..j+8 |
| `ego_starts_from_stop` | bool | `ego_v0 < 0.5` and `ego_v_max_fut > 1.5` |
| `ego_started_in_hist` | bool | `ego_v_hist_min < 0.5` and `ego_v0 >= 0.5` (was stopped in the history and is moving at t0) |
| `ego_dist_fut_m` | float | path length of the rear-axle positions j..j+8 |
| `ego_dyaw_fut_rad` | float | wrapped yaw(j+8) - yaw(j); yaw from `ego2global` (`atan2(R10, R00)`); + = left |
| `ego_x_end_m`, `ego_y_end_m` | float | position at j+8 in the t0 ego frame |

## Lead vehicle

Lead at frame k = the nearest `vehicle` in the corridor with `gap > 0`, **excluding oncoming** (agent velocity along ego x < -1.0 m/s; such agents set `oncoming_corridor_t0` instead). Stationary vehicles in the corridor DO count (queues, parked cars intruding on the strip); `lead_stationary` tells them apart.

| column | type | definition |
|---|---|---|
| `lead_t0` | bool | a lead exists at t0 |
| `lead_trk_t0` | str | its `track_token` ("" if none) |
| `lead_gap_m` | float | bumper gap at t0 (see Conventions) |
| `lead_speed_mps` | float | `hypot(vx, vy)` of the lead's velocity |
| `lead_vx_ego_mps` | float | lead's velocity component along ego x |
| `lead_closing_mps` | float | `ego_vx - lead_vx_ego` at t0 (>0 = closing) |
| `lead_time_gap_s` | float | `gap / ego_v0` if `ego_v0 > 0.5` else NaN |
| `lead_ttc_s` | float | `gap / closing` if `closing > 0.1` else NaN |
| `lead_stationary` | bool | lead exists and speed < 0.5 m/s |
| `lead_present_frac_win` | float | share of the 9 win frames that have a lead (each judged in its own ego frame) |
| `lead_present_all_win` | bool | a lead in all 9 win frames ("a lead exists over the whole window") |
| `lead_same_track_all_win` | bool | the **same track** is the nearest lead in all 9 win frames |
| `lead_appears_win` | bool | no lead at t0 but a lead in some future frame |
| `lead_gap_min_win_m`, `lead_ttc_min_win_s`, `lead_time_gap_min_win_s` | float | minimum over the win frames that have a lead (and a defined value) of gap / ttc / time gap, each in its own frame |
| `lead_track_found_frac` | float | share of win frames in which the t0 lead's track still exists in the annotations (any position) |
| `lead_speed_min_fut` | float m/s | min speed of the t0 lead track over j+1..j+8 |
| `lead_speed_drop_fut` | float m/s | `lead speed(t0) - min over future` (>0 = it slowed) |
| `lead_decel_max_mps2` | float >= 0 | max over the 8 steps of `(speed_k - speed_k+1)/0.5` for the t0 lead track (needs both frames) |
| `lead_stops_fut` | bool | t0 lead had speed >= 1.0 and falls below 0.5 within the future |
| `oncoming_corridor_t0` | bool | an oncoming vehicle (vx < -1.0) sits in the corridor with gap > 0 at t0 |
| `vru_corridor_t0`, `vru_corridor_win` | bool | a `pedestrian` or `bicycle` with `|y| < 1.8` and `0 < x < 30 m` at t0 / in any win frame |

## Cut-in

Candidate = a `vehicle` at t0 with `0 < x < 60` and `1.8 <= |y| <= 8` (beside the corridor, ahead).

| column | type | definition |
|---|---|---|
| `cutin` | bool | a candidate is, in some future frame k (j+1..j+8) and **in frame k's own ego frame**, inside the corridor with `gap > 0`, speed >= 1.0 m/s and not oncoming. **Caveat:** this also fires when the *ego* moves laterally toward a vehicle (lane change, turn) |
| `cutin_entry_step` | int | k - j of the first entry (1..8), -1 if none (when several candidates, the one with earliest step then smallest gap) |
| `cutin_gap_m`, `cutin_closing_mps`, `cutin_ttc_s`, `cutin_agent_speed_mps` | float | gap, closing speed (`ego_vx - agent_vx_ego`, frame k), TTC (closing > 0.1 else NaN) and speed of that agent at entry |
| `cutin_path` | bool | stricter, **path-based**: candidate's centre is >= 1.8 m from the ego's realised path at t0 and < 1.8 m from it (path = rear-axle positions j..j+8 as a polyline in the t0 frame; agent position mapped to the t0 frame by the ego poses) in some future frame while speed >= 1.0 and ahead of the ego (x > 0 in frame k). Insensitive to ego lateral motion, but the path is the *human's realised* path |
| `cutin_path_entry_step` | int | k - j, -1 if none |

## Traffic lights

`frames[j]["traffic_lights"]` = list of `(lane_connector_id, bool)`. **The bool is `is_red`**: `navsim/planning/scenario_builder/navsim_scenario.py:282-283` (`for lane_connector_id, is_red in ...traffic_lights: status = RED if is_red else GREEN`); PDM only blocks on RED and only for connectors in the route (`pdm_observation.py:299`). So `False` = green **or yellow or unknown** (no 3-state info exists in the pkl). The id is the nuplan map `lane_connectors` fid (checked: all 124 distinct ids seen in every 20th frame of the reference log are fids of that layer).

| column | type | definition |
|---|---|---|
| `tl_n` | int | list length at t0 (all controlled connectors the log reports around the ego, **not** only those on the ego's route) |
| `tl_present` | bool | `tl_n > 0` |
| `tl_n_red` | int | entries with bool True |
| `tl_any_red` | bool | `tl_n_red > 0`. **Weak signal** - the list spans every light-controlled connector in the neighbourhood, including cross traffic, so this is True at most signalised junctions regardless of the ego's own light |
| `tl_ctrl_ahead` | bool | map-based: some listed connector has its **entry point** (first point of the baseline) at `0 < x < 60 m`, `|y| < 3.5 m` in the t0 ego frame and entry heading within 35 deg of the ego heading |
| `tl_red_ahead`, `tl_green_ahead` | bool | the same, restricted to bool True / False. Approximates "the signal controlling my approach"; adjacent-lane connectors within 3.5 m can also match |
| `tl_red_ahead_dist_m` | float | min entry x over the red matches, NaN if none |
| `tl_ctrl_on_path`, `tl_red_on_path` | bool | the ego's realised rear-axle position at any win frame lies inside the polygon of a listed (resp. red) connector. Only fires when the ego actually *enters* the junction on that connector |

## Junction (map-based)

Geometry: nuplan maps v1.0 (`F:/Archive/devbox-C/navsim/data/maps/<city>/<ver>/map.gpkg`), layers `intersections` and `gen_lane_connectors_scaled_width_polygons`, reprojected to the map's UTM system (`meta` key `projectedCoordSystem`), as the nuplan `GPKGMapsDB` does. **Junction = inside an `intersections` polygon OR inside any lane-connector polygon.** Rear-axle point is used (not the footprint).

| column | type | definition |
|---|---|---|
| `map_ok` | bool | the city's map loaded; if False all map-based columns are False/NaN (reason in `run_report.json`) |
| `junction_t0` | bool | rear axle at t0 is in a junction |
| `junction_in_win` | bool | any win frame's rear axle is in a junction |
| `junction_dist_ahead_m` | float | smallest d in {0, 5, ..., 60} such that the point d m along the t0 heading (straight line, not the route) is in a junction; NaN if none within 60 m |

## Not provided (and why)

* **Route-aware lane/junction topology of the ego's lane** (which connector the planner *should* take): the pkl has `roadblock_ids` only; the entry-point matching above is a geometric proxy.
* **Yellow light state**: not in the pkl (bool only).
* **Selection / category labels**: deliberately absent - the frozen rule decides.

## Caveats measured on this table (details in SUMMARY.md)

* `cutin` (ego-frame) is liberal: 9.1 % of navtest tokens, but only 0.4 % under `cutin_path`; it also fires on crossing traffic and on ego-induced relative motion. Hand check: the independent recomputation reproduces entry step and gap for 4 tokens, but no ground-truth cut-in label exists to validate either flag.
* The junction polygon set includes the *scaled-width* lane-connector polygons (`gen_lane_connectors_scaled_width_polygons`), slightly wider than the lane; junction flags are True a little before the stop line.
* `lead_*` is a straight 3.6 m wide strip, not a lane: in a curve the nearest vehicle in the strip can be in the neighbouring lane, and a lead flickers in/out near |y| = 1.8 m (38 of the 41 "other" cruise-bundle tokens have an intermittent lead).
* 51 of 12,146 tokens have `dt_ok == False`; their finite-difference columns are less reliable.
