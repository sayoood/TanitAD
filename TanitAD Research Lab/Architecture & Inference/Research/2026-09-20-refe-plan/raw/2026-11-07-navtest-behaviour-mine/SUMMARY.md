# SUMMARY - navtest per-token behaviour features (DESCRIPTIVE ONLY)

> **Everything below is descriptive.** Thresholds are EXAMPLES chosen for orientation. The frozen selection rule (stream BEHAV, `BEHAVIOUR_SCORECARD.md`) decides which tokens form the suite. No token has been picked here.

## 1. Token accounting (pinned)

* navtest.yaml: **12146 tokens** (12146 unique), **136 log_names**; token-list sha256 `8d42fef68542f095d0e03894fdaf38babfdc89f3fd7cdc26080c3a6009cb4108`.
* **n tokens read = 12146**; n skipped = 0 (not found in any window: 0; failed `has_route` filter: 0; duplicates dropped: 0).
* logs processed 136, status {'ok': 136}; **logs that failed to unpickle / extract: 0** [].
* The pkl directory holds 147 files; 11 are not in navtest's 136 `log_names` and were NOT read (they belong to the OpenScene test split but not to navtest).
* maps unavailable for: none (all 4 cities loaded).
* `dt_ok` (all 12 frame gaps within 0.5 +- 0.05 s): 12095 / 12146.

## 2. Counts per example category (n = 12146)

| category (EXAMPLE threshold) | n | share |
|---|---:|---:|
| ego_brake_fd>=2.0 | 144 |   1.2 % |
| ego_brake_fd>=3.0 | 5 |   0.0 % |
| ego_brake_ax>=2.0 | 196 |   1.6 % |
| ego_accel_fd>=2.0 | 239 |   2.0 % |
| ego_stops_in_future | 1416 |  11.7 % |
| ego_starts_from_stop | 818 |   6.7 % |
| ego_stationary_window | 2 |   0.0 % |
| ego_started_in_hist | 815 |   6.7 % |
| lead_t0 | 5821 |  47.9 % |
| lead_time_gap_t0<2.0s | 756 |   6.2 % |
| lead_ttc_t0<=5s | 2628 |  21.6 % |
| lead_ttc_min_win<=3s | 2538 |  20.9 % |
| lead_gap_t0<10m | 2255 |  18.6 % |
| lead_stationary_t0 | 2586 |  21.3 % |
| lead_decel>=1.5 | 467 |   3.8 % |
| lead_speed_drop>=2.0 | 916 |   7.5 % |
| lead_stops_fut | 681 |   5.6 % |
| lead_appears_win | 2063 |  17.0 % |
| cutin(ego-frame) | 1100 |   9.1 % |
| cutin_path | 46 |   0.4 % |
| cutin_ego_frame_gap<20m | 407 |   3.4 % |
| vru_corridor_win | 1237 |  10.2 % |
| oncoming_corridor_t0 | 242 |   2.0 % |
| tl_present | 6912 |  56.9 % |
| tl_any_red(list) | 6344 |  52.2 % |
| tl_red_ahead(map) | 2099 |  17.3 % |
| tl_red_ahead & ego_stops/brake>=1.5 | 1039 |   8.6 % |
| tl_red_on_path | 1142 |   9.4 % |
| junction_t0 | 4147 |  34.1 % |
| junction_in_win | 7226 |  59.5 % |
| junction_ahead<=30m | 8765 |  72.2 % |
| cmd=left | 2501 |  20.6 % |
| cmd=right | 1575 |  13.0 % |
| cmd=straight | 8070 |  66.4 % |
| turn |dyaw|>=0.4rad | 2921 |  24.0 % |
| turn |dyaw|>=0.4 & junction_in_win | 2497 |  20.6 % |
| ANY of {brake>=2/stop/start/stationary, lead_hard, cutin, red_ahead, turn, vru} | 7951 |  65.5 % |
| cruise_like (v0>=5, decel<1, no event above) | 2478 |  20.4 % |

## 3. By map location

| map_location | n tokens | logs | lead_t0 | cutin | red_ahead(map) | junction_in_win | ego stops | cruise_like |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sg-one-north | 1914 | 13 | 153 | 44 | 64 | 1244 | 23 | 621 |
| us-ma-boston | 3731 | 51 | 2274 | 415 | 1037 | 2055 | 608 | 729 |
| us-nv-las-vegas-strip | 4064 | 50 | 2498 | 456 | 733 | 2669 | 577 | 441 |
| us-pa-pittsburgh-hazelwood | 2437 | 22 | 896 | 185 | 265 | 1258 | 208 | 687 |

## 4. Distributions (quantiles 0/5/25/50/75/95/100)

| column | n finite | quantiles |
|---|---:|---|
| ego_v0 | 12146 | 0.00, 0.23, 2.52, 5.05, 7.55, 10.87, 15.23 |
| ego_v_min_fut | 12146 | 0.00, 0.00, 1.23, 3.62, 6.34, 10.24, 14.98 |
| ego_decel_max_fd_mps2 | 12146 | 0.00, 0.00, 0.00, 0.26, 1.03, 1.61, 3.97 |
| ego_decel_max_ax_mps2 | 12146 | 0.00, 0.00, 0.00, 0.37, 1.12, 1.72, 3.97 |
| ego_accel_max_fd_mps2 | 12146 | 0.00, 0.00, 0.00, 0.22, 0.82, 1.72, 3.01 |
| lead_gap_m | 5821 | 1.31, 3.84, 7.00, 13.01, 23.43, 42.54, 53.43 |
| lead_time_gap_s | 5313 | 0.26, 1.62, 2.32, 3.13, 4.32, 11.48, 92.79 |
| lead_ttc_s | 4358 | 1.02, 2.16, 3.16, 4.32, 7.49, 28.33, 407.80 |
| lead_closing_mps | 5821 | -15.33, -2.50, 0.09, 1.93, 4.24, 7.64, 11.90 |
| lead_decel_max_mps2 | 5814 | 0.00, 0.00, 0.00, 0.00, 0.88, 1.74, 4.06 |
| lead_speed_drop_fut | 5814 | -1.56, -0.75, 0.00, 0.00, 1.01, 3.85, 10.99 |
| cutin_gap_m | 1100 | 1.31, 6.91, 14.07, 26.01, 39.68, 50.26, 54.34 |
| ego_dyaw_fut_rad | 12146 | -1.80, -0.78, -0.03, 0.00, 0.13, 0.95, 1.76 |
| junction_dist_ahead_m | 10248 | 0.00, 0.00, 0.00, 5.00, 20.00, 50.00, 60.00 |
| tl_red_ahead_dist_m | 2099 | 0.04, 5.62, 16.58, 27.32, 39.84, 54.27, 59.88 |

## 5. Driving-command decoding check (empirical)

`cmd_idx` is the argmax of the 4-vector. Mean realised yaw change over the 4 s future per index (index 0 should be left = positive, 2 right = negative):

| cmd_idx | n | mean ego_dyaw_fut_rad | share with |dyaw|>=0.4 |
|---|---:|---:|---:|
| 0 | 2501 | +0.645 | 71.4 % |
| 1 | 8070 | +0.006 | 1.7 % |
| 2 | 1575 | -0.636 | 63.4 % |

## 6. Sanity: the cruise bundle (`cruise_a0_per_token.json`)

Cruise mask in that bundle: `v0 > 5, |ax0| < 0.3, |GT v4 - v0| < 1 (analyze_ego_probe.py:96)`. Bundle tokens: 154; found in navtest features: **154**.

Definitions (this sanity check only): **free road** = no lead vehicle in the corridor in any of the 9 window frames; **steady lead** = lead at all 9 frames, |closing| < 1.5 m/s at t0, lead speed drop < 2 m/s, lead decel < 1.5 m/s^2, min time gap >= 1.0 s; **other** = everything else.

| class | n | share of 154 |
|---|---:|---:|
| free road | 108 | 70.1 % |
| steady lead | 5 | 3.2 % |
| other | 41 | 26.6 % |
| **free road + steady lead** | 113 | **73.4 %** |

Of the 154: ego stops in future 0, ego decel_fd >= 2.0 0, cutin 12, red-light-ahead 1, junction_in_win 84, in `cruise_like` 111.

Tokens in the bundle that read as 'other' (first 12): `6b2cda5004dc5df9`, `0490beca807c5e9f`, `5816a23cfee25d4e`, `3cd95c48caa05aaa`, `94c3ef5a3168570a`, `eed801b1cd035275`, `4109b79d84ea5053`, `6b19d4325f075ad0`, `6df89719fce05527`, `7e9f20074e1b5e68`, `2bd04a0902095129`, `a97ceb2edd0251f9`

### 6b. Why the 'other' bundle tokens are not free road / steady lead

| reason (not exclusive) | n of 41 |
|---|---:|
| lead present in only some window frames (0 < frac < 1) | 38 |
| lead in all 9 frames but not 'steady' (closing/drop/decel/time-gap rule) | 3 |
| of which lead stationary at t0 | 11 |
| ego-frame `cutin` flag | 11 |
| `cutin_path` flag | 0 |

Event flags over ALL 154 bundle tokens vs ALL navtest: `cutin` 7.8 % vs 9.1 %; `cutin_path` 0.0 % vs 0.4 %; `vru_corridor_win` 11.0 % vs 10.2 %; `oncoming_corridor_t0` 0.6 % vs 2.0 %; `junction_in_win` 54.5 % vs 59.5 %.

## 6c. Cut-in definition diagnostics (the two flags are NOT interchangeable)

| ego behaviour subset | n | `cutin` (ego-frame) | `cutin_path` |
|---|---:|---:|---:|
| all | 12146 | 1100 (9.1 %) | 46 (0.4 %) |
| ego straight (|dyaw|<0.1, |y_end|<1 m) | 5598 | 481 (8.6 %) | 12 (0.2 %) |
| ego turning (|dyaw|>=0.4) | 2921 | 183 (6.3 %) | 21 (0.7 %) |
| ego hard brake (fd>=2.0) | 144 | 14 (9.7 %) | 2 (1.4 %) |
| ego starts from stop | 818 | 66 (8.1 %) | 4 (0.5 %) |

`cutin` = any vehicle that was beside the corridor at t0 and is inside the (ego-relative) corridor later - it also fires on crossing traffic and on ego-induced relative motion. `cutin_path` = the vehicle moves into the ego's realised path. Neither is validated against a ground-truth cut-in label (none exists in the pkl); the hand check in section 7 shows the raw track.

## 6d. Other descriptive caveats

* Stationary leads are common: 2586 of 5821 tokens with a lead (44.4 %) have a lead moving < 0.5 m/s (queues, vehicles parked in the 1.8 m half-width strip). A lead-based rule should say whether it wants these.
* `tl_any_red` (raw list) is True on 52.2 % of tokens vs `tl_red_ahead` (map-matched approach) 17.3 %: the raw list is not a statement about the ego's own signal.
* `dt_ok` False on 51 tokens (frame gap outside 0.5 +- 0.05 s); their finite-difference features are less reliable (filter on `dt_ok == False`).

## 7. Hand verification: one token per category (recomputed straight from the raw pkl, separate code path)

Each block: the feature-table value vs a recomputation that reads `gt_boxes` corners / `ego_dynamic_state` directly. `OK` = agree within 1e-6.

### free_road_cruise
`token 36c6d1ea21015e5f`  log `2021.09.16.19.27.01_veh-45_01749_03230`  list_idx 1820 (pkl frame_idx 39)  cmd_idx 1  us-pa-pittsburgh-hazelwood
ego speed (j-3..j+8, m/s): 8.89, 8.72, 8.56, 8.39, 8.20, 8.07, 7.95, 7.74, 7.37, 6.95, 6.50, 6.11
| quantity | recomputed | table |
|---|---:|---:|
| ego_v0 | 8.3938 | 8.3938 |
| ego_v_min_fut | 6.1058 | 6.1058 |
| ego_decel_max_fd_mps2 | 0.8987 | 0.8987 |
table values: ego_decel_max_ax_mps2=1.025, ego_dist_fut_m=29.077, ego_dyaw_fut_rad=0.112, lead_time_gap_s=nan, lead_ttc_s=nan, lead_speed_drop_fut=nan, lead_decel_max_mps2=nan, cutin=False, cutin_entry_step=-1, cutin_gap_m=nan, cutin_path=False, tl_n=0, tl_n_red=0, tl_red_ahead=False, tl_red_ahead_dist_m=nan, junction_t0=False, junction_dist_ahead_m=nan
**agreement: OK**

### steady_lead
`token 94d209006f485164`  log `2021.06.28.18.03.27_veh-14_00620_01581`  list_idx 335 (pkl frame_idx 15)  cmd_idx 1  us-nv-las-vegas-strip
raw lead box t0 (x,y,L,W,heading) = (26.44, -0.30, 4.66, 2.07, 0.019); lead velocity (ego axes) = (11.50, 0.20) m/s; rear-edge x - 4.049 = gap
ego speed (j-3..j+8, m/s): 9.68, 10.36, 10.83, 11.16, 11.53, 11.90, 12.28, 12.62, 12.75, 12.66, 12.55, 12.51
| quantity | recomputed | table |
|---|---:|---:|
| ego_v0 | 11.1627 | 11.1627 |
| ego_v_min_fut | 11.5256 | 11.5256 |
| ego_decel_max_fd_mps2 | 0.2135 | 0.2135 |
| lead_gap_m | 20.0405 | 20.0405 |
| lead_closing_mps | -0.3399 | -0.3399 |
table values: ego_decel_max_ax_mps2=0.524, ego_dist_fut_m=47.108, ego_dyaw_fut_rad=0.003, lead_time_gap_s=1.795, lead_ttc_s=nan, lead_speed_drop_fut=-0.327, lead_decel_max_mps2=0.000, cutin=False, cutin_entry_step=-1, cutin_gap_m=nan, cutin_path=False, tl_n=15, tl_n_red=8, tl_red_ahead=False, tl_red_ahead_dist_m=nan, junction_t0=True, junction_dist_ahead_m=0.000
**agreement: OK**

### lead_braking
`token a417154cf84a5426`  log `2021.06.28.20.24.43_veh-38_03385_04952`  list_idx 2967 (pkl frame_idx 19)  cmd_idx 1  us-nv-las-vegas-strip
raw lead box t0 (x,y,L,W,heading) = (18.62, -0.13, 4.84, 2.09, -0.017); lead velocity (ego axes) = (3.91, -0.04) m/s; rear-edge x - 4.049 = gap
ego speed (j-3..j+8, m/s): 6.77, 6.65, 6.48, 6.15, 5.76, 5.28, 4.64, 3.86, 2.98, 2.10, 1.31, 0.73
| quantity | recomputed | table |
|---|---:|---:|
| ego_v0 | 6.1460 | 6.1460 |
| ego_v_min_fut | 0.7317 | 0.7317 |
| ego_decel_max_fd_mps2 | 1.7775 | 1.7775 |
| lead_gap_m | 12.1379 | 12.1379 |
| lead_closing_mps | 2.2335 | 2.2335 |
table values: ego_decel_max_ax_mps2=1.801, ego_dist_fut_m=14.002, ego_dyaw_fut_rad=-0.006, lead_time_gap_s=1.975, lead_ttc_s=5.434, lead_speed_drop_fut=3.912, lead_decel_max_mps2=1.984, cutin=False, cutin_entry_step=-1, cutin_gap_m=nan, cutin_path=False, tl_n=0, tl_n_red=0, tl_red_ahead=False, tl_red_ahead_dist_m=nan, junction_t0=True, junction_dist_ahead_m=0.000
**agreement: OK**

### ego_hard_brake
`token 8857612d0dad5e95`  log `2021.09.09.18.29.25_veh-39_00427_00556`  list_idx 94 (pkl frame_idx 29)  cmd_idx 1  us-ma-boston
independent cut-in recomputation (ego-frame, all vehicle tracks): {'track': '724ee061', 'step': 2, 'gap': 29.2665, 'y_t0': -5.12, 'y_entry': 0.55, 'speed': 6.18}
ego speed (j-3..j+8, m/s): 8.35, 8.65, 8.82, 8.93, 8.98, 8.62, 7.05, 5.06, 3.20, 1.68, 0.53, 0.07
| quantity | recomputed | table |
|---|---:|---:|
| ego_v0 | 8.9286 | 8.9286 |
| ego_v_min_fut | 0.0660 | 0.0660 |
| ego_decel_max_fd_mps2 | 3.9686 | 3.9686 |
| cutin_entry_step | 2.0000 | 2.0000 |
| cutin_gap_m | 29.2665 | 29.2665 |
table values: ego_decel_max_ax_mps2=3.973, ego_dist_fut_m=18.835, ego_dyaw_fut_rad=0.004, lead_time_gap_s=nan, lead_ttc_s=nan, lead_speed_drop_fut=nan, lead_decel_max_mps2=nan, cutin=True, cutin_entry_step=2, cutin_gap_m=29.266, cutin_path=False, tl_n=11, tl_n_red=5, tl_red_ahead=False, tl_red_ahead_dist_m=nan, junction_t0=False, junction_dist_ahead_m=25.000
**agreement: OK**

### ego_stops
`token e7295df63d0751d2`  log `2021.09.09.18.29.25_veh-39_01622_01766`  list_idx 134 (pkl frame_idx 19)  cmd_idx 1  us-ma-boston
ego speed (j-3..j+8, m/s): 8.27, 7.65, 6.80, 5.85, 4.76, 3.61, 2.53, 1.53, 0.72, 0.12, 0.01, 0.00
| quantity | recomputed | table |
|---|---:|---:|
| ego_v0 | 5.8470 | 5.8470 |
| ego_v_min_fut | 0.0015 | 0.0015 |
| ego_decel_max_fd_mps2 | 2.2983 | 2.2983 |
table values: ego_decel_max_ax_mps2=2.345, ego_dist_fut_m=7.413, ego_dyaw_fut_rad=-0.003, lead_time_gap_s=nan, lead_ttc_s=nan, lead_speed_drop_fut=nan, lead_decel_max_mps2=nan, cutin=False, cutin_entry_step=-1, cutin_gap_m=nan, cutin_path=False, tl_n=0, tl_n_red=0, tl_red_ahead=False, tl_red_ahead_dist_m=nan, junction_t0=False, junction_dist_ahead_m=15.000
**agreement: OK**

### ego_starts_from_stop
`token 290091fca02b5939`  log `2021.09.16.14.39.34_veh-42_00297_00935`  list_idx 731 (pkl frame_idx 6)  cmd_idx 1  us-pa-pittsburgh-hazelwood
independent cut-in recomputation (ego-frame, all vehicle tracks): {'track': '4ae00e1d', 'step': 3, 'gap': 20.8991, 'y_t0': 3.13, 'y_entry': 0.69, 'speed': 12.05}
ego speed (j-3..j+8, m/s): 0.00, 0.00, 0.03, 0.03, 0.44, 1.33, 2.31, 3.28, 4.25, 5.22, 6.12, 6.88
| quantity | recomputed | table |
|---|---:|---:|
| ego_v0 | 0.0276 | 0.0276 |
| ego_v_min_fut | 0.4429 | 0.4429 |
| ego_decel_max_fd_mps2 | 0.0000 | 0.0000 |
| cutin_entry_step | 3.0000 | 3.0000 |
| cutin_gap_m | 20.8991 | 20.8991 |
table values: ego_decel_max_ax_mps2=0.000, ego_dist_fut_m=13.254, ego_dyaw_fut_rad=0.127, lead_time_gap_s=nan, lead_ttc_s=nan, lead_speed_drop_fut=nan, lead_decel_max_mps2=nan, cutin=True, cutin_entry_step=3, cutin_gap_m=20.899, cutin_path=False, tl_n=0, tl_n_red=0, tl_red_ahead=False, tl_red_ahead_dist_m=nan, junction_t0=False, junction_dist_ahead_m=nan
**agreement: OK**

### cutin_path
`token 6162cfa976a1521f`  log `2021.09.09.18.04.06_veh-40_00743_01071`  list_idx 615 (pkl frame_idx 22)  cmd_idx 1  us-ma-boston
independent cut-in recomputation (ego-frame, all vehicle tracks): {'track': '4c1329af', 'step': 5, 'gap': 11.7384, 'y_t0': -2.63, 'y_entry': -1.54, 'speed': 11.1}
ego speed (j-3..j+8, m/s): 7.31, 7.84, 8.34, 8.84, 9.31, 9.77, 10.18, 10.47, 10.63, 10.71, 10.64, 10.40
| quantity | recomputed | table |
|---|---:|---:|
| ego_v0 | 8.8369 | 8.8369 |
| ego_v_min_fut | 9.3081 | 9.3081 |
| ego_decel_max_fd_mps2 | 0.4907 | 0.4907 |
| cutin_entry_step | 5.0000 | 5.0000 |
| cutin_gap_m | 11.7384 | 11.7384 |
table values: ego_decel_max_ax_mps2=0.597, ego_dist_fut_m=39.710, ego_dyaw_fut_rad=-0.075, lead_time_gap_s=nan, lead_ttc_s=nan, lead_speed_drop_fut=nan, lead_decel_max_mps2=nan, cutin=True, cutin_entry_step=5, cutin_gap_m=11.738, cutin_path=True, tl_n=8, tl_n_red=2, tl_red_ahead=False, tl_red_ahead_dist_m=nan, junction_t0=True, junction_dist_ahead_m=0.000
**agreement: OK**

### cutin_ego_frame
`token f2d4e4b013a3579e`  log `2021.09.09.14.18.22_veh-48_01298_01492`  list_idx 228 (pkl frame_idx 24)  cmd_idx 1  us-ma-boston
raw lead box t0 (x,y,L,W,heading) = (20.12, -1.22, 4.64, 1.99, -0.101); lead velocity (ego axes) = (2.47, -0.23) m/s; rear-edge x - 4.049 = gap
independent cut-in recomputation (ego-frame, all vehicle tracks): {'track': '698b6a19', 'step': 4, 'gap': 42.0179, 'y_t0': -3.53, 'y_entry': -1.64, 'speed': 3.09}
ego speed (j-3..j+8, m/s): 7.46, 7.14, 6.69, 6.18, 5.62, 5.03, 4.66, 4.48, 4.37, 4.30, 4.22, 4.14
| quantity | recomputed | table |
|---|---:|---:|
| ego_v0 | 6.1806 | 6.1806 |
| ego_v_min_fut | 4.1371 | 4.1371 |
| ego_decel_max_fd_mps2 | 1.1673 | 1.1673 |
| lead_gap_m | 13.6645 | 13.6645 |
| lead_closing_mps | 3.7113 | 3.7113 |
| cutin_entry_step | 4.0000 | 4.0000 |
| cutin_gap_m | 42.0179 | 42.0179 |
table values: ego_decel_max_ax_mps2=1.158, ego_dist_fut_m=18.055, ego_dyaw_fut_rad=-0.033, lead_time_gap_s=2.211, lead_ttc_s=3.682, lead_speed_drop_fut=0.150, lead_decel_max_mps2=0.301, cutin=True, cutin_entry_step=4, cutin_gap_m=42.018, cutin_path=False, tl_n=4, tl_n_red=4, tl_red_ahead=False, tl_red_ahead_dist_m=nan, junction_t0=True, junction_dist_ahead_m=0.000
**agreement: OK**

### red_ahead_stop
`token 7ef36a2139b45d9a`  log `2021.09.09.14.18.22_veh-48_01298_01492`  list_idx 34 (pkl frame_idx 30)  cmd_idx 1  us-ma-boston
ego speed (j-3..j+8, m/s): 5.99, 5.84, 5.65, 5.44, 5.08, 4.54, 3.86, 3.09, 2.21, 1.37, 0.54, 0.13
| quantity | recomputed | table |
|---|---:|---:|
| ego_v0 | 5.4437 | 5.4437 |
| ego_v_min_fut | 0.1296 | 0.1296 |
| ego_decel_max_fd_mps2 | 1.7484 | 1.7484 |
table values: ego_decel_max_ax_mps2=1.809, ego_dist_fut_m=11.138, ego_dyaw_fut_rad=-0.001, lead_time_gap_s=nan, lead_ttc_s=nan, lead_speed_drop_fut=nan, lead_decel_max_mps2=nan, cutin=False, cutin_entry_step=-1, cutin_gap_m=nan, cutin_path=False, tl_n=6, tl_n_red=4, tl_red_ahead=True, tl_red_ahead_dist_m=14.206, junction_t0=True, junction_dist_ahead_m=0.000
**agreement: OK**

### junction_turn
`token e10bd715492150e3`  log `2021.09.16.14.39.34_veh-42_00297_00935`  list_idx 779 (pkl frame_idx 15)  cmd_idx 2  us-pa-pittsburgh-hazelwood
ego speed (j-3..j+8, m/s): 1.90, 2.43, 2.94, 3.31, 3.52, 3.69, 3.73, 3.79, 3.88, 4.10, 4.45, 4.87
| quantity | recomputed | table |
|---|---:|---:|
| ego_v0 | 3.3096 | 3.3096 |
| ego_v_min_fut | 3.5226 | 3.5226 |
| ego_decel_max_fd_mps2 | 0.0000 | 0.0000 |
table values: ego_decel_max_ax_mps2=0.000, ego_dist_fut_m=15.171, ego_dyaw_fut_rad=-1.501, lead_time_gap_s=nan, lead_ttc_s=nan, lead_speed_drop_fut=nan, lead_decel_max_mps2=nan, cutin=False, cutin_entry_step=-1, cutin_gap_m=nan, cutin_path=False, tl_n=0, tl_n_red=0, tl_red_ahead=False, tl_red_ahead_dist_m=nan, junction_t0=True, junction_dist_ahead_m=0.000
**agreement: OK**
