# OPEN-LOOP panel — flagship v1 vs REF-C base, JUNCTION scene 7c72937c, empty

**Arms:** `flagship-v1` (A) vs `refc-base` (B) · windows A 170 / clusters 9 · B 170 / 9 · paired windows 170

⭐ **OPEN LOOP** — the ego follows the LOGGED trajectory; every frame is rendered at the pose the rig actually had and the model's plan is scored against the log's own future motion. **The plan is never executed.** This isolates perception + prediction from the control drift that is confounded with them in every closed-loop number.

`f_eff` A = 266.1706644208189, B = 266.1706644208189 (the canonicalization self-check against `F_REF`; both arms are 256px SQUARE and `_BasePolicy.canon` asserts `(1, 8, 9, 256, 256)` before any forward pass).

## ⛔ Degenerate by construction — measured, then marked

| arm | family | metric | measured max\|value\| | tol | confirmed | why |
|---|---|---|---|---|---|---|
| flagship-v1 | ADE | `dist_to_gt_traj_m` | 0 | 1e-06 | ✅ | literally abs(cross_track) in cl_metrics — the same measurement under a second name |
| flagship-v1 | LONGITUDINAL | `executed_speed_err_ms` | 0 | 1e-06 | ✅ | the ego speed IS the logged speed; no controller runs |
| flagship-v1 | LATERAL | `cross_track_abs_m` | 0 | 1e-06 | ✅ | the ego pose IS the logged pose, so its distance to the logged polyline is zero by construction |
| flagship-v1 | LATERAL | `cross_track_signed_m` | 0 | 1e-06 | ✅ | same quantity, signed |
| flagship-v1 | STRATEGIC | `route_corridor_departure_rate` | 0 | 1e-06 | ✅ | departure is |cross_track| > 2 m, and cross_track is pinned at 0 |
| refc-base | ADE | `dist_to_gt_traj_m` | 0 | 1e-06 | ✅ | literally abs(cross_track) in cl_metrics — the same measurement under a second name |
| refc-base | LONGITUDINAL | `executed_speed_err_ms` | 0 | 1e-06 | ✅ | the ego speed IS the logged speed; no controller runs |
| refc-base | LATERAL | `cross_track_abs_m` | 0 | 1e-06 | ✅ | the ego pose IS the logged pose, so its distance to the logged polyline is zero by construction |
| refc-base | LATERAL | `cross_track_signed_m` | 0 | 1e-06 | ✅ | same quantity, signed |
| refc-base | STRATEGIC | `route_corridor_departure_rate` | 0 | 1e-06 | ✅ | departure is |cross_track| > 2 m, and cross_track is pinned at 0 |

All pinned metrics confirmed at float tolerance. They are **setup, not result**, and are struck through in the family tables below.

⚠️ `manoeuvre_exec_eq_plan` — in OPEN loop the 'executed' manoeuvre is classified from the ego poses, which are the LOGGED poses — so this is `manoeuvre_plan_eq_logged` a second time, not an independent measurement of whether the arm executes what it selects. That question is only askable in CLOSED loop.

## ADE

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `ade_0_2s` | 2.1907 [1.6012, 2.8928] | 0.7863 [0.5683, 1.0297] | +1.4044 [+0.6592, +2.1885] | **separated** |
| `de_0_5s` | 0.8040 [0.6165, 1.0163] | 0.1252 [0.0870, 0.1721] | - |  |
| `de_1s` | 1.6927 [1.2569, 2.2007] | 0.4126 [0.2866, 0.5665] | - |  |
| `de_1_5s` | 2.6120 [1.8964, 3.4581] | 0.9233 [0.6481, 1.2366] | - |  |
| `de_2s` | 3.6542 [2.5988, 4.9205] | 1.6840 [1.2379, 2.1575] | - |  |
| ~~`dist_to_gt_traj_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |


## LONGITUDINAL

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `target_speed_err_ms` | -0.2762 [-1.1858, 0.5234] | 0.0718 [-0.0780, 0.2305] | - |  |
| `abs_target_speed_err_ms` | 1.5800 [1.2111, 1.9581] | 0.2154 [0.1407, 0.3070] | +1.3645 [+0.9765, +1.8099] | **separated** |
| ~~`executed_speed_err_ms`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| `along_track_ade_m` | 2.1039 [1.4559, 2.8544] | 0.6957 [0.4439, 0.9767] | +1.4082 [+0.6637, +2.1969] | **separated** |
| `along_track_0_5s_m` | 0.8004 [0.6109, 1.0135] | 0.1195 [0.0790, 0.1685] | - |  |
| `along_track_1s_m` | 1.6619 [1.2088, 2.1857] | 0.3827 [0.2450, 0.5460] | - |  |
| `along_track_1_5s_m` | 2.5159 [1.7415, 3.4137] | 0.8289 [0.5235, 1.1735] | - |  |
| `along_track_2s_m` | 3.4374 [2.2580, 4.8103] | 1.4517 [0.9176, 2.0156] | - |  |
| `headway_m` | 10.9361 [2.7911, 22.4149] | 10.9361 [2.7911, 22.4149] | - |  |
| `time_gap_s` | 9.4655 [1.0677, 22.6088] | 9.4655 [1.0677, 22.6088] | - |  |
| `min_headway_m` | -4.3684 [-4.3684, -3.2301] | -4.3684 [-4.3684, -3.2301] | - |  |
| `frac_time_gap_below_1s` | 0.4412 [0.1191, 0.7723] | 0.4412 [0.1191, 0.7723] | - |  |
| `frac_time_gap_below_0_5s` | 0.4412 [0.1191, 0.7723] | 0.4412 [0.1191, 0.7723] | - |  |
| `ttc_s_when_closing` | 2.9485 [0.3071, 7.5783] | 2.9485 [0.3071, 7.5783] | - |  |

* **A `lead_present_rate`** — `0.8`
* **B `lead_present_rate`** — `0.8`
* **A `ttc_defined_rate`** — `0.1838`
* **B `ttc_defined_rate`** — `0.1838`

## LATERAL

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `heading_err_rad` | 0.1495 [0.0689, 0.2513] | 0.1510 [0.0487, 0.2628] | -0.0015 [-0.0613, +0.0659] | not separated |
| `curvature_err_1pm` | 0.0161 [0.0070, 0.0265] | 0.0424 [0.0078, 0.0992] | -0.0263 [-0.0830, +0.0090] | not separated |
| `yawrate_err_rads` | 0.1129 [0.0315, 0.2242] | 0.1013 [0.0207, 0.2160] | +0.0115 [+0.0044, +0.0194] | **separated** |
| ~~`cross_track_abs_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| ~~`cross_track_signed_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | - | ⛔ setup, not result |
| `lateral_ade_m` | 0.3167 [0.1583, 0.4955] | 0.1809 [0.0586, 0.3319] | +0.1358 [+0.0847, +0.1944] | **separated** |


## TACTICAL

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `manoeuvre_plan_eq_logged` | 0.5118 [0.3067, 0.7302] | 0.4176 [0.2000, 0.6650] | +0.0941 [-0.0178, +0.2042] | not separated |
| `manoeuvre_exec_eq_plan` ⚠️ | 0.4000 [0.1000, 0.7778] | 0.3000 [0.0833, 0.6667] | - | alias of plan_eq_logged |
| `head_eq_plan` | 0.5118 [0.2721, 0.7435] | 0.5412 [0.3377, 0.7584] | - |  |
| `head_eq_logged` | 0.5882 [0.3733, 0.7990] | 0.3941 [0.1600, 0.6335] | - |  |

* **A `plan_class_share`** — `{"lane_keep": 0.2118, "turn_left": 0.3412, "turn_right": 0.0, "accelerate": 0.2588, "brake_stop": 0.1882}`
* **B `plan_class_share`** — `{"lane_keep": 0.3353, "turn_left": 0.3412, "turn_right": 0.0588, "accelerate": 0.1529, "brake_stop": 0.1118}`
* **A `logged_class_share`** — `{"lane_keep": 0.1353, "turn_left": 0.3235, "turn_right": 0.0, "accelerate": 0.5412, "brake_stop": 0.0}`
* **B `logged_class_share`** — `{"lane_keep": 0.1353, "turn_left": 0.3235, "turn_right": 0.0, "accelerate": 0.5412, "brake_stop": 0.0}`
* **A `head_class_share`** — `{"lane_keep": 0.1412, "turn_left": 0.3235, "turn_right": 0.0, "accelerate": 0.3941, "brake_stop": 0.1412}`
* **B `head_class_share`** — `{"lane_keep": 0.6471, "turn_left": 0.3529, "turn_right": 0.0, "accelerate": 0.0, "brake_stop": 0.0}`
* **A `head_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.0833, "recall": 0.087, "f1": 0.0851, "support_n_true": 23, "n_fires": 24}, "turn_left": {"precision": 0.7818, "recall": 0.7818, "f1": 0.7818, "support_n_true": 55, "n_fires": 55}, "turn_right": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "accelerate": {"precision": 0.8209, "recall": 0.5978, "f1": 0.6918, "support_n_true": 92, "n_fires": 67}, "brake_stop": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 24}, "_macro_f1": 0.3117, "_accuracy": 0.5882, "_majority_class_baseline_acc": 0.5412, "_n": 17`
* **B `head_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.2091, "recall": 1.0, "f1": 0.3459, "support_n_true": 23, "n_fires": 110}, "turn_left": {"precision": 0.7333, "recall": 0.8, "f1": 0.7652, "support_n_true": 55, "n_fires": 60}, "turn_right": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "accelerate": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 92, "n_fires": 0}, "brake_stop": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "_macro_f1": 0.2222, "_accuracy": 0.3941, "_majority_class_baseline_acc": 0.5412, "_n": 170, "_n_input"`
* **A `plan_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.25, "recall": 0.3913, "f1": 0.3051, "support_n_true": 23, "n_fires": 36}, "turn_left": {"precision": 0.7069, "recall": 0.7455, "f1": 0.7257, "support_n_true": 55, "n_fires": 58}, "turn_right": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "accelerate": {"precision": 0.8409, "recall": 0.4022, "f1": 0.5441, "support_n_true": 92, "n_fires": 44}, "brake_stop": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 32}, "_macro_f1": 0.315, "_accuracy": 0.5118, "_majority_class_baseline_acc": 0.5412, "_n": 170,`
* **B `plan_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.1228, "recall": 0.3043, "f1": 0.175, "support_n_true": 23, "n_fires": 57}, "turn_left": {"precision": 0.6724, "recall": 0.7091, "f1": 0.6903, "support_n_true": 55, "n_fires": 58}, "turn_right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 10}, "accelerate": {"precision": 0.9615, "recall": 0.2717, "f1": 0.4237, "support_n_true": 92, "n_fires": 26}, "brake_stop": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 19}, "_macro_f1": 0.2578, "_accuracy": 0.4176, "_majority_class_baseline_acc": 0.5412, "_n": 17`

## STRATEGIC

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| ~~`route_corridor_departure_rate`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| `logged_net_dyaw_rad` | 1.3799 [0.6622, 2.0180] | 1.3799 [0.6622, 2.0180] | - |  |
| `route_head_eq_logged` | 1.0000 [1.0000, 1.0000] ⛔CIRCULAR | 0.6452 [0.3694, 0.9154] | +0.3548 [+0.0846, +0.6306] | **separated** |
| `route_head_side_eq_graded_proxy` | 0.9235 [0.8254, 1.0000] ⛔CIRCULAR | 0.6765 [0.4105, 0.9263] | - |  |
| `route_progress_rel` | 0.9909 [0.9782, 0.9996] | 0.9909 [0.9782, 0.9996] | - |  |

* **A `route_logged_share`** — `{"left": 0.7118, "straight": 0.2, "right": 0.0, "unknown": 0.0882}`
* **B `route_logged_share`** — `{"left": 0.7118, "straight": 0.2, "right": 0.0, "unknown": 0.0882}`
* **A `route_head_share`** — `{"left": 0.7118, "straight": 0.2882, "right": 0.0, "CIRCULAR_NAV_ECHO": true, "do_not_quote": "the route head is a deterministic function of the nav input"}`
* **B `route_head_share`** — `{"left": 0.4647, "straight": 0.4, "right": 0.1353}`
* **A `route_derivation_reason_share`** — `{"tight_transient": 0.7118, "gray_zone": 0.0882, "road_following": 0.2}`
* **B `route_derivation_reason_share`** — `{"tight_transient": 0.7118, "gray_zone": 0.0882, "road_following": 0.2}`
* **A `route_label_valid_rate`** — `0.9118`
* **B `route_label_valid_rate`** — `0.9118`
* **A `route_head_nav_echo_check`** — `{"head_is_deterministic_function_of_nav": true, "identifiable": true, "n_distinct_nav": 2, "n_distinct_head": 2, "nav_to_head_map": {"1": [0], "0": [1]}, "n": 170, "verdict": "CIRCULAR \u2014 route_head_eq_logged above reproduces the nav command the policy was GIVEN and is NOT evidence of strategic skill; do not quote it"}`
* **B `route_head_nav_echo_check`** — `{"head_is_deterministic_function_of_nav": false, "identifiable": true, "n_distinct_nav": 2, "n_distinct_head": 3, "nav_to_head_map": {"1": [0, 1, 2], "0": [0, 1]}, "n": 170, "verdict": "not an echo \u2014 the head is not a function of nav on these windows"}`
* **A `route_head_side_per_class_PR`** — `{"straight": {"precision": 0.7347, "recall": 1.0, "f1": 0.8471, "support_n_true": 36, "n_fires": 49}, "left": {"precision": 1.0, "recall": 0.903, "f1": 0.949, "support_n_true": 134, "n_fires": 121}, "right": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "_macro_f1": 0.5987, "_accuracy": 0.9235, "_majority_class_baseline_acc": 0.7882, "_n": 170, "_n_input": 170, "_n_dropped_unpaired": 0, "CIRCULAR_NAV_ECHO": true, "do_not_quote": "the route head is a deterministic function of the nav input"}`
* **B `route_head_side_per_class_PR`** — `{"straight": {"precision": 0.5294, "recall": 1.0, "f1": 0.6923, "support_n_true": 36, "n_fires": 68}, "left": {"precision": 1.0, "recall": 0.5896, "f1": 0.7418, "support_n_true": 134, "n_fires": 79}, "right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 23}, "_macro_f1": 0.478, "_accuracy": 0.6765, "_majority_class_baseline_acc": 0.7882, "_n": 170, "_n_input": 170, "_n_dropped_unpaired": 0}`

## Estimator and caveats

* episode-cluster bootstrap over ROLLOUT STARTS. The clusters are disjoint segments of ONE clip, not independent episodes — the interval is the right estimator for the resampling unit available, and the unit is stated here so it is never mistaken for the 40-episode val bootstrap.
* WITHIN-SIM RELATIVE. REF-C open-loop ADE is 1.5157 on these NuRec reconstructions vs 0.4728 on real footage (3.21x OOD). Orderings survive; absolute rates do not.
* ✅ **grad-NCC is the only admissible render metric on these night clips.** PSNR, NCC **and MAE** are RETRACTED here — grad-NCC identifies the correct reference frame 5/5 on every arm while MAE/PSNR manage 1–4/5 with arm-dependent reliability.
* **Scope:** AlpaSim renderer **wire contract** with a gsplat backend. NOT `alpasim_runtime.simulate` — there is no AlpaSim collision/offroad score here.
* ⚠️ Report **precision alongside recall** for every rate, and state the denominator — the per-class PR blocks above carry both.