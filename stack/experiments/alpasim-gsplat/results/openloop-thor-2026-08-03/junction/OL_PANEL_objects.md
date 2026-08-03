# OPEN-LOOP panel — flagship v1 vs REF-C base, JUNCTION scene 7c72937c, objects

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
| `ade_0_2s` | 2.2431 [1.6524, 3.0405] | 0.7600 [0.5366, 0.9828] | +1.4831 [+0.8427, +2.2610] | **separated** |
| `de_0_5s` | 0.8202 [0.6129, 1.0758] | 0.1124 [0.0784, 0.1536] | - |  |
| `de_1s` | 1.7331 [1.2920, 2.3110] | 0.3941 [0.2638, 0.5304] | - |  |
| `de_1_5s` | 2.6829 [1.9815, 3.6365] | 0.8977 [0.6168, 1.1827] | - |  |
| `de_2s` | 3.7362 [2.7240, 5.1399] | 1.6358 [1.1889, 2.0802] | - |  |
| ~~`dist_to_gt_traj_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |


## LONGITUDINAL

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `target_speed_err_ms` | -0.3737 [-1.2913, 0.3988] | 0.0639 [-0.0665, 0.2088] | - |  |
| `abs_target_speed_err_ms` | 1.5897 [1.1881, 2.0622] | 0.1885 [0.1189, 0.2820] | +1.4013 [+0.9754, +1.9154] | **separated** |
| ~~`executed_speed_err_ms`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| `along_track_ade_m` | 2.1574 [1.5528, 2.9802] | 0.6785 [0.4316, 0.9369] | +1.4790 [+0.8340, +2.2657] | **separated** |
| `along_track_0_5s_m` | 0.8165 [0.6094, 1.0736] | 0.1077 [0.0723, 0.1508] | - |  |
| `along_track_1s_m` | 1.7037 [1.2570, 2.2977] | 0.3688 [0.2345, 0.5191] | - |  |
| `along_track_1_5s_m` | 2.5878 [1.8664, 3.5705] | 0.8140 [0.5190, 1.1355] | - |  |
| `along_track_2s_m` | 3.5218 [2.4373, 5.0229] | 1.4235 [0.8959, 1.9698] | - |  |
| `headway_m` | 8.9034 [0.9393, 18.8301] | 8.9034 [0.9393, 18.8301] | - |  |
| `time_gap_s` | 9.5788 [0.7598, 22.4537] | 9.5788 [0.7598, 22.4537] | - |  |
| `min_headway_m` | -4.3684 [-4.3684, -3.2301] | -4.3684 [-4.3684, -3.2301] | - |  |
| `frac_time_gap_below_1s` | 0.4651 [0.1329, 0.8134] | 0.4651 [0.1329, 0.8134] | - |  |
| `frac_time_gap_below_0_5s` | 0.4651 [0.1329, 0.8134] | 0.4651 [0.1329, 0.8134] | - |  |
| `ttc_s_when_closing` | 1.1481 [0.3071, 1.5686] | 1.1481 [0.3071, 1.5686] | - |  |

* **A `lead_present_rate`** — `0.7588235294117647`
* **B `lead_present_rate`** — `0.7588235294117647`
* **A `ttc_defined_rate`** — `0.1395`
* **B `ttc_defined_rate`** — `0.1395`

## LATERAL

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `heading_err_rad` | 0.1484 [0.0720, 0.2446] | 0.1436 [0.0525, 0.2433] | +0.0048 [-0.0536, +0.0647] | not separated |
| `curvature_err_1pm` | 0.0192 [0.0092, 0.0289] | 0.0402 [0.0087, 0.0887] | -0.0210 [-0.0676, +0.0088] | not separated |
| `yawrate_err_rads` | 0.1132 [0.0313, 0.2248] | 0.1006 [0.0196, 0.2158] | +0.0126 [+0.0043, +0.0212] | **separated** |
| ~~`cross_track_abs_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| ~~`cross_track_signed_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | - | ⛔ setup, not result |
| `lateral_ade_m` | 0.3353 [0.1703, 0.5185] | 0.1721 [0.0592, 0.3023] | +0.1631 [+0.1082, +0.2249] | **separated** |


## TACTICAL

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `manoeuvre_plan_eq_logged` | 0.4882 [0.2924, 0.7054] | 0.4412 [0.2282, 0.6721] | +0.0471 [-0.0588, +0.1385] | not separated |
| `manoeuvre_exec_eq_plan` ⚠️ | 0.5000 [0.2000, 0.8889] | 0.3000 [0.0833, 0.6667] | - | alias of plan_eq_logged |
| `head_eq_plan` | 0.5176 [0.3117, 0.7427] | 0.5882 [0.3757, 0.7948] | - |  |
| `head_eq_logged` | 0.5765 [0.3764, 0.7831] | 0.3941 [0.1600, 0.6335] | - |  |

* **A `plan_class_share`** — `{"lane_keep": 0.1706, "turn_left": 0.3588, "turn_right": 0.0, "accelerate": 0.2647, "brake_stop": 0.2059}`
* **B `plan_class_share`** — `{"lane_keep": 0.3353, "turn_left": 0.3941, "turn_right": 0.0353, "accelerate": 0.1412, "brake_stop": 0.0941}`
* **A `logged_class_share`** — `{"lane_keep": 0.1353, "turn_left": 0.3235, "turn_right": 0.0, "accelerate": 0.5412, "brake_stop": 0.0}`
* **B `logged_class_share`** — `{"lane_keep": 0.1353, "turn_left": 0.3235, "turn_right": 0.0, "accelerate": 0.5412, "brake_stop": 0.0}`
* **A `head_class_share`** — `{"lane_keep": 0.0941, "turn_left": 0.3294, "turn_right": 0.0, "accelerate": 0.3882, "brake_stop": 0.1882}`
* **B `head_class_share`** — `{"lane_keep": 0.6471, "turn_left": 0.3529, "turn_right": 0.0, "accelerate": 0.0, "brake_stop": 0.0}`
* **A `head_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support_n_true": 23, "n_fires": 16}, "turn_left": {"precision": 0.7857, "recall": 0.8, "f1": 0.7928, "support_n_true": 55, "n_fires": 56}, "turn_right": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "accelerate": {"precision": 0.8182, "recall": 0.587, "f1": 0.6835, "support_n_true": 92, "n_fires": 66}, "brake_stop": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 32}, "_macro_f1": 0.2953, "_accuracy": 0.5765, "_majority_class_baseline_acc": 0.5412, "_n": 170, "_n_input`
* **B `head_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.2091, "recall": 1.0, "f1": 0.3459, "support_n_true": 23, "n_fires": 110}, "turn_left": {"precision": 0.7333, "recall": 0.8, "f1": 0.7652, "support_n_true": 55, "n_fires": 60}, "turn_right": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "accelerate": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 92, "n_fires": 0}, "brake_stop": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "_macro_f1": 0.2222, "_accuracy": 0.3941, "_majority_class_baseline_acc": 0.5412, "_n": 170, "_n_input"`
* **A `plan_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.2759, "recall": 0.3478, "f1": 0.3077, "support_n_true": 23, "n_fires": 29}, "turn_left": {"precision": 0.6721, "recall": 0.7455, "f1": 0.7069, "support_n_true": 55, "n_fires": 61}, "turn_right": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "accelerate": {"precision": 0.7556, "recall": 0.3696, "f1": 0.4964, "support_n_true": 92, "n_fires": 45}, "brake_stop": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 35}, "_macro_f1": 0.3022, "_accuracy": 0.4882, "_majority_class_baseline_acc": 0.5412, "_n": 1`
* **B `plan_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.1579, "recall": 0.3913, "f1": 0.225, "support_n_true": 23, "n_fires": 57}, "turn_left": {"precision": 0.6269, "recall": 0.7636, "f1": 0.6885, "support_n_true": 55, "n_fires": 67}, "turn_right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 6}, "accelerate": {"precision": 1.0, "recall": 0.2609, "f1": 0.4138, "support_n_true": 92, "n_fires": 24}, "brake_stop": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 16}, "_macro_f1": 0.2655, "_accuracy": 0.4412, "_majority_class_baseline_acc": 0.5412, "_n": 170, "`

## STRATEGIC

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| ~~`route_corridor_departure_rate`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| `logged_net_dyaw_rad` | 1.3799 [0.6622, 2.0180] | 1.3799 [0.6622, 2.0180] | - |  |
| `route_head_eq_logged` | 1.0000 [1.0000, 1.0000] ⛔CIRCULAR | 0.7613 [0.5437, 0.9736] | +0.2387 [+0.0264, +0.4563] | **separated** |
| `route_head_side_eq_graded_proxy` | 0.9235 [0.8254, 1.0000] ⛔CIRCULAR | 0.7824 [0.5755, 0.9764] | - |  |
| `route_progress_rel` | 0.9909 [0.9782, 0.9996] | 0.9909 [0.9782, 0.9996] | - |  |

* **A `route_logged_share`** — `{"left": 0.7118, "straight": 0.2, "right": 0.0, "unknown": 0.0882}`
* **B `route_logged_share`** — `{"left": 0.7118, "straight": 0.2, "right": 0.0, "unknown": 0.0882}`
* **A `route_head_share`** — `{"left": 0.7118, "straight": 0.2882, "right": 0.0, "CIRCULAR_NAV_ECHO": true, "do_not_quote": "the route head is a deterministic function of the nav input"}`
* **B `route_head_share`** — `{"left": 0.5706, "straight": 0.3882, "right": 0.0412}`
* **A `route_derivation_reason_share`** — `{"tight_transient": 0.7118, "gray_zone": 0.0882, "road_following": 0.2}`
* **B `route_derivation_reason_share`** — `{"tight_transient": 0.7118, "gray_zone": 0.0882, "road_following": 0.2}`
* **A `route_label_valid_rate`** — `0.9118`
* **B `route_label_valid_rate`** — `0.9118`
* **A `route_head_nav_echo_check`** — `{"head_is_deterministic_function_of_nav": true, "identifiable": true, "n_distinct_nav": 2, "n_distinct_head": 2, "nav_to_head_map": {"1": [0], "0": [1]}, "n": 170, "verdict": "CIRCULAR \u2014 route_head_eq_logged above reproduces the nav command the policy was GIVEN and is NOT evidence of strategic skill; do not quote it"}`
* **B `route_head_nav_echo_check`** — `{"head_is_deterministic_function_of_nav": false, "identifiable": true, "n_distinct_nav": 2, "n_distinct_head": 3, "nav_to_head_map": {"1": [0, 1, 2], "0": [0, 1]}, "n": 170, "verdict": "not an echo \u2014 the head is not a function of nav on these windows"}`
* **A `route_head_side_per_class_PR`** — `{"straight": {"precision": 0.7347, "recall": 1.0, "f1": 0.8471, "support_n_true": 36, "n_fires": 49}, "left": {"precision": 1.0, "recall": 0.903, "f1": 0.949, "support_n_true": 134, "n_fires": 121}, "right": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "_macro_f1": 0.5987, "_accuracy": 0.9235, "_majority_class_baseline_acc": 0.7882, "_n": 170, "_n_input": 170, "_n_dropped_unpaired": 0, "CIRCULAR_NAV_ECHO": true, "do_not_quote": "the route head is a deterministic function of the nav input"}`
* **B `route_head_side_per_class_PR`** — `{"straight": {"precision": 0.5455, "recall": 1.0, "f1": 0.7059, "support_n_true": 36, "n_fires": 66}, "left": {"precision": 1.0, "recall": 0.7239, "f1": 0.8398, "support_n_true": 134, "n_fires": 97}, "right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 7}, "_macro_f1": 0.5152, "_accuracy": 0.7824, "_majority_class_baseline_acc": 0.7882, "_n": 170, "_n_input": 170, "_n_dropped_unpaired": 0}`

## Estimator and caveats

* episode-cluster bootstrap over ROLLOUT STARTS. The clusters are disjoint segments of ONE clip, not independent episodes — the interval is the right estimator for the resampling unit available, and the unit is stated here so it is never mistaken for the 40-episode val bootstrap.
* WITHIN-SIM RELATIVE. REF-C open-loop ADE is 1.5157 on these NuRec reconstructions vs 0.4728 on real footage (3.21x OOD). Orderings survive; absolute rates do not.
* ✅ **grad-NCC is the only admissible render metric on these night clips.** PSNR, NCC **and MAE** are RETRACTED here — grad-NCC identifies the correct reference frame 5/5 on every arm while MAE/PSNR manage 1–4/5 with arm-dependent reliability.
* **Scope:** AlpaSim renderer **wire contract** with a gsplat backend. NOT `alpasim_runtime.simulate` — there is no AlpaSim collision/offroad score here.
* ⚠️ Report **precision alongside recall** for every rate, and state the denominator — the per-class PR blocks above carry both.