# OPEN-LOOP: refc-base with-objects vs empty-road (identical logged poses)

**Arms:** `refc-base` (A) vs `refc-base` (B) · windows A 170 / clusters 9 · B 170 / 9 · paired windows 170

⭐ **OPEN LOOP** — the ego follows the LOGGED trajectory; every frame is rendered at the pose the rig actually had and the model's plan is scored against the log's own future motion. **The plan is never executed.** This isolates perception + prediction from the control drift that is confounded with them in every closed-loop number.

`f_eff` A = 266.0139895990537, B = 266.0139895990537 (the canonicalization self-check against `F_REF`; both arms are 256px SQUARE and `_BasePolicy.canon` asserts `(1, 8, 9, 256, 256)` before any forward pass).

## ⛔ Degenerate by construction — measured, then marked

| arm | family | metric | measured max\|value\| | tol | confirmed | why |
|---|---|---|---|---|---|---|
| refc-base | ADE | `dist_to_gt_traj_m` | 0 | 1e-06 | ✅ | literally abs(cross_track) in cl_metrics — the same measurement under a second name |
| refc-base | LONGITUDINAL | `executed_speed_err_ms` | 0 | 1e-06 | ✅ | the ego speed IS the logged speed; no controller runs |
| refc-base | LATERAL | `cross_track_abs_m` | 0 | 1e-06 | ✅ | the ego pose IS the logged pose, so its distance to the logged polyline is zero by construction |
| refc-base | LATERAL | `cross_track_signed_m` | 0 | 1e-06 | ✅ | same quantity, signed |
| refc-base | STRATEGIC | `route_corridor_departure_rate` | 0 | 1e-06 | ✅ | departure is |cross_track| > 2 m, and cross_track is pinned at 0 |
| refc-base | ADE | `dist_to_gt_traj_m` | 0 | 1e-06 | ✅ | literally abs(cross_track) in cl_metrics — the same measurement under a second name |
| refc-base | LONGITUDINAL | `executed_speed_err_ms` | 0 | 1e-06 | ✅ | the ego speed IS the logged speed; no controller runs |
| refc-base | LATERAL | `cross_track_abs_m` | 0 | 1e-06 | ✅ | the ego pose IS the logged pose, so its distance to the logged polyline is zero by construction |
| refc-base | LATERAL | `cross_track_signed_m` | 0 | 1e-06 | ✅ | same quantity, signed |
| refc-base | STRATEGIC | `route_corridor_departure_rate` | 0 | 1e-06 | ✅ | departure is |cross_track| > 2 m, and cross_track is pinned at 0 |

All pinned metrics confirmed at float tolerance. They are **setup, not result**, and are struck through in the family tables below.

⚠️ `manoeuvre_exec_eq_plan` — in OPEN loop the 'executed' manoeuvre is classified from the ego poses, which are the LOGGED poses — so this is `manoeuvre_plan_eq_logged` a second time, not an independent measurement of whether the arm executes what it selects. That question is only askable in CLOSED loop.

## ADE

| metric | refc-base | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `ade_0_2s` | 0.6238 [0.4144, 0.8403] | 0.6444 [0.4392, 0.8497] | -0.0206 [-0.0371, -0.0061] | **separated** |
| `de_0_5s` | 0.0762 [0.0530, 0.1021] | 0.0739 [0.0491, 0.0995] | - |  |
| `de_1s` | 0.3266 [0.2137, 0.4472] | 0.3310 [0.2159, 0.4504] | - |  |
| `de_1_5s` | 0.7591 [0.4961, 1.0329] | 0.7797 [0.5221, 1.0425] | - |  |
| `de_2s` | 1.3333 [0.8929, 1.7956] | 1.3930 [0.9684, 1.8284] | - |  |
| ~~`dist_to_gt_traj_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |


## LONGITUDINAL

| metric | refc-base | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `target_speed_err_ms` | 0.0254 [0.0016, 0.0477] | 0.0118 [-0.0086, 0.0291] | - |  |
| `abs_target_speed_err_ms` | 0.0601 [0.0518, 0.0695] | 0.0526 [0.0430, 0.0613] | +0.0075 [-0.0020, +0.0207] | not separated |
| ~~`executed_speed_err_ms`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| `along_track_ade_m` | 0.5870 [0.3691, 0.8039] | 0.6022 [0.3899, 0.8092] | -0.0152 [-0.0256, -0.0034] | **separated** |
| `along_track_0_5s_m` | 0.0732 [0.0495, 0.0996] | 0.0710 [0.0450, 0.0974] | - |  |
| `along_track_1s_m` | 0.3082 [0.1906, 0.4276] | 0.3082 [0.1862, 0.4303] | - |  |
| `along_track_1_5s_m` | 0.7107 [0.4329, 0.9869] | 0.7244 [0.4516, 0.9916] | - |  |
| `along_track_2s_m` | 1.2557 [0.8020, 1.7118] | 1.3051 [0.8773, 1.7268] | - |  |
| `headway_m` | 48.3899 [31.2727, 62.9700] | 48.3899 [31.2727, 62.9700] | - |  |
| `time_gap_s` | 3.0098 [1.8797, 4.6352] | 3.0098 [1.8797, 4.6352] | - |  |
| `min_headway_m` | -3.2112 [-3.2112, 38.8072] | -3.2112 [-3.2112, 38.8072] | - |  |
| `frac_time_gap_below_1s` | 0.1585 [0.0000, 0.3609] | 0.1585 [0.0000, 0.3609] | - |  |
| `frac_time_gap_below_0_5s` | 0.0854 [0.0000, 0.1714] | 0.0854 [0.0000, 0.1714] | - |  |
| `ttc_s_when_closing` | 3.6086 [2.1948, 5.9828] | 3.6086 [2.1948, 5.9828] | - |  |

* **A `lead_present_rate`** — `0.4823529411764706`
* **B `lead_present_rate`** — `0.4823529411764706`
* **A `ttc_defined_rate`** — `0.9756`
* **B `ttc_defined_rate`** — `0.9756`

## LATERAL

| metric | refc-base | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `heading_err_rad` | 0.0261 [0.0165, 0.0396] | 0.0277 [0.0179, 0.0413] | -0.0016 [-0.0051, +0.0013] | not separated |
| `curvature_err_1pm` | 0.0010 [0.0006, 0.0017] | 0.0011 [0.0006, 0.0017] | -0.0001 [-0.0002, +0.0000] | not separated |
| `yawrate_err_rads` | 0.0091 [0.0061, 0.0127] | 0.0085 [0.0057, 0.0122] | +0.0006 [-0.0001, +0.0016] | not separated |
| ~~`cross_track_abs_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| ~~`cross_track_signed_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | - | ⛔ setup, not result |
| `lateral_ade_m` | 0.1458 [0.1040, 0.1984] | 0.1644 [0.1138, 0.2246] | -0.0186 [-0.0470, +0.0026] | not separated |


## TACTICAL

| metric | refc-base | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `manoeuvre_plan_eq_logged` | 0.4000 [0.1597, 0.6402] | 0.4059 [0.1657, 0.6413] | -0.0059 [-0.0237, +0.0134] | not separated |
| `manoeuvre_exec_eq_plan` ⚠️ | 0.4000 [0.1111, 0.7273] | 0.4000 [0.1111, 0.7273] | - | alias of plan_eq_logged |
| `head_eq_plan` | 0.6059 [0.3432, 0.8429] | 0.5118 [0.2189, 0.7854] | - |  |
| `head_eq_logged` | 0.2294 [0.0579, 0.4444] | 0.1706 [0.0053, 0.3822] | - |  |

* **A `plan_class_share`** — `{"lane_keep": 0.9, "turn_left": 0.0, "turn_right": 0.0, "accelerate": 0.0118, "brake_stop": 0.0882}`
* **B `plan_class_share`** — `{"lane_keep": 0.8882, "turn_left": 0.0, "turn_right": 0.0, "accelerate": 0.0059, "brake_stop": 0.1059}`
* **A `logged_class_share`** — `{"lane_keep": 0.3824, "turn_left": 0.0, "turn_right": 0.0, "accelerate": 0.1765, "brake_stop": 0.4412}`
* **B `logged_class_share`** — `{"lane_keep": 0.3824, "turn_left": 0.0, "turn_right": 0.0, "accelerate": 0.1765, "brake_stop": 0.4412}`
* **A `head_class_share`** — `{"lane_keep": 0.6529, "turn_left": 0.0, "turn_right": 0.3471, "accelerate": 0.0, "brake_stop": 0.0}`
* **B `head_class_share`** — `{"lane_keep": 0.5647, "turn_left": 0.0, "turn_right": 0.4353, "accelerate": 0.0, "brake_stop": 0.0}`
* **A `head_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.3514, "recall": 0.6, "f1": 0.4432, "support_n_true": 65, "n_fires": 111}, "turn_left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "turn_right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 59}, "accelerate": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 30, "n_fires": 0}, "brake_stop": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 75, "n_fires": 0}, "_macro_f1": 0.0886, "_accuracy": 0.2294, "_majority_class_baseline_acc": 0.4412, "_n": 170, "_n_input": 170,`
* **B `head_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.3021, "recall": 0.4462, "f1": 0.3602, "support_n_true": 65, "n_fires": 96}, "turn_left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "turn_right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 74}, "accelerate": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 30, "n_fires": 0}, "brake_stop": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 75, "n_fires": 0}, "_macro_f1": 0.072, "_accuracy": 0.1706, "_majority_class_baseline_acc": 0.4412, "_n": 170, "_n_input": 170`
* **A `plan_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.3922, "recall": 0.9231, "f1": 0.5505, "support_n_true": 65, "n_fires": 153}, "turn_left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "turn_right": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "accelerate": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support_n_true": 30, "n_fires": 2}, "brake_stop": {"precision": 0.5333, "recall": 0.1067, "f1": 0.1778, "support_n_true": 75, "n_fires": 15}, "_macro_f1": 0.1456, "_accuracy": 0.4, "_majority_class_baseline_acc": 0.4412, "_n": 170, "_n_inpu`
* **B `plan_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.3974, "recall": 0.9231, "f1": 0.5556, "support_n_true": 65, "n_fires": 151}, "turn_left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "turn_right": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "accelerate": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support_n_true": 30, "n_fires": 1}, "brake_stop": {"precision": 0.5, "recall": 0.12, "f1": 0.1935, "support_n_true": 75, "n_fires": 18}, "_macro_f1": 0.1498, "_accuracy": 0.4059, "_majority_class_baseline_acc": 0.4412, "_n": 170, "_n_input"`

## STRATEGIC

| metric | refc-base | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| ~~`route_corridor_departure_rate`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| `logged_net_dyaw_rad` | -0.4760 [-0.6522, -0.2802] | -0.4760 [-0.6522, -0.2802] | - |  |
| `route_head_eq_logged` | 0.2712 [0.0156, 0.6818] ⛔degenerate ⚠️echo-unidentifiable | 0.2373 [0.0000, 0.6364] ⛔degenerate ⚠️echo-unidentifiable | +0.0339 [+0.0000, +0.0588] | not separated |
| `route_head_side_eq_graded_proxy` | 0.3706 [0.1447, 0.6191] ⚠️echo-unidentifiable | 0.4176 [0.1562, 0.6883] ⚠️echo-unidentifiable | - |  |
| `route_progress_rel` | 0.9995 [0.9991, 0.9999] | 0.9995 [0.9991, 0.9999] | - |  |

* **A `route_logged_share`** — `{"left": 0.0, "straight": 0.3471, "right": 0.0, "unknown": 0.6529}`
* **B `route_logged_share`** — `{"left": 0.0, "straight": 0.3471, "right": 0.0, "unknown": 0.6529}`
* **A `route_head_share`** — `{"left": 0.2706, "straight": 0.4471, "right": 0.2824, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`
* **B `route_head_share`** — `{"left": 0.2235, "straight": 0.4412, "right": 0.3353, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`
* **A `route_derivation_reason_share`** — `{"gray_zone": 0.6529, "road_following": 0.3471}`
* **B `route_derivation_reason_share`** — `{"gray_zone": 0.6529, "road_following": 0.3471}`
* **A `route_label_valid_rate`** — `0.3471`
* **B `route_label_valid_rate`** — `0.3471`
* **A `route_head_nav_echo_check`** — `{"head_is_deterministic_function_of_nav": null, "identifiable": false, "n_distinct_nav": 1, "n_distinct_head": 3, "nav_to_head_map": {"0": [0, 1, 2]}, "n": 170, "verdict": "UNIDENTIFIABLE \u2014 nav takes only 1 distinct value(s) on these 170 windows, so an echo cannot be distinguished from a constant head (the head takes 3 distinct value(s) here). Needs a scene where the nav command actually varies. NOT a clearance: do not read this as 'not an echo'."}`
* **B `route_head_nav_echo_check`** — `{"head_is_deterministic_function_of_nav": null, "identifiable": false, "n_distinct_nav": 1, "n_distinct_head": 3, "nav_to_head_map": {"0": [0, 1, 2]}, "n": 170, "verdict": "UNIDENTIFIABLE \u2014 nav takes only 1 distinct value(s) on these 170 windows, so an echo cannot be distinguished from a constant head (the head takes 3 distinct value(s) here). Needs a scene where the nav command actually varies. NOT a clearance: do not read this as 'not an echo'."}`
* **A `route_head_side_per_class_PR`** — `{"straight": {"precision": 0.1974, "recall": 0.625, "f1": 0.3, "support_n_true": 24, "n_fires": 76}, "left": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 46}, "right": {"precision": 1.0, "recall": 0.3288, "f1": 0.4948, "support_n_true": 146, "n_fires": 48}, "_macro_f1": 0.2649, "_accuracy": 0.3706, "_majority_class_baseline_acc": 0.8588, "_n": 170, "_n_input": 170, "_n_dropped_unpaired": 0, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`
* **B `route_head_side_per_class_PR`** — `{"straight": {"precision": 0.1867, "recall": 0.5833, "f1": 0.2828, "support_n_true": 24, "n_fires": 75}, "left": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 38}, "right": {"precision": 1.0, "recall": 0.3904, "f1": 0.5616, "support_n_true": 146, "n_fires": 57}, "_macro_f1": 0.2815, "_accuracy": 0.4176, "_majority_class_baseline_acc": 0.8588, "_n": 170, "_n_input": 170, "_n_dropped_unpaired": 0, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`

## Estimator and caveats

* episode-cluster bootstrap over ROLLOUT STARTS. The clusters are disjoint segments of ONE clip, not independent episodes — the interval is the right estimator for the resampling unit available, and the unit is stated here so it is never mistaken for the 40-episode val bootstrap.
* WITHIN-SIM RELATIVE. REF-C open-loop ADE is 1.5157 on these NuRec reconstructions vs 0.4728 on real footage (3.21x OOD). Orderings survive; absolute rates do not.
* ✅ **grad-NCC is the only admissible render metric on these night clips.** PSNR, NCC **and MAE** are RETRACTED here — grad-NCC identifies the correct reference frame 5/5 on every arm while MAE/PSNR manage 1–4/5 with arm-dependent reliability.
* **Scope:** AlpaSim renderer **wire contract** with a gsplat backend. NOT `alpasim_runtime.simulate` — there is no AlpaSim collision/offroad score here.
* ⚠️ Report **precision alongside recall** for every rate, and state the denominator — the per-class PR blocks above carry both.