# OPEN-LOOP panel — flagship v1 vs REF-C base, scene 00040136, objects

**Arms:** `flagship-v1` (A) vs `refc-base` (B) · windows A 170 / clusters 9 · B 170 / 9 · paired windows 170

⭐ **OPEN LOOP** — the ego follows the LOGGED trajectory; every frame is rendered at the pose the rig actually had and the model's plan is scored against the log's own future motion. **The plan is never executed.** This isolates perception + prediction from the control drift that is confounded with them in every closed-loop number.

`f_eff` A = 266.0139895990537, B = 266.0139895990537 (the canonicalization self-check against `F_REF`; both arms are 256px SQUARE and `_BasePolicy.canon` asserts `(1, 8, 9, 256, 256)` before any forward pass).

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
| `ade_0_2s` | 4.7304 [3.2677, 6.2121] | 0.6238 [0.4144, 0.8403] | +4.1066 [+2.6228, +5.6146] | **separated** |
| `de_0_5s` | 1.9850 [1.4345, 2.5180] | 0.0762 [0.0530, 0.1021] | - |  |
| `de_1s` | 3.8487 [2.7133, 4.9675] | 0.3266 [0.2137, 0.4472] | - |  |
| `de_1_5s` | 5.6915 [3.9162, 7.4828] | 0.7591 [0.4961, 1.0329] | - |  |
| `de_2s` | 7.3963 [4.9236, 9.9378] | 1.3333 [0.8929, 1.7956] | - |  |
| ~~`dist_to_gt_traj_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |


## LONGITUDINAL

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `target_speed_err_ms` | -4.0316 [-5.0732, -2.9663] | 0.0254 [0.0016, 0.0477] | - |  |
| `abs_target_speed_err_ms` | 4.0316 [2.9663, 5.0732] | 0.0601 [0.0518, 0.0695] | +3.9716 [+2.9051, +5.0106] | **separated** |
| ~~`executed_speed_err_ms`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| `along_track_ade_m` | 4.7119 [3.2407, 6.1994] | 0.5870 [0.3691, 0.8039] | +4.1250 [+2.6492, +5.6323] | **separated** |
| `along_track_0_5s_m` | 1.9845 [1.4340, 2.5177] | 0.0732 [0.0495, 0.0996] | - |  |
| `along_track_1s_m` | 3.8445 [2.7086, 4.9648] | 0.3082 [0.1906, 0.4276] | - |  |
| `along_track_1_5s_m` | 5.6722 [3.8908, 7.4682] | 0.7107 [0.4329, 0.9869] | - |  |
| `along_track_2s_m` | 7.3465 [4.8578, 9.9016] | 1.2557 [0.8020, 1.7118] | - |  |
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

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `heading_err_rad` | 0.0434 [0.0331, 0.0545] | 0.0261 [0.0165, 0.0396] | +0.0173 [+0.0067, +0.0289] | **separated** |
| `curvature_err_1pm` | 0.0018 [0.0014, 0.0024] | 0.0010 [0.0006, 0.0017] | +0.0008 [+0.0005, +0.0012] | **separated** |
| `yawrate_err_rads` | 0.0293 [0.0208, 0.0386] | 0.0091 [0.0061, 0.0127] | +0.0201 [+0.0102, +0.0304] | **separated** |
| ~~`cross_track_abs_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| ~~`cross_track_signed_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | - | ⛔ setup, not result |
| `lateral_ade_m` | 0.2394 [0.1781, 0.3159] | 0.1458 [0.1040, 0.1984] | +0.0936 [+0.0374, +0.1748] | **separated** |


## TACTICAL

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| `manoeuvre_plan_eq_logged` | 0.4353 [0.1610, 0.7133] | 0.4000 [0.1597, 0.6402] | +0.0353 [-0.4354, +0.5118] | not separated |
| `manoeuvre_exec_eq_plan` ⚠️ | 0.4000 [0.1000, 0.7778] | 0.4000 [0.1111, 0.7273] | - | alias of plan_eq_logged |
| `head_eq_plan` | 0.1824 [0.0157, 0.4261] | 0.6059 [0.3432, 0.8429] | - |  |
| `head_eq_logged` | 0.0353 [0.0053, 0.0738] | 0.2294 [0.0579, 0.4444] | - |  |

* **A `plan_class_share`** — `{"lane_keep": 0.0647, "turn_left": 0.0, "turn_right": 0.0059, "accelerate": 0.0, "brake_stop": 0.9294}`
* **B `plan_class_share`** — `{"lane_keep": 0.9, "turn_left": 0.0, "turn_right": 0.0, "accelerate": 0.0118, "brake_stop": 0.0882}`
* **A `logged_class_share`** — `{"lane_keep": 0.3824, "turn_left": 0.0, "turn_right": 0.0, "accelerate": 0.1765, "brake_stop": 0.4412}`
* **B `logged_class_share`** — `{"lane_keep": 0.3824, "turn_left": 0.0, "turn_right": 0.0, "accelerate": 0.1765, "brake_stop": 0.4412}`
* **A `head_class_share`** — `{"lane_keep": 0.0824, "turn_left": 0.0176, "turn_right": 0.3765, "accelerate": 0.3647, "brake_stop": 0.1588}`
* **B `head_class_share`** — `{"lane_keep": 0.6529, "turn_left": 0.0, "turn_right": 0.3471, "accelerate": 0.0, "brake_stop": 0.0}`
* **A `head_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.1429, "recall": 0.0308, "f1": 0.0506, "support_n_true": 65, "n_fires": 14}, "turn_left": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 3}, "turn_right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 64}, "accelerate": {"precision": 0.0645, "recall": 0.1333, "f1": 0.087, "support_n_true": 30, "n_fires": 62}, "brake_stop": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support_n_true": 75, "n_fires": 27}, "_macro_f1": 0.0275, "_accuracy": 0.0353, "_majority_class_baseline_acc": 0.4412, "_n": 170, "_n_inp`
* **B `head_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.3514, "recall": 0.6, "f1": 0.4432, "support_n_true": 65, "n_fires": 111}, "turn_left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "turn_right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 59}, "accelerate": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 30, "n_fires": 0}, "brake_stop": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 75, "n_fires": 0}, "_macro_f1": 0.0886, "_accuracy": 0.2294, "_majority_class_baseline_acc": 0.4412, "_n": 170, "_n_input": 170,`
* **A `plan_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.4545, "recall": 0.0769, "f1": 0.1316, "support_n_true": 65, "n_fires": 11}, "turn_left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "turn_right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 1}, "accelerate": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 30, "n_fires": 0}, "brake_stop": {"precision": 0.4367, "recall": 0.92, "f1": 0.5923, "support_n_true": 75, "n_fires": 158}, "_macro_f1": 0.1448, "_accuracy": 0.4353, "_majority_class_baseline_acc": 0.4412, "_n": 170, "_n_inp`
* **B `plan_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.3922, "recall": 0.9231, "f1": 0.5505, "support_n_true": 65, "n_fires": 153}, "turn_left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "turn_right": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "accelerate": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support_n_true": 30, "n_fires": 2}, "brake_stop": {"precision": 0.5333, "recall": 0.1067, "f1": 0.1778, "support_n_true": 75, "n_fires": 15}, "_macro_f1": 0.1456, "_accuracy": 0.4, "_majority_class_baseline_acc": 0.4412, "_n": 170, "_n_inpu`

## STRATEGIC

| metric | flagship-v1 | refc-base | paired Δ (A−B) | |
|---|---|---|---|---|
| ~~`route_corridor_departure_rate`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| `logged_net_dyaw_rad` | -0.4760 [-0.6522, -0.2802] | -0.4760 [-0.6522, -0.2802] | - |  |
| `route_head_eq_logged` | 1.0000 [1.0000, 1.0000] ⛔degenerate ⚠️echo-unidentifiable | 0.2712 [0.0156, 0.6818] ⛔degenerate ⚠️echo-unidentifiable | +0.7288 [+0.3182, +0.9844] | **separated** |
| `route_head_side_eq_graded_proxy` | 0.1412 [0.0000, 0.3905] ⚠️echo-unidentifiable | 0.3706 [0.1447, 0.6191] ⚠️echo-unidentifiable | - |  |
| `route_progress_rel` | 0.9995 [0.9991, 0.9999] | 0.9995 [0.9991, 0.9999] | - |  |

* **A `route_logged_share`** — `{"left": 0.0, "straight": 0.3471, "right": 0.0, "unknown": 0.6529}`
* **B `route_logged_share`** — `{"left": 0.0, "straight": 0.3471, "right": 0.0, "unknown": 0.6529}`
* **A `route_head_share`** — `{"left": 0.0, "straight": 1.0, "right": 0.0, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`
* **B `route_head_share`** — `{"left": 0.2706, "straight": 0.4471, "right": 0.2824, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`
* **A `route_derivation_reason_share`** — `{"gray_zone": 0.6529, "road_following": 0.3471}`
* **B `route_derivation_reason_share`** — `{"gray_zone": 0.6529, "road_following": 0.3471}`
* **A `route_label_valid_rate`** — `0.3471`
* **B `route_label_valid_rate`** — `0.3471`
* **A `route_head_nav_echo_check`** — `{"head_is_deterministic_function_of_nav": null, "identifiable": false, "n_distinct_nav": 1, "n_distinct_head": 1, "nav_to_head_map": {"0": [1]}, "n": 170, "verdict": "UNIDENTIFIABLE \u2014 nav takes only 1 distinct value(s) on these 170 windows, so an echo cannot be distinguished from a constant head (the head takes 1 distinct value(s) here). Needs a scene where the nav command actually varies. NOT a clearance: do not read this as 'not an echo'."}`
* **B `route_head_nav_echo_check`** — `{"head_is_deterministic_function_of_nav": null, "identifiable": false, "n_distinct_nav": 1, "n_distinct_head": 3, "nav_to_head_map": {"0": [0, 1, 2]}, "n": 170, "verdict": "UNIDENTIFIABLE \u2014 nav takes only 1 distinct value(s) on these 170 windows, so an echo cannot be distinguished from a constant head (the head takes 3 distinct value(s) here). Needs a scene where the nav command actually varies. NOT a clearance: do not read this as 'not an echo'."}`
* **A `route_head_side_per_class_PR`** — `{"straight": {"precision": 0.1412, "recall": 1.0, "f1": 0.2474, "support_n_true": 24, "n_fires": 170}, "left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "right": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 146, "n_fires": 0}, "_macro_f1": 0.0825, "_accuracy": 0.1412, "_majority_class_baseline_acc": 0.8588, "_n": 170, "_n_input": 170, "_n_dropped_unpaired": 0, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`
* **B `route_head_side_per_class_PR`** — `{"straight": {"precision": 0.1974, "recall": 0.625, "f1": 0.3, "support_n_true": 24, "n_fires": 76}, "left": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 46}, "right": {"precision": 1.0, "recall": 0.3288, "f1": 0.4948, "support_n_true": 146, "n_fires": 48}, "_macro_f1": 0.2649, "_accuracy": 0.3706, "_majority_class_baseline_acc": 0.8588, "_n": 170, "_n_input": 170, "_n_dropped_unpaired": 0, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`

## Estimator and caveats

* episode-cluster bootstrap over ROLLOUT STARTS. The clusters are disjoint segments of ONE clip, not independent episodes — the interval is the right estimator for the resampling unit available, and the unit is stated here so it is never mistaken for the 40-episode val bootstrap.
* WITHIN-SIM RELATIVE. REF-C open-loop ADE is 1.5157 on these NuRec reconstructions vs 0.4728 on real footage (3.21x OOD). Orderings survive; absolute rates do not.
* ✅ **grad-NCC is the only admissible render metric on these night clips.** PSNR, NCC **and MAE** are RETRACTED here — grad-NCC identifies the correct reference frame 5/5 on every arm while MAE/PSNR manage 1–4/5 with arm-dependent reliability.
* **Scope:** AlpaSim renderer **wire contract** with a gsplat backend. NOT `alpasim_runtime.simulate` — there is no AlpaSim collision/offroad score here.
* ⚠️ Report **precision alongside recall** for every rate, and state the denominator — the per-class PR blocks above carry both.