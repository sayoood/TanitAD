# OPEN-LOOP: flagship-v1 with-objects vs empty-road (identical logged poses)

**Arms:** `flagship-v1` (A) vs `flagship-v1` (B) · windows A 170 / clusters 9 · B 170 / 9 · paired windows 170

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
| flagship-v1 | ADE | `dist_to_gt_traj_m` | 0 | 1e-06 | ✅ | literally abs(cross_track) in cl_metrics — the same measurement under a second name |
| flagship-v1 | LONGITUDINAL | `executed_speed_err_ms` | 0 | 1e-06 | ✅ | the ego speed IS the logged speed; no controller runs |
| flagship-v1 | LATERAL | `cross_track_abs_m` | 0 | 1e-06 | ✅ | the ego pose IS the logged pose, so its distance to the logged polyline is zero by construction |
| flagship-v1 | LATERAL | `cross_track_signed_m` | 0 | 1e-06 | ✅ | same quantity, signed |
| flagship-v1 | STRATEGIC | `route_corridor_departure_rate` | 0 | 1e-06 | ✅ | departure is |cross_track| > 2 m, and cross_track is pinned at 0 |

All pinned metrics confirmed at float tolerance. They are **setup, not result**, and are struck through in the family tables below.

⚠️ `manoeuvre_exec_eq_plan` — in OPEN loop the 'executed' manoeuvre is classified from the ego poses, which are the LOGGED poses — so this is `manoeuvre_plan_eq_logged` a second time, not an independent measurement of whether the arm executes what it selects. That question is only askable in CLOSED loop.

## ADE

| metric | flagship-v1 | flagship-v1 | paired Δ (A−B) | |
|---|---|---|---|---|
| `ade_0_2s` | 4.7304 [3.2677, 6.2121] | 7.1479 [6.0645, 8.3711] | -2.4175 [-3.4848, -1.3549] | **separated** |
| `de_0_5s` | 1.9850 [1.4345, 2.5180] | 2.9434 [2.5306, 3.3662] | - |  |
| `de_1s` | 3.8487 [2.7133, 4.9675] | 5.7743 [4.9488, 6.6760] | - |  |
| `de_1_5s` | 5.6915 [3.9162, 7.4828] | 8.6018 [7.2978, 10.0661] | - |  |
| `de_2s` | 7.3963 [4.9236, 9.9378] | 11.2721 [9.4975, 13.4087] | - |  |
| ~~`dist_to_gt_traj_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |


## LONGITUDINAL

| metric | flagship-v1 | flagship-v1 | paired Δ (A−B) | |
|---|---|---|---|---|
| `target_speed_err_ms` | -4.0316 [-5.0732, -2.9663] | -5.9517 [-6.7660, -5.1379] | - |  |
| `abs_target_speed_err_ms` | 4.0316 [2.9663, 5.0732] | 5.9517 [5.1379, 6.7660] | -1.9201 [-2.7510, -1.0874] | **separated** |
| ~~`executed_speed_err_ms`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| `along_track_ade_m` | 4.7119 [3.2407, 6.1994] | 7.1374 [6.0540, 8.3617] | -2.4255 [-3.5012, -1.3617] | **separated** |
| `along_track_0_5s_m` | 1.9845 [1.4340, 2.5177] | 2.9431 [2.5303, 3.3659] | - |  |
| `along_track_1s_m` | 3.8445 [2.7086, 4.9648] | 5.7712 [4.9461, 6.6737] | - |  |
| `along_track_1_5s_m` | 5.6722 [3.8908, 7.4682] | 8.5910 [7.2860, 10.0577] | - |  |
| `along_track_2s_m` | 7.3465 [4.8578, 9.9016] | 11.2445 [9.4669, 13.3860] | - |  |
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

| metric | flagship-v1 | flagship-v1 | paired Δ (A−B) | |
|---|---|---|---|---|
| `heading_err_rad` | 0.0434 [0.0331, 0.0545] | 0.0469 [0.0366, 0.0554] | -0.0035 [-0.0139, +0.0052] | not separated |
| `curvature_err_1pm` | 0.0018 [0.0014, 0.0024] | 0.0028 [0.0018, 0.0037] | -0.0009 [-0.0019, -0.0001] | **separated** |
| `yawrate_err_rads` | 0.0293 [0.0208, 0.0386] | 0.0425 [0.0312, 0.0547] | -0.0132 [-0.0226, -0.0062] | **separated** |
| ~~`cross_track_abs_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| ~~`cross_track_signed_m`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | - | ⛔ setup, not result |
| `lateral_ade_m` | 0.2394 [0.1781, 0.3159] | 0.3077 [0.2478, 0.3732] | -0.0683 [-0.1338, -0.0029] | **separated** |


## TACTICAL

| metric | flagship-v1 | flagship-v1 | paired Δ (A−B) | |
|---|---|---|---|---|
| `manoeuvre_plan_eq_logged` | 0.4353 [0.1610, 0.7133] | 0.4412 [0.1302, 0.7707] | -0.0059 [-0.0710, +0.0785] | not separated |
| `manoeuvre_exec_eq_plan` ⚠️ | 0.4000 [0.1000, 0.7778] | 0.4000 [0.1000, 0.7778] | - | alias of plan_eq_logged |
| `head_eq_plan` | 0.1824 [0.0157, 0.4261] | 0.1529 [0.0419, 0.2789] | - |  |
| `head_eq_logged` | 0.0353 [0.0053, 0.0738] | 0.0765 [0.0134, 0.1728] | - |  |

* **A `plan_class_share`** — `{"lane_keep": 0.0647, "turn_left": 0.0, "turn_right": 0.0059, "accelerate": 0.0, "brake_stop": 0.9294}`
* **B `plan_class_share`** — `{"lane_keep": 0.0, "turn_left": 0.0, "turn_right": 0.0235, "accelerate": 0.0, "brake_stop": 0.9765}`
* **A `logged_class_share`** — `{"lane_keep": 0.3824, "turn_left": 0.0, "turn_right": 0.0, "accelerate": 0.1765, "brake_stop": 0.4412}`
* **B `logged_class_share`** — `{"lane_keep": 0.3824, "turn_left": 0.0, "turn_right": 0.0, "accelerate": 0.1765, "brake_stop": 0.4412}`
* **A `head_class_share`** — `{"lane_keep": 0.0824, "turn_left": 0.0176, "turn_right": 0.3765, "accelerate": 0.3647, "brake_stop": 0.1588}`
* **B `head_class_share`** — `{"lane_keep": 0.0706, "turn_left": 0.0647, "turn_right": 0.4706, "accelerate": 0.2647, "brake_stop": 0.1294}`
* **A `head_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.1429, "recall": 0.0308, "f1": 0.0506, "support_n_true": 65, "n_fires": 14}, "turn_left": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 3}, "turn_right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 64}, "accelerate": {"precision": 0.0645, "recall": 0.1333, "f1": 0.087, "support_n_true": 30, "n_fires": 62}, "brake_stop": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support_n_true": 75, "n_fires": 27}, "_macro_f1": 0.0275, "_accuracy": 0.0353, "_majority_class_baseline_acc": 0.4412, "_n": 170, "_n_inp`
* **B `head_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.25, "recall": 0.0462, "f1": 0.0779, "support_n_true": 65, "n_fires": 12}, "turn_left": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 11}, "turn_right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 80}, "accelerate": {"precision": 0.0444, "recall": 0.0667, "f1": 0.0533, "support_n_true": 30, "n_fires": 45}, "brake_stop": {"precision": 0.3636, "recall": 0.1067, "f1": 0.1649, "support_n_true": 75, "n_fires": 22}, "_macro_f1": 0.0592, "_accuracy": 0.0765, "_majority_class_baseline_acc": 0.4412, "_n": 170`
* **A `plan_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": 0.4545, "recall": 0.0769, "f1": 0.1316, "support_n_true": 65, "n_fires": 11}, "turn_left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "turn_right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 1}, "accelerate": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 30, "n_fires": 0}, "brake_stop": {"precision": 0.4367, "recall": 0.92, "f1": 0.5923, "support_n_true": 75, "n_fires": 158}, "_macro_f1": 0.1448, "_accuracy": 0.4353, "_majority_class_baseline_acc": 0.4412, "_n": 170, "_n_inp`
* **B `plan_vs_logged_per_class_PR`** — `{"lane_keep": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 65, "n_fires": 0}, "turn_left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "turn_right": {"precision": 0.0, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 4}, "accelerate": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 30, "n_fires": 0}, "brake_stop": {"precision": 0.4518, "recall": 1.0, "f1": 0.6224, "support_n_true": 75, "n_fires": 166}, "_macro_f1": 0.1245, "_accuracy": 0.4412, "_majority_class_baseline_acc": 0.4412, "_n": 170, "_n_input": 170, `

## STRATEGIC

| metric | flagship-v1 | flagship-v1 | paired Δ (A−B) | |
|---|---|---|---|---|
| ~~`route_corridor_departure_rate`~~ | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | ⛔ setup, not result |
| `logged_net_dyaw_rad` | -0.4760 [-0.6522, -0.2802] | -0.4760 [-0.6522, -0.2802] | - |  |
| `route_head_eq_logged` | 1.0000 [1.0000, 1.0000] ⛔degenerate ⚠️echo-unidentifiable | 1.0000 [1.0000, 1.0000] ⛔degenerate ⚠️echo-unidentifiable | +0.0000 [+0.0000, +0.0000] | not separated |
| `route_head_side_eq_graded_proxy` | 0.1412 [0.0000, 0.3905] ⚠️echo-unidentifiable | 0.1412 [0.0000, 0.3905] ⚠️echo-unidentifiable | - |  |
| `route_progress_rel` | 0.9995 [0.9991, 0.9999] | 0.9995 [0.9991, 0.9999] | - |  |

* **A `route_logged_share`** — `{"left": 0.0, "straight": 0.3471, "right": 0.0, "unknown": 0.6529}`
* **B `route_logged_share`** — `{"left": 0.0, "straight": 0.3471, "right": 0.0, "unknown": 0.6529}`
* **A `route_head_share`** — `{"left": 0.0, "straight": 1.0, "right": 0.0, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`
* **B `route_head_share`** — `{"left": 0.0, "straight": 1.0, "right": 0.0, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`
* **A `route_derivation_reason_share`** — `{"gray_zone": 0.6529, "road_following": 0.3471}`
* **B `route_derivation_reason_share`** — `{"gray_zone": 0.6529, "road_following": 0.3471}`
* **A `route_label_valid_rate`** — `0.3471`
* **B `route_label_valid_rate`** — `0.3471`
* **A `route_head_nav_echo_check`** — `{"head_is_deterministic_function_of_nav": null, "identifiable": false, "n_distinct_nav": 1, "n_distinct_head": 1, "nav_to_head_map": {"0": [1]}, "n": 170, "verdict": "UNIDENTIFIABLE \u2014 nav takes only 1 distinct value(s) on these 170 windows, so an echo cannot be distinguished from a constant head (the head takes 1 distinct value(s) here). Needs a scene where the nav command actually varies. NOT a clearance: do not read this as 'not an echo'."}`
* **B `route_head_nav_echo_check`** — `{"head_is_deterministic_function_of_nav": null, "identifiable": false, "n_distinct_nav": 1, "n_distinct_head": 1, "nav_to_head_map": {"0": [1]}, "n": 170, "verdict": "UNIDENTIFIABLE \u2014 nav takes only 1 distinct value(s) on these 170 windows, so an echo cannot be distinguished from a constant head (the head takes 1 distinct value(s) here). Needs a scene where the nav command actually varies. NOT a clearance: do not read this as 'not an echo'."}`
* **A `route_head_side_per_class_PR`** — `{"straight": {"precision": 0.1412, "recall": 1.0, "f1": 0.2474, "support_n_true": 24, "n_fires": 170}, "left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "right": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 146, "n_fires": 0}, "_macro_f1": 0.0825, "_accuracy": 0.1412, "_majority_class_baseline_acc": 0.8588, "_n": 170, "_n_input": 170, "_n_dropped_unpaired": 0, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`
* **B `route_head_side_per_class_PR`** — `{"straight": {"precision": 0.1412, "recall": 1.0, "f1": 0.2474, "support_n_true": 24, "n_fires": 170}, "left": {"precision": null, "recall": null, "f1": 0.0, "support_n_true": 0, "n_fires": 0}, "right": {"precision": null, "recall": 0.0, "f1": 0.0, "support_n_true": 146, "n_fires": 0}, "_macro_f1": 0.0825, "_accuracy": 0.1412, "_majority_class_baseline_acc": 0.8588, "_n": 170, "_n_input": 170, "_n_dropped_unpaired": 0, "NAV_ECHO_UNIDENTIFIABLE": true, "nav_echo_note": "nav takes only 1 distinct value(s) here, so the echo guard could not run. This number is NOT cleared of circularity."}`

## Estimator and caveats

* episode-cluster bootstrap over ROLLOUT STARTS. The clusters are disjoint segments of ONE clip, not independent episodes — the interval is the right estimator for the resampling unit available, and the unit is stated here so it is never mistaken for the 40-episode val bootstrap.
* WITHIN-SIM RELATIVE. REF-C open-loop ADE is 1.5157 on these NuRec reconstructions vs 0.4728 on real footage (3.21x OOD). Orderings survive; absolute rates do not.
* ✅ **grad-NCC is the only admissible render metric on these night clips.** PSNR, NCC **and MAE** are RETRACTED here — grad-NCC identifies the correct reference frame 5/5 on every arm while MAE/PSNR manage 1–4/5 with arm-dependent reliability.
* **Scope:** AlpaSim renderer **wire contract** with a gsplat backend. NOT `alpasim_runtime.simulate` — there is no AlpaSim collision/offroad score here.
* ⚠️ Report **precision alongside recall** for every rate, and state the denominator — the per-class PR blocks above carry both.