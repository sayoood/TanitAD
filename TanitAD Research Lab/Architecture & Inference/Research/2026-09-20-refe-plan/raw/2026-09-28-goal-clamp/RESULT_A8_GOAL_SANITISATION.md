> **CORRECTED READ, 2026-09-28 ~13:50 Berlin (coordinator).** The analyzer first read NOT PROVEN (09:08; kept verbatim as `RESULT_A8_GOAL_SANITISATION.as_first_read_0908_NOT_PROVEN.md`). Its only adverse component, `accel_mae_mps2`, was an instrument bug (`eval/proxy_eval.py::fam_direction` matched the substring "acc"; RETRACTION_LOG R26). Re-run with the fix on the same artifacts: ADOPT by the registered rule.
> **NOT DEPLOYED:** PI ruling 2026-09-28 -- "no hd map as input", clarified as no HD-map GEOMETRY for planning, navigation commands OK (the registered route goal stays). The lane-graph fallback is a new map use, so `--sanitize-goal` stays OFF in every evaluation.

# RESULT: SPEC_NAVTEST Amendment 8 (test-time goal sanitisation) -- **ADOPT** (both lower bounds > 0 and no longitudinal / lateral component separates adversely)

Registered 2026-09-28 06:32 (SPEC blob `8ace80cc`); snapshot 015 (md5 d7c59f4f), Amendment 7's repair ON, rule v1, both arms. NAVSIM v1 PDMS = ego pseudo-simulation of an open-loop plan against logged agents (W3's stamp). Every number is read from `result_a8.json`.

| gate | result | detail |
|---|---|---|
| (a) | PASS | `{"rows_on": 422, "triggered": 422, "max_abs_diff_vs_census_m": 0.0}` |
| (b) | PASS | `{"failed": [], "mutations": {"M1": true, "M2": true, "M3": true, "M4": true}, "landed_planner_sha256": "522a87ae2d57acc04442effec131e5085dc8d3595e5683c9b68554af2663a94f"}` |
| (c) | PASS | `{"executed_pose_equals_dumped_pick": "422/422", "pick_equals_rule_argmax": "422/422", "recorded_pick_equals_dump": "422/422", "rule": "navsim_v1", "repair_last_heading": true, "harness_C1_max_abs_delta": 0.0}` |
| (d) | PASS | `{"off": {"status": "PASS", "csv_valid_rows": 422, "seam_rows": 422}, "on": {"status": "PASS", "csv_valid_rows": 422, "seam_rows": 422}}` |
| (e) | PASS | `{"tokens": 422, "ego_identical": 422, "frames_identical": 422, "goal_differs_on_triggered": "422/422"}` |

| read | tokens | logs | PDMS OFF | PDMS ON | D = ON - OFF | 95 % CI |
|---|---|---|---|---|---|---|
| PRIMARY | 422 | 32 | 47.548 | 81.992 | **+34.444** | [+24.904, +43.745] |
| FRESH-LOG | 279 | 26 | 45.643 | 80.902 | **+35.259** | [+23.379, +44.711] |

Estimator: paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927.

## Four families (families6 on both gating seams; strategic UNAVAILABLE in NAVSIM by design)

Adverse separations ON vs OFF (an ON interval entirely on the worse side of OFF's; longitudinal / lateral components, GATING here): `{"adverse": [], "components_compared": 21, "strategic": "UNAVAILABLE in NAVSIM (no strategic decision scored; the arm has no strategic layer)"}`

| seam | speed_mae_mps | speed_bias_mps | along_mae_m | along_final_bias_m | accel_mae_mps2 | heading_mae_deg | yaw_rate_mae_degps | curvature_mae_1pm | cross_mae_m | cross_final_mae_m |
|---|---|---|---|---|---|---|---|---|---|---|
| refe_a8_off | 2.2249 | 1.4782 | 4.0585 | 5.4214 | 1.7462 | 9.5378 | 16.0315 | 0.1230 | 0.9534 | 2.3178 |
| refe_a8_on | 1.0799 | 0.1614 | 1.5780 | 0.7947 | 0.6860 | 4.7995 | 4.5769 | 0.0304 | 0.6969 | 1.7174 |
| refe_a8_clamp150 | 1.2752 | 0.5579 | 1.9862 | 1.6998 | 1.0398 | 9.3275 | 11.0265 | 0.0821 | 1.0013 | 2.5807 |
| refe_a8_straight | 1.0806 | 0.2648 | 1.6447 | 1.6638 | 0.6802 | 6.5305 | 4.7051 | 0.0275 | 0.8360 | 2.0218 |

| seam | lateral_decision | longitudinal_decision | maneuver_5way_collapsed | goal point error (m) | strategic |
|---|---|---|---|---|---|
| refe_a8_off | 0.6280 / 0.4230 | 0.5308 / 0.3099 | 0.4787 / 0.3444 | 9.4524 | UNAVAILABLE |
| refe_a8_on | 0.7725 / 0.5949 | 0.5853 / 0.3802 | 0.6256 / 0.5112 | 4.6419 | UNAVAILABLE |
| refe_a8_clamp150 | 0.6374 / 0.4418 | 0.5284 / 0.3044 | 0.5071 / 0.3771 | 5.8877 | UNAVAILABLE |
| refe_a8_straight | 0.7227 / 0.3841 | 0.5924 / 0.3892 | 0.5071 / 0.3739 | 5.2197 | UNAVAILABLE |

## Reported, not gating

`{"clamp150": {"n": 422, "pdms": 49.92179844366076, "D_vs_off": 2.3737655483804785, "ci95": [-2.084428218453042, 7.083900508465182], "status": "PASS"}, "straight": {"n": 422, "pdms": 71.4141624919943, "D_vs_off": 23.866129596714018, "ci95": [8.760469355549013, 37.57316015370546], "status": "PASS"}, "subscore_deltas_on_minus_off_x100": {"no_at_fault_collisions": 15.04739336492891, "drivable_area_compliance": 24.881516587677723, "ego_progress": 31.11310693875442, "time_to_collision_within_bound": 21.09004739336493, "comfort": 12.322274881516588, "driving_direction_compliance": 12.322274881516588}, "fallback_kinds_on": {"lane": 422}, "goal_p2_m_median": {"off": 151.00751621598505, "on": 60.035085976071045, "clamp150": 149.99998408379128, "straight": 66.21856689453125}}`

Selection tokens (exploratory, the 53 of the 1,123 the trigger fires on): `{"what": "EXPLORATORY (selection tokens): the sanitised goal on the 1,123's triggered tokens, CPU decode of the fp32 eval cache; OFF = M6's refe_m6_base_on (same snapshot, same repair)", "n": 53, "control_picks_equal_gpu_dump": "53/53", "control_goal_equals_eval_cache_goal": true, "harness": {"status": "PASS", "rows": 53}, "pdms_off_base_on": 35.522794679550955, "pdms_on": 81.60690057966369, "D_on_minus_off": 46.08410590011273, "ci95": [12.565753279025575, 59.66387441978474], "n_scored": 53, "n_logs": 9, "estimator": "paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927"}`

Analysed 2026-09-28T13:42:24 (Europe/Berlin).
