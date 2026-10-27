Contrast: DAG_s1 - M2b_s1. Tier: T1-class open loop (own held outputs; no recorded future action). Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes.

### TACTICAL, plan-derived decisions and goal point

| cell | DAG_s1 | M2b_s1 | DAG_s1 - M2b_s1 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.plan_derived.lateral_decision.accuracy` | 0.8370 | 0.8287 | +0.0083 [-0.0079, +0.0279] | 2,399 / 141 | no |
| `tactical.plan_derived.lateral_decision.kappa` | 0.6478 | 0.6324 | +0.0153 [-0.0148, +0.0500] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.accuracy` | 0.5607 | 0.5919 | -0.0313 [-0.0600, -0.0029] | 2,399 / 141 | **yes** (DAG_s1 worse) |
| `tactical.plan_derived.longitudinal_decision.kappa` | 0.3304 | 0.3698 | -0.0393 [-0.0802, +0.0009] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.accuracy` | 0.5602 | 0.5782 | -0.0179 [-0.0429, +0.0088] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.kappa` | 0.4333 | 0.4514 | -0.0181 [-0.0491, +0.0143] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_point_error_m` | 6.0567 | 5.8582 | +0.1985 [-0.0381, +0.4432] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_long_bias_m` | -1.2488 | -0.3763 | -0.8725 [-1.1927, -0.5645] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_lat_bias_m` | -0.0602 | 0.0185 | -0.0788 [-0.2495, +0.0973] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_bearing_mae_deg` | 3.9731 | 3.5884 | +0.3847 [-0.1485, +1.2492] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_bearing_bias_deg` | -0.8151 | -0.3515 | -0.4636 [-1.3801, +0.1463] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_range_ratio` | 0.9861 | 1.0220 | -0.0359 [-0.0579, -0.0107] | 2,367 / 141 | **yes** |

### TACTICAL, heads (lat / lon / goal)

| cell | DAG_s1 | M2b_s1 | DAG_s1 - M2b_s1 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.heads.lateral.macro_f1` | 0.8351 | 0.8292 | +0.0059 [-0.0188, +0.0374] | 2,107 / 139 | no |
| `tactical.heads.lateral.accuracy` | 0.9535 | 0.9544 | -0.0009 [-0.0079, +0.0051] | 2,107 / 139 | no |
| `tactical.heads.lateral.kappa` | 0.7321 | 0.7393 | -0.0071 [-0.0518, +0.0302] | 2,107 / 139 | no |
| `tactical.heads.longitudinal.macro_f1` | 0.5212 | 0.5095 | +0.0117 [-0.0368, +0.0600] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.accuracy` | 0.6295 | 0.6339 | -0.0045 [-0.0347, +0.0238] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.kappa` | 0.5201 | 0.5209 | -0.0008 [-0.0383, +0.0350] | 2,240 / 136 | no |
| `tactical.heads.goal.macro_f1` | 0.2747 | 0.2621 | +0.0126 [-0.0187, +0.0457] | 2,365 / 139 | no |
| `tactical.heads.goal.token_accuracy` | 0.9581 | 0.9582 | -0.0001 [-0.0020, +0.0019] | 2,365 / 139 | no |
| `tactical.heads.goal.per_token.FOLLOW_LANE` | 0.9727 | 0.9735 | -0.0008 [-0.0048, +0.0028] | 2,107 labelled, 1910 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_L` | 0.7767 | 0.7708 | +0.0059 [-0.0593, +0.0674] | 2,320 labelled, 47 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_R` | 0.7078 | 0.7442 | -0.0364 [-0.1078, +0.0186] | 2,320 labelled, 150 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_L` | 0.5000 | 0.2667 | +0.2333 (no interval) | 2,107 labelled, 11 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 2,107 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD` | 0.1474 | 0.1154 | +0.0320 [+0.0000, +0.0764] | 695 labelled, 85 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.STOP_POINT` | 0.2857 | 0.2156 | +0.0701 [-0.0210, +0.1600] | 2,246 labelled, 131 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.SPEED_BAND` | UNAVAILABLE | | | 0 labelled, 0 pos | no positive label for this token among its 0 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.CORRIDOR_OFFSET` | 0.3030 | 0.2754 | +0.0276 [-0.0157, +0.0712] | 695 labelled, 120 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.EVADE_IN_CORRIDOR` | 0.4773 | 0.4624 | +0.0148 [-0.1001, +0.1290] | 695 labelled, 110 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.OVERTAKE_VEHICLE` | 0.0000 | 0.2609 | -0.2609 (no interval) | 695 labelled, 20 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.MERGE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 25 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.GAP_TARGET` | 0.0328 | 0.0328 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 60 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.REACT_ON_ONCOMING` | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 50 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TAKE_EXIT_L` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TAKE_EXIT_R` | 0.0000 | 0.0625 | -0.0625 (no interval) | 695 labelled, 30 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_RED` | 0.5849 | 0.5275 | +0.0574 [-0.1202, +0.3696] | 720 labelled, 65 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_YELLOW` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_GREEN` | 0.4314 | 0.2727 | +0.1586 [+0.0044, +0.3526] | 695 labelled, 75 pos; 2,365 / 139 | **yes** (DAG_s1 better) |
| `tactical.heads.goal.per_token.LANE_CHANGE_L` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.LANE_CHANGE_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |

### STRATEGIC, S's checkpoint error

| cell | DAG_s1 | M2b_s1 | DAG_s1 - M2b_s1 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `strategic.position_error_m.30` | 6.9723 | 6.9802 | -0.0079 [-0.0748, +0.0673] | 2,322 / 137 | no |
| `strategic.position_error_m.50` | 7.9593 | 7.9384 | +0.0209 [-0.1190, +0.2089] | 2,319 / 137 | no |
| `strategic.position_error_m.80` | 11.2166 | 11.2141 | +0.0025 [-0.2667, +0.2615] | 2,294 / 136 | no |
| `strategic.heading_error_deg.30` | 4.1296 | 4.0437 | +0.0859 [-0.0974, +0.2608] | 2,322 / 137 | no |
| `strategic.heading_error_deg.50` | 6.9929 | 6.7187 | +0.2742 [-0.1693, +0.8685] | 2,319 / 137 | no |
| `strategic.heading_error_deg.80` | 13.3560 | 13.6014 | -0.2453 [-0.7404, +0.2492] | 2,294 / 136 | no |
