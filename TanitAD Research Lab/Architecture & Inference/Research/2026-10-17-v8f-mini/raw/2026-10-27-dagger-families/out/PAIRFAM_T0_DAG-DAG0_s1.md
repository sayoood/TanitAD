Contrast: DAG_s1 - DAG0_s1. Tier: T1-class open loop (own held outputs; no recorded future action). Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes.

### TACTICAL, plan-derived decisions and goal point

| cell | DAG_s1 | DAG0_s1 | DAG_s1 - DAG0_s1 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.plan_derived.lateral_decision.accuracy` | 0.8370 | 0.8479 | -0.0108 [-0.0200, -0.0025] | 2,399 / 141 | **yes** (DAG_s1 worse) |
| `tactical.plan_derived.lateral_decision.kappa` | 0.6478 | 0.6690 | -0.0212 [-0.0400, -0.0047] | 2,399 / 141 | **yes** (DAG_s1 worse) |
| `tactical.plan_derived.longitudinal_decision.accuracy` | 0.5607 | 0.5673 | -0.0067 [-0.0171, +0.0029] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.kappa` | 0.3304 | 0.3393 | -0.0089 [-0.0240, +0.0055] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.accuracy` | 0.5602 | 0.5652 | -0.0050 [-0.0154, +0.0050] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.kappa` | 0.4333 | 0.4389 | -0.0056 [-0.0185, +0.0071] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_point_error_m` | 6.0567 | 5.9556 | +0.1011 [+0.0071, +0.2008] | 2,399 / 141 | **yes** (DAG_s1 worse) |
| `tactical.goal_setting.goal_long_bias_m` | -1.2488 | -1.3154 | +0.0665 [-0.0452, +0.1722] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_lat_bias_m` | -0.0602 | -0.1506 | +0.0904 [+0.0089, +0.1738] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_bearing_mae_deg` | 3.9731 | 3.8303 | +0.1428 [-0.0376, +0.3917] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_bearing_bias_deg` | -0.8151 | -0.0549 | -0.7602 [-2.1853, +0.0291] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_range_ratio` | 0.9861 | 0.9903 | -0.0043 [-0.0114, +0.0019] | 2,367 / 141 | no |

### TACTICAL, heads (lat / lon / goal)

| cell | DAG_s1 | DAG0_s1 | DAG_s1 - DAG0_s1 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.heads.lateral.macro_f1` | 0.8351 | 0.8442 | -0.0091 [-0.0310, +0.0100] | 2,107 / 139 | no |
| `tactical.heads.lateral.accuracy` | 0.9535 | 0.9568 | -0.0033 [-0.0096, +0.0019] | 2,107 / 139 | no |
| `tactical.heads.lateral.kappa` | 0.7321 | 0.7563 | -0.0242 [-0.0693, +0.0111] | 2,107 / 139 | no |
| `tactical.heads.longitudinal.macro_f1` | 0.5212 | 0.5527 | -0.0315 [-0.0702, +0.0025] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.accuracy` | 0.6295 | 0.6384 | -0.0089 [-0.0246, +0.0059] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.kappa` | 0.5201 | 0.5315 | -0.0114 [-0.0297, +0.0069] | 2,240 / 136 | no |
| `tactical.heads.goal.macro_f1` | 0.2747 | 0.2750 | -0.0002 [-0.0486, +0.0211] | 2,365 / 139 | no |
| `tactical.heads.goal.token_accuracy` | 0.9581 | 0.9586 | -0.0005 [-0.0018, +0.0008] | 2,365 / 139 | no |
| `tactical.heads.goal.per_token.FOLLOW_LANE` | 0.9727 | 0.9737 | -0.0010 [-0.0037, +0.0014] | 2,107 labelled, 1910 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_L` | 0.7767 | 0.7810 | -0.0043 [-0.0399, +0.0404] | 2,320 labelled, 47 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_R` | 0.7078 | 0.7434 | -0.0356 [-0.1043, +0.0147] | 2,320 labelled, 150 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_L` | 0.5000 | 0.5263 | -0.0263 (no interval) | 2,107 labelled, 11 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 2,107 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD` | 0.1474 | 0.1263 | +0.0211 [-0.0022, +0.0695] | 695 labelled, 85 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.STOP_POINT` | 0.2857 | 0.3226 | -0.0369 [-0.0699, -0.0062] | 2,246 labelled, 131 pos; 2,365 / 139 | **yes** (DAG_s1 worse) |
| `tactical.heads.goal.per_token.SPEED_BAND` | UNAVAILABLE | | | 0 labelled, 0 pos | no positive label for this token among its 0 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.CORRIDOR_OFFSET` | 0.3030 | 0.3067 | -0.0037 [-0.0463, +0.0353] | 695 labelled, 120 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.EVADE_IN_CORRIDOR` | 0.4773 | 0.4142 | +0.0631 [-0.0112, +0.1345] | 695 labelled, 110 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.OVERTAKE_VEHICLE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 20 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.MERGE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 25 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.GAP_TARGET` | 0.0328 | 0.0328 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 60 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.REACT_ON_ONCOMING` | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 50 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TAKE_EXIT_L` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TAKE_EXIT_R` | 0.0000 | 0.0588 | -0.0588 (no interval) | 695 labelled, 30 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_RED` | 0.5849 | 0.5306 | +0.0543 [-0.0568, +0.2589] | 720 labelled, 65 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_YELLOW` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_GREEN` | 0.4314 | 0.4078 | +0.0236 [-0.0356, +0.1006] | 695 labelled, 75 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.LANE_CHANGE_L` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.LANE_CHANGE_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |

### STRATEGIC, S's checkpoint error

| cell | DAG_s1 | DAG0_s1 | DAG_s1 - DAG0_s1 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `strategic.position_error_m.30` | 6.9723 | 6.9713 | +0.0010 [-0.0314, +0.0331] | 2,322 / 137 | no |
| `strategic.position_error_m.50` | 7.9593 | 7.9401 | +0.0192 [-0.0622, +0.0988] | 2,319 / 137 | no |
| `strategic.position_error_m.80` | 11.2166 | 11.1435 | +0.0730 [-0.0738, +0.2236] | 2,294 / 136 | no |
| `strategic.heading_error_deg.30` | 4.1296 | 4.1223 | +0.0073 [-0.1896, +0.1821] | 2,322 / 137 | no |
| `strategic.heading_error_deg.50` | 6.9929 | 6.8909 | +0.1020 [-0.1231, +0.3227] | 2,319 / 137 | no |
| `strategic.heading_error_deg.80` | 13.3560 | 13.4232 | -0.0672 [-0.4166, +0.3149] | 2,294 / 136 | no |
