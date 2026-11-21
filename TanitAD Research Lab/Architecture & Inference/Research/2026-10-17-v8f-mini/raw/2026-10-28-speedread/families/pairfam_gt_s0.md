Contrast: gt - own. Tier: T1-class open loop (own held outputs; no recorded future action). Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes.

### TACTICAL, plan-derived decisions and goal point

| cell | gt | own | gt - own [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.plan_derived.lateral_decision.accuracy` | 0.8370 | 0.8441 | -0.0071 [-0.0192, +0.0042] | 2,399 / 141 | no |
| `tactical.plan_derived.lateral_decision.kappa` | 0.6541 | 0.6685 | -0.0143 [-0.0390, +0.0080] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.accuracy` | 0.6820 | 0.5661 | +0.1159 [+0.0875, +0.1450] | 2,399 / 141 | **yes** (gt better) |
| `tactical.plan_derived.longitudinal_decision.kappa` | 0.5158 | 0.3398 | +0.1760 [+0.1337, +0.2188] | 2,399 / 141 | **yes** (gt better) |
| `tactical.plan_derived.maneuver_5way_collapsed.accuracy` | 0.6544 | 0.5740 | +0.0805 [+0.0513, +0.1108] | 2,399 / 141 | **yes** (gt better) |
| `tactical.plan_derived.maneuver_5way_collapsed.kappa` | 0.5554 | 0.4529 | +0.1024 [+0.0648, +0.1416] | 2,399 / 141 | **yes** (gt better) |
| `tactical.goal_setting.goal_point_error_m` | 4.2068 | 6.4306 | -2.2239 [-2.5417, -1.9337] | 2,399 / 141 | **yes** (gt better) |
| `tactical.goal_setting.goal_long_bias_m` | -1.7201 | -2.4016 | +0.6815 [+0.1831, +1.2044] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_lat_bias_m` | -0.1363 | -0.0068 | -0.1294 [-0.2446, -0.0261] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_bearing_mae_deg` | 3.9591 | 4.1132 | -0.1541 [-0.6931, +0.3614] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_bearing_bias_deg` | -0.1320 | -0.6974 | +0.5654 [-0.5665, +2.3448] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_range_ratio` | 0.9455 | 0.9535 | -0.0080 [-0.0402, +0.0214] | 2,367 / 141 | no |

### TACTICAL, heads (lat / lon / goal)

| cell | gt | own | gt - own [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.heads.lateral.macro_f1` | 0.8269 | 0.8269 | +0.0000 [+0.0000, +0.0000] | 2,107 / 139 | no |
| `tactical.heads.lateral.accuracy` | 0.9544 | 0.9544 | +0.0000 [+0.0000, +0.0000] | 2,107 / 139 | no |
| `tactical.heads.lateral.kappa` | 0.7288 | 0.7288 | +0.0000 [+0.0000, +0.0000] | 2,107 / 139 | no |
| `tactical.heads.longitudinal.macro_f1` | 0.5015 | 0.5015 | +0.0000 [+0.0000, +0.0000] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.accuracy` | 0.6308 | 0.6308 | +0.0000 [+0.0000, +0.0000] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.kappa` | 0.5247 | 0.5247 | +0.0000 [+0.0000, +0.0000] | 2,240 / 136 | no |
| `tactical.heads.goal.macro_f1` | 0.2712 | 0.2712 | +0.0000 [+0.0000, +0.0000] | 2,365 / 139 | no |
| `tactical.heads.goal.token_accuracy` | 0.9562 | 0.9562 | +0.0000 [+0.0000, +0.0000] | 2,365 / 139 | no |
| `tactical.heads.goal.per_token.FOLLOW_LANE` | 0.9762 | 0.9762 | +0.0000 [+0.0000, +0.0000] | 2,107 labelled, 1910 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_L` | 0.7475 | 0.7475 | +0.0000 [+0.0000, +0.0000] | 2,320 labelled, 47 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_R` | 0.7211 | 0.7211 | +0.0000 [+0.0000, +0.0000] | 2,320 labelled, 150 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_L` | 0.6087 | 0.6087 | +0.0000 (no interval) | 2,107 labelled, 11 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 2,107 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD` | 0.1053 | 0.1053 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 85 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.STOP_POINT` | 0.3137 | 0.3137 | +0.0000 [+0.0000, +0.0000] | 2,246 labelled, 131 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.SPEED_BAND` | UNAVAILABLE | | | 0 labelled, 0 pos | no positive label for this token among its 0 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.CORRIDOR_OFFSET` | 0.2135 | 0.2135 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 120 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.EVADE_IN_CORRIDOR` | 0.4000 | 0.4000 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 110 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.OVERTAKE_VEHICLE` | 0.2609 | 0.2609 | +0.0000 (no interval) | 695 labelled, 20 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.MERGE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 25 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.GAP_TARGET` | 0.0328 | 0.0328 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 60 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.REACT_ON_ONCOMING` | 0.1132 | 0.1132 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 50 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TAKE_EXIT_L` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TAKE_EXIT_R` | 0.0588 | 0.0588 | +0.0000 (no interval) | 695 labelled, 30 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_RED` | 0.6016 | 0.6016 | +0.0000 [+0.0000, +0.0000] | 720 labelled, 65 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_YELLOW` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_GREEN` | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 75 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.LANE_CHANGE_L` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.LANE_CHANGE_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |

### STRATEGIC, S's checkpoint error

| cell | gt | own | gt - own [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `strategic.position_error_m.30` | 7.0141 | 7.0141 | +0.0000 [+0.0000, +0.0000] | 2,322 / 137 | no |
| `strategic.position_error_m.50` | 8.0368 | 8.0368 | +0.0000 [+0.0000, +0.0000] | 2,319 / 137 | no |
| `strategic.position_error_m.80` | 11.3682 | 11.3682 | +0.0000 [+0.0000, +0.0000] | 2,294 / 136 | no |
| `strategic.heading_error_deg.30` | 4.1894 | 4.1894 | +0.0000 [+0.0000, +0.0000] | 2,322 / 137 | no |
| `strategic.heading_error_deg.50` | 6.7680 | 6.7680 | +0.0000 [+0.0000, +0.0000] | 2,319 / 137 | no |
| `strategic.heading_error_deg.80` | 12.9855 | 12.9855 | +0.0000 [+0.0000, +0.0000] | 2,294 / 136 | no |
