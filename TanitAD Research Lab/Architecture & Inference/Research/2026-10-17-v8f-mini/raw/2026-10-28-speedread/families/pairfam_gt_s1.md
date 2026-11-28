Contrast: gt - own. Tier: T1-class open loop (own held outputs; no recorded future action). Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes.

### TACTICAL, plan-derived decisions and goal point

| cell | gt | own | gt - own [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.plan_derived.lateral_decision.accuracy` | 0.8454 | 0.8495 | -0.0042 [-0.0142, +0.0054] | 2,399 / 141 | no |
| `tactical.plan_derived.lateral_decision.kappa` | 0.6567 | 0.6634 | -0.0067 [-0.0281, +0.0133] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.accuracy` | 0.7332 | 0.5886 | +0.1446 [+0.1188, +0.1706] | 2,399 / 141 | **yes** (gt better) |
| `tactical.plan_derived.longitudinal_decision.kappa` | 0.5898 | 0.3670 | +0.2228 [+0.1852, +0.2614] | 2,399 / 141 | **yes** (gt better) |
| `tactical.plan_derived.maneuver_5way_collapsed.accuracy` | 0.6861 | 0.5873 | +0.0988 [+0.0738, +0.1254] | 2,399 / 141 | **yes** (gt better) |
| `tactical.plan_derived.maneuver_5way_collapsed.kappa` | 0.5930 | 0.4642 | +0.1289 [+0.0963, +0.1641] | 2,399 / 141 | **yes** (gt better) |
| `tactical.goal_setting.goal_point_error_m` | 4.0388 | 6.1025 | -2.0637 [-2.3514, -1.7739] | 2,399 / 141 | **yes** (gt better) |
| `tactical.goal_setting.goal_long_bias_m` | -0.7035 | -0.9872 | +0.2837 [-0.1308, +0.7242] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_lat_bias_m` | 0.0276 | 0.0578 | -0.0302 [-0.1403, +0.0734] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_bearing_mae_deg` | 3.5393 | 3.7561 | -0.2168 [-0.5473, +0.0024] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_bearing_bias_deg` | 0.7797 | 0.3330 | +0.4467 [-0.1977, +1.4937] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_range_ratio` | 1.0083 | 1.0315 | -0.0233 [-0.0522, +0.0034] | 2,367 / 141 | no |

### TACTICAL, heads (lat / lon / goal)

| cell | gt | own | gt - own [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.heads.lateral.macro_f1` | 0.8198 | 0.8198 | +0.0000 [+0.0000, +0.0000] | 2,107 / 139 | no |
| `tactical.heads.lateral.accuracy` | 0.9483 | 0.9483 | +0.0000 [+0.0000, +0.0000] | 2,107 / 139 | no |
| `tactical.heads.lateral.kappa` | 0.7219 | 0.7219 | +0.0000 [+0.0000, +0.0000] | 2,107 / 139 | no |
| `tactical.heads.longitudinal.macro_f1` | 0.4959 | 0.4959 | +0.0000 [+0.0000, +0.0000] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.accuracy` | 0.6196 | 0.6196 | +0.0000 [+0.0000, +0.0000] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.kappa` | 0.5034 | 0.5034 | +0.0000 [+0.0000, +0.0000] | 2,240 / 136 | no |
| `tactical.heads.goal.macro_f1` | 0.2316 | 0.2316 | +0.0000 [+0.0000, +0.0000] | 2,365 / 139 | no |
| `tactical.heads.goal.token_accuracy` | 0.9559 | 0.9559 | +0.0000 [+0.0000, +0.0000] | 2,365 / 139 | no |
| `tactical.heads.goal.per_token.FOLLOW_LANE` | 0.9690 | 0.9690 | +0.0000 [+0.0000, +0.0000] | 2,107 labelled, 1910 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_L` | 0.7238 | 0.7238 | +0.0000 [+0.0000, +0.0000] | 2,320 labelled, 47 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_R` | 0.7143 | 0.7143 | +0.0000 [+0.0000, +0.0000] | 2,320 labelled, 150 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_L` | 0.4000 | 0.4000 | +0.0000 (no interval) | 2,107 labelled, 11 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 2,107 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD` | 0.1165 | 0.1165 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 85 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.STOP_POINT` | 0.2112 | 0.2112 | +0.0000 [+0.0000, +0.0000] | 2,246 labelled, 131 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.SPEED_BAND` | UNAVAILABLE | | | 0 labelled, 0 pos | no positive label for this token among its 0 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.CORRIDOR_OFFSET` | 0.2625 | 0.2625 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 120 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.EVADE_IN_CORRIDOR` | 0.3810 | 0.3810 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 110 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.OVERTAKE_VEHICLE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 20 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.MERGE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 25 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.GAP_TARGET` | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 60 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.REACT_ON_ONCOMING` | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 50 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TAKE_EXIT_L` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TAKE_EXIT_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 30 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_RED` | 0.4828 | 0.4828 | +0.0000 [+0.0000, +0.0000] | 720 labelled, 65 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_YELLOW` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_GREEN` | 0.1395 | 0.1395 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 75 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.LANE_CHANGE_L` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.LANE_CHANGE_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |

### STRATEGIC, S's checkpoint error

| cell | gt | own | gt - own [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `strategic.position_error_m.30` | 7.0074 | 7.0074 | +0.0000 [+0.0000, +0.0000] | 2,322 / 137 | no |
| `strategic.position_error_m.50` | 8.0075 | 8.0075 | +0.0000 [+0.0000, +0.0000] | 2,319 / 137 | no |
| `strategic.position_error_m.80` | 11.1359 | 11.1359 | +0.0000 [+0.0000, +0.0000] | 2,294 / 136 | no |
| `strategic.heading_error_deg.30` | 4.0508 | 4.0508 | +0.0000 [+0.0000, +0.0000] | 2,322 / 137 | no |
| `strategic.heading_error_deg.50` | 6.8016 | 6.8016 | +0.0000 [+0.0000, +0.0000] | 2,319 / 137 | no |
| `strategic.heading_error_deg.80` | 13.0187 | 13.0187 | +0.0000 [+0.0000, +0.0000] | 2,294 / 136 | no |
