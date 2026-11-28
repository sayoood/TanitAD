Contrast: ego - own. Tier: T1-class open loop (own held outputs; no recorded future action). Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes.

### TACTICAL, plan-derived decisions and goal point

| cell | ego | own | ego - own [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.plan_derived.lateral_decision.accuracy` | 0.8454 | 0.8441 | +0.0013 [-0.0054, +0.0084] | 2,399 / 141 | no |
| `tactical.plan_derived.lateral_decision.kappa` | 0.6709 | 0.6685 | +0.0024 [-0.0106, +0.0169] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.accuracy` | 0.5619 | 0.5661 | -0.0042 [-0.0246, +0.0163] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.kappa` | 0.3312 | 0.3398 | -0.0086 [-0.0392, +0.0206] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.accuracy` | 0.5757 | 0.5740 | +0.0017 [-0.0183, +0.0221] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.kappa` | 0.4534 | 0.4529 | +0.0005 [-0.0238, +0.0250] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_point_error_m` | 6.0363 | 6.4306 | -0.3943 [-0.6422, -0.1651] | 2,399 / 141 | **yes** (ego better) |
| `tactical.goal_setting.goal_long_bias_m` | -1.7618 | -2.4016 | +0.6398 [+0.3316, +0.9665] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_lat_bias_m` | -0.0800 | -0.0068 | -0.0732 [-0.1386, -0.0091] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_bearing_mae_deg` | 4.0999 | 4.1132 | -0.0133 [-0.2417, +0.2332] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_bearing_bias_deg` | -0.5233 | -0.6974 | +0.1742 [-0.1519, +0.6312] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_range_ratio` | 0.9688 | 0.9535 | +0.0153 [+0.0050, +0.0255] | 2,367 / 141 | **yes** |

### TACTICAL, heads (lat / lon / goal)

| cell | ego | own | ego - own [95 % CI] | n (windows / episodes) | separated |
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

| cell | ego | own | ego - own [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `strategic.position_error_m.30` | 7.0141 | 7.0141 | +0.0000 [+0.0000, +0.0000] | 2,322 / 137 | no |
| `strategic.position_error_m.50` | 8.0368 | 8.0368 | +0.0000 [+0.0000, +0.0000] | 2,319 / 137 | no |
| `strategic.position_error_m.80` | 11.3682 | 11.3682 | +0.0000 [+0.0000, +0.0000] | 2,294 / 136 | no |
| `strategic.heading_error_deg.30` | 4.1894 | 4.1894 | +0.0000 [+0.0000, +0.0000] | 2,322 / 137 | no |
| `strategic.heading_error_deg.50` | 6.7680 | 6.7680 | +0.0000 [+0.0000, +0.0000] | 2,319 / 137 | no |
| `strategic.heading_error_deg.80` | 12.9855 | 12.9855 | +0.0000 [+0.0000, +0.0000] | 2,294 / 136 | no |
