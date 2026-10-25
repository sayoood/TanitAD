Contrast: X - B. Tier: T1-class open loop (own held outputs; no recorded future action). Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes.

### TACTICAL, plan-derived decisions and goal point

| cell | X | B | X - B [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.plan_derived.lateral_decision.accuracy` | 0.8441 | 0.8458 | -0.0017 [-0.0225, +0.0167] | 2,399 / 141 | no |
| `tactical.plan_derived.lateral_decision.kappa` | 0.6685 | 0.6652 | +0.0033 [-0.0396, +0.0392] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.accuracy` | 0.5661 | 0.5790 | -0.0129 [-0.0358, +0.0079] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.kappa` | 0.3398 | 0.3566 | -0.0168 [-0.0494, +0.0135] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.accuracy` | 0.5740 | 0.5811 | -0.0071 [-0.0346, +0.0179] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.kappa` | 0.4529 | 0.4586 | -0.0057 [-0.0389, +0.0254] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_point_error_m` | 6.4306 | 6.2810 | +0.1497 [-0.1038, +0.4086] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_long_bias_m` | -2.4016 | -2.2474 | -0.1542 [-0.4820, +0.1693] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_lat_bias_m` | -0.0068 | -0.1872 | +0.1804 [-0.0643, +0.4006] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_bearing_mae_deg` | 4.1132 | 4.0999 | +0.0132 [-1.1455, +0.9031] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_bearing_bias_deg` | -0.6974 | -0.5903 | -0.1072 [-1.2535, +0.8257] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_range_ratio` | 0.9535 | 0.9537 | -0.0002 [-0.0160, +0.0164] | 2,367 / 141 | no |

### TACTICAL, heads (lat / lon / goal)

| cell | X | B | X - B [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.heads.lateral.macro_f1` | 0.8269 | 0.8157 | +0.0112 [-0.0319, +0.0687] | 2,107 / 139 | no |
| `tactical.heads.lateral.accuracy` | 0.9544 | 0.9549 | -0.0005 [-0.0076, +0.0071] | 2,107 / 139 | no |
| `tactical.heads.lateral.kappa` | 0.7288 | 0.7240 | +0.0047 [-0.0429, +0.0573] | 2,107 / 139 | no |
| `tactical.heads.longitudinal.macro_f1` | 0.5015 | 0.4992 | +0.0024 [-0.0378, +0.0348] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.accuracy` | 0.6308 | 0.6085 | +0.0223 [-0.0090, +0.0554] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.kappa` | 0.5247 | 0.5012 | +0.0235 [-0.0153, +0.0639] | 2,240 / 136 | no |
| `tactical.heads.goal.macro_f1` | 0.2712 | 0.2618 | +0.0095 [-0.0181, +0.0723] | 2,365 / 139 | no |
| `tactical.heads.goal.token_accuracy` | 0.9562 | 0.9571 | -0.0009 [-0.0034, +0.0015] | 2,365 / 139 | no |
| `tactical.heads.goal.per_token.FOLLOW_LANE` | 0.9762 | 0.9752 | +0.0010 [-0.0032, +0.0054] | 2,107 labelled, 1910 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_L` | 0.7475 | 0.7045 | +0.0429 [-0.0700, +0.2038] | 2,320 labelled, 47 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_R` | 0.7211 | 0.7619 | -0.0408 [-0.0972, +0.0166] | 2,320 labelled, 150 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_L` | 0.6087 | 0.6364 | -0.0277 (no interval) | 2,107 labelled, 11 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 2,107 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD` | 0.1053 | 0.1010 | +0.0043 [+0.0000, +0.0170] | 695 labelled, 85 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.STOP_POINT` | 0.3137 | 0.3498 | -0.0361 [-0.1092, +0.0300] | 2,246 labelled, 131 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.SPEED_BAND` | UNAVAILABLE | | | 0 labelled, 0 pos | no positive label for this token among its 0 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.CORRIDOR_OFFSET` | 0.2135 | 0.2331 | -0.0196 [-0.1104, +0.0663] | 695 labelled, 120 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.EVADE_IN_CORRIDOR` | 0.4000 | 0.4699 | -0.0699 [-0.2133, +0.0622] | 695 labelled, 110 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.OVERTAKE_VEHICLE` | 0.2609 | 0.0952 | +0.1656 (no interval) | 695 labelled, 20 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.MERGE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 25 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.GAP_TARGET` | 0.0328 | 0.0000 | +0.0328 [+0.0000, +0.1081] | 695 labelled, 60 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.REACT_ON_ONCOMING` | 0.1132 | 0.0385 | +0.0747 [+0.0000, +0.2157] | 695 labelled, 50 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TAKE_EXIT_L` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TAKE_EXIT_R` | 0.0588 | 0.0556 | +0.0033 (no interval) | 695 labelled, 30 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_RED` | 0.6016 | 0.5273 | +0.0744 [-0.0575, +0.3282] | 720 labelled, 65 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_YELLOW` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_GREEN` | 0.0000 | 0.0250 | -0.0250 [-0.0870, +0.0000] | 695 labelled, 75 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.LANE_CHANGE_L` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.LANE_CHANGE_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |

### STRATEGIC, S's checkpoint error

| cell | X | B | X - B [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `strategic.position_error_m.30` | 7.0141 | 7.0231 | -0.0090 [-0.0721, +0.0590] | 2,322 / 137 | no |
| `strategic.position_error_m.50` | 8.0368 | 8.0122 | +0.0245 [-0.1547, +0.2015] | 2,319 / 137 | no |
| `strategic.position_error_m.80` | 11.3682 | 11.2059 | +0.1623 [-0.2843, +0.6321] | 2,294 / 136 | no |
| `strategic.heading_error_deg.30` | 4.1894 | 4.2002 | -0.0108 [-0.3329, +0.3380] | 2,322 / 137 | no |
| `strategic.heading_error_deg.50` | 6.7680 | 6.6864 | +0.0815 [-0.4630, +0.5949] | 2,319 / 137 | no |
| `strategic.heading_error_deg.80` | 12.9855 | 13.0540 | -0.0685 [-0.9740, +0.9320] | 2,294 / 136 | no |
