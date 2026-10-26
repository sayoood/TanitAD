Contrast: ORC_s1 - ORC0_s1. Tier: T1-class open loop (own held outputs; no recorded future action). Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes.

### TACTICAL, plan-derived decisions and goal point

| cell | ORC_s1 | ORC0_s1 | ORC_s1 - ORC0_s1 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.plan_derived.lateral_decision.accuracy` | 0.8429 | 0.8537 | -0.0108 [-0.0196, -0.0029] | 2,399 / 141 | **yes** (ORC_s1 worse) |
| `tactical.plan_derived.lateral_decision.kappa` | 0.6602 | 0.6779 | -0.0178 [-0.0366, -0.0005] | 2,399 / 141 | **yes** (ORC_s1 worse) |
| `tactical.plan_derived.longitudinal_decision.accuracy` | 0.5715 | 0.5677 | +0.0038 [-0.0088, +0.0183] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.kappa` | 0.3436 | 0.3391 | +0.0045 [-0.0144, +0.0251] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.accuracy` | 0.5723 | 0.5732 | -0.0008 [-0.0146, +0.0158] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.kappa` | 0.4469 | 0.4478 | -0.0009 [-0.0182, +0.0192] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_point_error_m` | 6.0302 | 5.9517 | +0.0785 [-0.0265, +0.1814] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_long_bias_m` | -0.9781 | -1.1751 | +0.1970 [+0.0680, +0.3209] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_lat_bias_m` | -0.1044 | -0.0537 | -0.0507 [-0.1352, +0.0353] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_bearing_mae_deg` | 3.9245 | 3.6224 | +0.3021 [+0.0258, +0.6524] | 2,367 / 141 | **yes** (ORC_s1 worse) |
| `tactical.goal_setting.goal_bearing_bias_deg` | -0.0868 | -0.0873 | +0.0005 [-0.3110, +0.3627] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_range_ratio` | 1.0106 | 0.9922 | +0.0185 [+0.0094, +0.0287] | 2,367 / 141 | **yes** |

### TACTICAL, heads (lat / lon / goal)

| cell | ORC_s1 | ORC0_s1 | ORC_s1 - ORC0_s1 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.heads.lateral.macro_f1` | 0.8537 | 0.8418 | +0.0120 [-0.0091, +0.0306] | 2,107 / 139 | no |
| `tactical.heads.lateral.accuracy` | 0.9578 | 0.9554 | +0.0024 [-0.0015, +0.0066] | 2,107 / 139 | no |
| `tactical.heads.lateral.kappa` | 0.7638 | 0.7489 | +0.0148 [-0.0083, +0.0387] | 2,107 / 139 | no |
| `tactical.heads.longitudinal.macro_f1` | 0.5026 | 0.4979 | +0.0046 [-0.0582, +0.0589] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.accuracy` | 0.6304 | 0.6214 | +0.0089 [-0.0097, +0.0284] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.kappa` | 0.5195 | 0.5109 | +0.0087 [-0.0143, +0.0327] | 2,240 / 136 | no |
| `tactical.heads.goal.macro_f1` | 0.2651 | 0.2865 | -0.0215 [-0.0439, +0.0061] | 2,365 / 139 | no |
| `tactical.heads.goal.token_accuracy` | 0.9584 | 0.9588 | -0.0004 [-0.0024, +0.0013] | 2,365 / 139 | no |
| `tactical.heads.goal.per_token.FOLLOW_LANE` | 0.9737 | 0.9737 | -0.0000 [-0.0022, +0.0024] | 2,107 labelled, 1910 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_L` | 0.7961 | 0.7736 | +0.0225 [-0.0296, +0.0717] | 2,320 labelled, 47 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_R` | 0.7419 | 0.7427 | -0.0007 [-0.0224, +0.0205] | 2,320 labelled, 150 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_L` | 0.4444 | 0.3750 | +0.0694 (no interval) | 2,107 labelled, 11 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_R` | 0.0000 | 0.1818 | -0.1818 (no interval) | 2,107 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD` | 0.0667 | 0.1414 | -0.0747 [-0.1469, +0.0000] | 695 labelled, 85 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.STOP_POINT` | 0.2299 | 0.2905 | -0.0606 [-0.1363, +0.0044] | 2,246 labelled, 131 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.SPEED_BAND` | UNAVAILABLE | | | 0 labelled, 0 pos | no positive label for this token among its 0 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.CORRIDOR_OFFSET` | 0.3503 | 0.3158 | +0.0345 [-0.0397, +0.1266] | 695 labelled, 120 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.EVADE_IN_CORRIDOR` | 0.4457 | 0.4659 | -0.0202 [-0.1285, +0.0875] | 695 labelled, 110 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.OVERTAKE_VEHICLE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 20 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.MERGE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 25 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.GAP_TARGET` | 0.0328 | 0.0328 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 60 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.REACT_ON_ONCOMING` | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 50 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TAKE_EXIT_L` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TAKE_EXIT_R` | 0.0000 | 0.1143 | -0.1143 (no interval) | 695 labelled, 30 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_RED` | 0.5347 | 0.6168 | -0.0822 [-0.2077, +0.0072] | 720 labelled, 65 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_YELLOW` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_GREEN` | 0.4200 | 0.4200 | +0.0000 [-0.1448, +0.1023] | 695 labelled, 75 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.LANE_CHANGE_L` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.LANE_CHANGE_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |

### STRATEGIC, S's checkpoint error

| cell | ORC_s1 | ORC0_s1 | ORC_s1 - ORC0_s1 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `strategic.position_error_m.30` | 6.9836 | 6.9734 | +0.0102 [-0.0264, +0.0488] | 2,322 / 137 | no |
| `strategic.position_error_m.50` | 8.0172 | 8.0194 | -0.0022 [-0.0813, +0.0781] | 2,319 / 137 | no |
| `strategic.position_error_m.80` | 11.2339 | 11.3127 | -0.0788 [-0.3270, +0.1337] | 2,294 / 136 | no |
| `strategic.heading_error_deg.30` | 4.2636 | 4.2653 | -0.0017 [-0.2482, +0.1878] | 2,322 / 137 | no |
| `strategic.heading_error_deg.50` | 6.9973 | 7.2285 | -0.2312 [-0.7013, +0.1262] | 2,319 / 137 | no |
| `strategic.heading_error_deg.80` | 13.3181 | 13.4372 | -0.1192 [-0.6331, +0.3461] | 2,294 / 136 | no |
