Contrast: DAG_s0 - M2b_s0. Tier: T1-class open loop (own held outputs; no recorded future action). Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes.

### TACTICAL, plan-derived decisions and goal point

| cell | DAG_s0 | M2b_s0 | DAG_s0 - M2b_s0 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.plan_derived.lateral_decision.accuracy` | 0.8508 | 0.8458 | +0.0050 [-0.0096, +0.0196] | 2,399 / 141 | no |
| `tactical.plan_derived.lateral_decision.kappa` | 0.6646 | 0.6652 | -0.0006 [-0.0332, +0.0308] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.accuracy` | 0.5773 | 0.5790 | -0.0017 [-0.0242, +0.0192] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.kappa` | 0.3528 | 0.3566 | -0.0038 [-0.0370, +0.0262] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.accuracy` | 0.5802 | 0.5811 | -0.0008 [-0.0254, +0.0229] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.kappa` | 0.4553 | 0.4586 | -0.0032 [-0.0335, +0.0267] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_point_error_m` | 5.6500 | 6.2810 | -0.6310 [-0.9125, -0.3577] | 2,399 / 141 | **yes** (DAG_s0 better) |
| `tactical.goal_setting.goal_long_bias_m` | -0.8347 | -2.2474 | +1.4127 [+1.0439, +1.7941] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_lat_bias_m` | 0.0351 | -0.1872 | +0.2223 [+0.0618, +0.3812] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_bearing_mae_deg` | 3.6282 | 4.0999 | -0.4717 [-1.4533, +0.1530] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_bearing_bias_deg` | -0.2987 | -0.5903 | +0.2915 [-0.9346, +1.4146] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_range_ratio` | 0.9983 | 0.9537 | +0.0446 [+0.0278, +0.0603] | 2,367 / 141 | **yes** |

### TACTICAL, heads (lat / lon / goal)

| cell | DAG_s0 | M2b_s0 | DAG_s0 - M2b_s0 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.heads.lateral.macro_f1` | 0.8347 | 0.8157 | +0.0190 [-0.0322, +0.0770] | 2,107 / 139 | no |
| `tactical.heads.lateral.accuracy` | 0.9530 | 0.9549 | -0.0019 [-0.0105, +0.0063] | 2,107 / 139 | no |
| `tactical.heads.lateral.kappa` | 0.7386 | 0.7240 | +0.0146 [-0.0370, +0.0688] | 2,107 / 139 | no |
| `tactical.heads.longitudinal.macro_f1` | 0.4944 | 0.4992 | -0.0047 [-0.0541, +0.0381] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.accuracy` | 0.6232 | 0.6085 | +0.0147 [-0.0123, +0.0438] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.kappa` | 0.5076 | 0.5012 | +0.0064 [-0.0269, +0.0421] | 2,240 / 136 | no |
| `tactical.heads.goal.macro_f1` | 0.2472 | 0.2618 | -0.0145 [-0.0402, +0.0170] | 2,365 / 139 | no |
| `tactical.heads.goal.token_accuracy` | 0.9565 | 0.9571 | -0.0006 [-0.0039, +0.0024] | 2,365 / 139 | no |
| `tactical.heads.goal.per_token.FOLLOW_LANE` | 0.9720 | 0.9752 | -0.0032 [-0.0084, +0.0015] | 2,107 labelled, 1910 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_L` | 0.7748 | 0.7045 | +0.0702 [-0.0588, +0.2430] | 2,320 labelled, 47 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_R` | 0.7161 | 0.7619 | -0.0458 [-0.1077, +0.0033] | 2,320 labelled, 150 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_L` | 0.3529 | 0.6364 | -0.2834 (no interval) | 2,107 labelled, 11 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 2,107 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD` | 0.1200 | 0.1010 | +0.0190 [-0.0096, +0.0652] | 695 labelled, 85 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.STOP_POINT` | 0.2317 | 0.3498 | -0.1181 [-0.2313, -0.0104] | 2,246 labelled, 131 pos; 2,365 / 139 | **yes** (DAG_s0 worse) |
| `tactical.heads.goal.per_token.SPEED_BAND` | UNAVAILABLE | | | 0 labelled, 0 pos | no positive label for this token among its 0 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.CORRIDOR_OFFSET` | 0.2442 | 0.2331 | +0.0111 [-0.0965, +0.1224] | 695 labelled, 120 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.EVADE_IN_CORRIDOR` | 0.4535 | 0.4699 | -0.0164 [-0.1403, +0.1004] | 695 labelled, 110 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.OVERTAKE_VEHICLE` | 0.0952 | 0.0952 | +0.0000 (no interval) | 695 labelled, 20 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.MERGE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 25 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.GAP_TARGET` | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 60 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.REACT_ON_ONCOMING` | 0.0000 | 0.0385 | -0.0385 [-0.1250, +0.0000] | 695 labelled, 50 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TAKE_EXIT_L` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TAKE_EXIT_R` | 0.1111 | 0.0556 | +0.0556 (no interval) | 695 labelled, 30 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_RED` | 0.4792 | 0.5273 | -0.0481 [-0.2682, +0.1032] | 720 labelled, 65 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_YELLOW` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_GREEN` | 0.1463 | 0.0250 | +0.1213 [+0.0004, +0.2769] | 695 labelled, 75 pos; 2,365 / 139 | **yes** (DAG_s0 better) |
| `tactical.heads.goal.per_token.LANE_CHANGE_L` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.LANE_CHANGE_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |

### STRATEGIC, S's checkpoint error

| cell | DAG_s0 | M2b_s0 | DAG_s0 - M2b_s0 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `strategic.position_error_m.30` | 7.0048 | 7.0231 | -0.0183 [-0.0745, +0.0373] | 2,322 / 137 | no |
| `strategic.position_error_m.50` | 8.1082 | 8.0122 | +0.0960 [-0.0644, +0.2674] | 2,319 / 137 | no |
| `strategic.position_error_m.80` | 11.6970 | 11.2059 | +0.4911 [+0.0093, +1.0719] | 2,294 / 136 | **yes** (DAG_s0 worse) |
| `strategic.heading_error_deg.30` | 4.2440 | 4.2002 | +0.0438 [-0.2037, +0.3003] | 2,322 / 137 | no |
| `strategic.heading_error_deg.50` | 7.1077 | 6.6864 | +0.4213 [-0.1062, +1.0461] | 2,319 / 137 | no |
| `strategic.heading_error_deg.80` | 13.9881 | 13.0540 | +0.9341 [-0.1986, +2.4094] | 2,294 / 136 | no |
