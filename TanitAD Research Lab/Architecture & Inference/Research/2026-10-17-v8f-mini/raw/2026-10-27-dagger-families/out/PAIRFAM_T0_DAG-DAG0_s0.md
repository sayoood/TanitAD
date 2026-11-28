Contrast: DAG_s0 - DAG0_s0. Tier: T1-class open loop (own held outputs; no recorded future action). Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes.

### TACTICAL, plan-derived decisions and goal point

| cell | DAG_s0 | DAG0_s0 | DAG_s0 - DAG0_s0 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.plan_derived.lateral_decision.accuracy` | 0.8508 | 0.8508 | +0.0000 [-0.0079, +0.0083] | 2,399 / 141 | no |
| `tactical.plan_derived.lateral_decision.kappa` | 0.6646 | 0.6672 | -0.0026 [-0.0191, +0.0147] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.accuracy` | 0.5773 | 0.5711 | +0.0063 [-0.0046, +0.0179] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.kappa` | 0.3528 | 0.3429 | +0.0099 [-0.0067, +0.0271] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.accuracy` | 0.5802 | 0.5736 | +0.0067 [-0.0058, +0.0196] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.kappa` | 0.4553 | 0.4478 | +0.0076 [-0.0084, +0.0243] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_point_error_m` | 5.6500 | 5.7020 | -0.0520 [-0.1318, +0.0252] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_long_bias_m` | -0.8347 | -1.0471 | +0.2124 [+0.1087, +0.3150] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_lat_bias_m` | 0.0351 | -0.0458 | +0.0809 [+0.0157, +0.1448] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_bearing_mae_deg` | 3.6282 | 4.0859 | -0.4577 [-0.8901, -0.1414] | 2,367 / 141 | **yes** (DAG_s0 better) |
| `tactical.goal_setting.goal_bearing_bias_deg` | -0.2987 | 0.0083 | -0.3070 [-1.2602, +0.8284] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_range_ratio` | 0.9983 | 0.9922 | +0.0060 [+0.0027, +0.0096] | 2,367 / 141 | **yes** |

### TACTICAL, heads (lat / lon / goal)

| cell | DAG_s0 | DAG0_s0 | DAG_s0 - DAG0_s0 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.heads.lateral.macro_f1` | 0.8347 | 0.8342 | +0.0005 [-0.0186, +0.0235] | 2,107 / 139 | no |
| `tactical.heads.lateral.accuracy` | 0.9530 | 0.9530 | +0.0000 [-0.0047, +0.0048] | 2,107 / 139 | no |
| `tactical.heads.lateral.kappa` | 0.7386 | 0.7374 | +0.0012 [-0.0254, +0.0311] | 2,107 / 139 | no |
| `tactical.heads.longitudinal.macro_f1` | 0.4944 | 0.5119 | -0.0175 [-0.0541, +0.0167] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.accuracy` | 0.6232 | 0.6402 | -0.0170 [-0.0326, -0.0009] | 2,240 / 136 | **yes** (DAG_s0 worse) |
| `tactical.heads.longitudinal.kappa` | 0.5076 | 0.5303 | -0.0227 [-0.0433, -0.0021] | 2,240 / 136 | **yes** (DAG_s0 worse) |
| `tactical.heads.goal.macro_f1` | 0.2472 | 0.2631 | -0.0159 [-0.0288, +0.0042] | 2,365 / 139 | no |
| `tactical.heads.goal.token_accuracy` | 0.9565 | 0.9574 | -0.0009 [-0.0021, +0.0003] | 2,365 / 139 | no |
| `tactical.heads.goal.per_token.FOLLOW_LANE` | 0.9720 | 0.9729 | -0.0009 [-0.0036, +0.0013] | 2,107 labelled, 1910 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_L` | 0.7748 | 0.7568 | +0.0180 [-0.0189, +0.0866] | 2,320 labelled, 47 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_R` | 0.7161 | 0.7320 | -0.0159 [-0.0379, +0.0093] | 2,320 labelled, 150 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_L` | 0.3529 | 0.6000 | -0.2471 (no interval) | 2,107 labelled, 11 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 2,107 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD` | 0.1200 | 0.1031 | +0.0169 [-0.0095, +0.0632] | 695 labelled, 85 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.STOP_POINT` | 0.2317 | 0.2754 | -0.0437 [-0.1137, +0.0143] | 2,246 labelled, 131 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.SPEED_BAND` | UNAVAILABLE | | | 0 labelled, 0 pos | no positive label for this token among its 0 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.CORRIDOR_OFFSET` | 0.2442 | 0.2414 | +0.0028 [-0.0255, +0.0312] | 695 labelled, 120 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.EVADE_IN_CORRIDOR` | 0.4535 | 0.5083 | -0.0548 [-0.1442, +0.0129] | 695 labelled, 110 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.OVERTAKE_VEHICLE` | 0.0952 | 0.0000 | +0.0952 (no interval) | 695 labelled, 20 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.MERGE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 25 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.GAP_TARGET` | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 60 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.REACT_ON_ONCOMING` | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 50 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TAKE_EXIT_L` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TAKE_EXIT_R` | 0.1111 | 0.1579 | -0.0468 (no interval) | 695 labelled, 30 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_RED` | 0.4792 | 0.4615 | +0.0176 [-0.0481, +0.1305] | 720 labelled, 65 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_YELLOW` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_GREEN` | 0.1463 | 0.1905 | -0.0441 [-0.1606, +0.0538] | 695 labelled, 75 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.LANE_CHANGE_L` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.LANE_CHANGE_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |

### STRATEGIC, S's checkpoint error

| cell | DAG_s0 | DAG0_s0 | DAG_s0 - DAG0_s0 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `strategic.position_error_m.30` | 7.0048 | 7.0131 | -0.0083 [-0.0354, +0.0187] | 2,322 / 137 | no |
| `strategic.position_error_m.50` | 8.1082 | 8.1026 | +0.0056 [-0.0525, +0.0662] | 2,319 / 137 | no |
| `strategic.position_error_m.80` | 11.6970 | 11.6685 | +0.0284 [-0.1305, +0.2028] | 2,294 / 136 | no |
| `strategic.heading_error_deg.30` | 4.2440 | 4.2580 | -0.0140 [-0.1327, +0.1158] | 2,322 / 137 | no |
| `strategic.heading_error_deg.50` | 7.1077 | 7.1877 | -0.0800 [-0.2549, +0.1043] | 2,319 / 137 | no |
| `strategic.heading_error_deg.80` | 13.9881 | 14.0434 | -0.0552 [-0.3549, +0.2732] | 2,294 / 136 | no |
