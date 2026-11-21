Contrast: ORC_s0 - ORC0_s0. Tier: T1-class open loop (own held outputs; no recorded future action). Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes.

### TACTICAL, plan-derived decisions and goal point

| cell | ORC_s0 | ORC0_s0 | ORC_s0 - ORC0_s0 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.plan_derived.lateral_decision.accuracy` | 0.8454 | 0.8266 | +0.0188 [+0.0017, +0.0358] | 2,399 / 141 | **yes** (ORC_s0 better) |
| `tactical.plan_derived.lateral_decision.kappa` | 0.6563 | 0.6171 | +0.0392 [+0.0055, +0.0725] | 2,399 / 141 | **yes** (ORC_s0 better) |
| `tactical.plan_derived.longitudinal_decision.accuracy` | 0.5698 | 0.5777 | -0.0079 [-0.0225, +0.0071] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.kappa` | 0.3431 | 0.3510 | -0.0079 [-0.0296, +0.0148] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.accuracy` | 0.5686 | 0.5665 | +0.0021 [-0.0159, +0.0192] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.kappa` | 0.4418 | 0.4364 | +0.0054 [-0.0177, +0.0270] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_point_error_m` | 5.8074 | 5.7126 | +0.0947 [-0.0311, +0.2287] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_long_bias_m` | -0.5592 | -0.9878 | +0.4287 [+0.2342, +0.6375] | 2,399 / 141 | **yes** |
| `tactical.goal_setting.goal_lat_bias_m` | -0.0010 | -0.0513 | +0.0502 [-0.0286, +0.1302] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_bearing_mae_deg` | 3.9644 | 3.9783 | -0.0139 [-0.5931, +0.4731] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_bearing_bias_deg` | -0.5949 | -0.2229 | -0.3720 [-1.4412, +0.4539] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_range_ratio` | 1.0325 | 0.9861 | +0.0465 [+0.0265, +0.0702] | 2,367 / 141 | **yes** |

### TACTICAL, heads (lat / lon / goal)

| cell | ORC_s0 | ORC0_s0 | ORC_s0 - ORC0_s0 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.heads.lateral.macro_f1` | 0.8437 | 0.8432 | +0.0005 [-0.0184, +0.0235] | 2,107 / 139 | no |
| `tactical.heads.lateral.accuracy` | 0.9544 | 0.9554 | -0.0009 [-0.0076, +0.0048] | 2,107 / 139 | no |
| `tactical.heads.lateral.kappa` | 0.7459 | 0.7467 | -0.0007 [-0.0323, +0.0316] | 2,107 / 139 | no |
| `tactical.heads.longitudinal.macro_f1` | 0.4701 | 0.5104 | -0.0403 [-0.0912, +0.0293] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.accuracy` | 0.6277 | 0.6415 | -0.0138 [-0.0303, +0.0027] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.kappa` | 0.5130 | 0.5324 | -0.0194 [-0.0413, +0.0020] | 2,240 / 136 | no |
| `tactical.heads.goal.macro_f1` | 0.2401 | 0.2697 | -0.0296 [-0.0571, -0.0049] | 2,365 / 139 | **yes** (ORC_s0 worse) |
| `tactical.heads.goal.token_accuracy` | 0.9569 | 0.9573 | -0.0004 [-0.0019, +0.0010] | 2,365 / 139 | no |
| `tactical.heads.goal.per_token.FOLLOW_LANE` | 0.9736 | 0.9740 | -0.0003 [-0.0034, +0.0024] | 2,107 labelled, 1910 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_L` | 0.7890 | 0.7636 | +0.0254 [-0.0186, +0.0782] | 2,320 labelled, 47 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_R` | 0.7279 | 0.7340 | -0.0061 [-0.0385, +0.0221] | 2,320 labelled, 150 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_L` | 0.3333 | 0.6667 | -0.3333 (no interval) | 2,107 labelled, 11 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 2,107 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.YIELD` | 0.1031 | 0.1031 | +0.0000 [-0.0045, +0.0036] | 695 labelled, 85 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.STOP_POINT` | 0.2989 | 0.2651 | +0.0338 [-0.0316, +0.0914] | 2,246 labelled, 131 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.SPEED_BAND` | UNAVAILABLE | | | 0 labelled, 0 pos | no positive label for this token among its 0 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.CORRIDOR_OFFSET` | 0.2500 | 0.2299 | +0.0201 [-0.0411, +0.0848] | 695 labelled, 120 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.EVADE_IN_CORRIDOR` | 0.4693 | 0.4556 | +0.0137 [-0.0383, +0.0685] | 695 labelled, 110 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.OVERTAKE_VEHICLE` | 0.0000 | 0.0952 | -0.0952 (no interval) | 695 labelled, 20 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.MERGE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 25 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.GAP_TARGET` | 0.0606 | 0.0000 | +0.0606 [+0.0000, +0.1861] | 695 labelled, 60 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.REACT_ON_ONCOMING` | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 695 labelled, 50 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TAKE_EXIT_L` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TAKE_EXIT_R` | 0.0000 | 0.1081 | -0.1081 (no interval) | 695 labelled, 30 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label for this token among its 695 labelled windows on this grid, so its F1 is undefined (the eval reports None) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_RED` | 0.4565 | 0.5000 | -0.0435 [-0.1921, +0.0171] | 720 labelled, 65 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_YELLOW` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_GREEN` | 0.1000 | 0.2299 | -0.1299 [-0.2883, +0.0000] | 695 labelled, 75 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.LANE_CHANGE_L` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |
| `tactical.heads.goal.per_token.LANE_CHANGE_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 10 pos | the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; the point values are reported |

### STRATEGIC, S's checkpoint error

| cell | ORC_s0 | ORC0_s0 | ORC_s0 - ORC0_s0 [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `strategic.position_error_m.30` | 7.0066 | 6.9841 | +0.0225 [-0.0087, +0.0637] | 2,322 / 137 | no |
| `strategic.position_error_m.50` | 8.0936 | 8.0859 | +0.0077 [-0.0545, +0.0757] | 2,319 / 137 | no |
| `strategic.position_error_m.80` | 11.6353 | 11.6716 | -0.0363 [-0.1967, +0.1121] | 2,294 / 136 | no |
| `strategic.heading_error_deg.30` | 4.2163 | 4.2264 | -0.0101 [-0.0989, +0.0756] | 2,322 / 137 | no |
| `strategic.heading_error_deg.50` | 7.1321 | 7.1867 | -0.0546 [-0.2258, +0.1034] | 2,319 / 137 | no |
| `strategic.heading_error_deg.80` | 13.9333 | 14.0768 | -0.1435 [-0.6418, +0.2594] | 2,294 / 136 | no |
