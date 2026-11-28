# NAV_FEED -- does feeding refcv8 the nav it was TRAINED on change the plan? (EMA-40k, os only)

**COMMITTED LABEL: `FEED-FIX-MATTERS`**

tier: T1 self-action OPEN LOOP on logged eval139 frames (NOT driving performance); the nav token / args are ORACLE-derived (v9 labels built offline from the ego's future path)
estimator: paired episode-cluster bootstrap (taniteval.ci.paired_episode_cluster_bootstrap), B 2000, seed 0, over the episodes present in the set: answers 'another draw of EPISODES' at a fixed checkpoint and a fixed inference seed; NOT another training run (H-ESTIM-SEED-1), NOT another inference run (seeds 0 and 1 are shown separately, never pooled)
evidence class: MEASURED (this run's dumps under out_dir; every cell cites n windows / n episodes)

rule inputs: separated-negative by seed [True, True], separated (any sign) by seed [True, True], control ok True, |delta| > F by seed [True, True], effect/floor {'s0': 16.74175824175824, 's1': 17.13736263736264}

## Primary: perframe_full - clip, ADE 6 s (m, lower = better), MISMATCH set

| seed | delta [95 % CI] n windows/episodes | arm mean | clip mean |
|---|---|---|---|
| 0 | -0.6094 [-0.8506, -0.3802] n 1157/59 **sep** | +1.6476 | +2.2570 |
| 1 | -0.6238 [-0.8537, -0.4093] n 1157/59 **sep** | +1.6485 | +2.2723 |

inference-seed floor F (clip s1 - clip s0, ADE 6 s, all sealed windows): +0.0364 m (|point| +0.0182, CI half-width +0.0364)

## Controls

* C1 (perframe_token - clip on the 300 CONTROL windows, ADE 6 s), seed 0: |delta| +0.000000 vs 2F +0.0728 -> PASS; plans bit-identical to clip: True
* C1 (perframe_token - clip on the 300 CONTROL windows, ADE 6 s), seed 1: |delta| +0.000000 vs 2F +0.0728 -> PASS; plans bit-identical to clip: True

* C2_clip_reroll_reproduces_the_banked_battery: `{"clip_s0_vs_banked_os_s0": {"mean_ade_6s_clip_reroll": 2.260171043466634, "mean_ade_6s_banked_os": 2.2453555674427745, "abs_diff_m": 0.014815476023859464, "banked_seed_replicate_abs_diff_m": 0.007757568326866693}, "clip_s1_vs_banked_os_s1": {"mean_ade_6s_clip_reroll": 2.278334477776788, "mean_ade_6s_banked_os": 2.237597999115908, "abs_diff_m": 0.04073647866088015, "banked_seed_replicate_abs_diff_m": 0.007757568326866693}}`

* C3_gt_and_dump_consistency: `{"clip_s0": {"gt_ego_frame_vs_banked_g_max_abs_m": 4.5835427968654585e-06, "os_2s_vs_plan_full_first4_max_abs_m": 0.0}, "clip_s1": {"gt_ego_frame_vs_banked_g_max_abs_m": 4.5835427968654585e-06, "os_2s_vs_plan_full_first4_max_abs_m": 0.0}, "perframe_token_s0": {"gt_ego_frame_vs_banked_g_max_abs_m": 4.5835427968654585e-06, "os_2s_vs_plan_full_first4_max_abs_m": 0.0}, "perframe_token_s1": {"gt_ego_frame_vs_banked_g_max_abs_m": 4.5835427968654585e-06, "os_2s_vs_plan_full_first4_max_abs_m": 0.0}, "perframe_full_s0": {"gt_ego_frame_vs_banked_g_max_abs_m": 4.5835427968654585e-06, "os_2s_vs_plan_full_first4_max_abs_m": 0.0}, "perframe_full_s1": {"gt_ego_frame_vs_banked_g_max_abs_m": 4.5835427968654585e-06, "os_2s_vs_plan_full_first4_max_abs_m": 0.0}}`

* token_feed_as_designed: `{"clip_s0": {"nav_cmd_equals_the_banked_clip_token": true, "manifest_has_no_nav_feed_block": true}, "clip_s1": {"nav_cmd_equals_the_banked_clip_token": true, "manifest_has_no_nav_feed_block": true}, "perframe_token_s0": {"nav_cmd_equals_the_v9_per_frame_token": true, "nav_cmd_clip_equals_the_banked_clip_token": true, "manifest_mode_matches": true, "r8_nav_fed_nonzero_rows_equal_known_rows": true}, "perframe_token_s1": {"nav_cmd_equals_the_v9_per_frame_token": true, "nav_cmd_clip_equals_the_banked_clip_token": true, "manifest_mode_matches": true, "r8_nav_fed_nonzero_rows_equal_known_rows": true}, "perframe_full_s0": {"nav_cmd_equals_the_v9_per_frame_token": true, "nav_cmd_clip_equals_the_banked_clip_token": true, "manifest_mode_matches": true, "r8_nav_fed_nonzero_rows_equal_known_rows": true}, "perframe_full_s1": {"nav_cmd_equals_the_v9_per_frame_token": true, "nav_cmd_clip_equals_the_banked_clip_token": true, "manifest_mode_matches": true, "r8_nav_fed_nonzero_rows_equal_known_rows": true}}`

* windows identical across dumps and equal to the sealed union: True; missing dumps: []


## MISMATCH set (clip token != per-frame token): perframe_full_minus_clip (A - B; error metrics lower = better, accuracy higher = better)

| family | metric | seed 0 | seed 1 |
|---|---|---|---|
| ADE | ade_2s | -0.0397 [-0.0618, -0.0183] n 1157/59 **sep** | -0.0418 [-0.0620, -0.0238] n 1157/59 **sep** |
| ADE | ade_6s | -0.6094 [-0.8506, -0.3802] n 1157/59 **sep** | -0.6238 [-0.8537, -0.4093] n 1157/59 **sep** |
| LONGITUDINAL | LON_speed_mae_mps | -0.3504 [-0.5552, -0.1710] n 909/56 **sep** | -0.3668 [-0.5633, -0.1917] n 909/56 **sep** |
| LONGITUDINAL | LON_along_abs_final_m | -2.1479 [-3.3765, -1.0172] n 909/56 **sep** | -2.2224 [-3.4059, -1.1359] n 909/56 **sep** |
| LONGITUDINAL | LON_along_signed_final_m | +5.0266 [+3.8969, +6.1657] n 909/56 **sep** | +5.0598 [+3.8857, +6.2457] n 909/56 **sep** |
| LONGITUDINAL | LON_false_decel_on_gt_cruise | -0.5361 [-0.6608, -0.3978] n 429/42 **sep** | -0.5338 [-0.6667, -0.3848] n 429/42 **sep** |
| LATERAL | LAT_cross_abs_final_m | -1.0294 [-1.6200, -0.5052] n 909/56 **sep** | -1.0496 [-1.7079, -0.4772] n 909/56 **sep** |
| LATERAL | LAT_heading_err_final_deg | -4.4039 [-6.4818, -2.5921] n 867/54 **sep** | -4.4169 [-6.6072, -2.5503] n 863/54 **sep** |
| LATERAL | LAT_cross_mae_m | -0.3788 [-0.6124, -0.1777] n 909/56 **sep** | -0.3893 [-0.6538, -0.1676] n 909/56 **sep** |
| LATERAL | LAT_heading_mae_deg | -1.9699 [-3.0622, -1.0343] n 890/55 **sep** | -1.9911 [-3.1578, -1.0020] n 889/55 **sep** |
| TACTICAL | TAC_head_lat_acc | +0.4337 [+0.3380, +0.5313] n 973/56 **sep** | +0.4337 [+0.3380, +0.5313] n 973/56 **sep** |
| TACTICAL | TAC_head_lon_acc | +0.1330 [-0.0071, +0.2611] n 932/55 | +0.1330 [-0.0071, +0.2611] n 932/55 |
| TACTICAL | TAC_traj_lat_correct | +0.0715 [+0.0388, +0.1059] n 909/56 **sep** | +0.0726 [+0.0360, +0.1089] n 909/56 **sep** |
| TACTICAL | TAC_traj_lon_correct | +0.1683 [+0.0844, +0.2566] n 909/56 **sep** | +0.1760 [+0.0923, +0.2597] n 909/56 **sep** |
| STRATEGIC | STR_route_acc | +0.0000 [+0.0000, +0.0000] n 820/50 | +0.0000 [+0.0000, +0.0000] n 820/50 |

## MISMATCH set (clip token != per-frame token): perframe_token_minus_clip (A - B; error metrics lower = better, accuracy higher = better)

| family | metric | seed 0 | seed 1 |
|---|---|---|---|
| ADE | ade_2s | -0.0385 [-0.0595, -0.0182] n 1157/59 **sep** | -0.0396 [-0.0596, -0.0210] n 1157/59 **sep** |
| ADE | ade_6s | -0.5980 [-0.8377, -0.3644] n 1157/59 **sep** | -0.6083 [-0.8362, -0.3923] n 1157/59 **sep** |
| LONGITUDINAL | LON_speed_mae_mps | -0.3455 [-0.5497, -0.1636] n 909/56 **sep** | -0.3598 [-0.5563, -0.1841] n 909/56 **sep** |
| LONGITUDINAL | LON_along_abs_final_m | -2.1210 [-3.3644, -0.9868] n 909/56 **sep** | -2.1664 [-3.3540, -1.0781] n 909/56 **sep** |
| LONGITUDINAL | LON_along_signed_final_m | +4.9759 [+3.8282, +6.1227] n 909/56 **sep** | +5.0209 [+3.8702, +6.2371] n 909/56 **sep** |
| LONGITUDINAL | LON_false_decel_on_gt_cruise | -0.5385 [-0.6621, -0.3994] n 429/42 **sep** | -0.5361 [-0.6682, -0.3883] n 429/42 **sep** |
| LATERAL | LAT_cross_abs_final_m | -1.0335 [-1.6112, -0.5099] n 909/56 **sep** | -1.0315 [-1.6944, -0.4569] n 909/56 **sep** |
| LATERAL | LAT_heading_err_final_deg | -4.3816 [-6.4654, -2.5507] n 867/54 **sep** | -4.3004 [-6.5009, -2.3785] n 862/54 **sep** |
| LATERAL | LAT_cross_mae_m | -0.3822 [-0.6133, -0.1821] n 909/56 **sep** | -0.3834 [-0.6465, -0.1599] n 909/56 **sep** |
| LATERAL | LAT_heading_mae_deg | -1.9687 [-3.0587, -1.0319] n 890/55 **sep** | -1.9451 [-3.1195, -0.9466] n 889/55 **sep** |
| TACTICAL | TAC_head_lat_acc | +0.4337 [+0.3380, +0.5313] n 973/56 **sep** | +0.4337 [+0.3380, +0.5313] n 973/56 **sep** |
| TACTICAL | TAC_head_lon_acc | +0.1330 [-0.0071, +0.2611] n 932/55 | +0.1330 [-0.0071, +0.2611] n 932/55 |
| TACTICAL | TAC_traj_lat_correct | +0.0693 [+0.0369, +0.1035] n 909/56 **sep** | +0.0726 [+0.0359, +0.1105] n 909/56 **sep** |
| TACTICAL | TAC_traj_lon_correct | +0.1628 [+0.0777, +0.2514] n 909/56 **sep** | +0.1749 [+0.0891, +0.2613] n 909/56 **sep** |
| STRATEGIC | STR_route_acc | +0.0000 [+0.0000, +0.0000] n 820/50 | +0.0000 [+0.0000, +0.0000] n 820/50 |

## MISMATCH set (clip token != per-frame token): perframe_full_minus_perframe_token (A - B; error metrics lower = better, accuracy higher = better)

| family | metric | seed 0 | seed 1 |
|---|---|---|---|
| ADE | ade_2s | -0.0011 [-0.0050, +0.0024] n 1157/59 | -0.0022 [-0.0046, +0.0001] n 1157/59 |
| ADE | ade_6s | -0.0114 [-0.0275, +0.0058] n 1157/59 | -0.0156 [-0.0381, +0.0044] n 1157/59 |
| LONGITUDINAL | LON_speed_mae_mps | -0.0050 [-0.0128, +0.0036] n 909/56 | -0.0070 [-0.0203, +0.0051] n 909/56 |
| LONGITUDINAL | LON_along_abs_final_m | -0.0269 [-0.0721, +0.0225] n 909/56 | -0.0561 [-0.1476, +0.0222] n 909/56 |
| LONGITUDINAL | LON_along_signed_final_m | +0.0507 [+0.0026, +0.1053] n 909/56 **sep** | +0.0389 [-0.0519, +0.1511] n 909/56 |
| LONGITUDINAL | LON_false_decel_on_gt_cruise | +0.0023 [+0.0000, +0.0082] n 429/42 | +0.0023 [-0.0085, +0.0155] n 429/42 |
| LATERAL | LAT_cross_abs_final_m | +0.0041 [-0.0157, +0.0231] n 909/56 | -0.0181 [-0.0732, +0.0150] n 909/56 |
| LATERAL | LAT_heading_err_final_deg | -0.0220 [-0.0600, +0.0076] n 872/55 | -0.0956 [-0.3173, +0.0286] n 866/54 |
| LATERAL | LAT_cross_mae_m | +0.0034 [-0.0050, +0.0129] n 909/56 | -0.0060 [-0.0263, +0.0065] n 909/56 |
| LATERAL | LAT_heading_mae_deg | -0.0011 [-0.0095, +0.0071] n 891/55 | -0.0459 [-0.1419, +0.0062] n 891/55 |
| TACTICAL | TAC_head_lat_acc | +0.0000 [+0.0000, +0.0000] n 973/56 | +0.0000 [+0.0000, +0.0000] n 973/56 |
| TACTICAL | TAC_head_lon_acc | +0.0000 [+0.0000, +0.0000] n 932/55 | +0.0000 [+0.0000, +0.0000] n 932/55 |
| TACTICAL | TAC_traj_lat_correct | +0.0022 [-0.0023, +0.0080] n 909/56 | +0.0000 [-0.0032, +0.0031] n 909/56 |
| TACTICAL | TAC_traj_lon_correct | +0.0055 [+0.0000, +0.0115] n 909/56 | +0.0011 [-0.0067, +0.0103] n 909/56 |
| STRATEGIC | STR_route_acc | +0.0000 [+0.0000, +0.0000] n 820/50 | +0.0000 [+0.0000, +0.0000] n 820/50 |

## CONTROL set (tokens equal): perframe_full_minus_clip (A - B; error metrics lower = better, accuracy higher = better)

| family | metric | seed 0 | seed 1 |
|---|---|---|---|
| ADE | ade_2s | +0.0036 [-0.0010, +0.0097] n 300/95 | +0.0024 [-0.0032, +0.0098] n 300/95 |
| ADE | ade_6s | -0.0185 [-0.0650, +0.0183] n 300/95 | -0.0253 [-0.0909, +0.0289] n 300/95 |
| LONGITUDINAL | LON_speed_mae_mps | -0.0219 [-0.0642, +0.0121] n 226/77 | -0.0269 [-0.0827, +0.0213] n 226/77 |
| LONGITUDINAL | LON_along_abs_final_m | -0.1448 [-0.3545, +0.0133] n 226/77 | -0.0391 [-0.2467, +0.1635] n 226/77 |
| LONGITUDINAL | LON_along_signed_final_m | +0.0971 [-0.0662, +0.3100] n 226/77 | +0.0334 [-0.1840, +0.2701] n 226/77 |
| LONGITUDINAL | LON_false_decel_on_gt_cruise | +0.0000 [+0.0000, +0.0000] n 59/42 | +0.0169 [+0.0000, +0.0526] n 59/42 |
| LATERAL | LAT_cross_abs_final_m | -0.0970 [-0.2657, +0.0505] n 226/77 | -0.2370 [-0.5362, -0.0096] n 226/77 **sep** |
| LATERAL | LAT_heading_err_final_deg | -0.1849 [-0.4628, +0.0705] n 178/73 | -0.3244 [-0.7536, -0.0056] n 177/73 **sep** |
| LATERAL | LAT_cross_mae_m | -0.0157 [-0.0537, +0.0217] n 226/77 | -0.0616 [-0.1376, -0.0006] n 226/77 **sep** |
| LATERAL | LAT_heading_mae_deg | -0.0654 [-0.1462, +0.0007] n 213/76 | -0.1360 [-0.4190, +0.0059] n 212/76 |
| TACTICAL | TAC_head_lat_acc | +0.0000 [+0.0000, +0.0000] n 245/85 | +0.0000 [+0.0000, +0.0000] n 245/85 |
| TACTICAL | TAC_head_lon_acc | +0.0000 [+0.0000, +0.0000] n 214/77 | +0.0000 [+0.0000, +0.0000] n 214/77 |
| TACTICAL | TAC_traj_lat_correct | +0.0044 [+0.0000, +0.0128] n 226/77 | +0.0044 [+0.0000, +0.0128] n 226/77 |
| TACTICAL | TAC_traj_lon_correct | +0.0265 [+0.0040, +0.0556] n 226/77 **sep** | +0.0221 [+0.0000, +0.0571] n 226/77 |
| STRATEGIC | STR_route_acc | +0.0000 [+0.0000, +0.0000] n 199/78 | +0.0000 [+0.0000, +0.0000] n 199/78 |

## CONTROL set (tokens equal): perframe_token_minus_clip (A - B; error metrics lower = better, accuracy higher = better)

| family | metric | seed 0 | seed 1 |
|---|---|---|---|
| ADE | ade_2s | +0.0000 [+0.0000, +0.0000] n 300/95 | +0.0000 [+0.0000, +0.0000] n 300/95 |
| ADE | ade_6s | +0.0000 [+0.0000, +0.0000] n 300/95 | +0.0000 [+0.0000, +0.0000] n 300/95 |
| LONGITUDINAL | LON_speed_mae_mps | +0.0000 [+0.0000, +0.0000] n 226/77 | +0.0000 [+0.0000, +0.0000] n 226/77 |
| LONGITUDINAL | LON_along_abs_final_m | +0.0000 [+0.0000, +0.0000] n 226/77 | +0.0000 [+0.0000, +0.0000] n 226/77 |
| LONGITUDINAL | LON_along_signed_final_m | +0.0000 [+0.0000, +0.0000] n 226/77 | +0.0000 [+0.0000, +0.0000] n 226/77 |
| LONGITUDINAL | LON_false_decel_on_gt_cruise | +0.0000 [+0.0000, +0.0000] n 59/42 | +0.0000 [+0.0000, +0.0000] n 59/42 |
| LATERAL | LAT_cross_abs_final_m | +0.0000 [+0.0000, +0.0000] n 226/77 | +0.0000 [+0.0000, +0.0000] n 226/77 |
| LATERAL | LAT_heading_err_final_deg | +0.0000 [+0.0000, +0.0000] n 178/73 | +0.0000 [+0.0000, +0.0000] n 177/73 |
| LATERAL | LAT_cross_mae_m | +0.0000 [+0.0000, +0.0000] n 226/77 | +0.0000 [+0.0000, +0.0000] n 226/77 |
| LATERAL | LAT_heading_mae_deg | +0.0000 [+0.0000, +0.0000] n 213/76 | +0.0000 [+0.0000, +0.0000] n 212/76 |
| TACTICAL | TAC_head_lat_acc | +0.0000 [+0.0000, +0.0000] n 245/85 | +0.0000 [+0.0000, +0.0000] n 245/85 |
| TACTICAL | TAC_head_lon_acc | +0.0000 [+0.0000, +0.0000] n 214/77 | +0.0000 [+0.0000, +0.0000] n 214/77 |
| TACTICAL | TAC_traj_lat_correct | +0.0000 [+0.0000, +0.0000] n 226/77 | +0.0000 [+0.0000, +0.0000] n 226/77 |
| TACTICAL | TAC_traj_lon_correct | +0.0000 [+0.0000, +0.0000] n 226/77 | +0.0000 [+0.0000, +0.0000] n 226/77 |
| STRATEGIC | STR_route_acc | +0.0000 [+0.0000, +0.0000] n 199/78 | +0.0000 [+0.0000, +0.0000] n 199/78 |

## CONTROL set (tokens equal): perframe_full_minus_perframe_token (A - B; error metrics lower = better, accuracy higher = better)

| family | metric | seed 0 | seed 1 |
|---|---|---|---|
| ADE | ade_2s | +0.0036 [-0.0010, +0.0097] n 300/95 | +0.0024 [-0.0032, +0.0098] n 300/95 |
| ADE | ade_6s | -0.0185 [-0.0650, +0.0183] n 300/95 | -0.0253 [-0.0909, +0.0289] n 300/95 |
| LONGITUDINAL | LON_speed_mae_mps | -0.0219 [-0.0642, +0.0121] n 226/77 | -0.0269 [-0.0827, +0.0213] n 226/77 |
| LONGITUDINAL | LON_along_abs_final_m | -0.1448 [-0.3545, +0.0133] n 226/77 | -0.0391 [-0.2467, +0.1635] n 226/77 |
| LONGITUDINAL | LON_along_signed_final_m | +0.0971 [-0.0662, +0.3100] n 226/77 | +0.0334 [-0.1840, +0.2701] n 226/77 |
| LONGITUDINAL | LON_false_decel_on_gt_cruise | +0.0000 [+0.0000, +0.0000] n 59/42 | +0.0169 [+0.0000, +0.0526] n 59/42 |
| LATERAL | LAT_cross_abs_final_m | -0.0970 [-0.2657, +0.0505] n 226/77 | -0.2370 [-0.5362, -0.0096] n 226/77 **sep** |
| LATERAL | LAT_heading_err_final_deg | -0.1849 [-0.4628, +0.0705] n 178/73 | -0.3244 [-0.7536, -0.0056] n 177/73 **sep** |
| LATERAL | LAT_cross_mae_m | -0.0157 [-0.0537, +0.0217] n 226/77 | -0.0616 [-0.1376, -0.0006] n 226/77 **sep** |
| LATERAL | LAT_heading_mae_deg | -0.0654 [-0.1462, +0.0007] n 213/76 | -0.1360 [-0.4190, +0.0059] n 212/76 |
| TACTICAL | TAC_head_lat_acc | +0.0000 [+0.0000, +0.0000] n 245/85 | +0.0000 [+0.0000, +0.0000] n 245/85 |
| TACTICAL | TAC_head_lon_acc | +0.0000 [+0.0000, +0.0000] n 214/77 | +0.0000 [+0.0000, +0.0000] n 214/77 |
| TACTICAL | TAC_traj_lat_correct | +0.0044 [+0.0000, +0.0128] n 226/77 | +0.0044 [+0.0000, +0.0128] n 226/77 |
| TACTICAL | TAC_traj_lon_correct | +0.0265 [+0.0040, +0.0556] n 226/77 **sep** | +0.0221 [+0.0000, +0.0571] n 226/77 |
| STRATEGIC | STR_route_acc | +0.0000 [+0.0000, +0.0000] n 199/78 | +0.0000 [+0.0000, +0.0000] n 199/78 |

## Exploratory split of the primary cell by (clip token -> per-frame token), perframe_full - clip, ADE 6 s (NOT an input to the label)

| group | seed 0 | seed 1 |
|---|---|---|
| FOLLOW->LEFT | +0.2547 [+0.0180, +0.3668] n 26/3 **sep** | -0.0739 [-0.1229, +0.1555] n 26/3 |
| FOLLOW->RIGHT | -0.8267 [-1.4712, -0.2721] n 76/5 **sep** | -0.8313 [-1.5523, -0.2037] n 76/5 **sep** |
| LEFT->FOLLOW | -1.1158 [-1.6461, -0.5662] n 280/12 **sep** | -1.0726 [-1.5754, -0.5386] n 280/12 **sep** |
| LEFT->RIGHT (opposite) | n/a (too few windows/episodes) | n/a (too few windows/episodes) |
| RIGHT->FOLLOW | -0.3604 [-0.5870, -0.1181] n 739/37 **sep** | -0.3836 [-0.6038, -0.1493] n 739/37 **sep** |
| RIGHT->LEFT (opposite) | -2.4510 [-5.0147, -0.4156] n 29/6 **sep** | -2.5483 [-5.2789, -0.1755] n 29/6 **sep** |

## Families that cannot be computed from these dumps

* **LONGITUDINAL distance-keeping (headway / time-gap / TTC to the lead agent)**: NOT COMPUTED here: the dumps carry no lead-block join (lead_block_common_grid needs the B1 lead block on a 0.2 s grid and the common instants with this 0.5 s plan grid); the v9 release's lead_gap columns are labels, not a plan-vs-GT distance-keeping instrument. Work item: join the B1 lead block to these dumps (taniteval.lead_metrics).
* **STRATEGIC route / goal-setting beyond the route-head accuracy**: NOT COMPUTED: needs the g_str goal-point head decode (not dumped as a row here)
