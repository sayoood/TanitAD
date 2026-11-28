**G0 = FAIL**; reasons: ['wrapper control failed: max rel 3.921e-03', 'M1 (no equalize) stayed inside the SMOOTH tolerance: the gate has no power -> VOID']; class counts {'MATCHED': 15, 'COUNT': 42, 'SMOOTH': 20, 'EXCLUDED': 1, 'STOCHASTIC': 5}; medians {'SMOOTH': 0.0009571043707606191, 'MATCHED': 0.0018559944739005392}

wrapper control (batch 4, micro 1+3, DDIM eps = 0 in both): max rel diff 0.003920739304811635 over 81 terms -> FAIL

strict load: missing 0, unexpected 0, 1101 keys, step 5000; param_breakdown equal to config.json: True (total 100,893,747); anchor file vs ckpt buffers max |Δ| 0.0

| term | class | in-run (Thor, step 5000) | dev-box mean over 8 seeds | sd | tolerance | verdict |
|---|---|---|---|---|---|---|
| `eval_agent_n_dropped` | COUNT | 0.0 | 0 | 0 | abs<=1e-5 | OK |
| `eval_agent_n_labelled` | COUNT | 15.125 | 15.125 | 0 | abs<=1e-5 | OK |
| `eval_agent_n_matched` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_agent_n_raw` | COUNT | 442.125 | 442.125 | 0 | abs<=1e-5 | OK |
| `eval_agent_n_target` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_agent_n_truncated` | COUNT | 0.0 | 0 | 0 | abs<=1e-5 | OK |
| `eval_agent_n_windows` | COUNT | 16.0 | 16 | 0 | abs<=1e-5 | OK |
| `eval_agent_rows_no_cam` | COUNT | 0.0 | 0 | 0 | abs<=1e-5 | OK |
| `eval_agent_rows_with_cam` | COUNT | 15.125 | 15.125 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_centre` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_cls` | COUNT | 1.96915 | 1.96915 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_dropped` | COUNT | 0.0 | 0 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_dropped_not_visible` | COUNT | 378.5 | 378.5 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_h` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_labelled` | COUNT | 15.125 | 15.125 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_matched` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_occ` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_rates` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_size` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_target` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_target_prefilter` | COUNT | 442.125 | 442.125 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_target_visible` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_windows` | COUNT | 16.0 | 16 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_yaw` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_box3d_n_z` | COUNT | 63.625 | 63.625 | 0 | abs<=1e-5 | OK |
| `eval_box3d_visible_filter` | COUNT | 1.0 | 1 | 0 | abs<=1e-5 | OK |
| `eval_ego_injected` | COUNT | 0.0 | 0 | 0 | abs<=1e-5 | OK |
| `eval_map_gt_drivable_frac` | COUNT | 0.35833 | 0.35833 | 0 | abs<=1e-5 | OK |
| `eval_map_n_labelled` | COUNT | 15.875 | 15.875 | 0 | abs<=1e-5 | OK |
| `eval_map_n_windows` | COUNT | 16.0 | 16 | 0 | abs<=1e-5 | OK |
| `eval_n_map_cells` | COUNT | 104563.5 | 104564 | 0 | abs<=1e-5 | OK |
| `eval_n_map_cells_seen` | COUNT | 117322.25 | 117322 | 0 | abs<=1e-5 | OK |
| `eval_n_map_cells_unobserved` | COUNT | 12758.75 | 12758.8 | 0 | abs<=1e-5 | OK |
| `eval_nav_injected` | COUNT | 1.0 | 1 | 0 | abs<=1e-5 | OK |
| `eval_slot_valid_frac` | COUNT | 0.94824 | 0.94824 | 0 | abs<=1e-5 | OK |
| `eval_tac_label_rows` | COUNT | 4.0 | 4 | 0 | abs<=1e-5 | OK |
| `eval_tac_label_v7` | COUNT | 1.0 | 1 | 0 | abs<=1e-5 | OK |
| `eval_tacv6_n_scene_mean` | COUNT | 580.0 | 580 | 0 | abs<=1e-5 | OK |
| `eval_tacv6_n_supervised_goal_cells` | COUNT | 37.5 | 37.5 | 0 | abs<=1e-5 | OK |
| `eval_tacv6_n_supervised_lat` | COUNT | 4.0 | 4 | 0 | abs<=1e-5 | OK |
| `eval_tacv6_n_supervised_lon` | COUNT | 4.0 | 4 | 0 | abs<=1e-5 | OK |
| `eval_windows` | COUNT | 128 | 128 | 0 | abs<=1e-5 | OK |
| `eval_goal_gate_grad` | EXCLUDED | 0.00549 |  |  |  | EXCLUDED |
| `eval_agent_centre` | MATCHED | 2.57125 | 2.57815 | 0 | rel<=0.15 (rel dev 2.68e-03) | OK |
| `eval_agent_cls` | MATCHED | 1.62724 | 1.64052 | 0 | rel<=0.15 (rel dev 8.16e-03) | OK |
| `eval_agent_presence` | MATCHED | 0.07787 | 0.07782 | 0 | rel<=0.15 (rel dev 6.42e-04) | OK |
| `eval_agent_size` | MATCHED | 1.19714 | 1.19557 | 0 | rel<=0.15 (rel dev 1.31e-03) | OK |
| `eval_agent_yaw` | MATCHED | 0.58824 | 0.58601 | 0 | rel<=0.15 (rel dev 3.79e-03) | OK |
| `eval_box3d` | MATCHED | 9.50372 | 9.50764 | 0 | rel<=0.15 (rel dev 4.12e-04) | OK |
| `eval_box3d_centre` | MATCHED | 2.43911 | 2.44323 | 0 | rel<=0.15 (rel dev 1.69e-03) | OK |
| `eval_box3d_cls` | MATCHED | 1.54825 | 1.54828 | 0 | rel<=0.15 (rel dev 1.94e-05) | OK |
| `eval_box3d_h` | MATCHED | 0.16329 | 0.16312 | 0 | rel<=0.15 (rel dev 1.04e-03) | OK |
| `eval_box3d_occ` | MATCHED | 0.00013 | 0.00013 | 0 | rel<=0.15 (rel dev 0.00e+00) | OK |
| `eval_box3d_presence` | MATCHED | 0.0768 | 0.07685 | 0 | rel<=0.15 (rel dev 6.51e-04) | OK |
| `eval_box3d_rates` | MATCHED | 6.47712 | 6.46842 | 0 | rel<=0.15 (rel dev 1.34e-03) | OK |
| `eval_box3d_size` | MATCHED | 1.19345 | 1.19806 | 0 | rel<=0.15 (rel dev 3.86e-03) | OK |
| `eval_box3d_yaw` | MATCHED | 0.60311 | 0.60189 | 0 | rel<=0.15 (rel dev 2.02e-03) | OK |
| `eval_box3d_z` | MATCHED | 0.24108 | 0.24194 | 0 | rel<=0.15 (rel dev 3.57e-03) | OK |
| `eval_anchor_acc` | SMOOTH | 0.59375 | 0.59375 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 0.00e+00) | OK |
| `eval_cls` | SMOOTH | 0.00233 | 0.00233 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 0.00e+00) | OK |
| `eval_goal2s_err_m` | SMOOTH | 4.79038 | 4.77622 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 2.96e-03) | OK |
| `eval_goal_gate` | SMOOTH | 0.05547 | 0.05547 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 0.00e+00) | OK |
| `eval_goal_tac` | SMOOTH | 12.55039 | 12.5189 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 2.51e-03) | OK |
| `eval_lat` | SMOOTH | 0.50641 | 0.5065 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.78e-04) | OK |
| `eval_lat_tac` | SMOOTH | 0.77799 | 0.77642 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 2.02e-03) | OK |
| `eval_lon` | SMOOTH | 1.20094 | 1.20226 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.10e-03) | OK |
| `eval_lon_tac` | SMOOTH | 1.30477 | 1.30234 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.86e-03) | OK |
| `eval_map` | SMOOTH | 0.7582 | 0.75813 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 9.23e-05) | OK |
| `eval_map_iou_drivable` | SMOOTH | 0.66459 | 0.66464 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 7.52e-05) | OK |
| `eval_map_pred_drivable_frac` | SMOOTH | 0.35076 | 0.35016 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.71e-03) | OK |
| `eval_map_pred_drivable_prob_mean` | SMOOTH | 0.37056 | 0.36988 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.84e-03) | OK |
| `eval_route` | SMOOTH | 1.10531 | 1.10533 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.81e-05) | OK |
| `eval_tac_v6` | SMOOTH | 0.07903 | 0.07897 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 7.59e-04) | OK |
| `eval_tacv6_goal_bce` | SMOOTH | 0.66514 | 0.66468 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 6.92e-04) | OK |
| `eval_tacv6_goal_conf_bce` | SMOOTH | 0.22084 | 0.22102 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 8.15e-04) | OK |
| `eval_tacv6_lat_ce` | SMOOTH | 0.68525 | 0.68519 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 8.76e-05) | OK |
| `eval_tacv6_lon_ce` | SMOOTH | 1.32327 | 1.3217 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.19e-03) | OK |
| `eval_tacv6_total_weighted` | SMOOTH | 0.07903 | 0.07897 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 7.59e-04) | OK |
| `eval_goal_score_absmean` | STOCHASTIC | 6.95675 | 6.94498 | 0.000934 | 99% PI [6.8721, 7.0179] | OK |
| `eval_law` | STOCHASTIC | 0.0002 | 0.00019375 | 5.18e-06 | 99% PI [0.0001726, 0.0002149] | OK |
| `eval_loss` | STOCHASTIC | 28.6892 | 28.7069 | 0.0253 | 99% PI [28.326, 29.088] | OK |
| `eval_sel_v3` | STOCHASTIC | 2.18423 | 2.1952 | 0.0243 | 99% PI [2.0832, 2.3072] | OK |
| `eval_traj` | STOCHASTIC | 0.64319 | 0.645375 | 0.0093 | 99% PI [0.60439, 0.68636] | OK |

**M1 (lift bank without equalize_bottom_rows, = refcv3_arm.py:1101)**: 0 SMOOTH term(s) leave tolerance:
