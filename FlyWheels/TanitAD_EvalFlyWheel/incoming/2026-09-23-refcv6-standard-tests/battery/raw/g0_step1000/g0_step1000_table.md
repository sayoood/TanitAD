**G0 = VOID**; reasons: ['M1 (no equalize) stayed inside the SMOOTH tolerance: the gate has no power -> VOID']; class counts {'MATCHED': 15, 'COUNT': 42, 'SMOOTH': 20, 'EXCLUDED': 1, 'STOCHASTIC': 5}; medians {'SMOOTH': 0.0002909970004923747, 'MATCHED': 0.00014741316304741844}

(as run, before the COUNT-rule correction: G0 = FAIL; reasons: ['eval_lon_tac [COUNT] in-run 1.48022 vs seeds mean 1.47915 (sd 0)', 'eval_tacv6_lon_ce [COUNT] in-run 1.70872 vs seeds mean 1.70848 (sd 0)', 'M1 (no equalize) stayed inside the SMOOTH tolerance: the gate has no power -> VOID'])

wrapper control (batch 4, micro 1+3, DDIM eps = 0 in both): max rel diff 0.000844279441418691 over 81 terms -> PASS

strict load: missing 0, unexpected 0, 1101 keys, step 1000; param_breakdown equal to config.json: True (total 100,893,747); anchor file vs ckpt buffers max |Δ| 0.0

| term | class | in-run (Thor, step 1000) | dev-box mean over 8 seeds | sd | tolerance | verdict |
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
| `eval_goal_gate_grad` | EXCLUDED | 0.00306 |  |  |  | EXCLUDED |
| `eval_agent_centre` | MATCHED | 3.20274 | 3.19821 | 0 | rel<=0.15 (rel dev 1.41e-03) | OK |
| `eval_agent_cls` | MATCHED | 1.99752 | 1.99742 | 0 | rel<=0.15 (rel dev 5.01e-05) | OK |
| `eval_agent_presence` | MATCHED | 0.08322 | 0.08318 | 0 | rel<=0.15 (rel dev 4.81e-04) | OK |
| `eval_agent_size` | MATCHED | 1.29275 | 1.29258 | 0 | rel<=0.15 (rel dev 1.32e-04) | OK |
| `eval_agent_yaw` | MATCHED | 0.71245 | 0.71044 | 0 | rel<=0.15 (rel dev 2.82e-03) | OK |
| `eval_box3d` | MATCHED | 10.78042 | 10.7815 | 0 | rel<=0.15 (rel dev 1.05e-04) | OK |
| `eval_box3d_centre` | MATCHED | 2.70565 | 2.70717 | 0 | rel<=0.15 (rel dev 5.62e-04) | OK |
| `eval_box3d_cls` | MATCHED | 1.97895 | 1.97915 | 0 | rel<=0.15 (rel dev 1.01e-04) | OK |
| `eval_box3d_h` | MATCHED | 0.17189 | 0.17198 | 0 | rel<=0.15 (rel dev 5.24e-04) | OK |
| `eval_box3d_occ` | MATCHED | 0.02088 | 0.0209 | 0 | rel<=0.15 (rel dev 9.58e-04) | OK |
| `eval_box3d_presence` | MATCHED | 0.08641 | 0.08642 | 0 | rel<=0.15 (rel dev 1.16e-04) | OK |
| `eval_box3d_rates` | MATCHED | 7.13562 | 7.13553 | 0 | rel<=0.15 (rel dev 1.26e-05) | OK |
| `eval_box3d_size` | MATCHED | 1.28579 | 1.286 | 0 | rel<=0.15 (rel dev 1.63e-04) | OK |
| `eval_box3d_yaw` | MATCHED | 0.6585 | 0.65763 | 0 | rel<=0.15 (rel dev 1.32e-03) | OK |
| `eval_box3d_z` | MATCHED | 0.31499 | 0.315 | 0 | rel<=0.15 (rel dev 3.17e-05) | OK |
| `eval_anchor_acc` | SMOOTH | 0.05469 | 0.05469 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 0.00e+00) | OK |
| `eval_cls` | SMOOTH | 0.05343 | 0.05341 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 3.74e-04) | OK |
| `eval_goal2s_err_m` | SMOOTH | 5.21278 | 5.23527 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 4.31e-03) | OK |
| `eval_goal_gate` | SMOOTH | 0.00303 | 0.00303 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 0.00e+00) | OK |
| `eval_goal_tac` | SMOOTH | 14.05123 | 14.0745 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.66e-03) | OK |
| `eval_lat` | SMOOTH | 0.51794 | 0.51806 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 2.32e-04) | OK |
| `eval_lat_tac` | SMOOTH | 0.96521 | 0.9666 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.44e-03) | OK |
| `eval_lon` | SMOOTH | 0.89348 | 0.89374 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 2.91e-04) | OK |
| `eval_lon_tac` | SMOOTH | 1.48022 | 1.47915 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 7.23e-04) | OK |
| `eval_map` | SMOOTH | 0.88997 | 0.88907 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.01e-03) | OK |
| `eval_map_iou_drivable` | SMOOTH | 0.58274 | 0.58315 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 7.04e-04) | OK |
| `eval_map_pred_drivable_frac` | SMOOTH | 0.29334 | 0.29359 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 8.52e-04) | OK |
| `eval_map_pred_drivable_prob_mean` | SMOOTH | 0.35712 | 0.3572 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 2.24e-04) | OK |
| `eval_route` | SMOOTH | 1.11143 | 1.11174 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 2.79e-04) | OK |
| `eval_tac_v6` | SMOOTH | 0.10704 | 0.10704 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 0.00e+00) | OK |
| `eval_tacv6_goal_bce` | SMOOTH | 0.9109 | 0.91101 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.21e-04) | OK |
| `eval_tacv6_goal_conf_bce` | SMOOTH | 0.46579 | 0.46593 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 3.01e-04) | OK |
| `eval_tacv6_lat_ce` | SMOOTH | 0.92925 | 0.92924 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.08e-05) | OK |
| `eval_tacv6_lon_ce` | SMOOTH | 1.70872 | 1.70848 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 1.40e-04) | OK |
| `eval_tacv6_total_weighted` | SMOOTH | 0.10704 | 0.10704 | 0 | abs<=1e-3 (|x|<0.1) else rel<=0.01 (rel dev 0.00e+00) | OK |
| `eval_goal_score_absmean` | STOCHASTIC | 6.81519 | 6.83624 | 0.000957 | 99% PI [6.7643, 6.9082] | OK |
| `eval_law` | STOCHASTIC | 0.00076 | 0.00076875 | 3.54e-06 | 99% PI [0.00074794, 0.00078956] | OK |
| `eval_loss` | STOCHASTIC | 34.26996 | 34.2399 | 0.0235 | 99% PI [33.81, 34.67] | OK |
| `eval_sel_v3` | STOCHASTIC | 3.25354 | 3.22487 | 0.0253 | 99% PI [3.0986, 3.3511] | OK |
| `eval_traj` | STOCHASTIC | 1.21898 | 1.21337 | 0.00913 | 99% PI [1.1673, 1.2594] | OK |

**M1 (lift bank without equalize_bottom_rows, = refcv3_arm.py:1101)**: 0 SMOOTH term(s) leave tolerance:
