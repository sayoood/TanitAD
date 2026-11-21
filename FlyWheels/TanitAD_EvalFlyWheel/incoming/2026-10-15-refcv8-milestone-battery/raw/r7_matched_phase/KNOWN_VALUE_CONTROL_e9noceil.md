# Known-value control: re-rolled refcv7-50,400 vs the banked `step50400_a6text`

* generated 2026-10-18T04:09:51; new tree `C:/Users/Admin/refcv8_eval_archive/battery/r7ctle9noceil_step50400_smoke`; banked tree `E:/refcv7_eval_kit/battery/step50400_a6text`; tolerance on `os`: 0.001 m
* **VERDICT: PASS**

## inference seed 0: 137 windows / 4 clips; gate_ok = **True**

| file:key | class | max abs diff | rows differing / rows | non-finite mismatches |
|---|---|---|---|---|
| dec:a_star | model | 0.000e+00 | 0 / 137 | 0 |
| dec:anchor_acc | model | 0.000e+00 | 0 / 137 | 0 |
| dec:ego_t0 | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:ep_poses | exact | 0.000e+00 | 0 / 4 | 0 |
| dec:fan_term_heading_nav_flipped | model | 0.000e+00 | 0 / 137 | 0 |
| dec:fan_term_heading_nav_shuffled | shuf | 6.250e+00 | 51 / 137 | 0 |
| dec:fan_term_heading_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:fan_term_heading_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:goal_dist_sel | model | 0.000e+00 | 0 / 137 | 0 |
| dec:goal_gate | model | 0.000e+00 | 0 / 137 | 0 |
| dec:goal_score_absmean | model | 2.802e-01 | 51 / 137 | 0 |
| dec:gstr_nav_flipped | model | 0.000e+00 | 0 / 137 | 0 |
| dec:gstr_nav_shuffled | shuf | 0.000e+00 | 0 / 137 | 0 |
| dec:gstr_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:gstr_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:gt_future_ext | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:gt_future_valid_ext | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:ha_controls | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:lat_label | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:lat_pred_nav_flipped | model | 0.000e+00 | 0 / 137 | 0 |
| dec:lat_pred_nav_shuffled | shuf | 7.000e+00 | 9 / 137 | 0 |
| dec:lat_pred_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:lat_pred_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:lon_label | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:lon_pred_nav_flipped | model | 0.000e+00 | 0 / 137 | 0 |
| dec:lon_pred_nav_shuffled | shuf | 6.000e+00 | 13 / 137 | 0 |
| dec:lon_pred_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:lon_pred_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:nav_cmd | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:nav_cmd_flip | info | 0.000e+00 | 0 / 137 | 0 |
| dec:nav_cmd_shuf | shuf | 2.000e+00 | 51 / 137 | 0 |
| dec:nav_injected_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:nav_injected_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:nav_valid | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:plan_full_nav_flipped | model | 0.000e+00 | 0 / 137 | 0 |
| dec:plan_full_nav_shuffled | shuf | 9.081e+00 | 51 / 137 | 0 |
| dec:plan_full_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:plan_full_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:pose_last | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:reach_keep_nav_flipped | model | 0.000e+00 | 0 / 137 | 0 |
| dec:reach_keep_nav_shuffled | shuf | 0.000e+00 | 0 / 137 | 0 |
| dec:reach_keep_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:reach_keep_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:route_label | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:route_pred_nav_flipped | model | 0.000e+00 | 0 / 137 | 0 |
| dec:route_pred_nav_shuffled | shuf | 0.000e+00 | 0 / 137 | 0 |
| dec:route_pred_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:route_pred_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_agrees_oracle | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_bank_nav_flipped | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_bank_nav_shuffled | shuf | 1.315e+01 | 4 / 137 | 0 |
| dec:sel_bank_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_bank_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_idx | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_idx_base | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_idx_nav_flipped | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_idx_nav_shuffled | shuf | 9.000e+00 | 4 / 137 | 0 |
| dec:sel_idx_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_idx_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_score_max | model | 0.000e+00 | 0 / 137 | 0 |
| ep:g | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha0 | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha0_ext | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:oracle_sel | model | 0.000e+00 | 0 / 137 | 0 |
| ep:os | gate | 0.000e+00 | 0 / 137 | 0 |
| ep:os_navflip | model | 0.000e+00 | 0 / 137 | 0 |
| ep:os_navshuf | shuf | 7.780e-01 | 51 / 137 | 0 |
| ep:os_navzero | model | 0.000e+00 | 0 / 137 | 0 |
| ep:v0 | exact | 0.000e+00 | 0 / 137 | 0 |

## inference seed 1: 137 windows / 4 clips; gate_ok = **True**

| file:key | class | max abs diff | rows differing / rows | non-finite mismatches |
|---|---|---|---|---|
| dec:a_star | model | 0.000e+00 | 0 / 137 | 0 |
| dec:anchor_acc | model | 0.000e+00 | 0 / 137 | 0 |
| dec:ego_t0 | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:ep_poses | exact | 0.000e+00 | 0 / 4 | 0 |
| dec:fan_term_heading_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:goal_dist_sel | model | 0.000e+00 | 0 / 137 | 0 |
| dec:goal_gate | model | 0.000e+00 | 0 / 137 | 0 |
| dec:goal_score_absmean | model | 0.000e+00 | 0 / 137 | 0 |
| dec:gstr_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:gt_future_ext | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:gt_future_valid_ext | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:ha_controls | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:lat_label | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:lat_pred_nav_shuffled | shuf | 0.000e+00 | 0 / 137 | 0 |
| dec:lat_pred_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:lat_pred_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:lon_label | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:lon_pred_nav_shuffled | shuf | 0.000e+00 | 0 / 137 | 0 |
| dec:lon_pred_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:lon_pred_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:nav_cmd | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:nav_cmd_shuf | shuf | 2.000e+00 | 51 / 137 | 0 |
| dec:nav_injected_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:nav_injected_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:nav_valid | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:plan_full_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:pose_last | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:reach_keep_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:route_label | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:route_pred_nav_shuffled | shuf | 0.000e+00 | 0 / 137 | 0 |
| dec:route_pred_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:route_pred_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_agrees_oracle | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_bank_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_idx | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_idx_base | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_idx_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_score_max | model | 0.000e+00 | 0 / 137 | 0 |
| ep:g | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha0 | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha0_ext | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:oracle_sel | model | 0.000e+00 | 0 / 137 | 0 |
| ep:os | gate | 0.000e+00 | 0 / 137 | 0 |
| ep:v0 | exact | 0.000e+00 | 0 / 137 | 0 |

## Diagnostic (NOT part of the verdict): the max-speed ceiling filter, ON vs OFF, new path vs banked roll

* seed 0: {"seed": 0, "n_windows": 137, "new_filter_OFF_vs_banked_os_max_abs": 0, "new_filter_OFF_vs_banked_plan_full_max_abs": 0, "new_filter_ON_vs_banked_os_max_abs": 0, "new_filter_ON_vs_banked_plan_full_max_abs": 0, "new_roll_windows_where_filter_changes_selection": 0, "banked_roll_n_windows": 4754, "banked_roll_windows_where_filter_changes_selection": 0, "banked_roll_filter_on_vs_off_traj_max_abs": 0.0, "reading": "filter-OFF arm of the new path reproduces the banked plan to <= tol"}
* seed 1: {"seed": 1, "status": "no filter_off arm in one of the two dumps (os-only roll, or the filter was off when rolled)"}
