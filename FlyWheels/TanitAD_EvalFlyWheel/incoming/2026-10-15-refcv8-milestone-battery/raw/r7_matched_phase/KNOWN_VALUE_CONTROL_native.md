# Known-value control: re-rolled refcv7-50,400 vs the banked `step50400_a6text`

* generated 2026-10-18T03:54:29; new tree `C:/Users/Admin/refcv8_eval_archive/battery/r7ctlnative_step50400_smoke`; banked tree `E:/refcv7_eval_kit/battery/step50400_a6text`; tolerance on `os`: 0.001 m
* **VERDICT: FAIL (-> every refcv7 milestone number from this path is VOID)**
  * seed 0 ep:os: max abs 3.676e+00 m > 0.001 m (36/137 windows differ; 0 non-finite mismatches)
  * seed 1 ep:os: max abs 3.474e+00 m > 0.001 m (34/137 windows differ; 0 non-finite mismatches)

## inference seed 0: 137 windows / 4 clips; gate_ok = **False**

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
| dec:goal_dist_sel | model | 2.509e+00 | 36 / 137 | 0 |
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
| dec:plan_full_nav_flipped | model | 2.755e+01 | 37 / 137 | 0 |
| dec:plan_full_nav_shuffled | shuf | 2.872e+01 | 70 / 137 | 0 |
| dec:plan_full_nav_true | model | 2.798e+01 | 36 / 137 | 0 |
| dec:plan_full_nav_zero | model | 3.660e+01 | 28 / 137 | 0 |
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
| dec:sel_agrees_oracle | model | 1.000e+00 | 27 / 137 | 0 |
| dec:sel_bank_nav_flipped | model | 3.078e+01 | 37 / 137 | 0 |
| dec:sel_bank_nav_shuffled | shuf | 3.078e+01 | 35 / 137 | 0 |
| dec:sel_bank_nav_true | model | 3.078e+01 | 36 / 137 | 0 |
| dec:sel_bank_nav_zero | model | 3.772e+01 | 28 / 137 | 0 |
| dec:sel_idx | model | 2.800e+01 | 36 / 137 | 0 |
| dec:sel_idx_base | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_idx_nav_flipped | model | 3.600e+01 | 37 / 137 | 0 |
| dec:sel_idx_nav_shuffled | shuf | 3.600e+01 | 35 / 137 | 0 |
| dec:sel_idx_nav_true | model | 2.800e+01 | 36 / 137 | 0 |
| dec:sel_idx_nav_zero | model | 6.700e+01 | 28 / 137 | 0 |
| dec:sel_score_max | model | 0.000e+00 | 0 / 137 | 0 |
| ep:g | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha0 | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha0_ext | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:oracle_sel | model | 0.000e+00 | 0 / 137 | 0 |
| ep:os | gate | 3.676e+00 | 36 / 137 | 0 |
| ep:os_navflip | model | 3.638e+00 | 37 / 137 | 0 |
| ep:os_navshuf | shuf | 3.822e+00 | 70 / 137 | 0 |
| ep:os_navzero | model | 7.557e+00 | 28 / 137 | 0 |
| ep:v0 | exact | 0.000e+00 | 0 / 137 | 0 |

## inference seed 1: 137 windows / 4 clips; gate_ok = **False**

| file:key | class | max abs diff | rows differing / rows | non-finite mismatches |
|---|---|---|---|---|
| dec:a_star | model | 0.000e+00 | 0 / 137 | 0 |
| dec:anchor_acc | model | 0.000e+00 | 0 / 137 | 0 |
| dec:ego_t0 | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:ep_poses | exact | 0.000e+00 | 0 / 4 | 0 |
| dec:fan_term_heading_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:goal_dist_sel | model | 2.248e+00 | 34 / 137 | 0 |
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
| dec:plan_full_nav_true | model | 2.669e+01 | 34 / 137 | 0 |
| dec:pose_last | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:reach_keep_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:route_label | exact | 0.000e+00 | 0 / 137 | 0 |
| dec:route_pred_nav_shuffled | shuf | 0.000e+00 | 0 / 137 | 0 |
| dec:route_pred_nav_true | model | 0.000e+00 | 0 / 137 | 0 |
| dec:route_pred_nav_zero | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_agrees_oracle | model | 1.000e+00 | 29 / 137 | 0 |
| dec:sel_bank_nav_true | model | 3.036e+01 | 34 / 137 | 0 |
| dec:sel_idx | model | 2.600e+01 | 34 / 137 | 0 |
| dec:sel_idx_base | model | 0.000e+00 | 0 / 137 | 0 |
| dec:sel_idx_nav_true | model | 2.600e+01 | 34 / 137 | 0 |
| dec:sel_score_max | model | 0.000e+00 | 0 / 137 | 0 |
| ep:g | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha0 | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:ha0_ext | exact | 0.000e+00 | 0 / 137 | 0 |
| ep:oracle_sel | model | 0.000e+00 | 0 / 137 | 0 |
| ep:os | gate | 3.474e+00 | 34 / 137 | 0 |
| ep:v0 | exact | 0.000e+00 | 0 / 137 | 0 |

## Diagnostic (NOT part of the verdict): the max-speed ceiling filter, ON vs OFF, new path vs banked roll

* seed 0: {"seed": 0, "n_windows": 137, "new_filter_OFF_vs_banked_os_max_abs": 0, "new_filter_OFF_vs_banked_plan_full_max_abs": 0, "new_filter_ON_vs_banked_os_max_abs": 3.6762771606445312, "new_filter_ON_vs_banked_plan_full_max_abs": 27.981958389282227, "new_roll_windows_where_filter_changes_selection": 36, "banked_roll_n_windows": 4754, "banked_roll_windows_where_filter_changes_selection": 0, "banked_roll_filter_on_vs_off_traj_max_abs": 0.0, "reading": "filter-OFF arm of the new path reproduces the banked plan to <= tol"}
* seed 1: {"seed": 1, "status": "no filter_off arm in one of the two dumps (os-only roll, or the filter was off when rolled)"}
