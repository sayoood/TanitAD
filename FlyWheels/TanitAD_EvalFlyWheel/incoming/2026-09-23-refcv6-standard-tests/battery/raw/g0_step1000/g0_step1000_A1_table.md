**G0-A1 = PASS** (SPEC AMENDMENT A1); reasons: none; inference seed 0

| mutation | the defect | terms outside tolerance | largest moves (in-run → mutated, rel) |
|---|---|---|---|
| m2_vmax_withheld | max-speed input withheld (`v_max_valid = 0`) | 6 | `eval_tacv6_goal_bce` 0.9109 → 1.0244 (12.5 %); `eval_tacv6_goal_conf_bce` 0.46579 → 0.48812 (4.8 %); `eval_tac_v6` 0.10704 → 0.11023 (3.0 %); `eval_tacv6_total_weighted` 0.10704 → 0.11023 (3.0 %) |
| m3_ego_hist_zeroed | ego-history window zeroed | 4 | `eval_anchor_acc` 0.05469 → 0.14062 (157.1 %); `eval_sel_v3` 3.2535 → 3.0424 (6.5 %); `eval_traj` 1.219 → 1.2682 (4.0 %); `eval_cls` 0.05343 → 0.05218 (2.3 %) |
| m4_no_goal_posweight_mask | tactical-goal pos_weight / class mask absent (refcv3_arm.load_model's state) | 4 | `eval_tacv6_goal_bce` 0.9109 → 0.67402 (26.0 %); `eval_tacv6_goal_conf_bce` 0.46579 → 0.5511 (18.3 %); `eval_tac_v6` 0.10704 → 0.09841 (8.1 %); `eval_tacv6_total_weighted` 0.10704 → 0.09841 (8.1 %) |
