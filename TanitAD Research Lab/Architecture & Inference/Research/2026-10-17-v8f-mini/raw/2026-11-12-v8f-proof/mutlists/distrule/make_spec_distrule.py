"""make_spec_distrule.py -- the mutation definitions of DISTRULE (the rule, the wrapper that applies it, the margin and quantile-map forms) and the check that every old text occurs exactly once.
  python make_spec_distrule.py   -> writes spec_distrule.json (pod paths) next to it
Checks: test_dist_rule.py (the lever's own, 22 tests) for the rule and the wrapper; test_margin_qcal.py (written for this list, 13 tests) for the margin and quantile-map forms."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "F:/TanitAD-scratch/v8proof/distrule"
TD = ["src/test_dist_rule.py"]
TM = ["src/test_margin_qcal.py"]
D = "test_dist_rule.py::"
Q = "test_margin_qcal.py::"
DR, WR, MC, QF, QM, MG, QC = ("dist_rule.py", "v8f_t1_eval_rule.py", "margin_calib.py", "qcal_fit.py", "qcal_map.py", "v8f_t1_eval_rule_margin.py", "v8f_t1_eval_rule_qcal.py")

# (id, file, old, new, expected tests, test files)
M = [
    ("d1_cap_ignores_the_lead_motion", DR, "a_c = (g0 + v_L * dt * i - np.maximum(D, tau * v_prev)) if use_cap", "a_c = (g0 - np.maximum(D, tau * v_prev)) if use_cap",
     [D + "test_r5_stopped_lead_10m_gives_the_braking_limited_profile_and_counts_saturated_steps"], TD),
    ("d2_margin_takes_the_smaller_of_D_and_tau_v", DR, "np.maximum(D, tau * v_prev)) if use_cap", "np.minimum(D, tau * v_prev)) if use_cap",
     [D + "test_r5_progress_never_decreases_the_floor_holds_and_the_lag_form_equals_the_card_recursion"], TD),
    ("d3_braking_floor_sign_flipped", DR, "np.maximum(0.0, v_prev - b_max * dt) * dt) if use_floor", "np.maximum(0.0, v_prev + b_max * dt) * dt) if use_floor",
     [D + "test_r5_progress_never_decreases_the_floor_holds_and_the_lag_form_equals_the_card_recursion"], TD),
    ("d4_floor_takes_the_minimum", DR, "a_i = np.maximum(a_f, np.minimum(a_w, a_c)) if floor_lifts", "a_i = np.minimum(a_f, np.minimum(a_w, a_c)) if floor_lifts",
     [D + "test_r5_progress_never_decreases_the_floor_holds_and_the_lag_form_equals_the_card_recursion"], TD),
    ("d5_activation_gap_threshold_60_to_600", DR, "G_MAX_M = 60.0 ", "G_MAX_M = 600.0 ",
     [D + "test_r5_cap_binds_only_where_it_should_a_lead_far_enough_ahead_changes_nothing"], TD),
    ("d6_lag_not_carried", DR, "lag = aP[:, i] - a_i", "lag = np.zeros(B)",
     [D + "test_r5_progress_never_decreases_the_floor_holds_and_the_lag_form_equals_the_card_recursion"], TD),
    ("d7_speed_not_updated_from_the_new_arc", DR, "v_prev = (a_i - a_prev) / dt", "v_prev = v0.copy()",
     [D + "test_r5_progress_never_decreases_the_floor_holds_and_the_lag_form_equals_the_card_recursion"], TD),
    ("d8_unchanged_steps_not_kept_bit_for_bit", DR, "pts[keep] = p64[b][keep]", "pts[keep] = pts[keep]",
     [D + "test_r5_a_faster_lead_leaves_the_plan_unchanged_bit_for_bit"], TD),
    ("d9_tau_constant_zero", DR, "TAU_S = 1.2\n", "TAU_S = 0.0\n",
     [D + "test_r5_progress_never_decreases_the_floor_holds_and_the_lag_form_equals_the_card_recursion"], TD),
    ("w1_progress_offset_dropped_from_the_source_gap", WR, "g0 = g_src + a_l - a_m", "g0 = g_src",
     [D + "test_e2e_label_with_a_stopped_lead_reduces_progress_push_increases_it_plumb_and_a_far_lead_change_nothing"], TD),
    ("w2_closing_speed_dropped_from_the_lead_speed", WR, "v_L = np.maximum(0.0, v_l + c_src)", "v_L = np.maximum(0.0, v_l)",
     [D + "test_e2e_label_with_a_stopped_lead_reduces_progress_push_increases_it_plumb_and_a_far_lead_change_nothing"], TD),
    ("w3_rule_computed_but_plan_not_returned", WR, "        return out.plan", "        return plan",
     [D + "test_e2e_label_with_a_stopped_lead_reduces_progress_push_increases_it_plumb_and_a_far_lead_change_nothing", D + "test_mutation_the_line_that_applies_the_rule_removed_turns_the_reduction_check_red"], TD),
    ("w4_rule_never_active", WR, "out = R.distance_rule(plan, v0, present, g_src, g0, v_L, m=self.m, **self.kw)", "out = R.distance_rule(plan, v0, np.zeros_like(present), g_src, g0, v_L, m=self.m, **self.kw)",
     [D + "test_e2e_label_with_a_stopped_lead_reduces_progress_push_increases_it_plumb_and_a_far_lead_change_nothing"], TD),
    ("w5_head_presence_threshold_0_5_to_0_0", WR, "present = (p >= HEAD_P_MIN) & ok", "present = (p >= 0.0) & ok",
     [D + "test_r6_head_source_equals_lead_dump_on_the_same_frames_and_r4_no_token_after_the_tick"], TD),
    ("w6_label_gap_upper_bound_removed", WR, "return bool(self.valid[r] == 1 and np.isfinite(g) and 0.0 <= g <= GAP_MAX_M)", "return bool(self.valid[r] == 1 and np.isfinite(g) and 0.0 <= g)",
     [D + "test_label_source_equals_the_training_label_arrays_row_for_row"], TD),
    ("w7_closing_run_step_break_removed", WR, "abs(self.gap[f + 1] - self.gap[f]) <= STEP_MAX_M:", "abs(self.gap[f + 1] - self.gap[f]) <= 1e9:",
     [D + "test_label_source_equals_the_training_label_arrays_row_for_row"], TD),
    ("w8_source_bad_ticks_stay_present", WR, "present = present & ~src_bad", "present = present",
     [D + "test_e2e_inactive_wrapper_equals_the_unwrapped_rollout_exactly_for_every_mode"], TD),
    ("w9_rule_flag_accepted_for_a_non_run_mode", WR, 'if mine.dist_rule != "off":', "if False:",
     [D + "test_cli_refusals"], TD),
    ("q1_map_above_last_knot_not_shifted", QM, "q[hi] = gg[hi] - (xs[-1] - ys[-1])", "q[hi] = gg[hi]",
     [Q + "test_qmap_below_between_above_and_min_with_the_head_gap"], TM),
    ("q2_map_not_capped_at_the_head_gap", QM, "out[fin] = np.minimum(gg, q)", "out[fin] = q",
     [Q + "test_qmap_below_between_above_and_min_with_the_head_gap"], TM),
    ("q3_map_below_first_knot_constant", QM, "q[lo] = ys[0] * gg[lo] / xs[0]", "q[lo] = ys[0]",
     [Q + "test_qmap_below_between_above_and_min_with_the_head_gap"], TM),
    ("q4_fit_level_20_to_50", QF, "Q_LEVEL = 20.0", "Q_LEVEL = 50.0",
     [Q + "test_fit_arm_gives_the_20th_percentile_map_monotone_below_the_diagonal"], TM),
    ("q5_fit_pav_removed", QF, "ys = np.minimum(pav(q, n), xs)", "ys = np.minimum(q, xs)",
     [Q + "test_fit_arm_gives_the_20th_percentile_map_monotone_below_the_diagonal"], TM),
    ("q6_fit_eval_file_refusal_removed", QF, 'if "_eval" in os.path.basename(p):', "if False:",
     [Q + "test_fit_arm_refuses_an_evaluation_file"], TM),
    ("q7_calib_round_up_becomes_round_down", MC, "return float(math.ceil(max(0.0, float(x)) * 2.0 - 1e-9) / 2.0)", "return float(math.floor(max(0.0, float(x)) * 2.0 - 1e-9) / 2.0)",
     [Q + "test_up_half_rounds_up_to_half_metres_and_not_below_zero"], TM),
    ("q8_calib_percentile_95_to_90", MC, "(50, 80, 95))", "(50, 80, 90))",
     [Q + "test_calib_arm_over_read_percentiles_and_recall_on_the_near_rows"], TM),
    ("q9_calib_near_range_40_to_400", MC, "NEAR_M = 40.0", "NEAR_M = 400.0",
     [Q + "test_calib_arm_over_read_percentiles_and_recall_on_the_near_rows"], TM),
    ("q10_margin_added_instead_of_subtracted", MG, "np.maximum(g - m, 0.0)", "np.maximum(g + m, 0.0)",
     [Q + "test_margin_lowers_only_finite_head_gaps_never_below_zero_and_zero_margin_is_identity"], TM),
    ("q11_margin_range_refusal_removed", MG, "if not (0.0 <= mine.gap_margin_m <= 30.0):", "if False:",
     [Q + "test_margin_main_refusals_and_it_installs_the_constants_and_the_head_read"], TM),
    ("q12_margin_head_read_not_installed", MG, "W.head_values = head_values_margin ", "W.head_values = W.head_values ",
     [Q + "test_margin_main_refusals_and_it_installs_the_constants_and_the_head_read"], TM),
    ("q13_qcal_map_not_applied", QC, "return p, QM.apply_map(g, MAP[\"xs\"], MAP[\"ys\"]).astype(g.dtype), c, ok", "return p, g, c, ok",
     [Q + "test_qcal_wrapper_applies_the_map_to_finite_head_gaps_only"], TM),
    ("q14_qcal_head_read_not_installed", QC, "W.head_values = head_values_qcal ", "W.head_values = W.head_values ",
     [Q + "test_qcal_main_refuses_bad_p_min_and_a_map_that_fails_and_installs_the_map_and_the_head_read"], TM),
]


def main():
    mut = []
    for mid, f, old, new, exp, tf in M:
        src = open(os.path.join(SRC, f), encoding="utf-8", newline="").read()
        n = src.count(old)
        print(mid, "old occurs", n)
        if n != 1:
            sys.exit("old text must occur once: " + mid)
        mut.append({"id": mid, "file": "src/" + f, "old": old, "new": new, "expect": exp, "test_files": tf})
    spec = {"pristine": "/workspace/v8proof/mutlists/distrule/tree_pristine", "work": "/workspace/v8proof/mutlists/distrule/tree_work",
            "env": {"DISTRULE_TREE": "/workspace/v8proof/tree_leadsup", "DISTRULE_SRC": "/workspace/v8proof/mutlists/distrule/tree_work/src"},
            "test_files": TD + TM,
            "ignore_failed": ["test_dist_rule.py::test_geometry_copy_equals_headway_read", "test_dist_rule.py::test_pod_script_refusals_layout_and_one_rollout"],
            "ignore_reason": "environment-bound on the pod: the first imports headway_read (a dev-box file), the second drives the pod script with a fake python and fake cgroup files",
            "mutations": mut}
    json.dump(spec, open(os.path.join(HERE, "spec_distrule.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
