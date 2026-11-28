"""run_mutations.py -- deliberate one-line mutations of imag_plan.py; each must turn a NAMED test of test_imag_plan.py red.

For every mutation: copy imag_plan.py to work/mut/<name>/imag_plan.py with exactly one line changed, run the named test(s) with IMAGPLAN_SRC_DIR pointing there, require a non-zero
exit AND the named test among the failures.  A control run on the unmutated file must be green (else the red proves nothing).  Result: work/mut/mutations.json.
Run:  python run_mutations.py      (CPU; about 1 minute; one pytest process at a time)
"""
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
MUT = os.path.join(HERE, "work", "mut")
SRC = open(os.path.join(HERE, "imag_plan.py"), encoding="utf-8").read()

MUTATIONS = [
    ("own_progress_sign", "gap = (g_src + a_off)[:, None, None]", "gap = (g_src - a_off)[:, None, None]", ["test_cand_cv_gap_formula_equals_the_hand_value"]),
    ("horizon_index_off_by_one", "KNOT_H = (5, 10) ", "KNOT_H = (4, 9) ", ["test_hand_boundary_exactly_one_candidate_boundary_is_safe"]),
    ("selection_picks_smallest_safe_c", "masked = np.where(safe, cs[None, :], -np.inf)\n    idx = np.argmax(masked, axis=1)", "masked = np.where(safe, cs[None, :], np.inf)\n    idx = np.argmin(masked, axis=1)",
     ["test_selection_largest_safe_c_and_zero_when_none_is_safe_and_set_order_does_not_matter"]),
    ("braking_floor_becomes_a_min", "a = np.maximum(c * aP[:, None, :], aB[:, None, :])", "a = np.minimum(c * aP[:, None, :], aB[:, None, :])", ["test_braking_floor_c0_equals_a_B"]),
    ("lead_travel_sign", "+ v_L[:, None, None] * h - a_h", "- v_L[:, None, None] * h - a_h", ["test_cand_cv_gap_formula_equals_the_hand_value"]),
    ("executed_action_frame_shifted_by_one", "out[1:, 0] = ch[1:m, 1]", "out[1:, 0] = ch[0:m - 1, 1]", ["test_actions_of_a_candidate_equal_what_the_pinned_commit_writes_bit_for_bit"]),
    ("logged_action_forward_becomes_backward", "out[:, 0] = (v[1:m + 1] - v[0:m]) / DT", "out[1:, 0] = (v[1:m] - v[0:m - 1]) / DT", ["test_actions_of_a_candidate_equal_what_the_pinned_commit_writes_bit_for_bit"]),
    ("imag_shift_sign", "gap = imag_raw + float(shift_m) + a_off[:, None, None]", "gap = imag_raw - float(shift_m) + a_off[:, None, None]", ["test_option_imag_shift_reaches_every_imagined_gap_and_cset_and_fallback_reach_the_choice"]),
    ("fallback_zero_never_applied", "c = np.where(anys, cs[np.maximum(idx, 0)], 0.0 if fallback == \"zero\" else np.nan)", "c = np.where(anys, cs[np.maximum(idx, 0)], np.nan)",
     ["test_selection_largest_safe_c_and_zero_when_none_is_safe_and_set_order_does_not_matter"]),
    ("yaw_limit_never_applied", "    if yaw_limit:", "    if False:", ["test_yaw_limit_bounds_a_heading_flip_and_leaves_a_physical_yaw_rate_untouched"]),
    # ---- candidate family decel (note ADD40_N3)
    ("decel_sign_of_d_in_v_d", "vP[:, None, :] - ds * DT * i)", "vP[:, None, :] + ds * DT * i)", ["test_decel_hand_values_constant_plan_speed_10_and_d_2"]),
    ("decel_max0_removed", "np.maximum(0.0, vP[:, None, :] - ds * DT * i)", "(vP[:, None, :] - ds * DT * i)",
     ["test_decel_monotone_in_d_speeds_never_negative_and_a_stopped_candidate_stays_stopped"]),
    ("decel_selection_picks_largest_safe_d", "idx = np.argmin(np.where(safe, ds[None, :], np.inf), axis=1)", "idx = np.argmax(np.where(safe, ds[None, :], -np.inf), axis=1)",
     ["test_decel_selection_smallest_safe_d_largest_d_when_none_is_safe_and_fallback_none_keeps_the_plan"]),
    ("decel_fallback_picks_smallest_d", "ds.max() if fallback == \"zero\"", "ds.min() if fallback == \"zero\"",
     ["test_decel_selection_smallest_safe_d_largest_d_when_none_is_safe_and_fallback_none_keeps_the_plan"]),
    ("decel_fallback_executes_smallest_d", "zero_i = int(np.argmax(cs))", "zero_i = int(np.argmin(cs))",
     ["test_decel_selection_smallest_safe_d_largest_d_when_none_is_safe_and_fallback_none_keeps_the_plan"]),
    ("decel_d0_not_the_plan_bit_for_bit", "    return np.where((np.asarray(dset, np.float64) == 0.0)[None, :, None], aP[:, None, :], a)", "    return a",
     ["test_decel_zero_is_the_plan_bit_for_bit_progress_knots_and_object"]),
    ("decel_actions_yaw_channel_overwritten", "        out[:, 0] = np.nan_to_num((vd[1:m + 1] - vd[0:m]) / DT).astype(np.float32)", "        out[:, 1] = np.nan_to_num((vd[1:m + 1] - vd[0:m]) / DT).astype(np.float32)",
     ["test_decel_actions_yaw_is_the_plan_acceleration_is_the_plan_minus_d_and_d0_is_bit_equal"]),
    ("decel_actions_step_index_shifted", "v[1:] - float(d) * DT * np.arange(1, m + 1))", "v[1:] - float(d) * DT * np.arange(0, m))",
     ["test_decel_actions_yaw_is_the_plan_acceleration_is_the_plan_minus_d_and_d0_is_bit_equal"]),
    ("decel_progress_uses_the_scale_family", "a_c = decel_progress(aP, cs) if family == \"decel\" else", "a_c = decel_progress(aP, cs) if False else",
     ["test_decel_hand_values_constant_plan_speed_10_and_d_2"]),
    ('imagkin_known_difference_dropped', 'gap = imag_raw[:, [i0], :] + float(shift_m) + a_off[:, None, None] + (a_h[:, [i0], :] - a_h)', 'gap = imag_raw[:, [i0], :] + float(shift_m) + a_off[:, None, None] + 0.0 * (a_h[:, [i0], :] - a_h)',
     ['test_imagkin_gap_is_the_imagined_level_of_the_plan_plus_the_known_differences', 'test_imagkin_with_a_planted_level_equals_cand_cv_and_refusals']),
    ('imagkin_known_difference_sign', 'gap = imag_raw[:, [i0], :] + float(shift_m) + a_off[:, None, None] + (a_h[:, [i0], :] - a_h)', 'gap = imag_raw[:, [i0], :] + float(shift_m) + a_off[:, None, None] - (a_h[:, [i0], :] - a_h)',
     ['test_imagkin_gap_is_the_imagined_level_of_the_plan_plus_the_known_differences', 'test_imagkin_with_a_planted_level_equals_cand_cv_and_refusals']),
    ('imagkin_reads_every_imagined_column', 'gap = imag_raw[:, [i0], :] + float(shift_m) + a_off[:, None, None] + (a_h[:, [i0], :] - a_h)', 'gap = imag_raw + float(shift_m) + a_off[:, None, None] + (a_h[:, [i0], :] - a_h)',
     ['test_imagkin_gap_is_the_imagined_level_of_the_plan_plus_the_known_differences', 'test_imagkin_with_a_planted_level_equals_cand_cv_and_refusals']),
    ('imagkin_shift_dropped', 'gap = imag_raw[:, [i0], :] + float(shift_m) + a_off[:, None, None] + (a_h[:, [i0], :] - a_h)', 'gap = imag_raw[:, [i0], :] + 0.0 * float(shift_m) + a_off[:, None, None] + (a_h[:, [i0], :] - a_h)',
     ['test_imagkin_gap_is_the_imagined_level_of_the_plan_plus_the_known_differences']),
]


def run(src_dir, tests, base):
    env = dict(os.environ, IMAGPLAN_SRC_DIR=src_dir, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    cmd = [PY, "-m", "pytest", os.path.join(HERE, "test_imag_plan.py"), "-q", "-p", "no:cacheprovider", f"--basetemp={base}", "-k", " or ".join(tests)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    out = r.stdout + r.stderr
    failed = sorted({ln.split("::")[-1].split(" ")[0] for ln in out.splitlines() if ln.startswith("FAILED ")})
    return r.returncode, failed, out.strip().splitlines()[-1] if out.strip() else ""


def main():
    os.makedirs(os.path.join(MUT, "pt"), exist_ok=True)
    res = {}
    all_tests = sorted({t for m in MUTATIONS for t in m[3]})
    rc, failed, last = run(HERE, all_tests, os.path.join(MUT, "pt", "control"))
    res["control_unmutated"] = {"rc": rc, "failed": failed, "summary": last}
    ok = rc == 0
    for name, find, repl, tests in MUTATIONS:
        if SRC.count(find) != 1:
            res[name] = {"error": f"the mutated line occurs {SRC.count(find)} times, not once"}
            ok = False
            continue
        d = os.path.join(MUT, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "imag_plan.py"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(SRC.replace(find, repl))
        rc, failed, last = run(d, tests, os.path.join(MUT, "pt", name))
        red = rc != 0 and all(t in failed for t in tests)
        res[name] = {"line_before": find, "line_after": repl, "tests": tests, "rc": rc, "failed": failed, "summary": last, "went_red": bool(red)}
        ok = ok and red
    res["all_mutations_red_and_control_green"] = bool(ok)
    with open(os.path.join(MUT, "mutations.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res.items():
        print(k, v if not isinstance(v, dict) else {x: v[x] for x in ("rc", "went_red", "failed") if x in v})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
