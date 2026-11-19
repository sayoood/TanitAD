"""make_spec_q.py -- the counted mutation list of source cand-cvq (note PREREG_V8F_PROOF_ADD40_N7 section 5) as a spec of mutlists/common/pod_mut_driver.py. Writes spec_qplan.json (pod paths).
Each old text must occur exactly once in the wrapper (checked here and again by the driver)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
F = "v8f_t1_eval_imagplan_q.py"
TF = ["src/test_imag_plan_q.py"]
Z = "test_imag_plan_q.py::"
T1 = Z + "test_the_min_of_a_hand_computed_case_and_the_gate_stays_on_the_head_gap"
T2 = Z + "test_a_quantile_constant_of_1e9_is_the_identity_of_cand_cv_and_a_constant_of_0_makes_the_gap_zero"
T3 = Z + "test_the_head_is_read_at_the_tick_frame_and_scaled_to_metres"
T4 = Z + "test_refusals_without_a_head_file_for_another_source_for_a_head_that_is_not_the_banked_or_not_in_band"
T5 = Z + "test_other_sources_do_not_see_the_head"
M = [
    ("m1_min_dropped_the_quantile_alone_is_the_gap", "g_h = QM.used_gap(g_h, q_h)", "g_h = q_h", [T1]),
    ("m2_max_instead_of_min", "g_h = QM.used_gap(g_h, q_h)", "g_h = np.maximum(g_h, q_h)", [T1]),
    ("m3_quantile_not_applied_the_head_gap_stays", "g_h = QM.used_gap(g_h, q_h)", "g_h = g_h", [T1, T2]),
    ("m4_head_applied_to_another_source_cand_cv", 'self.q_on = source == "cand-cvq"', 'self.q_on = source in ("cand-cvq", "cand-cv")', [T5]),
    ("m5_head_read_at_another_frame_window_first_frames_of_the_batch", "f = feats[:, -hist:]", "f = feats[:, :hist]", [T3]),
    ("m6_gate_moved_onto_the_used_gap", "g_gate, qh_c = gh_c, np.asarray(q_h, np.float64)", "g_gate, qh_c = None, np.asarray(q_h, np.float64)", [T1]),
    ("m7_constant_ignored_the_head_output_always_used", "if self.q_const is not None:", "if False:", [T2]),
    ("m8_refusal_of_cand_cvq_without_a_head_removed", "if self.q_on and qhead is None:", "if False:", [T4]),
    ("m9_head_given_to_another_source_not_refused", "if (not self.q_on) and (qhead is not None or q_const is not None):", "if False:", [T4]),
    ("m10_env_for_another_mode_not_refused", '        if f or c != "":', "        if False:", [T4]),
    ("m11_coverage_band_check_dropped", 'if rec.get("h5_check_coverage_in_0.15_0.25") is not True:', "if False:", [T4]),
    ("m12_head_sha256_check_dropped", 'if sha_head not in (rec.get("head_file_sha256") or {}).values():', "if False:", [T4]),
    ("m13_checkpoint_sha256_check_dropped", "if ckpt and os.path.isfile(ckpt) and QM.sha256_file(ckpt) != ck_rec:", "if False:", [T4]),
    ("m14_evaluation_split_path_not_refused", "if QM.is_eval_path(f):", "if False:", [T4]),
    ("m15_output_not_scaled_to_metres", "return (o * QM.GAP_SCALE_M).astype(np.float32)", "return (o * 1.0).astype(np.float32)", [T3]),
    ("m16_log_g_used_is_the_head_gap", "g_head=gh_c, q_head=qh_c, g_used=g_src,", "g_head=gh_c, q_head=qh_c, g_used=gh_c,", [T1]),
]


def main():
    src = open(os.path.join(HERE, F), encoding="utf-8", newline="").read()
    mut = []
    for mid, old, new, exp in M:
        n = src.count(old)
        print(mid, "old occurs", n)
        if n != 1:
            sys.exit("old text must occur once: " + mid)
        mut.append({"id": mid, "file": "src/" + F, "old": old, "new": new, "expect": exp, "test_files": TF})
    root = "/workspace/v8proof/imagplan_q/mut"
    spec = {"pristine": root + "/tree_pristine", "work": root + "/tree_work", "tmp_root": root + "/tmp",
            "env": {"DISTRULE_TREE": "/workspace/v8proof/tree_leadsup"}, "test_files": TF, "mutations": mut}
    json.dump(spec, open(os.path.join(HERE, "spec_qplan.json"), "w"), indent=1)
    print("wrote spec_qplan.json", len(mut))


if __name__ == "__main__":
    main()
