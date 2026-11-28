"""make_spec_actcons.py -- the mutation definitions of ACTCONS and the check that every old text occurs exactly once in the patched file.
  python make_spec_actcons.py   -> writes spec_actcons.json (pod paths) next to it"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHED = "F:/TanitAD-scratch/v8proof/actcons/patched"
AC = "stack/tanitad/models/act_cons.py"
TR = "stack/scripts/v8f_mini_train.py"
T = "test_v8f_act_cons.py::"

M = [
    ("m1_drop_kinematic_term", AC, "resid = g_off - g_ref + 0.5 * dr[None, :] * (T ** 2)", "resid = g_off - g_ref",
     [T + "test_term_equals_a_hand_computation", T + "test_planted_kinematic_predictor_reads_zero_and_the_two_mutations_read_red"]),
    ("m2_kinematic_term_sign_flipped", AC, "resid = g_off - g_ref + 0.5 * dr[None, :] * (T ** 2)", "resid = g_off - g_ref - 0.5 * dr[None, :] * (T ** 2)",
     [T + "test_term_equals_a_hand_computation", T + "test_planted_kinematic_predictor_reads_zero_and_the_two_mutations_read_red"]),
    ("m3_horizon_not_squared", AC, "resid = g_off - g_ref + 0.5 * dr[None, :] * (T ** 2)", "resid = g_off - g_ref + 0.5 * dr[None, :] * T",
     [T + "test_term_equals_a_hand_computation"]),
    ("m4_reference_uses_offset_actions", AC, "ref, _, _ = st.roll_pairs(zr, st.act_embed(a), ROLL_K)", "ref, _, _ = st.roll_pairs(zr, st.act_embed(a + off), ROLL_K)",
     [T + "test_term_equals_a_hand_computation"]),
    ("m5_lead_head_params_not_detached", AC, "params = {n: p.detach() for n, p in head.named_parameters()}", "params = {n: p for n, p in head.named_parameters()}",
     [T + "test_gradient_reaches_predictor_and_action_embed_and_exactly_zero_elsewhere", T + "test_probe_passes_and_goes_red_without_the_detach"]),
    ("m6_start_state_adapt_not_detached", AC, "z = wm.z1(adapt.detach().float()).float()", "z = wm.z1(adapt.float()).float()",
     [T + "test_gradient_reaches_predictor_and_action_embed_and_exactly_zero_elsewhere"]),
    ("m7_speed_bound_removed_from_valid", AC, "& (sp >= float(max_off) * SPEED_MULT))", "& (sp >= 0.0))",
     [T + "test_invalid_samples_give_exactly_zero_term_and_gradient_and_do_not_move_the_valid_ones"]),
    ("m8_gap_range_upper_bound_widened", AC, "GAP_LO_M, GAP_HI_M = 0.0, 60.0", "GAP_LO_M, GAP_HI_M = 0.0, 6000.0",
     [T + "test_invalid_samples_give_exactly_zero_term_and_gradient_and_do_not_move_the_valid_ones"]),
    ("m9_offset_range_one_sided", AC, "* 2.0 - 1.0) * float(max_off)", "* 2.0) * float(max_off)",
     [T + "test_offsets_dedicated_generator_range_and_global_rng_untouched"]),
    ("m10_term_not_added_to_total", TR, 'T["total"] = tot_ac + a.w_act_cons * ac_r["term"]', 'T["total"] = tot_ac + 0.0 * ac_r["term"]',
     [T + "test_total_moves_by_exactly_w_times_the_term"]),
    ("m11_weight_default_on", TR, 'ap.add_argument("--w-act-cons", dest="w_act_cons", type=float, default=0.0,', 'ap.add_argument("--w-act-cons", dest="w_act_cons", type=float, default=1.0,',
     [T + "test_default_builds_nothing_and_keeps_the_key_sets"]),
    ("m12_gate_does_not_refuse_max_without_weight", TR, "        if a.act_cons_max != AC.ACT_CONS_MAX_DEFAULT:", "        if False:",
     [T + "test_default_off_is_a_lever_that_declares_nothing_when_max_is_changed", T + "test_gate_refusals"]),
]


def main():
    mut = []
    for mid, f, old, new, exp in M:
        src = open(os.path.join(PATCHED, f.replace("stack/", "", 1)), encoding="utf-8", newline="").read()
        n = src.count(old)
        print(mid, "old occurs", n)
        if n != 1:
            sys.exit("old text must occur once: " + mid)
        mut.append({"id": mid, "file": f, "old": old, "new": new, "expect": exp})
    spec = {"pristine": "/workspace/v8proof/mutlists/actcons/tree_pristine", "work": "/workspace/v8proof/mutlists/actcons/tree_work",
            "test_files": ["stack/tests/test_v8f_act_cons.py"], "mutations": mut}
    json.dump(spec, open(os.path.join(HERE, "spec_actcons.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
