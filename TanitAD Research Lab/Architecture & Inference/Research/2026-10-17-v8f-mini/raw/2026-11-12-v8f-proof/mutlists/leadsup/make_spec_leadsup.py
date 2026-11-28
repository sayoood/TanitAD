"""make_spec_leadsup.py -- the mutation definitions of LEADSUP and the check that every old text occurs exactly once in the patched file.
  python make_spec_leadsup.py   -> writes spec_leadsup.json (pod paths) next to it"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHED = "F:/TanitAD-scratch/v8proof/spatial/leadsup/patched"
MI = "stack/tanitad/models/v8f_mini.py"
TR = "stack/scripts/v8f_mini_train.py"
DA = "stack/tanitad/data/v8f_mini_data.py"
T = "test_v8f_lead_sup.py::"

M = [
    ("m1_term_not_added_to_total", MI, 'total = total + cfg.w_lead_sup * T["lead_sup"] if cfg.lead_sup == "on" else total', 'total = total + 0.0 * T["lead_sup"] if cfg.lead_sup == "on" else total',
     [T + "test_the_total_gains_exactly_w_times_the_term_and_the_line_is_load_bearing"]),
    ("m2_window_mask_any_instead_of_all", MI, 'okw = batch["frame_ok"].unfold(1, hist, 1).all(-1)', 'okw = batch["frame_ok"].unfold(1, hist, 1).any(-1)',
     [T + "test_frame_alignment_labels_of_non_end_frames_change_nothing_end_frames_do"]),
    ("m3_label_frames_shifted_by_one", MI, "sl = slice(hist - 1, hist - 1 + fe)", "sl = slice(hist, hist + fe)",
     [T + "test_frame_alignment_labels_of_non_end_frames_change_nothing_end_frames_do"]),
    ("m4_gap_mask_ignores_missing_gap_labels", MI, "m_g = (okw & torch.isfinite(gap)).float()", "m_g = okw.float()",
     [T + "test_all_labels_masked_reads_exactly_zero_term_parts_and_gradients"]),
    ("m5_gap_part_divided_by_presence_count", MI, "t_g = (hg * m_g).sum() / m_g.sum().clamp_min(1.0)", "t_g = (hg * m_g).sum() / m_p.sum().clamp_min(1.0)",
     [T + "test_all_labels_masked_reads_exactly_zero_term_parts_and_gradients"]),
    ("m6_gap_output_scale_changed", MI, "LEAD_SUP_GAP_SCALE_M = 50.0", "LEAD_SUP_GAP_SCALE_M = 500.0",
     [T + "test_checkpoint_cfg_round_trip_loads_in_the_eval_builder_and_the_read_hook_runs"]),
    ("m7_head_always_reads_detached_state", MI, 'res["lead_out"] = self.lead_head(z1.detach() if self.lead_detach else z1)', 'res["lead_out"] = self.lead_head(z1.detach())',
     [T + "test_reach_term_alone_head_and_state_encoder_nonzero_everything_else_exactly_zero"]),
    ("m8_detach_flag_ignored", MI, 'self.lead_detach = cfg.lead_sup_detach == "on"', "self.lead_detach = False",
     [T + "test_detach_on_the_term_reaches_nothing_outside_the_head_exactly"]),
    ("m9_config_accepts_on_with_zero_weight", MI, "if not self.w_lead_sup > 0.0:", "if False:",
     [T + "test_flags_have_entries_defaults_and_the_config_refuses_what_it_cannot_build"]),
    ("m10_built_weight_lies", MI, 'lv["w_lead_sup"] = float(self.cfg.w_lead_sup) if lh is not None else 0.0', 'lv["w_lead_sup"] = 2.0 * float(self.cfg.w_lead_sup) if lh is not None else 0.0',
     [T + "test_declared_vs_built_goes_red_on_every_lie"]),
    ("m11_weight_flag_default_on", TR, 'ap.add_argument("--w-lead-sup", dest="w_lead_sup", type=float, default=0.0,', 'ap.add_argument("--w-lead-sup", dest="w_lead_sup", type=float, default=1.0,',
     [T + "test_flags_have_entries_defaults_and_the_config_refuses_what_it_cannot_build"]),
    ("m12_gate_accepts_on_with_zero_weight", TR, "    if not a.w_lead_sup > 0.0:", "    if False:",
     [T + "test_declared_vs_built_goes_red_on_every_lie"]),
    ("m13_gap_label_range_widened", DA, "LEAD_SUP_GAP_MAX_M = 60.0", "LEAD_SUP_GAP_MAX_M = 600.0",
     [T + "test_the_label_file_is_not_read_from_the_lead_head_arrays_and_gap_range_is_0_to_60"]),
    ("m14_closing_run_break_removed", DA, "LEAD_SUP_STEP_MAX_M = 3.0 ", "LEAD_SUP_STEP_MAX_M = 3000.0 ",
     [T + "test_planted_series_give_the_exact_slope_and_the_run_breaks_where_it_must"]),
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
    spec = {"pristine": "/workspace/v8proof/mutlists/leadsup/tree_pristine", "work": "/workspace/v8proof/mutlists/leadsup/tree_work",
            "test_files": ["stack/tests/test_v8f_lead_sup.py"], "mutations": mut}
    json.dump(spec, open(os.path.join(HERE, "spec_leadsup.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
