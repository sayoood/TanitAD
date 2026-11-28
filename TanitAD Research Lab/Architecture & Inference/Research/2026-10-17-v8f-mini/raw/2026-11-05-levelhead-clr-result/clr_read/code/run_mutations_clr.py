#!/usr/bin/env python3
"""run_mutations_clr.py -- mutation arms of the CLR reader, the chain gates, the pre-check and the wiring rule: each mutation edits ONE statement of a COPY of code/clr_read.py, clr_precheck.py or clr_wiring.py (every
replacement must match exactly once) and the named tests must go RED; the unmutated copy must be GREEN.  A rule that no test notices is a rule nobody checks.
usage: python -I code/run_mutations_clr.py   ->  logs/mutations_clr.log (last line MUTATIONS ALL RED)"""
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MUT = "F:/TanitAD-scratch/motion_c/clr_read/mut"
BT = "F:/TanitAD-scratch/motion_c/clr_read/pt/basetemp_mut"

READER = [      # (name, old, new, -k)   source code/clr_read.py, tests/test_clr_read.py, env CLR_READ_PATH
    ("m01_P_mean_form_instead_of_per_seed", '"PASS": bool(len(per) == len(SEEDS) and all(p["holds"] for p in per.values()))', '"PASS": bool(rr.primary_rule(cells)["PASS"])', "test_P_is_per_seed_not_the_mean"),
    ("m02_P_bar_exclusive", "sh >= SHARE_BAR", "sh > SHARE_BAR", "test_P_edges"),
    ("m03_leak_without_positive_logged_share", "sl is not None and sl > 0 and (float(sl)", "sl is not None and (float(sl)", "test_leak_borne_rule"),
    ("m04_leak_not_strict", "(float(sl) - srv) > LEAK_FRACTION * float(sl)", "(float(sl) - srv) >= LEAK_FRACTION * float(sl)", "test_leak_borne_rule"),
    ("m05_D_without_P_and_G", '"DOMINANT_COUNTED": bool(both and P and G)', '"DOMINANT_COUNTED": bool(both)', "test_D_needs_read_a_at_both_seeds_and_P_and_G"),
    ("m06_D_three_family_form_decides", 'all(v["DOMINANT"] for v in ra_by_seed.values()))\n    both3', 'all(v["DOMINANT_three_families_strategic_paused"] for v in ra_by_seed.values()))\n    both3', "test_D3_includes_the_strategic_family"),
    ("m07_F_logged_effect_decides", "F = rule_F(eff_ri, rep, eff_lg)", "F = rule_F(eff_lg, rep, eff_ri)", "test_floor_beside_is_the_replicate_pair"),
    ("m08_leak_does_not_override", "    if leak:\n        pf =", "    if False:\n        pf =", "test_leak_borne_overrides_the_label"),
    ("m09_attribution_needs_both_seeds", "elif n_attr_separated_better >= 1:\n        branch", "elif n_attr_separated_better >= 2:\n        branch", "test_decision_table_a2_outcomes"),
    ("m10_guards_always_pass", "g = rr.guards_rule(xy_by_seed)", 'g = {"PASS": True, "per_seed": {}}', "test_G_ade6_edge_and_noncrawl_rows"),
    ("m11_share_against_the_wrong_excess", 'cells_ri[k] = rr.m1_cell(b["RICLR_minus_RIC0"][M1], b["RIC0_minus_P10"][M1])', 'cells_ri[k] = rr.m1_cell(b["RICLR_minus_RIC0"][M1], b["RICLR_minus_P10"][M1])', "test_designed_arm_closes_exactly_alpha"),
    ("m12_logged_contrast_against_the_replicate", 'rows("CLR", "C0", M1)', 'rows("CLR", "C0REP", M1)', "test_reader_known_answers"),
    ("m13_no_refusal_on_failed_controls", "        if bad:\n            res[\"refused\"]", "        if False:\n            res[\"refused\"]", "test_production_without_cl_rollouts_refuses"),
    ("m14_strategic_family_dropped_from_the_table", 'FOUR_FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")', 'FOUR_FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL")', "test_family_table_counts_the_four_families"),
    ("m15_card_rules_pin_unchecked", 'ctl["card_rules"] = {"PASS": got == CARD_RULES_SHA,', 'ctl["card_rules"] = {"PASS": True,', "test_card_rules_control_fails_on_a_changed_card"),
    ("m16_attribution_against_the_c0_base", 'attr_ri[k] = rr.m1_cell(b["RICLR_minus_RIRAPRP01"][M1], b["RIRAPRP01_minus_P10"][M1])', 'attr_ri[k] = rr.m1_cell(b["RICLR_minus_RIC0"][M1], b["RIRAPRP01_minus_P10"][M1])',
     "test_branch_P_fails_attribution_not_separated_better_when_the_base_moved_as_much"),
    ("m17_logged_attribution_decides", 'A["n_seeds_separated_better_reindexed"], leak["LEAK_BORNE"])', 'A["n_seeds_separated_better_logged_beside"], leak["LEAK_BORNE"])', "test_the_reindexed_attribution_decides_not_the_logged_one"),
    ("m18_attribution_base_rolled_from_clr", 'RUN_OF["RIRAPRP01"] = "RAPRP01"', 'RUN_OF["RIRAPRP01"] = "CLR"', "test_expected_flags_are_the_cards_scene_modes"),
    ("m19_a2_card_unchecked", 'ctl["card_a2"] = {"PASS": bool(a2 == A2_CARD_SHA == fz),', 'ctl["card_a2"] = {"PASS": True,', "test_a2_card_control_fails_on_a_changed_card"),
    ("m20_uncommitted_cell_collapsed_into_the_committed_one", '        branch = "P-FAILS-ATTR-SEPARATED-BETTER"\n    else:', '        branch = "P-FAILS-ATTR-NOT-SEPARATED-BETTER"\n    else:', "test_decision_table_a2_outcomes"),
]
PRECHECK = [    # source code/clr_precheck.py, tests/test_clr_chain_gates.py, env CLR_PRECHECK_PATH
    ("p01_sha_vs_SHA256SUMS_unchecked", "if not sums or got != sums[0]:", "if False:", "test_precheck_refuses_sha_mismatch_with_sums_or_marker"),
    ("p02_sha_vs_box2_marker_unchecked", "elif ev != got:", "elif False:", "test_precheck_refuses_sha_mismatch_with_sums_or_marker"),
    ("p03_argv_change_unchecked", '"nothing_changed": not changed,', '"nothing_changed": True,', "test_precheck_argv_audit_is_exact"),
    ("p04_W_in_argv_unchecked", '"w_level_is_the_frozen_W": bool(added.get("--w-level") is not None and _same_num(added.get("--w-level"), w)),', '"w_level_is_the_frozen_W": True,', "test_precheck_argv_audit_is_exact"),
    ("p05_cfg_level_head_unchecked", 'if cfg.get("level_head") != "on":', "if False:", "test_precheck_refuses_a_checkpoint_whose_cfg_has_no_level_head"),
    ("p06_absent_ckpt_not_distinguished", 'return 0 if res["PASS"] else (5 if absent else 8)', 'return 0 if res["PASS"] else 8', "test_precheck_refuses_an_absent_checkpoint"),
    ("p07_recovery_lever_unchecked", 'c["recovery_lever_is_RAPRP01s"] = bool(rec(cl) == rec(c0) and p.replace(".", "", 1).isdigit() and float(p) > 0)', 'c["recovery_lever_is_RAPRP01s"] = True', "test_precheck_refuses_an_argv_without_the_recovery_lever"),
    ("p08_marker_recover_n_unchecked", "if not (isinstance(rn, (int, float)) and not isinstance(rn, bool) and rn > 0):", "if False:", "test_precheck_refuses_a_marker_without_recovery_evidence"),
    ("p09_attribution_base_sha_unchecked", "if not sb or gb != sb[0]:", "if False:", "test_precheck_checks_the_attribution_base_checkpoint"),
    ("p10_attribution_base_level_head_unchecked", 'if bk["cfg"].get("level_head") == "on":', "if False:", "test_precheck_checks_the_attribution_base_checkpoint"),
]
CONTROLS = [    # source code/clr_read.py, tests/test_clr_chain_gates.py, env CLR_READ_PATH
    ("c01_one_ckpt_per_seed_unchecked", 'ctl["one_checkpoint_per_seed"] = {"PASS": bool(all(len(v) == 1 for v in ckpts.values()) and len(ckpts) == 2 * len(SEEDS)),', 'ctl["one_checkpoint_per_seed"] = {"PASS": True,', "refuse_each_inconsistency and ckpt_two_shas"),
    ("c02_flags_unchecked", '"flags_as_expected": d["flags"].startswith(expected_flags(role))', '"flags_as_expected": True', "refuse_each_inconsistency and flags"),
    ("c03_header_scene_unchecked", '"header_scene": h["scene"] == NEW_ROLES[role][0]', '"header_scene": True', "refuse_each_inconsistency and scene_header"),
    ("c04_precheck_ckpt_vs_rollouts_unchecked", 'ckpts.get((s, "CLR"), set()) and pc["seeds"][f"s{s}"].get("raprp01_ckpt_sha256") in ckpts.get((s, "RAPRP01"), set()) for s in SEEDS)', 'ckpts.get((s, "CLR"), set()) and True for s in SEEDS)', "refuse_each_inconsistency and attribution_ckpt_vs_precheck"),
    ("c05_expected_run_dir_unchecked", '"ckpt_is_the_expected_run": f"/M2b_s{s}_recft_{RUN_OF[role]}/ckpt_last.pt" in d["ckpt"].replace("\\\\", "/"),', '"ckpt_is_the_expected_run": True,', "refuse_each_inconsistency and (cl_not_clr_run or attribution_base_ckpt or wrong_run_dir)"),
]
WIRING = [      # source code/clr_wiring.py, tests/test_clr_wiring.py, env CLR_WIRING_PATH
    ("w01_sensitive_edge_exclusive", "return SENSITIVE if x >= SENS_HI else", "return SENSITIVE if x > SENS_HI else", "test_thresholds_are_A1s"),
    ("w02_insensitive_edge_inclusive", "(INSENSITIVE if x < SENS_LO else WEAK)", "(INSENSITIVE if x <= SENS_LO else WEAK)", "test_thresholds_are_A1s"),
    ("w03_one_sensitive_seed_is_enough", "if len(labels) == 2 and vals == {SENSITIVE}:", "if SENSITIVE in vals:", "test_a_mixed_or_weak_reading"),
    ("w04_one_insensitive_seed_is_enough", "if len(labels) == 2 and vals == {INSENSITIVE}:", "if INSENSITIVE in vals:", "test_a_mixed_or_weak_reading"),
    ("w05_readout_controls_unchecked", 'ctl[k] = bool(per[k]["sensitivity_label_A1"] == per[k]["sensitivity_label_recomputed"] and not v["control_failures"])', "ctl[k] = True", "test_main_refuses_when_the_readouts_own_controls_failed"),
]
GROUPS = [("reader", READER, "code/clr_read.py", "test_clr_read.py", "CLR_READ_PATH"), ("precheck", PRECHECK, "code/clr_precheck.py", "test_clr_chain_gates.py", "CLR_PRECHECK_PATH"),
          ("controls", CONTROLS, "code/clr_read.py", "test_clr_chain_gates.py", "CLR_READ_PATH"), ("wiring", WIRING, "code/clr_wiring.py", "test_clr_wiring.py", "CLR_WIRING_PATH")]


def run(path, k, test, var):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")
    env[var] = path
    r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-x", f"{PK}/tests/{test}", "-k", k, "--basetemp", BT, "-p", "no:cacheprovider"], env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def main():
    os.makedirs(MUT, exist_ok=True)
    os.makedirs(os.path.dirname(BT), exist_ok=True)
    lines, ok, n = [], True, 0
    for gname, muts, srcrel, test, var in GROUPS:
        src_path = f"{PK}/{srcrel}"
        src = open(src_path, encoding="utf-8", newline="").read()
        rc, last = run(src_path, " or ".join(sorted({m[3] for m in muts})), test, var)
        lines.append(f"CLEAN ({gname})   rc={rc}  {last}")
        ok &= rc == 0
        for name, old, new, k in muts:
            c = src.count(old)
            if c != 1:
                lines.append(f"{name}: REPLACEMENT MATCHES {c} TIMES (must be 1)")
                ok = False
                continue
            p = f"{MUT}/{name}.py"
            open(p, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
            rc, last = run(p, k, test, var)
            ok &= rc != 0
            n += 1
            lines.append(f"{name}: {'RED' if rc != 0 else 'GREEN (NOT CAUGHT)'}  rc={rc}  [{k}]  {last}")
    lines.append(f"{n} mutations")
    lines.append("MUTATIONS ALL RED" if ok else "MUTATIONS NOT ALL RED")
    text = "\n".join(lines) + "\n"
    open(f"{PK}/logs/mutations_clr.log", "w", encoding="utf-8", newline="\n").write(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
