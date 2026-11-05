#!/usr/bin/env python3
"""run_mutations_cl.py -- mutation arms of the CL reader AND of the chain gates: each mutation edits ONE statement of a COPY of code/cl_read.py or code/cl_precheck.py (every replacement must match exactly once) and the named tests must go RED;
the unmutated copy must be GREEN.  A rule that no test notices is a rule nobody checks.   usage: python -I code/run_mutations_cl.py   ->  logs/mutations_cl.log (last line MUTATIONS ALL RED)"""
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MUT = "F:/TanitAD-scratch/motion_c/cl_read/mut"
BT = "F:/TanitAD-scratch/motion_c/cl_read/pt/basetemp_mut"
SRC = f"{PK}/code/cl_read.py"
PRE = f"{PK}/code/cl_precheck.py"

# (name, old, new, tests-that-must-go-RED  (-k expression))
MUTATIONS = [
    ("m01_P_mean_form_instead_of_per_seed", '"PASS": bool(len(per) == len(SEEDS) and all(p["holds"] for p in per.values()))', '"PASS": bool(rr.primary_rule(cells)["PASS"])', "test_P_is_per_seed_not_the_mean"),
    ("m02_P_bar_exclusive", "sh >= SHARE_BAR", "sh > SHARE_BAR", "test_P_edges"),
    ("m03_leak_without_positive_logged_share", "sl is not None and sl > 0 and (float(sl)", "sl is not None and (float(sl)", "test_leak_borne_rule"),
    ("m04_leak_not_strict", "(float(sl) - srv) > LEAK_FRACTION * float(sl)", "(float(sl) - srv) >= LEAK_FRACTION * float(sl)", "test_leak_borne_rule"),
    ("m05_D_without_P_and_G", '"DOMINANT_COUNTED": bool(both and P and G)', '"DOMINANT_COUNTED": bool(both)', "test_D_needs_read_a_at_both_seeds_and_P_and_G"),
    ("m06_D_three_family_form_decides", 'all(v["DOMINANT"] for v in ra_by_seed.values()))\n    both3', 'all(v["DOMINANT_three_families_strategic_paused"] for v in ra_by_seed.values()))\n    both3', "test_D3_includes_the_strategic_family"),
    ("m07_F_logged_effect_decides", "F = rule_F(eff_ri, rep, eff_lg)", "F = rule_F(eff_lg, rep, eff_ri)", "test_floor_beside_is_the_replicate_pair"),
    ("m08_leak_does_not_override", "    if leak:\n        pf =", "    if False:\n        pf =", "test_leak_borne_overrides_the_label"),
    ("m09_M1_moved_needs_both_seeds", "elif n_m1_separated_better >= 1:\n        branch", "elif n_m1_separated_better >= 2:\n        branch", "test_decision_table_section_5"),
    ("m10_guards_always_pass", "g = rr.guards_rule(xy_by_seed)", 'g = {"PASS": True, "per_seed": {}}', "test_G_ade6_edge_and_noncrawl_rows"),
    ("m11_share_against_the_wrong_excess", 'cells_ri[k] = rr.m1_cell(b["RICL_minus_RIC0"][M1], b["RIC0_minus_P10"][M1])', 'cells_ri[k] = rr.m1_cell(b["RICL_minus_RIC0"][M1], b["RICL_minus_P10"][M1])', "test_designed_arm_closes_exactly_alpha"),
    ("m12_logged_contrast_against_the_replicate", 'rows("CL", "C0", M1)', 'rows("CL", "C0REP", M1)', "test_reader_known_answers"),
    ("m13_no_refusal_on_failed_controls", "        if bad:\n            res[\"refused\"]", "        if False:\n            res[\"refused\"]", "test_production_without_cl_rollouts_refuses"),
    ("m14_strategic_family_dropped_from_the_table", 'FOUR_FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")', 'FOUR_FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL")', "test_family_table_counts_the_four_families"),
    ("m15_card_rules_pin_unchecked", 'ctl["card_rules"] = {"PASS": got == CARD_RULES_SHA,', 'ctl["card_rules"] = {"PASS": True,', "test_card_rules_control_fails_on_a_changed_card"),
]


# (name, old, new, -k, test file, env var that aims the tests at the mutated copy, source file)
PRECHECK_MUTATIONS = [
    ("p01_sha_vs_SHA256SUMS_unchecked", "if not sums or got != sums[0]:", "if False:", "test_precheck_refuses_sha_mismatch_with_sums_or_marker"),
    ("p02_sha_vs_box2_marker_unchecked", "elif ev != got:", "elif False:", "test_precheck_refuses_sha_mismatch_with_sums_or_marker"),
    ("p03_argv_change_unchecked", '"nothing_changed": not changed,', '"nothing_changed": True,', "test_precheck_argv_audit_is_exact"),
    ("p04_W_in_argv_unchecked", '"w_level_is_the_frozen_W": bool(added.get("--w-level") is not None and _same_num(added.get("--w-level"), w)),', '"w_level_is_the_frozen_W": True,', "test_precheck_argv_audit_is_exact"),
    ("p05_cfg_level_head_unchecked", 'if cfg.get("level_head") != "on":', "if False:", "test_precheck_refuses_a_checkpoint_whose_cfg_has_no_level_head"),
    ("p06_absent_ckpt_not_distinguished", 'return 0 if res["PASS"] else (5 if absent else 8)', 'return 0 if res["PASS"] else 8', "test_precheck_refuses_an_absent_checkpoint"),
]
CONTROL_MUTATIONS = [
    ("c01_one_ckpt_per_seed_unchecked", 'ctl["one_checkpoint_per_seed"] = {"PASS": bool(all(len(v) == 1 for v in ckpts.values()) and len(ckpts) == len(SEEDS)),', 'ctl["one_checkpoint_per_seed"] = {"PASS": True,', "refuse_each_inconsistency and ckpt_two_shas"),
    ("c02_flags_unchecked", '"flags_as_expected": d["flags"].startswith(expected_flags(role))', '"flags_as_expected": True', "refuse_each_inconsistency and flags"),
    ("c03_header_scene_unchecked", '"header_scene": h["scene"] == NEW_ROLES[role][0]', '"header_scene": True', "refuse_each_inconsistency and scene_header"),
    ("c04_precheck_ckpt_vs_rollouts_unchecked", 'same = all(pc["seeds"][f"s{s}"]["ckpt_sha256"] in ckpts.get(s, set()) for s in SEEDS)', "same = True", "refuse_each_inconsistency and precheck_ckpt"),
]


def run(path, k, test="test_cl_read.py", var="CL_READ_PATH"):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")
    env[var] = path
    r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-x", f"{PK}/tests/{test}", "-k", k, "--basetemp", BT, "-p", "no:cacheprovider"], env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def main():
    os.makedirs(MUT, exist_ok=True)
    os.makedirs(os.path.dirname(BT), exist_ok=True)
    src = open(SRC, encoding="utf-8", newline="").read()
    lines, ok = [], True
    rc, last = run(SRC, " or ".join(sorted({m[3] for m in MUTATIONS})))
    lines.append(f"CLEAN   rc={rc}  {last}")
    ok &= rc == 0
    for name, old, new, k in MUTATIONS:
        n = src.count(old)
        if n != 1:
            lines.append(f"{name}: REPLACEMENT MATCHES {n} TIMES (must be 1)")
            ok = False
            continue
        p = f"{MUT}/{name}.py"
        open(p, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        rc, last = run(p, k)
        red = rc != 0
        ok &= red
        lines.append(f"{name}: {'RED' if red else 'GREEN (NOT CAUGHT)'}  rc={rc}  [{k}]  {last}")
    pre_src = open(PRE, encoding="utf-8", newline="").read()
    for muts, srcs, test, var in ((PRECHECK_MUTATIONS, pre_src, "test_cl_chain_gates.py", "CL_PRECHECK_PATH"), (CONTROL_MUTATIONS, src, "test_cl_chain_gates.py", "CL_READ_PATH")):
        rc, last = run(PRE if var == "CL_PRECHECK_PATH" else SRC, " or ".join(sorted({m[3] for m in muts})), test, var)
        lines.append(f"CLEAN ({var})   rc={rc}  {last}")
        ok &= rc == 0
        for name, old, new, k in muts:
            n = srcs.count(old)
            if n != 1:
                lines.append(f"{name}: REPLACEMENT MATCHES {n} TIMES (must be 1)")
                ok = False
                continue
            p = f"{MUT}/{name}.py"
            open(p, "w", encoding="utf-8", newline="\n").write(srcs.replace(old, new))
            rc, last = run(p, k, test, var)
            ok &= rc != 0
            lines.append(f"{name}: {'RED' if rc != 0 else 'GREEN (NOT CAUGHT)'}  rc={rc}  [{k}]  {last}")
    lines.append("MUTATIONS ALL RED" if ok else "MUTATIONS NOT ALL RED")
    text = "\n".join(lines) + "\n"
    open(f"{PK}/logs/mutations_cl.log", "w", encoding="utf-8", newline="\n").write(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
