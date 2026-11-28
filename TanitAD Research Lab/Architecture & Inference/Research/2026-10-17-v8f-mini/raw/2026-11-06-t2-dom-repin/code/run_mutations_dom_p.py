"""Mutation run of rule R-SCN in `code/read_t2_dominance.py` (AGENT P, 2026-11-06).

Each mutation DELETES one guard (or breaks one wire) in a copy of the read's source, held in memory: the file on disk is never touched. The mutated module is swapped into
`tests/test_t2v8f_dominance_rscn.py` (its module global `RD`), and the designated tests of that mutation must go RED (any exception). An anchor that does not occur exactly once in the
source is an error, so a rewording of the read cannot turn a mutation into a no-op. The unmutated module is run first: every test must be green.

Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 python -I code/run_mutations_dom_p.py   (about 3 min; ends with `mutations RED: N of M` and `ZZ P-MUT-DONE ZZ`; rc 0 only if all are RED)"""
import inspect
import os
import pathlib
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/../tests")
import pytest  # noqa: E402

import read_t2_dominance as ORIG  # noqa: E402
import test_t2v8f_dominance_rscn as T  # noqa: E402

SRC_PATH = ORIG.__file__
SRC = open(SRC_PATH, encoding="utf-8", newline="").read()
SCRATCH = pathlib.Path(os.environ.get("P_MUT_SCRATCH", "F:/TanitAD-scratch/t2v8f_P/mut_run"))
C, D_, E_, A_, B_, I_, O_, F_ = ("test_red_c_unattributed_file", "test_red_d_exclusions_of_another_stage", "test_red_exclusion_entries_are_checked", "test_red_a_failed_cell_outside_the_excluded_scenes",
                                 "test_red_b_partial_scene_or_a_done_cell_in_an_excluded_scene", "test_red_included_cells_must_all_be_done", "test_red_overall_incomplete_only_under_the_conditions",
                                 "test_red_f_scene_excluded_in_stage4_only_left_in_p10")
MAIN3 = "test_main_with_the_three_flags_reads_the_refusal_surviving_scenes_and_discloses"
G3POS = "test_g3_positive_union_removed_from_both_arms"
P10 = "test_stage4_only_exclusion_is_removed_from_p10"
C0C = "test_stage2_only_exclusion_is_removed_from_c0c"

# (name, [(anchor, replacement)], [designated tests that must go RED])
MUT = [
    ("attributed_check_deleted", [('    if exc.get("verdict") != "ATTRIBUTED" or exc.get("problems"):', "    if False:")], [C]),
    ("manifest_sha_check_deleted", [('    if not m or exc.get("manifest_sha256") != m:', "    if False:")], [D_]),
    ("stage_label_check_deleted", [('    if exc.get("stage") != STAGE_LABEL[st] or res.get("stage") not in (None, STAGE_LABEL[st]):', "    if False:")], [D_]),
    ("result_binding_deleted", [("    if res_path:\n        bound =", "    if False:\n        bound =")], [D_]),
    ("idx_check_deleted", [('        if rec.get("idx") is None or int(rec["idx"]) != i:', "        if False:")], [E_, "test_builder_red_arms_on_stage3_and_stage4_layouts"]),
    ("scene_in_stage_check_deleted", [("        if i is None:\n            why.append(f\"RSCN_SCENE_NOT_IN_STAGE", "        if False:\n            why.append(f\"RSCN_SCENE_NOT_IN_STAGE")], [E_]),
    ("reason_check_deleted", [("        if rec.get(\"reason\") not in RSCN_REASONS:", "        if False:")], [E_]),
    ("scene_problems_check_deleted", [("        if rec.get(\"problems\"):", "        if False:")], [E_]),
    ("failed_outside_deleted", [("        if outside:", "        if False:")], [A_]),
    ("partial_scene_deleted", [("            if len(got) != len(set(got)) or set(got) != cells[p]:", "            if False:")], [B_]),
    ("included_done_count_deleted", [("        if done != need:", "        if False:")], [I_]),
    ("excluded_cell_done_on_disk_deleted", [('                if stt != "FAIL" or a != 0:', "                if False:")], [B_]),
    ("excluded_cells_absent_deleted", [("            if miss:\n                why.append(f\"RSCN_EXCLUDED_CELLS_ABSENT", "            if False:\n                why.append(f\"RSCN_EXCLUDED_CELLS_ABSENT")], [B_]),
    ("included_cell_state_deleted", [('                r = cell_state(d, range(_n_lines(f"{d}/records.jsonl")))', "                r = None")], [I_]),
    ("included_cell_absent_deleted", [("                if d is None:\n                    why.append(f\"RSCN_INCLUDED_CELL_ABSENT", "                if False:\n                    why.append(f\"RSCN_INCLUDED_CELL_ABSENT")], [I_]),
    ("overall_complete_check_deleted", [("        if ex or any(by for by in failed_by.values()):", "        if False:")], [O_]),
    ("overall_incomplete_check_deleted", [('    elif o == "INCOMPLETE":\n        if not ex:', '    elif o == "INCOMPLETE":\n        if False:')], [O_]),
    ("overall_other_deleted", [("        why.append(f\"G3_OVERALL_NOT_COMPLETE stage {st}: {o!r} (rule R-SCN accepts COMPLETE, or INCOMPLETE only with the conditions above)\")", "        pass")], [O_]),
    ("pass_skipped_deleted", [('    if ro.get("passes_skipped_disk"):', "    if False:")], [O_]),
    ("scenes_skipped_deleted", [('    if ro.get("scenes_skipped"):', "    if False:")], [O_]),
    ("cells_planned_deleted", [('    if ro.get("n_cells_planned") != len(all_cells) * len(rng):', "    if False:")], [O_]),
    ("scene_count_deleted", [('    if res.get("n_scenes_manifest") not in (None, len(rng)):', "    if False:")], [O_]),
    ("stage_starts_unknown_deleted", [("            if len(starts) != 1:\n                why.append(f\"RSCN_STARTS_UNKNOWN stage", "            if False:\n                why.append(f\"RSCN_STARTS_UNKNOWN stage")], [E_]),
    ("union_starts_unknown_deleted", [("        if len(ns) != 1:\n            probs.append(", "        if False:\n            probs.append(")], [E_]),
    ("flags_partial_deleted", [("    if missing:\n        probs.append(f\"RSCN_FLAGS_PARTIAL", "    if False:\n        probs.append(f\"RSCN_FLAGS_PARTIAL")], ["test_red_flags_all_three_or_none", "test_main_with_two_flags_or_none_is_gate_fail"]),
    ("rscn_problems_early_return_deleted", [('        if rs["problems"]:\n            return {"PASS": False, "why": list(rs["problems"])', '        if False:\n            return {"PASS": False, "why": list(rs["problems"])')], [A_, C]),
    ("post_condition_deleted", [('        why += rscn_pairs_on_excluded(out_recs, {u["idx"] for u in rs["union"].values()})', "        pass")], ["test_red_e_a_pairing_that_still_sits_on_an_excluded_scene", F_]),
    ("start_count_adjustment_deleted", [('        exp_l, exp_g = exp_l - rs["starts"]["lost_launch"], exp_g - rs["starts"]["lost_guard"]', "        pass")], [G3POS]),
    ("stage4_exclusion_not_removed_from_p10", [(
        '        want = {scene_ids[i][:8] for i in idxs} - union8\n        cells = [(d, recs) for d, recs in load_cells(roots[st], arm) if scene_of_cell(d) not in union8]',
        '        own = union8 if st == 4 else frozenset(rs["per_stage"][str(st)]["excluded"])\n        want = {scene_ids[i][:8] for i in idxs} - own\n        cells = [(d, recs) for d, recs in load_cells(roots[st], arm) if scene_of_cell(d) not in own]')],
     [G3POS, P10]),
    ("stage2_exclusion_not_removed_from_c0c", [(
        '        want = {scene_ids[i][:8] for i in idxs} - union8\n        cells = [(d, recs) for d, recs in load_cells(roots[st], arm) if scene_of_cell(d) not in union8]',
        '        own = frozenset(rs["per_stage"][str(st)]["excluded"])\n        want = {scene_ids[i][:8] for i in idxs} - own\n        cells = [(d, recs) for d, recs in load_cells(roots[st], arm) if scene_of_cell(d) not in own]')],
     [C0C, G3POS]),
    ("no_flags_overall_check_deleted", [('            if o != "COMPLETE":\n                why.append(f"G3_OVERALL_NOT_COMPLETE stage {st}: {o!r}")', "            pass")], ["test_without_the_flags_the_read_is_the_previous_pin"]),
    ("g2e_excluded_tabulation_deleted", [("            if excluded and sc in excluded:", "            if False:")], ["test_g2e_tabulates_excluded_scenes_and_still_needs_the_file_for_included_ones"]),
    ("run_does_not_compute_rscn", [("    rs = rscn_check(roots, results, sids, flags, EXPECT) if rscn_on and complete_sids else None", "    rs = None")], [MAIN3]),
    ("run_flags_do_not_reach_g3", [("exclusions=flags if rscn_on else None, rscn=rs)", "exclusions=None, rscn=None)")], [MAIN3]),
    ("run_skip_not_passed_to_scene_loader", [('skip={u["idx"] for u in rs["union"].values()} if rs is not None else ())', "skip=())")], [MAIN3]),
    ("run_excluded_not_passed_to_g2e", [("excluded=rscn_labels(rs))", "excluded=None)")], [MAIN3]),
    ("run_rscn_not_in_json", [('        res["r_scn"] = rs', "        pass")], [MAIN3]),
    ("run_expect_not_passed", [("g3_check(roots, results, sids, EXPECT, ", "g3_check(roots, results, sids, ")], [MAIN3]),
    ("md_section_deleted", [('    if rsd:\n        L += ["", "## Excluded scenes (rule R-SCN)"', '    if False:\n        L += ["", "## Excluded scenes (rule R-SCN)"')], [MAIN3]),
    ("main_flags_not_declared", [('    for k in ("exclusions2", "exclusions3", "exclusions4"):\n        ap.add_argument', '    for k in ():\n        ap.add_argument')], [MAIN3]),
]


def load(src):
    mod = types.ModuleType("read_t2_dominance")
    mod.__file__ = SRC_PATH
    exec(compile(src, SRC_PATH, "exec"), mod.__dict__)
    ORIG.init()
    mod.STATE.update(ORIG.STATE)
    return mod


_N = [0]


def run_test(mod, name):
    """-> None when green, else 'ExcType: message'."""
    f = getattr(T, name)
    T.RD = mod
    prm = inspect.signature(f).parameters
    kw, mp = {}, None
    _N[0] += 1
    d = SCRATCH / f"t{_N[0]:04d}"
    d.mkdir(parents=True)
    if "tmp_path" in prm:
        kw["tmp_path"] = d
    if "monkeypatch" in prm:
        mp = pytest.MonkeyPatch()
        kw["monkeypatch"] = mp
    try:
        f(**kw)
        return None
    except BaseException as e:                                              # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:150]}".replace("\n", " ")
    finally:
        if mp:
            mp.undo()
        T.RD = ORIG


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    names = [n for n, f in inspect.getmembers(T, inspect.isfunction) if n.startswith("test_") and f.__module__ == T.__name__]
    base = load(SRC)
    bad = {n: r for n in names if (r := run_test(base, n))}
    print(f"baseline: {len(names) - len(bad)} of {len(names)} tests green" + (f"; RED: {bad}" if bad else ""), flush=True)
    if bad:
        print("ZZ P-MUT-DONE ZZ")
        return 2
    red = 0
    for name, edits, designated in MUT:
        src = SRC
        for old, new in edits:
            if SRC.count(old) != 1:
                print(f"{name}: ANCHOR ERROR (occurs {SRC.count(old)} times): {old[:80]!r}", flush=True)
                return 3
            src = src.replace(old, new)
        mod = load(src)
        res = {t: run_test(mod, t) for t in designated}
        ok = all(v is not None for v in res.values())
        red += ok
        print(f"{name}: {'RED' if ok else 'GREEN  <-- not caught by ' + str([t for t, v in res.items() if v is None])}  " + "; ".join(f"{t[:40]} -> {(v or 'passed')[:110]}" for t, v in res.items()), flush=True)
    print(f"mutations RED: {red} of {len(MUT)}", flush=True)
    print("ZZ P-MUT-DONE ZZ", flush=True)
    return 0 if red == len(MUT) else 1


if __name__ == "__main__":
    sys.exit(main())
