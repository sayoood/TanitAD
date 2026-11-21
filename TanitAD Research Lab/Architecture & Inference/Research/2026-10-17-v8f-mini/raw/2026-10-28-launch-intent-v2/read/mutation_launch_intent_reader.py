#!/usr/bin/env python3
"""mutation_launch_intent_reader.py -- the mutation pass over the decision functions, the bar, the per-seed AND, the outcome table, the controls and the heavy-layer wiring of ``launch_intent_read.py``.

Each mutant is the reader with ONE change; ``test_launch_intent_read.py`` is run against it (``LAUNCH_INTENT_READER=<mutant>``) as a pytest subprocess with ``-x``; KILLED iff pytest exits non-zero.
A SURVIVOR is a finding (a test to add, or an equivalent mutant to name).  The unmutated reader must pass first.  Every anchor must occur exactly once in the reader.

    python mutation_launch_intent_reader.py --reader launch_intent_read.py --tests test_launch_intent_read.py --work <scratch> --out mutation_reader.json --prereg <PREREG file>
    python mutation_launch_intent_reader.py --reader launch_intent_read.py --check-anchors
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

MUTANTS = [
    # ---- the bar and its constants
    ("bar_factor", "BAR_FACTOR = 2.0 ", "BAR_FACTOR = 1.0 "),
    ("bar_f_ade6", "F_ADE6 = 0.1207 ", "F_ADE6 = 0.1 "),
    ("half_f_up", "HALF_F = 0.0604 ", "HALF_F = 0.0605 "),
    ("half_f_down", "HALF_F = 0.0604 ", "HALF_F = 0.0603 "),
    ("two_f", "TWO_F = 0.2415 ", "TWO_F = 0.1207 "),
    ("bar_f_on_every_row", 'max(F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label]))', "max(F_ADE6, abs(rf0[label]))"),
    ("bar_f_dropped", 'max(F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label]))', "abs(rf0[label])"),
    ("bar_rf0_dropped", 'max(F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label]))', 'F_ADE6 if label == "ADE6" else 0.0'),
    ("bar_min_not_max", "BAR_FACTOR * max(F_ADE6", "BAR_FACTOR * min(F_ADE6"),
    ("bar_signed_rf0", "abs(rf0[label]))", "rf0[label])"),
    ("clears_exclusive", "return bool(sep_better(label, row) and -worsening(label, row) >= bar)", "return bool(sep_better(label, row) and -worsening(label, row) > bar)"),
    ("clears_not_separated", "return bool(sep_better(label, row) and -worsening(label, row) >= bar)", "return bool(-worsening(label, row) >= bar)"),
    ("clears_reads_worse", "return bool(sep_better(label, row) and -worsening(label, row) >= bar)", "return bool(sep_worse(label, row) and -worsening(label, row) >= bar)"),
    ("direction_ignored", 'return d if DIRECTION[label] == "lower" else -d', "return d"),
    ("sep_worse_not_separated", 'return bool(_row(label, row)["separated"] and worsening(label, row) > 0)', "return bool(worsening(label, row) > 0)"),
    ("sep_worse_zero", 'return bool(_row(label, row)["separated"] and worsening(label, row) > 0)', 'return bool(_row(label, row)["separated"] and worsening(label, row) >= 0)'),
    ("sep_better_not_separated", 'return bool(_row(label, row)["separated"] and worsening(label, row) < 0)', "return bool(worsening(label, row) < 0)"),
    ("sep_better_zero", 'return bool(_row(label, row)["separated"] and worsening(label, row) < 0)', 'return bool(_row(label, row)["separated"] and worsening(label, row) <= 0)'),
    ("boot_float64_cast", "    d = np.asarray(d)  ", "    d = np.asarray(d, np.float64)  "),
    ("row_not_computed_ignored", 'if row is None or row.get("status") == "NOT COMPUTED":\n        raise RowNotComputed(label)', 'if row is None:\n        raise RowNotComputed(label)'),
    # ---- fires / domain / one pair
    ("fires_exclusive", "sep_worse(label, x_y) and worsening(label, x_y) >= bar)", "sep_worse(label, x_y) and worsening(label, x_y) > bar)"),
    ("fires_domain_dropped", "and not sep_worse(label, y_p10) and sep_worse(label, x_y)", "and sep_worse(label, x_y)"),
    ("fires_xp10_dropped", "return bool(sep_worse(label, x_p10) and not sep_worse(label, y_p10)", "return bool(not sep_worse(label, y_p10)"),
    ("fires_xy_not_separated", "and sep_worse(label, x_y) and worsening(label, x_y) >= bar)", "and worsening(label, x_y) >= bar)"),
    ("domain_exempts_ade6", "return [l for l in LABELS if not sep_worse(l, y_p10[l])]", 'return [l for l in LABELS if not sep_worse(l, y_p10[l]) and l != "ADE6"]'),
    ("domain_point_only", "return [l for l in LABELS if not sep_worse(l, y_p10[l])]", "return [l for l in LABELS if not worsening(l, y_p10[l]) > 0]"),
    ("f1_worse_not_separated", 'return bool(row["separated"] and float(row["delta"]) < 0)', 'return bool(float(row["delta"]) < 0)'),
    ("f1_worse_zero", 'return bool(row["separated"] and float(row["delta"]) < 0)', 'return bool(row["separated"] and float(row["delta"]) <= 0)'),
    ("f1_worse_sign", 'return bool(row["separated"] and float(row["delta"]) < 0)', 'return bool(row["separated"] and float(row["delta"]) > 0)'),
    ("compose_validation_dropped", '    for k_ in ("X_P10", "X_Y", "Y_P10"):', '    for k_ in ():'),
    ("crawl_reads_xy", 'crawl_rows = {l: not sep_worse(l, t["X_P10"][l]) for l in crawl}', 'crawl_rows = {l: not sep_worse(l, t["X_Y"][l]) for l in crawl}'),
    ("move_reads_xp10", 'move_rows = {l: clears(l, t["X_Y"][l], bars[l]) for l in crawl}', 'move_rows = {l: clears(l, t["X_P10"][l], bars[l]) for l in crawl}'),
    ("move_any", '"R-MOVE": all(move_rows.values())', '"R-MOVE": any(move_rows.values())'),
    ("crawl_any", '"R-CRAWL": all(crawl_rows.values())', '"R-CRAWL": any(crawl_rows.values())'),
    ("keep_exclusive", '"R-KEEP": bool(g["KEEP_point"] <= HALF_F)', '"R-KEEP": bool(g["KEEP_point"] < HALF_F)'),
    ("keep_reads_t0", '"R-KEEP": bool(g["KEEP_point"] <= HALF_F)', '"R-KEEP": bool(g["T0_point"] <= HALF_F)'),
    ("t0_exclusive", '"R-T0": bool(g["T0_point"] <= HALF_F)', '"R-T0": bool(g["T0_point"] < HALF_F)'),
    ("t0_reads_keep", '"R-T0": bool(g["T0_point"] <= HALF_F)', '"R-T0": bool(g["KEEP_point"] <= HALF_F)'),
    ("new_all_not_any", '"R-NEW": not any(fire_rows.values())', '"R-NEW": not all(fire_rows.values())'),
    ("nobase_all_not_any", '"R-NOBASE": not any(nobase_rows.values())', '"R-NOBASE": not all(nobase_rows.values())'),
    ("ade_reads_xp10", 'not sep_worse("ADE6", t["X_Y"]["ADE6"])', 'not sep_worse("ADE6", t["X_P10"]["ADE6"])'),
    ("tlat_ignored", '"R-TLAT": not f1_worse(g["R-TLAT"])', '"R-TLAT": True'),
    ("tlon_reads_tlat", '"R-TLON": not f1_worse(g["R-TLON"])', '"R-TLON": not f1_worse(g["R-TLAT"])'),
    ("nobase_exclusive", 'worsening(l, t["X_Y"][l]) >= bars[l]) for l in d_}', 'worsening(l, t["X_Y"][l]) > bars[l]) for l in d_}'),
    ("nobase_whole_labels", 'worsening(l, t["X_Y"][l]) >= bars[l]) for l in d_}', 'worsening(l, t["X_Y"][l]) >= bars[l]) for l in LABELS}'),
    ("fire_whole_labels", 'bars[l]) for l in d_}\n    nobase_rows', 'bars[l]) for l in LABELS}\n    nobase_rows'),
    ("nobase_xp10_dropped", 'bool(sep_worse(l, t["X_P10"][l]) and sep_worse(l, t["X_Y"][l]) and worsening', 'bool(sep_worse(l, t["X_Y"][l]) and worsening'),
    # ---- the per-seed AND over the two pairs
    ("seed_flags_or", 'bool(main["flags"][k] and rep["flags"][k])', 'bool(main["flags"][k] or rep["flags"][k])'),
    ("seed_flags_main_only", 'bool(main["flags"][k] and rep["flags"][k])', 'bool(main["flags"][k])'),
    ("seed_flags_rep_only", 'bool(main["flags"][k] and rep["flags"][k])', 'bool(rep["flags"][k])'),
    ("seed_guards_or", 'bool(main["guards"][g] and rep["guards"][g])', 'bool(main["guards"][g] or rep["guards"][g])'),
    ("seed_guards_main_only", 'bool(main["guards"][g] and rep["guards"][g])', 'bool(main["guards"][g])'),
    ("seed_guards_rep_only", 'bool(main["guards"][g] and rep["guards"][g])', 'bool(rep["guards"][g])'),
    # ---- the outcome table as one function
    ("refused_never", '"REFUSED": not ctrl,', '"REFUSED": False,'),
    ("harms_ignores_ctrl", '"HARMS": ctrl and not guards_held,', '"HARMS": not guards_held,'),
    ("guards_any", "guards_held = all(held[g][s] for g in GUARDS for s in SEEDS)", "guards_held = any(held[g][s] for g in GUARDS for s in SEEDS)"),
    ("guards_one_seed", "guards_held = all(held[g][s] for g in GUARDS for s in SEEDS)", "guards_held = all(held[g][0] for g in GUARDS)"),
    ("rm_both_any", "rm_both, rm_n, rc_both = all(rm[s] for s in SEEDS),", "rm_both, rm_n, rc_both = any(rm[s] for s in SEEDS),"),
    ("rc_both_any", "all(rc[s] for s in SEEDS)\n    adopt_ok", "any(rc[s] for s in SEEDS)\n    adopt_ok"),
    ("adopt_any", "adopt_ok = all(keep[s] and nobase[s] for s in SEEDS)", "adopt_ok = any(keep[s] and nobase[s] for s in SEEDS)"),
    ("adopt_keep_or_nobase", "adopt_ok = all(keep[s] and nobase[s] for s in SEEDS)", "adopt_ok = all(keep[s] or nobase[s] for s in SEEDS)"),
    ("adopt_nobase_dropped", "adopt_ok = all(keep[s] and nobase[s] for s in SEEDS)", "adopt_ok = all(keep[s] for s in SEEDS)"),
    ("adopt_keep_dropped", "adopt_ok = all(keep[s] and nobase[s] for s in SEEDS)", "adopt_ok = all(nobase[s] for s in SEEDS)"),
    ("fixed_adopted_rc_dropped", '"FIXED / ADOPTED": ctrl and guards_held and rm_both and rc_both and adopt_ok,', '"FIXED / ADOPTED": ctrl and guards_held and rm_both and adopt_ok,'),
    ("fixed_not_adopted_adopt_ok", '"FIXED / NOT ADOPTED": ctrl and guards_held and rm_both and rc_both and not adopt_ok,', '"FIXED / NOT ADOPTED": ctrl and guards_held and rm_both and rc_both and adopt_ok,'),
    ("partial_adopted_rc", '"PARTIAL / ADOPTED": ctrl and guards_held and rm_both and not rc_both and adopt_ok,', '"PARTIAL / ADOPTED": ctrl and guards_held and rm_both and rc_both and adopt_ok,'),
    ("partial_not_adopted_move", '"PARTIAL / NOT ADOPTED": ctrl and guards_held and rm_both and not rc_both and not adopt_ok,', '"PARTIAL / NOT ADOPTED": ctrl and guards_held and rm_n >= 1 and not rc_both and not adopt_ok,'),
    ("one_seed_ge1", '"PARTIAL / ONE SEED": ctrl and guards_held and rm_n == 1,', '"PARTIAL / ONE SEED": ctrl and guards_held and rm_n >= 1,'),
    ("no_effect_le1", '"NO EFFECT": ctrl and guards_held and rm_n == 0,', '"NO EFFECT": ctrl and guards_held and rm_n <= 1,'),
    ("no_effect_guards_dropped", '"NO EFFECT": ctrl and guards_held and rm_n == 0,', '"NO EFFECT": ctrl and rm_n == 0,'),
    ("decide_unplaced_empty_only", 'out = matching[0] if len(matching) == 1 else "UNPLACED"', 'out = matching[0] if matching else "UNPLACED"'),
    ("decide_adopts_all", '"adopts": out in ADOPTING', '"adopts": out in ADOPTING + ("PARTIAL / NOT ADOPTED",)'),
    ("adopting_one", 'ADOPTING = ("FIXED / ADOPTED", "PARTIAL / ADOPTED")', 'ADOPTING = ("FIXED / ADOPTED",)'),
    ("decide_arm_swaps_rc_rm", 'f("R-CRAWL"), f("R-MOVE"), f("R-KEEP"), f("R-NOBASE"))', 'f("R-MOVE"), f("R-CRAWL"), f("R-KEEP"), f("R-NOBASE"))'),
    ("decide_arm_swaps_keep_nobase", 'f("R-KEEP"), f("R-NOBASE"))', 'f("R-NOBASE"), f("R-KEEP"))'),
    ("decide_arm_ctrl_forced", "return decide_li(ctrl_ok, held,", "return decide_li(True, held,"),
    ("outcome_names_reordered", "OUTCOME_NAMES = tuple(r[0] for r in OUTCOME_TABLE)", "OUTCOME_NAMES = tuple(sorted(r[0] for r in OUTCOME_TABLE))"),
    # ---- committed next, the cascade, NEXT-LAUNCH
    ("next_a_b_swapped_first", 'return nxt[2:i].strip() if arm == "A" else nxt[i + 2:].strip()', 'return nxt[2:i].strip() if arm == "B" else nxt[i + 2:].strip()'),
    ("next_a_b_swapped_second", 'return nxt[i + 2:j].strip() if arm == "A" else nxt[j + 2:].strip()', 'return nxt[i + 2:j].strip() if arm == "B" else nxt[j + 2:].strip()'),
    ("next_as_row_dropped", 'm = re.fullmatch(r"as ([A-Z /]+)", nxt)', 'm = None'),
    ("cascade_a_decides", 'if a_block["OUTCOME"] not in SENDS_TO_B:', 'if a_block["OUTCOME"] in SENDS_TO_B:'),
    ("cascade_final_is_a", '"final_arm": "B", "final": b_block["OUTCOME"]', '"final_arm": "B", "final": a_block["OUTCOME"]'),
    ("cascade_b_promotes", '"final_arm": "A", "final": a_block["OUTCOME"], "b_status": ("not read"', '"final_arm": "B", "final": a_block["OUTCOME"], "b_status": ("not read"'),
    ("sends_to_b_no_harms", 'SENDS_TO_B = ("HARMS", "FIXED / NOT ADOPTED", "PARTIAL / NOT ADOPTED")', 'SENDS_TO_B = ("FIXED / NOT ADOPTED", "PARTIAL / NOT ADOPTED")'),
    ("sends_to_b_adds_one_seed", 'SENDS_TO_B = ("HARMS", "FIXED / NOT ADOPTED", "PARTIAL / NOT ADOPTED")', 'SENDS_TO_B = ("HARMS", "FIXED / NOT ADOPTED", "PARTIAL / NOT ADOPTED", "PARTIAL / ONE SEED")'),
    ("next_launch_inverted", '    if mlaunch_moved:\n        return "LOOP-STATE"', '    if not mlaunch_moved:\n        return "LOOP-STATE"'),
    ("unplaced_keeps_next", 'return {**d, "arm": arm, "condition": None, "committed_next": None, "note": UNPLACED_TEXT}', 'return {**d, "arm": arm, "condition": None, "committed_next": "adopt", "note": UNPLACED_TEXT}'),
    # ---- Read A on adoption
    ("reada_d1_exclusive", 'float(rows["ADE6"]["delta"]) <= -TWO_F)', 'float(rows["ADE6"]["delta"]) < -TWO_F)'),
    ("reada_d1_not_separated", 'd1 = bool(sep_better("ADE6", rows["ADE6"]) and', "d1 = bool(True and"),
    ("reada_d1_any_seed", 'd1 = all(per[f"s{s}"]["D1"] for s in SEEDS)', 'd1 = any(per[f"s{s}"]["D1"] for s in SEEDS)'),
    ("reada_d2_ignored", '"D2": not worse,', '"D2": True,'),
    ("reada_d2_any_seed", 'd2 = all(per[f"s{s}"]["D2"] for s in SEEDS)', 'd2 = any(per[f"s{s}"]["D2"] for s in SEEDS)'),
    ("reada_d3_any", '"D3": all(fams.values())', '"D3": any(fams.values())'),
    ("reada_d3_any_seed", 'd3 = all(per[f"s{s}"]["D3"] for s in SEEDS)', 'd3 = any(per[f"s{s}"]["D3"] for s in SEEDS)'),
    ("reada_family_dropped", 'for f in ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")}', 'for f in ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL")}'),
    ("reada_crawl_ignored", 'crawl_pass = not any(per[f"s{s}"]["M1_sep_worse"] or per[f"s{s}"]["L2_sep_worse"] for s in SEEDS)', "crawl_pass = True"),
    ("reada_crawl_m1_only", 'per[f"s{s}"]["M1_sep_worse"] or per[f"s{s}"]["L2_sep_worse"] for s in SEEDS)', 'per[f"s{s}"]["M1_sep_worse"] for s in SEEDS)'),
    ("reada_no_margin_label", 'if not d1:\n        label = "NO MARGIN"', 'if d1:\n        label = "NO MARGIN"'),
    # ---- floors, zeros, families
    ("floors_signed", 'max(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS)', 'max(float(_row(l, tables[s][l])["delta"]) for s in SEEDS)'),
    ("floors_min", 'max(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS)', 'min(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS)'),
    ("floors_one_seed", 'max(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS)', 'max(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS[:1])'),
    ("zero_empty_ok", "return bool(comp) and all(", "return all("),
    ("zero_hi_unchecked", 'r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 for r in comp)', 'r["delta"] == 0.0 and r["lo"] == 0.0 for r in comp)'),
    ("zero_lo_unchecked", 'r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 for r in comp)', 'r["delta"] == 0.0 and r["hi"] == 0.0 for r in comp)'),
    ("zero_delta_unchecked", 'r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 for r in comp)', 'r["lo"] == 0.0 and r["hi"] == 0.0 for r in comp)'),
    ("strategic_paused", 'out["STRATEGIC"]["paused"] = False', 'out["STRATEGIC"]["paused"] = True'),
    ("families_row_filter", "labs = [l for l in LABELS if FAMILY[l] == fam]", "labs = [l for l in LABELS if FAMILY[l] != fam]"),
    # ---- the slow-draw flag
    ("slow_shift_value", "SLOW_SHIFT = 0.5 ", "SLOW_SHIFT = 0.4 "),
    ("slow_shift_inclusive", "cells[n][0] < -thr for n in SPEED_CELLS)", "cells[n][0] <= -thr for n in SPEED_CELLS)"),
    ("slow_ci_touching_zero", "slow = all(cells[n][2] < 0.0 and", "slow = all(cells[n][2] <= 0.0 and"),
    ("slow_any_cell", "slow = all(cells[n][2] < 0.0 and cells[n][0] < -thr for n in SPEED_CELLS)", "slow = any(cells[n][2] < 0.0 and cells[n][0] < -thr for n in SPEED_CELLS)"),
    ("slow_ci_ignored", "slow = all(cells[n][2] < 0.0 and cells[n][0] < -thr for n in SPEED_CELLS)", "slow = all(cells[n][0] < -thr for n in SPEED_CELLS)"),
    ("fast_shift_inclusive", "cells[n][0] > thr for n in SPEED_CELLS)", "cells[n][0] >= thr for n in SPEED_CELLS)"),
    ("fast_ci_touching_zero", "fast = all(cells[n][1] > 0.0 and", "fast = all(cells[n][1] >= 0.0 and"),
    ("fast_any_cell", "fast = all(cells[n][1] > 0.0 and cells[n][0] > thr for n in SPEED_CELLS)", "fast = any(cells[n][1] > 0.0 and cells[n][0] > thr for n in SPEED_CELLS)"),
    ("slow_label_swapped", 'return {"label": "SLOW-T" if slow else ("FAST-T" if fast else None)', 'return {"label": "FAST-T" if slow else ("SLOW-T" if fast else None)'),
    ("speed_cells_set", 'SPEED_CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")', 'SPEED_CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms")'),
    ("speed_cells_sign", "cx[:, CONS_NAMES.index(n)] - cy[:, CONS_NAMES.index(n)]", "cy[:, CONS_NAMES.index(n)] - cx[:, CONS_NAMES.index(n)]"),
    ("boot_percentile", "np.percentile(bs, 2.5)", "np.percentile(bs, 5.0)"),
    ("boot_percentile_hi", "np.percentile(bs, 97.5)", "np.percentile(bs, 95.0)"),
    ("boot_ratio_of_means", "bs.append(s[pick].sum() / max(n[pick].sum(), 1))", "bs.append(s[pick].sum() / max(n.sum(), 1))"),
    ("boot_default_rng_seed", "rng = np.random.default_rng(0) if rng is None else rng", "rng = np.random.default_rng(1) if rng is None else rng"),
    ("n_boot", "N_BOOT, N_WINDOWS, N_EPISODES = 2000,", "N_BOOT, N_WINDOWS, N_EPISODES = 1000,"),
    # ---- K-PAIR / K-REGISTERED / K-INIT
    ("flag_diff_ignores_nothing", 'def flag_diff(fa, fb, ignore=("--out",)):', "def flag_diff(fa, fb, ignore=()):"),
    ("norm_not_numeric", "            return repr(float(x))", "            return str(x)"),
    ("one_value_multi_ok", "if not occ or len(occ) != 1 or len(occ[0]) != 1:", "if not occ:"),
    ("one_value_multi_values_ok", "if not occ or len(occ) != 1 or len(occ[0]) != 1:", "if not occ or len(occ) != 1:"),
    ("rep_seed_map", "REP_SEED = {0: 10, 1: 11}", "REP_SEED = {0: 10, 1: 10}"),
    ("kpair_diff", "        if d != [LEVER]:\n            bad.append(f\"{role}_s{seed}'s argv differs from its pair control's", "        if False:\n            bad.append(f\"{role}_s{seed}'s argv differs from its pair control's"),
    ("kpair_w_unchecked", "        if not ok or v != lever_w(role):\n            bad.append(f\"{LEVER} {v!r} on {role}_s{seed}", "        if False:\n            bad.append(f\"{LEVER} {v!r} on {role}_s{seed}"),
    ("kpair_b_vs_a_skipped", 'if role in ("B", "Brep") and a_flags is not None:', 'if False and a_flags is not None:'),
    ("kpair_rep_seed", 'want["--seed"] = [[str(REP_SEED[seed])]]', 'want["--seed"] = [[str(seed)]]'),
    ("kpair_unknown_role", 'return False, [f"unknown role {role!r}"]', 'return True, [f"unknown role {role!r}"]'),
    ("kreg_w_acc_unchecked", '    if not ok_w or v_w != W_ACC:', '    if False:'),
    ("kreg_bank_unchecked", '    if not ok_b or v_b != OACC_BANK_SHA[seed]:', '    if False:'),
    ("kreg_dagger_seed_unchecked", '    if not ok_d or v_d != str(seed):', '    if False:'),
    ("kreg_composition_allowed", '    if "--ego-perturb-with-dagger" in flags:', '    if False:'),
    ("kreg_ego_p", '("--ego-perturb-p", 0.0)', '("--ego-perturb-p", 0.1)'),
    ("kreg_swap_p", '("--dagger-p", SWAP_P)', '("--dagger-p", 0.2)'),
    ("kreg_seed_not_run_seed", '("--seed", run_seed_of(role, seed))', '("--seed", seed)'),
    ("kreg_steps", '("--steps", STEPS)', '("--steps", 100)'),
    ("kreg_arm", '    if not ok or v != "M2b":', '    if False:'),
    ("kreg_init_unchecked", '    if not ok or not str(v).replace("\\\\", "/").endswith(f"/M2b_s{seed}/ckpt_last.pt"):', '    if False:'),
    ("kreg_init_seed", 'endswith(f"/M2b_s{seed}/ckpt_last.pt")', 'endswith("/ckpt_last.pt")'),
    ("kreg_lever_unchecked", '        if not ok or v != lever_w(role):\n            bad.append(f"{LEVER} {v!r}, registered {lever_w(role)} on {role}")', '        if False:\n            bad.append(f"{LEVER} {v!r}, registered {lever_w(role)} on {role}")'),
    ("kreg_lever_on_control", "    elif LEVER in flags:", "    elif False:"),
    ("kinit_empty_ok", "    if not init_values:\n        return False,", "    if False:\n        return False,"),
    ("kinit_value_unchecked", "if str(v) != M2B_CKPT_SHA[seed]]", "if False]"),
    # ---- K-SHARE / K-MASK
    ("share_formula", "return ((1.0 + w) * n) / ((1.0 + w) * n + (eligible - n))", "return (w * n) / (w * n + (eligible - n))"),
    ("share_tol_wide", "SHARE_TOL = 0.02", "SHARE_TOL = 0.03"),
    ("share_tol_narrow", "SHARE_TOL = 0.02", "SHARE_TOL = 0.01"),
    ("share_target_a", 'SHARE_TARGET = {"A": 1.0 / 3.0,', 'SHARE_TARGET = {"A": 1.0 / 4.0,'),
    ("share_w_unchecked", "    if w is None or abs(float(w) - float(W_ARM[arm])) > 1e-12:", "    if w is None:"),
    ("share_w_ratio_loose", "abs(float(w) - float(W_ARM[arm])) > 1e-12", "abs(float(w) - float(W_ARM[arm])) > 1.0"),
    ("share_rows_unchecked", "    if rows != e:", "    if False:"),
    ("share_recorded_unchecked", "abs(float(rec) - sh) > 1e-9", "abs(float(rec) - sh) > 1.0"),
    ("share_counts_unchecked", "e > 0 and 0 <= n <= e):", "e > 0):"),
    ("share_control_prefix", 'str(k).startswith("launch")', 'str(k).startswith("launch_n")'),
    ("share_control_w", 'levers["launch_intent_w"] != 0.0', 'levers["launch_intent_w"] != 1.0'),
    ("share_control_counts", '    if levers.get("launch_n") not in (None, 0) or levers.get("launch_eligible") not in (None, 0):', "    if False:"),
    ("mask_tol", "MASK_TOL = 0.02", "MASK_TOL = 0.03"),
    ("mask_tol_narrow", "MASK_TOL = 0.02", "MASK_TOL = 0.01"),
    ("mask_f", "F_LAUNCH = 49618 / 306566", "F_LAUNCH = 50000 / 306566"),
    ("mask_window_list", '    if summary.get("train_window_list_sha256") != WINDOW_LIST_SHA:', "    if False:"),
    ("mask_no_fraction_ok", '    else:\n        bad.append(f"launch_n {n!r} / launch_eligible {e!r}: no fraction")', "    else:\n        pass"),
    # ---- K-BUILD
    ("build_empty_ok", "    if not rec:\n        return False,", "    if False:\n        return False,"),
    ("build_commit_len", 'r"[0-9a-f]{40}"', 'r"[0-9a-f]{3,40}"'),
    ("build_base_unchecked", '    if not str(rec.get("base_commit") or "").startswith(BUILD_BASE_COMMIT):', "    if False:"),
    ("build_missing_sha_ok", '        if not fr or not fr.get("sha256_launch") or not fr.get("sha256_base"):', "        if False:"),
    ("build_byte_identical_data", "            if f in BYTE_IDENTICAL_FILES:", "            if False:"),
    ("build_no_hunk_ok", "            if not hunks:", "            if False:"),
    ("build_default_off", 'if h.get("default_off") is not True or not h.get("behind_flag") or h.get("bit_identity_test") != "PASS":', 'if not h.get("behind_flag") or h.get("bit_identity_test") != "PASS":'),
    ("build_flag_unnamed", 'if h.get("default_off") is not True or not h.get("behind_flag") or h.get("bit_identity_test") != "PASS":', 'if h.get("default_off") is not True or h.get("bit_identity_test") != "PASS":'),
    ("build_bit_identity", 'if h.get("default_off") is not True or not h.get("behind_flag") or h.get("bit_identity_test") != "PASS":', 'if h.get("default_off") is not True or not h.get("behind_flag"):'),
    ("build_files_set", 'BUILD_FILES = ("stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py")', 'BUILD_FILES = ("stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py")'),
    ("build_identical_files_set", 'BYTE_IDENTICAL_FILES = ("stack/tanitad/data/v8f_mini_data.py",)', "BYTE_IDENTICAL_FILES = ()"),
    # ---- K-REP-LICENCE
    ("lic_done", '    if sm.get("done") is not True:', "    if False:"),
    ("lic_swap_tol", "SWAP_P, SWAP_TOL = 3000, 1e-4, 100, 0.2, 0.1, 0.02", "SWAP_P, SWAP_TOL = 3000, 1e-4, 100, 0.2, 0.1, 0.03"),
    ("lic_swap_tol_narrow", "SWAP_P, SWAP_TOL = 3000, 1e-4, 100, 0.2, 0.1, 0.02", "SWAP_P, SWAP_TOL = 3000, 1e-4, 100, 0.2, 0.1, 0.01"),
    ("lic_swap_pair", '    if not (_count(dn) and _count(de) and de > 0 and 0 <= dn <= de):', "    if False:"),
    ("lic_ego_n", '    if sm.get("ego_perturb_n") != 0:', "    if False:"),
    ("lic_composition", '    if lev.get("ego_perturb_with_dagger") not in (False, None) or sm.get("ego_perturb_with_dagger") not in (False, None):', "    if False:"),
    ("lic_stream", '    if lev.get("ego_perturb_stream") not in (0, None):', "    if False:"),
    ("lic_token_vs_oacc", '        if tk.get(k) != oacc["token"].get(k) or not tk.get(k):', "        if False:"),
    ("lic_init_registered", '    if tk.get("init_ckpt_sha256") != M2B_CKPT_SHA[seed]:', "    if False:"),
    ("lic_bank_registered", '    if tk.get("bank_sha256") != OACC_BANK_SHA[seed]:', "    if False:"),
    ("lic_summary_vs_oacc", '        if sm.get(k) != oacc["summary"].get(k) or not sm.get(k):', "        if False:"),
    ("lic_token_excluded", '    if tk.get("excluded_sha256") != oacc["token"].get("excluded_sha256") or not tk.get("excluded_sha256"):', "    if False:"),
    ("lic_w_acc_oacc", 'float(tk.get("w_acc")) == float(W_ACC) == float(oacc["token"].get("w_acc"))', 'float(tk.get("w_acc")) == float(W_ACC)'),
    ("lic_w_acc_flag", '    if not ok or w != W_ACC:\n        bad.append(f"REP --w-acc', '    if False:\n        bad.append(f"REP --w-acc'),
    ("lic_rep_commit", "    if not rc.startswith(REP_BUILD_COMMIT):", "    if False:"),
    ("lic_oacc_commit", "    if not oc.startswith(CTRL_BUILD_COMMIT):", "    if False:"),
    # ---- the PREREG pin
    ("pin_sha", "    if pin and hashlib.sha256(raw).hexdigest() != pin:", "    if False:"),
    ("pin_crlf", 'raw = open(path, "rb").read().replace(b"\\r\\n", b"\\n")', 'raw = open(path, "rb").read()'),
    ("pin_row_present", "            if r not in lines:", "            if False:"),
    ("pin_predicate_table", '("per-seed predicate table", PREREG_PREDICATE_ROWS)', '("per-seed predicate table", ())'),
    ("pin_bar_row", '("bar row", (PREREG_BAR_ROW,))', '("bar row", ())'),
    ("pin_outcome_table", '("outcome table", PREREG_OUTCOME_ROWS)', '("outcome table", ())'),
    # ---- discovery
    ("needs_t0_fresh", 'keys = ("t0",) if role in ROLES_BANKED else ("t0", "t0_fresh")', 'keys = ("t0",)'),
    ("needs_t1_logged", 'for k in (("t1",) if role in ROLES_BANKED else ("t1", "t1_logged")):', 'for k in ("t1",):'),
    ("discover_episode_count", 'len(glob.glob(os.path.join(p, "ep*.npz"))) < N_EPISODES:', "False:"),
    ("discover_state", '"state": "PRESENT" if not miss else "PARTIAL"', '"state": "PRESENT"'),
    # ---- the heavy-layer wiring (exercised through main_read on the fake heavy layer)
    ("core_zero_rep_dropped", 'ctl.add("K-ZERO:REP-REP", all(is_exact_zero(ident["REP"][s]) for s in SEEDS),', 'ctl.add("K-ZERO:REP-REP", True or all(is_exact_zero(ident["REP"][s]) for s in SEEDS),'),
    ("core_zero_rf_dropped", 'ctl.add("K-ZERO:RF-of-REP-as-CTRL", all(v == 0.0 for v in floors_from({s: ident["REP"][s] for s in SEEDS}).values()),', 'ctl.add("K-ZERO:RF-of-REP-as-CTRL", True or all(v == 0.0 for v in floors_from({s: ident["REP"][s] for s in SEEDS}).values()),'),
    ("core_zero_role_dropped", 'ctl.add(f"K-ZERO:{role}-{role}:s{s}", is_exact_zero(ident[role][s]),', 'ctl.add(f"K-ZERO:{role}-{role}:s{s}", True,'),
    ("core_rf_table", 'tabs["REP_CTRL"] = H.contrast("REP", "CTRL")', 'tabs["REP_CTRL"] = H.contrast("CTRL", "REP")'),
    ("core_pair_control_main", '"main": (x, "CTRL")', '"main": (x, "REP")'),
    ("core_pair_control_rep", '"rep": (x + "rep", "REP")}', '"rep": (x + "rep", "CTRL")}'),
    ("core_p10_of_control", 'tabs["CTRL_P10"], tabs["REP_P10"] = H.contrast("CTRL", "P10"), H.contrast("REP", "P10")', 'tabs["CTRL_P10"], tabs["REP_P10"] = H.contrast("CTRL", "P10"), H.contrast("CTRL", "P10")'),
    ("core_slow_rng", "slow_draw(H.speed(px, py, s) if speed_rng is None else H.speed(px, py, s, speed_rng))", "slow_draw(H.speed(py, px, s) if speed_rng is None else H.speed(px, py, s, speed_rng))"),
    ("core_bars_default", "    bars = bars or bar_table(rf0)", "    bars = bars or bar_table({l: 0.0 for l in LABELS})"),
    ("core_b_needs_both", 'xs = ["A"] + (["B"] if all(', 'xs = ["A"] + (["B"] if any('),
    ("compose_y_p10_of_x", '"Y_P10": res["tabs"][f"{py}_P10"][s]}', '"Y_P10": res["tabs"][f"{px}_P10"][s]}'),
    ("compose_x_y_of_p10", '"X_Y": res["tabs"][f"{px}_{py}"][s],', '"X_Y": res["tabs"][f"{px}_P10"][s],'),
    ("compose_bars_v1", 'bars = res["floors"]["bar"]\n    for x in res["xs"]:', 'bars = {l: 2.0 * max(F_ADE6 if l == "ADE6" else 0.0, abs(res["floors"]["RF0"][l]), abs(res["floors"]["RF"][l])) for l in LABELS}\n    for x in res["xs"]:'),
    ("compose_ctl_ok", 'decide_arm(ctl_ok, per_seed)', 'decide_arm(True, per_seed)'),
    ("main_prereg_rc", 'print("LAUNCH-INTENT READ REFUSED (nothing written): the PREREG pin:", *bad, sep="\\n  ", file=sys.stderr)\n        return 2', 'print("LAUNCH-INTENT READ REFUSED (nothing written): the PREREG pin:", *bad, sep="\\n  ", file=sys.stderr)\n        return 0'),
    ("main_prereg_optional", 'bad = prereg_problems(a.prereg, a.prereg_sha or None) if a.prereg else ["--prereg is required"]\n    if bad:\n        print("LAUNCH-INTENT READ REFUSED', 'bad = prereg_problems(a.prereg, a.prereg_sha or None) if a.prereg else []\n    if bad:\n        print("LAUNCH-INTENT READ REFUSED'),
    ("main_missing_rc", '        print(json.dumps(miss, indent=1), file=sys.stderr)\n        return 4', '        print(json.dumps(miss, indent=1), file=sys.stderr)\n        return 0'),
    ("main_b_half_present_rc", 'file=sys.stderr)' + chr(10) + '        return 4' + chr(10) + '    bad = prereg_problems', 'file=sys.stderr)' + chr(10) + '        return 0' + chr(10) + '    bad = prereg_problems'),
    ("main_b_half_present_check", '    if any(v["state"] != "MISSING" for v in b_slots.values()) and not all(v["state"] == "PRESENT" for v in b_slots.values()):', "    if False:"),
    ("main_b_half_present_all", 'and not all(v["state"] == "PRESENT" for v in b_slots.values()):', 'and not any(v["state"] == "PRESENT" for v in b_slots.values()):'),
    ("main_b_half_present_none", '    if any(v["state"] != "MISSING" for v in b_slots.values()) and', '    if all(v["state"] != "MISSING" for v in b_slots.values()) and'),
    ("main_static_control", 'ctl.add("STATIC", not sp, sp or "staged readers, Read A\'s banked inputs, DAG-ORC\'s PREREG text, RF0 pin")\n    pr, rf0, det = H.rf0(rollout_files(man))', 'ctl.add("STATIC", True, sp or "staged readers, Read A\'s banked inputs, DAG-ORC\'s PREREG text, RF0 pin")\n    pr, rf0, det = H.rf0(rollout_files(man))'),
    ("main_kp0_control", 'ctl.add("K-P0", not pr, pr or {"RF0_sha256": RF0_SHA256[:16] + "...", **det})\n    ctl.run("K-BUILD"', 'ctl.add("K-P0", True, pr or {"RF0_sha256": RF0_SHA256[:16] + "...", **det})\n    ctl.run("K-BUILD"'),
    ("main_kbuild_control", 'ctl.run("K-BUILD", lambda: k_build(man.get("build")))', 'ctl.add("K-BUILD", True, None)'),
    ("main_rf0_files", "pr, rf0, det = H.rf0(rollout_files(man))", "pr, rf0, det = H.rf0([])"),
    ("main_rowerror_not_caught", "    except (RowNotComputed, OSError, KeyError, ValueError) as e:", "    except (OSError, KeyError, ValueError) as e:"),
    ("main_refused_rc", "            json.dump({\"OUTCOME\": \"REFUSED\"", "            json.dump({\"OUTCOME\": \"REFUSED\""),  # placeholder replaced below (kept out of the list by the filter)
    ("result_refused_return", '                       "note": "no row is reported (PREREG section 10)", "provenance": provenance(a, t_start, a.manifest)}, f, indent=1, default=str)\n        return 2', '                       "note": "no row is reported (PREREG section 10)", "provenance": provenance(a, t_start, a.manifest)}, f, indent=1, default=str)\n        return 0'),
    ("result_final_arm", 'final = comp[casc["final_arm"]]["decision"]', 'final = comp["A"]["decision"]'),
    ("result_unplaced_rc", 'return 3 if final["OUTCOME"] == "UNPLACED" else 0', "return 0"),
    ("result_next_launch_tuple", 'final["OUTCOME"] in ("PARTIAL / ONE SEED", "NO EFFECT", "PARTIAL / ADOPTED")', 'final["OUTCOME"] in ("NO EFFECT", "PARTIAL / ADOPTED")'),
    ("result_next_launch_no_effect", 'final["OUTCOME"] in ("PARTIAL / ONE SEED", "NO EFFECT", "PARTIAL / ADOPTED")', 'final["OUTCOME"] in ("PARTIAL / ONE SEED", "PARTIAL / ADOPTED")'),
    ("result_next_launch_adopted", 'final["OUTCOME"] in ("PARTIAL / ONE SEED", "NO EFFECT", "PARTIAL / ADOPTED")', 'final["OUTCOME"] in ("PARTIAL / ONE SEED", "NO EFFECT")'),
    ("result_next_launch_b", '(final["arm"] == "B" and final["OUTCOME"] in SENDS_TO_B)', '(final["arm"] == "A" and final["OUTCOME"] in SENDS_TO_B)'),
    ("result_read_a_main", 'read_a(res["tabs"][f"{x}_P10"])', 'read_a(res["tabs"][f"{x}rep_P10"])'),
    ("result_read_a_only_adopting", 'for x in res["xs"] if comp[x]["decision"]["adopts"]}', 'for x in res["xs"]}'),
    ("result_floors", '"floors": res["floors"],', '"floors": {},'),
    ("result_strategic_text", '"strategic_family": "NOT PAUSED for v8F (PI ruling 2026-10-17)"', '"strategic_family": "PAUSED"'),
    ("beside_slow_pairs", 'for pair, pxy in pair_roles(x).items() for s in SEEDS}}', 'for pair, pxy in list(pair_roles(x).items())[:1] for s in SEEDS}}'),
    ("beside_rf", '"RF": res["floors"]["RF"][l]', '"RF": res["floors"]["RF0"][l]'),
    ("families_block_pairs", 'for pair, (px, py) in pair_roles(x).items()} for s in SEEDS}', 'for pair, (px, py) in list(pair_roles(x).items())[:1]} for s in SEEDS}'),
    ("rollout_roles", "    for role in ROLES_NEW:\n        for s in SEEDS:\n            e = ((man.get", "    for role in ROLES_BANKED:\n        for s in SEEDS:\n            e = ((man.get"),
    ("rollout_logged_lane", 'for lane in ("t1", "t1_logged"):', 'for lane in ("t1",):'),
    ("trim_hidden_rows", 'for l, r in rows.items() if not str(l).startswith("_")}', "for l, r in rows.items()}"),
]
MUTANTS = [m for m in MUTANTS if m[0] != "main_refused_rc"]


def apply(src, old, new):
    if src.count(old) != 1:
        raise SystemExit(f"mutant anchor occurs {src.count(old)} times (need exactly 1): {old[:100]!r}")
    return src.replace(old, new, 1)


def pytest_run(tests, reader, env_extra, tmp):
    env = dict(os.environ, LAUNCH_INTENT_READER=reader, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", **env_extra)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", tmp, tests], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=900,
                       encoding="utf-8", errors="replace")
    failed = [ln.split(" - ")[0].replace("FAILED ", "").replace("ERROR ", "").strip() for ln in p.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    return {"rc": p.returncode, "failed": failed, "seconds": round(time.time() - t0, 1), "tail": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--tests", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--prereg", default="")
    ap.add_argument("--check-anchors", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated mutant names (re-run survivors)")
    a = ap.parse_args(argv)
    src = open(a.reader, encoding="utf-8", newline="").read()
    if a.check_anchors:
        bad = [(n, src.count(o)) for n, o, nw in MUTANTS if src.count(o) != 1 or o == nw]
        names = [n for n, _, _ in MUTANTS]
        dup = sorted({n for n in names if names.count(n) > 1})
        print(f"{len(MUTANTS)} mutants; bad anchors: {bad}; duplicate names: {dup}")
        return 0 if not bad and not dup else 1
    if not (a.tests and a.work and a.out):
        raise SystemExit("--tests, --work and --out are required")
    extra = {"LI_PREREG": a.prereg} if a.prereg else {}
    os.makedirs(os.path.join(a.work, "pytmp"), exist_ok=True)
    res = {"baseline": pytest_run(a.tests, a.reader, extra, os.path.join(a.work, "pytmp", "baseline")), "mutants": {}}
    print("baseline:", res["baseline"], flush=True)
    if res["baseline"]["rc"] != 0:
        json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
        print("ZZ-MUTATION-STOP baseline does not pass")
        return 2
    only = set(filter(None, a.only.split(",")))
    for name, old, new in MUTANTS:
        if only and name not in only:
            continue
        d = os.path.join(a.work, "mutants", name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "launch_intent_read.py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(apply(src, old, new))
        r = pytest_run(a.tests, path, extra, os.path.join(a.work, "pytmp", name))
        r["killed"] = r["rc"] != 0
        res["mutants"][name] = r
        print(f"{'KILLED  ' if r['killed'] else 'SURVIVED'} {name}: {r['failed'][:1]} ({r['seconds']}s)", flush=True)
    res["n_mutants"], res["n_killed"] = len(res["mutants"]), sum(r["killed"] for r in res["mutants"].values())
    res["survivors"] = sorted(n for n, r in res["mutants"].items() if not r["killed"])
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"ZZ-MUTATION-DONE {res['n_killed']}/{res['n_mutants']} killed; survivors={res['survivors']}", flush=True)
    return 0 if not res["survivors"] else 1


if __name__ == "__main__":
    sys.exit(main())
