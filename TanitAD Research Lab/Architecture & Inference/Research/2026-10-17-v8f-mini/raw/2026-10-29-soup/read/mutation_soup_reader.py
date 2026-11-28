#!/usr/bin/env python3
"""mutation_soup_reader.py -- the mutation pass over soup_read.py: the coded constants, the per-seed AND, the outcome table, SG and its routing, K-SOUP-REC, K-TREE, the PREREG-TEXT check, the
composition (read_core) and the heavy-layer wiring (SoupHeavy, make_heavy, main), in the style of mutation_launch_intent_reader.py.

Each mutant is soup_read.py with ONE textual change (its anchor must occur exactly once); the reader's two test files are run against it (SOUP_READ_DIR=<mutant dir>) as ONE pytest subprocess with -x;
KILLED iff pytest exits non-zero.  A SURVIVOR is a finding: a test to add, or an equivalent mutant to NAME (EQUIVALENT below, each with its reason).  The unmutated reader must pass first.
Zero training, CPU only, one pytest at a time.

    python mutation_soup_reader.py --check-anchors
    python mutation_soup_reader.py --work <scratch dir> --out ../evidence/mutation_reader.json [--only name,name]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
READER = os.path.join(W, "read", "soup_read.py")
TESTS = [os.path.join(W, "read", "test_soup_read.py"), os.path.join(W, "read", "test_soup_read_heavy.py")]

MUTANTS = [
    # ---- coded constants
    ("pairs_main_control", 'PAIRS = {"main": ("SOUP", "CTRL"), "rep": ("SOUPR", "A")}', 'PAIRS = {"main": ("SOUP", "REP"), "rep": ("SOUPR", "A")}'),
    ("pairs_rep_control_is_arep", 'PAIRS = {"main": ("SOUP", "CTRL"), "rep": ("SOUPR", "A")}', 'PAIRS = {"main": ("SOUP", "CTRL"), "rep": ("SOUPR", "Arep")}'),
    ("pairs_rep_control_is_ctrl", 'PAIRS = {"main": ("SOUP", "CTRL"), "rep": ("SOUPR", "A")}', 'PAIRS = {"main": ("SOUP", "CTRL"), "rep": ("SOUPR", "CTRL")}'),
    ("pairs_main_soup_is_soupr", 'PAIRS = {"main": ("SOUP", "CTRL"), "rep": ("SOUPR", "A")}', 'PAIRS = {"main": ("SOUPR", "CTRL"), "rep": ("SOUPR", "A")}'),
    ("members_soup", '"SOUP": ("CTRL", "REP"),', '"SOUP": ("CTRL", "CTRL"),'),
    ("members_soupr", '"SOUPR": ("A", "Arep"),', '"SOUPR": ("A", "A"),'),
    ("members_soup4_short", '"SOUP4": ("CTRL", "REP", "A", "Arep")}', '"SOUP4": ("CTRL", "REP", "A")}'),
    ("soup_roles_no_soup4", 'SOUP_ROLES = ("SOUP", "SOUPR", "SOUP4")', 'SOUP_ROLES = ("SOUP", "SOUPR")'),
    ("member_roles_no_arep", 'MEMBER_ROLES = ("CTRL", "REP", "A", "Arep")', 'MEMBER_ROLES = ("CTRL", "REP", "A")'),
    ("spec_name_soupr", '"SOUPR": "SOUPR_s{s}"', '"SOUPR": "SOUP_s{s}"'),
    ("spec_member_rep", '"REP": "REP_{s}"', '"REP": "A_{s}"'),
    ("donor_soupr", 'DONOR = {"SOUP": "CTRL", "SOUPR": "A", "SOUP4": "CTRL"}', 'DONOR = {"SOUP": "CTRL", "SOUPR": "Arep", "SOUP4": "CTRL"}'),
    ("guards_no_new", 'GUARDS = ("R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW")', 'GUARDS = ("R-TLAT", "R-TLON", "R-ADE", "R-T0")'),
    ("guards_no_t0", 'GUARDS = ("R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW")', 'GUARDS = ("R-TLAT", "R-TLON", "R-ADE", "R-NEW")'),
    ("seeds_one", "SEEDS = (0, 1)", "SEEDS = (0,)"),
    ("k_rc_tables_short", '"A_P10": ("A", "P10"),\n               "Arep_P10": ("Arep", "P10")}', '"A_P10": ("A", "P10")}'),
    ("k_rc_tables_swapped", '"A_CTRL": ("A", "CTRL")', '"A_CTRL": ("CTRL", "A")'),
    ("k_rc_tables_rep_ctrl", '"REP_CTRL": ("REP", "CTRL")', '"REP_CTRL": ("CTRL", "REP")'),
    ("tool_digest", 'TOOL_DIGEST = "579183a3', 'TOOL_DIGEST = "579183a4'),
    ("li_reader_sha", 'LI_READER_SHA = "fb3d2d7d', 'LI_READER_SHA = "fb3d2d7e'),
    ("claim_row", 'CLAIM_ROW = "H-V8F-SOUP-1"', 'CLAIM_ROW = "H-V8F-SOUP-2"'),
    ("tier_t1_closed_loop", "T1-kin (self-action OPEN loop;", "T1-kin (self-action closed loop;"),
    ("estimator_b", "B=2000, seed 0, 141 episodes", "B=1000, seed 0, 141 episodes"),
    ("adopting_one", 'ADOPTING = ("DOMINANT / ADOPTED", "MOVED / ADOPTED")', 'ADOPTING = ("DOMINANT / ADOPTED",)'),
    ("routed_no_no_effect", 'ROUTED = ("HARMS", "MOVED / NOT ADOPTED", "ONE SEED", "NO EFFECT")', 'ROUTED = ("HARMS", "MOVED / NOT ADOPTED", "ONE SEED")'),
    ("routed_adds_dominant", 'ROUTED = ("HARMS", "MOVED / NOT ADOPTED", "ONE SEED", "NO EFFECT")', 'ROUTED = ("HARMS", "MOVED / NOT ADOPTED", "ONE SEED", "NO EFFECT", "DOMINANT / NOT ADOPTED")'),
    ("next_soup_swapped", 'NEXT_SOUP = {"BARRIER": "ENS2", "FLAT": "SOUP-K"}', 'NEXT_SOUP = {"BARRIER": "SOUP-K", "FLAT": "ENS2"}'),
    ("table_cell_one_seed", "MV at exactly one seed", "MV at one seed or more"),
    ("table_cell_moved_adopted_next", "the rows that still fail D2 go to SOUP-K on the pure recipe", "the rows that still fail D2 go to ENS2 on the pure recipe"),
    ("table_cell_harms_cond", '"controls pass; H fails at ≥ 1 seed"', '"controls pass; H fails at both seeds"'),
    # ---- the landed reader's pin
    ("load_li_pin_off", "    if pin is not None and got != pin:\n        raise SystemExit", "    if False:\n        raise SystemExit"),
    ("load_li_pin_inverted", "    if pin is not None and got != pin:\n        raise SystemExit", "    if pin is not None and got == pin:\n        raise SystemExit"),
    # ---- per_seed_booleans
    ("h_any", 'out[s] = {"H": all(m["guards"][g] and r["guards"][g] for g in GUARDS),', 'out[s] = {"H": any(m["guards"][g] and r["guards"][g] for g in GUARDS),'),
    ("h_or", 'all(m["guards"][g] and r["guards"][g] for g in GUARDS)', 'all(m["guards"][g] or r["guards"][g] for g in GUARDS)'),
    ("h_main_only", 'all(m["guards"][g] and r["guards"][g] for g in GUARDS)', 'all(m["guards"][g] for g in GUARDS)'),
    ("h_rep_only", 'all(m["guards"][g] and r["guards"][g] for g in GUARDS)', 'all(r["guards"][g] for g in GUARDS)'),
    ("mv_or", '"MV": bool(m["flags"]["R-MOVE"] and r["flags"]["R-MOVE"]),', '"MV": bool(m["flags"]["R-MOVE"] or r["flags"]["R-MOVE"]),'),
    ("mv_main_only", '"MV": bool(m["flags"]["R-MOVE"] and r["flags"]["R-MOVE"]),', '"MV": bool(m["flags"]["R-MOVE"]),'),
    ("mv_rep_only", '"MV": bool(m["flags"]["R-MOVE"] and r["flags"]["R-MOVE"]),', '"MV": bool(r["flags"]["R-MOVE"]),'),
    ("ad_no_main_keep", 'bool(m["flags"]["R-KEEP"] and m["flags"]["R-NOBASE"] and r["flags"]["R-KEEP"] and r["flags"]["R-NOBASE"])', 'bool(m["flags"]["R-NOBASE"] and r["flags"]["R-KEEP"] and r["flags"]["R-NOBASE"])'),
    ("ad_no_main_nobase", 'bool(m["flags"]["R-KEEP"] and m["flags"]["R-NOBASE"] and r["flags"]["R-KEEP"] and r["flags"]["R-NOBASE"])', 'bool(m["flags"]["R-KEEP"] and r["flags"]["R-KEEP"] and r["flags"]["R-NOBASE"])'),
    ("ad_no_rep_keep", 'bool(m["flags"]["R-KEEP"] and m["flags"]["R-NOBASE"] and r["flags"]["R-KEEP"] and r["flags"]["R-NOBASE"])', 'bool(m["flags"]["R-KEEP"] and m["flags"]["R-NOBASE"] and r["flags"]["R-NOBASE"])'),
    ("ad_no_rep_nobase", 'bool(m["flags"]["R-KEEP"] and m["flags"]["R-NOBASE"] and r["flags"]["R-KEEP"] and r["flags"]["R-NOBASE"])', 'bool(m["flags"]["R-KEEP"] and m["flags"]["R-NOBASE"] and r["flags"]["R-KEEP"])'),
    ("ad_keep_or_nobase", 'bool(m["flags"]["R-KEEP"] and m["flags"]["R-NOBASE"] and r["flags"]["R-KEEP"] and r["flags"]["R-NOBASE"])', 'bool((m["flags"]["R-KEEP"] or m["flags"]["R-NOBASE"]) and r["flags"]["R-KEEP"] and r["flags"]["R-NOBASE"])'),
    ("dm_or", '"DM": bool(dm[s]["SOUP"] and dm[s]["SOUPR"])}', '"DM": bool(dm[s]["SOUP"] or dm[s]["SOUPR"])}'),
    ("dm_soup_only", '"DM": bool(dm[s]["SOUP"] and dm[s]["SOUPR"])}', '"DM": bool(dm[s]["SOUP"])}'),
    ("dm_soupr_only", '"DM": bool(dm[s]["SOUP"] and dm[s]["SOUPR"])}', '"DM": bool(dm[s]["SOUPR"])}'),
    # ---- the outcome table as one function
    ("or_h_any", 'h = all(b[s]["H"] for s in SEEDS)', 'h = any(b[s]["H"] for s in SEEDS)'),
    ("or_dm_any", 'dm = all(b[s]["DM"] for s in SEEDS)', 'dm = any(b[s]["DM"] for s in SEEDS)'),
    ("or_ad_any", 'ad = all(b[s]["AD"] for s in SEEDS)', 'ad = any(b[s]["AD"] for s in SEEDS)'),
    ("or_mv_n_max", 'mv_n = sum(bool(b[s]["MV"]) for s in SEEDS)', 'mv_n = max(bool(b[s]["MV"]) for s in SEEDS)'),
    ("or_refused_never", '"REFUSED": not ctrl,', '"REFUSED": False,'),
    ("or_harms_ignores_ctrl", '"HARMS": bool(ctrl and not h),', '"HARMS": bool(not h),'),
    ("or_dom_adopted_ad_dropped", '"DOMINANT / ADOPTED": bool(ctrl and h and dm and ad),', '"DOMINANT / ADOPTED": bool(ctrl and h and dm),'),
    ("or_dom_not_adopted_inverted", '"DOMINANT / NOT ADOPTED": bool(ctrl and h and dm and not ad),', '"DOMINANT / NOT ADOPTED": bool(ctrl and h and dm and ad),'),
    ("or_dom_not_adopted_h_dropped", '"DOMINANT / NOT ADOPTED": bool(ctrl and h and dm and not ad),', '"DOMINANT / NOT ADOPTED": bool(ctrl and dm and not ad),'),
    ("or_moved_adopted_mv_ge1", 'not dm and mv_n == 2 and ad),', 'not dm and mv_n >= 1 and ad),'),
    ("or_moved_adopted_dm_dropped", '"MOVED / ADOPTED": bool(ctrl and h and not dm and mv_n == 2 and ad),', '"MOVED / ADOPTED": bool(ctrl and h and mv_n == 2 and ad),'),
    ("or_moved_not_adopted_inverted", 'not dm and mv_n == 2 and not ad),', 'not dm and mv_n == 2 and ad),'),
    ("or_one_seed_ge1", '"ONE SEED": bool(ctrl and h and not dm and mv_n == 1),', '"ONE SEED": bool(ctrl and h and not dm and mv_n >= 1),'),
    ("or_one_seed_ctrl_dropped", '"ONE SEED": bool(ctrl and h and not dm and mv_n == 1),', '"ONE SEED": bool(h and not dm and mv_n == 1),'),
    ("or_no_effect_le1", '"NO EFFECT": bool(ctrl and h and not dm and mv_n == 0),', '"NO EFFECT": bool(ctrl and h and not dm and mv_n <= 1),'),
    ("or_no_effect_h_dropped", '"NO EFFECT": bool(ctrl and h and not dm and mv_n == 0),', '"NO EFFECT": bool(ctrl and not dm and mv_n == 0),'),
    ("decide_unplaced_empty_only", 'out = matching[0] if len(matching) == 1 else "UNPLACED"', 'out = matching[0] if matching else "UNPLACED"'),
    ("decide_adopts_first_only", '"adopts": out in ADOPTING,', '"adopts": out in ADOPTING[:1],'),
    ("decide_ctrl_true", '"controls_pass": bool(ctrl),', '"controls_pass": True,'),
    ("decide_per_seed_s0", '"per_seed": {f"s{s}": dict(b[s]) for s in SEEDS}}', '"per_seed": {f"s{s}": dict(b[0]) for s in SEEDS}}'),
    ("decide_matching_list", '"matching_rows_in_table_order": matching,', '"matching_rows_in_table_order": [],'),
    # ---- SG and its routing
    ("sg_shape_check_off", "    if ms.shape[1] != x.shape[0] or len(eid) != x.shape[0]:", "    if False:"),
    ("sg_eid_check_off", "    if ms.shape[1] != x.shape[0] or len(eid) != x.shape[0]:", "    if ms.shape[1] != x.shape[0]:"),
    ("sg_x_check_off", "    if ms.shape[1] != x.shape[0] or len(eid) != x.shape[0]:", "    if len(eid) != x.shape[0]:"),
    ("sg_mask_any", "mk = np.isfinite(x) & np.isfinite(ms).all(axis=0)", "mk = np.isfinite(x) & np.isfinite(ms).any(axis=0)"),
    ("sg_mask_no_x", "mk = np.isfinite(x) & np.isfinite(ms).all(axis=0)", "mk = np.isfinite(ms).all(axis=0)"),
    ("sg_mask_no_members", "mk = np.isfinite(x) & np.isfinite(ms).all(axis=0)", "mk = np.isfinite(x)"),
    ("sg_median", "ms[:, mk].mean(axis=0)", "np.median(ms[:, mk], axis=0)"),
    ("sg_sum_not_mean", "ms[:, mk].mean(axis=0)", "ms[:, mk].sum(axis=0)"),
    ("sg_args_swapped", "r = pb(x[mk], ms[:, mk].mean(axis=0), np.asarray(eid)[mk], n_boot=n_boot, seed=0)", "r = pb(ms[:, mk].mean(axis=0), x[mk], np.asarray(eid)[mk], n_boot=n_boot, seed=0)"),
    ("sg_seed_1", "np.asarray(eid)[mk], n_boot=n_boot, seed=0)", "np.asarray(eid)[mk], n_boot=n_boot, seed=1)"),
    ("sg_n_boot_dropped", "np.asarray(eid)[mk], n_boot=n_boot, seed=0)", "np.asarray(eid)[mk], n_boot=2000, seed=0)"),
    ("sg_separated_and", '"separated": bool(lo > 0 or hi < 0),', '"separated": bool(lo > 0 and hi < 0),'),
    ("sg_barrier_ge", '"barrier": bool(lo > 0),', '"barrier": bool(lo >= 0),'),
    ("sg_barrier_hi", '"barrier": bool(lo > 0),', '"barrier": bool(hi > 0),'),
    ("sg_gain_le", '"gain": bool(hi < 0),', '"gain": bool(hi <= 0),'),
    ("sg_gain_lo", '"gain": bool(hi < 0),', '"gain": bool(lo < 0),'),
    ("sg_counts_swapped", '"n_windows": int(mk.sum()), "n_dropped": int((~mk).sum())}', '"n_windows": int((~mk).sum()), "n_dropped": int(mk.sum())}'),
    ("route_needs_soup_only", 'need = [f"{x}:s{s}" for x in ("SOUP", "SOUPR") for s in SEEDS]', 'need = [f"{x}:s{s}" for x in ("SOUP",) for s in SEEDS]'),
    ("route_none_inverted", "    if any(cells.get(k) is None for k in need):\n        return None", "    if all(cells.get(k) is None for k in need):\n        return None"),
    ("route_barrier_all", 'return "BARRIER" if any(cells[k]["barrier"] for k in need) else "FLAT"', 'return "BARRIER" if all(cells[k]["barrier"] for k in need) else "FLAT"'),
    ("route_swapped", 'return "BARRIER" if any(cells[k]["barrier"] for k in need) else "FLAT"', 'return "FLAT" if any(cells[k]["barrier"] for k in need) else "BARRIER"'),
    ("next_routed_inverted", "    if outcome not in ROUTED:", "    if outcome in ROUTED:"),
    ("next_route_none_inverted", "    if route is None:\n        return {\"committed_next\": row[2], \"next_soup\": \"UNDECIDED", "    if route is not None:\n        return {\"committed_next\": row[2], \"next_soup\": \"UNDECIDED"),
    ("next_always_flat", '"next_soup": NEXT_SOUP[route], "route": route}', '"next_soup": NEXT_SOUP["FLAT"], "route": route}'),
    ("next_unknown_row", "    if row is None:\n        return {\"committed_next\": None, \"next_soup\": None}", "    if row is not None:\n        return {\"committed_next\": None, \"next_soup\": None}"),
    # ---- K-SOUP-REC
    ("rec_name_off", '    if rec.get("name") != name:', "    if False:"),
    ("rec_output_off", '    if rec.get("output_sha256") != file_sha:', "    if False:"),
    ("rec_members_off", '    if rec.get("members") != list(s["members"]) or rec.get("donor") != s["donor"]:', "    if False:"),
    ("rec_donor_off", '    if rec.get("members") != list(s["members"]) or rec.get("donor") != s["donor"]:', '    if rec.get("members") != list(s["members"]):'),
    ("rec_member_pin_off", '        if rec.get("member_sha256", {}).get(m) != spec["members"][m]["sha256"]:', "        if False:"),
    ("rec_builder_off", '    if rec.get("builder_sha256") != builder_sha:', "    if False:"),
    ("rec_basin_b1_off", '    if not (kb.get("PASS") and kb.get("B1") and kb.get("B2")):', '    if not (kb.get("PASS") and kb.get("B2")):'),
    ("rec_basin_b2_off", '    if not (kb.get("PASS") and kb.get("B1") and kb.get("B2")):', '    if not (kb.get("PASS") and kb.get("B1")):'),
    ("rec_basin_pass_off", '    if not (kb.get("PASS") and kb.get("B1") and kb.get("B2")):', '    if not (kb.get("B1") and kb.get("B2")):'),
    ("rec_mid_off", '    if not (rec.get("K-MID") or {}).get("PASS"):', "    if False:"),
    ("rec_weights_half", '    if rec.get("weights") != [1.0 / len(s["members"])] * len(s["members"]):', '    if rec.get("weights") != [0.5] * len(s["members"]):'),
    ("rec_weights_off", '    if rec.get("weights") != [1.0 / len(s["members"])] * len(s["members"]):', "    if False:"),
    ("rec_cfg_off", "    if donor_cfg_canon is not None and ckpt_cfg_canon != donor_cfg_canon:", "    if False:"),
    ("rec_cfg_none_ok", "    if donor_cfg_canon is not None and ckpt_cfg_canon != donor_cfg_canon:", "    if ckpt_cfg_canon != donor_cfg_canon:"),
    ("rec_opt_off", "    if not opt_is_none:", "    if False:"),
    # ---- K-TREE and the rollout file list
    ("tree_not_dict_off", "    if not isinstance(done, dict):", "    if False:"),
    ("tree_no_after", 'for k in ("ckpt_sha256", "ckpt_sha256_before", "ckpt_sha256_after"):', 'for k in ("ckpt_sha256", "ckpt_sha256_before"):'),
    ("tree_no_before", 'for k in ("ckpt_sha256", "ckpt_sha256_before", "ckpt_sha256_after"):', 'for k in ("ckpt_sha256", "ckpt_sha256_after"):'),
    ("tree_no_ckpt", 'for k in ("ckpt_sha256", "ckpt_sha256_before", "ckpt_sha256_after"):', 'for k in ("ckpt_sha256_before", "ckpt_sha256_after"):'),
    ("tree_digest_off", '    if done.get("tool_digest") != tool_digest:', "    if False:"),
    ("tree_smoke_off", '    if done.get("smoke") is not False or done.get("max_windows") is not None:', '    if done.get("max_windows") is not None:'),
    ("tree_window_limit_off", '    if done.get("smoke") is not False or done.get("max_windows") is not None:', '    if done.get("smoke") is not False:'),
    ("tree_default_digest", "def k_tree(done, ckpt_sha, tool_digest=TOOL_DIGEST):", "def k_tree(done, ckpt_sha, tool_digest=None):"),
    ("files_no_arep", 'def rollout_files(man, roles=("A", "Arep") + SOUP_ROLES):', 'def rollout_files(man, roles=("A",) + SOUP_ROLES):'),
    ("files_no_soups", 'def rollout_files(man, roles=("A", "Arep") + SOUP_ROLES):', 'def rollout_files(man, roles=("A", "Arep")):'),
    ("files_one_lane", '            for lane in ("t1", "t1_logged"):\n                t = e.get(lane) or {}', '            for lane in ("t1",):\n                t = e.get(lane) or {}'),
    ("files_no_ticks", '[t.get("json") or "", t.get("ticks") or ""]', '[t.get("json") or ""]'),
    ("files_no_json", '[t.get("json") or "", t.get("ticks") or ""]', '[t.get("ticks") or ""]'),
    ("files_glob_all_npz", 'os.path.join(t.get("dump_dir") or "", "ep*.npz")', 'os.path.join(t.get("dump_dir") or "", "*.npz")'),
    ("files_one_seed", "    for role in roles:\n        for s in SEEDS:\n            e = ((man.get", "    for role in roles:\n        for s in SEEDS[:1]:\n            e = ((man.get"),
    # ---- PREREG-TEXT
    ("prereg_missing_off", "    if not os.path.exists(path):", "    if False:"),
    ("prereg_pin_off", "    if pin is not None and got != pin:\n        probs.append", "    if False:\n        probs.append"),
    ("prereg_pin_inverted", "    if pin is not None and got != pin:\n        probs.append", "    if pin is not None and got == pin:\n        probs.append"),
    ("prereg_no_end_split", '.split("## 10.", 1)[-1].split("## 11.", 1)[0]', '.split("## 10.", 1)[-1]'),
    ("prereg_rows_any_pipe", '        if ln.startswith("| **"):', '        if ln.startswith("|"):'),
    ("prereg_cells_ge4", "            if len(cells) >= 3:", "            if len(cells) >= 4:"),
    ("prereg_extra_off", "    extra = set(rows) - set(OUTCOME_NAMES)", "    extra = set()"),
    ("prereg_cond_ignored", "        if rows.get(name) != (strip_md(cond), strip_md(nxt)):", "        if rows.get(name) is None or rows[name][1] != strip_md(nxt):"),
    ("prereg_next_ignored", "        if rows.get(name) != (strip_md(cond), strip_md(nxt)):", "        if rows.get(name) is None or rows[name][0] != strip_md(cond):"),
    ("prereg_strip_md", '    return t.replace("**", "").strip()', '    return t.replace("**", "")'),
    ("families_note_target", 'fb["LATERAL"]["note"] = "the crawl rows', 'fb["TACTICAL"]["note"] = "the crawl rows'),
    # ---- read_core wiring
    ("core_records_dropped_arep", "for r in MEMBER_ROLES}\n    LI.run_record_controls", "for r in MEMBER_ROLES[:3]}\n    LI.run_record_controls"),
    ("core_record_controls_off", "    LI.run_record_controls(ctl, trs)", "    pass"),
    ("core_member_slots_one_seed", "    for r in MEMBER_ROLES:\n        for s in SEEDS:\n            H.load_slot", "    for r in MEMBER_ROLES:\n        for s in SEEDS[:1]:\n            H.load_slot"),
    ("core_soups_any_seed", 'all((man["arms"].get(x) or {}).get(str(s)) for s in SEEDS)', 'any((man["arms"].get(x) or {}).get(str(s)) for s in SEEDS)'),
    ("core_need_soup_only", 'for need in ("SOUP", "SOUPR"):', 'for need in ("SOUP",):'),
    ("core_soup_slots_one_seed", "    for x in soups:\n        for s in SEEDS:\n            H.load_soup_slot", "    for x in soups:\n        for s in SEEDS[:1]:\n            H.load_soup_slot"),
    ("core_global_off", "    H.global_controls(ctl)", "    pass"),
    ("core_prepare_off", "    H.prepare(ctl)\n", "    pass\n"),
    ("core_k_rc_off", "    H.k_rc(ctl)", "    pass"),
    ("core_bars_ignore_given", "    bars = bars or LI.bar_table(rf0)", "    bars = LI.bar_table(rf0)"),
    ("core_bars_zero", "    bars = bars or LI.bar_table(rf0)", "    bars = bars or LI.bar_table({l: 0.0 for l in LI.LABELS})"),
    ("core_rf_table_swapped", '"REP_CTRL": H.contrast("REP", "CTRL")', '"REP_CTRL": H.contrast("CTRL", "REP")'),
    ("core_ctrl_p10_of_rep", '"CTRL_P10": H.contrast("CTRL", "P10")', '"CTRL_P10": H.contrast("REP", "P10")'),
    ("core_a_p10_of_ctrl", '"A_P10": H.contrast("A", "P10")}', '"A_P10": H.contrast("CTRL", "P10")}'),
    ("core_rf_from_ctrl_p10", 'rf = LI.floors_from({s: tabs["REP_CTRL"][s] for s in SEEDS})', 'rf = LI.floors_from({s: tabs["CTRL_P10"][s] for s in SEEDS})'),
    ("core_rf_one_seed", 'rf = LI.floors_from({s: tabs["REP_CTRL"][s] for s in SEEDS})', 'rf = LI.floors_from({s: tabs["REP_CTRL"][s] for s in SEEDS[:1]})'),
    ("core_zero_no_controls", 'for role in soups + ["CTRL", "A"]:', "for role in soups:"),
    ("core_zero_no_a", 'for role in soups + ["CTRL", "A"]:', 'for role in soups + ["CTRL"]:'),
    ("core_zero_unchecked", 'ctl.add(f"K-ZERO:{role}-{role}:s{s}", LI.is_exact_zero(z[s]),', 'ctl.add(f"K-ZERO:{role}-{role}:s{s}", True,'),
    ("core_zero_one_seed", '        z = H.contrast(role, role)\n        for s in SEEDS:', '        z = H.contrast(role, role)\n        for s in SEEDS[:1]:'),
    ("core_x_p10_is_xy", 'tabs[f"{x}_P10"], tabs[f"{x}_{y}"] = H.contrast(x, "P10"), H.contrast(x, y)', 'tabs[f"{x}_P10"], tabs[f"{x}_{y}"] = H.contrast(x, y), H.contrast(x, "P10")'),
    ("core_xy_is_yx", 'tabs[f"{x}_P10"], tabs[f"{x}_{y}"] = H.contrast(x, "P10"), H.contrast(x, y)', 'tabs[f"{x}_P10"], tabs[f"{x}_{y}"] = H.contrast(x, "P10"), H.contrast(y, x)'),
    ("core_guard_swapped", 'g = H.guard_rows(ctl, x, y, s, tabs[f"{x}_{y}"])', 'g = H.guard_rows(ctl, y, x, s, tabs[f"{x}_{y}"])'),
    ("core_guard_wrong_table", 'g = H.guard_rows(ctl, x, y, s, tabs[f"{x}_{y}"])', 'g = H.guard_rows(ctl, x, y, s, tabs[f"{x}_P10"])'),
    ("core_t_y_p10_of_x", '"Y_P10": tabs[f"{y}_P10"][s]}', '"Y_P10": tabs[f"{x}_P10"][s]}'),
    ("core_t_x_y_of_p10", '"X_Y": tabs[f"{x}_{y}"][s], "Y_P10"', '"X_Y": tabs[f"{x}_P10"][s], "Y_P10"'),
    ("core_t_x_p10_of_xy", '{"X_P10": tabs[f"{x}_P10"][s],', '{"X_P10": tabs[f"{x}_{y}"][s],'),
    ("core_compose_zero_bars", "pairs[s][pname] = LI.compose_pair(t, g, bars)", "pairs[s][pname] = LI.compose_pair(t, g, LI.bar_table(rf0))"),
    ("core_g_rows_dropped", "            g_rows[(pname, s)] = g\n", "            g_rows[(pname, s)] = {}\n"),
    ("core_soup4_off", '    if "SOUP4" in soups:', "    if False:"),
    ("core_soup4_ctrl_swapped", 'tabs["SOUP4_P10"], tabs["SOUP4_CTRL"] = H.contrast("SOUP4", "P10"), H.contrast("SOUP4", "CTRL")', 'tabs["SOUP4_P10"], tabs["SOUP4_CTRL"] = H.contrast("SOUP4", "CTRL"), H.contrast("SOUP4", "P10")'),
    ("core_beside_soup_rep", 'H.contrast("SOUP", "REP"), H.contrast("SOUPR", "Arep"), H.contrast("SOUPR", "CTRL")', 'H.contrast("SOUP", "CTRL"), H.contrast("SOUPR", "Arep"), H.contrast("SOUPR", "CTRL")'),
    ("core_beside_soupr_arep", 'H.contrast("SOUP", "REP"), H.contrast("SOUPR", "Arep"), H.contrast("SOUPR", "CTRL")', 'H.contrast("SOUP", "REP"), H.contrast("SOUPR", "A"), H.contrast("SOUPR", "CTRL")'),
    ("core_beside_soupr_ctrl", 'H.contrast("SOUP", "REP"), H.contrast("SOUPR", "Arep"), H.contrast("SOUPR", "CTRL")', 'H.contrast("SOUP", "REP"), H.contrast("SOUPR", "Arep"), H.contrast("SOUPR", "REP")'),
    ("core_read_a_decisive_only", "for x in soups}\n    dm = ", "for x in soups[:2]}\n    dm = "),
    ("core_dm_no_d2", 'read_a[x]["per_seed"][f"s{s}"]["D1"] and read_a[x]["per_seed"][f"s{s}"]["D2"] and read_a[x]["per_seed"][f"s{s}"]["D3"])', 'read_a[x]["per_seed"][f"s{s}"]["D1"] and read_a[x]["per_seed"][f"s{s}"]["D3"])'),
    ("core_dm_no_d1", 'read_a[x]["per_seed"][f"s{s}"]["D1"] and read_a[x]["per_seed"][f"s{s}"]["D2"] and read_a[x]["per_seed"][f"s{s}"]["D3"])', 'read_a[x]["per_seed"][f"s{s}"]["D2"] and read_a[x]["per_seed"][f"s{s}"]["D3"])'),
    ("core_dm_no_d3", 'read_a[x]["per_seed"][f"s{s}"]["D1"] and read_a[x]["per_seed"][f"s{s}"]["D2"] and read_a[x]["per_seed"][f"s{s}"]["D3"])', 'read_a[x]["per_seed"][f"s{s}"]["D1"] and read_a[x]["per_seed"][f"s{s}"]["D2"])'),
    ("core_dm_d1_or_d2", 'read_a[x]["per_seed"][f"s{s}"]["D1"] and read_a[x]["per_seed"][f"s{s}"]["D2"] and read_a[x]["per_seed"][f"s{s}"]["D3"])', 'read_a[x]["per_seed"][f"s{s}"]["D1"] or read_a[x]["per_seed"][f"s{s}"]["D2"] and read_a[x]["per_seed"][f"s{s}"]["D3"])'),
    ("core_dm_soup_only", 'for x in ("SOUP", "SOUPR")} for s in SEEDS}', 'for x in ("SOUP",)} for s in SEEDS}'),
    ("core_sg_members", "sg[f\"{x}:s{s}\"] = H.sg(x, MEMBERS[x], s)", "sg[f\"{x}:s{s}\"] = H.sg(x, MEMBERS[\"SOUP\"], s)"),
    ("core_sg_error_filled", "                sg[f\"{x}:s{s}\"] = None\n                sg_errors", "                sg[f\"{x}:s{s}\"] = {\"barrier\": False}\n                sg_errors"),
    ("core_sg_error_unrecorded", "                sg_errors[f\"{x}:s{s}\"] = repr(e)", "                pass"),
    ("core_slow_swapped", "slow[k_] = LI.slow_draw(H.speed(x, y, s))", "slow[k_] = LI.slow_draw(H.speed(y, x, s))"),
    ("core_slow_error_unrecorded", "slow[k_], slow_errors[k_] = None, repr(e)", "slow[k_], slow_errors[k_] = None, None"),
    ("core_slow_key", 'k_ = f"{pname}:{x}-{y}:s{s}"', 'k_ = f"{pname}:{y}-{x}:s{s}"'),
    ("core_finish_off", "    H.finish(ctl)\n    b = per_seed_booleans", "    b = per_seed_booleans"),
    ("core_ctrl_forced", "    d = decide(not ctl.failed(), b)", "    d = decide(True, b)"),
    ("core_route_dropped", "    d.update(committed_next(d[\"OUTCOME\"], route))", "    d.update(committed_next(d[\"OUTCOME\"], None))"),
    ("core_floors_rf0", '"floors": {"RF0": rf0, "RF": rf, "bar": bars}', '"floors": {"RF0": rf0, "RF": rf0, "bar": bars}'),
    # ---- the records
    ("rec_refused_failed_empty", '"failed": {k: v["detail"] for k, v in ctl.failed().items()},', '"failed": {},'),
    ("rec_refused_note", '"note": "no row is reported (PREREG section 10)"', '"note": ""'),
    ("rec_result_route", '"sg_errors": res["sg_errors"], "route": res["route"],', '"sg_errors": res["sg_errors"], "route": None,'),
    ("rec_result_families_pair", "for p, (x, y) in PAIRS.items() for s in SEEDS}", "for p, (x, y) in list(PAIRS.items())[:1] for s in SEEDS}"),
    ("rec_result_strategic", '"strategic_family": "NOT PAUSED for v8F (PI ruling 2026-10-17)"', '"strategic_family": "PAUSED"'),
    # ---- make_heavy / SoupHeavy
    ("heavy_builder_pin_off", "    if sha256_file(soup_build_path) != soup_build_sha:", "    if False:"),
    ("heavy_builder_import_off", "    if os.path.normcase(os.path.abspath(SB.__file__)) != os.path.normcase(os.path.abspath(soup_build_path)):", "    if False:"),
    ("heavy_standin_default", "{}, [], standin, banked_li_read, {}", "{}, [], True, banked_li_read, {}"),
    ("heavy_work_dir_dropped", "super().__init__(orc_dir=orc_dir, work_dir=work_dir, n_boot=n_boot)", "super().__init__(orc_dir=orc_dir, n_boot=n_boot)"),
    ("heavy_n_boot_dropped", "super().__init__(orc_dir=orc_dir, work_dir=work_dir, n_boot=n_boot)", "super().__init__(orc_dir=orc_dir, work_dir=work_dir)"),
    ("heavy_dev_roles", '            if role in ("CTRL", "A"):', '            if role in ("CTRL",):'),
    ("heavy_dev_roles_all", '            if role in ("CTRL", "A"):', '            if role in ("CTRL", "A", "REP", "Arep"):'),
    ("heavy_dev_required_off", "                if not td:", "                if False:"),
    ("heavy_dev_ckpt_unchecked", 'ctl.add(f"T0DEV-CKPT:{role}_s{s}", ok,', 'ctl.add(f"T0DEV-CKPT:{role}_s{s}", True,'),
    ("heavy_dev_key", "self.dev[(role, s)] = {", "self.dev[(role, 0)] = {"),
    ("heavy_dev_beside_sign", 'float(np.nanmean(darr["ade6"]) - np.nanmean(rec["t0arr"]["ade6"]))', 'float(np.nanmean(rec["t0arr"]["ade6"]) - np.nanmean(darr["ade6"]))'),
    ("heavy_standin_branch", "            if self.standin:\n                self.na += [f\"K-SOUP-REC", "            if not self.standin:\n                self.na += [f\"K-SOUP-REC"),
    ("heavy_standin_ckpt_path", "ck_path = f\"{e['train_dir']}/ckpt_last.pt\"", "ck_path = f\"{e['soup_dir']}/ckpt_last.pt\""),
    ("heavy_real_ckpt_path", "ck_path = f\"{e['soup_dir']}/ckpt_last.pt\"\n            return", "ck_path = f\"{e['train_dir']}/ckpt_last.pt\"\n            return"),
    ("heavy_name_of_role", 'name, tag = SPEC_NAME[role].format(s=s), f"{role}_s{s}"', 'name, tag = SPEC_NAME[role].format(s=0), f"{role}_s{s}"'),
    ("heavy_rec_sha_from_record", "probs = k_soup_rec(srec, self.spec, name, sha256_file(ckp),", "probs = k_soup_rec(srec, self.spec, name, srec.get('output_sha256'),"),
    ("heavy_rec_builder_from_record", "sha256_file(soup_build_path), SB._canon(dck", "srec.get('builder_sha256'), SB._canon(dck"),
    ("heavy_rec_opt_ignored", 'SB._canon(ck["cfg"]), ck.get("opt") is None)', 'SB._canon(ck["cfg"]), True)'),
    ("heavy_rec_cfg_ignored", 'SB._canon(dck["cfg"]), SB._canon(ck["cfg"]),', 'SB._canon(dck["cfg"]), SB._canon(dck["cfg"]),'),
    ("heavy_verify_root", "v = SB.verify(self.spec, name, os.path.dirname(d))", "v = SB.verify(self.spec, name, d)"),
    ("heavy_mid_unchecked", 'ctl.add(f"K-MID:{tag}", v["PASS"], v["problems"])', 'ctl.add(f"K-MID:{tag}", True, v["problems"])'),
    ("heavy_basin_unchecked", 'ctl.add(f"K-BASIN:{tag}", kb["PASS"],', 'ctl.add(f"K-BASIN:{tag}", True,'),
    ("heavy_basin_name", "kb = SB.basin_for(self.spec, name)\n            ctl.add", "kb = SB.basin_for(self.spec, 'SOUP_s0')\n            ctl.add"),
    ("run_t1cfg_limit", 'DR.g_t1cfg(tag, rec["t1json"], ckpt_tail, "causal", pk.meta)', 'DR.g_t1cfg(tag, rec["t1json"], ckpt_tail, "logged", pk.meta)'),
    ("run_t1cfg_logged_limit", 'DR.g_t1cfg(tag, lj, ckpt_tail, "logged", lpk.meta)', 'DR.g_t1cfg(tag, lj, ckpt_tail, "causal", lpk.meta)'),
    ("run_t1cfg_logged_json", 'DR.g_t1cfg(tag, lj, ckpt_tail, "logged", lpk.meta)', 'DR.g_t1cfg(tag, rec["t1json"], ckpt_tail, "logged", lpk.meta)'),
    ("run_k4_unchecked", 'ctl.add(f"K4:{tag}", ok, det)', 'ctl.add(f"K4:{tag}", True, det)'),
    ("run_pack_unchecked", 'ctl.add(f"{k}:{tag}", ok, det)', 'ctl.add(f"{k}:{tag}", True, det)'),
    ("run_light_unchecked", 'ctl.add(f"{k}-logged:{tag}", ok, det)', 'ctl.add(f"{k}-logged:{tag}", True, det)'),
    ("run_t0f_ckpt_inverted", '== ckpt_tail, f"fresh T0 eval json', '!= ckpt_tail, f"fresh T0 eval json'),
    ("run_t0f_ckpt_unchecked", 'ctl.add(f"T0F-CKPT:{tag}", DR.tail2(str(t0fj.get("ckpt", ""))) == ckpt_tail,', 'ctl.add(f"T0F-CKPT:{tag}", True,'),
    ("run_t0_ckpt_unchecked", 'ctl.add(f"T0-CKPT:{tag}", t0_ok,', 'ctl.add(f"T0-CKPT:{tag}", True,'),
    ("run_windows_ws_unchecked", 'and np_.array_equal(d["clip_index"], pk.ci) and np_.array_equal(d["ws"], pk.t0)),', 'and np_.array_equal(d["clip_index"], pk.ci)),'),
    ("run_windows_ci_unchecked", 'and np_.array_equal(d["clip_index"], pk.ci) and np_.array_equal(d["ws"], pk.t0)),', 'and np_.array_equal(d["ws"], pk.t0)),'),
    ("run_windows_n_unchecked", 'bool(len(d["ws"]) == pk.N and', 'bool(True and'),
    ("run_grid_windows", "bool(pk.N == LI.N_WINDOWS and len(np_.unique(d[\"eid\"])) == LI.N_EPISODES)", "bool(len(np_.unique(d[\"eid\"])) == LI.N_EPISODES)"),
    ("run_grid_episodes", "bool(pk.N == LI.N_WINDOWS and len(np_.unique(d[\"eid\"])) == LI.N_EPISODES)", "bool(pk.N == LI.N_WINDOWS)"),
    ("run_r2_tolerance", '<= 1e-6, {"reader_delta"', '<= 1e-2, {"reader_delta"'),
    ("run_r2_tolerance_tight", '<= 1e-6, {"reader_delta"', '<= 1e-9, {"reader_delta"'),
    ("run_r2_no_json", "                if js is None:\n                    return False,", "                if js is None:\n                    return True,"),
    ("run_r2_swapped_arms", 'A_, B_ = self.K.comps(d["cl"], d["g"])["ade6_m"], self.K.comps(d["fl"], d["g"])["ade6_m"]', 'A_, B_ = self.K.comps(d["fl"], d["g"])["ade6_m"], self.K.comps(d["cl"], d["g"])["ade6_m"]'),
    ("run_r2_seed", "n_boot=nb, seed=0)\n                return abs", "n_boot=nb, seed=1)\n                return abs"),
    ("run_ktree_one_lane", 'for lane, ent in (("t1", t1), ("t1_logged", lg)):', 'for lane, ent in (("t1", t1),):'),
    ("run_ktree_logged_only", 'for lane, ent in (("t1", t1), ("t1_logged", lg)):', 'for lane, ent in (("t1_logged", lg),):'),
    ("run_ktree_no_file_ok", '                    if not dn or not os.path.isfile(dn):\n                        bad[lane] = f"no done record {dn!r}"\n                        continue', '                    if not dn or not os.path.isfile(dn):\n                        continue'),
    ("run_ktree_probs_dropped", "                    if probs:\n                        bad[lane] = probs", "                    if False:\n                        bad[lane] = probs"),
    ("run_ktree_want", "                want = sha256_file(ck_path)", "                want = sha256_file(t1['json'])"),
    ("run_record_not_stored", "            self.recs.setdefault(role, {})[s] = rec\n            return rec\n\n        # ---- weights-only", "            return rec\n\n        # ---- weights-only"),
    ("global_standin_branch", "            if self.standin:\n                self.na += [\"K-XINIT:s0\"", "            if not self.standin:\n                self.na += [\"K-XINIT:s0\""),
    ("global_xinit_full", 'bool(not full["PASS"] and not b2["PASS"] and not b1["PASS"])', 'bool(not b2["PASS"] and not b1["PASS"])'),
    ("global_xinit_b2", 'bool(not full["PASS"] and not b2["PASS"] and not b1["PASS"])', 'bool(not full["PASS"] and not b1["PASS"])'),
    ("global_xinit_b1", 'bool(not full["PASS"] and not b2["PASS"] and not b1["PASS"])', 'bool(not full["PASS"] and not b2["PASS"])'),
    ("global_xinit_detail_swapped", 'SB.basin_for(self.spec, name, cache, clauses=("B2",)), SB.basin_for(self.spec, name, cache, clauses=("B1",))', 'SB.basin_for(self.spec, name, cache, clauses=("B1",)), SB.basin_for(self.spec, name, cache, clauses=("B2",))'),
    ("global_xinit_one_seed", "            for s in SEEDS:\n                name = f\"XINIT_s{s}\"", "            for s in SEEDS[:1]:\n                name = f\"XINIT_s{s}\""),
    ("global_kzero_member", 'm = self.spec["members"]["DAGOACC_0"]\n                ck, _', 'm = self.spec["members"]["DAGOACC_1"]\n                ck, _'),
    ("global_kzero_unchecked", "                return kz[\"PASS\"], kz[\"per_n\"]", "                return True, kz[\"per_n\"]"),
    ("contrast_memo_off", "            if (x, y) not in self._cc:", "            if True:"),
    ("krc_rounded", "LI.row_diffs(LI.trim(t[s]), banked[\"read_a\"][name][f\"s{s}\"])]", "LI.row_diffs(LI.trim(t[s]), banked[\"read_a\"][name][f\"s{s}\"]) if False]"),
    ("krc_one_seed", "for s in SEEDS for d in LI.row_diffs", "for s in SEEDS[:1] for d in LI.row_diffs"),
    ("krc_unchecked", 'ctl.add(f"K-RC:{name}", not diffs, diffs[:10])', 'ctl.add(f"K-RC:{name}", True, diffs[:10])'),
    ("krc_rf_unchecked", 'ctl.add("K-RC:RF", rf == banked["floors"]["RF"],', 'ctl.add("K-RC:RF", True,'),
    ("krc_rf_from_wrong_table", 'rf = LI.floors_from({s: got["REP_CTRL"][s] for s in SEEDS})', 'rf = LI.floors_from({s: got["A_CTRL"][s] for s in SEEDS})'),
    ("swap_t0arr_dropped", '"t0arr": self.dev[(y, s)]["t0arr"], "t0cons": self.dev[(y, s)]["t0cons"]}', '"t0arr": keep["t0arr"], "t0cons": self.dev[(y, s)]["t0cons"]}'),
    ("swap_t0cons_dropped", '"t0arr": self.dev[(y, s)]["t0arr"], "t0cons": self.dev[(y, s)]["t0cons"]}', '"t0arr": self.dev[(y, s)]["t0arr"], "t0cons": keep["t0cons"]}'),
    ("swap_wrong_key", '"t0arr": self.dev[(y, s)]["t0arr"], "t0cons": self.dev[(y, s)]["t0cons"]}', '"t0arr": self.dev[(y, 0)]["t0arr"], "t0cons": self.dev[(y, s)]["t0cons"]}'),
    ("guard_not_restored", "return super().guard_rows(ctl, x, y, s, tab_xy)\n            finally:\n                self.recs[y][s] = keep", "return super().guard_rows(ctl, x, y, s, tab_xy)\n            finally:\n                pass"),
    ("speed_not_restored", "return super().speed(x, y, s, rng)\n            finally:\n                self.recs[y][s] = keep", "return super().speed(x, y, s, rng)\n            finally:\n                pass"),
    ("guard_no_swap", "        def guard_rows(self, ctl, x, y, s, tab_xy):\n            keep = self._with_dev_control(y, s)", "        def guard_rows(self, ctl, x, y, s, tab_xy):\n            keep = self.recs[y][s]"),
    ("speed_no_swap", "        def speed(self, x, y, s, rng=None):\n            keep = self._with_dev_control(y, s)", "        def speed(self, x, y, s, rng=None):\n            keep = self.recs[y][s]"),
    ("speed_rng_dropped", "return super().speed(x, y, s, rng)", "return super().speed(x, y, s)"),
    ("sg_cl_not_cl", 'a = K.comps(dx["cl"], dx["g"])["ade6_m"]', 'a = K.comps(dx["fl"], dx["g"])["ade6_m"]'),
    ("sg_eid_check_off_heavy", '                if not np.array_equal(self.recs[m][s]["dump"]["eid"], dx["eid"]):', "                if False:"),
    ("sg_n_boot_heavy", "return soup_gain(a, ms, dx[\"eid\"], T.PB, self.n_boot)", "return soup_gain(a, ms, dx[\"eid\"], T.PB, 2000)"),
    # ---- main
    ("main_prereg_gate_off", "    if probs:\n        print(\"REFUSED: PREREG-TEXT\", probs)\n        return 2", "    if False:\n        print(\"REFUSED: PREREG-TEXT\", probs)\n        return 2"),
    ("main_prereg_rc", "        print(\"REFUSED: PREREG-TEXT\", probs)\n        return 2", "        print(\"REFUSED: PREREG-TEXT\", probs)\n        return 0"),
    ("main_li_pin_dropped", "LI = load_li(a.li_reader, a.li_reader_sha)", "LI = load_li(a.li_reader, None)"),
    ("main_heavy_args", "make_heavy(LI, a.orc_dir, a.spec, a.soup_build, a.soup_build_sha, a.banked_li_read, work_dir=a.work_dir or None)", "make_heavy(LI, a.orc_dir, a.spec, a.soup_build, a.soup_build_sha, a.banked_li_read, work_dir=a.work_dir or None, standin=True)"),
    ("main_heavy_work_dir", "work_dir=a.work_dir or None)", "work_dir=a.work_dir)"),
    ("main_static_unchecked", 'ctl.add("STATIC", not sp,', 'ctl.add("STATIC", True,'),
    ("main_kp0_unchecked", 'ctl.add("K-P0", not pr,', 'ctl.add("K-P0", True,'),
    ("main_rf0_files", "pr, rf0, det = H.rf0(rollout_files(man))", "pr, rf0, det = H.rf0([])"),
    ("main_no_rf0_off", "    if not rf0:", "    if False:"),
    ("main_no_rf0_rc", "        print(\"SOUP READ REFUSED: K-P0\", file=sys.stderr)\n        return 2", "        print(\"SOUP READ REFUSED: K-P0\", file=sys.stderr)\n        return 0"),
    ("main_no_rf0_writes_nothing", "        write(refused_record(refused_d, ctl, prov()))\n        print(\"SOUP READ REFUSED: K-P0\"", "        print(\"SOUP READ REFUSED: K-P0\""),
    ("main_catch_valueerror", "except (LI.RowNotComputed, OSError, KeyError, ValueError, getattr(getattr(H, \"DR\", None), \"Refusal\", ValueError)) as e:", "except (LI.RowNotComputed, OSError, KeyError, getattr(getattr(H, \"DR\", None), \"Refusal\", ValueError)) as e:"),
    ("main_catch_keyerror", "except (LI.RowNotComputed, OSError, KeyError, ValueError, getattr(getattr(H, \"DR\", None), \"Refusal\", ValueError)) as e:", "except (LI.RowNotComputed, OSError, ValueError, getattr(getattr(H, \"DR\", None), \"Refusal\", ValueError)) as e:"),
    ("main_catch_oserror", "except (LI.RowNotComputed, OSError, KeyError, ValueError, getattr(getattr(H, \"DR\", None), \"Refusal\", ValueError)) as e:", "except (LI.RowNotComputed, KeyError, ValueError, getattr(getattr(H, \"DR\", None), \"Refusal\", ValueError)) as e:"),
    ("main_catch_rownotcomputed", "except (LI.RowNotComputed, OSError, KeyError, ValueError, getattr(getattr(H, \"DR\", None), \"Refusal\", ValueError)) as e:", "except (OSError, KeyError, ValueError, getattr(getattr(H, \"DR\", None), \"Refusal\", KeyError)) as e:"),
    ("main_catch_refusal", "except (LI.RowNotComputed, OSError, KeyError, ValueError, getattr(getattr(H, \"DR\", None), \"Refusal\", ValueError)) as e:", "except (LI.RowNotComputed, OSError, KeyError, ValueError) as e:"),
    ("main_catch_rc", "        print(f\"SOUP READ REFUSED (nothing written): {type(e).__name__}: {e}\", file=sys.stderr)\n        return 2", "        print(f\"SOUP READ REFUSED (nothing written): {type(e).__name__}: {e}\", file=sys.stderr)\n        return 0"),
    ("main_refused_check", '    if d["OUTCOME"] == "REFUSED":', "    if False:"),
    ("main_refused_rc", "        print(\"SOUP READ REFUSED: failed controls:\"", "        return 0\n        print(\"SOUP READ REFUSED: failed controls:\""),
    ("main_refused_full_record", "        write(refused_record(d, ctl, prov()))\n        print(\"SOUP READ REFUSED: failed", "        write(result_record(LI, res, ctl, prov()))\n        print(\"SOUP READ REFUSED: failed"),
    ("main_unplaced_rc", 'return 3 if d["OUTCOME"] == "UNPLACED" else 0', "return 0"),
    ("main_prov_reader_sha", '"reader_sha256": sha256_file(os.path.abspath(__file__)),', '"reader_sha256": a.li_reader_sha,'),
    ("main_prov_manifest", '"manifest_sha256": sha256_file(a.manifest)', '"manifest_sha256": None'),
    ("main_prov_prereg", '"prereg_sha256": a.prereg_sha,', '"prereg_sha256": None,'),
    ("main_out_dir", "        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)\n", "        pass\n"),
    ("main_output_written", "    write(result_record(LI, res, ctl, prov()))\n    print(\"SOUP READ\"", "    print(\"SOUP READ\""),
]

# A mutant that cannot change behaviour is NAMED, with the reason; the pass requires every other mutant killed.
EQUIVALENT = {
    "run_r2_seed": "R2-ADE6 compares only the paired POINT estimate `delta` (a mean of per-window differences) with the run json's; the bootstrap seed moves only lo / hi, so seed 0 or 1 cannot change the control's verdict.",
    "run_windows_n_unchecked": "`len(d['ws']) == pk.N` is implied by `np.array_equal(d['ws'], pk.t0)` whenever the pack's window count N is the length of its window array t0 (true of the stub Pack; the clause is DAG-ORC's own "
                               "line, copied; NOT proven here for the real PackReader, which this pass does not load): a dump with another window count already fails array_equal.",
}


def apply(src, old, new):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"mutant anchor occurs {n} times (need exactly 1): {old[:100]!r}")
    return src.replace(old, new, 1)


def pytest_run(reader_dir, tmp):
    env = dict(os.environ, SOUP_READ_DIR=reader_dir, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", tmp] + TESTS, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       timeout=900, encoding="utf-8", errors="replace")
    failed = [ln.split(" - ")[0].replace("FAILED ", "").replace("ERROR ", "").strip() for ln in p.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    return {"rc": p.returncode, "failed": failed, "seconds": round(time.time() - t0, 1), "tail": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--check-anchors", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated mutant names (re-run survivors)")
    a = ap.parse_args(argv)
    src = open(READER, encoding="utf-8", newline="").read()
    names = [n for n, _, _ in MUTANTS]
    if a.check_anchors:
        bad = [(n, src.count(o)) for n, o, nw in MUTANTS if src.count(o) != 1 or o == nw]
        dup = sorted({n for n in names if names.count(n) > 1})
        print(f"{len(MUTANTS)} mutants; bad anchors: {bad}; duplicate names: {dup}")
        return 0 if not bad and not dup else 1
    if not (a.work and a.out):
        raise SystemExit("--work and --out are required")
    os.makedirs(os.path.join(a.work, "pytmp"), exist_ok=True)
    res = {"reader": READER.replace("\\", "/"), "reader_sha256": hashlib.sha256(open(READER, "rb").read()).hexdigest(), "baseline": None, "mutants": {}, "equivalent_named": {}}
    base = os.path.join(a.work, "mutants", "_baseline")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "soup_read.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    res["baseline"] = pytest_run(base, os.path.join(a.work, "pytmp", "baseline"))
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
        with open(os.path.join(d, "soup_read.py"), "w", encoding="utf-8", newline="\n") as f:
            f.write(apply(src, old, new))
        r = pytest_run(d, os.path.join(a.work, "pytmp", name))
        r["killed"] = r["rc"] != 0
        res["mutants"][name] = r
        print(f"{'KILLED  ' if r['killed'] else 'SURVIVED'} {name}: {r['failed'][:1]} ({r['seconds']}s)", flush=True)
    res["n_mutants"], res["n_killed"] = len(res["mutants"]), sum(r["killed"] for r in res["mutants"].values())
    surv = sorted(n for n, r in res["mutants"].items() if not r["killed"])
    res["equivalent_named"] = {n: EQUIVALENT[n] for n in surv if n in EQUIVALENT}
    res["survivors"] = [n for n in surv if n not in EQUIVALENT]
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"ZZ-MUTATION-DONE {res['n_killed']}/{res['n_mutants']} killed; named equivalent={sorted(res['equivalent_named'])}; UNEXPLAINED survivors={res['survivors']}", flush=True)
    return 0 if not res["survivors"] else 1


if __name__ == "__main__":
    sys.exit(main())
