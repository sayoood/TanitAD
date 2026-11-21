#!/usr/bin/env python3
"""mutation_speedsub_reader.py -- the mutation pass over the decision function, the guards, the controls and the floor rule of ``speedsub_t1_read.py``.

Each mutant is the reader with ONE line changed; ``test_speedsub_t1_read.py`` is run against it (``SPEEDSUB_READER=<mutant>``) as a pytest subprocess; KILLED iff pytest exits non-zero. A SURVIVOR is a finding.
The unmutated reader must pass first.

    python mutation_speedsub_reader.py --reader speedsub_t1_read.py --tests test_speedsub_t1_read.py --work <scratch> --out mutation_reader.json [--prereg <PREREG file>]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

MUTANTS = [
    ("bar_is_f", "BAR = F_ADE6 / 2.0", "BAR = F_ADE6"),
    ("bar_two_f", "TWO_F = 0.2415", "TWO_F = 0.1207"),
    ("hit_exclusive_point", "float(row[\"delta\"]) <= -BAR", "float(row[\"delta\"]) < -BAR"),
    ("hit_ci_touches_zero", "float(row[\"hi\"]) < 0.0 and", "float(row[\"hi\"]) <= 0.0 and"),
    ("hit_ignores_ci", "return bool(computed(row) and float(row[\"hi\"]) < 0.0 and float(row[\"delta\"]) <= -BAR)", "return bool(computed(row) and float(row[\"delta\"]) <= -BAR)"),
    ("guard_runs_two", "GUARD_RUNS_MIN = 3", "GUARD_RUNS_MIN = 2"),
    ("guard_runs_four", "GUARD_RUNS_MIN = 3", "GUARD_RUNS_MIN = 4"),
    ("direction_ignored", "return d if DIRECTION[label] == \"lower\" else -d", "return d"),
    ("sep_worse_not_separated", "return bool(computed(row) and row[\"separated\"] and worsening(label, row) > 0)", "return bool(computed(row) and worsening(label, row) > 0)"),
    ("g_worse_any_run", "return any(len(v) >= GUARD_RUNS_MIN for v in per.values()), {l: v for l, v in per.items() if v}", "return any(len(v) >= 1 for v in per.values()), {l: v for l, v in per.items() if v}"),
    ("g_new_any_ctrl_seed", "        if all(e):\n            detail[l]", "        if any(e):\n            detail[l]"),
    ("g_new_identity_both", "            if not all(i):\n                fires.append(l)", "            if not any(i):\n                fires.append(l)"),
    ("g_new_reads_rep", "for r in CTRL_RUNS]\n        i = [sep_worse(l, rows_ip[r].get(l)) for r in CTRL_RUNS]", "for r in RUNS]\n        i = [sep_worse(l, rows_ip[r].get(l)) for r in CTRL_RUNS]"),
    ("g_ade_wrong_direction", "bad = [r for r in RUNS if sep_worse(\"ADE6\", ade_ei[r])]", "bad = [r for r in RUNS if sep_better(\"ADE6\", ade_ei[r])]"),
    ("g_ade_ctrl_only", "bad = [r for r in RUNS if sep_worse(\"ADE6\", ade_ei[r])]", "bad = [r for r in CTRL_RUNS if sep_worse(\"ADE6\", ade_ei[r])]"),
    ("g_viol_one_run", "return bool(AMENDMENT_1_GVIOL and len(bad) >= GUARD_RUNS_MIN), bad", "return bool(AMENDMENT_1_GVIOL and len(bad) >= 1), bad"),
    ("g_viol_switch_ignored", "return bool(AMENDMENT_1_GVIOL and len(bad) >= GUARD_RUNS_MIN), bad", "return bool(len(bad) >= GUARD_RUNS_MIN), bad"),
    ("g_viol_wrong_direction", "bad = [r for r in RUNS if sep_worse(VIOL, viol_ei.get(r))]", "bad = [r for r in RUNS if sep_better(VIOL, viol_ei.get(r))]"),
    ("decide_controls_ignored", "if not controls_ok or not runs_present:", "if not runs_present:"),
    ("decide_runs_ignored", "if not controls_ok or not runs_present:", "if not controls_ok:"),
    ("decide_pass_before_harms", "    if fired:\n        out = \"HARMS\"\n    elif n_hit == len(RUNS):\n        out = \"PASS\"", "    if n_hit == len(RUNS):\n        out = \"PASS\"\n    elif fired:\n        out = \"HARMS\""),
    ("decide_pass_three_hits", "elif n_hit == len(RUNS):\n        out = \"PASS\"", "elif n_hit >= 3:\n        out = \"PASS\""),
    ("decide_partial_any_ctrl", "elif ctrl_both:\n        out = \"PARTIAL\"", "elif any(hits[r] for r in CTRL_RUNS):\n        out = \"PARTIAL\""),
    ("decide_partial_reps", "ctrl_both = all(hits[r] for r in CTRL_RUNS)", "ctrl_both = all(hits[r] for r in RUNS[2:])"),
    ("decide_null_labels_swapped", "label = \"NULL-FLAT\" if n_hit == 0 else \"NULL-ONE\"", "label = \"NULL-ONE\" if n_hit == 0 else \"NULL-FLAT\""),
    ("read_a_d1_bar", "float(rows[\"ADE6\"][\"delta\"]) <= -TWO_F", "float(rows[\"ADE6\"][\"delta\"]) <= -BAR"),
    ("read_a_d2_ignored", "d2 = all(per[r][\"D2\"] for r in CTRL_RUNS)", "d2 = True"),
    ("read_a_d3_any", "d3 = all(per[r][\"D3\"] for r in CTRL_RUNS)", "d3 = any(per[r][\"D3\"] for r in CTRL_RUNS)"),
    ("read_a_crawl_ignored", "crawl_pass = not any(per[r][\"M1_sep_worse\"] or per[r][\"L2_sep_worse\"] for r in CTRL_RUNS)", "crawl_pass = True"),
    ("read_a_one_seed", "d1 = all(per[r][\"D1\"] for r in CTRL_RUNS)", "d1 = any(per[r][\"D1\"] for r in CTRL_RUNS)"),
    ("floor_half_to_one", "g_c[r] <= 0.5 * g_id[r]", "g_c[r] <= 1.0 * g_id[r]"),
    ("floor_positive_gid", "g_id[r] > 0.0 and", "g_id[r] >= 0.0 and"),
    ("floor_any_row", "\"verdict\": \"COLLAPSE\" if all(per.values()) else \"NOT COLLAPSE\"", "\"verdict\": \"COLLAPSE\" if any(per.values()) else \"NOT COLLAPSE\""),
    ("floor_rows", "FLOOR_ROWS = (\"ADE6\", \"yaw-rate MAE\", \"curvature MAE\")", "FLOOR_ROWS = (\"ADE6\", \"yaw-rate MAE\")"),
    ("k_zero_delta_unchecked", "r[\"delta\"] == 0.0 and r[\"lo\"] == 0.0 and r[\"hi\"] == 0.0", "r[\"lo\"] == 0.0 and r[\"hi\"] == 0.0"),
    ("k_zero_empty_ok", "return not bad and any(computed(r) for r in rows.values()), bad", "return not bad, bad"),
    ("k_reg_any", "return all(sep_worse(\"ADE6\", ade_si[r]) for r in RUNS)", "return any(sep_worse(\"ADE6\", ade_si[r]) for r in RUNS)"),
    ("k_id_tolerance", "TOL_K_ID, TOL_K_REC = 1e-4, 1e-5", "TOL_K_ID, TOL_K_REC = 1e-2, 1e-5"),
    ("k_id_max_to_min", "return bool(d.max() <= tol and len(d) == len(cl_banked))", "return bool(d.min() <= tol and len(d) == len(cl_banked))"),
    ("ade6_max_not_mean", "axis=-1).mean(1)", "axis=-1).max(1)"),
    ("k_rec_tolerance", "TOL_K_ID, TOL_K_REC = 1e-4, 1e-5", "TOL_K_ID, TOL_K_REC = 1e-4, 1e-1"),
    ("k_rec_used_only", "return bool(max(d_used, d_o) <= tol)", "return bool(d_used <= tol)"),
    ("k_reach_egoc_min", "REACH_EGOC_MIN, REACH_SHUF_MIN = 0.99, 0.95", "REACH_EGOC_MIN, REACH_SHUF_MIN = 0.5, 0.95"),
    ("k_reach_shuf_min", "REACH_EGOC_MIN, REACH_SHUF_MIN = 0.99, 0.95", "REACH_EGOC_MIN, REACH_SHUF_MIN = 0.99, 0.5"),
    ("k_reach_reads_own", "f2 = moved_fraction(egoshufc[\"o_std\"], egoc[\"o_std\"])", "f2 = moved_fraction(egoshufc[\"o_std\"], egoshufc[\"o_std\"])"),
    ("k_shuf_fixed_points", "return bool(nfix <= max(3, 0.01 * n)), nfix", "return True, nfix"),
    ("k_shuf_fixed_points_pct", "return bool(nfix <= max(3, 0.01 * n)), nfix", "return bool(nfix <= max(3, 0.1 * n)), nfix"),
    ("k_shuf_rows_unchecked", "and m.get(\"perm_n\") == SHUF_N and fix_ok and rows_ok)", "and m.get(\"perm_n\") == SHUF_N and fix_ok)"),
    ("k_shuf_seed", "SHUF_SEED, SHUF_N = 20261029, 2399", "SHUF_SEED, SHUF_N = 20261030, 2399"),
    ("k_nonspeed_tolerant", "all(v == 0.0 for v in mx.values())", "all(v <= 1e-3 for v in mx.values())"),
    ("k_floor_allclose", "a.tobytes() == b.tobytes()", "np.allclose(a, b)"),
    ("k_sha_prefix_unchecked", "if not str(meta.get(\"ckpt_sha256\", \"\")).startswith(CKPT_PREFIX[run]):", "if False:"),
    ("k_sha_logged_readout", "        want[\"readout_sha256\"] = pins[\"readout_banked\"][\"sha256\"]", "        want[\"readout_sha256\"] = pins[\"readout_c\"][\"sha256\"]"),
    ("k_sha_tool_unchecked", "\"tool_sha256\": pins[\"tool_sha256\"], ", ""),
    ("moved_tolerance", "def moved_fraction(a, b, tol=1e-6):", "def moved_fraction(a, b, tol=10.0):"),
    ("k_id_shape_guard", "if np.shape(cl_run) != np.shape(cl_banked) or np.shape(g)[0] != np.shape(cl_banked)[0]:", "if False:"),
    ("light_nonspeed_dropped", "add(\"K-NONSPEED\", *k_nonspeed({f\"{run}:{m}\": r for m, r in recs.items()}))", "add(\"K-NONSPEED\", True, None)"),
    ("light_sha_dropped", "add(f\"K-SHA:{m}\", *k_sha(r[\"meta\"], run, pins))", "add(f\"K-SHA:{m}\", True, None)"),
    ("light_align_rc_dropped", "r[\"meta\"][\"aligned\"] and r[\"meta\"][\"tool_rc\"] == 0, r[\"meta\"][\"alignment_with_tool_ticks\"])", "r[\"meta\"][\"aligned\"], r[\"meta\"][\"alignment_with_tool_ticks\"])"),
    ("light_rec_dropped", "add(\"K-REC\", *k_rec(recs[\"egoc\"], lambda ego: ro.predict_raw(None, ego)))", "add(\"K-REC\", True, None)"),
    ("light_reach_dropped", "add(\"K-REACH\", *k_reach(recs[\"egoc\"], recs[\"egoshufc\"]))", "add(\"K-REACH\", True, None)"),
    ("light_reach_smoke_dropped", "add(\"K-REACH\", f >= REACH_EGOC_MIN, ", "add(\"K-REACH\", True, "),
    ("light_shuf_dropped", "add(\"K-SHUF\", *k_shuf(recs[\"egoshufc\"]))", "add(\"K-SHUF\", True, None)"),
    ("light_floor_dropped", "add(f\"K-FLOOR:{m}\", *k_floor(d[\"fl\"], banked[\"fl\"]))", "add(f\"K-FLOOR:{m}\", True, None)"),
    ("light_win_gt_unchecked", "for k in (\"clip_index\", \"ws\", \"eid\", \"g\", \"v0\")), \"windows and GT bit-equal", "for k in (\"clip_index\", \"ws\", \"eid\", \"v0\")), \"windows and GT bit-equal"),
    ("light_id_dropped", "add(\"K-ID\", *k_id(dumps[\"identity\"][\"cl\"], banked[\"cl\"], banked[\"g\"]))", "add(\"K-ID\", True, None)"),
    ("light_preflight_skips_banked", "need_banked=(kind == \"preflight\"))", "need_banked=False)"),
    ("light_stop_rc", "json.dumps({k: v[\"ok\"] for k, v in ctl.items()}), flush=True)\n    return 0 if ok else 1", "json.dumps({k: v[\"ok\"] for k, v in ctl.items()}), flush=True)\n    return 0"),
    ("light_missing_rc", "STOP missing {p}\")\n            return 2", "STOP missing {p}\")\n            return 0"),
    ("light_smoke_reads_shuf", "modes = (\"identity\", \"egoc\", \"egoshufc\") if kind == \"preflight\" else (\"identity\", \"egoc\")", "modes = (\"identity\", \"egoc\", \"egoshufc\")"),
    ("light_smoke_root", "os.path.join(a.out_root, \"\" if kind == \"preflight\" else \"SMOKE\", a.run)", "os.path.join(a.out_root, \"\", a.run)"),
    ("manifest_beside_on_rep", "for m in MODES + (BESIDE_MODES if fam == \"CTRL\" else ()):", "for m in MODES + BESIDE_MODES:"),
    ("manifest_tag", "tag = f\"SST1_{r}_{m}\"", "tag = f\"SST_{r}_{m}\""),
    ("manifest_rec_path", "\"rec\": f\"{d}/ticks/speedsub_{tag}.npz\"", "\"rec\": f\"{d}/speedsub_{tag}.npz\""),
    ("manifest_ticks_path", "\"ticks\": f\"{d}/ticks/ticks_{tag}.npz\"", "\"ticks\": f\"{d}/ticks_{tag}.npz\""),
    ("beside_swapped", "\"egol\", \"identity_l\", {s: dumps[s][\"identity_l\"] for s in SEEDS})", "\"identity_l\", \"egol\", {s: dumps[s][\"identity_l\"] for s in SEEDS})"),
    ("beside_ref_wrong", "{s: dumps[s][\"identity_l\"] for s in SEEDS})", "{s: dumps[s][\"egol\"] for s in SEEDS})"),
    ("beside_hidden_rows_kept", "if not str(l).startswith(\"_\")} for s in SEEDS}}", "} for s in SEEDS}}"),
    ("main_viol_ol_row", "viol_ei = {r: tabs[\"egoc-identity\"][r][VIOL] for r in RUNS}", "viol_ei = {r: tabs[\"egoc-identity\"][r][VIOL_OL] for r in RUNS}"),
    ("main_ade_from_p10", "ade_ei = {r: tabs[\"egoc-identity\"][r][\"ADE6\"] for r in RUNS}", "ade_ei = {r: tabs[\"egoc-P10\"][r][\"ADE6\"] for r in RUNS}"),
    ("main_gnew_swapped", "g_new(tabs[\"egoc-P10\"], tabs[\"identity-P10\"])", "g_new(tabs[\"identity-P10\"], tabs[\"egoc-P10\"])"),
    ("main_gworse_p10", "g_worse(tabs[\"egoc-identity\"])", "g_worse(tabs[\"egoc-P10\"])"),
    ("main_reada_table", "read_a(tabs[\"egoc-P10\"])", "read_a(tabs[\"egoc-identity\"])"),
    ("main_reada_always", "if dec[\"OUTCOME\"] == \"PASS\":", "if True:"),
    ("main_kreg_table", "tabs[\"egoshufc-identity\"][r].get(\"ADE6\")", "tabs[\"egoc-identity\"][r].get(\"ADE6\")"),
    ("main_kzero_table", "is_exact_zero(tabs[\"identity-identity\"][r])", "is_exact_zero(tabs[\"identity-P10\"][r])"),
    ("main_kzero_dropped", "add(f\"K-ZERO:{r}\", ok, det)", "add(f\"K-ZERO:{r}\", True, det)"),
    ("main_kreg_dropped", "add(\"K-REG\", *k_reg({r: tabs[\"egoshufc-identity\"][r].get(\"ADE6\") for r in RUNS}))", "add(\"K-REG\", True, None)"),
    ("main_controls_forced", "controls_ok = all(ctl.values())", "controls_ok = True"),
    ("main_gviol_dropped", "guards = {\"G-WORSE\": gw, \"G-NEW\": gn, \"G-ADE\": ga, \"G-VIOL\": gv}", "guards = {\"G-WORSE\": gw, \"G-NEW\": gn, \"G-ADE\": ga}"),
    ("main_rc_refused", "return 2 if res[\"OUTCOME\"] == \"REFUSED\" else 0", "return 0"),
    ("main_floor_targets_swapped", "((\"REP_id\", \"CTRL_id\"), g_id), ((\"REP_c\", \"CTRL_c\"), g_c)", "((\"REP_id\", \"CTRL_id\"), g_c), ((\"REP_c\", \"CTRL_c\"), g_id)"),
    ("main_floor_s0_s1_swapped", "\"s0 (decisive, beside)\": cross[\"s0\"], \"s1 (reported)\": cross[\"s1\"]", "\"s0 (decisive, beside)\": cross[\"s1\"], \"s1 (reported)\": cross[\"s0\"]"),
    ("main_missing_rc", "        return 4\n", "        return 0\n"),
    ("main_prereg_rc", "file=sys.stderr)\n        return 2", "file=sys.stderr)\n        return 0"),
    ("main_prereg_optional", "bad = prereg_problems(a.prereg, a.prereg_sha) if a.prereg else [\"--prereg is required\"]\n    pins = ", "bad = prereg_problems(a.prereg, a.prereg_sha) if a.prereg else []\n    pins = "),
    ("main_kreach_plan_dropped", "add(f\"K-REACH-PLAN:{r}\", d_plan >= REACH_EGOC_MIN,", "add(f\"K-REACH-PLAN:{r}\", True,"),
    ("main_kreach_dropped", "add(f\"K-REACH:{r}\", *k_reach(recs[r][\"egoc\"], recs[r][\"egoshufc\"]))", "add(f\"K-REACH:{r}\", True, None)"),
    ("main_kshuf_dropped", "add(f\"K-SHUF:{r}\", *k_shuf(recs[r][\"egoshufc\"]))", "add(f\"K-SHUF:{r}\", True, None)"),
    ("main_kfloor_dropped", "add(f\"K-FLOOR:{r}:{mode}\", *k_floor(d[\"fl\"], ref[\"fl\"]))", "add(f\"K-FLOOR:{r}:{mode}\", True, None)"),
    ("main_kid_dropped", "add(f\"K-ID:{r}\", *k_id(dumps[r][\"identity\"][\"cl\"], dumps[r][\"banked\"][\"cl\"], dumps[r][\"banked\"][\"g\"]))", "add(f\"K-ID:{r}\", True, None)"),
    ("main_kid_against_egoc", "k_id(dumps[r][\"identity\"][\"cl\"], dumps[r][\"banked\"][\"cl\"], dumps[r][\"banked\"][\"g\"])", "k_id(dumps[r][\"egoc\"][\"cl\"], dumps[r][\"banked\"][\"cl\"], dumps[r][\"banked\"][\"g\"])"),
    ("main_knonspeed_ctrl_only", "for r in RUNS for m in MODES}))", "for r in CTRL_RUNS for m in MODES}))"),
    ("main_kwin_dropped", "add(f\"K-WIN:{r}:{mode}\", same,", "add(f\"K-WIN:{r}:{mode}\", True,"),
    ("main_ksha_dropped", "add(f\"K-SHA:{r}:{mode}\", ok, det)", "add(f\"K-SHA:{r}:{mode}\", True, det)"),
    ("main_kalign_rc_dropped", "recs[r][mode][\"meta\"][\"aligned\"] and recs[r][mode][\"meta\"][\"tool_rc\"] == 0", "recs[r][mode][\"meta\"][\"aligned\"]"),
    ("main_heads_controls_dropped", "add(f\"{k}:{r}:{mode}\", okp, detp)", "add(f\"{k}:{r}:{mode}\", True, detp)"),
    ("main_krec_dropped", "add(f\"K-REC:{r}\", *k_rec(recs[r][\"egoc\"], lambda ego, ro=ro: ro.predict_raw(None, ego)))", "add(f\"K-REC:{r}\", True, None)"),
    ("main_rep_family_is_ctrl", "fam = {\"CTRL\": CTRL_RUNS, \"REP\": (\"REP_s0\", \"REP_s1\")}", "fam = {\"CTRL\": CTRL_RUNS, \"REP\": CTRL_RUNS}"),
    ("main_runs_present_false", "dec = decide_speedsub(controls_ok, True, hits, guards)", "dec = decide_speedsub(controls_ok, False, hits, guards)"),
]


def apply(src, old, new):
    if src.count(old) != 1:
        raise SystemExit(f"mutant anchor occurs {src.count(old)} times (need exactly 1): {old[:90]!r}")
    return src.replace(old, new, 1)


def pytest_run(tests, reader, env_extra, tmp):
    env = dict(os.environ, SPEEDSUB_READER=reader, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", **env_extra)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", tmp, tests], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)
    failed = [ln.split(" - ")[0].replace("FAILED ", "").replace("ERROR ", "").strip() for ln in p.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    return {"rc": p.returncode, "failed": failed, "seconds": round(time.time() - t0, 1), "tail": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prereg", default="")
    a = ap.parse_args(argv)
    src = open(a.reader, encoding="utf-8").read()
    extra = {"SPEEDSUB_PREREG": a.prereg} if a.prereg else {}
    os.makedirs(os.path.join(a.work, "pytmp"), exist_ok=True)
    res = {"baseline": pytest_run(a.tests, a.reader, extra, os.path.join(a.work, "pytmp", "baseline")), "mutants": {}}
    print("baseline:", res["baseline"], flush=True)
    if res["baseline"]["rc"] != 0:
        json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
        print("ZZ-MUTATION-STOP baseline does not pass")
        return 2
    for name, old, new in MUTANTS:
        d = os.path.join(a.work, "mutants", name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "speedsub_t1_read.py")
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
