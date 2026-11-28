"""run_mutations_tools.py -- mutation runner of the three tools of card ADD83 that have a --selftest: make_surrogate.py, check_shufclose.py, read_shufclose10k.py.
Each mutation changes ONE fragment of one tool in a COPY inside tmp_mut2/ (the real file is never touched); the --selftest of the copy has to FAIL (non-zero return code). The unmutated selftests (control) have
to pass before and after. Sequential, one process at a time, CPU only.
  bash ../ops/pycap.sh run_mutations_tools.py [--out MUTATIONS_TOOLS.json]
Writes MUTATIONS_TOOLS.json (the list, the result of each run, the control runs). Exit 0 = every mutation red and the control runs green."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUT = HERE / "tmp_mut2"
V = HERE.parent

# name -> (tool file, what it breaks, exact text of ONE fragment, its replacement)
MUTATIONS = {
    "t01_map_neighbours_not_exchanged": ("make_surrogate.py", "neighbours map to themselves", "pos[order[a]], pos[order[a + 1]] = order[a + 1], order[a]", "pos[order[a]], pos[order[a + 1]] = order[a], order[a + 1]"),
    "t02_ties_by_descending_clip_index": ("make_surrogate.py", "ties are broken by the larger clip index", "order = np.lexsort((clip_ids, np.asarray(share, np.float64)))", "order = np.lexsort((-clip_ids, np.asarray(share, np.float64)))"),
    "t03_cycle_has_a_fixed_point": ("make_surrogate.py", "the last three do not form a cycle", "pos[a], pos[b], pos[c] = b, c, a", "pos[a], pos[b], pos[c] = a, c, b"),
    "t04_share_threshold_strict": ("make_surrogate.py", "p equal to 0.5 does not count", 'counts = {int(c): int(((ci == c) & (p >= 0.5)).sum()) for c in ids}', 'counts = {int(c): int(((ci == c) & (p > 0.5)).sum()) for c in ids}'),
    "t05_surrogate_frame_shifted": ("make_surrogate.py", "the surrogate takes the frame after", "tgt = np.array([img[int(c)] for c in est_ci], np.int64) * STRIDE + est_fr", "tgt = np.array([img[int(c)] for c in est_ci], np.int64) * STRIDE + est_fr + 1"),
    "t06_u1_values_not_judged": ("make_surrogate.py", "u1 does not judge the values", 'd["rows_equal_estimate_rows"] and bad == 0', 'd["rows_equal_estimate_rows"]'),
    "t07_u4_band_not_judged": ("make_surrogate.py", "u4 passes whatever r is", "ok = bool(np.isfinite(r_s) and -R_BAND <= r_s <= R_BAND)", "ok = True"),
    "t08_identity_table_is_the_surrogate": ("make_surrogate.py", "the identity table holds the surrogate", 'idt = {"clip_index": ec.copy(), "frame": ef.copy(), "est": ev.copy()}', 'idt = {"clip_index": ec.copy(), "frame": ef.copy(), "est": sv.copy()}'),
    "t09_const_table_not_2": ("make_surrogate.py", "the planted table is 1.0", 'est=z0 + 2.0)', 'est=z0 + 1.0)'),
    "t10_existing_out_folder_accepted": ("make_surrogate.py", "an existing out folder is used", '    if os.path.exists(out_dir):\n        raise SystemExit("REFUSED: %s exists" % out_dir)\n    CF = load_fit(fit_dir)\n    os.makedirs(out_dir)\n', '    CF = load_fit(fit_dir)\n    os.makedirs(out_dir, exist_ok=True)\n'),
    "c01_u0a_closing_not_judged": ("check_shufclose.py", "u0a ignores the closing column", 'c["c_src_bad"] == 0 and c["est_ok_mismatch"] == 0', 'c["est_ok_mismatch"] == 0'),
    "c02_u0a_est_ok_not_judged": ("check_shufclose.py", "u0a ignores the rows marked full", 'c["c_src_bad"] == 0 and c["est_ok_mismatch"] == 0', 'c["c_src_bad"] == 0'),
    "c03_u0a_plans_not_judged": ("check_shufclose.py", "u0a ignores the first plans", 'and bool(pc["plan_largest_diff"] <= tol)', ''),
    "c04_u0b_tolerance": ("check_shufclose.py", "u0b allows a difference of 1e-3", "ok = worst == 0.0 and not prob", "ok = worst <= 1e-3 and not prob"),
    "c05_u0c_short_looks_not_judged": ("check_shufclose.py", "u0c ignores the looks without a full history", "ok = ok and a2 and a0 and avl and ae", "ok = ok and a2 and avl and ae"),
    "c06_line_hash_not_judged": ("check_shufclose.py", "the file hash of the run line is not compared", "and m_.group(5) == table_sha)", ")"),
    "c07_refusal_return_code": ("check_shufclose.py", "a return code of 0 passes the refusal check", "return bool((rc is None or rc != 0) and", "return bool(True and"),
    "c08_refusal_names": ("check_shufclose.py", "the flag names are not required in the refusal text", 'and all(n in text for n in names))', ')'),
    "r01_sd_upper_bound": ("read_shufclose10k.py", "the u3 sd ratio may reach 2.0", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.15, 0.90", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 2.0, 0.15, 0.90"),
    "r02_sd_lower_bound": ("read_shufclose10k.py", "the u3 sd ratio may go down to 0.5", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.15, 0.90", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.50, 1.25, 0.15, 0.90"),
    "r03_mean_bound": ("read_shufclose10k.py", "the u3 mean may reach 0.5", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.15, 0.90", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.50, 0.90"),
    "r04_share_bound": ("read_shufclose10k.py", "the u3 share may be 0.80", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.15, 0.90", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.15, 0.80"),
    "r05_gate_without_mean": ("read_shufclose10k.py", "the u3 gate does not judge the mean", "gate[e] = bool(c1 and c2 and c3)", "gate[e] = bool(c1 and c3)"),
    "r06_c_word_includes_not_separated": ("read_shufclose10k.py", "NOT-SEPARATED counts as LOWER in the verdict", '"AGE C LOWER", None if w["C"] is None else w["C"] == "LOWER")', '"AGE C LOWER", None if w["C"] is None else w["C"] in ("LOWER", "NOT-SEPARATED"))'),
    "r07_u_not_judged": ("read_shufclose10k.py", "U HIGHER does not stop the verdict", '("AGE U not HIGHER", nh(w["U"]))]', '("AGE U not HIGHER", True)]'),
    "r08_beside_name_of_another_pair": ("read_shufclose10k.py", "the beside name is that of SHUF-VS-NO-RULE", 'sr = words["AGE"]["SHUF-VS-RR-" + e]["C"]', 'sr = words["AGE"]["SHUF-VS-NO-RULE-" + e]["C"]'),
    "r09_marks_swapped": ("read_shufclose10k.py", "the N10 mark sits on L25", 'MARK_N10 if e == "N10" else MARK_L25', 'MARK_L25 if e == "N10" else MARK_N10'),
    "r10_age_gate_ignored": ("read_shufclose10k.py", "the verdict is written without the u3 gate", "        if not age_ok.get(e):\n            why[key]", "        if False:\n            why[key]"),
    "r11_u2_tolerance": ("read_shufclose10k.py", "u2 allows 1e-3", "TOL_U2 = 1e-9", "TOL_U2 = 1e-3"),
    "r12_u2_pattern_not_judged": ("read_shufclose10k.py", "u2 does not compare est_ok with the table", "pattern_bad = int(((found_all & np.isfinite(tv_all)) != eo).sum())", "pattern_bad = 0"),
    "r13_u2_first_look_not_required": ("read_shufclose10k.py", "u2 takes tick 0 as the first look of the mid-look log", 'first = 0 if tag == "rule" else (int(kk.min()) if len(kk) else 0)', "first = 0"),
    "r14_sur_hashes_not_compared": ("read_shufclose10k.py", "the surrogate tables are not compared with the record", "bad = [n for n, h in shas.items() if files.get(n) != h]", "bad = []"),
    "r15_sur_u4_not_judged": ("read_shufclose10k.py", "u4 of the record is not judged", 'u4_ok = bool(all(e in u4 and u4[e].get("passed") is True for e in ESTS))', "u4_ok = True"),
    "r16_est_line_hash_not_judged": ("read_shufclose10k.py", "the est line file hash is not compared with the table", 'sha_ok = bool(pe is not None and pe["file_sha256"] == shas.get((f[len("RRe_S"):], s)))', "sha_ok = True"),
    "r17_estclose_card_not_judged": ("read_shufclose10k.py", "any read is taken as that of card ADD79", 'if ej.get("card") != PARENT_CARD:', "if False:"),
    "r18_pair_surrogate_vs_other": ("read_shufclose10k.py", "SHUF-VS-RR is read against no rule", '("SHUF-VS-RR", "RRe_S" + _e, "RR")', '("SHUF-VS-RR", "RRe_S" + _e, "R0K")'),
    "r19_f_names_dropped": ("read_shufclose10k.py", "no F name is written", 'for kind in ("C", "U", "F"):', 'for kind in ("C", "U"):'),
    "r20_x1_fallback_not_judged": ("read_shufclose10k.py", "x1 does not judge the closing 0.0 without a full history", 'and r.get("fallback_closing_exactly_zero") is True', ""),
}


def run_selftest(tool_path, tag, timeout=900):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", SC_ESTREAD_DIR=str(V / "estclose_read"), CLOSEREAD_DIR=str(V / "closeread"), ECR_INTEG_DIR=str(V / "integ10k"))
    t0 = time.time()
    r = subprocess.run([sys.executable, str(tool_path), "--selftest"], env=env, capture_output=True, text=True, encoding="utf-8", timeout=timeout, cwd=str(tool_path.parent))
    out = r.stdout + r.stderr
    return {"rc": r.returncode, "seconds": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-2:]}


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(HERE / "MUTATIONS_TOOLS.json"))
    a = ap.parse_args(argv)
    stamp = time.strftime("%H%M%S")
    names = [n for n in MUTATIONS if not a.only or n.split("_")[0] in a.only.split(",") or n in a.only.split(",")]
    res = {"mutations": {}, "controls_before": {}, "controls_after": {}}
    tools = sorted({MUTATIONS[n][0] for n in names})
    for t in tools:
        res["controls_before"][t] = run_selftest(HERE / t, "before")
        print("control before", t, res["controls_before"][t]["rc"], flush=True)
    for n in names:
        tool, what, old, new = MUTATIONS[n]
        base = (HERE / tool).read_text(encoding="utf-8")
        if base.count(old) != 1:
            raise SystemExit("mutation %s: the fragment occurs %d times, expected exactly 1" % (n, base.count(old)))
        d = MUT / (n + "_" + stamp)
        d.mkdir(parents=True, exist_ok=True)
        f = d / tool
        f.write_text(base.replace(old, new), encoding="utf-8", newline="\n")
        r = run_selftest(f, n)
        r["what"], r["old"], r["new"], r["tool"] = what, old, new, tool
        r["red"] = r["rc"] != 0
        res["mutations"][n] = r
        print("%-40s %s" % (n, "RED" if r["red"] else "GREEN (not caught)"), flush=True)
    for t in tools:
        res["controls_after"][t] = run_selftest(HERE / t, "after")
        print("control after", t, res["controls_after"][t]["rc"], flush=True)
    ok = all(v["rc"] == 0 for v in res["controls_before"].values()) and all(v["rc"] == 0 for v in res["controls_after"].values()) and all(v["red"] for v in res["mutations"].values())
    res["red"], res["total"] = sum(1 for v in res["mutations"].values() if v["red"]), len(res["mutations"])
    res["all_red_and_controls_green"] = ok
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print("ZZ MUTATIONS-TOOLS", "PASS" if ok else "FAIL", "%d of %d red" % (res["red"], res["total"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
