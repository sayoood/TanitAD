"""run_mutations_cc.py -- mutation runner of the tool kit of card ADD80 (CONSTCLOSE10K): make_const_files.py, check_constclose.py, read_constclose10k.py (suite "selftest": the copy's --selftest must return a
non-zero code) and make_constclose_queue.py (suite "queue": test_constclose_queue.py with CC_TOOL_DIR at the copy must go RED). Each mutation is applied to a COPY of the tool inside tmp_mut/<tag>/<name>/ (the
real files are never touched). The unmutated tools (control) must be GREEN before and after. Sequential, CPU only. Nothing is deleted or moved: the copies stay.

  PYCAP_MB=3000 bash ops/pycap.sh run_mutations_cc.py --out MUTATIONS_CC.json [--only a,b] [--tag NAME]
Exit 0 = every mutation red and both control runs green.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUT, PT = HERE / "tmp_mut", HERE / "tmp_pt"
ROOT = HERE.parent

# name: (tool file, old fragment, new fragment, what)
TABLE = {
    # ---- read_constclose10k.py
    "r01_k2_tolerance_wide": ("read_constclose10k.py", "TOL_K2 = 1e-6", "TOL_K2 = 1e-1", "k2 accepts 0.1 m/s"),
    "r02_coverage_bar_low": ("read_constclose10k.py", "COV_BAR = 0.99", "COV_BAR = 0.40", "the coverage bar is 0.40"),
    "r03_age_c_not_lower": ("read_constclose10k.py", '("AGE C LOWER", None if a["C"] is None else a["C"] == "LOWER")', '("AGE C LOWER", None if a["C"] is None else a["C"] in NOT_HIGHER)', "the verdict accepts an AGE C that is not LOWER"),
    "r04_fresh_c_unchecked": ("read_constclose10k.py", '("FRESH C not HIGHER", nh(f["C"]))', '("FRESH C not HIGHER", True)', "the verdict ignores FRESH C"),
    "r05_age_u_unchecked": ("read_constclose10k.py", '("AGE U not HIGHER", nh(a["U"]))', '("AGE U not HIGHER", True)', "the verdict ignores AGE U"),
    "r06_fresh_family_unchecked": ("read_constclose10k.py", '("FRESH no family WORSE", no_worse(f["fam"]))', '("FRESH no family WORSE", True)', "the verdict ignores a WORSE family at FRESH"),
    "r07_one_of_three_always": ("read_constclose10k.py", "if len(supported) == 1:", "if len(supported) >= 1:", "the one-of-three mark is written when two or three are SUPPORTED"),
    "r08_f_beside_dropped": ("read_constclose10k.py", "            if fw is not None:\n", "            if False:\n", "the F name is not written beside the verdict"),
    "r09_match_ignores_f": ("read_constclose10k.py", "None if a is None or b is None else bool(a and b)", "None if a is None or b is None else bool(a)", "a value matches E whatever F reads"),
    "r10_not_beyond_needs_all": ("read_constclose10k.py", "if any(v is True for v in m.values()):", "if all(v is True for v in m.values()):", "NOT-BEYOND needs all three values to match"),
    "r11_not_beyond_name_swapped": ("read_constclose10k.py", 'names[key] = "%s-%s-NOT-BEYOND-A-CONSTANT" % (PRE, e)', 'names[key] = "%s-%s-BEYOND-A-CONSTANT" % (PRE, e)', "a match is named BEYOND"),
    "r12_not_separated_not_a_match": ("read_constclose10k.py", "return None if word is None else bool(word in NOT_HIGHER)", 'return None if word is None else bool(word == "LOWER")', "NOT-SEPARATED does not count as not HIGHER"),
    "r13_e_age_gate_ignored": ("read_constclose10k.py", '        if not gate.get("AGE"):', "        if False:", "E verdicts are written below the coverage bar at AGE"),
    "r14_verdict_gate_ignored": ("read_constclose10k.py", "        if notnamed:", "        if False:", "a verdict is written with a setting below the coverage bar"),
    "r15_names_gate_ignored": ("read_constclose10k.py", '        if not gate.get(st):\n            why["COVERAGE_" + st]', '        if False:\n            why["COVERAGE_" + st]', "names are written below the coverage bar"),
    "r16_gate_text_share_dropped": ("read_constclose10k.py", "sl >= COV_BAR and stx >= COV_BAR)", "sl >= COV_BAR)", "the gate ignores the run-text share"),
    "r17_k0_hash_unchecked": ("read_constclose10k.py", 'all(t is not None and t["file_sha256"] == sh for t in lines.values())', 'all(t is not None for t in lines.values())', "k0 does not compare the file hash of the run line"),
    "r18_k0_value_unchecked": ("read_constclose10k.py", "and all(v == CVAL[c] for v in vals) and all(t is not None", "and all(t is not None", "k0 does not check the returned value"),
    "r19_k2_value_unchecked": ("read_constclose10k.py", "d = np.abs(ce[eo] - CVAL[c])", "d = np.abs(ce[eo] - ce[eo])", "k2 compares the estimate with itself"),
    "r20_k2_nan_unchecked": ("read_constclose10k.py", "nan_ok = bool(np.all(np.isnan(ce[~eo])))", "nan_ok = True", "k2 does not check NaN without a full history"),
    "r21_k2_first_tick_not_required": ("read_constclose10k.py", "and not bad.any() and n_first > 0 and nan_ok)", "and not bad.any() and nan_ok)", "k2 does not need a compared first-tick row"),
    "r22_x1_fallback_unchecked": ("read_constclose10k.py", " and np.all(cs[short] == 0.0))", ")", "x1 does not check the 0.0 fallback"),
    "r23_x1_rr_unchecked": ("read_constclose10k.py", 'g1 = bool(r and r["rows"] > 0 and r["present_nonzero"] == 0 and r["max_abs_closing"] == 0.0 and r["closing_finite_where_present"])', "g1 = bool(r)", "x1 does not check that RR feeds 0"),
    "r24_m1_counts_unchecked": ("read_constclose10k.py", "good = bool(n and base and n == base)", "good = bool(n and base)", "m1 does not compare the row counts"),
    "r25_r1_dump_count_unchecked": ("read_constclose10k.py", "good = EC.run_text_ok(p, preroll, wrapper) and n_files.get((f, s)) == n_clips", "good = EC.run_text_ok(p, preroll, wrapper)", "r1 does not count the dump files"),
    "r26_failed_control_keeps_payload": ("read_constclose10k.py", '    if res["controls_all_pass"]:\n        res.update(payload)', "    if payload:\n        res.update(payload)", "a failed control still carries names and values"),
    "r27_estclose_card_unchecked": ("read_constclose10k.py", '    if ej.get("card") != PARENT_CARD:', "    if False:", "a file of another card is admitted"),
    "r28_estclose_not_all_pass_admitted": ("read_constclose10k.py", 'if ej.get("controls_all_pass") is not True:', 'if ej.get("controls_all_pass") is None:', "a read of card ADD79 with a failed control is admitted"),
    "r29_pair_order_swapped": ("read_constclose10k.py", 'p["CONST-VS-RR-" + c] = ("RRe_" + c, "RR")', 'p["CONST-VS-RR-" + c] = ("RR", "RRe_" + c)', "the pair is RR minus RRe instead of RRe minus RR"),
    "r30_name_order_changed": ("read_constclose10k.py", 'return "%s-%s-%s-%s-%s" % (PRE, pair, setting, kind, word)', 'return "%s-%s-%s-%s-%s" % (PRE, pair, kind, setting, word)', "setting and kind swapped in a name"),
    "r31_run_names_changed": ("read_constclose10k.py", '"RQ50e%sKF5%s" % (c, suf) for c in VALUES}', '"RQ50e%sKF10%s" % (c, suf) for c in VALUES}', "the run names of the constant runs are wrong"),
    "r32_value_changed": ("read_constclose10k.py", 'CVAL = {"C10": 1.0, "C20": 2.0, "C40": 4.0}', 'CVAL = {"C10": 1.0, "C20": 2.5, "C40": 4.0}', "the value of C20 is 2.5"),
    "r33_form_of_changed": ("read_constclose10k.py", 'FORM_OF.update({"RRe_" + c: "RRe" for c in VALUES})', 'FORM_OF.update({"RRe_" + c: "RRh" for c in VALUES})', "the constant runs are registered as form RRh"),
    # ---- check_constclose.py
    "c01_closing_tolerance": ("check_constclose.py", "np.all(cs[full] == value)", "np.allclose(cs[full], value, atol=0.2)", "K1a accepts 1.9"),
    "c02_c_est_nan_unchecked": ("check_constclose.py", "np.all(ce[eo] == value) and np.all(np.isnan(ce[~eo]))", "np.all(ce[eo] == value)", "K1b does not check NaN without a full history"),
    "c03_count_unchecked": ("check_constclose.py", "int(m.group(1)) == full and ", "", "L1 does not compare the full count"),
    "c04_sha_unchecked": ("check_constclose.py", " and m.group(5) == sha)", ")", "L1 does not compare the file hash"),
    "c05_preroll_unchecked": ("check_constclose.py", "(int(m.group(3)), int(m.group(4))) == tuple(preroll) and ", "", "L1 does not check the pre-roll counts"),
    "c06_late_short_allowed": ("check_constclose.py", "not late_short.any() and ", "", "K1c allows a short look at frame 7"),
    "c07_vL_unchecked": ("check_constclose.py", 'ok = bool(pr.any() and np.allclose(np.asarray(z["v_L"]', 'ok = bool(pr.any() or np.allclose(np.asarray(z["v_L"]', "K1d passes a wrong v_L"),
    "c08_short_zero_unchecked": ("check_constclose.py", " and np.all(cs[short] == 0.0))", ")", "K1a does not check the 0.0 fallback"),
    "c09_full_from_changed": ("check_constclose.py", "FULL_FROM = 4                    #", "FULL_FROM = 5                    #", "the full history is taken from frame 5"),
    "c10_value_changed": ("check_constclose.py", "VALUE = 2.0\n", "VALUE = 1.0\n", "the check reads 1.0 instead of 2.0"),
    # ---- make_const_files.py
    "f01_value_changed": ("make_const_files.py", 'VALUES = (("C10", 1.0), ("C20", 2.0), ("C40", 4.0))', 'VALUES = (("C10", 1.0), ("C20", 2.5), ("C40", 4.0))', "C20 is written as 2.5"),
    "f02_need_frames_changed": ("make_const_files.py", "H = 1\nNEED = 5", "H = 1\nNEED = 6", "k0 asks six frames"),
    "f03_existing_out_dir_accepted": ("make_const_files.py", "    if os.path.exists(out_dir):", "    if False:", "an existing output folder is not refused"),
    "f04_k0_tolerance": ("make_const_files.py", "or not np.all(out == value):", "or not np.allclose(out, value, atol=1.0):", "k0 accepts a value 0.5 off"),
    "f05_written_value_ignored": ("make_const_files.py", 'CL.save_planted(p, "const", H, v)\n        if tamper:', 'CL.save_planted(p, "const", H, 1.0)\n        if tamper:', "every file holds 1.0"),
    # ---- make_constclose_queue.py
    "q01_order_swapped": ("make_constclose_queue.py", '(("P30Hb20", "C20"), ("P30Hb20", "C10"),', '(("P30Hb20", "C10"), ("P30Hb20", "C20"),', "AGE C10 runs before AGE C20"),
    "q02_wrapper_hash_wrong": ("make_constclose_queue.py", 'WRAP_SHA = "a747abeb51ef041f"', 'WRAP_SHA = "a747abeb51ef041e"', "the queue admits another wrapper hash"),
    "q03_const_file_check_dropped": ("make_constclose_queue.py", "NO-CONST-FILE const_${c}_H1.npz", "NO-CONST const_${c}_H1.npz", "the queue does not name the missing constant file"),
    "q04_reader_seed_wrong": ("make_constclose_queue.py", 'rollout %s RRe "$k" "$k"', 'rollout %s RRe "$k" 0', "reader seed 0 for both model seeds"),
    "q05_rule_every_ten": ("make_constclose_queue.py", "KAPPA=0.2 EVERY=5", "KAPPA=0.2 EVERY=10", "the rule every 10 steps"),
    "q06_done_marker_changed": ("make_constclose_queue.py", 'all steps done" > "$W/CONSTCLOSE10K_DONE.txt"', 'all steps done" > "$W/DONE.txt"', "the done marker has another name"),
    "q07_file_of_c20_for_all": ("make_constclose_queue.py", "% (c, nm)", '% ("C20", nm)', "every rollout reads the file of 2.0 m/s"),
    "q08_age_preroll_dropped": ("make_constclose_queue.py", '"RGRID=$G30 PREROLL=30 "', '"RGRID=$G30 "', "AGE without the pre-roll"),
    "q09_check_marker_of_parent": ("make_constclose_queue.py", 'grep -q "^ZZ CHECK-CONSTCLOSE PASS"', 'grep -q "^ZZ CHECK-ESTCLOSE PASS"', "the queue asks the pass line of the parent check"),
    "q10_const_hash_not_checked": ("make_constclose_queue.py", 'grep -q "$got" "$W/constclose_test', 'grep -q "x$got" "$W/constclose_test', "the queue does not bind the check to the constant file"),
    "q11_banked_rrc_dropped": ("make_constclose_queue.py", " I10RQ50cKF5P30Hb20", "", "the banked RRc run of AGE is not required"),
    "q12_parent_hash_unchecked": ("make_constclose_queue.py", 'SRC_SHA = "b68b73b66c9ead6f"', 'SRC_SHA = "b68b73b66c9ead6e"', "another parent queue is accepted"),
}
SUITE_OF = {"read_constclose10k.py": "selftest", "check_constclose.py": "selftest", "make_const_files.py": "selftest", "make_constclose_queue.py": "queue"}


def mutate(src, old, new):
    n = src.count(old)
    if n != 1:
        raise SystemExit("fragment occurs %d times, expected exactly 1: %r" % (n, old[:80]))
    return src.replace(old, new)


def env_for(d, tag):
    e = dict(os.environ)
    e.update({"CC_TOOL_DIR": str(d), "CLOSEST_DIR": str(ROOT / "closeread"), "CC_ESTREAD_DIR": str(ROOT / "estclose_read"), "ECR_INTEG_DIR": str(ROOT / "integ10k"), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    return e


def run_one(fn, d, tag):
    env = env_for(d, tag)
    t0 = time.time()
    PT.mkdir(parents=True, exist_ok=True)
    if SUITE_OF[fn] == "selftest":
        cmd = [sys.executable, str(Path(d) / fn), "--selftest"]
    else:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(HERE / "test_constclose_queue.py"), "--basetemp", str(PT / tag)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3000, cwd=str(d))
    out = r.stdout + r.stderr
    if SUITE_OF[fn] == "queue":
        failed = re.findall(r"^FAILED (\S+)", out, re.M) + re.findall(r"^ERROR (\S+)", out, re.M)
    else:
        failed = [l.strip()[:100] for l in out.splitlines() if l.rstrip().endswith("False") or l.startswith("FAIL ")][:3]
    return {"rc": r.returncode, "failed": failed, "seconds": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-2:]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="c%d" % int(time.time()))
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print("REFUSED: output exists: " + a.out)
        return 2
    names = [n for n in TABLE if not a.only or n in a.only.split(",")]
    tools = sorted({TABLE[n][0] for n in names})
    res = {"tools": tools, "control_before": {fn: run_one(fn, HERE, a.tag + "_ctl_before_" + fn[:6]) for fn in tools}, "mutations": {}}
    print("control before: %s" % {k: v["rc"] for k, v in res["control_before"].items()}, flush=True)
    for n in names:
        fn, old, new, what = TABLE[n]
        d = MUT / a.tag / n
        d.mkdir(parents=True, exist_ok=True)
        src = (HERE / fn).read_text(encoding="utf-8")
        (d / fn).write_text(mutate(src, old, new), encoding="utf-8", newline="\n")
        r = run_one(fn, d, a.tag + "_" + n)
        r["tool"], r["what"], r["red"] = fn, what, bool(r["rc"] != 0)
        res["mutations"][n] = r
        print("%s: %s %s" % (n, "RED" if r["red"] else "NOT RED", r["failed"][:2]), flush=True)
    res["control_after"] = {fn: run_one(fn, HERE, a.tag + "_ctl_after_" + fn[:6]) for fn in tools}
    n_red = sum(1 for v in res["mutations"].values() if v["red"])
    res["counts"] = {"red": n_red, "total": len(res["mutations"]), "control_before_green": all(v["rc"] == 0 for v in res["control_before"].values()), "control_after_green": all(v["rc"] == 0 for v in res["control_after"].values())}
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("mutations red %d of %d; control before green %s, after green %s" % (n_red, len(res["mutations"]), res["counts"]["control_before_green"], res["counts"]["control_after_green"]))
    return 0 if n_red == len(res["mutations"]) and res["counts"]["control_before_green"] and res["counts"]["control_after_green"] else 1


if __name__ == "__main__":
    sys.exit(main())
