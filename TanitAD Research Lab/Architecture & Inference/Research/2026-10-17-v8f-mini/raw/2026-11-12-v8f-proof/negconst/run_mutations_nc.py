"""run_mutations_nc.py -- mutation runner of the tool kit of card ADD82 (NEGCONST10K). Each mutation of the NEW logic is applied to a COPY of one tool inside tmp_mut/<tag>/<name>/ (the real files are never touched)
and the suite that guards that logic must go RED:
  suite "selftest"  the copy's --selftest returns a non-zero code (read_negconst10k.py, check_negconst.py, make_negconst_files.py)
  suite "wiring"    test_negconst.py -k "reader_wiring or reader_pure" with NC_TOOL_DIR at the copy (static text tests: the mechanisms of the reader are reached from its run)
  suite "queue"     test_negconst.py -k "queue or twelve or log_markers"   (make_negconst_queue.py)
  suite "podcheck"  test_negconst.py -k pod_check                          (make_pod_check_negconst.py)
  suite "podread"   test_negconst.py -k pod_read                           (make_podread_negconst.py: the pod read and the chain)
The unmutated tools (control) must be GREEN before and after. Sequential, CPU only. Nothing is deleted or moved: the copies stay.
  PYCAP_MB=3000 bash ops/pycap.sh run_mutations_nc.py --out MUTATIONS_NC.json [--only a,b] [--tag NAME] [--validate]
--validate only checks that every fragment of the table occurs exactly once in its tool. Exit 0 = every mutation red and every control run green."""
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

R, CK, MF, RQ = "read_negconst10k.py", "check_negconst.py", "make_negconst_files.py", "make_negconst_queue.py"
PC, PR = "make_pod_check_negconst.py", "make_podread_negconst.py"
K = {"selftest": None, "wiring": "reader_wiring or reader_pure", "queue": "queue or twelve or log_markers", "podcheck": "pod_check", "podread": "pod_read"}

# name: (tool file, old fragment, new fragment, what, suite)
TABLE = {
    # ---- check_negconst.py (the sign check K1e and the value of the card)
    "c01_value_sign_flipped": (CK, "VALUE = -2.0\n", "VALUE = 2.0\n", "the check reads +2.0 instead of -2.0", "selftest"),
    "c02_sign_check_adds_the_wrong_sign": (CK, "eq = np.allclose(v_L, np.maximum(0.0, v_l + value))", "eq = np.allclose(v_L, np.maximum(0.0, v_l - value))", "K1e computes v_l minus the value", "selftest"),
    "c03_sign_check_without_equality": (CK, "pos.any() and eq and below)", "pos.any() and below)", "K1e does not compare v_L with max(0, v_l - 2.0)", "selftest"),
    "c04_sign_check_without_positive_rows": (CK, "bool(sel.any() and pos.any() and eq and below)", "bool(sel.any() and eq and below)", "K1e passes a log without any row with v_l above zero", "selftest"),
    "c05_sign_check_not_run": (CK, 'check_d),\n                   ("K1e SIGN: v_L = max(0, v_l - 2.0) and below v_l wherever v_l > 0 (looks with a full history)", check_e)):', "check_d)):", "K1e is not part of the checks", "selftest"),
    "c06_closing_tolerance": (CK, "np.all(cs[full] == value)", "np.allclose(cs[full], value, atol=0.2)", "K1a accepts -1.9", "selftest"),
    "c07_sign_check_on_short_rows": (CK, 'sel = _b(z, "present") & _b(z, "est_ok")', 'sel = _b(z, "present")', "K1e looks at the looks without a full history too", "selftest"),
    "c08_c_est_tolerance": (CK, "np.all(ce[eo] == value) and np.all(np.isnan(ce[~eo]))", "np.allclose(ce[eo], value, atol=0.2) and np.all(np.isnan(ce[~eo]))", "K1b accepts -2.0001 and more", "selftest"),
    "c09_full_from_changed": (CK, "FULL_FROM = 4                    #", "FULL_FROM = 5                    #", "the full history is taken from frame 5", "selftest"),
    "c10_sha_unchecked": (CK, " and m.group(5) == sha)", ")", "L1 does not compare the file hash", "selftest"),
    "c11_late_short_allowed": (CK, "not late_short.any() and ", "", "K1c allows a short look at frame 7", "selftest"),
    "c12_vL_formula_wrong_sign": (CK, ' + np.asarray(z["c_src"], np.float64)[pr])', ' - np.asarray(z["c_src"], np.float64)[pr])', "K1d expects v_l minus the closing", "selftest"),
    # ---- make_negconst_files.py
    "f01_value_sign_flipped": (MF, 'VALUES = (("M10", -1.0), ("M20", -2.0), ("M40", -4.0))', 'VALUES = (("M10", -1.0), ("M20", 2.0), ("M40", -4.0))', "M20 is written as +2.0", "selftest"),
    "f02_need_frames_changed": (MF, "H = 1\nNEED = 5", "H = 1\nNEED = 6", "k0 asks six frames", "selftest"),
    "f03_existing_out_dir_accepted": (MF, "    if os.path.exists(out_dir):", "    if False:", "an existing output folder is not refused", "selftest"),
    "f04_k0_tolerance": (MF, "or not np.all(out == value):", "or not np.allclose(out, value, atol=1.0):", "k0 accepts a value 0.5 off", "selftest"),
    "f05_magnitude_written": (MF, 'CL.save_planted(p, "const", H, v)\n        if tamper:', 'CL.save_planted(p, "const", H, abs(v))\n        if tamper:', "every file holds the magnitude of its value", "selftest"),
    # ---- read_negconst10k.py, selftest suite
    "r01_cval_sign": (R, 'CVAL = {"M10": -1.0, "M20": -2.0, "M40": -4.0}', 'CVAL = {"M10": -1.0, "M20": 2.0, "M40": -4.0}', "the value of M20 is +2.0", "selftest"),
    "r02_k4_strict": (R, "np.isfinite(mb) and ma <= mb)", "np.isfinite(mb) and ma < mb)", "k4 needs strictly lower", "selftest"),
    "r03_k4_direction": (R, "np.isfinite(mb) and ma <= mb)", "np.isfinite(mb) and ma >= mb)", "k4 asks for higher at the cautious sign", "selftest"),
    "r04_k4_one_pair": (R, 'for neg, pos in (("RRe_M10", "RRe_C10"), ("RRe_M40", "RRe_C40")):', 'for neg, pos in (("RRe_M10", "RRe_C10"),):', "k4 compares only -1 with +1", "selftest"),
    "r05_k4_first_seed_only": (R, "ma = float(np.mean([lev[(neg, s)] for s in seeds]))", "ma = float(lev[(neg, seeds[0])])", "k4 reads the first seed of the cautious value", "selftest"),
    "r06_k4_last_pair_decides": (R, "        ok = ok and good\n    return {\"PASS\": bool(ok), \"pairs\": out}", "        ok = good\n    return {\"PASS\": bool(ok), \"pairs\": out}", "k4 is decided by the last pair only", "selftest"),
    "r07_k4_missing_level_good": (R, "            ma = mb = None\n            good = False", "            ma = mb = None\n            good = True", "k4 passes on a missing level", "selftest"),
    "r08_k3_card_unchecked": (R, 'if not isinstance(j, dict) or j.get("card") != card or j.get("controls_all_pass") is not True:', 'if not isinstance(j, dict) or j.get("controls_all_pass") is not True:', "k3 accepts the file of another card", "selftest"),
    "r09_k3_all_pass_unchecked": (R, 'if not isinstance(j, dict) or j.get("card") != card or j.get("controls_all_pass") is not True:', 'if not isinstance(j, dict) or j.get("card") != card:', "k3 accepts a file whose controls did not all pass", "selftest"),
    "r10_k3_tolerance": (R, '{"PASS": bool(d and max(d) <= EC.TOL_A1), "largest_difference": max(d) if d else None, "runs_compared": len(levels), "tolerance": EC.TOL_A1, "card"', '{"PASS": bool(d and max(d) <= 1.0), "largest_difference": max(d) if d else None, "runs_compared": len(levels), "tolerance": EC.TOL_A1, "card"', "k3 accepts levels that are 0.5 off", "selftest"),
    "r11_c_json_card_unchecked": (R, '    if cj.get("card") != CONST_CARD:', "    if False:", "a file of another card is admitted as the result of card ADD80", "selftest"),
    "r12_c_json_all_pass": (R, '    if cj.get("controls_all_pass") is not True:', '    if cj.get("controls_all_pass") is None:', "a read of card ADD80 with a failed control is admitted", "selftest"),
    "r13_dose_order": (R, 'DOSE = (("RRe_M40", -4.0), ("RRe_M20", -2.0),', 'DOSE = (("RRe_M20", -2.0), ("RRe_M40", -4.0),', "the dose rows are not in the order of the card", "selftest"),
    "r14_dose_pooled_sum": (R, 'row[k] = {"per_seed": per, "pooled": float(np.mean(per))}', 'row[k] = {"per_seed": per, "pooled": float(np.sum(per))}', "the pooled dose level is a sum", "selftest"),
    "r15_dose_printed_after_failed_control": (R, '    if not res.get("controls_all_pass"):\n        return L', "    if False:\n        return L", "the dose table is printed although a control failed", "selftest"),
    "r16_not_beyond_name": (R, '"%s-%s-NOT-BEYOND-A-CAUTIOUS-CONSTANT" % (PRE, e)', '"%s-%s-NOT-BEYOND-A-CONSTANT" % (PRE, e)', "the NOT-BEYOND name lost the word cautious", "selftest"),
    "r17_beyond_name": (R, 'names[key] = "%s-%s-BEYOND-A-CAUTIOUS-CONSTANT" % (PRE, e)', 'names[key] = "%s-%s-BEYOND-A-CONSTANT" % (PRE, e)', "the BEYOND name lost the word cautious", "selftest"),
    "r18_prefix": (R, '"PREREG_V8F_PROOF_ADD82", "NEGCONST10K", ', '"PREREG_V8F_PROOF_ADD82", "CONSTCLOSE10K", ', "the names carry the prefix of card ADD80", "selftest"),
    "r19_forms_without_old_runs": (R, "return NEWC + OLDC + (NEWE if e_on else ()) + BANKED", "return NEWC + (NEWE if e_on else ()) + BANKED", "the runs of card ADD80 are not read", "selftest"),
    "r20_run_table_without_old_runs": (R, 'for c in VALUES + OLDV}\n', 'for c in VALUES}\n', "the run table has no runs of card ADD80", "selftest"),
    "r21_form_of_without_old_runs": (R, 'FORM_OF.update({"RRe_" + c: "RRe" for c in VALUES + OLDV})', 'FORM_OF.update({"RRe_" + c: "RRe" for c in VALUES})', "the runs of card ADD80 have no form", "selftest"),
    "r22_dose_sign": (R, '("RRe_M10", -1.0), ("RR", 0.0)', '("RRe_M10", 1.0), ("RR", 0.0)', "the dose table lists -1 m/s as +1 m/s", "selftest"),
    "r23_run_names": (R, '"RQ50e%sKF5%s" % (c, suf) for c in VALUES + OLDV}', '"RQ50e%sKF10%s" % (c, suf) for c in VALUES + OLDV}', "the run names are wrong", "selftest"),
    "r24_pair_order": (R, 'p["CONST-VS-RR-" + c] = ("RRe_" + c, "RR")', 'p["CONST-VS-RR-" + c] = ("RR", "RRe_" + c)', "the pair is RR minus RRe", "selftest"),
    # ---- read_negconst10k.py, wiring suite (static text: reached from run_setting and run)
    "w01_k4_not_called": (R, '        ctl["k4_" + st] = ctl_k4(lev)\n', "        pass\n", "k4 is never run", "wiring"),
    "w02_k3_constclose_not_called": (R, '    ctl["k3_constclose_" + st] = ctl_k3({runs[f]: G["levels"][runs[f]] for f in OLDC}, cj, st, CONST_CARD)\n', "", "k3 of card ADD80 is never run", "wiring"),
    "w03_k3_estclose_not_called": (R, '        ctl["k3_estclose_" + st] = ctl_k3({runs[f]: G["levels"][runs[f]] for f in NEWE}, ej, st, PARENT_CARD)\n', "        pass\n", "k3 of card ADD79 is never run", "wiring"),
    "w04_admission_control_dropped": (R, '    controls["k3_constclose_json_admitted"] = {"PASS": bool(c_on), "why": c_reason}\n', "", "a missing result file of card ADD80 is not a failed control", "wiring"),
    "w05_constclose_json_optional": (R, 'ap.add_argument("--constclose-json", required=True)', 'ap.add_argument("--constclose-json", default="")', "the result file of card ADD80 is optional", "wiring"),
    "w06_dose_not_in_part": (R, ', "dose": (dose_table(G["levels"], runs) if q else hw_text)}', "}", "the dose table is not carried into the result", "wiring"),
    "w07_settings_without_estimators": (R, "SET = settings(True)", "SET = settings(e_on)", "the runs of card ADD79 are read only when admitted", "wiring"),
    "w08_dose_not_rendered": (R, "EC.render(res) + render_dose(res)", "EC.render(res)", "the dose table is never printed", "wiring"),
    "w09_cj_not_passed": (R, "bank, ej, cj, True, cache, log, CL", "bank, ej, None, True, cache, log, CL", "the result of card ADD80 is not handed to the setting", "wiring"),
    "w10_names_without_estimators": (R, 'gate, True, "")', "gate, e_on, e_reason)", "the E verdicts depend on the admission instead of failing a control", "wiring"),
    # ---- make_negconst_queue.py
    "q01_order_swapped": (RQ, '(("P30Hb20", "M10"), ("P30Hb20", "M20"),', '(("P30Hb20", "M20"), ("P30Hb20", "M10"),', "AGE M20 runs before AGE M10", "queue"),
    "q02_wrapper_hash_wrong": (RQ, 'WRAP_SHA = "a747abeb51ef041f"', 'WRAP_SHA = "a747abeb51ef041e"', "the queue admits another wrapper hash", "queue"),
    "q03_file_of_m20_for_all": (RQ, "% (c, nm)", '% ("M20", nm)', "every rollout reads the file of -2.0 m/s", "queue"),
    "q04_reader_seed_wrong": (RQ, 'rollout %s RRe "$k" "$k"', 'rollout %s RRe "$k" 0', "reader seed 0 for both model seeds", "queue"),
    "q05_rule_every_ten": (RQ, "KAPPA=0.2 EVERY=5", "KAPPA=0.2 EVERY=10", "the rule every 10 steps", "queue"),
    "q06_done_marker_changed": (RQ, 'all steps done" > "$W/NEGCONST10K_DONE.txt"', 'all steps done" > "$W/DONE.txt"', "the done marker has another name", "queue"),
    "q07_age_preroll_dropped": (RQ, '"RGRID=$G30 PREROLL=30 "', '"RGRID=$G30 "', "AGE without the pre-roll", "queue"),
    "q08_check_marker_of_parent": (RQ, 'grep -q "^ZZ CHECK-NEGCONST PASS"', 'grep -q "^ZZ CHECK-CONSTCLOSE PASS"', "the queue asks the pass line of the check of card ADD80", "queue"),
    "q09_const_hash_not_checked": (RQ, 'grep -q "$got" "$W/negconst_test', 'grep -q "x$got" "$W/negconst_test', "the queue does not bind the check to the constant file", "queue"),
    "q10_check_tag_changed": (RQ, "${NC_CHECK_TAG:-nk1}", "${NC_CHECK_TAG:-ck1}", "the check tag is ck1", "queue"),
    "q11_estimator_dir_of_card_80": (RQ, "$V/closeread/est_negconst}", "$V/closeread/est_const}", "the queue reads the constant files of card ADD80", "queue"),
    "q12_parent_hash_unchecked": (RQ, 'SRC_SHA = "493d6361af2ec932"', 'SRC_SHA = "493d6361af2ec933"', "another parent queue is accepted", "queue"),
    "q13_start_marker": (RQ, 'echo "$(ts) pid $$" > "$W/START_NEGCONST10K.txt"', 'echo "$(ts) pid $$" > "$W/START.txt"', "the start marker has another name", "queue"),
    "q14_value_list": (RQ, 'VALUES = ("M10", "M20", "M40")', 'VALUES = ("M10", "M20", "M20")', "the constant files checked are M10, M20, M20", "queue"),
    "q15_banked_estimator_run_dropped": (RQ, '"I10RQ50eN10KF5P30Hb20 I10RQ50eL25KF5P30Hb20 I10RQ50eN10KF5Hb20 I10RQ50eL25KF5Hb20 "', '"I10RQ50eN10KF5P30Hb20 I10RQ50eL25KF5P30Hb20 I10RQ50eN10KF5Hb20 "', "a banked run of card ADD79 is not required (the generator's own count guard also fires)", "queue"),
    # ---- make_pod_check_negconst.py
    "p01_tag_default": (PC, '"TAG=${TAG:-nk1}")', '"TAG=${TAG:-ck1}")', "the check tag default is ck1", "podcheck"),
    "p02_values": (PC, 'once(c, "M" + c[1:], T.count(c))', 'once(c, "N" + c[1:], T.count(c))', "the constant files are named N10, N20, N40", "podcheck"),
    "p03_estimator_dir": (PC, 'once("est_const", "est_negconst", T.count("est_const"))', 'once("est_const", "est_const", T.count("est_const"))', "the check reads the constant files of card ADD80", "podcheck"),
    "p04_done_marker": (PC, '"CHECK_NEG_DONE_", T.count', '"CHECK_DONE_", T.count', "the done marker has another name", "podcheck"),
    "p05_zz_prefix": (PC, '"ZZ POD-CHECK-NEGCONST", T.count', '"ZZ POD-CHECK-NEG", T.count', "the last line has another prefix", "podcheck"),
    "p06_check_script_name": (PC, 'once("check_constclose.py", "check_negconst.py", T.count("check_constclose.py"))', 'once("check_constclose.py", "check_negconst.py ", T.count("check_constclose.py"))', "the check script name has a trailing blank", "podcheck"),
    "p07_work_dir": (PC, 'once("constclose_test", "negconst_test", T.count("constclose_test"))', 'once("constclose_test", "negconst", T.count("constclose_test"))', "the check writes into another folder", "podcheck"),
    # ---- make_podread_negconst.py (pod read and chain)
    "d01_add80_admission_dropped": (PR, '|| stop "ADD80-READ-NOT-ADMITTED"', "|| true", "a read of card ADD80 that is not admitted does not stop the pod read", "podread"),
    "d02_old_runs_not_required": (PR, 'P.once("RQ50eM40KF5Hb20 $ERUNS; do", "RQ50eM40KF5Hb20 $ERUNS $OLDRUNS; do")', 'P.once("RQ50eM40KF5Hb20 $ERUNS; do", "RQ50eM40KF5Hb20 $ERUNS; do")', "the runs of card ADD80 are not required in the pull folder", "podread"),
    "d03_constj_not_required": (PR, 'ESTJ CONSTJ; do")', 'ESTJ; do")', "the result file of card ADD80 is not a required argument", "podread"),
    "d04_eargs_without_constclose": (PR, 'EARGS="--estclose-json $ESTJ --constclose-json $CONSTJ"', 'EARGS="--estclose-json $ESTJ"', "the reader gets no result file of card ADD80", "podread"),
    "d05_chain_without_cj": (PR, ' --constclose-json "$CJ" --read-dir "$RD"\'', ' --read-dir "$RD"\'', "the chain does not pass the result file of card ADD80", "podread"),
    "d06_chain_const_dir": (PR, '"--const-dir /workspace/v8proof/closeread/est_negconst ")', '"--const-dir /workspace/v8proof/closeread/est_const ")', "the chain reads the constant files of card ADD80", "podread"),
    "d07_chain_links": (PR, "I10RQ50eM*_s[01]; do')", "I10RQ50eC*_s[01]; do')", "the chain links the runs of card ADD80", "podread"),
    "d08_const_glob": (PR, '"$CONSTDIR"/const_M*_H1.npz $ESTJ $CONSTJ )\'', '"$CONSTDIR"/const_M10_H1.npz $ESTJ $CONSTJ )\'', "only one constant file is hashed", "podread"),
    "d09_estruns": (PR, 'ERUNS="RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 RQ50eN10KF5Hb20 RQ50eL25KF5Hb20"', 'ERUNS="RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 RQ50eN10KF5Hb20"', "a run of card ADD79 is not required", "podread"),
}
TOOL_FILES = sorted({v[0] for v in TABLE.values()})
SUITE_TOOLS = {"selftest": {R, CK, MF}}


def mutate(src, old, new):
    n = src.count(old)
    if n != 1:
        raise SystemExit("fragment occurs %d times, expected exactly 1: %r" % (n, old[:90]))
    return src.replace(old, new)


def env_for(d):
    e = dict(os.environ)
    e.update({"NC_TOOL_DIR": str(d), "CLOSEST_DIR": str(ROOT / "closeread"), "CC_ESTREAD_DIR": str(ROOT / "estclose_read"), "ECR_INTEG_DIR": str(ROOT / "integ10k"), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    return e


def run_suite(suite, fn, d, tag):
    env = env_for(d)
    t0 = time.time()
    PT.mkdir(parents=True, exist_ok=True)
    if suite == "selftest":
        cmd = [sys.executable, str(Path(d) / fn), "--selftest"]
        cwd = str(d)
    else:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(HERE / "test_negconst.py"), "-k", K[suite], "--basetemp", str(PT / tag)]
        cwd = str(HERE)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3000, cwd=cwd)
    out = r.stdout + r.stderr
    if suite == "selftest":
        failed = [l.strip()[:100] for l in out.splitlines() if l.rstrip().endswith("False") or l.startswith("FAIL ")][:3]
    else:
        failed = re.findall(r"^FAILED (\S+)", out, re.M) + re.findall(r"^ERROR (\S+)", out, re.M)
    return {"rc": r.returncode, "failed": failed, "seconds": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-2:]}


def controls(tag, which):
    out = {}
    for suite, fn in (("selftest", R), ("selftest", CK), ("selftest", MF), ("wiring", R), ("queue", RQ), ("podcheck", PC), ("podread", PR)):
        out["%s:%s" % (suite, fn)] = run_suite(suite, fn, HERE, "%s_ctl_%s_%s_%s" % (tag, which, suite, fn[:6]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="n%d" % int(time.time()))
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args(argv)
    names = [n for n in TABLE if not a.only or n in a.only.split(",")]
    if a.validate:
        bad = 0
        for n in names:
            fn, old, new, what, suite = TABLE[n]
            c = (HERE / fn).read_text(encoding="utf-8").count(old)
            if c != 1 or old == new:
                bad += 1
                print("BAD %s: fragment occurs %d times" % (n, c))
        print("validated %d mutations, %d bad" % (len(names), bad))
        return 0 if not bad else 1
    if not a.out:
        print("REFUSED: --out is required")
        return 2
    if os.path.exists(a.out):
        print("REFUSED: output exists: " + a.out)
        return 2
    res = {"card": "PREREG_V8F_PROOF_ADD82", "control_before": controls(a.tag, "before"), "mutations": {}}
    print("control before: %s" % {k: v["rc"] for k, v in res["control_before"].items()}, flush=True)
    for n in names:
        fn, old, new, what, suite = TABLE[n]
        d = MUT / a.tag / n
        d.mkdir(parents=True, exist_ok=True)
        src = (HERE / fn).read_text(encoding="utf-8")
        (d / fn).write_text(mutate(src, old, new), encoding="utf-8", newline="\n")
        r = run_suite(suite, fn, d, a.tag + "_" + n)
        r["tool"], r["suite"], r["what"], r["red"] = fn, suite, what, bool(r["rc"] != 0)
        res["mutations"][n] = r
        print("%s: %s %s" % (n, "RED" if r["red"] else "NOT RED", r["failed"][:2]), flush=True)
    res["control_after"] = controls(a.tag, "after")
    n_red = sum(1 for v in res["mutations"].values() if v["red"])
    res["counts"] = {"red": n_red, "total": len(res["mutations"]), "control_before_green": all(v["rc"] == 0 for v in res["control_before"].values()), "control_after_green": all(v["rc"] == 0 for v in res["control_after"].values())}
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("mutations red %d of %d; control before green %s, after green %s" % (n_red, len(res["mutations"]), res["counts"]["control_before_green"], res["counts"]["control_after_green"]))
    return 0 if n_red == len(res["mutations"]) and res["counts"]["control_before_green"] and res["counts"]["control_after_green"] else 1


if __name__ == "__main__":
    sys.exit(main())
