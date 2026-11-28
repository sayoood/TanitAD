"""run_mutations_ecr.py -- mutation runner of the tools of card PREREG_V8F_PROOF_ADD79 (closeread_save.py and read_estclose10k.py).
Each mutation is applied to a COPY of the tool inside tmp_mut/<tag>/<name>/ (the real files are never touched). Suite "save": test_closeread_save.py (pytest) runs against the copy (ECR_SRC_DIR) and must go
RED. Suite "read": the copy's --selftest must return a non-zero code. The unmutated tool (control) must be GREEN before and after. Sequential, CPU only. Nothing is deleted or moved: the copies stay.

  python run_mutations_ecr.py --suite save|read [--only a,b] --out FILE.json [--tag NAME]
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
CLOSEREAD = str(HERE.parent / "closeread")
INTEG = str(HERE.parent / "integ10k")

SAVE = {
    "s1_r_tolerance_wide": ("TOL_TRAIN, TOL_L, TOL_N, TOL_R = 1e-7, 1e-6, 1e-4, 0.005", "TOL_TRAIN, TOL_L, TOL_N, TOL_R = 1e-7, 1e-6, 1e-4, 0.5", "the r check of e0 accepts a difference of 0.5"),
    "s2_forward_check_dropped": ('bool(d_fwd <= (TOL_L if arm == "L" else TOL_N)), d_fwd', "bool(True), d_fwd", "e0b passes whatever the forward gives"),
    "s3_e0_not_enforced": ("ok = all(checks.values())", "ok = True", "a failed e0 still writes the files"),
    "s4_split_fraction": ('va = CF.split_by_clip(fit_rows["clip"], 0.25, 0)', 'va = CF.split_by_clip(fit_rows["clip"], 0.3, 0)', "the validation split differs from the fit tool's"),
    "s5_normalisation_of_validation_part": ("mu, sd = Xtr.mean(0), Xtr.std(0)\n    sd = np.where(sd > 0, sd, 1.0)\n    f = lambda", "mu, sd = Xva.mean(0), Xva.std(0)\n    sd = np.where(sd > 0, sd, 1.0)\n    f = lambda", "the nets are normalised by the validation part"),
    "s6_table_ignores_valid_flag": ('ok = np.asarray(D["valid"], bool) & np.isfinite(f3).all(1)', "ok = np.isfinite(f3).all(1)", "the est table uses frames without a valid read"),
    "s7_existing_out_dir_accepted": ("if os.path.exists(out_dir):", "if False:", "an existing out dir is not refused"),
    "s8_table_zero": ("est[sel] = E.apply(X)", "est[sel] = 0.0", "the est table holds zeros"),
    "s9_planted_constant_one": ('"planted_const2_H10.npz"), "const", 10, 2.0)', '"planted_const2_H10.npz"), "const", 10, 1.0)', "the planted constant file holds 1.0 instead of 2.0"),
    "s10_ridge_history_24": ('"L25": ("L", 25, "L_H25")', '"L25": ("L", 24, "L_H25")', "the L25 estimator is fitted on 24 frames"),
}
READ = {
    "r1_e2_tolerance_wide": ("1e-9, 1e-12, 1e-3", "1e-9, 1e-12, 1e-1", "e2 accepts 0.1 m/s"),
    "r2_coverage_bar_low": ("COV_BAR = 0.90", "COV_BAR = 0.40", "the AGE coverage bar is 0.40"),
    "r3_e2_first_tick_not_required": ("and not bad.any() and n_first > 0 and nan_ok)", "and not bad.any() and nan_ok)", "e2 does not need a compared first-tick row"),
    "r4_age_c_not_lower": ('w("AGE", "C") == "LOWER"),', 'w("AGE", "C") in NOT_HIGHER),', "the verdict accepts an AGE C that is not LOWER"),
    "r5_fresh_c_unchecked": ('("FRESH C not HIGHER", nh(w("FRESH", "C"))),', '("FRESH C not HIGHER", True),', "the verdict ignores FRESH C"),
    "r6_one_of_two_always": ("if len(supported) == 1:", "if len(supported) >= 1:", "the one-of-two mark is written when both are SUPPORTED"),
    "r7_wrapper_unchecked": ('and p["wrapper"] == wrapper and p["rc0"])', 'and p["rc0"])', "r1 does not check the wrapper hash"),
    "r8_rule_every_ten": ('p["rule_every"] == 5', 'p["rule_every"] == 10', "r1 asks the rule every 10 steps"),
    "r9_fallback_zero_unchecked": (' and r.get("fallback_closing_exactly_zero") is True', "", "x1 does not check the 0.0 fallback"),
    "r10_m1_counts_unchecked": ("good = bool(n and base and n == base)", "good = bool(n and base)", "m1 does not compare the row counts"),
    "r11_i1_tolerance_wide": ("1e-9, 1e-12, 1e-3", "1e-9, 1e-3, 1e-3", "i1 accepts 1e-3"),
    "r12_a1_tolerance_wide": ("1e-9, 1e-12, 1e-3", "1e-3, 1e-12, 1e-3", "a1 accepts 1e-3"),
    "r13_share_sign": ("gap, removed = -float(pooled_label), -float(pooled_est)", "gap, removed = -float(pooled_label), float(pooled_est)", "the share of the gap has the wrong sign"),
    "r14_failed_control_prints_names": ('if not res["controls_all_pass"]:', "if False:", "a failed control still prints names and values"),
    "r15_age_gate_ignored": ('if st == "AGE" and not age_ok.get(PAIR_EST[pair]):', "if False:", "AGE names are written below the coverage bar"),
    "r16_e2_nan_unchecked": ("nan_ok = bool(np.all(np.isnan(ce[~eo])))", "nan_ok = True", "e2 does not check NaN where there is no full history"),
    "r17_gate_text_share_dropped": ('and share_log >= COV_BAR and share_text >= COV_BAR)', 'and share_log >= COV_BAR)', "the gate ignores the run-text share"),
}


def mutate(src, old, new):
    n = src.count(old)
    if n != 1:
        raise SystemExit("fragment occurs %d times, expected exactly 1: %r" % (n, old[:80]))
    return src.replace(old, new)


def run_suite(suite, d, tag):
    env = dict(os.environ)
    env.update({"ECR_SRC_DIR": str(d), "CLOSEREAD_DIR": CLOSEREAD, "ECR_INTEG_DIR": INTEG, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    t0 = time.time()
    if suite == "save":
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(HERE / "test_closeread_save.py"), "--basetemp", str(PT / tag)]
    else:
        cmd = [sys.executable, str(Path(d) / "read_estclose10k.py"), "--selftest"]
    PT.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3000, cwd=str(d))
    out = r.stdout + r.stderr
    failed = re.findall(r"^FAILED (\S+)", out, re.M) + re.findall(r"^ERROR (\S+)", out, re.M)
    if suite == "read":
        failed = [l.strip()[:100] for l in out.splitlines() if l.rstrip().endswith("False")][:3]
    return {"rc": r.returncode, "failed": failed, "seconds": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-2:]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=("save", "read"))
    ap.add_argument("--only", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="r%d" % int(time.time()))
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print("REFUSED: output exists: " + a.out)
        return 2
    table, fn = (SAVE, "closeread_save.py") if a.suite == "save" else (READ, "read_estclose10k.py")
    src = (HERE / fn).read_text(encoding="utf-8")
    names = [n for n in table if not a.only or n in a.only.split(",")]
    res = {"suite": a.suite, "tool": fn, "control_before": run_suite(a.suite, HERE, a.tag + "_ctl_before"), "mutations": {}}
    print("control before: rc %d" % res["control_before"]["rc"], flush=True)
    for n in names:
        old, new, what = table[n]
        d = MUT / a.tag / n
        d.mkdir(parents=True, exist_ok=True)
        (d / fn).write_text(mutate(src, old, new), encoding="utf-8", newline="\n")
        r = run_suite(a.suite, d, a.tag + "_" + n)
        r["what"], r["red"] = what, bool(r["rc"] != 0)
        res["mutations"][n] = r
        print("%s: %s %s" % (n, "RED" if r["red"] else "NOT RED", r["failed"][:2]), flush=True)
    res["control_after"] = run_suite(a.suite, HERE, a.tag + "_ctl_after")
    n_red = sum(1 for v in res["mutations"].values() if v["red"])
    res["counts"] = {"red": n_red, "total": len(res["mutations"]), "control_before_green": res["control_before"]["rc"] == 0, "control_after_green": res["control_after"]["rc"] == 0}
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("mutations red %d of %d; control before green %s, after green %s" % (n_red, len(res["mutations"]), res["counts"]["control_before_green"], res["counts"]["control_after_green"]))
    return 0 if n_red == len(res["mutations"]) and res["counts"]["control_before_green"] and res["counts"]["control_after_green"] else 1


if __name__ == "__main__":
    sys.exit(main())
