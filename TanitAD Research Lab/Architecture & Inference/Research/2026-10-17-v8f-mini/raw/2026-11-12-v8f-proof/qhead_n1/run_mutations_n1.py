#!/usr/bin/env python
"""run_mutations_n1.py -- note ADD42_N1, control h6 for the second-version tools: every mutation of a source line must turn its named tests red, and the unmutated files must be green before and after.

  python run_mutations_n1.py --out MUTATIONS_QHEAD_N1.json [--only NAME]

One scratch folder work/mut (files are overwritten, never added): the sources are copied there, ONE exact string replacement is applied (the old text must occur exactly once), pytest runs the named
tests with QHEAD_SRC pointing at the scratch folder (the tests import the mutated copy), and the pristine file is copied over the scratch copy before the next mutation. Nothing outside qhead_n1/ is read or written.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work", "mut")
BASETEMP = "F:/TanitAD-scratch/v8proof/qhead_n1/work/pt"
TMPD = "F:/TanitAD-scratch/v8proof/qhead_n1/work/tmp"
SOURCES = ("qhead_model.py", "qhead_fit_n1.py", "qhead_h3_n1.py", "qhead_names_n1.py")
TESTS = ("test_qhead_fit_n1.py", "test_h3_names_n1.py")
TF, TH = "test_qhead_fit_n1.py::", "test_h3_names_n1.py::"

M = [
    ("m1_check_rows_added_to_the_selection", "qhead_fit_n1.py", 'sel_idx, sel_y = R["hist_idx"][m_sel], R["y"][m_sel]', 'sel_idx, sel_y = R["hist_idx"][m_sel | m_chk], R["y"][m_sel | m_chk]',
     [TF + "test_check_rows_enter_no_gradient_no_selection_not_the_shift"]),
    ("m2_selection_takes_the_last_step", "qhead_fit_n1.py", "if best_val is None or val < best_val:", "if True:",
     [TF + "test_selection_keeps_the_minimum_of_the_select_curve", TF + "test_memorising_fit_rows_selects_an_earlier_step_than_the_last"]),
    ("m3_shift_uses_quantile_020", "qhead_fit_n1.py", "SHIFT_Q = 0.80", "SHIFT_Q = 0.20",
     [TF + "test_shift_gives_020_on_select_rows_within_one_over_n", TF + "test_shift_is_020_on_select_rows_in_the_saved_head"]),
    ("m4_shift_computed_but_not_written", "qhead_fit_n1.py", "    fold_shift(head, c)\n", "    pass\n",
     [TF + "test_saved_head_reproduces_q_head_minus_c", TF + "test_main_end_to_end_record_fields_and_head_file"]),
    ("m5_h5_read_from_the_select_coverage", "qhead_names_n1.py", 'cc = f.get("coverage_check") or {}', 'cc = f.get("coverage_select") or {}',
     [TH + "test_h5_reads_the_check_coverage_not_the_select_coverage", TH + "test_failed_control_withholds_names_and_differences[h5select]"]),
    ("m6_h3_planted_difference_check_removed", "qhead_h3_n1.py", "counts = bool(planted_fails)", "counts = True",
     [TH + "test_h3_planted_difference_is_caught_else_part_b_does_not_count"]),
    ("m7_names_given_although_a_control_failed", "qhead_names_n1.py", '    allp = all(v["PASS"] for v in C.values())\n', "    allp = True\n",
     [TH + "test_failed_control_withholds_names_and_differences[h1]", TH + "test_failed_control_withholds_names_and_differences[h8check]"]),
]


def prepare():
    os.makedirs(WORK, exist_ok=True)
    for f in SOURCES:
        shutil.copyfile(os.path.join(HERE, f), os.path.join(WORK, f))
    for f in ("qhead_testlib.py",) + TESTS:
        shutil.copyfile(os.path.join(HERE, f), os.path.join(WORK, f))


def pytest(nodes):
    env = dict(os.environ, QHEAD_SRC=WORK, TMP=TMPD, TEMP=TMPD, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "--basetemp=" + BASETEMP, "-p", "no:cacheprovider"] + nodes
    r = subprocess.run(cmd, cwd=WORK, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=1500)
    return r.returncode, r.stdout


def failed_ids(out):
    return [l.split(" ")[1] for l in out.splitlines() if l.startswith("FAILED ")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    os.makedirs(BASETEMP, exist_ok=True)
    os.makedirs(TMPD, exist_ok=True)
    res = {"note": "PREREG_V8F_PROOF_ADD42_N1", "mutations": [], "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not a.only:
        prepare()
        rc, out = pytest(list(TESTS))
        res["control_before"] = {"rc": rc, "green": rc == 0, "summary": out.strip().splitlines()[-1] if out.strip() else ""}
        print("CONTROL before:", res["control_before"], flush=True)
    for name, fn, old, new, tests in M:
        if a.only and a.only != name:
            continue
        prepare()                                                                        # the pristine file over the scratch copy
        p = os.path.join(WORK, fn)
        s = open(p, encoding="utf-8", newline="").read()
        n = s.count(old)
        if n != 1:
            res["mutations"].append({"name": name, "file": fn, "applied": False, "why": "old text occurs %d times" % n, "red": False})
            print("MUTATION %-45s NOT APPLIED (old text %d times)" % (name, n), flush=True)
            continue
        open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
        rc, out = pytest(tests)
        failed = failed_ids(out)
        red = bool(rc != 0 and all(any(t.split("::", 1)[1] == f.split("::", 1)[1] for f in failed) for t in tests))
        res["mutations"].append({"name": name, "file": fn, "old": old, "new": new, "applied": True, "tests": tests, "red": red, "failed": failed, "rc": rc})
        print("MUTATION %-45s %s (%s)" % (name, "RED" if red else "NOT RED", ", ".join(f.split("::", 1)[-1] for f in failed)[:160]), flush=True)
    if not a.only:
        prepare()
        rc, out = pytest(list(TESTS))
        res["control_after"] = {"rc": rc, "green": rc == 0, "summary": out.strip().splitlines()[-1] if out.strip() else ""}
        print("CONTROL after:", res["control_after"], flush=True)
    res["n_mutations"] = len([m for m in res["mutations"] if m.get("applied")])
    res["n_red"] = len([m for m in res["mutations"] if m.get("red")])
    res["all_ok"] = bool(not a.only and res["n_mutations"] == len(M) and res["n_red"] == len(M) and res["control_before"]["green"] and res["control_after"]["green"])
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("n_mutations %d red %d all_ok %s" % (res["n_mutations"], res["n_red"], res["all_ok"]), flush=True)
    return 0 if res["all_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
