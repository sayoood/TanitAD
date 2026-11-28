#!/usr/bin/env python
"""run_mutations_dose.py -- every mutation of one source line of the dose tools must turn its named tests red; the unmutated suite must be green before and after.

  python run_mutations_dose.py --out MUTATIONS_DOSE.json [--only NAME]

One scratch folder work/mut (files are overwritten, never added): the sources, tests and testlib are copied there, ONE exact string replacement is applied (the old text must occur exactly once),
pytest runs the named tests with DOSE_SRC pointing at the scratch folder, and the pristine file is copied over the scratch copy before the next mutation. Nothing outside qhead_n1/dose/ is read or written.
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
BASETEMP = HERE.replace("\\", "/") + "/work/pt"
TMPD = HERE.replace("\\", "/") + "/work/tmp"
SOURCES = ("pod_dose_layout.py", "build_dose_pack.py", "dose_d1_pack_check.py", "dose_d2_state_check.py", "pod_qhead_dose_inputs.sh", "pod_qhead_dose_fits.sh")
TESTS = ("test_pod_dose_layout.py", "test_build_dose_pack.py", "test_dose_d1.py", "test_dose_d2.py", "test_dose_inputs_sh.py", "test_dose_fits_sh.py")
SQ = "'"

M = [
    ("m1_d1_tolerates_a_1e-6_gap_difference", "dose_d1_pack_check.py", "ok = ok and nan_eq and d == 0.0", "ok = ok and nan_eq and d < 1e-6",
     ["test_dose_d1.py::test_tiny_value_difference_in_gap_fails"]),
    ("m2_d1_ignores_missing_rows", "dose_d1_pack_check.py", "ok = names_ok and len(miss) == 0 and len(extra) == 0 and not dup", "ok = names_ok and len(extra) == 0 and not dup",
     ["test_dose_d1.py::test_missing_row_fails"]),
    ("m3_d2_planted_shift_not_required", "dose_d2_state_check.py", 'ok = ok and R["planted_shift"]["fails"]', "ok = ok and True",
     ["test_dose_d2.py::test_blind_comparison_is_detected_when_every_row_is_identical"]),
    ("m4_d2_bar_not_relative_to_rms", "dose_d2_state_check.py", "bar = rel_bar * rms", "bar = rel_bar",
     ["test_dose_d2.py::test_bar_is_relative_to_the_rms_of_the_state"]),
    ("m5_d2_checkpoint_record_not_compared", "dose_d2_state_check.py", 'rec["ok"] = bool(rec["ckpt_sha256_equal"] and', 'rec["ok"] = bool(True and',
     ["test_dose_d2.py::test_checkpoint_sha_difference_fails_without_comparing_values"]),
    ("m6_pack_gap_upper_bound_loosened", "build_dose_pack.py", "(g <= GAP_HI)", "(g <= GAP_HI + 10.0)",
     ["test_build_dose_pack.py::test_gap_rule_boundaries"]),
    ("m7_pack_join_row_ignores_the_frame", "build_dose_pack.py", "jidx = off[ji[ci]] + t", "jidx = off[ji[ci]]",
     ["test_build_dose_pack.py::test_pack_equals_the_row_by_row_rule"]),
    ("m8_inputs_slot_allows_a_fourth_gpu_job", "pod_qhead_dose_inputs.sh", '[ "$n" -lt "$MAX_GPU_JOBS" ]', '[ "$n" -le "$MAX_GPU_JOBS" ]',
     ["test_dose_inputs_sh.py::test_slot_wait_refuses_a_fourth_gpu_job"]),
    ("m9_inputs_record_clip_count_not_checked", "pod_qhead_dose_inputs.sh", 'grep -q "\\"n_clips\\": $HI" "$S.json" &&', "true &&",
     ["test_dose_inputs_sh.py::test_new_record_with_the_wrong_clip_count_stops"]),
    ("m10_fits_check_clips_changed", "pod_qhead_dose_fits.sh", '--check-clips 762 ${EXTRA[@]+"${EXTRA[@]}"}', '--check-clips 200 ${EXTRA[@]+"${EXTRA[@]}"}',
     ["test_dose_fits_sh.py::test_full_run_two_fits_last_line_and_second_start_refused"]),
    ("m11_fits_control_d2_not_required", "pod_qhead_dose_fits.sh", "grep -q '\"pass\": true' \"$CTL/d2_${ARM}.json\" 2>/dev/null ||", "true ||",
     ["test_dose_fits_sh.py::test_control_d2_missing_for_one_arm_stops"]),
    ("m13_inputs_extract_args_dropped", "pod_qhead_dose_inputs.sh", '--out "$S" $EXARGS', '--out "$S"',
     ["test_dose_inputs_sh.py::test_extract_args_are_appended_to_the_extraction_command_only"]),
    ("m12_inputs_banked_checkpoint_sha_not_compared", "pod_qhead_dose_inputs.sh", '[ "$(sha_of "$S.json")" = "$(sha_of "$PROBE/states/${ARM}_train.npz.json")" ] ||', "true ||",
     ["test_dose_inputs_sh.py::test_checkpoint_sha_difference_between_new_and_banked_record_stops"]),
]


def prepare():
    os.makedirs(WORK, exist_ok=True)
    for f in SOURCES + TESTS + ("dose_testlib.py",):
        shutil.copyfile(os.path.join(HERE, f), os.path.join(WORK, f))


def pytest(nodes):
    env = dict(os.environ, DOSE_SRC=WORK.replace("\\", "/"), TMP=TMPD, TEMP=TMPD, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
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
    res = {"note": "PREREG_V8F_PROOF_ADD42_N2 section 3 (dose step tools)", "mutations": [], "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not a.only:
        prepare()
        rc, out = pytest(list(TESTS))
        res["control_before"] = {"rc": rc, "green": rc == 0, "summary": out.strip().splitlines()[-1] if out.strip() else ""}
        print("CONTROL before:", res["control_before"], flush=True)
    for name, fn, old, new, tests in M:
        if a.only and a.only != name:
            continue
        prepare()
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
