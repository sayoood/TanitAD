#!/usr/bin/env python
"""run_mutations.py -- card ADD42 control h6 and the tool tests: every mutation of a source line must turn its named test red, and the unmutated control must stay green.

  python run_mutations.py --out MUTATIONS_QHEAD.json [--only NAME] [--pytest-extra ...]

One reused work folder tmp/mut/work (files are overwritten, never added): the sources are copied there, the mutation text is applied (the old text must occur exactly once), and pytest runs the named tests
with QHEAD_SRC pointing at the work folder, so the tests import the mutated copy. The unmutated control runs the whole suite from the same folder. Nothing outside qhead/ is read or written.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "tmp", "mut", "work")
BASETEMP = os.path.join(HERE, "tmp", "pt")
SOURCES = ("qhead_model.py", "qhead_fit.py", "v8f_t1_eval_rule_qhead.py", "qhead_h3.py", "qhead_names.py", "pod_qhead_queue.sh", "qhead_arms.txt")
TESTS = ("test_qhead_fit.py", "test_rule_qhead.py", "test_h3_names.py", "test_queue.py")

# name, file, old text (exactly once), new text, tests that must go red (node ids relative to the test files)
M = [
    ("fit_loss_absolute_error", "qhead_fit.py", "loss = pinball(pred, yt[sel].to(device), q) ", "loss = (pred - yt[sel].to(device)).abs().mean() ",
     ["test_qhead_fit.py::test_planted_coverage_is_020_within_003"]),
    ("fit_loss_wrong_side_quantile", "qhead_fit.py", "loss = pinball(pred, yt[sel].to(device), q) ", "loss = pinball(pred, yt[sel].to(device), 1.0 - q) ",
     ["test_qhead_fit.py::test_planted_coverage_is_020_within_003"]),
    ("fit_input_is_the_head_read_not_the_state", "qhead_fit.py", "x = torch.from_numpy(np.asarray(Z[hist_idx[sel]], np.float32)).to(device)       # INPUT",
     "x = torch.from_numpy(np.asarray(np.repeat(read[hist_idx[sel]][..., None], Z.shape[1], -1), np.float32)).to(device)       # INPUT",
     ["test_qhead_fit.py::test_planted_conditional_coverage_follows_the_state"]),
    ("fit_gradcheck_loss_detached", "qhead_fit.py", "loss = pinball(head(x)[:, 0, 0], yy, q)", "loss = pinball(head(x)[:, 0, 0].detach(), yy, q)",
     ["test_qhead_fit.py::test_grad_check_h6_nonzero_new_exactly_zero_outside"]),
    ("fit_gradcheck_control_param_in_graph", "qhead_fit.py", "loss = pinball(head(x)[:, 0, 0], yy, q)", "loss = pinball(head(x)[:, 0, 0], yy, q) + ctrl.sum()",
     ["test_qhead_fit.py::test_grad_check_h6_nonzero_new_exactly_zero_outside"]),
    ("fit_gradcheck_banked_head_in_graph", "qhead_fit.py", "loss = pinball(head(x)[:, 0, 0], yy, q)", "loss = pinball(head(x)[:, 0, 0], yy, q) + sum(p.sum() for p in held.values())",
     ["test_qhead_fit.py::test_grad_check_h6_nonzero_new_exactly_zero_outside"]),
    ("fit_accepts_evaluation_paths", "qhead_model.py", "    return any(\"eval\" in s.lower() for s in parts[-2:])", "    return False",
     ["test_qhead_fit.py::test_h4_refuses_evaluation_split_paths"]),
    ("fit_uses_the_held_out_clips", "qhead_fit.py", "    fit_m = ~held\n", "    fit_m = np.ones_like(held)\n",
     ["test_qhead_fit.py::test_main_end_to_end_writes_head_and_record"]),
    ("rule_gap_used_is_the_head_read", "v8f_t1_eval_rule_qhead.py", "used = QM.used_gap(g, q)", "used = g",
     ["test_rule_qhead.py::test_used_never_above_head_and_cuts_somewhere", "test_rule_qhead.py::test_gap_used_is_min_of_head_read_and_quantile_read"]),
    ("rule_gap_used_is_max", "qhead_model.py", "np.where(np.isfinite(q), np.minimum(g, q), g)", "np.where(np.isfinite(q), np.maximum(g, q), g)",
     ["test_rule_qhead.py::test_used_never_above_head_and_cuts_somewhere"]),
    ("rule_quantile_reads_the_first_frames", "v8f_t1_eval_rule_qhead.py", "        f = feats[:, -hist:]\n", "        f = feats[:, :hist]\n",
     ["test_rule_qhead.py::test_q_reads_the_last_five_frames_not_others"]),
    ("rule_equals_head_mode_is_not_the_head_read", "v8f_t1_eval_rule_qhead.py", "return np.array(g_head, copy=True)", "return np.array(g_head, copy=True) - 1.0",
     ["test_rule_qhead.py::test_equals_head_mode_gives_head_read_bit_for_bit"]),
    ("rule_checkpoint_check_skipped", "v8f_t1_eval_rule_qhead.py", "            if sha != rec[\"states_record\"][\"ckpt_sha256\"]:", "            if False:",
     ["test_rule_qhead.py::test_main_refuses_checkpoint_mismatch_and_tampered_head"]),
    ("rule_threshold_not_installed", "v8f_t1_eval_rule_qhead.py", "    W.HEAD_P_MIN = float(mine.head_p_min)\n", "    pass\n",
     ["test_rule_qhead.py::test_main_installs_the_read_and_writes_side_file"]),
    ("h3_recomputed_min_is_max", "qhead_h3.py", "u_re = np.minimum(g_re, q_re)", "u_re = np.maximum(g_re, q_re)",
     ["test_h3_names.py::test_h3_passes_on_consistent_data_and_reports_beside_numbers"]),
    ("names_sign_reversed", "qhead_names.py", "n[\"U_against_map\"] = \"QHEAD-CUTS-MORE\" if both(uqc, -1)", "n[\"U_against_map\"] = \"QHEAD-CUTS-MORE\" if both(uqc, +1)",
     ["test_h3_names.py::test_names_all_controls_pass_gives_names_and_beside"]),
    ("names_given_despite_failed_control", "qhead_names.py", "    allp = all(v[\"PASS\"] for v in C.values())\n", "    allp = True\n",
     ["test_h3_names.py::test_failed_control_withholds_names_and_differences"]),
    ("queue_slot_ignores_gpu_count", "pod_qhead_queue.sh", "[ \"$n\" -lt \"$MAX_GPU_JOBS\" ] && [ \"$g\" -gt \"$MIN_FREE_GB\" ]", "[ \"$g\" -gt \"$MIN_FREE_GB\" ]",
     ["test_queue.py::test_slot_wait_refuses_a_fourth_gpu_job"]),
    ("queue_second_start_allowed", "pod_qhead_queue.sh", [("for m in QHEAD_DONE.txt QHEAD_STOP.txt QHEAD_STARTED.txt; do", "for m in QHEAD_DONE.txt QHEAD_STOP.txt; do"), ("set -o noclobber; ", "")],
     ["test_queue.py::test_second_start_is_refused_by_marker"]),
]


def prepare():
    os.makedirs(WORK, exist_ok=True)
    for f in SOURCES:
        shutil.copyfile(os.path.join(HERE, f), os.path.join(WORK, f))


def pytest(nodes, extra):
    env = dict(os.environ, QHEAD_SRC=WORK, DISTRULE_DIR=os.path.normpath(os.path.join(HERE, "..", "distrule")), CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "--basetemp", BASETEMP, "-p", "no:cacheprovider"] + nodes + extra
    r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=900)
    return r.returncode, r.stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a, extra = ap.parse_known_args(argv)
    os.makedirs(BASETEMP, exist_ok=True)
    res = {"card": "PREREG_V8F_PROOF_ADD42", "mutations": [], "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for ent in M:
        name, fn, old = ent[:3]
        new, tests = (ent[3], ent[4]) if len(ent) == 5 else (None, ent[3])
        if a.only and a.only != name:
            continue
        prepare()
        p = os.path.join(WORK, fn)
        s = open(p, encoding="utf-8", newline="").read()
        pairs = old if isinstance(old, list) else [(old, new)]
        n = min(s.count(o) for o, _ in pairs)
        if any(s.count(o) != 1 for o, _ in pairs):
            res["mutations"].append({"name": name, "file": fn, "applied": False, "why": "old text occurs %d times" % n, "red": False})
            print("MUTATION %-45s NOT APPLIED (old text %d times)" % (name, n), flush=True)
            continue
        for o, nw in pairs:
            s = s.replace(o, nw)
        open(p, "w", encoding="utf-8", newline="").write(s)
        rc, out = pytest(tests, extra)
        failed = [l.split(" ")[1].split(" - ")[0] for l in out.splitlines() if l.startswith("FAILED ")]
        red = bool(rc != 0 and all(any(t.split("::")[1] in f for f in failed) for t in tests))
        res["mutations"].append({"name": name, "file": fn, "applied": True, "tests": tests, "red": red, "failed": failed, "rc": rc})
        print("MUTATION %-45s %s (%s)" % (name, "RED" if red else "NOT RED", ", ".join(f.split("::")[-1] for f in failed)[:140]), flush=True)
    if not a.only:
        prepare()
        rc, out = pytest(list(TESTS), extra)
        res["control_unmutated"] = {"rc": rc, "green": rc == 0, "summary": out.strip().splitlines()[-1] if out.strip() else ""}
        print("CONTROL unmutated suite:", res["control_unmutated"], flush=True)
    res["n_mutations"] = len([m for m in res["mutations"] if m.get("applied")])
    res["n_red"] = len([m for m in res["mutations"] if m.get("red")])
    res["all_ok"] = bool(not a.only and res["n_mutations"] == len(M) and res["n_red"] == len(M) and res["control_unmutated"]["green"])
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("n_mutations %d red %d all_ok %s" % (res["n_mutations"], res["n_red"], res["all_ok"]), flush=True)
    return 0 if res["all_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
