# -*- coding: utf-8 -*-
"""WP-D step 17b -- THE PLANNER-SIDE SEED FLOOR (rewritten in Python, no heredoc).

⚠️ WHY REWRITTEN: the shell version was authored through a QUOTED heredoc, which
ate one backslash level, so `dumps\\$1` became `dumps\$1` -- a LITERAL `$1`. The
25-minute rollout ran perfectly and wrote 40 episodes into a directory named
`dumps$1`. The dump was not lost, but the guard correctly refused to pair,
because it asserts on the ARTIFACT (file count) and not on the exit code, which
was 0. Building the paths in Python removes the quoting layer entirely.

WHY THIS ARM EXISTS: A3 (step 11) shows the REPRESENTATION probe cannot resolve
the lever on this rig. That leaves `F2` -- "the planner regressed, separably, on
8 of 9 T1 metrics" -- as the evidence E-BEV-AUX-1's refutation rests on. ⛔ F2 was
measured on ONE-SEED arms, and CLAUDE.md is explicit that a separated CI from
one-seed arms is NECESSARY AND NOT SUFFICIENT (H-ESTIM-SEED-1). D0b (zero levers
moved) and D0c (seed only) go through the IDENTICAL T1 arm as D0 and D1, so
F2's +0.02610 ADE can be read against the rig's own run-to-run noise.
"""
import os
import subprocess
import sys

PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
TOOL = r"C:\Users\Admin\wpd-run\taniteval\tools\refcv3_arm.py"
EPS = r"C:\Users\Admin\tanitad-data\refav1-eval141\eps"
LAB = r"C:\Users\Admin\wpd-probe\labels\s2_labels_v7.2_eval.jsonl.gz"
LEAD = r"C:\Users\Admin\wpd-probe\leadblk\b1_eval_lead_block.npz"
DUMPS = r"C:\Users\Admin\wpd-probe\dumps"
CKPT = r"C:\Users\Admin\wpd-probe\ckpt"
RAW = r"C:\Users\Admin\wpd-probe\raw"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="6")


def n_eps(arm):
    d = os.path.join(DUMPS, arm)
    return len([f for f in os.listdir(d) if f.startswith("ep") and f.endswith(".npz")]) \
        if os.path.isdir(d) else 0


def dump(arm, ck, cfg):
    have = n_eps(arm)
    if have == 40:
        print(f"[17b] {arm}: 40/40 already banked -- SKIPPING the rollout "
              f"(re-running would burn 23 GPU-min to reproduce what is on disk)")
        return True
    out = os.path.join(DUMPS, arm)
    print(f"[17b] === DUMP {arm} (have {have}/40) ===", flush=True)
    cmd = [PY, "-u", TOOL, "--ckpt", ck, "--config", cfg, "--episodes", EPS,
           "--labels", LAB, "--lead-block", LEAD, "--arm", arm, "--grid", "2s",
           "--device", "cuda", "--episodes-n", "40", "--window-stride", "2",
           "--dump-dir", out, "--dump-only",
           "--out", os.path.join(RAW, arm + ".json")]
    log = os.path.join(RAW, f"planner_{arm}.log")
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=ENV)
    got = n_eps(arm)
    # ⛔ ASSERT ON THE ARTIFACT, NEVER THE EXIT CODE. rc was 0 for the run that
    # produced 0 files; the file count is the only thing that settles it.
    print(f"[17b] {arm}: rc={rc} FILES={got}/40 log={log}")
    if got != 40:
        print(f"[17b] REFUSING to pair on a partial dump ({got}/40)")
        return False
    return True


ok = True
ok &= dump("wpdD0b", os.path.join(CKPT, "ckpt_D0b.pt"), os.path.join(CKPT, "config_D0b.json"))
ok &= dump("wpdD0c", os.path.join(CKPT, "ckpt_D0c.pt"), os.path.join(CKPT, "config_D0c.json"))
if not ok:
    sys.exit(7)

for arm in ("wpdD0b", "wpdD0c"):
    print(f"\n[17b] === PAIR {arm} vs wpdD0 -- THE PLANNER FLOOR ===", flush=True)
    out = os.path.join(RAW, f"paired_floor_{arm}_vs_D0.json")
    rc = subprocess.call(
        [PY, os.path.join(r"C:\Users\Admin\wpd-probe\code", "b9_pair_5b.py"),
         "--a", os.path.join(DUMPS, arm), "--b", os.path.join(DUMPS, "wpdD0"),
         "--n-boot", "2000", "--out", out], env=ENV)
    print(f"[17b] pair {arm} rc={rc} artifact_exists={os.path.exists(out)}")
print("B17B_DONE")
