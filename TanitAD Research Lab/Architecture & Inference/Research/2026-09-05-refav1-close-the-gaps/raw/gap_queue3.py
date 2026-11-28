"""CLOSE-THE-GAPS arm queue v2 -- A3 (goal-conditioned cost) + A1 + A2.

⚠️ NEW FILE, not an edit of v1: python compiles a script at import, so editing a
running one changes nothing, and bash reads lazily enough that editing a live
shell script makes it execute garbage. v1 launched NOTHING before being
replaced (its log shows only `deferring`), so nothing is orphaned.

TWO THINGS CHANGED FROM v1.

1. ⭐ IT RUNS FROM AN ISOLATED TREE, `C:/Users/Admin/tanitad-ctg`, not from the
   live mirror `C:/Users/Admin/tanitad-wt`. The sibling turn-asymmetry queue
   launches its arms from the mirror; putting the new `--w-kappa-by-goal` code
   there would change the code under a stream that never asked for it. The
   isolated tree is a byte copy of the mirror with exactly two files replaced
   (`refa_v1.py`, `refav1_arm.py`) -- and the flag-OFF path is proven
   bit-identical by 394 passing refa_v1 tests plus
   `test_refa_v1_goal_kappa_cost.py::test_a_off_is_bit_identical`, so A1/A2
   remain comparable to the banked frontier.

2. ⭐ THE A3 ARMS ARE FIRST. `gkappa` is the arm that could produce refav1's
   first configuration that drives accurately AND turns; `gkappa_inv` is its
   deliberate-regression control.

DEFERENCE (unchanged, and it is the load-bearing part): this queue launches only
when BOTH (a) fewer than MAX_ARMS arms are live, counted by --dump-dir and never
by process, and (b) `ta_queue3.py`'s launcher is GONE. If both queues launched on
a free slot they would place a third arm on an 8.2 GB card whose measured
per-arm footprint is ~2.8 GB (6737 MiB used by two at 01:11) -- an OOM, not a
slowdown. ⛔ A failed probe is never read as "no arms".

ASCII only in print() (cp1252 dev box).
"""
import json
import os
import re
import subprocess
import sys
import time

PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
ROOT = "C:/Users/Admin/tanitad-ctg"
TOOL = ROOT + "/taniteval/tools/refav1_arm.py"
P4 = "C:/Users/Admin/refav1_margin/p4out"
VEND = "64.29715042415070"
WK15 = "15.11245"
POLL_S = 20
MAX_ARMS = 2
SIBLING = "ta_queue3.py"

COMMON = [
    "--ckpt", "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt",
    "--config", "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json",
    "--cache", "C:/Users/Admin/refav1_margin/p4/fp8",
    "--episodes", "C:/Users/Admin/refav1_margin/p4/eps",
    "--labels", "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz",
    "--nav", "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz",
    "--device", "cuda", "--episodes-n", "0", "--window-stride", "16",
    "--no-navshuf", "--no-lead-block", "--cost-metric", "ccos",
    "--plan-seed", "0",
]

#: PRIORITY ORDER -- a killed queue still yields the arm that answers the PI.
#: (tag, W_KAPPA scalar, extra flags)
#: ⭐ `gkappa` carries the SAME 15.11245 on LANE_KEEP windows that `wk15` did,
#: which makes its LANE_KEEP column a WITHIN-ARM control against the banked
#: `wk15` dump -- a known-value control that costs no extra GPU.
PLAN = [
    # ⭐ RE-ORDERED 01:40: Thor is ALREADY running `G_gkappa` and `G_gkappa_inv`
    # in its two free slots, so the dev box must COMPLEMENT rather than
    # duplicate. A1 and A2 are unanswered; they go first.
    ("kammshift",  "0.0", ["--kamm-mu", "0.7", "--a-sustain-mode", "a0_shift"]),
    ("wk1",        "1.0", []),
    # `gkappa` stays on the list AFTER them, deliberately: a dev-box copy is a
    # CROSS-RIG REPLICATE of the A3 finding and, unlike the Thor arm, it can be
    # paired in-rig against `ccos_argmax`, `kamm07` AND `wk15` -- all banked
    # here. That is worth a GPU hour, but not the FIRST one.
    ("gkappa",     WK15, ["--w-kappa-by-goal", WK15 + ",0.0"]),
    ("wk3",        "3.0", []),
    ("wk7",        "7.0", []),
]

PS = ["powershell.exe", "-NoProfile", "-Command",
      "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "ForEach-Object { $_.CommandLine }"]


def ts():
    return time.strftime("%H:%M:%S")


def probe():
    """(live arm tags, sibling_alive) or (None, None) if the probe failed.

    ONE powershell call answers BOTH questions, so the two facts can never come
    from different seconds and disagree.
    """
    try:
        r = subprocess.run(PS, capture_output=True, text=True, timeout=240)
    except Exception:                                            # noqa: BLE001
        return None, None
    if r.returncode != 0:
        return None, None
    tags, sib = set(), False
    for line in r.stdout.splitlines():
        if SIBLING in line:
            sib = True
        if "refav1_arm" not in line:
            continue
        m = re.search(r"--dump-dir\s+(\S+)", line)
        if m:
            tags.add(os.path.basename(m.group(1).strip('"'))[5:])
    return tags, sib


def record_ok(tag):
    rec = "%s/rec_%s.json" % (P4, tag)
    if not (os.path.isfile(rec) and os.path.getsize(rec) > 0):
        return False
    try:
        json.load(open(rec, encoding="utf-8"))
        return True
    except Exception:                                            # noqa: BLE001
        return False


def launch(tag, wk, extra):
    log = os.path.join(P4, tag + ".log")
    cmd = ([PY, TOOL] + COMMON +
           ["--cost-weights", "0.0,%s,%s" % (wk, VEND)] + extra +
           ["--dump-dir", "%s/dump_%s" % (P4, tag),
            "--out", "%s/rec_%s.json" % (P4, tag),
            "--arm", "refav1-21109-p4-" + tag])
    env = dict(os.environ, OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8",
               PYTHONPATH=ROOT + "/stack;" + ROOT + "/taniteval")
    fh = open(log, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                         cwd=ROOT)
    print("%s  LAUNCHED %s (W_KAPPA=%s extra=%s) pid=%d"
          % (ts(), tag, wk, " ".join(extra), p.pid), flush=True)
    return p


def main():
    assert os.path.isfile(TOOL), TOOL
    todo = [x for x in PLAN if not record_ok(x[0])]
    print("%s  queue v2 start; tool=%s" % (ts(), TOOL), flush=True)
    print("%s  todo=%s" % (ts(), [t for t, _, _ in todo]), flush=True)
    running = {}
    last = [""]
    deadline = time.time() + 11 * 3600
    while (todo or running) and time.time() < deadline:
        for tag in [t for t, p in running.items() if p.poll() is not None]:
            running.pop(tag)
            print("%s  FINISHED %s record=%s" % (ts(), tag, record_ok(tag)),
                  flush=True)
            print("ZZGAPDONE-%sZZ" % tag, flush=True)
        tags, sib = probe()
        if tags is None:
            print("%s  probe FAILED -- NOT reading that as 'no arms'" % ts(),
                  flush=True)
        elif todo:
            if sib:
                line = "deferring: %s alive; %d of mine queued (live=%s)" % (
                    SIBLING, len(todo), sorted(tags))
            elif len(tags) < MAX_ARMS:
                tag, wk, extra = todo.pop(0)
                running[tag] = launch(tag, wk, extra)
                line = ""
                last[0] = ""
            else:
                line = "%d/%d slots busy (%s); %d of mine queued" % (
                    len(tags), MAX_ARMS, sorted(tags), len(todo))
            if line and line != last[0]:
                print("%s  %s" % (ts(), line), flush=True)
                last[0] = line
        time.sleep(POLL_S)
    print("%s  ALL DONE: %s" % (ts(), {t: record_ok(t) for t, _, _ in PLAN}),
          flush=True)
    print("ZZGAPALLDONEZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
