"""Turn-asymmetry panel: wait for the FOREIGN arms, then run my two rounds.

⛔ THE GATE COUNTS ARMS, NOT PROCESSES. One arm is 2-4 `python.exe` entries (a
parent and its children, all carrying the full command line), so a gate on
"processes <= 2" can never open. The identity of an arm here is its
`--dump-dir`, which is unique per arm and is what the count is taken over.

⛔ AND THE GATE IS SELF-IMMUNE: my own dump dirs are excluded by name, so once
round 1 launches the waiter does not deadlock on itself.

⛔ It gates on the ABSENCE of foreign arms, never on a success marker: an arm
that dies leaves no marker and a marker-waiter would poll forever
(`wait-loops-must-match-failure`). A probe that FAILS is not read as "no arms".
"""
import json
import os
import re
import subprocess
import sys
import time

PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
TOOL = "C:/Users/Admin/tanitad-wt/taniteval/tools/refav1_arm.py"
P4 = "C:/Users/Admin/refav1_margin/p4out"
SP = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(SP, "panel_turn75.json")
VEND = "64.29715042415070"

COMMON = [
    "--ckpt", "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt",
    "--config", "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json",
    "--cache", "C:/Users/Admin/refav1_margin/p4/fp8",
    "--episodes", "C:/Users/Admin/refav1_margin/p4/eps",
    "--labels", "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz",
    "--nav", "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz",
    "--device", "cuda", "--episodes-n", "0",
    "--window-list", PANEL,
    "--no-navshuf", "--no-lead-block", "--cost-metric", "ccos",
]

#: (tag, W_KAPPA, plan_seed). Round 1 answers the PI's question on its own;
#: round 2 makes it a claim about the PENALTY rather than about `wk15`.
ROUNDS = [
    [("ta_wk15_s0", "15.11245", "0"), ("ta_wk15_s1", "15.11245", "1")],
    [("ta_ccos_s0", "0.0", "0"), ("ta_ccos_s1", "0.0", "1")],
]
MINE = {t for r in ROUNDS for t, _, _ in r}

PS = ["powershell.exe", "-NoProfile", "-Command",
      "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "ForEach-Object { $_.CommandLine }"]


def live_arms():
    """(set of foreign arm dump-dir basenames, set of MY live tags) or None."""
    try:
        r = subprocess.run(PS, capture_output=True, text=True, timeout=180)
    except Exception:                                            # noqa: BLE001
        return None
    tags = set()
    for line in r.stdout.splitlines():
        if "refav1_arm" not in line:
            continue
        m = re.search(r"--dump-dir\s+(\S+)", line)
        if m:
            tags.add(os.path.basename(m.group(1).strip('"'))[5:])   # strip dump_
    return tags


def launch(tag, wk, seed):
    log = os.path.join(P4, tag + ".log")
    cmd = ([PY, TOOL] + COMMON +
           ["--cost-weights", "0.0,%s,%s" % (wk, VEND),
            "--plan-seed", seed,
            "--dump-dir", "%s/dump_%s" % (P4, tag),
            "--out", "%s/rec_%s.json" % (P4, tag),
            "--arm", "refav1-21109-turnasym-" + tag])
    env = dict(os.environ, OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8",
               PYTHONPATH="C:/Users/Admin/tanitad-wt/stack;"
                          "C:/Users/Admin/tanitad-wt/taniteval")
    fh = open(log, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                         cwd="C:/Users/Admin/tanitad-wt")
    print("%s  LAUNCHED %s pid=%d -> %s" % (ts(), tag, p.pid, log), flush=True)
    return p


def ts():
    return time.strftime("%H:%M:%S")


def wait_for_free(exclude, deadline):
    """Block until no FOREIGN arm is live. `exclude` are tags that are mine."""
    while time.time() < deadline:
        tags = live_arms()
        if tags is None:
            print("%s  probe FAILED -- NOT reading that as 'no arms'" % ts(), flush=True)
            time.sleep(90)
            continue
        foreign = sorted(t for t in tags if t not in exclude)
        if not foreign:
            print("%s  GPU free (0 foreign arms)" % ts(), flush=True)
            return True
        print("%s  waiting on %d foreign arm(s): %s" % (ts(), len(foreign), foreign),
              flush=True)
        time.sleep(120)
    return False


def main():
    assert os.path.isfile(PANEL), PANEL
    doc = json.load(open(PANEL))
    print("panel: %d windows, strata %s" % (doc["n"], doc["strata"]), flush=True)
    deadline = time.time() + 6 * 3600
    for ri, round_ in enumerate(ROUNDS, 1):
        if not wait_for_free(MINE, deadline):
            print("%s  DEADLINE -- round %d not started" % (ts(), ri), flush=True)
            return 2
        procs = [launch(t, wk, s) for t, wk, s in round_]
        for p in procs:
            p.wait()
        for t, _, _ in round_:
            rec = "%s/rec_%s.json" % (P4, t)
            ok = os.path.isfile(rec) and os.path.getsize(rec) > 0
            if ok:
                try:
                    json.load(open(rec, encoding="utf-8"))
                except Exception as e:                           # noqa: BLE001
                    ok = "UNPARSEABLE: %s" % e
            print("%s  round %d %s record: %s" % (ts(), ri, t, ok), flush=True)
        print("ZZROUND%dDONEZZ" % ri, flush=True)
    print("ZZALLDONEZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
