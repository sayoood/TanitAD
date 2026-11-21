"""Turn-asymmetry panel, SLOT-BASED gate (replaces ta_queue.py).

⛔ WHY v2 EXISTS. v1 gated on the ABSENCE of every foreign arm and then launched
TWO of mine. MEASURED 2026-09-05 23:05: the sibling stream keeps feeding new arms
(`combined` -> `wk15_ladder` -> `lonshift`), so a zero-foreign gate can STARVE
INDEFINITELY while the box is only half loaded. And launching two of mine on top
of a live foreign arm would break the measured 2-arm ceiling.

⇒ v2 gates on SLOTS: the box runs at most `MAX_ARMS` arms, counted as ARMS (a
dump-dir), never as processes -- one arm is 2-4 `python.exe` entries and a gate
on processes can never open. It launches ONE of mine per free slot, in priority
order, so a killed run still yields the arm that answers the PI's question.

⚠️ The file is NEW, not an edit of the running one: bash and python both read a
script lazily enough that editing a live one is how a shell executes garbage.
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
POLL_S = 15                       # tight: the sibling stream re-fills a freed slot within seconds
MAX_ARMS = 2                      # the measured dev-box ceiling

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

#: priority order. Round 1 (the seed pair at wk15) answers the PI's question on
#: its own; round 2 makes it a claim about the PENALTY rather than about wk15.
PLAN = [("ta_wk15_s0", "15.11245", "0"),
        ("ta_wk15_s1", "15.11245", "1"),
        ("ta_ccos_s0", "0.0", "0"),
        ("ta_ccos_s1", "0.0", "1")]
MINE = {t for t, _, _ in PLAN}

PS = ["powershell.exe", "-NoProfile", "-Command",
      "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "ForEach-Object { $_.CommandLine }"]


def ts():
    return time.strftime("%H:%M:%S")


def live_arms():
    """set of live arm tags (by --dump-dir basename), or None if the probe failed."""
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
            tags.add(os.path.basename(m.group(1).strip('"'))[5:])
    return tags


def launch(tag, wk, seed):
    log = os.path.join(P4, tag + ".log")
    cmd = ([PY, TOOL] + COMMON +
           ["--cost-weights", "0.0,%s,%s" % (wk, VEND), "--plan-seed", seed,
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


def record_ok(tag):
    rec = "%s/rec_%s.json" % (P4, tag)
    if not (os.path.isfile(rec) and os.path.getsize(rec) > 0):
        return False
    try:
        json.load(open(rec, encoding="utf-8"))
        return True
    except Exception:                                            # noqa: BLE001
        return False


def main():
    assert os.path.isfile(PANEL), PANEL
    print("panel: %s" % json.load(open(PANEL))["strata"], flush=True)
    todo = [x for x in PLAN if not record_ok(x[0])]
    running = {}                                    # tag -> Popen
    last_line = [""]
    deadline = time.time() + 9 * 3600
    while (todo or running) and time.time() < deadline:
        for tag in [t for t, p in running.items() if p.poll() is not None]:
            running.pop(tag)
            print("%s  FINISHED %s record=%s" % (ts(), tag, record_ok(tag)),
                  flush=True)
            print("ZZARMDONE-%sZZ" % tag, flush=True)
        tags = live_arms()
        if tags is None:
            print("%s  probe FAILED -- NOT reading that as 'no arms'" % ts(),
                  flush=True)
        elif todo:
            free = MAX_ARMS - len(tags)
            if free > 0:
                tag, wk, seed = todo.pop(0)
                running[tag] = launch(tag, wk, seed)
            else:
                line = "%d/%d slots busy (%s); %d of mine queued" % (
                    len(tags), MAX_ARMS, sorted(tags), len(todo))
                if line != last_line[0]:          # a 15 s poll must not spam
                    print("%s  %s" % (ts(), line), flush=True)
                    last_line[0] = line
        time.sleep(POLL_S)
    print("%s  ALL DONE: %s" % (ts(), {t: record_ok(t) for t, _, _ in PLAN}),
          flush=True)
    print("ZZALLDONEZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
