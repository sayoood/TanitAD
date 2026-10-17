#!/usr/bin/env python3
"""bounded_run.py -- run ONE command under a watchdog: a wall-clock timeout AND a bound on its live descendants.

Why it exists (2026-10-16): running SFT-4R's G-SUITE spawned a RUNAWAY process tree (~80 python processes, RAM fell to 8.6 GB) because a suite script re-entered the
suite that contains it (raw/2026-10-16-sft4r-readiness/READINESS.md, "The recursion").  Every run of these tests goes through here, so a repeat is a bounded, logged event.

usage:  python bounded_run.py --timeout 900 --max-children 20 --log out.txt [--cwd DIR] [--env K=V ...] -- <cmd> [args...]

* the child's stdout+stderr go to --log (a file, so no pipe can fill and stall it);
* the watchdog polls the child's descendants every 0.2 s (psutil);  more than --max-children live descendants, or --timeout seconds, and it is killed;
* KILLS ONLY BY EXPLICIT PID: the pids it collected from the child's own tree, leaves first (psutil.Process(pid).kill()).  Never by image name, never by command-line pattern -- other agents'
  python processes share this box;
* last line printed (also appended to --log):  ZZBOUNDED rc=<rc> peak_children=<n> secs=<s> killed=<none|timeout|children> survivors=<n>
* exit code: the child's own rc;  124 = killed for the timeout;  125 = killed for the descendant bound.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import psutil


def kill_tree(root: psutil.Process, pids_seen: set, log_path: str = "") -> int:
    """Kill the pids in `pids_seen` (+ whatever the tree holds now), leaves first, each by its own PID.  Returns the number still alive afterwards.
    Before the first kill the tree's command lines are appended to `log_path` (what the runaway WAS, as evidence)."""
    victims = []
    try:
        victims = root.children(recursive=True)
    except psutil.Error:
        pass
    if log_path:
        with open(log_path, "ab") as lf:
            lf.write(b"\n--- TREE AT KILL (pid ppid cmd[:170]) ---\n")
            for p in [root] + victims:
                try:
                    cl = " ".join(p.cmdline())[:170]
                    lf.write((f"{p.pid} {p.ppid()} {cl}" + "\n").encode("utf-8", "replace"))
                except psutil.Error:
                    pass
    procs = {p.pid: p for p in victims}
    for pid in pids_seen:
        if pid not in procs:
            try:
                procs[pid] = psutil.Process(pid)
            except psutil.Error:
                pass
    order = sorted(procs.values(), key=lambda p: -p.pid)       # newest pids first ~ leaves first
    for p in order:
        try:
            p.kill()
        except psutil.Error:
            pass
    try:
        root.kill()
    except psutil.Error:
        pass
    time.sleep(0.5)
    alive = 0
    for p in list(procs.values()) + [root]:
        try:
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                alive += 1
        except psutil.Error:
            pass
    return alive


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-children", type=int, default=20)
    ap.add_argument("--log", required=True)
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--env", action="append", default=[], help="K=V (repeatable)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("no command", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    t0 = time.time()
    peak, killed, survivors, seen = 0, "none", 0, set()
    with open(a.log, "ab") as lf:
        proc = subprocess.Popen(cmd, cwd=a.cwd, env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        root = psutil.Process(proc.pid)
        while True:
            rc = proc.poll()
            try:
                kids = root.children(recursive=True)
            except psutil.Error:
                kids = []
            for k in kids:
                seen.add(k.pid)
            n = len(kids)
            peak = max(peak, n)
            if rc is not None:
                break
            if n > a.max_children:
                killed = "children"
                survivors = kill_tree(root, seen, a.log)
                break
            if time.time() - t0 > a.timeout:
                killed = "timeout"
                survivors = kill_tree(root, seen, a.log)
                break
            time.sleep(0.2)
        rc = proc.wait() if killed == "none" else (125 if killed == "children" else 124)
    secs = round(time.time() - t0, 1)
    line = f"ZZBOUNDED rc={rc} peak_children={peak} secs={secs} killed={killed} survivors={survivors}"
    with open(a.log, "ab") as lf:
        lf.write(("\n" + line + "\n").encode())
    print(line, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
