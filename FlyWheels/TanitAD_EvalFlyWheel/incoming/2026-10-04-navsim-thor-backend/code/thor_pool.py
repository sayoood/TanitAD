#!/usr/bin/env python3
"""Thor NavSim CPU scoring pool: <= 6 scorers, RAM-gated launches, a PID-only watchdog.  (any python3)

    python3 thor_pool.py --jobs /dev/shm/navsim/pool/jobs.jsonl --state /dev/shm/navsim/pool/state.json

``jobs.jsonl``: one JSON object per line, ``{"name": "...", "argv": ["/path/python", "driver.py", ...],
"cwd": "...", "log": "..."}``. Each job is launched as ``nice -n 19 ionice -c3 setsid <argv>`` with
``CUDA_VISIBLE_DEVICES=""`` (children inherit niceness, io class and the empty GPU list).

POLICY (EvalFlyWheel brief 2026-10-04, all scorer-side -- this process never touches a PID it did
not start, and never pattern-matches a process table):
  ⚠ DEFAULTS TIGHTENED 2026-10-04 by the coordinator after a GLOBAL kernel OOM on Thor at 22:57:19
  (another agent's 16.4 GB CPU process): 4 scorers, launch floor 35 GB, SIGTERM floor 20 GB.
  (a) at most ``--max-jobs`` (default 4) jobs alive at once;
  (b) a job is LAUNCHED only if ``MemAvailable >= --launch-min-gb`` (default 35) AND at least
      ``--settle-s`` seconds have passed since the previous launch (a scorer's RSS grows during its
      scene-loader init, so back-to-back launches would all read the same pre-launch number);
  (c) WATCHDOG every ``--poll-s``: if ``MemAvailable < --kill-below-gb`` (default 20) every live job
      tree is sent SIGTERM -- by the PIDs this pool recorded plus their descendants read from
      ``/proc/<pid>/task/*/children`` (i.e. processes it started), never by name;
  (d) if ``/home/nvidia/NAVSIM_YIELD`` exists: launch nothing, SIGTERM live jobs the same way, exit.
A job killed by (c)/(d) is recorded ``KILLED_<why>`` and is NOT relaunched (its own count guards
refuse the partial output anyway); ``--requeue-killed`` re-adds it once memory recovers.
⚠ On Thor, MemAvailable does not see GPU allocations reliably (CLAUDE.md Thor trap) -- that is why
the thresholds are wide (30 GB to launch, 15 GB to kill) rather than tight.

State: ``state.json`` (rewritten every poll) + an opaque one-line marker printed every poll,
``ZZ<done>-<fail>-<alive>-<queued>-<availGB>ZZ``, so a client never greps for words its own command
line contains.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time

YIELD = "/home/nvidia/NAVSIM_YIELD"


def mem_available_gb() -> float:
    with open("/proc/meminfo", encoding="ascii") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 2**20
    raise RuntimeError("MemAvailable missing")


def descendants(pid: int) -> list:
    """Descendants of ``pid`` from the PPID field of ``/proc/<n>/stat`` (a parent-pointer tree walk, NOT a
    pattern match on command lines).

    ⛔ FIXED 2026-10-05: the first version read ``/proc/<p>/task/<t>/children``, which DOES NOT EXIST on Thor's
    kernel (CONFIG_PROC_CHILDREN off; MEASURED: ``ls /proc/self/task/*/children`` -> No such file). It
    silently returned [] and the watchdog SIGTERMed only the driver, ORPHANING the devkit wrapper that held
    the RAM. Caught when two orphans kept appending to a rerun's call log."""
    ppid = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat", encoding="ascii", errors="replace") as fh:
                s = fh.read()
            ppid[int(d)] = int(s[s.rindex(")") + 2:].split()[1])
        except (OSError, ValueError):
            pass
    kids = {}
    for c, p in ppid.items():
        kids.setdefault(p, []).append(c)
    out, stack = [], [pid]
    while stack:
        for k in kids.get(stack.pop(), []):
            out.append(k)
            stack.append(k)
    return out


def term_tree(pid: int) -> list:
    """SIGTERM the job's whole PROCESS GROUP (every job is started under ``setsid``, so pgid == the recorded
    pid, and every subprocess it spawns inherits that group), then -- belt and braces -- every descendant
    found by the PPID walk. Returns the PIDs signalled."""
    sent = []
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGTERM)
            sent.append(-pid)
    except (ProcessLookupError, PermissionError):
        pass
    for p in descendants(pid)[::-1] + [pid]:
        try:
            os.kill(p, signal.SIGTERM)
            sent.append(p)
        except ProcessLookupError:
            pass
    return sent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--max-jobs", type=int, default=4)
    ap.add_argument("--launch-min-gb", type=float, default=35.0)
    ap.add_argument("--kill-below-gb", type=float, default=20.0)
    ap.add_argument("--settle-s", type=float, default=45.0)
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--requeue-killed", action="store_true")
    a = ap.parse_args()
    jobs = [json.loads(ln) for ln in open(a.jobs, encoding="utf-8") if ln.strip()]
    queue = list(jobs)
    live: dict = {}                     # name -> (Popen, job, t_start)
    done: dict = {}                     # name -> record
    events: list = []
    t_last_launch = 0.0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with open(a.state + ".pid", "w", encoding="ascii") as fh:
        fh.write(str(os.getpid()))

    def write_state(avail):
        st = {"t": time.time(), "pool_pid": os.getpid(), "mem_available_gb": round(avail, 2),
              "alive": {n: {"pid": p.pid, "t_start": t0, "descendants": descendants(p.pid)}
                        for n, (p, _j, t0) in live.items()},
              "queued": [j["name"] for j in queue], "done": done, "events": events[-200:],
              "policy": {k: getattr(a, k) for k in ("max_jobs", "launch_min_gb", "kill_below_gb",
                                                    "settle_s", "poll_s")}}
        tmp = a.state + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(st, fh, indent=1)
        os.replace(tmp, a.state)

    yielding = False
    while queue or live:
        avail = mem_available_gb()
        # reap
        for n in list(live):
            p, j, t0 = live[n]
            rc = p.poll()
            if rc is not None:
                rec = done.get(n) or {}
                rec.update({"rc": rc, "wall_s": round(time.time() - t0, 1),
                            "status": rec.get("status", "EXITED")})
                done[n] = rec
                del live[n]
                events.append({"t": time.time(), "event": "exit", "job": n, "rc": rc})
        # (d) yield / (c) watchdog
        why = "NAVSIM_YIELD" if os.path.exists(YIELD) else ("MEM_LT_%gGB" % a.kill_below_gb
                                                          if avail < a.kill_below_gb else None)
        if why:
            for n, (p, j, t0) in live.items():
                if done.get(n, {}).get("status", "").startswith("KILLED"):
                    continue
                sent = term_tree(p.pid)
                done[n] = {"status": f"KILLED_{why}", "signalled": sent, "avail_gb": round(avail, 2)}
                events.append({"t": time.time(), "event": "sigterm", "job": n, "why": why, "pids": sent})
                if a.requeue_killed and why != "NAVSIM_YIELD":
                    queue.append(j)
            if why == "NAVSIM_YIELD":
                yielding = True
                queue = []
        # (a)+(b) launch
        if (not yielding and queue and len(live) < a.max_jobs and avail >= a.launch_min_gb
                and time.time() - t_last_launch >= a.settle_s and not os.path.exists(YIELD)):
            j = queue.pop(0)
            logf = open(j["log"], "a", encoding="utf-8")
            cmd = ["nice", "-n", "19", "ionice", "-c3", "setsid"] + j["argv"]
            p = subprocess.Popen(cmd, cwd=j.get("cwd", "/dev/shm/navsim"), env=env, stdout=logf,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            live[j["name"]] = (p, j, time.time())
            t_last_launch = time.time()
            events.append({"t": t_last_launch, "event": "launch", "job": j["name"], "pid": p.pid,
                           "avail_gb": round(avail, 2), "cmd": cmd})
        write_state(avail)
        n_ok = sum(1 for r in done.values() if r.get("rc") == 0 and r.get("status") == "EXITED")
        n_bad = len(done) - n_ok
        print(f"ZZ{n_ok}-{n_bad}-{len(live)}-{len(queue)}-{avail:.0f}ZZ", flush=True)
        time.sleep(a.poll_s)
    write_state(mem_available_gb())
    print("ZZPOOLENDZZ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
