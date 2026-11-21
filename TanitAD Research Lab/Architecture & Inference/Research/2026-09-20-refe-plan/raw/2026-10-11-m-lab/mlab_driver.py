#!/usr/bin/env python3
"""M-LAB driver: runs the measurement arms in sequence, each as N independent worker PROCESSES (mlab_worker.py), while a
sampler records (1 Hz) every worker's RSS and the machine's busy logical CPUs minus the workers' own share (= the background
load from every other process: C3 harness scoring, the video render, ...). Nothing else on the box is touched.

    python mlab_driver.py [--only <arm-tag>] [--list]         resumable: an arm whose runs/<tag>/summary.json exists is skipped
Arms are listed in ARMS in priority order, so a stopped chain still yields the primary numbers first.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
V11 = "E:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
MAPS = "E:/Archive/devbox-C/navsim/data/maps"
NCPU = psutil.cpu_count(logical=True)

# (tag, plan, mode, nworkers, extra worker flags)
ARMS = (
    [(f"cached_r{r}_n1", f"cached_r{r}", "cached", 1, []) for r in (1, 2, 3)]
    + [(f"build_r{r}_n1", f"build_r{r}", "build", 1, []) for r in (1, 2, 3)]
    + [(f"cached_r1_n{n}", "cached_r1", "cached", n, []) for n in (2, 4, 8)]
    + [(f"build_r3_n{n}", "build_r3", "build", n, []) for n in (2, 4)]
    + [("cached_r2_retain_n1", "cached_r2", "cached", 1, ["--retain"])]
    + [(f"single_r{r}_n1", f"single_r{r}", "build", 1, ["--single"]) for r in (1, 2, 3)]
    # added after the first arms showed a loaded box (6-23 background logical CPUs): more independent runs, and these
    # (and cached_r3 onwards) record the per-set background load, so c can be read against load
    # r4-r6 are PINNED to one P-core thread each (logical 4, 6, 8 = thread 0 of three distinct P-cores; E-cores are logical 16-23 on
    # this 12900F), so a run is not moved onto a slower E-core by a saturated box; the sibling thread is whatever the box runs there
    + [(f"cached_r{r}_n1", f"cached_r{r}", "cached", 1, ["--pin", str(p)]) for r, p in ((4, 4), (5, 6), (6, 8))]
    # paired pinned-vs-unpinned on IDENTICAL tokens: the pinned replays of cached_r1 and build_r1
    + [("cached_r1p_n1", "cached_r1", "cached", 1, ["--pin", "10"]), ("build_r1p_n1", "build_r1", "build", 1, ["--pin", "12"])]
)


def worker_env():
    e = dict(os.environ)
    e.update(PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
             NUMEXPR_NUM_THREADS="1", REFE_SIM_HZ="10", NUPLAN_MAPS_ROOT=MAPS, NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
             PYTHONPATH=f"{V11};{os.path.join(PKG, 'refe')}")
    return e


def snapshot_others(skip_pids, window=2.0, top=8):
    """CPU% (of one logical CPU) of every other process over a short window; python ones named."""
    procs = {}
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if p.pid in skip_pids:
                continue
            procs[p.pid] = (p, p.cpu_times(), p.info["name"])
        except (psutil.Error, OSError):
            pass
    t0 = time.perf_counter()
    time.sleep(window)
    dt = time.perf_counter() - t0
    rows = []
    for pid, (p, c0, name) in procs.items():
        try:
            c1 = p.cpu_times()
            u = ((c1.user + c1.system) - (c0.user + c0.system)) / dt * 100.0
            if u >= 2.0:
                cmd = ""
                try:
                    cmd = " ".join(p.cmdline())[:150]
                except (psutil.Error, OSError):
                    pass
                rows.append({"pid": pid, "name": name, "cpu_pct_of_1_logical": round(u, 1), "cmd": cmd})
        except (psutil.Error, OSError):
            pass
    rows.sort(key=lambda r: -r["cpu_pct_of_1_logical"])
    return rows[:top]


class Sampler(threading.Thread):
    def __init__(self, pids):
        super().__init__(daemon=True)
        self.pids = pids
        self.stop = threading.Event()
        self.rows = []

    def resolve(self, pid):
        """Windows venv python.exe is a LAUNCHER STUB that spawns the real interpreter as a child; the process that does
        the work (and that mlab_worker.py measures) is the child. Sample the child when it exists, else the stub."""
        try:
            p = psutil.Process(pid)
            kids = p.children(recursive=True)
            return kids[0] if kids else p
        except psutil.Error:
            return None

    def run(self):
        procs, last_w = {}, {}
        last_sys = psutil.cpu_times()
        t_last = time.perf_counter()
        while not self.stop.wait(1.0):
            now = time.perf_counter()
            dt = now - t_last
            sy = psutil.cpu_times()
            busy = ((sy.user + sy.system) - (last_sys.user + last_sys.system)) / dt        # busy logical CPUs, system-wide
            wk, rss = 0.0, {}
            for lp in self.pids:
                p = procs.get(lp)
                if p is None or p.pid == lp:                                             # not yet resolved to the real child
                    q = self.resolve(lp)
                    if q is not None and (p is None or q.pid != p.pid):
                        procs[lp] = p = q
                        last_w.pop(lp, None)
                if p is None:
                    continue
                try:
                    c = p.cpu_times()
                    cur = c.user + c.system
                    wk += (cur - last_w.get(lp, cur)) / dt
                    last_w[lp] = cur
                    rss[p.pid] = p.memory_info().rss
                except psutil.Error:
                    pass
            self.rows.append({"t": round(now, 2), "busy_logical": round(busy, 2), "workers_logical": round(wk, 2),
                              "background_logical": round(max(busy - wk, 0.0), 2), "rss": rss,
                              "mem_available_mb": round(psutil.virtual_memory().available / 1e6)})
            last_sys, t_last = sy, now


def run_arm(tag, plan, mode, n, extra):
    out_dir = os.path.join(HERE, "runs", tag)
    summ = os.path.join(out_dir, "summary.json")
    if os.path.exists(summ):
        print(f"[skip] {tag} (summary exists)", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    env = worker_env()
    bg_before = snapshot_others(set())
    procs, outs = [], []
    t0 = time.perf_counter()
    for i in range(n):
        o = os.path.join(out_dir, f"worker{i}.json")
        outs.append(o)
        cmd = [NV_PY, os.path.join(HERE, "mlab_worker.py"), "--plan", plan, "--shard", str(i), "--nshards", str(n),
               "--mode", mode, "--out", o] + extra
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(os.path.join(out_dir, f"worker{i}.log"), "w"),
                                      stderr=subprocess.STDOUT))
    sm = Sampler([p.pid for p in procs])
    sm.start()
    rcs = [p.wait() for p in procs]
    wall = time.perf_counter() - t0
    sm.stop.set()
    sm.join(3)
    bg_after = snapshot_others({p.pid for p in procs})
    res = {"tag": tag, "plan": plan, "mode": mode, "nworkers": n, "extra": extra, "rcs": rcs, "wall_s": wall,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ncpu_logical": NCPU,
           "bg_top_before": bg_before, "bg_top_after": bg_after,
           "samples": sm.rows, "worker_outputs": [os.path.basename(o) for o in outs]}
    json.dump(res, open(summ, "w"), indent=0)
    bgl = [r["background_logical"] for r in sm.rows] or [0]
    print(f"[done] {tag} n={n} wall={wall:.0f}s rcs={rcs} background busy logical CPUs mean={sum(bgl) / len(bgl):.2f} "
          f"max={max(bgl):.2f} (of {NCPU})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for t in ARMS:
            print(t)
        return 0
    for tag, plan, mode, n, extra in ARMS:
        if a.only and a.only != tag:
            continue
        run_arm(tag, plan, mode, n, extra)
    print("ZZMLAB_CHAIN_DONE", time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
