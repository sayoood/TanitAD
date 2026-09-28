#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chrome-trace (torch.profiler export) -> per step: wall, GPU busy (union of kernel / memcpy /
memset intervals), and WHERE THE GPU WAS IDLE: every idle gap on the GPU timeline is attributed to
the innermost user annotation (`phase:` / `fn:` / `mod:` scopes the harness inserted) active on the
MAIN (Python) thread at the gap's midpoint. That answers "what is the CPU doing while the GPU
waits" -- the mechanism behind a low GPU-utilisation sample.

Also: GPU kernel time per main-thread scope (kernels whose launching runtime call lies inside the
scope's CPU span, linked by correlation id), i.e. FORWARD cost per scope; backward kernels are
reported per backward-thread annotation (checkpoint recompute scopes) and otherwise as one block.
"""
from __future__ import annotations

import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path


def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main(trace: str, out: str):
    d = json.load(open(trace, encoding="utf-8"))
    ev = d["traceEvents"] if isinstance(d, dict) else d
    steps = sorted((e for e in ev if str(e.get("name", "")).startswith("ProfilerStep#")),
                   key=lambda e: e["ts"])
    main_tid = steps[0]["tid"] if steps else None
    gpu = [(e["ts"], e["ts"] + e.get("dur", 0)) for e in ev
           if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ann = [e for e in ev if e.get("cat") == "user_annotation"
           and str(e.get("name", "")).startswith(("phase:", "fn:", "mod:"))]
    ann_main = [e for e in ann if e.get("tid") == main_tid]
    ann_other = [e for e in ann if e.get("tid") != main_tid]
    # correlation: runtime launch (cpu side) -> kernel
    rt = {}
    for e in ev:
        if e.get("cat") == "cuda_runtime":
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                rt[c] = (e["ts"], e.get("tid"))
    kern = [(e["ts"], e.get("dur", 0), (e.get("args") or {}).get("correlation")) for e in ev
            if e.get("cat") == "kernel"]

    def innermost(anns, t):
        best = None
        for a in anns:
            if a["ts"] <= t <= a["ts"] + a.get("dur", 0):
                if best is None or a.get("dur", 0) < best.get("dur", 0):
                    best = a
        return best["name"] if best else "<no scope>"
    res = {"n_steps": len(steps), "steps": []}
    idle_by_scope_all = defaultdict(float)
    for s in steps:
        t0, t1 = s["ts"], s["ts"] + s["dur"]
        g = union([(max(a, t0), min(b, t1)) for a, b in gpu if b > t0 and a < t1])
        busy = sum(b - a for a, b in g)
        gaps, prev = [], t0
        for a, b in g:
            if a > prev:
                gaps.append((prev, a))
            prev = max(prev, b)
        if prev < t1:
            gaps.append((prev, t1))
        am = [a for a in ann_main if a["ts"] < t1 and a["ts"] + a.get("dur", 0) > t0]
        idle = defaultdict(float)
        for a, b in gaps:
            if b - a < 50:                     # < 50 us: launch jitter, not a phase
                idle["<gaps<50us>"] += b - a
                continue
            # split long gaps at annotation boundaries: sample at 1 ms resolution
            t = a
            while t < b:
                u = min(b, t + 1000)
                idle[innermost(am, (t + u) / 2)] += u - t
                t = u
        for k, v in idle.items():
            idle_by_scope_all[k] += v
        res["steps"].append({"name": s["name"], "wall_ms": s["dur"] / 1e3,
                             "gpu_busy_ms": busy / 1e3, "busy_frac": busy / s["dur"],
                             "idle_ms_by_scope_top": sorted(((k, round(v / 1e3, 2))
                                                             for k, v in idle.items()),
                                                            key=lambda x: -x[1])[:15]})
    # GPU kernel time by the thread that LAUNCHED it: main thread (forward, loss, optimiser) vs
    # the autograd engine thread (backward + checkpoint recompute)
    by_thread = defaultdict(float)
    t_lo = steps[0]["ts"] if steps else 0
    t_hi = (steps[-1]["ts"] + steps[-1]["dur"]) if steps else 0
    for ts, dur, c in kern:
        if not (t_lo <= ts <= t_hi) or c not in rt:
            continue
        by_thread["main" if rt[c][1] == main_tid else "autograd/other"] += dur
    res["kernel_ms_per_step_by_launch_thread"] = {k: round(v / 1e3 / max(len(steps), 1), 1)
                                                  for k, v in by_thread.items()}
    res["idle_ms_per_step_by_scope"] = sorted(((k, round(v / 1e3 / max(len(steps), 1), 2))
                                               for k, v in idle_by_scope_all.items()),
                                              key=lambda x: -x[1])[:30]
    walls = [x["wall_ms"] for x in res["steps"]]
    busys = [x["gpu_busy_ms"] for x in res["steps"]]
    res["wall_ms_mean"] = sum(walls) / max(len(walls), 1)
    res["gpu_busy_ms_mean"] = sum(busys) / max(len(busys), 1)
    res["busy_frac_overall"] = sum(busys) / max(sum(walls), 1e-9)
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_steps", "wall_ms_mean", "gpu_busy_ms_mean",
                                          "busy_frac_overall")}))
    for st in res["steps"]:
        print(st["name"], round(st["wall_ms"]), "busy", round(st["busy_frac"], 3))
    print("idle ms/step by CPU scope:")
    for k, v in res["idle_ms_per_step_by_scope"][:20]:
        print(f"  {v:9.1f}  {k}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
