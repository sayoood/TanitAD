#!/usr/bin/env python3
"""One-screen status of the step-50,400 NavSim scoring (any python with psutil). Reads ARTIFACTS only.

    python code/status50k7.py [--json]

Per split/arm: the count guard's status, valid rows, PDMS/EPDMS summary where the guard reads PASS, the
governor's pauses; the runner / queue tails; the slot files; the GPU lock; free RAM and free commit.
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
MS = os.path.join(PKG, "raw", "milestones", "step50400")
sys.path.insert(0, HERE)


def jget(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:                                                     # noqa: BLE001
        return None


def row(name, c):
    if c is None:
        return f"  {name:42s} (no counts yet)"
    g = c.get("governor") or {}
    s = c.get("summary_x100_4dp") or {}
    # a FAILED guard's summary is PARTIAL work that reads like a result: show a number only beside PASS
    extra = f" PDMS={s.get('PDMS')}" if (s and c.get("status") == "PASS") else ""
    n = c.get("csv_valid_rows")
    return (f"  {name:42s} {c.get('status'):5s} valid_rows={n}{extra}"
            f"  pauses={g.get('n_pauses')} paused_s={g.get('paused_s_total')}")


def main() -> int:
    out = []
    for split, pat in (("navtest (runner)", "scores_navtest/*/*.counts.json"),
                       ("navhard (runner)", "scores_navhard/*.counts.json"),
                       ("warmup (queue)", "scores_warmup/*.counts.json"),
                       ("LEGAL navtest", "vmaxoff_legal/scores_navtest/*/*.counts.json"),
                       ("LEGAL navhard", "vmaxoff_legal/scores_navhard/*.counts.json")):
        out.append(f"{split}:")
        for p in sorted(glob.glob(os.path.join(MS, pat))):
            out.append(row(os.path.basename(p).replace(".counts.json", ""), jget(p)))
    for name in ("runner.log", "scores_warmup/queue.log", "vmaxoff_legal/vmaxoff_legal.log"):
        p = os.path.join(MS, name)
        if os.path.exists(p):
            out.append(f"{name} (tail):")
            out += ["  " + ln.rstrip()[:200] for ln in open(p, encoding="utf-8", errors="replace").readlines()[-4:]]
    try:
        import psutil
        out.append(f"RAM available {psutil.virtual_memory().available / 2 ** 20:.0f} MB")
    except Exception:                                                     # noqa: BLE001
        pass
    sd = "C:/Users/Admin/qland/work/refcv7/navsim_score_slots"
    out.append("slots: " + ", ".join(sorted(os.listdir(sd))) if os.path.isdir(sd) else "slots: -")
    try:
        out.append("GPU lock: " + open("C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock", encoding="utf-8").read()[:200])
    except OSError:
        out.append("GPU lock: free")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
