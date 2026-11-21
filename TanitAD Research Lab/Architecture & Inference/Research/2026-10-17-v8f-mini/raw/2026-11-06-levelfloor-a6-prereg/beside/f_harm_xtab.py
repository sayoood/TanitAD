#!/usr/bin/env python3
"""f_harm_xtab.py -- descriptive cross-tab after the A5 read (never deciding): per window, does the floor bind on a tick whose own v0 >= V (the rollout's measured speed at that replan)?
Windows whose bound ticks all have v0 < V approximate what a per-tick gate v0 < V would keep. Also splits those kept windows by the human's state at the bound ticks.
usage: python -I f_harm_xtab.py <out_json>"""
import json, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/a6_diag")
import f_harm_split as H
res = {}
for s in (0, 1):
    base, _ = H.L5.LFO.reused_arrays(s)
    for tag in (f"RILFO100_s{s}", f"RILF100_s{s}"):
        t = np.load(f"{H.OUT}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=False)
        vlog, ok = H.LFR.logged_speed(t["gt"], t["gt_ok"])
        win, v0, bd = t["win"], t["v0"], t["lf_bind"]
        nw = int(win.max()) + 1
        arm = H.rr.read_arrays(f"{H.MC}/lf5_read/raw/arrays/{tag}.npz")
        out = {}
        for V in (5.2, 6.0, 7.0):
            hi = np.zeros(nw, bool); lo = np.zeros(nw, bool); lo_slow = np.zeros(nw, bool); lo_go = np.zeros(nw, bool)
            np.logical_or.at(hi, win, bd & (v0 >= V))
            np.logical_or.at(lo, win, bd & (v0 < V))
            np.logical_or.at(lo_slow, win, bd & (v0 < V) & ok & (vlog < 3.0))
            np.logical_or.at(lo_go, win, bd & (v0 < V) & ok & (vlog >= 5.2))
            kept = lo & ~hi
            cls = {"kept_by_gate": kept, "has_bound_tick_at_or_above": hi, "kept_human_lt3": kept & lo_slow, "kept_human_ge5.2": kept & lo_go}
            out[str(V)] = {"sizes": {k: int(v.sum()) for k, v in cls.items()}, "rows": H.split(arm, base["RIC0"], cls),
                           "share_bound_ticks_below": float((bd & (v0 < V)).sum() / max(1, bd.sum()))}
        res[tag] = out
        for V, o in out.items():
            r = o["rows"]
            print(tag, "V", V, "bound ticks below V %.3f" % o["share_bound_ticks_below"], o["sizes"], "| kept: ADE6 %+.4f M1 %+.4f DKV %+.4f XT %+.4f | dropped: ADE6 %+.4f M1 %+.4f DKV %+.4f XT %+.4f | kept lt3 ADE6 %+.4f ge5.2 ADE6 %+.4f" % (
                r["ADE6"]["kept_by_gate"], r["M1 crawl share"]["kept_by_gate"], r["DK-VIOL"]["kept_by_gate"], r["cross-track MAE"]["kept_by_gate"],
                r["ADE6"]["has_bound_tick_at_or_above"], r["M1 crawl share"]["has_bound_tick_at_or_above"], r["DK-VIOL"]["has_bound_tick_at_or_above"], r["cross-track MAE"]["has_bound_tick_at_or_above"],
                r["ADE6"]["kept_human_lt3"], r["ADE6"]["kept_human_ge5.2"]))
json.dump(res, open(sys.argv[1], "w", encoding="utf-8", newline="\n"), indent=1)
