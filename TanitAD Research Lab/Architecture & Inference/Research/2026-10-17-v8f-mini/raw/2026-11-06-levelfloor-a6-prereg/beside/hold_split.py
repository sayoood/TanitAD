#!/usr/bin/env python3
"""hold_split.py -- descriptive, after the A5 read (never deciding): how much of each row's delta sits in windows where the A5 floor asked for more than max(v0, Vc) on a bound tick
(the part a hold floor v_f = min(kappa v_hat, max(v0, Vc)) would change), for Vc in 5.5 and 6.0 m/s.
usage: python -I hold_split.py <out_json>"""
import json, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/a6_diag")
import f_harm_split as H
res = {}
for s in (0, 1):
    base, _ = H.L5.LFO.reused_arrays(s)
    for tag in (f"RILFO100_s{s}", f"RILF100_s{s}", f"RILFN100x10_s{s}"):
        t = np.load(f"{H.OUT}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=False)
        win, v0, bd, vf = t["win"], t["v0"], t["lf_bind"], t["lf_vf"].astype(np.float64)
        nw = int(win.max()) + 1
        arm = H.rr.read_arrays(f"{H.MC}/lf5_read/raw/arrays/{tag}.npz")
        out = {}
        for Vc in (5.5, 6.0):
            above = bd & (vf > np.maximum(v0, Vc) + 0.25)
            w = np.zeros(nw, bool); np.logical_or.at(w, win, above)
            bw = np.zeros(nw, bool); np.logical_or.at(bw, win, bd)
            cls = {"changed_by_hold": w, "bound_unchanged": bw & ~w}
            r = H.split(arm, base["RIC0"], cls)
            out[str(Vc)] = {"share_bound_ticks_above": float(above.sum() / max(1, bd.sum())), "n_windows_changed": int(w.sum()), "rows": r}
            print(tag, "Vc", Vc, "bound ticks above %.3f" % out[str(Vc)]["share_bound_ticks_above"], "windows", int(w.sum()), " ".join("%s %+.4f/%+.4f" % (k[:8], v["changed_by_hold"], v["bound_unchanged"]) for k, v in r.items()))
        res[tag] = out
json.dump(res, open(sys.argv[1], "w", encoding="utf-8", newline="\n"), indent=1)
