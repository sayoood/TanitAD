#!/usr/bin/env python3
"""f_harm_split.py -- descriptive split of the A5 harm rows by window class, computed AFTER the A5 read (never deciding; it chooses which F change the A6 card names).

For a row k the reader's delta is mean over valid windows of (a_k[arm] - a_k[base]). Here the same per-window differences are summed inside window classes and divided by the
same n, so the class contributions add up to the reader's delta exactly. Classes come from the arm's own tick dump (3 ticks per window, aligned by `win`):
  * bound: the floor bound on at least one tick of the window;
  * v0 band at the window's first tick: < 5.2 m/s (the M1 side) or >= 5.2 m/s;
  * human state: logged route speed (lf_read.logged_speed, as A4/A5) < 3 m/s, 3-5.2, >= 5.2, or no complete route;
  * cap: the lead cap fired on a tick of the window;
  * over: on a bound tick, v_f exceeded the logged speed (the estimate's overshoot; zero by construction for LFO100).
Also, per bound tick, whether F's progress target passed the end of O's own 3 s plan (extrapolation along the last segment).
usage: python -I f_harm_split.py <out_json>"""
import importlib.util
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
MC = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
OUT = "F:/TanitAD-scratch/motion_c/lf5_read/out"
ROWS = ("ADE6", "along-track MAE", "speed MAE", "DK-VIOL", "cross-track MAE", "M1 crawl share")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L5 = _load("lf5_read_diag", f"{MC}/lf5_read/code/lf5_read.py")
L5.init()
LFR, rr = L5.LFR, L5.rr


def win_classes(tag):
    t = np.load(f"{OUT}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=False)
    win, v0 = t["win"], t["v0"]
    vlog, ok = LFR.logged_speed(t["gt"], t["gt_ok"]) if "gt_ok" in t.files else LFR.logged_speed(t["gt"])
    nw = int(win.max()) + 1
    first = np.full(nw, -1)
    for i in range(len(win) - 1, -1, -1):
        first[win[i]] = i
    bound = np.zeros(nw, bool); cap = np.zeros(nw, bool); over = np.zeros(nw, bool); extra = np.zeros(nw, bool)
    np.logical_or.at(bound, win, t["lf_bind"])
    np.logical_or.at(cap, win, t["lf_cap"])
    np.logical_or.at(over, win, t["lf_bind"] & ok & (t["lf_vf"] > vlog + 0.25))
    # extrapolation: the floored plan's final along-path distance exceeds O's raw plan arc length by > 0.5 m
    arc = lambda p: np.linalg.norm(np.diff(np.concatenate([np.zeros((p.shape[0], 1, 2)), p], 1), axis=1), axis=-1).sum(1)
    ex = t["lf_bind"] & (arc(t["plan"]) > arc(t["plan_raw"]) + 0.5)
    np.logical_or.at(extra, win, ex)
    v0w, vlw, okw = v0[first], vlog[first], ok[first]
    human = np.where(~okw, "noroute", np.where(vlw < 3.0, "lt3", np.where(vlw < 5.2, "3to5.2", "ge5.2")))
    return {"bound": bound, "cap": cap, "over": over, "extra": extra, "v0lo": v0w < 5.2, "human": human,
            "n_tick_bound": int(t["lf_bind"].sum()), "n_tick_extra": int(ex.sum())}


def split(arm_arr, base_arr, cls):
    labs = [str(x) for x in arm_arr["labels"]]
    res = {}
    for r in ROWS:
        i = labs.index(r)
        a, b = arm_arr[f"a_{i}"], base_arr[f"a_{i}"]
        okp = np.isfinite(a) & np.isfinite(b)
        d = np.where(okp, a - b, 0.0)
        n = okp.sum()
        out = {"delta": float(d.sum() / n), "n": int(n)}
        for name, m in cls.items():
            out[name] = round(float(d[m].sum() / n), 5)
        res[r] = out
    return res


def main():
    res = {"note": __doc__.split("\n\n")[0], "arms": {}}
    for s in (0, 1):
        base, _ = L5.LFO.reused_arrays(s)
        for tag in (f"RILFO100_s{s}", f"RILF100_s{s}", f"RILFN100x10_s{s}"):
            arm = rr.read_arrays(f"{MC}/lf5_read/raw/arrays/{tag}.npz")
            c = win_classes(tag)
            B, V, H = c["bound"], c["v0lo"], c["human"]
            cls = {"bound": B, "unbound": ~B,
                   "bound_v0lt5.2": B & V, "bound_v0ge5.2": B & ~V,
                   "bound_cap": B & c["cap"], "bound_nocap": B & ~c["cap"],
                   "bound_over": B & c["over"], "bound_notover": B & ~c["over"],
                   "bound_extra": B & c["extra"], "bound_noextra": B & ~c["extra"]}
            for h in ("lt3", "3to5.2", "ge5.2", "noroute"):
                cls[f"bound_human_{h}"] = B & (H == h)
            sizes = {k: int(v.sum()) for k, v in cls.items()}
            res["arms"][tag] = {"window_class_sizes": sizes, "n_tick_bound": c["n_tick_bound"], "n_tick_extrapolated": c["n_tick_extra"], "rows_vs_RIC0": split(arm, base["RIC0"], cls)}
    json.dump(res, open(sys.argv[1], "w", encoding="utf-8", newline="\n"), indent=1)
    for tag, v in res["arms"].items():
        print("==", tag, "bound windows", v["window_class_sizes"]["bound"], "ticks bound", v["n_tick_bound"], "extrapolated", v["n_tick_extrapolated"])
        for r, o in v["rows_vs_RIC0"].items():
            print("  %-16s %+.4f | v0<5.2 %+.4f v0>=5.2 %+.4f | cap %+.4f | over %+.4f | extra %+.4f | human lt3 %+.4f 3-5.2 %+.4f ge5.2 %+.4f noroute %+.4f | unbound %+.4f" % (
                r, o["delta"], o["bound_v0lt5.2"], o["bound_v0ge5.2"], o["bound_cap"], o["bound_over"], o["bound_extra"], o["bound_human_lt3"], o["bound_human_3to5.2"], o["bound_human_ge5.2"], o["bound_human_noroute"], o["unbound"]))


if __name__ == "__main__":
    main()
