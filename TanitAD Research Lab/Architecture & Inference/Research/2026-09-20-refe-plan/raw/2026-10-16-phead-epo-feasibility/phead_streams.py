#!/usr/bin/env python3
"""data-order sensitivity of the closed-loop head arms: Delta J_G (lam 1 - lam 0, same restriction) for streams 0, 1, 2 (cached-feature path, K = 50), f against the ONLY closed-loop full-arm denominator (stream 0, banked),
and the open-loop f per stream (adam_sim replay, K = 50, regime P) from open_out/.  -> streams.json"""
import glob, json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import phead_common as PC
FR = pickle.load(open(os.path.join(HERE, "full_ref.pkl"), "rb")); POOLS = ("train", "val"); LOGS = FR["logs"]
dF = {p: FR[(1.0, 50)][p]["J"] - FR[(0.0, 50)][p]["J"] for p in POOLS}
out = {"closed_cached_dJ_by_stream": {}, "open_loop_f_by_stream": {}}
for g in ("PH_C_HEAD", "PH_B_SIX", "PH_A_EPROW"):
    for s in (0, 1, 2):
        pa = [os.path.join(HERE, d, f"closed_s{s}_P_{g}_R_lr1_lam{l}{t}.pkl") for l in (1, 0) for d, t in (("closed_cached", ""), )]
        if s == 0:
            pa = [os.path.join(HERE, "closed_out", f"closed_s0_P_{g}_R_lr1_lam{l}.pkl") if os.path.exists(os.path.join(HERE, "closed_out", f"closed_s0_P_{g}_R_lr1_lam{l}.pkl")) else os.path.join(HERE, "closed_cached", f"closed_s0_P_{g}_R_lr1_lam{l}.pkl") for l in (1, 0)]
        if not all(os.path.exists(x) for x in pa):
            continue
        r1, r0 = pickle.load(open(pa[0], "rb")), pickle.load(open(pa[1], "rb"))
        rec = {}
        for p in POOLS:
            d = r1["snaps"][50]["eval"][p]["J"] - r0["snaps"][50]["eval"][p]["J"]
            b = PC.boot_ratio(LOGS[p], d, dF[p])
            rec[p] = {"dJ_G": b["mean_num"], "f_vs_stream0_FULL": b["f"], "f_ci": b["ci_f"]}
        rec["f_min"] = min(rec[p]["f_vs_stream0_FULL"] for p in POOLS)
        out["closed_cached_dJ_by_stream"][f"{g}|s{s}"] = rec
o = {}
for pth in sorted(glob.glob(os.path.join(HERE, "open_out", "open_*.pkl"))):
    o.update(pickle.load(open(pth, "rb"))["runs"])
for (s, rg, g, kind, lrm, lam, k), v in sorted(o.items()):
    if lam != 1.0 or "eval" not in v or kind != "R":
        continue
    r0, f1, f0 = o.get((s, rg, g, kind, lrm, 0.0, k)), o.get((s, rg, "FULL", "R", 1.0, 1.0, k)), o.get((s, rg, "FULL", "R", 1.0, 0.0, k))
    if not (r0 and f1 and f0 and "eval" in r0 and "eval" in f1 and "eval" in f0):
        continue
    rec = {}
    for p in POOLS:
        b = PC.boot_ratio(LOGS[p], v["eval"][p]["J"] - r0["eval"][p]["J"], f1["eval"][p]["J"] - f0["eval"][p]["J"])
        rec[p] = {"dJ_G": b["mean_num"], "dJ_FULL": b["mean_den"], "f": b["f"]}
    rec["f_min"] = min(rec[p]["f"] for p in POOLS)
    out["open_loop_f_by_stream"][f"{rg}|{g}|s{s}|K{k}"] = rec
json.dump(out, open(os.path.join(HERE, "streams.json"), "w"), indent=1)
for k, v in out["closed_cached_dJ_by_stream"].items():
    print("closed", k, {p: round(v[p]["dJ_G"], 4) for p in POOLS}, "f_min", round(v["f_min"], 3))
for k, v in out["open_loop_f_by_stream"].items():
    print("open", k, {p: (round(v[p]["dJ_G"], 4), round(v[p]["dJ_FULL"], 4), round(v[p]["f"], 3)) for p in POOLS}, "f_min", round(v["f_min"], 3))
