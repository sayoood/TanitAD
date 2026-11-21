"""nr_matched_ade_crosscheck.py -- an INDEPENDENT numpy re-computation of the matched-instrument S2 / S6 ADE (no torch, no pair_read_r8, no taniteval).

Reads only the banked panels + extras and the per-tree results of `nr_matched_instrument.py read`, and checks that the OFF read's refcv8 ADE mean and
its paired delta vs `b_refcv7_50400_s0` are what a plain `mean_k ||P_k - G_k||` gives on the OFF plans.  S2 plan source = the banked panel's own
`os_filteroff` arm (built by the battery from the extras; a DIFFERENT code path from this tool's writer); S6 plan source = extras `filter_off.traj`
at the S6 panel's `grid.slots` (the S6 panel carries no filter-off arm).  Tolerance 2e-6 m on a mean (float32 inputs, float64 accumulation).

    python nr_matched_ade_crosscheck.py            # CPU, numpy only; writes raw/nr_matched_instrument/ADE_CROSSCHECK.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
OUT = HERE.parent / "raw" / "nr_matched_instrument"
TREES = ["step5000_unc", "step5000ema_unc", "step10500_unc", "step10500ema_unc",
         "step15000_unc", "step15000ema_unc", "step20000_unc", "step20000ema_unc"]
TOL = 2e-6


def ade(P, G):
    return np.linalg.norm(P.astype(np.float64) - G.astype(np.float64), axis=-1).mean(1)


def main() -> int:
    res = {"tool": "nr_matched_ade_crosscheck.py", "tolerance_m": TOL, "trees": {}}
    worst = 0.0
    for t in TREES:
        r = json.load(open(OUT / "trees" / f"{t}.json", encoding="utf-8"))
        root = os.path.join(ARCHIVE, t)
        ex = np.load(os.path.join(root, "dump_s0", "refcv6_extras.npz"))
        key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
        foff = ex["filter_off.traj"]
        per = {}
        for surf, pd in (("S2", "panel_s0"), ("S6", "s6_s0")):
            man = json.load(open(os.path.join(root, pd, "manifest.json"), encoding="utf-8"))
            slots = list(man["grid"]["slots"])
            ep_of = {int(e["file_index"]): int(e["episode_index"]) for e in man["episodes"]}
            a_off, a_on, a_ref = [], [], []
            for f in sorted(glob.glob(os.path.join(root, pd, "ep*.npz"))):
                z = np.load(f)
                if surf == "S2":
                    off = z["os_filteroff"]
                else:
                    rows = [key[(ep_of[int(os.path.basename(f)[2:5])], int(w))] for w in z["ws"]]
                    off = foff[rows][:, slots]
                a_off.append(ade(off, z["g"]))
                a_on.append(ade(z["os"], z["g"]))
                a_ref.append(ade(z["b_refcv7_50400_s0"], z["g"]))
            a_off, a_on, a_ref = map(np.concatenate, (a_off, a_on, a_ref))
            rd = r["surfaces"][surf]["read_OFF"]["rows"]["ade_m"]["os_minus_refcv7_50400"]
            rs_on = r["surfaces"][surf]["read_ON_rewritten"]["rows_os_minus_refcv7_50400"]["ade_m"]
            d = {"n": int(len(a_off)),
                 "refcv8_OFF_mean_numpy": float(a_off.mean()), "refcv8_OFF_mean_tool": rd["mean_new"],
                 "refcv8_ON_mean_numpy": float(a_on.mean()), "refcv8_ON_mean_tool": rs_on["mean_new"],
                 "refcv7_mean_numpy": float(a_ref.mean()), "refcv7_mean_tool": rd["mean_ref"],
                 "delta_OFF_numpy": float((a_off - a_ref).mean()), "delta_OFF_tool": rd["delta"],
                 "delta_ON_numpy": float((a_on - a_ref).mean()), "delta_ON_tool": rs_on["delta"]}
            d["max_abs_mean_diff"] = max(abs(d["refcv8_OFF_mean_numpy"] - d["refcv8_OFF_mean_tool"]), abs(d["refcv8_ON_mean_numpy"] - d["refcv8_ON_mean_tool"]),
                                         abs(d["refcv7_mean_numpy"] - d["refcv7_mean_tool"]))
            d["delta_rounding_ok"] = bool(abs(d["delta_OFF_numpy"] - d["delta_OFF_tool"]) <= 6e-5 and abs(d["delta_ON_numpy"] - d["delta_ON_tool"]) <= 6e-5)
            d["OFF_differs_from_ON"] = bool(abs(d["refcv8_OFF_mean_numpy"] - d["refcv8_ON_mean_numpy"]) > 1e-4)
            worst = max(worst, d["max_abs_mean_diff"])
            per[surf] = d
        res["trees"][t] = per
    res["worst_abs_mean_diff"] = worst
    res["PASS"] = bool(worst <= TOL and all(v["delta_rounding_ok"] and v["OFF_differs_from_ON"] for p in res["trees"].values() for v in p.values()))
    json.dump(res, open(OUT / "ADE_CROSSCHECK.json", "w", encoding="utf-8"), indent=1)
    print("PASS" if res["PASS"] else "FAIL", "worst |mean diff| =", worst)
    return 0 if res["PASS"] else 5


if __name__ == "__main__":
    sys.exit(main())
