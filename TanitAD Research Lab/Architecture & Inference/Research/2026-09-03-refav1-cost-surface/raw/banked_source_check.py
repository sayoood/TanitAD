"""Third probe: does the banked `source = "cem"` on 34/140 windows carry any real
plan? Read the banked decisions directly — controls, cost, source — and express every
cost in units of the float32 cosine ULP (5.9604645e-08)."""
import glob
import json
import os

import numpy as np

ULP = 5.9604645e-08
NAMES = ["cem", "baseline:cv", "baseline:hold_v0", "baseline:proposal",
         "baseline:decel_1.5"]
BASE = r"C:\Users\Admin\refav1_eval_slice"
out = {}
for dd in ("t1_dump", "t1_dump_ep2"):
    src, cost, ctl = [], [], []
    for f in sorted(glob.glob(os.path.join(BASE, dd, "decisions", "ep*.npz"))):
        d = np.load(f, allow_pickle=True)
        src.append(d["plan_source_cl"])
        cost.append(d["plan_cost_cl"])
        ctl.append(d["cl_controls"])
    src = np.concatenate(src)
    cost = np.concatenate(cost).astype(np.float64)
    ctl = np.concatenate(ctl)
    blk = {"n": int(src.size),
           "max_abs_control_over_all_windows": float(np.abs(ctl).max()),
           "n_windows_with_any_nonzero_control": int(
               (np.abs(ctl).max(axis=(1, 2)) > 0).sum()),
           "by_source": {}}
    for s in np.unique(src):
        m = src == s
        blk["by_source"][NAMES[int(s)]] = {
            "n": int(m.sum()),
            "cost_min": float(cost[m].min()), "cost_med": float(np.median(cost[m])),
            "cost_max": float(cost[m].max()),
            "cost_min_in_ulps": float(cost[m].min() / ULP),
            "cost_max_in_ulps": float(cost[m].max() / ULP),
            "max_abs_control": float(np.abs(ctl[m]).max())}
    blk["distinct_costs_in_ulps"] = sorted(
        {round(float(c) / ULP, 4) for c in cost})
    out[dd] = blk
    print(f"== {dd}  n={blk['n']}  windows with ANY non-zero control: "
          f"{blk['n_windows_with_any_nonzero_control']}  "
          f"max|control| over all = {blk['max_abs_control_over_all_windows']:.3e}")
    for k, v in blk["by_source"].items():
        print(f"   {k:24s} n={v['n']:4d}  cost [{v['cost_min']:.4e}, "
              f"{v['cost_max']:.4e}] = [{v['cost_min_in_ulps']:+.2f}, "
              f"{v['cost_max_in_ulps']:+.2f}] ULPs   max|ctrl| {v['max_abs_control']:.1e}")
    print(f"   distinct costs, in ULPs: {blk['distinct_costs_in_ulps']}")

p = os.path.join(BASE, "costsurface", "banked_source_check.json")
with open(p, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("\nwritten:", p)
