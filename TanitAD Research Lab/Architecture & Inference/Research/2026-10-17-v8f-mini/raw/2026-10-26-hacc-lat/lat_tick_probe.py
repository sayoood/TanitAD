"""DESCRIPTIVE probe (decides nothing; TRAIN tick samples only): where HACC's T1-kin lateral excess lives, plan vs execution.
Per tick (0 = logged state, 1-2 = own executed state) and measured speed band: curvature MAE over the first 10 knots (the part a tick
executes) of HACC's plan and of P10's plan against the tick GT, and the plan-vs-P curvature gap with no GT at all.
python lat_tick_probe.py <t1k export root> <t1k_read.py> <ticks npz> <out json>"""
import ast
import json
import math
import sys

import numpy as np

ROOT, READER, TK, OUT = sys.argv[1:5]
sys.path.insert(0, ROOT + "/taniteval")
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB  # noqa: E402

ns = {"np": np, "torch": torch, "FF": FF, "PB": PB, "math": math, "DT": 0.2}
for node in ast.parse(open(READER, encoding="utf-8").read()).body:
    if isinstance(node, ast.FunctionDef) and node.name == "comps":
        exec(compile(ast.Module([node], []), READER, "exec"), ns)
comps = ns["comps"]
z = np.load(TK)
ok = z["gt_ok"]
plan, P, gt, v0, tick = z["plan"][ok], z["P"][ok], z["gt"][ok], z["v0"][ok], z["tick"][ok]
K = 10
cH, cP = comps(plan[:, :K], gt[:, :K]), comps(P[:, :K], gt[:, :K])
cHP = comps(plan[:, :K], P[:, :K])
bands = [(0, 2), (2, 5.2), (5.2, 8.6), (8.6, 99)]
res = {"what": "TRAIN tick samples, first 10 knots; curvature MAE (1/m) vs the tick GT; HACC plan vs P10 plan", "n": int(ok.sum()), "cells": []}
for t in (0, 1, 2):
    for lo, hi in bands:
        m = (tick == t) & (v0 >= lo) & (v0 < hi)
        if m.sum() < 20:
            continue
        h = float(np.nanmean(cH["curvature_mae_1pm"][m])); p = float(np.nanmean(cP["curvature_mae_1pm"][m]))
        res["cells"].append({"tick": t, "v0_band": [lo, hi], "n": int(m.sum()), "hacc_curv_mae": round(h, 4), "p10_curv_mae": round(p, 4),
                             "hacc_minus_p10": round(h - p, 4), "hacc_vs_p10_plan_curv_gap": round(float(np.nanmean(cHP["curvature_mae_1pm"][m])), 4),
                             "hacc_ade2": round(float(np.nanmean(cH["ade2_m"][m])), 3), "p10_ade2": round(float(np.nanmean(cP["ade2_m"][m])), 3)})
json.dump(res, open(OUT, "w"), indent=1)
print(f"{'tick':>4} {'v0 band':>10} {'n':>6} {'HACC':>7} {'P10':>7} {'H-P':>8} {'gap':>7} {'ADE2 H/P':>12}")
for c in res["cells"]:
    print(f"{c['tick']:>4} {str(c['v0_band']):>10} {c['n']:>6} {c['hacc_curv_mae']:>7} {c['p10_curv_mae']:>7} {c['hacc_minus_p10']:>+8} {c['hacc_vs_p10_plan_curv_gap']:>7} {c['hacc_ade2']:>6}/{c['p10_ade2']}")
