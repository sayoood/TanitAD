#!/usr/bin/env python3
"""lever_map_d3.py -- DESCRIPTIVE D3, run AFTER READ.json fixed the PREREG_V8F_HSPEC outcome. NOT a bar, NOT an arm.

Is part of the remaining second-derivative roughness in the per-step WEIGHTS w themselves (HBLEND's PAV plateaus and steps, e.g. along w 0.120 -> 0.178 -> 0.311 -> 0.429 over 2.0-2.6 s)?
A step in w multiplies O's residual and injects a second difference ~ w'' r that no per-coefficient v can touch. D3 replaces w by ONE fixed smoothing (5-knot moving average, edge-nearest, clipped to
[0, 1]; no sweep) at v = 1 (= H) and at the stage-2 v, and reads accel / curvature / ADE against P10 and H. Hypothesis-generating only.
"""
import json, os, sys, time
sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1"); os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy.ndimage import uniform_filter1d
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v8f_hspec as S
import lever_map as L
H = S.H
KEYS = L.KEYS
res = {"label": "DESCRIPTIVE D3 (hypothesis-generating): w replaced by a 5-knot moving average (clipped to [0,1]); one setting, no sweep; run after the HSPEC outcome was fixed", "per_seed": {}}
t0 = time.time()
for seed in S.SEEDS:
    W = S.load_world(seed)
    ws = np.clip(uniform_filter1d(W["w"], size=5, axis=0, mode="nearest"), 0.0, 1.0)
    v2 = np.asarray(json.load(open(f"{HERE}/fit_hs2_{seed}.json"))["v"])
    r = {"w_max_abs_second_diff": {"w": float(np.abs(np.diff(W["w"], 2, axis=0)).max()), "w_smooth": float(np.abs(np.diff(ws, 2, axis=0)).max())},
         "w_mean_abs_second_diff": {"w": float(np.abs(np.diff(W["w"], 2, axis=0)).mean()), "w_smooth": float(np.abs(np.diff(ws, 2, axis=0)).mean())}}
    for nm, (w_, v_) in {"H_wsmooth_v1": (ws, np.ones((S.NB, 2))), "HS2_wsmooth_v2": (ws, v2)}.items():
        plan = S.hs_plan(W["P"], W["c"], w_, v_, W["B30"])
        r[nm] = L.rows(plan, W, 2000)
        x = r[nm]["vs_P10"]
        print(f"[d3 {seed}] {nm:16s} " + " | ".join(f"{k.replace('fw_','')[:11]} {x[k]['mean']:+.4f}[{x[k]['lo']:+.4f},{x[k]['hi']:+.4f}]{'B' if x[k]['better'] else 'W' if x[k]['worse'] else '-'}" for k in ("ade2", "ade6", "fw_lon_accel_mae", "fw_lat_curvature_mae_1pm")), flush=True)
    res["per_seed"][seed] = r
res["seconds"] = time.time() - t0
H.jdump(res, f"{HERE}/lever_map_d3.json")
