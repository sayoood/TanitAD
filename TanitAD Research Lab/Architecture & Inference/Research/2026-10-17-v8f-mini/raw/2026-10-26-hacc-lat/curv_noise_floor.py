"""INSTRUMENT probe (no model arm is read): the canonical lateral rows' noise floor. The eval-141 GT tracks are perturbed by i.i.d. Gaussian
position noise of sigma in {2, 5, 10, 20} mm per 0.2 s knot, and the perturbed GT is scored against the clean GT with the reader's own
row function (four_families' geometry and masks). The rows are reported per GT-speed band. A row whose noise floor at a few millimetres is
comparable to the arm differences it is asked to rank is noise-dominated in that band.
python curv_noise_floor.py <t1k export root> <t1k_read.py> <a T1KIN dump dir> <out json>"""
import ast
import glob
import json
import math
import sys

import numpy as np

ROOT, READER, DUMP, OUT = sys.argv[1:5]
sys.path.insert(0, ROOT + "/taniteval")
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB  # noqa: E402

ns = {"np": np, "torch": torch, "FF": FF, "PB": PB, "math": math, "DT": 0.2}
for node in ast.parse(open(READER, encoding="utf-8").read()).body:
    if isinstance(node, ast.FunctionDef) and node.name == "comps":
        exec(compile(ast.Module([node], []), READER, "exec"), ns)
comps = ns["comps"]
G = np.concatenate([np.load(f)["g"] for f in sorted(glob.glob(DUMP + "/ep*.npz"))]).astype(np.float64)
step = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(G), 1, 2)), G], 1), axis=1), axis=-1)
vbar = step.mean(1) / 0.2                                                       # the window's mean GT speed (m/s)
bands = [(0, 2), (2, 5.2), (5.2, 8.6), (8.6, 11.8), (11.8, 99)]
rng = np.random.default_rng(0)
res = {"what": "noise floor of the canonical rows: perturbed GT vs clean GT, eval-141 GT (n windows %d)" % len(G), "bands_mps": bands, "sigma": {}}
for sig in (0.002, 0.005, 0.01, 0.02):
    Xn = G + rng.normal(0.0, sig, G.shape)
    C = comps(Xn, G)
    row = {}
    for lo, hi in bands:
        m = (vbar >= lo) & (vbar < hi)
        row[f"{lo}-{hi}"] = {"n": int(m.sum()),
                             "curvature_mae_1pm": round(float(np.nanmean(C["curvature_mae_1pm"][m])), 4),
                             "yaw_rate_mae_degps": round(float(np.nanmean(C["yaw_rate_mae_degps"][m])), 3),
                             "heading_mae_deg": round(float(np.nanmean(C["heading_mae_deg"][m])), 3),
                             "cross_mae_m": round(float(np.nanmean(C["cross_mae_m"][m])), 4)}
    row["pooled"] = {"curvature_mae_1pm": round(float(np.nanmean(C["curvature_mae_1pm"])), 4),
                     "yaw_rate_mae_degps": round(float(np.nanmean(C["yaw_rate_mae_degps"])), 3)}
    res["sigma"][f"{sig * 1000:g}mm"] = row
json.dump(res, open(OUT, "w"), indent=1)
for s, row in res["sigma"].items():
    print(f"sigma {s:>5}: " + "  ".join(f"[{k}] curv {v['curvature_mae_1pm']} yaw {v['yaw_rate_mae_degps']}" for k, v in row.items() if k != "pooled")
          + f"  | pooled curv {row['pooled']['curvature_mae_1pm']} yaw {row['pooled']['yaw_rate_mae_degps']}")
