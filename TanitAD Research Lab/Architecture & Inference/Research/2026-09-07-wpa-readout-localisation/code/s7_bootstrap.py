# -*- coding: utf-8 -*-
"""WP-A step 7 -- PAIRED episode-cluster bootstrap over the 30 test clips.

The programme's decision-grade interval (taniteval/ci.py's estimator family):
resample TEST EPISODES with replacement, recompute the metric, and for two arms
use the PAIRED difference on the same resample.

⚠️ AND SAY WHAT IT ANSWERS. This interval asks "would another draw of EPISODES
say this?" -- NOT "would another training run say this?" (H-ESTIM-SEED-1).
The second question is answered separately by the REPLICATE arms
(oracle_s0_t1 / _t2, same split, training seed only), and both are reported.
"""
import json
import math
import os
import sys

sys.path.insert(0, r"C:\Users\Admin\wpa-readout\stack_snapshot")

import numpy as np
import torch

from tanitad.data.bev_raster import BEVGrid, cell_centers_xy, fov_mask

BANK = r"C:\Users\Admin\wpa-readout\bank\k8r1"
OUT = r"C:\Users\Admin\wpa-readout\raw\bootstrap.json"
NBOOT = 2000
SEED = 0
dev = "cuda" if torch.cuda.is_available() else "cpu"

idx = json.load(open(os.path.join(BANK, "idx.json"), encoding="utf-8"))
clips = idx["clips"]; plan = np.asarray(idx["plan"], dtype=np.int64)
N = int(idx["N"]); TH, TW = idx["token_grid"]
GH, GW = idx["grid"]; CELL = float(idx["cell_m"])
grid = BEVGrid(x_fwd_m=float(idx["x_fwd_m"]), y_half_m=float(idx["y_half_m"]),
               cell_m=CELL)
Yall = np.asarray(np.load(os.path.join(BANK, "y.npy"), mmap_mode="r"))

# reproduce s6's addressable-cell set EXACTLY (same formulas, same constants)
PATCH, W_PX, H_PX, F_REF, H_CAM = 16, 640, 256, 305.5774907364391, 1.5
X_m, Y_m = cell_centers_xy(grid)
az = np.arctan2(Y_m, X_m)
colf = (F_REF * az + W_PX / 2.0) / PATCH - 0.5
rng_m = np.sqrt(X_m ** 2 + Y_m ** 2)
rowf = (H_PX / 2.0 + F_REF * H_CAM / np.maximum(rng_m, 1e-6)) / PATCH - 0.5
valid = (fov_mask(grid, math.radians(60.0)) & (colf >= 0) & (colf <= TW - 1)
         & (rowf >= 0) & (rowf <= TH - 1))
vidx = np.flatnonzero(valid.reshape(-1))
NC = len(vidx)

rng = np.random.default_rng(SEED)
order = rng.permutation(len(clips))
te_c = set(order[:30].tolist())
cl = plan[:, 0]
ti = np.where(np.isin(cl, list(te_c)))[0]
tclip = cl[ti]
uclip = np.unique(tclip)
Yt = torch.from_numpy(Yall[ti][:, vidx].copy()).float().to(dev)
print(f"[boot] test rows {len(ti)} over {len(uclip)} clips, cells {NC}, "
      f"base rate {float(Yt.mean()):.6f}")

xs = torch.from_numpy(X_m.reshape(-1)[vidx].astype(np.float32)).to(dev)
ys = torch.from_numpy(Y_m.reshape(-1)[vidx].astype(np.float32)).to(dev)
band = (xs >= 15.0) & (xs < 30.0)

def ap_of(scores, labels):
    o = torch.argsort(scores, descending=True)
    l = labels[o]
    tp = torch.cumsum(l, 0); fp = torch.cumsum(1 - l, 0)
    npos = l.sum()
    if float(npos) == 0:
        return float("nan")
    rec = tp / npos
    dr = torch.diff(rec, prepend=torch.zeros(1, device=rec.device))
    return float(((tp / (tp + fp)) * dr).sum())

def lat_of(P, T):
    t = T[:, band]; p = P[:, band].clamp(min=0)
    mt = t.sum(1); ok = mt > 0
    if int(ok.sum()) == 0:
        return float("nan")
    yv = ys[band]
    e = ((t * yv).sum(1) / mt.clamp(min=1e-9)
         - (p * yv).sum(1) / p.sum(1).clamp(min=1e-9)).abs()
    return float(e[ok].mean())

arms = {}
for f in sorted(os.listdir(BANK)):
    if f.startswith("or_") and f.endswith(".npy"):
        nm = f[3:-4]
        P = torch.from_numpy(np.load(os.path.join(BANK, f))).float().to(dev)
        assert P.shape == Yt.shape, (nm, P.shape, Yt.shape)
        arms[nm] = P
print("[boot] arms:", list(arms))

# per-clip row index lists
rows_by_clip = {int(c): np.flatnonzero(tclip == c) for c in uclip}
g = np.random.default_rng(SEED)
draws = [np.concatenate([rows_by_clip[int(c)] for c in
                         g.choice(uclip, len(uclip), replace=True)])
         for _ in range(NBOOT)]

point = {k: {"ap": ap_of(P.reshape(-1), Yt.reshape(-1)),
             "lat_15_30_m": lat_of(P, Yt)} for k, P in arms.items()}
for k, v in point.items():
    print(f"[point] {k:10s} AP {v['ap']:.4f}  latMAE15-30 {v['lat_15_30_m']:.3f} m")

REF = "orc_16x40"
res = {}
for k, P in arms.items():
    if k == REF:
        continue
    d_ap, d_lat = [], []
    for dr_ in draws:
        r = torch.from_numpy(dr_).to(dev)
        Ys, Pa, Pb = Yt[r], arms[REF][r], P[r]
        d_ap.append(ap_of(Pa.reshape(-1), Ys.reshape(-1))
                    - ap_of(Pb.reshape(-1), Ys.reshape(-1)))
        d_lat.append(lat_of(Pa, Ys) - lat_of(Pb, Ys))
    d_ap = np.array(d_ap); d_lat = np.array(d_lat)
    lo, hi = np.percentile(d_ap, [2.5, 97.5])
    llo, lhi = np.percentile(d_lat, [2.5, 97.5])
    res[k] = {"d_ap_mean": float(d_ap.mean()), "d_ap_ci95": [float(lo), float(hi)],
              "d_ap_separated": bool(lo > 0 or hi < 0),
              "d_lat_mean_m": float(d_lat.mean()),
              "d_lat_ci95_m": [float(llo), float(lhi)],
              "d_lat_separated": bool(llo > 0 or lhi < 0),
              "ratio_ap": float(point[REF]["ap"] / point[k]["ap"]) if point[k]["ap"] else None}
    print(f"[paired {REF} - {k:10s}] dAP {d_ap.mean():+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}] sep={res[k]['d_ap_separated']}   "
          f"dLat {d_lat.mean():+.3f} m [{llo:+.3f}, {lhi:+.3f}] "
          f"sep={res[k]['d_lat_separated']}", flush=True)

# replicate spread across TRAINING seeds (the other variance)
rep = {}
for ts in ("", "_t1", "_t2"):
    p = os.path.join(r"C:\Users\Admin\wpa-readout\raw",
                     f"oracle_s0{ts}.json" if ts else "oracle_s0.json")
    if os.path.isfile(p):
        d = json.load(open(p, encoding="utf-8"))
        for k, v in d["arms"].items():
            rep.setdefault(k, []).append(v["ap_test"])
noise = {k: {"aps": v, "range": float(max(v) - min(v)), "mean": float(np.mean(v)),
             "n_seeds": len(v)} for k, v in rep.items() if len(v) > 1}
print("[replicate noise floor, AP]",
      json.dumps({k: round(v["range"], 4) for k, v in noise.items()}))

json.dump({"_evidence_class": "MEASURED (ours; artifact = this file + the bank)",
           "estimator": "PAIRED episode-cluster bootstrap over the 30 test clips, "
                        f"{NBOOT} resamples",
           "what_it_answers": "would another DRAW OF EPISODES say this? It does "
                              "NOT answer 'would another TRAINING RUN say this?' "
                              "-- that is the replicate block below "
                              "(H-ESTIM-SEED-1).",
           "reference_arm": REF, "n_test_clips": int(len(uclip)),
           "n_test_rows": int(len(ti)), "n_cells": int(NC),
           "point": point, "paired_vs_16x40": res,
           "replicate_training_seeds": noise},
          open(OUT, "w", encoding="utf-8"), indent=1)
print("WROTE", OUT)
