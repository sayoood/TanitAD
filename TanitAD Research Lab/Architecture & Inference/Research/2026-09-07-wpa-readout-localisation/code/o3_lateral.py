# -*- coding: utf-8 -*-
"""LATERAL family for the oracle recheck, recomputed POST-HOC from the banked
per-arm test scores (`raw/test_scores.npz`) -- no GPU, no retraining.

⭐ WHY SEPARATELY: `E-READOUT-CEILING-1` quotes a lateral-MAE column beside the AP
ladder (1.372 m at 15-30 m for 16x40, 2.669 m for 4x4).  Those are mirrored-address
numbers too, and an AP-only recheck would leave them un-re-measured -- which is
exactly the "ADE alone" failure the four-families rule forbids, one metric family
across.  The estimator is s6_oracle.py's own `lateral_mae`, verbatim.
"""
import json
import math
import os
import sys

sys.path.insert(0, r"C:\Users\Admin\wpa-readout\stack_snapshot")
import numpy as np

from tanitad.data.bev_raster import BEVGrid, cell_centers_xy, fov_mask

BANK = r"C:\Users\Admin\wpa-readout\bank\k8r1"
RAW = r"C:\Users\Admin\wpa-readout\oracle_recheck\raw"
sys.stdout.reconfigure(encoding="utf-8")

idx = json.load(open(os.path.join(BANK, "idx.json"), encoding="utf-8"))
clips = idx["clips"]; plan = np.asarray(idx["plan"], dtype=np.int64)
TH, TW = idx["token_grid"]
PATCH, W_PX, H_PX, F_REF, H_CAM = 16, 640, 256, 305.5774907364391, 1.5
grid = BEVGrid(x_fwd_m=float(idx["x_fwd_m"]), y_half_m=float(idx["y_half_m"]),
               cell_m=float(idx["cell_m"]))
X_m, Y_m = cell_centers_xy(grid)
az = np.arctan2(Y_m, X_m)
rng_m = np.sqrt(X_m ** 2 + Y_m ** 2)
rowf = (H_PX / 2.0 + F_REF * H_CAM / np.maximum(rng_m, 1e-6)) / PATCH - 0.5
valid = fov_mask(grid, math.radians(60.0)) & (rowf >= 0) & (rowf <= TH - 1)
for sg in (-1.0, +1.0):
    cf = (W_PX / 2.0 + sg * F_REF * az) / PATCH - 0.5
    valid = valid & (cf >= 0) & (cf <= TW - 1)
vidx = np.flatnonzero(valid.reshape(-1))
xs, ys = X_m.reshape(-1)[vidx], Y_m.reshape(-1)[vidx]
NC = len(vidx)

Yall = np.asarray(np.load(os.path.join(BANK, "y.npy"), mmap_mode="r"))
Yv = Yall[:, vidx]
rng = np.random.default_rng(0)
order = rng.permutation(len(clips))
te_c = set(order[:30].tolist())
ti = np.where(np.isin(plan[:, 0], list(te_c)))[0]
T = Yv[ti].astype(np.float64)
print(f"[lateral] cells {NC}  test rows {len(ti)}", flush=True)

BANDS = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0)]


def lateral_mae(prob):
    """s6_oracle.py:194-207 verbatim, in numpy: per range band, the |error| between
    the TRUE occupancy centroid's y and the PREDICTED probability-weighted y."""
    out = {}
    for lo, hi in BANDS:
        m = (xs >= lo) & (xs < hi)
        t, p = T[:, m], np.clip(prob[:, m], 0, None)
        mt = t.sum(1); ok = mt > 0
        if int(ok.sum()) == 0:
            out[f"{lo:.0f}-{hi:.0f}m"] = {"n": 0}
            continue
        yv = ys[m]
        e = np.abs((t * yv).sum(1) / np.maximum(mt, 1e-9)
                   - (p * yv).sum(1) / np.maximum(p.sum(1), 1e-9))
        out[f"{lo:.0f}-{hi:.0f}m"] = {"n": int(ok.sum()),
                                      "mae_y_m": float(e[ok].mean()),
                                      "med_y_m": float(np.median(e[ok]))}
    return out


Z = np.load(os.path.join(RAW, "test_scores.npz"))
res = {k: lateral_mae(Z[k].astype(np.float64)) for k in Z.files}
json.dump({"_evidence_class": "MEASURED (ours; post-hoc from raw/test_scores.npz)",
           "estimator": "s6_oracle.py lateral_mae, verbatim",
           "n_test_rows": int(len(ti)), "n_cells": int(NC), "bands_m": BANDS,
           "arms": res},
          open(os.path.join(RAW, "lateral.json"), "w", encoding="utf-8"), indent=1)

print(f"\n| arm | 0-15 m | 15-30 m | 30-45 m | 45-60 m |")
print("|---|---|---|---|---|")
for k in sorted(res):
    r = res[k]
    print(f"| `{k}` | " + " | ".join(
        f"{r[f'{lo:.0f}-{hi:.0f}m'].get('mae_y_m', float('nan')):.3f} m"
        for lo, hi in BANDS) + " |")
