# -*- coding: utf-8 -*-
"""WP-A step 4 -- LOCALISATION probe, parameterised so it can actually resolve.

WHY THIS EXISTS AND s2 DOES NOT STAND
-------------------------------------
s2 regressed the DENSE 7,680-cell raster.  Its controls refused it:
the CONSTANT control scored AP 0.03402 while the full 16x40 arm scored 0.0305 --
i.e. every arm was at or below "predict the per-cell marginal", and lambda for the
81,920-dim arm pinned at the grid maximum (1e5).  That is the 2026-08-22 signature
(n << d -> validation correctly picks maximal regularisation -> every arm reads
the no-information value) and it is a statement about the PARAMETERISATION, not
about the readout.  s2 is banked as a NEGATIVE RESULT ABOUT THE PROBE.

WHAT CHANGES: the target is now LOW-DIMENSIONAL and LOCALISATION-SHAPED, and the
score is relative to the constant predictor, so the no-information value is
EXACTLY 0.0000 by construction.

TARGETS
  y_cent_all     mass-weighted LATERAL centroid of GT occupancy, whole grid  [m]
  y_cent_15_30   the same in the 15-30 m band                                [m]
  y_near         lateral y of the NEAREST occupied cell                      [m]
  x_near         forward x of the nearest occupied cell   [m]  <- a RANGE target;
                 rows are pooled 4x in EVERY arm, so column count should move it
                 far less than the lateral ones.  A contrast, not a headline.
  n_occ          log1p(#occupied cells)  <- ⭐ NON-SPATIAL CONTENT CONTROL.
                 If the column ladder separates THIS too, the difference is not
                 about spatial addressing and the whole reading is wrong.
  prof64         the 64-bin LATERAL occupancy profile (range-marginalised)

SCORE  R2_vs_constant = 1 - SSE_model / SSE_constant, constant = the FIT-split
mean.  ⇒ the constant control reads EXACTLY 0.0000; a NEGATIVE value means the
arm is worse than knowing nothing.  MAE is reported in METRES beside it.
⛔ Not an eval tier: no trajectory is produced.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, r"C:\Users\Admin\wpa-readout\stack_snapshot")

import numpy as np
import torch

from tanitad.data.bev_raster import BEVGrid, fov_mask

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default=r"C:\Users\Admin\wpa-readout\bank\k8r1")
ap.add_argument("--out", default=r"C:\Users\Admin\wpa-readout\raw\bearing_k8.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--mlp", action="store_true", help="add the nonlinear arm")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)
idx = json.load(open(os.path.join(a.bank, "idx.json"), encoding="utf-8"))
clips = idx["clips"]
plan = np.asarray(idx["plan"], dtype=np.int64)
N = int(idx["N"]); TH, TW = idx["token_grid"]; DM = int(idx["d_model"])
GH, GW = idx["grid"]; CELL = float(idx["cell_m"])
grid = BEVGrid(x_fwd_m=float(idx["x_fwd_m"]), y_half_m=float(idx["y_half_m"]),
               cell_m=CELL)
tok = np.load(os.path.join(a.bank, "tok.npy"), mmap_mode="r")
pix = np.load(os.path.join(a.bank, "pix.npy"), mmap_mode="r")
Y = np.asarray(np.load(os.path.join(a.bank, "y.npy"), mmap_mode="r"))

HALF = math.radians(60.0)                # ⚠ HALF angle: 60 deg -> a 120 deg field
fovm = fov_mask(grid, HALF)
print(f"[fov] in-field {int(fovm.sum())}/{fovm.size} "
      f"({100*fovm.mean():.2f} %)  half_angle 60 deg -> FOV 120 deg")

# ---- targets ---------------------------------------------------------------
Yg = torch.from_numpy(Y.reshape(N, GH, GW).copy()).to(torch.float32)
xc = (torch.arange(GH, dtype=torch.float32) + 0.5) * CELL          # 0.25..59.75
yc = -grid.y_half_m + (torch.arange(GW, dtype=torch.float32) + 0.5) * CELL

def centroid_y(mask_rows):
    m = Yg[:, mask_rows]                       # [N, r, GW]
    tot = m.sum(dim=(1, 2))
    cy = (m * yc).sum(dim=(1, 2)) / tot.clamp(min=1e-9)
    return cy, tot > 0

rows_all = torch.ones(GH, dtype=torch.bool)
rows_b = (xc >= 15.0) & (xc < 30.0)
y_all, ok_all = centroid_y(rows_all)
y_b, ok_b = centroid_y(rows_b)

# nearest occupied cell
occ = Yg > 0
any_row = occ.any(dim=2)                                          # [N, GH]
first = torch.where(any_row.any(dim=1),
                    any_row.float().argmax(dim=1), torch.zeros(N, dtype=torch.long))
ok_near = any_row.any(dim=1)
x_near = xc[first]
rowsel = occ[torch.arange(N), first].float()                      # [N, GW]
y_near = (rowsel * yc).sum(1) / rowsel.sum(1).clamp(min=1e-9)
n_occ = torch.log1p(Yg.sum(dim=(1, 2)))
prof64 = Yg.sum(dim=1)                                            # [N, 64]
prof64 = prof64 / prof64.sum(1, keepdim=True).clamp(min=1e-9)

TARGETS = {
    "y_cent_all":   (y_all[:, None], ok_all, "m"),
    "y_cent_15_30": (y_b[:, None],   ok_b,   "m"),
    "y_near":       (y_near[:, None], ok_near, "m"),
    "x_near":       (x_near[:, None], ok_near, "m"),
    "n_occ":        (n_occ[:, None], torch.ones(N, dtype=torch.bool), "log1p"),
    "prof64":       (prof64, ok_all, "frac"),
}
for k, (t, ok, u) in TARGETS.items():
    print(f"[target] {k:12s} dim {t.shape[1]:3d}  n_valid {int(ok.sum()):6d}"
          f"  sd {float(t[ok].std()):.4f} {u}")

# ---- split -----------------------------------------------------------------
rng = np.random.default_rng(a.seed)
order = rng.permutation(len(clips))
n_te, n_va = 30, 25
te_c, va_c = set(order[:n_te].tolist()), set(order[n_te:n_te + n_va].tolist())
cl = plan[:, 0]
fi = torch.tensor(np.where(~np.isin(cl, list(te_c | va_c)))[0])
vi = torch.tensor(np.where(np.isin(cl, list(va_c)))[0])
ti = torch.tensor(np.where(np.isin(cl, list(te_c)))[0])
print(f"[split] clips {len(clips)-n_te-n_va}/{n_va}/{n_te}  "
      f"rows {len(fi)}/{len(vi)}/{len(ti)}")

ARMS = [("tok_16x40", "tok", (1, 1)), ("tok_8x20", "tok", (2, 2)),
        ("tok_4x8", "tok", (4, 5)), ("tok_4x4", "tok", (4, 10)),
        ("tok_4x2", "tok", (4, 20)), ("tok_1x1", "tok", (16, 40)),
        ("pix_16x40", "pix", (1, 1)), ("pix_8x20", "pix", (2, 2)),
        ("pix_4x4", "pix", (4, 10))]

def build_X(kind, ker):
    src = tok if kind == "tok" else pix
    C = DM if kind == "tok" else int(idx["in_channels"])
    d = C * (TH // ker[0]) * (TW // ker[1])
    X = torch.empty(N, d, dtype=torch.float32)
    for s0 in range(0, N, 1024):
        x = torch.from_numpy(np.asarray(src[s0:s0 + 1024], dtype=np.float32))
        x = x.transpose(1, 2).reshape(-1, C, TH, TW)
        if ker != (1, 1):
            x = torch.nn.functional.avg_pool2d(x, ker)
        X[s0:s0 + x.shape[0]] = x.flatten(1)
    return X

def gram(X, ra, rb, chunk=2048):
    K = torch.empty(len(ra), len(rb), dtype=torch.float32, device=dev)
    for s in range(0, len(ra), chunk):
        xa = X[ra[s:s + chunk]].to(dev)
        for u in range(0, len(rb), chunk):
            xb = X[rb[u:u + chunk]].to(dev)
            K[s:s + xa.shape[0], u:u + xb.shape[0]] = xa @ xb.T
            del xb
        del xa
    return K

LAM = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]

def r2_vs_constant(pred, true, const):
    sse = ((pred - true) ** 2).sum()
    ssc = ((const - true) ** 2).sum()
    return float(1.0 - sse / ssc)

res = {}
t0all = time.time()
for name, kind, ker in ARMS:
    t0 = time.time()
    X = build_X(kind, ker)
    d = X.shape[1]
    X = X - X[fi].mean(0, keepdim=True)
    Kff = gram(X, fi, fi); Kvf = gram(X, vi, fi); Ktf = gram(X, ti, fi)
    eye = torch.eye(len(fi), device=dev)
    arm = {"d": int(d), "az_cols": TW // ker[1],
           "deg_per_col": 120.0 / (TW // ker[1]),
           "lat_width_at_30m": 2 * 30 * math.tan(math.radians(60.0 / (TW // ker[1]))),
           "n_fit": int(len(fi)), "targets": {}}
    for tname, (T, ok, unit) in TARGETS.items():
        f2 = fi[ok[fi]]; v2 = vi[ok[vi]]; t2 = ti[ok[ti]]
        mf = torch.isin(fi, f2); mv = torch.isin(vi, v2); mt = torch.isin(ti, t2)
        Tf = T[f2].to(dev); Tv = T[v2].to(dev); Tt = T[t2].to(dev)
        mu = Tf.mean(0, keepdim=True)
        best = None
        # the Cholesky is refit on the rows VALID FOR THIS TARGET, so a target
        # with fewer valid frames is not silently solved on the wrong system.
        Kff2 = Kff[mf][:, mf]
        eye2 = torch.eye(int(mf.sum()), device=dev)
        for lam in LAM:
            try:
                L2 = torch.linalg.cholesky(Kff2 + lam * eye2)
            except Exception:
                continue
            alpha = torch.cholesky_solve(Tf - mu, L2)
            pv = Kvf[mv][:, mf] @ alpha + mu
            r2v = r2_vs_constant(pv, Tv, mu.expand_as(Tv))
            if best is None or r2v > best[1]:
                best = (lam, r2v, alpha, mf)
            del L2
        del Kff2, eye2
        lam, r2v, alpha, mf = best
        pt = Ktf[mt][:, mf] @ alpha + mu
        r2t = r2_vs_constant(pt, Tt, mu.expand_as(Tt))
        mae = float((pt - Tt).abs().mean())
        mae_c = float((mu.expand_as(Tt) - Tt).abs().mean())
        arm["targets"][tname] = {"lambda": lam, "r2_val": r2v, "r2_test": r2t,
                                 "mae_test": mae, "mae_constant": mae_c,
                                 "unit": unit, "n_test": int(mt.sum())}
        np.save(os.path.join(a.bank, f"bp_{name}_{tname}.npy"),
                pt.to(torch.float16).cpu().numpy())
        np.save(os.path.join(a.bank, f"bt_{tname}.npy"),
                Tt.to(torch.float16).cpu().numpy())
        np.save(os.path.join(a.bank, f"bc_{tname}.npy"),
                np.asarray(cl[t2.cpu().numpy()]))
    r = arm["targets"]
    print(f"[{name:10s}] d={d:6d} az={arm['az_cols']:2d}  "
          f"y_all R2 {r['y_cent_all']['r2_test']:+.4f} MAE {r['y_cent_all']['mae_test']:.3f}m | "
          f"y15-30 R2 {r['y_cent_15_30']['r2_test']:+.4f} MAE {r['y_cent_15_30']['mae_test']:.3f}m | "
          f"y_near R2 {r['y_near']['r2_test']:+.4f} | x_near R2 {r['x_near']['r2_test']:+.4f} | "
          f"n_occ R2 {r['n_occ']['r2_test']:+.4f} | prof64 R2 {r['prof64']['r2_test']:+.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    res[name] = arm
    del X, Kff, Kvf, Ktf, eye
    torch.cuda.empty_cache()

# ---- SHUFFLED control: features vs targets permuted ACROSS clips -----------
X = build_X("tok", (1, 1)); X = X - X[fi].mean(0, keepdim=True)
Kff = gram(X, fi, fi); Ktf = gram(X, ti, fi)
shuf = {}
for tname in ("y_cent_all", "n_occ"):
    T, ok, unit = TARGETS[tname]
    f2 = fi[ok[fi]]; t2 = ti[ok[ti]]
    mf = torch.isin(fi, f2); mt = torch.isin(ti, t2)
    Tf = T[f2].to(dev); Tt = T[t2].to(dev)
    p = torch.from_numpy(rng.permutation(int(mf.sum())).copy()).to(dev)
    Tf = Tf[p]
    mu = Tf.mean(0, keepdim=True)
    lam = res["tok_16x40"]["targets"][tname]["lambda"]
    L2 = torch.linalg.cholesky(Kff[mf][:, mf] + lam * torch.eye(int(mf.sum()), device=dev))
    alpha = torch.cholesky_solve(Tf - mu, L2)
    pt = Ktf[mt][:, mf] @ alpha + mu
    shuf[tname] = {"r2_test": r2_vs_constant(pt, Tt, mu.expand_as(Tt)),
                   "lambda": lam}
print("[control shuffled]", json.dumps(shuf))
print("[control constant] r2_test = 0.000000 BY CONSTRUCTION "
      "(the score's denominator IS the constant predictor)")

out = {"_evidence_class": "MEASURED (ours; artifact = this file + the bank)",
       "tier": "NOT APPLICABLE - representation probe, no trajectory produced",
       "function_class": "LINEAR (exact dual ridge, linear kernel); a linear "
                         "negative bounds LINEAR decodability only",
       "score": "R2_vs_constant = 1 - SSE_model/SSE_constant, constant = FIT mean "
                "-> the constant control reads EXACTLY 0.0000",
       "bank": a.bank, "ckpt": idx["ckpt"], "step": idx["step"], "seed": a.seed,
       "n_clips": len(clips), "n_rows": N,
       "split_clips": {"fit": len(clips) - n_te - n_va, "val": n_va, "test": n_te},
       "fov": {"half_angle_deg": 60.0, "hfov_deg": 120.0,
               "in_field_cells": int(fovm.sum()), "cells": int(fovm.size),
               "in_field_frac": float(fovm.mean())},
       "arms": res,
       "controls": {"constant_r2": 0.0, "shuffled": shuf},
       "test_clips": sorted(int(c) for c in te_c)}
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
print("WROTE", a.out, f"{time.time()-t0all:.0f}s")
