# -*- coding: utf-8 -*-
"""WP-A step 2 -- the readout-pooling LADDER, one probe, one split, one seed.

QUESTION: how much BEV agent localisation does the 16x40 token grid carry that
a pooled readout does not?

ARMS  (all the SAME encoder tokens; ONLY the pooling differs)
  tok_16x40   full token grid                        40 az cols  ->  3.0 deg/col
  tok_8x20    AvgPool2d(2,2)   = refcv5's ResNet map  20 az cols  ->  6.0 deg/col
  tok_4x8     AvgPool2d(4,5)   = v7-tiny's DEPLOYED    8 az cols  -> 15.0 deg/col
  tok_4x4     AvgPool2d(4,10)  = v1/v6 flagship        4 az cols  -> 30.0 deg/col
  tok_2x4 / tok_1x1            ladder ends
  z_4x8_proj  the DEPLOYED compact state: proj(pool_4x5(tok)) -- prices the
              128->64 projection on top of the pooling
CONTROLS (each must read a KNOWN value or the panel is inadmissible)
  pix_16x40 / pix_4x4   raw-patch-mean PIXEL floor at the same two geometries
  constant              fit-split per-cell mean  -> AP == base rate EXACTLY
  allzero               -> IoU 0, F1 0, AP == base rate; accuracy printed
  shuffled              targets permuted ACROSS clips -> must collapse to constant

ESTIMATOR: exact ridge in the DUAL (linear kernel), so arm A gets its full
81,920 dims with no PCA distortion and every arm uses the identical estimator.
lambda is chosen on an INNER-VAL split drawn from FIT clips only -- never on test
(the 2026-08-22 failure: lambda on test picks maximal regularisation and every arm
reads the no-information value).

⛔ NOT an eval tier. No model produces a trajectory here; this is a
REPRESENTATION probe, so a T0/T1 stamp would be a category error.
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
ap.add_argument("--out", default=r"C:\Users\Admin\wpa-readout\raw\panel_k8.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--boot", type=int, default=1000)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)
idx = json.load(open(os.path.join(a.bank, "idx.json"), encoding="utf-8"))
clips = idx["clips"]
plan = np.asarray(idx["plan"], dtype=np.int64)          # [N, 2] (clip_i, row)
N = int(idx["N"])
TH, TW = idx["token_grid"]
DM = int(idx["d_model"])
GH, GW = idx["grid"]                                     # 120, 64
CELL = float(idx["cell_m"])
grid = BEVGrid(x_fwd_m=float(idx["x_fwd_m"]), y_half_m=float(idx["y_half_m"]),
               cell_m=CELL)
print(f"[bank] N={N} tokens {TH}x{TW}x{DM} target {GH}x{GW} ({GH*GW} cells)")

tok = np.load(os.path.join(a.bank, "tok.npy"), mmap_mode="r")
pix = np.load(os.path.join(a.bank, "pix.npy"), mmap_mode="r")
Y = np.load(os.path.join(a.bank, "y.npy"), mmap_mode="r")
assert tok.shape == (N, TH * TW, DM) and Y.shape == (N, GH * GW)

# ---- episode-disjoint split, deterministic ---------------------------------
rng = np.random.default_rng(a.seed)
order = rng.permutation(len(clips))
n_te, n_va = 30, 25
te_c = set(order[:n_te].tolist())
va_c = set(order[n_te:n_te + n_va].tolist())
fi = torch.tensor([i for i in range(N) if plan[i, 0] not in te_c and plan[i, 0] not in va_c])
vi = torch.tensor([i for i in range(N) if plan[i, 0] in va_c])
ti = torch.tensor([i for i in range(N) if plan[i, 0] in te_c])
print(f"[split] clips fit {len(clips)-n_te-n_va} / val {n_va} / test {n_te}  "
      f"rows fit {len(fi)} / val {len(vi)} / test {len(ti)}")

Yt = torch.from_numpy(np.asarray(Y)).to(torch.float32)   # [N, 7680] on CPU
base_rate_te = float(Yt[ti].mean())
base_rate_fit = float(Yt[fi].mean())
print(f"[occupancy] test base rate {base_rate_te:.6f}  ->  ALL-ZERO accuracy "
      f"{100*(1-base_rate_te):.4f} %   (fit {base_rate_fit:.6f})")

fovm = torch.from_numpy(fov_mask(grid, math.radians(120.0)).reshape(-1).copy())
print(f"[fov] in-field cells {int(fovm.sum())}/{GH*GW} "
      f"({100*float(fovm.float().mean()):.2f} %)")
base_rate_te_fov = float(Yt[ti][:, fovm].mean())
print(f"[occupancy] test base rate IN-FOV {base_rate_te_fov:.6f}")

# ---- geometry of the ladder ------------------------------------------------
HFOV = 120.0
def az_per_col(nc):
    return HFOV / nc
def lat_width_at(nc, R):
    return 2.0 * R * math.tan(math.radians(az_per_col(nc) / 2.0))

ARMS = [
    ("tok_16x40", "tok", (1, 1)),
    ("tok_8x20",  "tok", (2, 2)),
    ("tok_4x8",   "tok", (4, 5)),
    ("tok_4x4",   "tok", (4, 10)),
    ("tok_2x4",   "tok", (8, 10)),
    ("tok_1x1",   "tok", (16, 40)),
    ("pix_16x40", "pix", (1, 1)),
    ("pix_4x4",   "pix", (4, 10)),
]

def build_X(kind, ker):
    """[N, d] float32 CPU features for one arm."""
    src = tok if kind == "tok" else pix
    C = DM if kind == "tok" else int(idx["in_channels"])
    d = C * (TH // ker[0]) * (TW // ker[1])
    X = torch.empty(N, d, dtype=torch.float32)
    B = 1024
    for s0 in range(0, N, B):
        x = torch.from_numpy(np.asarray(src[s0:s0 + B], dtype=np.float32))
        x = x.transpose(1, 2).reshape(-1, C, TH, TW)
        if ker != (1, 1):
            x = torch.nn.functional.avg_pool2d(x, ker)
        X[s0:s0 + x.shape[0]] = x.flatten(1)
    return X

# ---- fast AP + metrics -----------------------------------------------------
def average_precision(scores, labels):
    """AP = sum (R_i - R_{i-1}) * P_i over the score ranking. Constant scores
    give exactly the base rate, which is the control's known value."""
    o = torch.argsort(scores, descending=True)
    l = labels[o]
    tp = torch.cumsum(l, 0)
    fp = torch.cumsum(1.0 - l, 0)
    prec = tp / (tp + fp)
    npos = float(l.sum())
    if npos == 0:
        return float("nan")
    rec = tp / npos
    dr = torch.diff(rec, prepend=torch.zeros(1, device=rec.device))
    return float((prec * dr).sum())

def best_f1(scores, labels, n_thr=200):
    st = max(1, scores.numel() // 200000)
    q = torch.quantile(scores[::st].float(),
                       torch.linspace(0.90, 0.99999, n_thr, device=scores.device))
    best, bt = 0.0, float("nan")
    npos = labels.sum()
    for t in q:
        p = (scores >= t).float()
        tp = (p * labels).sum()
        f1 = float(2 * tp / (p.sum() + npos + 1e-9))
        if f1 > best:
            best, bt = f1, float(t)
    return best, bt

BANDS = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0)]
xc = (torch.arange(GH, dtype=torch.float32) + 0.5) * CELL
yc = -grid.y_half_m + (torch.arange(GW, dtype=torch.float32) + 0.5) * CELL

def lateral_centroid_err(pred, true, dev):
    """Mass-weighted lateral centroid |dy| in metres, per range band.
    `pred` and `true` are [n, GH*GW] on `dev`."""
    out = {}
    P = pred.reshape(-1, GH, GW).clamp(min=0.0)
    T = true.reshape(-1, GH, GW)
    yv = yc.to(dev)
    for lo, hi in BANDS:
        r = (xc >= lo) & (xc < hi)
        r = r.to(dev)
        tb, pb = T[:, r], P[:, r]
        mt = tb.sum(dim=(1, 2))
        ok = mt > 0
        if int(ok.sum()) == 0:
            out[f"{lo:.0f}-{hi:.0f}m"] = {"n": 0}
            continue
        cy_t = (tb * yv).sum(dim=(1, 2)) / mt.clamp(min=1e-9)
        mp = pb.sum(dim=(1, 2)).clamp(min=1e-9)
        cy_p = (pb * yv).sum(dim=(1, 2)) / mp
        e = (cy_t[ok] - cy_p[ok]).abs()
        out[f"{lo:.0f}-{hi:.0f}m"] = {"n": int(ok.sum()),
                                      "mae_y_m": float(e.mean()),
                                      "med_y_m": float(e.median())}
    return out

# ---- dual ridge ------------------------------------------------------------
LAMBDAS = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]

Yf = Yt[fi].to(dev)
Yv = Yt[vi].to(dev)
Yte = Yt[ti].to(dev)
fovm_d = fovm.to(dev)

def gram(X, rows_a, rows_b, chunk=2048):
    """K[a, b] = Xa @ Xb.T on GPU, streaming from CPU float32."""
    Ka = torch.empty(len(rows_a), len(rows_b), dtype=torch.float32, device=dev)
    for s in range(0, len(rows_a), chunk):
        xa = X[rows_a[s:s + chunk]].to(dev, non_blocking=True)
        for u in range(0, len(rows_b), chunk):
            xb = X[rows_b[u:u + chunk]].to(dev, non_blocking=True)
            Ka[s:s + xa.shape[0], u:u + xb.shape[0]] = xa @ xb.T
            del xb
        del xa
    return Ka

results = {}
t_all = time.time()
for name, kind, ker in ARMS:
    t0 = time.time()
    X = build_X(kind, ker)                       # [N, d] CPU float32
    d = X.shape[1]
    mu = X[fi].mean(0, keepdim=True)
    X = X - mu                                   # centre on FIT only
    Kff = gram(X, fi, fi)
    Kvf = gram(X, vi, fi)
    Ktf = gram(X, ti, fi)
    eye = torch.eye(len(fi), device=dev)
    ymu = Yf.mean(0, keepdim=True)
    Yfc = Yf - ymu
    best = None
    for lam in LAMBDAS:
        try:
            L = torch.linalg.cholesky(Kff + lam * eye)
        except Exception:
            continue
        alpha = torch.cholesky_solve(Yfc, L)
        pv = Kvf @ alpha + ymu
        apv = average_precision(pv[:, fovm_d].reshape(-1),
                                Yv[:, fovm_d].reshape(-1))
        if best is None or apv > best[1]:
            best = (lam, apv, alpha)
        del L
    lam, ap_val, alpha = best
    pt = Ktf @ alpha + ymu
    ap_all = average_precision(pt.reshape(-1), Yte.reshape(-1))
    ap_fov = average_precision(pt[:, fovm_d].reshape(-1), Yte[:, fovm_d].reshape(-1))
    f1, thr = best_f1(pt[:, fovm_d].reshape(-1), Yte[:, fovm_d].reshape(-1))
    p = (pt[:, fovm_d] >= thr).float()
    yy = Yte[:, fovm_d]
    inter = float((p * yy).sum())
    iou = inter / float((p + yy - p * yy).sum() + 1e-9)
    loc = lateral_centroid_err(pt, Yte, dev)
    results[name] = {"d": int(d), "n_fit": int(len(fi)), "n_val": int(len(vi)),
                     "n_test": int(len(ti)), "lambda": lam,
                     "ap_val_infov": ap_val, "ap_test_all": ap_all,
                     "ap_test_infov": ap_fov, "f1_infov": f1, "iou_infov": iou,
                     "thr": thr, "lateral": loc,
                     "az_cols": TW // ker[1],
                     "deg_per_col": az_per_col(TW // ker[1]),
                     "lat_width_at_30m": lat_width_at(TW // ker[1], 30.0),
                     "wall_s": round(time.time() - t0, 1)}
    print(f"[{name:10s}] d={d:6d} lam={lam:<7g} AP_fov={ap_fov:.4f} "
          f"AP_all={ap_all:.4f} F1={f1:.4f} IoU={iou:.4f} "
          f"latMAE@15-30={loc['15-30m'].get('mae_y_m', float('nan')):.3f} m "
          f"({time.time()-t0:.0f}s)", flush=True)
    # bank test predictions for the paired bootstrap
    np.save(os.path.join(a.bank, f"pred_{name}.npy"),
            pt.to(torch.float16).cpu().numpy())
    del X, Kff, Kvf, Ktf, alpha, pt
    torch.cuda.empty_cache()

# ---- CONTROLS --------------------------------------------------------------
ctrl = {}
# constant: fit per-cell mean
cmean = Yf.mean(0, keepdim=True).expand(len(ti), -1)
ctrl["constant"] = {
    "ap_test_infov": average_precision(cmean[:, fovm_d].reshape(-1),
                                       Yte[:, fovm_d].reshape(-1)),
    "ap_test_all": average_precision(cmean.reshape(-1), Yte.reshape(-1)),
    "lateral": lateral_centroid_err(cmean, Yte, dev)}
# all-zero
zero = torch.zeros_like(cmean)
ctrl["allzero"] = {
    "ap_test_infov": average_precision(zero[:, fovm_d].reshape(-1),
                                       Yte[:, fovm_d].reshape(-1)),
    "accuracy_all": float((Yte == 0).float().mean()),
    "accuracy_infov": float((Yte[:, fovm_d] == 0).float().mean()),
    "iou": 0.0, "f1": 0.0}
np.save(os.path.join(a.bank, "pred_constant.npy"),
        cmean.to(torch.float16).cpu().numpy())

# shuffled: best arm's features vs targets permuted ACROSS clips
X = build_X("tok", (1, 1))
X = X - X[fi].mean(0, keepdim=True)
perm = torch.from_numpy(rng.permutation(len(fi)).copy()).to(dev)
Kff = gram(X, fi, fi); Ktf = gram(X, ti, fi)
eye = torch.eye(len(fi), device=dev)
Yfs = Yf[perm]
ymu = Yfs.mean(0, keepdim=True)
L = torch.linalg.cholesky(Kff + results["tok_16x40"]["lambda"] * eye)
alpha = torch.cholesky_solve(Yfs - ymu, L)
pts = Ktf @ alpha + ymu
ctrl["shuffled_tok_16x40"] = {
    "ap_test_infov": average_precision(pts[:, fovm_d].reshape(-1),
                                       Yte[:, fovm_d].reshape(-1)),
    "lateral": lateral_centroid_err(pts, Yte, dev)}
del X, Kff, Ktf, alpha, pts
torch.cuda.empty_cache()

print("[controls]", json.dumps({k: {kk: vv for kk, vv in v.items()
                                    if kk != "lateral"}
                                for k, v in ctrl.items()}, indent=1))

out = {"_evidence_class": "MEASURED (ours; artifact = this file + the bank)",
       "tier": "NOT APPLICABLE - representation probe, no trajectory is produced",
       "function_class": "LINEAR (exact dual ridge, linear kernel)",
       "bank": a.bank, "ckpt": idx["ckpt"], "step": idx["step"],
       "seed": a.seed, "n_clips": len(clips),
       "split_clips": {"fit": len(clips) - n_te - n_va, "val": n_va, "test": n_te},
       "base_rate_test_all": base_rate_te,
       "base_rate_test_infov": base_rate_te_fov,
       "allzero_accuracy_all": float((Yte == 0).float().mean()),
       "grid": [GH, GW], "cell_m": CELL,
       "arms": results, "controls": ctrl,
       "test_clip_idx": sorted(int(c) for c in te_c),
       "test_rows_clip": plan[ti, 0].tolist()}
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
print("WROTE", a.out, f"total {time.time()-t_all:.0f}s")
