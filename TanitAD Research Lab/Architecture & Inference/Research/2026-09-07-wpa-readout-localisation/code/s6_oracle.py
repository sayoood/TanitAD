# -*- coding: utf-8 -*-
"""WP-A step 6 -- the READOUT CEILING under a PERFECT front-end (ORACLE arm).

⛔⛔ THIS IS AN ORACLE. Its input is BUILT FROM THE TARGET. It is NOT a
perception result and must never be quoted as one. It answers exactly one
question, which is the question WP-A was asked:

   ⭐ IF the encoder were PERFECT, how much BEV agent localisation would each
      READOUT GEOMETRY still deliver -- and how much does the pool destroy?

WHY IT IS NEEDED. s5 measured the real v7-tiny trunk: the head FITS its training
split (AP 0.205 vs the pos-only control's 0.036) and NONE of it transfers
(test AP 0.029 vs pos_only 0.033). With no transferable signal at the TOP of the
ladder there is nothing for the pool to destroy, so the ladder cannot be ranked
on that trunk. The oracle removes the encoder from the question entirely.

CONSTRUCTION. The GT BEV occupancy is re-projected into the encoder's own
(row = elevation, col = azimuth) token grid:

    col  = (f_ref * atan2(y, x) + W/2) / patch        cylindrical -> LINEAR in az
    row  = (v0 + f_ref * h_cam / range) / patch       ground-plane inverse depth

⭐ POSITION LIVES ONLY IN THE ADDRESS. The map's VALUE is presence / count --
never a coordinate -- because "the trajectory's geometry is the address, not a
shared latent" is the property under test. Painting az or range as a channel
value would let a 1x1 map recover position and would measure nothing.

Then every arm is avg_pool(k) -> nearest-upsample to 16x40 -> the SAME head
(identical architecture, params, steps, seed), so the ONLY difference between
arms is what the pool destroyed.

LADDER (⭐ the two axes are separated, because the designer needs the attribution)
  16x40  full            | 16x4  columns pooled ONLY  | 4x40  rows pooled ONLY
  8x20   refcv5's map    | 4x8   v7-tiny DEPLOYED     | 4x4   v1/v6 flagship
CONTROLS (known values)
  pos_only  features zeroed -> the marginal field only
  allzero   AP == base rate EXACTLY, IoU 0, F1 0
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
from torch import nn

from tanitad.data.bev_raster import BEVGrid, cell_centers_xy, fov_mask

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default=r"C:\Users\Admin\wpa-readout\bank\k8r1")
ap.add_argument("--out", default=r"C:\Users\Admin\wpa-readout\raw\oracle.json")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--cells", type=int, default=2048)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0, help="SPLIT seed -- keep fixed")
ap.add_argument("--train-seed", type=int, default=None,
                help="TRAINING seed only; the split is untouched, so this is a "
                     "true replicate arm (H-ESTIM-SEED-1)")
ap.add_argument("--h-cam", type=float, default=1.5)
ap.add_argument("--arms", default="")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
idx = json.load(open(os.path.join(a.bank, "idx.json"), encoding="utf-8"))
clips = idx["clips"]; plan = np.asarray(idx["plan"], dtype=np.int64)
N = int(idx["N"]); TH, TW = idx["token_grid"]
GH, GW = idx["grid"]; CELL = float(idx["cell_m"])
PATCH, W_PX, H_PX, F_REF = 16, 640, 256, 305.5774907364391
grid = BEVGrid(x_fwd_m=float(idx["x_fwd_m"]), y_half_m=float(idx["y_half_m"]),
               cell_m=CELL)
Yall = np.asarray(np.load(os.path.join(a.bank, "y.npy"), mmap_mode="r"))

# ---- address: BEV cell -> (token row, token col) ---------------------------
X_m, Y_m = cell_centers_xy(grid)
az = np.arctan2(Y_m, X_m)
colf = (F_REF * az + W_PX / 2.0) / PATCH - 0.5
rng_m = np.sqrt(X_m ** 2 + Y_m ** 2)
vpx = H_PX / 2.0 + F_REF * a.h_cam / np.maximum(rng_m, 1e-6)
rowf = vpx / PATCH - 0.5
in_fov = fov_mask(grid, math.radians(60.0))
valid = in_fov & (colf >= 0) & (colf <= TW - 1) & (rowf >= 0) & (rowf <= TH - 1)
vidx = np.flatnonzero(valid.reshape(-1))
NC = len(vidx)
ri = np.clip(np.rint(rowf.reshape(-1)[vidx]), 0, TH - 1).astype(np.int64)
ci = np.clip(np.rint(colf.reshape(-1)[vidx]), 0, TW - 1).astype(np.int64)
tokflat = torch.from_numpy(ri * TW + ci).to(dev)
print(f"[address] in-field {int(in_fov.sum())}  addressable {NC}/{valid.size} "
      f"({100*valid.mean():.2f} %)")
print(f"[rows] token rows used {sorted(set(ri.tolist()))}  "
      f"(h_cam {a.h_cam} m, v0 {H_PX/2:.0f}px)  distinct (row,col) cells "
      f"{len(set(zip(ri.tolist(), ci.tolist())))} of {TH*TW}")
u, cnt = np.unique(ri * TW + ci, return_counts=True)
print(f"[address] BEV cells per token cell: median {np.median(cnt):.1f} "
      f"max {cnt.max()}  -- the 16x40 grid's OWN quantisation")

# positional embedding (all that pos_only gets)
xs = X_m.reshape(-1)[vidx]; ys = Y_m.reshape(-1)[vidx]; azv = az.reshape(-1)[vidx]
pf = [xs / 60.0, ys / 16.0, np.log1p(xs) / 4.2, azv / 1.05, rng_m.reshape(-1)[vidx] / 60.0]
for k in (1, 2, 4, 8):
    pf += [np.sin(k * azv), np.cos(k * azv), np.sin(k * xs / 20.0), np.cos(k * xs / 20.0)]
POS = torch.from_numpy(np.stack(pf, 1).astype(np.float32)).to(dev)
PD = POS.shape[1]

# bilinear column-sampling matrix (the head's own lookup)
c = np.clip(colf.reshape(-1)[vidx], 0, TW - 1 - 1e-6)
c0 = np.floor(c).astype(np.int64); w1 = c - c0; c1 = np.minimum(c0 + 1, TW - 1)
S = np.zeros((NC, TW), dtype=np.float32)
S[np.arange(NC), c0] += 1.0 - w1
S[np.arange(NC), c1] += w1
S_t = torch.from_numpy(S).to(dev)

Yv = torch.from_numpy(Yall[:, vidx].copy())
base = float(Yv.float().mean())
print(f"[occupancy] addressable base rate {base:.6f} -> ALL-ZERO accuracy "
      f"{100*(1-base):.4f} %")

# ---- the ORACLE map: GT occupancy scattered into the token grid ------------
ORAC = torch.zeros(N, 2, TH * TW, dtype=torch.float16)
_y = Yv.float()
for s in range(0, N, 512):
    blk = _y[s:s + 512].to(dev)                                  # [b, NC]
    m = torch.zeros(blk.shape[0], TH * TW, device=dev)
    m.index_add_(1, tokflat, blk)                                # COUNT
    ORAC[s:s + blk.shape[0], 0] = (m > 0).to(torch.float16).cpu()   # presence
    ORAC[s:s + blk.shape[0], 1] = torch.log1p(m).to(torch.float16).cpu()
ORAC = ORAC.reshape(N, 2, TH, TW)
print(f"[oracle] mean presence {float(ORAC[:, 0].float().mean()):.5f}  "
      f"frames with any {int((ORAC[:,0].reshape(N,-1).sum(1)>0).sum())}/{N}")
assert float(ORAC[:, 0].float().mean()) > 1e-4, "ORACLE MAP IS DEGENERATE"

# ---- split -----------------------------------------------------------------
rng = np.random.default_rng(a.seed)
order = rng.permutation(len(clips))
n_te, n_va = 30, 25
te_c, va_c = set(order[:n_te].tolist()), set(order[n_te:n_te + n_va].tolist())
cl = plan[:, 0]
fi = np.where(~np.isin(cl, list(te_c | va_c)))[0]
vi = np.where(np.isin(cl, list(va_c)))[0]
ti = np.where(np.isin(cl, list(te_c)))[0]
print(f"[split] rows {len(fi)}/{len(vi)}/{len(ti)}")

ARMS = [("orc_16x40", (1, 1)), ("orc_8x20", (2, 2)), ("orc_16x4", (1, 10)),
        ("orc_4x40", (4, 1)), ("orc_4x8", (4, 5)), ("orc_4x4", (4, 10)),
        ("orc_4x2", (4, 20)), ("orc_1x1", (16, 40)), ("pos_only", (1, 1))]
if a.arms:
    keep = set(a.arms.split(","))
    ARMS = [x for x in ARMS if x[0] in keep]

D_RED = 16
class Head(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.red = nn.Conv2d(cin, D_RED, 1)
        self.mix = nn.Sequential(nn.Conv2d(D_RED, D_RED, 3, padding=1), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(TH * D_RED + PD, 192), nn.GELU(),
                                 nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, fmap, cellsel, zero_feat=False):
        b = fmap.shape[0]
        x = self.red(fmap)
        x = x + self.mix(x)
        x = x.reshape(b, TH * D_RED, TW)
        g = torch.einsum("bdw,kw->bkd", x, S_t[cellsel])
        if zero_feat:
            g = torch.zeros_like(g)
        p = POS[cellsel].unsqueeze(0).expand(b, -1, -1)
        return self.mlp(torch.cat([g, p], -1)).squeeze(-1)

def feat(rows, ker):
    x = ORAC[rows].to(torch.float32)
    if ker != (1, 1):
        x = torch.nn.functional.avg_pool2d(x, ker)
        x = torch.nn.functional.interpolate(x, size=(TH, TW), mode="nearest")
    return x

def average_precision(scores, labels):
    o = torch.argsort(scores, descending=True)
    l = labels[o]
    tp = torch.cumsum(l, 0); fp = torch.cumsum(1 - l, 0)
    rec = tp / l.sum()
    dr = torch.diff(rec, prepend=torch.zeros(1, device=rec.device))
    return float(((tp / (tp + fp)) * dr).sum())

xs_t = torch.from_numpy(xs.astype(np.float32)).to(dev)
ys_t = torch.from_numpy(ys.astype(np.float32)).to(dev)
BANDS = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0)]

def lateral_mae(prob, true):
    out = {}
    for lo, hi in BANDS:
        m = (xs_t >= lo) & (xs_t < hi)
        t, p = true[:, m], prob[:, m].clamp(min=0)
        mt = t.sum(1); ok = mt > 0
        if int(ok.sum()) == 0:
            out[f"{lo:.0f}-{hi:.0f}m"] = {"n": 0}; continue
        yv = ys_t[m]
        e = ((t * yv).sum(1) / mt.clamp(min=1e-9)
             - (p * yv).sum(1) / p.sum(1).clamp(min=1e-9)).abs()
        out[f"{lo:.0f}-{hi:.0f}m"] = {"n": int(ok.sum()), "mae_y_m": float(e[ok].mean()),
                                      "med_y_m": float(e[ok].median())}
    return out

@torch.no_grad()
def evaluate(head, rows, ker, zf):
    head.eval(); P, T = [], []
    allc = torch.arange(NC, device=dev)
    for s in range(0, len(rows), 48):
        r = rows[s:s + 48]
        P.append(torch.sigmoid(head(feat(r, ker).to(dev), allc, zf)).cpu())
        T.append(Yv[r].float())
    head.train()
    return torch.cat(P).to(dev), torch.cat(T).to(dev)

res = {}
t_all = time.time()
TS = a.seed if a.train_seed is None else a.train_seed
for name, ker in ARMS:
    torch.manual_seed(TS); np.random.seed(TS)
    zf = (name == "pos_only")
    head = Head(2).to(dev)
    npar = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - base) / base], device=dev))
    g = np.random.default_rng(TS)
    t0 = time.time()
    for step in range(a.steps):
        rows = np.sort(g.choice(fi, a.batch, replace=False))
        cs = torch.from_numpy(g.choice(NC, a.cells, replace=False)).to(dev)
        y = Yv[rows][:, cs.cpu()].float().to(dev)
        loss = lossf(head(feat(rows, ker).to(dev), cs, zf), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    Pv, Tv = evaluate(head, vi, ker, zf)
    qs = torch.quantile(Pv.reshape(-1)[::7].float(),
                        torch.linspace(0.90, 0.9999, 120, device=dev))
    bf, bt = 0.0, 0.5
    for t in qs:
        pr = (Pv >= t).float()
        f1 = float(2 * (pr * Tv).sum() / (pr.sum() + Tv.sum() + 1e-9))
        if f1 > bf:
            bf, bt = f1, float(t)
    Pt, Tt = evaluate(head, ti, ker, zf)
    apt = average_precision(Pt.reshape(-1), Tt.reshape(-1))
    pr = (Pt >= bt).float(); inter = float((pr * Tt).sum())
    iou = inter / float((pr + Tt - pr * Tt).sum() + 1e-9)
    f1 = float(2 * inter / (pr.sum() + Tt.sum() + 1e-9))
    lat = lateral_mae(Pt, Tt)
    nc_ = TW // ker[1]
    res[name] = {"params": int(npar), "pool": list(ker), "az_cols": nc_,
                 "rows": TH // ker[0], "deg_per_col": 120.0 / nc_,
                 "lat_width_at_30m": 2 * 30 * math.tan(math.radians(60.0 / nc_)),
                 "ap_test": apt, "f1_test": f1, "iou_test": iou,
                 "thr_from_val": bt, "f1_val": bf, "lateral": lat,
                 "wall_s": round(time.time() - t0, 1)}
    print(f"[{name:10s}] rows={TH//ker[0]:2d} az={nc_:2d}  AP {apt:.4f}  F1 {f1:.4f}  "
          f"IoU {iou:.4f}  latMAE 0-15 {lat['0-15m'].get('mae_y_m',float('nan')):.3f} "
          f"15-30 {lat['15-30m'].get('mae_y_m',float('nan')):.3f} "
          f"30-45 {lat['30-45m'].get('mae_y_m',float('nan')):.3f} "
          f"45-60 {lat['45-60m'].get('mae_y_m',float('nan')):.3f} m ({time.time()-t0:.0f}s)",
          flush=True)
    np.save(os.path.join(a.bank, f"or_{name}.npy"), Pt.to(torch.float16).cpu().numpy())
    del head, opt; torch.cuda.empty_cache()

Tt_all = Yv[ti].float().to(dev)
az0 = {"ap": average_precision(torch.zeros_like(Tt_all).reshape(-1), Tt_all.reshape(-1)),
       "accuracy": float((Tt_all == 0).float().mean()), "iou": 0.0, "f1": 0.0,
       "base_rate_test": float(Tt_all.mean())}
print("[allzero]", json.dumps(az0))

json.dump({"_evidence_class": "MEASURED (ours; artifact = this file + the bank)",
           "tier": "NOT APPLICABLE - representation probe, no trajectory produced",
           "ORACLE": "⛔ the feature map is BUILT FROM THE TARGET. This is a "
                     "READOUT-CEILING measurement, NOT a perception result.",
           "position_in_address_only": "the map's values are presence/count; no "
                                       "coordinate is painted as a value",
           "row_model": {"v0_px": H_PX / 2, "h_cam_m": a.h_cam,
                         "formula": "v = v0 + f_ref*h/range (ground-plane inverse "
                                    "depth); a MODEL of the vertical axis, stated"},
           "projection_scope": "column LINEAR in azimuth: OUR 256x640 cylindrical "
                               "corpus, f_ref 305.577, FOV 120 deg; a pinhole "
                               "formula gives 92.6 deg and does not travel",
           "bank": a.bank, "split_seed": a.seed, "train_seed": TS, "steps": a.steps,
           "n_addressable_cells": int(NC), "base_rate": base,
           "bev_cells_per_token_cell": {"median": float(np.median(cnt)),
                                        "max": int(cnt.max()),
                                        "n_token_cells_used": int(len(u))},
           "arms": res, "controls": {"allzero": az0}},
          open(a.out, "w", encoding="utf-8"), indent=1)
print("WROTE", a.out, f"{time.time()-t_all:.0f}s")
