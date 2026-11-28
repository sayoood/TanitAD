# -*- coding: utf-8 -*-
"""WP-A step 5 -- THE DECISIVE PROBE: azimuth-INDEXED occupancy decoding.

This is DiffusionDrive's coupling (1) in miniature: the BEV cell's own geometry
is the ADDRESS into the image feature map.  For BEV cell (x, y):

    az = atan2(y, x)                       (ego frame, +x fwd, +y left)
    u  = f_ref * az + W/2                  ⭐ CYLINDRICAL -> column is LINEAR in
                                             azimuth.  Scoped to OUR 256x640
                                             f_ref 305.577 corpus; a pinhole
                                             formula gives 92.6 deg and is wrong here.
    c  = u / patch                         -> a fractional token column in [0, 40)

The probe bilinearly samples the feature map at column ``c`` over ALL 16 rows
(row = elevation = the monocular range cue) and feeds that, plus a positional
embedding of (x, y), to a SHARED MLP that emits that cell's occupancy logit.

⭐ WHY THIS ANSWERS WP-A AND THE LINEAR PROBE DID NOT
Every arm is ``avg_pool(k) -> nearest-upsample back to 16x40``.  The head is
then BYTE-IDENTICAL in architecture, parameter count, optimiser, steps and seed
across arms, so the ONLY difference between arms is the information the pool
destroyed.  A 4-column arm literally cannot tell two cells apart when their
azimuths fall in the same 30 deg wedge -- except through the positional
embedding, which is exactly what the ``pos_only`` control isolates.

CONTROLS -- each must read a KNOWN value
  pos_only    features ZEROED: the head can only learn the MARGINAL occupancy
              field.  Any arm that does not beat this has added nothing.
  allzero     AP == the positive base rate EXACTLY; IoU 0; accuracy printed.
  shuffled    features taken from a RANDOM OTHER FRAME -> must fall to pos_only.
  pix_*       raw 9-channel patch means: the GEOMETRY ceiling with no trunk.
⛔ Never accuracy: occupancy is ~1.7 % of in-field cells.
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
from torch import nn

from tanitad.data.bev_raster import BEVGrid, cell_centers_xy, fov_mask

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default=r"C:\Users\Admin\wpa-readout\bank\k8r1")
ap.add_argument("--out", default=r"C:\Users\Admin\wpa-readout\raw\indexed_k8.json")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--cells", type=int, default=1536, help="cells sampled per frame in training")
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arms", default="")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
idx = json.load(open(os.path.join(a.bank, "idx.json"), encoding="utf-8"))
clips = idx["clips"]; plan = np.asarray(idx["plan"], dtype=np.int64)
N = int(idx["N"]); TH, TW = idx["token_grid"]; DM = int(idx["d_model"])
CIN = int(idx["in_channels"]); GH, GW = idx["grid"]; CELL = float(idx["cell_m"])
PATCH = 16
W_PX, F_REF = 640, 305.5774907364391
grid = BEVGrid(x_fwd_m=float(idx["x_fwd_m"]), y_half_m=float(idx["y_half_m"]),
               cell_m=CELL)

tok = np.load(os.path.join(a.bank, "tok.npy"), mmap_mode="r")
pix = np.load(os.path.join(a.bank, "pix.npy"), mmap_mode="r")
Yall = np.asarray(np.load(os.path.join(a.bank, "y.npy"), mmap_mode="r"))

# ---- the ADDRESS: BEV cell -> fractional token column ----------------------
X_m, Y_m = cell_centers_xy(grid)                       # [120, 64] each
az = np.arctan2(Y_m, X_m)                              # +y left -> +az left
u = F_REF * az + W_PX / 2.0                            # cylindrical: LINEAR
col = u / PATCH - 0.5                                  # token-centre coords
in_fov = fov_mask(grid, math.radians(60.0))
valid = in_fov & (col >= 0) & (col <= TW - 1)
print(f"[address] in-field {int(in_fov.sum())}/{in_fov.size}  "
      f"addressable {int(valid.sum())}/{valid.size} "
      f"({100*valid.mean():.2f} %)  col range {col[valid].min():.2f}..{col[valid].max():.2f}")
vidx = np.flatnonzero(valid.reshape(-1))
NC = len(vidx)

# bilinear sampling matrix over COLUMNS: S [NC, TW]
c = np.clip(col.reshape(-1)[vidx], 0, TW - 1 - 1e-6)
c0 = np.floor(c).astype(np.int64); w1 = c - c0; c1 = np.minimum(c0 + 1, TW - 1)
S = np.zeros((NC, TW), dtype=np.float32)
S[np.arange(NC), c0] += 1.0 - w1
S[np.arange(NC), c1] += w1
S_t = torch.from_numpy(S).to(dev)

# positional embedding of the cell itself (this is ALL pos_only gets)
xs = X_m.reshape(-1)[vidx]; ys = Y_m.reshape(-1)[vidx]
pf = [xs / 60.0, ys / 16.0, np.log1p(xs) / 4.2, az.reshape(-1)[vidx] / 1.05,
      np.sqrt(xs ** 2 + ys ** 2) / 60.0]
for k in (1, 2, 4, 8):
    pf += [np.sin(k * az.reshape(-1)[vidx]), np.cos(k * az.reshape(-1)[vidx]),
           np.sin(k * xs / 20.0), np.cos(k * xs / 20.0)]
POS = torch.from_numpy(np.stack(pf, 1).astype(np.float32)).to(dev)   # [NC, 21]
PD = POS.shape[1]
print(f"[probe] addressable cells {NC}  pos dim {PD}")

Yv = torch.from_numpy(Yall[:, vidx].copy())            # [N, NC] uint8, CPU
base = float(Yv.float().mean())
print(f"[occupancy] addressable-cell base rate {base:.6f} -> "
      f"ALL-ZERO accuracy {100*(1-base):.4f} %")

# ---- split -----------------------------------------------------------------
rng = np.random.default_rng(a.seed)
order = rng.permutation(len(clips))
n_te, n_va = 30, 25
te_c, va_c = set(order[:n_te].tolist()), set(order[n_te:n_te + n_va].tolist())
cl = plan[:, 0]
fi = np.where(~np.isin(cl, list(te_c | va_c)))[0]
vi = np.where(np.isin(cl, list(va_c)))[0]
ti = np.where(np.isin(cl, list(te_c)))[0]
print(f"[split] clips {len(clips)-n_te-n_va}/{n_va}/{n_te}  rows {len(fi)}/{len(vi)}/{len(ti)}")

ARMS = [("tok_16x40", "tok", (1, 1)), ("tok_8x20", "tok", (2, 2)),
        ("tok_4x8", "tok", (4, 5)), ("tok_4x4", "tok", (4, 10)),
        ("tok_4x2", "tok", (4, 20)), ("tok_1x1", "tok", (16, 40)),
        ("pix_16x40", "pix", (1, 1)), ("pix_8x20", "pix", (2, 2)),
        ("pix_4x4", "pix", (4, 10)),
        ("pos_only", "tok", (1, 1)), ("shuffled", "tok", (1, 1))]
if a.arms:
    keep = set(a.arms.split(","))
    ARMS = [x for x in ARMS if x[0] in keep]

D_RED = 16                       # 1x1 reduction before the column lookup
class Head(nn.Module):
    """IDENTICAL for every arm -- same shapes, same params, same init."""
    def __init__(self, cin):
        super().__init__()
        self.red = nn.Conv2d(cin, D_RED, 1)
        self.mix = nn.Sequential(nn.Conv2d(D_RED, D_RED, 3, padding=1), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(TH * D_RED + PD, 192), nn.GELU(),
                                 nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, fmap, cellsel, zero_feat=False):
        """fmap [B, C, 16, 40] -> logits [B, len(cellsel)]"""
        b = fmap.shape[0]
        x = self.red(fmap)
        x = x + self.mix(x)
        x = x.reshape(b, TH * D_RED, TW)                       # [B, 16*Dr, 40]
        s = S_t[cellsel]                                       # [K, 40]
        g = torch.einsum("bdw,kw->bkd", x, s)                  # [B, K, 16*Dr]
        if zero_feat:
            g = torch.zeros_like(g)
        p = POS[cellsel].unsqueeze(0).expand(b, -1, -1)
        return self.mlp(torch.cat([g, p], -1)).squeeze(-1)

def load_feat(src_np, rows, ker, C):
    x = torch.from_numpy(np.asarray(src_np[rows], dtype=np.float32))
    x = x.transpose(1, 2).reshape(-1, C, TH, TW)
    if ker != (1, 1):
        x = torch.nn.functional.avg_pool2d(x, ker)
        # ⭐ unpool back to 16x40 so the head is IDENTICAL for every arm and the
        # ONLY difference between arms is what the pool destroyed.
        x = torch.nn.functional.interpolate(x, size=(TH, TW), mode="nearest")
    return x

def average_precision(scores, labels):
    o = torch.argsort(scores, descending=True)
    l = labels[o]
    tp = torch.cumsum(l, 0); fp = torch.cumsum(1 - l, 0)
    prec = tp / (tp + fp); npos = l.sum()
    rec = tp / npos
    dr = torch.diff(rec, prepend=torch.zeros(1, device=rec.device))
    return float((prec * dr).sum())

xs_t = torch.from_numpy(xs.astype(np.float32)).to(dev)
ys_t = torch.from_numpy(ys.astype(np.float32)).to(dev)
BANDS = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0)]

def lateral_mae(prob, true):
    out = {}
    for lo, hi in BANDS:
        m = (xs_t >= lo) & (xs_t < hi)
        t = true[:, m]; p = prob[:, m].clamp(min=0)
        mt = t.sum(1); ok = mt > 0
        if int(ok.sum()) == 0:
            out[f"{lo:.0f}-{hi:.0f}m"] = {"n": 0}; continue
        yv = ys_t[m]
        cy_t = (t * yv).sum(1) / mt.clamp(min=1e-9)
        cy_p = (p * yv).sum(1) / p.sum(1).clamp(min=1e-9)
        e = (cy_t[ok] - cy_p[ok]).abs()
        out[f"{lo:.0f}-{hi:.0f}m"] = {"n": int(ok.sum()), "mae_y_m": float(e.mean()),
                                      "med_y_m": float(e.median())}
    return out

@torch.no_grad()
def evaluate(head, rows, src_np, ker, C, zero_feat, shuffle_rows):
    head.eval()
    P, T = [], []
    allc = torch.arange(NC, device=dev)
    for s in range(0, len(rows), 32):
        r = rows[s:s + 32]
        fr = shuffle_rows[s:s + 32] if shuffle_rows is not None else r
        f = load_feat(src_np, fr, ker, C).to(dev)
        P.append(torch.sigmoid(head(f, allc, zero_feat)).cpu())
        T.append(Yv[r].float())
    head.train()
    return torch.cat(P).to(dev), torch.cat(T).to(dev)

res = {}
t_all = time.time()
for name, kind, ker in ARMS:
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    src_np = tok if kind == "tok" else pix
    C = DM if kind == "tok" else CIN
    zero_feat = (name == "pos_only")
    head = Head(C).to(dev)
    nparam = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    pw = torch.tensor([(1 - base) / base], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    g = np.random.default_rng(a.seed)
    t0 = time.time()
    for step in range(a.steps):
        rows = np.sort(g.choice(fi, a.batch, replace=False))
        frows = np.sort(g.choice(fi, a.batch, replace=False)) if name == "shuffled" else rows
        f = load_feat(src_np, frows, ker, C).to(dev)
        cs = torch.from_numpy(g.choice(NC, a.cells, replace=False)).to(dev)
        y = Yv[rows][:, cs.cpu()].float().to(dev)
        loss = lossf(head(f, cs, zero_feat), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % max(1, a.steps // 4) == 0 or step == a.steps - 1:
            print(f"    step {step:5d} loss {float(loss):.5f}", flush=True)
    # --- inner-val: choose the operating threshold there, never on test ------
    sr = np.sort(g.choice(fi, len(vi), replace=True)) if name == "shuffled" else None
    Pv, Tv = evaluate(head, vi, src_np, ker, C, zero_feat, sr)
    qs = torch.quantile(Pv.reshape(-1)[::7].float(),
                        torch.linspace(0.90, 0.9999, 120, device=dev))
    bf, bt = 0.0, 0.5
    for t in qs:
        pr = (Pv >= t).float()
        f1 = float(2 * (pr * Tv).sum() / (pr.sum() + Tv.sum() + 1e-9))
        if f1 > bf:
            bf, bt = f1, float(t)
    fit_probe = np.sort(g.choice(fi, min(600, len(fi)), replace=False))
    Pf, Tf_ = evaluate(head, fit_probe, src_np, ker, C, zero_feat,
                       np.sort(g.choice(fi, len(fit_probe), replace=False))
                       if name == "shuffled" else None)
    ap_fit = average_precision(Pf.reshape(-1), Tf_.reshape(-1))
    sr = np.sort(g.choice(fi, len(ti), replace=True)) if name == "shuffled" else None
    Pt, Tt = evaluate(head, ti, src_np, ker, C, zero_feat, sr)
    ap = average_precision(Pt.reshape(-1), Tt.reshape(-1))
    pr = (Pt >= bt).float()
    inter = float((pr * Tt).sum())
    iou = inter / float((pr + Tt - pr * Tt).sum() + 1e-9)
    f1 = float(2 * inter / (pr.sum() + Tt.sum() + 1e-9))
    lat = lateral_mae(Pt, Tt)
    res[name] = {"params": int(nparam), "az_cols": TW // ker[1],
                 "deg_per_col": 120.0 / (TW // ker[1]),
                 "lat_width_at_30m": 2 * 30 * math.tan(math.radians(60.0 / (TW // ker[1]))),
                 "ap_test": ap, "ap_fit": ap_fit, "f1_test": f1, "iou_test": iou, "thr_from_val": bt,
                 "f1_val": bf, "lateral": lat, "steps": a.steps,
                 "n_test_frames": int(len(ti)), "n_cells": int(NC),
                 "wall_s": round(time.time() - t0, 1)}
    np.save(os.path.join(a.bank, f"ix_{name}.npy"), Pt.to(torch.float16).cpu().numpy())
    print(f"[{name:10s}] az={res[name]['az_cols']:2d} par={nparam/1e3:5.1f}k  "
          f"AP {ap:.4f} (fit {ap_fit:.4f})  F1 {f1:.4f}  IoU {iou:.4f}  "
          f"latMAE 0-15 {lat['0-15m'].get('mae_y_m', float('nan')):.3f} "
          f"15-30 {lat['15-30m'].get('mae_y_m', float('nan')):.3f} "
          f"30-45 {lat['30-45m'].get('mae_y_m', float('nan')):.3f} m "
          f"({time.time()-t0:.0f}s)", flush=True)
    del head, opt
    torch.cuda.empty_cache()

Tt_all = Yv[ti].float().to(dev)
zero = torch.zeros_like(Tt_all)
allzero = {"ap": average_precision(zero.reshape(-1), Tt_all.reshape(-1)),
           "accuracy": float((Tt_all == 0).float().mean()),
           "iou": 0.0, "f1": 0.0,
           "base_rate_test": float(Tt_all.mean())}
print("[allzero]", json.dumps(allzero))

out = {"_evidence_class": "MEASURED (ours; artifact = this file + the bank)",
       "tier": "NOT APPLICABLE - representation probe, no trajectory produced",
       "function_class": "NONLINEAR (conv 1x1 + 3x3 residual, 3-layer MLP head, "
                         "identical arch/params/steps/seed for every arm)",
       "probe": "azimuth-INDEXED: the BEV cell's own geometry addresses the "
                "feature map (DiffusionDrive coupling (1) in miniature)",
       "pool_then_unpool": "every arm is avg_pool(k) then nearest-upsample to "
                           "16x40, so the head is byte-identical and the ONLY "
                           "difference is what the pool destroyed",
       "projection_scope": "column LINEAR in azimuth holds for OUR 256x640 "
                           "cylindrical corpus at f_ref 305.577 (FOV 120 deg); "
                           "a pinhole formula gives 92.6 deg and does not travel",
       "bank": a.bank, "ckpt": idx["ckpt"], "step": idx["step"], "seed": a.seed,
       "steps": a.steps, "batch": a.batch, "lr": a.lr,
       "n_addressable_cells": int(NC), "n_grid_cells": int(GH * GW),
       "in_field_cells": int(in_fov.sum()),
       "base_rate_addressable": base,
       "split_clips": {"fit": len(clips) - n_te - n_va, "val": n_va, "test": n_te},
       "arms": res, "controls": {"allzero": allzero},
       "negatives_state": "a negative cell is NOT 'seen-and-empty'. It is "
                          "'no obstacle.offline cuboid covers this cell centre', "
                          "which merges seen-and-empty with agent-occluded. "
                          "Out-of-field cells are EXCLUDED by fov_mask (two-state), "
                          "so the third state is still missing programme-wide."}
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
print("WROTE", a.out, f"{time.time()-t_all:.0f}s")
