# -*- coding: utf-8 -*-
"""WP-D step 4 -- RE-READ WP-A's OWN BANKED TRUNK THROUGH THE CORRECTED ADDRESS.

⛔ WHY THIS EXISTS. WP-D's entire premise is WP-A's dissociation: *"on the real
planning-trained trunk the fit ladder does not transfer -- test AP 0.027-0.034
against a 0.0325 marginal"* ⇒ *"the map does not yet contain agents, so WP-B
built today would index into an empty room."*

But WP-A's address is

    u = W/2 + f_ref * az                      (s5_indexed.py)

while `bev_aux.azimuth_column` AND `psg_targets.azimuth_column` -- two
independent implementations in this repo that AGREE with each other -- both use

    col = (hfov/2 - az) / daz                 ⇒  u = W/2 - f_ref * az

These are MIRRORS. On WP-D's bank the raw-pixel arm reads VAL AP 0.048265 under
the second and 0.031789 under the first: a 1.52x gap on an UNLEARNED
representation, identical head, identical seed, one sign flipped.

⇒ this script runs WP-A's BANKED FEATURES, unchanged, under BOTH signs, so the
question *"was WP-A's dissociation an artifact of a mirrored address?"* is
answered on WP-A's own bytes rather than by analogy from a different trunk.

⭐ The tell that motivated it, visible in WP-A's own banked panel: its BEST token
arm is `tok_1x1` (AP 0.0335) -- the arm that average-pools the whole 16x40 map to
a SINGLE value and therefore carries NO azimuth information at all -- while
`tok_16x40`, which carries the most, is the WORST (0.0295). Under a correct
address more azimuth resolution cannot hurt; under a mirrored one, destroying the
address is an improvement, because a mirrored feature actively misleads.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn

SNAP = r"C:\Users\Admin\wpd-probe\snap"
sys.path.insert(0, SNAP)
from tanitad.data.bev_raster import BEVGrid, cell_centers_xy, fov_mask
from tanitad.refs.refc_bev_aux import _average_precision as AP

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default=r"C:\Users\Admin\wpa-readout\bank\k8r1")
ap.add_argument("--out", default=r"C:\Users\Admin\wpd-probe\raw\wpa_recheck.json")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--cells", type=int, default=1536)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

idx = json.load(open(os.path.join(a.bank, "idx.json"), encoding="utf-8"))
clips = idx["clips"]; plan = np.asarray(idx["plan"], dtype=np.int64)
TH, TW = idx["token_grid"]; DM = int(idx["d_model"]); CIN = int(idx["in_channels"])
PATCH = 640 // TW                      # 16 at the 16x40 grid
W_PX, F_REF, HFOV = 640, 305.5774907364391, 120.0
print(f"[wpa-bank] token_grid {TH}x{TW} d_model {DM} patch {PATCH} "
      f"ckpt {idx['ckpt']} step {idx['step']}", flush=True)

grid = BEVGrid(x_fwd_m=float(idx["x_fwd_m"]), y_half_m=float(idx["y_half_m"]),
               cell_m=float(idx["cell_m"]))
X_m, Y_m = cell_centers_xy(grid)
az_full = np.arctan2(Y_m, X_m)

def address(sign):
    return lambda az: (W_PX / 2.0 + sign * F_REF * az) / PATCH - 0.5

ok = fov_mask(grid, math.radians(HFOV / 2.0))
for sg in (-1, 1):
    cc = address(sg)(az_full)
    ok &= (cc >= 0) & (cc <= TW - 1)
vidx = np.flatnonzero(ok.reshape(-1))
NC = len(vidx)
xs, ys = X_m.reshape(-1)[vidx], Y_m.reshape(-1)[vidx]
azv = az_full.reshape(-1)[vidx]

Yall = np.asarray(np.load(os.path.join(a.bank, "y.npy"), mmap_mode="r"))
Yv = torch.from_numpy(Yall[:, vidx].copy())

def build_S(sign):
    c = np.clip(address(sign)(azv), 0, TW - 1 - 1e-6)
    c0 = np.floor(c).astype(np.int64); w1 = c - c0
    c1 = np.minimum(c0 + 1, TW - 1)
    S = np.zeros((NC, TW), dtype=np.float32)
    S[np.arange(NC), c0] += 1.0 - w1
    S[np.arange(NC), c1] += w1
    return torch.from_numpy(S).to(dev)

S_BY_SIGN = {"prog": build_S(-1), "wpa": build_S(1)}
S_t = S_BY_SIGN["prog"]

pf = [xs / 60.0, ys / 16.0, np.log1p(np.clip(xs, 0, None)) / 4.2, azv / 1.05,
      np.sqrt(xs ** 2 + ys ** 2) / 60.0]
for k in (1, 2, 4, 8):
    pf += [np.sin(k * azv), np.cos(k * azv), np.sin(k * xs / 20.0), np.cos(k * xs / 20.0)]
POS = torch.from_numpy(np.stack(pf, 1).astype(np.float32)).to(dev)
PD = POS.shape[1]

rng = np.random.default_rng(a.seed)
order = rng.permutation(len(clips))
n_te, n_va = 30, 25
te_c, va_c = set(order[:n_te].tolist()), set(order[n_te:n_te + n_va].tolist())
cl = plan[:, 0]
fi = np.where(~np.isin(cl, list(te_c | va_c)))[0]
vi = np.where(np.isin(cl, list(va_c)))[0]
ti = np.where(np.isin(cl, list(te_c)))[0]
Ttest = Yv[ti].float().to(dev)
_yt = (Yv[ti].numpy() > 0)
n_pos_test = int(np.count_nonzero(_yt)); n_scored = int(_yt.size)
base_test = n_pos_test / n_scored
del _yt
print(f"[split] rows {len(fi)}/{len(vi)}/{len(ti)}  cells {NC}", flush=True)
print(f"[base] test base rate {base_test:.9f} = {n_pos_test}/{n_scored}", flush=True)
ap_const = AP(np.full(n_scored, 0.37), Ttest.reshape(-1).cpu().numpy())
print(f"[CONTROL] constant AP {ap_const:.12f}  base {base_test:.12f}  "
      f"exact={ap_const == base_test}", flush=True)
if ap_const != base_test:
    raise SystemExit("REFUSING: the constant control does not read the base rate")

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

SRC = {"tok": np.load(os.path.join(a.bank, "tok.npy"), mmap_mode="r"),
       "pix": np.load(os.path.join(a.bank, "pix.npy"), mmap_mode="r")}

POOL = {}
def load_feat(kind, rows):
    C = CIN if kind == "pix" else DM
    x = torch.from_numpy(np.asarray(SRC[kind][rows], dtype=np.float32))
    x = x.transpose(1, 2).reshape(-1, C, TH, TW)
    ker = POOL.get("ker", (1, 1))
    if ker != (1, 1):
        # ⭐ pool THEN nearest-upsample back to 16x40, exactly as WP-A's ladder does,
        # so the head is byte-identical across rungs and the ONLY difference is what
        # the pool destroyed.
        x = torch.nn.functional.avg_pool2d(x, ker)
        x = torch.nn.functional.interpolate(x, size=(TH, TW), mode="nearest")
    return x

@torch.no_grad()
def evaluate(head, rows, kind, zero_feat):
    head.eval(); P = []
    allc = torch.arange(NC, device=dev)
    for s in range(0, len(rows), 32):
        P.append(torch.sigmoid(head(load_feat(kind, rows[s:s + 32]).to(dev),
                                    allc, zero_feat)).cpu())
    head.train()
    return torch.cat(P).to(dev)

def run(name, kind, sign, zero_feat=False, ker=(1, 1)):
    global S_t
    S_t = S_BY_SIGN[sign]
    POOL["ker"] = ker
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    C = CIN if kind == "pix" else DM
    head = Head(C).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    base_fit = float(Yv[fi].float().mean())
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(1 - base_fit) / base_fit], device=dev))
    g = np.random.default_rng(a.seed)
    t0 = time.time()
    for step in range(a.steps):
        rows = np.sort(g.choice(fi, a.batch, replace=False))
        cs = torch.from_numpy(g.choice(NC, a.cells, replace=False)).to(dev)
        y = Yv[rows][:, cs.cpu()].float().to(dev)
        loss = lossf(head(load_feat(kind, rows).to(dev), cs, zero_feat), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    apv = AP(evaluate(head, vi, kind, zero_feat).reshape(-1).cpu().numpy(),
             Yv[vi].float().reshape(-1).numpy())
    apt = AP(evaluate(head, ti, kind, zero_feat).reshape(-1).cpu().numpy(),
             Ttest.reshape(-1).cpu().numpy())
    del head, opt
    torch.cuda.empty_cache()
    print(f"[{name:16s}] AP_val {apv:.4f}  AP_test {apt:.4f}  ({time.time()-t0:.0f}s)",
          flush=True)
    return {"ap_val": apv, "ap_test": apt}

res = {}
for sg in ("prog", "wpa"):
    res[f"pix_{sg}"] = run(f"pix/{sg}", "pix", sg)
    res[f"tok_{sg}"] = run(f"tok/{sg}", "tok", sg)
res["pos_only"] = run("pos_only", "tok", "prog", zero_feat=True)

# ⭐⭐ THE FALSIFIABLE PREDICTION THE MIRROR DIAGNOSIS MAKES.
# WP-A's banked ladder is MONOTONE-WRONG: its BEST token arm is `tok_1x1` (0.0335),
# which pools ALL azimuth away, and its WORST is `tok_16x40` (0.0295), which keeps
# the most. Under a CORRECT address more azimuth resolution cannot hurt, so the
# ladder must INVERT. Under the mirrored one it should stay wrong. Both are run.
LADDER = [("16x40", (1, 1)), ("8x20", (2, 2)), ("4x4", (4, 10)), ("1x1", (16, 40))]
for sg in ("prog", "wpa"):
    for nm, ker in LADDER:
        res[f"ladder_{sg}_{nm}"] = run(f"ladder/{sg}/{nm}", "tok", sg, ker=ker)
POOL["ker"] = (1, 1)

out = {"_evidence_class": "MEASURED (ours; artifact = this file + WP-A's own bank "
                          "C:/Users/Admin/wpa-readout/bank/k8r1)",
       "tier": "NOT APPLICABLE - frozen-feature representation probe",
       "question": ("was E-READOUT-CEILING-1's dissociation an artifact of a "
                    "MIRRORED azimuth address? Run on WP-A's OWN banked v6 "
                    "features, both signs, everything else identical."),
       "bank": a.bank, "ckpt": idx["ckpt"], "step": idx["step"],
       "token_grid": [TH, TW], "d_model": DM, "patch_px": PATCH,
       "n_cells": NC, "n_test_rows": int(len(ti)), "n_scored_cells": n_scored,
       "base_rate_test": base_test, "n_pos_test": n_pos_test,
       "constant_control_ap": ap_const,
       "constant_equals_base_rate_exactly": ap_const == base_test,
       "wpa_banked_reference": {"tok_16x40": 0.0295, "tok_8x20": 0.0330,
                                "tok_1x1": 0.0335, "pix_16x40": 0.0270,
                                "pos_only": 0.0325,
                                "note": "read from wpa-readout/raw/indexed_k8.json"},
       "arms": res, "seed": a.seed, "steps": a.steps,
       "ladder_prediction": ("under a CORRECT address more azimuth resolution cannot "
                             "hurt, so AP must FALL as the pool coarsens; WP-A's banked "
                             "ladder RISES (16x40 0.0295 -> 1x1 0.0335), which is what a "
                             "MIRRORED address produces because a mirrored feature "
                             "actively misleads and destroying it helps."),
       "wpa_banked_ladder": {"tok_16x40": 0.0295, "tok_8x20": 0.0330,
                             "tok_4x8": 0.0318, "tok_4x4": 0.0327,
                             "tok_4x2": 0.0332, "tok_1x1": 0.0335}}
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
print("WROTE", a.out)
