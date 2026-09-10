# -*- coding: utf-8 -*-
"""WP-D step 2 -- the PRE-REGISTERED frozen-feature probe on D0 / D1 / D2.

WP-A section 4.2's azimuth-INDEXED occupancy decoder, unchanged in shape, re-run on
REF-C's REAL 8x20 stride-32 map.  The BEV cell's own geometry is the ADDRESS
into the feature map (DiffusionDrive coupling (1) in miniature):

    az = atan2(y, x)                  ego frame, +x fwd, +y LEFT
    u  = f_ref * az + W/2             CYLINDRICAL -> column LINEAR in azimuth
    c  = u / 32 - 0.5                 -> fractional token column in [0, 20)

Every arm shares one head architecture, parameter count, optimiser, step count
and seed, so the ONLY difference between arms is the information in the trunk.

TWO TARGET GEOMETRIES, both scored, because they answer different questions:
  cart   WP-A's Cartesian GRID_DEFAULT raster -- the PRIMARY read, because it is
         the instrument section 5A's reference values were measured on, and because
         it is NOT the parameterisation D1's aux head was trained against.  A
         win here is transfer; a win only on `pol` would be teaching to the test.
  pol    the polar 24x20 target the aux head actually optimised, with its
         occlusion IGNORE mask.  Secondary, and labelled as such.

CONTROLS, each of which MUST read a KNOWN value:
  constant  AP EXACTLY the base rate over scored cells (tie-group AP).  If this
            does not hold the METRIC is wrong and no arm number is admissible.
  allzero   IoU 0.000, F1 0.000, AP = base rate
  perfect   AP 1.0
  pos_only  features ZEROED -> the marginal occupancy field (bar A1's reference)
  pix       raw 9-channel cell means -> the raw-input floor (bar A2)
  shuf      D1's features from a RANDOM OTHER frame -> must fall to pos_only

No headline number is an ACCURACY: occupancy is ~2-3 % of scored cells, so an
all-zero predictor scores ~97 %.
Not an eval tier: no trajectory is produced (section 5A says so explicitly).
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
from tanitad.data import bev_aux as BA
from tanitad.refs.refc_bev_aux import _average_precision as AP  # TIE-GROUP safe
from taniteval.ci import episode_index, _draws

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default=r"C:\Users\Admin\wpd-probe\bank")
ap.add_argument("--out", default=r"C:\Users\Admin\wpd-probe\raw\panel.json")
ap.add_argument("--geom", default="cart", choices=["cart", "pol"])
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--cells", type=int, default=1536)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-boot", type=int, default=400)
ap.add_argument("--boot-cell-stride", type=int, default=2)
ap.add_argument("--arms", default="")
# ⛔⛔ THE AZIMUTH SIGN IS NOT A STYLE CHOICE -- IT IS THE M1 "MIRRORED WORLD"
# MUTANT, and the two conventions in this repo DISAGREE:
#   prog : u = W/2 - f_ref*az   -> image column 0 is azimuth +hfov/2 (LEFT).
#          This is `bev_aux.azimuth_column` AND `psg_targets.azimuth_column`,
#          two independent implementations that agree, and it is the convention
#          D1's aux head was TRAINED under (so the polar target's column j maps
#          onto feature column j -- the "zero resampling" registration).
#   wpa  : u = W/2 + f_ref*az   -> image column 0 is azimuth -hfov/2 (RIGHT).
#          This is WP-A's `s5_indexed.py` address, i.e. the MIRROR of the above.
# `auto` picks the one the RAW-PIXEL arm prefers on the VAL split -- an unlearned
# representation with a direct geometric link to the target, so this is an
# instrument check with a known expected relationship, decided on fit+val and
# NEVER on test.
ap.add_argument("--az-sign", default="auto", choices=["auto", "prog", "wpa"])
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

idx = json.load(open(os.path.join(a.bank, "idx.json"), encoding="utf-8"))
clips = idx["clips"]; plan = np.asarray(idx["plan"], dtype=np.int64)
N = int(idx["N"]); TH, TW = idx["token_grid"]; FD = int(idx["feat_dim"])
PATCH, W_PX, F_REF = 32, 640, 305.5774907364391
POL = BA.PolarBEVSpec(**{k: v for k, v in idx["polar_spec"].items()
                         if k in ("n_az", "n_rng", "r_max_m", "hfov_deg",
                                  "r_min_m", "sub_r", "sub_az")})

# ---------------------------------------------------------------- address ---
def address(sign):
    """az [rad] -> fractional feature column. sign=-1 is `prog`, +1 is `wpa`."""
    return lambda az: (W_PX / 2.0 + sign * F_REF * az) / PATCH - 0.5

if a.geom == "cart":
    grid = BEVGrid(x_fwd_m=float(idx["x_fwd_m"]), y_half_m=float(idx["y_half_m"]),
                   cell_m=float(idx["cell_m"]))
    X_m, Y_m = cell_centers_xy(grid)
    az_full = np.arctan2(Y_m, X_m)
    in_f = fov_mask(grid, math.radians(POL.hfov_deg / 2.0))
    # the addressable set is SIGN-INDEPENDENT (the two conventions are mirrors of
    # each other about the field centre), so the arms are scored on the SAME cells
    ok = in_f & (np.abs(az_full) <= math.radians(POL.hfov_deg / 2.0) - 1e-9)
    for sg in (-1, 1):
        cc = address(sg)(az_full)
        ok &= (cc >= 0) & (cc <= TW - 1)
    vidx = np.flatnonzero(ok.reshape(-1))
    xs, ys = X_m.reshape(-1)[vidx], Y_m.reshape(-1)[vidx]
    azv = az_full.reshape(-1)[vidx]
    Y_ALL = np.asarray(np.load(os.path.join(a.bank, "y_cart.npy"), mmap_mode="r"))
    Yv = torch.from_numpy(Y_ALL[:, vidx].copy())
    SUP = None                                  # every addressable cell is scored
else:
    # POLAR: `bev_aux.sample_lattice` builds cell (i, j) at azimuth
    #   az = hfov/2 - (j + 0.5) * daz            <- column 0 is +hfov/2 = LEFT
    # so this MUST be built from that formula, not from a mirrored one.
    r_c = (np.arange(POL.n_rng) + 0.5) * POL.dr_m
    az_c = np.radians(POL.hfov_deg / 2.0 - (np.arange(POL.n_az) + 0.5) * POL.daz_deg)
    R, AZ = np.meshgrid(r_c, az_c, indexing="ij")
    xs = (R * np.cos(AZ)).reshape(-1); ys = (R * np.sin(AZ)).reshape(-1)
    azv = AZ.reshape(-1)
    vidx = np.arange(POL.n_rng * POL.n_az)
    Yv = torch.from_numpy(np.asarray(np.load(os.path.join(a.bank, "y_pol.npy"),
                                             mmap_mode="r")).copy())
    SUP = torch.from_numpy(np.asarray(np.load(os.path.join(a.bank, "m_pol.npy"),
                                              mmap_mode="r")).copy().astype(bool))
NC = len(vidx)

def build_S(sign):
    c = np.clip(address(sign)(azv), 0, TW - 1 - 1e-6)
    c0 = np.floor(c).astype(np.int64); w1 = c - c0
    c1 = np.minimum(c0 + 1, TW - 1)
    S = np.zeros((NC, TW), dtype=np.float32)
    S[np.arange(NC), c0] += 1.0 - w1
    S[np.arange(NC), c1] += w1
    return torch.from_numpy(S).to(dev)

# ⭐ the POLAR identity check: under `prog` the polar target's column j must land
# on feature column j EXACTLY (that is what "zero resampling" means).
if a.geom == "pol":
    cprog = address(-1)(azv).reshape(POL.n_rng, POL.n_az)
    ident = float(np.abs(cprog - np.arange(POL.n_az)[None, :]).max())
    print(f"[registration] max |col(prog) - j| over the polar grid = {ident:.6f} "
          f"(0 means the column-to-column registration is EXACT)", flush=True)

S_BY_SIGN = {"prog": build_S(-1), "wpa": build_S(1)}
S_t = S_BY_SIGN["prog"]

pf = [xs / 60.0, ys / 16.0, np.log1p(np.clip(xs, 0, None)) / 4.2, azv / 1.05,
      np.sqrt(xs ** 2 + ys ** 2) / 60.0]
for k in (1, 2, 4, 8):
    pf += [np.sin(k * azv), np.cos(k * azv), np.sin(k * xs / 20.0), np.cos(k * xs / 20.0)]
POS = torch.from_numpy(np.stack(pf, 1).astype(np.float32)).to(dev)
PD = POS.shape[1]

# ---------------------------------------------------------------- split -----
rng = np.random.default_rng(a.seed)
order = rng.permutation(len(clips))
n_te, n_va = 30, 25
te_c, va_c = set(order[:n_te].tolist()), set(order[n_te:n_te + n_va].tolist())
cl = plan[:, 0]
fi = np.where(~np.isin(cl, list(te_c | va_c)))[0]
vi = np.where(np.isin(cl, list(va_c)))[0]
ti = np.where(np.isin(cl, list(te_c)))[0]
EID_TEST = cl[ti]                       # episode id per TEST row -> the cluster
sup_te = SUP[ti].to(dev) if SUP is not None else None
Ttest = Yv[ti].float().to(dev)
# ⛔ THE BASE RATE MUST BE EXACT, FROM INTEGER COUNTS. Accumulating it as a
# float32 `.mean()` over ~2e7 cells loses ~4e-10, and the constant control -- which
# compares for EQUALITY against a literal -- then refuses a metric that is
# perfectly correct. MEASURED here on the first run: constant AP
# 0.01610680071310928 vs a float32 base rate 0.016106801107525826. The control was
# right to fire; the REFERENCE was the wrong one, and rounding the comparison
# instead of fixing the reference is how a real tie-break defect would slip past.
_yt = (Yv[ti].numpy() > 0)
_mt = SUP[ti].numpy().astype(bool) if SUP is not None else np.ones_like(_yt, dtype=bool)
n_pos_test = int(np.count_nonzero(_yt & _mt))
n_scored = int(np.count_nonzero(_mt))
base_test = n_pos_test / n_scored
del _yt, _mt
print(f"[geom {a.geom}] cells {NC}  pos_dim {PD}  test rows {len(ti)} "
      f"episodes {len(set(EID_TEST.tolist()))}", flush=True)
print(f"[split] clips fit/val/test {len(clips)-n_te-n_va}/{n_va}/{n_te}  "
      f"rows {len(fi)}/{len(vi)}/{len(ti)}", flush=True)
print(f"[base] test base rate {base_test:.9f} = {n_pos_test}/{n_scored} scored cells "
      f"-> an ALL-ZERO predictor would score {100*(1-base_test):.4f} % ACCURACY, "
      f"which is why no headline number here is an accuracy", flush=True)

# ---- CONTROL #0: the metric itself. A constant score MUST read the base rate.
def scored(Pt, Tt):
    """-> (score_1d, target_1d) over the SUPERVISED cells only."""
    if sup_te is None:
        return Pt.reshape(-1), Tt.reshape(-1)
    return Pt[sup_te], Tt[sup_te]

_s, _t = scored(torch.full_like(Ttest, 0.37), Ttest)
_sn, _tn = _s.cpu().numpy(), _t.cpu().numpy()
ap_const = AP(_sn, _tn)
ap_zero = AP(np.zeros_like(_sn), _tn)
ap_one = AP(np.ones_like(_sn), _tn)
ap_perfect = AP(_tn.astype(np.float64), _tn)
print(f"[CONTROL] constant(0.37) AP {ap_const:.12f}   base rate {base_test:.12f}   "
      f"exact_match={ap_const == base_test}", flush=True)
print(f"[CONTROL] allzero AP {ap_zero:.12f}  allone AP {ap_one:.12f}  "
      f"perfect AP {ap_perfect:.12f}", flush=True)
if not (ap_const == base_test == ap_zero == ap_one):
    raise SystemExit(
        f"REFUSING: the constant control reads {ap_const!r} but the base rate is "
        f"{base_test!r}. The METRIC is wrong (the tie-break defect R-2026-09-07-"
        f"ap-ties), not the model. Fix the metric before trusting any arm.")
if ap_perfect != 1.0:
    raise SystemExit(f"REFUSING: perfect ranker reads {ap_perfect!r}, not 1.0")

# ---------------------------------------------------------------- head ------
D_RED = 16
class Head(nn.Module):
    """IDENTICAL for every arm -- same shapes, same params, same init, same seed."""
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

SRC = {}
def src(kind):
    if kind not in SRC:
        f = "pix.npy" if kind == "pix" else f"tok_{kind}.npy"
        SRC[kind] = np.load(os.path.join(a.bank, f), mmap_mode="r")
    return SRC[kind]

def load_feat(kind, rows):
    C = 9 if kind == "pix" else FD
    x = torch.from_numpy(np.asarray(src(kind)[rows], dtype=np.float32))
    return x.transpose(1, 2).reshape(-1, C, TH, TW)

ARMS = [("tok_D0", "D0", False, False), ("tok_D1", "D1", False, False),
        ("tok_D2", "D2", False, False), ("pix", "pix", False, False),
        ("pos_only", "D0", True, False), ("shuf_D1", "D1", False, True)]
# A3's REPLICATE arms, appended ONLY when the bank actually holds them, so a run
# over the original bank takes the identical code path it took on 2026-09-08.
# `run_arm` re-seeds torch AND numpy AND its own Generator at entry, so an arm's
# result does not depend on which arms ran before it -- appending is safe.
for _rep in ("D0b", "D0c"):
    if os.path.exists(os.path.join(a.bank, f"tok_{_rep}.npy")):
        ARMS.append((f"tok_{_rep}", _rep, False, False))
# ⛔ THE CONSTANT CONTROL FOR THE FLOOR ITSELF. `tok_D0dup` reads the IDENTICAL
# feature file as `tok_D0` -- same bytes, same head, same seed, same split, same
# invocation -- so the TRUE value of |AP(D0dup) - AP(D0)| is KNOWN TO BE ZERO.
# Whatever it reads is the INSTRUMENT floor, and no arm gap smaller than it is a
# gap. Without this, A3's floor cannot be separated from the probe's own noise.
if os.path.exists(os.path.join(a.bank, "tok_D0.npy")):
    ARMS.append(("tok_D0dup", "D0", False, False))
if a.arms:
    keep = set(a.arms.split(",")); ARMS = [x for x in ARMS if x[0] in keep]

@torch.no_grad()
def evaluate(head, rows, kind, zero_feat, shuffle_rows):
    head.eval(); P = []
    allc = torch.arange(NC, device=dev)
    for s in range(0, len(rows), 32):
        r = rows[s:s + 32]
        fr = shuffle_rows[s:s + 32] if shuffle_rows is not None else r
        P.append(torch.sigmoid(head(load_feat(kind, fr).to(dev), allc, zero_feat)).cpu())
    head.train()
    return torch.cat(P).to(dev)

res, PT = {}, {}

def run_arm(name, kind, zero_feat, shuf, sign, store=True):
    """Train ONE probe arm end-to-end and score it. `sign` selects the address
    convention; everything else is identical across arms by construction."""
    global S_t
    S_t = S_BY_SIGN[sign]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    C = 9 if kind == "pix" else FD
    head = Head(C).to(dev)
    nparam = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    base_fit = float(Yv[fi].float().mean())
    pw = torch.tensor([(1 - base_fit) / max(base_fit, 1e-9)], device=dev)
    g = np.random.default_rng(a.seed)
    t0 = time.time()
    for step in range(a.steps):
        rows = np.sort(g.choice(fi, a.batch, replace=False))
        frows = np.sort(g.choice(fi, a.batch, replace=False)) if shuf else rows
        cs = torch.from_numpy(g.choice(NC, a.cells, replace=False)).to(dev)
        y = Yv[rows][:, cs.cpu()].float().to(dev)
        logit = head(load_feat(kind, frows).to(dev), cs, zero_feat)
        ew = nn.functional.binary_cross_entropy_with_logits(
            logit, y, reduction="none", pos_weight=pw)
        if SUP is not None:
            m = SUP[rows][:, cs.cpu()].to(dev).float()
            loss = (ew * m).sum() / m.sum().clamp(min=1)
        else:
            loss = ew.mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % max(1, a.steps // 3) == 0 or step == a.steps - 1:
            print(f"    [{name}/{sign}] step {step:5d} loss {float(loss):.5f}", flush=True)
    # ---- VAL: the threshold AND the address convention are decided HERE -----
    svr = np.sort(g.choice(fi, len(vi), replace=True)) if shuf else None
    Pv = evaluate(head, vi, kind, zero_feat, svr)
    Tv = Yv[vi].float().to(dev)
    mv = SUP[vi].to(dev) if SUP is not None else None
    pv_s = Pv[mv] if mv is not None else Pv.reshape(-1)
    tv_s = Tv[mv] if mv is not None else Tv.reshape(-1)
    ap_val = AP(pv_s.cpu().numpy(), tv_s.cpu().numpy())
    qs = torch.quantile(pv_s[::7].float(),
                        torch.linspace(0.90, 0.9999, 120, device=dev))
    bf, bt = 0.0, 0.5
    for t in qs:
        pr = (pv_s >= t).float()
        f1 = float(2 * (pr * tv_s).sum() / (pr.sum() + tv_s.sum() + 1e-9))
        if f1 > bf:
            bf, bt = f1, float(t)
    if not store:
        del head, opt
        torch.cuda.empty_cache()
        return {"ap_val": ap_val, "f1_val": bf, "wall_s": round(time.time() - t0, 1)}
    # ---- TEST: scored, never tuned on ---------------------------------------
    str_ = np.sort(g.choice(fi, len(ti), replace=True)) if shuf else None
    Pt = evaluate(head, ti, kind, zero_feat, str_)
    PT[name] = Pt
    s_, t_ = scored(Pt, Ttest)
    ap_test = AP(s_.cpu().numpy(), t_.cpu().numpy())
    pr = (s_ >= bt).float()
    inter = float((pr * t_).sum())
    iou = inter / float((pr + t_ - pr * t_).sum() + 1e-9)
    f1 = float(2 * inter / (pr.sum() + t_.sum() + 1e-9))
    res[name] = {"ap_test": ap_test, "ap_val": ap_val, "f1_test": f1, "iou_test": iou,
                 "thr_from_val": bt, "f1_val": bf, "head_params": int(nparam),
                 "az_sign": sign,
                 "n_test_rows": int(len(ti)),
                 "n_test_episodes": int(len(set(EID_TEST.tolist()))),
                 "n_scored_cells": n_scored, "n_cells_per_row": int(NC),
                 "d_raw_per_row": int(TH * TW * C),
                 "d_head_input_per_cell": int(TH * D_RED + PD),
                 "steps": a.steps, "wall_s": round(time.time() - t0, 1)}
    np.save(os.path.join(a.bank, f"pt_{a.geom}_{name}.npy"),
            Pt.to(torch.float16).cpu().numpy())
    print(f"[{name:9s}] AP_test {ap_test:.4f}  AP_val {ap_val:.4f}  F1 {f1:.4f}  "
          f"IoU {iou:.4f}  par {nparam/1e3:.1f}k  n_scored {n_scored}  "
          f"d_raw {TH*TW*C}  ({time.time()-t0:.0f}s)", flush=True)
    del head, opt
    torch.cuda.empty_cache()

# ---- INSTRUMENT CHECK: which address convention actually registers? --------
# Decided on the RAW-PIXEL arm's VAL AP. Pixels are an unlearned representation
# with a direct geometric link to the target, so the correct address must beat
# its mirror. Fit + val only; the test split is untouched here.
if a.az_sign == "auto":
    sign_probe = {sg: run_arm(f"__signcheck_{sg}", "pix", False, False, sg, store=False)
                  for sg in ("prog", "wpa")}
    AZ_SIGN = max(sign_probe, key=lambda k: sign_probe[k]["ap_val"])
    print(f"[az-sign] pix val AP  prog {sign_probe['prog']['ap_val']:.6f}  "
          f"wpa {sign_probe['wpa']['ap_val']:.6f}  -> CHOSEN {AZ_SIGN}", flush=True)
else:
    AZ_SIGN, sign_probe = a.az_sign, {}

for _name, _kind, _zf, _sh in ARMS:
    run_arm(_name, _kind, _zf, _sh, AZ_SIGN)


# ------------------------------------------- PAIRED episode-cluster bootstrap
# ⛔ AP is a RANKING statistic and does not decompose per window, so it cannot be
# handed to `paired_episode_cluster_bootstrap` as a per-window mean.  The
# RESAMPLING here is taniteval.ci's own -- `episode_index` + `_draws`, imported,
# not reimplemented -- and only the STATISTIC recomputed inside each draw
# differs: the pooled tie-group AP over the drawn episodes' cells.  Paired: both
# arms see the SAME draws, so shared episode difficulty cancels inside each draw.
st = a.boot_cell_stride
sel_cells = np.arange(0, NC, st)
Tb = Ttest[:, sel_cells].cpu().numpy()
Mb = sup_te[:, sel_cells].cpu().numpy() if sup_te is not None else None
uniq, idx_by_ep = episode_index(EID_TEST)
draws = list(_draws(uniq, idx_by_ep, a.n_boot, a.seed))
print(f"[boot] {len(draws)} draws over {len(uniq)} test episodes, "
      f"cell stride {st} -> {Tb.shape[1]} cells/row", flush=True)

def draw_aps(name):
    P = PT[name][:, sel_cells].cpu().numpy()
    out = np.empty(len(draws), dtype=np.float64)
    for i, s in enumerate(draws):
        p, t = P[s].ravel(), Tb[s].ravel()
        if Mb is not None:
            m = Mb[s].ravel(); p, t = p[m], t[m]
        out[i] = AP(p, t)
    return out

t0 = time.time()
DA = {n: draw_aps(n) for n in res}
print(f"[boot] {time.time()-t0:.0f}s", flush=True)

def paired(x, y):
    d = DA[x] - DA[y]
    lo, hi = np.percentile(d, [2.5, 97.5])
    pt = res[x]["ap_test"] - res[y]["ap_test"]
    return {"delta": round(float(pt), 6), "lo": round(float(lo), 6),
            "hi": round(float(hi), 6), "separated": bool(lo > 0 or hi < 0),
            "p_delta_gt0": round(float((d > 0).mean()), 4),
            "n_boot": len(draws), "n_episodes": int(len(uniq)),
            "estimator": "paired episode-cluster bootstrap (taniteval.ci._draws "
                         "resampling; statistic = pooled tie-group AP per draw)",
            "answers": "would another draw of EPISODES say this? -- NOT whether "
                       "another TRAINING RUN would (that is D0b)"}

PAIRS = [("tok_D1", "pos_only", "A1 marginal"), ("tok_D1", "pix", "A2 raw-pixel floor"),
         ("tok_D1", "tok_D0", "A3 the lever vs aux-off"),
         ("tok_D1", "tok_D2", "A4 information vs shuffled target"),
         ("tok_D0", "pos_only", "D0 marginal (context)"),
         ("tok_D0", "pix", "D0 vs pixels (context)"),
         ("tok_D2", "pos_only", "D2 marginal (context)"),
         ("tok_D2", "tok_D0", "D2 vs D0 (context)"),
         ("shuf_D1", "pos_only", "shuffled-frame control -> must NOT separate up"),
         # ---- A3's replicate floor (H-ESTIM-SEED-1). These are the ONLY pairs
         # that answer "would another TRAINING RUN say this?"; every pair above
         # answers "would another draw of EPISODES say this?" and is blind to it.
         ("tok_D0b", "tok_D0", "A3 FLOOR f_same: D0's flags, D0's SEED, zero levers"),
         ("tok_D0c", "tok_D0", "A3 FLOOR f_seed: D0's flags, seed 1 (prereg s4)"),
         ("tok_D0c", "tok_D0b", "A3 context: replicate vs replicate"),
         ("tok_D1", "tok_D0b", "A3 context: the lever vs the same-seed replicate"),
         ("tok_D2", "tok_D0b", "A3 context: shuffled target vs the replicate"),
         ("tok_D0dup", "tok_D0", "CONTROL: IDENTICAL features -> MUST read 0.000000")]
pairs = {}
for x, y, why in PAIRS:
    if x in res and y in res:
        pairs[f"{x}_minus_{y}"] = dict(paired(x, y), why=why)
        p = pairs[f"{x}_minus_{y}"]
        print(f"[pair] {why:34s} {x} - {y}: {p['delta']:+.5f} "
              f"[{p['lo']:+.5f}, {p['hi']:+.5f}] separated={p['separated']}", flush=True)

out = {"_evidence_class": "MEASURED (ours; artifact = this file + "
                          "C:/Users/Admin/wpd-probe/bank + this json)",
       "tier": "NOT APPLICABLE - frozen-feature representation probe, no "
               "trajectory is produced (PREREG_WPD_BEV_AUX.md section 5A)",
       "geometry": a.geom,
       "geometry_note": ("cart = WP-A's Cartesian GRID_DEFAULT, the PRIMARY read "
                         "and NOT the parameterisation D1 trained on; "
                         "pol = the polar 24x20 target the aux head optimised, "
                         "with its occlusion IGNORE mask (secondary)"),
       "function_class": "NONLINEAR (1x1 conv + 3x3 residual + 3-layer MLP), "
                         "identical arch/params/steps/seed across arms",
       "projection_scope": "column LINEAR in azimuth holds for OUR 256x640 "
                           "cylindrical corpus at f_ref 305.577 (HFOV 120 deg); "
                           "a pinhole formula gives 92.6 deg and does not travel",
       "az_sign_chosen": AZ_SIGN,
       "az_sign_probe": sign_probe,
       "az_sign_note": ("`prog` = bev_aux.azimuth_column AND psg_targets."
                        "azimuth_column (two independent implementations that "
                        "AGREE): image column 0 is azimuth +hfov/2 = LEFT, and it "
                        "is the convention D1's aux head TRAINED under. `wpa` = "
                        "WP-A's s5_indexed.py address, which is its MIRROR. The "
                        "choice is settled by the RAW-PIXEL arm's VAL AP -- an "
                        "unlearned representation, fit+val only, test untouched. "
                        "This is the M1 'mirrored world' mutant as a live "
                        "question, not a style preference."),
       "bank": a.bank, "trunks": idx["trunks"], "align_with_wpa": idx["align_with_wpa"],
       "seed": a.seed, "steps": a.steps, "batch": a.batch, "cells": a.cells, "lr": a.lr,
       "n_cells_per_row": int(NC), "n_scored_cells_test": n_scored,
       "base_rate_test": base_test, "n_pos_test": n_pos_test,
       "all_zero_accuracy_pct": round(100 * (1 - base_test), 4),
       "controls": {"constant_score_ap": ap_const, "base_rate": base_test,
                    "constant_equals_base_rate_exactly": ap_const == base_test,
                    "allzero_ap": ap_zero, "allone_ap": ap_one,
                    "perfect_ranker_ap": ap_perfect,
                    "note": "the constant control is the only thing standing "
                            "between this panel and the R-2026-09-07-ap-ties "
                            "defect, in which a no-information arm scored ABOVE "
                            "its own base rate"},
       "split_clips": {"fit": len(clips) - n_te - n_va, "val": n_va, "test": n_te},
       "arms": res, "pairs": pairs,
       "boot_cell_stride": st, "n_boot": len(draws)}
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
print("WROTE", a.out)
