# -*- coding: utf-8 -*-
"""WP-D step 5 -- the PAIRED episode-cluster bootstrap on AP, done EXACTLY and fast.

⛔ AP is a RANKING statistic: it does not decompose per window, so it cannot be
handed to `taniteval.ci.paired_episode_cluster_bootstrap` as a per-window mean.
The RESAMPLING here is taniteval.ci's OWN -- `episode_index` + `_draws`, imported,
not reimplemented -- and only the STATISTIC recomputed inside each draw differs:
the pooled TIE-GROUP AP over the drawn episodes' cells.  PAIRED: every arm sees
the SAME draws, so shared episode difficulty cancels inside each draw.

⭐ THE SPEED-UP IS EXACT, NOT AN APPROXIMATION, AND THAT MATTERS.  A draw only
changes HOW MANY TIMES each row is counted; it never changes a score.  So the
global descending sort and the tie-group boundaries are computed ONCE per arm,
and a draw is a re-WEIGHTING of that fixed order:

    w      = count_of_row[row_of_cell]                (integer multiplicity)
    tp(g)  = cumsum(w*y) at tie-group ends
    n(g)   = cumsum(w)   at tie-group ends
    AP     = sum_g  (tp(g)/n(g)) * dtp(g)/n_pos

This is the SAME number a naive re-sort would produce -- pinned below by an
exact agreement check against a re-sorted AP on a sample of draws -- at ~1/13 the
cost, so the CI uses EVERY scored cell instead of a subsample.
"""
import argparse
import json
import os
import sys

import numpy as np

SNAP = r"C:\Users\Admin\wpd-probe\snap"
sys.path.insert(0, SNAP)
from tanitad.refs.refc_bev_aux import _average_precision as AP
from taniteval.ci import episode_index, _draws

argp = argparse.ArgumentParser()
argp.add_argument("--bank", default=r"C:\Users\Admin\wpd-probe\bank")
argp.add_argument("--geom", default="cart")
argp.add_argument("--out", required=True)
argp.add_argument("--n-boot", type=int, default=2000)
argp.add_argument("--seed", type=int, default=0)
argp.add_argument("--arms", default="tok_D0,tok_D1,tok_D2,pix,pos_only,shuf_D1")
a = argp.parse_args()

idx = json.load(open(os.path.join(a.bank, "idx.json"), encoding="utf-8"))
plan = np.asarray(idx["plan"], dtype=np.int64)
clips = idx["clips"]
rng = np.random.default_rng(a.seed)
order = rng.permutation(len(clips))
n_te, n_va = 30, 25
te_c = set(order[:n_te].tolist())
cl = plan[:, 0]
ti = np.where(np.isin(cl, list(te_c)))[0]
EID = cl[ti]
R = len(ti)

if a.geom == "cart":
    Y = np.asarray(np.load(os.path.join(a.bank, "y_cart.npy"), mmap_mode="r"))[ti]
    M = None
else:
    Y = np.asarray(np.load(os.path.join(a.bank, "y_pol.npy"), mmap_mode="r"))[ti]
    M = np.asarray(np.load(os.path.join(a.bank, "m_pol.npy"), mmap_mode="r"))[ti].astype(bool)

ARMS = a.arms.split(",")
P0 = np.load(os.path.join(a.bank, f"pt_{a.geom}_{ARMS[0]}.npy"))
NCELL = P0.shape[1]
if a.geom == "cart":                       # the probe scored a CELL SUBSET
    import math
    from tanitad.data.bev_raster import BEVGrid, cell_centers_xy, fov_mask
    grid = BEVGrid(x_fwd_m=float(idx["x_fwd_m"]), y_half_m=float(idx["y_half_m"]),
                   cell_m=float(idx["cell_m"]))
    X_m, Y_m = cell_centers_xy(grid)
    az_full = np.arctan2(Y_m, X_m)
    ok = fov_mask(grid, math.radians(60.0)) & (np.abs(az_full) <= math.radians(60.0) - 1e-9)
    for sg in (-1, 1):
        cc = (320.0 + sg * 305.5774907364391 * az_full) / 32.0 - 0.5
        ok &= (cc >= 0) & (cc <= 19)
    vidx = np.flatnonzero(ok.reshape(-1))
    assert len(vidx) == NCELL, (len(vidx), NCELL)
    Y = Y[:, vidx]
y = (Y.reshape(-1) > 0)
mask = M.reshape(-1) if M is not None else None
rowid = np.repeat(np.arange(R, dtype=np.int64), NCELL)
if mask is not None:
    y, rowid = y[mask], rowid[mask]
print(f"[boot] geom {a.geom}  rows {R}  cells/row {NCELL}  scored {y.size}  "
      f"pos {int(y.sum())}", flush=True)

uniq, idx_by_ep = episode_index(EID)
draws = [np.asarray(s) for s in _draws(uniq, idx_by_ep, a.n_boot, a.seed)]
CNT = np.zeros((len(draws), R), dtype=np.float64)
for i, s in enumerate(draws):
    CNT[i] = np.bincount(s, minlength=R)
print(f"[boot] {len(draws)} draws over {len(uniq)} episodes", flush=True)

PRE = {}
for arm in ARMS:
    P = np.load(os.path.join(a.bank, f"pt_{a.geom}_{arm}.npy")).reshape(-1).astype(np.float64)
    if mask is not None:
        P = P[mask]
    o = np.argsort(-P, kind="mergesort")
    so, yo, ro = P[o], y[o].astype(np.float32), rowid[o].astype(np.int32)
    ends = np.flatnonzero(np.r_[np.diff(so) != 0.0, True])
    PRE[arm] = (yo, ro, ends)
    print(f"[pre] {arm}: {len(ends)} tie groups over {so.size} cells", flush=True)
    del P, o, so

def ap_draw(arm, cnt):
    """Pooled tie-group AP for one bootstrap draw, by RE-WEIGHTING the fixed
    global order. ⛔ A leading tie group can contain ONLY cells from rows this
    draw did not select, giving n == 0. That group's tp is 0 and its recall
    increment is 0, so its contribution is 0 -- but `tp/n` evaluates 0/0 = NaN
    and `.sum()` then returns NaN for the WHOLE draw, silently poisoning the
    percentiles. Guarded explicitly rather than left to a warning."""
    yo, ro, ends = PRE[arm]
    w = cnt[ro]
    tp = np.cumsum(w * yo)[ends]
    n = np.cumsum(w)[ends]
    npos = tp[-1]
    if npos <= 0:
        return float("nan")
    d = np.diff(np.r_[0.0, tp]) / npos
    prec = np.zeros_like(tp)
    np.divide(tp, n, out=prec, where=n > 0)
    v = float((prec * d).sum())
    if not np.isfinite(v):
        raise SystemExit(f"REFUSING: non-finite AP for {arm} on a draw")
    return v

# ⛔ EXACTNESS CHECK: the re-weighting must reproduce a genuine re-sorted AP.
# Verified on real draws, not asserted. Cells are contiguous per row BEFORE the
# supervised mask, so a row's slice is arithmetic; under a mask we keep the
# per-row index lists once and index into them.
_arm0 = ARMS[0]
_P0 = np.load(os.path.join(a.bank, f"pt_{a.geom}_{_arm0}.npy")).reshape(-1).astype(np.float64)
_y0 = (Y.reshape(-1) > 0)
if mask is not None:
    _P0, _y0 = _P0[mask], _y0[mask]
    _off = np.r_[0, np.cumsum(np.bincount(rowid, minlength=R))]
    _rowslice = [np.arange(_off[r], _off[r + 1]) for r in range(R)]
else:
    _rowslice = None
chk = []
for i in (0, 1, 2):
    cnt = CNT[i]
    rows = np.repeat(np.arange(R), cnt.astype(np.int64))
    if _rowslice is None:
        take = (rows[:, None] * NCELL + np.arange(NCELL)).ravel()
    else:
        take = np.concatenate([_rowslice[r] for r in rows])
    chk.append((ap_draw(_arm0, cnt), AP(_P0[take], _y0[take])))
    del take
del _P0, _y0
print("[exactness] fast vs re-sorted AP on 3 real draws:", flush=True)
for f, s in chk:
    print(f"   fast {f:.12f}   resorted {s:.12f}   equal={f == s}", flush=True)
if not all(abs(f - s) < 1e-12 for f, s in chk):
    raise SystemExit("REFUSING: the fast bootstrap does not reproduce a re-sorted AP")

DA = {}
for arm in ARMS:
    DA[arm] = np.array([ap_draw(arm, CNT[i]) for i in range(len(draws))])
    print(f"[draws] {arm}: point {ap_draw(arm, np.ones(R)):.6f} "
          f"boot mean {DA[arm].mean():.6f}", flush=True)

POINT = {arm: ap_draw(arm, np.ones(R)) for arm in ARMS}

def paired(x, yname):
    d = DA[x] - DA[yname]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": round(float(POINT[x] - POINT[yname]), 6),
            "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "separated": bool(lo > 0 or hi < 0),
            "p_delta_gt0": round(float((d > 0).mean()), 4),
            "n_boot": len(draws), "n_episodes": int(len(uniq)),
            "estimator": ("paired episode-cluster bootstrap -- taniteval.ci."
                          "_draws resampling, statistic = pooled tie-group AP"),
            "answers": ("would another DRAW OF EPISODES say this? It does NOT "
                        "answer whether another TRAINING RUN would -- that is "
                        "what the D0b replicate arm measures (H-ESTIM-SEED-1).")}

PAIRS = [("tok_D1", "pos_only", "A1 marginal (bar: >= +0.010, separated)"),
         ("tok_D1", "pix", "A2 raw-pixel floor (bar: > 0, separated)"),
         ("tok_D1", "tok_D0", "A3 the lever vs aux-off (bar: >= 3x replicate floor)"),
         ("tok_D1", "tok_D2", "A4 information vs shuffled target (bar: >= 3x floor)"),
         ("tok_D0", "pos_only", "CONTEXT: does the AUX-OFF trunk already carry agents?"),
         ("tok_D0", "pix", "CONTEXT: aux-off trunk vs the raw-pixel floor"),
         ("tok_D2", "tok_D0", "CONTEXT: the ZERO-INFORMATION target vs aux-off"),
         ("tok_D2", "pos_only", "CONTEXT: shuffled-target trunk vs marginal"),
         ("pix", "pos_only", "CONTEXT: do raw pixels beat the marginal?"),
         ("shuf_D1", "pos_only", "CONTROL: random-frame features must NOT gain")]
pairs = {}
for x, yn, why in PAIRS:
    if x in DA and yn in DA:
        pairs[f"{x}_minus_{yn}"] = dict(paired(x, yn), why=why)
        p = pairs[f"{x}_minus_{yn}"]
        print(f"[pair] {why:52s} {p['delta']:+.5f} "
              f"[{p['lo']:+.5f}, {p['hi']:+.5f}] sep={p['separated']}", flush=True)

json.dump({"_evidence_class": "MEASURED (ours; artifact = this file + the bank's "
                              "pt_*.npy score matrices)",
           "tier": "NOT APPLICABLE - frozen-feature representation probe",
           "geom": a.geom, "n_boot": len(draws), "n_episodes": int(len(uniq)),
           "n_test_rows": R, "n_cells_per_row": NCELL, "n_scored_cells": int(y.size),
           "n_pos": int(y.sum()), "base_rate": float(y.sum() / y.size),
           "exactness_check": [{"fast": f, "resorted": s, "equal": f == s}
                               for f, s in chk],
           "point_ap": POINT, "pairs": pairs},
          open(a.out, "w", encoding="utf-8"), indent=1)
print("WROTE", a.out)
