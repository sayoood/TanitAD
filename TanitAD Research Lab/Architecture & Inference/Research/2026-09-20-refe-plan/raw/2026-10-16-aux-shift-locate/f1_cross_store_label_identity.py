#!/usr/bin/env python3
"""H-a, the sharpest data-level test: do the two label stores AGREE where they describe the same place?  CPU, read-only.

nuPlan logs revisit the same streets, so some navtest frames sit (almost) exactly on a navtrain frame's pose (same map, same heading).  A label is a deterministic function of (map, rear-axle pose), so two frames whose
poses differ by a few cm must have (almost) identical fine masks regardless of which store holds them.  If the stores differed in CONSTRUCTION (frame, cell offset, sign, scale, origin, map version), the
CROSS-store agreement at a given pose distance would fall below the WITHIN-store agreement at the same distance.
Pairs: for navtest frame j, its nearest navtrain frame (same map) in (x, y, 20 cos h, 20 sin h).  Controls: (i) WITHIN train: train frame -> nearest train frame of ANOTHER log; (ii) WITHIN navtest: likewise;
(iii) DELIBERATE REGRESSIONS on the cross-store pairs: the navtrain label shifted by one fine cell (0.5 m) along x, and along y, and sign-flipped -- all must read clearly below the real pairs.
Agreement = share of the 160x128 fine cells with the same sign of the SDF (the mask the head is trained on is a function of exactly that)."""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C  # noqa: E402
import d1_place_novelty as D1  # noqa: E402

P_tr = D1.poses(f"{C.AUXDIR}/label_store")
P_nt = D1.poses(f"{C.AUXDIR}/label_store_navtest")
src_tr, src_nt = C.open_train_labels(), C.open_navtest_labels()
rng = np.random.default_rng(20261016)


def arr(P):
    keys = list(P.keys())
    maps = np.array([P[k][0] for k in keys])
    f = np.array([[P[k][1], P[k][2], 20 * np.cos(P[k][3]), 20 * np.sin(P[k][3])] for k in keys])
    logs = np.array([k[0] for k in keys])
    return keys, maps, f, logs


k_tr, m_tr, f_tr, l_tr = arr(P_tr)
k_nt, m_nt, f_nt, l_nt = arr(P_nt)


def pairs_cross(maxd):
    out = []
    for m in ("LV", "BOS", "PIT", "SG"):
        it = np.flatnonzero(m_tr == m)
        jn = np.flatnonzero(m_nt == m)
        tree = cKDTree(f_tr[it])
        d, ix = tree.query(f_nt[jn])
        for a, b_, dd in zip(jn, ix, d):
            if dd <= maxd:
                out.append((a, it[b_], dd))
    return out


def pairs_within(f, m, logs, keys, maxd, n_sample):
    out = []
    for mp in ("LV", "BOS", "PIT", "SG"):
        idx = np.flatnonzero(m == mp)
        tree = cKDTree(f[idx])
        pick = rng.choice(idx, min(n_sample, len(idx)), replace=False)
        d, ix = tree.query(f[pick], k=12)
        for a, dd, ii in zip(pick, d, ix):
            for dj, j in zip(dd, ii):
                if idx[j] != a and logs[idx[j]] != logs[a] and dj <= maxd:      # nearest frame of ANOTHER log
                    out.append((a, idx[j], dj))
                    break
    return out


def fine(src, key):
    return src.label(key[0], key[1]) > 0


def agree(pairs, ka, sa, kb, sb, shift=None, flip=False, cap=1500):
    pairs = [pairs[i] for i in rng.permutation(len(pairs))[:cap]]
    r, ds = [], []
    for a, b_, dd in pairs:
        A, Bm = fine(sa, ka[a]), fine(sb, kb[b_])
        if flip:
            Bm = ~Bm
        if shift is not None:
            Bm = np.roll(Bm, shift[0], axis=0) if shift[0] else Bm
            Bm = np.roll(Bm, shift[1], axis=1) if shift[1] else Bm
        r.append(float((A == Bm).mean()))
        ds.append(dd)
    return np.array(r), np.array(ds)


MAXD = 0.5
cross = pairs_cross(MAXD)
within_tr = pairs_within(f_tr, m_tr, l_tr, k_tr, MAXD, 4000)
within_nt = pairs_within(f_nt, m_nt, l_nt, k_nt, MAXD, 4000)
res = {"maxd_4d_m": MAXD, "n_pairs": {"cross_store": len(cross), "within_train": len(within_tr), "within_navtest": len(within_nt)}}
print(res, flush=True)
bins = [0, 0.1, 0.2, 0.3, 0.5]
def summarize(r, d):
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d > lo) & (d <= hi) if lo > 0 else (d <= hi)
        out[f"{lo}-{hi}"] = dict(n=int(m.sum()), mean_agreement=(float(r[m].mean()) if m.any() else None), p10=(float(np.percentile(r[m], 10)) if m.any() else None))
    out["all"] = dict(n=len(r), mean_agreement=float(r.mean()), p10=float(np.percentile(r, 10)), share_ge_0p98=float((r >= 0.98).mean()))
    return out
r_c, d_c = agree(cross, k_nt, src_nt, k_tr, src_tr)
res["cross_store"] = summarize(r_c, d_c)
r_w, d_w = agree(within_tr, k_tr, src_tr, k_tr, src_tr)
res["within_train"] = summarize(r_w, d_w)
r_n, d_n = agree(within_nt, k_nt, src_nt, k_nt, src_nt)
res["within_navtest"] = summarize(r_n, d_n)
# deliberate regressions on the cross-store pairs (same sample: same rng state is not reused, so re-draw with cap=len to compare on the same pairs)
rs = {}
for nm, kw in (("shift_x_+1_fine_cell_0.5m", dict(shift=(1, 0))), ("shift_y_+1_fine_cell_0.5m", dict(shift=(0, 1))), ("shift_x_+4_fine_cells_2m", dict(shift=(4, 0))), ("sign_flipped", dict(flip=True))):
    r_, d_ = agree(cross, k_nt, src_nt, k_tr, src_tr, **kw)
    rs[nm] = dict(mean_agreement=float(r_.mean()), p10=float(np.percentile(r_, 10)))
res["deliberate_regressions_on_cross_pairs"] = rs
res["verdict_rule"] = "cross-store mean agreement within 0.005 of the within-store means at the same distance bins, and every deliberate regression at least 0.02 lower"
for k in ("cross_store", "within_train", "within_navtest"):
    print(k, json.dumps(res[k]), flush=True)
print(json.dumps(rs, indent=1), flush=True)
C.jdump(res, os.path.join(C.HERE, "f1_cross_store_label_identity.json"))
print("ZZF1_DONE")
