#!/usr/bin/env python3
"""PAIRED episode-cluster CI for CURVATURE MAE (cl - ha0) -- the interval the
refav1 records do NOT carry.

⛔ THE CONTROL THAT MUST READ A KNOWN VALUE: the per-window mean recomputed here
must reproduce the record's own `four_families.lateral.curvature_mae_1pm` for the
SAME arm. If it does not, this estimator is measuring something else and NOTHING
below is quotable. Both numbers are printed side by side, always.

Estimator: `taniteval.ci.paired_episode_cluster_bootstrap` -- imported, never
edited (it is a sibling's file). It answers "would another draw of EPISODES say
this?" and NOT "would another INFERENCE RUN say this?"; the seed spread is the
instrument for the second question and is reported separately.
ASCII only.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\tanitad-ctg\taniteval")
from taniteval import ci as CI                                    # noqa: E402

P4 = r"C:\Users\Admin\refav1_margin\p4out"
MIN_DS = 0.05          # refav1's own min step length gate (lateral.min_ds_m)


def curvature_per_window(path):
    """|kappa| per step from a [n, K, 2] path, then the per-window mean of the
    ABS DIFFERENCE is taken by the caller. Curvature from the discrete turn
    angle over arc length: kappa_k = dtheta_k / ds_k."""
    p = np.asarray(path, dtype=np.float64)
    d = np.diff(p, axis=1)                       # [n, K-1, 2]
    ds = np.linalg.norm(d, axis=-1)              # [n, K-1]
    th = np.arctan2(d[..., 1], d[..., 0])        # [n, K-1]
    dth = np.diff(th, axis=1)                     # [n, K-2]
    dth = (dth + np.pi) % (2 * np.pi) - np.pi
    seg = 0.5 * (ds[:, :-1] + ds[:, 1:])          # [n, K-2]
    with np.errstate(divide="ignore", invalid="ignore"):
        k = dth / seg
    valid = seg >= MIN_DS
    return k, valid


def load(dumpdir, key):
    out, eps = [], []
    for f in sorted(glob.glob(dumpdir + "/ep*.npz")):
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            continue
        a = z[key]
        out.append(a)
        eps += [os.path.basename(f)] * len(a)
    return (np.concatenate(out, 0) if out else None), eps


def per_window_curv_mae(dumpdir, arm, ref="g"):
    pa, eps = load(dumpdir, arm)
    pg, _ = load(dumpdir, ref)
    if pa is None or pg is None:
        return None, None
    ka, va = curvature_per_window(pa)
    kg, vg = curvature_per_window(pg)
    m = va & vg
    n = m.sum(1)
    with np.errstate(invalid="ignore"):
        w = np.where(n > 0, (np.abs(ka - kg) * m).sum(1) / np.maximum(n, 1), np.nan)
    return w, eps


def main():
    rows = []
    for tag, dump in [(t, os.path.join(P4, "dump_%s" % t)) for t in sys.argv[1:]]:
        if not os.path.isdir(dump):
            dump = os.path.join(r"C:\Users\Admin\wkfront\out", "dump_%s" % tag)
        rec = os.path.join(P4, "rec_%s.json" % tag)
        if not os.path.exists(rec):
            rec = os.path.join(r"C:\Users\Admin\wkfront\out", "rec_%s.json" % tag)
        banked = None
        if os.path.exists(rec):
            d = json.load(open(rec, encoding="utf-8"))
            banked = d["arms"]["cl"]["four_families"]["lateral"]["curvature_mae_1pm"]
        cl, eps = per_window_curv_mae(dump, "cl")
        h0, _ = per_window_curv_mae(dump, "ha0")
        if cl is None:
            print("%-12s DUMP ABSENT %s" % (tag, dump))
            continue
        keep = ~(np.isnan(cl) | np.isnan(h0))
        e = [x for x, k in zip(eps, keep) if k]
        r = CI.paired_episode_cluster_bootstrap(cl[keep], h0[keep], e,
                                                n_boot=2000, seed=0)
        mine = float(np.nanmean(cl))
        ok = "MATCH" if (banked is not None and abs(mine - banked) < 5e-4) else "MISMATCH"
        print("%-12s recomputed cl curv MAE %.6f  vs BANKED %.6f  -> %s   "
              "| ha0 %.6f | paired (cl-ha0) delta %+.6f [%+.6f, %+.6f] separated=%s  n=%d/%d"
              % (tag, mine, banked if banked is not None else float("nan"), ok,
                 float(np.nanmean(h0)), r["delta"], r["lo"], r["hi"],
                 r["separated"], int(keep.sum()), len(cl)))
        rows.append((tag, mine, r))
    return rows


if __name__ == "__main__":
    main()
