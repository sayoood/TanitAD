#!/usr/bin/env python
"""qmap_fit.py -- card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE), part 3: the monotone 0.20-quantile map of the label gap on the reader's gap read.

Input: the rows file of `tok_reader.py extract` (held-out clips of the corpus training cache only). Fit rows: held-out clips whose hash value is divisible by 20 (column `fit`);
check rows: the other held-out clips. The form is the map of card ADD31 (distrule/qcal_fit.py: bins, min rows per knot, pool adjacent violators, ys <= xs; qcal_map.py is the same file).
Coverage (control k9): the share of check rows (label present, reader history complete) with a label gap below the mapped value Q(g_reader) must lie in [0.15, 0.25].

  python qmap_fit.py --rows ROWS.npz --arm LR16_s0 --out-dir DIR            -> DIR/QMAP_<arm>.json (+ .sha256 line inside DIR/QMAP_<arm>.json.sha256)
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, os.path.join(HERE, "..", "distrule")):
    if os.path.isfile(os.path.join(cand, "qcal_map.py")) and cand not in sys.path:
        sys.path.insert(0, cand)
import qcal_map as QM  # noqa: E402

EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
MIN_ROWS = 200
Q_LEVEL = 20.0
BAND = (0.15, 0.25)


def pav(y, w):
    y = [float(v) for v in y]
    w = [float(v) for v in w]
    blocks = [[y[i], w[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-15:
            a, b = blocks[i], blocks[i + 1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1], a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, n in blocks:
        out.extend([v] * n)
    return np.asarray(out, np.float64)


def raw_map(g, xs, ys):
    """Q(g) without the min with g (the map of QM.apply_map before the min)."""
    g = np.asarray(g, np.float64)
    q = np.interp(g, xs, ys)
    lo = g < xs[0]
    q[lo] = ys[0] * g[lo] / xs[0]
    hi = g > xs[-1]
    q[hi] = g[hi] - (xs[-1] - ys[-1])
    return q


def usable(R, which):
    """boolean mask of rows of the fit (which='fit') or the check (which='check') clips with a present label, a finite label gap and a finite reader gap."""
    base = np.asarray(R["lab_pres"]).astype(bool) & np.isfinite(np.asarray(R["lab_gap"], np.float64)) & np.isfinite(np.asarray(R["g"], np.float64))
    fit = np.asarray(R["fit"]).astype(bool)
    return base & (fit if which == "fit" else ~fit)


def fit_map(gh, gt):
    b = np.digitize(gh, EDGES[1:])
    groups = [np.where(b == i)[0] for i in range(len(EDGES))]
    merged = []
    for gidx in groups:
        if merged and gidx.size < MIN_ROWS:
            merged[-1] = np.concatenate([merged[-1], gidx])
        else:
            merged.append(gidx)
    while len(merged) > 1 and merged[0].size < MIN_ROWS:
        merged[1] = np.concatenate([merged[0], merged[1]])
        del merged[0]
    xs = np.array([np.median(gh[m]) for m in merged])
    q = np.array([np.percentile(gt[m], Q_LEVEL) for m in merged])
    n = np.array([m.size for m in merged])
    if len(xs) < 2 or not np.all(np.diff(xs) > 0) or xs[0] <= 0:
        raise SystemExit("REFUSED: knot positions not positive and increasing: %s" % xs)
    ys = np.maximum(np.minimum(pav(q, n), xs), 0.0)
    QM.check_map(xs, ys)
    return xs, ys, q, n


def coverage(gh, gt, xs, ys):
    """share of rows whose label gap is below the mapped value Q(g_reader) (raw) and below g_used = min(g, Q(g))."""
    if gh.size == 0:
        return {"n": 0, "share_below_Q": float("nan"), "share_below_used": float("nan")}
    return {"n": int(gh.size), "share_below_Q": float((gt < raw_map(gh, xs, ys)).mean()), "share_below_used": float((gt < QM.apply_map(gh, xs, ys)).mean())}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fit_from_rows(R, arm="X"):
    mf, mc = usable(R, "fit"), usable(R, "check")
    g, lg = np.asarray(R["g"], np.float64), np.asarray(R["lab_gap"], np.float64)
    if mf.sum() < 2 * MIN_ROWS:
        raise SystemExit("REFUSED: %d fit rows (need at least %d)" % (mf.sum(), 2 * MIN_ROWS))
    xs, ys, q, n = fit_map(g[mf], lg[mf])
    cov = coverage(g[mc], lg[mc], xs, ys)
    cov["pass"] = bool(cov["n"] > 0 and BAND[0] <= cov["share_below_Q"] <= BAND[1])
    ci = np.asarray(R["ci"]).astype(np.int64)
    fit_clips = set(ci[mf].tolist())
    check_clips = set(ci[mc].tolist())
    return {"arm": arm, "card": "PREREG_V8F_PROOF_ADD46", "xs": [float(v) for v in xs], "ys": [float(v) for v in ys], "raw_q20": [float(v) for v in q], "n_rows_per_knot": [int(v) for v in n],
            "n_fit_rows": int(mf.sum()), "n_check_rows": int(mc.sum()), "n_fit_clips": len(fit_clips), "n_check_clips": len(check_clips), "fit_check_clips_disjoint": bool(not (fit_clips & check_clips)),
            "edges_m": list(EDGES), "min_rows": MIN_ROWS, "q_level": Q_LEVEL, "band": list(BAND), "coverage": cov, "unit_check": QM.unit_check(xs, ys), "tier": "T0"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    R = np.load(a.rows, allow_pickle=True)
    rec = fit_from_rows(R, a.arm)
    rec["rows_file"] = os.path.basename(a.rows)
    rec["rows_sha256"] = sha256_file(a.rows)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "QMAP_%s.json" % a.arm)
    if os.path.exists(out):
        raise SystemExit("REFUSED: %s exists (a map is fitted once)" % out)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    with open(out + ".sha256", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(sha256_file(out) + "  " + os.path.basename(out) + "\n")
    print("QMAP %s fit rows %d check rows %d coverage %.3f (band %s pass %s) unit %s" % (a.arm, rec["n_fit_rows"], rec["n_check_rows"], rec["coverage"]["share_below_Q"], BAND, rec["coverage"]["pass"], rec["unit_check"]["pass"]))
    return 0 if rec["coverage"]["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
