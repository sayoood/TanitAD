#!/usr/bin/env python
"""qcal_fit.py -- card PREREG_V8F_PROOF_ADD31 (QCAL), section 3: per checkpoint the map from the head gap to the 20th percentile of the label gap,
from the TRAIN probe clips only.  It reads no evaluation file (control q4: every path is checked and listed with its sha256).

  python qcal_fit.py --arms LS_s0,LS_s1,LSD_s0,LSD_s1 --out-dir F:/TanitAD-scratch/v8proof/distrule/qcal
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import margin_calib as MC  # noqa: E402   (row matching and hashing as card ADD30)
import qcal_map as QM  # noqa: E402

EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
MIN_ROWS = 200
Q_LEVEL = 20.0


def pav(y, w):
    """Pool adjacent violators: the non-decreasing fit of y with weights w."""
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


def fit_arm(arm, in_dir, states_dir, pack, files):
    lead_p = os.path.join(in_dir, arm + "_train_lead.npz")
    st_p = os.path.join(states_dir, arm + "_train.npz")
    for p in (lead_p, st_p):
        if "_eval" in os.path.basename(p):
            raise SystemExit("REFUSED: an evaluation file in the fit: " + p)
        files[p.replace("\\", "/")] = MC.sha256_file(p)
    L = np.load(lead_p, allow_pickle=True)
    S = np.load(st_p, allow_pickle=True)
    s_names = np.asarray(S["clips"]).astype(str)[np.asarray(S["clip_idx"]).astype(np.int64)]
    s_t = np.asarray(S["t"]).astype(np.int64)
    rows = np.array([pack["index"].get((nm, int(t)), -1) for nm, t in zip(s_names, s_t)], np.int64)
    g_h = np.asarray(L["gap_m"], np.float64)
    ok = (rows >= 0) & np.asarray(L["has_history"]).astype(bool) & np.isfinite(g_h)
    gl = np.full(rows.size, np.nan)
    pl = np.zeros(rows.size, bool)
    gl[rows >= 0] = pack["gap"][rows[rows >= 0]]
    pl[rows >= 0] = pack["pres"][rows[rows >= 0]] == 1
    use = ok & pl & np.isfinite(gl)
    gh, gt = g_h[use], gl[use]
    b = np.digitize(gh, EDGES[1:])                      # bin 0 = below 5 m (negatives included), last bin = 50 m and above
    groups = [np.where(b == i)[0] for i in range(len(EDGES))]
    merged = []
    for gidx in groups:
        if merged and gidx.size < MIN_ROWS:
            merged[-1] = np.concatenate([merged[-1], gidx])
        else:
            merged.append(gidx)
    while len(merged) > 1 and merged[0].size < MIN_ROWS:   # a thin first bin goes into the next one
        merged[1] = np.concatenate([merged[0], merged[1]])
        del merged[0]
    xs = np.array([np.median(gh[m]) for m in merged])
    q = np.array([np.percentile(gt[m], Q_LEVEL) for m in merged])
    n = np.array([m.size for m in merged])
    if not np.all(np.diff(xs) > 0) or xs[0] <= 0:
        raise SystemExit("REFUSED: %s knot positions not positive and increasing: %s" % (arm, xs))
    ys = np.minimum(pav(q, n), xs)
    ys = np.maximum(ys, 0.0)
    QM.check_map(xs, ys)
    used = QM.apply_map(gh, xs, ys)
    return {"arm": arm, "xs": [float(v) for v in xs], "ys": [float(v) for v in ys], "raw_q20": [float(v) for v in q], "n_rows_per_knot": [int(v) for v in n],
            "n_rows": int(use.sum()), "edges_m": list(EDGES), "min_rows": MIN_ROWS, "q_level": Q_LEVEL,
            "train_share_used_above_label": float((used > gt).mean()), "train_share_head_above_label": float((gh > gt).mean()),
            "train_mean_shift_m": float((gh - used).mean()), "unit_check": QM.unit_check(xs, ys)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="LS_s0,LS_s1,LSD_s0,LSD_s1")
    ap.add_argument("--in-dir", default=MC.ROOT + "/spatial/leadsup/read/in")
    ap.add_argument("--states-dir", default="C:/Users/Admin/v8proof_c/s1states")
    ap.add_argument("--pack", default=MC.ROOT + "/spatial/leadsup/read/train_pack.npz")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    if "eval" in os.path.basename(a.pack):
        raise SystemExit("REFUSED: the pack must be the train pack")
    os.makedirs(a.out_dir, exist_ok=True)
    P = np.load(a.pack, allow_pickle=True)
    names = np.asarray(P["clips"]).astype(str)[np.asarray(P["clip_idx"]).astype(np.int64)]
    t = np.asarray(P["t"]).astype(np.int64)
    pack = {"index": {(nm, int(tt)): i for i, (nm, tt) in enumerate(zip(names, t))}, "gap": np.asarray(P["gap"], np.float64), "pres": np.asarray(P["pres"], np.float64)}
    pack_sha = MC.sha256_file(a.pack)
    for arm in a.arms.split(","):
        out_p = os.path.join(a.out_dir, arm + ".json")
        if os.path.exists(out_p):
            raise SystemExit("REFUSED: %s exists (the maps are fixed once)" % out_p)
        files = {a.pack.replace("\\", "/"): pack_sha}
        r = fit_arm(arm, a.in_dir, a.states_dir, pack, files)
        r["card"] = "PREREG_V8F_PROOF_ADD31"
        r["files_sha256"] = files
        r["q4_only_train_files"] = bool(not [p for p in files if "_eval" in os.path.basename(p)])
        with open(out_p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(r, f, indent=1, sort_keys=True)
        print("%-7s rows %d knots %d | xs %s | ys %s | train share used above label %.3f (head above label %.3f) | mean shift %.2f m | unit %s q4 %s" % (
            arm, r["n_rows"], len(r["xs"]), " ".join("%.1f" % v for v in r["xs"]), " ".join("%.1f" % v for v in r["ys"]), r["train_share_used_above_label"],
            r["train_share_head_above_label"], r["train_mean_shift_m"], r["unit_check"]["pass"], r["q4_only_train_files"]))
    ident = os.path.join(a.out_dir, "identity.json")
    if not os.path.exists(ident):
        with open(ident, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"xs": [1.0, 100.0], "ys": [1.0, 100.0], "note": "identity map, control q1"}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
