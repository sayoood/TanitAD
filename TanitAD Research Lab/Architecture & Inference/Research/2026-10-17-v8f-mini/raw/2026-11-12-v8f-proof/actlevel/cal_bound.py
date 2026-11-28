#!/usr/bin/env python
"""cal_bound.py -- T0, descriptive: the first read of lever (a) of note PREREG_V8F_PROOF_ADD37_N7 section 5 ("how much of the level error is a constant offset").
Reader dumps of card ADD36 (imaghead/dump/DUMP_<arm>_s<k>.npz), windows with a lead at t and at t + 2.0 s and a finite rollout read. Per arm and base:
  raw            mean absolute error of the rollout read (logged actions) against the label gap at t + 2.0 s
  best shift     the same after subtracting the MEDIAN signed error of these same windows (the L1-optimal constant: an upper bound on any shift fitted elsewhere)
  best affine    the same after a least-absolute-deviations fit a + b * read on these same windows (iteratively reweighted least squares, 60 rounds; an upper bound on any
                 affine map fitted elsewhere); its scale b multiplies the response of the read
Ratios are against the raw error of RLFT at the same base on the same windows; bar of note N4: 1.15. Names nothing. Written by the lead, 2026-11-19.
Usage: cal_bound.py <dump dir> <out json>
"""
import io
import json
import sys

import numpy as np

dd, out_path = sys.argv[1], sys.argv[2]
res, ref = {}, {}
for arm in ("RLFT", "RLAm", "RLVa", "RLVc", "RLVd", "RLVb", "RLAh", "RLA3"):
    res[arm] = []
    for s in (0, 1):
        d = np.load("%s/DUMP_%s_s%d.npz" % (dd, arm, s), allow_pickle=True)
        y, g, r = d["gap_t2"].astype(float), d["gap_t"].astype(float), d["roll_logged"][:, 1, 1].astype(float)
        m = np.isfinite(y) & np.isfinite(g) & np.isfinite(r)
        y, r = y[m], r[m]
        raw = float(np.abs(r - y).mean())
        sh = float(np.abs(r - y - np.median(r - y)).mean())
        X = np.column_stack([np.ones(len(r)), r])
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        for _ in range(60):
            e = np.abs(y - X @ w)
            ww = 1.0 / np.maximum(e, 1e-3)
            w = np.linalg.lstsq(X * ww[:, None] ** 0.5, y * ww ** 0.5, rcond=None)[0]
        lad = float(np.abs(y - X @ w).mean())
        if arm == "RLFT":
            ref[s] = raw
        rec = {"seed": s, "n": int(m.sum()), "mae_raw": round(raw, 4), "signed_bias_mean": round(float((r - y).mean()), 4), "mae_best_shift_in_sample": round(sh, 4),
               "mae_best_affine_in_sample": round(lad, 4), "affine_scale": round(float(w[1]), 3), "bar_1.15x_RLFT_raw": round(1.15 * ref[s], 4),
               "ratio_raw": round(raw / ref[s], 4), "ratio_best_shift": round(sh / ref[s], 4), "ratio_best_affine": round(lad / ref[s], 4)}
        res[arm].append(rec)
        print(arm, s, json.dumps(rec))
with io.open(out_path, "w", encoding="utf-8", newline="\n") as fh:
    json.dump({"tier": "T0 (in-sample bounds on the eval dumps of the reader; names nothing; never driving performance)",
               "strategic": "STRATEGIC: NOT COMPUTED (n = 0; level of the imagined lead gap, no strategic instrument)", "arms": res}, fh, indent=1)
print("ZZ CAL-BOUND DONE")
