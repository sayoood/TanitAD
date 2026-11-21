#!/usr/bin/env python3
"""D4 L3 follow-up (Master Mind 2026-10-04 after D2): quantify the oracle leak of the per-window realised-max
ceiling AND of NON-ORACLE options, with the same OOF test as d4_measure.py section D (target = the window's own
max v over [NOW+2, NOW+6] s; 5-fold clip-grouped OLS; recovered = share of the future information v0 lacks).

Non-oracle candidates read ONLY the ego's PAST (rows <= NOW) or a constant:
  N0  no input (constant)                                        -> recovered 0 by construction (control)
  N1  past-20 s realised max, snapped UP to the 8-step road-law ladder
  N2  N1 with an URBAN FLOOR of 50 km/h (a stopped/slow ego never reads below the urban default -- the
      "slow ego => low limit" artefact D2 measured: 74.0 % of intersection clips <= 30 km/h)
  N3  coarse 3-level road-type proxy from N2: urban (<= 50 km/h) / rural (70-100) / motorway (>= 120)
  N2u N2 with UNKNOWN (all-zero) on 45 % of windows (the NavSim navtest no-limit rate, INHERITED)
Oracle reference: O1 per-window [NOW, NOW+6 s] max snapped up (8-step) -- the per-window realised max.
Also reports, per candidate, the share of windows where the human's own [NOW, NOW+6 s] max EXCEEDS the fed
ceiling (an obedient planner would then be barred from the expert's speed).
Writes raw/d4_vmax_nonoracle.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d4_lib as L  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.environ["D4_SCRATCH"]
MAN = os.path.join(SCR, "train_v2manifest.pt")
LAD = np.array(L.LADDER8_KMH, float) / 3.6


def r2(y, yh):
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def oof(y, X, folds):
    yh = np.zeros_like(y)
    for f in range(5):
        w, *_ = np.linalg.lstsq(X[folds != f], y[folds != f], rcond=None)
        yh[folds == f] = X[folds == f] @ w
    return r2(y, yh)


def onehot(i, n):
    o = np.zeros((len(i), n))
    o[np.arange(len(i)), i] = 1
    return o


def main():
    man = torch.load(MAN, map_location="cpu", weights_only=False)
    Y, V0, P20, F6, FO = [], [], [], [], []
    for i, cid in enumerate(man["clip_id"]):
        P = man["poses"][i].numpy().astype(np.float64)
        B = L.window_block(P)
        v = B["v"]
        r = B["r"]
        full = B["valid"][:, 60]
        Vv = np.where(B["valid"], B["V"], -np.inf)
        Y.append(np.where(full, Vv[:, 20:61].max(1), np.nan))
        F6.append(Vv.max(1))
        V0.append(B["v0"])
        P20.append(np.array([v[max(0, rr - 200):rr + 1].max() for rr in r]))
        FO.append(np.full(B["n"], int(L.sha12(cid), 16) % 5))
    Y, V0, P20, F6, FO = map(np.concatenate, (Y, V0, P20, F6, FO))
    m = np.isfinite(Y)
    y, v0, p20, f6, folds = Y[m], V0[m], P20[m], F6[m], FO[m]
    base = np.hstack([np.ones((len(y), 1)), v0[:, None]])
    r0 = oof(y, base, folds)
    rng = np.random.default_rng(2)
    b_n1 = L.snap_up(p20, L.LADDER8_KMH)
    b_n2 = L.snap_up(np.maximum(p20, 50 / 3.6), L.LADDER8_KMH)
    n3 = np.where(LAD[b_n2] <= 50 / 3.6 + 1e-9, 0, np.where(LAD[b_n2] <= 100 / 3.6 + 1e-9, 1, 2))
    unk = rng.random(len(y)) < 0.45
    b_o1 = L.snap_up(f6[:], L.LADDER8_KMH)
    cands = {
        "N0_no_input": (base, None),
        "N1_past20s_max_bin8": (np.hstack([base, onehot(b_n1, 8)]), LAD[b_n1]),
        "N2_past20s_max_bin8_urban_floor50": (np.hstack([base, onehot(b_n2, 8)]), LAD[b_n2]),
        "N3_coarse_urban_rural_motorway": (np.hstack([base, onehot(n3, 3)]), np.array([50, 100, 130.0])[n3] / 3.6),
        "N2u_N2_unknown_p0.45": (np.hstack([base, onehot(b_n2, 8) * (~unk)[:, None]]), np.where(unk, np.inf, LAD[b_n2])),
        "O1_ORACLE_window_0_6s_max_bin8": (np.hstack([base, onehot(b_o1, 8)]), LAD[b_o1]),
    }
    out = {"_evidence": "MEASURED (D4), TRAIN manifest md5 3c9f8bc8..., n windows with a full 6 s future",
           "n": int(len(y)), "r2_v0_only": round(r0, 4), "rows": {}}
    for k, (X, lim) in cands.items():
        rr = oof(y, X, folds)
        row = {"oof_r2": round(rr, 4), "recovered": round((rr - r0) / (1 - r0), 4)}
        if lim is not None:
            row["human_0_6s_max_exceeds_ceiling"] = round(float((f6 > lim + 1e-6).mean()), 4)
            row["human_exceeds_by_gt_2ms"] = round(float((f6 > lim + 2.0).mean()), 4)
        out["rows"][k] = row
    out["n3_class_shares"] = {nm: round(float((n3 == i).mean()), 4) for i, nm in enumerate(("urban", "rural", "motorway"))}
    json.dump(out, open(os.path.join(HERE, "raw", "d4_vmax_nonoracle.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
