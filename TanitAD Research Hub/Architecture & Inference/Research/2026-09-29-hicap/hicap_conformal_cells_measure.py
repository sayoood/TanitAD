#!/usr/bin/env python3
"""HiCAP conformal keep-set on real windows (numpy only, torch-free).

Question: does the split-conformal keep-set of Proposition (conformal cascade) reach its nominal coverage on
EPISODE-DISJOINT data, and how many tactical cells does it remove?  Only inference-time quantities enter the
score: the ego speed v0 (a binned state) and a train-fitted cell frequency table.

Data: the committed val-40 window dump taniteval/results/windows_refc-xl-30k.pt (881 windows / 40 episodes;
gt [W,4,2] at 0.5/1/1.5/2 s in the ego frame; speed = v0 in m/s).  Because only the val windows are committed,
the 40 episodes are split at random into FIT (frequency table), CALIBRATION (conformal threshold) and TEST,
R = 500 times; no window of a test episode is ever seen by the fit or the calibration.  This is a mechanism
check on a small set, NOT a ship-ready guarantee (that needs the 4,572-clip train split).

Cells are a KINEMATIC PROXY for the v7 tactical cell (label side may use the future path): longitudinal in
{HOLD, BRAKE, CRUISE, ACCEL} from the speed change over the last 0.5 s segment versus v0; lateral in
{KEEP, NUDGE_L, NUDGE_R, TURN_L, TURN_R} from the heading change and the lateral offset at 2 s.  The thresholds
(dv -1.0 / +1.0 m/s, turn 0.30 rad, nudge 0.6 m, hold 0.5 m/s) are chosen here, are NOT the label emitter's,
and only the coverage/pruning mechanism is being measured, not the labels.

Scores compared: (a) prior-only  s = log q(cell);  (b) speed-conditional  s = log q(cell | v0 bin).
Keep-set (split conformal, marginal coverage): calibration conformity = -log q at the true cell; the threshold is
the ceil((1-alpha)(n+1))-th smallest of the nonconformity -s(true); a cell is kept iff -s(cell) <= threshold.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "TanitAD Research Hub/Architecture & Inference/Research")
from reff_v0_coverage import WINDOWS, load_pt  # noqa: E402

DT = 0.5
LON = ["HOLD", "BRAKE", "CRUISE", "ACCEL"]
LAT = ["KEEP", "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R"]
NC = len(LON) * len(LAT)
V_BINS = np.array([0.5, 3.0, 8.0, 14.0])          # 5 speed bins: <0.5, .5-3, 3-8, 8-14, >=14 m/s
R, SEED = 500, 0
ALPHAS = (0.05, 0.10, 0.20)


def cells_from_gt(gt, v0):
    pts = np.concatenate([np.zeros_like(gt[:, :1, :]), gt], axis=1)                 # [W,5,2]
    d = np.diff(pts, axis=1)
    seg = np.linalg.norm(d, axis=-1) / DT                                             # [W,4]
    hd = np.arctan2(d[..., 1], d[..., 0])
    dpsi = (hd[:, -1] - hd[:, 0] + np.pi) % (2 * np.pi) - np.pi
    y2 = gt[:, -1, 1]
    dv = seg[:, -1] - v0
    lon = np.where((v0 < 0.5) & (seg[:, -1] < 0.5), 0, np.where(dv < -1.0, 1, np.where(dv > 1.0, 3, 2)))
    lat = np.where(np.abs(dpsi) > 0.30, np.where(dpsi > 0, 3, 4),
                   np.where(np.abs(y2) > 0.6, np.where(y2 > 0, 1, 2), 0))
    return lat * len(LON) + lon


def fit_tables(cell, vbin, idx, alpha_smooth=0.5):
    n_v = len(V_BINS) + 1
    cnt = np.full((n_v, NC), alpha_smooth)
    np.add.at(cnt, (vbin[idx], cell[idx]), 1.0)
    cond = cnt / cnt.sum(1, keepdims=True)
    uncond = cnt.sum(0) / cnt.sum()
    return np.log(uncond), np.log(cond)


def keep_sets(logq_rows, thr):
    return (-logq_rows) <= thr[:, None] if np.ndim(thr) else (-logq_rows) <= thr


def main():
    win = load_pt(WINDOWS)
    gt = np.asarray(win["gt"], dtype=np.float64)
    v0 = np.asarray(win["speed"], dtype=np.float64).reshape(-1)
    eid = np.asarray(win["eid"]).reshape(-1)
    cell = cells_from_gt(gt, v0)
    vbin = np.digitize(v0, V_BINS)
    eps = np.unique(eid)
    ep_idx = {e: np.flatnonzero(eid == e) for e in eps}
    counts = np.bincount(cell, minlength=NC)
    out = {"n_windows": int(len(cell)), "n_episodes": int(len(eps)), "n_cells_proxy": NC,
           "populated_cells": int((counts > 0).sum()),
           "cell_share": {f"{LAT[c // len(LON)]}|{LON[c % len(LON)]}": round(float(counts[c] / counts.sum()), 4)
                          for c in np.argsort(-counts)[:10]},
           "splits": R, "split_sizes_episodes": "fit 14 / calibration 12 / test 14",
           "note": "kinematic-proxy cells; val-only mechanism check; not a ship-ready guarantee",
           "results": {}}
    rng = np.random.default_rng(SEED)
    acc = {(m, a): {"cov": [], "size": [], "prune": []} for m in ("prior_only", "speed_conditional") for a in ALPHAS}
    for _ in range(R):
        perm = rng.permutation(eps)
        fit_e, cal_e, test_e = perm[:14], perm[14:26], perm[26:]
        fit_i = np.concatenate([ep_idx[e] for e in fit_e])
        cal_i = np.concatenate([ep_idx[e] for e in cal_e])
        test_i = np.concatenate([ep_idx[e] for e in test_e])
        lq_u, lq_c = fit_tables(cell, vbin, fit_i)
        for name in ("prior_only", "speed_conditional"):
            if name == "prior_only":
                score_cal = np.broadcast_to(lq_u, (len(cal_i), NC))
                score_te = np.broadcast_to(lq_u, (len(test_i), NC))
            else:
                score_cal = lq_c[vbin[cal_i]]
                score_te = lq_c[vbin[test_i]]
            nonconf = -score_cal[np.arange(len(cal_i)), cell[cal_i]]
            n = len(nonconf)
            for a in ALPHAS:
                k = int(np.ceil((1 - a) * (n + 1)))
                thr = np.sort(nonconf)[min(k, n) - 1]
                keep = (-score_te) <= thr
                acc[(name, a)]["cov"].append(float(keep[np.arange(len(test_i)), cell[test_i]].mean()))
                acc[(name, a)]["size"].append(float(keep.sum(1).mean()))
                acc[(name, a)]["prune"].append(float(1.0 - keep.mean()))
    for (name, a), v in acc.items():
        cov = np.array(v["cov"])
        out["results"][f"{name}|alpha={a}"] = {
            "nominal_coverage": round(1 - a, 3),
            "coverage_mean": round(float(cov.mean()), 4),
            "coverage_sd_over_splits": round(float(cov.std(ddof=1)), 4),
            "coverage_p05_split": round(float(np.percentile(cov, 5)), 4),
            "share_of_splits_below_nominal": round(float((cov < 1 - a).mean()), 3),
            "mean_cells_kept_of_20": round(float(np.mean(v["size"])), 2),
            "mean_prune_fraction": round(float(np.mean(v["prune"])), 4)}
    json.dump(out, sys.stdout, indent=1)


ALPHAS_SET = ALPHAS
if __name__ == "__main__":
    main()
