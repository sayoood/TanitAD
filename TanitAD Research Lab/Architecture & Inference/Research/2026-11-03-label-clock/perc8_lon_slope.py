#!/usr/bin/env python
"""r9clk probe 3 -- model-side corroboration on the banked PERC8 packs (dev box, CPU, read-only).

For every held-out window (sha12, ws) of `perc_det.pkl`, the Hungarian-matched TRUE POSITIVE pairs (pred slot <-> GT row)
are rebuilt from the pack (the pack stores `matched` per slot and the GT rows, not the pairing; the pairing is the
minimum-total-distance assignment among the matched slots and the `pos` GT rows, and is CHECKED against the pack's own
`pair_err` multiset). The signed longitudinal error  dx = pred_x - gt_x  is regressed on the ego speed v at the window's
NOW row (poses[ws + w - 1, 3] of the eval manifest).

  LAG hypothesis + a model that predicts where the object IS in the image:  dx = -0.2 s * v   (slope -0.2)
  slope ~ 0: either no lag, or the model learned the lag -- this probe alone cannot decide (brief).

The probe's sensitivity is shown by a KNOWN-ANSWER control: the GT x is shifted by a known +0.2 s * v in software; the
fitted slope must move by exactly -0.2.
"""
import argparse
import hashlib
import json
import pickle

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def fit(v, dx):
    A = np.stack([np.ones_like(v), v], 1)
    (a, b), *_ = np.linalg.lstsq(A, dx, rcond=None)
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=8)
    ap.add_argument("--gate-m", type=float, default=2.0)
    a = ap.parse_args()
    man = torch.load(a.manifest, map_location="cpu", weights_only=False)
    poses = {sha12(c): man["poses"][i].double().numpy() for i, c in enumerate(man["clip_id"])}
    recs = pickle.load(open(a.packs, "rb"))
    res = {"packs": a.packs, "n_windows": len(recs), "gate_m": a.gate_m}
    for w in sorted({a.w, 6, 8, 10}):
        rows = []   # (episode idx, v, pred_x, gt_x, pred_y, gt_y, window, dist)
        n_pairs_ok = n_pairs_bad = 0
        for rec in recs:
            if not rec.get("box3d"):
                continue          # 222 of 4,754 windows carry no box3d pack (4,532 do)
            pk = rec["box3d"][0]
            m = np.flatnonzero(pk["matched"])
            pos = np.flatnonzero(np.asarray(pk["pos"]))
            if len(m) == 0 or len(pos) == 0:
                continue
            P, G = pk["xy"][m].astype(np.float64), pk["gt_xy"][pos].astype(np.float64)
            dist = np.hypot(P[:, None, 0] - G[None, :, 0], P[:, None, 1] - G[None, :, 1])
            r, c = linear_sum_assignment(dist)
            d = dist[r, c]
            if w == a.w:
                ref = np.sort(np.asarray(pk["pair_err"], np.float64))
                if len(ref) == len(d) and np.allclose(np.sort(d), ref, atol=2e-3):
                    n_pairs_ok += 1
                else:
                    n_pairs_bad += 1
            keep = d <= a.gate_m
            now = rec["ws"] + w - 1
            v = float(poses[rec["sha12"]][now, 3])
            for rr, cc, kk in zip(r, c, keep):
                if kk:
                    rows.append((rec["sha12"], v, P[rr, 0], G[cc, 0], P[rr, 1], G[cc, 1]))
        eps = np.array([hash(x[0]) for x in rows])
        v = np.array([x[1] for x in rows])
        dx = np.array([x[2] - x[3] for x in rows])
        gx = np.array([x[3] for x in rows])
        dy = np.array([x[4] - x[5] for x in rows])
        out = {"n_tp_pairs": len(rows), "n_clips": int(len(set(eps))), "v_mean": float(v.mean()), "v_std": float(v.std())}
        # all TP pairs, then bands by GT range (the model has most skill 0-40 m)
        for name, sel in (("all", np.ones(len(v), bool)), ("gt_x_0_20", (gx >= 0) & (gx < 20)),
                          ("gt_x_20_40", (gx >= 20) & (gx < 40)), ("gt_x_0_40_moving_ego_v>3", (gx >= 0) & (gx < 40) & (v > 3))):
            if sel.sum() < 50:
                continue
            a0, b0 = fit(v[sel], dx[sel])
            u = np.unique(eps[sel])
            rng = np.random.default_rng(0)
            bs = []
            idx_by = {e: np.flatnonzero(sel & (eps == e)) for e in u}
            for _ in range(500):
                pick = rng.choice(u, len(u))
                ii = np.concatenate([idx_by[e] for e in pick])
                bs.append(fit(v[ii], dx[ii])[1])
            # KNOWN-ANSWER control: GT x made 0.2 s old (gt' = gt + 0.2 s * v for a static object seen from the ego
            # frame of 0.2 s earlier)  =>  dx' = pred - gt' = dx - 0.2 v  =>  the fitted slope must move by exactly -0.2
            _, bk = fit(v[sel], (dx - 0.2 * v)[sel])
            out[name] = {"n": int(sel.sum()), "intercept_m": a0, "slope_s": b0, "slope_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                         "mean_dx_m": float(dx[sel].mean()), "mean_abs_dy_m": float(np.abs(dy[sel]).mean()),
                         "CONTROL_slope_if_GT_were_0.2s_old(expect=slope-0.2)": bk,
                         "CONTROL_shift_error_vs_expected": float(bk - (b0 - 0.2))}
        if w == a.w:
            out["pairing_reconstruction"] = {"windows_pair_err_multiset_reproduced": n_pairs_ok, "windows_not": n_pairs_bad}
        res[f"w{w}"] = out
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
