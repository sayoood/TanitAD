#!/usr/bin/env python3
"""Fit the P10 prior that v8F-mini's operative residual is taken against -- in the CORRECT kd_x unit.

ORCHESTRATOR CORRECTION 2026-10-17: ``actions[:, 0]`` is a road-wheel STEER ANGLE (arctan(2.9 kappa)); the banked
refav1 kd_x / P10 integrated it as a curvature (ha0_ext 5.356 m vs 2.838 m). So the prior is refit here with
``action_units="steer"`` through ``taniteval.floors_kin`` (the shared, tested implementation -- nothing is
re-implemented): the SAME recipe as the banked P10 (``probe_te.py::fit_mlp_kin``: MLP 8 -> 256 -> 60, wd in {1e-4, 1e-2}
and epoch count by 4-fold grouped inner CV, 600 train clips at stride 8, the first 600 by sha12 -- a nested prefix of
the S4 train order), on the v2 MANIFEST poses/actions (frame i + 2). Also refits the LEGACY unit as a PIPELINE
CONTROL: it must land near the banked 2.9257 m, else this script is not the banked recipe.

Outputs (--out-dir): p10_prior.pt (the frozen prior), p10_fit.json (selection + eval numbers), p10_eval_pred6.npy
[2399, 30, 2] (the refit-P10 baseline predictions on the trunk-probe grid) and p10_legacy_eval_pred6.npy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from taniteval import floors_kin as fk  # noqa: E402

PKG = os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini")


def _base(kin, units):
    ha0 = fk.ha0_paths(kin[:, 0], D.DT, D.HORIZON)
    hx = fk.ha0_ext_paths(kin, D.DT, D.HORIZON, action_units=units)
    return fk.kd_x(ha0, hx).astype(np.float64)


def build_rows(poses_by_clip, clips, stride, units, t_first=3):
    kf, y, base, g, grp = [], [], [], [], []
    for gi, c in enumerate(clips):
        P, A = poses_by_clip[c]
        Tg = (len(P) + 1) // 2
        ts = np.arange(t_first, Tg - D.HORIZON, stride)
        ts = ts[2 * ts < len(P)]
        if len(ts) == 0:
            continue
        kin, gt, _ = D.window_kin(P, A, ts)
        ok = np.isfinite(kin).all(1) & np.isfinite(gt).all((1, 2))
        kin, gt = kin[ok], gt[ok]
        b = _base(kin, units)
        kf.append(fk.kin_features(kin))
        base.append(b)
        g.append(gt)
        grp.append(np.full(len(kin), gi))
        y.append((gt - b).reshape(len(gt), -1))
    cat = np.concatenate
    return cat(kf), cat(y), cat(base), cat(g), cat(grp)


def eval_rows(eval_kin, grid, units):
    z = np.load(eval_kin)
    off = np.concatenate([[0], np.cumsum(z["n"])])
    kf, base, g = [], [], []
    for i, c in enumerate(z["clip"].tolist()):
        P = z["poses"][off[i]:off[i + 1]].astype(np.float64)
        A = z["actions"][off[i]:off[i + 1]].astype(np.float64)
        kin, gt, _ = D.window_kin(P, A, grid["windows"][c])
        kf.append(fk.kin_features(kin))
        base.append(_base(kin, units))
        g.append(gt)
    return np.concatenate(kf), np.concatenate(base), np.concatenate(g)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt")
    ap.add_argument("--eval-kin", default=os.path.join(PKG, "raw", "eval141_kin.npz"))
    ap.add_argument("--eval-grid", default=os.path.join(PKG, "raw", "eval_grid_windows.json"))
    ap.add_argument("--clips", type=int, default=600)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(PKG, "raw", "p10"))
    ap.add_argument("--skip-legacy", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    m = torch.load(a.manifest, weights_only=False)
    byc = {c: D.full_from_manifest(p.numpy(), ac.numpy()) for c, p, ac in zip(m["clip_id"], m["poses"], m["actions"])}
    clips = sorted(byc, key=D.sha12)[:a.clips]
    grid = json.load(open(a.eval_grid))
    res = {"manifest": a.manifest, "n_manifest_clips": len(byc), "fit_clips": len(clips), "stride": a.stride,
           "seed": a.seed, "banked_legacy_p10": {"ade2": 0.2229, "ade6": 2.9257, "note": "kd_x integrated in the WRONG unit"}}
    for units in (("steer",) if a.skip_legacy else ("steer", "kappa")):
        kf, y, base, g, grp = build_rows(byc, clips, a.stride, units)
        net, st, best = fk.fit_p10(kf, y, base, g, grp, seed=a.seed)
        kfe, be, ge = eval_rows(a.eval_kin, grid, units)
        pred = fk.predict_p10((net, st, best), kfe, be)
        ade6, fde6 = fk.ade(pred, ge), fk.fde(pred, ge)
        ade2 = fk.ade(pred[:, :10], ge[:, :10])
        res[units] = {"n_fit_rows": int(len(kf)), "best": [int(best[0]), float(best[1]), int(best[2])],
                      "eval_n": int(len(ge)), "ade6_m": float(ade6.mean()), "fde6_m": float(fde6.mean()),
                      "ade2_first10knots_m": float(ade2.mean()),
                      "kd_x_ade6_m": float(fk.ade(be, ge).mean()),
                      "acts": fk.acts_check(pred, D.DT)}
        np.save(os.path.join(a.out_dir, "p10_eval_pred6.npy" if units == "steer" else "p10_legacy_eval_pred6.npy"),
                pred.astype(np.float32))
        if units == "steer":
            prior = M.P10Prior.from_fit((net, st, best))
            M.save_p10(prior, os.path.join(a.out_dir, "p10_prior.pt"),
                       {"units": "steer", "fit_clips": len(clips), "stride": a.stride, "best": res[units]["best"],
                        "eval_ade6_m": res[units]["ade6_m"]})
            np.save(os.path.join(a.out_dir, "eval_base6_steer.npy"), be.astype(np.float32))
    res["seconds"] = time.time() - t0
    json.dump(res, open(os.path.join(a.out_dir, "p10_fit.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("steer", "kappa", "banked_legacy_p10")}, indent=1, default=str)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
