#!/usr/bin/env python3
"""S2 -- THE GOAL-ERROR CURVE (priority 2): how accurate must a goal be?

Injects a controlled error into the TRUE 2 s goal, rebuilds the reference exactly
as `fanc_goal.py` does, re-picks from the SAME real fan, and measures how much of
the oracle's -0.2705 m survives.

    ghat = g + delta ;  R = ghat[:,None,:] * [0.25,0.5,0.75,1.0]
    pick = argmin_c  mean_i || fan[w,c,i] - R[w,i] ||        (mean-over-waypoint L2)
    recovery = (A0 - realised) / (A0 - realised(delta=0))     A0 = 0.4714

Families:  ISO(sigma)  isotropic 2-D Gaussian, sigma = PER-AXIS SD in metres
           LONG(sigma) along-track (component 0) only
           LAT(sigma)  cross-track (component 1) only
           SHRINK(a)   ghat = gbar + a*(g - gbar)  -- the REALISTIC regression-head
                       error model (biased toward the mean, not isotropic)

Every point is quoted in METRES of goal error -> METRES of ade_0_2s. No
correlations, no R^2 (retraction class CORRELATION-WITHOUT-SLOPE).

The per-window value that enters the bootstrap is the SEED-MEAN realised ADE, so
the interval is on the expected realised error under goal noise.

Usage:  python gi_sweep.py
"""
from __future__ import annotations

import json

import numpy as np

from gi_common import (OUT, ci_paired, ci_single, eid_str, goal_reference,
                       load_refc_fan, pick_nearest_to, r4, SEED)

SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
ALPHAS = [1.0, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
N_SEEDS = 16
#: FIXED per-family seed offsets. `hash(str)` is SALTED per process
#: (PYTHONHASHSEED), so using it here would make the curve irreproducible
#: run-to-run while looking perfectly stable within one run.
KIND_OFFSET = {"ISO": 11, "LONG": 23, "LAT": 37, "SHRINK": 53,
               "NC1_pure_noise": 71, "NC2_shuffled_goal": 89}


def realised_for(goal, fan, err4):
    idx = pick_nearest_to(goal_reference(goal), fan)
    return err4[np.arange(len(goal)), idx]


def sweep_cell(g, fan, err4, kind, param, n_seeds=N_SEEDS):
    """-> (per-window seed-mean realised ADE, per-window seed-mean radial goal
    error, per-window seed-RMS radial goal error)."""
    W = len(g)
    acc = np.zeros(W)
    rad = np.zeros(W)
    rad2 = np.zeros(W)
    for s in range(n_seeds):
        rng = np.random.default_rng(SEED + 1000 * s + KIND_OFFSET[kind])
        if kind == "ISO":
            d = rng.normal(0.0, param, size=(W, 2))
        elif kind == "LONG":
            d = np.stack([rng.normal(0.0, param, W), np.zeros(W)], 1)
        elif kind == "LAT":
            d = np.stack([np.zeros(W), rng.normal(0.0, param, W)], 1)
        elif kind == "SHRINK":
            d = (param - 1.0) * (g - g.mean(0)[None])          # deterministic
        elif kind == "NC1_pure_noise":
            d = rng.multivariate_normal(np.zeros(2), np.cov(g.T), W) \
                + (g.mean(0)[None] - g)
        elif kind == "NC2_shuffled_goal":
            d = g[rng.permutation(W)] - g
        else:
            raise ValueError(kind)
        acc += realised_for(g + d, fan, err4)
        r = np.linalg.norm(d, axis=-1)
        rad += r
        rad2 += r ** 2
        if kind == "SHRINK":                # deterministic -> one seed suffices
            return acc, rad, np.sqrt(rad2)
    return acc / n_seeds, rad / n_seeds, np.sqrt(rad2 / n_seeds)


def interp_x_at(y_target, xs, ys):
    """First crossing of `y_target` on a (monotone-ish decreasing) curve."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 - y_target) * (y1 - y_target) <= 0 and y0 != y1:
            f = (y0 - y_target) / (y0 - y1)
            return float(xs[i] + f * (xs[i + 1] - xs[i]))
    return None


def main() -> None:
    d = load_refc_fan("xl")
    fan = d["fan"].numpy().astype(np.float64)
    gt = d["gt"].numpy().astype(np.float64)
    cv = d["cv"].numpy().astype(np.float64)
    sel = d["sel"].numpy()
    eid = eid_str(d)
    W = len(gt)
    g = gt[:, -1]
    err4 = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)
    a0 = err4[np.arange(W), sel]
    A0 = float(a0.mean())
    v0_goal = realised_for(g, fan, err4)
    R0 = float(v0_goal.mean())
    HEADROOM = A0 - R0
    v_cv = err4[np.arange(W), pick_nearest_to(cv, fan)]

    res = {"_stream": "2026-07-27-goal-input", "_stage": "S2 goal-error curve",
           "_estimator": "paired_episode_cluster_bootstrap B=2000, unit=episode",
           "_units": ("sigma = PER-AXIS SD in metres; radial_rms = RMS of "
                      "||delta||; every recovery is metres of ade_0_2s"),
           "_n_seeds": N_SEEDS,
           "A0_as_trained": r4(A0), "R_goal2s_oracle": r4(R0),
           "headroom_m": r4(HEADROOM), "R_cv": r4(v_cv.mean()),
           "families": {}, "controls": {}}

    # --- FIDELITY A: sigma = 0 must reproduce the committed 0.2009 exactly ---
    res["controls"]["fidelity_A_sigma0_realised"] = r4(R0)
    res["controls"]["fidelity_A_ok"] = bool(abs(R0 - 0.2009) < 5e-5)

    for kind, grid in (("ISO", SIGMAS), ("LONG", SIGMAS), ("LAT", SIGMAS),
                       ("SHRINK", ALPHAS)):
        rows = []
        print(f"\n=== {kind} ===")
        print(f"{'param':>7s} {'rad_rms_m':>9s} {'rad_mean_m':>10s} "
              f"{'realised':>9s} {'recovery':>9s} {'vs A0':>9s} sep")
        for p in grid:
            v, rmean, rrms = sweep_cell(g, fan, err4, kind, p)
            rec = (A0 - v.mean()) / HEADROOM
            cip = ci_paired(v, a0, eid)
            rows.append({"param": p,
                         "radial_rms_m": r4(rrms.mean()),
                         "radial_mean_m": r4(rmean.mean()),
                         "realised": ci_single(v, eid),
                         "recovery": r4(rec),
                         "vs_as_trained": cip,
                         "vs_R_cv": ci_paired(v, v_cv, eid)})
            print(f"{p:7.3f} {rrms.mean():9.4f} {rmean.mean():10.4f} "
                  f"{v.mean():9.4f} {rec:+9.4f} {cip['delta']:+9.4f} "
                  f"{'SEP' if cip['separated'] else '-'}")
        x_rms = [r["radial_rms_m"] for r in rows]
        y = [r["recovery"] for r in rows]
        mono = all(y[i] >= y[i + 1] - 1e-9 for i in range(len(y) - 1))
        res["families"][kind] = {
            "rows": rows,
            "sigma_50_radial_rms_m": interp_x_at(0.5, x_rms, y),
            "sigma_0_radial_rms_m": interp_x_at(0.0, x_rms, y),
            "monotone_decreasing": bool(mono),
            "min_recovery": r4(min(y)),
        }
        print(f"  sigma_50 (radial RMS m) = "
              f"{res['families'][kind]['sigma_50_radial_rms_m']}")
        print(f"  sigma_0  (radial RMS m) = "
              f"{res['families'][kind]['sigma_0_radial_rms_m']}   "
              f"monotone={mono}  min_recovery={min(y):+.4f}")

    # --- negative controls --------------------------------------------------
    for kind in ("NC1_pure_noise", "NC2_shuffled_goal"):
        v, rmean, rrms = sweep_cell(g, fan, err4, kind, 0.0)
        rec = (A0 - v.mean()) / HEADROOM
        res["controls"][kind] = {"realised": ci_single(v, eid),
                                 "radial_rms_m": r4(rrms.mean()),
                                 "recovery": r4(rec),
                                 "vs_as_trained": ci_paired(v, a0, eid),
                                 "recovery_le_zero": bool(rec <= 0)}
        print(f"\n{kind}: realised {v.mean():.4f}  radial_rms {rrms.mean():.2f} m  "
              f"recovery {rec:+.4f}  (must be <= 0)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gi_sweep.json").write_text(json.dumps(res, indent=2))
    print("\nwrote", OUT / "gi_sweep.json")


if __name__ == "__main__":
    main()
