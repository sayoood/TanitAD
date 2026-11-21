#!/usr/bin/env python3
"""S3b -- REPLICATION of the requirement and the head arm on all three REF-C fans.

The parent stream replicated the ORACLE effect x3 (xl +88.0 %, base +87.8 %,
small +91.3 %). This replicates the two things that decide the verdict:

  * the REQUIREMENT  -- sigma_0 / sigma_50 of the ISO goal-error curve
  * the ACHIEVEMENT  -- a FAN-INDEPENDENT latent-only goal head
                        (`H_ridge_lat_ego_raw`: F_lat + F_ego only, no selector
                        answer, no fan confidences) fed through the same rule

A fan-independent head is the right object here: a strategic brain must emit a
goal WITHOUT consulting the tactical selector's answer.

Usage:  python gi_replicate.py
"""
from __future__ import annotations

import json

import numpy as np

from gi_common import (OUT, ci_paired, ci_single, eid_str, goal_reference,
                       load_refc_fan, pick_nearest_to, r4)
from gi_sweep import SIGMAS, interp_x_at, sweep_cell

HEAD_ARM = "H_ridge_lat_ego_raw"


def main() -> None:
    taut = json.loads((OUT / "gi_tautology.json").read_text())
    ghat = np.asarray(taut["preds"][HEAD_ARM], float)

    res = {"_stream": "2026-07-27-goal-input",
           "_stage": "S3b replication on 3 REF-C fans",
           "_head_arm": HEAD_ARM,
           "_head_note": ("fan-INDEPENDENT: F_lat(23 latent projections) + "
                          "F_ego(9 kinematics) only. Out-of-fold, 5 "
                          "episode-disjoint folds."),
           "fans": {}}

    print(f"{'fan':6s} {'N':>4s} {'A0':>7s} {'oracle':>7s} {'goal2s':>7s} "
          f"{'sig50':>6s} {'sig0':>6s} {'headRMS':>8s} {'realised':>8s} "
          f"{'recov':>7s} {'vs A0':>8s} sep")
    for arm in ("xl", "base", "small"):
        d = load_refc_fan(arm)
        fan = d["fan"].numpy().astype(np.float64)
        gt = d["gt"].numpy().astype(np.float64)
        sel = d["sel"].numpy()
        eid = eid_str(d)
        W = len(gt)
        g = gt[:, -1]
        err4 = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)
        a0 = err4[np.arange(W), sel]
        A0 = float(a0.mean())
        v_or = err4[np.arange(W), pick_nearest_to(goal_reference(g), fan)]
        R0 = float(v_or.mean())
        HR = A0 - R0

        xs, ys = [], []
        for s in SIGMAS:
            v, _rm, rrms = sweep_cell(g, fan, err4, "ISO", s)
            xs.append(float(rrms.mean()))
            ys.append((A0 - v.mean()) / HR)
        s50 = interp_x_at(0.5, xs, ys)
        s0 = interp_x_at(0.0, xs, ys)

        e = np.linalg.norm(ghat - g, axis=-1)
        rms = float(np.sqrt((e ** 2).mean()))
        v_h = err4[np.arange(W), pick_nearest_to(goal_reference(ghat), fan)]
        rec = (A0 - v_h.mean()) / HR
        cip = ci_paired(v_h, a0, eid)

        res["fans"][arm] = {
            "n_anchors": int(fan.shape[1]), "n_windows": W,
            "A0_as_trained": r4(A0), "oracle_in_fan": r4(err4.min(1).mean()),
            "R_goal2s": ci_single(v_or, eid), "headroom_m": r4(HR),
            "sigma_50_radial_rms_m": r4(s50), "sigma_0_radial_rms_m": r4(s0),
            "head_radial_rms_m": r4(rms),
            "head_realised": ci_single(v_h, eid),
            "head_recovery": r4(rec), "head_vs_as_trained": cip,
            "head_past_break_even": bool(rms > s0),
            "accuracy_factor_needed": r4(rms / s0),
            "iso_curve_radial_rms_m": [r4(x) for x in xs],
            "iso_curve_recovery": [r4(y) for y in ys],
        }
        print(f"{arm:6s} {fan.shape[1]:4d} {A0:7.4f} {err4.min(1).mean():7.4f} "
              f"{R0:7.4f} {s50:6.3f} {s0:6.3f} {rms:8.3f} {v_h.mean():8.4f} "
              f"{rec:+7.3f} {cip['delta']:+8.4f} "
              f"{'SEP' if cip['separated'] else '-'}")

    (OUT / "gi_replicate.json").write_text(json.dumps(res, indent=2))
    print("\nwrote", OUT / "gi_replicate.json")


if __name__ == "__main__":
    main()
