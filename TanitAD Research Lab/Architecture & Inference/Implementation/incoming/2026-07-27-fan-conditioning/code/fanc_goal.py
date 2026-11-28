#!/usr/bin/env python3
"""S6 -- GOAL CONDITIONING, tested as an ORACLE UPPER BOUND.

⚠️ Standing confound (V5_PLAN §4, CLAUDE.md): REF-C evaluates with
`nav_cmd=None`, so its decoder never had a working route input. A goal arm must
therefore be a real contrast, not a footnote. PhysicalAI-AV also carries NO map,
lane graph, junction annotation or route signal (settled at five probes), so a
*realisable* goal conditioner cannot be built from this corpus at all.

What CAN be built is the UPPER BOUND: condition on the TRUE future goal. If an
ORACLE goal does not move the realised pick, no realisable goal conditioner can,
and the direction is closed without a GPU-week.

Three references of increasing goal knowledge, all scored by the same
nearest-anchor rule on the SAME real fan:

  R_cv        constant velocity            (no goal)
  R_goal2s    exact 2 s GOAL POSITION, constant-velocity path to it   (oracle)
  R_goalfull  the full GT trajectory                                  (= ceiling)

`R_goal2s` is the operative arm: it knows WHERE to be in 2 s and nothing about
how to get there. That is exactly what a route/goal input supplies.

Usage:  python fanc_goal.py
"""
from __future__ import annotations

import json

import numpy as np

from fanc_common import (OUT, T_HORIZON, WP_STEPS, ade, ci_paired, ci_single,
                         eid_str, load_refc_fan, r4)


def main() -> None:
    res = {"_stream": "2026-07-27-fan-conditioning",
           "_estimator": "paired_episode_cluster_bootstrap B=2000, unit=episode",
           "_bars": {"CONFIRM": 0.4907, "STRONG": 0.4271}}
    t = np.asarray(WP_STEPS, float) * (T_HORIZON / max(WP_STEPS))
    frac = (t / T_HORIZON)[None, :, None]

    for arm in ("xl", "base", "small"):
        d = load_refc_fan(arm)
        fan = d["fan"].numpy().astype(np.float64)
        gt = d["gt"].numpy().astype(np.float64)
        cv = d["cv"].numpy().astype(np.float64)
        sel = d["sel"].numpy()
        eid = eid_str(d)
        W = len(gt)
        err = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)
        as_trained = err[np.arange(W), sel]

        refs = {
            "R_cv": cv,
            "R_goal2s": gt[:, -1][:, None, :] * frac,     # oracle goal, CV path
            "R_goalfull": gt,                             # full oracle
        }
        blk = {"n_anchors": int(fan.shape[1]),
               "as_trained": ci_single(as_trained, eid),
               "oracle_in_fan": ci_single(err.min(1), eid),
               "arms": {}}
        for k, R in refs.items():
            dd = np.linalg.norm(fan - R[:, None], axis=-1).mean(-1)
            idx = dd.argmin(1)
            v = err[np.arange(W), idx]
            blk["arms"][k] = {
                "reference_own_ade": r4(ade(R, gt).mean()),
                "realised": ci_single(v, eid),
                "vs_as_trained": ci_paired(v, as_trained, eid),
                "vs_oracle_in_fan": ci_paired(v, err.min(1), eid),
                "beats_CONFIRM_0.4907": bool(ci_single(v, eid)["mean"] < 0.4907),
                "beats_STRONG_0.4271": bool(ci_single(v, eid)["mean"] < 0.4271),
            }
        # the goal-vs-no-goal contrast, the arm the brief demands
        blk["goal_vs_nogoal"] = ci_paired(
            err[np.arange(W), np.linalg.norm(
                fan - refs["R_goal2s"][:, None], axis=-1).mean(-1).argmin(1)],
            err[np.arange(W), np.linalg.norm(
                fan - refs["R_cv"][:, None], axis=-1).mean(-1).argmin(1)], eid)
        res[f"refc_{arm}"] = blk

        print(f"\n=== REF-C-{arm.upper()} N={fan.shape[1]} "
              f"as-trained {blk['as_trained']['mean']:.4f} "
              f"oracle {blk['oracle_in_fan']['mean']:.4f}")
        print(f"{'arm':12s} {'ref own':>8s} {'realised':>9s} {'CI':>18s} "
              f"{'vs A0':>9s} sep  CONFIRM STRONG")
        for k, a in blk["arms"].items():
            r = a["realised"]
            print(f"{k:12s} {a['reference_own_ade']:8.4f} {r['mean']:9.4f} "
                  f"[{r['lo']:.4f},{r['hi']:.4f}] {a['vs_as_trained']['delta']:+9.4f} "
                  f"{'SEP' if a['vs_as_trained']['separated'] else ' - ':>3s}  "
                  f"{str(a['beats_CONFIRM_0.4907']):>7s} {str(a['beats_STRONG_0.4271']):>6s}")
        g = blk["goal_vs_nogoal"]
        print(f"goal - nogoal: {g['delta']:+.4f} [{g['lo']:+.4f},{g['hi']:+.4f}] "
              f"sep={g['separated']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fanc_goal.json").write_text(json.dumps(res, indent=2))
    print("\nwrote", OUT / "fanc_goal.json")


if __name__ == "__main__":
    main()
