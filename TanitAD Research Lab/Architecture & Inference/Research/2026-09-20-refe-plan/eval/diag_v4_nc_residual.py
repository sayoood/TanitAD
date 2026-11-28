#!/usr/bin/env python3
"""WHY does the teacher's collision label still disagree with NAVSIM on half-speed copies after --nc-at-fault?

Hypothesis (stated before this ran): NAVSIM scores a plan AFTER tracking it with its LQR controller + kinematic bicycle
from the ego's CURRENT speed, while the teacher injects the plan's poses directly. A 0.5x copy asks for an immediate
halving of the speed, which the tracker cannot follow, so NAVSIM's simulated ego runs AHEAD of the plan and meets the
scene at different places and times than the teacher's plan does.
Test: the max distance between the plan and NAVSIM's simulated rear axle (the SAME simulation the labeller already runs
for drivable area and comfort, `navsim_dac.simulated_states`), compared between copies where the at-fault label and
NAVSIM DISAGREE and copies where they AGREE, per factor; and the same for the originals as the control.

    python eval/diag_v4_nc_residual.py [--arm sets_ncaf]
Writes eval/raw/e6_sub200_ep015/v4_nc_residual.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))
import validate_slow_labels_v4 as V  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="sets_ncaf")
    ap.add_argument("--per-group", type=int, default=60, help="max slots simulated per (group, agree) cell")
    a = ap.parse_args()
    import navsim_dac as ND
    import navtrain_scenarios as NS
    import onpolicy_label_v4 as L4
    T = np.load(V.TABLE)
    tok = [str(t) for t in T["token"]]
    ix = {t: i for i, t in enumerate(tok)}
    top = np.load(V.SLOW_RANKS)["top"]
    D = np.load(V.STOP_DUMP, allow_pickle=False)
    Hc, stop_h = V.harness_copy_csvs(), V.read_csv(V.STOP_CSV)
    tl = json.load(open(V.TOKENS, encoding="utf-8"))["token_log"]
    dbs = NS.index_dbs()
    lines = V._lines(V.arm_dirs(a.arm)[1])
    cells: dict = {}
    for r in lines:
        i = ix[r["token"]]
        sl = r["slow"]
        so = {s: (sv, fv) for s, sv, fv in zip(sl["slots"], sl["src"], sl["factor"])}
        ti = [int(x) for x in top[i]]
        P = D["traj"][i].astype(np.float64)
        for slot in range(64):
            t = r["targets"][slot]
            if slot in so:
                s, f = so[slot]
                if f == 0.0:
                    continue
                rr = ti.index(s) if s in ti else None
                h = (Hc.get((f, "r", rr), {}).get(r["token"]) if rr is not None else None) or \
                    Hc.get((f, "p", s), {}).get(r["token"])
                if not h or not h[0]:
                    continue
                g, nav_nc, traj = V.ftag(f), h[1][0], L4.copy_traj(P, s, f)
            else:
                g, nav_nc, traj = "orig", T["sub"][i, slot, 0], P[slot]
            lab_ok = float(t["collision.NuPlanCollision.info"]) < 0.5
            agree = lab_ok == (nav_nc >= 0.999)
            cells.setdefault((g, agree), []).append((r["token"], traj))
    rng = np.random.default_rng(20260927)
    out = {}
    scs_cache: dict = {}
    for (g, agree), items in sorted(cells.items()):
        pick = [items[k] for k in rng.permutation(len(items))[:a.per_group]]
        dev = []
        for token, traj in pick:
            sc = scs_cache.get(token)
            if sc is None:
                lg = tl[token]
                sc = {x.scenario_name: x for x in NS.build_scenarios_for_log(dbs[lg], [token])}[token]
                scs_cache[token] = sc
            ego = sc.get_ego_state_at_iteration(0)
            sim = ND.simulated_states(ND.to_navsim(traj)[None], ego)[0]            # [41, state] map frame, 0.1 s
            ra = ego.rear_axle
            c, s_ = np.cos(ra.heading), np.sin(ra.heading)
            dx, dy = sim[:, 0] - ra.x, sim[:, 1] - ra.y
            sim_rel = np.stack([c * dx + s_ * dy, -s_ * dx + c * dy], -1)          # ego frame
            plan10 = ND.at_10hz(ND.to_navsim(traj))[:, :2]                          # the plan on the same 0.1 s grid
            dev.append(float(np.linalg.norm(sim_rel - plan10, axis=1).max()))
        out[f"{g}_{'agree' if agree else 'DISAGREE'}"] = {
            "slots_in_cell": len(items), "simulated": len(dev),
            "max_plan_vs_sim_m_median": round(float(np.median(dev)), 3) if dev else None,
            "max_plan_vs_sim_m_p90": round(float(np.percentile(dev, 90)), 3) if dev else None}
        print(f"  {g:5s} {'agree   ' if agree else 'DISAGREE'} n {len(items):5d}  simulated {len(dev):3d}  max |plan - "
              f"NAVSIM sim| median {out[f'{g}_' + ('agree' if agree else 'DISAGREE')]['max_plan_vs_sim_m_median']} m",
              flush=True)
    res = {"_label": "DIAGNOSTIC (exploratory): plan vs NAVSIM-simulated ego, by label agreement", "arm": a.arm,
           "cells": out}
    p = os.path.join(HERE, "raw", f"e6_{V.NAME}", "v4_nc_residual.json")
    json.dump(res, open(p, "w", encoding="utf-8"), indent=1)
    print(f"ZZV4NCRES_DONE -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
