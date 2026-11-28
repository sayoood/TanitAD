#!/usr/bin/env python3
"""DISCRIMINATING DIAGNOSTIC (2026-10-16): is a rank-K twin the ONSET of a lane change (it reaches the rank-K lane later) or a perturbation
that never commits? Roll the same twin 10 s instead of 4 s, route HELD vs per-step re-ranking (legacy), and read the lateral fraction
f(t) = d_logged / (d_logged + d_twin) against the twin's and the logged PDM centrelines at 1..10 s. Diagnostic only: writes no bank row.
    run from /workspace/ops/d7fix/pkg/refe with the env of probe_run.sh:
    $PY long_rollout.py --bank <dir> --arm held|legacy --out <jsonl> --mc <parts dir> --code-dir <patched> [--log-shard i/N] [--span-s 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _min_speed(spec, ego, ms0):
    """None -> the config default; a float -> that floor; 'v0max' -> max(frame speed, default) (addendum 5)."""
    if spec is None:
        return ms0
    if spec == "v0max":
        return max(float(ego[6]), ms0)
    return float(spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True); ap.add_argument("--arm", choices=("held", "legacy"), required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--mc", required=True); ap.add_argument("--code-dir", required=True)
    ap.add_argument("--refe-dir", required=True); ap.add_argument("--pkg-code-dir", required=True)
    ap.add_argument("--log-shard", default=None); ap.add_argument("--span-s", type=float, default=10.0)
    ap.add_argument("--goal-horizon", type=float, default=12.0)
    ap.add_argument("--goal-min-speed", default=None, help="LEVER (addendum 5): float m/s or v0max; unset = the config default")
    a = ap.parse_args()
    os.environ["REFE_ROUTE_HOLD"] = "1" if a.arm == "held" else "0"
    sys.path[:0] = [a.code_dir, a.refe_dir, a.pkg_code_dir]
    import numpy as np
    import shapely
    import route_lane_rank_patch as P
    import build_teacher_rollouts as BR
    import navtrain_scenarios as NS
    import online_labeller as OL
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    assert P.install(1) and hasattr(P, "held_route")
    rows = [json.loads(l) for l in open(Path(a.bank) / "targets_rank1.jsonl", encoding="utf-8")]
    logs = sorted({r["log_name"] for r in rows})
    if a.log_shard:
        i, n = (int(x) for x in a.log_shard.split("/"))
        logs = logs[i::n]
    BR.TEACHER_DEVICE = "cpu"
    planner = BR.build_planner("cpu")
    MS0 = float(planner._feature_builder.config.route_goal_min_speed_mps)
    dbs = NS.index_dbs()
    fo = open(a.out, "a", encoding="utf-8")
    for lg in logs:
        rs = [r for r in rows if r["log_name"] == lg]
        scs = {sc.scenario_name: sc for sc in NS.build_scenarios_for_log(dbs[lg], sorted({r["token"] for r in rs}))}
        for r in rs:
            k = int(r["aug"]["value"])
            sc = scs.get(r["token"])
            rec = {"log_name": lg, "token": r["token"], "k": k, "arm": a.arm, "v0": r["ego"][6]}
            tw_p, lg_p = OL.parts_path(a.mc, lg, r["token"], {"kind": "lane_rank", "value": k}), OL.parts_path(a.mc, lg, r["token"], None)
            if sc is None or not tw_p.is_file() or not lg_p.is_file():
                rec["error"] = "missing scenario or parts"; fo.write(json.dumps(rec) + "\n"); continue
            dt = float(sc.database_interval)
            span = int(round(a.span_s / dt))
            controller = BR.build_controller(sc, "cpu")
            P.set_rank(k)
            planner._feature_builder.config.route_goal_horizon_s = a.goal_horizon
            rec["goal_min_speed"] = planner._feature_builder.config.route_goal_min_speed_mps = _min_speed(a.goal_min_speed, r["ego"], MS0)
            planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                                     mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            t0 = time.time()
            try:
                poses = BR.rollout_from_frame(sc, planner, controller, int(r.get("step", 0)), span)
            except Exception as exc:                                   # noqa: BLE001
                BR._reset_planner_state(planner)
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"; fo.write(json.dumps(rec) + "\n"); continue
            if poses is None:
                rec["error"] = f"scenario too short for {span} steps"; fo.write(json.dumps(rec) + "\n"); continue
            twl, lgl = OL.load_parts(tw_p)[3].linestring, OL.load_parts(lg_p)[3].linestring
            xy = shapely.points(np.array([[p.rear_axle.x, p.rear_axle.y] for p in poses], dtype=np.float64))
            d_lg, d_tw = shapely.distance(lgl, xy), shapely.distance(twl, xy)
            f = d_lg / np.maximum(d_lg + d_tw, 1e-6)
            per_s = int(round(1.0 / dt))
            rec["f_at_s"] = [round(float(f[min(per_s * s - 1, len(f) - 1)]), 3) for s in range(1, int(a.span_s) + 1)]
            rec["d_twin_at_s"] = [round(float(d_tw[min(per_s * s - 1, len(f) - 1)]), 2) for s in range(1, int(a.span_s) + 1)]
            h = getattr(BR.rollout_from_frame, "last_hold", None)
            rec["hold_misses"] = None if h is None else h["misses"]
            rec["rollout_s"] = round(time.time() - t0, 2)
            fo.write(json.dumps(rec) + "\n"); fo.flush()
    fo.close()
    print("ZZLONG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
