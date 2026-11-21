#!/usr/bin/env python3
"""Census, CPU only, NO model: for each navtest token of the 1,123, the ego's distance to the route polyline that
`REFePlanner._goal_for` builds (planner.py:461-463 -> augment_routes._route_with_lane_rank, rank 0), and the goal the
planner would pass. It answers: does 'ego farther than D m from its own route' fire on EXACTLY the goal-p2 > 300 m
tokens, and on how many others (the draft Amendment 8's trigger, measured before any confirmation token is read)?
    python raw/2026-09-28-goal-clamp/route_cover_census.py [--tokens <W3-format json>] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "eval"))
sys.path.insert(0, str(PKG / "refe"))
sys.path.insert(0, str(PKG / "code"))
import eval_checkpoint as EC  # noqa: E402


def seg_dist(P):
    """min distance from the origin (the ego) to the polyline's segments"""
    a, b = P[:-1], P[1:]
    ab = b - a
    t = np.clip(-(a * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab, axis=1).min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=str(PKG / "raw" / "2026-09-28-m6b-tangent" / "tokens_1123.json"))
    ap.add_argument("--out", default=str(HERE / "route_cover_census.json"))
    a = ap.parse_args()
    if os.environ.get("REFE_RC_CHILD") != "1":
        # the child runs with cwd=eval/, so a RELATIVE --out / --tokens is resolved HERE, against the caller's cwd
        # (MEASURED 2026-09-28 01:59: a relative --out made the full-navtest census compute for 9 min, then die at the
        # final write with FileNotFoundError)
        sys.argv = [sys.argv[0], "--tokens", str(Path(a.tokens).resolve()), "--out", str(Path(a.out).resolve())]
        env = EC.env_driverl()
        env["REFE_RC_CHILD"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), str(PKG / "refe"), str(PKG / "eval"),
                                             str(PKG / "code")])
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), *sys.argv[1:]], cwd=str(PKG / "eval"), env=env)
    import augment_routes as A
    import navtrain_scenarios as NS
    import refe_navtest_seam as SEAM
    from nuplan.planning.script import driverl_runtime_map_features as M  # noqa: F401
    tj = json.load(open(a.tokens, encoding="utf-8"))
    by_log: dict = {}
    for t in tj["tokens"]:
        by_log.setdefault(tj["token_log"][t], []).append(t)
    rows = {}
    for lg, toks in sorted(by_log.items()):
        try:
            for sc in NS.build_scenarios_for_log(os.path.join(SEAM.TEST_DB_DIR, f"{lg}.db"), toks, history_rows=1,
                                                 future_rows=80):
                ego = sc.get_ego_state_at_iteration(0)
                ids = list(sc.get_route_roadblock_ids() or [])
                poly, _ = A._route_with_lane_rank(sc.map_api, ids, A._anchor_from_ego(ego), 0)
                d = seg_dist(np.asarray(poly, float)) if poly is not None else None
                rows[sc._initial_lidar_token] = {"log": lg, "ego_to_route_m": d, "no_route": poly is None}
        except Exception as e:                                         # a log that cannot be opened is reported, never
            for t in toks:                                             # silently dropped
                rows.setdefault(t, {"log": lg, "error": repr(e)[:200]})
        print(f"  {lg}: {sum(1 for r in rows.values() if r['log'] == lg)} / {len(toks)}", flush=True)
    d = np.array([r["ego_to_route_m"] for r in rows.values() if r.get("ego_to_route_m") is not None])
    out = {"n_tokens": len(tj["tokens"]), "n_measured": int(d.size),
           "n_no_route": sum(1 for r in rows.values() if r.get("no_route")),
           "n_error": sum(1 for r in rows.values() if "error" in r),
           "ego_to_route_m_quantiles": {q: float(np.percentile(d, q)) for q in (50, 90, 99, 99.9, 100)} if d.size else {},
           "count_gt": {str(x): int((d > x).sum()) for x in (5, 10, 20, 50, 100, 300)}, "rows": rows}
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("ZZROUTECENSUS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
