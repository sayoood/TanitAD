#!/usr/bin/env python3
"""CPU only, NO model, no GPU: for EVERY navtest token, the goal `REFePlanner._goal_for` would pass (the teacher's
`route_goal_positions` on the route polyline: 2 points, horizon 12 s, min speed 5 m/s), the ego's speed, and -- for the OLD
Amendment-8 trigger only -- the ego's distance to that polyline. It lets a trigger that reads the NAV GOAL ITSELF (and v0) be
compared with the old one on all 12,146 tokens, label-free.
    python goal_census.py [--tokens <W3 json>] [--out <json>]
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

HORIZON_S, MIN_SPEED, N_GOAL = 12.0, 5.0, 2


def seg_dist(P):
    a, b = P[:-1], P[1:]
    ab = b - a
    t = np.clip(-(a * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab, axis=1).min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="D:/Projects/TanitAD/data/refe_navtest/navtest_all_tokens.json")
    ap.add_argument("--out", default=str(HERE / "goal_census_navtest_full.json"))
    a = ap.parse_args()
    if os.environ.get("REFE_GC_CHILD") != "1":
        sys.argv = [sys.argv[0], "--tokens", str(Path(a.tokens).resolve()), "--out", str(Path(a.out).resolve())]
        env = EC.env_driverl()
        env["REFE_GC_CHILD"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), str(PKG / "refe"), str(PKG / "eval"), str(PKG / "code")])
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), *sys.argv[1:]], cwd=str(PKG / "eval"), env=env)
    import torch
    import augment_routes as A
    import navtrain_scenarios as NS
    import refe_navtest_seam as SEAM
    from driverl.datatypes.goal_position_utils import route_goal_positions
    from nuplan.planning.script import driverl_runtime_map_features as M  # noqa: F401
    tj = json.load(open(a.tokens, encoding="utf-8"))
    toks = tj["tokens"] if isinstance(tj, dict) else tj
    tl = tj.get("token_log") if isinstance(tj, dict) else None
    if not tl:
        import gzip
        exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
        tl = {t: exp[t]["log_name"] for t in toks}
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(tl[t], []).append(t)
    rows = {}
    for lg, ts in sorted(by_log.items()):
        try:
            for sc in NS.build_scenarios_for_log(os.path.join(SEAM.TEST_DB_DIR, f"{lg}.db"), ts, history_rows=1, future_rows=80):
                ego = sc.get_ego_state_at_iteration(0)
                ids = list(sc.get_route_roadblock_ids() or [])
                poly, _ = A._route_with_lane_rank(sc.map_api, ids, A._anchor_from_ego(ego), 0)
                v = ego.dynamic_car_state.rear_axle_velocity_2d
                r = {"log": lg, "v": [float(v.x), float(v.y)], "no_route": poly is None}
                if poly is not None:
                    rp = torch.as_tensor(np.asarray(poly)[None], dtype=torch.float32)
                    mask = torch.as_tensor((np.abs(np.asarray(poly)).sum(-1) > 0)[None])
                    g = route_goal_positions(torch.zeros(1, 1, 2), torch.tensor([[[float(v.x), float(v.y)]]], dtype=torch.float32),
                                             rp, mask, horizon_s=HORIZON_S, min_speed_mps=MIN_SPEED, num_goal_positions=N_GOAL)
                    r["goal"] = [float(x) for x in g.reshape(-1).tolist()]
                    r["ego_to_route_m"] = seg_dist(np.asarray(poly, float))
                rows[sc._initial_lidar_token] = r
        except Exception as e:
            for t in ts:
                rows.setdefault(t, {"log": lg, "error": repr(e)[:200]})
        print(f"  {lg}: {sum(1 for r in rows.values() if r['log'] == lg)} / {len(ts)}", flush=True)
    json.dump({"n_tokens": len(toks), "n_rows": len(rows), "n_error": sum(1 for r in rows.values() if "error" in r),
               "horizon_s": HORIZON_S, "min_speed_mps": MIN_SPEED, "n_goal": N_GOAL, "rows": rows},
              open(a.out, "w", encoding="utf-8", newline="\n"), indent=0)
    print("ZZGOALCENSUS_DONE", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
