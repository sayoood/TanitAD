#!/usr/bin/env python3
"""WHY is the goal input 349-480 m ahead on 20 navtest tokens? Re-derive `REFePlanner._goal_for` step by step
(refe/planner.py:449-474 -> code/augment_routes.py:46-70 `_route_with_lane_rank` -> nuplan-devkit
driverl_runtime_map_features.py `_route_start_index` :881, `_choose_route_edge` :835, `_fit_route_polyline` :206 ->
DriveRL goal_position_utils.py:512 `route_goal_positions`) for one token of each affected log and a normal control, from
the scenario DB and the map. CPU only. Writes goal_trace.json.
"""
from __future__ import annotations

import json
import math
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

TOKENS = {"2021.08.30.13.45.25_veh-40_01116_01336": "3efebf87894a552e",
          "2021.09.09.17.18.51_veh-48_00098_00328": "5b24b678ff67555d"}


def main() -> int:
    if os.environ.get("REFE_GT_CHILD") != "1":
        env = EC.env_driverl()
        env["REFE_GT_CHILD"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), str(PKG / "refe"), str(PKG / "eval"),
                                             str(PKG / "code")])
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__)], cwd=str(PKG / "eval"), env=env)
    import torch
    import augment_routes as A
    import navtrain_scenarios as NS
    import refe_navtest_seam as SEAM
    from driverl.datatypes.goal_position_utils import route_goal_positions
    from nuplan.common.maps.abstract_map import SemanticMapLayer
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.planning.script import driverl_runtime_map_features as M
    rows = {}
    for l in open("D:/Projects/TanitAD/data/refe_proxy/cache_eval_ep015/rows.jsonl", encoding="utf-8"):
        r = json.loads(l)
        rows[r["key"][1]] = r
    ctrl = next(k for k, r in rows.items() if r["key"][0] not in TOKENS and 40 < np.linalg.norm(r["goal"][2:4]) < 90)
    todo = dict(TOKENS)
    todo[rows[ctrl]["key"][0]] = ctrl
    out = {}
    for lg, tok in todo.items():
        sc = next(iter(NS.build_scenarios_for_log(os.path.join(SEAM.TEST_DB_DIR, f"{lg}.db"), [tok], history_rows=1,
                                                  future_rows=80)))
        ids = list(sc.get_route_roadblock_ids() or [])
        mp = sc.map_api
        ego = sc.get_ego_state_at_iteration(0)
        anchor = A._anchor_from_ego(ego)
        rbs = [(rid, M._route_roadblock_for_id(mp, rid)) for rid in ids]
        per = []
        for rid, rb in rbs:
            edges = [e for e in (getattr(rb, "interior_edges", []) or []) if M._edge_centerline_coords(e)] if rb else []
            dmin = min((min(math.hypot(x - anchor.x, y - anchor.y) for x, y in M._edge_centerline_coords(e))
                        for e in edges), default=float("inf"))
            score = min((M._route_edge_anchor_score(e, anchor) for e in edges), default=float("inf"))
            per.append({"id": str(rid), "found": rb is not None, "n_edges": len(edges), "min_dist_m": round(dmin, 1),
                        "anchor_score": round(score, 1)})
        start = M._route_start_index([rb for _r, rb in rbs if rb is not None], anchor)
        near = mp.get_proximal_map_objects(Point2D(anchor.x, anchor.y), 3.0,
                                           [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR])
        ego_rbs = [str(o.id) for lay in near.values() for o in lay]
        poly, _ = A._route_with_lane_rank(mp, ids, anchor, 0)
        rec = {"token": tok, "n_route_roadblocks": len(ids), "route_roadblocks": per, "route_start_index": start,
               "ego_on_roadblocks_within_3m": ego_rbs,
               "ego_roadblock_in_route": any(r in [str(i) for i in ids] for r in ego_rbs)}
        if poly is not None:
            P = np.asarray(poly, float)
            d = np.linalg.norm(P, axis=1)
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            rp = torch.as_tensor(P[None], dtype=torch.float32)
            mask = torch.as_tensor((np.abs(P).sum(-1) > 0)[None])
            v = ego.dynamic_car_state.rear_axle_velocity_2d
            g = route_goal_positions(torch.zeros(1, 1, 2), torch.tensor([[[float(v.x), float(v.y)]]]), rp, mask,
                                     horizon_s=12.0, min_speed_mps=5.0, num_goal_positions=2).reshape(-1).numpy()
            rec.update({"poly_first_point_m": [round(float(x), 1) for x in P[0]], "poly_first_dist_m": round(float(d[0]), 1),
                        "poly_min_dist_to_ego_m": round(float(d.min()), 1), "poly_argmin_index": int(d.argmin()),
                        "poly_length_m": round(float(seg.sum()), 1), "poly_last_point_m": [round(float(x), 1) for x in P[-1]],
                        "goal_rederived": [round(float(x), 1) for x in g],
                        "goal_in_eval_cache": [round(float(x), 1) for x in rows[tok]["goal"]]})
        out[lg] = rec
        print(lg, json.dumps(rec)[:1500], flush=True)
    json.dump(out, open(HERE / "goal_trace.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZGOALTRACE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
