#!/usr/bin/env python3
"""zt3_geom.py -- per-step drivable-area geometry of P / S / H on NAVSIM v1.1's OWN pipeline (PREREG_ZT3 section 3).

    PYTHONPATH=D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896 \
    C:/Users/Admin/navsim-crun/venv/Scripts/python.exe zt3_geom.py --jobs ../data/jobs.npz --shard 0/2 --out ../data/geom

Per token, exactly as `navsim/evaluate/pdm_score.py:83-140` (and `tracked_dump.py` of the aux-veto feasibility package):
  raw  = transform_trajectory + get_trajectory_as_array  (the plan's own interpolated 41 states)
  sim  = PDMSimulator.simulate_proposals(raw, ego_state) (LQR + kinematic bicycle) -- what the scorer judges
and on both: the 4 footprint corners vs ROADBLOCK/INTERSECTION/DRIVABLE_AREA/CARPARK polygons of the metric
cache (pdm_scorer.py:284-286 rule, via zt3_lib.nondrivable_steps), lane/connector/intersection membership of
the box centre, and the distance of out-corners to the drivable union. Nothing here picks or scores a plan.
"""
from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zt3_lib as L  # noqa: E402

MC_ROOT = Path("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of tokens (debug)")
    ap.add_argument("--mutate", default="", help="deliberate regression for the self-test: 'empty_poly' | 'shift500'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))

    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.planner.ml_planner.transform_utils import _get_fixed_timesteps, _se2_vel_acc_to_ego_state
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

    future = TrajectorySampling(num_poses=40, interval_length=0.1)
    model_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
    simulator = PDMSimulator(future)

    J = np.load(a.jobs)
    toks = [str(t) for t in J["token"]]
    rows = list(range(len(toks)))
    if a.only:
        want = set(a.only.split(","))
        rows = [r for r in rows if toks[r] in want]
    rows = [r for k, r in enumerate(rows) if k % sn == si]
    if a.limit:
        rows = rows[: a.limit]
    K = J["traj"].shape[1]
    n = len(rows)
    out = {"row": np.array(rows, np.int32), "token": np.array([toks[r] for r in rows], dtype="U16"),
           "nd_sim": np.zeros((n, K, 41), bool), "nd_raw": np.zeros((n, K, 41), bool), "out4": np.zeros((n, K, 41, 4), bool),
           "cxy": np.full((n, K, 41, 2), np.nan, np.float32), "psi": np.full((n, K, 41), np.nan, np.float32),
           "depth": np.zeros((n, K, 41), np.float32), "inter_c": np.zeros((n, K, 41), bool),
           "inter_dist_out": np.full((n, K, 41), np.inf, np.float32), "corners_AP": np.full((n, 41, 4, 2), np.nan, np.float32),
           "origin": np.zeros((n, 3)), "sec": np.zeros(n)}
    lanes_js = {}
    t_start = time.time()
    for q, r in enumerate(rows):
        t0 = time.time()
        t, lg = toks[r], str(J["log"][r])
        mc = pickle.loads(lzma.decompress(open(MC_ROOT / lg / "unknown" / t / "metric_cache.pkl", "rb").read()))
        ego = mc.ego_state
        vp = ego.car_footprint.vehicle_parameters
        dm = mc.drivable_area_map
        types = dm.map_types
        idcs = dm.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
                                           SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
        if a.mutate == "empty_poly":
            idcs = []
        lane_idx = dm.get_indices_of_map_type([SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
        inter_idx = dm.get_indices_of_map_type([SemanticMapLayer.INTERSECTION])
        geoms = dm._geometries
        driv_union = unary_union([geoms[i] for i in idcs]) if len(idcs) else None
        inter_union = unary_union([geoms[i] for i in inter_idx]) if len(inter_idx) else None
        timesteps = _get_fixed_timesteps(ego, model_sampling.time_horizon, model_sampling.interval_length)
        have = np.nonzero(J["have"][r])[0]
        arrs = []
        for j in have:
            p = J["traj"][r, j].astype(np.float64)
            if a.mutate == "shift500":
                p = p.copy()
                p[:, 1] += 500.0
            absolute = relative_to_absolute_poses(ego.rear_axle, [StateSE2.deserialize(x) for x in p])
            agents = [_se2_vel_acc_to_ego_state(s, [0.0, 0.0], [0.0, 0.0], ts, vp) for s, ts in zip(absolute, timesteps)]
            arrs.append(get_trajectory_as_array(InterpolatedTrajectory([ego] + agents), future, ego.time_point))
        raw = np.stack(arrs)
        sim = simulator.simulate_proposals(raw, ego)
        x0, y0, h0 = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        ch, sh = np.cos(h0), np.sin(h0)

        def to_ego(c):
            d = c - np.array([x0, y0])
            return np.stack([d[..., 0] * ch + d[..., 1] * sh, -d[..., 0] * sh + d[..., 1] * ch], -1)

        def corner_flags(states):
            coords = state_array_to_coords_array(states, vp)                     # [k, 41, 5, 2] global
            inp = dm.points_in_polygons(coords).transpose(1, 2, 0, 3)            # [k, 41, n_poly, 5]
            cin = inp[:, :, idcs, :4] if len(idcs) else np.zeros(inp.shape[:2] + (0, 4), bool)
            return coords, cin

        c_sim, cin_sim = corner_flags(sim)
        _, cin_raw = corner_flags(raw)
        nd_sim = L.nondrivable_steps(cin_sim)
        nd_raw = L.nondrivable_steps(cin_raw)
        out4 = ~cin_sim.any(axis=-2)                                             # [k, 41, 4]
        centre_g = c_sim[:, :, :4, :].mean(axis=2)                               # [k, 41, 2] (PREREG: mean of 4 corners)
        cen_in = dm.points_in_polygons(centre_g[:, :, None, :]).transpose(1, 2, 0, 3)[..., 0]   # [k, 41, n_poly]
        lanes_tok = {}
        for jj, j in enumerate(have):
            out["nd_sim"][q, j] = nd_sim[jj]
            out["nd_raw"][q, j] = nd_raw[jj]
            out["out4"][q, j] = out4[jj]
            out["cxy"][q, j] = to_ego(centre_g[jj])
            out["psi"][q, j] = sim[jj, :, StateIndex.HEADING] - h0
            out["inter_c"][q, j] = cen_in[jj][:, inter_idx].any(axis=-1) if len(inter_idx) else False
            lanes_tok[int(j)] = [[str(dm._tokens[i]) for i in lane_idx if cen_in[jj, s, i]] for s in range(41)]
            for s in np.nonzero(out4[jj].any(axis=-1))[0]:
                pts = [Point(*c_sim[jj, s, c]) for c in range(4) if out4[jj, s, c]]
                if driv_union is not None:
                    out["depth"][q, j, s] = max(driv_union.distance(pt) for pt in pts)
                if inter_union is not None:
                    out["inter_dist_out"][q, j, s] = min(inter_union.distance(pt) for pt in pts)
            if j == 0:
                out["corners_AP"][q] = to_ego(c_sim[jj, :, :4, :])
        lanes_js[t] = {"lanes": lanes_tok, "route": [str(x) for x in mc.route_lane_ids]}
        out["origin"][q] = [x0, y0, h0]
        out["sec"][q] = time.time() - t0
        if q % 25 == 0 or q == n - 1:
            print(f"[{q + 1}/{n}] {t} {out['sec'][q]:.1f}s  elapsed {time.time() - t_start:.0f}s", flush=True)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    tag = f"geom_{si}of{sn}" + (f"_{a.mutate}" if a.mutate else "") + ("_only" if a.only else "")
    np.savez_compressed(od / f"{tag}.npz", **out)
    with lzma.open(od / f"{tag}_lanes.json.xz", "wt", encoding="utf-8") as f:
        json.dump(lanes_js, f)
    print(f"ZZDONE {tag} n={n} wall={time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
