#!/usr/bin/env python3
"""tracked_dump.py -- re-derive NAVSIM v1.1's DAC for the C3 table's proposals FROM ITS OWN CODE, and dump the footprints it judges.

Runs in the NAVSIM venv (python 3.9) with the v1.1 source the harness used (PYTHONPATH = navsim-3e8291bf...):
    PYTHONPATH=D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896 \
    C:/Users/Admin/navsim-crun/venv/Scripts/python.exe tracked_dump.py --stride 7 --shard 0/4 --out C:/Users/Admin/tmpauxv/tracked

Per token (all 64 proposals), exactly `navsim/evaluate/pdm_score.py:83-140`, batched:
  raw states  = transform_trajectory + get_trajectory_as_array   (the proposal's own interpolated 41-step trajectory)
  sim states  = PDMSimulator.simulate_proposals(raw, ego_state)  (LQR tracker + kinematic bicycle) -- what the scorer judges
  D0          = PDMScorer.score_proposals(sim, ...)._multi_metrics[DRIVABLE_AREA]      -> must equal the table's DAC (reproduction)
  D1          = the SAME non-drivable rule (pdm_scorer.py:284-286) applied to the RAW states (no tracker) with NAVSIM's own polygons
  corners     = state_array_to_coords_array(...) minus the centre, in the EGO frame (rear-axle origin of the initial ego state)
Nothing here reads a model output other than the 64 stored proposals; nothing is picked.
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

PK = Path(r"C:/Users/Admin/AppData/Local/Temp/auxv/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
C3 = PK / "raw/2026-10-11-c3-selection-headroom"
MC_ROOT = Path("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))

    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    from nuplan.planning.simulation.planner.ml_planner.transform_utils import _get_fixed_timesteps, _se2_vel_acc_to_ego_state
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, MultiMetricIndex
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    future = TrajectorySampling(num_poses=40, interval_length=0.1)
    model_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
    simulator = PDMSimulator(future)
    scorer = PDMScorer(future, PDMScorerConfig())
    corner_idx = [i for i in range(len(BBCoordsIndex)) if i != int(BBCoordsIndex.CENTER)]

    z = np.load(C3 / "table.npz")
    tj = json.load(open(C3 / "tokens.json"))
    tok = [str(t) for t in z["token"]]
    order = sorted(range(len(tok)), key=lambda i: tok[i])
    sel = [i for r, i in enumerate(order) if r % a.stride == 0]
    sel = [i for k, i in enumerate(sel) if k % sn == si]
    if a.limit:
        sel = sel[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {"idx": [], "token": [], "origin": [], "d0_sim_poly": [], "d1_raw_poly": [], "corners_sim_ego": [], "corners_raw_ego": [], "vel": [], "sec": []}
    t_start = time.time()
    for n, i in enumerate(sel):
        t0 = time.time()
        t = tok[i]
        lg = tj["token_log"][t]
        mc = pickle.loads(lzma.decompress(open(MC_ROOT / lg / "unknown" / t / "metric_cache.pkl", "rb").read()))
        ego = mc.ego_state
        props = z["proposals"][i].astype(np.float64)                       # [64, 8, 3]
        timesteps = _get_fixed_timesteps(ego, model_sampling.time_horizon, model_sampling.interval_length)
        arrs = []
        for p in props:
            absolute = relative_to_absolute_poses(ego.rear_axle, [StateSE2.deserialize(q) for q in p])
            agents = [_se2_vel_acc_to_ego_state(s, [0.0, 0.0], [0.0, 0.0], ts, ego.car_footprint.vehicle_parameters) for s, ts in zip(absolute, timesteps)]
            arrs.append(get_trajectory_as_array(InterpolatedTrajectory([ego] + agents), future, ego.time_point))
        raw = np.stack(arrs)                                                # [64, 41, S]
        sim = simulator.simulate_proposals(raw, ego)
        scorer.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
        d0 = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA].copy()   # [64]
        vp = ego.car_footprint.vehicle_parameters
        dm = mc.drivable_area_map
        idcs = dm.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])

        def poly_dac(states):
            coords = state_array_to_coords_array(states, vp)                # [P, T, 5, 2]
            inp = dm.points_in_polygons(coords).transpose(1, 2, 0, 3)       # [P, T, n_poly, 5]
            cin = inp[..., :-1]
            nondriv = (cin[:, :, idcs].sum(axis=-2) > 0).sum(axis=-1) < 4   # pdm_scorer.py:284-286
            return (~nondriv.any(axis=-1)).astype(np.float64), coords

        d1, c_raw = poly_dac(raw)
        _, c_sim = poly_dac(sim)
        x0, y0, h0 = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        ch, sh = np.cos(h0), np.sin(h0)

        def to_ego(c):                                                      # global -> ego (rear-axle) frame
            d = c[..., corner_idx, :] - np.array([x0, y0])
            return np.stack([d[..., 0] * ch + d[..., 1] * sh, -d[..., 0] * sh + d[..., 1] * ch], -1).astype(np.float32)

        rows["idx"].append(i)
        rows["token"].append(t)
        rows["origin"].append([x0, y0, h0])
        rows["d0_sim_poly"].append(d0)
        rows["d1_raw_poly"].append(d1)
        rows["corners_sim_ego"].append(to_ego(c_sim))
        rows["corners_raw_ego"].append(to_ego(c_raw))
        rows["vel"].append([ego.dynamic_car_state.rear_axle_velocity_2d.x, ego.dynamic_car_state.rear_axle_velocity_2d.y])
        rows["sec"].append(time.time() - t0)
        print(f"[{n + 1}/{len(sel)}] {t} {time.time() - t0:.1f}s  d0==table {int((d0 == z['sub'][i][:, 1]).sum())}/64", flush=True)
    np.savez(out / f"tracked_{si}of{sn}.npz", **{k: (np.array(v) if k != "token" else np.array(v, dtype="U16")) for k, v in rows.items()})
    print(f"ZZDONE shard {si}/{sn} n={len(sel)} wall={time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
