#!/usr/bin/env python3
"""trk_dump.py -- G0-TRK instrument (PREREG_VDAC_P6_A2 sec.1/3): NAVSIM v1.1 tracked states for P6's 64 proposals on all navtest tokens.

Adapted from the banked 2026-10-16 tracked_dump.py (same NAVSIM path pdm_score.py:83-140). Runs in the NAVSIM venv (py3.9):
  PYTHONPATH=<navsim-3e8291bf...> C:/Users/Admin/navsim-crun/venv/Scripts/python.exe trk_dump.py --shard 0/2 --out DIR
Per token (sorted-token order, round-robin shards): raw xyh [64,41,3], tracked xyh [64,41,3] (float64), d0 (NAVSIM DAC on tracked),
d1 (same rule on raw), origin (rear-axle x,y,h), vehicle params. Written in chunks (resumable). Reads NO model output beyond the stored
proposals; selects nothing, scores nothing but the DAC identity.
Also: rebuild_corners(xyh, vp) = NAVSIM state_array_to_coords_array (4 corners, centre excluded), optionally in the ego frame.
"""
from __future__ import annotations

import argparse
import hashlib
import lzma
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROPOSALS = "F:/TanitAD_p6read_work/pass2/proposals_p6final.npz"
TABLE = "F:/tanitad_scratch/selq2read/table_p6final_c.npz"
TABLE_SHA_PREFIX = "3bda2080"
MC_ROOTS = ["F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest", "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"]
DAC_COL = 1  # MultiMetricIndex.DRIVABLE_AREA; table 'sub' [N,64,6], col 1 == DAC
CHUNK = 250
NCORNER_CHECK = 20  # first tokens of each chunk keep NAVSIM's own global-frame corners for the rebuild check


class VP:
    """Minimal stand-in carrying the three fields state_array_to_coords_array reads."""
    def __init__(self, half_length, half_width, rear_axle_to_center):
        self.half_length, self.half_width, self.rear_axle_to_center = half_length, half_width, rear_axle_to_center


def rebuild_corners(xyh, vp, origin=None):
    """xyh [...,T,3] (global x,y,heading) -> corners [...,T,4,2] via NAVSIM state_array_to_coords_array (centre dropped).
    vp: object with half_length/half_width/rear_axle_to_center, or the stored vp row [hl,hw,r2c,...].
    If origin=(x0,y0,h0) is given, returns them in that rear-axle frame (float64)."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex
    if not hasattr(vp, "half_length"):
        vp = VP(float(vp[0]), float(vp[1]), float(vp[2]))
    idx = [i for i in range(len(BBCoordsIndex)) if i != int(BBCoordsIndex.CENTER)]
    xyh = np.asarray(xyh, dtype=np.float64)
    lead = xyh.shape[:-2]
    flat = xyh.reshape((-1,) + xyh.shape[-2:])
    st = np.zeros(flat.shape[:2] + (10,))
    st[..., :3] = flat
    c = state_array_to_coords_array(st, vp)[..., idx, :].reshape(lead + (xyh.shape[-2], 4, 2))
    if origin is not None:
        x0, y0, h0 = origin
        d = c - np.array([x0, y0])
        ch, sh = np.cos(h0), np.sin(h0)
        c = np.stack([d[..., 0] * ch + d[..., 1] * sh, -d[..., 0] * sh + d[..., 1] * ch], -1)
    return c


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="only the first N sorted tokens (before sharding)")
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
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    mc_root = next((Path(r) for r in MC_ROOTS if Path(r).is_dir()), None)
    if mc_root is None:
        sys.exit("FATAL metric cache not found in " + str(MC_ROOTS))
    future = TrajectorySampling(num_poses=40, interval_length=0.1)
    model_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
    simulator = PDMSimulator(future)
    scorer = PDMScorer(future, PDMScorerConfig())

    tab = np.load(TABLE, allow_pickle=True)
    prp = np.load(PROPOSALS, allow_pickle=True)
    tok = [str(t) for t in tab["token"]]
    assert [str(t) for t in prp["token"]] == tok, "proposal/table token order differ"
    logs = [str(l) for l in tab["log"]]
    sub = tab["sub"]
    order = sorted(range(len(tok)), key=lambda i: tok[i])
    if a.limit:
        order = order[: a.limit]
    mine = [i for k, i in enumerate(order) if k % sn == si]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    props_all = prp["proposals"]
    t_start = time.time()
    nbad = 0
    for c0 in range(0, len(mine), CHUNK):
        part = out / f"trk_{si}of{sn}_part{c0 // CHUNK:04d}.npz"
        if part.exists():
            continue
        rows = {k: [] for k in ("idx", "token", "origin", "vp", "raw_xyh", "trk_xyh", "d0", "d1", "sec")}
        corners_chk = []
        for n, i in enumerate(mine[c0:c0 + CHUNK]):
            t0 = time.time()
            t = tok[i]
            mc = pickle.loads(lzma.decompress(open(mc_root / logs[i] / "unknown" / t / "metric_cache.pkl", "rb").read()))
            ego = mc.ego_state
            props = props_all[i].astype(np.float64)
            timesteps = _get_fixed_timesteps(ego, model_sampling.time_horizon, model_sampling.interval_length)
            arrs = []
            for p in props:
                absolute = relative_to_absolute_poses(ego.rear_axle, [StateSE2.deserialize(q) for q in p])
                agents = [_se2_vel_acc_to_ego_state(s, [0.0, 0.0], [0.0, 0.0], ts, ego.car_footprint.vehicle_parameters) for s, ts in zip(absolute, timesteps)]
                arrs.append(get_trajectory_as_array(InterpolatedTrajectory([ego] + agents), future, ego.time_point))
            raw = np.stack(arrs)
            sim = simulator.simulate_proposals(raw, ego)
            scorer.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
            d0 = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA].copy()
            vp = ego.car_footprint.vehicle_parameters
            dm = mc.drivable_area_map
            idcs = dm.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
            coords = state_array_to_coords_array(raw, vp)
            inp = dm.points_in_polygons(coords).transpose(1, 2, 0, 3)
            nondriv = (inp[..., :-1][:, :, idcs].sum(axis=-2) > 0).sum(axis=-1) < 4  # pdm_scorer.py:284-286
            d1 = (~nondriv.any(axis=-1)).astype(np.float64)
            ok = int((d0 == sub[i][:, DAC_COL]).sum())
            nbad += 64 - ok
            rows["idx"].append(i)
            rows["token"].append(t)
            rows["origin"].append([ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading])
            rows["vp"].append([vp.half_length, vp.half_width, vp.rear_axle_to_center, vp.length, vp.width, vp.wheel_base])
            rows["raw_xyh"].append(raw[..., :3])
            rows["trk_xyh"].append(sim[..., :3])
            rows["d0"].append(d0)
            rows["d1"].append(d1)
            rows["sec"].append(time.time() - t0)
            if n < NCORNER_CHECK:
                corners_chk.append(state_array_to_coords_array(sim, vp))
            print(f"[{c0 + n + 1}/{len(mine)}] {t} {time.time() - t0:.1f}s d0==table {ok}/64", flush=True)
        tmp = str(part) + ".tmp.npz"
        np.savez(tmp, idx=np.array(rows["idx"]), token=np.array(rows["token"], dtype="U16"), origin=np.array(rows["origin"]),
                 vp=np.array(rows["vp"]), raw_xyh=np.stack(rows["raw_xyh"]), trk_xyh=np.stack(rows["trk_xyh"]),
                 d0=np.stack(rows["d0"]), d1=np.stack(rows["d1"]), sec=np.array(rows["sec"]), corners_chk_global=np.stack(corners_chk))
        Path(tmp).replace(part)
    (out / f"DONE_{si}of{sn}").write_text(f"n={len(mine)} d0_table_mismatch={nbad} wall={time.time() - t_start:.0f}s\n")
    print(f"ZZTRK_DONE shard {si}/{sn} n={len(mine)} d0_table_mismatch={nbad} wall={time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
