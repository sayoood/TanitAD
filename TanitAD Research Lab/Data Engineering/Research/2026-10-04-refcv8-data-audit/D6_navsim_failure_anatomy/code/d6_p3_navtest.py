"""D6 P3 (navtest) -- exact re-score + perfect-fix oracles on the navsim-1.1 devkit (single-stage PDMS v1).   NAVSIM VENV, CPU only.

    PYTHONPATH="D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896;C:/Users/Admin/navsim-crun/nuplan-devkit" \
    OMP_NUM_THREADS=2 C:/Users/Admin/navsim-crun/venv/Scripts/python.exe d6_p3_navtest.py --hooks <navtest A1 30k hooks.json> \
        --tokens <file> --out <jsonl> [--shard i/n] [--limit N]

Per token: (1) the exact devkit re-score of the banked A1 plan (v1.1 ``pdm_score``, the SAME objects the official navtest run used:
``PDMSimulator`` / ``PDMScorer`` with the yaml defaults, non-reactive observation as cached) with the first non-drivable step, the
PDM-Closed reference's DAC, and the first at-fault collision event; (2) SPEED oracle: the same path at 0.8 / 0.6 / 0.4 x the speed, full
score; (3) SNAP oracle: DAC / DDC (map-only) for B075 / B150 / FULL centreline snaps and speed-only V80 / V60 / V40.  Oracle literals as
d6_p3_oracles.py.  The re-score must reproduce the banked row (control).
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import pickle
import sys
import time

import numpy as np

for k, v in {"NUPLAN_MAP_VERSION": "nuplan-maps-v1.0", "NUPLAN_MAPS_ROOT": "D:/Archive/devbox-C/navsim/data/maps",
             "NAVSIM_EXP_ROOT": "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1", "OPENSCENE_DATA_ROOT": "D:/Archive/devbox-C/navsim/data/openscene"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_geomutil as U                                             # noqa: E402

from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES   # noqa: E402
from nuplan.planning.metrics.utils.collision_utils import CollisionType   # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling   # noqa: E402
from navsim.common.dataclasses import Trajectory                     # noqa: E402
from navsim.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array   # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig   # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type   # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator   # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex, MultiMetricIndex, StateIndex, WeightedMetricIndex   # noqa: E402

CACHE = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
SAMP = TrajectorySampling(time_horizon=4, interval_length=0.5)
KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score")


def nc_events(scorer, idx):
    obs = scorer._observation
    collided = list(obs.collided_track_ids)
    ev = []
    for t in range(scorer.proposal_sampling.num_poses + 1):
        poly = scorer._ego_polygons[idx, t]
        inter = obs[t].query(np.array([poly]), predicate="intersects")
        if len(inter) == 0 or len(inter[0]) == 0:
            continue
        for _pi, gi in zip(inter[0], inter[1]):
            token = obs[t].tokens[gi]
            if (obs.red_light_token in token) or (token in collided):
                continue
            multi = bool(scorer._ego_areas[idx, t, EgoAreaIndex.MULTIPLE_LANES])
            nond = bool(scorer._ego_areas[idx, t, EgoAreaIndex.NON_DRIVABLE_AREA])
            to = obs.unique_objects[token]
            ctype = get_collision_type(scorer._states[idx, t], poly, to, obs[t][token])
            front = ctype in (CollisionType.ACTIVE_FRONT_COLLISION, CollisionType.STOPPED_TRACK_COLLISION)
            lat = ctype == CollisionType.ACTIVE_LATERAL_COLLISION
            at_fault = bool(front or ((multi or nond) and lat))
            st = scorer._states[idx, t]
            c = obs[t][token].centroid
            h = st[StateIndex.HEADING]
            dx, dy = c.x - st[StateIndex.X], c.y - st[StateIndex.Y]
            vel = getattr(to, "velocity", None)
            ev.append({"t_idx": t, "ctype": ctype.name, "obj_type": to.tracked_object_type.name,
                       "obj_speed": float(math.hypot(vel.x, vel.y)) if vel is not None else float("nan"),
                       "ego_speed": float(math.hypot(st[StateIndex.VELOCITY_X], st[StateIndex.VELOCITY_Y])),
                       "rel_lon": float(math.cos(-h) * dx - math.sin(-h) * dy), "rel_lat": float(math.sin(-h) * dx + math.cos(-h) * dy),
                       "at_fault": at_fault, "ego_multi_or_nondrivable": bool(multi or nond), "agent_type": to.tracked_object_type in AGENT_TYPES})
            if not at_fault:
                collided.append(token)
    return ev


def map_only(scorer, simulator, mc, plan, ps):
    ini = mc.ego_state
    tr = Trajectory(poses=np.asarray(plan, dtype=np.float32), trajectory_sampling=SAMP)
    arr = get_trajectory_as_array(transform_trajectory(tr, ini), ps, ini.time_point)
    sim = simulator.simulate_proposals(arr[None], ini)
    scorer.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
    return float(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 0]), float(scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hooks", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    toks = [l.strip() for l in open(a.tokens) if l.strip()]
    toks = [t for i, t in enumerate(toks) if i % n == k]
    if a.limit:
        toks = toks[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["token"])
            except Exception:                                       # noqa: BLE001
                pass
    hooks = {c["token"]: c for c in json.load(open(a.hooks, encoding="utf-8"))["pdm_score_calls"]}
    ps = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(ps)
    scorer = PDMScorer(ps, PDMScorerConfig())
    # token -> log (the metric cache is keyed by log): from the navtest inputs export
    import gzip
    inp = json.load(gzip.open("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz", "rt", encoding="utf-8"))["tokens"]
    log_of = {t: r["log_name"] for t, r in inp.items()}
    del inp
    cnt = 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for tok in toks:
            if tok in done:
                continue
            t0 = time.time()
            h = hooks[tok]
            rec = {"token": tok}
            try:
                with lzma.open(f"{CACHE}/{log_of[tok]}/unknown/{tok}/metric_cache.pkl", "rb") as f:
                    mc = pickle.load(f)
                plan = np.asarray(h["agent_poses"], dtype=np.float64)
                res = pdm_score(mc, Trajectory(poses=plan.astype(np.float32), trajectory_sampling=SAMP), ps, simulator, scorer)
                row = {kk: float(getattr(res, kk)) for kk in KEYS}
                banked = h["row"]
                rec["repro_max_abs"] = max(abs(row[kk] - float(banked[kk])) for kk in KEYS)
                ea = scorer._ego_areas
                nd = ea[1, :, EgoAreaIndex.NON_DRIVABLE_AREA]
                rec["nd_first"] = int(np.argmax(nd)) if nd.any() else -1
                rec["ref_dac"] = float(1.0 - ea[0, :, EgoAreaIndex.NON_DRIVABLE_AREA].any())
                rec["nc_events"] = nc_events(scorer, 1)
                ct = scorer._collision_time_idcs[1]
                rec["nc_first_at_fault_t"] = int(ct) if np.isfinite(ct) else -1
                v = {}
                for s in U.SPEED_SCALES:
                    pl = U.scale_speed(plan, s)
                    r2 = pdm_score(mc, Trajectory(poses=pl.astype(np.float32), trajectory_sampling=SAMP), ps, simulator, scorer)
                    v[f"S{s}"] = {kk: float(getattr(r2, kk)) for kk in KEYS}
                    v[f"S{s}"]["end_dist"] = float(np.hypot(*pl[-1, :2]))
                rec["speed"] = v
                sn = {"ID": map_only(scorer, simulator, mc, plan, ps)}
                for name, b in U.SNAP_BOUNDS:
                    sn[name] = map_only(scorer, simulator, mc, U.snap_plan(mc, plan, "bound", b), ps)
                sn["FULL"] = map_only(scorer, simulator, mc, U.snap_plan(mc, plan, "full"), ps)
                for name, s in U.V_VARIANTS:
                    sn[name] = map_only(scorer, simulator, mc, U.scale_speed(plan, s), ps)
                rec["snap"] = {kk: {"dac": vv[0], "ddc": vv[1]} for kk, vv in sn.items()}
                rec["status"] = "OK"
            except Exception as e:                                  # noqa: BLE001
                rec["status"] = "ERR"
                rec["err"] = repr(e)[:300]
            rec["wall_s"] = round(time.time() - t0, 3)
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            cnt += 1
    print("DONE", cnt, "->", a.out)


if __name__ == "__main__":
    main()
