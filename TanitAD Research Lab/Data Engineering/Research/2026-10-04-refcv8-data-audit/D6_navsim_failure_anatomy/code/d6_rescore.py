"""D6 -- re-score banked refcv7 NavSim plans against the LOCAL metric cache and keep what the
official CSV throws away: WHEN the drivable-area test first fails, whether it fails at t=0 (initial
state), what the at-fault collision was (type / object / speeds), the human + PDM-closed reference
trajectories in the ego frame (the stage-2 hooks bank no human poses), and the plan's lateral offset
from the route centerline.   NAVSIM VENV (C:/Users/Admin/navsim-crun/venv), CPU only, one token at a time.

    PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit" \
    OMP_NUM_THREADS=2 C:/Users/Admin/navsim-crun/venv/Scripts/python.exe d6_rescore.py \
        --hooks <..._hooks.json> --tokens <tokens.txt> --out <out.jsonl> [--human] [--limit N]

The scoring path is the devkit's own ``navsim.evaluate.pdm_score.pdm_score`` with the SAME objects the
official run instantiated from the hydra config (``PDMSimulator``, ``PDMScorer`` with
``human_penalty_filter=True``, ``NavsimIDMTrafficAgents`` with the yaml's IDM parameters), so the
re-scored sub-scores MUST equal the banked ones -- that equality is the control (``repro`` field).
Resumable: tokens already in --out are skipped.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import lzma
import pickle
from pathlib import Path

ENV_DEFAULTS = {
    "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
    "NUPLAN_MAPS_ROOT": "C:/Users/Admin/navsim/data/maps",
    "NAVSIM_EXP_ROOT": "C:/Users/Admin/navsim/exp",
    "NAVSIM_DEVKIT_ROOT": "C:/Users/Admin/navsim/devkit",
    "OPENSCENE_DATA_ROOT": "C:/Users/Admin/navsim/data/openscene",
}
for k, v in ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)

import numpy as np
from shapely.geometry import Point

from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import (pdm_score, transform_trajectory, get_trajectory_as_array)
from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex, StateIndex
from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents

CACHE = "C:/Users/Admin/navsim-crun/exp/metric_cache_navhard_two_stage"
SUBS = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
        "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
        "lane_keeping", "history_comfort")


def build_objects():
    ps = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(ps)
    scorer = PDMScorer(ps, PDMScorerConfig(human_penalty_filter=True))
    # the yaml's idm_agents_observation block (navsim_IDM_traffic_agents.yaml), verbatim
    idm = NavsimIDMAgents(target_velocity=10, min_gap_to_lead_agent=1.0, headway_time=1.5, accel_max=1.0,
                          decel_max=2.0, open_loop_detections_types=[], minimum_path_length=20,
                          planned_trajectory_samples=None, planned_trajectory_sample_interval=None,
                          radius=100, add_open_loop_parked_vehicles=True, idm_snap_threshold=3.0)
    policy = NavsimIDMTrafficAgents(ps, idm)
    return ps, simulator, scorer, policy


def cache_path(log_name: str, token: str) -> str:
    return f"{CACHE}/{log_name}/unknown/{token}/metric_cache.pkl"


def to_ego(arr_xyh, x0, y0, h0):
    """global (x, y, heading) rows -> ego frame of (x0, y0, h0)."""
    a = np.asarray(arr_xyh, dtype=np.float64)
    dx, dy = a[:, 0] - x0, a[:, 1] - y0
    c, s = math.cos(-h0), math.sin(-h0)
    out = np.stack([c * dx - s * dy, s * dx + c * dy, (a[:, 2] - h0 + np.pi) % (2 * np.pi) - np.pi], axis=1)
    return out


def lateral_offsets(centerline, pts_xy):
    """signed lateral offset (left +) of points from the route centerline + centerline heading there."""
    ls = centerline.linestring
    res = []
    for x, y in pts_xy:
        p = Point(x, y)
        s = ls.project(p)
        q = ls.interpolate(s)
        q2 = ls.interpolate(min(s + 1.0, ls.length))
        q1 = ls.interpolate(max(s - 1.0, 0.0))
        tx, ty = q2.x - q1.x, q2.y - q1.y
        n = math.hypot(tx, ty) or 1.0
        tx, ty = tx / n, ty / n
        lat = (-ty) * (x - q.x) + tx * (y - q.y)
        res.append((float(lat), float(s), float(math.atan2(ty, tx))))
    return res


def score_states_only(scorer, simulator, metric_cache, traj_ego: Trajectory, ps):
    """DAC / DDC bits of ONE trajectory (no environment needed: both are map-only metrics)."""
    ini = metric_cache.ego_state
    tr = transform_trajectory(traj_ego, ini)
    arr = get_trajectory_as_array(tr, ps, ini.time_point)
    sim = simulator.simulate_proposals(arr[None, ...], ini)
    res = scorer.score_proposals(sim, metric_cache.observation, metric_cache.centerline,
                                 metric_cache.route_lane_ids, metric_cache.drivable_area_map,
                                 metric_cache.map_parameters, None, metric_cache.past_human_trajectory)[0]
    nd = scorer._ego_areas[0, :, EgoAreaIndex.NON_DRIVABLE_AREA].copy()
    onc = scorer._ego_areas[0, :, EgoAreaIndex.ONCOMING_TRAFFIC].copy()
    return {"dac": float(res["drivable_area_compliance"].iloc[0]),
            "ddc": float(res["driving_direction_compliance"].iloc[0]),
            "nd_first": int(np.argmax(nd)) if nd.any() else -1,
            "nd_n": int(nd.sum()), "nd_bits": "".join("1" if b else "0" for b in nd)}, sim[0]


def nc_detail(scorer, idx: int):
    """Re-run the devkit's at-fault-collision loop for ONE proposal and keep the events."""
    obs = scorer._observation
    collided = list(copy.deepcopy(obs.collided_track_ids))
    events = []
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
            front_or_stopped = ctype in (CollisionType.ACTIVE_FRONT_COLLISION, CollisionType.STOPPED_TRACK_COLLISION)
            lateral = ctype == CollisionType.ACTIVE_LATERAL_COLLISION
            at_fault = bool(front_or_stopped or ((multi or nond) and lateral))
            st = scorer._states[idx, t]
            c = obs[t][token].centroid
            h = st[StateIndex.HEADING]
            dx, dy = c.x - st[StateIndex.X], c.y - st[StateIndex.Y]
            lon = math.cos(-h) * dx - math.sin(-h) * dy
            lat = math.sin(-h) * dx + math.cos(-h) * dy
            vel = getattr(to, "velocity", None)
            ospd = float(math.hypot(vel.x, vel.y)) if vel is not None else float("nan")
            events.append({"t_idx": t, "ctype": ctype.name, "obj_type": to.tracked_object_type.name,
                           "obj_speed": ospd, "ego_speed": float(math.hypot(st[StateIndex.VELOCITY_X], st[StateIndex.VELOCITY_Y])),
                           "rel_lon": float(lon), "rel_lat": float(lat), "ego_multi_or_nondrivable": bool(multi or nond),
                           "ego_nondrivable": nond, "at_fault": at_fault, "agent_type": to.tracked_object_type in AGENT_TYPES})
            if at_fault:
                pass
            else:
                collided.append(token)
    return events


def global_from_ego(rows_xyh, x0, y0, h0):
    a = np.asarray(rows_xyh, dtype=np.float64)
    c, s = math.cos(h0), math.sin(h0)
    return [(x0 + c * r[0] - s * r[1], y0 + s * r[0] + c * r[1]) for r in a]


def geom_features(mc, ps, raw_plans: dict, with_human: bool = True):
    """Arm-independent route features + the lateral offset of every arm's RAW plan from the route
    centerline.  No scoring, no environment: ~0.3 s/token.  ``raw_plans`` = {arm_label: [8,3] ego-frame}."""
    ini = mc.ego_state
    x0, y0, h0 = ini.rear_axle.x, ini.rear_axle.y, ini.rear_axle.heading
    out = {"ego": {"speed": float(ini.dynamic_car_state.speed), "acc": float(ini.dynamic_car_state.acceleration)},
           "cl_len": float(mc.centerline.length)}
    ref_arr = get_trajectory_as_array(mc.trajectory, ps, ini.time_point)
    ref_ego = to_ego(ref_arr[:, [StateIndex.X, StateIndex.Y, StateIndex.HEADING]], x0, y0, h0)
    out["ref_xyh_0p5s"] = ref_ego[5::5].round(4).tolist()
    # NB: the cached PDM-closed InterpolatedTrajectory carries NO velocities (they read 0.0) -> finite-difference
    xy = ref_arr[:, [StateIndex.X, StateIndex.Y]]
    out["ref_speed_0p5s"] = [float(np.hypot(*(xy[i] - xy[i - 5])) / 0.5) for i in range(5, 41, 5)]
    out["cl_ref"] = lateral_offsets(mc.centerline, [(r[StateIndex.X], r[StateIndex.Y]) for r in ref_arr[5::5]])
    # the ROUTE itself, sampled at fixed arc-lengths ahead of the ego's projection (independent of any plan or of the
    # reference's speed): heading relative to the ego heading + ego-frame xy.  d = 0 is the projection point.
    ls = mc.centerline.linestring
    s0 = float(ls.project(Point(x0, y0)))
    out["s0"] = s0
    out["cl_remaining_m"] = float(ls.length - s0)
    out["lat0"] = lateral_offsets(mc.centerline, [(x0, y0)])[0][0]
    ahead = []
    dam = mc.drivable_area_map
    for d in (0, 5, 10, 15, 20, 25, 30, 40, 50, 60):
        sd = min(s0 + d, ls.length)
        q = ls.interpolate(sd)
        qa, qb = ls.interpolate(min(sd + 1.0, ls.length)), ls.interpolate(max(sd - 1.0, 0.0))
        hd = math.atan2(qa.y - qb.y, qa.x - qb.x)
        e = to_ego([[q.x, q.y, hd]], x0, y0, h0)[0]
        try:
            inter = bool(dam.is_in_layer(Point(q.x, q.y), SemanticMapLayer.INTERSECTION))
        except Exception:                                                    # noqa: BLE001
            inter = None
        ahead.append([d, float(e[0]), float(e[1]), float(e[2]), inter])
    out["cl_ahead"] = ahead      # rows [d_m, x_ego, y_ego, heading_rel_ego, centerline_point_in_INTERSECTION_polygon]
    try:
        out["ego_in_intersection"] = bool(dam.is_in_layer(Point(x0, y0), SemanticMapLayer.INTERSECTION))
    except Exception:                                                        # noqa: BLE001
        out["ego_in_intersection"] = None
    out["cl_plan_raw"] = {}
    for arm, pl in raw_plans.items():
        out["cl_plan_raw"][arm] = lateral_offsets(mc.centerline, global_from_ego(pl, x0, y0, h0))
    if with_human:
        try:
            hp = np.asarray(mc.human_trajectory.poses, dtype=np.float64)
            out["human_xyh"] = hp.round(4).tolist()
            out["cl_human"] = lateral_offsets(mc.centerline, global_from_ego(hp, x0, y0, h0))
        except Exception:                                                    # noqa: BLE001
            out["human_xyh"] = None
    return out, (x0, y0, h0)


class Snap:
    """Wrap ``scorer.score_proposals`` and snapshot the PLAN-proposal state right after the FIRST call per
    token: for ORIGINAL scenes ``pdm_score`` calls it a second time for the human filter, which would
    otherwise overwrite ``_ego_areas`` / mutate the observation."""

    def __init__(self, scorer):
        self.scorer, self.orig = scorer, scorer.score_proposals
        self.calls, self.snap = 0, None

    def reset(self):
        self.calls, self.snap = 0, None

    def __call__(self, *args, **kw):
        out = self.orig(*args, **kw)
        self.calls += 1
        if self.calls == 1:
            sc = self.scorer
            self.snap = {"ea": sc._ego_areas.copy(), "coll_t": sc._collision_time_idcs.copy(),
                         "nc": nc_detail(sc, 1)}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hooks", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--human", action="store_true", help="also score the human trajectory (DAC/DDC only)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--geom-only", action="store_true", help="no scoring: route features + raw-plan offsets only")
    ap.add_argument("--extra-hooks", default="", help="geom-only: label=path[,label=path] extra arms' hooks files")
    a = ap.parse_args()
    toks = [l.strip() for l in open(a.tokens) if l.strip()]
    if a.limit:
        toks = toks[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["token"])
            except Exception:
                pass
    hooks = {c["token"]: c for c in json.load(open(a.hooks, encoding="utf-8"))["pdm_score_calls"]}
    extra = {}
    for kv in [x for x in a.extra_hooks.split(",") if x]:
        lab, path = kv.split("=", 1)
        extra[lab] = {c["token"]: c for c in json.load(open(path, encoding="utf-8"))["pdm_score_calls"]}
    ps, simulator, scorer, policy = build_objects()
    snap = Snap(scorer)
    scorer.score_proposals = snap
    n = 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for tok in toks:
            if tok in done:
                continue
            t0 = time.time()
            h = hooks[tok]
            rec = {"token": tok, "log_name": h["log_name"], "scene_type": h["scene_type"]}
            try:
                with lzma.open(cache_path(h["log_name"], tok), "rb") as f:
                    mc = pickle.load(f)
                if a.geom_only:
                    plans = {"MAIN": np.asarray(h["agent_poses"], dtype=np.float64)}
                    for lab, hk in extra.items():
                        plans[lab] = np.asarray(hk[tok]["agent_poses"], dtype=np.float64)
                    g, _ = geom_features(mc, ps, plans)
                    rec.update(g)
                    rec["status"] = "OK_GEOM"
                    rec["wall_s"] = round(time.time() - t0, 3)
                    fo.write(json.dumps(rec) + "\n")
                    fo.flush()
                    n += 1
                    continue
                plan = np.asarray(h["agent_poses"], dtype=np.float32)
                traj = Trajectory(poses=plan, trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5))
                snap.reset()
                row, sim_states = pdm_score(mc, traj, ps, simulator, scorer, policy)
                resc = {k: float(row[k].iloc[0]) for k in SUBS}
                rec["resc"] = resc
                banked = h["row"]
                rec["repro"] = {k: abs(resc[k] - float(banked[k])) for k in SUBS}
                ini = mc.ego_state
                x0, y0, h0 = ini.rear_axle.x, ini.rear_axle.y, ini.rear_axle.heading
                rec["ego"] = {"speed": float(ini.dynamic_car_state.speed),
                              "acc": float(ini.dynamic_car_state.acceleration)}
                # per-timestep area bits of the PLAN proposal (idx 1; idx 0 is the PDM-closed reference)
                ea = snap.snap["ea"]
                for name, idx in (("nd", EgoAreaIndex.NON_DRIVABLE_AREA), ("onc", EgoAreaIndex.ONCOMING_TRAFFIC),
                                  ("mul", EgoAreaIndex.MULTIPLE_LANES)):
                    b = ea[1, :, idx]
                    rec[f"{name}_bits"] = "".join("1" if x else "0" for x in b)
                    rec[f"{name}_first"] = int(np.argmax(b)) if b.any() else -1
                    b0 = ea[0, :, idx]
                    rec[f"ref_{name}_first"] = int(np.argmax(b0)) if b0.any() else -1
                rec["ref_dac"] = float(1.0 - ea[0, :, EgoAreaIndex.NON_DRIVABLE_AREA].any())
                # NC events for the plan proposal (snapshotted inside the first score_proposals call)
                rec["nc_events"] = snap.snap["nc"]
                ct = snap.snap["coll_t"][1]
                rec["nc_first_at_fault_t"] = int(ct) if np.isfinite(ct) else -1
                rec["raw_dac"] = float(1.0 - ea[1, :, EgoAreaIndex.NON_DRIVABLE_AREA].any())
                # geometry in the ego frame
                sim_ego = to_ego(sim_states[:, [StateIndex.X, StateIndex.Y, StateIndex.HEADING]], x0, y0, h0)
                rec["sim_xyh_0p5s"] = sim_ego[5::5].round(4).tolist()          # 8 rows: 0.5..4.0 s
                rec["sim_speed_0p5s"] = [float(math.hypot(sim_states[i, StateIndex.VELOCITY_X], sim_states[i, StateIndex.VELOCITY_Y]))
                                         for i in range(5, 41, 5)]
                rec["plan_xyh"] = plan.round(4).tolist()
                ref_arr = get_trajectory_as_array(mc.trajectory, ps, ini.time_point)
                ref_ego = to_ego(ref_arr[:, [StateIndex.X, StateIndex.Y, StateIndex.HEADING]], x0, y0, h0)
                rec["ref_xyh_0p5s"] = ref_ego[5::5].round(4).tolist()
                hum = None
                try:
                    hp = np.asarray(mc.human_trajectory.poses, dtype=np.float64)
                    rec["human_xyh"] = hp.round(4).tolist()
                    hum = mc.human_trajectory
                except Exception as e:                                       # noqa: BLE001
                    rec["human_xyh"] = None
                    rec["human_err"] = repr(e)[:120]
                # lateral offsets from the route centerline (plan, human, reference)
                glob_plan = sim_ego  # ego frame; convert back to global for projection
                gl = []
                for r in sim_states[5::5]:
                    gl.append((r[StateIndex.X], r[StateIndex.Y]))
                rec["cl_len"] = float(mc.centerline.length)
                rec["cl_plan"] = lateral_offsets(mc.centerline, gl)
                gref = [(r[StateIndex.X], r[StateIndex.Y]) for r in ref_arr[5::5]]
                rec["cl_ref"] = lateral_offsets(mc.centerline, gref)
                if hum is not None:
                    htr = transform_trajectory(hum, ini)
                    harr = get_trajectory_as_array(htr, ps, ini.time_point)
                    ghum = [(r[StateIndex.X], r[StateIndex.Y]) for r in harr[5::5]]
                    rec["cl_human"] = lateral_offsets(mc.centerline, ghum)
                    if a.human:
                        hres, _ = score_states_only(scorer, simulator, mc, hum, ps)
                        rec["human_score"] = hres
                g, _ = geom_features(mc, ps, {"MAIN": np.asarray(h["agent_poses"], dtype=np.float64)})
                rec["geom"] = {k: v for k, v in g.items() if k in ("cl_plan_raw", "ref_speed_0p5s")}
                rec["status"] = "OK"
            except Exception as e:                                           # noqa: BLE001
                rec["status"] = "ERR"
                rec["err"] = repr(e)[:300]
            rec["wall_s"] = round(time.time() - t0, 3)
            fo.write(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
            fo.flush()
            n += 1
    print("DONE", n, "new tokens ->", a.out)


if __name__ == "__main__":
    main()
