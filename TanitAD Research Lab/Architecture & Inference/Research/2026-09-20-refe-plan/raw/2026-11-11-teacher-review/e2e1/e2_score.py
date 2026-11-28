#!/usr/bin/env python3
"""E2 (teacher review s6): tracker-transparent scoring of the 1,018 banked held-out navtrain teacher targets.

Per (log, token) of `d1_navtrain.json` (the 1,018 with a banked 10 Hz rank-0 target), through NAVSIM v1.1's OWN
PDMScorer (NC, DAC, TTC, EP, comfort) on four arms, scored in ONE scorer call against PDM-Closed (EP normaliser):
  teacher_tracked      the target's 20 poses -> ND.to_navsim (8 poses, NO A7 repair: identical to p4_d1_navtrain) -> LQR + bicycle (sim8)
  teacher_transparent  the target's 20 poses treated as EXECUTED: cubic spline through (t=0 origin, 20 poses at 0.2 s) -> 41 states
                       at 0.1 s with analytic velocity/acceleration/yaw rate; NO LQR
  human_tracked        the logged ego future, 8 poses at 0.5 s -> LQR + bicycle (NAVSIM's standard human row)
  human_transparent    the logged ego future sampled at 0.2 s (20 poses) -> the SAME transparent builder as the teacher
Usage: python e2_score.py --refe <dir> --d1 <d1_navtrain.json> --out <jsonl> --shard i/N [--limit n]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

BANK = "E:/Projects/TanitAD/data/refe_navtrain10"
DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
WHEELBASE = 3.089


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def transparent_states(rel20: np.ndarray, ego, state0: np.ndarray) -> np.ndarray:
    """20 rear-axle poses (ego frame, t = 0.2..4.0) treated as executed -> [41, 11] PDM state array (world frame).
    Cubic spline per coordinate through t = 0 (origin) + the 20 poses; first-derivative clamp at t = 0 to the logged
    initial velocity / yaw rate, at t = 4 to the last finite difference. Velocity/acceleration/yaw terms are the spline's
    analytic derivatives, rotated into the body frame (nuPlan's rear-axle convention). No tracker."""
    from scipy.interpolate import CubicSpline
    p = np.asarray(rel20, np.float64)
    t_k = np.arange(21) * 0.2
    pts = np.vstack([np.zeros((1, 3)), p])
    pts[:, 2] = np.unwrap(pts[:, 2])
    v0 = ego.dynamic_car_state.rear_axle_velocity_2d
    w0 = ego.dynamic_car_state.angular_velocity
    d0 = np.array([v0.x, v0.y, w0])
    d1 = (pts[-1] - pts[-2]) / 0.2
    t_q = np.arange(41) * 0.1
    pos, vel, acc = np.zeros((41, 3)), np.zeros((41, 3)), np.zeros((41, 3))
    for c in range(3):
        cs = CubicSpline(t_k, pts[:, c], bc_type=((1, d0[c]), (1, d1[c])))
        pos[:, c], vel[:, c], acc[:, c] = cs(t_q), cs(t_q, 1), cs(t_q, 2)
    ra = ego.rear_axle
    ch, sh = math.cos(ra.heading), math.sin(ra.heading)
    out = np.zeros((41, 11))
    out[:, 0] = ra.x + pos[:, 0] * ch - pos[:, 1] * sh
    out[:, 1] = ra.y + pos[:, 0] * sh + pos[:, 1] * ch
    out[:, 2] = ra.heading + pos[:, 2]
    c, s = np.cos(pos[:, 2]), np.sin(pos[:, 2])
    out[:, 3] = c * vel[:, 0] + s * vel[:, 1]               # body-frame (rear axle) velocity
    out[:, 4] = -s * vel[:, 0] + c * vel[:, 1]
    out[:, 5] = c * acc[:, 0] + s * acc[:, 1]
    out[:, 6] = -s * acc[:, 0] + c * acc[:, 1]
    spd = np.maximum(np.hypot(out[:, 3], out[:, 4]), 0.1)
    out[:, 7] = np.arctan(WHEELBASE * vel[:, 2] / spd)       # tire steering angle from yaw rate
    out[:, 8] = np.gradient(out[:, 7], 0.1)
    out[:, 9] = vel[:, 2]
    out[:, 10] = acc[:, 2]
    out[0] = state0                                          # t = 0 is the logged ego state
    return out


def human_rel(scenario, ego, n_states: int = 40):
    """Logged ego future at 0.1 s (n=40) as rear-axle poses in the ego frame [40, 3]."""
    fut = scenario.get_ego_future_trajectory(0, 4.0, n_states)
    ra = ego.rear_axle
    c, s = math.cos(-ra.heading), math.sin(-ra.heading)
    out = []
    for st in fut:
        q = st.rear_axle
        dx, dy = q.x - ra.x, q.y - ra.y
        out.append((dx * c - dy * s, dx * s + dy * c, wrap(q.heading - ra.heading)))
    return np.asarray(out)


def score_states(parts, states_tracked_in, extra_states):
    """NAVSIM scorer on [PDM-Closed, *tracked arms (simulated), *transparent arms (given as executed)]."""
    import navsim_pdm_targets as PT
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    S = PT._setup()
    pdm_traj, ego, obs, centerline, lanes, dam = parts
    inp = [get_trajectory_as_array(pdm_traj, S["samp"], ego.time_point)]
    for p8 in states_tracked_in:
        it = transform_trajectory(Trajectory(np.asarray(p8, np.float32), S["samp8"]), ego)
        inp.append(get_trajectory_as_array(it, S["samp"], ego.time_point))
    sim = S["sim"].simulate_proposals(np.stack(inp, 0), ego)
    if extra_states:
        sim = np.concatenate([sim, np.stack(extra_states, 0)], 0)
    sc = S["scorer"]
    sc.score_proposals(sim, obs, centerline, lanes, dam)
    mult = sc._multi_metrics.prod(axis=0)
    raw = sc._progress_raw * mult
    ttc = np.asarray(sc._weighted_metrics[WeightedMetricIndex.TTC])
    com = np.asarray(sc._weighted_metrics[WeightedMetricIndex.COMFORTABLE])
    res = []
    for j in range(1, sim.shape[0]):
        m = max(raw[0], raw[j])
        ep = raw[j] / m if m > 5.0 else (1.0 if mult[j] != 0 else 0.0)
        pdms = mult[j] * (5 * ep + 5 * ttc[j] + 2 * com[j]) / 12.0
        res.append({"nc": float(sc._multi_metrics[MultiMetricIndex.NO_COLLISION][j]),
                    "dac": float(sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA][j]),
                    "mult": float(mult[j]),
                    "ep": float(ep), "ttc": float(ttc[j]), "c": float(com[j]), "pdms": float(pdms)})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe", required=True)
    ap.add_argument("--d1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.refe)
    import navsim_dac as ND
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    from navsim_pdm.pdm_array_representation import ego_state_to_state_array
    si, sn = (int(x) for x in a.shard.split("/"))
    d1 = json.load(open(a.d1, encoding="utf-8"))["rows"]
    keys = sorted({(r["log"], r["token"]) for r in d1})
    bank = {}
    want = set(keys)
    for fp in glob.glob(f"{BANK}/**/targets_rank0.jsonl", recursive=True):
        for ln in open(fp, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            k = (r.get("log_name"), r.get("token"))
            if k in want and int(r.get("sim_hz", 0)) == 10 and int(r.get("step", 0)) == 0:
                bank[k] = r["traj"]
    by = {}
    for k in keys:
        if k in bank:
            by.setdefault(k[0], []).append(k[1])
    logs = sorted(by)[si::sn]
    dbs = NS.index_dbs(db_root=DB_ROOT)
    n_done = 0
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg in logs:
            toks = by[lg]
            if not os.path.exists(dbs.get(lg, "")):
                for t in toks:
                    fo.write(json.dumps({"log": lg, "token": t, "error": "no local DB"}) + "\n")
                continue
            for sc in NS.build_scenarios_for_log(dbs[lg], toks):
                tok = sc._initial_lidar_token
                rec = {"log": lg, "token": tok}
                try:
                    parts = PT.metric_cache_parts(sc)
                    ego = parts[1]
                    s0 = ego_state_to_state_array(ego)
                    t20 = np.asarray(bank[(lg, tok)], np.float64)
                    hum = human_rel(sc, ego)                              # [40, 3] at 0.1 s
                    h8 = hum[4::5][:8]                                    # 0.5 s .. 4.0 s
                    h20 = hum[1::2][:20]                                  # 0.2 s .. 4.0 s
                    res = score_states(parts, [ND.to_navsim(t20), h8],
                                       [transparent_states(t20, ego, s0), transparent_states(h20, ego, s0)])
                    for nm, r in zip(("teacher_tracked", "human_tracked", "teacher_transparent", "human_transparent"), res):
                        rec[nm] = r
                except Exception as e:                                    # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n_done += 1
                if a.limit and n_done >= a.limit:
                    return 0
    print("ZZE2_SHARD_DONE", a.shard, n_done, round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
