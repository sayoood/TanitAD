#!/usr/bin/env python3
"""TF-N2 probe (navtrain held-out ONLY, never navtest): EP mechanism + label-side lever arms, one row per token.

Population: the 1,705 navtrain held-out teacher rollouts of TF-2 (1,018, `p5s_cubic.npz` col 0) and TF-2b (687,
`tf2b/rows/roll_s*.jsonl`). Scorer: W-TRANSP's own functions (`eval/instruments/transp_score.py`, imported, unchanged:
`transparent_states`, `human_rel`, `world_to_ego8`, `lead_track`) on the nuPlan-DB rebuild of the metric-cache parts (the
navtrain route; no harness cache exists for navtrain, see `../n1/a1/WTRANSP_DIAG.md`). The row format is W-TRANSP's
(`arms.<name>.{tracked,transparent,fam}`, `shared`), so `families_run.py` reads it unchanged.

Added per arm: `raw` (the plan's raw centerline progress x mult, metres) and per token `mech` (PDM-Closed raw progress
raw0, its 4 s path and speeds, red light on the route, the teacher's own 4 s path and end speed).

Arms (all from the teacher's own 20 poses `t20`; none reads a harness score, none reads the log future):
  T        t20 as is (to_navsim 8 poses)                                      control (N1-T form)
  C        Tc20s = iterate20(t20, cubic)                                      baseline (N1-C form)
  R        P5-R lambda* = 3.0, warm start Tc20s (p5r_labels.solve)            lever: regularised inversion
  W105     Tc20s of the progressively time-warped path, tau(t) = t (1 + 0.05 t / 4)      lever: speed-profile correction
  W110     same, 0.10                                                          dose-response
  human    the DB's logged future (W-TRANSP --human-arm form)                  reference
  RW105    (--extra) P5-R of the W105 path, warm start = its Tc20s     lever: the two label levers combined
  Wreg     Tc20s of the warp with factor -0.30 (a deliberately SLOWED plan)    deliberate regression: EP must fall
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
TF = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity"
P5R = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5r"
LAM = 3.0                     # P5-R lambda* (RESULT_P5R.md)
WARPS = {"W105": 0.05, "W110": 0.10, "Wreg": -0.30}


def load_keys(w):
    z = np.load(os.path.join(w, "p5s", "p5s_cubic.npz"))
    t20 = {}
    for i, (lg, tok) in enumerate(zip(z["logs"].tolist(), z["tokens"].tolist())):
        t20[(lg, tok)] = (np.asarray(z["labels20_t20_tc20_tw"][i, 0], np.float64),
                          np.asarray(z["labels20_t20_tc20_tw"][i, 1], np.float64), "tf2")
    for s in (0, 1):
        for ln in open(os.path.join(TF, "tf2b", "rows", f"roll_s{s}.jsonl"), encoding="utf-8"):
            r = json.loads(ln)
            if "traj20" in r:
                t20[(r["log"], r["token"])] = (np.asarray(r["traj20"], np.float64), None, "tf2b")
    return t20


def warp(t20, k):
    """Progressive time warp along the teacher's OWN path: pose at t := teacher(tau(t)), tau(t) = t (1 + k t / 4).
    tau'(0) = 1 (the t0 speed is kept), tau(4) = 4 (1 + k). Beyond 4 s: constant velocity / yaw rate of the last 0.2 s."""
    t_src = np.concatenate([[0.0], 0.2 * np.arange(1, 21)])
    p = np.vstack([np.zeros((1, 3)), t20])
    p[:, 2] = np.unwrap(p[:, 2])
    t_new = 0.2 * np.arange(1, 21)
    tau = t_new * (1.0 + k * t_new / 4.0)
    out = np.zeros((20, 3))
    vel = (p[-1] - p[-2]) / 0.2
    for i, ta in enumerate(tau):
        if ta <= 4.0:
            for c in range(3):
                out[i, c] = np.interp(ta, t_src, p[:, c])
        else:
            out[i] = p[-1] + (ta - 4.0) * vel
    return out


def score_arm(TS, parts, ego, s0, plan20, plan8=None):
    """W-TRANSP score_one_arm semantics (tracked = to_navsim(plan20) or plan8, transparent = plan20 at 0.2 s) + raw progress."""
    import navsim_dac as ND
    import navsim_pdm_targets as PT
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    S = PT._setup()
    pdm_traj, _, obs, centerline, lanes, dam = parts
    tr8 = plan8 if plan8 is not None else ND.to_navsim(plan20)
    tst = TS.transparent_states(plan20, ego, s0, 0.2)
    inp = [get_trajectory_as_array(pdm_traj, S["samp"], ego.time_point)]
    it = transform_trajectory(Trajectory(np.asarray(tr8, np.float32), S["samp8"]), ego)
    inp.append(get_trajectory_as_array(it, S["samp"], ego.time_point))
    sim = S["sim"].simulate_proposals(np.stack(inp, 0), ego)
    sim_tr = sim[1].copy()
    sim = np.concatenate([sim, tst[None]], 0)
    sc = S["scorer"]
    sc.score_proposals(sim, obs, centerline, lanes, dam)
    mult = sc._multi_metrics.prod(axis=0)
    raw = sc._progress_raw * mult
    ttc = np.asarray(sc._weighted_metrics[WeightedMetricIndex.TTC])
    com = np.asarray(sc._weighted_metrics[WeightedMetricIndex.COMFORTABLE])
    ddc = np.asarray(sc._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION])
    res = []
    for j in (1, 2):
        m = max(raw[0], raw[j])
        ep = raw[j] / m if m > 5.0 else (1.0 if mult[j] != 0 else 0.0)
        res.append({"nc": float(sc._multi_metrics[MultiMetricIndex.NO_COLLISION][j]),
                    "dac": float(sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA][j]),
                    "ddc": float(ddc[j]), "mult": float(mult[j]), "ep": float(ep), "ttc": float(ttc[j]),
                    "c": float(com[j]), "pdms": float(mult[j] * (5 * ep + 5 * ttc[j] + 2 * com[j]) / 12.0),
                    "raw": float(sc._progress_raw[j]), "raw0": float(sc._progress_raw[0]), "mult0": float(mult[0])})
    fam = {"reference8": np.asarray(tr8, np.float64).tolist(), "exec_tracked8": TS.world_to_ego8(sim_tr, ego).tolist(),
           "exec_transparent8": TS.world_to_ego8(tst, ego).tolist()}
    return {"tracked": res[0], "transparent": res[1], "fam": fam}, sim


def mech(sc, parts, sim, t20):
    """Per-token EP mechanism fields (no arm score is used): PDM-Closed's own 4 s motion and the red-light state."""
    pdm_traj, ego, obs, centerline, lanes, dam = parts
    st = sim[0]                                   # PDM-Closed simulated states [41, 11]: x, y, heading, vx, vy, ...
    d = np.linalg.norm(np.diff(st[:, :2], axis=0), axis=1)
    sp = np.hypot(st[:, 3], st[:, 4])
    red = 0
    try:
        from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
        lane_ids = set(str(x) for x in lanes)
        for tl in sc.get_traffic_light_status_at_iteration(0):
            if tl.status == TrafficLightStatusType.RED and str(tl.lane_connector_id) in lane_ids:
                red += 1
    except Exception:                                                            # noqa: BLE001
        red = -1
    tp = np.vstack([np.zeros((1, 2)), t20[:, :2]])
    tv = np.linalg.norm(np.diff(tp, axis=0), axis=1) / 0.2
    return {"pdm_path4_m": float(d.sum()), "pdm_v0": float(sp[0]), "pdm_v2": float(sp[20]), "pdm_v4": float(sp[40]),
            "pdm_amax": float(np.max(np.diff(sp) / 0.1)), "red_on_route": red,
            "t_path4_m": float(np.linalg.norm(np.diff(tp, axis=0), axis=1).sum()), "t_v4": float(tv[-1]),
            "t_vmax": float(tv.max()), "t_amax": float(np.max(np.diff(np.concatenate([[float(ego.dynamic_car_state.speed)], tv])) / 0.2))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True, help="dir holding refe/ eval/ p5s/ (export of the tip)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--sample", type=int, default=0, help="random sample of n keys (seed 20261112); 0 = all")
    ap.add_argument("--exclude", default="", help="jsonl whose tokens are skipped (e.g. an earlier sample)")
    ap.add_argument("--max-s", type=float, default=0.0, help="stop cleanly after this many seconds")
    ap.add_argument("--extra", action="store_true", help="only the combined arm RW105 = P5-R of the W105 path (warm start W105)")
    a = ap.parse_args()
    for p in (os.path.join(a.w, "refe"), os.path.join(a.w, "eval"), os.path.join(a.w, "eval", "instruments"), P5R):
        sys.path.insert(0, p)
    os.environ["P5C_PKG"] = a.w
    import navsim_dac as ND
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    import transp_score as TS
    import p5r_labels as R
    from p4_tracker_comp import tracked8
    from p5_target_labels import iterate20
    from navsim_pdm.pdm_array_representation import ego_state_to_state_array
    t20s = load_keys(a.w)
    keys = sorted(t20s)
    if a.sample:
        keys = sorted(random.Random(20261112).sample(keys, a.sample))
    skip = set()
    for fp in [x for x in a.exclude.split(",") if x]:
        for ln in open(fp, encoding="utf-8"):
            try:
                skip.add(json.loads(ln)["token"])
            except Exception:                                                    # noqa: BLE001
                pass
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                done.add(r["token"])
    si, sn = (int(x) for x in a.shard.split("/"))
    todo = [k for k in keys if k[1] not in done and k[1] not in skip and int(hashlib.sha1(k[1].encode()).hexdigest(), 16) % sn == si]
    by = {}
    for lg, tok in todo:
        by.setdefault(lg, []).append(tok)
    dbs = NS.index_dbs(db_root=DB_ROOT)
    t0, n = time.time(), 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg in sorted(by):
            if a.max_s and time.time() - t0 > a.max_s:
                break
            if not os.path.exists(dbs.get(lg, "")):
                for t in by[lg]:
                    fo.write(json.dumps({"log": lg, "token": t, "error": "no local DB"}) + "\n")
                continue
            for sc in NS.build_scenarios_for_log(dbs[lg], by[lg]):
                if a.max_s and time.time() - t0 > a.max_s:
                    break
                tok = sc._initial_lidar_token
                t20, tc_bank, src = t20s[(lg, tok)]
                rec = {"log": lg, "token": tok, "src": src, "arms": {}}
                ts = time.time()
                try:
                    parts = PT.metric_cache_parts(sc)
                    ego = parts[1]
                    s0 = ego_state_to_state_array(ego)
                    ctl = {"knot_exact": True, "knot_checks": 0}
                    tc = iterate20(ND, t20, ND.to_navsim(t20), ego, ctl, None, "cubic")[0]
                    rec["cderiv_m"] = None if tc_bank is None else float(np.abs(tc - tc_bank).max())
                    track = lambda Pb: tracked8(ND, Pb, ego)                    # noqa: E731
                    if a.extra:
                        wp = warp(t20, WARPS["W105"])
                        w105 = iterate20(ND, wp, ND.to_navsim(wp), ego, ctl, None, "cubic")[0]
                        arms = {"RW105": R.solve(wp, w105 - wp, [LAM], track, ND.to_navsim)[LAM]["label"]}
                    else:
                        rl = R.solve(t20, tc - t20, [LAM], track, ND.to_navsim)[LAM]["label"]
                        arms = {"T": t20, "C": tc, "R": rl}
                    for nm, k in ({} if a.extra else WARPS).items():
                        wp = warp(t20, k)
                        arms[nm] = iterate20(ND, wp, ND.to_navsim(wp), ego, ctl, None, "cubic")[0]
                    hum = TS.human_rel(sc, ego)
                    sim_c = None
                    for nm, p20 in arms.items():
                        rec["arms"][nm], sim = score_arm(TS, parts, ego, s0, p20)
                        if nm == "C":
                            sim_c = sim
                    if a.extra:
                        rec["t_s"] = round(time.time() - ts, 3)
                        fo.write(json.dumps(rec) + "\n")
                        fo.flush()
                        n += 1
                        continue
                    rec["arms"]["human"], _ = score_arm(TS, parts, ego, s0, hum[1::2][:20].copy(), hum[4::5][:8].copy())
                    v0 = ego.dynamic_car_state.rear_axle_velocity_2d
                    rec["shared"] = {"v0": float(math.hypot(v0.x, v0.y)), "human8": hum[4::5][:8].tolist(),
                                     "lead": TS.lead_track(sc, ego)}
                    rec["mech"] = mech(sc, parts, sim_c, t20)
                    rec["t_s"] = round(time.time() - ts, 3)
                except Exception as e:                                           # noqa: BLE001
                    rec = {"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n += 1
    print("ZZTFN2_PROBE_DONE", a.shard, n, round(time.time() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
