#!/usr/bin/env python3
"""W-TRANSP + W-FAM row producer: navtest-scale, sharded, resumable, CPU-only PDMS scorer with a tracker-transparent arm.

For every token, and for EVERY arm's submitted plan (one or more --plans files), it makes one NAVSIM-v1.1 PDMScorer call (the vendored
simulator and scorer, as E2 / TF-1 do) on  [PDM-Closed (EP normaliser), plan LQR-TRACKED, plan TRANSPARENT (executed as given, no
tracker)]  and writes ONE JSON row per token: {"log","token","t_s","shared":{v0,human8,lead},"arms":{name:{tracked,transparent,fam,..}}}.
`fam` + `shared` are the payload that families_run.py turns into the four metric families.

Plan file (--plans), npz:
    tokens   [N]            str   NAVSIM token
    logs     [N]            str   optional (else --token-log is used)
    plans20  [N,20,3]       float 20 rear-axle poses (x,y,heading), ego frame at t0, t = 0.2 .. 4.0 s   (the P5-C / TF-N1 form)
    plans8   [N,8,3]        float optional: 8 poses at 0.5 s, ego frame. If present it is what the TRACKED arm is handed
                                  (the "seam" form); the transparent arm is then built from plans8 (dt 0.5) unless plans20 exists.
  At least one of plans20 / plans8 is required. jsonl is also accepted: {"token","log","plan20"|"plan8"}.
  The tracked arm for a plans20-only file is `navsim_dac.to_navsim(plans20)` (E2 / TF-1 conversion), after the optional A7 repair
  (`--a7 1`: last heading := previous pose's heading; DEFAULT 0 = score the plan exactly as given).

Usage (one shard per process; at most 2 processes):
  PYTHONPATH=<navsim devkit 3e8291b>;<refe dir>  python transp_score.py --refe <refe dir> --plans [name=]<plans.npz> [...] --out <rows.jsonl> \
      --shard i/N [--split navtrain|navtest] [--token-log <navtest_inputs.json.gz>] [--controls] [--limit n] [--metric-cache-root <metric_cache_navtest>]
Resumable: rows already in --out without an "error" are skipped; failures are re-attempted. Never reads a navtest token unless
the plan file names it.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
WHEELBASE = 3.089
LEAD_LAT_M = 2.0          # = taniteval lead_metrics.LEAD_LAT_M
LEAD_MAX_GAP_M = 80.0     # = taniteval lead_source.LEAD_MAX_GAP_M
SPLIT_ROWS = {"navtrain": (None, None), "navtest": (1, 80)}   # (history_rows, future_rows): navtest as p4_tracker_comp.py:99


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def transparent_states(rel: np.ndarray, ego, state0: np.ndarray, dt: float = 0.2) -> np.ndarray:
    """E2's builder (`e2_score.py:49-84`), the arithmetic UNCHANGED, with the pose spacing `dt` made a parameter (0.2 = E2).

    n rear-axle poses (ego frame, t = dt..n*dt) treated as EXECUTED -> [41, 11] PDM state array (world frame, 0.1 s).
    Cubic spline per coordinate through t=0 (origin) + the n poses; first-derivative clamp at t=0 to the logged initial
    velocity / yaw rate, at t=n*dt to the last finite difference. Velocity / acceleration / yaw terms are the spline's analytic
    derivatives rotated into the body frame. No tracker. Row 0 is the logged ego state."""
    from scipy.interpolate import CubicSpline
    p = np.asarray(rel, np.float64)
    n = p.shape[0]
    t_k = np.arange(n + 1) * dt
    pts = np.vstack([np.zeros((1, 3)), p])
    pts[:, 2] = np.unwrap(pts[:, 2])
    v0 = ego.dynamic_car_state.rear_axle_velocity_2d
    w0 = ego.dynamic_car_state.angular_velocity
    d0 = np.array([v0.x, v0.y, w0])
    d1 = (pts[-1] - pts[-2]) / dt
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
    out[:, 3] = c * vel[:, 0] + s * vel[:, 1]
    out[:, 4] = -s * vel[:, 0] + c * vel[:, 1]
    out[:, 5] = c * acc[:, 0] + s * acc[:, 1]
    out[:, 6] = -s * acc[:, 0] + c * acc[:, 1]
    spd = np.maximum(np.hypot(out[:, 3], out[:, 4]), 0.1)
    out[:, 7] = np.arctan(WHEELBASE * vel[:, 2] / spd)
    out[:, 8] = np.gradient(out[:, 7], 0.1)
    out[:, 9] = vel[:, 2]
    out[:, 10] = acc[:, 2]
    out[0] = state0
    return out


def human_rel(scenario, ego, n_states: int = 40):
    """E2's `human_rel` verbatim: logged ego future at 0.1 s (n=40) as rear-axle poses in the ego frame [40, 3]."""
    fut = scenario.get_ego_future_trajectory(0, 4.0, n_states)
    ra = ego.rear_axle
    c, s = math.cos(-ra.heading), math.sin(-ra.heading)
    out = []
    for st in fut:
        q = st.rear_axle
        dx, dy = q.x - ra.x, q.y - ra.y
        out.append((dx * c - dy * s, dx * s + dy * c, wrap(q.heading - ra.heading)))
    return np.asarray(out)


def world_to_ego8(states: np.ndarray, ego) -> np.ndarray:
    """[41, >=3] world-frame PDM states -> the 8 poses at 0.5 s in the t0 rear-axle frame [8, 3]."""
    s = np.asarray(states)[5::5][:8, :3]
    ra = ego.rear_axle
    c, sn = math.cos(-ra.heading), math.sin(-ra.heading)
    dx, dy = s[:, 0] - ra.x, s[:, 1] - ra.y
    return np.stack([dx * c - dy * sn, dx * sn + dy * c, wrap(s[:, 2] - ra.heading)], -1)


def lead_track(sc, ego, n: int = 8, dt: float = 0.5) -> dict:
    """The lead vehicle for distance-keeping, from the nuPlan log's tracked objects (GT detections), in the t0 rear-axle frame.

    Lead := nearest VEHICLE ahead of the ego at t0 with |lateral| < LEAD_LAT_M and rear-face gap in [0, LEAD_MAX_GAP_M]
    (the corridor/gap rule of taniteval lead_metrics / lead_source). Its track is the SAME track_token at t = dt..n*dt.
    ⚠ Origin is the REAR AXLE (NAVSIM's pose frame), gap := along - length/2 is rear-axle-to-lead-rear-face."""
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    step = float(sc.database_interval)
    its = [int(round(k * dt / step)) for k in range(n + 1)]
    ra = ego.rear_axle
    c, s = math.cos(-ra.heading), math.sin(-ra.heading)

    def to_ego(x, y):
        dx, dy = x - ra.x, y - ra.y
        return dx * c - dy * s, dx * s + dy * c

    def vehicles(it):
        return sc.get_tracked_objects_at_iteration(iteration=it).tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)

    best, best_gap = None, None
    for o in vehicles(its[0]):
        x, y = to_ego(o.center.x, o.center.y)
        gap = x - o.box.length / 2.0
        if gap >= 0.0 and abs(y) < LEAD_LAT_M and gap <= LEAD_MAX_GAP_M and (best is None or gap < best_gap):
            best, best_gap = o, gap
    if best is None:
        return {"state": "NO_LEAD", "leads": [[None, None]] * n, "len": None}
    tok = best.track_token
    pts = []
    for k in range(1, n + 1):
        hit = [o for o in vehicles(its[k]) if o.track_token == tok]
        pts.append(list(to_ego(hit[0].center.x, hit[0].center.y)) if hit else [None, None])
    return {"state": "LEAD", "leads": pts, "len": float(best.box.length), "gap0_m": float(best_gap)}


def score_arms(parts, tracked_plans8, executed_states):
    """E2/TF-1 `score` : [PDM-Closed, *tracked_plans8 (LQR-simulated), *executed_states (as given)] in ONE scorer call.
    Returns (per-arm sub-score dicts for arms 1.., simulated states of the tracked arms [T,41,11])."""
    import navsim_pdm_targets as PT
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    S = PT._setup()
    pdm_traj, ego, obs, centerline, lanes, dam = parts
    inp = [get_trajectory_as_array(pdm_traj, S["samp"], ego.time_point)]
    for p8 in tracked_plans8:
        it = transform_trajectory(Trajectory(np.asarray(p8, np.float32), S["samp8"]), ego)
        inp.append(get_trajectory_as_array(it, S["samp"], ego.time_point))
    sim = S["sim"].simulate_proposals(np.stack(inp, 0), ego)
    n_tr = len(tracked_plans8)
    sim_tracked = sim[1:1 + n_tr].copy()
    if executed_states:
        sim = np.concatenate([sim, np.stack(executed_states, 0)], 0)
    sc = S["scorer"]
    sc.score_proposals(sim, obs, centerline, lanes, dam)
    mult = sc._multi_metrics.prod(axis=0)
    raw = sc._progress_raw * mult
    ttc = np.asarray(sc._weighted_metrics[WeightedMetricIndex.TTC])
    com = np.asarray(sc._weighted_metrics[WeightedMetricIndex.COMFORTABLE])
    ddc = np.asarray(sc._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION])
    out = []
    for j in range(1, sim.shape[0]):
        m = max(raw[0], raw[j])
        ep = raw[j] / m if m > 5.0 else (1.0 if mult[j] != 0 else 0.0)
        pdms = mult[j] * (5 * ep + 5 * ttc[j] + 2 * com[j]) / 12.0
        out.append({"nc": float(sc._multi_metrics[MultiMetricIndex.NO_COLLISION][j]),
                    "dac": float(sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA][j]),
                    "ddc": float(ddc[j]), "mult": float(mult[j]), "ep": float(ep), "ttc": float(ttc[j]),
                    "c": float(com[j]), "pdms": float(pdms)})
    return out, sim_tracked


def load_plans(path: str) -> dict:
    """-> {token: {"log": str|None, "plan20": [20,3]|None, "plan8": [8,3]|None}}"""
    out = {}
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        toks = [str(t) for t in z["tokens"]]
        logs = [str(x) for x in z["logs"]] if "logs" in z.files else [None] * len(toks)
        p20 = z["plans20"] if "plans20" in z.files else None
        p8 = z["plans8"] if "plans8" in z.files else None
        if p20 is None and p8 is None:
            raise SystemExit("plans file needs plans20 and/or plans8")
        for i, t in enumerate(toks):
            out[t] = {"log": logs[i], "plan20": None if p20 is None else np.asarray(p20[i], np.float64),
                      "plan8": None if p8 is None else np.asarray(p8[i], np.float64)}
    else:
        for ln in open(path, encoding="utf-8"):
            if not ln.strip():
                continue
            r = json.loads(ln)
            out[str(r["token"])] = {"log": r.get("log"),
                                    "plan20": np.asarray(r["plan20"], np.float64) if r.get("plan20") is not None else None,
                                    "plan8": np.asarray(r["plan8"], np.float64) if r.get("plan8") is not None else None}
    return out


def load_token_log(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    d = json.load(op(path, "rt", encoding="utf-8"))
    d = d.get("tokens", d)
    return {t: (v["log_name"] if isinstance(v, dict) else v) for t, v in d.items()}


def done_tokens(out_path: str) -> set:
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" not in r and "token" in r:
                done.add(r["token"])
    return done


def plan_views(p: dict, a7: bool, shift_lat: float):
    """-> (plan8 for the tracked arm [8,3], (rel poses, dt) for the transparent builder, the plan20 used or None)"""
    import navsim_dac as ND
    p20, p8 = p["plan20"], p["plan8"]
    if p20 is not None:
        p20 = p20.copy()
        if a7:
            p20[-1, 2] = p20[-2, 2]
        if shift_lat:
            p20[:, 1] += shift_lat
    if p8 is not None:
        p8 = p8.copy()
        if a7:
            p8[-1, 2] = p8[-2, 2]
        if shift_lat:
            p8[:, 1] += shift_lat
    tracked8 = p8 if p8 is not None else ND.to_navsim(p20)
    transp = (p20, 0.2) if p20 is not None else (p8, 0.5)
    return tracked8, transp


def score_one_arm(parts, ego, s0, p, a7: bool, shift_lat: float, controls: bool, want_fam: bool) -> dict:
    """One arm's plan on one token -> {"tracked","transparent",[controls],[fam]}. Two scorer calls at most per arm; PDM-Closed and the
    metric-cache parts (the expensive part, ~1-2 s/token) are shared across arms by the caller."""
    tr8, (trel, tdt) = plan_views(p, a7, shift_lat)
    tstates = transparent_states(trel, ego, s0, tdt)
    res, sim_tr = score_arms(parts, [tr8], [tstates])
    out = {"tracked": res[0], "transparent": res[1]}
    if controls:
        # (iii-a) the LQR-simulated states fed into the executed slot must reproduce the tracked score;
        # (iii-b) the plan shifted 0.5 m left must change the scores (DAC on >= 1 token over the run).
        sh_p = dict(p)
        for k in ("plan20", "plan8"):
            if p[k] is not None:
                q = p[k].copy()
                q[:, 1] += 0.5
                sh_p[k] = q
        tr8s, (trels, tdts) = plan_views(sh_p, a7, shift_lat)
        cres, _ = score_arms(parts, [tr8s], [sim_tr[0], transparent_states(trels, ego, s0, tdts)])
        out["ctl_sim_as_executed"], out["ctl_shift05_tracked"], out["ctl_shift05_transparent"] = cres[1], cres[0], cres[2]
    if want_fam:
        out["fam"] = {"reference8": np.asarray(tr8, np.float64).tolist(),
                      "exec_tracked8": world_to_ego8(sim_tr[0], ego).tolist(),
                      "exec_transparent8": world_to_ego8(tstates, ego).tolist()}
    return out


def load_cache_parts(root: str, log: str, token: str):
    """(pdm_traj, ego, obs, centerline, route_lane_ids, drivable_area_map) exactly as navsim.evaluate.pdm_score reads a stored MetricCache.

    The harness's own file `<root>/<log>/unknown/<token>/metric_cache.pkl` (lzma pickle). A missing file RAISES, naming the path: there is
    deliberately NO fallback to the nuPlan-DB rebuild, because that rebuild is the not-harness-identical route (WTRANSP_DIAG.md)."""
    import lzma
    import pickle
    p = os.path.join(root, log, "unknown", token, "metric_cache.pkl")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"--metric-cache-root: no harness metric cache for token {token} at {p}")
    with lzma.open(p, "rb") as fh:
        mc = pickle.loads(fh.read())
    return (mc.trajectory, mc.ego_state, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refe", required=True, help="dir with navsim_dac.py, navtrain_scenarios.py, navsim_pdm_targets.py, navsim_pdm/")
    ap.add_argument("--plans", nargs="+", required=True,
                    help="one or more plan files, optionally name=path. EVERY arm is scored in the same pass over a token, so the "
                         "PDM-Closed / metric-cache work (~1-2 s/token, the dominant cost) is paid once for all arms. "
                         "All files must carry the same tokens (an arm missing a token is an error).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--split", choices=sorted(SPLIT_ROWS), default="navtrain")
    ap.add_argument("--token-log", default=None, help="token->log json(.gz) (navtest_inputs.json.gz) when the plans carry no logs")
    ap.add_argument("--db-root", default=DB_ROOT)
    ap.add_argument("--a7", type=int, default=0)
    ap.add_argument("--shift-lat", type=float, default=0.0, help="RED control: shift every plan pose by this many m to the left")
    ap.add_argument("--controls", action="store_true", help="also score the control arms per arm (sim_as_executed, shifted 0.5 m)")
    ap.add_argument("--human-arm", action="store_true",
                    help="add an arm named 'human': plan20 = logged future at 0.2 s, plan8 = the 8 logged poses (NAVSIM's standard human "
                         "row for the tracked arm, E2's builder for the transparent arm). Then reference8 == human8 exactly.")
    ap.add_argument("--no-fam", action="store_true", help="skip the family payload (human future, lead track)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--metric-cache-root", default="",
                    help="A1 FIX (WTRANSP_DIAG.md): hand the scorer the HARNESS'S OWN stored MetricCache parts (PDM-Closed, ego, observation, "
                         "centerline, lanes, drivable map; <root>/<log>/unknown/<token>/metric_cache.pkl) instead of rebuilding them from the "
                         "nuPlan DB. NAVTEST ONLY (navtrain has no harness cache). The DB scenario is still built for the human future and the "
                         "lead track. Default empty = the DB rebuild, output unchanged. A missing cache file is an error, never a fallback.")
    a = ap.parse_args()
    sys.path.insert(0, a.refe)
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    from navsim_pdm.pdm_array_representation import ego_state_to_state_array
    si, sn = (int(x) for x in a.shard.split("/"))
    arms: dict = {}
    for spec in a.plans:
        if "=" in spec:
            nm, fp = spec.split("=", 1)
        else:
            nm, fp = os.path.splitext(os.path.basename(spec))[0], spec
        arms[nm] = load_plans(fp)
    names = list(arms)
    base = arms[names[0]]
    for nm in names[1:]:
        if set(arms[nm]) != set(base):
            raise SystemExit(f"arm {nm} does not carry the same tokens as arm {names[0]}")
    tlog = load_token_log(a.token_log) if a.token_log else {}
    for t, p in base.items():
        p["log"] = p["log"] or tlog.get(t)
        if p["log"] is None:
            raise SystemExit(f"no log for token {t}: give logs in the plans file or --token-log")
    done = done_tokens(a.out)
    # shard over SAMPLES by a STABLE token hash, so a resume (or a different pending set) never moves a token between shards
    todo = [t for t in sorted(base) if t not in done and int(hashlib.sha1(t.encode()).hexdigest(), 16) % sn == si]
    by: dict = {}
    for t in todo:
        by.setdefault(base[t]["log"], []).append(t)
    dbs = NS.index_dbs(db_root=a.db_root)
    hr, fr = SPLIT_ROWS[a.split]
    kw = {} if hr is None else {"history_rows": hr, "future_rows": fr}
    n_done, t0 = 0, time.time()
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:           # a killed run can leave a torn last line: terminate it,
        with open(a.out, "rb") as fb:                                  # so the next row does not get glued onto it
            fb.seek(-1, 2)
            torn = fb.read(1) != b"\n"
        if torn:
            with open(a.out, "a", encoding="utf-8") as fb:
                fb.write("\n")
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg in sorted(by):
            if not os.path.exists(dbs.get(lg, "")):
                for t in by[lg]:
                    fo.write(json.dumps({"log": lg, "token": t, "error": "no local DB"}) + "\n")
                continue
            for sc in NS.build_scenarios_for_log(dbs[lg], by[lg], **kw):
                tok = sc._initial_lidar_token
                rec = {"log": lg, "token": tok, "arms": {}}
                ts = time.time()
                try:
                    parts = load_cache_parts(a.metric_cache_root, lg, tok) if a.metric_cache_root else PT.metric_cache_parts(sc)
                    ego = parts[1]
                    s0 = ego_state_to_state_array(ego)
                    hum = human_rel(sc, ego) if ((not a.no_fam) or a.human_arm) else None
                    todo_arms = [(nm, arms[nm][tok]) for nm in names]
                    if a.human_arm:
                        todo_arms.append(("human", {"log": lg, "plan20": hum[1::2][:20].copy(), "plan8": hum[4::5][:8].copy()}))
                    for nm, p in todo_arms:
                        # the human arm is never A7-repaired or shifted: it is the reference, not a submission
                        isH = a.human_arm and nm == "human"
                        rec["arms"][nm] = score_one_arm(parts, ego, s0, p, False if isH else bool(a.a7),
                                                        0.0 if isH else a.shift_lat, a.controls and not isH, not a.no_fam)
                    if not a.no_fam:
                        v0 = ego.dynamic_car_state.rear_axle_velocity_2d
                        rec["shared"] = {"v0": float(math.hypot(v0.x, v0.y)), "human8": hum[4::5][:8].tolist(),
                                         "lead": lead_track(sc, ego)}
                    rec["t_s"] = round(time.time() - ts, 3)
                except Exception as e:                                    # noqa: BLE001
                    rec = {"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n_done += 1
                if a.limit and n_done >= a.limit:
                    print("ZZTRANSP_SHARD_DONE", a.shard, n_done, round(time.time() - t0, 1))
                    return 0
    print("ZZTRANSP_SHARD_DONE", a.shard, n_done, round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
