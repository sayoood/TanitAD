"""Part 1 probe: real-map centreline for a pilot_v5 state, curve_speed as shipped (v5) vs heading-over-arc curvature.
usage: curve_speed_probe.py IDX [IDX ...] --out FILE.json     (CPU only; one process)
Builds the SAME objects as gym_env.GymState (build_scenario, make_planner, plan_all, planner0._centerline) but skips the
speed sweep / evaluator (no effect on the centreline). cl_pts / cl_s are built exactly as RewardEvaluator.__init__ does."""
import os, sys, json, gzip, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
V5 = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v5/scorer_gym"
sys.path.insert(0, V5)
import numpy as np
import gym_env as g
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from shapely.geometry import Point
RAW = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v5/raw"
STATES = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v5/PREREG_REWARD_1_STATES.json"


FAST = False


def build_cl(st):
    sc = g.build_scenario(st)
    ego0 = sc.initial_ego_state
    p = g.make_planner(sc)
    if FAST:
        # same two calls plan_all makes before the centreline exists (route correction, then _update_proposal_manager); skips proposal simulation
        p._route_roadblock_correction(ego0)
        p._drivable_area_map = g.PDMDrivableMap.from_simulation(p._map_api, ego0, p._map_radius)
        p._update_proposal_manager(ego0)
    else:
        obs0 = sc.get_tracked_objects_at_iteration(0)
        tl0 = list(sc.get_traffic_light_status_at_iteration(0))
        it0 = SimulationIteration(ego0.time_point, 0)
        hist = SimulationHistoryBuffer.initialize_from_list(2, [ego0], [obs0], 0.1)
        p.plan_all(PlannerInput(it0, hist, tl0))
    limit = float(p._get_starting_lane(ego0).speed_limit_mps or 15.0)
    cl = p._centerline
    pts = np.asarray(cl.discrete_path and [[q.x, q.y, q.heading] for q in cl.discrete_path])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts[:, :2], axis=0).T))])
    s0 = float(cl.project(Point(ego0.rear_axle.x, ego0.rear_axle.y)))
    return dict(pts=pts, s=s, limit=max(limit, 2.0), s0=s0, cl_length=float(cl.length))


def curve_v5(pts, cl_s, s0, ahead=30.0, a_lat=2.0):
    """verbatim copy of gym_env.RewardEvaluator.curve_speed (returns speed and k)"""
    m = (cl_s >= s0) & (cl_s <= s0 + ahead)
    h = pts[m, 2]
    if len(h) < 3:
        return 99.0, 0.0
    dh = np.abs(np.diff(np.unwrap(h)))
    ds = np.maximum(np.diff(cl_s[m]), 1e-3)
    k = float(np.max(dh / ds))
    return (99.0 if k < 1e-4 else math.sqrt(a_lat / k)), k


def kmax_detail(pts, cl_s, s0, ahead=30.0):
    m = np.where((cl_s >= s0) & (cl_s <= s0 + ahead))[0]
    if len(m) < 3:
        return None
    h = np.unwrap(pts[m, 2]); ds_raw = np.diff(cl_s[m]); ds = np.maximum(ds_raw, 1e-3)
    dh = np.abs(np.diff(h)); kk = dh / ds
    j = int(np.argmax(kk))
    lo, hi = max(0, j - 2), min(len(m), j + 4)
    return dict(k=float(kk[j]), dh_rad=float(dh[j]), ds_raw=float(ds_raw[j]), ds_used=float(ds[j]),
                pts=[dict(i=int(m[q]), s=float(cl_s[m[q]]), x=float(pts[m[q], 0]), y=float(pts[m[q], 1]), h=float(pts[m[q], 2])) for q in range(lo, hi)])


def curve_arc(pts, cl_s, s0, arc=3.0, ahead=30.0, a_lat=2.0, step=0.5):
    """heading-based curvature over a fixed arc length: resample every `step` m, k = max |h(s+arc)-h(s)|/arc (unwrapped heading)."""
    ss = np.arange(s0, min(s0 + ahead, cl_s[-1]), step)
    if len(ss) < 3:
        return 99.0, 0.0
    hu = np.unwrap(pts[:, 2])
    hh = np.interp(np.append(ss, ss[-1] + arc), cl_s, hu)
    n = int(round(arc / step))
    if len(hh) <= n:
        return 99.0, 0.0
    k = float(np.max(np.abs(hh[n:] - hh[:-n]) / (n * step)))
    return (99.0 if k < 1e-4 else math.sqrt(a_lat / k)), k


def load_rec(idx):
    sub = "run_main" if idx < 20 else "ladder_20_160"
    for l in gzip.open(f"{RAW}/{sub}/states.jsonl.gz", "rt", encoding="utf-8"):
        r = json.loads(l)
        if r.get("idx") == idx:
            return r
    return None


def cycles(rec, pts, s, s0, limit):
    cyc = []
    for name, d in rec["ladder"].items():
        cum = 0.0
        for r in d["logs"]:
            cum += r["progress_legacy"] * limit * 0.5      # approx: exact unless progress clipped at 1.5
            se = s0 + cum
            v5, k5 = curve_v5(pts, s, se); va, ka = curve_arc(pts, s, se)
            cyc.append(dict(rung=name, cycle=r["cycle"], s_end=se, lead=r["lead_gap"], conf=r["conf_ttc"],
                            active_stored=r["active"]["q_free"], hard=r["hard"], k_v5=k5, vstar_v5=min(limit, v5),
                            k_arc3=ka, vstar_arc3=min(limit, va)))
    return cyc


def analyse(idx):
    sts = json.load(open(STATES, encoding="utf-8"))
    st = sts[idx]
    rec = load_rec(idx)
    c = build_cl(st)
    pts, s, s0, limit = c["pts"], c["s"], c["s0"], c["limit"]
    ds = np.diff(s)
    out = dict(idx=idx, log=st["log"], n_pts=len(pts), cl_len=float(s[-1]), s0=s0, limit=limit,
               stored_limit=None if rec is None else rec["limit"],
               ds_min=float(ds.min()), ds_median=float(np.median(ds)), ds_max=float(ds.max()),
               n_ds_lt_1e3=int((ds < 1e-3).sum()), n_ds_lt_1e2=int((ds < 1e-2).sum()),
               dh_max_rad=float(np.abs(np.diff(np.unwrap(pts[:, 2]))).max()))
    out["from_ego_window"] = dict(v5=curve_v5(pts, s, s0), arc3=curve_arc(pts, s, s0), arc5=curve_arc(pts, s, s0, arc=5.0), arc2=curve_arc(pts, s, s0, arc=2.0))
    worst, kv5, karc = None, 0.0, 0.0
    for q in np.arange(s0, min(s[-1], s0 + 90.0), 1.0):   # cycles end <= ~60 m past s0 (state 108: 59.5 m)
        d = kmax_detail(pts, s, q)
        if d and (worst is None or d["k"] > worst["k"]):
            worst = dict(d, window_s0=float(q))
        karc = max(karc, curve_arc(pts, s, q)[1])
    out["scan_k_v5_max"] = None if worst is None else worst["k"]
    out["scan_k_arc3_max"] = karc
    out["worst_window"] = worst
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "cl_fast" if FAST else "cl"), exist_ok=True)
    np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "cl_fast" if FAST else "cl", f"cl_{idx}.npz"), pts=pts, s=s, s0=s0, limit=limit)
    if rec is not None:
        cyc = cycles(rec, pts, s, s0, limit)
        out["cycles"] = cyc
        pred = [(c_["vstar_v5"] >= 1.0, c_["active_stored"]) for c_ in cyc if c_["lead"] is None and c_["conf"] is None]
        out["pred_vs_stored_free_cycles"] = dict(
            n=len(pred), agree=int(sum(a == b for a, b in pred)),
            pred_inactive_stored_inactive=int(sum((not a) and (not b) for a, b in pred)),
            pred_active_stored_inactive=int(sum(a and (not b) for a, b in pred)),
            pred_inactive_stored_active=int(sum((not a) and b for a, b in pred)))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--fast" in a:
        a.remove("--fast"); FAST = True
    o = a.index("--out"); outf = a[o + 1]; idxs = [int(x) for x in a[:o]]
    res = {}
    if os.path.exists(outf):
        res = json.load(open(outf, encoding="utf-8"))
    for i in idxs:
        try:
            res[str(i)] = analyse(i)
        except Exception as e:
            import traceback
            res[str(i)] = dict(idx=i, error=repr(e), tb=traceback.format_exc()[-800:])
        json.dump(res, open(outf, "w", encoding="utf-8"), default=float)
        print(i, "done", flush=True)
