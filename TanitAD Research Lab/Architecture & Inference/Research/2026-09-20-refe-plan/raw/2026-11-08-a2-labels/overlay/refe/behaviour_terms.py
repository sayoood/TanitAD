"""A2 -- BEHAVIOUR-TERM labels (q_free, q_gap, q_osc, q_stall, q_comf) for every proposal, on the SAME 4 s non-reactive simulation the PDM labeller runs.

Stream A2-LABELS, 2026-11-08.  Design: SCORER_GYM_DESIGN.md section 2.5 (the PI's behaviour rules), rung A2 of section 2.6.  TRAINING LABELS ONLY: no term
here may enter the model's inputs at inference (leak test: A2_PREREG_NOTES.md; mutation = shuffle the labels across tokens).  Default OFF everywhere it is wired.

INPUT = exactly what `navsim_pdm_targets.score_plans` already holds: the simulated states `sim` [M+1, 41, 11] (row 0 = PDM-Closed's own tracked trajectory, rows 1..M the
proposals; 0.1 s, StateIndex layout) and the metric-cache `parts` (pdm_traj, ego, obs, centerline, lanes, dam).  Nothing is re-simulated.

OUTPUT (per proposal, arrays of length M):
  q_free   phantom braking   1 if the mean speed on free-road steps >= 0.9 v*, linear to 0.3 at 0.5 v*          (design 2.5, identical formula to the pilot)
  q_gap    distance keeping  time gap to the lead agent: 1 in [1.5, 2.5] s, linear to 0.5 at 1.0 and at 3.5 s, 0.2 below 0.8 s   (pilot's piecewise form)
  q_osc    speed oscillation 0.8 when the smoothed a_lon flips sign > 2 times among samples with |a_lon| > 0.5 m/s^2 inside the 4 s window
  q_stall  stall when clear  0.5 when v < 0.5 m/s for > 2 s while the corridor ahead has NO object and no mover enters it within 4 s
  q_comf   comfort           0.7 ** (number of violated nuPlan comfort channels), the channels evaluated by NAVSIM's OWN filtered metrics (see COMFORT NOTE)
  act_*    boolean: the term was APPLICABLE to that proposal (q_free: >= 5 of 9 sampled steps free road; q_gap: >= 1 lead step; q_osc/q_stall/q_comf: always
           applicable, `fired_*` below says whether q < 1).     fired_* = q < 1.
  diagnostics: n_free_steps, n_lead_steps, min_time_gap_s, v_bar_free, vstar [M], stall_run_s, comf_viol [M, 6] (bool, True = channel violated)

WHAT DIFFERS FROM THE PILOT (`pilot/scorer_gym/gym_env.py`), and why -- each is a flagged INTERPRETATION, a pre-registration candidate:
  * COMFORT NOTE.  The pilot built q_comf from RAW 10 Hz finite differences of the plant's acceleration/yaw (gym_env.py:331-348) and it fired on EVERY arm including
    PDM-Closed (M_GYM_1 HANDOFF caveat d).  NAVSIM's own comfort metric (`pdm_comfort_metrics.py`) never differences raw samples: it Savitzky-Golay-filters the
    acceleration (poly 2, window = the whole 41-step trajectory) and takes jerk / yaw rate / yaw accel as SG derivatives.  The differencing of a noisy tracker acceleration at
    10 Hz multiplies the noise by 1/dt = 10, which is the artefact.  This module therefore calls NAVSIM's channel functions one by one (identity: q_comf == 1 <=> NAVSIM's
    `c` == 1, asserted in tests) and measures its own activation on PDM-Closed (measure/RESULT_A2_ACTIVATION.md).  The pilot's detector is also reimplemented
    (`pilot_style_comfort`) ONLY so the two can be compared on the same states.
  * v*.  The design's v* = min(speed limit, route-curvature speed) needs the map, which the cached parts do NOT carry.  Mode "limit" uses it when the caller supplies it
    (`speed_limit_mps`; needs a D9 parts format bump -- A2_PREREG_NOTES).  Otherwise mode "ref_mean": v* = PDM-Closed's own mean speed over the same free-road steps.
    So q_free reads "did this proposal brake on a free road relative to what PDM-Closed itself does there"; it can never penalise a proposal that is as fast as the reference.
  * the lead / conflict geometry is the pilot's corridor (route centreline slice max(2.5 v, 20 m) ahead, half-width 1.5 m) evaluated against the cached constant-velocity
    occupancy forecast (PDMObservation), at 9 samples (every 0.5 s) of the 4 s window.
KNOWN LIMITS (UNVERIFIED rates until measured on the pod): no stop-sign / yield-sign information is in the parts, so a legal stop at a stop sign with a clear corridor reads
as q_stall = 0.5; a red light is an object in the corridor (token contains obs.red_light_token) so it correctly blocks "free" and "stall".
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional

import numpy as np

VERSION = "behaviour/1"
ORDER = ("q_free", "q_gap", "q_osc", "q_stall", "q_comf")        # column order of pack(): [M, 5]
COMFORT_CHANNELS = ("lon_acc", "lat_acc", "mag_jerk", "lon_jerk", "yaw_acc", "yaw_rate")   # = ego_is_comfortable's function order

# every constant below is a PRE-REGISTRATION CANDIDATE, not a tuned value (design 2.5); none may be tuned on navtest
C = dict(
    stride=5,                  # steps of 0.1 s between sampled steps -> 9 samples over the 4 s window
    corridor_half_w=1.5,       # m
    lead_min_m=20.0, lead_headway_s=2.5,
    lead_cos_min=0.5,          # a mover counts as LEAD when cos(angle(its velocity, ego heading)) > this, else it is a CONFLICT/crosser
    mover_speed=0.3,           # m/s: slower than this = static
    conflict_horizon_s=4.0,
    free_min_steps=5,          # of 9 sampled steps
    vstar_min=1.0,             # m/s: below this the reference is itself ~stopped, q_free is not applicable
    free_full=0.9, free_floor_ratio=0.5, free_floor_q=0.3,
    osc_window=9, osc_amp=0.5, osc_flips=2, osc_q=0.8,    # SG window 0.9 s, poly 2 (a 41-sample window would make >2 flips impossible)
    stall_v=0.5, stall_min_steps=22, stall_q=0.5,         # > 2.0 s = at least 22 consecutive 0.1 s samples
    comf_q=0.7,
)
# state columns (navsim StateIndex)
X, Y, H, VX, VY, AX, AY, ST, SR, W, WA = range(11)


# ------------------------------------------------------------------------------------------------ scene adapter
class Scene:
    """What the terms need from a scene, behind a tiny interface so tests can fake it with plain shapely.
        centerline : .linestring, .project(points), .substring(a, b), .length          (navsim PDMPath)
        occ(i)     : -> (tokens list, geometries ndarray) at 10 Hz index i (clipped)     (navsim PDMObservation[i])
        velocity   : {token: (vx, vy)} for moving objects (absent/None = static)
        red_token  : substring that marks a red-light pseudo-object;  collided: tokens the ego already overlaps at t0 (ignored, as the scorer does)
        route_area : optional shapely geometry = union of the ON-ROUTE lane / lane-connector polygons.  When given, only objects whose CENTRE is inside it can be a lead, a blocker or a
                     conflict (`intersects` was tried first: a car 0.2 m beside the lane still touched the lane polygon) (design 2.5: "on the route lane").  MEASURED need (2026-11-08, 13 trainval states): without it a parked car or kerb object that grazes the
                     1.5 m corridor reads as a lead and drove PDM-Closed's own q_gap to 0.22 in a scene.
    """

    def __init__(self, centerline, occ, velocity: dict, red_token: str, collided=(), n_occ: int = 52, route_area=None):
        self.route_area = route_area
        if route_area is not None:
            import shapely
            shapely.prepare(route_area)
        self.centerline, self._occ, self.velocity = centerline, occ, velocity
        self.red_token, self.collided, self.n_occ = red_token, set(collided), int(n_occ)
        self._tree_cache = {}

    def occ(self, i: int):
        return self._occ(int(min(max(i, 0), self.n_occ - 1)))


def scene_from_parts(parts) -> Scene:
    """navsim_pdm_targets' parts tuple -> Scene (reads PDMObservation: [i] occupancy maps, unique_objects, collided_track_ids)."""
    _traj, _ego, obs, centerline, lanes, dam = parts
    route_area = None
    try:                                                              # on-route lane polygons from the cached drivable-area map (tokens = lane ids)
        import shapely
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        keep = set(map(str, lanes))
        polys = [g for t, ty, g in zip(dam.tokens, dam.map_types, dam._geometries)
                 if str(t) in keep and ty in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR)]
        route_area = shapely.union_all(polys) if polys else None
    except Exception:                                                 # noqa: BLE001 -- no map info => no filter (the unfiltered behaviour, flagged by route_area=None)
        route_area = None
    vel = {}
    for tok, o in obs.unique_objects.items():
        v = getattr(o, "velocity", None)
        vel[tok] = (float(v.x), float(v.y)) if v is not None else None

    def occ(i):
        m = obs[i]
        return m.tokens, m._geometries, m

    return Scene(centerline, occ, vel, obs.red_light_token, obs.collided_track_ids, n_occ=len(obs._global_to_local_idcs), route_area=route_area)


# ------------------------------------------------------------------------------------------------ small pieces
def speed(states) -> np.ndarray:
    return np.hypot(states[..., VX], states[..., VY])


def sg_longitudinal_accel(v: np.ndarray, dt: float = 0.1, window: Optional[int] = None) -> np.ndarray:
    """Smoothed longitudinal acceleration from the SPEED profile: SG derivative (poly 2).  Used for q_osc only."""
    from scipy.signal import savgol_filter
    w = int(window or C["osc_window"])
    w = min(w, v.shape[-1] - (1 - v.shape[-1] % 2))                 # odd, <= length
    return savgol_filter(v, window_length=w, polyorder=2, deriv=1, delta=dt, axis=-1)


def q_gap_of(gap_m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-step distance-keeping factor.  v < 1 m/s: 1 if the bumper gap >= 1 m else 0.2.  Else time gap tg = gap / v (pilot's piecewise form of design 2.5)."""
    gap_m, v = np.asarray(gap_m, float), np.asarray(v, float)
    tg = gap_m / np.maximum(v, 1e-6)
    q = np.select([tg < 0.8, tg < 1.0, tg < 1.5, tg <= 2.5, tg <= 3.5],
                  [0.2, 0.2 + 0.3 * (tg - 0.8) / 0.2, 0.5 + 0.5 * (tg - 1.0) / 0.5, 1.0, 1.0 - 0.5 * (tg - 2.5)], 0.5)
    return np.where(v < 1.0, np.where(gap_m >= 1.0, 1.0, 0.2), q)


def q_free_of(vbar: np.ndarray, vstar: np.ndarray) -> np.ndarray:
    vstar = np.maximum(np.asarray(vstar, float), 1e-6)
    lo, hi = C["free_floor_ratio"] * vstar, C["free_full"] * vstar
    return np.clip(1.0 - (1.0 - C["free_floor_q"]) * (hi - vbar) / (hi - lo), C["free_floor_q"], 1.0)


def comfort_violations(states: np.ndarray) -> np.ndarray:
    """[M, 6] bool, True = channel VIOLATED, by NAVSIM's own filtered channel functions (COMFORT NOTE).  states [M, 41, 11]."""
    try:
        from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_comfort_metrics as CM
    except Exception:                                                   # noqa: BLE001 -- the vendored byte-identical copy (refe/navsim_pdm) needs only scipy
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from navsim_pdm import pdm_comfort_metrics as CM
    t = np.arange(states.shape[1], dtype=np.float64) * 0.1
    fns = (CM._compute_lon_acceleration, CM._compute_lat_acceleration, CM._compute_jerk_metric, CM._compute_lon_jerk_metric, CM._compute_yaw_accel, CM._compute_yaw_rate)
    return ~np.stack([np.asarray(f(states, t), bool) for f in fns], -1)


def pilot_style_comfort(states: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """The PILOT's detector (raw 10 Hz differences, gym_env.py:331-348), reimplemented ONLY to compare it with NAVSIM's on the same states.  [M, 6] bool, True = violated,
    in COMFORT_CHANNELS order (lon_acc, lat_acc, mag_jerk, lon_jerk, yaw_acc, yaw_rate)."""
    ax, ay, w = states[..., AX], states[..., AY], states[..., W]
    jx, jy = np.diff(ax, axis=-1) / dt, np.diff(ay, axis=-1) / dt
    out = np.zeros((states.shape[0], 6), bool)
    out[:, 0] = ((ax < -4.05) | (ax > 2.40)).any(-1)
    out[:, 1] = (np.abs(ay) > 4.89).any(-1)
    out[:, 2] = (np.hypot(jx, jy) > 8.37).any(-1)
    out[:, 3] = (np.abs(jx) > 4.13).any(-1)
    out[:, 4] = (np.abs(np.diff(w, axis=-1) / dt) > 1.93).any(-1)
    out[:, 5] = (np.abs(w) > 0.95).any(-1)
    return out


# ------------------------------------------------------------------------------------------------ geometry pass
def _geometry_pass(states: np.ndarray, scene: Scene, ego_coords: np.ndarray):
    """For each proposal m and sampled step i (every `stride` samples): the free-road flag, lead gap, conflict flag, blocker flag.
    ego_coords [M, 41, 5, 2] (navsim BBCoordsIndex layout: FL, RL, RR, FR, centre).  Returns dict of [M, S] arrays."""
    import shapely
    from shapely.geometry import LineString
    M, T = states.shape[:2]
    idx = np.arange(0, T, C["stride"])
    S = len(idx)
    cl = scene.centerline
    free = np.zeros((M, S), bool)
    valid = np.zeros((M, S), bool)
    lead_gap = np.full((M, S), np.nan)
    conflict = np.zeros((M, S), bool)
    blocked = np.zeros((M, S), bool)
    v_all = speed(states)
    # corridors
    corr = np.empty((M, S), dtype=object)
    for m in range(M):
        pts = shapely.points(states[m, idx, X], states[m, idx, Y])
        s0 = np.asarray(cl.project(pts), float)
        for i in range(S):
            L = max(C["lead_headway_s"] * v_all[m, idx[i]], C["lead_min_m"])
            a, b = s0[i], min(s0[i] + L, cl.length)
            g = None
            if b - a > 1.0 and np.isfinite(a):
                ls = cl.substring(a, b)
                if isinstance(ls, LineString) and ls.length > 0.5:
                    g = ls.buffer(C["corridor_half_w"], cap_style="flat")
            corr[m, i] = g
            valid[m, i] = g is not None
    horizon = int(round(C["conflict_horizon_s"] / (C["stride"] * 0.1)))        # 8 samples of 0.5 s
    ego_poly = shapely.polygons(ego_coords[:, idx, :4, :]) if ego_coords is not None else None   # [M, S]
    for i in range(S):
        base = int(idx[i])
        ok = [m for m in range(M) if corr[m, i] is not None]
        if not ok:
            continue
        cg = np.array([corr[m, i] for m in ok], dtype=object)
        # ---- now (j = 0): who is in the corridor AHEAD of the ego
        toks, geoms, occ = scene.occ(base)
        if len(toks):
            hit_c, hit_o = occ.query(cg, predicate="intersects")
        else:
            hit_c, hit_o = np.array([], int), np.array([], int)
        now_tokens = [set() for _ in ok]
        for c, o in zip(hit_c, hit_o):
            m = ok[c]
            tok = toks[o]
            if tok in scene.collided:
                continue
            geom = geoms[o]
            if scene.route_area is not None and scene.red_token not in tok and not scene.route_area.contains(geom.centroid):
                continue                                                            # not on the route lane
            hd = states[m, base, H]
            cx, cy = geom.centroid.x - states[m, base, X], geom.centroid.y - states[m, base, Y]
            if cx * math.cos(hd) + cy * math.sin(hd) <= 0.0:                        # behind the rear axle: not ahead
                continue
            now_tokens[c].add(tok)
            red = scene.red_token in tok
            vel = None if red else scene.velocity.get(tok)
            sp = math.hypot(*vel) if vel else 0.0
            is_lead = (not red) and (sp < C["mover_speed"] or (vel[0] * math.cos(hd) + vel[1] * math.sin(hd)) / sp > C["lead_cos_min"])
            if is_lead:
                gap = float(ego_poly[m, i].distance(geom)) if ego_poly is not None else float(geom.distance(shapely.points(states[m, base, X], states[m, base, Y])))
                if not (gap >= lead_gap[m, i]):                                     # NaN-safe min
                    lead_gap[m, i] = gap
            else:
                blocked[m, i] = True                                                # red light, or a crosser already inside the corridor
        # ---- future (j = 1..8): a MOVER (not now inside) that enters the corridor within the horizon = conflict
        for j in range(1, horizon + 1):
            tj = base + j * C["stride"]
            toks_j, geoms_j, occ_j = scene.occ(tj)
            if not len(toks_j):
                continue
            hc, ho = occ_j.query(cg, predicate="intersects")
            for c, o in zip(hc, ho):
                tok = toks_j[o]
                if tok in now_tokens[c] or tok in scene.collided or scene.red_token in tok:
                    continue
                if scene.route_area is not None and not scene.route_area.contains(geoms_j[o].centroid):
                    continue
                vel = scene.velocity.get(tok)
                if vel is None or math.hypot(*vel) < C["mover_speed"]:
                    continue
                conflict[ok[c], i] = True
    for m in range(M):
        for i in range(S):
            if valid[m, i]:
                free[m, i] = (not np.isfinite(lead_gap[m, i])) and (not blocked[m, i]) and (not conflict[m, i])
    # "free" also requires that nothing at all sits in the corridor ahead (static non-lead objects are leads with gap; red lights set `blocked`)
    return dict(free=free, valid=valid, lead_gap=lead_gap, conflict=conflict, blocked=blocked, idx=idx, v_s=v_all[:, idx])


# ------------------------------------------------------------------------------------------------ the terms
def behaviour_terms(sim: np.ndarray, parts, *, speed_limit_mps: Optional[float] = None, scene: Optional[Scene] = None, vehicle_params=None) -> dict:
    """sim [M+1, 41, 11] (row 0 = PDM-Closed) + parts -> the per-proposal terms for rows 1..M (see module docstring).  Pure function of its arguments."""
    sim = np.asarray(sim, np.float64)
    ref, st = sim[0], sim[1:]
    M = st.shape[0]
    scene = scene or scene_from_parts(parts)
    if vehicle_params is None:
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        vehicle_params = get_pacifica_parameters()
    try:
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    except Exception:                                                       # noqa: BLE001
        from navsim_pdm.pdm_array_representation import state_array_to_coords_array
    coords = state_array_to_coords_array(st, vehicle_params)
    g = _geometry_pass(st, scene, coords)
    free, lead_gap, v_s = g["free"], g["lead_gap"], g["v_s"]
    idx = g["idx"]
    S = len(idx)
    v_ref_s = speed(ref[None])[0, idx]

    # ---- q_free
    n_free = free.sum(1)
    act_free = np.zeros(M, bool)
    vbar = np.zeros(M)
    vstar = np.zeros(M)
    for m in range(M):
        if n_free[m] >= C["free_min_steps"]:
            f = free[m]
            vbar[m] = v_s[m, f].mean()
            vstar[m] = float(speed_limit_mps) if speed_limit_mps is not None else v_ref_s[f].mean()      # mode "limit" | "ref_mean"
            act_free[m] = vstar[m] >= C["vstar_min"]
    q_free = np.ones(M)
    q_free[act_free] = q_free_of(vbar[act_free], vstar[act_free])

    # ---- q_gap
    lead = np.isfinite(lead_gap)
    n_lead = lead.sum(1)
    act_gap = n_lead >= 1
    qg = np.where(lead, q_gap_of(np.nan_to_num(lead_gap, nan=1e9), v_s), np.nan)
    q_gap = np.ones(M)
    q_gap[act_gap] = np.nanmean(qg[act_gap], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tgs = np.where(lead, lead_gap / np.maximum(v_s, 1e-6), np.inf)
    min_tg = tgs.min(1)

    # ---- q_osc  (smoothed a_lon from the speed profile; sign flips among |a| > amp samples)
    a = sg_longitudinal_accel(speed(st))
    q_osc = np.ones(M)
    flips = np.zeros(M, int)
    for m in range(M):
        big = a[m][np.abs(a[m]) > C["osc_amp"]]
        flips[m] = int(np.sum(np.sign(big[1:]) != np.sign(big[:-1]))) if len(big) > 1 else 0
    q_osc[flips > C["osc_flips"]] = C["osc_q"]

    # ---- q_stall
    v = speed(st)
    slow = v < C["stall_v"]
    q_stall = np.ones(M)
    run_s = np.zeros(M)
    clear_s = free                                                          # free = valid & nothing in the corridor & no mover entering within 4 s
    for m in range(M):
        best = 0
        t = 0
        T = slow.shape[1]
        while t < T:
            if slow[m, t]:
                u = t
                while u + 1 < T and slow[m, u + 1]:
                    u += 1
                n = u - t + 1
                if n >= C["stall_min_steps"]:
                    smp = [i for i in range(S) if t <= idx[i] <= u]
                    if smp and all(clear_s[m, i] for i in smp):
                        best = max(best, n)
                t = u + 1
            else:
                t += 1
        run_s[m] = best * 0.1
        if best:
            q_stall[m] = C["stall_q"]

    # ---- q_comf
    viol = comfort_violations(st)
    q_comf = C["comf_q"] ** viol.sum(1)

    q = dict(q_free=q_free, q_gap=q_gap, q_osc=q_osc, q_stall=q_stall, q_comf=q_comf)
    act = dict(q_free=act_free, q_gap=act_gap, q_osc=np.ones(M, bool), q_stall=np.ones(M, bool), q_comf=np.ones(M, bool))
    out = {k: np.asarray(v_, np.float64) for k, v_ in q.items()}
    out.update({"act_" + k[2:]: act[k] for k in q})
    out.update({"fired_" + k[2:]: q[k] < 1.0 for k in q})
    out.update(n_free_steps=n_free, n_lead_steps=n_lead, min_time_gap_s=min_tg, v_bar_free=vbar, vstar=vstar, stall_run_s=run_s, comf_viol=viol, osc_flips=flips,
               q_prod=np.prod(np.stack([q[k] for k in ORDER], -1), -1))
    return out


def pack(t: dict) -> tuple:
    """-> (q [M, 5] float32 in ORDER, applicable [M, 5] bool)."""
    q = np.stack([np.asarray(t[k], np.float32) for k in ORDER], -1)
    a = np.stack([np.asarray(t["act_" + k[2:]], bool) for k in ORDER], -1)
    return q, a
