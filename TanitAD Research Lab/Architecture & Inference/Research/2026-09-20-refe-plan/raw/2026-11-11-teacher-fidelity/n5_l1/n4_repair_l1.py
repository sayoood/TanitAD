"""n4_repair_l1.py - L1: TWO-SIDED lateral repair of the teacher TARGET (TF-N5 candidate). CPU only. Imports the PINNED n4_repair / n4_geom; copies nothing.

REPAIR-D (n4_repair.repair_target) shifts the target along ONE return vector (worst-outside corner) and ping-pongs on a two-sided corridor
(ANATOMY_25: sign flips >= 2 in 16 of 20 non-guard residuals). L1 reads BOTH boundaries at once:
  1. label = labels(target)['R'] ; score ; the SAME certificate as REPAIR-D (n4_repair.certified: DAC = 1 and NC/DDC/TTC/C not worse) -> return.
  2. if no corner is outside any more (ext_dist <= 0) and the certificate still refuses: CERT_BLOCKED, return None (as REPAIR-D).
  3. for every simulated step k = 1..40 compute the FEASIBLE LATERAL INTERVAL [lo_k, hi_k] (m, along the vehicle left normal at that step): the set
     of lateral offsets d for which all 4 footprint corners lie inside the drivable union eroded by `margin` (0.05 m, as REPAIR-D). The interval is the
     0.01 m grid run containing d = 0, else the run nearest to 0. An EMPTY set at some step is retried at `fallback_margin` for that step (recorded);
     still empty -> INFEASIBLE flag, return None (corridor narrower than footprint + margin).
  4. per target pose j (t = 0.2 (j+1)) the bounds are the intersection of the intervals of steps 2j+1, 2j+2. When 0 lies outside a bound, the
     approached side is pulled `centre_m` deeper, capped at the half width (centre it when narrow). A minimum-total-variation ramp s_1..s_20
     (s_0 = 0, small second-difference term so it ramps from t0 like REPAIR-D) is found inside the bounds (LP, scipy HiGHS, deterministic).
  5. the step is s_j * n_j (left normal of the EXECUTED heading), rotated to the ego frame; on a SIGN FLIP of the dominant correction versus the
     previous iteration the step is HALVED (cumulative if damp_cum). It is added to the target; repeat at most `iters` (5) times.
sign = -1 is the deliberate-regression arm. Zero correction leaves the target, hence the label, bit-identical."""
import functools
import inspect
import math

import n5_env  # noqa: F401  (must precede numpy)
import numpy as np
import shapely
from scipy.optimize import linprog

import n4_repair as RP

MARGIN = RP.MARGIN
CAP = RP.CAP
GRID_RES = 0.01
DMAX = 1.5
CENTRE_M = 0.05
FALLBACK_MARGIN = 0.005
POSES = [2 * j + 2 for j in range(20)]      # sim step of target pose j (0.1 s steps, 0.2 s poses)


def _corners_heading(cx, state41):
    from navsim_pdm.pdm_array_representation import state_array_to_coords_array
    c = state_array_to_coords_array(state41[None], cx.vp)[0][:, :4, :]       # [41, 4, 2]
    return c, np.asarray(state41[:, 2], float)


def step_intervals(drivable, corners, heading, margin, dmax=DMAX, res=GRID_RES):
    """-> lo[41], hi[41] (nan where empty at this margin), nonempty[41]. Offsets along the left normal (-sin h, cos h) of each step."""
    n = int(round(dmax / res))
    d = np.arange(-n, n + 1) * res
    er = drivable.buffer(-margin) if margin > 0 else drivable
    if er.is_empty:
        return np.full(41, np.nan), np.full(41, np.nan), np.zeros(41, bool)
    shapely.prepare(er)
    nv = np.stack([-np.sin(heading), np.cos(heading)], -1)                                   # [41, 2]
    pts = corners[:, :, None, :] + d[None, None, :, None] * nv[:, None, None, :]             # [41, 4, G, 2]
    ok = shapely.contains_xy(er, pts[..., 0], pts[..., 1]).all(1)                            # [41, G]
    lo, hi, ne = np.full(41, np.nan), np.full(41, np.nan), np.zeros(41, bool)
    for k in range(41):
        idx = np.flatnonzero(ok[k])
        if idx.size == 0:
            continue
        ne[k] = True
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        if ok[k, n]:
            r = next(r for r in runs if r[0] <= n <= r[-1])
        else:
            r = min(runs, key=lambda r: min(abs(d[r[0]]), abs(d[r[-1]])))
        lo[k], hi[k] = d[r[0]], d[r[-1]]
    return lo, hi, ne


def pose_bounds(lo, hi, centre_m=CENTRE_M):
    """per-pose bounds from the step intervals (window intersection) + one-sided centring -> blo[20], bhi[20], n_window_conflicts"""
    blo, bhi, nconf = np.zeros(20), np.zeros(20), 0
    for j in range(20):
        ks = (2 * j + 1, 2 * j + 2)
        a, b = max(lo[k] for k in ks), min(hi[k] for k in ks)
        if a > b:
            nconf += 1
            a = b = 0.5 * (a + b)
        if a > 0:                                   # 0 below the interval: approach from below, stay deeper by centre_m (<= half width)
            a = min(a + centre_m, 0.5 * (a + b))
        elif b < 0:
            b = max(b - centre_m, 0.5 * (a + b))
        blo[j], bhi[j] = a, b
    return blo, bhi, nconf


def taut_ramp(blo, bhi, eps=0.01):
    """minimum-total-variation s_1..s_20 inside [blo, bhi], s_0 = 0, plus eps * sum |second difference| (ramps from t0). Deterministic LP."""
    if np.all(blo <= 0.0) and np.all(bhi >= 0.0):
        return np.zeros(20)
    nv, nt, nu = 20, 20, 20
    N = nv + nt + nu
    c = np.concatenate([np.zeros(nv), np.ones(nt), eps * np.ones(nu)])
    A = []

    def put(row, j, coef):                          # coef * s_j ; s_0 (and s_-1) are the constant 0
        if j >= 1:
            row[j - 1] += coef

    for j in range(1, 21):
        for sg in (+1, -1):
            row = np.zeros(N)
            put(row, j, sg)
            put(row, j - 1, -sg)
            row[nv + j - 1] = -1
            A.append(row)
            row = np.zeros(N)
            put(row, j, sg)
            put(row, j - 1, -2 * sg)
            put(row, j - 2, sg)
            row[nv + nt + j - 1] = -1
            A.append(row)
    bounds = [(float(blo[j]), float(bhi[j])) for j in range(20)] + [(0, None)] * (nt + nu)
    r = linprog(c, A_ub=np.array(A), b_ub=np.zeros(len(A)), bounds=bounds, method="highs")
    if r.status != 0:
        return np.clip(np.zeros(20), blo, bhi)
    return np.asarray(r.x[:20], float)


def l1_step(cx, state41, margin=MARGIN, fallback_margin=FALLBACK_MARGIN, centre_m=CENTRE_M, sign=1.0):
    """one L1 correction from one simulated state array [41, 11] -> (d_ego[20, 2], info). d_ego is None when INFEASIBLE."""
    corners, heading = _corners_heading(cx, state41)
    lo, hi, ne = step_intervals(cx.drivable, corners, heading, margin)
    relaxed = []
    if not ne[1:].all():
        lo2, hi2, ne2 = step_intervals(cx.drivable, corners, heading, fallback_margin)
        for k in range(1, 41):
            if not ne[k] and ne2[k]:
                lo[k], hi[k], ne[k] = lo2[k], hi2[k], True
                relaxed.append(k)
    bad = [k for k in range(1, 41) if not ne[k]]
    info = {"relaxed_steps": relaxed, "infeasible_steps": bad, "infeasible": bool(bad)}
    if bad:
        return None, info
    blo, bhi, nconf = pose_bounds(lo, hi, centre_m)
    s = taut_ramp(blo, bhi)
    hp = heading[POSES]
    nrm = np.stack([-np.sin(hp), np.cos(hp)], -1)                                            # [20, 2] world
    h0 = cx.ego.rear_axle.heading
    c0, s0 = math.cos(-h0), math.sin(-h0)
    R = np.array([[c0, -s0], [s0, c0]])
    d_ego = (s[:, None] * nrm) @ R.T
    info.update({"window_conflicts": int(nconf), "s_max_abs_m": float(np.abs(s).max()), "s_dominant": float(s[int(np.argmax(np.abs(s)))]),
                 "interval_width_min_m": float(np.min(hi[1:] - lo[1:]))})
    return d_ego * sign, info


def repair_l1(cx, t20, base, margin=MARGIN, iters=CAP, sign=1.0, cert=RP.certified, centre_m=CENTRE_M, fallback_margin=FALLBACK_MARGIN,
              damp=0.5, damp_cum=False, info=None):
    """L1 on the teacher target t20; same call shape as n4_repair.repair_target (cx, t20, base) -> result dict or None.
    info (optional dict) receives: status in {CERTIFIED, PASSED_AT_0, CERT_BLOCKED, INFEASIBLE, CAP_EXHAUSTED}, per-iteration records, n_flips."""
    info = {} if info is None else info
    info.update({"iters": [], "n_flips": 0, "status": None})
    tgt = np.array(t20, float)
    total = np.zeros(2)
    prev_dom, nflip = 0.0, 0

    def fin(res, status, it):
        info["status"] = status
        if res is None:
            return None
        return {"plan": res[0], "m": res[1], "param": {"iters": it, "total_shift_ego_m": total.round(4).tolist(), "n_flips": nflip}, "n_scored": it + 1}

    for it in range(iters):
        lab = cx.labels(tgt)["R"]
        res, sim = cx.score([lab])
        if cert(res[0], base):
            return fin((lab, res[0]), "PASSED_AT_0" if it == 0 else "CERTIFIED", it)
        d = cx.ext_dist(sim[0])
        if d.max() <= 0:
            return fin(None, "CERT_BLOCKED", it)
        step, inf = l1_step(cx, sim[0], margin, fallback_margin, centre_m, sign)
        rec = dict(inf, it=it, ext_max_m=float(d.max()))
        info["iters"].append(rec)
        if step is None:
            return fin(None, "INFEASIBLE", it)
        dom = inf["s_dominant"]
        factor = 1.0
        if prev_dom * dom < 0:
            nflip += 1
            factor = damp ** (nflip if damp_cum else 1)
            rec["damped"] = True
        info["n_flips"] = nflip
        if dom != 0:
            prev_dom = dom
        rec["factor"] = factor
        tgt = np.array(tgt, float)
        tgt[:, :2] += factor * step
        total += factor * step[-1]
    lab = cx.labels(tgt)["R"]
    res, _ = cx.score([lab])
    if cert(res[0], base):
        return fin((lab, res[0]), "CERTIFIED", iters)
    return fin(None, "CAP_EXHAUSTED", iters)


def make_l1(margin=MARGIN, iters=CAP, sign=1.0, cert="full", centre_m=CENTRE_M, fallback_margin=FALLBACK_MARGIN, damp=0.5, damp_cum=False):
    """the repair object an arm runs: a functools.partial with EXPLICIT parameters (readback() reads what the built object uses)"""
    return functools.partial(repair_l1, margin=margin, iters=iters, sign=sign, cert=RP.CERTS[cert], centre_m=centre_m, fallback_margin=fallback_margin,
                             damp=damp, damp_cum=damp_cum)


def readback(fn) -> dict:
    """parameters as the BUILT object will use them (partial keywords, else the function defaults)"""
    f = fn.func if isinstance(fn, functools.partial) else fn
    kw = dict(fn.keywords) if isinstance(fn, functools.partial) else {}
    sig = inspect.signature(f).parameters
    get = lambda k: kw[k] if k in kw else sig[k].default                      # noqa: E731
    out = {k: get(k) for k in ("margin", "iters", "sign", "centre_m", "fallback_margin", "damp", "damp_cum")}
    out["cert"] = next((n for n, c in RP.CERTS.items() if c is get("cert")), "UNKNOWN")
    out["target"] = f"{f.__module__}.{f.__name__}"
    return out
