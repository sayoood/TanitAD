"""N4-DAC map-certified repairs for a privileged-teacher label (20 poses @0.2 s, ego frame). The teacher has the map; the student never does.
Every repair returns the FIRST certified candidate in a fixed, smallest-change-first order, or None. Certificate (vs the base label's own
tracked scores, same scorer call): DAC = 1, NC >= base, DDC >= base, TTC >= base, C >= base.  Methods:
  B  blind lateral offset      ramp(t/1 s) * dy, dy in +-{0.10, 0.20, 0.30, 0.50} m
  H  heading nudge             rotate the plan about the ego origin by +-{0.5, 1, 2, 3} deg (heading channel rotated too)
  A  informed shift (label)    shift = ramp(t/t_v) * (ext + margin) * unit(return vector of the worst-outside corner), iterated <= 5x, applied to the LABEL
  D  informed shift (target)   the same shift applied to the TEACHER target t20, then the label pipeline is re-run (Tc20s iteration + P5-R lambda 3.0):
                               the DAC margin inside the P5-R solve (P5-R inverts the tracker toward the shifted, margin-bearing target)
All shifts are along the SCORER-computed return vector, so the geometry is exactly what DAC tests (4 corners vs the union of ROADBLOCK /
INTERSECTION / CARPARK polygons). Regression arms (tests): sign-flipped A must stay failing; zero shift must be bit-identical."""
import math
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

import n4_geom as G

MARGIN = 0.05
CAP = 5
BLIND = [(s * d) for d in (0.10, 0.20, 0.30, 0.50) for s in (+1, -1)]
HEAD_DEG = [(s * d) for d in (0.5, 1.0, 2.0, 3.0) for s in (+1, -1)]
T20 = 0.2 * np.arange(1, 21)


def certified(c, b):
    return c["dac"] == 1.0 and c["nc"] >= b["nc"] and c["ddc"] >= b["ddc"] - 1e-12 and c["ttc"] >= b["ttc"] - 1e-12 and c["c"] >= b["c"] - 1e-12


def certified_dac(c, b):
    """N4-DM certificate: tracked DAC = 1 only (map and ego state; no NC / DDC / TTC / C comparison, so no agent futures)"""
    return c["dac"] == 1.0


CERTS = {"full": certified, "dac": certified_dac}


def ramp(t_ref):
    return np.clip(T20 / max(t_ref, 0.2), 0.0, 1.0)


def apply_shift(plan20, d_ego, t_ref, sign=1.0):
    p = np.array(plan20, float)
    p[:, :2] += sign * ramp(t_ref)[:, None] * np.asarray(d_ego)[None, :]
    return p


def return_vector(cx, state41):
    """worst-outside corner over the 41 steps -> (ext_m, t_s, unit return vector in the EGO frame at t0)  or None if inside."""
    from navsim_pdm.pdm_array_representation import state_array_to_coords_array
    d = cx.ext_dist(state41)
    if d.max() <= 0:
        return None
    k = int(np.argmax(d))
    cs = state_array_to_coords_array(state41[None], cx.vp)[0][k, :4]
    far = max(cs, key=lambda p: cx.drivable.distance(Point(*p)))
    near = nearest_points(cx.drivable, Point(*far))[0]
    v = np.array([near.x - far[0], near.y - far[1]])
    h0 = cx.ego.rear_axle.heading
    c, s = math.cos(-h0), math.sin(-h0)
    v_ego = np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])
    n = np.linalg.norm(v)
    return float(d[k]), 0.1 * k, v_ego / max(n, 1e-9)


def _first_cert(cx, cands, base):
    res, _ = cx.score(cands)
    for i, r in enumerate(res):
        if certified(r, base):
            return i, r
    return None, None


def repair_blind(cx, label, base):
    cands = [apply_shift(label, [0.0, dy], 1.0) for dy in BLIND]
    i, r = _first_cert(cx, cands, base)
    return None if i is None else {"plan": cands[i], "m": r, "param": BLIND[i], "n_scored": len(cands)}


def repair_heading(cx, label, base):
    cands = []
    for deg in HEAD_DEG:
        a = math.radians(deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        p = np.array(label, float)
        p[:, :2] = p[:, :2] @ R.T
        p[:, 2] = G.wrap(p[:, 2] + a)
        cands.append(p)
    i, r = _first_cert(cx, cands, base)
    return None if i is None else {"plan": cands[i], "m": r, "param": HEAD_DEG[i], "n_scored": len(cands)}


def repair_informed(cx, label, base, margin=MARGIN, iters=5, sign=1.0):
    """A: iterate the shift on the label itself; the first certified iterate, or None."""
    p = np.array(label, float)
    total = np.zeros(2)
    for it in range(iters):
        res, sim = cx.score([p])
        if certified(res[0], base):
            return {"plan": p, "m": res[0], "param": {"iters": it, "total_shift_ego_m": total.round(4).tolist()}, "n_scored": it + 1}
        rv = return_vector(cx, sim[0])
        if rv is None:                       # inside the area but another factor (NC / TTC / C) is worse: no DAC shift can help
            return None
        ext, t_v, u = rv
        d = (ext + margin) * u
        total += sign * d
        p = apply_shift(p, d, t_v, sign)
    res, _ = cx.score([p])
    if certified(res[0], base):
        return {"plan": p, "m": res[0], "param": {"iters": iters, "total_shift_ego_m": total.round(4).tolist()}, "n_scored": iters + 1}
    return None


def repair_target(cx, t20, base, margin=MARGIN, iters=CAP, sign=1.0, cert=certified):
    """D: iterate the shift on the TEACHER TARGET; the label is rebuilt by the unchanged label pipeline (Tc20s + P5-R lambda 3).
    sign = +1 the frozen repair; -1 = N4-X (deliberate regression); cert = `certified` (N4-D) or `certified_dac` (N4-DM).
    The defaults are the development behaviour, bit for bit (sign 1.0 multiplies by 1.0)."""
    tgt = np.array(t20, float)
    total = np.zeros(2)
    for it in range(iters):
        lab = cx.labels(tgt)["R"]
        res, sim = cx.score([lab])
        if cert(res[0], base):
            return {"plan": lab, "m": res[0], "param": {"iters": it, "total_shift_ego_m": total.round(4).tolist()}, "n_scored": it + 1}
        rv = return_vector(cx, sim[0])
        if rv is None:
            return None
        ext, t_v, u = rv
        d = (ext + margin) * u
        total += sign * d
        tgt = apply_shift(tgt, d, t_v, sign)
    lab = cx.labels(tgt)["R"]
    res, _ = cx.score([lab])
    if cert(res[0], base):
        return {"plan": lab, "m": res[0], "param": {"iters": iters, "total_shift_ego_m": total.round(4).tolist()}, "n_scored": iters + 1}
    return None


METHODS = {"B": lambda cx, lab, t20, base: repair_blind(cx, lab, base),
           "H": lambda cx, lab, t20, base: repair_heading(cx, lab, base),
           "A": lambda cx, lab, t20, base: repair_informed(cx, lab, base),
           "D": lambda cx, lab, t20, base: repair_target(cx, t20, base)}
