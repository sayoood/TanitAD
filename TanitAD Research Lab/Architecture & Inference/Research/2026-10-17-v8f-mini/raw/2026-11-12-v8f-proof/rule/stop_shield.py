"""stop_shield.py -- the RULESHIELD braking-envelope shield (PREREG_V8F_PROOF_ADD3), a pure numpy function.

It is the algorithm machine-checked as VC-C1 / VC-C2 in fv/FV_SPEC_V1.md (fv/fv_check.py lines 55-66), step for step:
    vb = max(v - q, 0)           (q = a*dt)
    uc = max(u, vb)              (u: the proposal, ANY real)
    d2 = d - v*dt
    v' = uc if B(uc) <= d2 else vb
    B(x) = Bcf(x, n) = dt*((n+1)*x - q*n*(n+1)/2) with n = ceil(x/q) - 1 for x > 0 (the steps of the braking recursion B(x) = x*dt + B(max(x-q,0)), B(0) = 0)
Conclusions held at every step when the invariant B(v) <= d holds on entry: B(v') <= d', d' >= 0, v' >= vb.
Evidence class: code of the theorem's encoding; the tests are in test_stop_shield.py.
"""
import math

import numpy as np

A_BRAKE = 3.0      # m/s2, fixed before data (card)
EPS_N = 1e-9


def brake_dist(v, q, dt):
    """Closed-form braking distance Bcf(v, n), n = ceil(v/q) - 1, for v > 0; 0 for v <= 0.  Scalar or array."""
    v = np.asarray(v, np.float64)
    vp = np.maximum(v, 0.0)
    k = np.ceil(vp / q - EPS_N)            # number of positive terms v, v-q, ... (tolerates the float ratio landing a hair above an integer)
    k = np.where(vp > 0, np.maximum(k, 1.0), 0.0)
    n = k - 1.0
    out = dt * ((n + 1.0) * vp - q * n * (n + 1.0) / 2.0)
    return np.where(vp > 0, out, 0.0)


def brake_dist_rec(v, q, dt):
    """The recursion B(v) = v*dt + B(max(v-q, 0)), B(0) = 0 (scalar; the reference the closed form must equal)."""
    s, v = 0.0, float(v)
    while v > 0.0:
        s += v * dt
        v = max(v - q, 0.0)
    return s


def shield_step(v, u, d, q, dt, fallback=True):
    """One step.  -> (v', d').  v >= 0 is the held speed, u the proposal (any real), d the remaining budget.
    fallback=False is the mutation M-C1a (v' = uc: the B(uc) <= d' test and the vb branch removed)."""
    vb = max(v - q, 0.0)
    uc = max(u, vb)
    d2 = d - v * dt
    if not fallback:
        return uc, d2
    vn = uc if float(brake_dist(uc, q, dt)) <= d2 else vb
    return vn, d2


def shield_speeds(s, d0, q, dt, fallback=True):
    """Shield a speed profile s[0..n-1] (segment speeds).  Segment 0 runs at its own speed (the start state v_0 = s[0]); step k offers s[k+1] as the proposal.
    -> (v [n], feasible, d_left).  If B(v_0) > d0 the start is INFEASIBLE: every step brakes at q (v' = vb)."""
    s = np.asarray(s, np.float64)
    n = len(s)
    v = np.empty(n)
    v[0] = max(float(s[0]), 0.0)
    d = float(d0)
    feasible = float(brake_dist(v[0], q, dt)) <= d
    for k in range(n - 1):
        if feasible:
            v[k + 1], d = shield_step(v[k], float(s[k + 1]), d, q, dt, fallback)
        else:
            v[k + 1], d = max(v[k] - q, 0.0), d - v[k] * dt
    d_left = d - v[n - 1] * dt           # budget left after the last segment
    return v, feasible, d_left


# ------------------------------------------------------------------------------------------------ the plan path
def path_of(plan):
    """The plan's own path: points [0, plan[0..n-1]] (tick frame, origin = ego now); cumulative arc L; unit direction of the final (extension) ray."""
    P = np.vstack([np.zeros((1, 2)), np.asarray(plan, np.float64)])
    seg = np.diff(P, axis=0)
    sl = np.hypot(seg[:, 0], seg[:, 1])
    L = np.concatenate([[0.0], np.cumsum(sl)])
    nz = np.nonzero(sl > 1e-9)[0]
    if len(nz):
        j = nz[-1]
        ext = seg[j] / sl[j]
    else:
        ext = np.array([1.0, 0.0])
    return P, L, sl, ext


def seg_speeds(plan, dt):
    """Along-path segment speeds of the plan (the 0.2 s knots), segment k covers knot k-1 -> knot k (knot -1 = the origin)."""
    return path_of(plan)[2] / dt


def point_at(P, L, ext, s):
    """Position at arc length s (array) on the plan path; beyond the last knot the path is extended straight along its last direction; s < 0 clamps to 0."""
    s = np.maximum(np.asarray(s, np.float64), 0.0)
    out = np.empty(s.shape + (2,))
    flat, o = s.reshape(-1), out.reshape(-1, 2)
    for i, x in enumerate(flat):
        if x >= L[-1]:
            o[i] = P[-1] + ext * (x - L[-1])
        else:
            j = int(np.searchsorted(L, x, side="right")) - 1
            j = min(max(j, 0), len(L) - 2)
            w = (x - L[j]) / max(L[j + 1] - L[j], 1e-12)
            o[i] = P[j] + w * (P[j + 1] - P[j])
    return out


def retime(plan, v, dt):
    """Re-time the plan along its OWN path with the segment speeds v: knot k = path(sum_{i<=k} v_i*dt).  -> [n, 2]."""
    P, L, _, ext = path_of(plan)
    return point_at(P, L, ext, np.cumsum(np.asarray(v, np.float64) * dt))


def rest_arc(plan, xy, ext_len=300.0):
    """Arc length (on the extended plan path) of the point nearest to the rest point xy (tick frame).  The cue distance d_0 of both variants."""
    P, L, _, ext = path_of(plan)
    Pe = np.vstack([P, P[-1] + ext * ext_len])
    Le = np.concatenate([L, [L[-1] + ext_len]])
    xy = np.asarray(xy, np.float64)
    best, arc = np.inf, 0.0
    for j in range(len(Pe) - 1):
        a, b = Pe[j], Pe[j + 1]
        ab = b - a
        n2 = float(ab @ ab)
        t = 0.0 if n2 < 1e-18 else min(max(float((xy - a) @ ab) / n2, 0.0), 1.0)
        dd = float(np.hypot(*(a + t * ab - xy)))
        if dd < best - 1e-12:
            best, arc = dd, Le[j] + t * math.sqrt(n2)
    return arc, best


def dist_to_path(P, L, ext, pts, ext_len=300.0):
    """Distance of each point to the extended plan path (for control s3)."""
    Pe = np.vstack([P, P[-1] + ext * ext_len])
    out = np.empty(len(pts))
    for i, xy in enumerate(np.asarray(pts, np.float64)):
        best = np.inf
        for j in range(len(Pe) - 1):
            a, b = Pe[j], Pe[j + 1]
            ab = b - a
            n2 = float(ab @ ab)
            t = 0.0 if n2 < 1e-18 else min(max(float((xy - a) @ ab) / n2, 0.0), 1.0)
            best = min(best, float(np.hypot(*(a + t * ab - xy))))
        out[i] = best
    return out


def shield_plan(plan, d0, a=A_BRAKE, dt=0.2, fallback=True):
    """Shield one plan [30, 2] with cue distance d0.  -> dict(plan=[30,2] shielded, v=shielded speeds, feasible, s_null, s_shield, d_left)."""
    q = a * dt
    s = seg_speeds(plan, dt)
    v, feas, d_left = shield_speeds(s, d0, q, dt, fallback)
    return {"plan": retime(plan, v, dt), "v": v, "feasible": bool(feas), "s_null": float(s.sum() * dt), "s_shield": float(v.sum() * dt), "d_left": float(d_left), "v_null": s}
