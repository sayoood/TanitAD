"""dist_rule.py -- the DISTRULE projection (card PREREG_V8F_PROOF_ADD26 section 2), a pure numpy function, batched over windows.

A plan [B, N, 2] (N knots at 0.2 s in the ego frame of the tick, origin not in the array) is slowed along its own path so that the gap to an imagined
constant-speed lead keeps a bound.  Definitions that the card leaves open are in DISTRULE_DEFS.md (E3: arc lengths, lag form, saturated steps).
Tier T1-kin (kinematic rollout on the model's own actions, logged frames); a diagnostic, never driving performance.

  distance_rule(plan, v0, present, g_src, g0, v_L, m=10) -> RuleOut
    plan     [B, N, 2]  the tick plans (any float dtype)
    v0       [B]        rolled ego speed at the tick (m/s)
    present  [B] bool   the source says a lead is present
    g_src    [B]        the source gap (m); the rule is inactive above 60 m
    g0       [B]        gap of the rolled ego = g_src + a_l - a_m (m)
    v_L      [B]        imagined lead speed = max(0, v_l + c_src) (m/s)
  An inactive window keeps its plan bit for bit; when no window is active the SAME array object comes back.

The progress geometry (`_path_frame`, `logged_arc`, `progress`) is a verbatim copy of headway/headway_read.py lines 62-111 (the pod has no headway_read.py);
test_dist_rule.py asserts the copy equals the original.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

DT = 0.2
D_M = 3.0          # card section 2
TAU_S = 1.2
B_MAX = 6.0
G_MAX_M = 60.0     # the rule is inactive above this source gap
RAY = 1000.0       # headway_read.RAY
PUSH_MPS = 0.5     # control r3: the planted advance


# ============================================================================================ progress geometry (copy of headway_read.py:62-111)
def _path_frame(g):
    """g [N,30,2] -> vertices V [N,33,2] (back ray, P0..P30, forward ray), arc A [N,33], segment lengths L [N,32], unit directions U [N,32,2], non-degenerate nd [N,32]."""
    g = np.asarray(g, np.float64)
    n = len(g)
    P = np.concatenate([np.zeros((n, 1, 2)), g], axis=1)
    d = P[:, 1:] - P[:, :-1]
    ln = np.linalg.norm(d, axis=-1)
    nd = ln > 1e-9
    first = np.argmax(nd, axis=1)
    last = (d.shape[1] - 1) - np.argmax(nd[:, ::-1], axis=1)
    anyn = nd.any(1)
    ux = np.array([1.0, 0.0])
    ar = np.arange(n)
    u0 = np.where(anyn[:, None], d[ar, first] / np.maximum(ln[ar, first], 1e-300)[:, None], ux)
    u1 = np.where(anyn[:, None], d[ar, last] / np.maximum(ln[ar, last], 1e-300)[:, None], ux)
    V = np.concatenate([(P[:, 0] - RAY * u0)[:, None], P, (P[:, -1] + RAY * u1)[:, None]], axis=1)
    seg = V[:, 1:] - V[:, :-1]
    L = np.linalg.norm(seg, axis=-1)
    Lnd = L > 1e-9
    U = np.where(Lnd[..., None], seg / np.maximum(L, 1e-300)[..., None], 0.0)
    A = np.concatenate([np.full((n, 1), -RAY), np.zeros((n, 1)), np.cumsum(L[:, 1:-1], axis=1)], axis=1)
    A = np.concatenate([A, (A[:, -1] + L[:, -1])[:, None]], axis=1)
    return V, A, L, U, Lnd


def logged_arc(g):
    """a_l(k), k = 1..30: arc coordinate of the logged points (cumulative length of the logged path). [N,30]"""
    _, A, _, _, _ = _path_frame(g)
    return A[:, 2:32].copy()


def progress(g, pts):
    """a(q) for pts [N,M,2]: arc coordinate of the orthogonal projection on the nearest non-degenerate segment of the extended logged path. [N,M]"""
    V, A, L, U, nd = _path_frame(g)
    pts = np.asarray(pts, np.float64)
    a = V[:, None, :-1, :]
    ab = (V[:, 1:] - V[:, :-1])[:, None]
    q = pts[:, :, None, :]
    l2 = (L ** 2)[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.clip(((q - a) * ab).sum(-1) / np.where(l2 > 0, l2, 1.0), 0.0, 1.0)
    proj = a + t[..., None] * ab
    dist = np.linalg.norm(q - proj, axis=-1)
    dist = np.where(nd[:, None, :], dist, np.inf)
    j = np.argmin(dist, axis=-1)
    jj = j[..., None]
    arc = np.take_along_axis(np.broadcast_to(A[:, None, :-1], dist.shape), jj, -1)[..., 0]
    tt = np.take_along_axis(t, jj, -1)[..., 0]
    ll = np.take_along_axis(np.broadcast_to(L[:, None, :], dist.shape), jj, -1)[..., 0]
    return arc + tt * ll


# ============================================================================================ the plan path
def plan_path(plan):
    """One plan [N,2] -> (P [N+1,2] incl. the origin, a_P [N+1] cumulative arc, ext [2] unit direction of the extension ray)."""
    P = np.vstack([np.zeros((1, 2)), np.asarray(plan, np.float64)])
    seg = np.diff(P, axis=0)
    sl = np.hypot(seg[:, 0], seg[:, 1])
    L = np.concatenate([[0.0], np.cumsum(sl)])
    nz = np.nonzero(sl > 1e-9)[0]
    ext = seg[nz[-1]] / sl[nz[-1]] if len(nz) else np.array([1.0, 0.0])
    return P, L, ext


def sample_path(P, L, ext, s):
    """Points of the plan path at arc lengths s [M] (>= 0); beyond the last knot the path goes straight along `ext`."""
    s = np.maximum(np.asarray(s, np.float64), 0.0)
    out = np.empty((len(s), 2))
    for i, x in enumerate(s):
        if x >= L[-1]:
            out[i] = P[-1] + ext * (x - L[-1])
        else:
            j = min(max(int(np.searchsorted(L, x, side="right")) - 1, 0), len(L) - 2)
            w = (x - L[j]) / max(L[j + 1] - L[j], 1e-12)
            out[i] = P[j] + w * (P[j + 1] - P[j])
    return out


class RuleOut(NamedTuple):
    plan: np.ndarray        # [B, N, 2], same dtype as the input; the input object itself when no window is active
    active: np.ndarray      # [B] bool
    n_sat: np.ndarray       # [B] int, saturated steps over all N steps (cap would hold below the braking floor)
    n_sat_exec: np.ndarray  # [B] int, the same over the first m (executed) steps
    n_lift: np.ndarray      # [B] int, steps where the floor is above the plan own wish (a_f > a_w: a plan that wants to slow faster than B_MAX is lifted)
    changed: np.ndarray     # [B] bool, plan differs from the input
    a_new: np.ndarray       # [B, N] the new arc lengths a'(i), i = 1..N (NaN for inactive windows)


def distance_rule(plan, v0, present, g_src, g0, v_L, m=10, D=D_M, tau=TAU_S, b_max=B_MAX, use_cap=True, use_floor=True, push_mps=None, dt=DT, floor_lifts=True) -> RuleOut:
    """The projection of card section 2 (forward over i, lag form, see DISTRULE_DEFS E3).  `use_cap=False, use_floor=False` is the identity probe (mode plumb);
    `floor_lifts=False` (builder option, NOT the card: a' = min(a_w, max(a_f, a_c)), the floor limits the cap but never lifts a plan above its own wish, DISTRULE_DEFS E3b);
    `push_mps` replaces the cap and the floor by a planted advance a'(i) = a_P(i) + push_mps * dt * i (control r3)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return _distance_rule(plan, v0, present, g_src, g0, v_L, m, D, tau, b_max, use_cap, use_floor, push_mps, dt, floor_lifts)


def _distance_rule(plan, v0, present, g_src, g0, v_L, m, D, tau, b_max, use_cap, use_floor, push_mps, dt, floor_lifts) -> RuleOut:
    plan_in = plan
    plan = np.asarray(plan)
    B, N = plan.shape[0], plan.shape[1]
    v0 = np.asarray(v0, np.float64)
    g_src, g0, v_L = (np.asarray(x, np.float64) for x in (g_src, g0, v_L))
    with np.errstate(invalid="ignore"):
        active = (np.asarray(present, bool) & np.isfinite(g_src) & np.isfinite(g0) & np.isfinite(v_L) & np.isfinite(v0) & (g_src <= G_MAX_M))
    zi = np.zeros(B, np.int64)
    if not active.any():
        return RuleOut(plan_in, active, zi.copy(), zi.copy(), zi.copy(), np.zeros(B, bool), np.full((B, N), np.nan))
    p64 = plan.astype(np.float64)
    aP = np.zeros((B, N + 1))
    PP, EXT = [], []
    for b in range(B):
        P, L, ext = plan_path(p64[b])
        aP[b] = L
        PP.append(P)
        EXT.append(ext)
    steps = np.arange(1, N + 1)
    lag = np.zeros(B)                                   # l(i-1) = a_P(i-1) - a'(i-1)
    v_prev = v0.copy()                                  # v'(i-1)
    a_new = np.zeros((B, N))
    n_sat, n_sat_exec, n_lift = zi.copy(), zi.copy(), zi.copy()
    a_prev = np.zeros(B)                                # a'(i-1)
    for i in steps:
        a_w = aP[:, i] - lag
        if push_mps is not None:
            a_i = aP[:, i] + push_mps * dt * i                  # the plan advanced along its own path by push_mps times the elapsed time
            sat = lift = np.zeros(B, bool)
        else:
            a_c = (g0 + v_L * dt * i - np.maximum(D, tau * v_prev)) if use_cap else np.full(B, np.inf)
            a_f = (a_prev + np.maximum(0.0, v_prev - b_max * dt) * dt) if use_floor else np.full(B, -np.inf)
            a_i = np.maximum(a_f, np.minimum(a_w, a_c)) if floor_lifts else np.minimum(a_w, np.maximum(a_f, a_c))
            sat = active & (a_c < a_f) & (a_c <= a_w)
            lift = active & (a_f > a_w) & bool(floor_lifts)
        n_sat += (sat & active).astype(np.int64)
        n_lift += (lift & active).astype(np.int64)
        if i <= m:
            n_sat_exec += (sat & active).astype(np.int64)
        lag = aP[:, i] - a_i
        v_prev = (a_i - a_prev) / dt
        a_prev = a_i
        a_new[:, i - 1] = a_i
    out = plan.copy()
    changed = np.zeros(B, bool)
    for b in np.nonzero(active)[0]:
        a_b = a_new[b]
        keep = a_b == aP[b, 1:]                         # an unchanged step is the original knot (bit for bit)
        pts = sample_path(PP[b], aP[b], EXT[b], a_b)
        pts[keep] = p64[b][keep]
        new = pts.astype(plan.dtype)
        changed[b] = bool((new != plan[b]).any())
        out[b] = new
    a_new = np.where(active[:, None], a_new, np.nan)
    if not changed.any():
        return RuleOut(plan_in, active, n_sat, n_sat_exec, n_lift, changed, a_new)
    return RuleOut(out, active, n_sat, n_sat_exec, n_lift, changed, a_new)
