"""Pure numpy metrics for the route-following / fan / box diagnostics (SPEC.md sec. 3-7). No torch, no model.

Frame: ego at NOW, x forward, y LEFT, metres; paths are [..., 8, 2] at the slots (5,10,15,20,30,40,50,60) ticks.
"""
from __future__ import annotations

import math

import numpy as np

HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
SLOT_T_S = np.array([h * 0.1 for h in HORIZONS])
STALL_M = 0.05
TAU_C = 0.18063741505146028            # the run's own --nav-compliance-tau-rad (10.35 deg)
TURN_DEG, STRAIGHT_DEG, HEAD_AGREE_DEG = 30.0, 10.0, 15.0
MIN_LEN_M = 5.0
LAT_NAMES = ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC", "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R")
LAT_SIDE = np.array([0, 1, -1, 0, 1, -1, 1, -1])           # side of each tactical lat class
NAV_SIDE = {0: 0, 1: 1, 2: -1, 3: 0}                         # follow, left, right, straight


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def terminal_heading(P):
    """theta of the slot-50 -> slot-60 segment; 0.0 when it is shorter than 0.05 m (compliance_target)."""
    P = np.asarray(P, np.float64)
    d = P[..., -1, :] - P[..., -2, :]
    th = np.arctan2(d[..., 1], d[..., 0])
    return np.where(np.hypot(d[..., 0], d[..., 1]) < STALL_M, 0.0, th)


def dir_class(theta, tau=TAU_C):
    th = np.asarray(theta, np.float64)
    return np.where(th >= tau, 1, np.where(th <= -tau, -1, 0)).astype(np.int8)


def path_length(P):
    P = np.asarray(P, np.float64)
    Q = np.concatenate([np.zeros(P.shape[:-2] + (1, 2)), P], axis=-2)
    return np.linalg.norm(np.diff(Q, axis=-2), axis=-1).sum(-1)


def gt_class(gt, gt_valid):
    """-> (cls, theta): cls 'turnL','turnR','straight','gentle','unclassified' per SPEC sec. 3."""
    th = terminal_heading(gt)
    ln = path_length(gt)
    ok = np.asarray(gt_valid)[..., -1] & (ln >= MIN_LEN_M)
    deg = np.degrees(np.abs(th))
    cls = np.full(th.shape, "unclassified", dtype=object)
    cls[ok & (deg >= TURN_DEG) & (th > 0)] = "turnL"
    cls[ok & (deg >= TURN_DEG) & (th < 0)] = "turnR"
    cls[ok & (deg < STRAIGHT_DEG)] = "straight"
    cls[ok & (deg >= STRAIGHT_DEG) & (deg < TURN_DEG)] = "gentle"
    return cls, th


def ade_fde(P, gt, gt_valid):
    """P [..., 8, 2] (may carry an extra candidate axis before the slot axis), gt [W, 8, 2]."""
    P = np.asarray(P, np.float64)
    gt = np.asarray(gt, np.float64)
    v = np.asarray(gt_valid, bool)
    if P.ndim == gt.ndim + 1:                       # [W, N, 8, 2]
        gt = gt[:, None]
        v = v[:, None]
    d = np.linalg.norm(P - gt, axis=-1)
    vv = np.broadcast_to(v, d.shape)
    n = vv.sum(-1)
    ade = np.where(n > 0, (d * vv).sum(-1) / np.maximum(n, 1), np.nan)
    fde = np.where(vv[..., -1], d[..., -1], np.nan)
    return ade, fde, d


def rankdata(x):
    """average ranks (ties averaged), 1-based."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def circ_std(theta):
    t = np.asarray(theta, np.float64)
    R = np.hypot(np.cos(t).mean(), np.sin(t).mean())
    return float(np.sqrt(-2.0 * np.log(max(R, 1e-12))))


# ------------------------------------------------------------------------------------------------- #
# bootstrap (episode clusters)                                                                       #
# ------------------------------------------------------------------------------------------------- #
def cluster_index(ep):
    eps, inv = np.unique(np.asarray(ep).astype(str), return_inverse=True)
    return eps, inv


def boot_ratio(num, den, ep, B=2000, seed=0, draws=None):
    """ratio of sums with episode-cluster bootstrap; num/den per window (den 0 = not counted)."""
    eps, inv = cluster_index(ep)
    E = len(eps)
    N = np.bincount(inv, weights=np.asarray(num, np.float64), minlength=E)
    D = np.bincount(inv, weights=np.asarray(den, np.float64), minlength=E)
    if draws is None:
        draws = np.random.default_rng(seed).integers(0, E, (B, E))
    cnt = np.stack([np.bincount(d, minlength=E) for d in draws])        # [B, E]
    bn, bd = cnt @ N, cnt @ D
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = bn / bd
    pt = N.sum() / D.sum() if D.sum() > 0 else np.nan
    return pt, bs, draws


def ci95(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [None, None]


def make_draws(ep, B=2000, seed=0):
    eps, _ = cluster_index(ep)
    return np.random.default_rng(seed).integers(0, len(eps), (B, len(eps)))


def boot_mean(x, ep, mask=None, draws=None, B=2000, seed=0):
    x = np.asarray(x, np.float64)
    m = np.isfinite(x) if mask is None else (np.asarray(mask, bool) & np.isfinite(x))
    return boot_ratio(np.where(m, x, 0.0), m.astype(float), ep, B=B, seed=seed, draws=draws)


# ------------------------------------------------------------------------------------------------- #
# rotated BEV boxes                                                                                  #
# ------------------------------------------------------------------------------------------------- #
def box_corners(cx, cy, l, w, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = 0.5 * l, 0.5 * w
    loc = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]], np.float64)      # CCW
    R = np.array([[c, -s], [s, c]])
    return loc @ R.T + np.array([cx, cy])


def _poly_area(P):
    if len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip(subject, clipper):
    """Sutherland-Hodgman; both CCW convex."""
    out = [tuple(p) for p in subject]
    n = len(clipper)
    for i in range(n):
        a, b = clipper[i], clipper[(i + 1) % n]
        inp = out
        out = []
        if not inp:
            break

        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

        def inter(p, q):
            x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
            x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < 1e-15:
                return q
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        s = inp[-1]
        for e in inp:
            if inside(e):
                if not inside(s):
                    out.append(inter(s, e))
                out.append(e)
            elif inside(s):
                out.append(inter(s, e))
            s = e
    return np.array(out, np.float64) if out else np.zeros((0, 2))


def bev_iou(b1, b2):
    """b = (cx, cy, l, w, yaw)."""
    P, Q = box_corners(*b1), box_corners(*b2)
    a1, a2 = _poly_area(P), _poly_area(Q)
    if a1 <= 0 or a2 <= 0:
        return 0.0
    if np.hypot(b1[0] - b2[0], b1[1] - b2[1]) > 0.5 * (math.hypot(b1[2], b1[3]) + math.hypot(b2[2], b2[3])):
        return 0.0
    I = _poly_area(_clip(P, Q))
    return float(I / (a1 + a2 - I))


def nms_keep(p, xy, mode, thr, boxes=None, min_p=0.0):
    """score-ordered greedy suppression over slots with p >= min_p. mode 'none'|'centre'|'iou'.
    centre: suppress if centre distance <= thr (m); iou: suppress if BEV IoU > thr ('iou0' == any overlap: thr 0)."""
    p = np.asarray(p, np.float64)
    cand = np.nonzero(p >= min_p)[0]
    keep = np.zeros(len(p), bool)
    if mode == "none":
        keep[cand] = True
        return keep
    order = cand[np.argsort(-p[cand], kind="mergesort")]
    kept = []
    for i in order:
        ok = True
        for j in kept:
            if mode == "centre":
                if math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]) <= thr:
                    ok = False
                    break
            else:
                if bev_iou(boxes[i], boxes[j]) > thr:
                    ok = False
                    break
        if ok:
            kept.append(i)
    keep[kept] = True
    return keep
