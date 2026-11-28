"""zt3_lib.py -- pure, numpy-only definitions for PREREG_ZT3.md (every threshold here is the pre-registered one).

Used by zt3_jobs.py / zt3_analyze.py (driverl-eval venv) and zt3_geom.py (NAVSIM py3.9 venv). Unit- and
mutation-tested by test_zt3.py. Nothing here reads data.
"""
from __future__ import annotations

import numpy as np

# ---- pre-registered constants (PREREG_ZT3.md section 3) -------------------------------------------------------
NEAR_M = 15.0
TURN_DEG = 15.0          # S_TURNS
MAN_DEG = 30.0           # man(X)
E_M = 0.5                # |e*| class threshold
GRAZE_M = 0.5
S_STOP_M = 1.0
MOVE_M = 5.0             # WRONG_MANOEUVRE needs both paths >= 5 m
EXC_JUNC_M = 2.0
TSTAR_BINS = [(0, 10), (10, 20), (20, 30), (30, 41)]      # steps of 0.1 s: [0,1) [1,2) [2,3) [3,4] s
CURV_BINS = [(0.0, 15.0), (15.0, 45.0), (45.0, 1e9)]       # |dpsi_H| degrees
CLASSES = ("S_STOP", "WRONG_MANOEUVRE", "CORNER_CUT", "WIDE_TURN", "STRAIGHT_DRIFT", "RUN_LONG", "ON_PATH")
HEADS = ("NC", "DAC", "EP", "TTC", "C", "DDC", "lane")


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float64)))      # identical to eval/rl_val_eval.sigmoid


def selection_score(logits, rule):
    """[..., M, C] logits -> [..., M] selection score; identical algebra to eval/rl_val_eval.select."""
    p = sigmoid(logits)
    s = p[..., 0] * p[..., 1] * (5.0 * p[..., 2] + 5.0 * p[..., 3] + 2.0 * p[..., 4]) / 12.0
    if rule == "navsim_v1_lane":
        s = s * p[..., 6]
    elif rule != "navsim_v1":
        raise ValueError(rule)
    return s


def best_nonzero(score, pdms):
    """S = argmax of the selection score over proposals with PDMS > 0; -1 if none.  score/pdms [M]."""
    ok = np.asarray(pdms) > 0
    if not ok.any():
        return -1
    return int(np.argmax(np.where(ok, score, -np.inf)))


def rank_of(score, k):
    """1-based rank of proposal k in descending score (ties broken by index, like argmax)."""
    order = np.lexsort((np.arange(len(score)), -np.asarray(score)))
    return int(np.nonzero(order == k)[0][0]) + 1


def term_decomposition(logits_p, logits_s, rule):
    """log s(P) - log s(S) split into NC, DAC, W(=(5EP+5TTC+2C)/12), lane.  Sums exactly to the total."""
    pp, ps = sigmoid(logits_p), sigmoid(logits_s)
    w = lambda p: (5.0 * p[2] + 5.0 * p[3] + 2.0 * p[4]) / 12.0
    d = {"NC": np.log(pp[0]) - np.log(ps[0]), "DAC": np.log(pp[1]) - np.log(ps[1]), "W": np.log(w(pp)) - np.log(w(ps))}
    d["lane"] = (np.log(pp[6]) - np.log(ps[6])) if rule == "navsim_v1_lane" else 0.0
    return {k: float(v) for k, v in d.items()}


def nondrivable_steps(corner_in):
    """corner_in [..., T, n_poly, 4] bool (corner inside drivable polygon j) -> [..., T] bool non-drivable.
    pdm_scorer.py:284-286: a step is non-drivable when fewer than 4 of the 4 corners lie in >= 1 polygon."""
    return (np.asarray(corner_in).sum(axis=-2) > 0).sum(axis=-1) < 4


def first_true(flags):
    """first index of a True in a 1-D bool array, -1 if none."""
    idx = np.flatnonzero(np.asarray(flags))
    return int(idx[0]) if idx.size else -1


def arc_length(xy, upto=None):
    """cumulative path length of [T,2] points through index `upto` (inclusive); whole path if None."""
    xy = np.asarray(xy, np.float64)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return float(cum[-1] if upto is None else cum[upto])


def wrap_deg(a):
    return (np.asarray(a, np.float64) + 180.0) % 360.0 - 180.0


def manoeuvre(dpsi_deg, thr=MAN_DEG):
    return "LEFT" if dpsi_deg >= thr else ("RIGHT" if dpsi_deg <= -thr else "STRAIGHT")


def signed_offset(pt, poly):
    """Signed perpendicular distance of pt [2] from polyline poly [T,2] (nearest segment; the LAST segment is
    extended beyond the end, the first is NOT extended backwards). Positive = LEFT of the local direction.
    Returns (offset, arc length of the projection on poly)."""
    pt = np.asarray(pt, np.float64)
    P = np.asarray(poly, np.float64)
    seg = np.diff(P, axis=0)
    L = np.linalg.norm(seg, axis=-1)
    keep = L > 1e-6
    if not keep.any():                                   # degenerate (standstill) polyline
        return float("nan"), 0.0
    A, D, Ls = P[:-1][keep], seg[keep], L[keep]
    cum = np.concatenate([[0.0], np.cumsum(Ls)])[:-1]
    t = ((pt - A) * D).sum(-1) / (Ls ** 2)
    t_cl = np.clip(t, 0.0, None)
    t_cl[:-1] = np.minimum(t_cl[:-1], 1.0)               # only the last segment extends beyond its end
    proj = A + t_cl[:, None] * D
    dist = np.linalg.norm(pt - proj, axis=-1)
    j = int(np.argmin(dist))
    cross = D[j, 0] * (pt[1] - A[j, 1]) - D[j, 1] * (pt[0] - A[j, 0])
    return float(np.sign(cross) * dist[j]), float(cum[j] + t_cl[j] * Ls[j])


def side_of(out4):
    """out4 [4] bool in BBCoordsIndex order (FL, RL, RR, FR) -> LEFT / RIGHT / MIXED / NONE."""
    o = np.asarray(out4, bool)
    if not o.any():
        return "NONE"
    left, right = o[0] or o[1], o[2] or o[3]
    return "LEFT" if left and not right else ("RIGHT" if right and not left else "MIXED")


def classify(L_P, L_S, dpsi_P, dpsi_S, e_star, s_star_P, e_thr=E_M, man_thr=MAN_DEG, turn_thr=TURN_DEG):
    """The pre-registered pattern class (first match wins)."""
    if L_S < S_STOP_M:
        return "S_STOP"
    if L_P >= MOVE_M and L_S >= MOVE_M and manoeuvre(dpsi_P, man_thr) != manoeuvre(dpsi_S, man_thr):
        return "WRONG_MANOEUVRE"
    s_turns = abs(dpsi_S) >= turn_thr
    if not np.isfinite(e_star):
        return "ON_PATH" if s_star_P <= L_S else "RUN_LONG"
    if abs(e_star) >= e_thr:
        if s_turns:
            return "CORNER_CUT" if np.sign(e_star) == np.sign(dpsi_S) else "WIDE_TURN"
        return "STRAIGHT_DRIFT"
    return "RUN_LONG" if s_star_P > L_S else "ON_PATH"


def route_diff(lanes_P, lanes_S, t_star):
    """ROUTE_DIFF: at some step k < t* P's centre lies in >= 1 lane/connector polygon and NONE of them is ever
    entered by S's centre.  lanes_*: list (per step) of lists of polygon ids."""
    s_all = set(x for step in lanes_S for x in step)
    for k in range(max(t_star, 0)):
        here = set(lanes_P[k])
        if here and not (here & s_all):
            return True
    return False


def within_token_auc(p, y):
    """Mann-Whitney AUC of score p separating y=1 from y=0 (ties count 0.5); nan if one class is empty."""
    p, y = np.asarray(p, np.float64), np.asarray(y).astype(bool)
    pos, neg = p[y], p[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))
