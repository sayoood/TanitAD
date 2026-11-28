"""zt5_lib.py -- pure-numpy definitions of PREREG_ZT5 (every constant here is the pre-registered one).

Imported by zt5_sim.py / zt5_dev.py / zt5_nav.py (NAVSIM py3.9 venv), zt5_analyze.py and zt5_seam_fix.py (any numpy venv),
and mutation-tested by test_zt5.py.  Reads no data.
"""
from __future__ import annotations

import numpy as np

# ---- pre-registered constants (PREREG_ZT5 sections 3-4) -----------------------------------------------------------
TAN_CHORD_MIN_M = 1.0          # F_c interior knots: central chord length needed to define the tangent
TAN_END_SEG_MIN_M = 0.5        # F_c last knot: both last segments at least this long
K0_MIN_M = 0.5                 # CF-S: first knot must be this far for kappa0
RHO_GRID = (0.99, 0.97, 0.95, 0.90)
SWITCH_MIN_M = 0.10            # F_b: delta(pick) - delta(j*) needed to switch
COMP_ITERS = 3                 # F_a = P4's compensate
MAN_DEG = 30.0
STRAIGHT_DEG = 2.0             # descriptor: local turn below this = straight
RULE = "navsim_v1_lane"
WHEELBASE_PACIFICA = 3.089     # nuPlan get_pacifica_parameters().wheel_base
MAX_STEER = np.pi / 3


def wrap(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float64)))      # == eval/rl_val_eval.sigmoid


def selection_score(logits, rule=RULE):
    """[..., M, C] logits -> [..., M]; identical algebra to eval/rl_val_eval.select."""
    p = sigmoid(logits)
    s = p[..., 0] * p[..., 1] * (5.0 * p[..., 2] + 5.0 * p[..., 3] + 2.0 * p[..., 4]) / 12.0
    if rule == "navsim_v1_lane":
        s = s * p[..., 6]
    elif rule != "navsim_v1":
        raise ValueError(rule)
    return s


def tangents(p8):
    """[8, 3] executed knots (t = 0.5..4 s, ego frame, origin at t = 0) -> (tau [8], defined [8] bool), PREREG F_c."""
    p = np.asarray(p8, np.float64)
    xy = np.concatenate([np.zeros((1, 2)), p[:, :2]], 0)              # knot 0 = origin
    tau = np.full(8, np.nan)
    ok = np.zeros(8, bool)
    for k in range(1, 8):                                                # knots 1..7 (array index k-1)
        c = xy[k + 1] - xy[k - 1]
        if np.hypot(*c) >= TAN_CHORD_MIN_M:
            tau[k - 1], ok[k - 1] = np.arctan2(c[1], c[0]), True
    a, b = xy[7] - xy[6], xy[8] - xy[7]
    if np.hypot(*a) >= TAN_END_SEG_MIN_M and np.hypot(*b) >= TAN_END_SEG_MIN_M:
        ta, tb = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0])
        tau[7], ok[7] = tb + 0.5 * wrap(tb - ta), True
    return tau, ok


def fix_heading_tangent(p8):
    """F_c: headings := path tangent where defined (nearest branch to the original heading); x, y bit-identical."""
    p = np.array(p8, dtype=np.float64, copy=True)
    tau, ok = tangents(p)
    p[ok, 2] = p[ok, 2] + wrap(tau[ok] - p[ok, 2])
    return p


def heading_inconsistency(p8):
    """max over knots with a defined tangent of |psi_k - tau_k| (rad); nan if none defined."""
    p = np.asarray(p8, np.float64)
    tau, ok = tangents(p)
    if not ok.any():
        return float("nan")
    return float(np.abs(wrap(p[ok, 2] - tau[ok])).max())


def kappa0(p8):
    """CF-S: curvature of the circle through the origin, tangent to +x, through the first knot."""
    x1, y1 = float(p8[0][0]), float(p8[0][1])
    r2 = x1 * x1 + y1 * y1
    return 0.0 if np.sqrt(r2) < K0_MIN_M else 2.0 * y1 / r2


def steer_from_kappa(k, wheelbase=WHEELBASE_PACIFICA):
    return float(np.clip(np.arctan(wheelbase * k), -MAX_STEER, MAX_STEER))


def self_deviation(sim_xy, raw_xy):
    """delta = max over steps of |S - R| (time-indexed, rear axle). [T,2] each -> float."""
    return float(np.linalg.norm(np.asarray(sim_xy, np.float64) - np.asarray(raw_xy, np.float64), axis=-1).max())


def local_frame(raw_xy, raw_psi):
    """unit tangent [T,2] of R from central finite differences (one-sided at the ends); R's heading where it barely moves."""
    xy = np.asarray(raw_xy, np.float64)
    d = np.gradient(xy, axis=0)
    n = np.linalg.norm(d, axis=-1)
    psi = np.where(n > 0.005, np.arctan2(d[:, 1], d[:, 0]), np.asarray(raw_psi, np.float64))
    return np.stack([np.cos(psi), np.sin(psi)], -1), psi


def decompose(sim_xy, raw_xy, raw_psi, t=None, win=10):
    """along / cross (left +) of S - R at step t (default: the step of max |S - R|) in R's local frame, and the cross
    component signed INSIDE (+) / OUTSIDE (-) of R's local turn over +-win steps (0 when |turn| < 2 deg)."""
    S, R = np.asarray(sim_xy, np.float64), np.asarray(raw_xy, np.float64)
    e = S - R
    if t is None:
        t = int(np.argmax(np.linalg.norm(e, axis=-1)))
    u, psi = local_frame(R, raw_psi)
    nrm = np.stack([-u[:, 1], u[:, 0]], -1)
    along, cross = float(e[t] @ u[t]), float(e[t] @ nrm[t])
    turn = float(wrap(psi[min(t + win, len(psi) - 1)] - psi[max(t - win, 0)]))
    sgn = 0.0 if abs(np.degrees(turn)) < STRAIGHT_DEG else float(np.sign(turn))
    return {"t": t, "along": along, "cross": cross, "cross_inside": cross * sgn if sgn else 0.0,
            "turn_deg": float(np.degrees(turn)), "straight": sgn == 0.0}


def families(sim_xy, sim_psi, raw_xy, raw_psi, dt=0.1):
    """LONGITUDINAL speed MAE, LATERAL mean |cross| and heading MAE vs R's path tangent (deg), and delta."""
    S, R = np.asarray(sim_xy, np.float64), np.asarray(raw_xy, np.float64)
    vS = np.linalg.norm(np.diff(S, axis=0), axis=-1) / dt
    vR = np.linalg.norm(np.diff(R, axis=0), axis=-1) / dt
    u, psi = local_frame(R, raw_psi)
    nrm = np.stack([-u[:, 1], u[:, 0]], -1)
    cross = np.abs(((S - R) * nrm).sum(-1))
    hmae = np.degrees(np.abs(wrap(np.asarray(sim_psi, np.float64) - psi)))
    return {"speed_mae": float(np.abs(vS - vR).mean()), "cross_mae": float(cross.mean()),
            "heading_mae_deg": float(hmae.mean()), "delta": self_deviation(S, R)}


def manoeuvre(final_heading_rad, thr=MAN_DEG):
    d = float(np.degrees(wrap(final_heading_rad)))
    return "LEFT" if d >= thr else ("RIGHT" if d <= -thr else "STRAIGHT")


def fb_pick(score, delta, pick, rho):
    """F_b: among {j : s_j >= rho * max s}, argmin delta (ties: higher s, then lower index); switch iff
    delta[pick] - delta[j*] >= SWITCH_MIN_M."""
    s, d = np.asarray(score, np.float64), np.asarray(delta, np.float64)
    cand = np.flatnonzero(s >= rho * s.max())
    order = np.lexsort((cand, -s[cand], d[cand]))                      # primary delta, then -s, then index
    j = int(cand[order[0]])
    return j if (d[pick] - d[j]) >= SWITCH_MIN_M else int(pick)


def comp_step(r8, p8, tracked8):
    """one P4 compensation update r <- r + (p - track(r)) at the 8 knots (xy, wrapped heading)."""
    r = np.array(r8, np.float64, copy=True)
    e = np.concatenate([p8[:, :2] - tracked8[:, :2], wrap(p8[:, 2] - tracked8[:, 2])[:, None]], -1)
    r = r + e
    r[:, 2] = wrap(r[:, 2])
    return r
