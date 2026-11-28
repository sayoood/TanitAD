"""H-ST-RECOVERYLABEL zero-training prototype: a kinematically feasible recovery label for an off-route own state.

PROTOTYPE, lives in the bank_presteps package, NOT in stack/. Numpy only. Implements FROZEN_CARDS.md readings L1-L8
(sha256 in ../FROZEN_CARDS.sha256). Nothing here is tuned on data: every constant is fixed in the frozen file.

Geometry (world frame, 0.2 s grid, H = 30 waypoints, label k = 1..30 relative to the own state at k = 0)  [AMENDMENT 1 design]:
  * speed profile v_l: starts at the OWN speed, follows the human speed v_h from the matched row r' with a rate limit
    0.99 * A_MAX once unmerged, and follows the human unconditionally once equal (so zero speed offset == human, bit for bit);
  * label distance l[k] = sum_i 0.5 (v_l[i-1] + v_l[i]) * dt  (the label MOVES at v_l along its own path: timing is consistent by construction);
  * path = cubic Hermite blend from the own pose (tangent = own heading) to the human polyline point at the human's row r'+k_j
    (tangent = the polyline's central chord there), THEN the human polyline from that point; the label sits at path arc l[k];
  * target ladder k_j in LADDER; the first rung whose RECOVERY PHASE (steps until the label has arrived at the target, +1) passes the
    independent feasibility checker is emitted; a rung the label cannot reach within 6 s is infeasible.
"""
from __future__ import annotations

import math

import numpy as np

DT = 0.2
H = 30
A_MAX = 4.0           # feasible_decode.A_MAX_MPS2
KAPPA_MAX = 0.2       # feasible_decode.KAPPA_MAX_1PM
MU_KAMM = 0.7         # feasible_decode.MU_KAMM
G_MPS2 = 9.80665
MIN_SPEED = 0.5       # feasible_decode.MIN_SPEED_MPS (curvature denominator floor)
MARGIN = 1e-2         # feasible_decode.MARGIN
A_LIM = A_MAX * (1.0 - MARGIN)
K_LIM = KAPPA_MAX * (1.0 - MARGIN)
F_LIM = MU_KAMM * G_MPS2 * (1.0 - MARGIN)
LADDER = (5, 10, 15, 20, 25, 30)
N_HERMITE = 161
STOP_STEP_M = 1e-9    # a step shorter than this holds the previous heading (feasible_decode's stopped-step rule, in spirit)

STATUS_EMITTED = "emitted"
STATUS_IDENTITY = "emitted_identity"     # zero state offset: the human label itself
REFUSALS = ("refused_infeasible", "no_logged_future", "nonfinite", "path_exhausted", "degenerate")


# ------------------------------------------------------------------------------------------------ frames
def to_ego(points_world, pose):
    """World points [..., 2] -> the ego frame of ``pose`` = (x, y, yaw). Same expression as taniteval.floors_kin.gt_waypoints."""
    p = np.asarray(points_world, np.float64)
    d = p - np.asarray(pose[:2], np.float64)
    c, s = np.cos(-pose[2]), np.sin(-pose[2])
    return np.stack([d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c], axis=-1)


def human_label(grid, r):
    """The human's GT label at logged row r: ``grid`` [T, >=3] (x, y, yaw) on the 0.2 s grid -> [30, 2] in the ego frame of row r.
    Identical arithmetic to ``taniteval.floors_kin.gt_waypoints(poses, [r], 1..30)[0]``."""
    P = np.asarray(grid, np.float64)
    r = int(r)
    d = P[r + 1:r + 1 + H, :2] - P[r, :2]
    c, s = np.cos(-P[r, 2]), np.sin(-P[r, 2])
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], axis=-1)


# ------------------------------------------------------------------------------------------------ independent feasibility checker
def controls(points, dt: float = DT):
    """points [n, 2] (index 0 first) -> (speed [n-1], accel [n-2], lat_acc [n-2], kappa [n-2]) in feasible_decode.recover_controls'
    convention (speed = chord / dt; heading = atan2 of the step, a zero-length step HOLDS the previous heading; yaw_rate = wrapped
    heading difference / dt; v_mid = max(speed[k], MIN_SPEED); kappa = yaw_rate / v_mid; lat_acc = v_mid * yaw_rate)."""
    p = np.asarray(points, np.float64)
    d = p[1:] - p[:-1]
    ln = np.hypot(d[:, 0], d[:, 1])
    speed = ln / dt
    head = np.arctan2(d[:, 1], d[:, 0])
    for i in range(len(head)):
        if ln[i] < STOP_STEP_M:
            head[i] = head[i - 1] if i > 0 else 0.0
    dth = head[1:] - head[:-1]
    dth = np.arctan2(np.sin(dth), np.cos(dth))
    yaw = dth / dt
    vmid = np.maximum(speed[:-1], MIN_SPEED)
    accel = (speed[1:] - speed[:-1]) / dt
    return speed, accel, vmid * yaw, yaw / vmid


def violations(label_own, v_own: float, upto: int, a_lim=A_LIM, k_lim=K_LIM, f_lim=F_LIM):
    """The feasibility checker. ``label_own`` [30, 2] in the OWN frame (heading 0); a virtual predecessor (-v_own*dt, 0) carries the own
    speed and heading into the first step. Controls are derived from [pred; origin; label 1 .. upto] (``upto`` <= 30) and compared with the
    limits. Returns dict(n_checked, accel, curv, fric, any, max_*)."""
    upto = int(min(max(upto, 2), H))
    pts = np.concatenate([[[-float(v_own) * DT, 0.0]], [[0.0, 0.0]], np.asarray(label_own, np.float64)[:upto]])
    sp, ac, lat, kap = controls(pts)
    fr = np.hypot(ac, lat)
    out = {"n_checked": int(len(ac)),
           "accel": int((np.abs(ac) > a_lim).sum()), "curv": int((np.abs(kap) > k_lim).sum()), "fric": int((fr > f_lim).sum()),
           "max_abs_accel": float(np.abs(ac).max()), "max_abs_kappa": float(np.abs(kap).max()), "max_friction": float(fr.max())}
    out["any"] = bool(out["accel"] or out["curv"] or out["fric"])
    return out


# ------------------------------------------------------------------------------------------------ the labeller
def label_speed(v_own: float, v_h, a_lim: float = A_LIM, clip: bool = True):
    """v_l [31]: starts at the own speed; follows the human once equal; otherwise moves toward the human at <= a_lim per second.
    ``clip=False`` is the deliberate-regression variant: the label speed JUMPS to the human's at once."""
    vh = np.asarray(v_h, np.float64)
    v = np.empty(len(vh))
    v[0] = v_own
    lim = a_lim * DT
    for k in range(1, len(vh)):
        if not clip or v[k - 1] == vh[k - 1]:
            v[k] = vh[k]
        else:
            diff = vh[k] - v[k - 1]
            v[k] = vh[k] if abs(diff) <= lim else v[k - 1] + math.copysign(lim, diff)
    return v


def _polyline(P):
    """Cumulative arc s [n] of polyline points P [n, 2]."""
    seg = np.hypot(*(P[1:] - P[:-1]).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _at_arc(P, s_all, s):
    """Points of polyline P at arc positions s (array)."""
    return np.stack([np.interp(s, s_all, P[:, 0]), np.interp(s, s_all, P[:, 1])], axis=-1)


def _tangent_at(P, s_all, s0, half: float = 0.6):
    """Unit tangent of the polyline at arc s0, as the CENTRAL chord across [s0 - half, s0 + half] (clipped to the path), which for a
    polyline with a vertex at s0 is the symmetric direction rather than one of the two segment directions (a half-step turn-angle bias).
    Falls back to the nearest non-degenerate segment direction; None if the whole path is a point."""
    lo, hi = max(0.0, s0 - half), min(float(s_all[-1]), s0 + half)
    if hi - lo > 1e-9:
        a = _at_arc(P, s_all, np.array([lo, hi]))
        d = a[1] - a[0]
        n = math.hypot(d[0], d[1])
        if n > 1e-9:
            return d / n
    i = int(np.searchsorted(s_all, s0, side="right")) - 1
    i = min(max(i, 0), len(P) - 2)
    for j in list(range(i, len(P) - 1)) + list(range(i - 1, -1, -1)):
        d = P[j + 1] - P[j]
        n = math.hypot(d[0], d[1])
        if n > 1e-9:
            return d / n
    return None


def _hermite(q, t0, T, t1):
    """Cubic Hermite from q (tangent t0) to T (tangent t1), tangent magnitudes = chord; returns (pts [N,2], arc [N])."""
    c = float(np.hypot(*(T - q)))
    u = np.linspace(0.0, 1.0, N_HERMITE)[:, None]
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    pts = h00 * q + h10 * (c * t0) + h01 * T + h11 * (c * t1)
    pts[0], pts[-1] = q, T                      # exact endpoints
    return pts, _polyline(pts)


def recovery_label(grid, v_log, r: int, own_pose, v_own: float, *, clip: bool = True, check: bool = True, ladder=LADDER):
    """Recovery label for an own state at ``own_pose`` = (x, y, yaw) [world] with speed ``v_own``, matched to logged row ``r``.

    ``grid`` [T, >=3] logged (x, y, yaw) on the 0.2 s grid; ``v_log`` [T] logged speed (the native track's v column on the same rows).
    Returns dict: status, label_own [30, 2] (own frame), label_world [31, 2] (k = 0..30), k_rej, viol (checker dict or None), n_rungs_tried.
    ``clip=False, check=False`` is the 'no-clip' deliberate-regression labeller (speed jump, first rung emitted unchecked)."""
    G = np.asarray(grid, np.float64)
    T = len(G)
    r = int(r)
    q = np.asarray(own_pose[:2], np.float64)
    psi = float(own_pose[2])
    out = {"status": None, "label_own": None, "label_world": None, "k_rej": None, "viol": None, "n_rungs_tried": 0}
    if r < 0 or r + H > T - 1:
        out["status"] = "no_logged_future"
        return out
    Ph = G[r:, :2]                                   # human polyline from the matched row
    vh = np.asarray(v_log, np.float64)[r:r + H + 1]
    if not (np.isfinite(Ph[:H + 1]).all() and np.isfinite(vh).all() and np.isfinite(G[r, 2])
            and np.isfinite(q).all() and math.isfinite(psi) and math.isfinite(v_own)):
        out["status"] = "nonfinite"
        return out
    # --- identity branch: the own state IS the logged state (bit-equal pose and speed) -> the human label itself
    if q[0] == G[r, 0] and q[1] == G[r, 1] and psi == G[r, 2] and float(v_own) == vh[0]:
        out.update(status=STATUS_IDENTITY, label_world=Ph[:H + 1].copy(), k_rej=0)
        out["label_own"] = to_ego(Ph[1:H + 1], (q[0], q[1], psi))
        return out
    s_h_all = _polyline(Ph)
    s_end = float(s_h_all[-1])
    vl = label_speed(float(v_own), vh, clip=clip)
    ell = np.concatenate([[0.0], np.cumsum(0.5 * (vl[:-1] + vl[1:]) * DT)])       # label distance travelled at k = 0..30
    t0 = np.array([math.cos(psi), math.sin(psi)])
    tried = 0
    any_infeasible = False
    last = "degenerate"
    for k_j in ladder:
        tried += 1
        S_star = float(s_h_all[k_j])                                               # arc of the human's position at row r + k_j
        T_star = Ph[k_j]
        t1 = _tangent_at(Ph, s_h_all, S_star)
        if t1 is None or float(np.hypot(*(T_star - q))) < 1e-9:
            last = "degenerate"
            continue
        hp, ha = _hermite(q, t0, T_star, t1)
        B = float(ha[-1])
        if float(ell[H]) > B + (s_end - S_star) + 1e-9:
            last = "path_exhausted"
            continue
        arrived = ell >= B                                                          # k at which the label has reached the target
        k_e = int(np.argmax(arrived)) if arrived.any() else None
        # position at label distance l: on the Hermite while l <= B, on the human polyline (arc S* + l - B) afterwards
        on_h = ell <= B
        pos = np.empty((H + 1, 2))
        if on_h.any():
            pos[on_h] = _at_arc(hp, ha, ell[on_h])
        if (~on_h).any():
            pos[~on_h] = _at_arc(Ph, s_h_all, S_star + (ell[~on_h] - B))
        pos[0] = q
        Lo = to_ego(pos[1:], (q[0], q[1], psi))
        if k_e is None:                                                             # the label never reaches the target within 6 s
            last = "refused_infeasible"
            any_infeasible = True
            continue
        v = violations(Lo, float(v_own), k_e + 1) if check else None
        if (not check) or (not v["any"]):
            out.update(status=STATUS_EMITTED, label_world=pos, label_own=Lo, k_rej=int(k_e), viol=v, k_target=int(k_j))
            out["n_rungs_tried"] = tried
            return out
        any_infeasible = True
    out["status"] = "refused_infeasible" if any_infeasible else last
    out["n_rungs_tried"] = tried
    return out


def label_ignore_offset(grid, v_log, r: int, own_pose, v_own: float):
    """DELIBERATE-REGRESSION arm: ignores the offset entirely. The label is the human's path in the WORLD frame (as if the car were on
    it), expressed in the own frame like any other label. It is what the existing human-label pipeline implies for an off-route state."""
    G = np.asarray(grid, np.float64)
    r = int(r)
    if r < 0 or r + H > len(G) - 1:
        return {"status": "no_logged_future", "label_own": None, "label_world": None, "k_rej": None, "viol": None, "n_rungs_tried": 0}
    Ph = G[r:r + H + 1, :2]
    Lw = Ph.copy()
    Lo = to_ego(Lw[1:], (own_pose[0], own_pose[1], own_pose[2]))
    return {"status": STATUS_EMITTED, "label_own": Lo, "label_world": Lw, "k_rej": 0, "viol": None, "n_rungs_tried": 0}


def gate_violation_count(label_own, v_own, upto: int) -> int:
    """G-REC component: number of checker violations (accel + curvature + friction) in the first ``upto`` label steps."""
    v = violations(label_own, v_own, upto)
    return int(v["accel"] + v["curv"] + v["fric"])
