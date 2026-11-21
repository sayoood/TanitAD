"""POST-HOC variant V2 of the H-ST-RECOVERYLABEL prototype (see FROZEN_AMENDMENT_2.md): the target ladder is an ARC LOOKAHEAD along the human
polyline, not the human's position at row r' + k_j. Everything else (speed profile, label distance, Hermite blend, checker, identity branch)
is labeller.py's, imported unchanged. V1 (labeller.py) stays the registered result; V2 is reported beside it and labelled post-hoc.
"""
from __future__ import annotations

import math

import numpy as np

import labeller as L

ARC_LADDER = (6.0, 10.0, 15.0, 22.0, 32.0, 45.0, 60.0)      # metres of human-polyline arc ahead of the matched row; fixed in Amendment 2 before any V2 output was read


def recovery_label(grid, v_log, r: int, own_pose, v_own: float, *, clip: bool = True, check: bool = True, ladder=ARC_LADDER):
    G = np.asarray(grid, np.float64)
    T = len(G)
    r = int(r)
    q = np.asarray(own_pose[:2], np.float64)
    psi = float(own_pose[2])
    H = L.H
    out = {"status": None, "label_own": None, "label_world": None, "k_rej": None, "viol": None, "n_rungs_tried": 0}
    if r < 0 or r + H > T - 1:
        out["status"] = "no_logged_future"
        return out
    Ph = G[r:, :2]
    vh = np.asarray(v_log, np.float64)[r:r + H + 1]
    if not (np.isfinite(Ph[:H + 1]).all() and np.isfinite(vh).all() and np.isfinite(G[r, 2]) and np.isfinite(q).all() and math.isfinite(psi) and math.isfinite(v_own)):
        out["status"] = "nonfinite"
        return out
    if q[0] == G[r, 0] and q[1] == G[r, 1] and psi == G[r, 2] and float(v_own) == vh[0]:
        out.update(status=L.STATUS_IDENTITY, label_world=Ph[:H + 1].copy(), k_rej=0)
        out["label_own"] = L.to_ego(Ph[1:H + 1], (q[0], q[1], psi))
        return out
    s_h_all = L._polyline(Ph)
    s_end = float(s_h_all[-1])
    vl = L.label_speed(float(v_own), vh, clip=clip)
    ell = np.concatenate([[0.0], np.cumsum(0.5 * (vl[:-1] + vl[1:]) * L.DT)])
    t0 = np.array([math.cos(psi), math.sin(psi)])
    tried, any_infeasible, last = 0, False, "degenerate"
    for S_star in ladder:
        tried += 1
        if S_star > s_end:
            last = "path_exhausted"
            continue
        T_star = L._at_arc(Ph, s_h_all, np.array([S_star]))[0]
        t1 = L._tangent_at(Ph, s_h_all, S_star)
        if t1 is None or float(np.hypot(*(T_star - q))) < 1e-9:
            last = "degenerate"
            continue
        hp, ha = L._hermite(q, t0, T_star, t1)
        B = float(ha[-1])
        if float(ell[H]) > B + (s_end - S_star) + 1e-9:
            last = "path_exhausted"
            continue
        arrived = ell >= B
        k_e = int(np.argmax(arrived)) if arrived.any() else None
        on_h = ell <= B
        pos = np.empty((H + 1, 2))
        if on_h.any():
            pos[on_h] = L._at_arc(hp, ha, ell[on_h])
        if (~on_h).any():
            pos[~on_h] = L._at_arc(Ph, s_h_all, S_star + (ell[~on_h] - B))
        pos[0] = q
        Lo = L.to_ego(pos[1:], (q[0], q[1], psi))
        if k_e is None:
            any_infeasible = True
            continue
        v = L.violations(Lo, float(v_own), k_e + 1) if check else None
        if (not check) or (not v["any"]):
            out.update(status=L.STATUS_EMITTED, label_world=pos, label_own=Lo, k_rej=int(k_e), viol=v, k_target=float(S_star))
            out["n_rungs_tried"] = tried
            return out
        any_infeasible = True
    out["status"] = "refused_infeasible" if any_infeasible else last
    out["n_rungs_tried"] = tried
    return out
