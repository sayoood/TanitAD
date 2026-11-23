"""H-ST-RECOVERYLABEL V1 (registered): a kinematically feasible recovery label for an off-route own state, ported VERBATIM into stack/ (PREREG_V8F_ROLLHIST, the label lever ``--dagger-label recovery_v1``).

SOURCE (research prototype, bank_presteps/recovery/labeller.py): sha256 deb258d14f3e47fb53d267fd3c6e543a74ce520f2857fcc7b408eb6ecd4dac60. The body from ``from __future__`` to ``gate_violation_count`` below is that file byte for byte (``stack/tests/test_v8f_recovery.py`` compares the
function sources' sha256 with the digest pinned there); only this docstring and the INTEGRATION section at the end are new. V2 (labeller_v2.py, post-hoc) is NOT here.

THE SHIPPED LABEL IS NOT THE V1 POSE-RECOVERY LABEL (MM design change 2026-11-22): ``swap_rows`` re-anchors the own poses onto the LOGGED NOW at r', so the own pose offset is invisible to every input and a label that
recovers it would be unpredictable. The trainer's ``--dagger-label speed_v1`` uses ``speed_label`` below (ONE speed source: the human polyline's own chords; the logged speed column is not used), which REUSES from V1 only ``violations`` (the feasibility checker),
``_polyline`` / ``_at_arc`` and ``human_label``. ``recovery_label`` / ``label_ignore_offset`` stay as the verbatim, tested research code and are no longer reachable from the trainer.

Geometry (world frame, 0.2 s grid, H = 30 waypoints, label k = 1..30 relative to the own state at k = 0)  [AMENDMENT 1 design]:
  * speed profile v_l: starts at the OWN speed, follows the human speed v_h from the matched row r' with a rate limit 0.99 * A_MAX once unmerged, and follows the human unconditionally once equal (so zero speed offset == human, bit for bit);
  * label distance l[k] = sum_i 0.5 (v_l[i-1] + v_l[i]) * dt  (the label MOVES at v_l along its own path: timing is consistent by construction);
  * path = cubic Hermite blend from the own pose (tangent = own heading) to the human polyline point at the human's row r'+k_j (tangent = the polyline's central chord there), THEN the human polyline from that point;
  * target ladder k_j in LADDER; the first rung whose RECOVERY PHASE passes the independent feasibility checker is emitted; a rung the label cannot reach within 6 s is infeasible.
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


# ================================================================================================ INTEGRATION (new; the labeller above is verbatim)
DAGGER_LABELS = ("human", "speed_v1", "kin_v1")  # the ``--dagger-label`` choices; "human" is the default and is today's swap bit for bit; "kin_v1" = KINLAB (section at the end of this file)
EMITTED = (STATUS_EMITTED, STATUS_IDENTITY)
_OK = 0                                         # v8f_dagger.OK (not imported: this module stays numpy-only)
SPEED_STATUSES = (STATUS_EMITTED, STATUS_IDENTITY, "no_logged_future", "nonfinite", "path_exhausted", "arcs_decreasing", "refused_infeasible")


def arc_from_label(label_own):
    """``s_h`` re-derived from a label: [N, 30, 2] (the sample's frame, knots 1..30) -> float32 [N, 30], the path length from NOW to knot i + 1 (the same quantity as ``v8f_mini_data.human_arc``)."""
    L = np.asarray(label_own, np.float64)
    pts = np.concatenate([np.zeros((L.shape[0], 1, 2)), L], axis=1)
    seg = np.hypot(pts[:, 1:, 0] - pts[:, :-1, 0], pts[:, 1:, 1] - pts[:, :-1, 1])
    return np.cumsum(seg, axis=1).astype(np.float32)


A_LABEL = 3.0                                    # m/s2: the label's speed-change rate (0.75 of A_MAX; the margin absorbs the chord-against-arc difference on curves; fixed by principle, not from data)


def label_steps(v_own: float, u_h, a_rate: float = None, clip: bool = True):
    """The label's step speeds u_l [K+1] from the human's step speeds ``u_h`` [K+1] (index 0 unused: the own speed stands there). u_l[0] = v_own; for k >= 1 the label moves toward u_h[k] by at most ``a_rate`` * dt and,
    from the first k at which |u_h[k] - u_l[k-1]| <= a_rate * dt (or u_l[k-1] == u_h[k]), follows u_h unconditionally. ``clip=False`` is the deliberate-regression variant: the speed JUMPS to the human's at k = 1."""
    a = A_LABEL if a_rate is None else a_rate
    uh = np.asarray(u_h, np.float64)
    ul = np.empty(len(uh))
    ul[0] = v_own
    lim = a * DT
    merged = False
    for k in range(1, len(uh)):
        if not clip or merged or ul[k - 1] == uh[k]:
            ul[k], merged = uh[k], True
        else:
            diff = uh[k] - ul[k - 1]
            if abs(diff) <= lim:
                ul[k], merged = uh[k], True
            else:
                ul[k] = ul[k - 1] + math.copysign(lim, diff)
    return ul


def speed_label(grid, r: int, v_own: float, *, clip: bool = True, check: bool = True, a_rate: float = None):
    """The SPEED-CONSISTENT label (``--dagger-label speed_v1``, ONE speed source) for an own state matched (ok) to logged row ``r``, in the HUMAN'S ego frame at r' (= the frame of a re-anchored swapped sample).

    ``grid`` [T, >=3] logged (x, y, yaw); ``v_own`` the own speed (the one the model reads). P = the human polyline (rows r', r'+1, ... in the frame at r'), s its cumulative arc; the human's step speeds are the polyline's own chords,
    u_h[k] = (s[k] - s[k-1]) / dt. The label's step speeds u_l = ``label_steps(v_own, u_h)`` (rate-limited at A_LABEL from the own speed); the label arc L[k] = sum_{i<=k} u_l[i] dt; label point k = P at arc L[k]. After the merge
    L[k] - s[k] is constant: the label is the human's path shifted along itself. The logged speed column is NOT used anywhere.
    Refused (the caller keeps the human label) when L runs past the polyline's end (path_exhausted), L is not non-decreasing (arcs_decreasing), or ``violations(label, v_own, 30)`` finds any (refused_infeasible; ``viol`` carries the terms).
    Identity: u_l[k] == u_h[k] for every k >= 1 -> ``human_label(grid, r)`` itself. ``clip=False`` and ``check=False`` are the deliberate-regression variants.
    Returns dict(status, label [30, 2] | None, delta [31] | None (= L - s), u_h1 | None (= u_h[1]), viol | None)."""
    G = np.asarray(grid, np.float64)
    r = int(r)
    out = {"status": None, "label": None, "delta": None, "u_h1": None, "viol": None}
    if r < 0 or r + H > len(G) - 1:
        out["status"] = "no_logged_future"
        return out
    if not (np.isfinite(G[r:r + H + 1, :3]).all() and math.isfinite(float(v_own))):
        out["status"] = "nonfinite"
        return out
    Gp = G[r:]
    P = to_ego(Gp[:, :2], (Gp[0, 0], Gp[0, 1], Gp[0, 2]))                           # the human polyline in the human's frame at r'
    s = _polyline(P)
    uh = np.concatenate([[0.0], (s[1:H + 1] - s[:H]) / DT])                         # u_h[k] = (s[k] - s[k-1]) / dt, k = 1..30
    out["u_h1"] = float(uh[1])
    ul = label_steps(float(v_own), uh, a_rate=a_rate, clip=clip)
    L = np.concatenate([[0.0], np.cumsum(ul[1:]) * DT])
    out["delta"] = L - s[:H + 1]
    if np.array_equal(ul[1:], uh[1:]):                                              # identity: bit-equal to today's label
        out.update(status=STATUS_IDENTITY, label=human_label(G, r))
        return out
    if float(L[-1]) > float(s[-1]) + 1e-9:
        out["status"] = "path_exhausted"
        return out
    if check and bool((L[1:] < L[:-1]).any()):
        out["status"] = "arcs_decreasing"
        return out
    lab = _at_arc(P, s, L)[1:]
    if check:
        v = violations(lab, float(v_own), H)
        out["viol"] = v
        if v["any"]:
            out["status"] = "refused_infeasible"
            return out
    out.update(status=STATUS_EMITTED, label=lab)
    return out


def label_bank(arrays: dict, ci, clip_names, read_clip, *, log=None, clip: bool = True, check: bool = True) -> dict:
    """``speed_label`` for every bank record whose matcher reason is ok (once, at bank load; deterministic float64, no RNG). Off-route / end_edge / nonfinite records get no label (they stay cut, as today).

    ``arrays`` the bank record arrays (``DaggerBank.a``: ``p10f`` [N, F, 8], ``r``, ``reason``, ``tick``); ``ci`` the trainer's join index per record; ``clip_names`` the join's clip names; ``read_clip(name)`` ->
    ``{"pose": [T, 4]}`` (``ClipStore.get_kin``: x, y, yaw, v of the LOGGED 0.2 s grid). The own speed is ``p10f[:, -1, 0]`` (V1's source).
    Returns ``{"slot": int32 [N] (-1 = none), "label": float64 [n_emit, 30, 2], "status_counts", "n_labelled", "census": ...}``; ``census`` = per tick {ok, relabelled, identity, refused by reason} + p50 / p95 of
    |v_own - u_h[1]| (u_h = the human polyline's own chord speeds) and of |L[30] - s[30]| over the ok records that could be read."""
    reason = np.asarray(arrays["reason"])
    p10f, rr = arrays["p10f"], np.asarray(arrays["r"], np.int64)
    tick = np.asarray(arrays["tick"], np.int64) if "tick" in arrays else np.zeros(len(reason), np.int64)
    ci = np.asarray(ci, np.int64)
    todo = np.flatnonzero(reason == _OK)
    order = todo[np.argsort(ci[todo], kind="stable")]
    slot = np.full(len(reason), -1, np.int32)
    labels, counts, n_done = [], {}, 0
    st = np.full(len(reason), -1, np.int8)
    dsp, d30 = np.full(len(reason), np.nan), np.full(len(reason), np.nan)
    cur, grid = None, None
    vt = np.zeros(len(reason), np.int8)                                              # checker terms of a refused_infeasible record: 1 accel, 2 curv, 4 fric (bit set)
    for i in order:
        c = int(ci[i])
        if c != cur:
            d = read_clip(str(clip_names[c]))
            pg = np.asarray(d["pose"], np.float64)
            grid, cur = pg[:, :3], c
        v_own, r = float(p10f[i, -1, 0]), int(rr[i])
        res = speed_label(grid, r, v_own, clip=clip, check=check)
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        st[i] = SPEED_STATUSES.index(res["status"])
        if res["u_h1"] is not None:
            dsp[i] = abs(v_own - res["u_h1"])
        if res["viol"] is not None:
            vt[i] = (1 if res["viol"]["accel"] else 0) | (2 if res["viol"]["curv"] else 0) | (4 if res["viol"]["fric"] else 0)
        if res["delta"] is not None:
            d30[i] = abs(res["delta"][H])
        if res["status"] in EMITTED:
            slot[i] = len(labels)
            labels.append(res["label"])
        n_done += 1
        if log is not None and n_done % 100000 == 0:
            log(f"[speed_v1] labelled {n_done}/{len(order)}")
    lab = np.stack(labels).astype(np.float64) if labels else np.zeros((0, H, 2))
    return {"slot": slot, "label": lab, "status_counts": counts, "n_labelled": int(n_done), "census": speed_census(reason, tick, st, dsp, d30, vt)}


def speed_census(reason, tick, st, dsp, d30, vt=None) -> dict:
    """The bank census of ``--dagger-label speed_v1``: per tick, counts of ok records, relabelled (emitted, delta != 0), identity, and refused by reason; refused_infeasible by checker term (accel / curv / fric, not exclusive; ``vt`` bitmask 1/2/4); p50 / p95 of |v_own - u_h[1]| and |L[30] - s[30]|; the relabelled share of ok records."""
    ok = np.asarray(reason) == _OK
    tick = np.asarray(tick)
    pct = lambda a: ({"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "n": int(len(a))} if len(a) else {"p50": None, "p95": None, "n": 0})
    E, I = SPEED_STATUSES.index(STATUS_EMITTED), SPEED_STATUSES.index(STATUS_IDENTITY)
    out = {"per_tick": {}}
    for t in sorted(set(int(x) for x in tick[ok])) or [0]:
        m = ok & (tick == t)
        row = {"ok": int(m.sum()), "relabelled": int((m & (st == E)).sum()), "identity": int((m & (st == I)).sum())}
        row["refused"] = {k: int((m & (st == i)).sum()) for i, k in enumerate(SPEED_STATUSES) if k not in EMITTED and (m & (st == i)).any()}
        if vt is not None:
            row["refused_by_term"] = {"accel": int((m & (vt & 1 > 0)).sum()), "curv": int((m & (vt & 2 > 0)).sum()), "fric": int((m & (vt & 4 > 0)).sum())}
        out["per_tick"][str(t)] = row
    out["abs_v_own_minus_u_h1"] = pct(dsp[ok & np.isfinite(dsp)])
    out["abs_delta30"] = pct(d30[ok & np.isfinite(d30)])
    n_ok = int(ok.sum())
    out["relabelled_share_of_ok"] = (int((ok & (st == E)).sum()) / n_ok) if n_ok else None
    return out


# ================================================================================================ KINLAB (``--dagger-label kin_v1``)
"""KINLAB: the KINEMATIC label (MM design, BRIEF_BUILD_KINLAB 2026-11-23, binding; every constant below is that brief's, none is fitted to data).

WHY. In an own-state sample the model READS the own kinematics (kin3 = (v, a, kappa), ``p10f`` derived from it, ``base`` = O's prior path), but ``gt`` follows the human path with the human's heading and curvature from NOW (``speed_v1`` changes only the arc
timing). MEASURED (phase 1, RH bank s0 shard 0, n = 372,791): own-vs-human yaw-rate gap p50 1.84 / p95 23.5 deg/s (human-to-human p95 1.78), acceleration gap p50 1.55 / p95 6.42 m/s2 (human-to-human p95 0.58). The label contradicts the prior path the model
is given. ``kin_v1`` makes the label START where the inputs say the car is going and then merge into the human's path.

THE LABEL (per matcher-ok record; the sample frame = the human's frame at r', origin = NOW, heading 0):
  1. own start (v_o, a_o, kappa_o) = ``p10f[-1, 0:3]`` of the record (the exact values the model reads);
  2. reference = the record's ``speed_v1`` label (its step speeds u_l[k] and the human polyline P it samples); if ``speed_label`` refuses the record, ``kin_v1`` refuses it too (reason ``speed_<status>``);
  3. unicycle integration, sub-step H_SUB = 0.05 s for T_END = 6.0 s (120 sub-steps), output every 0.2 s (30 points) = ``gt``; ``s_h`` is re-derived by ``arc_from_label`` exactly as under speed_v1.
     blend w(t) = min(1, t / T_B), T_B = 1.0 s.  Longitudinal a(t) = (1 - w) a_o + w a_rule(t), |a| <= A_LIM, v >= 0, where a_rule(t) = clip((u_h(t) - v(t)) / DT, +-A_LABEL) is the speed_v1 RULE evaluated on the INTEGRATED state v(t)
     (u_h(t) = the human's step speed, linearly interpolated between the 0.2 s knots; knot k = u_h[k], knot 0 = u_h[1]). [PHASE 3 (MM construction fix, BRIEF_BUILD_KINLAB3 2026-11-23): phase 2 used the speed_v1 per-step accelerations, which never pull the
     integrated speed back to the reference, so the merge tail jumped the speed (98.6 % of the acceleration refusals sat on the merge step).]
     Lateral kappa(t) = (1 - w) kappa_o + w kappa_pp(t): pure pursuit to P, lookahead at arc s* + L_d from the closest point s*, L_d = max(LD_MIN, LD_GAIN * v), kappa_pp = 2 sin(alpha) / L_d; |kappa| <= K_LIM, and hypot(a, v^2 kappa) <= F_LIM by scaling kappa down first.
  4. merge: at the first output point with w = 1, lateral distance to P < MERGE_LAT_M, heading error to P's tangent < MERGE_HEAD_RAD AND |v - u_h| <= MERGE_SPEED_MS (all at the same output point), the remaining points are P at the cumulative arc
     s*_m + sum of the step speeds of the speed_v1 rule started from the merged v (``label_steps(v_m, u_h[k_m:])``: toward u_h[k] by at most A_LABEL * DT per step, then u_h once equal; no step changes v by more than A_LABEL * DT).
  5. refusal (the caller keeps the speed_v1 label of that record, else the human label), each counted: ``refused_nonfinite``; ``refused_not_merged`` (not merged by t = 6.0 s); ``refused_path_exhausted`` (a lookahead or the merged arc runs past P's end);
     ``refused_infeasible`` (``violations`` of the final label, from v_o); ``speed_<status>`` (the reference itself was refused).
  6. identity rule: |kappa_o - kappa_ref0| <= IDENT_TOL and |a_o - a_ref0| <= IDENT_TOL -> the record's label is the speed_v1 label bit for bit. a_ref0 = (u_l[1] - u_l[0]) / DT; kappa_ref0 = the reference label's first-step path curvature
     (the turn between its chords 0->p1 and p1->p2 over their mean length; 0 when a chord is shorter than STOP_STEP_M).
"""
T_B = 1.0               # s, the blend horizon
LD_MIN = 4.0            # m, the lookahead floor
LD_GAIN = 1.0           # s, the lookahead's speed gain (L_d = max(LD_MIN, LD_GAIN * v))
MERGE_LAT_M = 0.10      # m
MERGE_HEAD_RAD = math.radians(1.0)
MERGE_SPEED_MS = 0.05   # m/s, the merge's speed condition |v - u_h| (phase 3)
H_SUB = 0.05            # s, the integration sub-step (DT / H_SUB = 4 sub-steps per output point)
T_END = 6.0             # s = H * DT
IDENT_TOL = 1e-6        # the identity rule's tolerance on kappa and a
SUB_PER_STEP = int(round(DT / H_SUB))
KIN_REFUSALS = ("refused_nonfinite", "refused_not_merged", "refused_path_exhausted", "refused_infeasible")
KIN_STATUSES = ("kin_emitted", "identity_rule") + KIN_REFUSALS + tuple("speed_" + s for s in ("no_logged_future", "nonfinite", "path_exhausted", "arcs_decreasing", "refused_infeasible"))
assert abs(SUB_PER_STEP * H_SUB - DT) < 1e-12 and int(round(T_END / H_SUB)) == H * SUB_PER_STEP


def _wrap(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def _closest(Pl, sl, q, i0: int):
    """Closest point of the polyline (lists ``Pl`` [(x, y)], ``sl`` arc) to q = (x, y), searching forward from segment i0 (monotone: the pursuit never goes back along the path) and stopping 3 segments past the best one.
    Returns (segment, arc s*, distance, segment heading)."""
    n = len(Pl) - 1
    best = None
    i = i0
    qx, qy = q
    while i < n:
        ax, ay = Pl[i]
        dx, dy = Pl[i + 1][0] - ax, Pl[i + 1][1] - ay
        d2 = dx * dx + dy * dy
        t = 0.0 if d2 < 1e-18 else min(1.0, max(0.0, ((qx - ax) * dx + (qy - ay) * dy) / d2))
        fx, fy = ax + t * dx, ay + t * dy
        dist = math.hypot(qx - fx, qy - fy)
        if best is None or dist < best[2] - 1e-12:
            best = (i, sl[i] + t * math.sqrt(d2), dist)
        elif i > best[0] + 3:
            break
        i += 1
    if best is None:                                   # a one-point polyline: nothing to follow
        return i0, sl[-1], math.hypot(qx - Pl[-1][0], qy - Pl[-1][1]), 0.0
    j = best[0]
    hd = None
    for jj in list(range(j, n)) + list(range(j - 1, -1, -1)):          # the heading of the nearest non-degenerate segment
        dx, dy = Pl[jj + 1][0] - Pl[jj][0], Pl[jj + 1][1] - Pl[jj][1]
        if dx * dx + dy * dy > 1e-18:
            hd = math.atan2(dy, dx)
            break
    return best[0], best[1], best[2], (0.0 if hd is None else hd)


def _point_at(Pl, sl, s):
    """The polyline point at arc s (clamped), linear interpolation."""
    import bisect
    i = min(max(bisect.bisect_right(sl, s) - 1, 0), len(Pl) - 2)
    d = sl[i + 1] - sl[i]
    t = 0.0 if d < 1e-18 else (s - sl[i]) / d
    return Pl[i][0] + t * (Pl[i + 1][0] - Pl[i][0]), Pl[i][1] + t * (Pl[i + 1][1] - Pl[i][1])


def ref_start(label, u_l):
    """(a_ref0, kappa_ref0): the reference's first-step acceleration and curvature (the identity rule's comparison values)."""
    a_ref0 = (float(u_l[1]) - float(u_l[0])) / DT
    p = np.concatenate([[[0.0, 0.0]], np.asarray(label, np.float64)[:2]])
    d = p[1:] - p[:-1]
    c = np.hypot(d[:, 0], d[:, 1])
    if c[0] < STOP_STEP_M or c[1] < STOP_STEP_M:
        return a_ref0, 0.0
    return a_ref0, _wrap(math.atan2(d[1, 1], d[1, 0]) - math.atan2(d[0, 1], d[0, 0])) / (0.5 * (c[0] + c[1]))


def kin_label(grid, r: int, kin_own, *, check: bool = True, _speed=None):
    """The KINEMATIC label (``--dagger-label kin_v1``) of an own state matched (ok) to logged row ``r``; ``kin_own`` = (v_o, a_o, kappa_o) = ``p10f[-1, 0:3]``. ``grid`` [T, >=3] the logged (x, y, yaw) on the 0.2 s grid.
    Returns dict(status, label [30, 2] | None, speed_status, speed_label [30, 2] | None (the fallback), t_merge | None, max_dev | None (max over the output points of the distance to the human polyline), viol | None).
    ``status`` is one of KIN_STATUSES: kin_emitted / identity_rule carry ``label``; a refusal carries ``label`` None and the caller keeps ``speed_label`` (else the human label)."""
    G = np.asarray(grid, np.float64)
    r = int(r)
    v_o, a_o, k_o = (float(kin_own[0]), float(kin_own[1]), float(kin_own[2]))
    out = {"status": None, "label": None, "speed_status": None, "speed_label": None, "t_merge": None, "max_dev": None, "viol": None}
    sp = speed_label(G, r, v_o) if _speed is None else _speed
    out["speed_status"], out["speed_label"] = sp["status"], sp["label"]
    if sp["status"] not in EMITTED:
        out["status"] = "speed_" + sp["status"]
        out["speed_label"] = None
        return out
    if not (math.isfinite(v_o) and math.isfinite(a_o) and math.isfinite(k_o)):
        out["status"] = "refused_nonfinite"
        return out
    Gp = G[r:]
    P = to_ego(Gp[:, :2], (Gp[0, 0], Gp[0, 1], Gp[0, 2]))                            # the reference polyline, as speed_label builds it
    s = _polyline(P)
    uh = np.concatenate([[0.0], (s[1:H + 1] - s[:H]) / DT])
    ul = label_steps(v_o, uh)
    a_ref0, k_ref0 = ref_start(sp["label"], ul)
    if abs(k_o - k_ref0) <= IDENT_TOL and abs(a_o - a_ref0) <= IDENT_TOL:             # the identity rule: the label IS the speed_v1 label
        out.update(status="identity_rule", label=sp["label"])
        return out
    uh_k = uh.copy()
    uh_k[0] = uh[1]                                                                   # knot 0 of the interpolated human step speed (u_h[0] is unused: the first step's speed stands there)
    Pl, sl = [(float(x), float(y)) for x, y in P], [float(x) for x in s]
    s_end = sl[-1]
    x = y = th = 0.0
    v = v_o
    seg = 0
    pts = np.empty((H, 2))
    max_dev = 0.0
    merged_k = None
    s_m = v_m = 0.0
    for n in range(H * SUB_PER_STEP):
        t = n * H_SUB
        w = min(1.0, t / T_B)
        k = n // SUB_PER_STEP
        fr = (n % SUB_PER_STEP) / SUB_PER_STEP
        u_h_t = (1.0 - fr) * float(uh_k[k]) + fr * float(uh_k[k + 1])                 # the human step speed at t, linear between the 0.2 s knots
        a_rule = min(max((u_h_t - v) / DT, -A_LABEL), A_LABEL)                        # the speed_v1 rule on the integrated state
        a = min(max((1.0 - w) * a_o + w * a_rule, -A_LIM), A_LIM)
        if w > 0.0:
            seg, s_star, _, _ = _closest(Pl, sl, (x, y), seg)
            Ld = max(LD_MIN, LD_GAIN * v)
            if s_star + Ld > s_end:                                                   # the lookahead runs past the logged path
                out["status"] = "refused_path_exhausted"
                return out
            lx, ly = _point_at(Pl, sl, s_star + Ld)
            kpp = 2.0 * math.sin(_wrap(math.atan2(ly - y, lx - x) - th)) / Ld
        else:
            kpp = 0.0
        kap = min(max((1.0 - w) * k_o + w * kpp, -K_LIM), K_LIM)
        if math.hypot(a, v * v * kap) > F_LIM:                                        # the friction circle: kappa gives way first
            kap = math.copysign(math.sqrt(max(F_LIM * F_LIM - a * a, 0.0)) / (v * v), kap)
        v1 = max(0.0, v + a * H_SUB)
        vm = 0.5 * (v + v1)
        th1 = th + vm * kap * H_SUB
        thm = 0.5 * (th + th1)
        x += vm * math.cos(thm) * H_SUB
        y += vm * math.sin(thm) * H_SUB
        v, th = v1, th1
        if (n + 1) % SUB_PER_STEP == 0:
            kk = (n + 1) // SUB_PER_STEP                                              # output point kk (1..30)
            pts[kk - 1] = (x, y)
            seg, s_star, dist, hd = _closest(Pl, sl, (x, y), seg)
            max_dev = max(max_dev, dist)
            if (n + 1) * H_SUB >= T_B and dist < MERGE_LAT_M and abs(_wrap(th - hd)) < MERGE_HEAD_RAD and abs(v - float(uh_k[kk])) <= MERGE_SPEED_MS:
                merged_k, s_m, v_m = kk, s_star, v
                break
    if not (np.isfinite(pts[:max(merged_k or 0, 1)]).all() and math.isfinite(max_dev)):
        out["status"] = "refused_nonfinite"
        return out
    if merged_k is None:
        out["status"] = "refused_not_merged"
        return out
    out["t_merge"] = merged_k * DT
    ul_t = label_steps(v_m, uh_k[merged_k:])                                          # the speed_v1 rule from the merged v: ul_t[j] = the step speed of output point merged_k + j
    arc = s_m
    for kk in range(merged_k + 1, H + 1):                                             # P at the cumulative arc (position- and speed-continuous)
        arc += float(ul_t[kk - merged_k]) * DT
        if arc > s_end + 1e-9:
            out["status"] = "refused_path_exhausted"
            return out
        pts[kk - 1] = _point_at(Pl, sl, arc)
    if not np.isfinite(pts).all():
        out["status"] = "refused_nonfinite"
        return out
    out["max_dev"] = float(max_dev)
    if check:
        vv = violations(pts, v_o, H)
        out["viol"] = vv
        if vv["any"]:
            out["status"] = "refused_infeasible"
            return out
    out.update(status="kin_emitted", label=pts)
    return out


def kin_label_bank(arrays: dict, ci, clip_names, read_clip, *, log=None, check: bool = True) -> dict:
    """``kin_label`` for every bank record whose matcher reason is ok (once, at bank load; deterministic float64, no RNG). Off-route / end_edge / nonfinite records get no label (they stay cut, exactly as under speed_v1: the eligible index is equal under
    all three label kinds).

    Same arguments as ``label_bank``. The own state is ``p10f[:, -1, 0:3]``. A refused record keeps its ``speed_v1`` label (when the reference itself was not refused), else the human label (slot -1).
    Returns ``{"slot": int32 [N] (-1 = none), "label": float64 [n_emit, 30, 2], "is_kin": bool [N] (the record's label is a kin_v1 one: status kin_emitted), "status_counts", "n_labelled", "census"}``."""
    reason = np.asarray(arrays["reason"])
    p10f, rr = arrays["p10f"], np.asarray(arrays["r"], np.int64)
    tick = np.asarray(arrays["tick"], np.int64) if "tick" in arrays else np.zeros(len(reason), np.int64)
    ci = np.asarray(ci, np.int64)
    todo = np.flatnonzero(reason == _OK)
    order = todo[np.argsort(ci[todo], kind="stable")]
    N = len(reason)
    slot = np.full(N, -1, np.int32)
    is_kin = np.zeros(N, bool)
    st = np.full(N, -1, np.int8)
    t_merge, max_dev = np.full(N, np.nan), np.full(N, np.nan)
    d30, d5, a30 = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    vt = np.zeros(N, np.int8)                                                         # checker terms of a refused_infeasible record: 1 accel, 2 curv, 4 fric (bit set)
    labels, counts, n_done = [], {}, 0
    cur, grid = None, None
    for i in order:
        c = int(ci[i])
        if c != cur:
            d = read_clip(str(clip_names[c]))
            pg = np.asarray(d["pose"], np.float64)
            grid, cur = pg[:, :3], c
        res = kin_label(grid, int(rr[i]), np.asarray(p10f[i, -1, :3], np.float64), check=check)
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        st[i] = KIN_STATUSES.index(res["status"])
        if res["status"] == "refused_infeasible":
            vt[i] = (1 if res["viol"]["accel"] else 0) | (2 if res["viol"]["curv"] else 0) | (4 if res["viol"]["fric"] else 0)
        if res["status"] in ("kin_emitted", "identity_rule"):
            slot[i] = len(labels)
            labels.append(res["label"])
            is_kin[i] = res["status"] == "kin_emitted"
        elif res["speed_label"] is not None:                                          # the fallback: today's speed_v1 label of this record
            slot[i] = len(labels)
            labels.append(res["speed_label"])
        if res["status"] == "kin_emitted":
            sl_ = res["speed_label"]
            t_merge[i], max_dev[i] = res["t_merge"], res["max_dev"]
            d30[i] = float(np.hypot(*(res["label"][H - 1] - sl_[H - 1])))
            d5[i] = float(np.hypot(*(res["label"][4] - sl_[4])))
            a30[i] = float(arc_from_label(res["label"][None])[0, H - 1] - arc_from_label(sl_[None])[0, H - 1])
        n_done += 1
        if log is not None and n_done % 50000 == 0:
            log(f"[kin_v1] labelled {n_done}/{len(order)}")
    lab = np.stack(labels).astype(np.float64) if labels else np.zeros((0, H, 2))
    return {"slot": slot, "label": lab, "is_kin": is_kin, "status_counts": counts, "n_labelled": int(n_done), "census": kin_census(reason, tick, st, t_merge, max_dev, d30, d5, a30, vt)}


def kin_census(reason, tick, st, t_merge, max_dev, d30, d5, a30, vt=None) -> dict:
    """The bank census of ``--dagger-label kin_v1``: per tick and pooled, counts of ok records, relabelled (kin_emitted), identity (the identity rule), refused by reason; the relabelled share of ok; and over the relabelled records the merge time (s), the max
    deviation of the label from the human polyline (m), |P_kin[30] - P_speed[30]| and |P_kin[5] - P_speed[5]| (m; point distance), and the arc difference at knot 30 (m, absolute)."""
    ok = np.asarray(reason) == _OK
    tick = np.asarray(tick)
    pct = lambda a: ({"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "n": int(len(a))} if len(a) else {"p50": None, "p95": None, "p99": None, "n": 0})
    E, I = KIN_STATUSES.index("kin_emitted"), KIN_STATUSES.index("identity_rule")
    out = {"per_tick": {}}
    for t in sorted(set(int(x) for x in tick[ok])) + ["pooled"]:
        m = ok if t == "pooled" else ok & (tick == t)
        row = {"ok": int(m.sum()), "relabelled": int((m & (st == E)).sum()), "identity": int((m & (st == I)).sum())}
        row["refused"] = {k: int((m & (st == i)).sum()) for i, k in enumerate(KIN_STATUSES) if i not in (E, I) and (m & (st == i)).any()}
        if vt is not None:                                                            # refused_infeasible by checker term (not exclusive)
            row["refused_infeasible_by_term"] = {"accel": int((m & (vt & 1 > 0)).sum()), "curv": int((m & (vt & 2 > 0)).sum()), "fric": int((m & (vt & 4 > 0)).sum())}
        out["per_tick"][str(t)] = row
    k = ok & (st == E)
    out["merge_time_s"], out["max_dev_m"] = pct(t_merge[k]), pct(max_dev[k])
    out["abs_dpoint30_kin_vs_speed_m"], out["abs_dpoint5_kin_vs_speed_m"] = pct(d30[k]), pct(d5[k])
    out["abs_darc30_kin_vs_speed_m"] = pct(np.abs(a30[k]))
    n_ok = int(ok.sum())
    out["relabelled_share_of_ok"] = (int(k.sum()) / n_ok) if n_ok else None
    return out
