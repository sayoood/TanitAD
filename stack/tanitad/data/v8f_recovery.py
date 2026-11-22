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
DAGGER_LABELS = ("human", "speed_v1")           # the ``--dagger-label`` choices; "human" is the default and is today's swap bit for bit
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
