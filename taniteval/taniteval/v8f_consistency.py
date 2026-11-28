"""v8F-mini metrics: CONSISTENCY, the nav-only kinematic checkpoint baseline, tactical macro-F1.

Pre-registration: ``Project Steering/PREREG_V8F_MINI.md`` (committed 2026-10-17, BEFORE any data).
Every literal below is in :data:`RULES`; the four violation rules are the PREREG's, verbatim in intent:

* **SPEED** -- the planned max speed is more than 0.5 m/s above min(limit, SPEED_BAND v_hi).
* **STOP**  -- the planned speed is more than 1.0 m/s at the stop distance, or the plan passes the stop point by
  more than 2 m.
* **TURN**  -- the sign of the planned heading change differs from the goal side, while a TURN goal is active.
* **LEAD**  -- the minimum planned time gap is more than 0.2 s below the constraint.

PURE NUMPY, no torch, no repo import: the numbers it produces must be reproducible from arrays alone. The torch
penalties in ``tanitad/models/v8f_mini.py`` are tested to AGREE with these functions (hinge > 0 <=> violation).

INTERPRETATIONS (the PREREG one-liners are silent on these; each is frozen HERE, before any data, and listed in
``Research/2026-10-17-v8f-mini/RESULT.md`` for the Master Mind to confirm; none can be tuned after the fact):

I1. The plan is the 6 s path on the 0.2 s trunk-probe grid, ``[N, 30, 2]`` ego-frame metres (x forward, y left),
    knot j at tau = 0.2 j (j = 1..30), origin at tau = 0.
I2. The SPEED / STOP / LEAD constraints are labels defined on the band ``[NOW+2, NOW+8]``; they are evaluated on the
    plan's overlap with it, ``tau >= 2.0`` (knots j >= 10 for knot-valued rules, intervals starting at tau >= 2.0 for
    speed). Without this, the GT human path violates SPEED whenever it brakes (v at NOW > max_B v).
I3. STOP is active iff the STOP_POINT goal is set with weight 1, ``lon_d_reach_m`` (= d_stop, arc NOW -> stop
    point) is finite and the stop is inside the plan horizon (``lon_t_reach_s`` <= 6.0). "Speed at the stop
    distance" is the speed of the plan interval holding arc d_stop; a plan that never reaches d_stop has no speed
    violation (it stopped short); "passes" is total plan arc > d_stop + 2.0.
I4. TURN is active iff a TURN_L / TURN_R goal bit has weight 1 and the turn segment overlaps the plan by >= 1.0 s,
    ``[max(0, t_start), min(6, t_end)]``. The planned heading change is psi(t1) - psi(t0) from the path tangents
    (psi(0) = 0, the ego frame); sign 0 inside a 1 deg dead band; a violation is sign != goal side (so a plan
    that does not turn at all during an active TURN goal violates it).
I5. LEAD: the lead's longitudinal position is taken from the logged (human) run -- ``lead_gap_h + s_h`` -- and the
    planned gap is that minus the plan's arc; ``tg = max(gap, 0) / max(v_plan, 1.0)`` (the builder's own time-gap
    definition); the constraint is the label ``lead_tg_min_s``. A window with no lead data is inactive.
I6. CONSISTENCY(a) = fraction of ALL evaluated windows violating ANY active rule (a window with no active rule
    cannot violate); per-rule rates over all windows AND over that rule's active windows are both reported.
    Self-consistency(b) = 1 - the same fraction against the model's OWN predicted constraints.
I7. The nav-only baseline's turn: the PREREG says "go straight to d, then follow an R = 12 m arc to the nav side"
    without an exit angle. The arc is a 90 deg junction turn (length pi/2 * 12 m), then straight. 90 deg is an
    assumption; the empirical median |nav_dyaw_next_deg| of the TRAIN release is printed in the label-join report.
"""
from __future__ import annotations

import math

import numpy as np

DT = 0.2
K = 30                          # knots (6 s)
RULES = {
    "dt_s": DT, "horizon_s": 6.0, "band_start_s": 2.0,
    "speed_tol_ms": 0.5,
    "stop_speed_ms": 1.0, "stop_pass_m": 2.0,
    "turn_deadband_deg": 1.0, "turn_min_overlap_s": 1.0,
    "lead_tg_tol_s": 0.2, "lead_v_floor_ms": 1.0,
    "nav_baseline_radius_m": 12.0, "nav_baseline_turn_deg": 90.0,
    "ckpt_arcs_m": (30.0, 50.0, 80.0),
}
RULE_NAMES = ("SPEED", "STOP", "TURN", "LEAD")
_BAND_KNOT = int(round(RULES["band_start_s"] / DT))        # 10: first knot inside the band (tau = 2.0)


# ------------------------------------------------------------------------------------------------ kinematics
def plan_kinematics(xy: np.ndarray, dt: float = DT) -> dict:
    """``xy`` [N, K, 2] -> arc ``s`` [N, K+1] (knot 0 = origin), interval speed ``v`` [N, K] (interval j-1 ends at
    knot j), heading ``psi`` [N, K+1] (rad; knot 0 = 0, knot j = tangent of interval j-1; carried across ~stationary
    intervals, unwrapped along time)."""
    xy = np.asarray(xy, np.float64)
    n, k, _ = xy.shape
    P = np.concatenate([np.zeros((n, 1, 2)), xy], 1)
    seg = np.diff(P, axis=1)
    L = np.linalg.norm(seg, axis=-1)
    s = np.concatenate([np.zeros((n, 1)), np.cumsum(L, 1)], 1)
    psi = np.zeros((n, k + 1))
    cur = np.zeros(n)
    for j in range(k):
        mov = L[:, j] > 1e-6
        ang = np.arctan2(seg[:, j, 1], seg[:, j, 0])
        # unwrap against the carried heading
        ang = cur + (ang - cur + np.pi) % (2 * np.pi) - np.pi
        cur = np.where(mov, ang, cur)
        psi[:, j + 1] = cur
    return {"s": s, "v": L / dt, "psi": psi}


def _interp_knots(vals: np.ndarray, tau: np.ndarray, dt: float = DT) -> np.ndarray:
    """Per-row linear interpolation of ``vals`` [N, K+1] (knot j at tau = j*dt) at per-row times ``tau`` [N]."""
    x = np.clip(np.asarray(tau, np.float64) / dt, 0.0, vals.shape[1] - 1.0)
    i0 = np.minimum(np.floor(x).astype(int), vals.shape[1] - 2)
    f = x - i0
    r = np.arange(len(vals))
    return vals[r, i0] * (1 - f) + vals[r, i0 + 1] * f


# ------------------------------------------------------------------------------------------------ the rules
def violations(plan_xy: np.ndarray, c: dict, rules: dict = RULES) -> dict:
    """Evaluate the four frozen rules.

    ``c`` (constraint dict; the SAME schema for GT constraints and the model's own predicted ones):
      ``limit_ms`` [N] (NaN = unknown), ``v_hi_ms`` [N] (NaN),
      ``stop_active`` [N] bool, ``d_stop`` [N], ``t_stop`` [N],
      ``turn_side`` [N] in {-1 (R), 0 (none), +1 (L)}, ``turn_t_start`` [N], ``turn_t_end`` [N],
      ``lead_tg_min`` [N] (NaN = no lead constraint), ``lead_gap_h`` [N, K] (NaN = no data), ``s_h`` [N, K].
    Returns ``{rule: {"active": bool[N], "violated": bool[N]}}`` plus ``"any_active"`` / ``"any_violated"``."""
    plan_xy = np.asarray(plan_xy, np.float64)
    n = len(plan_xy)
    kin = plan_kinematics(plan_xy, rules["dt_s"])
    s, v, psi = kin["s"], kin["v"], kin["psi"]
    out = {}

    # ---- SPEED
    cap = np.fmin(np.asarray(c["limit_ms"], np.float64), np.asarray(c["v_hi_ms"], np.float64))
    act = np.isfinite(cap)
    vmax = v[:, _BAND_KNOT:].max(1)                      # intervals starting at tau >= 2.0
    out["SPEED"] = {"active": act, "violated": act & (vmax > np.where(act, cap, 0.0) + rules["speed_tol_ms"]),
                    "vmax": vmax, "cap": cap}

    # ---- STOP
    d = np.asarray(c["d_stop"], np.float64)
    act = (np.asarray(c["stop_active"], bool) & np.isfinite(d)
           & (np.asarray(c["t_stop"], np.float64) <= rules["horizon_s"] + 1e-9))
    s_end = s[:, -1]
    reached = act & (s_end >= d)
    j = np.array([min(max(int(np.searchsorted(s[i], d[i], side="right")) - 1, 0), v.shape[1] - 1) if reached[i] else 0
                  for i in range(n)])
    v_at = v[np.arange(n), j]
    passes = act & (s_end > np.where(act, d, 0.0) + rules["stop_pass_m"])
    out["STOP"] = {"active": act, "violated": (reached & (v_at > rules["stop_speed_ms"])) | passes,
                   "v_at_stop": np.where(reached, v_at, np.nan), "s_end": s_end}

    # ---- TURN
    side = np.asarray(c["turn_side"], np.int64)
    t0 = np.maximum(0.0, np.nan_to_num(np.asarray(c["turn_t_start"], np.float64), nan=np.inf))
    t1 = np.minimum(rules["horizon_s"], np.nan_to_num(np.asarray(c["turn_t_end"], np.float64), nan=-np.inf))
    act = (side != 0) & ((t1 - t0) >= rules["turn_min_overlap_s"] - 1e-9)
    t0c, t1c = np.where(act, t0, 0.0), np.where(act, t1, 0.0)
    dpsi = np.degrees(_interp_knots(psi, t1c, rules["dt_s"]) - _interp_knots(psi, t0c, rules["dt_s"]))
    sgn = np.where(np.abs(dpsi) < rules["turn_deadband_deg"], 0, np.sign(dpsi)).astype(np.int64)
    out["TURN"] = {"active": act, "violated": act & (sgn != side), "dpsi_deg": np.where(act, dpsi, np.nan)}

    # ---- LEAD
    tgm = np.asarray(c["lead_tg_min"], np.float64)
    gap_h, s_h = np.asarray(c["lead_gap_h"], np.float64), np.asarray(c["s_h"], np.float64)
    gap_plan = gap_h + s_h - s[:, 1:]
    tg = np.maximum(gap_plan, 0.0) / np.maximum(v, rules["lead_v_floor_ms"])       # knot j <-> interval j-1
    tg = np.where(np.isfinite(gap_h), tg, np.nan)[:, _BAND_KNOT - 1:]               # knots j >= 10 (tau >= 2.0)
    have = np.isfinite(tg).any(1)
    tg_min = np.where(have, np.nanmin(np.where(np.isfinite(tg), tg, np.inf), 1), np.nan)
    act = np.isfinite(tgm) & have
    out["LEAD"] = {"active": act, "violated": act & (tg_min < np.where(act, tgm, 0.0) - rules["lead_tg_tol_s"]),
                   "tg_min_plan": tg_min}

    out["any_active"] = np.any([out[r]["active"] for r in RULE_NAMES], 0)
    out["any_violated"] = np.any([out[r]["violated"] for r in RULE_NAMES], 0)
    return out


def consistency(plan_xy, c, eid=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """The CONSISTENCY metric: per-rule + any-rule violation rate (PREREG I6). With ``eid`` each rate carries the
    episode-cluster bootstrap CI (``taniteval.ci``). The variance answered is *which episodes are drawn* only."""
    V = violations(plan_xy, c)
    n = len(V["any_violated"])
    res = {"n_windows": n, "rules": RULES,
           "any": {"rate_all": float(V["any_violated"].mean()), "n_violated": int(V["any_violated"].sum()),
                   "n_active": int(V["any_active"].sum()),
                   "rate_over_active": float(V["any_violated"].sum() / max(V["any_active"].sum(), 1))}}
    for r in RULE_NAMES:
        a, w = V[r]["active"], V[r]["violated"]
        res[r] = {"rate_all": float(w.mean()), "n_violated": int(w.sum()), "n_active": int(a.sum()),
                  "rate_over_active": float(w.sum() / max(a.sum(), 1))}
    if eid is not None:
        from taniteval.ci import episode_cluster_bootstrap
        res["any"]["ci"] = episode_cluster_bootstrap(V["any_violated"].astype(np.float64), eid,
                                                     n_boot=n_boot, seed=seed)
    res["_per_window"] = {"any_violated": V["any_violated"], **{r: V[r]["violated"] for r in RULE_NAMES}}
    return res


def paired_consistency_delta(viol_a: np.ndarray, viol_b: np.ndarray, eid, n_boot: int = 2000, seed: int = 0) -> dict:
    """``rate(a) - rate(b)`` on the same windows, paired episode-cluster bootstrap."""
    from taniteval.ci import paired_episode_cluster_bootstrap
    return paired_episode_cluster_bootstrap(np.asarray(viol_a, np.float64), np.asarray(viol_b, np.float64), eid,
                                            n_boot=n_boot, seed=seed)


# ------------------------------------------------------------------------------------------------ constraints
def constraints_from_v9(L: dict, limit_ms: np.ndarray, lead_gap_h: np.ndarray, s_h: np.ndarray) -> dict:
    """GT constraint dict from per-window v9 label arrays ``L`` (names as in ``v9_labels_*.npz`` rows, plus
    ``goal_y`` / ``goal_w`` unpacked to ``goal_y_bits`` / ``goal_w_bits`` ints). ``limit_ms`` from the v8 sidecar."""
    from_bit = lambda arr, name_idx: ((arr >> name_idx) & 1).astype(bool)
    gy, gw = np.asarray(L["goal_y"], np.int64), np.asarray(L["goal_w"], np.int64)
    GOAL = GOAL22
    def bit(nm):
        i = GOAL.index(nm)
        return from_bit(gy, i) & from_bit(gw, i)
    tl, tr = bit("TURN_L"), bit("TURN_R")
    side = np.where(tl, 1, np.where(tr, -1, 0))
    return {"limit_ms": np.asarray(limit_ms, np.float64), "v_hi_ms": np.asarray(L["v_hi_ms"], np.float64),
            "stop_active": bit("STOP_POINT"), "d_stop": np.asarray(L["lon_d_reach_m"], np.float64),
            "t_stop": np.asarray(L["lon_t_reach_s"], np.float64),
            "turn_side": side, "turn_t_start": np.asarray(L["turn_t_start_s"], np.float64),
            "turn_t_end": np.asarray(L["turn_t_end_s"], np.float64),
            "lead_tg_min": np.where(np.asarray(L["lead_valid"]) == 1, np.asarray(L["lead_tg_min_s"], np.float64), np.nan),
            "lead_gap_h": np.asarray(lead_gap_h, np.float64), "s_h": np.asarray(s_h, np.float64)}


GOAL22 = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD", "STOP_POINT",
          "SPEED_BAND", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR", "OVERTAKE_VEHICLE", "MERGE", "GAP_TARGET",
          "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED",
          "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")  # = v9_labels.GOAL22
LON9_FOLLOW = 3                 # index of FOLLOW in LON9 = (HOLD, CREEP, STOP, FOLLOW, DECELERATE, ACCELERATE, KEEP)


def constraints_from_pred(p: dict, limit_ms: np.ndarray, lead_gap_h: np.ndarray, s_h: np.ndarray) -> dict:
    """The model's OWN constraint dict (self-consistency, PREREG (b)) from decoded T-head outputs ``p``:
    ``goal_prob`` [N, 22], ``lon_logits`` [N, 7], ``cons`` dict of de-standardised regressions
    (``v_hi_ms, lon_d_reach_m, lon_t_reach_s, turn_t_start_s, turn_t_end_s, lead_tg_min_s``). A goal is 'set' at
    prob > 0.5; the lead constraint is active iff the argmax longitudinal class is FOLLOW."""
    gp = np.asarray(p["goal_prob"])
    tl, tr = gp[:, GOAL22.index("TURN_L")] > 0.5, gp[:, GOAL22.index("TURN_R")] > 0.5
    side = np.where(tl & (gp[:, 1] >= gp[:, 2]), 1, np.where(tr, -1, 0))
    cn = p["cons"]
    follow = np.asarray(p["lon_logits"]).argmax(-1) == LON9_FOLLOW
    return {"limit_ms": np.asarray(limit_ms, np.float64), "v_hi_ms": np.asarray(cn["v_hi_ms"], np.float64),
            "stop_active": gp[:, GOAL22.index("STOP_POINT")] > 0.5,
            "d_stop": np.asarray(cn["lon_d_reach_m"], np.float64), "t_stop": np.asarray(cn["lon_t_reach_s"], np.float64),
            "turn_side": side, "turn_t_start": np.asarray(cn["turn_t_start_s"], np.float64),
            "turn_t_end": np.asarray(cn["turn_t_end_s"], np.float64),
            "lead_tg_min": np.where(follow, np.asarray(cn["lead_tg_min_s"], np.float64), np.nan),
            "lead_gap_h": np.asarray(lead_gap_h, np.float64), "s_h": np.asarray(s_h, np.float64)}


# ------------------------------------------------------------------------------------------------ nav-only baseline
def _arc_point(path: np.ndarray, a: float) -> tuple:
    """Point + tangent angle at arc ``a`` of a polyline ``path`` [M, 2] (origin included), straight extrapolation."""
    seg = np.diff(path, axis=0)
    L = np.linalg.norm(seg, axis=1)
    S = np.concatenate([[0.0], np.cumsum(L)])
    nz = np.nonzero(L > 1e-9)[0]
    if len(nz) == 0:
        return (a, 0.0, 0.0)
    if a <= S[-1]:
        j = int(np.clip(np.searchsorted(S, a, side="right") - 1, 0, len(L) - 1))
        while L[j] <= 1e-9 and j + 1 < len(L):
            j += 1
        p = path[j] + (a - S[j]) / L[j] * seg[j]
        return (float(p[0]), float(p[1]), float(math.atan2(seg[j, 1], seg[j, 0])))
    jj = nz[-1]
    u = seg[jj] / L[jj]
    p = path[-1] + (a - S[-1]) * u
    return (float(p[0]), float(p[1]), float(math.atan2(u[1], u[0])))


def nav_only_checkpoints(p10_xy: np.ndarray, nav_token: np.ndarray, nav_d: np.ndarray,
                         rules: dict = RULES) -> np.ndarray:
    """The nav-only kinematic checkpoint baseline (PREREG H-MINI-STRAT-1).

    ``p10_xy`` [N, 30, 2] -- P10's 6 s path; ``nav_token`` [N] {0 FOLLOW, 1 TURN_L, 2 TURN_R};
    ``nav_d`` [N] distance to the manoeuvre (m; NaN = unknown -> no turn is injected).
    Extrapolate the P10 path to 30 / 50 / 80 m of arc (straight along its last tangent beyond its end); when nav
    announces a TURN at distance d, follow the P10 path to arc d, then an R = 12 m arc to the nav side (a 90 deg
    junction turn, then straight -- interpretation I7). Returns [N, 3, 3] = (x, y, psi_rad) per arc."""
    R, ang_tot = rules["nav_baseline_radius_m"], math.radians(rules["nav_baseline_turn_deg"])
    arcs = rules["ckpt_arcs_m"]
    out = np.zeros((len(p10_xy), len(arcs), 3))
    for i in range(len(p10_xy)):
        path = np.concatenate([np.zeros((1, 2)), np.asarray(p10_xy[i], np.float64)], 0)
        tok, d = int(nav_token[i]), float(nav_d[i])
        turn = tok in (1, 2) and np.isfinite(d)
        if turn:
            d = max(d, 0.0)
            x0, y0, h0 = _arc_point(path, d)
            sgn = 1.0 if tok == 1 else -1.0
        for m, a in enumerate(arcs):
            if not turn or a <= d:
                out[i, m] = _arc_point(path, a)
                continue
            u = a - d
            arc_len = R * ang_tot
            if u <= arc_len:
                th = u / R
                dx, dy = R * math.sin(th), sgn * R * (1.0 - math.cos(th))
                psi = h0 + sgn * th
            else:
                th = ang_tot
                ex, ey = R * math.sin(th), sgn * R * (1.0 - math.cos(th))
                rest = u - arc_len
                dx, dy = ex + rest * math.cos(sgn * th), ey + rest * math.sin(sgn * th)
                psi = h0 + sgn * th
            c, s_ = math.cos(h0), math.sin(h0)
            out[i, m] = (x0 + c * dx - s_ * dy, y0 + s_ * dx + c * dy, psi)
    return out


def checkpoint_errors(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    """Per-checkpoint position error [N, 3] (m, NaN where the label is invalid) and heading error (deg, wrapped)."""
    pe = np.linalg.norm(pred[..., :2] - gt[..., :2], axis=-1)
    dh = np.degrees(np.abs((pred[..., 2] - gt[..., 2] + np.pi) % (2 * np.pi) - np.pi))
    ok = np.asarray(valid, bool)
    return {"pos_err_m": np.where(ok, pe, np.nan), "head_err_deg": np.where(ok, dh, np.nan)}


# ------------------------------------------------------------------------------------------------ tactical
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, ignore: int = -100) -> dict:
    """Macro-F1 over the classes PRESENT in the labelled cells (IGNORE masked), with the confusion matrix."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    m = (y_true != ignore) & (y_true >= 0) & (y_true < n_classes)
    yt, yp = y_true[m], y_pred[m]
    cm = np.zeros((n_classes, n_classes), np.int64)
    np.add.at(cm, (yt, yp), 1)
    f1 = np.full(n_classes, np.nan)
    for c in range(n_classes):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
        if cm[c].sum() > 0:                                         # class present among the labels
            f1[c] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return {"macro_f1": float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan"), "per_class_f1": f1,
            "confusion": cm, "n": int(m.sum()), "support": cm.sum(1)}


def majority_baseline(y_true: np.ndarray, n_classes: int, ignore: int = -100) -> dict:
    """Macro-F1 of always predicting the most frequent labelled class."""
    y = np.asarray(y_true).ravel()
    y = y[(y != ignore) & (y >= 0) & (y < n_classes)]
    maj = int(np.bincount(y, minlength=n_classes).argmax()) if len(y) else 0
    r = macro_f1(y, np.full(len(y), maj), n_classes, ignore)
    r["majority_class"] = maj
    return r
