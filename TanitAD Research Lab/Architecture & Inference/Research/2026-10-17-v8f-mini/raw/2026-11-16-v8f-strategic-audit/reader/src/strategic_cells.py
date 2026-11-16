"""STRATEGIC cells from the NVIDIA nav text (numpy only; staged, not landed).

Cells (all per window; NaN = not scorable, never 0):
  S_ADH   route adherence: an announced L/R manoeuvre whose point lies inside the plan's own arc; 1 if the plan turns the
          announced side by >= theta over the arc [0, d_text + COMPLETE_M]. Signed turn (deg, side-positive) is the continuous twin.
  S_FALSE false turn: no manoeuvre announced within the plan's arc + margin; 1 if the plan turns by >= theta either side.
  S_CKPT  S's route checkpoint (30 / 50 / 80 m of arc, NOW frame) heading agrees with the announced side (v8f-mini only).
Ground truth: navigation.parquet text parsed by navtok (class, distance); coarse side only.
The path integrator is hwm_plan.integrate (lazy import). The prototype's own copy stays as _proto_integrate, used only by a test that pins the two as equal.
"""
from __future__ import annotations

import numpy as np

MU, G_ACC, VMAX, DT = 0.7, 9.81, 40.0, 0.2
COMPLETE_M = 20.0          # HYPOTHESIS: arc allowance past the announced point for the turn to complete
THETA_DEG = {"STD": 25.0, "SLIGHT": 10.0}   # HYPOTHESIS thresholds; pre-register before a decisive read
FALSE_MARGIN_M = 30.0


def _proto_integrate(v0, plan, dt=DT):
    n, K, _ = plan.shape
    v = np.array(v0, np.float64)
    psi, x, y = np.zeros(n), np.zeros(n), np.zeros(n)
    PSI, XY = [psi.copy()], [np.stack([x, y], 1)]
    amax = MU * G_ACC
    for t in range(K):
        a, w = np.clip(plan[:, t, 0], -amax, amax), plan[:, t, 1]
        wm = amax / np.maximum(v, 1.0)
        w = np.clip(w, -wm, wm)
        x = x + v * np.cos(psi) * dt
        y = y + v * np.sin(psi) * dt
        v = np.clip(v + a * dt, 0.0, VMAX)
        psi = psi + w * dt
        PSI.append(psi.copy())
        XY.append(np.stack([x, y], 1))
    return np.stack(PSI, 1), np.stack(XY, 1)


def integrate(v0, plan):
    """(psi [N, K+1], xy [N, K+1, 2]) of a raw (a, yaw rate) sequence from NOW, by the production kinematic model hwm_plan.integrate (friction and speed bounds)."""
    import hwm_plan
    _, psi, xy, _ = hwm_plan.integrate(v0, plan)
    return psi, xy


def parse_side(vocab_name: str):
    """(side +1 left / -1 right / 0 none, theta_deg, is_uturn) from a navtok class name. UTURN and non-TURN classes -> side 0."""
    n = str(vocab_name)
    if not n.startswith("TURN"):
        return 0, np.nan, n.startswith("UTURN")
    th = THETA_DEG["SLIGHT"] if "SLIGHT" in n else THETA_DEG["STD"]
    return (1 if n.endswith("_L") else -1 if n.endswith("_R") else 0), th, False


def side_table(vocab):
    s = np.array([parse_side(v)[0] for v in vocab], np.int8)
    th = np.array([parse_side(v)[1] for v in vocab], np.float64)
    return s, th


def arc_len(xy):
    return np.concatenate([np.zeros((xy.shape[0], 1)), np.cumsum(np.linalg.norm(np.diff(xy, axis=1), axis=-1), 1)], 1)


def heading_at_arc(psi, xy, arc_m):
    """psi at arc length arc_m (per window); NaN where the path is shorter."""
    s = arc_len(xy)
    k = (s >= arc_m[:, None]).argmax(1)
    reach = s[:, -1] >= arc_m
    return np.where(reach, psi[np.arange(len(psi)), k], np.nan), reach


def cells(psi, xy, side, theta_deg, d_text, text_ok):
    """psi [N, K+1], xy [N, K+1, 2] of ONE path (plan, logged or hold) from NOW in the ego frame; side/theta/d_text/text_ok per window.
    Returns dict of [N] arrays: s_adh, s_adh_signed_deg, s_false. A window with no text row (text_ok False) is NaN in every cell."""
    N = psi.shape[0]
    adh, sgn, fal = (np.full(N, np.nan) for _ in range(3))
    s_end = arc_len(xy)[:, -1]
    turn = text_ok & (side != 0) & np.isfinite(d_text)
    arc = np.where(turn, d_text + COMPLETE_M, np.nan)
    h, reach = heading_at_arc(psi, xy, np.where(turn, arc, 0.0))
    m = turn & reach
    sgn[m] = np.degrees(h[m]) * side[m]
    adh[m] = (sgn[m] >= theta_deg[m]).astype(float)
    # false turn: no TURN_* announced inside the path + margin (side == 0 or announced far), and the path is scorable over its full arc
    far = text_ok & ((side == 0) | ~np.isfinite(d_text) | (d_text > s_end + FALSE_MARGIN_M))
    net = np.degrees(np.abs(psi[:, -1]))
    fal[far] = (net[far] >= THETA_DEG["STD"]).astype(float)
    return {"s_adh": adh, "s_adh_signed_deg": sgn, "s_false": fal}


def ckpt_cell(ck_psi_rad, arcs_m, side, theta_deg, d_text, text_ok, ck_ok):
    """S_CKPT: ck_psi_rad [N, 3] heading of S's checkpoint at arcs_m (30/50/80). Scorable at arc a where a TURN_* is announced with d_text + COMPLETE_M <= a."""
    N = ck_psi_rad.shape[0]
    out = np.full((N, len(arcs_m)), np.nan)
    for m, a in enumerate(arcs_m):
        ok = text_ok & ck_ok[:, m] & (side != 0) & np.isfinite(d_text) & (d_text + COMPLETE_M <= a)
        out[ok, m] = (np.degrees(ck_psi_rad[ok, m]) * side[ok] >= theta_deg[ok]).astype(float)
    return out


# ---------------------------------------------------------------------------------------------- report text shared by hwm_eval, assemble_arm and verdict_actsens (one source)
STRATEGIC_STATE = "REPORTED IN FULL (PI 2026-11-16: the strategic pause ended); never a pass by itself"
STRATEGIC_TIER_STAMP = ("T0 HWM-ORACLE diagnostic on the oracle rows: the goal that conditioned the plan is the ENCODED TRUE FUTURE, so a cell on those plans measures the world model's use of an oracle goal, "
                        "not a strategic decision and never a driving number. Rows of the predicted-goal arms (hwm_plan, flat_plan) are T1-kin")
STRATEGIC_BAND_EMPTY = ("empty by construction: a STRATEGIC cell reads the whole plan arc against the announced manoeuvre distance (median 127 m ahead, NAVPROBE), "
                        "so only the full 6 s band is filled; the 0-2 s and 2-6 s bands carry n = 0")
STRATEGIC_SCOPE = ("S_ADH: an announced L/R turn at distance d_text with d_text + COMPLETE_M inside the plan's own arc; 1 if the plan's side-signed heading change over [0, d_text + COMPLETE_M] reaches theta; "
                   "s_adh_deg is the same heading change in signed degrees (side-positive), the continuous twin. S_FALSE: no turn announced within the plan arc + 30 m, 1 if the plan's net heading change reaches 25 deg either side; "
                   "the text does not announce bends or ramps the logged path takes, so S_FALSE is read ONLY as plan minus logged-path rate, never as an absolute. COMPLETE_M, the thetas and the 30 m margin are HYPOTHESIS. "
                   "Ground truth is the NVIDIA navigation text (navtok table, flag 0 rows only; every other row is NaN). The text and the trained nav input both derive from the drive's own path, so the cells are optimistic as route-following evidence. "
                   "Estimator: paired episode-cluster bootstrap (taniteval/ci.py), which answers 'another draw of episodes' only")
STRATEGIC_NOT_COMPUTED = {
    "STRATEGIC.intent_6_30s": "n = 0: no arm outputs an intent or goal at 6-30 s (the HWM plan is 6 s; v8f-mini's S emits route checkpoints out to 80 m only). Would-have n, MEASURED count on the 994 ACTSENS windows by time to the announced "
                              "manoeuvre d / v0 (strategic-audit count_n.py): 6-12 s 32 windows, 12-30 s 69, over 30 s 20. Unblocks with a PI go on a strategic head and training",
    "STRATEGIC.derived_goals": "n = 0: no derived-goal output exists on the HWM planner. Would-have n: n/a (no output or label is defined; STRATEGIC_PLAN section 2)",
    "STRATEGIC.detected_info": "n = 0: no detected-information output exists on the HWM planner (the HWM ctx token is an input, not scored). Would-have n: n/a (STRATEGIC_PLAN section 2)",
    "STRATEGIC.S_CKPT": "structural n = 0: the HWM planner has no S head, so no route checkpoint exists to score. S_CKPT is read on v8f-mini arms only; would-have n on the eval141 grid, MEASURED "
                        "(proto_eval141_counts.json): 46 / 120 / 214 windows (18 / 22 / 24 episodes) at 30 / 50 / 80 m of arc",
}
