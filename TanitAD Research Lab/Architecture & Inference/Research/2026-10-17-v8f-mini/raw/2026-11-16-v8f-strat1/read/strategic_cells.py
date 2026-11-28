"""STRATEGIC cells from the NVIDIA nav text (CPU prototype, numpy only; staged, not landed).

Cells (all per window; NaN = not scorable, never 0):
  S_ADH   route adherence: an announced L/R manoeuvre whose point lies inside the plan's own arc; 1 if the plan turns the
          announced side by >= theta over the arc [0, d_text + COMPLETE_M]. Signed turn (deg, side-positive) is the continuous twin.
  S_FALSE false turn: no manoeuvre announced within the plan's arc + margin; 1 if the plan turns by >= theta either side.
  S_CKPT  S's route checkpoint (30 / 50 / 80 m of arc, NOW frame) heading agrees with the announced side (v8f-mini only).
Ground truth: navigation.parquet text parsed by navtok (class, distance); coarse side only.
Integrator below is a copy of hwm_plan.integrate (hwm_plan.py:523); production imports it instead.
"""
from __future__ import annotations

import numpy as np

MU, G_ACC, VMAX, DT = 0.7, 9.81, 40.0, 0.2
COMPLETE_M = 20.0          # HYPOTHESIS: arc allowance past the announced point for the turn to complete
THETA_DEG = {"STD": 25.0, "SLIGHT": 10.0}   # HYPOTHESIS thresholds; pre-register before a decisive read
FALSE_MARGIN_M = 30.0


def integrate(v0, plan, dt=DT):
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
