"""WP-A Stage 3 — the INDEPENDENT derivation that validates the v9 release (SPEC §9.V).

⛔ INDEPENDENCE RULE (CLAUDE.md "a cross-check must be derived independently of the value it checks"): this module
imports NOTHING from `build_v9_labels.py` or the v7 builder, and where the log offers a second channel it uses it:
  * heading  = the PATH TANGENT of the positions (resampled every 1 m of arc) — not the quaternion yaw;
  * arc      = the integral of the SPEED channel over time — not the position chords;
  * radius   = from the provider's own CURVATURE channel (∫|κ| ds over the segment) — not arc / Δψ;
  * stop     = speed from POSITION differences — not the |v| channel.
Thresholds are written here as literals from road geometry (they coincide with the SPEC's where the SPEC states a
physical bar, which is what makes agreement meaningful).
"""
from __future__ import annotations

import math

import numpy as np

TURN_DEG = 30.0          # a turn: >= 30 deg of heading excursion inside the band
STOP_V = 0.5             # standstill
GAP_MAX = 1.0            # admissible log gap (s)
DS_TAN = 1.0             # tangent resampling step (m)
RATE_ON = 6.0            # sustained yaw-rate threshold for segment timing (deg/s) — the PI-accepted literal
MIN_SEG = 0.8
MIN_PATH_M = 5.0         # D2's stationary rule: a band path shorter than this cannot carry a turn


class Log:
    """Native egomotion samples; raw time from the first sample."""

    def __init__(self, path):
        import pandas as pd
        df = pd.read_parquet(path, columns=["timestamp", "x", "y", "vx", "vy", "vz", "curvature"])
        t = df["timestamp"].to_numpy(np.float64) / 1e6
        self.t = t - t[0]
        self.x = df["x"].to_numpy(np.float64)
        self.y = df["y"].to_numpy(np.float64)
        self.v = np.linalg.norm(df[["vx", "vy", "vz"]].to_numpy(np.float64), axis=1)
        self.kappa = df["curvature"].to_numpy(np.float64)
        # arc from the SPEED channel (trapezoid), independent of the position chords
        self.arc_v = np.concatenate([[0.0], np.cumsum(0.5 * (self.v[1:] + self.v[:-1]) * np.diff(self.t))])
        # arc from POSITIONS, used only to parameterise the tangent heading
        self.arc_p = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))])
        sg = np.arange(0.0, self.arc_p[-1] + 1e-9, DS_TAN)
        keep = np.concatenate([[True], np.diff(self.arc_p) > 1e-9])
        xs = np.interp(sg, self.arc_p[keep], self.x[keep])
        ys = np.interp(sg, self.arc_p[keep], self.y[keep])
        h = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
        k = np.ones(5) / 5.0                              # 5 m smoothing of the tangent heading
        self.s_grid = sg
        # CUSPS (forward <-> reverse) from the POSITION path alone: consecutive 1-m segments turning by > 150 deg.
        # Disclosed post-hoc (2026-10-04): at a cusp the path tangent flips 180 deg against the body heading, so
        # no tangent-based turn reading is defined there (the 3 inverted eval clips).
        segd = np.arctan2(np.diff(ys), np.diff(xs))
        jump = np.abs((np.diff(segd) + np.pi) % (2 * np.pi) - np.pi)
        sc = sg[1:-1][jump > np.radians(150.0)] if len(segd) > 1 else np.array([])
        self.cusp_t = np.interp(sc, self.arc_p[keep], self.t[keep]) if sc.size else np.array([])
        self.psi_s = np.convolve(np.pad(h, 2, mode="edge"), k, mode="valid") if len(h) >= 5 else h
        # speed from POSITIONS (an independent stop detector), 0.3 s smoothing
        dt = np.diff(self.t)
        vp = np.concatenate([[0.0], np.hypot(np.diff(self.x), np.diff(self.y)) / np.maximum(dt, 1e-6)])
        n = 30 if np.median(dt) < 0.02 else 3
        self.v_pos = np.convolve(vp, np.ones(n) / n, mode="same")

    @property
    def t_end(self):
        return float(self.t[-1])

    def psi_t(self, t):
        """tangent heading at raw time(s) t (via the position arc reached at t)."""
        s = np.interp(t, self.t, self.arc_p)
        return np.interp(s, self.s_grid, self.psi_s)

    def cusp_in(self, a, b):
        return bool(np.any((self.cusp_t >= a) & (self.cusp_t <= b)))

    def h_obs(self, now, horizon=8.0):
        i = max(int(np.searchsorted(self.t, now, "right")) - 1, 0)
        g = np.diff(self.t[i:])
        big = np.nonzero(g > GAP_MAX)[0]
        t_gap = self.t[i + big[0]] if big.size else math.inf
        return max(0.0, min(horizon, t_gap - now, self.t_end - now))


def band_turn(L: Log, now: float, h: float, stationary_rule: bool = True):
    """(theta_deg, observed) — the largest tangent-heading excursion over [NOW+2, NOW+h] from its value at NOW+2."""
    if h < 2.2:
        return np.nan, False
    # ⛔ POST-HOC VALIDATOR REPAIR (2026-10-04, disclosed in RESULT): D2's own independent rule "band path < 5 m ⇒
    # stationary ⇒ no turn" (d2_lib.MIN_PATH_M). Without it the PATH TANGENT is undefined at standstill (position
    # jitter read as 112-137° 'turns' on the two stationary eval clips) — a defect of this channel, not of the builder.
    path = float(np.interp(now + h, L.t, L.arc_v) - np.interp(now + 2.0, L.t, L.arc_v))
    if stationary_rule and path < MIN_PATH_M:
        return 0.0, True
    tt = now + np.arange(2.0, h + 1e-9, 0.1)
    d = np.degrees(L.psi_t(tt) - L.psi_t(now + 2.0))
    j = int(np.argmax(np.abs(d)))
    return float(d[j]), True


def band_stop(L: Log, now: float, h: float):
    """(stop_in_band, t_stop_rel, d_stop_speed_arc) using the POSITION speed for the event and the SPEED arc for d."""
    if h < 2.2:
        return None, np.nan, np.nan
    va = float(np.interp(now + 2.0, L.t, L.v_pos))
    if va <= STOP_V:
        return None, np.nan, np.nan            # already stopped at the band start: not a STOP event (SPEC)
    m = (L.t >= now + 2.0) & (L.t <= now + h) & (L.v_pos <= STOP_V)
    if not m.any():
        return False, np.nan, np.nan
    ts = float(L.t[np.nonzero(m)[0][0]])
    d = float(np.interp(ts, L.t, L.arc_v) - np.interp(now, L.t, L.arc_v))
    return True, ts - now, d


def segments(L: Log, t0: float, t1: float):
    """Sustained-rate segments of the TANGENT heading on [t0, t1]: (t_start, t_end, dpsi_deg, R_curv_m)."""
    tt = np.arange(max(t0, 0.0), min(t1, L.t_end) + 1e-9, 0.1)
    if len(tt) < 10:
        return []
    psi = L.psi_t(tt)
    rate = np.degrees(np.gradient(psi, 0.1))
    rate = np.convolve(rate, np.ones(5) / 5.0, mode="same")
    act = np.abs(rate) >= RATE_ON
    out, i = [], 0
    while i < len(tt):
        if act[i]:
            j = i
            while j + 1 < len(tt) and act[j + 1]:
                j += 1
            if (j - i + 1) * 0.1 >= MIN_SEG:
                d = math.degrees(psi[j] - psi[i])
                if abs(d) >= 20.0:
                    # radius from the provider CURVATURE channel: arc / integral |kappa| ds
                    m = (L.t >= tt[i]) & (L.t <= tt[j])
                    if m.sum() > 2:
                        s = np.interp(L.t[m], L.t, L.arc_v)
                        ik = float(np.trapezoid(np.abs(L.kappa[m]), s)) if hasattr(np, "trapezoid") else float(np.trapz(np.abs(L.kappa[m]), s))
                        Lseg = float(s[-1] - s[0])
                        R = Lseg / ik if ik > 1e-6 else np.inf
                    else:
                        R = np.nan
                    out.append((float(tt[i]), float(tt[j] + 0.1), d, R))
            i = j + 1
        else:
            i += 1
    return out
