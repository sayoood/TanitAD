#!/usr/bin/env python3
"""strategic_ck.py -- time-indexed strategic checkpoints (band-labels stream A, priority 1).

For every 0.2 s cache step j of a clip (raw v2ep frame k = 2 j, log time now = grid_start_s + k * dt_s, the v9 clock
rule), the ego route point at NOW + h for h in 6, 9, ..., 30 s, as (x, y, heading) in the ego frame at NOW
(x forward, y left, heading in rad relative to the NOW heading), with a validity mask.

Smoothing is the v9 ``rc_*`` smoothing (``build_v9_labels.py`` section 6): the 10 Hz track is resampled at 0.5 m in arc
length and Gaussian-smoothed with sigma = 8 m (kernel +-3 sigma, renormalised at the start). The difference from the
rc checkpoint is only the INDEX: rc reads the smoothed path at a fixed arc ahead (30 / 50 / 80 m); here the arc is the
one the ego reached at NOW + h (``s_at(track, now + h)``), so the checkpoint is a function of time.

Validity of one (frame, horizon):
  1. now >= 0 (the log has a sample at or before NOW; a clip whose camera grid starts before the log is not read by
     clamping);
  2. now + h <= log end;
  3. no log gap above g_max (1.0 s, v9 literal) overlaps [now, now + h];
  4. the arc is inside the smoothed path with a complete forward kernel (arc <= s_hi_valid, as rc_point);
  5. the chord sagitta over the kernel footprint [arc - 3 sigma, arc + 3 sigma] is at most 0.25 m (the v9 literal).
     (rc checks the whole stretch [s_now - 3 sigma, s_now + L + 3 sigma]; a time checkpoint is one point whose value
     depends on its own kernel footprint, so only the footprint is checked. ``ck_why`` records which rule failed.)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import build_v9_verbatim as B  # noqa: E402  (verbatim copy of build_v9_labels.py, md5 recorded in the manifest)

HORIZONS_S = tuple(float(h) for h in range(6, 31, 3))     # 6, 9, ..., 30
WHY = {0: "ok", 1: "now_before_log", 2: "beyond_log_end", 3: "log_gap", 4: "arc_beyond_kernel", 5: "sagitta", 6: "no_path"}


def _gap_ok(ts: np.ndarray, now: np.ndarray, T: np.ndarray, g_max: float) -> np.ndarray:
    """True where no native-sample gap above g_max overlaps (now, T). Same rule as ``build_v9_labels.max_gap``
    (gaps between samples lo..hi, lo = last sample at or before ``now``, hi = first sample at or after ``T``)."""
    g = np.diff(ts)
    big = np.nonzero(g > g_max)[0]
    ok = np.ones(T.shape, bool)
    for i in big:
        a, b = ts[i], ts[i + 1]
        ok &= ~((b > now[:, None]) & (a < T))
    return ok


def _sagitta_footprint_ok(sg: np.ndarray, s_nat: np.ndarray, s_lo: np.ndarray, s_hi: np.ndarray, lim: float) -> np.ndarray:
    """Vectorised ``chord_sagitta_max(...) <= lim`` for many (s_lo, s_hi): the same index range as the original
    (native samples whose arc is in [s_lo, s_hi], widened by one before and one after, chords between them)."""
    n = len(s_nat)
    i0 = np.searchsorted(s_nat, s_lo, "left")
    i1 = np.searchsorted(s_nat, s_hi, "right") - 1
    ok = np.zeros(s_lo.shape, bool)
    # sg[i] = chord i -> i+1 sagitta; original takes chords of indices a..b where a = max(i0-1,0), b = min(i1+1, n-1)
    # (np.diff over the widened index set gives chords a .. b-1).
    flat = np.nonzero(np.isfinite(s_lo))
    for idx in zip(*flat):
        a_, b_ = i0[idx], i1[idx]
        if b_ - a_ + 1 < 2:                                  # original: fewer than 2 samples -> inf -> not ok
            continue
        a = max(a_ - 1, 0)
        b = min(b_ + 1, n - 1)
        if b <= a:
            continue
        ok[idx] = float(sg[a:b].max()) <= lim
    return ok


def chord_sagitta_array(log) -> np.ndarray:
    c = np.hypot(np.diff(log.x), np.diff(log.y))
    return c * np.abs(np.diff(log.yaw)) / 8.0


def checkpoints(log, nows, horizons=HORIZONS_S, *, tr=None, path=None, s_nat=None, g_max=None):
    """Checkpoints for log times ``nows`` [n]. Returns (xyh [n, H, 3] float64 (NaN where invalid), valid [n, H] bool,
    why [n, H] uint8 first failing rule)."""
    L = B.LITERALS
    g_max = L["g_max_s"] if g_max is None else g_max
    nows = np.asarray(nows, np.float64)
    h = np.asarray(horizons, np.float64)
    tr = tr or B.track10(log)
    try:
        path = path or B.smooth_path(tr, L["rc_sigma_m"])
    except ValueError:        # a clip that never moves: fewer than 3 arc-grid points, no path and no heading
        n_, H_ = len(np.atleast_1d(nows)), len(np.atleast_1d(horizons))
        return np.full((n_, H_, 3), np.nan), np.zeros((n_, H_), bool), np.full((n_, H_), 6, np.uint8)
    s_nat = B._ro(B.s_at(tr, log.ts)) if s_nat is None else s_nat
    n, H = len(nows), len(h)
    x0, y0, psi0, _ = B.sample(log, nows)
    T = nows[:, None] + h[None, :]
    s_t = B.s_at(tr, T)
    px, py, pp = (np.interp(s_t, path.s, a) for a in (path.x, path.y, path.psi))
    c, s = np.cos(psi0)[:, None], np.sin(psi0)[:, None]
    dx, dy = px - x0[:, None], py - y0[:, None]
    xl, yl = c * dx + s * dy, -s * dx + c * dy
    rel = (pp - psi0[:, None] + np.pi) % (2 * np.pi) - np.pi
    why = np.zeros((n, H), np.uint8)

    def mark(bad, code):
        w = (why == 0) & bad
        why[w] = code

    mark(np.broadcast_to((nows < -1e-9)[:, None], (n, H)), 1)
    mark(T > log.t_end + 1e-9, 2)
    mark(~_gap_ok(log.ts, nows, T, g_max), 3)
    mark((s_t > path.s_hi_valid) | (s_t < path.s[0]), 4)
    foot = L["rc_kernel_sigmas"] * path.sigma
    chk = why == 0
    lo = np.where(chk, s_t - foot, np.nan)
    hi = np.where(chk, s_t + foot, np.nan)
    sag_ok = _sagitta_footprint_ok(chord_sagitta_array(log), s_nat, lo, hi, L["rc_sagitta_max_m"])
    mark(chk & ~sag_ok, 5)
    valid = why == 0
    xyh = np.stack([xl, yl, rel], axis=-1)
    xyh[~valid] = np.nan
    return xyh, valid, why


def load_ego(path: str):
    return B.load_log(path)


def max_gap_matrix(ts: np.ndarray, nows: np.ndarray, horizons=HORIZONS_S) -> np.ndarray:
    """Largest native-sample gap (s) among the gaps overlapping [now, now + h]; NaN where now + h is past the log end
    (the same sample range as ``build_v9_labels.max_gap``)."""
    h = np.asarray(horizons, np.float64)
    g = np.diff(ts)
    out = np.full((len(nows), len(h)), np.nan)
    for i, now in enumerate(nows):
        lo = max(int(np.searchsorted(ts, now, "right")) - 1, 0)
        for j, hh in enumerate(h):
            T = now + hh
            if T > ts[-1] + 1e-9 or now < ts[0] - 1e-9:
                continue
            hi = min(int(np.searchsorted(ts, T, "left")), len(ts) - 1)
            out[i, j] = 0.0 if hi <= lo else g[lo:hi].max()
    return out
