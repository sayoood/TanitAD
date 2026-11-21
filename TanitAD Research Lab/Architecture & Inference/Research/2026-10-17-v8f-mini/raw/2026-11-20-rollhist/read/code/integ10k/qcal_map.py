#!/usr/bin/env python
"""qcal_map.py -- card PREREG_V8F_PROOF_ADD31 (QCAL): the gap map shared by the fit script (dev box) and the rollout tool (pod).

g_used = min(g_head, Q(g_head)), Q piecewise linear through the knots (xs, ys), xs increasing, ys non-decreasing, ys <= xs:
  below the first knot  Q(g) = ys[0] * g / xs[0]
  between knots         linear
  above the last knot   Q(g) = g - (xs[-1] - ys[-1])
A value that is not finite stays as it is.
"""
import json

import numpy as np


def load_map(path):
    d = json.load(open(path, encoding="utf-8"))
    xs, ys = np.asarray(d["xs"], np.float64), np.asarray(d["ys"], np.float64)
    check_map(xs, ys)
    return xs, ys


def check_map(xs, ys):
    if xs.ndim != 1 or xs.shape != ys.shape or xs.size < 2:
        raise ValueError("map needs at least two knots")
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        raise ValueError("map knots must be finite")
    if not (xs[0] > 0 and np.all(np.diff(xs) > 0)):
        raise ValueError("knot positions must be positive and increasing")
    if not (np.all(np.diff(ys) >= 0) and np.all(ys <= xs + 1e-12) and ys[0] >= 0):
        raise ValueError("knot values must be non-decreasing, not negative and not above their positions")


def apply_map(g, xs, ys):
    g = np.asarray(g)
    out = np.array(g, dtype=np.float64, copy=True)
    fin = np.isfinite(out)
    gg = out[fin]
    q = np.interp(gg, xs, ys)
    lo = gg < xs[0]
    q[lo] = ys[0] * gg[lo] / xs[0]
    hi = gg > xs[-1]
    q[hi] = gg[hi] - (xs[-1] - ys[-1])
    out[fin] = np.minimum(gg, q)
    return out


def unit_check(xs, ys):
    """Control q5: knots reproduce their values, the map is non-decreasing, g_used never above g_head."""
    k = apply_map(xs, xs, ys)
    grid = np.arange(0.0, 150.0001, 0.05)
    u = apply_map(grid, xs, ys)
    r = {"knots_reproduce": bool(np.max(np.abs(k - ys)) <= 1e-9), "non_decreasing": bool(np.all(np.diff(u) >= -1e-9)), "never_above_head": bool(np.all(u <= grid + 1e-9)),
         "nan_stays": bool(np.isnan(apply_map(np.array([np.nan, 5.0]), xs, ys)[0]))}
    r["pass"] = bool(all(r.values()))
    return r
