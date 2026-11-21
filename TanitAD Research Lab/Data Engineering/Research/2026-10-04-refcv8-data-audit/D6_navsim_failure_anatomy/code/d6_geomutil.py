"""D6 -- pure-geometry helpers shared by the navtest oracle harness (numpy + shapely only; same definitions as d6_p3_oracles.py).
A copy rather than an import so the harness runs under the navsim-1.1 tree, where d6_rescore.py (navsim-2 API) cannot be imported."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point

SPEED_SCALES = (0.8, 0.6, 0.4)
SNAP_BOUNDS = (("B075", 0.75), ("B150", 1.5))
V_VARIANTS = (("V80", 0.8), ("V60", 0.6), ("V40", 0.4))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def to_ego(rows_xyh, x0, y0, h0):
    a = np.asarray(rows_xyh, dtype=np.float64)
    dx, dy = a[:, 0] - x0, a[:, 1] - y0
    c, s = math.cos(-h0), math.sin(-h0)
    return np.stack([c * dx - s * dy, s * dx + c * dy, wrap(a[:, 2] - h0)], axis=1)


def scale_speed(plan, s):
    p = np.asarray(plan, dtype=np.float64)
    if s <= 0.0:
        return np.zeros_like(p)
    xy = np.vstack([[0.0, 0.0], p[:, :2]])
    yaw = np.concatenate([[0.0], np.unwrap(p[:, 2])])
    seg = np.hypot(*np.diff(xy, axis=0).T)
    a = np.concatenate([[0.0], np.cumsum(seg)])
    tgt = s * a[1:]
    out = np.zeros_like(p)
    out[:, 0] = np.interp(tgt, a, xy[:, 0])
    out[:, 1] = np.interp(tgt, a, xy[:, 1])
    out[:, 2] = np.interp(tgt, a, yaw)
    return out


def centerline_frame(mc, plan):
    ini = mc.ego_state
    x0, y0, h0 = ini.rear_axle.x, ini.rear_axle.y, ini.rear_axle.heading
    c, sn = math.cos(h0), math.sin(h0)
    ls = mc.centerline.linestring
    rows = []
    for x, y, _h in np.asarray(plan, dtype=np.float64):
        gx, gy = x0 + c * x - sn * y, y0 + sn * x + c * y
        s = float(ls.project(Point(gx, gy)))
        q = ls.interpolate(s)
        qa, qb = ls.interpolate(min(s + 1.0, ls.length)), ls.interpolate(max(s - 1.0, 0.0))
        tx, ty = qa.x - qb.x, qa.y - qb.y
        n = math.hypot(tx, ty) or 1.0
        tx, ty = tx / n, ty / n
        lat = (-ty) * (gx - q.x) + tx * (gy - q.y)
        rows.append((gx, gy, s, lat, q.x, q.y, tx, ty))
    return rows, (x0, y0, h0)


def snap_plan(mc, plan, mode, bound=None):
    rows, (x0, y0, h0) = centerline_frame(mc, plan)
    plan = np.asarray(plan, dtype=np.float64)
    out = plan.copy()
    ls = mc.centerline.linestring
    s_prev = -1e9
    for i, (gx, gy, s, lat, qx, qy, tx, ty) in enumerate(rows):
        if mode == "bound":
            lat2 = lat - math.copysign(min(abs(lat), bound), lat)
            nx, ny = gx - (lat - lat2) * (-ty), gy - (lat - lat2) * tx
            hh = plan[i, 2]
        else:
            s = max(s, s_prev)
            s_prev = s
            q = ls.interpolate(s)
            qa, qb = ls.interpolate(min(s + 1.0, ls.length)), ls.interpolate(max(s - 1.0, 0.0))
            nx, ny = q.x, q.y
            hh = wrap(math.atan2(qa.y - qb.y, qa.x - qb.x) - h0)
        e = to_ego([[nx, ny, 0.0]], x0, y0, h0)[0]
        out[i, 0], out[i, 1], out[i, 2] = e[0], e[1], hh
    return out
