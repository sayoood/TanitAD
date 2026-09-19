"""Quantitative validation of an estimated trajectory.

The residual against the GNSS fixes that were *used* in the fit is not an error
metric -- a filter that simply interpolates its own measurements scores well on
it.  The metric that matters is **hold-out**: drop a subset of the fixes, refit,
and measure the error where the filter was flying blind.  That is what
:func:`holdout_validation` does, and it is the number to quote.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geo import bearing_to_math_angle, wrap_pi
from .trajectory import estimate_trajectory


@dataclass
class ValidationReport:
    holdout: dict = field(default_factory=dict)
    insample: dict = field(default_factory=dict)
    consistency: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def __str__(self):
        def blk(title, d):
            if not d:
                return ""
            body = "\n".join(f"    {k:26s} {v}" for k, v in d.items())
            return f"  {title}\n{body}\n"
        out = ["[VALIDATION]"]
        out.append(blk("hold-out (GNSS not seen by the filter)", self.holdout))
        out.append(blk("in-sample (fit residual)", self.insample))
        out.append(blk("internal consistency", self.consistency))
        for n in self.notes:
            out.append(f"  note: {n}")
        return "\n".join(x for x in out if x)


def _pos_err(traj, gps, idx):
    t = gps["seconds_elapsed"].to_numpy(dtype=float)[idx]
    p = traj.enu.forward(gps["latitude"].to_numpy(dtype=float)[idx],
                         gps["longitude"].to_numpy(dtype=float)[idx],
                         gps["altitude"].to_numpy(dtype=float)[idx]
                         if "altitude" in gps.columns else None)[:, :2]
    inside = (t >= traj.t[0]) & (t <= traj.t[-1])
    t, p = t[inside], p[inside]
    E = np.interp(t, traj.t, traj.E)
    N = np.interp(t, traj.t, traj.N)
    return np.hypot(E - p[:, 0], N - p[:, 1]), t, inside


def _stats(e, unit=""):
    if len(e) == 0:
        return "n/a"
    return (f"mean={e.mean():.2f}{unit}  rms={np.sqrt((e ** 2).mean()):.2f}{unit}  "
            f"p95={np.percentile(e, 95):.2f}{unit}  max={e.max():.2f}{unit}  n={len(e)}")


def holdout_validation(session, vframe, k: int = 5, cfg=None, enu=None) -> dict:
    """k-fold hold-out over the GNSS fixes.

    Fold ``i`` removes every ``k``-th fix starting at ``i``, refits the whole
    session, and scores the estimate at exactly those removed fixes.  With a
    1 Hz receiver and ``k=5`` the filter must bridge ~5 s of GNSS outage, so
    this measures genuine dead-reckoning quality, not interpolation.
    """
    import copy

    gps_full = session["gps"]
    n = len(gps_full)
    errs, verrs, herrs = [], [], []

    for i in range(k):
        keep = np.ones(n, dtype=bool)
        keep[i::k] = False
        if keep.sum() < 8:
            continue
        sub = copy.copy(session)
        sub.sensors = dict(session.sensors)
        sub.sensors["gps"] = gps_full[keep].reset_index(drop=True)
        try:
            tr = estimate_trajectory(sub, vframe, cfg=cfg, enu=enu)
        except Exception:
            continue
        held = np.nonzero(~keep)[0]
        e, t_h, inside = _pos_err(tr, gps_full, held)
        errs.append(e)
        if "speed" in gps_full.columns:
            v_true = gps_full["speed"].to_numpy(dtype=float)[held][inside]
            verrs.append(np.interp(t_h, tr.t, tr.speed) - v_true)
        if "bearing" in gps_full.columns and "speed" in gps_full.columns:
            v_true = gps_full["speed"].to_numpy(dtype=float)[held][inside]
            b = bearing_to_math_angle(gps_full["bearing"].to_numpy(dtype=float)[held][inside])
            m = v_true >= 2.5
            if m.any():
                psi = np.interp(t_h, tr.t, np.unwrap(tr.heading))
                herrs.append(np.rad2deg(np.abs(wrap_pi(psi[m] - b[m]))))

    out = {}
    if errs:
        out["position (m)"] = _stats(np.concatenate(errs), " m")
    if verrs:
        out["speed (m/s)"] = _stats(np.abs(np.concatenate(verrs)), " m/s")
    if herrs:
        out["heading (deg, v>=2.5)"] = _stats(np.concatenate(herrs), " deg")
    out["folds"] = f"{k} (each holds out every {k}th fix ~ {k:.0f} s outage)"
    return out


def internal_consistency(traj) -> dict:
    """Checks that need no external reference at all."""
    out = {}
    dt = traj.t[1] - traj.t[0]

    # speed implied by differentiating position must match the speed state
    v_num = np.hypot(np.gradient(traj.E, dt), np.gradient(traj.N, dt))
    d = v_num - traj.speed
    out["|d(pos)/dt| vs speed state"] = f"rms={np.sqrt((d ** 2).mean()):.4f} m/s"

    # direction of travel must match the heading state (non-holonomic check)
    m = traj.speed > 2.0
    if m.sum() > 10:
        course = np.arctan2(np.gradient(traj.N, dt), np.gradient(traj.E, dt))
        dh = np.rad2deg(np.abs(wrap_pi(course[m] - traj.heading[m])))
        out["course vs heading (v>2)"] = f"rms={np.sqrt((dh ** 2).mean()):.3f} deg"

    out["speed non-negative"] = f"min={traj.speed.min():.3f} m/s"
    out["pos 1-sigma"] = (f"median={np.median(traj.pos_std()):.2f} m  "
                          f"max={traj.pos_std().max():.2f} m")
    a = np.gradient(traj.speed, dt)
    out["longitudinal accel"] = f"p99={np.percentile(np.abs(a), 99):.2f} m/s^2  max={np.abs(a).max():.2f}"
    yr = np.rad2deg(traj.yaw_rate)
    out["yaw rate"] = f"p99={np.percentile(np.abs(yr), 99):.1f} deg/s  max={np.abs(yr).max():.1f}"
    return out


def validate(session, vframe, traj, k: int = 5, cfg=None) -> ValidationReport:
    rep = ValidationReport()
    gps = session["gps"]

    e, _, _ = _pos_err(traj, gps, np.arange(len(gps)))
    rep.insample["position (m)"] = _stats(e, " m")
    if "speed" in gps.columns:
        t = gps["seconds_elapsed"].to_numpy(dtype=float)
        m = (t >= traj.t[0]) & (t <= traj.t[-1])
        dv = np.interp(t[m], traj.t, traj.speed) - gps["speed"].to_numpy(dtype=float)[m]
        rep.insample["speed (m/s)"] = _stats(np.abs(dv), " m/s")

    rep.holdout = holdout_validation(session, vframe, k=k, cfg=cfg, enu=traj.enu)
    rep.consistency = internal_consistency(traj)

    if traj.meta.get("accel_tier") == 4:
        rep.notes.append(
            "no usable longitudinal accelerometer: speed is GNSS-driven "
            f"({traj.meta.get('accel_source')})")
    return rep
