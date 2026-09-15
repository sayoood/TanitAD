"""Recover *true* longitudinal acceleration from whatever the phone exported.

Why this module exists
----------------------
Android's ``Accelerometer`` virtual sensor ("linear acceleration") is not the
raw accelerometer: it is the specific force minus a *continuously re-estimated*
gravity vector.  That estimator has a time constant of order a second, so any
acceleration sustained for longer than that leaks into the gravity estimate and
is subtracted away.  Measured on the 2025-08-11 session, integrating its forward
component over 75 s gives **-0.8 m/s** while the vehicle actually gained
**+7.8 m/s**, and its correlation with the GPS-derived acceleration is -0.06.
It carries road vibration and little else.

Feeding that to a fusion filter as the propagation input is worse than useless:
it injects noise while contributing no real longitudinal information.

Tiers, best first:

1.  ``TotalAcceleration`` (raw specific force) with gravity removed via the
    ``Orientation`` quaternion.  The attitude comes from gyro-led fusion, so
    sustained horizontal acceleration does not corrupt it the way an
    accel-led gravity estimator is corrupted.
2.  ``TotalAcceleration`` minus the ``Gravity`` stream -- same caveat as the
    linear channel, but kept as a fallback.
3.  ``Accelerometer`` (linear) -- short-term dynamics only.
4.  Nothing usable: the input is set to zero and the process noise is raised so
    speed becomes a random walk driven entirely by GNSS.  With RTS smoothing
    over 1 Hz Doppler speed this is still a perfectly good estimate; it is
    simply honest about where the information comes from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AccelSource:
    a_fwd: np.ndarray        # forward acceleration on the caller's time grid (m/s^2)
    tier: int
    name: str
    quality: float           # |corr| against GPS-derived dv/dt, NaN if unavailable
    usable: bool

    def __str__(self):
        q = "n/a" if not np.isfinite(self.quality) else f"{self.quality:.3f}"
        return (f"AccelSource(tier={self.tier}, {self.name}, quality={q}, "
                f"usable={self.usable}, std={self.a_fwd.std():.3f} m/s^2)")


def quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame ``v`` into the body frame given body->world ``q``.

    ``q`` is (n, 4) as (w, x, y, z); ``v`` is (3,) or (n, 3).
    """
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    v = np.broadcast_to(np.asarray(v, dtype=float), (len(q), 3))
    # body = R(q)^T world
    t = 2.0 * np.cross(np.stack([-x, -y, -z], axis=1), v)
    return v + w[:, None] * t + np.cross(np.stack([-x, -y, -z], axis=1), t)


def longitudinal_acceleration(session, vframe, t_grid: np.ndarray,
                              g0: float = 9.80665,
                              min_quality: float = 0.15) -> AccelSource:
    """Best available forward acceleration, resampled onto ``t_grid``."""
    def resamp(df, cols):
        t = df["seconds_elapsed"].to_numpy(dtype=float)
        return np.stack([np.interp(t_grid, t, df[c].to_numpy(dtype=float)) for c in cols], axis=1)

    candidates = []

    if session.has("total_accel") and session.has("orientation"):
        F = resamp(session["total_accel"], "xyz")
        ori = session["orientation"]
        to = ori["seconds_elapsed"].to_numpy(dtype=float)
        Q = np.stack([np.interp(t_grid, to, ori[c].to_numpy(dtype=float))
                      for c in ("qw", "qx", "qy", "qz")], axis=1)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
        g_body = quat_rotate_inv(Q, np.array([0.0, 0.0, g0]))   # world Z is up (ENU)
        candidates.append((1, "total_accel - R(q)^T*g", F - g_body))

    if session.has("total_accel") and session.has("gravity"):
        F = resamp(session["total_accel"], "xyz")
        G = resamp(session["gravity"], "xyz")
        candidates.append((2, "total_accel - gravity", F - G))

    if session.has("accel"):
        candidates.append((3, "accel (android linear)", resamp(session["accel"], "xyz")))

    # reference for scoring: GPS-derived longitudinal acceleration
    ref_t = ref_a = None
    if session.has("gps"):
        gps = session["gps"]
        tv = gps["seconds_elapsed"].to_numpy(dtype=float)
        vv = gps["speed"].to_numpy(dtype=float)
        fin = np.isfinite(tv) & np.isfinite(vv)
        if fin.sum() > 8 and (np.max(vv[fin]) - np.min(vv[fin])) > 1.0:
            tv, vv = tv[fin], vv[fin]
            ref_t, ref_a = tv, np.gradient(vv, tv)

    best = None
    for tier, name, A in candidates:
        a_f = vframe.to_vehicle(A)[:, 0]
        qual, signed = np.nan, np.nan
        if ref_t is not None:
            a_at = np.interp(ref_t, t_grid, a_f)
            if np.std(a_at) > 1e-6 and np.std(ref_a) > 1e-6:
                signed = float(np.corrcoef(a_at, ref_a)[0, 1])
                qual = abs(signed)
        # A *negative* correlation means the forward axis points backwards, so
        # the signal is good and the frame is wrong -- integrating it makes the
        # speed run the wrong way.  Reject rather than rectify: the frame should
        # be fixed upstream, and silently negating here would hide that.
        if np.isfinite(signed) and signed < -min_quality:
            name = f"{name} REVERSED (r={signed:+.2f}) - vehicle frame is wrong"
            qual = float(qual)
            cand = AccelSource(a_fwd=a_f, tier=tier, name=name, quality=qual, usable=False)
        else:
            # Unknown quality is *not* a licence to trust the stream.  Defaulting
            # unverifiable to usable is how a reversed accelerometer reached the
            # filter and turned a 3 m trajectory into a 77 m one.
            cand = AccelSource(a_fwd=a_f, tier=tier, name=name, quality=qual,
                               usable=bool(np.isfinite(qual) and qual >= min_quality))
        if best is None:
            best = cand
        # prefer a lower tier, but only if it is not demonstrably worse
        if np.isfinite(qual) and np.isfinite(best.quality) and qual > best.quality + 0.05:
            best = cand

    if best is None:
        return AccelSource(np.zeros(len(t_grid)), 4, "none", np.nan, False)

    if not best.usable:
        # Do not propagate on a signal that carries no longitudinal information.
        q = "unverifiable" if not np.isfinite(best.quality) else f"quality={best.quality:.2f}"
        return AccelSource(np.zeros(len(t_grid)), 4,
                           f"disabled ({best.name}, {q})", best.quality, False)
    return best
