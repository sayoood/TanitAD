"""Curvature, drivability-constrained smoothing, and steering-wheel angle.

Why curvature is the right variable
-----------------------------------
The exported path showed a small lateral wobble -- ~0.042 m RMS, 0.24 m peak,
measured on straight stretches against a constant-curvature reference.  It is
*not* GNSS: switching the position measurement off entirely leaves it unchanged
at 0.0418 m.  It is 0.3-1 Hz content in the gyroscope, and it survives every
filter-tuning knob because the gyro is the propagation *input*, not a
measurement.

Whether that content is real is a question about the vehicle, not the filter.
Expressed as a steering rate it is the honest test:

    steering-wheel rate  =  d(kappa)/dt * wheelbase * steering_ratio

At the raw settings the 99th percentile came out at **313 deg/s**, sustained
through ordinary straight-line driving.  A driver holding a lane does not saw
the wheel back and forth at that rate; that is mount flex and body motion, not
a path the car ever drove.

So the fix is a constraint, not a tuning parameter: smooth ``kappa`` until the
implied steering rate is physically executable, then re-integrate.  Smoothing is
zero-phase and the whole session is available, so this adds **no lag** -- it is
the same past-and-future argument that motivates the RTS pass.

Vehicle
-------
Defaults are for the **Audi A6 e-tron**: wheelbase 2.946 m.  The steering ratio
is quoted as ~15.9:1 for the conventional rack; the car can be ordered with
progressive (variable-ratio) steering, in which case a single number is an
approximation and the estimate is best treated as accurate in shape and
approximate in scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AUDI_A6_ETRON = dict(
    name="Audi A6 e-tron",
    wheelbase_m=2.946,
    steering_ratio=15.9,
    understeer_gradient=0.0025,   # rad of extra steer per m/s^2 of lateral accel
    max_steer_wheel_deg=520.0,    # lock to lock / 2
    track_m=1.64,
)


@dataclass
class SteeringResult:
    t: np.ndarray
    curvature: np.ndarray          # 1/m, signed (+ = left turn)
    road_wheel_deg: np.ndarray     # average front-wheel steer angle
    wheel_deg: np.ndarray          # steering-wheel angle
    wheel_rate_deg_s: np.ndarray
    sigma_wheel_deg: np.ndarray   # 1-sigma, propagated from speed + gyro noise
    valid: np.ndarray              # False where speed is too low to observe steering
    vehicle: dict = field(default_factory=dict)

    def summary(self) -> str:
        v = self.wheel_deg[self.valid]
        r = np.abs(self.wheel_rate_deg_s[self.valid])
        if not len(v):
            return "SteeringResult: no valid samples"
        return (f"Steering [{self.vehicle.get('name', '?')}, L={self.vehicle.get('wheelbase_m')} m, "
                f"ratio {self.vehicle.get('steering_ratio')}:1]\n"
                f"  wheel angle : {v.min():+7.1f} .. {v.max():+7.1f} deg  "
                f"(p99 |.| = {np.percentile(np.abs(v), 99):.1f})\n"
                f"  wheel rate  : p99 {np.percentile(r, 99):6.1f} deg/s   max {r.max():6.1f} deg/s\n"
                f"  curvature   : max |k| {np.abs(self.curvature[self.valid]).max():.4f} 1/m "
                f"(min radius {1.0 / max(np.abs(self.curvature[self.valid]).max(), 1e-9):.1f} m)\n"
                f"  1-sigma     : median {np.median(self.sigma_wheel_deg[self.valid]):5.1f} deg "
                f"(p90 {np.percentile(self.sigma_wheel_deg[self.valid], 90):.1f})\n"
                f"  valid       : {100 * self.valid.mean():.1f} % of samples")


# --------------------------------------------------------------------------- #
def path_curvature(speed, yaw_rate, v_eps: float = 1.5):
    """Signed path curvature from speed and yaw rate, regularised at low speed.

    ``kappa = omega / v`` blows up as the vehicle stops.  Using
    ``omega * v / (v^2 + v_eps^2)`` instead tends smoothly to zero at standstill
    and to ``omega / v`` once moving, which is the honest statement: when the
    car is not moving, its path has no curvature to measure.  (The steering wheel
    may well be turned -- that is simply not observable from motion.)
    """
    speed = np.asarray(speed, dtype=float)
    yaw_rate = np.asarray(yaw_rate, dtype=float)
    return yaw_rate * speed / (speed ** 2 + v_eps ** 2)


def _zero_phase_lowpass(x, fs, fc):
    from scipy.signal import butter, sosfiltfilt
    fc = min(fc, 0.45 * fs)
    return sosfiltfilt(butter(2, fc, "low", fs=fs, output="sos"), x)


def drivable_curvature(t, speed, yaw_rate, wheelbase: float, steering_ratio: float,
                       max_wheel_rate_deg_s: float = 180.0, v_eps: float = 1.5,
                       fc_lo: float = 0.15, fc_hi: float = 5.0):
    """Smooth curvature until the implied steering rate is physically executable.

    Returns ``(kappa_smooth, fc_used, stats)``.  The cutoff is found by bisection
    on the p99 steering rate rather than being guessed, so the criterion is a
    vehicle limit and not a magic number.  Filtering is zero-phase.
    """
    t = np.asarray(t, dtype=float)
    fs = 1.0 / np.median(np.diff(t))
    kap = path_curvature(speed, yaw_rate, v_eps)

    def wheel_rate_p99(k):
        dk = np.gradient(k, t)
        return float(np.percentile(np.abs(np.rad2deg(dk * wheelbase * steering_ratio)), 99))

    raw = wheel_rate_p99(kap)
    if raw <= max_wheel_rate_deg_s:
        return kap, None, dict(raw_p99=raw, final_p99=raw)

    lo, hi = fc_lo, fc_hi
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if wheel_rate_p99(_zero_phase_lowpass(kap, fs, mid)) > max_wheel_rate_deg_s:
            hi = mid
        else:
            lo = mid
    kap_s = _zero_phase_lowpass(kap, fs, lo)
    return kap_s, lo, dict(raw_p99=raw, final_p99=wheel_rate_p99(kap_s))


def reintegrate(traj, kappa, blend_hz: float = 0.08):
    """Rebuild heading and position from a smoothed curvature profile.

    The heading is re-integrated from ``v * kappa`` so the path is drivable by
    construction.  Position is then re-integrated along that heading, and the
    low-frequency difference against the RTS solution is added back, so absolute
    accuracy (which GNSS owns) is preserved while the high-frequency shape
    (which the vehicle model owns) comes from the smooth curvature.

    Returns ``(E, N, psi)``.
    """
    t = traj.t
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    v = traj.speed

    psi0 = float(np.unwrap(traj.heading)[0])
    psi = psi0 + np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] * kappa[1:] + v[:-1] * kappa[:-1]) * dt)])
    # keep the long-term heading tied to the estimate; only replace its fine structure
    dpsi = np.unwrap(traj.heading) - psi
    psi = psi + _zero_phase_lowpass(dpsi, fs, blend_hz)

    dE = v * np.cos(psi)
    dN = v * np.sin(psi)
    E = traj.E[0] + np.concatenate([[0.0], np.cumsum(0.5 * (dE[1:] + dE[:-1]) * dt)])
    N = traj.N[0] + np.concatenate([[0.0], np.cumsum(0.5 * (dN[1:] + dN[:-1]) * dt)])
    E = E + _zero_phase_lowpass(traj.E - E, fs, blend_hz)
    N = N + _zero_phase_lowpass(traj.N - N, fs, blend_hz)
    return E, N, psi


def apply_drivability(traj, vehicle: dict | None = None,
                      max_wheel_rate_deg_s: float = 180.0, blend_hz: float = 0.08):
    """Replace a trajectory's path with its drivable equivalent, in place-ish.

    .. warning::
       **Off by default, and it should stay that way unless you have measured
       the trade on your own data.**  Low-passing curvature at the cutoff needed
       to hit a 180 deg/s steering-rate limit (0.366 Hz on the 08-11 session)
       also smears genuine sharp manoeuvres.  Measured there: it rotates the
       heading by up to 4.7 deg through the junction turn and displaces the 5 s
       ego-window endpoint by up to **2.48 m** (median 0.18 m, p95 0.94 m) --
       enough to swing the projected ribbon clean off the road.  It was bought
       to suppress a 0.042 m ripple.  That is a bad trade, and the ripple was
       mostly a *rendering* artefact anyway (integer-pixel rasterisation, see
       ``viz.draw_bev_cv``), not a property of the trajectory.

       Rate-limiting is still the right thing for the *steering read-out*, where
       it costs nothing -- :func:`estimate_steering` does that internally
       without touching the path.

    Returns ``(traj, info)`` where ``traj`` is a shallow copy with E/N/psi
    rewritten and ``info`` records what the constraint had to do.
    """
    import copy

    veh = {**AUDI_A6_ETRON, **(vehicle or {})}
    kap, fc, st = drivable_curvature(traj.t, traj.speed, traj.yaw_rate,
                                     veh["wheelbase_m"], veh["steering_ratio"],
                                     max_wheel_rate_deg_s=max_wheel_rate_deg_s)
    out = copy.copy(traj)
    out.x = traj.x.copy()
    E, N, psi = reintegrate(traj, kap, blend_hz=blend_hz)
    from .trajectory import IE, IN, IPSI
    out.x[:, IE], out.x[:, IN], out.x[:, IPSI] = E, N, psi
    out.meta = dict(traj.meta)
    out.meta.update(drivability_fc_hz=fc, wheel_rate_p99_before=st["raw_p99"],
                    wheel_rate_p99_after=st["final_p99"])
    return out, st | dict(fc=fc)


# --------------------------------------------------------------------------- #
def estimate_steering(traj, vehicle: dict | None = None, v_min: float = 1.5,
                      v_eps: float = 1.5, use_understeer: bool = True,
                      max_wheel_rate_deg_s: float | None = 180.0) -> SteeringResult:
    """Steering-wheel angle from the reconstructed motion.

    Kinematic bicycle model with a linear understeer correction::

        delta      = atan(L * kappa)  +  K_us * a_lat
        wheel      = delta * steering_ratio

    The ``atan`` (rather than ``L * kappa``) matters only at very small radii but
    costs nothing.  The understeer term accounts for the extra steer a real car
    needs at speed to hold a given radius; it is linear in lateral acceleration,
    which is the standard single-track result.

    Below ``v_min`` the samples are flagged invalid rather than extrapolated:
    steering is genuinely unobservable from a stationary vehicle's motion.
    """
    veh = {**AUDI_A6_ETRON, **(vehicle or {})}
    v = traj.speed
    kap = path_curvature(v, traj.yaw_rate, v_eps)

    # Rate-limit the curvature used for the *read-out* only.  A steering wheel
    # that jitters at 240 deg/s on a straight road is obviously not what the
    # driver did, and the gauge should not display it.  Crucially this does not
    # touch the trajectory -- see the warning on apply_drivability for what
    # happens when you let this constraint rewrite the path.
    if max_wheel_rate_deg_s:
        kap, _, _ = drivable_curvature(traj.t, v, traj.yaw_rate, veh["wheelbase_m"],
                                       veh["steering_ratio"],
                                       max_wheel_rate_deg_s=max_wheel_rate_deg_s,
                                       v_eps=v_eps)

    delta = np.arctan(veh["wheelbase_m"] * kap)
    if use_understeer:
        delta = delta + veh["understeer_gradient"] * (v * traj.yaw_rate)

    wheel = np.rad2deg(delta) * veh["steering_ratio"]
    wheel = np.clip(wheel, -veh["max_steer_wheel_deg"], veh["max_steer_wheel_deg"])
    rate = np.gradient(wheel, traj.t)

    # This angle is *derived*, not measured, so its error is inherited.  With
    # kappa = omega / v, the relative error is the relative error of the yaw rate
    # plus that of the speed.  The gyro is excellent here (0.84 deg RMS heading
    # against held-out GNSS), so speed dominates -- and speed is GNSS-driven on
    # this device, at ~1.3 m/s RMS.  That is 12 % at 11 m/s but 40 % at 3 m/s,
    # which is exactly where the sharpest steering happens.  Worth carrying.
    from .trajectory import IV
    sigma_v = np.sqrt(np.maximum(traj.P[:, IV, IV], 1e-9))
    sigma_w = 0.004                     # rad/s, gyro after smoothing
    rel = np.sqrt((sigma_w / np.maximum(np.abs(traj.yaw_rate), 1e-3)) ** 2
                  + (sigma_v / np.maximum(v, v_eps)) ** 2)
    sigma_wheel = np.abs(wheel) * np.clip(rel, 0.0, 2.0)

    return SteeringResult(t=traj.t, curvature=kap, road_wheel_deg=np.rad2deg(delta),
                          wheel_deg=wheel, wheel_rate_deg_s=rate,
                          sigma_wheel_deg=sigma_wheel,
                          valid=v >= v_min, vehicle=veh)


def steering_at(res: SteeringResult, t):
    """Interpolate the steering result at arbitrary times."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    w = np.interp(t, res.t, res.wheel_deg)
    ok = np.interp(t, res.t, res.valid.astype(float)) > 0.5
    return w, ok
