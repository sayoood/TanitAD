"""Session-level trajectory estimation: EKF forward pass + RTS backward pass.

Architecture note -- why this is done once per *session* and not per frame
------------------------------------------------------------------------
The previous pipeline re-solved a fresh filter inside each frame's 5 s window,
2247 times per recording.  That is wrong in three ways:

1.  **No past.**  The window ran ``[t_frame, t_frame + 5 s]``, so every solve
    started from a cold, zero-information state exactly at the frame.  The brief
    asks for backward *and* forward optimisation; only forward existed.
2.  **Inconsistency.**  Neighbouring frames are 33 ms apart but were solved
    independently, so their estimates disagree.  Differentiating the exported
    positions across frames yields a velocity that does not match the exported
    velocity, and biases learned from it are not physical.
3.  **Wasted information.**  A 5 s window sees at most ~5 GPS fixes.  The
    session has ~78.  Estimating IMU biases from 5 fixes is hopeless; over
    78 s they are well observed.

Estimating the whole session once and then *slicing* per frame fixes all three:
every frame automatically carries the full past and the full future, and
adjacent frames are guaranteed mutually consistent because they are views of a
single smoothed solution.

State
-----
``x = [E, N, v, psi, b_omega, b_a]`` in a local ENU tangent plane.

* ``E, N``   position (m)
* ``v``      forward speed (m/s), non-holonomic: the vehicle moves along its heading
* ``psi``    heading (rad, CCW from East)
* ``b_omega`` gyro yaw bias (rad/s)
* ``b_a``    forward accelerometer bias (m/s^2)

Propagation uses the IMU as a control input (accelerometer + gyro), and GNSS
position / speed / bearing enter as measurements.  The RTS pass then
redistributes future information backwards, which is what turns a causal
estimate into a *ground truth* estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .accel_source import longitudinal_acceleration
from .geo import LocalENU, bearing_to_math_angle, wrap_pi

IE, IN, IV, IPSI, IBW, IBA = range(6)
NX = 6


@dataclass
class TrajectoryResult:
    t: np.ndarray            # (N,) session seconds_elapsed
    x: np.ndarray            # (N, 6) smoothed state
    P: np.ndarray            # (N, 6, 6) smoothed covariance
    enu: LocalENU
    x_filt: np.ndarray       # (N, 6) forward-only (causal) estimate, for diagnostics
    meta: dict

    @property
    def E(self):    return self.x[:, IE]
    @property
    def N(self):    return self.x[:, IN]
    @property
    def speed(self): return self.x[:, IV]
    @property
    def heading(self): return self.x[:, IPSI]

    @property
    def yaw_rate(self):
        return np.gradient(np.unwrap(self.x[:, IPSI]), self.t)

    def pos_std(self):
        """1-sigma horizontal position uncertainty (m)."""
        return np.sqrt(self.P[:, IE, IE] + self.P[:, IN, IN])


# --------------------------------------------------------------------------- #
# Measurement assembly
# --------------------------------------------------------------------------- #
def _build_measurements(gps, enu, t_grid, cfg):
    """Snap each GPS fix to the nearest grid index and package its measurements."""
    t_g = gps["seconds_elapsed"].to_numpy(dtype=float)
    lat = gps["latitude"].to_numpy(dtype=float)
    lon = gps["longitude"].to_numpy(dtype=float)
    alt = gps["altitude"].to_numpy(dtype=float) if "altitude" in gps.columns else None
    pos = enu.forward(lat, lon, alt)[:, :2]

    v = gps["speed"].to_numpy(dtype=float) if "speed" in gps.columns else None
    brg = gps["bearing"].to_numpy(dtype=float) if "bearing" in gps.columns else None

    hacc = (gps["horizontalAccuracy"].to_numpy(dtype=float)
            if "horizontalAccuracy" in gps.columns else np.full(len(t_g), cfg["hacc_default"]))
    hacc = np.where(hacc > 0.1, hacc, cfg["hacc_default"])
    sacc = (gps["speedAccuracy"].to_numpy(dtype=float)
            if "speedAccuracy" in gps.columns else np.zeros(len(t_g)))
    sacc = np.where(sacc > 0.05, sacc, cfg["sacc_default"])
    bacc = (gps["bearingAccuracy"].to_numpy(dtype=float)
            if "bearingAccuracy" in gps.columns else np.zeros(len(t_g)))

    dt = t_grid[1] - t_grid[0]
    idx = np.clip(np.round((t_g - t_grid[0]) / dt).astype(int), 0, len(t_grid) - 1)
    inside = (t_g >= t_grid[0] - dt) & (t_g <= t_grid[-1] + dt)

    # A zero-velocity update may only fire when the vehicle is *actually* standing
    # still, which means the neighbouring fixes must be near zero too.  Trusting a
    # single isolated 0.0 punches a sharp notch straight to zero in the middle of a
    # crawl -- visible on the 08-11 session at t~48 s and t~54 s -- and a spurious
    # hard decelerate/accelerate pair is exactly the kind of artefact that would
    # teach a driving model wrong dynamics.
    if v is not None:
        below = v < cfg["zupt_speed"]
        stationary = below.copy()
        stationary[1:-1] = below[:-2] & below[1:-1] & below[2:]
        stationary[0] = below[0] & below[min(1, len(below) - 1)]
        stationary[-1] = below[-1] & below[max(-2, -len(below))]
    else:
        stationary = None

    meas = {}
    for k in range(len(t_g)):
        if not inside[k]:
            continue
        i = int(idx[k])
        entry = meas.setdefault(i, [])
        entry.append(("pos", pos[k], np.array([hacc[k] ** 2, hacc[k] ** 2])))
        if v is not None and np.isfinite(v[k]):
            entry.append(("spd", float(v[k]), float(sacc[k] ** 2),
                          bool(stationary[k]) if stationary is not None else False))
        # A bearing is only meaningful above a few m/s, and only if the receiver
        # reports a finite accuracy for it.  Devices that publish
        # bearingAccuracy == 0 for every fix are reporting "unknown", not
        # "perfect" -- treat that as a missing uncertainty, not a missing bearing.
        if brg is not None and v is not None and v[k] >= cfg["bearing_min_speed"]:
            sigma = np.deg2rad(bacc[k]) if bacc[k] > 0 else np.deg2rad(cfg["bacc_default_deg"])
            sigma = float(np.clip(sigma * cfg["bacc_inflate"],
                                  np.deg2rad(2.0), np.deg2rad(60.0)))
            entry.append(("brg", float(bearing_to_math_angle(brg[k])), sigma ** 2))
    return meas


# --------------------------------------------------------------------------- #
# Filter
# --------------------------------------------------------------------------- #
DEFAULT_CFG = dict(
    dt=0.01,
    sigma_accel=0.6,         # forward-accel process noise (m/s^2 / sqrt(Hz))
    sigma_lat=0.05,          # across-track model error (m/s / sqrt(Hz)); small,
                             # because a car is non-holonomic -- it cannot
                             # translate sideways off its own heading
    sigma_accel_no_imu=0.8,  # used when no usable longitudinal accel exists;
                             # tuned on GNSS hold-out subject to a physical
                             # cap on p99 longitudinal acceleration
    sigma_gyro=0.02,         # yaw-rate process noise (rad/s / sqrt(Hz))
    gyro_lowpass_hz=1.5,     # see estimate_trajectory: kills mount-flex ripple
    sigma_bw_rw=2e-4,        # gyro bias random walk (rad/s / s)
    sigma_ba_rw=5e-3,        # accel bias random walk (m/s^2 / s)
    hacc_default=8.0,
    sacc_default=0.6,
    bacc_default_deg=12.0,
    bacc_inflate=1.0,        # see below: the gyro beats GNSS bearing short-term
    bearing_min_speed=2.5,
    # A stationary receiver does not report 0.00 m/s -- Doppler noise keeps it
    # jittering a few tenths.  A 0.25 m/s threshold therefore almost never fires
    # on real data (a synthetic stop with 0.4 m/s speed noise triggered it in
    # 0 of 8 fixes), so the standstill constraint was effectively dead code.
    zupt_speed=0.7,          # GNSS speed below which the vehicle is called stopped
    zupt_sigma=0.25,         # how hard the zero-velocity update is applied
    zupt_noise_scale=0.05,   # process-noise multiplier while stopped
    min_speed=0.0,           # speed is non-negative: a forward dashcam never reverses
    min_speed_sigma=0.05,
    zupt_max_gap=3.0,        # don't call the vehicle stopped across a long GNSS gap
    gate_chi2=16.0,          # innovation gate (2-DOF, ~4 sigma)
    hacc_inflate=1.0,        # inflating the reported accuracy measurably hurt
                             # hold-out position; trust the receiver's own number
)


def _predict(x, P, a_f, omega, dt, cfg, stationary=False):
    E, N, v, psi, bw, ba = x
    a = a_f - ba
    w = omega - bw
    c, s = np.cos(psi), np.sin(psi)

    xn = np.array([
        E + v * c * dt + 0.5 * a * c * dt * dt,
        N + v * s * dt + 0.5 * a * s * dt * dt,
        v + a * dt,
        psi + w * dt,
        bw,
        ba,
    ])

    F = np.eye(NX)
    F[IE, IV] = c * dt
    F[IE, IPSI] = -(v * dt + 0.5 * a * dt * dt) * s
    F[IE, IBA] = -0.5 * c * dt * dt
    F[IN, IV] = s * dt
    F[IN, IPSI] = (v * dt + 0.5 * a * dt * dt) * c
    F[IN, IBA] = -0.5 * s * dt * dt
    F[IV, IBA] = -dt
    F[IPSI, IBW] = -dt

    # Continuous white-noise-acceleration model, discretised exactly.  Building
    # this block by hand is easy to get wrong: the position/velocity cross term
    # must satisfy Q_pp * Q_vv >= Q_pv^2 or Q stops being positive semi-definite,
    # the covariance goes indefinite and both the gain and the RTS pass degrade
    # into nonsense.  The dt^3/3, dt^2/2, dt triple below is PSD by construction.
    # A stationary vehicle cannot accelerate much, cannot slip sideways, and --
    # this is the one that matters -- cannot rotate.  Left at driving-level
    # process noise, the heading random-walks freely through a stop because it is
    # unobservable there: position carries no heading information at v = 0 and the
    # GNSS bearing is gated off.  Over an 8 s stop that alone wandered ~5 deg.
    sc = cfg["zupt_noise_scale"] if stationary else 1.0
    sa2 = (cfg["sigma_accel"] * sc) ** 2
    Q = np.zeros((NX, NX))
    u = np.array([c, s])
    Q[np.ix_((IE, IN), (IE, IN))] = sa2 * np.outer(u, u) * (dt ** 3 / 3.0)
    Q[IE, IV] = Q[IV, IE] = sa2 * c * (dt ** 2 / 2.0)
    Q[IN, IV] = Q[IV, IN] = sa2 * s * (dt ** 2 / 2.0)
    Q[IV, IV] = sa2 * dt
    # lateral model error (tyre slip, mount flex): keeps P non-singular across-track
    lat = np.array([-s, c])
    Q[np.ix_((IE, IN), (IE, IN))] += ((cfg["sigma_lat"] * sc) ** 2) * np.outer(lat, lat) * dt
    Q[IPSI, IPSI] = ((cfg["sigma_gyro"] * sc) ** 2) * dt
    Q[IBW, IBW] = (cfg["sigma_bw_rw"] ** 2) * dt
    Q[IBA, IBA] = (cfg["sigma_ba_rw"] ** 2) * dt

    Pn = F @ P @ F.T + Q
    Pn = 0.5 * (Pn + Pn.T)
    return xn, Pn, F


def _update_scalar(x, P, H, innov, R, gate=None):
    """Scalar EKF update with a *self-healing* robust gate.

    A hard chi-square gate that simply discards outliers is a trap here.  At a
    traffic stop the zero-velocity update drives P_vv to almost nothing; when the
    car pulls away the GPS speed innovation is then enormous relative to S, every
    update is rejected, and the filter never recovers -- on this session the
    speed stayed pinned near zero for the last 25 s and drifted negative.

    Instead an outlier is *down-weighted*: R is inflated by NIS/gate so the
    update still moves the state, just less.  Persistent evidence therefore wins
    eventually, which is the behaviour you want, while a single blunder is
    still suppressed.  This is the standard Huber-style robust Kalman variant.
    """
    S = float(H @ P @ H.T + R)
    if S <= 0:
        return x, P, False
    nis = innov * innov / S
    accepted = True
    if gate is not None and nis > gate:
        R = R * (nis / gate)
        S = float(H @ P @ H.T + R)
        accepted = False
    K = (P @ H.T) / S
    x = x + K * innov
    A = np.eye(NX) - np.outer(K, H)
    P = A @ P @ A.T + np.outer(K, K) * R          # Joseph form: stays symmetric PSD
    P = 0.5 * (P + P.T)
    return x, P, accepted


def estimate_trajectory(session, vframe, t_lo=None, t_hi=None, cfg=None,
                        enu: LocalENU | None = None) -> TrajectoryResult:
    """Run the forward EKF and the RTS smoother over an entire session."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    dt = cfg["dt"]

    acc, gyr, gps = session["accel"], session["gyro"], session["gps"]
    ta = acc["seconds_elapsed"].to_numpy(dtype=float)

    t_start = max(acc["seconds_elapsed"].iloc[0], gyr["seconds_elapsed"].iloc[0])
    t_end = min(acc["seconds_elapsed"].iloc[-1], gyr["seconds_elapsed"].iloc[-1])
    if t_lo is not None:
        t_start = max(t_start, t_lo)
    if t_hi is not None:
        t_end = min(t_end, t_hi)
    t_grid = np.arange(t_start, t_end, dt)
    n = len(t_grid)
    if n < 50:
        raise ValueError("session too short to estimate a trajectory")

    # --- IMU inputs in the vehicle frame ---------------------------------- #
    ta = acc["seconds_elapsed"].to_numpy(dtype=float)
    tw = gyr["seconds_elapsed"].to_numpy(dtype=float)
    W = np.stack([np.interp(t_grid, tw, gyr[c].to_numpy(dtype=float)) for c in "xyz"], axis=1)
    omega = vframe.to_vehicle(W)[:, 2]         # yaw rate about up

    # A road vehicle's yaw dynamics live below ~1.5 Hz -- on this session 98.2 %
    # of the gyro's yaw-rate power is under 0.5 Hz.  What is left above that is
    # body roll and windshield-mount flex, and integrating it puts a ~0.3 deg
    # wobble into the heading that shows up as visible ripple in the exported
    # path (0.3 deg is 0.25 m of lateral error 45 m ahead).  The measurement is
    # the *input* to the propagation, so no amount of tuning on the measurement
    # side removes it -- it has to be filtered here.  Zero-phase, so no lag.
    if cfg.get("gyro_lowpass_hz"):
        from scipy.signal import butter, sosfiltfilt
        fs = 1.0 / dt
        fc = min(cfg["gyro_lowpass_hz"], 0.45 * fs)
        omega = sosfiltfilt(butter(2, fc, "low", fs=fs, output="sos"), omega)

    asrc = longitudinal_acceleration(session, vframe, t_grid)
    a_f = asrc.a_fwd
    if not asrc.usable:
        # No trustworthy longitudinal input: model speed as a random walk whose
        # noise covers real vehicle accelerations, and let GNSS drive it.
        cfg = {**cfg, "sigma_accel": max(cfg["sigma_accel"], cfg["sigma_accel_no_imu"])}

    # --- anchor the tangent plane ------------------------------------------ #
    if enu is None:
        enu = LocalENU(float(gps["latitude"].iloc[0]), float(gps["longitude"].iloc[0]),
                       float(gps["altitude"].iloc[0]) if "altitude" in gps.columns else 0.0)
    meas = _build_measurements(gps, enu, t_grid, cfg)

    # --- initial state ------------------------------------------------------ #
    # iOS reports -1 for an unavailable speed or course, which the loader turns
    # into NaN.  Seeding the filter from row 0 regardless makes the whole state
    # NaN from the first step -- exactly what happened on the Rose Ave recording,
    # where the opening fixes have speed = -1 and bearing = -1.  Seed from the
    # first fix that actually carries a finite value.
    g0 = gps.iloc[0]
    p0 = enu.forward(float(g0["latitude"]), float(g0["longitude"]))[0, :2]

    def _first_finite(col, default):
        if col not in gps.columns:
            return default
        v = gps[col].to_numpy(dtype=float)
        ok = np.nonzero(np.isfinite(v))[0]
        return float(v[ok[0]]) if len(ok) else default

    v0 = max(_first_finite("speed", 0.0), 0.0)
    b0 = _first_finite("bearing", None)
    psi0 = float(bearing_to_math_angle(b0)) if b0 is not None else 0.0

    x = np.array([p0[0], p0[1], v0, psi0, 0.0, 0.0])
    P = np.diag([25.0, 25.0, 4.0, (np.pi / 2) ** 2, 1e-3, 0.5])

    x_f = np.zeros((n, NX));   P_f = np.zeros((n, NX, NX))
    x_p = np.zeros((n, NX));   P_p = np.zeros((n, NX, NX))
    F_s = np.zeros((n, NX, NX))
    n_used = 0

    x_f[0], P_f[0] = x, P
    x_p[0], P_p[0] = x, P
    F_s[0] = np.eye(NX)

    # Standstill spans the whole gap between two stationary GNSS fixes, not just
    # the instants they arrive: the constraint has to hold at every propagation
    # step in between or the state drifts away and gets yanked back once a second.
    stat_grid = np.zeros(n, dtype=bool)
    stat_idx = sorted(i for i, lst in meas.items()
                      if any(mm[0] == "spd" and len(mm) > 3 and mm[3] for mm in lst))
    for a, b in zip(stat_idx, stat_idx[1:]):
        if (b - a) * dt <= cfg["zupt_max_gap"]:
            stat_grid[a:b + 1] = True
    for i in stat_idx:
        stat_grid[i] = True

    for k in range(1, n):
        x, P, F = _predict(x_f[k - 1], P_f[k - 1], a_f[k - 1], omega[k - 1], dt, cfg,
                           stationary=bool(stat_grid[k]))
        x_p[k], P_p[k], F_s[k] = x, P, F

        if stat_grid[k]:
            H = np.zeros(NX); H[IV] = 1.0
            x, P, _ = _update_scalar(x, P, H, float(0.0 - x[IV]), cfg["zupt_sigma"] ** 2)

        for m in meas.get(k, []):
            kind, z, R = m[0], m[1], m[2]
            if kind == "pos":
                Rp = np.asarray(R) * (cfg["hacc_inflate"] ** 2)
                for j, ax in enumerate((IE, IN)):
                    H = np.zeros(NX); H[ax] = 1.0
                    x, P, ok = _update_scalar(x, P, H, float(z[j] - x[ax]), float(Rp[j]),
                                              gate=cfg["gate_chi2"])
                    n_used += ok
            elif kind == "spd":
                H = np.zeros(NX); H[IV] = 1.0
                x, P, ok = _update_scalar(x, P, H, float(z - x[IV]), float(R),
                                          gate=cfg["gate_chi2"])
                n_used += ok
                if len(m) > 3 and m[3]:
                    H = np.zeros(NX); H[IV] = 1.0
                    x, P, _ = _update_scalar(x, P, H, float(0.0 - x[IV]), cfg["zupt_sigma"] ** 2)
            elif kind == "brg":
                H = np.zeros(NX); H[IPSI] = 1.0
                x, P, ok = _update_scalar(x, P, H, float(wrap_pi(z - x[IPSI])), float(R),
                                          gate=cfg["gate_chi2"])
                n_used += ok

        # Nothing in the unicycle model forbids negative speed -- it just means
        # driving backwards -- but a forward-facing dashcam never does, and a
        # negative excursion flips the heading interpretation of every exported
        # window.  Enforce it as a pseudo-measurement so the covariance stays
        # consistent instead of clamping the state behind the filter's back.
        if x[IV] < cfg["min_speed"]:
            H = np.zeros(NX); H[IV] = 1.0
            x, P, _ = _update_scalar(x, P, H, float(cfg["min_speed"] - x[IV]),
                                     cfg["min_speed_sigma"] ** 2)
        x_f[k], P_f[k] = x, P

    # --- RTS backward pass -------------------------------------------------- #
    x_s = x_f.copy()
    P_s = P_f.copy()
    for k in range(n - 2, -1, -1):
        Pp = P_p[k + 1]
        try:
            C = P_f[k] @ F_s[k + 1].T @ np.linalg.inv(Pp)
        except np.linalg.LinAlgError:
            C = P_f[k] @ F_s[k + 1].T @ np.linalg.pinv(Pp)
        d = x_s[k + 1] - x_p[k + 1]
        d[IPSI] = wrap_pi(d[IPSI])          # heading residuals must not wrap around
        x_s[k] = x_f[k] + C @ d
        P_s[k] = P_f[k] + C @ (P_s[k + 1] - Pp) @ C.T

    x_s[:, IPSI] = np.unwrap(x_s[:, IPSI])

    meta = dict(n=n, dt=dt, n_updates=int(n_used), n_gps=len(gps),
                accel_tier=asrc.tier, accel_source=asrc.name,
                accel_quality=float(asrc.quality), accel_usable=bool(asrc.usable),
                t_start=float(t_grid[0]), t_end=float(t_grid[-1]),
                lat0=enu.lat0, lon0=enu.lon0, alt0=enu.alt0)
    return TrajectoryResult(t=t_grid, x=x_s, P=P_s, enu=enu, x_filt=x_f, meta=meta)


# --------------------------------------------------------------------------- #
# Per-frame extraction in the ego frame
# --------------------------------------------------------------------------- #
def to_vehicle_reference(ego: dict, longitudinal_m: float, lateral_m: float) -> dict:
    """Re-express a phone-referenced ego window about the vehicle reference point.

    GNSS locates the *phone*, so the raw trajectory is the phone's path.  The
    vehicle body does not follow it: a rigid body's points trace different
    curves, and a point mounted ``L`` metres ahead of the rear axle swings wide
    through every turn.  With the phone at vehicle-frame ``(L forward, W left)``
    the reference point's path is

        ref(t) = phone(t) - R(psi(t)) . (L, W)

    -- a *rotated, per-point* transform.  Shifting the path by a constant ``-W``
    instead (what the renderer used to do) is only right while the heading is
    constant; the residual is

        dy = -L.sin(psi) + W.(1 - cos psi)

    which is zero on straight road and grows with heading change.  Measured on
    the Rose Ave recording with L = 2.4 m that reaches **2.57 m** at the 5 s
    horizon -- more than a lane width -- while every straight-road check still
    showed centimetres.  That is exactly the error signature of a ribbon that
    looks right until the car turns.

    Speed is recomputed along the transformed path: in a turn the phone and the
    reference point genuinely move at different speeds (``v_phone - v_ref`` is
    of order ``omega * L``, ~0.6 m/s at 0.3 rad/s).
    """
    L, W = float(longitudinal_m), float(lateral_m)
    if L == 0.0 and W == 0.0:
        return ego
    out = dict(ego)
    psi = np.asarray(ego["yaw"], dtype=float)
    c, s = np.cos(psi), np.sin(psi)
    out["x"] = np.asarray(ego["x"], dtype=float) + L * (1.0 - c) + W * s
    out["y"] = np.asarray(ego["y"], dtype=float) - L * s + W * (1.0 - c)
    t = np.asarray(ego["t"], dtype=float)
    if len(t) > 2:
        sp = np.hypot(np.gradient(out["x"], t), np.gradient(out["y"], t))
        out["speed"] = sp
        out["speed_ref"] = float(np.interp(0.0, t, sp))
    out["reference"] = "vehicle"
    out["mount_longitudinal_m"] = L
    out["mount_lateral_m"] = W
    return out


def ego_trajectory(traj: TrajectoryResult, t_ref: float,
                   t_past: float = 3.0, t_future: float = 5.0,
                   dt_out: float = 0.1, standstill_speed: float = 0.15,
                   t_future_standstill: float = 1.0) -> dict | None:
    """Slice the smoothed session trajectory into the ego frame at ``t_ref``.

    Returns positions in FLU vehicle coordinates at ``t_ref`` -- x forward,
    y left, both zero at ``t_ref`` -- with negative ``t`` in the past and
    positive in the future.  This is the layout end-to-end driving models expect.

    **Standstill.**  A stopped vehicle's 5 s future is five seconds of zeros: it
    carries no trajectory information, and at a long light it can dominate the
    exported set with identical degenerate samples.  When the vehicle is
    genuinely stationary the future window is shortened to
    ``t_future_standstill`` and the record is flagged ``standstill=True``.

    The threshold is deliberately strict -- ``standstill_speed`` defaults to
    0.15 m/s, well below the 0.7 m/s the ZUPT uses -- because *low speed is not
    standstill*.  A vehicle creeping at 1 m/s still covers 5 m over the horizon
    and its path is exactly the kind of manoeuvre worth keeping.
    """
    t = traj.t
    if t_ref < t[0] or t_ref > t[-1]:
        return None

    v_ref = float(np.interp(t_ref, t, traj.x[:, IV]))
    standstill = bool(abs(v_ref) < standstill_speed)
    if standstill:
        t_future = min(t_future, float(t_future_standstill))

    ts = np.arange(-t_past, t_future + dt_out / 2, dt_out)
    tq = t_ref + ts
    valid = (tq >= t[0]) & (tq <= t[-1])

    E = np.interp(tq, t, traj.x[:, IE])
    N = np.interp(tq, t, traj.x[:, IN])
    v = np.interp(tq, t, traj.x[:, IV])
    psi = np.interp(tq, t, np.unwrap(traj.x[:, IPSI]))
    psi_r = float(np.interp(t_ref, t, np.unwrap(traj.x[:, IPSI])))
    E_r = float(np.interp(t_ref, t, traj.x[:, IE]))
    N_r = float(np.interp(t_ref, t, traj.x[:, IN]))

    c, s = np.cos(-psi_r), np.sin(-psi_r)
    dE, dN = E - E_r, N - N_r
    fwd = c * dE - s * dN
    left = s * dE + c * dN

    yaw = wrap_pi(psi - psi_r)
    yaw_rate = np.gradient(np.unwrap(psi), tq)
    pos_std = np.interp(tq, t, traj.pos_std())

    return dict(
        t_ref=float(t_ref),
        t=ts[valid],
        x=fwd[valid],            # +forward, metres
        y=left[valid],           # +left, metres
        speed=v[valid],
        yaw=yaw[valid],
        yaw_rate=yaw_rate[valid],
        pos_std=pos_std[valid],
        heading_ref=psi_r,
        speed_ref=v_ref,
        standstill=standstill,
        t_future_s=float(t_future),
        complete=bool(valid.all()),
    )
