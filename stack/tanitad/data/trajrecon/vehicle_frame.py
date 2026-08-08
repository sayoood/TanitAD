"""Estimate the fixed phone -> vehicle rotation for a rigidly mounted device.

The phone sits in a cradle at an arbitrary, unknown attitude.  Every downstream
step needs its measurements in the vehicle frame (FLU: x forward, y left,
z up), so that rotation has to be recovered from the data.

The previous implementation solved this by correlating each raw IMU axis against
a GPS derivative *inside every 5 s window* and keeping the single best-scoring
axis.  Three things go wrong with that:

* only one axis is used, so any mount that is not near-axis-aligned is
  mis-modelled by up to 45 degrees;
* the winning axis can change between neighbouring windows, which makes the
  exported trajectory discontinuous from one frame to the next; and
* it needs GPS, so it silently degrades to a hard-coded default exactly when
  GPS is bad -- which is when you least want a default.

Here the rotation is estimated **once per session** from all the data:

*   **Up** comes from the gravity vector, which is directly measured and needs
    no GPS at all.
*   **Left** comes from the centripetal relation ``a_lat = v * omega_up``: the
    lateral axis is the horizontal direction whose acceleration correlates with
    the yaw rate.  Also GPS-free.
*   **Forward** closes the right-handed triad, with the sign fixed by the known
    camera boresight (the rear camera looks out of ``-z_phone``, i.e. forward).

GPS, when healthy, is used only to *verify* the result, not to derive it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleFrame:
    R_vp: np.ndarray        # 3x3, maps a phone-frame vector into the vehicle frame
    up_phone: np.ndarray    # unit up axis, in phone coordinates
    left_phone: np.ndarray
    fwd_phone: np.ndarray
    lateral_score: float    # |corr| between a_lat and v*omega -- confidence in "left"
    tilt_stability_deg: float   # angular wobble of the gravity direction over the session
    fwd_sign_source: str

    def to_vehicle(self, v_phone: np.ndarray) -> np.ndarray:
        """(n, 3) phone-frame vectors -> (n, 3) vehicle-frame FLU vectors."""
        return np.asarray(v_phone, dtype=float) @ self.R_vp.T

    def summary(self) -> str:
        return (
            f"VehicleFrame  lateral_score={self.lateral_score:.3f}  "
            f"gravity_wobble={self.tilt_stability_deg:.2f} deg  fwd_sign={self.fwd_sign_source}\n"
            f"  fwd_phone  = {np.round(self.fwd_phone, 4)}\n"
            f"  left_phone = {np.round(self.left_phone, 4)}\n"
            f"  up_phone   = {np.round(self.up_phone, 4)}"
        )


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _sign_from_gnss(session, t, W, up, min_speed: float = 3.0, min_turn: float = 0.30):
    """+1 / -1: does rotation about ``up`` agree with the GNSS heading rate?

    Returns ``(sign, note)``.  The check needs the vehicle to have actually
    turned; with too little turning the correlation is meaningless and the sign
    is left alone, which the note records so it is visible downstream.
    """
    from .geo import bearing_to_math_angle

    if not session.has("gps"):
        return 1.0, "sign unverified: no GNSS"
    g = session["gps"]
    if "bearing" not in g.columns or "speed" not in g.columns:
        return 1.0, "sign unverified: no bearing"
    tb = g["seconds_elapsed"].to_numpy(dtype=float)
    v = g["speed"].to_numpy(dtype=float)
    b = g["bearing"].to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(b) & (v > min_speed)
    if ok.sum() < 20:
        return 1.0, "sign unverified: too few moving fixes"
    psi = np.unwrap(bearing_to_math_angle(b[ok]))
    dpsi = np.gradient(psi, tb[ok])
    w = np.interp(tb[ok], t, W @ up)
    if np.std(dpsi) < min_turn * np.pi / 180.0 or np.std(w) < 1e-4:
        return 1.0, "sign unverified: too little turning"
    r = float(np.corrcoef(w, dpsi)[0, 1])
    if not np.isfinite(r) or abs(r) < 0.15:
        return 1.0, f"sign unverified: weak agreement (r={r:+.2f})"
    return (1.0 if r > 0 else -1.0), f"sign from GNSS (r={r:+.2f})"


def estimate_vehicle_frame(session, t_lo: float | None = None, t_hi: float | None = None,
                           min_speed_for_lateral: float = 2.0) -> VehicleFrame:
    """Recover the phone->vehicle rotation from a whole session."""
    acc = session["accel"]      # linear acceleration, gravity already removed
    gyr = session["gyro"]

    t = acc["seconds_elapsed"].to_numpy(dtype=float)
    m = np.ones_like(t, dtype=bool)
    if t_lo is not None:
        m &= t >= t_lo
    if t_hi is not None:
        m &= t <= t_hi
    t = t[m]
    A = acc[["x", "y", "z"]].to_numpy(dtype=float)[m]

    tw = gyr["seconds_elapsed"].to_numpy(dtype=float)
    W = np.stack([np.interp(t, tw, gyr[c].to_numpy(dtype=float)) for c in "xyz"], axis=1)

    # --- up ---------------------------------------------------------------- #
    if session.has("gravity"):
        grv = session["gravity"]
        tg = grv["seconds_elapsed"].to_numpy(dtype=float)
        G = np.stack([np.interp(t, tg, grv[c].to_numpy(dtype=float)) for c in "xyz"], axis=1)
        up = _unit(G.mean(axis=0))
        Gu = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
        wobble = float(np.rad2deg(np.arccos(np.clip(Gu @ up, -1, 1))).std())
        up_src = "gravity"
        # Android's gravity vector points *up* (a phone face-up reads +9.81 z);
        # iOS CoreMotion points it *down*.  Assuming one of them silently inverts
        # the yaw-rate sign, so the filter integrates every turn backwards --
        # measured on the Rose Ave recording as corr(omega, GNSS dpsi/dt) = -0.43,
        # which produced 300 m of episodic position error on otherwise clean data.
        # The sign is therefore *measured* rather than assumed.
        s, note = _sign_from_gnss(session, t, W, up)
        if s < 0:
            up = -up
            Gu = -Gu
        up_src = f"gravity ({note})"
    else:
        # No gravity stream (e.g. only the linear-acceleration channel was
        # exported).  A road vehicle rotates almost exclusively in yaw, so the
        # dominant principal axis of the gyro signal *is* the vertical.
        Wc = W - W.mean(axis=0)
        _, _, vt = np.linalg.svd(Wc, full_matrices=False)
        up = _unit(vt[0])
        # sign: yaw rate about "up" must agree with the GPS bearing rate
        if session.has("gps"):
            gps = session["gps"]
            tb = gps["seconds_elapsed"].to_numpy(dtype=float)
            if "bearing" in gps.columns and len(tb) > 5:
                psi = np.unwrap(np.deg2rad(90.0 - gps["bearing"].to_numpy(dtype=float)))
                dpsi = np.interp(t, tb, np.gradient(psi, tb))
                if np.sum((W @ up) * dpsi) < 0:
                    up = -up
        Gu = np.repeat(up[None, :], len(t), axis=0)
        wobble = float("nan")
        up_src = "gyro-PCA"

    # --- left -------------------------------------------------------------- #
    # Yaw rate about the *downward* vertical.  Android reports gravity pointing
    # up, hence the sign flip; the gyro-PCA fallback is already signed as "up".
    omega_up = np.sum(W * Gu, axis=1) if up_src == "gyro-PCA" else -np.sum(W * Gu, axis=1)
    A_h = A - np.outer(A @ up, up)              # horizontal linear acceleration

    # optional speed gate: centripetal acceleration only exists while moving
    w = np.ones(len(t))
    if session.has("gps"):
        gps = session["gps"]
        if "speed" in gps.columns:
            v_i = np.interp(t, gps["seconds_elapsed"].to_numpy(dtype=float),
                            gps["speed"].to_numpy(dtype=float))
            if np.ptp(v_i) > 0.5:               # only trust a GPS speed that actually varies
                w = (v_i > min_speed_for_lateral).astype(float)
                if w.sum() < 0.1 * len(w):
                    w = np.ones(len(t))

    # a_lat = v * omega_up  =>  the left axis maximises <a_h . l, omega_up>
    num = (A_h * (omega_up * w)[:, None]).sum(axis=0)
    num = num - up * (num @ up)                 # keep it horizontal
    left = _unit(num)

    proj = A_h @ left
    denom = np.sqrt(np.sum((proj * w) ** 2) * np.sum((omega_up * w) ** 2))
    score = float(abs(np.sum(proj * omega_up * w ** 2) / denom)) if denom > 1e-12 else 0.0

    # --- forward ----------------------------------------------------------- #
    # FLU is right-handed with x = y cross z
    fwd = _unit(np.cross(left, up))

    # The rear camera looks along -z of the phone body frame.  Whichever way the
    # phone is cradled, that boresight has a positive forward component.
    cam_axis = np.array([0.0, 0.0, -1.0])
    sign_src = "camera-boresight"
    if fwd @ cam_axis < 0:
        fwd, left = -fwd, -left

    # If GPS is healthy, prefer its (unambiguous) longitudinal-acceleration sign.
    if session.has("gps"):
        gps = session["gps"]
        tv = gps["seconds_elapsed"].to_numpy(dtype=float)
        vv = gps["speed"].to_numpy(dtype=float)
        # A single NaN makes np.ptp return NaN, the test False, and the whole
        # check silently skipped -- which is how the Rose Ave recording ended up
        # with a *reversed* forward axis (corr(a_fwd, dv/dt) = -0.77) resolved by
        # the weaker camera-boresight fallback.  One missing sample must not
        # disable a safety check.
        fin = np.isfinite(tv) & np.isfinite(vv)
        if fin.sum() > 5 and (np.max(vv[fin]) - np.min(vv[fin])) > 1.0:
            tv, vv = tv[fin], vv[fin]
            dv = np.gradient(vv, tv)
            dv_i = np.interp(t, tv, dv)
            a_f = A_h @ fwd
            if np.std(a_f) > 1e-6 and np.std(dv_i) > 1e-6:
                c = float(np.corrcoef(a_f, dv_i)[0, 1])
                if abs(c) > 0.15:
                    sign_src = f"gps-accel (r={c:+.2f})"
                    if c < 0:
                        fwd, left = -fwd, -left

    R = np.stack([fwd, left, up], axis=0)
    # re-orthonormalise (guards against tiny numerical drift)
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        R[2] *= -1.0

    return VehicleFrame(R_vp=R, up_phone=up, left_phone=left, fwd_phone=fwd,
                        lateral_score=score, tilt_stability_deg=wobble,
                        fwd_sign_source=f"{sign_src} | up={up_src}")
