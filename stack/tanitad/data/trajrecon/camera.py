"""Camera model and calibration for projecting the trajectory into the image.

Frames
------
* **Vehicle (FLU)** -- x forward, y left, z up, origin on the road under the
  rear axle.
* **Camera (OpenCV)** -- x right, y down, z along the optical axis.

The nominal vehicle->camera rotation for a forward-looking camera is therefore::

    x_cam = -y_veh,   y_cam = -z_veh,   z_cam =  x_veh

and the real mount differs from it by small yaw / pitch / roll offsets.

Calibrating the mount from the data
-----------------------------------
Two of those three offsets are directly observable and do not need a
calibration target.  While the vehicle drives straight, the optical flow of the
static world radiates from the **focus of expansion** -- the image point where
the velocity vector pierces the image plane.  For a forward-moving vehicle that
direction *is* "straight ahead", so the FOE pixel gives the camera's yaw and
pitch relative to the vehicle's forward axis directly.

Roll is not constrained by the FOE (a radial flow field is roll-symmetric).  It
is taken from the measured gravity direction instead, which pins the horizon.

Focal length is the one intrinsic that matters for this projection.  It is
estimated from the gyroscope: during a yaw at rate ``omega`` the image shifts by
``f * omega * dt`` pixels, so regressing measured image shift against the
(trustworthy) gyro yaw rate yields ``f`` in pixels.  Segments with high yaw rate
are used, because there rotation dominates the translation-induced flow that
otherwise biases the estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

R_CV_NOMINAL = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


def _rot(yaw, pitch, roll):
    """Small-offset rotation applied in the camera frame (yaw-pitch-roll, rad)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    # Signs are chosen so that the boresight of the resulting rotation lands at
    # image (cx + fx*tan(yaw)/cos(pitch), cy + fy*tan(pitch)) -- i.e. positive
    # yaw moves it right and positive pitch moves it *down*, matching how the
    # focus of expansion is measured.  Getting the pitch sign backwards puts the
    # reported horizon on the opposite side of the principal point from the FOE
    # that produced it.
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])     # about camera Y (down) -> yaw
    Rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])     # about camera X (right) -> pitch
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])     # about optical axis -> roll
    return Rz @ Rx @ Ry


@dataclass
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float = 1.25        # camera height above the road
    lateral_m: float = 0.0        # +left of the vehicle centreline
    longitudinal_m: float = 0.0   # +forward of the trajectory origin
    yaw: float = 0.0              # mount offsets, radians
    pitch: float = 0.0
    roll: float = 0.0
    source: dict = field(default_factory=dict)

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @property
    def R_cv(self):
        """Vehicle -> camera rotation."""
        return _rot(self.yaw, self.pitch, self.roll) @ R_CV_NOMINAL

    @property
    def t_v(self):
        """Camera position expressed in the vehicle frame."""
        return np.array([self.longitudinal_m, self.lateral_m, self.height_m])

    def hfov_deg(self):
        return float(np.rad2deg(2.0 * np.arctan(self.width / (2.0 * self.fx))))

    # ---------------------------------------------------------------- #
    def to_camera(self, pts_v: np.ndarray) -> np.ndarray:
        pts_v = np.atleast_2d(np.asarray(pts_v, dtype=float))
        return (pts_v - self.t_v) @ self.R_cv.T

    def project(self, pts_v: np.ndarray, min_z: float = 0.35):
        """Vehicle-frame points -> pixels.

        Returns ``(uv, valid)``.  ``valid`` is False for points at or behind the
        image plane; those pixels are meaningless and must never be drawn -- a
        point 1 cm behind the camera projects to a finite pixel far outside the
        frame and would otherwise paint a spurious line across the image.
        """
        pc = self.to_camera(pts_v)
        z = pc[:, 2]
        valid = z > min_z
        zs = np.where(valid, z, 1.0)
        u = self.fx * pc[:, 0] / zs + self.cx
        v = self.fy * pc[:, 1] / zs + self.cy
        return np.stack([u, v], axis=1), valid

    def horizon_v(self) -> float:
        """Image row of the horizon (the vanishing line of the road plane)."""
        d = self.R_cv @ np.array([1.0, 0.0, 0.0])
        return float(self.fy * d[1] / d[2] + self.cy) if abs(d[2]) > 1e-9 else float("nan")

    def summary(self) -> str:
        return (f"CameraModel {self.width}x{self.height}  f=({self.fx:.1f},{self.fy:.1f}) px  "
                f"HFOV={self.hfov_deg():.1f} deg\n"
                f"  mount: yaw={np.rad2deg(self.yaw):+.2f} pitch={np.rad2deg(self.pitch):+.2f} "
                f"roll={np.rad2deg(self.roll):+.2f} deg   height={self.height_m:.2f} m\n"
                f"  horizon row={self.horizon_v():.1f} px   sources={self.source}")


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #
def nominal_camera(width: int, height: int, hfov_deg: float = 66.0) -> CameraModel:
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return CameraModel(width=width, height=height, fx=fx, fy=fx,
                       cx=width / 2.0, cy=height / 2.0,
                       source={"intrinsics": f"nominal HFOV={hfov_deg} deg"})


def estimate_focal_from_gyro(t_cam, cam_yaw_rate, t_gyro, gyro_yaw_rate,
                             proc_width: int, full_width: int,
                             assumed_f_px: float, min_rate: float = 0.10):
    """Refine the focal length by regressing image yaw rate against the gyro.

    ``cam_yaw_rate`` must have been produced with ``assumed_f_px``; the ratio of
    the true to the assumed focal length is the regression slope's reciprocal.
    Only samples above ``min_rate`` rad/s are used: there the rotational flow
    dominates the translational flow that biases the estimate at low yaw rates.

    Magnitudes only: a focal length is positive, and the *sign* of the
    correlation just records whether the image-yaw and gyro-yaw conventions
    happen to agree (they differ by whether yaw is measured about "up" or
    "down").  Testing the signed correlation would throw away a perfectly good
    r = -0.91 fit.
    """
    g = np.interp(t_cam, t_gyro, gyro_yaw_rate)
    ok = np.isfinite(cam_yaw_rate) & (np.abs(g) > min_rate)
    if ok.sum() < 50:
        return None, 0.0, int(ok.sum())
    a = cam_yaw_rate[ok]
    b = g[ok]
    slope = float(np.sum(a * b) / np.sum(a * a))     # gyro ~ slope * cam
    r = float(np.corrcoef(a, b)[0, 1])
    f_proc = assumed_f_px * abs(slope)
    return f_proc * (full_width / proc_width), abs(r), int(ok.sum())


# --------------------------------------------------------------------------- #
# Extrinsics from the focus of expansion
# --------------------------------------------------------------------------- #
def estimate_foe(video, cam: CameraModel, t_windows, proc_width: int = 640,
                 max_frames: int = 900, min_track: int = 40,
                 cam_flow=None, max_rot_rate: float = 0.035):
    """Locate the focus of expansion over straight-driving windows.

    Each tracked feature's displacement defines a line through the FOE.  Stacking
    ``n^T p = n^T p_i`` for every flow vector (with ``n`` the normal to the flow
    direction) gives an over-determined linear system solved in the least-squares
    sense, with two IRLS re-weighting rounds to suppress independently moving
    objects such as oncoming traffic.

    The flow-line construction is only valid for **pure translation**.  Any
    rotation between the pair translates every vector bodily and drags the fitted
    FOE with it -- and a windshield mount pitches constantly over road bumps.
    Filtering on the vehicle's yaw rate alone does not catch that, because the
    bumps are vertical.  When ``cam_flow`` is supplied, frame pairs whose own
    measured yaw/pitch/roll rate exceeds ``max_rot_rate`` are dropped; on this
    footage that moved the fitted horizon by ~200 px, from clearly wrong to
    consistent with the visible vanishing point.
    """
    import cv2
    from .timesync import _iter_gray_frames

    scale = proc_width / video.width
    pw, ph = proc_width, int(round(video.height * scale / 2) * 2)
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    feat = dict(maxCorners=700, qualityLevel=0.01, minDistance=8, blockSize=7)
    mask = np.zeros((ph, pw), dtype=np.uint8)
    mask[int(0.10 * ph):int(0.88 * ph), :] = 255

    want = np.zeros(len(video.pts), dtype=bool)
    for lo, hi in t_windows:
        want |= (video.pts >= lo) & (video.pts <= hi)

    if cam_flow is not None:
        t_mid, _, comps = cam_flow
        quiet_t = t_mid[np.all(np.abs(comps) < max_rot_rate, axis=1)]
        if len(quiet_t) > 50:
            # a pair is usable only if its own midpoint was rotationally quiet
            mid = 0.5 * (video.pts[:-1] + video.pts[1:])
            near = np.abs(mid[:, None] - quiet_t[None, :]).min(axis=1) < 1e-6
            keep_pair = np.zeros(len(video.pts), dtype=bool)
            keep_pair[1:] = near
            keep_pair[:-1] |= near
            want &= keep_pair

    idxs = np.nonzero(want)[0]
    if len(idxs) < 10:
        return None
    if len(idxs) > max_frames:
        idxs = idxs[np.linspace(0, len(idxs) - 1, max_frames).astype(int)]
    first, last = int(idxs[0]), int(idxs[-1]) + 1
    keep = np.zeros(last - first, dtype=bool)
    keep[idxs - first] = True

    src = (video.iter_gray(pw, ph, first, last - first) if hasattr(video, "iter_gray")
           else _iter_gray_frames(video.path, pw, ph, first, last - first))

    A_rows, b_rows = [], []
    prev = None
    for k, frame in enumerate(src):
        if not keep[k]:
            prev = None
            continue
        if prev is None:
            prev = frame
            continue
        p0 = cv2.goodFeaturesToTrack(prev, mask=mask, **feat)
        prev_frame, prev = prev, frame
        if p0 is None or len(p0) < min_track:
            continue
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, p0, None, **lk)
        if p1 is None or st is None or st.sum() < min_track:
            continue
        a = p0[st.ravel() == 1].reshape(-1, 2)
        b = p1[st.ravel() == 1].reshape(-1, 2)
        d = b - a
        L = np.linalg.norm(d, axis=1)
        m = L > 0.7                       # ignore sub-pixel noise near the FOE
        if m.sum() < min_track:
            continue
        a, d = a[m], d[m]
        n = np.stack([-d[:, 1], d[:, 0]], axis=1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        A_rows.append(n)
        b_rows.append(np.sum(n * a, axis=1))

    if not A_rows:
        return None
    A = np.concatenate(A_rows)
    bb = np.concatenate(b_rows)
    w = np.ones(len(bb))
    foe = None
    for _ in range(3):
        Aw = A * w[:, None]
        foe, *_ = np.linalg.lstsq(Aw, bb * w, rcond=None)
        r = np.abs(A @ foe - bb)
        s = 1.4826 * np.median(r) + 1e-6
        w = 1.0 / (1.0 + (r / (3.0 * s)) ** 2)      # Cauchy re-weighting

    foe_full = foe / scale
    inl = float((np.abs(A @ foe - bb) < 3.0 * (1.4826 * np.median(np.abs(A @ foe - bb)) + 1e-6)).mean())
    return dict(foe_px=foe_full, n_flow=len(bb), inlier_frac=inl,
                yaw=float(np.arctan2(foe_full[0] - cam.cx, cam.fx)),
                pitch=float(np.arctan2(foe_full[1] - cam.cy, cam.fy)))


def vanishing_point_hough(video, frame_indices, canny=(60, 180), min_len: int = 90):
    """Independent vanishing-point estimate from straight edges (no motion used).

    Cross-check for :func:`estimate_foe`.  The two are derived from completely
    different signals -- one from inter-frame motion, one from single-frame
    geometry -- so agreement between them is real evidence that the mount angles
    are right, and disagreement is a red flag worth surfacing.

    Note the two measure subtly different things: this returns the vanishing
    point of the *street*, while the FOE is the direction the *vehicle* is
    actually travelling.  A car sitting slightly angled in its lane separates
    them by a degree or so, which is expected, not an error.
    """
    import cv2

    pts = []
    for idx in frame_indices:
        img = video.read_bgr(int(idx)) if hasattr(video, "read_bgr") else None
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, *canny)
        e[:int(0.25 * img.shape[0]), :] = 0            # sky carries no useful lines
        lines = cv2.HoughLinesP(e, 1, np.pi / 360, 80,
                                minLineLength=min_len, maxLineGap=12)
        if lines is None:
            continue
        A, b = [], []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = float(x2 - x1), float(y2 - y1)
            ang = abs(np.degrees(np.arctan2(dy, dx)))
            if ang < 12 or ang > 78:                   # keep oblique lines only
                continue
            n = np.array([-dy, dx]); n /= np.linalg.norm(n)
            A.append(n); b.append(float(n @ np.array([x1, y1], dtype=float)))
        if len(A) < 8:
            continue
        A = np.array(A); b = np.array(b); w = np.ones(len(b))
        vp = None
        for _ in range(4):
            vp, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
            r = np.abs(A @ vp - b)
            s = 1.4826 * np.median(r) + 1e-6
            w = 1.0 / (1.0 + (r / (2.5 * s)) ** 2)
        pts.append(vp)
    if not pts:
        return None
    P = np.array(pts)
    return dict(vp_px=np.median(P, axis=0), n_frames=len(P),
                spread_px=float(np.median(np.abs(P - np.median(P, axis=0)).sum(axis=1))))


def straight_windows(traj, t_video_start: float, video_duration: float,
                     max_yaw_rate: float = 0.05, min_speed: float = 4.0,
                     min_len: float = 1.0):
    """Video-time intervals where the vehicle drives straight and reasonably fast."""
    ok = (np.abs(traj.yaw_rate) < max_yaw_rate) & (traj.speed > min_speed)
    out = []
    i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            # clip to the video's own extent *before* testing the length, or a
            # window that lies entirely past the end of the clip survives the
            # test and then selects no frames at all
            lo = max(0.0, traj.t[i] - t_video_start)
            hi = min(video_duration, traj.t[j] - t_video_start)
            if hi - lo >= min_len:
                out.append((lo, hi))
            i = j + 1
        else:
            i += 1
    return out


def calibrate_camera(video, session, sync, traj=None, cam_flow=None, vframe=None,
                     height_m: float = 1.25, hfov_deg: float = 66.0,
                     refine_focal: bool = True, refine_foe: bool = True,
                     hough_frames: int = 40) -> CameraModel:
    """Build a CameraModel, refining focal length and mount angles from the data."""
    from .timesync import gyro_yaw_rate

    cam = nominal_camera(video.width, video.height, hfov_deg)

    if refine_focal and cam_flow is not None and session.has("gyro"):
        t_cam, _, comps = cam_flow
        if session.has("gravity"):
            tg, wg = gyro_yaw_rate(session)
        elif vframe is not None:
            # no gravity stream: take the yaw axis from the estimated vehicle frame
            gy = session["gyro"]
            tg = gy["seconds_elapsed"].to_numpy(dtype=float)
            wg = vframe.to_vehicle(gy[["x", "y", "z"]].to_numpy(dtype=float))[:, 2]
        else:
            tg = wg = None
        proc_w = 480
        assumed_f = (proc_w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        if tg is None:
            cam.source["intrinsics_note"] = "no gravity stream and no vehicle frame - kept nominal"
        else:
            f_new, r, n = estimate_focal_from_gyro(
                t_cam + sync.t_video_start, comps[:, 0], tg, wg, proc_w, video.width, assumed_f)
            if f_new is not None and r > 0.4 and 0.4 * cam.fx < f_new < 2.5 * cam.fx:
                cam.fx = cam.fy = float(f_new)
                cam.source["intrinsics"] = f"gyro-regressed f (r={r:.2f}, n={n})"
            else:
                cam.source["intrinsics_note"] = (
                    f"gyro focal refinement rejected (r={r:.2f}, n={n}) - kept nominal")

    if refine_foe and traj is not None:
        wins = straight_windows(traj, sync.t_video_start, video.duration)
        if not wins:
            cam.source["extrinsics_note"] = (
                "no straight-driving window inside the analysed clip - kept nominal mount")
        else:
            foe = estimate_foe(video, cam, wins, cam_flow=cam_flow)
            if foe is None:
                cam.source["extrinsics_note"] = "FOE fit produced no usable flow - kept nominal"
            elif foe["n_flow"] <= 2000:
                cam.source["extrinsics_note"] = (
                    f"only {foe['n_flow']} flow vectors (<2000) - kept nominal mount")
            elif abs(foe["yaw"]) > np.deg2rad(15) or abs(foe["pitch"]) > np.deg2rad(20):
                # A dashcam mount is never more than a few degrees off; a large
                # solution means the FOE fit was captured by moving objects.
                cam.source["extrinsics_note"] = (
                    f"FOE implausible (yaw={np.rad2deg(foe['yaw']):.1f} "
                    f"pitch={np.rad2deg(foe['pitch']):.1f} deg) - kept nominal")
            else:
                cam.yaw = foe["yaw"]
                cam.pitch = foe["pitch"]
                cam.source["extrinsics"] = (
                    f"FOE at ({foe['foe_px'][0]:.0f},{foe['foe_px'][1]:.0f}) px, "
                    f"{foe['n_flow']} flow vectors, inliers={foe['inlier_frac']:.2f}")

                # Independent geometric cross-check.  The per-frame Hough VP is
                # noisy -- individual frames on this footage scatter by 150-290 px
                # -- so it is a coarse sanity check on the FOE, not a competing
                # measurement, and it needs a decent number of frames before its
                # median settles.  Its own scatter is reported so the agreement
                # figure can be read in context.
                if hasattr(video, "read_bgr"):
                    idxs = np.linspace(0, len(video.pts) - 1, hough_frames).astype(int)
                    vp = vanishing_point_hough(video, idxs)
                    if vp is not None:
                        d = float(np.hypot(*(vp["vp_px"] - foe["foe_px"])))
                        cam.source["foe_vs_hough_vp"] = (
                            f"Hough VP ({vp['vp_px'][0]:.0f},{vp['vp_px'][1]:.0f}) px over "
                            f"{vp['n_frames']} frames (own scatter {vp['spread_px']:.0f} px); "
                            f"{d:.0f} px from FOE (~{np.rad2deg(d / cam.fx):.1f} deg)"
                            + ("" if d < 0.12 * video.width else "  <-- CHECK MOUNT"))
    cam.height_m = height_m
    return cam


# --------------------------------------------------------------------------- #
# Per-frame body attitude
# --------------------------------------------------------------------------- #
def body_attitude_deviation(session, vframe, highpass_hz: float = 0.05):
    """Instantaneous pitch/roll deviation of the body from its session mean.

    The mount angles from :func:`calibrate_camera` are a *session average*.  The
    car itself pitches over every bump and rolls in every corner, and the ground
    projection is brutally sensitive to that: with h = 1.34 m, a 0.35 deg pitch
    error -- the measured 1-sigma on this session -- moves the projected ground
    point at 30 m by **4.2 m**, and at 45 m by 9.4 m.  Holding pitch fixed is why
    the far end of the ribbon swims about between neighbouring frames.

    The deviation is obtained by integrating the gyro and high-passing away the
    drift.  Only the *deviation* is wanted -- the mean is already in the mount
    calibration -- so the notorious weakness of gyro integration is exactly the
    part being discarded.  Zero-phase, so no lag is introduced.

    Returns ``(t, d_pitch_noseup, d_roll)`` in radians.
    """
    from scipy.signal import butter, sosfiltfilt

    gy = session["gyro"]
    t = gy["seconds_elapsed"].to_numpy(dtype=float)
    W = vframe.to_vehicle(gy[["x", "y", "z"]].to_numpy(dtype=float))
    fs = 1.0 / np.median(np.diff(t))
    sos = butter(2, highpass_hz, "high", fs=fs, output="sos")

    def integ(w):
        p = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
        return sosfiltfilt(sos, p)

    # FLU: a positive rotation about +y (left) drops the nose, hence the sign
    return t, integ(-W[:, 1]), integ(W[:, 0])


def camera_at(cam, d_pitch: float = 0.0, d_roll: float = 0.0):
    """Copy of ``cam`` with an instantaneous body attitude applied.

    ``d_pitch`` is nose-up.  A nose-up body lifts the camera boresight, which in
    this convention (positive pitch drives the boresight *down* the image) means
    subtracting it.
    """
    import copy
    c = copy.copy(cam)
    c.pitch = cam.pitch - d_pitch
    c.roll = cam.roll + d_roll
    return c
