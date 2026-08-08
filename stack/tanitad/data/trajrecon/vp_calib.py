"""Camera roll and pitch from the **vertical** vanishing point.

Roll is the one rotation nothing else here can measure.  The focus of expansion
is blind to it by construction -- a radial flow field is roll-symmetric.  The
road-plane homography can see it in principle but not in practice: ``roll ~ n_x``
is the plane normal's *lateral* tilt, and a road corridor 27 m long by 6.4 m wide
is depth-dominated, so the estimate came back +0.66 deg with a 95 % CI of
[-0.46, +1.38] -- indistinguishable from zero.

Vertical world lines fix it, and they have exactly the leverage the road lacks:
lamp posts, sign poles, building corners and fence posts are spread across the
full image *width*.  Their common vanishing point is the world's "up" direction
seen by the camera, which is the same quantity as the road-plane normal --
reached from completely different evidence.  Static image structure, no motion,
no trajectory, no IMU.

Given the vanishing point ``v`` in homogeneous pixels, ``d = K^-1 v`` is that
direction in camera coordinates, and the same closed form used for the plane
normal applies::

    pitch = asin(d_z)
    roll  = atan2(d_x, -d_y)

In principle this yields a third estimate of pitch too -- but in practice it is
worthless, and the reason is worth stating because it is structural rather than
bad luck.  Write the vanishing point homogeneously as ``v = (vx, vy, w)``.  Then
``d_z = w``, so ``pitch = asin(w/|d|)`` depends *entirely* on the **convergence**
of the vertical lines.  For a roughly level camera those lines are almost
parallel, ``w`` is tiny, and it is swamped by pixel noise and by any residual
lens distortion -- which bends straight lines and biases convergence
systematically.  Roll, by contrast, is the common *direction* of the same lines,
which is robust however near infinity the point lies.

Measured on the 08-11 session (91 frames, 2491 vertical lines, 100 % inliers):

======  ===========  ====================  ==========================
param   estimate     95 % CI               verdict
======  ===========  ====================  ==========================
roll    -0.20 deg    [-0.53, +0.81]        **use** -- tightest available
pitch   +10.32 deg   [+9.62, +11.18]       **reject** -- ill-conditioned, 3.7 deg from two agreeing methods
======  ===========  ====================  ==========================

So the three methods divide cleanly along their conditioning: the focus of
expansion gives yaw, the road plane gives pitch and height, and vertical lines
give roll.  Each is used only where it is well posed.

**Roll is now settled.** Two independent estimates -- road plane +0.66 deg
[-0.46, +1.38] and vertical VP -0.20 deg [-0.53, +0.81] -- both straddle zero.
The long-standing assumption of roll = 0 is not merely convenient, it is
confirmed, and the ``ground_calib`` fit that once confidently reported +0.30 to
+0.79 deg was reading its own noise.

Working in homogeneous coordinates matters.  For a roughly level camera the
vertical vanishing point is near infinity -- vertical lines project almost
parallel -- and any formulation that needs finite pixel coordinates falls apart
there.  The null-space form handles it without a special case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VPCalibResult:
    roll_deg: float
    pitch_deg: float
    roll_ci_deg: tuple
    pitch_ci_deg: tuple
    n_frames: int
    n_lines: int
    inlier_frac: float
    roll_series_deg: np.ndarray
    pitch_series_deg: np.ndarray

    def summary(self) -> str:
        return (f"VerticalVP from {self.n_frames} frames / {self.n_lines} vertical lines\n"
                f"  roll  = {self.roll_deg:+.2f} deg   95% CI "
                f"[{self.roll_ci_deg[0]:+.2f}, {self.roll_ci_deg[1]:+.2f}]\n"
                f"  pitch = {self.pitch_deg:+.2f} deg   95% CI "
                f"[{self.pitch_ci_deg[0]:+.2f}, {self.pitch_ci_deg[1]:+.2f}]\n"
                f"  inliers {self.inlier_frac:.0%}")


def direction_to_roll_pitch(d):
    """Camera-frame world-up direction -> (roll, pitch) in radians."""
    d = np.asarray(d, dtype=float).ravel()
    d = d / (np.linalg.norm(d) + 1e-12)
    if d[1] > 0:                      # world "up" must point up in the image (y is down)
        d = -d
    pitch = float(np.arcsin(np.clip(d[2], -1.0, 1.0)))
    roll = float(np.arctan2(d[0], -d[1]))
    return roll, pitch


def vertical_lines(img, min_len_frac: float = 0.055, max_tilt_deg: float = 28.0,
                   horizon_v: float | None = None):
    """Detect near-vertical line segments, as homogeneous image lines.

    Returns ``(lines, segs)`` with ``lines`` an (n, 3) array of ``p1 x p2``.
    Segments below the horizon are dropped: the road surface produces long
    near-vertical edges (lane markings, kerbs, tyre tracks) that converge on the
    *road* vanishing point, not the vertical one, and they would drag the fit.
    """
    import cv2

    h, w = img.shape[:2]
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_len = min_len_frac * h

    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        seg = lsd.detect(g)[0]
        seg = np.zeros((0, 4)) if seg is None else seg.reshape(-1, 4)
    except Exception:                                   # LSD absent in some builds
        e = cv2.Canny(g, 60, 180)
        hl = cv2.HoughLinesP(e, 1, np.pi / 720, 60,
                             minLineLength=int(min_len), maxLineGap=6)
        seg = np.zeros((0, 4)) if hl is None else hl.reshape(-1, 4).astype(float)

    if len(seg) == 0:
        return np.zeros((0, 3)), np.zeros((0, 4))

    x1, y1, x2, y2 = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    # angle away from image-vertical
    tilt = np.degrees(np.arctan2(np.abs(dx), np.abs(dy) + 1e-9))
    keep = (length > min_len) & (tilt < max_tilt_deg)
    if horizon_v is not None:
        keep &= (np.minimum(y1, y2) < horizon_v)        # must reach above the horizon
    seg = seg[keep]
    if len(seg) == 0:
        return np.zeros((0, 3)), seg

    p1 = np.stack([seg[:, 0], seg[:, 1], np.ones(len(seg))], axis=1)
    p2 = np.stack([seg[:, 2], seg[:, 3], np.ones(len(seg))], axis=1)
    lines = np.cross(p1, p2)
    lines /= (np.linalg.norm(lines[:, :2], axis=1, keepdims=True) + 1e-12)
    return lines, seg


def fit_vanishing_point(lines, iters: int = 400, thresh: float = 2.0, rng=None):
    """Robust vanishing point as the homogeneous null point of a line pencil.

    RANSAC on pairs, scored by point-line distance, then a least-squares
    refinement on the inliers.  Returns ``(v, inlier_mask)`` with ``v``
    homogeneous and unit-norm -- it stays valid when the point is at infinity,
    which is the normal case for a level camera.
    """
    n = len(lines)
    if n < 8:
        return None, None
    rng = rng or np.random.default_rng(0)

    best_v, best_in = None, None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        v = np.cross(lines[i], lines[j])
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            continue
        v = v / nv
        # distance from a line to the point; scale-free via the point's own norm
        d = np.abs(lines @ v) / (np.linalg.norm(v[:2]) + 1e-9)
        inl = d < thresh
        if best_in is None or inl.sum() > best_in.sum():
            best_v, best_in = v, inl
    if best_v is None or best_in.sum() < 6:
        return None, None

    L = lines[best_in]
    _, _, vt = np.linalg.svd(L)
    v = vt[-1]
    return v / (np.linalg.norm(v) + 1e-12), best_in


def estimate(video, cam, frame_indices, horizon_pad: float = 0.0, rng=None):
    """Vertical-VP roll and pitch, medianed over frames with a bootstrap CI."""
    Kinv = np.linalg.inv(cam.K)
    hv = cam.horizon_v()
    rolls, pitches, nl, inl = [], [], 0, []

    for fi in frame_indices:
        img = video.read_bgr(int(fi)) if hasattr(video, "read_bgr") else None
        if img is None:
            continue
        lines, seg = vertical_lines(img, horizon_v=hv + horizon_pad)
        if len(lines) < 8:
            continue
        v, mask = fit_vanishing_point(lines, rng=rng)
        if v is None:
            continue
        d = Kinv @ v
        r, p = direction_to_roll_pitch(d)
        # a windscreen mount is never wildly rolled; reject fits that say so
        if abs(np.rad2deg(r)) > 20 or abs(np.rad2deg(p)) > 25:
            continue
        rolls.append(np.rad2deg(r)); pitches.append(np.rad2deg(p))
        nl += len(lines); inl.append(mask.mean())

    if len(rolls) < 6:
        return None
    R = np.array(rolls); P = np.array(pitches)
    rng2 = np.random.default_rng(1)

    def ci(x):
        m = np.array([np.median(rng2.choice(x, len(x), replace=True)) for _ in range(3000)])
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    return VPCalibResult(roll_deg=float(np.median(R)), pitch_deg=float(np.median(P)),
                         roll_ci_deg=ci(R), pitch_ci_deg=ci(P),
                         n_frames=len(R), n_lines=nl, inlier_frac=float(np.mean(inl)),
                         roll_series_deg=R, pitch_series_deg=P)


def apply(cam, res: VPCalibResult, set_roll: bool = True, set_pitch: bool = False,
          max_roll_deg: float = 8.0):
    """Write the vertical-VP roll onto a camera model.

    ``set_pitch`` is off by default for the same reason as in
    :mod:`trajlib.plane_calib`: the focus of expansion determines pitch from
    3 x 10^5 flow vectors and this estimate is more valuable as an independent
    check on it than as a replacement.  Roll is the opposite case -- nothing else
    measures it, so this is the estimate that stands.
    """
    import copy
    c = copy.copy(cam)
    applied = False
    if set_roll and abs(res.roll_deg) <= max_roll_deg:
        c.roll = np.deg2rad(res.roll_deg)
        applied = True
    if set_pitch:
        c.pitch = np.deg2rad(res.pitch_deg)
    c.source = dict(cam.source)
    c.source["vertical_vp"] = (
        f"{res.n_frames} frames/{res.n_lines} lines: roll {res.roll_deg:+.2f} "
        f"[{res.roll_ci_deg[0]:+.2f},{res.roll_ci_deg[1]:+.2f}], "
        f"pitch {res.pitch_deg:+.2f} [{res.pitch_ci_deg[0]:+.2f},{res.pitch_ci_deg[1]:+.2f}] deg"
        + ("" if applied else "  (roll NOT applied: outside plausible range)"))
    return c
