"""Camera roll, pitch and height from road-plane homography decomposition.

Why this replaces the road-track least-squares fit
--------------------------------------------------
The earlier attempt (``ground_calib``) fitted four extrinsic parameters so as to
minimise the residual between tracked road points and the motion the trajectory
predicts.  That is ill-conditioned: the longitudinal channel carries a large LK
tracking bias, pitch and height are exactly the parameters that absorb it, and
the lateral channel cannot see pitch or height at all.  Refits with different
channel choices disagreed by 0.5 deg in roll and 0.23 m in height while reducing
the cost by 4 %.

This estimates the geometry **directly** instead of fitting a correction to it.
For a camera moving over a plane, the inter-frame homography factorises as

    H  =  R + (t/d) n^T

and decomposing it returns ``n``, the **road-plane normal in camera
coordinates**.  That normal *is* the camera's roll and pitch relative to the
road -- an output of the geometry, not a parameter tuned against a biased
residual.

With the camera convention used in :mod:`trajlib.camera`
(``R_cv = Rz(roll) Rx(pitch) Ry(yaw) @ R_CV_NOMINAL``) the road normal, which is
"up" in the vehicle frame, maps to::

    n_cam = [ sin(roll)cos(pitch),  -cos(roll)cos(pitch),  sin(pitch) ]

so the extraction is closed-form::

    pitch = asin(n_z)
    roll  = atan2(n_x, -n_y)

Yaw does not appear -- rotation about the plane normal leaves the plane
invariant -- which is exactly the right division of labour: **yaw comes from the
focus of expansion, roll and pitch from the plane.**  Pitch is produced by both,
so the two methods cross-validate each other on a shared parameter.

The decomposition also yields ``t/d``; combined with the true baseline ``|t|``
from the trajectory it gives ``d`` -- the **camera height** -- which the FOE
cannot provide at all.

And because a normal is recovered per frame-pair, the spread is not noise to be
averaged away: its median is the static mount, and its variation is the body
pitching over bumps.

Measured outcome on the 08-11 session (93 usable pairs, 4-frame baseline)
------------------------------------------------------------------------
======  ==========  ======================  ============  ==============
param   median      bootstrap 95 % CI       split-half    verdict
======  ==========  ======================  ============  ==============
pitch   +6.98 deg   [+6.71, +7.24]          0.52 deg      **corroborates the FOE** (+6.62 deg, 0.36 deg apart)
height  1.167 m     [1.100, 1.223]          0.025 m       **adopt** -- replaces a guessed 1.25 m
roll    +0.66 deg   [-0.46, +1.38]          0.49 deg      **reject** -- CI spans zero
======  ==========  ======================  ============  ==============

Two things follow.  First, pitch is now confirmed by two independent methods, so
the FOE value stands and the earlier ``ground_calib`` fit that shifted it was
wrong.  Second, roll is *not* measurable this way, and the reason is geometric
rather than a matter of noise: ``roll ~= n_x``, the plane normal's **lateral**
tilt, and a road corridor 27 m long by 6.4 m wide is depth-dominated, so there
is little lateral leverage.  Assuming roll = 0 is statistically indistinguishable
from this estimate; recovering it properly needs a different observable, such as
the vertical vanishing point from poles and building edges.

Note the split-half difference is comparable to the bootstrap CI width for the
angles, which means the bootstrap is optimistic -- consecutive frame pairs are
correlated.  Height is the exception: 0.025 m across halves is genuinely stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlaneCalibResult:
    roll_deg: float
    pitch_deg: float
    height_m: float
    roll_mad_deg: float          # robust spread across frame pairs
    pitch_mad_deg: float
    height_mad_m: float
    n_pairs: int
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    roll_series_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    pitch_series_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    height_series_m: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        return (f"PlaneCalib from {self.n_pairs} road homographies\n"
                f"  roll   = {self.roll_deg:+.2f} deg   (robust spread +/-{self.roll_mad_deg:.2f})\n"
                f"  pitch  = {self.pitch_deg:+.2f} deg   (robust spread +/-{self.pitch_mad_deg:.2f})\n"
                f"  height = {self.height_m:.3f} m     (robust spread +/-{self.height_mad_m:.3f})")


def _mad(x):
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def normal_to_roll_pitch(n):
    """Road-plane normal in camera coordinates -> (roll, pitch) in radians.

    Inverse of ``n_cam = [sin(r)cos(p), -cos(r)cos(p), sin(p)]``.  The normal is
    flipped so that it points *up* (negative y in the camera frame, where y is
    down); the decomposition is sign-ambiguous and only one sign is physical for
    a road beneath the camera.
    """
    n = np.asarray(n, dtype=float).ravel()
    n = n / (np.linalg.norm(n) + 1e-12)
    if n[1] > 0:
        n = -n
    pitch = float(np.arcsin(np.clip(n[2], -1.0, 1.0)))
    roll = float(np.arctan2(n[0], -n[1]))
    return roll, pitch


def decompose_pair(a, b, K, baseline_m, max_tilt_deg=25.0, ransac_px=1.5):
    """Homography between two sets of road points -> (roll, pitch, height).

    Returns ``None`` when the fit or the disambiguation fails.
    """
    import cv2

    if len(a) < 14:
        return None
    H, mask = cv2.findHomography(a.reshape(-1, 1, 2), b.reshape(-1, 1, 2),
                                 cv2.RANSAC, ransac_px, maxIters=4000, confidence=0.995)
    if H is None or mask is None or int(mask.sum()) < 12:
        return None

    n_sol, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
    if n_sol == 0:
        return None

    best = None
    for i in range(n_sol):
        roll, pitch = normal_to_roll_pitch(Ns[i])
        if abs(np.rad2deg(pitch)) > max_tilt_deg or abs(np.rad2deg(roll)) > max_tilt_deg:
            continue
        tn = float(np.linalg.norm(Ts[i]))          # this is |t| / d
        if tn < 1e-6:
            continue
        d = baseline_m / tn
        if not (0.6 < d < 2.6):                    # a windscreen camera is ~1-2 m up
            continue
        # among survivors prefer the one whose rotation is closest to identity:
        # over a 0.27 s baseline the camera barely rotates
        ang = float(np.rad2deg(np.arccos(np.clip((np.trace(Rs[i]) - 1) / 2, -1, 1))))
        score = ang
        if best is None or score < best[0]:
            best = (score, roll, pitch, d)
    if best is None:
        return None
    return np.rad2deg(best[1]), np.rad2deg(best[2]), best[3]


def estimate_from_tracks(rows, cam, min_baseline_m: float = 0.8):
    """Run the decomposition over pre-collected road-feature track pairs.

    ``rows`` are the tuples produced by
    :func:`trajlib.ground_calib.collect_road_tracks`:
    ``(pts_a, pts_b, dp, dpsi, t0, t1)`` with ``dp`` the vehicle displacement
    between the two frames, which supplies the metric baseline that turns
    ``t/d`` into a height.
    """
    K = cam.K
    rolls, pitches, heights, ts = [], [], [], []
    for a, b, dp, dpsi, t0, t1 in rows:
        base = float(np.hypot(dp[0], dp[1]))
        if base < min_baseline_m:
            continue
        out = decompose_pair(np.asarray(a, dtype=np.float32),
                             np.asarray(b, dtype=np.float32), K, base)
        if out is None:
            continue
        r, p, d = out
        rolls.append(r); pitches.append(p); heights.append(d); ts.append(0.5 * (t0 + t1))

    if len(rolls) < 8:
        return None
    rolls = np.array(rolls); pitches = np.array(pitches)
    heights = np.array(heights); ts = np.array(ts)
    order = np.argsort(ts)
    return PlaneCalibResult(
        roll_deg=float(np.median(rolls)), pitch_deg=float(np.median(pitches)),
        height_m=float(np.median(heights)),
        roll_mad_deg=_mad(rolls), pitch_mad_deg=_mad(pitches), height_mad_m=_mad(heights),
        n_pairs=len(rolls), t=ts[order], roll_series_deg=rolls[order],
        pitch_series_deg=pitches[order], height_series_m=heights[order])


def apply(cam, res: PlaneCalibResult, set_height: bool = True, set_roll: bool = True,
          set_pitch: bool = False):
    """Write the plane-derived extrinsics onto a camera model.

    ``set_pitch`` defaults to False: the focus of expansion already determines
    pitch from 3 x 10^5 flow vectors, and the point of estimating it here is to
    have an *independent* check on that number rather than to overwrite it.
    Enable it only once the two agree.
    """
    import copy
    c = copy.copy(cam)
    if set_roll:
        c.roll = np.deg2rad(res.roll_deg)
    if set_pitch:
        c.pitch = np.deg2rad(res.pitch_deg)
    if set_height:
        c.height_m = res.height_m
    c.source = dict(cam.source)
    c.source["plane_calib"] = (
        f"road homography x{res.n_pairs}: roll {res.roll_deg:+.2f}+/-{res.roll_mad_deg:.2f}, "
        f"pitch {res.pitch_deg:+.2f}+/-{res.pitch_mad_deg:.2f} deg, "
        f"h {res.height_m:.2f}+/-{res.height_mad_m:.2f} m")
    return c
