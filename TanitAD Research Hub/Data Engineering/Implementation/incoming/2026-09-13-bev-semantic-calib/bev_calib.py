#!/usr/bin/env python3
"""bev_calib — camera calibration by BEV agreement of semantic ground features.

THE PROBLEM THIS EXISTS FOR
---------------------------
On the 2026-08-08 recording every automatic calibrator declined or disagreed:

    plane_calib   height 1.691 m   DECLINED (spread +/-0.56 over 14 homographies)
    scale_calib   f*h 1608.1       DECLINED (spread 62% over 1576 tracks)
    FOE pitch -2.93 deg   vs   lane-VP horizon row 523.4 px   ->   2.29 deg apart

and the ground plane cannot separate focal length from height in principle
(``scale_calib.py``: lateral back-projection sees ``h`` alone, longitudinal sees
only the product ``f*h``). The shipped overlay was wrong in yaw, pitch, height
and lateral offset simultaneously.

THE IDEA (Sayed, 2026-09-13)
----------------------------
Segment ground semantics per pixel -- asphalt, lane lines, painted symbols --
then treat calibration as an OPTIMISATION: with the vehicle's motion known, the
correct calibration is the one under which the same static ground features, seen
from many frames, land on TOP OF EACH OTHER in a common bird's-eye view. Wrong
calibration smears them.

This module is that objective and its optimiser. The segmentation front end is
deliberately NOT hard-wired: it consumes per-frame arrays of "ink" pixel
coordinates, so a SAM3/SAM2 mask, a classical top-hat threshold, or hand labels
all drop in unchanged.

WHY IT IS WELL POSED -- and where it is NOT
-------------------------------------------
**Metric ego-motion is what makes scale observable, and it kills the obvious
degeneracy.** Scale the height by ``k`` and every back-projected point scales by
``k`` too -- but the pose transform between frames translates by the *unscaled*
metric motion from GNSS/IMU. Two observations of one static point land at
``k*x1`` and ``k*x2 + s``; they coincide only when ``k*(x1-x2) = s``, i.e. only at
``k = 1``. So "collapse the height to zero and everything overlaps" is NOT a free
optimum: collapsing drives each frame's points to its own origin, and the metric
translation then pulls those clusters APART.

⚠️ **But f and h stay degenerate on a perfectly straight path.** Lateral position
scales with ``h`` and longitudinal with ``f*h``; on straight driving a static
point's lateral coordinate is constant in every frame, so scaling all lateral
values by ``k`` leaves it just as self-consistent. Separation needs the path to
TURN, which mixes the two axes -- or an external lateral metric. Two things
supply that, and they are conveniently orthogonal:

    painted line WIDTH  (lateral, ~0.15 m standard)  ->  pins h alone, no f
    dash PITCH          (longitudinal, standardised) ->  pins f*h  ->  with h, f

``test_bev_calib.py`` measures this rather than assuming it: it recovers known
parameters from synthetic data on a straight path and on a curved one, and
reports which parameters come back in each case.

THE OBJECTIVE (corrected 2026-09-13 after the synthetic test refuted v1)
-------------
Accumulate each frame's ink into the anchor grid separately, blur, and maximise
the CROSS-FRAME overlap with the self-term removed::

    agreement = sum_cells [ (sum_i m_i)^2 - sum_i m_i^2 ]

⚠️ The self-term removal is not a detail. The first version of this module scored
plain concentration and was REFUTED by its own synthetic test: collapsing the
height to 0.05 m scored 15x BETTER than the truth, because total concentration is
maximised by shrinking every frame's footprint whether or not the frames agree.
Excluding the diagonal means a collapsed solution -- each frame's blob in its own
place -- scores zero. See :func:`agreement`.

CONVENTION -- identical to ``trajlib.camera``, verified
-------------------------------------------------------
    R_cv = Rz(roll) @ Rx(pitch) @ Ry(yaw) @ R_CV_NOMINAL
    t_v  = [longitudinal, lateral, height]
    ray  = [(u-cx)/fx, (v-cy)/fy, 1]  ;  d = ray @ R_cv  ;  s = -height / d_z
    ground point in the VEHICLE frame = t_v[:2] + s * d[:2]

Getting this wrong would calibrate against a different camera model than the one
that renders, so ``test_bev_calib.py`` pins it against ``trajlib.camera`` directly.
"""
from __future__ import annotations

import numpy as np

R_CV_NOMINAL = np.array([[0.0, -1.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [1.0, 0.0, 0.0]])

#: optimisation order; `calibrate` selects a subset of these by name.
PARAMS = ("yaw", "pitch", "roll", "height", "lateral", "longitudinal",
          "fx", "cx", "cy")

DEFAULTS = dict(yaw=0.0, pitch=0.0, roll=0.0, height=1.20, lateral=0.0,
                longitudinal=2.10, fx=1478.3, cx=960.0, cy=540.0)


def rot_ypr(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Mount rotation in the camera frame (radians). Mirrors ``camera._rot``."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def ground_from_pixels(uv: np.ndarray, P: dict) -> np.ndarray:
    """Pixels -> ground points (z=0) in the VEHICLE frame, metres.

    Returns an ``(n, 2)`` array of ``[forward, left]``. Rays that do not strike
    the ground ahead of the camera are dropped, so the output may be shorter
    than the input -- that is intended: a calibration that tips the ground plane
    behind the camera should lose those points and score worse for it.
    """
    R = rot_ypr(P["yaw"], P["pitch"], P["roll"]) @ R_CV_NOMINAL
    ray = np.stack([(uv[:, 0] - P["cx"]) / P["fx"],
                    (uv[:, 1] - P["cy"]) / P["fx"],      # fy == fx
                    np.ones(len(uv))], axis=1)
    d = ray @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -P["height"] / d[:, 2]
    ok = np.isfinite(s) & (s > 0)
    g = np.empty((int(ok.sum()), 2))
    g[:, 0] = d[ok, 0] * s[ok] + P["longitudinal"]
    g[:, 1] = d[ok, 1] * s[ok] + P["lateral"]
    return g


def project_ground(xy: np.ndarray, P: dict) -> np.ndarray:
    """Vehicle-frame ground points -> pixels. The inverse of the above.

    Used by the synthetic test to render observations, so the test exercises the
    same convention the objective consumes.
    """
    R = rot_ypr(P["yaw"], P["pitch"], P["roll"]) @ R_CV_NOMINAL
    p = np.stack([xy[:, 0] - P["longitudinal"],
                  xy[:, 1] - P["lateral"],
                  np.full(len(xy), -P["height"])], axis=1)
    pc = p @ R.T
    z = pc[:, 2]
    ok = z > 0.35
    uv = np.full((len(xy), 2), np.nan)
    uv[ok, 0] = P["fx"] * pc[ok, 0] / z[ok] + P["cx"]
    uv[ok, 1] = P["fx"] * pc[ok, 1] / z[ok] + P["cy"]
    return uv


def to_anchor(xy: np.ndarray, pose: tuple) -> np.ndarray:
    """Vehicle-frame points at a later time -> the anchor frame.

    ``pose = (tx, ty, psi)`` is that frame's vehicle origin expressed in the
    anchor's frame, which is exactly what ``trajectory.jsonl``'s per-frame window
    provides. THE TRANSLATION IS METRIC AND DOES NOT SCALE WITH THE CALIBRATION
    -- that is the whole reason the scale is observable.
    """
    tx, ty, psi = pose
    c, s = np.cos(psi), np.sin(psi)
    out = np.empty_like(xy)
    out[:, 0] = c * xy[:, 0] - s * xy[:, 1] + tx
    out[:, 1] = s * xy[:, 0] + c * xy[:, 1] + ty
    return out


class BevGrid:
    """Anchor-frame occupancy grid over a fixed physical window."""

    def __init__(self, x_range=(4.0, 45.0), y_range=(-8.0, 8.0), cell=0.08):
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.cell = cell
        self.nx = int(round((self.x1 - self.x0) / cell))
        self.ny = int(round((self.y1 - self.y0) / cell))

    def accumulate_one(self, p: np.ndarray) -> np.ndarray:
        H = np.zeros((self.nx, self.ny), dtype=np.float64)
        if len(p):
            ix = ((p[:, 0] - self.x0) / self.cell).astype(np.int64)
            iy = ((p[:, 1] - self.y0) / self.cell).astype(np.int64)
            m = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            np.add.at(H, (ix[m], iy[m]), 1.0)
        return H

    def accumulate(self, pts_list) -> np.ndarray:
        H = np.zeros((self.nx, self.ny), dtype=np.float64)
        for p in pts_list:
            H += self.accumulate_one(p)
        return H


def _blur(H: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Separable Gaussian. Smooths the objective so a gradient-free optimiser is
    not fighting the grid's own quantisation steps."""
    if sigma_cells <= 0:
        return H
    r = max(1, int(round(3 * sigma_cells)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, H)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)


def agreement(obs, poses, P: dict, grid: BevGrid, sigma_cells: float = 1.0) -> float:
    """Mean pairwise NORMALISED cross-correlation of the per-frame BEV maps,
    scaled by the fraction of ink that actually landed on the grid.

    Higher is better. Two things in that sentence are load-bearing, and both were
    learned by being wrong:

    **1. Cross-frame only — the self-term is excluded.** v1 maximised plain
    concentration ``sum_cells (sum_i m_i)^2``. MEASURED 2026-09-13: a collapsed
    height of 0.05 m scored 3.61e-04 against the truth's 2.31e-05, **15x better**,
    because shrinking the height shrinks every frame's own footprint and total
    concentration does not care whether frames agree.

    **2. NORMALISED, not raw, correlation.** Removing the self-term alone was not
    enough -- collapse still won 7.69e-05 to 2.02e-05, **3.8x**. Diagnosed rather
    than guessed: at h=0.05 the whole 4-140 m scene compresses into a **5.54 m**
    span (16.20 m at the truth) and 96.7% of the ink lands on the grid instead of
    14.3%, so consecutive frames' dense blobs overlap heavily at 3 m spacing. The
    mass is identical; it is merely denser. A raw inner product rewards exactly
    that. NCC divides it out::

        score = mean_{i<j} <H_i, H_j> / (||H_i|| ||H_j||)   x   on_grid_fraction

    With unit-normalised maps the pairwise sum has a closed form,
    ``(||sum_i H_i||^2 - K) / (K(K-1))``, so this costs one pass, not K^2.

    The ``on_grid_fraction`` factor is what stops the opposite cheat: flinging the
    ink off the grid until only a self-consistent handful remains.
    """
    n_total = sum(len(o) for o in obs)
    if n_total == 0:
        return 0.0
    maps, on_grid = [], 0
    for o, ps in zip(obs, poses):
        g = to_anchor(ground_from_pixels(o, P), ps)
        Hi = grid.accumulate_one(g)
        on_grid += int(Hi.sum())
        Hi = _blur(Hi, sigma_cells)
        n = np.sqrt((Hi * Hi).sum())
        if n > 0:
            maps.append(Hi / n)
    K = len(maps)
    if K < 2:
        return 0.0
    total = np.zeros_like(maps[0])
    for Hi in maps:
        total += Hi
    pairwise = (float((total * total).sum()) - K) / (K * (K - 1))
    return pairwise * (on_grid / float(n_total))


#: kept as the historical name so existing callers do not silently get the
#: broken metric back; it now delegates to the corrected one.
sharpness = agreement


def calibrate(obs, poses, P0: dict, free, bounds=None, grid: BevGrid = None,
              sigma_cells: float = 1.0, maxiter: int = 400, seed: int = 0):
    """Maximise BEV agreement over the named free parameters.

    ``free`` is a tuple of names from :data:`PARAMS`. Everything else is held at
    its value in ``P0``. Returns ``(P_best, info)``.

    Powell is used deliberately: the objective is a histogram score and is only
    piecewise smooth, so a gradient-based method chases quantisation noise.
    """
    from scipy.optimize import minimize

    grid = grid or BevGrid()
    bounds = bounds or {}
    x0 = np.array([P0[k] for k in free], dtype=float)
    scale = np.array([_SCALE.get(k, 1.0) for k in free])

    hist = []

    def unpack(x):
        P = dict(P0)
        for k, v in zip(free, x * scale):
            P[k] = float(v)
        return P

    def neg(x):
        P = unpack(x)
        for k, (lo, hi) in bounds.items():
            if k in free and not (lo <= P[k] <= hi):
                return 1e6
        s = sharpness(obs, poses, P, grid, sigma_cells)
        hist.append(s)
        return -s

    res = minimize(neg, x0 / scale, method="Powell",
                   options=dict(maxiter=maxiter, xtol=1e-4, ftol=1e-6))
    P = unpack(res.x)
    return P, dict(score=float(-res.fun), score0=float(sharpness(obs, poses, P0, grid, sigma_cells)),
                   nfev=int(res.nfev), n_eval=len(hist), success=bool(res.success))


#: per-parameter step scaling, so Powell takes comparable steps in radians,
#: metres and pixels rather than crawling in one and thrashing in another.
_SCALE = dict(yaw=0.01, pitch=0.01, roll=0.01, height=0.05,
              lateral=0.05, longitudinal=0.1, fx=50.0, cx=20.0, cy=20.0)
