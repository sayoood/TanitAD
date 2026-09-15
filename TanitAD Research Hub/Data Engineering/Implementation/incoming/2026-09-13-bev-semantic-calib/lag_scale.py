#!/usr/bin/env python3
"""Measure the LONGITUDINAL SCALE directly, by racing the BEV against the odometer.

WHY THE AGREEMENT OBJECTIVE CANNOT DO THIS (measured, not assumed)
------------------------------------------------------------------
``run_real.py``'s BEV agreement determines the extrinsics well (yaw -7.01 +
pitch -0.64 beat the shipped nominal by +41%) but its ``f`` scan is MONOTONE to
the boundary::

    f  900  1100  1250  1356  1478  1550  1628  1800  2100  2500   (h pinned 1.03)
    s  .016 .026  .036  .044  .054  .059  .064  .076  .092  .100

and the joint fit ran to ``fx 2588`` of a 2600 bound. I tested two explanations
and REFUTED both by measurement: it is not the ``on_grid_fraction`` factor (that
stayed 0.932-0.966 while the NCC itself rose 2.15e-2 -> 1.00e-1), and it is not
non-ground ink (restricting to the road columns made the contrast WORSE, +41% ->
-16%). The mechanism is structural: **raising f stretches the reconstruction
longitudinally, so the FIXED 2.2 m of inter-frame travel becomes relatively
smaller and the maps agree more.** Any objective that scores how well stacked
maps overlap rewards that stretch. A band-pass at a fixed physical scale reduced
it (peak 2500 -> 2100 px) but did not remove it.

⇒ **Stop scoring overlap. Measure the stretch itself.**

THE ESTIMATOR
-------------
Reconstruct each frame's ground ink in ITS OWN vehicle frame -- no pose applied.
A static ground feature at true longitudinal ``X`` in frame 0 is at ``X - D`` in
a frame ``D`` metres later. Under an assumed calibration both are scaled by the
same factor ``s = (f_a h_a) / (f_t h_t)``, so the two reconstructions differ by a
pure translation of ``D * s``::

    s = (measured BEV lag) / (odometer displacement D)
    f*h  =  f_a * h_a / s

The ego displacement is METRIC and does not scale with the calibration -- that is
the whole ruler. And crucially the estimator reads a **lag**, not an overlap
level, so stretching the map cannot flatter it: stretch by 2x and the lag doubles
too, which is exactly the thing being reported.

The same construction measures the horizon row for free. If the assumed horizon
``v_h`` is wrong by ``delta`` rows the reconstruction is not a pure scale but a
Mobius map, and the lag then VARIES with range::

    x_hat = A*X / (T - delta*X)      A = f_a h_a,  T = f_t h_t

so ``lag(x_hat)`` is constant only when ``delta = 0``. Binning the lag by range
turns a pitch error into a visible slope. :func:`fit_scale_and_horizon` fits both.

WHAT PROVIDES THE SIGNAL -- and the honest limit
------------------------------------------------
⚠️ **A continuous lane line carries NO longitudinal information.** Slide it along
itself and it is unchanged, so on a straight road with solid lines this estimator
is blind by construction. The signal comes from DASH ENDS and transverse ink
(seams, patches, shadow edges). :func:`pair_lag` therefore reports a peak
CONTRAST alongside every lag, and pairs whose correlation has no distinct peak
are dropped rather than averaged in.

⚠️ **The coverage envelope correlates at lag 0 and would swamp the ground signal.**
Both frames reconstruct the same image band (rows 0.55H-0.88H) into the same
range of ``x_hat``, so the two maps share an identical trapezoid of support. That
is a nuisance signal with a huge amplitude sitting exactly where a null result
would be -- it would bias every lag toward 0 and hence every ``s`` toward 0 and
every focal toward infinity, which is the same failure as the agreement scan.
:func:`highpass` removes it before correlating, and ``--self-test`` verifies that
a known injected scale still comes back.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bev_calib as BC                                              # noqa: E402


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------
def horizon_row(P: dict) -> float:
    """Image row of the horizon at ``u = cx``, for this calibration.

    ⚠️ **It depends on ``fx``.** The ground ray is ``[(u-cx)/f, (v-cy)/f, 1]``, so
    the vanishing condition at ``u = cx`` is ``(v-cy) R12 / f + R22 = 0``, i.e.
    ``v_h = cy - f R22/R12``. Changing the focal length while holding the pitch
    fixed therefore MOVES THE HORIZON, and the reconstruction is then not a
    stretched copy of the truth but a Mobius map of it.

    This is not a footnote; it invalidated this module's first self-test. Injecting
    "f x 1.80" at fixed pitch moved the horizon by ``(f_t - f_a) * pitch`` = 13 px,
    which at 30 m range is a 24% distortion of the reconstructed range -- so the
    test's expectation of ``s = 1.80`` was simply WRONG and the estimator was
    convicted of a bias it may not have. Same class as every other trap in this
    programme: a probe that answers a different question than the one asked.

    ⇒ The honest free parameters are ``(v_h, f*h, h)``, not ``(pitch, f, h)``.
    ``v_h`` is measurable on its own from the lane vanishing point (523.4 px,
    [513.3, 534.8] on this recording); ``f*h`` is what the lag test measures; ``h``
    comes from the painted line width. :func:`pitch_for_horizon` is the change of
    variables that keeps them separate.
    """
    R = BC.rot_ypr(P["yaw"], P["pitch"], P["roll"]) @ BC.R_CV_NOMINAL
    if abs(R[1, 2]) < 1e-12:
        return np.inf
    return float(P["cy"] - P["fx"] * R[2, 2] / R[1, 2])


def pitch_for_horizon(P: dict, v_h: float) -> float:
    """Pitch that puts the horizon at row ``v_h`` for this ``fx``. Radians."""
    from scipy.optimize import brentq
    f = lambda p: horizon_row({**P, "pitch": p}) - v_h
    lo, hi = np.deg2rad(-20.0), np.deg2rad(20.0)
    if f(lo) * f(hi) > 0:
        raise ValueError(f"no pitch in +/-20 deg puts the horizon at {v_h}")
    return float(brentq(f, lo, hi, xtol=1e-10))


def with_horizon(P: dict, fx: float, height: float, v_h: float) -> dict:
    """A calibration at the given focal and height whose horizon stays at ``v_h``."""
    Q = dict(P)
    Q["fx"] = float(fx)
    Q["height"] = float(height)
    Q["pitch"] = pitch_for_horizon(Q, v_h)
    return Q


def highpass(H: np.ndarray, cell: float, scale_m: float = 1.5) -> np.ndarray:
    """Remove the smooth coverage envelope, keep marking-scale structure.

    Subtracting a Gaussian-blurred copy at ``scale_m`` keeps everything finer
    than that (dash ends ~0.5 m, line widths ~0.15 m) and removes everything
    coarser (the ~32 m support trapezoid, the 1/range ink-density ramp). The
    scale is in METRES, not cells, so it does not silently change meaning when
    the grid resolution changes.
    """
    sig = max(scale_m / cell, 0.5)
    return H - BC._blur(H, sig)


def _rect_sumsq(S: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> float:
    """Sum of squares over ``[r0,r1) x [c0,c1)`` from an integral image ``S``."""
    if r1 <= r0 or c1 <= c0:
        return 0.0
    return float(S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0])


def _ncc_profile(A: np.ndarray, B: np.ndarray, i_lo: int, i_hi: int, j_max: int,
                 rows: tuple | None = None):
    """NCC over longitudinal lags ``i_lo..i_hi``, maximised over lateral lag.

    One FFT for every numerator and an integral image for every denominator, so
    the cost is O(N log N) instead of O(N x lags).

    ``rows=(r0, r1)`` restricts the TEMPLATE to a slab of ``A``, which is how the
    lag is measured separately at several ranges. That is not a refinement: a
    lag that changes with range is the signature of a wrong horizon row, and
    without it a pitch error is indistinguishable from a focal error -- the two
    call for opposite repairs.
    """
    nx, ny = A.shape
    r0, r1 = rows if rows else (0, nx)
    Am = A
    if rows:
        Am = np.zeros_like(A)
        Am[r0:r1] = A[r0:r1]
    fx_, fy_ = 2 * nx, 2 * ny                      # pad: linear, not circular
    C = np.fft.irfft2(np.fft.rfft2(Am, s=(fx_, fy_))
                      * np.conj(np.fft.rfft2(B, s=(fx_, fy_))), s=(fx_, fy_))
    SA = np.zeros((nx + 1, ny + 1)); SA[1:, 1:] = np.cumsum(np.cumsum(A * A, 0), 1)
    SB = np.zeros((nx + 1, ny + 1)); SB[1:, 1:] = np.cumsum(np.cumsum(B * B, 0), 1)

    prof = np.full(i_hi - i_lo + 1, -np.inf)
    for i in range(i_lo, i_hi + 1):
        # template rows [r0,r1) of A line up with rows [r0-i, r1-i) of B
        ar0, ar1 = max(r0, i), min(r1, nx)
        br0, br1 = max(0, r0 - i), min(nx - i, r1 - i)
        if ar1 <= ar0 or br1 <= br0:
            continue
        best = -np.inf
        for j in range(-j_max, j_max + 1):
            num = C[i, j]                           # negative j wraps to the tail
            na2 = _rect_sumsq(SA, ar0, ar1, max(j, 0), min(ny, ny + j))
            nb2 = _rect_sumsq(SB, br0, br1, max(0, -j), min(ny, ny - j))
            if na2 <= 0 or nb2 <= 0:
                continue
            best = max(best, num / np.sqrt(na2 * nb2))
        prof[i - i_lo] = best
    return prof


def pair_lag(A: np.ndarray, B: np.ndarray, cell: float,
             lag_lo_m: float, lag_hi_m: float, dy_max_m: float = 0.8,
             rows: tuple | None = None):
    """Longitudinal lag from A to B, by normalised cross-correlation.

    ``A`` and ``B`` are ``(nx, ny)`` high-passed BEV maps in each frame's OWN
    vehicle frame. Returns ``(lag_m, contrast, r)`` or ``None``.

    ⚠️ **The correlation is normalised BY THE OVERLAP, not by the whole map.**
    Shifting by ``i`` cells leaves only ``nx-i`` rows in the inner product, so a
    denominator computed once over the whole map makes ``r(i)`` decay mechanically
    with lag and drags the argmax toward zero.

    ⚠️ **That fix alone did NOT remove the shrink-toward-unity bias the first
    self-test showed, and my diagnosis of it was WRONG.** Overlap normalisation
    changed ``f x 1.40`` from ``s = 1.233`` to ``1.234`` -- no effect. The real
    cause was in the TEST: injecting a focal error at fixed pitch also moves the
    horizon row (see :func:`horizon_row`), so the reconstruction was a Mobius map
    of the truth and ``s = factor`` was never the right expectation. Both changes
    are kept -- the overlap normalisation is correct on its own terms -- but the
    record of which one mattered is kept too.

    The search is bracketed to ``[lag_lo_m, lag_hi_m]``. That bracket is a real
    assumption and it is stated rather than hidden: dashed markings are periodic,
    so the correlation has near-equal peaks every dash period and an unbracketed
    argmax would pick an arbitrary one. The bracket is set from the odometer,
    wide enough that landing in its interior is informative and landing at an edge
    is reported as a failure by the caller.

    A small lateral search absorbs the residual yaw/lateral motion the caller
    could not derotate exactly; the returned lag is the longitudinal component
    at the best lateral offset.
    """
    nx, ny = A.shape
    i_lo = max(1, int(np.floor(lag_lo_m / cell)))
    i_hi = int(np.ceil(lag_hi_m / cell))
    j_max = int(round(dy_max_m / cell))
    # keep at least half the map in every comparison: a tiny overlap makes the
    # normalised score high-variance and manufactures spurious peaks at the far
    # end of the bracket -- the mirror image of the bug above.
    i_hi = min(i_hi, nx // 2)
    if i_hi <= i_lo + 2:
        return None

    # ⚠️ The profile starts at lag ZERO even though the bracket does not, purely
    # so the STATIC REFERENCE is available. Content that is fixed in the IMAGE --
    # guardrail, rock face, bonnet, any rig artefact -- back-projects to the same
    # range in both frames and correlates at lag 0 with no competition from the
    # ground. MEASURED on the real recording: at D = 2.99 m the profile decays
    # monotonically from r = 0.211 at lag 0, and at the true displacement it is
    # r = 0.062 -- a shoulder, not a peak. Without this guard the argmax simply
    # slid to the bracket's lower edge and the module reported f = 3645 px
    # (HFOV 29.5 deg) with a tight bootstrap, which is the most dangerous kind of
    # wrong answer: confidently precise.
    prof_full = _ncc_profile(A, B, 0, i_hi, j_max, rows=rows)
    if not np.isfinite(prof_full).any():
        return None
    static_r = float(np.nanmax(prof_full[:max(1, i_lo // 3)]))
    prof = prof_full[i_lo:]
    i_best = int(np.argmax(prof)) + i_lo
    r_best = float(prof[i_best - i_lo])
    if not np.isfinite(r_best) or r_best <= 0:
        return None
    if r_best <= static_r:
        return None                 # the ground never beat the static content
    # contrast against the rest of the bracket: a real ground lock is a PEAK,
    # not merely the largest value of a flat noisy profile.
    k = i_best - i_lo
    mask = np.ones(len(prof), bool)
    mask[max(0, k - 2):k + 3] = False
    floor = np.median(prof[mask]) if mask.sum() >= 3 else 0.0
    spread = np.std(prof[mask]) if mask.sum() >= 3 else 0.0
    contrast = (r_best - floor) / spread if spread > 0 else np.inf

    # sub-cell refinement on the longitudinal profile
    lag = i_best * cell
    if 0 < k < len(prof) - 1:
        y0, y1, y2 = prof[k - 1], prof[k], prof[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            lag = (i_best + 0.5 * (y0 - y2) / den) * cell
    # a peak within two cells of either bracket end is the bracket's answer, not
    # the data's: widened from "exactly at the end", which let the whole real-data
    # run through at s = 0.35-0.47 sitting one cell inside the lower edge.
    edge = (i_best <= i_lo + 2) or (i_best >= i_hi - 2)
    return dict(lag_m=float(lag), contrast=float(contrast), r=float(r_best),
                static_r=float(static_r), at_edge=bool(edge))


def own_frame_bev(uv: np.ndarray, P: dict, grid: BC.BevGrid, psi: float = 0.0):
    """One frame's ink in its own vehicle frame, optionally derotated by ``psi``.

    Derotation uses the ODOMETER's relative heading only. It carries no scale, so
    it cannot leak the quantity being measured into the answer.
    """
    g = BC.ground_from_pixels(uv, P)
    if psi:
        c, s = np.cos(psi), np.sin(psi)
        g = np.stack([c * g[:, 0] - s * g[:, 1], s * g[:, 0] + c * g[:, 1]], axis=1)
    return grid.accumulate_one(g)


def measure_scale(anchors, P: dict, grid: BC.BevGrid, min_D=1.5, max_D=7.0,
                  min_contrast=3.0, hp_m=1.5, brk=(0.35, 2.2), n_bands=3,
                  verbose=False):
    """Scale ``s`` from every usable frame pair across every anchor.

    ``max_D`` is capped at 7 m for a reason that is measured, not aesthetic: the
    lag bracket spans ``(brk[1]-brk[0]) * D`` metres, and once that exceeds one
    dash period (13 m for French T1) the correlation contains a second, equally
    good alias and the argmax becomes arbitrary. 7 m keeps the window at 13 m.

    Returns ``(records, summary)``. ``records`` holds one dict per accepted pair
    so the fit, the outlier behaviour and the range dependence can all be
    inspected rather than taken on trust.
    """
    cell = grid.cell
    recs = []
    for ai, (obs, poses, fno) in enumerate(anchors):
        maps = [None] * len(obs)
        for i in range(len(obs)):
            for k in range(i + 1, len(obs)):
                dx = poses[k][0] - poses[i][0]
                dy = poses[k][1] - poses[i][1]
                dpsi = poses[k][2] - poses[i][2]
                D = float(np.hypot(dx, dy))
                if not (min_D <= D <= max_D):
                    continue
                if maps[i] is None:
                    maps[i] = highpass(own_frame_bev(obs[i], P, grid), cell, hp_m)
                Bm = highpass(own_frame_bev(obs[k], P, grid, psi=dpsi), cell, hp_m)
                nx = maps[i].shape[0]
                # one lag per RANGE SLAB, plus the whole-map lag. Slabs start
                # past the largest lag so every slab has a valid counterpart in B.
                slabs = [(None, None)]
                if n_bands > 1:
                    b0 = int(np.ceil(brk[1] * D / cell))
                    edges = np.linspace(b0, nx, n_bands + 1).astype(int)
                    slabs += [(int(edges[b]), int(edges[b + 1])) for b in range(n_bands)
                              if edges[b + 1] - edges[b] > 20]
                # ⚠️ a slab's EFFECTIVE range is not its geometric centre. Ink
                # density falls roughly as 1/range^2, so the correlation peak is
                # set by the near part of the slab. MEASURED in synthetic: using
                # the centre made the fitted horizon error come out ~30% short
                # (s at x~30 m read 1.311 against the Mobius prediction 1.408,
                # with the same shortfall at every slab). The ink-weighted mean
                # removes that, and costs one cumulative sum.
                dens = np.abs(maps[i]).sum(axis=1)
                xs = grid.x0 + (np.arange(nx) + 0.5) * cell
                for r0, r1 in slabs:
                    rows = None if r0 is None else (r0, r1)
                    if rows is not None:
                        w = dens[r0:r1]
                        if w.sum() <= 0 or w.sum() < 0.02 * dens.sum():
                            continue            # too little ink to locate anything
                        rng = float((w * xs[r0:r1]).sum() / w.sum())
                    else:
                        rng = np.nan
                    res = pair_lag(maps[i], Bm, cell, brk[0] * D, brk[1] * D, rows=rows)
                    if res is None or res["at_edge"] or res["contrast"] < min_contrast:
                        continue
                    res.update(anchor=ai, frame=fno, i=i, k=k, D_m=D,
                               s=res["lag_m"] / D, band=(-1 if rows is None else
                                                         slabs.index((r0, r1)) - 1),
                               range_m=float(rng))
                    recs.append(res)
                    if verbose:
                        tag = "all " if rows is None else f"x~{rng:4.1f}"
                        print(f"      a{ai} {i}->{k} {tag}  D {D:5.2f} m  "
                              f"lag {res['lag_m']:5.2f} m  s {res['s']:.3f}  "
                              f"contrast {res['contrast']:.1f}")
    whole = [r for r in recs if r["band"] < 0]      # headline uses the full map
    if not whole:
        return recs, None
    s = np.array([r["s"] for r in whole])
    w = np.array([r["D_m"] for r in whole])         # longer baselines are sharper
    summary = dict(n=len(whole), n_band=len(recs) - len(whole),
                   s_median=float(np.median(s)),
                   s_mean_w=float((s * w).sum() / w.sum()),
                   s_iqr=float(np.percentile(s, 75) - np.percentile(s, 25)),
                   s_p16=float(np.percentile(s, 16)),
                   s_p84=float(np.percentile(s, 84)))
    return recs, summary


def fit_mobius(recs, A_assumed: float, n_boot: int = 2000, seed: int = 0):
    """Solve for ``f*h`` AND the horizon error together, from the range-resolved lags.

    THE MODEL, derived rather than fitted by eye. With ``A = f_a h_a`` assumed,
    ``T = f_t h_t`` true and ``delta = v_h,true - v_h,assumed`` in pixels, a ground
    point at true range ``X`` reconstructs at ``x = A X / (delta X + T)``. Its
    displacement over a known odometer step ``D`` is then::

        lag(x) / D  =  (A - delta * x)^2 / (T * A)                     [exact to O(lag/x)]

    so ``sqrt(s)`` is LINEAR IN RANGE::

        sqrt(s)  =  sqrt(A/T)  -  (delta / sqrt(T A)) * x
                    ^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^
                    intercept      slope

    and one weighted least-squares gives both::

        T      = A / intercept^2
        delta  = -slope * A / intercept

    ⭐ **This is what separates the focal length from the pitch.** Read at a single
    range they are one number -- which is exactly why every earlier estimator in
    this programme returned a manifold instead of a value. Read at several ranges
    the horizon error CURVES the scale and the focal error only SHIFTS it, and the
    two come apart. MEASURED in synthetic: a +-10 px horizon error moves the
    whole-map ``s`` by -17%/+17%, i.e. an unmodelled horizon uncertainty of the
    size of our own lane-VP interval would have swamped the focal answer.

    Returns the fit with an anchor-cluster bootstrap interval, or ``None``.
    """
    band = [r for r in recs
            if r["band"] >= 0 and np.isfinite(r["range_m"]) and r["s"] > 0]
    if len(band) < 8 or len({round(r["range_m"]) for r in band}) < 3:
        return None

    def solve(rows):
        """Huber-robust weighted line fit.

        ⚠️ Plain WLS is NOT enough here, and the synthetic test is what showed it:
        with the horizon assumed 10 px LOW the far slabs compress, lose ink and
        lose contrast, and a handful of them come back at ``s`` 0.38-0.50 against
        a true ~0.73. Least squares chases those and the recovered horizon error
        came out ``-3.3 px`` against a true ``+10`` -- wrong SIGN, which would have
        sent the pitch the wrong way. Huber weights bound each residual's pull.
        """
        x = np.array([r["range_m"] for r in rows])
        y = np.sqrt(np.array([r["s"] for r in rows]))
        w0 = np.clip(np.array([r["contrast"] for r in rows]), 0.0, 20.0)
        X = np.stack([np.ones_like(x), x], axis=1)
        w = w0.copy()
        beta = None
        for _ in range(12):
            sw = np.sqrt(w)[:, None]
            beta = np.linalg.lstsq(X * sw, y * sw[:, 0], rcond=None)[0]
            res = y - X @ beta
            mad = np.median(np.abs(res - np.median(res)))
            scale = max(1.4826 * mad, 1e-3)
            w = w0 * np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(res), 1e-9))
        a0, a1 = float(beta[0]), float(beta[1])
        if a0 <= 1e-6:
            return None
        return a0, a1, A_assumed / a0 ** 2, -a1 * A_assumed / a0

    base = solve(band)
    if base is None:
        return None
    a0, a1, T, delta = base

    by_anchor = {}
    for r in band:
        by_anchor.setdefault(r["anchor"], []).append(r)
    keys = list(by_anchor)
    boot = []
    if len(keys) >= 3:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            rows = [r for p in rng.choice(len(keys), len(keys), replace=True)
                    for r in by_anchor[keys[p]]]
            got = solve(rows)
            if got:
                boot.append((got[2], got[3]))
    out = dict(intercept=a0, slope=a1, fh=float(T), delta_px=float(delta),
               n=len(band), n_clusters=len(keys),
               range_span=[float(min(r["range_m"] for r in band)),
                           float(max(r["range_m"] for r in band))])
    if len(boot) >= 100:
        bt = np.array(boot)
        out["fh_ci"] = [float(np.percentile(bt[:, 0], 2.5)),
                        float(np.percentile(bt[:, 0], 97.5))]
        out["delta_ci"] = [float(np.percentile(bt[:, 1], 2.5)),
                           float(np.percentile(bt[:, 1], 97.5))]
    return out


def fit_scale_and_horizon(recs, P, grid):
    """Is the lag CONSTANT in range (pure scale) or does it slope (horizon error)?

    Reported, not corrected: a slope here says the pitch is still wrong, which is
    a different repair from changing ``f``. The test is a weighted least-squares
    of ``s`` on the pair's mean range, with the slope's own standard error, so
    "no slope" is a measurement and not an eyeball.
    """
    band = [r for r in recs if r["band"] >= 0 and np.isfinite(r["range_m"])]
    if len(band) < 6 or len({round(r["range_m"], 1) for r in band}) < 2:
        return None
    x = np.array([r["range_m"] for r in band])      # MEASURED slab centre, not a proxy
    y = np.array([r["s"] for r in band])
    X = np.stack([np.ones_like(x), x - x.mean()], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(x) - 2, 1)
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    return dict(s0=float(beta[0]), slope=float(beta[1]),
                slope_se=float(np.sqrt(cov[1, 1])),
                slope_t=float(beta[1] / np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else 0.0,
                x_span=[float(x.min()), float(x.max())])


def bootstrap_s(recs, n_boot=2000, seed=0):
    """Episode-style cluster bootstrap over ANCHORS, not pairs.

    Pairs inside one anchor share frames, ink and a stretch of road, so they are
    not independent; resampling pairs would report an interval several times too
    narrow. This is the same reason the programme's decision-grade interval
    clusters on episodes.
    """
    by_anchor = {}
    for r in recs:
        if r["band"] < 0:
            by_anchor.setdefault(r["anchor"], []).append(r["s"])
    keys = list(by_anchor)
    if len(keys) < 3:
        return None
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        vals = np.concatenate([by_anchor[keys[p]] for p in pick])
        out.append(np.median(vals))
    out = np.sort(out)
    return dict(lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)),
                n_clusters=len(keys))


# --------------------------------------------------------------------------
# synthetic self-test — the estimator must recover a KNOWN injected scale
# --------------------------------------------------------------------------
def synth_anchors(P_true, n_anchors=4, n_frames=8, dt=0.15, speed=22.0,
                  dash=3.0, gap=10.0, seed=0):
    """A straight dashed road, rendered through ``P_true``.

    The dashes are what make the longitudinal lag observable at all, so the test
    exercises exactly the property the real data may or may not have.
    """
    rng = np.random.default_rng(seed)
    anchors = []
    for a in range(n_anchors):
        phase = rng.uniform(0, dash + gap)
        lanes = [-1.75 + rng.uniform(-0.2, 0.2), 1.75 + rng.uniform(-0.2, 0.2)]
        # world ink: dashed lines + sparse asphalt speckle
        xs = np.arange(0.0, 400.0, 0.05)
        keep = ((xs + phase) % (dash + gap)) < dash
        pts = []
        for ly in lanes:
            xx = xs[keep]
            for off in np.linspace(-0.07, 0.07, 4):
                pts.append(np.stack([xx, np.full_like(xx, ly + off)], axis=1))
        spk = np.stack([rng.uniform(0, 400, 4000), rng.uniform(-6, 6, 4000)], axis=1)
        pts.append(spk)
        world = np.concatenate(pts)

        obs, poses = [], []
        for f in range(n_frames):
            D = speed * dt * f
            local = world.copy()
            local[:, 0] -= D
            uv = BC.project_ground(local, P_true)
            ok = (np.isfinite(uv[:, 0]) & (uv[:, 0] > 0) & (uv[:, 0] < 1920)
                  & (uv[:, 1] > 0.55 * 1080) & (uv[:, 1] < 0.88 * 1080))
            if ok.sum() < 200:
                continue
            obs.append(uv[ok])
            poses.append((D, 0.0, 0.0))
        if len(obs) >= 4:
            anchors.append((obs, poses, a))
    return anchors


def self_test() -> int:
    P_true = dict(BC.DEFAULTS)
    P_true.update(yaw=np.deg2rad(-7.0), pitch=np.deg2rad(-0.6), height=1.03, fx=1560.0,
                  lateral=-0.12, longitudinal=2.10)
    v_h = horizon_row(P_true)
    anchors = synth_anchors(P_true, n_anchors=6)
    grid = BC.BevGrid(x_range=(6.0, 40.0), y_range=(-8.0, 8.0), cell=0.06)
    fh_true = P_true["fx"] * P_true["height"]
    print(f"synthetic: {len(anchors)} anchors, "
          f"{np.mean([len(o) for o,_,_ in anchors]):.1f} frames each")
    print(f"truth: f {P_true['fx']:.0f}  h {P_true['height']:.3f}  f*h {fh_true:.1f}"
          f"  horizon row {v_h:.2f}\n")

    ok = True
    # ---- A. pure f*h error at the TRUE horizon -----------------------------
    # The factors bracket the whole plausible focal band and then some: an
    # estimator that only worked near the assumed value would be certifying the
    # assumption. The horizon is held at its true row, so the reconstruction is a
    # pure stretch and "expect s = factor" is exactly right.
    print("A. pure f*h error, horizon held at the truth  (expect s == factor)")
    for factor in (0.55, 0.70, 1.00, 1.40, 1.80):
        P = with_horizon(P_true, P_true["fx"] * factor, P_true["height"], v_h)
        recs, summ = measure_scale(anchors, P, grid, min_contrast=2.0)
        if summ is None:
            print(f"   f*h x {factor:.2f}: NO usable pairs — estimator failed")
            ok = False
            continue
        fh_est = P["fx"] * P["height"] / summ["s_median"]
        err = fh_est / fh_true - 1.0
        flag = "OK " if abs(err) < 0.06 else "BAD"
        print(f"   {flag} f*h x {factor:.2f}  ->  s {summ['s_median']:.3f} "
              f"(expect {factor:.3f})   f*h recovered {fh_est:7.1f} "
              f"vs true {fh_true:.1f}   err {100*err:+.1f}%   n={summ['n']}")
        ok &= abs(err) < 0.06

    # ---- B. the Mobius fit must recover a WRONG horizon AND still get f*h ----
    # A horizon error and a focal error are the same number at a single range.
    # This is the test that they come apart across range -- if it fails, no focal
    # value from this module is admissible, because it would be reporting the
    # pitch error as a focal length.
    print("\nB. joint (f*h, horizon) recovery -- the test that separates them")
    for dv in (-10.0, 0.0, +10.0):
        P = with_horizon(P_true, P_true["fx"], P_true["height"], v_h + dv)
        recs, summ = measure_scale(anchors, P, grid, min_contrast=2.0, n_bands=4)
        mb = fit_mobius(recs, P["fx"] * P["height"]) if recs else None
        if mb is None:
            print(f"   horizon {dv:+5.1f} px: too few range-resolved pairs to fit")
            ok = False
            continue
        d_err = mb["delta_px"] - (-dv)          # delta = v_h,true - v_h,assumed
        fh_err = mb["fh"] / fh_true - 1.0
        good = abs(d_err) < 4.0 and abs(fh_err) < 0.08
        print(f"   {'OK ' if good else 'BAD'} assumed horizon {v_h+dv:6.1f} "
              f"(delta true {-dv:+5.1f} px)  ->  delta {mb['delta_px']:+6.1f} px "
              f"(err {d_err:+5.1f})   f*h {mb['fh']:7.1f} (err {100*fh_err:+5.1f}%)"
              f"   naive whole-map s would say f*h {P['fx']*P['height']/summ['s_median']:7.1f}")
        ok &= good

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", type=pathlib.Path, default=None)
    ap.add_argument("--anchors", type=int, default=14)
    ap.add_argument("--span", type=float, default=1.2,
                    help="seconds per anchor. Longer than run_real's 0.6 s on purpose: "
                         "the lag test WANTS displacement, and unlike the agreement "
                         "objective it does not need the frames to share most of their ground.")
    ap.add_argument("--step", type=float, default=0.15)
    ap.add_argument("--max-pts", type=int, default=6000)
    ap.add_argument("--cell", type=float, default=0.06)
    ap.add_argument("--height", type=float, default=1.03)
    ap.add_argument("--fx", type=float, default=1478.3)
    ap.add_argument("--yaw", type=float, default=-7.01)
    ap.add_argument("--horizon", type=float, default=523.4,
                    help="assumed horizon ROW, not a pitch. The pitch is then solved "
                         "for. 523.4 px is the lane-VP measurement on this recording "
                         "([513.3, 534.8]); the pipeline's own pitch puts it at 464.4.")
    ap.add_argument("--pitch", type=float, default=None,
                    help="override: fix the pitch in degrees instead of the horizon row")
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--min-contrast", type=float, default=3.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    import run_real as RR
    run = a.run or RR.RUN
    recs_j = RR.load_records(run)
    anchors = RR.build_anchors(recs_j, run, a.anchors, a.span, a.step, a.max_pts)
    grid = BC.BevGrid(x_range=(6.0, 40.0), y_range=(-8.0, 8.0), cell=a.cell)
    P = dict(RR.NOMINAL)
    P.update(yaw=np.deg2rad(a.yaw), height=a.height, fx=a.fx)
    if a.pitch is not None:
        P["pitch"] = np.deg2rad(a.pitch)
    else:
        P["pitch"] = pitch_for_horizon(P, a.horizon)
    print(f"anchors {len(anchors)}   frames/anchor "
          f"{np.mean([len(o) for o,_,_ in anchors]):.1f}   cell {a.cell} m")
    print(f"assumed: yaw {a.yaw}  h {a.height}  f {a.fx}  => f*h {a.fx*a.height:.1f}")
    print(f"         pitch {np.rad2deg(P['pitch']):+.3f} deg  "
          f"=> horizon row {horizon_row(P):.2f}\n")

    recs, summ = measure_scale(anchors, P, grid, min_contrast=a.min_contrast,
                               n_bands=a.bands, verbose=a.verbose)
    if summ is None:
        print("NO pair produced a distinct correlation peak.")
        print("That is a real answer: this road has too little longitudinal ink "
              "for the lag test. It is NOT a licence to fall back on the biased "
              "agreement scan.")
        return 1

    fh_a = a.fx * a.height
    fh = fh_a / summ["s_median"]
    print(f"accepted pairs {summ['n']} over {len({r['anchor'] for r in recs})} anchors")
    print(f"  s  median {summ['s_median']:.3f}   IQR {summ['s_iqr']:.3f}   "
          f"p16-p84 [{summ['s_p16']:.3f}, {summ['s_p84']:.3f}]")
    print(f"  =>  f*h = {fh_a:.1f} / {summ['s_median']:.3f} = {fh:.1f} px*m")
    print(f"  =>  with h = {a.height:.3f} m :  f = {fh/a.height:7.1f} px  "
          f"(HFOV {np.rad2deg(2*np.arctan(1920/(2*fh/a.height))):.1f} deg)")

    bs = bootstrap_s(recs)
    out = dict(summary=summ, fh_assumed=fh_a, fh_measured=fh,
               height=a.height, f_measured=fh / a.height, pairs=recs)
    if bs:
        out["bootstrap_s"] = bs
        print(f"  anchor-cluster bootstrap on s: [{bs['lo']:.3f}, {bs['hi']:.3f}] "
              f"({bs['n_clusters']} clusters)  =>  f*h in "
              f"[{fh_a/bs['hi']:.0f}, {fh_a/bs['lo']:.0f}]  =>  f in "
              f"[{fh_a/bs['hi']/a.height:.0f}, {fh_a/bs['lo']/a.height:.0f}] px")

    hz = fit_scale_and_horizon(recs, P, grid)
    if hz:
        out["range_slope"] = hz
        print(f"\n  range dependence of s: slope {hz['slope']:+.4f} /m "
              f"(SE {hz['slope_se']:.4f}, t {hz['slope_t']:+.2f}) over "
              f"x {hz['x_span'][0]:.1f}-{hz['x_span'][1]:.1f} m")

    band = (1356, 1628)
    f_meas = fh / a.height

    # ---- the joint answer: f*h and the horizon together ---------------------
    mb = fit_mobius(recs, fh_a)
    if mb:
        out["mobius"] = mb
        v_a = horizon_row(P)
        f_mb = mb["fh"] / a.height
        print(f"\n  JOINT (f*h, horizon) fit over {mb['n']} range-resolved lags, "
              f"x {mb['range_span'][0]:.1f}-{mb['range_span'][1]:.1f} m, "
              f"{mb['n_clusters']} anchors")
        print(f"    horizon row  {v_a:.1f} assumed  {mb['delta_px']:+.1f} px  "
              f"->  {v_a + mb['delta_px']:.1f}"
              + (f"   95% CI [{v_a+mb['delta_ci'][0]:.1f}, {v_a+mb['delta_ci'][1]:.1f}]"
                 if "delta_ci" in mb else ""))
        print(f"    f*h          {mb['fh']:.1f} px*m"
              + (f"   95% CI [{mb['fh_ci'][0]:.0f}, {mb['fh_ci'][1]:.0f}]"
                 if "fh_ci" in mb else ""))
        print(f"    f (h={a.height:.3f})  {f_mb:.0f} px   "
              f"(HFOV {np.rad2deg(2*np.arctan(1920/(2*f_mb))):.1f} deg)"
              + (f"   95% CI [{mb['fh_ci'][0]/a.height:.0f}, "
                 f"{mb['fh_ci'][1]/a.height:.0f}]" if "fh_ci" in mb else ""))
        print(f"    S21 FE plausible band {band[0]}-{band[1]} px: "
              f"{'INSIDE' if band[0] <= f_mb <= band[1] else 'OUTSIDE'}")
        f_meas = f_mb

    print(f"\n  whole-map (horizon assumed exact) would say f = {fh/a.height:.0f} px: "
          f"{'INSIDE' if band[0] <= fh/a.height <= band[1] else 'OUTSIDE'} the band")
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
