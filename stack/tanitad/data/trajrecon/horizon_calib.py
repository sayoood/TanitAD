"""Horizon row from the fact that a lane does not change width with range.

⛔ WHY THIS EXISTS. The horizon was the one parameter of the 2026-08-08 calibration
that no instrument could pin, and it cost a wrong render. Every direct estimate
needed ``f*h``, which the ground plane cannot separate (``scale_calib``), and the
two containment metrics built to settle the geometry are both **structurally blind
to it**:

* road containment is degenerate in the horizon — the ribbon/road width ratio
  falls 0.197 -> 0.176 as the horizon rises 440 -> 500 **while its score rises**;
* lane containment scores a *difference* of clearances, in which a scale error
  cancels exactly.

Both return "fine" for a 9 px horizon error, and a 9 px error is not cosmetic: the
drawn width is wrong by ``(v - v_h_true)/(v - v_h_used)``, which was **3.7 % at
10 m and 11.1 % at 30 m** — the ribbon **flares outward with range**, and a flare
is exactly what "the trajectory is leaving the road" looks like.

⭐ THE SIGNATURE, AND IT NEEDS NO KNOWN LANE WIDTH. The lane is the same width at
8 m as at 30 m. The lateral scale at image row ``v`` is ``h / (v - v_h)`` — the
focal length cancels — so a lane of constant metric width ``W`` subtends

    w_px(v) = (W / h) * (v - v_h)

**a straight line through the horizon row.** Therefore:

    v_h = the x-INTERCEPT of a line fitted to (row, lane width in px)
    W/h = its SLOPE

No scan, no assumed width, no focal length, and the width's *value* never enters
the horizon estimate — only its constancy. *(This is the same shape as the
``clear_L`` diagnostic that settled the yaw: parallel means flat.)*

⚠️ WHAT IT DOES AND DOES NOT CLAIM.

* It returns an **EFFECTIVE horizon for the road it was measured on.** A constant
  road grade over the fitted range acts as a horizon offset and is absorbed here.
  That is correct for placing a drawing on *that* road and it is **not a claim
  about the camera's mounting pitch** — do not quote it as one.
* It returns ``lane_width_over_height`` (the slope), **not** a lane width.
  Recovering metres needs ``h``, so ``f*h`` stays exactly as degenerate as it was.
* MEASURED on 2026-08-08 (2258 observations, rows 538-811): ``v_h = 463.9``,
  R^2 0.975, against an independent scan of width-vs-range flatness that put the
  flat band at **460-464**. The value in use before this was 472.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["HorizonFit", "horizon_from_lane_width"]


@dataclass(frozen=True)
class HorizonFit:
    """Result of :func:`horizon_from_lane_width`.

    ``ok`` is False when the fit was refused; ``reason`` then says why, and the
    caller must fall back rather than use the numbers.
    """

    ok: bool
    horizon_row: float           # x-intercept: the row where lane width -> 0
    lane_width_over_height: float  # slope, = W / h  (dimensionless)
    r2: float
    n_used: int
    n_total: int
    row_span: float
    reason: str = ""

    def lane_width_m(self, height_m: float) -> float:
        """Lane width in metres, GIVEN a camera height. Not a free measurement."""
        return float(self.lane_width_over_height * height_m)


def horizon_from_lane_width(
    rows,
    widths_px,
    *,
    min_rows: int = 40,
    min_row_span: float = 60.0,
    trim_sd: float = 2.5,
    iters: int = 5,
    min_r2: float = 0.60,
    row_bounds: tuple[float, float] | None = None,
) -> HorizonFit:
    """Fit ``w_px = (W/h) * (v - v_h)`` and return the horizon row ``v_h``.

    Parameters
    ----------
    rows, widths_px
        Paired observations: the image row of a lane-width measurement, and that
        width in pixels. They may come from many frames pooled together — the fit
        only needs the lane's width to be constant in METRES, not in pixels.
    trim_sd, iters
        Robust re-fitting: points more than ``trim_sd`` robust sd from the current
        line are dropped and the line re-fitted. Paint misdetections (a shoulder
        edge line, a joint, a vehicle) are **outliers with leverage**, and a
        single one at a far row moves an x-intercept a long way.
    row_bounds
        Optional ``(lo, hi)`` sanity window for the answer. A fit landing outside
        it is refused rather than returned, because an x-intercept extrapolated
        beyond the observed rows is the failure mode of this estimator.

    ⚠️ THE INTERCEPT IS AN EXTRAPOLATION and its precision degrades sharply as the
    observed row span shrinks toward the horizon — that is why ``min_row_span``
    is enforced and why ``row_span`` is returned for the caller to judge.
    """
    v = np.asarray(rows, dtype=float).ravel()
    w = np.asarray(widths_px, dtype=float).ravel()
    if v.shape != w.shape:
        raise ValueError(f"rows and widths_px differ in length: {v.shape} vs {w.shape}")
    good = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[good], w[good]
    n_total = int(v.size)
    if n_total < min_rows:
        return HorizonFit(False, float("nan"), float("nan"), float("nan"),
                          0, n_total, 0.0,
                          f"only {n_total} usable observations (need {min_rows})")
    span = float(v.max() - v.min())
    if span < min_row_span:
        return HorizonFit(False, float("nan"), float("nan"), float("nan"),
                          0, n_total, span,
                          f"row span {span:.0f} px too short (need {min_row_span:.0f}); "
                          "the intercept would be extrapolated far beyond the data")

    keep = np.ones(n_total, dtype=bool)
    slope = intercept = float("nan")
    for _ in range(max(1, iters)):
        if keep.sum() < min_rows:
            break
        A = np.stack([np.ones(keep.sum()), v[keep]], axis=1)
        intercept, slope = np.linalg.lstsq(A, w[keep], rcond=None)[0]
        resid = w - (intercept + slope * v)
        med = float(np.median(resid))
        sd = 1.4826 * float(np.median(np.abs(resid - med)))
        if not np.isfinite(sd) or sd <= 0:
            break
        nxt = np.abs(resid - med) < trim_sd * sd
        if nxt.sum() < min_rows or np.array_equal(nxt, keep):
            keep = nxt if nxt.sum() >= min_rows else keep
            break
        keep = nxt

    if not np.isfinite(slope) or slope <= 0:
        return HorizonFit(False, float("nan"), float("nan"), float("nan"),
                          int(keep.sum()), n_total, span,
                          f"slope {slope:.4g} is not positive — lane width must GROW "
                          "toward the bottom of the image; the inputs are inconsistent")

    resid = w[keep] - (intercept + slope * v[keep])
    denom = float(np.var(w[keep]))
    r2 = 1.0 - float(np.var(resid)) / denom if denom > 0 else 0.0
    if r2 < min_r2:
        return HorizonFit(False, float("nan"), float("nan"), r2,
                          int(keep.sum()), n_total, span,
                          f"R^2 {r2:.3f} below {min_r2:.2f} — the width is not linear "
                          "in row, so a single horizon does not describe these data")

    v_h = float(-intercept / slope)
    if row_bounds is not None and not (row_bounds[0] <= v_h <= row_bounds[1]):
        return HorizonFit(False, v_h, float(slope), r2, int(keep.sum()), n_total, span,
                          f"horizon {v_h:.1f} outside the admissible window "
                          f"[{row_bounds[0]:.0f}, {row_bounds[1]:.0f}]")
    return HorizonFit(True, v_h, float(slope), r2, int(keep.sum()), n_total, span)
