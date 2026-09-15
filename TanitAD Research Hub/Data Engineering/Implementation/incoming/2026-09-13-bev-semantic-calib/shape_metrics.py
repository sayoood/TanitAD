#!/usr/bin/env python3
"""Pin h and f from the SHAPE of road markings in the BEV — the external metric.

WHY THIS EXISTS
---------------
MEASURED 2026-09-13 on the real recording: BEV cross-frame agreement determines
the EXTRINSICS well (yaw -7.01 + pitch -0.64 beat the shipped nominal by +41%)
but does NOT determine height or focal -- the f*h manifold is only 2.09x peaked
against 9-17x in synthetic, and an unconstrained fit runs to its bounds
(fx 2588 of a 2600 limit). Every image-only estimator in this programme has now
failed on that same direction:

    plane_calib   height 1.691 m   DECLINED (spread +/-0.56)
    scale_calib   f*h 1608.1       DECLINED (spread 62%)
    BEV agreement peak h 0.95 m    only 2.09x contrast

So stop asking the optimiser and supply the missing metric. Road markings are
manufactured to a standard, and the two quantities they fix are orthogonal in
exactly the way f and h are:

    painted line WIDTH   is LATERAL       ->  y = (u-cx)*h/(v-v_h)   ->  h ALONE, no f
    dash PERIOD          is LONGITUDINAL  ->  x = f*h/(v-v_h)        ->  f*h

Measure both in the BEV, compare to the standard, and h and f both fall out.

⚠️ THE STANDARD IS AN INPUT, NOT A RESULT. This module does not hard-code a
national value and then report agreement with it -- that would be circular. It
MEASURES the width and period at a given calibration, reports them in metres, and
leaves the comparison explicit via ``--line-width-m`` / ``--dash-period-m``. Where
the standard is uncertain, the measured period is itself diagnostic: dash periods
are strongly quantised (French T1 lane lines are 3 m mark + 10 m gap = 13 m), so a
measurement near a standard value corroborates the calibration, and one far from
every standard says the calibration is wrong rather than the road unusual.

HOW THE TWO MEASUREMENTS ARE MADE
---------------------------------
Build the stacked BEV at a candidate calibration, then:

  * **lateral profile** -- sum the ink along x. Lane lines appear as peaks in y.
    Each peak's full width at half maximum, in metres, is the painted line width
    (plus a blur term, which is why the blur is reported alongside it).
  * **longitudinal autocorrelation** -- take the ink inside one line's y-band and
    autocorrelate along x. The first strong non-zero lag is the dash period.

Both scale simply: width ∝ h, period ∝ f*h. So a single BEV at any reasonable
starting calibration yields the CORRECTION factors directly, with no search.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bev_calib as BC                                              # noqa: E402
import run_real as RR                                               # noqa: E402


def anchor_bev(obs, poses, P, grid, sigma=1.0) -> np.ndarray:
    """BEV for ONE anchor.

    ⚠️ Deliberately NOT pooled across anchors. Each anchor sits at a different
    place on the road and the car wanders within its lane, so the lane lines are
    at different lateral offsets per anchor. MEASURED 2026-09-13: pooling 13
    anchors smeared the lateral profile into a SINGLE band of FWHM 0.360 m against
    a ~0.15 m painted line, destroying the very quantity being measured. Shape is
    measured per anchor and aggregated afterwards.
    """
    tot = None
    for o, ps in zip(obs, poses):
        H = grid.accumulate_one(BC.to_anchor(BC.ground_from_pixels(o, P), ps))
        tot = H if tot is None else tot + H
    return BC._blur(tot, sigma) if tot is not None else None


def lateral_bands(H: np.ndarray, grid: BC.BevGrid, min_rel=0.35):
    """Lane-line candidates from the lateral profile, with FWHM widths in metres."""
    prof = H.sum(axis=0)                       # sum over x -> profile in y
    prof = prof - np.median(prof)
    prof[prof < 0] = 0
    if prof.max() <= 0:
        return []
    peaks = []
    for i in range(1, len(prof) - 1):
        if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > min_rel * prof.max():
            half = prof[i] / 2.0
            lo = i
            while lo > 0 and prof[lo] > half:
                lo -= 1
            hi = i
            while hi < len(prof) - 1 and prof[hi] > half:
                hi += 1
            peaks.append(dict(y_m=grid.y0 + (i + 0.5) * grid.cell,
                              fwhm_m=(hi - lo) * grid.cell,
                              strength=float(prof[i] / prof.max()),
                              idx=i))
    # drop peaks whose FWHM overlaps a stronger neighbour (one line, two bumps)
    peaks.sort(key=lambda p: -p["strength"])
    keep = []
    for p in peaks:
        if all(abs(p["y_m"] - q["y_m"]) > 0.5 for q in keep):
            keep.append(p)
    return sorted(keep, key=lambda p: p["y_m"])


def dash_period(H: np.ndarray, grid: BC.BevGrid, band_idx: int, half_cells: int = 4,
                lo_m: float = 4.0, hi_m: float = 25.0):
    """Autocorrelation of the ink along x inside one lane-line band.

    ⚠️ Searches for the STRONGEST peak inside a plausible dash-period range, not
    the first peak past a short dead zone. MEASURED 2026-09-13: a 1.0 m dead zone
    plus first-peak selection returned 1.88 m -- ink granularity, not a dash --
    which then implied a 4.6 deg HFOV. Real dash periods are 6-14 m (French T1 is
    3 m mark + 10 m gap = 13 m), so the window is 4-25 m.
    """
    lo = max(0, band_idx - half_cells)
    hi = min(H.shape[1], band_idx + half_cells + 1)
    sig = H[:, lo:hi].sum(axis=1).astype(float)
    sig = sig - sig.mean()
    if np.allclose(sig, 0):
        return None
    ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
    ac /= ac[0]
    i0, i1 = int(round(lo_m / grid.cell)), min(len(ac) - 2, int(round(hi_m / grid.cell)))
    if i1 <= i0 + 1:
        return None
    cand = [(float(ac[i]), i) for i in range(i0, i1)
            if ac[i] >= ac[i - 1] and ac[i] > ac[i + 1]]
    if not cand:
        return None
    strength, i = max(cand)
    if strength < 0.15:
        return None
    return (i * grid.cell, strength)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=RR.RUN)
    ap.add_argument("--anchors", type=int, default=14)
    ap.add_argument("--span", type=float, default=0.6)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--max-pts", type=int, default=4000)
    ap.add_argument("--cell", type=float, default=0.04, help="finer grid: we are measuring widths")
    ap.add_argument("--line-width-m", type=float, default=0.15,
                    help="painted line width to compare against (NOT assumed in the measurement)")
    ap.add_argument("--dash-period-m", type=float, default=13.0,
                    help="dash+gap period to compare against (French T1 = 3 m + 10 m)")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    recs = RR.load_records(a.run)
    anchors = RR.build_anchors(recs, a.run, a.anchors, a.span, a.step, a.max_pts)
    grid = BC.BevGrid(x_range=(6.0, 38.0), y_range=(-8.0, 8.0), cell=a.cell)
    P = dict(RR.NOMINAL)
    P.update(yaw=np.deg2rad(-7.01), pitch=np.deg2rad(-0.64), height=1.21, fx=1478.3)
    print(f"anchors {len(anchors)}   grid cell {a.cell} m")
    print(f"reference calibration: yaw -7.01  pitch -0.64  h {P['height']}  f {P['fx']}\n")

    # ---- per-anchor measurement, then aggregate -----------------------------
    W, D, nb = [], [], []
    for obs, poses, fno in anchors:
        H = anchor_bev(obs, poses, P, grid)
        if H is None:
            continue
        bands = lateral_bands(H, grid)
        nb.append(len(bands))
        if not bands:
            continue
        for b in bands:
            W.append(b["fwhm_m"])
        strongest = max(bands, key=lambda b: b["strength"])
        dp = dash_period(H, grid, strongest["idx"])
        if dp is not None:
            D.append(dp[0])
    print(f"per-anchor: bands found {nb}  (total {len(W)} lines, {len(D)} dash periods)")
    if not W:
        print("no lane-line bands in any anchor — cannot measure shape")
        return 1

    blur_fwhm = 2.355 * 1.0 * a.cell
    w_med = float(np.median(W))
    w_iqr = float(np.percentile(W, 75) - np.percentile(W, 25))
    w_corr = float(np.sqrt(max(w_med ** 2 - blur_fwhm ** 2, 1e-6)))
    k_h = a.line_width_m / w_corr
    print(f"\nLATERAL  (pins h alone, no f)   n = {len(W)} lines")
    print(f"    line FWHM   median {w_med:.3f} m   IQR {w_iqr:.3f} m   "
          f"(blur contributes {blur_fwhm:.3f} m)")
    print(f"    deblurred   {w_corr:.3f} m   vs standard {a.line_width_m:.3f} m")
    print(f"    ->  h x {k_h:.3f}   =>  h = {P['height'] * k_h:.3f} m")

    print(f"\nLONGITUDINAL  (pins f*h)   n = {len(D)} periods")
    k_fh = None
    if not D:
        print("    no periodic structure found in the 4-25 m window")
    else:
        d_med = float(np.median(D))
        d_iqr = float(np.percentile(D, 75) - np.percentile(D, 25))
        print(f"    dash period median {d_med:.2f} m   IQR {d_iqr:.2f} m   "
              f"values {[round(x,1) for x in sorted(D)]}")
        if d_iqr > 0.35 * d_med:
            print(f"    ⚠️ IQR is {100*d_iqr/d_med:.0f}% of the median — the anchors do NOT agree on a")
            print(f"       period, so this is not a reliable measurement of a standard dash.")
        k_fh = a.dash_period_m / d_med
        print(f"    vs standard {a.dash_period_m:.2f} m  ->  f*h x {k_fh:.3f}"
              f"   =>  f*h = {P['fx'] * P['height'] * k_fh:.0f} px*m")
        f_new = P["fx"] * P["height"] * k_fh / (P["height"] * k_h)
        print(f"    with h above  =>  f = {f_new:.0f} px "
              f"(HFOV {np.rad2deg(2 * np.arctan(1920 / (2 * f_new))):.1f} deg)")
        if not (900 <= f_new <= 2600):
            print(f"    ⚠️ that focal is outside any plausible S21 FE value (1356-1628 px);")
            print(f"       treat the chain as REFUTED rather than as a calibration.")

    out = dict(n_lines=len(W), n_periods=len(D), bands_per_anchor=nb,
               width_fwhm_median_m=w_med, width_fwhm_iqr_m=w_iqr,
               width_deblurred_m=w_corr, k_h=float(k_h),
               h_implied=float(P["height"] * k_h),
               dash_period_median_m=(float(np.median(D)) if D else None),
               k_fh=(None if k_fh is None else float(k_fh)))
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
