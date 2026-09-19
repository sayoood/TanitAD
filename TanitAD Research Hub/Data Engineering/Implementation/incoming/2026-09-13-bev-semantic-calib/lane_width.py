#!/usr/bin/env python3
"""Height from LATERAL road geometry — the one thing the motion cannot give us.

WHY THIS IS THE MISSING HALF, AND WHY IT IS NOT OPTIONAL
--------------------------------------------------------
Back-projection onto the road plane sees the two axes differently
(``scale_calib``'s own derivation, re-verified here)::

    lateral       y = (u - cx) * h / (v - v_h)        -> h ALONE, no f
    longitudinal  x = f * h / (v - v_h)               -> the PRODUCT f*h

So motion against the odometer can only ever deliver ``f*h``, and ``f`` needs a
lateral metre-stick. That is not a textbook claim here, it is MEASURED on this
recording: ``flow_scale.observability`` differentiates the actual reprojection on
the actual tracks and returns singular values ``[8766, 997, 676, 24.7]`` with the
near-null direction being ``height -0.82, yaw +0.57``. Height is 350x less
observable than ``f*h`` from the motion. No optimiser fixes that.

WHAT IS MEASURED, AND WHAT IS ASSUMED
-------------------------------------
Two lateral quantities, both read off a SINGLE frame's back-projection:

  * **lane width** -- the spacing between adjacent lane-line peaks;
  * **painted line width** -- each peak's own FWHM.

Both scale linearly with ``h``, so measuring either in reconstructed metres and
dividing by its true value gives the correction directly, with no search.

⚠️ **The true values are INPUTS and they are declared, not discovered.**
``--lane-width-m`` (3.50 m, French motorway) and ``--line-width-m`` (0.15 m) are
the assumptions this module runs on; change them and every number moves
proportionally. What the module contributes is the MEASUREMENT and its spread.
Where the two disagree, that disagreement is the result -- it says the
reconstruction is not similar to the road, which no choice of ``h`` can fix.

⚠️ **Single frame, never stacked.** MEASURED 2026-09-13: pooling 13 anchors smeared
the lateral profile into one band of FWHM 0.360 m against a ~0.15 m line, because
the car wanders within its lane and each anchor sits at a different offset. The
stacked measurement then implied ``h = 0.59 m``, which is not a camera height, it
is a smear. Every profile here comes from one frame.

⚠️ **No width censoring.** ``lane_calib`` accepts a lane only if ``2.6 < w < 4.6`` m
at its assumed height, which makes the measured width nearly uninformative about
that height -- the filter enforces the answer. The bracket here is 2.0-6.0 m, it
is applied to the RECONSTRUCTED spacing, and the fraction of candidates it removes
is reported so the censoring is visible instead of implicit.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bev_calib as BC                                              # noqa: E402
import lag_scale as LS                                              # noqa: E402


def lateral_profile(uv: np.ndarray, P: dict, x_win=(8.0, 25.0), y_win=(-8.0, 8.0),
                    cell=0.03, smooth=1.0):
    """Ink summed along the road, as a function of lateral offset.

    ``x_win`` starts at 8 m and stops at 25 m on purpose: nearer than that the
    bonnet and the ridge band's lower cut intrude, and beyond it a 0.15 m line is
    thinner than one pixel row, so its reconstructed width is set by the imaging
    rather than by the paint.
    """
    g = BC.ground_from_pixels(uv, P)
    m = (g[:, 0] >= x_win[0]) & (g[:, 0] <= x_win[1]) \
        & (g[:, 1] >= y_win[0]) & (g[:, 1] <= y_win[1])
    g = g[m]
    if len(g) < 80:
        return None, None
    ny = int(round((y_win[1] - y_win[0]) / cell))
    idx = ((g[:, 1] - y_win[0]) / cell).astype(int)
    idx = idx[(idx >= 0) & (idx < ny)]
    prof = np.bincount(idx, minlength=ny).astype(float)
    if smooth > 0:
        prof = BC._blur(prof[:, None], smooth)[:, 0]
    ys = y_win[0] + (np.arange(ny) + 0.5) * cell
    return ys, prof


def peaks_with_fwhm(ys, prof, min_rel=0.30, min_sep_m=0.6):
    """Lane-line candidates: local maxima with their full width at half maximum."""
    base = np.median(prof)
    p = np.clip(prof - base, 0, None)
    if p.max() <= 0:
        return []
    cell = float(ys[1] - ys[0])
    out = []
    for i in range(1, len(p) - 1):
        if p[i] >= p[i - 1] and p[i] > p[i + 1] and p[i] > min_rel * p.max():
            half = p[i] / 2.0
            lo = i
            while lo > 0 and p[lo] > half:
                lo -= 1
            hi = i
            while hi < len(p) - 1 and p[hi] > half:
                hi += 1
            out.append(dict(y_m=float(ys[i]), fwhm_m=float((hi - lo) * cell),
                            strength=float(p[i] / p.max())))
    out.sort(key=lambda d: -d["strength"])
    keep = []
    for d in out:
        if all(abs(d["y_m"] - e["y_m"]) > min_sep_m for e in keep):
            keep.append(d)
    return sorted(keep, key=lambda d: d["y_m"])


def cluster_bootstrap(vals, groups, n_boot=4000, seed=0):
    """95% interval, resampling FRAMES rather than measurements.

    Several spacings can come from one frame and they share that frame's ink, its
    stretch of road and its calibration residual. Resampling measurements would
    report an interval several times too narrow -- the same reason this
    programme's decision-grade interval clusters on episodes.
    """
    by = {}
    for v, g in zip(vals, groups):
        by.setdefault(g, []).append(v)
    keys = list(by)
    if len(keys) < 3:
        return None
    rng = np.random.default_rng(seed)
    out = [np.median(np.concatenate([by[keys[p]] for p in
                                     rng.choice(len(keys), len(keys), replace=True)]))
           for _ in range(n_boot)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=None)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--height", type=float, default=1.03)
    ap.add_argument("--fx", type=float, default=1478.3)
    ap.add_argument("--yaw", type=float, default=-7.01)
    ap.add_argument("--horizon", type=float, default=523.4)
    ap.add_argument("--lane-width-m", type=float, default=3.50,
                    help="ASSUMED true lane width (French motorway). An input.")
    ap.add_argument("--line-width-m", type=float, default=0.15,
                    help="ASSUMED true painted line width. An input.")
    ap.add_argument("--cell", type=float, default=0.03)
    ap.add_argument("--bracket", type=float, nargs=2, default=[2.0, 6.0],
                    help="reconstructed spacings accepted as a lane, in metres. "
                         "Stated because it censors: the fraction removed is printed.")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    import run_real as RR
    run = a.run or RR.RUN
    recs = RR.load_records(run)
    fdir = run / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > 8.0
              and (fdir / f"{int(r['frame']):06d}.jpg").exists()]
    picks = np.linspace(0, len(usable) - 1, a.frames).astype(int)

    P = dict(RR.NOMINAL)
    P.update(yaw=np.deg2rad(a.yaw), height=a.height, fx=a.fx)
    P["pitch"] = LS.pitch_for_horizon(P, a.horizon)
    print(f"assumed: yaw {a.yaw}  h {a.height}  f {a.fx}  horizon {a.horizon}")
    print(f"inputs:  lane width {a.lane_width_m} m   line width {a.line_width_m} m")
    print(f"frames: {len(picks)} single-frame profiles (never stacked)\n")

    from trajlib import lane_calib as LC
    rng = np.random.default_rng(0)
    widths, w_grp, lines, l_grp = [], [], [], []
    n_cand, n_kept, n_prof, npk = 0, 0, 0, []
    for pi in picks:
        f = int(usable[pi]["frame"])
        img = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H = img.shape[0]
        u, v = LC._ridge_points(img, int(0.55 * H), int(0.95 * H))
        if len(u) < 100:
            continue
        uv = np.stack([u, v], axis=1).astype(float)
        ys, prof = lateral_profile(uv, P, cell=a.cell)
        if ys is None:
            continue
        n_prof += 1
        pk = peaks_with_fwhm(ys, prof)
        npk.append(len(pk))
        for d in pk:
            lines.append(d["fwhm_m"])
            l_grp.append(f)
        for i in range(len(pk) - 1):
            w = pk[i + 1]["y_m"] - pk[i]["y_m"]
            n_cand += 1
            if a.bracket[0] <= w <= a.bracket[1]:   # stated bracket; its cost is printed
                widths.append(w)
                w_grp.append(f)
                n_kept += 1

    print(f"profiles used {n_prof}   peaks/frame "
          f"median {int(np.median(npk)) if npk else 0}   "
          f"adjacent-pair candidates {n_cand}, kept in the 2.0-6.0 m bracket "
          f"{n_kept} ({100*n_kept/max(n_cand,1):.0f}%)")
    out = dict(n_profiles=n_prof, n_candidates=n_cand, n_kept=n_kept,
               height_assumed=a.height, fx_assumed=a.fx, horizon=a.horizon)

    if widths:
        wm = float(np.median(widths))
        ci = cluster_bootstrap(widths, w_grp)
        k = a.lane_width_m / wm
        print(f"\nLANE WIDTH   n = {len(widths)} spacings over "
              f"{len(set(w_grp))} frames")
        print(f"    measured  {wm:.3f} m   IQR "
              f"{np.percentile(widths,75)-np.percentile(widths,25):.3f} m"
              + (f"   95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
        print(f"    vs the assumed {a.lane_width_m:.2f} m  ->  h x {k:.3f}  "
              f"=>  h = {a.height*k:.3f} m"
              + (f"   [{a.height*a.lane_width_m/ci[1]:.3f}, "
                 f"{a.height*a.lane_width_m/ci[0]:.3f}]" if ci else ""))
        out["lane_width"] = dict(measured_m=wm, ci=ci, n=len(widths),
                                 n_frames=len(set(w_grp)), k_h=k,
                                 h_implied=a.height * k)
    else:
        print("\nLANE WIDTH   no adjacent peak pair survived — cannot measure")

    if lines:
        blur_fwhm = 2.355 * 1.0 * a.cell
        lm = float(np.median(lines))
        lc = np.sqrt(max(lm ** 2 - blur_fwhm ** 2, 1e-6))
        ci = cluster_bootstrap(lines, l_grp)
        k = a.line_width_m / lc
        print(f"\nLINE WIDTH   n = {len(lines)} peaks over {len(set(l_grp))} frames")
        print(f"    measured FWHM {lm:.3f} m   deblurred {lc:.3f} m   "
              f"(blur contributes {blur_fwhm:.3f} m)"
              + (f"   95% CI on FWHM [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
        print(f"    vs the assumed {a.line_width_m:.3f} m  ->  h x {k:.3f}  "
              f"=>  h = {a.height*k:.3f} m")
        out["line_width"] = dict(fwhm_m=lm, deblurred_m=float(lc), ci=ci,
                                 n=len(lines), k_h=float(k),
                                 h_implied=float(a.height * k))

    if "lane_width" in out and "line_width" in out:
        h1, h2 = out["lane_width"]["h_implied"], out["line_width"]["h_implied"]
        print(f"\nTWO LATERAL METRICS, SAME AXIS:  lane {h1:.3f} m   line {h2:.3f} m"
              f"   ratio {max(h1,h2)/max(min(h1,h2),1e-9):.2f}x")
        if max(h1, h2) / max(min(h1, h2), 1e-9) > 1.35:
            print("    ⚠️ they DISAGREE. Both are pure lateral measurements of the same"
                  "\n       reconstruction, so no height reconciles them — the"
                  "\n       reconstruction is not similar to the road. Treat neither as"
                  "\n       a height until the disagreement is explained.")
        else:
            print("    they agree: two independent lateral rulers give the same height.")
        out["h_agreement_ratio"] = float(max(h1, h2) / max(min(h1, h2), 1e-9))

    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
