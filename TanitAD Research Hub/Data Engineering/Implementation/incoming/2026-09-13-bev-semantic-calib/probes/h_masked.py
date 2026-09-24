#!/usr/bin/env python3
"""Camera height from lane width — REDONE with the road mask (`R-2026-09-14-foliage`).

§50's h = 1.586 m is withdrawn: its lane width came from an unmasked ridge detector that
was finding the sunlit bank, not the paint. This repeats the measurement with
`road_mask` in front of it, and with the pair search CONSTRAINED so that a "lane" whose
two sides are not parallel cannot be returned at all.

THE ALGEBRA (unchanged, and it is why this works without a focal length). Eliminating
range between ``u = cx + f(y-lat)/x`` and ``v = v_h + f·h/x``::

    Delta_y  =  Delta_u · h / (v - v_h)

so with ``q = v - v_h`` every road-parallel line is ``u = u_vp + m·q`` with ``m = y/h``,
and two boundaries of one lane share ``u_vp`` and differ by ``m_R - m_L = w/h``. Two
free parameters, no focal length, and a non-parallel pair is not expressible.

    lane width in units of h   =  |m_R - m_L|          <- measured
    h                          =  3.5 m / that          <- Sayed's lane width
    horizon                    =  the v_h that makes the width range-INDEPENDENT

⚠️ THE PREVIEW IS PART OF THE PROCEDURE. Run ``probes/road_mask.py`` and look at the
annotated frames before believing anything here. That is the rule the foliage
retraction exists to enforce, and it is the step whose absence cost a day.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import run_real as RR                                                  # noqa: E402
from road_mask import calib, road_mask, masked_ridges                  # noqa: E402
from lane_pair_hough import lane_pair                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx", type=float, default=1533.0)
    ap.add_argument("--height-mask", type=float, default=1.586,
                    help="height used ONLY to project the coarse mask")
    ap.add_argument("--horizon", type=float, default=448.4)
    ap.add_argument("--yaw", type=float, default=-5.35)
    ap.add_argument("--lane-m", type=float, default=3.5)
    ap.add_argument("--x-lo", type=float, default=10.0)
    ap.add_argument("--x-hi", type=float, default=38.0)
    ap.add_argument("--y-left", type=float, default=4.2)
    ap.add_argument("--y-right", type=float, default=2.6)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    P = calib(fx=a.fx, height=a.height_mask, horizon=a.horizon, yaw=a.yaw)
    fh = a.fx * a.height_mask
    rows = np.arange(a.horizon + fh / a.x_hi, a.horizon + fh / a.x_lo, 1.0)
    u_grid = np.arange(560.0, 1160.0, 3.0)
    m_grid = np.arange(-4.0, 4.0001, 0.01)
    # admissible separation in units of h: a 2.6-4.4 m lane over a 1.2-2.0 m camera
    sep_lo, sep_hi = 2.6 / 2.0, 4.4 / 1.2

    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    fdir = RR.RUN / "frames"
    avail = [f for f in sorted(int(p.stem) for p in fdir.glob("*.jpg"))
             if (r := recs.get(f)) and r["complete"] and r["speed_ms"] > 8]
    pick = [avail[i] for i in np.linspace(0, len(avail) - 1,
                                          min(a.n, len(avail))).astype(int)]
    mask = None
    DM, UV, FR, PAIRS = [], [], [], []
    npts = []
    for f in pick:
        g = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if mask is None:
            mask = road_mask(P, g.shape, y_left=a.y_left, y_right=a.y_right,
                             x_range=(8.0, a.x_hi + 2.0))
            print(f"mask: {100*float(mask.mean())/255:.1f} % of the frame, "
                  f"rows {np.nonzero(mask.any(1))[0].min()}-"
                  f"{np.nonzero(mask.any(1))[0].max()}")
        pts = masked_ridges(g, mask, rows, fh, a.horizon, a.fx)
        npts.append(len(pts))
        if len(pts) < 40:
            continue
        r = lane_pair(pts, a.horizon, u_grid, m_grid, sep_lo, sep_hi)
        if r is None:
            continue
        uv, mL, mR, nL, nR = r
        DM.append(abs(mR - mL)); UV.append(uv); FR.append(f)
        PAIRS.append((uv, mL, mR))
    DM = np.asarray(DM); UV = np.asarray(UV); FR = np.asarray(FR)
    mad = lambda z: 1.4826 * np.median(np.abs(z - np.median(z)))
    print(f"{len(pick)} frames, median {np.median(npts):.0f} masked ridge points each")
    print(f"lane pair found in {len(DM)}/{len(pick)} ({100*len(DM)/len(pick):.0f} %)\n")
    if len(DM) < 40:
        print("too few pairs")
        return 1

    print("  |m_R - m_L| histogram  (= lane width in units of the camera height h)")
    hist, edges = np.histogram(DM, bins=np.arange(1.0, 4.01, 0.1))
    for c, lo, hi in zip(hist, edges[:-1], edges[1:]):
        if c:
            bar = "#" * int(round(46 * c / max(hist.max(), 1)))
            print(f"    {lo:4.1f}-{hi:4.1f}  n {c:4d}  {bar}   (h = {a.lane_m/((lo+hi)/2):.2f} m)")
    peak = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
    sel = np.abs(DM - peak) < 0.45
    print(f"\n  peak bin at {peak:.2f}; {sel.sum()} of {len(DM)} pairs within +/-0.45 of it")
    dm = float(np.median(DM[sel]))
    h_hat = a.lane_m / dm
    rng = np.random.default_rng(0)
    fu = np.unique(FR[sel])
    bs = []
    for _ in range(600):
        p = rng.choice(fu, len(fu), replace=True)
        idx = np.concatenate([np.flatnonzero(FR == q) for q in p])
        v = DM[idx]; v = v[np.abs(v - peak) < 0.45]
        if len(v) >= 20:
            bs.append(a.lane_m / np.median(v))
    print(f"\n  ⭐ |m_R - m_L| = {dm:.4f}  ->  CAMERA HEIGHT h = {a.lane_m} / {dm:.4f} "
          f"= {h_hat:.3f} m")
    if bs:
        print(f"     frame-cluster bootstrap 95% CI "
              f"[{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}] m")
    print(f"     (withdrawn §50 value was 1.586 m, from an unmasked detector)")
    yaw = np.rad2deg(np.arctan((UV[sel] - 960.0) / a.fx))
    print(f"\n  free cross-check, camera yaw from u_vp: {np.median(yaw):+.2f} deg"
          f"   robust sd {mad(yaw):.2f}   (rendered at {a.yaw:+.2f})")

    # ⛔ NO HORIZON SCAN HERE, deliberately. Re-scaling |m_R - m_L| at a trial v_h
    # measures nothing: the separation is a property of the fit that was already solved
    # AT the adopted horizon, so the "scan" would just re-divide one number and report
    # its own arithmetic. A real horizon measurement needs the pair RE-SOLVED at each
    # trial v_h, which is a separate run. Not done, not reported.
    print(f"\n  (no horizon scan: re-scaling a separation fitted at one v_h measures")
    print(f"   nothing — the pair must be re-solved at each trial horizon. Separate run.)")

    if a.json:
        a.json.write_text(json.dumps(dict(
            n_frames=len(pick), n_pairs=int(len(DM)), frac=len(DM) / len(pick),
            peak=float(peak), dm=dm, height_m=float(h_hat),
            ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None,
            yaw_from_uvp=float(np.median(yaw)), lane_m=a.lane_m), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
