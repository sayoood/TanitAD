#!/usr/bin/env python3
"""The ROAD's vanishing point, straight from the pixels — no calibration input.

WHY THIS IS THE ONE INSTRUMENT THAT IS NOT CIRCULAR
---------------------------------------------------
Every estimator in this directory so far takes a calibration in and gives a
calibration out, and three of them turned out to confirm whatever they were handed
(``lane_residual.py`` reported the same lane width at h=1.17 and h=1.427; the
association windows in ``fit_to_paint`` and ``joint_fit`` centre on a prediction).

A vanishing point does not. It is a pure image-space quantity: fit straight lines to
road-parallel paint and structure, intersect them. NO focal length, NO height, NO
horizon, NO ego pose goes in. What comes out is:

    v_vp  = the HORIZON ROW, measured.        (flow said 427-448; paint-consistency
                                               said 465-485; the pipeline's lane-VP
                                               said 523.4 — they cannot all be right)
    u_vp  = cx + f*tan(yaw), so the CAMERA YAW once f is chosen.

WHY POOLING ACROSS FRAMES IS VALID (and why the earlier per-frame probe failed).
``per_frame_vp.py`` demanded both lane lines in one frame and got a VP in 3 frames of
260, because the right line is dashed and yields 8 points. But a road-parallel line at
ground offset ``y0`` images as::

    col = cx + f*tan(theta) + y0*(v - v_h)/h

so at ``v = v_h`` the column is ``cx + f*tan(theta)`` — INDEPENDENT OF y0. Every
road-parallel line in every frame passes through the same point regardless of where
the car sits in the lane. So the left solid line ALONE, pooled over frames, pins the
VP; and the barrier, the shoulder edge and the far dashed segments all reinforce it
instead of being nuisances.

⚠️ EIS is on (confirmed by Sayed). Electronic stabilisation rotates the frame, which
MOVES the vanishing point frame to frame. The per-frame scatter reported below is the
direct measurement of how much — and therefore of how well any FIXED calibration can
ever do on this recording. That is a number the programme does not yet have.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import run_real as RR                                               # noqa: E402


def road_segments(gray, r0, r1, min_len=55, max_gap=14):
    """Long straight bright-ridge segments plausibly parallel to the road.

    The slope filter is physical, not cosmetic: a road-parallel line has image slope
    ``dcol/drow = y0/h``, so with h ~ 1.6 m and lateral offsets from 0.5 m (the
    corridor edge) to 8 m (the far barrier) the admissible band is ~0.3 to 5.0.
    Horizontal structure (bridges, shadows, the far treeline) sits outside it, and so
    do the near-vertical poles.
    """
    from trajlib import lane_calib as LC
    u, v = LC._ridge_points(gray, r0, r1)
    if len(u) < 60:
        return []
    H, W = gray.shape
    mask = np.zeros((H, W), np.uint8)
    mask[np.clip(v, 0, H - 1).astype(int), np.clip(u, 0, W - 1).astype(int)] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    lines = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=32,
                            minLineLength=min_len, maxLineGap=max_gap)
    if lines is None:
        return []
    out = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dv = float(y2 - y1)
        if abs(dv) < 25:
            continue
        s = abs((x2 - x1) / dv)
        if not (0.25 <= s <= 5.5):
            continue
        out.append((float(x1), float(y1), float(x2), float(y2)))
    return out


def seg_to_line(s):
    """Homogeneous line through a segment, normalised."""
    p1 = np.array([s[0], s[1], 1.0])
    p2 = np.array([s[2], s[3], 1.0])
    L = np.cross(p1, p2)
    n = np.hypot(L[0], L[1])
    return L / n if n > 1e-9 else None


def fit_vp(lines):
    """Cauchy-reweighted least-squares vanishing point (the pipeline's own scheme)."""
    L = np.asarray(lines, float)
    if len(L) < 2:
        return None
    w = np.ones(len(L))
    v = np.array([0.0, 0.0, 1.0])
    for _ in range(14):
        _, _, vt = np.linalg.svd(L * w[:, None])
        v = vt[-1]
        if abs(v[2]) < 1e-12:
            return None
        v = v / v[2]
        r = np.abs(L @ v)
        s = 1.4826 * np.median(r) + 1e-9
        w = 1.0 / np.sqrt(1.0 + (r / (2.5 * s)) ** 2)
    return v[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--max-yawrate", type=float, default=0.6, help="deg/s; straight only")
    ap.add_argument("--cx", type=float, default=960.0)
    ap.add_argument("--focals", type=float, nargs="+", default=[1442., 1600., 1666., 1741., 1900.])
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    recs = RR.load_records(RR.RUN)
    fdir = RR.RUN / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > 8.0
              and (fdir / f"{int(r['frame']):06d}.jpg").exists()]
    keep = []
    for r in usable:
        t = np.asarray(r["t"], float); yw = np.asarray(r["yaw"], float)
        m = np.abs(t) < 0.4
        if m.sum() >= 3 and abs(np.rad2deg(np.polyfit(t[m], yw[m], 1)[0])) <= a.max_yawrate:
            keep.append(r)
    print(f"{len(keep)} straight frames of {len(usable)} usable "
          f"(|yaw rate| <= {a.max_yawrate} deg/s)")

    per_frame, all_lines, nseg = [], [], []
    for pi in np.linspace(0, len(keep) - 1, min(a.frames, len(keep))).astype(int):
        f = int(keep[pi]["frame"])
        g = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        H = g.shape[0]
        segs = road_segments(g, int(0.48 * H), int(0.92 * H))
        L = [seg_to_line(s) for s in segs]
        L = [x for x in L if x is not None]
        nseg.append(len(L))
        all_lines.extend(L)
        if len(L) >= 2:
            vp = fit_vp(L)
            if vp is not None and 0 < vp[0] < 1920 and 200 < vp[1] < 900:
                per_frame.append((f, float(vp[0]), float(vp[1])))

    print(f"road-parallel segments per frame: median {np.median(nseg):.0f}, "
          f"{sum(1 for n in nseg if n >= 2)}/{len(nseg)} frames with >= 2")

    out = {}
    if len(all_lines) >= 20:
        vp = fit_vp(all_lines)
        print(f"\nPOOLED over {len(all_lines)} segments from {len(nseg)} frames")
        print(f"  vanishing point  u = {vp[0]:.1f}   v = {vp[1]:.1f}")
        # frame-cluster bootstrap: resample FRAMES, not segments
        idx, start = [], 0
        for n in nseg:
            idx.append((start, start + n)); start += n
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(400):
            pick = rng.integers(0, len(idx), len(idx))
            sub = [all_lines[i] for k in pick for i in range(*idx[k])]
            if len(sub) < 20:
                continue
            q = fit_vp(sub)
            if q is not None:
                bs.append(q)
        if bs:
            B = np.asarray(bs)
            lo_u, hi_u = np.percentile(B[:, 0], [2.5, 97.5])
            lo_v, hi_v = np.percentile(B[:, 1], [2.5, 97.5])
            print(f"  frame-cluster bootstrap 95% CI   u [{lo_u:.1f}, {hi_u:.1f}]"
                  f"   v [{lo_v:.1f}, {hi_v:.1f}]   (n={len(bs)})")
            out["ci"] = dict(u=[float(lo_u), float(hi_u)], v=[float(lo_v), float(hi_v)])
        out["pooled"] = dict(u=float(vp[0]), v=float(vp[1]), n_lines=len(all_lines))
        print(f"\n  => HORIZON ROW = {vp[1]:.1f} px   (this is a direct measurement:"
              f" no f, no h, no ego pose entered it)")
        print(f"  => CAMERA YAW, by assumed focal:")
        for f_ in a.focals:
            print(f"       f = {f_:6.0f} px  ->  yaw = "
                  f"{np.rad2deg(np.arctan((vp[0] - a.cx) / f_)):+.2f} deg")
        out["yaw_by_focal"] = {str(f_): float(np.rad2deg(np.arctan((vp[0] - a.cx) / f_)))
                               for f_ in a.focals}

    if len(per_frame) >= 20:
        P = np.asarray([[p[1], p[2]] for p in per_frame])
        mu, mv = np.median(P[:, 0]), np.median(P[:, 1])
        su = 1.4826 * np.median(np.abs(P[:, 0] - mu))
        sv = 1.4826 * np.median(np.abs(P[:, 1] - mv))
        print(f"\nPER-FRAME vanishing point, {len(per_frame)} frames")
        print(f"  u  median {mu:7.1f}   robust sd {su:5.1f} px"
              f"   10-90% [{np.percentile(P[:,0],10):.0f}, {np.percentile(P[:,0],90):.0f}]")
        print(f"  v  median {mv:7.1f}   robust sd {sv:5.1f} px"
              f"   10-90% [{np.percentile(P[:,1],10):.0f}, {np.percentile(P[:,1],90):.0f}]")
        print(f"\n  the u scatter IS the per-frame yaw scatter: at f=1666, "
              f"sd {np.rad2deg(np.arctan(su/1666)):.2f} deg")
        print(f"  => a FIXED calibration carries a lateral error of "
              f"{40*su/1666:.2f} m at 40 m from this alone (1 sd)")
        out["per_frame"] = dict(n=len(per_frame), u_median=float(mu), v_median=float(mv),
                                u_sd=float(su), v_sd=float(sv),
                                frames=[[int(p[0]), p[1], p[2]] for p in per_frame])
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
