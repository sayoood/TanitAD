#!/usr/bin/env python3
"""Find the lane as a PAIR, not as two lines that happen to be near each other.

WHY THE PREVIOUS DETECTOR FAILED (§77). It fitted lines independently by RANSAC, took
the strongest on each side of the path, and then checked whether they were parallel.
On this recording the strongest thing to the right is frequently the shoulder, the
kerb or the barrier, so the check threw away 94 % of frames — and when it did not
(the looser gate) it reported a "lane" of 6.41 m at 10 m growing to 18.28 m at 40 m,
and a corridor placement that swung 2.15 m and changed sign on the gate choice.

THE FIX IS TO USE THE CONSTRAINT IN THE SEARCH INSTEAD OF AFTER IT. Every
road-parallel line passes through the vanishing point, so with ``q = v - v_h``::

    u(v)  =  u_vp  +  m · q          and     m = y / h

Two boundaries of one lane therefore share ``u_vp`` and differ in slope by exactly
``m_R - m_L = w / h``. That is **two free parameters, not four** — and a pair that is
not parallel, or not one lane apart, cannot be expressed at all.

So: for each candidate ``u_vp`` every ridge point votes for a slope ``m = (u - u_vp)/q``;
a lane shows as two peaks in that histogram at the right separation. The winner is the
``(u_vp, m_L, m_R)`` with the most support. It is a Hough accumulator over the pair,
and it cannot return a non-parallel "lane" by construction.

WHAT IT YIELDS, with no calibration beyond the horizon row:

    lane width                w = h · |m_R - m_L|
    camera offset from centre     = h · (m_L + m_R) / 2      (h cancels against w)
    camera yaw                    = atan((u_vp - cx) / f)    — a free cross-check
                                    against the -5.30 deg measured in §55
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
from overlay_far import ridge_cols, ridge_width_px                     # noqa: E402


def lane_pair(pts, horizon, u_grid, m_grid, sep_lo, sep_hi, tol_px=3.0, min_each=12):
    """(u_vp, m_L, m_R, n_L, n_R) for the best-supported parallel pair, or None."""
    v = pts[:, 0]
    q = v - horizon
    good = q > 20.0
    if good.sum() < 2 * min_each:
        return None
    u = pts[good, 1]
    q = q[good]
    best = None
    dm = m_grid[1] - m_grid[0]
    for uv in u_grid:
        m = (u - uv) / q
        # a point supports slope m0 if |m - m0| * q < tol_px, i.e. the tolerance in
        # SLOPE is range-dependent -- which is right: a 3 px error at q = 50 is a much
        # bigger slope error than at q = 250.
        hist, _ = np.histogram(m, bins=np.append(m_grid - 0.5 * dm, m_grid[-1] + 0.5 * dm))
        if hist.max() < min_each:
            continue
        k = int(round(sep_lo / dm))
        k2 = int(round(sep_hi / dm))
        for off in range(k, k2 + 1):
            a = hist[:-off] if off else hist
            b = hist[off:] if off else hist
            s = a + b
            i = int(np.argmax(s))
            if a[i] < min_each or b[i] < min_each:
                continue
            score = int(s[i])
            if best is None or score > best[0]:
                best = (score, float(uv), float(m_grid[i]), float(m_grid[i + off]),
                        int(a[i]), int(b[i]))
    if best is None:
        return None
    # refine: least squares on the supporting points, pair constraint kept
    _, uv, mL, mR, nL, nR = best
    for _ in range(3):
        mm = (u - uv) / q
        inL = np.abs(mm - mL) * q < tol_px
        inR = np.abs(mm - mR) * q < tol_px
        if inL.sum() < min_each or inR.sum() < min_each:
            break
        # u = uv + m*q  ->  solve for uv, mL, mR jointly (linear)
        A = np.zeros((int(inL.sum() + inR.sum()), 3))
        rhs = np.concatenate([u[inL], u[inR]])
        A[:, 0] = 1.0
        A[: int(inL.sum()), 1] = q[inL]
        A[int(inL.sum()):, 2] = q[inR]
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        uv, mL, mR = float(sol[0]), float(sol[1]), float(sol[2])
        nL, nR = int(inL.sum()), int(inR.sum())
    return uv, mL, mR, nL, nR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx", type=float, default=1533.0)
    ap.add_argument("--height", type=float, default=1.586)
    ap.add_argument("--horizon", type=float, default=448.4)
    ap.add_argument("--x-lo", type=float, default=10.0)
    ap.add_argument("--x-hi", type=float, default=40.0)
    ap.add_argument("--lane-lo", type=float, default=2.6, help="admissible lane width, m")
    ap.add_argument("--lane-hi", type=float, default=4.2)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    fh = a.fx * a.height
    rows = np.arange(a.horizon + fh / a.x_hi, a.horizon + fh / a.x_lo, 1.0)
    u_grid = np.arange(560.0, 1160.0, 3.0)
    m_grid = np.arange(-4.0, 4.0001, 0.01)
    sep_lo, sep_hi = a.lane_lo / a.height, a.lane_hi / a.height

    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    fdir = RR.RUN / "frames"
    avail = [f for f in sorted(int(p.stem) for p in fdir.glob("*.jpg"))
             if (r := recs.get(f)) and r["complete"] and r["speed_ms"] > 8]
    pick = [avail[i] for i in np.linspace(0, len(avail) - 1, min(a.n, len(avail))).astype(int)]
    print(f"{len(pick)} frames; pair search over u_vp {u_grid[0]:.0f}-{u_grid[-1]:.0f} px "
          f"and a lane of {a.lane_lo}-{a.lane_hi} m\n")

    UV, W, OFF, NL, NR, T = [], [], [], [], [], []
    for f in pick:
        g = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        pts = []
        for v in rows:
            x = fh / max(v - a.horizon, 1e-3)
            for c in ridge_cols(g[int(round(v))], ridge_width_px(x, a.fx)):
                pts.append((float(v), float(c)))
        if len(pts) < 50:
            continue
        r = lane_pair(np.asarray(pts, float), a.horizon, u_grid, m_grid, sep_lo, sep_hi)
        if r is None:
            continue
        uv, mL, mR, nL, nR = r
        w = a.height * abs(mR - mL)
        if not (a.lane_lo <= w <= a.lane_hi):
            continue
        UV.append(uv); W.append(w); NL.append(nL); NR.append(nR)
        OFF.append(a.height * 0.5 * (mL + mR))
        T.append(float(recs[f]["t_session_s"]))

    UV = np.asarray(UV); W = np.asarray(W); OFF = np.asarray(OFF); T = np.asarray(T)
    mad = lambda z: 1.4826 * np.median(np.abs(z - np.median(z)))
    print(f"lane pair found in {len(UV)}/{len(pick)} frames "
          f"({100*len(UV)/max(len(pick),1):.0f} %), median support {np.median(NL):.0f} "
          f"left / {np.median(NR):.0f} right")
    if len(UV) < 40:
        print("still too few — the pair constraint did not rescue it")
        return 1
    print(f"\n  lane width        {np.median(W):.2f} m   robust sd {mad(W):.2f}"
          f"   10-90% [{np.percentile(W,10):.2f}, {np.percentile(W,90):.2f}]")
    print(f"  camera offset     {np.median(OFF):+.2f} m from lane centre"
          f"   robust sd {mad(OFF):.2f}")
    yaw = np.rad2deg(np.arctan((UV - 960.0) / a.fx))
    print(f"  u_vp              {np.median(UV):.0f} px  ->  camera yaw "
          f"{np.median(yaw):+.2f} deg   robust sd {mad(yaw):.2f}")
    print(f"     (§55 measured the yaw at -5.30 deg from the LEFT line alone —"
          f" this is a free cross-check)")
    rng = np.random.default_rng(0)
    bs = [np.median(OFF[rng.integers(0, len(OFF), len(OFF))]) for _ in range(500)]
    print(f"\n  camera offset 95% CI [{np.percentile(bs,2.5):+.2f}, "
          f"{np.percentile(bs,97.5):+.2f}] m")
    if a.json:
        a.json.write_text(json.dumps(dict(
            n=len(UV), frac=len(UV) / len(pick), lane_w=float(np.median(W)),
            lane_w_sd=float(mad(W)), offset=float(np.median(OFF)),
            offset_sd=float(mad(OFF)),
            offset_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            u_vp=float(np.median(UV)), yaw_deg=float(np.median(yaw)),
            series=[[float(t), float(o), float(w)] for t, o, w in zip(T, OFF, W)]),
            indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
