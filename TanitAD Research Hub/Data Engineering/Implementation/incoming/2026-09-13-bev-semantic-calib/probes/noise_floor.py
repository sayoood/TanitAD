#!/usr/bin/env python3
"""The honest noise floor of the lane-angle metric — two ways that do not share a flaw.

WHY. §68 put the instrument noise at 0.063 deg from an odd/even ROW split, and §73
showed why that is too optimistic: interleaved rows are adjacent samples of the SAME
paint over the SAME span, so they share every systematic the fit has and differ only in
pixel noise. Every residual number in this document is quoted against that floor, so
the floor has to be right before Step 3 is worth running at all.

TWO ESTIMATORS, chosen so that a flaw in one does not appear in the other:

**A — LEFT LINE vs RIGHT LINE.** Both boundaries of the lane are road-parallel, and at
``v = v_h`` the ``y/h`` term vanishes, so BOTH give the same angle: ``u(v_h) = cx +
f·tan(theta)``. Different paint, different pixels, the same quantity, the same full
span. Their difference is pure error. And their CORRELATION decomposes the variance
directly: with a real common signal ``S`` and per-line noise ``N``,

    corr(theta_L, theta_R) = S² / (S² + N²)

so the correlation and the variance together separate "real" from "my ruler" without
assuming either.

**B — THE STRUCTURE FUNCTION.** ``D(tau) = rms(theta(t+tau) − theta(t))`` over lag.
Independent per-frame noise contributes a constant ``sqrt(2)·N`` at every lag, including
the shortest; anything real must grow from zero with lag, because the road and the
camera cannot change in 33 ms. So ``D(tau→0)/sqrt(2)`` is the noise floor, read off
without needing a second line at all.

⚠️ NO CONTINUITY GATE is used here. The gate in the delivered metric constrains each
frame's fit to land near the previous frame's, which would suppress exactly the
frame-to-frame variation being measured and drive the floor artificially low. Identity
is instead fixed per frame by side-of-path and the physical slope band.
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
import bev_calib as BC, lag_scale as LS, run_real as RR                # noqa: E402
from overlay_far import ridge_cols, ridge_width_px                     # noqa: E402


def ransac_line(P, minin=16, minsep=60, iters=900, seed=0):
    rng = np.random.default_rng(seed)
    n = len(P)
    if n < minin + 6:
        return None, None
    best, bn = None, 0
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        if abs(P[i, 0] - P[j, 0]) < minsep:
            continue
        m = (P[j, 1] - P[i, 1]) / (P[j, 0] - P[i, 0])
        if not (0.2 <= abs(m) <= 4.0):
            continue
        b = P[i, 1] - m * P[i, 0]
        k = int((np.abs(P[:, 1] - (m * P[:, 0] + b)) < 3.0).sum())
        if k > bn:
            bn, best = k, (m, b)
    if best is None or bn < minin:
        return None, None
    m, b = best
    for _ in range(3):
        inl = P[np.abs(P[:, 1] - (m * P[:, 0] + b)) < 3.0]
        if len(inl) < minin or np.ptp(inl[:, 0]) < minsep:
            return None, None
        m, b = np.polyfit(inl[:, 0], inl[:, 1], 1)
    return (float(m), float(b)), np.abs(P[:, 1] - (m * P[:, 0] + b)) < 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx", type=float, default=1533.0)
    ap.add_argument("--height", type=float, default=1.586)
    ap.add_argument("--horizon", type=float, default=448.4)
    ap.add_argument("--yaw", type=float, default=-5.35)
    ap.add_argument("--x-lo", type=float, default=10.0)
    ap.add_argument("--x-hi", type=float, default=40.0)
    ap.add_argument("--max-frames", type=int, default=812)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    P0 = dict(RR.NOMINAL)
    P0.update(fx=a.fx, height=a.height, lateral=-0.126, yaw=np.deg2rad(a.yaw))
    P0["pitch"] = LS.pitch_for_horizon(P0, a.horizon)
    fh = a.fx * a.height
    rows = np.arange(a.horizon + fh / a.x_hi, a.horizon + fh / a.x_lo, 1.0)
    # the ego path's column at a near row, used only to say which side a line is on
    v_ref = a.horizon + fh / 12.0
    u_ref = float(BC.project_ground(np.array([[12.0, 0.0]]), P0)[0][0])

    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    fdir = RR.RUN / "frames"
    avail = sorted(int(p.stem) for p in fdir.glob("*.jpg"))

    def straight(f):
        r = recs.get(f)
        if r is None or not r["complete"] or r["speed_ms"] < 12:
            return False
        t = np.asarray(r["t"], float); yw = np.asarray(r["yaw"], float)
        m = np.abs(t) < 0.4
        return m.sum() >= 3 and abs(np.rad2deg(np.polyfit(t[m], yw[m], 1)[0])) <= 1.2

    runs, cur = [], []
    for f in avail:
        if straight(f):
            cur.append(f)
        else:
            if len(cur) > len(runs):
                runs = cur
            cur = []
    if len(cur) > len(runs):
        runs = cur
    seq = runs[: a.max_frames]
    print(f"longest straight run: {len(runs)} consecutive frames, using {len(seq)}")
    print(f"span {a.x_lo:.0f}-{a.x_hi:.0f} m = rows {rows.min():.0f}-{rows.max():.0f}, "
          f"path column at 12 m = {u_ref:.0f}\n")

    T, TL, TR, NL, NR = [], [], [], [], []
    for f in seq:
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
        P = np.asarray(pts, float)
        lines, rem = [], P
        for k in range(5):
            ln, inl = ransac_line(rem, seed=k + 1)
            if ln is None:
                break
            lines.append((ln, int(inl.sum())))
            rem = rem[~inl]
            if len(rem) < 30:
                break
        left = [(ln, n) for ln, n in lines if ln[0] * v_ref + ln[1] < u_ref]
        right = [(ln, n) for ln, n in lines if ln[0] * v_ref + ln[1] >= u_ref]
        if not left or not right:
            continue
        (ml, bl), nl = max(left, key=lambda z: z[1])
        (mr, br), nr = max(right, key=lambda z: z[1])
        tl = np.rad2deg(np.arctan((ml * a.horizon + bl - 960.0) / a.fx))
        tr = np.rad2deg(np.arctan((mr * a.horizon + br - 960.0) / a.fx))
        if abs(tl) > 20 or abs(tr) > 20:
            continue
        T.append(float(recs[f]["t_session_s"])); TL.append(tl); TR.append(tr)
        NL.append(nl); NR.append(nr)

    T = np.asarray(T); TL = np.asarray(TL); TR = np.asarray(TR)
    print(f"{len(T)} frames with BOTH a left and a right full-span line "
          f"(median inliers {np.median(NL):.0f} / {np.median(NR):.0f})\n")
    if len(T) < 60:
        print("too few — cannot set a floor")
        return 1

    mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))

    # ---- A: left vs right ---------------------------------------------------- #
    d = TL - TR
    sd_d = mad(d)
    N_line = sd_d / np.sqrt(2.0)
    r_lr = float(np.corrcoef(TL, TR)[0, 1])
    sd_l, sd_r = mad(TL), mad(TR)
    print("A — LEFT vs RIGHT LINE, same frame, same full span")
    print(f"    theta_L rms {sd_l:.3f} deg     theta_R rms {sd_r:.3f} deg")
    print(f"    difference  {sd_d:.3f} deg   median offset {np.median(d):+.3f} deg")
    print(f"    correlation {r_lr:+.3f}")
    print(f"    => per-line NOISE  N = {N_line:.3f} deg")
    S2 = 0.5 * (sd_l ** 2 + sd_r ** 2) - N_line ** 2
    S = float(np.sqrt(max(S2, 0.0)))
    print(f"    => REAL common signal S = {S:.3f} deg"
          f"   (corr predicts S²/(S²+N²) = {S**2/(S**2+N_line**2+1e-12):.2f}"
          f" against the measured {max(r_lr,0):.2f})")

    # ---- B: structure function ------------------------------------------------ #
    # ⚠️ ESTIMATOR A IS NOT A NOISE ESTIMATE ON THIS RECORDING, and B is what proves it:
    # if the per-frame noise really were 1.8 deg, two CONSECUTIVE frames would differ by
    # sqrt(2)*1.8 = 2.6 deg. They differ by 0.17 deg. So the left-right disagreement is
    # not per-frame noise -- it is a systematic difference between the two boundaries.
    #
    # And its median part is diagnostic rather than noise. theta = atan((m*v_h + b - cx)/f),
    # so d(theta)/d(v_h) = m/f -- and the two boundaries have OPPOSITE slopes, so a horizon
    # error moves them in OPPOSITE directions. With m_L - m_R = -3.5/h = -2.21 (the slope
    # difference measured in §54), an offset of theta_L - theta_R implies
    #     delta_v_h = (theta_L - theta_R) * f / (m_L - m_R)
    # The left-right offset is therefore a HORIZON measurement, not an error bar.
    print("\nB — STRUCTURE FUNCTION of the LEFT-line angle, D(tau) = rms[theta(t+tau)-theta(t)]")
    print("    independent per-frame noise gives a constant sqrt(2)*N at EVERY lag;")
    print("    anything real must grow from zero, because nothing moves in 33 ms.")
    th = TL
    print(f"    {'lag':>10}{'pairs':>8}{'D(tau)':>10}{'D/sqrt2':>10}")
    out_sf = []
    order = np.argsort(T); Ts, ths = T[order], th[order]
    for lag_s in (0.033, 0.067, 0.10, 0.133, 0.20, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        tol = max(0.016, 0.05 * lag_s)
        dd = []
        j0 = 0
        for i in range(len(Ts)):
            target = Ts[i] + lag_s
            j = np.searchsorted(Ts, target - tol, side="left")
            while j < len(Ts) and Ts[j] <= target + tol:
                dd.append(ths[j] - ths[i]); j += 1
        if len(dd) < 25:
            continue
        D = mad(np.asarray(dd))
        out_sf.append((lag_s, len(dd), D))
        print(f"    {lag_s:>8.3f} s{len(dd):>8}{D:>10.3f}{D/np.sqrt(2):>10.3f} deg")
    N_sf = out_sf[0][2] / np.sqrt(2) if out_sf else float("nan")
    Dsat = max(d_ for _, _, d_ in out_sf) if out_sf else float("nan")
    S_sf = float(np.sqrt(max(Dsat ** 2 - out_sf[0][2] ** 2, 0.0)) / np.sqrt(2)) if out_sf else 0.0
    print(f"\n    floor (shortest lag)  N = {N_sf:.3f} deg")
    print(f"    saturation            D = {Dsat:.3f} deg  ->  real variation S = {S_sf:.3f} deg")

    # ⚠️ theta is in DEGREES here; the small-angle relation needs radians.
    dvh = float(np.deg2rad(np.median(d)) * a.fx / (-3.5 / a.height))
    print(f"\n    left-right offset {np.median(d):+.3f} deg reads as a HORIZON error of"
          f" {dvh:+.1f} px")
    print(f"    i.e. horizon {a.horizon + dvh:.1f} rather than {a.horizon:.1f}"
          f"   (row-flow alone preferred 438)")
    N_line, S = N_sf, S_sf

    print("\n" + "=" * 72)
    print("WHAT THIS MEANS FOR EVERY RESIDUAL QUOTED IN THIS DOCUMENT")
    print(f"  noise floor on ONE lane angle      N = {N_line:.3f} deg"
          f"  = {40*np.tan(np.deg2rad(N_line)):.2f} m at 40 m")
    print(f"  real per-frame variation           S = {S:.3f} deg"
          f"  = {40*np.tan(np.deg2rad(S)):.2f} m at 40 m")
    if S < N_line:
        print("  ⇒ THE RULER IS BIGGER THAN THE SIGNAL. Most of the 'residual spread'")
        print("    reported against the 0.063 deg floor is measurement, not overlay error.")
    else:
        print("  ⇒ the signal exceeds the ruler; the residual is real and worth chasing.")
    if a.json:
        a.json.write_text(json.dumps(dict(
            n=len(T), sd_left=float(sd_l), sd_right=float(sd_r), sd_diff=float(sd_d),
            corr=r_lr, noise_deg=float(N_line), signal_deg=S, horizon_implied=dvh,
            structure=[[float(x), int(n), float(d_)] for x, n, d_ in out_sf]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
