#!/usr/bin/env python3
"""Video/sensor sync, measured by racing IMAGE rotation against the TRAJECTORY.

WHY IT MATTERS HERE, AND WHY NOTHING ELSE IN THIS DIRECTORY WOULD HAVE CAUGHT IT
--------------------------------------------------------------------------------
Every calibration instrument here projects a STRAIGHT line at +-1.75 m in the
vehicle frame. The rendered corridor does not: it follows the FUTURE TRAJECTORY.
So a timing error between video and sensors moves the far end of the corridor --
Sayed's *"cuts road markings at large distances"* -- while leaving every residual
test in this directory reading zero. The two are blind to each other by
construction.

Scale of the effect: the corridor at range ``x`` shows where the car will be at
``T = x/v`` seconds. Mis-time by ``dt`` and the lateral error is about
``kappa v^2 T dt``. On this clip (``|kappa|`` up to 0.0034 1/m, v 22 m/s), at 40 m
(T = 1.8 s) a 0.1 s error is **0.30 m** -- and it is PROPORTIONAL TO CURVATURE, so
it appears on bends and vanishes on straights. That signature is the test.

WHY THE TWO EARLIER ATTEMPTS FAILED
------------------------------------
* ``scan_time_offset.py`` scanned BEV agreement against a pose offset: FLAT,
  max/min 1.05x. It could not resolve the offset and said so.
* image yaw rate vs the gyro: no distinct peak, best |r| 0.312 at +4.40 s against
  the pipeline's +3.30 s, and r NEGATIVE at the pipeline's own lag.
* ``eis_vs_gyro.py`` tracked "distant" content in rows 0.12-0.38 H and got |r| < 0.04
  -- because on this road those rows hold NEAR hillside and roadside trees, so
  translation dominated (15 px rms measured against a 2.4 px rotation prediction).

This one fixes that last point: distant content is sought in the CENTRAL columns
only, where the road runs to the horizon and the hillsides are out of frame. And it
correlates against the TRAJECTORY's yaw rate (already fused and smooth) rather than
the raw gyro, because the trajectory is what the corridor is actually drawn from --
so the lag this finds is the lag that matters.
"""
import sys, argparse, json, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import run_real as RR

LK = dict(winSize=(31, 31), maxLevel=5,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=900, help="CONSECUTIVE frames")
    ap.add_argument("--start-frac", type=float, default=0.10)
    ap.add_argument("--fx", type=float, default=1666.0)
    ap.add_argument("--rows", type=float, nargs=2, default=[0.30, 0.42],
                    help="above the horizon: distant content")
    ap.add_argument("--cols", type=float, nargs=2, default=[0.34, 0.66],
                    help="CENTRAL only — the edges hold near hillside")
    ap.add_argument("--max-lag", type=float, default=1.0, help="seconds")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    by = {int(r["frame"]): r for r in recs}
    usable = sorted(k for k, r in by.items()
                    if r["complete"] and r["speed_ms"] > 8.0
                    and (fdir / f"{k:06d}.jpg").exists())
    i0 = int(a.start_frac * len(usable))
    sel = usable[i0:i0 + a.frames]
    fps = 29.922

    img_rate, traj_rate, tt = [], [], []
    prev = None
    for f in sel:
        p = fdir / f"{f:06d}.jpg"
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            prev = None
            continue
        if prev is not None and prev[0] == f - 1:
            H, W = g.shape
            mask = np.zeros((H, W), np.uint8)
            mask[int(a.rows[0] * H):int(a.rows[1] * H),
                 int(a.cols[0] * W):int(a.cols[1] * W)] = 255
            p0 = cv2.goodFeaturesToTrack(prev[1], maxCorners=400, qualityLevel=0.01,
                                         minDistance=8, mask=mask, blockSize=7)
            if p0 is not None and len(p0) >= 25:
                p1, s1, _ = cv2.calcOpticalFlowPyrLK(prev[1], g, p0, None, **LK)
                pb, s0, _ = cv2.calcOpticalFlowPyrLK(g, prev[1], p1, None, **LK)
                ok = (s1.ravel() == 1) & (s0.ravel() == 1)
                ok &= np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 0.8
                if ok.sum() >= 15:
                    du = float(np.median((p1.reshape(-1, 2) - p0.reshape(-1, 2))[ok, 0]))
                    # distant content: du ~= -f * omega_yaw * dt
                    img_rate.append(-du * fps / a.fx)
                    r = by[f]
                    t = np.asarray(r["t"], float); yw = np.asarray(r["yaw"], float)
                    m = np.abs(t) < 0.35
                    traj_rate.append(float(np.polyfit(t[m], yw[m], 1)[0]) if m.sum() >= 3
                                     else np.nan)
                    tt.append(f / fps)
        prev = (f, g)

    img_rate = np.asarray(img_rate); traj_rate = np.asarray(traj_rate)
    m = np.isfinite(img_rate) & np.isfinite(traj_rate)
    img_rate, traj_rate = img_rate[m], traj_rate[m]
    if len(img_rate) < 120:
        print(f"only {len(img_rate)} paired samples — cannot decide")
        return 1
    # ⚠️ BAND-PASS BOTH, or the lag is not identifiable. MEASURED: correlating the
    # raw rates gave r ramping monotonically from -0.284 at -0.97 s to -0.587 at
    # +0.97 s, still rising at the edge (peak/p95 = 1.01x) -- a boundary, not a peak.
    # Both series share a slow drift (the road's overall curvature), and a shared
    # trend correlates at every lag, so the low frequencies carry no timing
    # information at all and drown the frequencies that do. The pipeline's own
    # timesync band-passes 0.15-4.0 Hz for exactly this reason.
    def bandpass(x, fs, lo_s=3.0, hi_s=0.20):
        x = x - x.mean()
        klo = max(3, int(round(lo_s * fs)) | 1)
        khi = max(3, int(round(hi_s * fs)) | 1)
        slow = np.convolve(x, np.ones(klo) / klo, mode="same")
        y = x - slow                                  # high-pass: kill the shared trend
        return np.convolve(y, np.ones(khi) / khi, mode="same")   # low-pass: kill noise
    a_ = bandpass(img_rate, fps)[30:-30]
    b_ = bandpass(traj_rate, fps)[30:-30]
    print(f"{len(a_)} paired samples over {len(a_)/fps:.1f} s of consecutive frames")
    print(f"  image  yaw rate rms {np.rad2deg(np.sqrt((a_**2).mean())):.3f} deg/s")
    print(f"  traj   yaw rate rms {np.rad2deg(np.sqrt((b_**2).mean())):.3f} deg/s\n")
    L = int(a.max_lag * fps)
    print(f"  {'lag (s)':>9} {'r':>8}")
    best = None
    rows = []
    for lag in range(-L, L + 1):
        if lag >= 0:
            x, y = a_[lag:], b_[:len(b_) - lag] if lag else b_
        else:
            x, y = a_[:len(a_) + lag], b_[-lag:]
        n = min(len(x), len(y))
        if n < 100:
            continue
        r = float(np.corrcoef(x[:n], y[:n])[0, 1])
        rows.append((lag / fps, r))
        if best is None or abs(r) > abs(best[1]):
            best = (lag / fps, r)
    for lag, r in rows[::max(1, len(rows) // 14)]:
        mark = "  <== best" if abs(r - best[1]) < 1e-12 else ""
        print(f"  {lag:+9.3f} {r:+8.3f}{mark}")
    print(f"\n  best correlation r = {best[1]:+.3f} at lag {best[0]:+.3f} s")
    rr = np.array([r for _, r in rows])
    print(f"  |r| median {np.median(np.abs(rr)):.3f}  p95 {np.percentile(np.abs(rr),95):.3f}"
          f"   peak/p95 = {abs(best[1])/max(np.percentile(np.abs(rr),95),1e-9):.2f}x")
    # a lag read off a monotone ramp is the search bound, not a measurement
    at_edge = abs(abs(best[0]) - a.max_lag) < 1.5 / fps
    contrast = abs(best[1]) / max(np.percentile(np.abs(rr), 95), 1e-9)
    if at_edge or contrast < 1.15:
        print(f"\n  ⛔ NOT A PEAK: |r| is {'still rising at the search bound' if at_edge else 'flat across lags'}"
              f" (peak/p95 = {contrast:.2f}x).")
        print("     The lag is the bound, not a measurement. Reporting an offset from")
        print("     this would be the same boundary-running error as the focal scan.")
    elif abs(best[1]) < 0.35:
        print("\n  ⚠️ too weak to decide — this channel cannot resolve the sync.")
    elif abs(best[0]) <= 1.5 / fps:
        print(f"\n  => the image and the trajectory are IN SYNC to within one frame "
              f"({1000/fps:.0f} ms). A timing error is NOT what bends the far corridor.")
    else:
        err = abs(best[0])
        print(f"\n  => OFFSET of {best[0]:+.3f} s ({best[0]*fps:+.1f} frames). At 40 m "
              f"(T=1.8 s) and kappa 0.0034 1/m that is {0.0034*22**2*1.8*err:.2f} m of")
        print(f"     lateral error at the far end of the corridor — and it scales with "
              f"curvature, so it shows on bends and not on straights.")
    if a.json:
        a.json.write_text(json.dumps(dict(n=int(len(a_)), best_lag_s=best[0],
                                          best_r=best[1],
                                          lags=[l for l, _ in rows],
                                          r=[r for _, r in rows]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
