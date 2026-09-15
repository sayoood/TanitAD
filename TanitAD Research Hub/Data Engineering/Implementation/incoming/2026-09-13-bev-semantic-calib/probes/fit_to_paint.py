#!/usr/bin/env python3
"""Fit (height, lateral, yaw) so the projected lane lands ON the paint.

`f*h` is HELD at the flow measurement (2444.6 px*m) and the horizon at 465 px --
those came from ego motion and are not up for renegotiation here. `f` is not free
either: it follows as `f*h / h`. What is free is exactly what motion cannot see:

    height   -> the lateral SCALE      (lane too wide/narrow)
    lateral  -> a constant shift       (corridor off-centre at every range)
    yaw      -> a shift growing with range

⚠️ The fitted `lateral` is NOT purely a mount offset. It absorbs the mean lane
position the driver actually held, because the reference is "the lane centre,
averaged over frames". For rendering that is what you want; as a mount
measurement it is an upper bound on the mount's own offset.

⚠️ The 3.50 m lane width is an INPUT. Every height here scales with it: at 3.00 m
the fitted height drops by the same 14%.
"""
import sys, argparse, json, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bev_calib as BC, lag_scale as LS, run_real as RR


def collect(n_frames, rows=(0.50, 0.95), max_yawrate=None):
    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > 8.0
              and (fdir / f"{int(r['frame']):06d}.jpg").exists()]
    # ⚠️ CURVATURE IS A YAW CONFOUND. On a curve the lane centre genuinely moves
    # laterally with range, which is indistinguishable from a mount-yaw error in a
    # residual-vs-range fit. MEASURED: this clip reaches |k| = 0.0034 1/m (radius
    # 294 m), which displaces the lane by 15^2/(2*294) = 0.38 m over the usable
    # 8-15 m window -- the same size as the residual being fitted. Straight frames
    # only, when a yaw is being read off.
    if max_yawrate is not None:
        keep = []
        for r in usable:
            t = np.asarray(r["t"], float); yw = np.asarray(r["yaw"], float)
            m = np.abs(t) < 0.35
            if m.sum() < 3:
                continue
            rate = np.polyfit(t[m], yw[m], 1)[0]
            if abs(np.rad2deg(rate)) <= max_yawrate:
                keep.append(r)
        usable = keep
    from trajlib import lane_calib as LC
    out = []
    for pi in np.linspace(0, len(usable) - 1, n_frames).astype(int):
        f = int(usable[pi]["frame"])
        img = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape[:2]
        u, v = LC._ridge_points(img, int(rows[0] * H), int(rows[1] * H))
        if len(u) < 80:
            continue
        ink = np.zeros((H, W), np.uint8)
        ink[np.clip(v, 0, H - 1).astype(int), np.clip(u, 0, W - 1).astype(int)] = 1
        out.append(ink)
    return out


def cost(inks, fh, vh, h, lat, yaw, ranges, half_lane, assoc_m=1.1, c_m=0.25):
    P = dict(RR.NOMINAL)
    P.update(height=h, fx=fh / h, lateral=lat, yaw=yaw)
    try:
        P["pitch"] = LS.pitch_for_horizon(P, vh)
    except Exception:
        return 1e9, 0
    pts = np.array([[x, s * half_lane] for x in ranges for s in (+1, -1)])
    uv = BC.project_ground(pts, P)
    tot, n = 0.0, 0
    for ink in inks:
        Hh, Ww = ink.shape
        for (x, _), (cu, cv_) in zip(pts, uv):
            if not np.isfinite(cu) or not np.isfinite(cv_):
                continue
            r, c = int(round(cv_)), int(round(cu))
            if not (0 <= r < Hh):
                continue
            mpp = x / P["fx"]
            hw = int(round(assoc_m / mpp))
            lo, hi = max(0, c - hw), min(Ww, c + hw + 1)
            if hi <= lo:
                continue
            band = ink[max(0, r - 2):r + 3, lo:hi].sum(axis=0)
            if band.max() <= 0:
                continue
            cols = np.flatnonzero(band >= band.max())
            d = (float(cols.mean()) + lo - c) * mpp
            tot += np.log1p((d / c_m) ** 2)      # Cauchy: a wrong line must not pull
            n += 1
    return (tot / max(n, 1) if n else 1e9), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--fh", type=float, default=2444.6)
    ap.add_argument("--horizon", type=float, default=465.0)
    ap.add_argument("--half-lane", type=float, default=1.75)
    ap.add_argument("--max-yawrate", type=float, default=None)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()
    inks = collect(a.frames, max_yawrate=a.max_yawrate)
    ranges = [8., 9., 10., 11., 12.]   # both edges associate >60% only here
    print(f"{len(inks)} frames of ridge ink;  f*h held at {a.fh}, horizon at {a.horizon}")
    print(f"lane width assumed {2*a.half_lane:.2f} m  (an INPUT — every height scales with it)\n")

    from scipy.optimize import minimize
    def neg(z):
        h, lat, yawd = float(z[0]), float(z[1]), float(z[2])
        if not (0.8 <= h <= 2.4 and -1.5 <= lat <= 1.5 and -15 <= yawd <= 1):
            return 1e9
        return cost(inks, a.fh, a.horizon, h, lat, np.deg2rad(yawd), ranges, a.half_lane)[0]

    best = None
    for h0 in (1.30, 1.50, 1.70):
        for y0 in (-7.01, -6.2):
            r = minimize(neg, [h0, -0.12, y0], method="Nelder-Mead",
                         options=dict(maxfev=260, xatol=1e-3, fatol=1e-6))
            if best is None or r.fun < best.fun:
                best = r
    h, lat, yawd = best.x
    c0, n0 = cost(inks, a.fh, a.horizon, 1.70, -0.12, np.deg2rad(-7.01), ranges, a.half_lane)
    c1, n1 = cost(inks, a.fh, a.horizon, h, lat, np.deg2rad(yawd), ranges, a.half_lane)
    print(f"  rendered  h 1.700  lateral -0.120  yaw -7.010   cost {c0:.5f}  (n {n0})")
    print(f"  FITTED    h {h:.3f}  lateral {lat:+.3f}  yaw {yawd:+.3f}   cost {c1:.5f}  (n {n1})")
    print(f"  => f = f*h / h = {a.fh/h:.0f} px   "
          f"(HFOV {np.rad2deg(2*np.arctan(1920/(2*a.fh/h))):.1f} deg)"
          f"{'   INSIDE the 1356-1628 device band' if 1356 <= a.fh/h <= 1628 else '   OUTSIDE the band'}")
    print(f"  cost improvement {100*(1-c1/c0):.1f}%")
    if a.json:
        a.json.write_text(json.dumps(dict(height=float(h), lateral=float(lat),
                                          yaw_deg=float(yawd), fx=float(a.fh/h),
                                          fh=a.fh, horizon=a.horizon,
                                          cost_rendered=float(c0), cost_fitted=float(c1),
                                          n_frames=len(inks), lane_width=2*a.half_lane), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
