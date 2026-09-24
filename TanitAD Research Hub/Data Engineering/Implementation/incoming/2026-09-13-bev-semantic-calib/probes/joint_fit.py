#!/usr/bin/env python3
"""Fit yaw, horizon, f*h and lateral TOGETHER against the paint.

WHY, AND WHY EVERY EARLIER FIT LEFT AN ERROR BEHIND
----------------------------------------------------
Every parameter in this session was fitted ALONE and then frozen while a coupled one
moved underneath it:

    f fitted, then h moved            -> f stale
    yaw fitted at horizon 465,
        then the horizon moved to 485 -> yaw stale by 1.0 deg
    yaw refitted alone at 485         -> still 0.8 deg of residual angle on the
                                         delivered video

They are coupled through one equation. A road point at range ``x``, lateral ``y``
lands at::

    v = v_h + f*h / x          u = cx + f * (y - lat) / x

so yaw rotates the lane, ``v_h`` slides the range scale, ``f*h`` stretches it, and
``lat`` shifts it — and a change in any one can be partly absorbed by the others.
Fitting them one at a time walks the valley instead of finding its floor.

THE OBJECTIVE. The SOLID left line only (the dashed right one yields 8 samples a
frame against 19, and past 10 m the associator locks onto the shoulder). Straight
frames only, so road curvature cannot masquerade as yaw. Residual = metres from the
projected lane edge to the nearest paint, Cauchy-weighted so a mis-association
cannot drag the fit.

⚠️ ``h`` is NOT free. Nothing in this recording measures it (every lateral estimator
turned out to confirm whatever height it was handed), so it stays pinned by the VW
Caddy mount geometry and the lens FOV, and ``f = f*h / h`` follows.
"""
import sys, argparse, json, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bev_calib as BC, lag_scale as LS, run_real as RR

RANGES = [8., 10., 12., 15., 18., 22.]


def load_inks(n_frames, max_yawrate=0.6):
    from trajlib import lane_calib as LC
    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > 8.0
              and (fdir / f"{int(r['frame']):06d}.jpg").exists()]
    keep = []
    for r in usable:
        t = np.asarray(r["t"], float); yw = np.asarray(r["yaw"], float)
        m = np.abs(t) < 0.4
        if m.sum() >= 3 and abs(np.rad2deg(np.polyfit(t[m], yw[m], 1)[0])) <= max_yawrate:
            keep.append(r)
    out = []
    for pi in np.linspace(0, len(keep) - 1, n_frames).astype(int):
        f = int(keep[pi]["frame"])
        img = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        u, v = LC._ridge_points(img, int(0.50 * H), int(0.95 * H))
        if len(u) < 120:
            continue
        ink = np.zeros((H, W), np.uint8)
        ink[np.clip(v, 0, H - 1).astype(int), np.clip(u, 0, W - 1).astype(int)] = 1
        out.append(ink)
    return out


def residuals(inks, P, half_lane, assoc_m=1.0):
    pts = np.array([[x, half_lane] for x in RANGES])
    uv = BC.project_ground(pts, P)
    res = []
    for ink in inks:
        H, W = ink.shape
        for (x, _), (cu, cv_) in zip(pts, uv):
            if not np.isfinite(cu) or not np.isfinite(cv_):
                continue
            r, c = int(round(cv_)), int(round(cu))
            if not (0 <= r < H):
                continue
            mpp = x / P["fx"]
            hw = int(round(assoc_m / mpp))
            lo, hi = max(0, c - hw), min(W, c + hw + 1)
            if hi <= lo:
                continue
            band = ink[max(0, r - 2):r + 3, lo:hi].sum(axis=0)
            if band.max() <= 0:
                continue
            cols = np.flatnonzero(band >= band.max())
            res.append(((float(cols.mean()) + lo - c) * mpp, x))
    return res


def cost(inks, fh, vh, yawd, lat, h, half_lane, n_ref, c_m=0.20):
    """⚠️ EQUAL WEIGHT PER RANGE, on the median residual at each range.

    Summing a robust loss over every point looks natural and is wrong here: the
    sample collapses from 141 points at 8 m to 19 at 22 m, so the near field
    outvotes the far field 7:1 and the fit buys a lower level by accepting a worse
    ANGLE. MEASURED: that objective cut the cost 15% while the residual slope went
    from +0.0038 to +0.0270 m/m (0.22 deg -> 1.55 deg), i.e. it made exactly the
    defect being chased worse.

    What "the corridor is parallel to the lane" means is that the residual is the
    SAME at every range. So take one median per range and weight the ranges
    equally; far ranges then carry the angular information they actually hold.
    """
    P = dict(RR.NOMINAL)
    P.update(fx=fh / h, height=h, lateral=lat, yaw=np.deg2rad(yawd))
    try:
        P["pitch"] = LS.pitch_for_horizon(P, vh)
    except Exception:
        return 1e9
    r = residuals(inks, P, half_lane)
    if not r:
        return 1e9
    tot, used = 0.0, 0
    for x in RANGES:
        v = [d for d, xx in r if xx == x]
        if len(v) < 10:
            tot += 1.0                       # a dropped range costs, so the fit
            continue                         # cannot buy flatness by losing data
        tot += float(np.median(v)) ** 2
        used += 1
    if used < 4:
        return 1e9
    return tot / len(RANGES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--height", type=float, default=1.622)
    ap.add_argument("--half-lane", type=float, default=1.75)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()
    inks = load_inks(a.frames)
    n_ref = len(inks) * len(RANGES)
    print(f"{len(inks)} straight frames, h pinned at {a.height} m, "
          f"{len(RANGES)} ranges {RANGES[0]:.0f}-{RANGES[-1]:.0f} m\n")

    from scipy.optimize import minimize
    def neg(z):
        fh, vh, yawd, lat = z[0] * 1000, z[1] * 100, z[2], z[3]
        if not (1800 < fh < 3600 and 420 < vh < 540 and -11 < yawd < -3 and -1.2 < lat < 0.6):
            return 1e9
        return cost(inks, fh, vh, yawd, lat, a.height, a.half_lane, n_ref)

    start = (2702.6, 485.0, -7.80, -0.126)
    c0 = neg([start[0] / 1000, start[1] / 100, start[2], start[3]])
    print(f"  as rendered   f*h {start[0]:.0f}  horizon {start[1]:.0f}  "
          f"yaw {start[2]:+.2f}  lateral {start[3]:+.3f}   cost {c0:.5f}")
    best = None
    for fh0 in (2450., 2700., 2900.):
        for vh0 in (460., 485., 505.):
            for y0 in (-8.4, -7.8, -7.0):
                r = minimize(neg, [fh0 / 1000, vh0 / 100, y0, -0.126],
                             method="Nelder-Mead",
                             options=dict(maxfev=420, xatol=1e-4, fatol=1e-7))
                if best is None or r.fun < best.fun:
                    best = r
    fh, vh, yawd, lat = best.x[0] * 1000, best.x[1] * 100, best.x[2], best.x[3]
    print(f"  JOINT FIT     f*h {fh:.0f}  horizon {vh:.1f}  yaw {yawd:+.2f}  "
          f"lateral {lat:+.3f}   cost {best.fun:.5f}  ({100*(1-best.fun/c0):+.0f}%)")
    print(f"                f = f*h/h = {fh/a.height:.0f} px  "
          f"(HFOV {np.rad2deg(2*np.arctan(1920/(2*fh/a.height))):.1f} deg, "
          f"crop {fh/a.height/1442:.2f}x)")

    # the residual profile at the fit — flat is what we want
    P = dict(RR.NOMINAL)
    P.update(fx=fh / a.height, height=a.height, lateral=lat, yaw=np.deg2rad(yawd))
    P["pitch"] = LS.pitch_for_horizon(P, vh)
    r = residuals(inks, P, a.half_lane)
    print(f"\n  residual by range at the joint fit:")
    xs, sl = [], []
    for x in RANGES:
        v = [d for d, xx in r if xx == x]
        if len(v) >= 10:
            print(f"    {x:5.0f} m  {np.median(v):+7.3f} m  (n {len(v):4d})")
            xs.append(x); sl.append(np.median(v))
    if len(xs) >= 3:
        s = np.polyfit(xs, sl, 1)[0]
        print(f"    slope {s:+.4f} m/m = {np.rad2deg(np.arctan(abs(s))):.2f} deg "
              f"-> {40*abs(s):.2f} m at 40 m")
    if a.json:
        a.json.write_text(json.dumps(dict(fh=fh, horizon=vh, yaw_deg=yawd, lateral=lat,
                                          height=a.height, fx=fh / a.height,
                                          cost0=c0, cost=best.fun), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
