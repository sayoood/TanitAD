#!/usr/bin/env python3
"""Where does the MODEL put the lane edge, versus where the PAINT actually is?

The definitive test of a road-plane calibration, and the one that does not depend
on identifying which peak is which. For a set of ranges, project the model's lane
edges (y = +-half_lane) into the image and measure, in metres, how far the nearest
painted marking is. The residual's SHAPE names the defective parameter:

    constant in range            -> lateral offset
    grows in proportion to range -> yaw
    grows with range, same sign
      on both edges, symmetric   -> height (the lateral scale)
    left and right diverge       -> the lane is not the assumed width

Association is bounded to +-`assoc_m` so a residual can never be manufactured by
locking onto the next line over; unassociated samples are reported, not dropped
silently.
"""
import sys, argparse, json, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bev_calib as BC, lag_scale as LS, run_real as RR


def residuals_one(img, P, ranges, half_lane, assoc_m=1.1):
    H, W = img.shape[:2]
    from trajlib import lane_calib as LC
    u, v = LC._ridge_points(img, int(0.50 * H), int(0.95 * H))
    if len(u) < 80:
        return []
    ink = np.zeros((H, W), np.uint8)
    ink[np.clip(v, 0, H - 1).astype(int), np.clip(u, 0, W - 1).astype(int)] = 1
    out = []
    for x in ranges:
        for side, y in (("L", +half_lane), ("R", -half_lane)):   # +y = left
            uv = BC.project_ground(np.array([[x, y]]), P)[0]
            if not np.isfinite(uv).all():
                continue
            r, c = int(round(uv[1])), int(round(uv[0]))
            if not (0 <= r < H):
                continue
            m_per_px = x / P["fx"]                    # local lateral scale
            halfw = int(round(assoc_m / m_per_px))
            lo, hi = max(0, c - halfw), min(W, c + halfw + 1)
            if hi <= lo:
                continue
            band = ink[max(0, r - 2):r + 3, lo:hi].sum(axis=0)
            if band.max() <= 0:
                out.append(dict(x=float(x), side=side, resid_m=None))
                continue
            # centroid of the strongest run, so a thick line does not bias to an edge
            cols = np.flatnonzero(band >= max(1, band.max()))
            du = float(cols.mean() + lo - c)
            out.append(dict(x=float(x), side=side, resid_m=float(du * m_per_px)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--height", type=float, default=1.70)
    ap.add_argument("--fx", type=float, default=1438.0)
    ap.add_argument("--yaw", type=float, default=-7.01)
    ap.add_argument("--horizon", type=float, default=465.0)
    ap.add_argument("--lateral", type=float, default=-0.12)
    ap.add_argument("--half-lane", type=float, default=1.75)
    ap.add_argument("--max-yawrate", type=float, default=None,
                    help="deg/s; keep only near-straight frames when reading a yaw")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    max_yawrate = a.max_yawrate
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
    P = dict(RR.NOMINAL)
    P.update(yaw=np.deg2rad(a.yaw), height=a.height, fx=a.fx, lateral=a.lateral)
    P["pitch"] = LS.pitch_for_horizon(P, a.horizon)
    ranges = [5., 6., 7., 8., 10., 12., 15., 20.]
    print(f"f {a.fx}  h {a.height}  horizon {a.horizon}  yaw {a.yaw}  lateral {a.lateral}"
          f"   half-lane {a.half_lane} m\n")

    acc = {}
    for pi in np.linspace(0, len(usable) - 1, a.frames).astype(int):
        f = int(usable[pi]["frame"])
        img = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for d in residuals_one(img, P, ranges, a.half_lane):
            acc.setdefault((d["x"], d["side"]), []).append(d["resid_m"])

    print(f"{'range':>6}  {'LEFT edge':>26}   {'RIGHT edge':>26}")
    print(f"{'':6}  {'median   n   assoc%':>26}   {'median   n   assoc%':>26}")
    rows = []
    for x in ranges:
        cells = []
        for side in ("L", "R"):
            v = acc.get((x, side), [])
            ok = [q for q in v if q is not None]
            cells.append((np.median(ok) if len(ok) >= 8 else np.nan, len(ok),
                          100 * len(ok) / max(len(v), 1)))
        rows.append((x, cells))
        print(f"{x:6.0f}  {cells[0][0]:+8.3f} m {cells[0][1]:5d} {cells[0][2]:6.0f}%"
              f"   {cells[1][0]:+8.3f} m {cells[1][1]:5d} {cells[1][2]:6.0f}%")

    xs = np.array([r[0] for r in rows])
    L = np.array([r[1][0][0] for r in rows]); R = np.array([r[1][1][0] for r in rows])
    ok = np.isfinite(L) & np.isfinite(R)
    if ok.sum() >= 3:
        mid = 0.5 * (L + R)          # + = model lane sits LEFT of the paint
        halfw = 0.5 * (L - R)        # + = model lane NARROWER than the paint
        sl = np.polyfit(xs[ok], mid[ok], 1)
        print(f"\n  CENTRE offset  = {np.polyval(sl, 0):+.3f} m at x=0, "
              f"slope {sl[0]:+.4f} m/m  ->  residual yaw {np.rad2deg(np.arctan(sl[0])):+.2f} deg")
        print(f"  HALF-WIDTH err = " + "  ".join(f"{h:+.2f}" for h in halfw[ok]) +
              f"   (median {np.median(halfw[ok]):+.3f} m)")
        print(f"  => the painted lane is {2*(a.half_lane - np.median(halfw[ok])):.2f} m wide "
              f"at the assumed height {a.height} m")
        print(f"  => for a TRUE 3.50 m lane, height should be "
              f"{a.height * 3.50 / (2*(a.half_lane - np.median(halfw[ok]))):.3f} m")
        if a.json:
            a.json.write_text(json.dumps(dict(x=xs.tolist(), left=L.tolist(),
                                              right=R.tolist(), params=vars(a)|{}, ), indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
