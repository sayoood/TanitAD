#!/usr/bin/env python3
"""Does ONE calibration fit every frame? Measure the horizon and yaw PER FRAME.

Sayed: *"I did not find a config which satisfies all frames."* This decides whether
that is a fitting failure or a property of the recording.

On a straight, planar road the two lane lines meet at the vanishing point, and that
point IS the camera's attitude with respect to the road surface: its ROW is the
horizon (pitch), its COLUMN is the yaw. Both are read per frame, from the image
alone, with no calibration assumed beyond a rough association window.

⚠️ Why not the gravity/orientation sensor. Device pitch relative to GRAVITY swings
7.6 deg peak-to-peak on this clip, but most of that is road GRADE -- and on a grade
the car and the road ahead tilt together, so the camera-to-road geometry is
unchanged. Grade cancels; suspension pitch and crest/sag do not. The vanishing point
measures the residual that actually matters, which the orientation sensor cannot
separate.

⚠️ Straight frames only. A curve has no single vanishing point for the two lines.
"""
import sys, argparse, json, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bev_calib as BC, lag_scale as LS, run_real as RR


def vp_one(img, P, half_lane, x0=8.0, x1=26.0, assoc_m=0.9):
    from trajlib import lane_calib as LC
    H, W = img.shape[:2]
    u, v = LC._ridge_points(img, int(0.50 * H), int(0.80 * H))
    if len(u) < 120:
        return None
    fits = {}
    for side, y in (("L", +half_lane), ("R", -half_lane)):
        pts = []
        for x in np.arange(x0, x1 if side == "L" else x1 + 6.0, 0.4):
            uv = BC.project_ground(np.array([[x, y]]), P)[0]
            if not np.isfinite(uv).all():
                continue
            r, c = uv[1], uv[0]
            hw = assoc_m / (x / P["fx"])
            sel = (np.abs(v - r) <= 2.0) & (np.abs(u - c) <= hw)
            if sel.sum() >= 2:
                pts.append((float(np.median(u[sel])), float(r)))
        # ⚠️ asymmetric on purpose. MEASURED: the SOLID left line yields a median of
        # 19 samples per frame and fits to 1.39 px; the DASHED right line yields 8,
        # so a symmetric >=10 gate rejected EVERY frame. The threshold follows the
        # markings, not a preference for round numbers.
        if len(pts) < (10 if side == "L" else 6):
            return None
        pts = np.asarray(pts)
        # u = a*v + b   (lane lines are near-vertical in the image)
        a_, b_ = np.polyfit(pts[:, 1], pts[:, 0], 1)
        resid = pts[:, 0] - (a_ * pts[:, 1] + b_)
        if np.std(resid) > 4.0:            # not a straight line: curve or junk
            return None
        fits[side] = (a_, b_, len(pts))
    (aL, bL, nL), (aR, bR, nR) = fits["L"], fits["R"]
    if abs(aL - aR) < 1e-6:
        return None
    v_vp = (bR - bL) / (aL - aR)
    u_vp = aL * v_vp + bL
    return float(u_vp), float(v_vp), nL + nR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=260)
    ap.add_argument("--fx", type=float, default=1713.0)
    ap.add_argument("--height", type=float, default=1.427)
    ap.add_argument("--horizon", type=float, default=465.0)
    ap.add_argument("--yaw", type=float, default=-6.80)
    ap.add_argument("--lateral", type=float, default=-0.126)
    ap.add_argument("--half-lane", type=float, default=1.75)
    ap.add_argument("--max-yawrate", type=float, default=0.5)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    P = dict(RR.NOMINAL)
    P.update(fx=a.fx, height=a.height, lateral=a.lateral, yaw=np.deg2rad(a.yaw))
    P["pitch"] = LS.pitch_for_horizon(P, a.horizon)

    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > 8.0
              and (fdir / f"{int(r['frame']):06d}.jpg").exists()]
    keep = []
    for r in usable:
        t = np.asarray(r["t"], float); yw = np.asarray(r["yaw"], float)
        m = np.abs(t) < 0.4
        if m.sum() < 3:
            continue
        if abs(np.rad2deg(np.polyfit(t[m], yw[m], 1)[0])) <= a.max_yawrate:
            keep.append(r)

    rows, cols, fr = [], [], []
    for pi in np.linspace(0, len(keep) - 1, a.frames).astype(int):
        f = int(keep[pi]["frame"])
        img = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        got = vp_one(img, P, a.half_lane)
        if got is None:
            continue
        cols.append(got[0]); rows.append(got[1]); fr.append(f)
    if len(rows) < 25:
        print(f"only {len(rows)} frames yielded a vanishing point — cannot decide")
        return 1
    rows = np.asarray(rows); cols = np.asarray(cols); fr = np.asarray(fr)
    keepm = (rows > 300) & (rows < 700)
    rows, cols, fr = rows[keepm], cols[keepm], fr[keepm]

    yaws = np.rad2deg(np.arctan((cols - P["cx"]) / a.fx))
    print(f"{len(rows)} straight frames with a usable vanishing point "
          f"(of {len(keep)} straight frames)\n")
    print(f"  assumed for the whole clip:  horizon {a.horizon:.1f} px   yaw {a.yaw:+.2f} deg\n")
    for name, v, unit in (("horizon row", rows, "px"), ("implied yaw", yaws, "deg")):
        print(f"  {name}: median {np.median(v):+8.2f} {unit}   "
              f"p10 {np.percentile(v,10):+8.2f}   p90 {np.percentile(v,90):+8.2f}   "
              f"sd {np.std(v):6.2f}")
    spread = np.percentile(rows, 90) - np.percentile(rows, 10)
    print(f"\n  horizon p10-p90 spread = {spread:.1f} px")
    print(f"  a fixed horizon is therefore wrong by up to +-{spread/2:.0f} px on "
          f"individual frames,")
    print(f"  which at 15 m displaces the projected lane by "
          f"{abs(15 - a.fx*a.height/((a.horizon+spread/2) - a.horizon + a.fx*a.height/15)):.2f} m.")
    if spread > 25:
        print("\n  => NO SINGLE STATIC CALIBRATION CAN SATISFY EVERY FRAME. The camera's")
        print("     attitude to the ROAD genuinely moves during the clip. A per-clip")
        print("     constant is the wrong model; the horizon has to be tracked per frame.")
    else:
        print("\n  => the attitude is stable; a per-clip constant is an adequate model and")
        print("     the per-frame disagreement has some other cause.")
    if a.json:
        a.json.write_text(json.dumps(dict(n=len(rows), horizon_median=float(np.median(rows)),
                                          horizon_p10=float(np.percentile(rows, 10)),
                                          horizon_p90=float(np.percentile(rows, 90)),
                                          horizon_sd=float(np.std(rows)),
                                          yaw_median=float(np.median(yaws)),
                                          yaw_sd=float(np.std(yaws)),
                                          frames=fr.tolist(),
                                          rows=rows.tolist(), yaws=yaws.tolist()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
