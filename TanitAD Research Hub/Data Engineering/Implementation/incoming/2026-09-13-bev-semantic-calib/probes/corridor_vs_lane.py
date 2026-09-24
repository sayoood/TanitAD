#!/usr/bin/env python3
"""Does the TRAJECTORY-drawn corridor drift off the lane with range? Measure it.

Sayed: *"the overlay corridor cuts road markings at large distances."* Part 9 blamed
the trajectory's heading error (hold-out rms 0.53 deg -> 0.37 m at 40 m). That was an
INFERENCE from a hold-out statistic, not a measurement of the symptom.

⚠️ Every other instrument in this directory projects a STRAIGHT line at a fixed
lateral offset in the vehicle frame. The rendered corridor does not -- it is a ribbon
around the FUTURE PATH. So a heading error in the trajectory bends the corridor while
leaving every straight-line residual reading zero. This probe is the only one that
tests what is actually drawn.

THE TEST. The corridor's left edge should sit a CONSTANT distance from the left
painted line at every range, because both run along the lane. Any slope of that
distance against range is the angle between the corridor and the lane. Measuring it
for the trajectory ribbon AND for a straight ribbon separates the two causes:

    straight ribbon slope  -> calibration (yaw, horizon)
    trajectory  minus straight -> the trajectory's own heading error

Only the left line is used: it is solid, and fits to 1.39 px against the dashed
right line's 8 samples a frame.
"""
import sys, argparse, json, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bev_calib as BC, lag_scale as LS, run_real as RR


def corridor_point(rec, x_target, half_w, straight=False):
    """Lateral position of the ribbon's LEFT edge at longitudinal range ``x_target``."""
    t = np.asarray(rec["t"], float)
    px = np.asarray(rec["x"], float)
    py = np.asarray(rec["y"], float)
    yaw = np.asarray(rec["yaw"], float)
    fut = t >= 0
    if fut.sum() < 5:
        return None
    px, py, yaw = px[fut], py[fut], yaw[fut]
    if px[-1] < x_target:
        return None
    if straight:
        return x_target, half_w            # the reference: a straight ribbon
    yq = float(np.interp(x_target, px, py))
    wq = float(np.interp(x_target, px, yaw))
    # ribbon edge is offset along the LEFT NORMAL of the path, exactly as viz does
    nx_, ny_ = -np.sin(wq), np.cos(wq)
    return x_target + nx_ * half_w, yq + ny_ * half_w


def residual_at(img_ink, P, gx, gy, assoc_m=1.1):
    """Signed distance, in metres, from a projected ground point to the nearest paint."""
    H, W = img_ink.shape
    uv = BC.project_ground(np.array([[gx, gy]]), P)[0]
    if not np.isfinite(uv).all():
        return None
    r, c = int(round(uv[1])), int(round(uv[0]))
    if not (0 <= r < H):
        return None
    mpp = max(gx, 1.0) / P["fx"]
    hw = int(round(assoc_m / mpp))
    lo, hi = max(0, c - hw), min(W, c + hw + 1)
    if hi <= lo:
        return None
    band = img_ink[max(0, r - 2):r + 3, lo:hi].sum(axis=0)
    if band.max() <= 0:
        return None
    cols = np.flatnonzero(band >= band.max())
    return float((float(cols.mean()) + lo - c) * mpp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=260)
    ap.add_argument("--fx", type=float, default=1666.0)
    ap.add_argument("--height", type=float, default=1.622)
    ap.add_argument("--horizon", type=float, default=485.0)
    ap.add_argument("--yaw", type=float, default=-6.80)
    ap.add_argument("--lateral", type=float, default=-0.126)
    # ⚠️ 1.75 m, the LANE half-width, not 0.90 m the vehicle half-width. The probe
    # predicts where the PAINTED LINE should be if the corridor tracks the lane, so
    # the association window sits on the paint. MEASURED with 0.90: the window had to
    # bridge the 0.85 m from ribbon edge to line and jumped features instead --
    # residuals of +0.384 m at 10 m and -0.790 m at 14 m, and the sample collapsed
    # from n=199 to n=11 by 18 m.
    ap.add_argument("--half-width", type=float, default=1.75)
    ap.add_argument("--ranges", type=float, nargs="+", default=[10., 14., 18., 22., 26.])
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    P = dict(RR.NOMINAL)
    P.update(fx=a.fx, height=a.height, lateral=a.lateral, yaw=np.deg2rad(a.yaw))
    P["pitch"] = LS.pitch_for_horizon(P, a.horizon)
    from trajlib import lane_calib as LC
    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > 8.0
              and (fdir / f"{int(r['frame']):06d}.jpg").exists()]

    slopes = {"trajectory ribbon": [], "straight ribbon": []}
    prof = {k: {x: [] for x in a.ranges} for k in slopes}
    for pi in np.linspace(0, len(usable) - 1, a.frames).astype(int):
        rec = usable[pi]
        f = int(rec["frame"])
        img = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        u, v = LC._ridge_points(img, int(0.50 * H), int(0.95 * H))
        if len(u) < 120:
            continue
        ink = np.zeros((H, W), np.uint8)
        ink[np.clip(v, 0, H - 1).astype(int), np.clip(u, 0, W - 1).astype(int)] = 1
        for mode, straight in (("trajectory ribbon", False), ("straight ribbon", True)):
            xs, rs = [], []
            for x in a.ranges:
                pt = corridor_point(rec, x, a.half_width, straight)
                if pt is None:
                    continue
                d = residual_at(ink, P, pt[0], pt[1])
                if d is None:
                    continue
                xs.append(x); rs.append(d); prof[mode][x].append(d)
            if len(xs) >= 3:
                s, _ = np.polyfit(xs, rs, 1)
                if abs(s) < 0.20:
                    slopes[mode].append(s)

    print(f"calibration: f {a.fx}  h {a.height}  horizon {a.horizon}  yaw {a.yaw}\n")
    print("Distance from the ribbon's LEFT edge to the left painted line, by range.")
    print("Parallel to the lane => constant. A slope IS the corridor-to-lane angle.\n")
    print(f"  {'range':>7}" + "".join(f"{m:>22}" for m in slopes))
    for x in a.ranges:
        row = f"  {x:5.0f} m"
        for m in slopes:
            v_ = prof[m][x]
            row += f"{np.median(v_):+15.3f} m (n{len(v_):3d})" if len(v_) >= 10 else f"{'--':>22}"
        print(row)
    print()
    out = {}
    for m in slopes:
        s = np.asarray(slopes[m])
        if len(s) < 20:
            print(f"  {m}: only {len(s)} frames — not enough")
            continue
        ang = np.rad2deg(np.arctan(np.median(np.abs(s))))
        out[m] = dict(n=int(len(s)), median_slope=float(np.median(s)),
                      median_abs_slope=float(np.median(np.abs(s))), angle_deg=float(ang))
        print(f"  {m:22s} n {len(s):3d}   median |slope| {np.median(np.abs(s)):.4f} m/m"
              f"   = {ang:.2f} deg   -> {40*np.tan(np.deg2rad(ang)):.2f} m at 40 m")
    if len(out) == 2:
        t_ = out["trajectory ribbon"]["angle_deg"]; s_ = out["straight ribbon"]["angle_deg"]
        print(f"\n  trajectory adds {t_ - s_:+.2f} deg over a straight ribbon "
              f"({40*np.tan(np.deg2rad(t_)) - 40*np.tan(np.deg2rad(s_)):+.2f} m at 40 m)")
        if t_ > s_ * 1.25:
            print("  => the TRAJECTORY is the dominant term. Part 9's diagnosis holds:")
            print("     no calibration change fixes this; the heading estimate must improve.")
        elif t_ < s_ * 0.8:
            print("  => the trajectory is BETTER than a straight ribbon — it is tracking the")
            print("     road. The residual drift is CALIBRATION, not the trajectory.")
        else:
            print("  => they are comparable: the trajectory is NOT the dominant term, so")
            print("     Part 9's diagnosis is NOT supported as the main cause.")
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
