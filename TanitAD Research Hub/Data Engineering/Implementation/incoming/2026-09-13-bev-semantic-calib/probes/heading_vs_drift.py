#!/usr/bin/env python3
"""Is the per-frame attitude swing the CAR steering, or the calibration failing?

Sayed could not find one configuration that satisfies every frame, and the
per-frame residual slope does spread by 4.6 deg (p10-p90). Two readings, opposite
consequences:

  (a) the calibration is wrong and varies  -> keep fitting;
  (b) the CAR's heading relative to the lane varies -> there is nothing to fit, and
      a static calibration is already correct.

They are separable, because (b) makes a prediction (a) does not. If the vehicle is
yawed by theta relative to the lane, it is CROSSING the lane at v*sin(theta), so the
lateral offset must change at that rate::

    d(lateral offset)/dt  =  v * tan(theta)  =  v * (residual slope)

Both sides are measured here, independently, on consecutive frames. A regression
slope near 1 means the attitude swing is the car actually steering, and the overlay
is right to follow it.
"""
import sys, argparse, json, pathlib, importlib.util
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bev_calib as BC, lag_scale as LS, run_real as RR

_spec = importlib.util.spec_from_file_location(
    "lr", str(pathlib.Path(__file__).resolve().parent / "lane_residual.py"))
lr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-frac", type=float, default=0.30)
    ap.add_argument("--n", type=int, default=260, help="CONSECUTIVE frames")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fx", type=float, default=1713.0)
    ap.add_argument("--height", type=float, default=1.427)
    ap.add_argument("--horizon", type=float, default=465.0)
    ap.add_argument("--yaw", type=float, default=-6.80)
    ap.add_argument("--lateral", type=float, default=-0.126)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    P = dict(RR.NOMINAL)
    P.update(fx=a.fx, height=a.height, lateral=a.lateral, yaw=np.deg2rad(a.yaw))
    P["pitch"] = LS.pitch_for_horizon(P, a.horizon)
    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    by = {int(r["frame"]): r for r in recs}
    usable = sorted(k for k, r in by.items()
                    if r["complete"] and r["speed_ms"] > 8.0
                    and (fdir / f"{k:06d}.jpg").exists())
    i0 = int(a.start_frac * len(usable))
    sel = usable[i0:i0 + a.n * a.stride:a.stride]
    ranges = [8., 10., 12., 15., 20.]
    fps = 29.922

    t, off, slope, spd = [], [], [], []
    for f in sel:
        img = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        d = [q for q in lr.residuals_one(img, P, ranges, 1.75)
             if q["side"] == "L" and q["resid_m"] is not None]
        if len(d) < 4:
            continue
        x = np.array([q["x"] for q in d]); y = np.array([q["resid_m"] for q in d])
        s, b = np.polyfit(x, y, 1)
        if abs(s) > 0.15:
            continue
        t.append(f / fps); slope.append(s); off.append(b + s * 12.0)
        spd.append(float(by[f]["speed_ms"]))
    t, off, slope, spd = map(np.asarray, (t, off, slope, spd))
    if len(t) < 30:
        print(f"only {len(t)} frames usable — cannot decide")
        return 1

    # smooth lightly: the residual carries per-frame detection noise, the drift does not
    k = 5
    ker = np.ones(k) / k
    offs_s = np.convolve(off, ker, mode="valid")
    t_s = np.convolve(t, ker, mode="valid")
    sl_s = np.convolve(slope, ker, mode="valid")
    v_s = np.convolve(spd, ker, mode="valid")
    drift = np.gradient(offs_s, t_s)              # m/s, measured
    pred = v_s * sl_s                              # m/s, predicted from the attitude
    m = np.isfinite(drift) & np.isfinite(pred)
    drift, pred = drift[m], pred[m]
    if len(drift) < 20:
        print("too few samples after smoothing")
        return 1
    A = float(np.sum(pred * drift) / np.sum(pred * pred))
    r = float(np.corrcoef(pred, drift)[0, 1])
    print(f"{len(t)} consecutive frames (stride {a.stride}), {len(drift)} paired samples\n")
    print(f"  measured lateral drift   rms {np.sqrt((drift**2).mean()):.3f} m/s")
    print(f"  predicted from attitude  rms {np.sqrt((pred**2).mean()):.3f} m/s")
    print(f"  regression  drift = {A:.2f} x predicted     r = {r:+.3f}\n")
    if r > 0.45 and 0.4 < A < 2.2:
        print("  => THE ATTITUDE SWING IS THE CAR STEERING. The vehicle really is yawed")
        print("     relative to the lane by the amount the overlay shows, and it really is")
        print("     crossing the lane at the matching rate. A static calibration is the")
        print("     right model; no configuration can make the corridor lane-parallel in")
        print("     every frame, because the CAR is not lane-parallel in every frame.")
    elif r < 0.25:
        print("  => NO relationship. The per-frame attitude swing is NOT the car steering,")
        print("     so it is the geometry -- a per-frame calibration term is needed.")
    else:
        print("  => partial. Some of the swing is real steering, some is not.")
    if a.json:
        a.json.write_text(json.dumps(dict(n=len(t), n_paired=int(len(drift)),
                                          regression=A, r=r,
                                          drift_rms=float(np.sqrt((drift**2).mean())),
                                          pred_rms=float(np.sqrt((pred**2).mean()))), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
