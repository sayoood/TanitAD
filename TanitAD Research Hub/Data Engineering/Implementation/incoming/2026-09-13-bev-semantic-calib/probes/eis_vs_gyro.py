#!/usr/bin/env python3
"""Is electronic image stabilisation ON? Race the image against the gyroscope.

WHY THIS IS THE RIGHT TEST, AND WHY EARLIER ONES FAILED
--------------------------------------------------------
Sayed: *"I did not find a config which satisfies all frames."* If EIS is active the
camera RE-FRAMES every frame to cancel handshake, so the effective rotation -- and
the principal point -- differ frame to frame. No single static calibration can then
satisfy all of them, and looking for one is looking for something that does not
exist.

The test uses the one part of the image the calibration cannot confuse: **content
far enough away that translation does not move it.** A hillside 500 m off shifts in
the frame almost entirely because the CAMERA ROTATED, so for distant features

    du ~= -f * omega_yaw * dt        dv ~= -f * omega_pitch * dt

With EIS off, the measured shift matches the gyro's prediction and the regression
slope is 1. With EIS on the shift is actively cancelled and the slope collapses
toward 0. The slope IS the stabilisation strength.

⚠️ An earlier attempt in this programme declared EIS "UNVERIFIED -- both roll and
yaw channels too noisy". That one band-passed a whole-image rotation estimate. This
one uses thousands of tracked distant points per frame and regresses against the
gyro directly, which is a far stronger estimator of the same quantity.

⚠️ The gyro's axis naming is NOT assumed. Sensor Logger writes columns z,y,x and the
mapping to image axes depends on device orientation, so every axis is regressed
against both image components and the best pairing is reported. Assuming a mapping
is how you get a null result from a real effect.
"""
import sys, argparse, json, pathlib, csv
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import run_real as RR

LK = dict(winSize=(31, 31), maxLevel=5,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))


def load_gyro(path):
    t, w = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["seconds_elapsed"]))
            w.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return np.asarray(t), np.asarray(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=pathlib.Path,
                    default=pathlib.Path("/root/trajdata/scratch3/2026-08-08_14-19-54-android"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--fx", type=float, default=1713.0)
    ap.add_argument("--t-video-start", type=float, default=3.3043,
                    help="the pipeline's own sync offset")
    ap.add_argument("--sky-rows", type=float, nargs=2, default=[0.12, 0.38],
                    help="fraction of image height holding DISTANT content only")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    gt, gw = load_gyro(a.raw / "Gyroscope.csv")
    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    usable = [r for r in recs if r["complete"] and (fdir / f"{int(r['frame']):06d}.jpg").exists()]
    fps = 29.922
    dt = 1.0 / fps

    du, dv, W = [], [], []
    for pi in np.linspace(0, len(usable) - 1, a.frames).astype(int):
        f0 = int(usable[pi]["frame"])
        p0, p1 = fdir / f"{f0:06d}.jpg", fdir / f"{f0+1:06d}.jpg"
        if not p1.exists():
            continue
        i0 = cv2.imread(str(p0), cv2.IMREAD_GRAYSCALE)
        i1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
        if i0 is None or i1 is None:
            continue
        H, Wd = i0.shape
        mask = np.zeros((H, Wd), np.uint8)
        mask[int(a.sky_rows[0] * H):int(a.sky_rows[1] * H), :] = 255
        p = cv2.goodFeaturesToTrack(i0, maxCorners=500, qualityLevel=0.01,
                                    minDistance=10, mask=mask, blockSize=7)
        if p is None or len(p) < 30:
            continue
        q, s1, _ = cv2.calcOpticalFlowPyrLK(i0, i1, p, None, **LK)
        b, s0, _ = cv2.calcOpticalFlowPyrLK(i1, i0, q, None, **LK)
        ok = (s1.ravel() == 1) & (s0.ravel() == 1) \
             & (np.linalg.norm((b - p).reshape(-1, 2), axis=1) < 0.7)
        if ok.sum() < 25:
            continue
        d = (q.reshape(-1, 2) - p.reshape(-1, 2))[ok]
        # frame f0 is at video time f0/fps, i.e. sensor time f0/fps + t_video_start
        ts = f0 / fps + a.t_video_start
        m = (gt >= ts - dt / 2) & (gt <= ts + dt / 2)
        if m.sum() < 1:
            continue
        du.append(float(np.median(d[:, 0])))
        dv.append(float(np.median(d[:, 1])))
        W.append(gw[m].mean(axis=0))
    du, dv, W = np.asarray(du), np.asarray(dv), np.asarray(W)
    if len(du) < 40:
        print(f"only {len(du)} usable frames — cannot decide")
        return 1

    print(f"{len(du)} frame pairs, distant content in rows "
          f"{a.sky_rows[0]:.2f}-{a.sky_rows[1]:.2f} H, f = {a.fx:.0f} px\n")
    print("measured image shift of DISTANT content vs each gyro axis.")
    print("slope 1.00 => the image follows the gyro, EIS is OFF.")
    print("slope << 1 => the rotation is being cancelled, EIS is ON.\n")
    print(f"  {'pair':>12} {'slope':>8} {'r':>7}   {'predicted px rms':>17} {'measured px rms':>16}")
    best = None
    for iax, axname in enumerate("xyz"):
        pred = -a.fx * W[:, iax] * dt
        for meas, mname in ((du, "du"), (dv, "dv")):
            if np.std(pred) < 1e-9:
                continue
            slope = float(np.sum(pred * meas) / np.sum(pred * pred))
            r = float(np.corrcoef(pred, meas)[0, 1])
            print(f"  gyro.{axname} -> {mname} {slope:8.3f} {r:+7.3f}   "
                  f"{np.sqrt((pred**2).mean()):14.2f} px {np.sqrt((meas**2).mean()):14.2f} px")
            if best is None or abs(r) > abs(best[2]):
                best = (f"gyro.{axname}->{mname}", slope, r,
                        float(np.sqrt((pred ** 2).mean())),
                        float(np.sqrt((meas ** 2).mean())))
    name, slope, r, prms, mrms = best
    print(f"\n  strongest pairing: {name}   slope {slope:.3f}   r {r:+.3f}")
    if abs(r) < 0.35:
        print("  ⚠️ the correlation is too weak to decide. Either the sync is wrong or the")
        print("     distant-content assumption is failing; NOT evidence either way.")
    elif abs(slope) > 0.75:
        print("  => the image follows the gyro. EIS is OFF; a single static calibration")
        print("     is the right model and the per-frame disagreement is something else.")
    elif abs(slope) < 0.45:
        print(f"  => rotation is being CANCELLED ({100*(1-abs(slope)):.0f}% suppressed).")
        print("     EIS IS ON. No single static calibration can satisfy every frame,")
        print("     because the camera re-frames each one. Re-record with stabilisation")
        print("     off, or estimate a per-frame rotation and take it out.")
    else:
        print("  => partial suppression — consistent with EIS acting on some frequencies.")
    if a.json:
        a.json.write_text(json.dumps(dict(n=len(du), best=name, slope=slope, r=r,
                                          pred_rms_px=prms, meas_rms_px=mrms), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
