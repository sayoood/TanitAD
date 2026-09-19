#!/usr/bin/env python3
"""STEP 3 — does the corridor's predicted deviation actually come true?

THE QUESTION, sharpened by §72 and §74. The per-frame corridor-to-lane angle is real
(25x the 0.088 deg noise floor), it is not the camera (common-mode <= 0.25 deg), and it
varies on a 1-3 s timescale. So it is either the vehicle genuinely pointing off the lane
-- in which case the corridor is CORRECT and the rejected frames show real driving --
or it is error in the calibration/trajectory.

Those two are distinguishable, because a car pointed off the lane GOES somewhere:

    a heading of psi_rel off the lane  ==>  a lateral displacement of x(tau)*tan(psi_rel)
    relative to the lane after travelling x(tau)

So measure the angle now, and check the lateral position later.

TWO QUANTITIES, BOTH FROM THE PAINT, NEITHER NEEDING A CALIBRATION
------------------------------------------------------------------
    psi_rel(t) = theta_L(t) - median(theta_L)     heading relative to the lane
                 (the median removes the mount yaw empirically, so the adopted
                  -5.35 deg is not assumed)

    L(t) = 1.75 * (m_L + m_R) / |m_L - m_R|       lateral offset from lane centre

The second one carries **no calibration at all**: ``h`` cancels against
``|m_L - m_R| = 3.5/h``. No horizon, no focal length, no ego motion.

PRE-REGISTERED, both outcomes committed before running:
  * dL_measured tracks x*tan(psi_rel) with slope ~ +-1  =>  the angle is REAL VEHICLE
    HEADING. The corridor is right, the residual is the car, and the remaining work is
    presentation (say so in the overlay), not geometry.
  * slope ~ 0 or no correlation  =>  the angle does NOT predict where the car goes, so
    it is not heading. It is calibration or trajectory error, and the corridor is wrong.

⚠️ A third possibility that must not be mistaken for the first: L and psi_rel come from
the SAME two line fits, so a line misidentification moves both. It moves them
differently (psi from the intercept at the horizon, L from the slope SUM), but the
control that settles it is the LAG STRUCTURE -- a shared per-frame artefact predicts
tau = 0 and dies immediately, while real heading predicts a displacement that GROWS
with tau. The slope is therefore reported against tau, not at one lag.
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
from noise_floor import ransac_line                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx", type=float, default=1533.0)
    ap.add_argument("--height", type=float, default=1.586)
    ap.add_argument("--horizon", type=float, default=448.4)
    ap.add_argument("--yaw", type=float, default=-5.35)
    ap.add_argument("--lane-m", type=float, default=3.5)
    ap.add_argument("--x-lo", type=float, default=10.0)
    ap.add_argument("--x-hi", type=float, default=40.0)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    P0 = dict(RR.NOMINAL)
    P0.update(fx=a.fx, height=a.height, lateral=-0.126, yaw=np.deg2rad(a.yaw))
    P0["pitch"] = LS.pitch_for_horizon(P0, a.horizon)
    fh = a.fx * a.height
    rows = np.arange(a.horizon + fh / a.x_hi, a.horizon + fh / a.x_lo, 1.0)
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
    print(f"longest straight run: {len(runs)} consecutive frames\n")

    T, TH, LAT, SPD, TX, TY = [], [], [], [], [], []
    for f in runs:
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
        (ml, bl), _ = max(left, key=lambda z: z[1])
        (mr, br), _ = max(right, key=lambda z: z[1])
        dm = ml - mr
        # the pair must BE one lane: |m_L - m_R| = 3.5/h ~ 2.2 (§54's cluster)
        if not (1.6 < abs(dm) < 3.2):
            continue
        th = np.rad2deg(np.arctan((ml * a.horizon + bl - 960.0) / a.fx))
        if abs(th) > 20:
            continue
        lat = a.lane_m / 2.0 * (ml + mr) / abs(dm)
        r = recs[f]
        t = np.asarray(r["t"], float)
        T.append(float(r["t_session_s"])); TH.append(th); LAT.append(lat)
        SPD.append(float(r["speed_ms"]))
        TX.append((t, np.asarray(r["x"], float), np.asarray(r["y"], float)))
    T = np.asarray(T); TH = np.asarray(TH); LAT = np.asarray(LAT); SPD = np.asarray(SPD)
    print(f"{len(T)} frames with a valid single-lane pair "
          f"(|m_L-m_R| in 1.6-3.2, i.e. one lane at h~1.6 m)\n")
    if len(T) < 80:
        print("too few")
        return 1

    psi = np.deg2rad(TH - np.median(TH))            # heading relative to the lane, rad
    print(f"psi_rel: median removed ({np.median(TH):+.2f} deg = the mount yaw, "
          f"measured not assumed), rms {np.rad2deg(psi).std():.2f} deg")
    print(f"lateral offset L: median {np.median(LAT):+.2f} m, "
          f"robust sd {1.4826*np.median(np.abs(LAT-np.median(LAT))):.3f} m\n")

    print("DOES THE ANGLE PREDICT THE LATER LATERAL POSITION?")
    print(f"  {'lag':>7}{'pairs':>7}{'predicted rms':>15}{'measured rms':>14}"
          f"{'slope':>9}{'corr':>8}{'skill':>8}")
    out = []
    for lag in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        pred, meas = [], []
        for i in range(len(T)):
            j = np.searchsorted(T, T[i] + lag)
            if j >= len(T) or abs(T[j] - T[i] - lag) > 0.05:
                continue
            t_, xw, yw = TX[i]
            if t_.max() < lag:
                continue
            dist = float(np.interp(lag, t_, xw))     # metres travelled forward
            pred.append(dist * np.tan(psi[i]))
            meas.append(LAT[j] - LAT[i])
        if len(pred) < 40:
            continue
        pr = np.asarray(pred); me = np.asarray(meas)
        sl = float(np.polyfit(pr, me, 1)[0])
        rr = float(np.corrcoef(pr, me)[0, 1])
        skill = 1.0 - np.var(me - sl * pr) / np.var(me)
        print(f"  {lag:>5.2f} s{len(pr):>7}{np.std(pr):>13.3f} m{np.std(me):>12.3f} m"
              f"{sl:>9.2f}{rr:>8.3f}{skill:>8.2f}")
        out.append(dict(lag=lag, n=len(pr), pred_rms=float(np.std(pr)),
                        meas_rms=float(np.std(me)), slope=sl, corr=rr,
                        skill=float(skill)))

    print("\n  PRE-REGISTERED READING:")
    print("    |slope| ~ 1 and growing pairs  => the angle IS real vehicle heading;")
    print("       the corridor is right and the residual is the car.")
    print("    slope ~ 0                      => the angle does not predict where the")
    print("       car goes, so it is calibration/trajectory error and the corridor is wrong.")
    print("    ⚠️ a shared per-frame fit artefact would show at the SHORTEST lag and")
    print("       fade; real heading GROWS with lag. Read the trend, not one row.")
    if a.json:
        a.json.write_text(json.dumps(dict(
            n=len(T), mount_yaw_deg=float(np.median(TH)),
            psi_rms_deg=float(np.rad2deg(psi).std()),
            lat_median=float(np.median(LAT)), rows=out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
