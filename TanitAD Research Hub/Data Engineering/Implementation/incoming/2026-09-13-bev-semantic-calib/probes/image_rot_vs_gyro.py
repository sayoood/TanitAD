#!/usr/bin/env python3
"""STEP 1 — is the per-frame residual the CAMERA or the CAR?

§68-69 established that the ~1 deg of per-frame camera-to-lane yaw variation is REAL
(split-half instrument noise 0.063 deg), is not curvature aliasing, and does not
correlate with the vehicle's yaw rate. What it did not do is separate the two things
that remain: the car's genuine yaw relative to the lane, and actual rotation of the
camera (EIS, or the body on its suspension).

THE INSTRUMENT, AND WHY IT IS DEPTH-FREE
----------------------------------------
For a rigid scene, the flow at an image point splits exactly::

    flow_i  =  A(x_i) · omega   +   s_i · e_i          s_i >= 0 unknown, e_i known

where the first term is ROTATIONAL (3 parameters, **independent of depth**) and the
second is TRANSLATIONAL, which for a known translation direction lies along the
**radial direction from the focus of expansion**, with a magnitude that carries the
unknown 1/Z. The vehicle's translation direction is known from the trajectory, so
``e_i`` is known — and therefore:

    **the component of each point's flow PERPENDICULAR to e_i is purely rotational,
    whatever the depth of that point.**

One scalar equation per point, three unknowns, hundreds of points, no depth anywhere.
That is why this works where the plane homography did not: no planarity assumption, no
forward-motion degeneracy, and every feature in the frame is usable rather than only
the ones on the road.

THE TEST IS BAND-SPECIFIC, AND THAT MATTERS
-------------------------------------------
The vehicle's own turning rotates the camera too, and BOTH the image and the gyro see
it. EIS is a high-pass compensator: it suppresses shake and passes slow turning. So
"image rotation vs gyro" compared broadband cannot answer the question — it would show
agreement driven by the turning whether or not EIS is active. The discriminator is the
RATIO of image to gyro amplitude **as a function of frequency**:

    EIS OFF  ->  ratio ~ 1 at every frequency
    EIS ON   ->  ratio ~ 1 slow, and << 1 in the shake band

PRE-REGISTERED, both outcomes committed before running:
  * ratio stays near 1 into the shake band  => the camera is rigid with the phone; the
    residual is the car and/or the trajectory, and a per-frame ATTITUDE fix is NOT the
    lever. Go to Step 3 (trajectory), do not build Step 2.
  * ratio falls well below 1 in the shake band => EIS is rotating frames, a fixed
    extrinsic cannot track it, and Step 2 (per-frame attitude from far-field
    correspondences) is the fix.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bev_calib as BC, lag_scale as LS, run_real as RR                # noqa: E402


def load_gyro(run):
    rows = list(csv.DictReader(open(run / "sensors" / "gyro.csv")))
    t = np.array([float(r["seconds_elapsed"]) for r in rows])
    w = np.stack([np.array([float(r[c]) for r in rows]) for c in ("x", "y", "z")], 1)
    o = np.argsort(t)
    return t[o], w[o]


def rot_jacobian(x, y, f):
    """d(flow)/d(omega) at normalised image coords (x, y), in PIXELS per radian."""
    Au = np.stack([f * x * y, -f * (1.0 + x * x), f * y], axis=1)
    Av = np.stack([f * (1.0 + y * y), -f * x * y, -f * x], axis=1)
    return Au, Av


def solve_omega(a, b, foe, fx, cx, cy, iters=8, c_px=1.0, foe_iters=4):
    """Robust depth-free rotation, with the FOE SOLVED FOR rather than assumed.

    ⛔ WHY THE FOE CANNOT BE ASSUMED. The perpendicular-to-radial trick is exact, but
    it is exquisitely sensitive to the FOE whenever the translational flow is large.
    A near-road point moves ~100 px between frames; if the FOE is off by 20 px at a
    radial distance of 300 px, the induced perpendicular component is 100 x 20/300 =
    6.7 px — against a genuine rotational signal of about 1.5 px at 0.03 rad/s over
    33 ms. The error is FOUR TIMES the signal.

    MEASURED with the FOE taken from the calibration: the recovered rotation had
    |r| <= 0.09 against the gyro in EVERY band, including 0-0.5 Hz where the vehicle's
    own turning must appear in both. That is not a result about EIS, it is a broken
    instrument — and the gyro side is validated (r = 0.992 against the trajectory's
    yaw rate), so the fault was here.

    The fix is to alternate: solve the rotation at the current FOE, de-rotate the flow,
    and re-fit the FOE as the least-squares intersection of the de-rotated flow lines
    (which are radial from the true FOE by construction). Three or four passes.
    """
    e_flow = b - a
    x = (a[:, 0] - cx) / fx
    y = (a[:, 1] - cy) / fx
    Au, Av = rot_jacobian(x, y, fx)
    om = np.zeros(3)
    for _ in range(foe_iters):
        e = a - foe
        n = np.linalg.norm(e, axis=1)
        ok = n > 25.0
        if ok.sum() < 30:
            return None, 0, foe
        eu = e[ok] / n[ok, None]
        perp = np.stack([-eu[:, 1], eu[:, 0]], axis=1)
        M = perp[:, 0:1] * Au[ok] + perp[:, 1:2] * Av[ok]
        r = np.einsum("ij,ij->i", e_flow[ok], perp)
        w = np.ones(len(r))
        for _ in range(iters):
            Mw = M * w[:, None]
            om, *_ = np.linalg.lstsq(Mw, r * w, rcond=None)
            res = r - M @ om
            s = 1.4826 * np.median(np.abs(res)) + 1e-9
            w = 1.0 / np.sqrt(1.0 + (res / (c_px * s)) ** 2)
        # de-rotate, then intersect the residual flow lines: they are radial from the
        # true FOE, so sum_i (perp_i . (foe - a_i))^2 is minimised at it.
        d2 = e_flow - np.stack([Au @ om, Av @ om], axis=1)
        L = np.linalg.norm(d2, axis=1)
        keep = L > 1.5
        if keep.sum() < 30:
            break
        u = d2[keep] / L[keep, None]
        pp = np.stack([-u[:, 1], u[:, 0]], axis=1)
        wf = np.minimum(L[keep], 60.0)                 # long vectors localise it best
        A_ = pp * wf[:, None]
        b_ = np.einsum("ij,ij->i", pp, a[keep]) * wf
        try:
            foe_new, *_ = np.linalg.lstsq(A_, b_, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(foe_new)) or np.linalg.norm(foe_new - foe) < 0.2:
            foe = foe_new if np.all(np.isfinite(foe_new)) else foe
            break
        foe = foe_new
    return om, int(ok.sum()), foe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--n", type=int, default=1200, help="consecutive frames")
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--fx", type=float, default=1533.0)
    ap.add_argument("--height", type=float, default=1.586)
    ap.add_argument("--horizon", type=float, default=448.4)
    ap.add_argument("--yaw", type=float, default=-5.35)
    ap.add_argument("--bonnet-row", type=int, default=832)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    P = dict(RR.NOMINAL)
    P.update(fx=a.fx, height=a.height, lateral=-0.126, yaw=np.deg2rad(a.yaw))
    P["pitch"] = LS.pitch_for_horizon(P, a.horizon)
    R_cv = BC.rot_ypr(P["yaw"], P["pitch"], P["roll"]) @ BC.R_CV_NOMINAL
    cx, cy = P["cx"], P["cy"]

    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    fdir = RR.RUN / "frames"
    gt, gw = load_gyro(RR.RUN)
    FPS = 29.9219

    feat = dict(maxCorners=700, qualityLevel=0.004, minDistance=10, blockSize=7)
    lk = dict(winSize=(21, 21), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    avail = sorted(int(p.stem) for p in fdir.glob("*.jpg"))
    lo = max(a.start, avail[0])
    seq = [f for f in avail if lo <= f < lo + a.n and (f + a.gap) in set(avail)]
    print(f"{len(seq)} consecutive frame pairs from {seq[0]} (gap {a.gap} = "
          f"{a.gap/FPS*1000:.0f} ms)\n")

    prev = None
    ts, OM, NP, GY, FOE = [], [], [], [], []
    mask = None
    for f0 in seq:
        g0 = prev if prev is not None and prev[0] == f0 else None
        if g0 is None:
            im0 = cv2.imread(str(fdir / f"{f0:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        else:
            im0 = g0[1]
        im1 = cv2.imread(str(fdir / f"{f0+a.gap:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        prev = (f0 + a.gap, im1)
        if im0 is None or im1 is None:
            continue
        if mask is None:
            mask = np.zeros(im0.shape, np.uint8)
            mask[: a.bonnet_row, :] = 255
        r = recs.get(f0)
        if r is None or not r["complete"] or r["speed_ms"] < 6.0:
            continue
        t = np.asarray(r["t"], float)
        dt = a.gap / FPS
        if t.min() > 0 or t.max() < dt:
            continue
        dx = float(np.interp(dt, t, np.asarray(r["x"], float)))
        dy = float(np.interp(dt, t, np.asarray(r["y"], float)))
        if np.hypot(dx, dy) < 0.05:
            continue
        # FOE = the image of the direction of travel (a point at infinity)
        dv = np.array([dx, dy, 0.0]); dv /= np.linalg.norm(dv)
        dc = R_cv @ dv
        if dc[2] <= 1e-6:
            continue
        foe = np.array([cx + a.fx * dc[0] / dc[2], cy + a.fx * dc[1] / dc[2]])

        p0 = cv2.goodFeaturesToTrack(im0, mask=mask, **feat)
        if p0 is None or len(p0) < 60:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(im0, im1, p0, None, **lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(im1, im0, p1, None, **lk)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & \
               (np.linalg.norm(p0.reshape(-1, 2) - pb.reshape(-1, 2), axis=1) < 0.7)
        A, B = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        if len(A) < 60:
            continue
        om, npts, foe_fit = solve_omega(A, B, foe, a.fx, cx, cy)
        if om is None:
            continue
        t0 = float(r["t_session_s"])
        m = (gt >= t0) & (gt < t0 + dt)
        if m.sum() < 1:
            continue
        ts.append(t0); OM.append(om / dt); NP.append(npts); GY.append(gw[m].mean(0))
        FOE.append(foe_fit)

    OM = np.asarray(OM); GY = np.asarray(GY); ts = np.asarray(ts)
    FOE = np.asarray(FOE)
    print(f"{len(OM)} usable pairs, median {np.median(NP):.0f} points per solve")
    print(f"fitted FOE: u {np.median(FOE[:,0]):.1f} +/- {1.4826*np.median(np.abs(FOE[:,0]-np.median(FOE[:,0]))):.1f}"
          f"   v {np.median(FOE[:,1]):.1f} +/- {1.4826*np.median(np.abs(FOE[:,1]-np.median(FOE[:,1]))):.1f}"
          f"   (calibration predicts u ~ {960-1533*np.tan(np.deg2rad(5.35)):.0f}, v ~ {a.horizon:.0f})\n")
    if len(OM) < 100:
        print("too few to say anything")
        return 1

    names_i = ["image wx (pitch)", "image wy (yaw)", "image wz (roll)"]
    names_g = ["gyro x", "gyro y", "gyro z"]
    print("correlation, image rotation rate vs gyro rate (broadband):")
    print("        " + "".join(f"{n:>12}" for n in names_g))
    C = np.zeros((3, 3))
    for i in range(3):
        row = ""
        for j in range(3):
            C[i, j] = np.corrcoef(OM[:, i], GY[:, j])[0, 1]
            row += f"{C[i,j]:>12.3f}"
        print(f"{names_i[i]:>16}" + row)
    pair = [(i, int(np.argmax(np.abs(C[i])))) for i in range(3)]
    print("\nbest axis pairing and amplitude ratio (rad/s per rad/s):")
    for i, j in pair:
        sl = float(np.polyfit(GY[:, j], OM[:, i], 1)[0])
        print(f"  {names_i[i]:>16} <- {names_g[j]}   r {C[i,j]:+.3f}   slope {sl:+.3f}"
              f"   rms image {np.std(OM[:,i]):.4f} vs gyro {np.std(GY[:,j]):.4f} rad/s")

    # ---- the discriminator: amplitude ratio BY FREQUENCY BAND ----------------- #
    def band(x, lo_hz, hi_hz, fs):
        X = np.fft.rfft(x - x.mean())
        fr = np.fft.rfftfreq(len(x), 1.0 / fs)
        X[(fr < lo_hz) | (fr >= hi_hz)] = 0
        return np.fft.irfft(X, n=len(x))

    fs = 1.0 / float(np.median(np.diff(ts)))
    print(f"\nsampling {fs:.1f} Hz; ⭐ the discriminator — amplitude ratio by band:")
    print(f"  {'band':>14}{'image rms':>13}{'gyro rms':>12}{'ratio':>9}{'corr':>9}")
    out_bands = {}
    for lo_hz, hi_hz in ((0.0, 0.5), (0.5, 1.5), (1.5, 4.0), (4.0, min(10.0, fs / 2))):
        if hi_hz <= lo_hz:
            continue
        rows = []
        for i, j in pair:
            bi = band(OM[:, i], lo_hz, hi_hz, fs)
            bg = band(GY[:, j], lo_hz, hi_hz, fs)
            if np.std(bg) < 1e-7:
                continue
            rows.append((np.std(bi), np.std(bg), np.std(bi) / np.std(bg),
                         float(np.corrcoef(bi, bg)[0, 1])))
        if not rows:
            continue
        ri = float(np.mean([r[0] for r in rows])); rg = float(np.mean([r[1] for r in rows]))
        rr = float(np.mean([r[2] for r in rows])); rc = float(np.mean([r[3] for r in rows]))
        print(f"  {f'{lo_hz:.1f}-{hi_hz:.1f} Hz':>14}{ri:>13.5f}{rg:>12.5f}{rr:>9.2f}{rc:>9.3f}")
        out_bands[f"{lo_hz}-{hi_hz}"] = dict(image_rms=ri, gyro_rms=rg, ratio=rr, corr=rc)

    print("\n  PRE-REGISTERED READING:")
    print("    ratio ~1 in every band      => camera rigid with the phone, EIS not acting")
    print("                                   => the residual is the car / the trajectory,")
    print("                                      Step 2 is NOT the lever, go to Step 3")
    print("    ratio << 1 in the shake band => EIS is rotating frames; a fixed extrinsic")
    print("                                   cannot track it => build Step 2")
    if a.json:
        a.json.write_text(json.dumps(dict(
            n=len(OM), corr=C.tolist(), pairing=[[int(i), int(j)] for i, j in pair],
            bands=out_bands, fs=fs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
