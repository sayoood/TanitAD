#!/usr/bin/env python3
"""Is electronic image stabilisation (EIS) active on this recording?

WHY THIS DECIDES WHETHER BEV CALIBRATION IS EVEN VALID
------------------------------------------------------
The whole method assumes ONE mount rotation shared by every frame. EIS breaks
that assumption at the root: it re-crops and counter-rotates each frame to cancel
handshake, so the effective extrinsics become per-frame and no single (yaw, pitch,
roll) can describe the clip. The mp4 carries no EIS flag -- only
``com.android.version=16`` -- so it has to be measured.

THE TEST, and why ROLL is the right channel
-------------------------------------------
Compare the rotation the CAMERA actually underwent (recovered from image motion)
against the rotation the PHONE underwent (gyro). Without EIS they are the same
rigid body and the two agree. With EIS the camera sees LESS rotation than the
gyro reports, because that is precisely what stabilisation removes.

⚠️ Yaw and pitch are the wrong channels for this. ``image_angular_rate``'s own
docstring warns that forward translation leaks into its ``tx``/``ty``, and
MEASURED 2026-08-08 that leak inflated camera yaw to p50 5.42 deg/s against the
gyro's 1.46 -- a 3.7x discrepancy that has nothing to do with EIS. **Roll is
immune**: a pure forward translation produces no in-plane image rotation, so the
roll channel carries rotation and only rotation.

So the discriminator is the regression of camera roll rate on gyro roll rate:

    slope ~ 1, good r   ->  NO EIS (or EIS not touching roll): the method is valid
    slope << 1          ->  EIS IS ATTENUATING rotation: per-frame extrinsics,
                            and a single-rotation fit is measuring an average that
                            describes no individual frame

Prior expectation from the 2026-08-08 numbers: camera roll p50 1.87 deg/s vs gyro
1.81 deg/s -- nearly equal, which points at NO EIS. That is a comparison of
marginal distributions though, not of matched samples; this does the matched,
time-aligned regression.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=pathlib.Path,
                    default=pathlib.Path("/root/trajdata/scratch_session"))
    ap.add_argument("--video", type=pathlib.Path, required=True)
    ap.add_argument("--gyro", type=pathlib.Path, required=True)
    ap.add_argument("--t-video-start", type=float, required=True,
                    help="sync offset: session_time = video_time + this")
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--start", type=float, default=5.0)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    from trajlib import timesync as TS

    video = TS.probe_video(str(a.video))
    t_mid, _, comps = TS.image_angular_rate(video, t_start=a.start, t_dur=a.dur)
    cam_roll = comps[:, 2]                       # in-plane rotation, translation-immune
    cam_yaw, cam_pitch = comps[:, 0], comps[:, 1]

    g = pd.read_csv(a.gyro)
    tg = g["seconds_elapsed"].to_numpy(float)
    wg = g[["x", "y", "z"]].to_numpy(float)

    # camera roll is rotation about the OPTICAL AXIS; for a roughly forward-facing
    # phone that is the gyro axis most aligned with the direction of travel. Pick
    # it by correlation rather than by assuming the mount orientation.
    ts = t_mid + a.t_video_start
    best = None
    for k, name in enumerate("xyz"):
        for sign in (+1, -1):
            gi = sign * np.interp(ts, tg, wg[:, k])
            ok = np.isfinite(cam_roll) & np.isfinite(gi)
            if ok.sum() < 50:
                continue
            r = float(np.corrcoef(cam_roll[ok], gi[ok])[0, 1])
            if best is None or r > best[0]:
                best = (r, k, sign, gi, ok)
    if best is None:
        print("not enough overlapping samples")
        return 1
    r, k, sign, gi, ok = best
    slope, icept = np.polyfit(gi[ok], cam_roll[ok], 1)

    print(f"samples {int(ok.sum())}   gyro axis {'+-'[sign < 0]}{'xyz'[k]} "
          f"(chosen by correlation, not assumed)")
    print(f"  camera roll rate  p50 |.| {np.rad2deg(np.nanpercentile(np.abs(cam_roll[ok]),50)):.2f} deg/s")
    print(f"  gyro  roll rate   p50 |.| {np.rad2deg(np.nanpercentile(np.abs(gi[ok]),50)):.2f} deg/s")
    print(f"  regression  camera = {slope:.3f} * gyro + {np.rad2deg(icept):.3f} deg/s   r = {r:.3f}")
    print(f"  (for reference, the translation-contaminated channels: "
          f"yaw p50 {np.rad2deg(np.nanpercentile(np.abs(cam_yaw),50)):.2f}, "
          f"pitch p50 {np.rad2deg(np.nanpercentile(np.abs(cam_pitch),50)):.2f} deg/s)")

    if r < 0.3:
        verdict = "INCONCLUSIVE — camera and gyro roll barely correlate; neither claim is supported"
    elif slope < 0.6:
        verdict = (f"EIS LIKELY ACTIVE — camera sees only {slope:.0%} of the gyro's roll. "
                   f"Extrinsics are per-frame and a single-rotation fit describes no frame.")
    elif slope > 0.85:
        verdict = ("NO EIS ATTENUATION IN ROLL — camera and phone rotate together, so a "
                   "single mount rotation is a valid model for this clip.")
    else:
        verdict = (f"PARTIAL — slope {slope:.2f}. Some attenuation; treat a single-rotation "
                   f"fit as approximate and consider a per-frame rotation residual.")
    print(f"\nVERDICT: {verdict}")

    if a.json:
        a.json.write_text(json.dumps(dict(
            n=int(ok.sum()), gyro_axis=f"{'+-'[sign < 0]}{'xyz'[k]}", slope=float(slope),
            intercept_deg=float(np.rad2deg(icept)), r=float(r), verdict=verdict,
            cam_roll_p50_deg=float(np.rad2deg(np.nanpercentile(np.abs(cam_roll[ok]), 50))),
            gyro_roll_p50_deg=float(np.rad2deg(np.nanpercentile(np.abs(gi[ok]), 50))),
        ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
