#!/usr/bin/env python3
"""Estimate the video/sensor time offset from BEV agreement itself.

WHY THIS EXISTS
---------------
MEASURED 2026-09-13, and it is the reason this script was written. Trying to
verify the pipeline's ``t_video_start = 3.3043 s`` by cross-correlating the
camera's image-derived yaw rate against the gyro (the pipeline's own band-pass,
0.15-4.0 Hz) produced **no distinct peak**::

    best |r| = 0.312 at lag +4.40 s        (pipeline uses +3.30 s)
    r at the pipeline's lag  = -0.193      (negative!)
    |r| over all lags: p50 0.111  p95 0.243  max 0.312
    max / p95 = 1.28x   ->  indistinguishable from the noise floor

⚠️ This does NOT prove the pipeline's sync is wrong. The pipeline refines the lag
on the 3-vector across three bands with a coarse+fine scheme and an agreement
test, which is a stronger estimator than the scalar correlation above, and its
``sync_score`` is not a Pearson r, so 0.47 and 0.31 are not comparable. What it
does establish is that the sync **cannot be independently confirmed from this
channel** -- so the BEV method should not simply trust it.

It does not have to. BEV agreement is itself a function of the offset: pair each
frame's ink with the ego motion from the WRONG moment and the markings smear
exactly as they do under a wrong calibration. Scanning the offset and taking the
peak turns the sync from an assumption into a measurement, in the same currency
as everything else here.

HOW THE OFFSET ENTERS
---------------------
Relative poses come from the anchor record's own window, so a global offset does
not corrupt a window internally -- it changes WHICH record is paired with a given
video frame. Over a near-constant-speed stretch that is a mild error (speed here
spans 19.7-23.3 m/s, so ~15% of displacement at worst), which is why the method
is only weakly sensitive to it. The scan measures that sensitivity rather than
assuming it is negligible.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bev_calib as BC                                              # noqa: E402
import run_real as RR                                               # noqa: E402


def anchors_at_offset(recs, run, n_anchors, span_s, step_s, max_pts, dt_off, seed=0):
    """Anchors whose POSES are read `dt_off` seconds away from the frame time."""
    rng = np.random.default_rng(seed)
    by_frame = {int(r["frame"]): r for r in recs}
    frames_dir = run / "frames"
    usable = [r for r in recs
              if r["complete"] and r["speed_ms"] > 8.0
              and (frames_dir / f"{int(r['frame']):06d}.jpg").exists()]
    picks = np.linspace(0, len(usable) - 1, n_anchors).astype(int)
    offsets = np.arange(0.0, span_s + 1e-9, step_s)
    fps = 29.922
    out = []
    for pi in picks:
        a = usable[pi]
        t = np.asarray(a["t"], float)
        ax, ay, ayaw = (np.asarray(a[k], float) for k in ("x", "y", "yaw"))
        obs, poses = [], []
        for d in offsets:
            fno = int(round(a["frame"] + d * fps))
            p = frames_dir / f"{fno:06d}.jpg"
            if fno not in by_frame or not p.exists():
                continue
            # the ONLY thing the offset changes: where in the window we read the pose
            tq = d + dt_off
            if not (t.min() <= tq <= t.max()):
                continue
            uv = RR.ridge_ink(p, max_pts, rng)
            if len(uv) < 200:
                continue
            obs.append(uv)
            poses.append((float(np.interp(tq, t, ax) - np.interp(dt_off, t, ax)),
                          float(np.interp(tq, t, ay) - np.interp(dt_off, t, ay)),
                          float(np.interp(tq, t, ayaw) - np.interp(dt_off, t, ayaw))))
        if len(obs) >= 4:
            out.append((obs, poses, int(a["frame"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=RR.RUN)
    ap.add_argument("--anchors", type=int, default=6)
    ap.add_argument("--span", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--max-pts", type=int, default=3000)
    ap.add_argument("--params", type=pathlib.Path, default=None,
                    help="JSON from run_real.py; uses its fitted calibration")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    recs = RR.load_records(a.run)
    P = dict(RR.NOMINAL)
    if a.params and a.params.exists():
        fit = json.loads(a.params.read_text()).get("fit")
        if fit:
            P.update({k: float(v) for k, v in fit.items()})
            print("using the fitted calibration from", a.params)
    grid = BC.BevGrid(x_range=(5.0, 40.0), y_range=(-10.0, 10.0), cell=0.08)

    print("\nBEV agreement vs pose time offset  (0.00 = the pipeline's sync)")
    rows = []
    for dt in np.arange(-1.5, 1.51, 0.25):
        anc = anchors_at_offset(recs, a.run, a.anchors, a.span, a.step, a.max_pts, float(dt))
        s = RR.multi_agreement(anc, P, grid) if anc else 0.0
        rows.append((float(dt), s, len(anc)))
        print(f"    dt = {dt:+.2f} s   anchors {len(anc):2d}   agreement {s:.6e}")
    rows_s = np.array([r[1] for r in rows])
    dts = np.array([r[0] for r in rows])
    peak = float(dts[rows_s.argmax()])
    ratio = float(rows_s.max() / max(rows_s.min(), 1e-12))
    print(f"\n  peak at dt = {peak:+.2f} s   max/min = {ratio:.2f}x")
    if abs(peak) <= 0.25:
        print("  => consistent with the pipeline's sync; BEV sees no offset error.")
    else:
        print(f"  => BEV prefers a pose shifted by {peak:+.2f} s. At ~22 m/s that is "
              f"{abs(peak)*22:.0f} m of travel — worth reconciling with the sync.")
    if ratio < 1.15:
        print("  ⚠️ the scan is nearly FLAT: this data cannot resolve the offset, so the "
              "peak location is not evidence either way.")
    if a.json:
        a.json.write_text(json.dumps(dict(dt=dts.tolist(), score=rows_s.tolist(),
                                          peak_dt=peak, ratio=ratio), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
