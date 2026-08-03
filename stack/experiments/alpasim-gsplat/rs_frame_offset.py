#!/usr/bin/env python3
"""Are we rendering the RIGHT INSTANT? A frame-alignment probe.

WHY
---
The phase sweep in `rs_sweep.py` is monotone in the wrong direction. Rendering earlier
keeps helping, and it does not stop at the shutter start:

    g_p1.00 (shutter END, production) 0.3241  ->  g_p0.50 0.3289  ->  g_p0.00 0.3313
    ->  g_pm0.50 (EXTRAPOLATED, half a readout BEFORE the shutter opens) 0.3332

"The shutter-start pose is the best single pose" cannot explain a point outside the
shutter interval scoring better than the interval's own endpoint. A **systematic offset
between the rig trajectory and the reference video** can. One readout is 30.559 ms and
one video frame is ~33 ms, so a **one-frame index offset and a one-readout pose offset
are nearly the same displacement** — which is exactly why the phase sweep alone cannot
tell them apart.

THE PROBE
---------
Render rig frame `f` **once**, exactly as production does (shutter-END pose, actors at
the shutter-END time), and score it against video frames `f-K … f+K`. Report the argmax.

* argmax at 0 for every frame -> alignment is right, and the phase result is a genuine
  (small) sub-frame pose preference.
* argmax consistently at the same non-zero offset -> the rig and the video are indexed
  differently, and **every render fidelity number in the programme has been measured
  against the wrong reference frame.**

This is a strictly harder negative control than the standard one: the competing frames
are the IMMEDIATE NEIGHBOURS, not frames spread across the clip.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from render_quality import CAM, build_renderer, grad_ncc, load_refs, parse_arm
from rs_sweep import CONFIGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="chosen", choices=sorted(CONFIGS))
    ap.add_argument("--k", type=int, default=3, help="offsets scanned: -k .. +k")
    ap.add_argument("--n-frames-auto", type=int, default=12)
    ap.add_argument("--loader-dir", default=None)
    a = ap.parse_args()

    if a.loader_dir:
        import sys
        sys.path.insert(0, a.loader_dir)
    scene = Path(a.scene_dir).expanduser()
    from nurec_loader import RigTrajectories
    rig = RigTrajectories(scene / "rig_trajectories.json")
    cam = rig.camera(CAM)
    n_clip = rig.n_frames(CAM)
    # keep k frames of headroom at both ends so every offset exists for every frame
    frames = sorted(set(int(round(x)) for x in
                        np.linspace(a.k, n_clip - 1 - a.k, a.n_frames_auto)))
    need = {f + d for f in frames for d in range(-a.k, a.k + 1)}
    refs = load_refs(scene / f"{CAM}.mp4", need, (int(cam.width), int(cam.height)))
    missing = sorted(need - set(refs))
    if missing:
        raise SystemExit(f"reference frames not decodable: {missing}")

    r, _ = build_renderer(scene, parse_arm(CONFIGS[a.config]), a.loader_dir)
    for _ in range(3):
        r.render(r.gt_cam_to_nre(frames[0]))

    # video timing, so the offset can be quoted in MILLISECONDS as well as frames
    t0 = rig.frame_timestamps_us(CAM, frames[0])
    t1 = rig.frame_timestamps_us(CAM, frames[0] + 1)
    frame_period_ms = (float(t1[1]) - float(t0[1])) / 1000.0
    readout_ms = (float(t0[1]) - float(t0[0])) / 1000.0

    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "config": a.config, "config_spec": CONFIGS[a.config], "k": a.k,
           "frames": frames, "clip_n_frames": n_clip,
           "frame_period_ms": round(frame_period_ms, 4),
           "readout_ms": round(readout_ms, 4),
           "note": ("render is PRODUCTION: shutter-END pose, actors at shutter-END time. "
                    "Only the reference frame index varies."),
           "per_frame": {}}
    offsets = list(range(-a.k, a.k + 1))
    argmax = []
    for f in frames:
        img = r.render(r.gt_cam_to_nre(f),
                       actor_time_us=float(r.frame_timestamps_us(f)[1]))[0]
        sc = {d: round(grad_ncc(img, refs[f + d]), 4) for d in offsets}
        best = max(sc, key=lambda d: sc[d])
        argmax.append(best)
        out["per_frame"][int(f)] = {"grad_ncc_by_offset": sc, "argmax_offset": best,
                                    "gain_vs_offset0": round(sc[best] - sc[0], 4)}
        print(f"[off] f{f:<4} " + "  ".join(f"{d:+d}:{sc[d]:.4f}" for d in offsets)
              + f"   argmax {best:+d}", flush=True)

    cnt = Counter(argmax)
    mean_by_off = {d: round(float(np.mean(
        [out["per_frame"][f]["grad_ncc_by_offset"][d] for f in frames])), 4)
        for d in offsets}
    out["argmax_histogram"] = {int(k): int(v) for k, v in sorted(cnt.items())}
    out["mean_grad_ncc_by_offset"] = mean_by_off
    out["best_mean_offset"] = int(max(mean_by_off, key=lambda d: mean_by_off[d]))
    out["mean_gain_at_best_offset"] = round(
        mean_by_off[out["best_mean_offset"]] - mean_by_off[0], 4)
    Path(a.out).expanduser().write_text(json.dumps(out, indent=1))

    print(f"\nframe period {frame_period_ms:.3f} ms, shutter readout {readout_ms:.3f} ms "
          f"({readout_ms / frame_period_ms:.3f} of a frame)")
    print(f"mean grad-NCC by reference offset: {mean_by_off}")
    print(f"argmax histogram over {len(frames)} frames: "
          f"{dict(sorted(cnt.items()))}")
    print(f"BEST mean offset = {out['best_mean_offset']:+d} frames "
          f"({out['mean_gain_at_best_offset']:+.4f} grad-NCC vs offset 0)")
    print("\nVERDICT INPUT: argmax pinned at 0 on every frame => alignment is correct and "
          "the phase result is a real sub-frame preference.\nA consistent non-zero argmax "
          "=> the rig trajectory and the reference video are indexed differently.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
