#!/usr/bin/env python3
"""WHERE the rolling-shutter 161x actually goes, and WHAT it changes — measured.

Two questions the quality sweep cannot answer, because it only sees finished frames:

**1. How many gaussians survive projection?**  `Cameras.cuh:357` shows the validity
handling is ASYMMETRIC:

    GLOBAL  -> `return {image_point_start, valid_start};`   validity KEPT
    ROLLING -> `return {image_points_rs_prev, true};`       validity DISCARDED

and `world_gaussian_to_image_gaussian_unscented_transform_shutter_pose` culls a gaussian
outright (`radii = 0`) if ANY of its 7 sigma points is invalid, because
`require_all_sigma_points_valid` defaults to True. So the rolling-shutter render may not
be "the same scene, swept" — it may be A DIFFERENT, LARGER SET OF GAUSSIANS. This probe
calls `fully_fused_projection_with_ut` directly and COUNTS them. A source reading is a
hypothesis; the count is the evidence.

**2. Is the cost pose-dependent or ordering/thermal?**  The sweep measured two ZERO-MOTION
rolling-shutter arms — geometrically identical to a global render, differing only in
which of the two shutter poses was used — at 155 ms and 1694 ms. An 11x spread between
two poses 0.48 m apart is not credible as physics until the alternative (clock ramping,
ordering) is excluded, so every configuration here is timed INTERLEAVED and repeated.

Usage:
    python rs_cost_probe.py --scene-dir <scene> --out results.json [--config chosen]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from render_quality import CAM, build_renderer, parse_arm
from rs_sweep import CONFIGS


def project_counts(r, cam_to_nre, cam_to_nre_end=None, require_all=True,
                   margin=None, near=0.05, far=2000.0):
    """Call gsplat's UT projection DIRECTLY and count what survives.

    Returns the number of gaussians with a non-zero radius (i.e. that will be binned
    into tiles and rasterized) plus the total screen area they claim, which is what the
    per-pixel blending loop actually pays for.
    """
    import torch
    from gsplat.cuda._wrapper import (RollingShutterType,
                                      fully_fused_projection_with_ut)
    p = r.ut_defaults()
    prev = (p.require_all_sigma_points_valid, p.in_image_margin_factor)
    p.require_all_sigma_points_valid = bool(require_all)
    if margin is not None:
        p.in_image_margin_factor = float(margin)
    try:
        K = torch.from_numpy(r._K_for(r.cam).copy()[None]).to(r.device)
        vm = torch.from_numpy(
            np.linalg.inv(cam_to_nre)[None].astype(np.float32)).to(r.device)
        kw = {}
        if cam_to_nre_end is not None:
            kw = dict(rolling_shutter=RollingShutterType[str(r.cam.shutter_type)],
                      viewmats_rs=torch.from_numpy(
                          np.linalg.inv(cam_to_nre_end)[None].astype(np.float32)
                      ).to(r.device))
        radii = fully_fused_projection_with_ut(
            r.means, r.quats, r.scales, r.opac, vm, K, r.width, r.height,
            eps2d=0.3, near_plane=near, far_plane=far, radius_clip=0.0,
            calc_compensations=False, camera_model="ftheta",
            ftheta_coeffs=r._ftheta_coeffs_for(r.cam), **kw)[0]
        rr = radii.reshape(-1, 2).float()
        alive = (rr > 0).all(-1)
        area = (4.0 * rr[:, 0] * rr[:, 1])[alive]
        return {"n_total": int(rr.shape[0]), "n_alive": int(alive.sum()),
                "frac_alive": round(float(alive.float().mean()), 5),
                "sum_bbox_area_px": float(area.sum()),
                "median_bbox_area_px": float(area.median()) if alive.any() else 0.0,
                "max_radius_px": float(rr[alive].max()) if alive.any() else 0.0}
    finally:
        p.require_all_sigma_points_valid, p.in_image_margin_factor = prev


def timed(fn, reps=3):
    """Median of `reps` calls, plus every sample, so a bimodal result is visible."""
    xs = []
    for _ in range(reps):
        t = time.time()
        fn()
        xs.append((time.time() - t) * 1000.0)
    return {"median_ms": round(float(np.median(xs)), 2),
            "samples_ms": [round(x, 2) for x in xs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="chosen", choices=sorted(CONFIGS))
    ap.add_argument("--frames", default="0,150,300,450")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--loader-dir", default=None)
    a = ap.parse_args()

    if a.loader_dir:
        import sys
        sys.path.insert(0, a.loader_dir)
    scene = Path(a.scene_dir).expanduser()
    frames = [int(x) for x in a.frames.split(",")]
    r, _ = build_renderer(scene, parse_arm(CONFIGS[a.config]), a.loader_dir)

    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "config": a.config, "config_spec": CONFIGS[a.config],
           "shutter_type": str(r.cam.shutter_type),
           "n_static_gaussians": int(r.n_gauss), "reps": a.reps, "frames": frames,
           "per_frame": {}}

    # warm every kernel variant before ANY number is recorded
    c0s, c0e = r.gt_cam_to_nre_pair(frames[0])
    for _ in range(3):
        r.render(c0e)
        r.render(c0s, cam_to_nre_end=c0e)
        r.render(c0e, cam_to_nre_end=c0e)

    for f in frames:
        cs, ce = r.gt_cam_to_nre_pair(f)
        ts = float(r.frame_timestamps_us(f)[1])
        cfgs = {
            "global_end": dict(pose=ce, end=None),
            "global_start": dict(pose=cs, end=None),
            "rs_zero_end": dict(pose=ce, end=ce),
            "rs_zero_start": dict(pose=cs, end=cs),
            "rs_sweep": dict(pose=cs, end=ce),
        }
        row = {}
        # INTERLEAVED: one rep of every config, then the next rep — so a clock ramp or a
        # thermal drift hits every config equally instead of penalising whichever ran first
        samples = {k: [] for k in cfgs}
        for _ in range(a.reps):
            for k, c in cfgs.items():
                t = time.time()
                r.render(c["pose"], actor_time_us=ts, cam_to_nre_end=c["end"])
                samples[k].append((time.time() - t) * 1000.0)
        for k in cfgs:
            row[k] = {"median_ms": round(float(np.median(samples[k])), 2),
                      "samples_ms": [round(x, 2) for x in samples[k]]}
        # ---- what the projection stage actually admits --------------------------
        row["counts"] = {
            "global_end_require_all": project_counts(r, ce),
            "global_end_any_sigma": project_counts(r, ce, require_all=False),
            "global_end_margin2.0": project_counts(r, ce, margin=2.0),
            "rs_zero_end": project_counts(r, ce, cam_to_nre_end=ce),
            "rs_sweep": project_counts(r, cs, cam_to_nre_end=ce),
        }
        out["per_frame"][int(f)] = row
        print(f"[cost] f{f:<4} " + "  ".join(
            f"{k}={row[k]['median_ms']:.0f}ms" for k in cfgs), flush=True)
        print(f"        alive: " + "  ".join(
            f"{k}={v['n_alive']}({v['frac_alive']:.3f})"
            for k, v in row["counts"].items()), flush=True)

    Path(a.out).expanduser().write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")

    # ---- summary across frames ----------------------------------------------------
    print(f"\n{'config':<22}{'median ms (over frames)':>26}")
    for k in ("global_end", "global_start", "rs_zero_end", "rs_zero_start", "rs_sweep"):
        v = [out["per_frame"][str(f) if str(f) in out["per_frame"] else f][k]["median_ms"]
             for f in frames]
        print(f"{k:<22}{np.median(v):>16.1f}   per-frame {[round(x) for x in v]}")
    print(f"\n{'projection gate':<26}{'n_alive':>10}{'frac':>8}{'sum bbox px':>16}")
    for k in out["per_frame"][frames[0]]["counts"]:
        vs = [out["per_frame"][f]["counts"][k] for f in frames]
        print(f"{k:<26}{int(np.median([x['n_alive'] for x in vs])):>10}"
              f"{np.median([x['frac_alive'] for x in vs]):>8.3f}"
              f"{np.median([x['sum_bbox_area_px'] for x in vs]):>16.3e}")


if __name__ == "__main__":
    main()
