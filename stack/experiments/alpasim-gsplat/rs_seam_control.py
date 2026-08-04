#!/usr/bin/env python3
"""Is the sliced shutter losing quality because of GEOMETRY, or because of my SEAMS?

THE PROBLEM THIS EXISTS TO SOLVE
--------------------------------
`rs_sweep.py` measured grad-NCC falling MONOTONICALLY as the slice count rises
(s3 0.3274 -> s64 0.3146 on the deployed config) — i.e. the more faithfully the
rolling-shutter pose sweep is reproduced, the WORSE the render scores. Before that can
be read as "the pose sweep is worthless", one confound has to be removed:

    an N-slice composite has N-1 HORIZONTAL SEAMS, and grad-NCC is a GRADIENT metric.

Each seam is a one-row discontinuity that the reference does not contain, so the slice
count is confounded with the seam count and the two must be separated before either is
quoted.

THE CONTROL
-----------
Score every arm twice: on the full frame, and on the frame with +-`--mask` rows around
each of THAT ARM'S OWN band boundaries deleted. The baseline is re-scored on the
IDENTICAL deleted-row set, so the pair is like-for-like on the same pixels — masking
changes the metric's value, and comparing a masked arm to an unmasked baseline would
manufacture exactly the artifact we are trying to remove.

Read the result as:
* seam-masked delta stays negative  -> the pose sweep genuinely does not help.
* seam-masked delta goes to ~0/positive -> the decline was my compositing, and a
  feathered composite would recover it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from frame_align import scene_ref_offset
from render_quality import CAM, build_renderer, grad_ncc, load_refs, parse_arm
from rs_sweep import CONFIGS


def band_edges(H: int, n: int) -> np.ndarray:
    return np.linspace(0, H, n + 1).round().astype(int)


def keep_rows(H: int, n: int, mask: int) -> np.ndarray:
    """Row indices surviving a +-`mask` cut around every INTERIOR band boundary."""
    keep = np.ones(H, bool)
    for y in band_edges(H, n)[1:-1]:
        keep[max(0, y - mask):min(H, y + mask)] = False
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="chosen", choices=sorted(CONFIGS))
    ap.add_argument("--slices", default="2,4,8,16,32,64")
    ap.add_argument("--mask", type=int, default=3, help="rows cut each side of a seam")
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
    frames = sorted(set(int(round(x)) for x in
                        np.linspace(0, n_clip - 1, a.n_frames_auto)))
    ref_offset = scene_ref_offset(scene, n_clip)   # R-2026-08-03-k, per scene
    refs = load_refs(scene / f"{CAM}.mp4", frames, (int(cam.width), int(cam.height)),
                     ref_offset=ref_offset)
    r, _ = build_renderer(scene, parse_arm(CONFIGS[a.config]), a.loader_dir)
    H = r.height
    ns = [int(x) for x in a.slices.split(",")]

    for _ in range(3):
        r.render(r.gt_cam_to_nre(frames[0]))

    # baseline (shutter-END global) and every sliced arm, rendered once per frame
    base_img, sliced = {}, {n: {} for n in ns}
    for f in frames:
        cs, ce = r.gt_cam_to_nre_pair(f)
        ts0, ts1 = (float(x) for x in r.frame_timestamps_us(f)[:2])
        base_img[f] = r.render(ce, actor_time_us=ts1)[0]
        for n in ns:
            sliced[n][f] = r.render_rs_sliced(cs, ce, n, actor_time_us_start=ts0,
                                              actor_time_us_end=ts1)[0]
        print(f"[seam] rendered f{f}", flush=True)

    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "config": a.config, "config_spec": CONFIGS[a.config], "frames": frames,
           "mask_rows_each_side": a.mask, "height": H, "arms": []}
    for n in ns:
        keep = keep_rows(H, n, a.mask)
        full_a, full_b, msk_a, msk_b = [], [], [], []
        for f in frames:
            ref, ia, ib = refs[f], sliced[n][f], base_img[f]
            full_a.append(grad_ncc(ia, ref))
            full_b.append(grad_ncc(ib, ref))
            # identical row set for BOTH arms — that is the whole point of the control
            msk_a.append(grad_ncc(np.ascontiguousarray(ia[keep]),
                                  np.ascontiguousarray(ref[keep])))
            msk_b.append(grad_ncc(np.ascontiguousarray(ib[keep]),
                                  np.ascontiguousarray(ref[keep])))
        row = {"n_slices": n, "n_seams": n - 1,
               "rows_dropped": int((~keep).sum()),
               "full_sliced": round(float(np.mean(full_a)), 4),
               "full_baseline": round(float(np.mean(full_b)), 4),
               "full_delta": round(float(np.mean(np.array(full_a) - np.array(full_b))), 4),
               "masked_sliced": round(float(np.mean(msk_a)), 4),
               "masked_baseline": round(float(np.mean(msk_b)), 4),
               "masked_delta": round(float(np.mean(np.array(msk_a) - np.array(msk_b))), 4)}
        out["arms"].append(row)
        print(f"[seam] n={n:<3} seams={n - 1:<3} dropped={row['rows_dropped']:>4} rows | "
              f"FULL {row['full_sliced']:.4f} vs {row['full_baseline']:.4f} "
              f"({row['full_delta']:+.4f}) | MASKED {row['masked_sliced']:.4f} vs "
              f"{row['masked_baseline']:.4f} ({row['masked_delta']:+.4f})", flush=True)

    Path(a.out).expanduser().write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")
    print("\nVERDICT INPUT: if masked_delta stays negative as n grows, the pose sweep "
          "itself does not help.\nIf masked_delta rises to ~0 while full_delta falls, "
          "the decline was the seams, not the geometry.")


if __name__ == "__main__":
    main()
