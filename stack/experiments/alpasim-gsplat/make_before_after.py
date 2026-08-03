#!/usr/bin/env python3
"""BEFORE/AFTER side-by-side at MATCHED frames, built from a `render_quality.py` run dir.

Reads the `render_<arm>_f<frame>.npz` files a quality run banks (render, alpha, ref) and
lays out one row per frame:  BEFORE | AFTER | REFERENCE | alpha(BEFORE) | alpha(AFTER).

The header prints the run directory and both arms' headline numbers, because the failure
this guards against is a headline table copied from a superseded run — the image must
carry its own provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def label_bar(width, text, h=44, scale=0.85):
    import cv2
    bar = np.zeros((h, width, 3), np.uint8)
    cv2.putText(bar, text, (12, int(h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)
    return bar


def alpha_img(a):
    import cv2
    return cv2.applyColorMap((np.clip(a, 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_VIRIDIS)[:, :, ::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()

    import cv2
    run = Path(a.run_dir)
    rep = json.loads((run / "report.json").read_text())
    arms = {s["arm"]: s for s in rep["arms"]}
    for nm in (a.before, a.after):
        if nm not in arms:
            raise SystemExit(f"arm {nm!r} not in {run}; have {sorted(arms)}")

    npz = sorted(run.glob(f"render_{a.before}_f*.npz"))
    if not npz:
        raise SystemExit(f"no render_{a.before}_f*.npz in {run}")
    rows, frames = [], []
    for pb in npz:
        f = pb.stem.split("_f")[-1]
        pa = run / f"render_{a.after}_f{f}.npz"
        if not pa.exists():
            continue
        B, A = np.load(pb), np.load(pa)
        imgs = [B["render"], A["render"], B["ref"], alpha_img(B["alpha"]),
                alpha_img(A["alpha"])]
        if a.scale != 1.0:
            imgs = [cv2.resize(x, None, fx=a.scale, fy=a.scale,
                               interpolation=cv2.INTER_AREA) for x in imgs]
        pad = np.full((imgs[0].shape[0], 6, 3), 255, np.uint8)
        strip = np.concatenate([x for p in imgs for x in (p, pad)][:-1], 1)
        w = strip.shape[1]
        rows.append(np.concatenate([
            label_bar(w, f"frame {f}     BEFORE ({a.before})  |  AFTER ({a.after})  |  "
                         f"REFERENCE (NuRec mp4)  |  alpha BEFORE  |  alpha AFTER",
                      h=36, scale=0.7),
            strip], 0))
        frames.append(f)
    if not rows:
        raise SystemExit("no matched frames between the two arms")

    sb, sa = arms[a.before], arms[a.after]
    w = rows[0].shape[1]
    head = np.concatenate([
        label_bar(w, "NuRec render quality — BEFORE vs AFTER (matched frames)", h=52,
                  scale=1.0),
        label_bar(w, f"run_dir {rep['run_dir']}   scene {Path(rep['scene_dir']).name}   "
                     f"metric: gradient-NCC (PSNR/NCC retracted on this clip)", h=34,
                  scale=0.62),
        label_bar(w, f"BEFORE {a.before:<22} gradNCC {sb['grad_ncc_mean']:.4f}  "
                     f"neg-control margin {sb['neg_margin_mean']:+.4f}  "
                     f"mean_alpha {sb['mean_alpha']:.4f}  MAE {sb['mae_full']:.4f}  "
                     f"{sb['raster_ms_median']:.0f} ms/frame", h=34, scale=0.62),
        label_bar(w, f"AFTER  {a.after:<22} gradNCC {sa['grad_ncc_mean']:.4f}  "
                     f"neg-control margin {sa['neg_margin_mean']:+.4f}  "
                     f"mean_alpha {sa['mean_alpha']:.4f}  MAE {sa['mae_full']:.4f}  "
                     f"{sa['raster_ms_median']:.0f} ms/frame", h=34, scale=0.62),
        label_bar(w, f"DELTA  gradNCC {sa['grad_ncc_mean'] - sb['grad_ncc_mean']:+.4f} "
                     f"({100 * (sa['grad_ncc_mean'] / sb['grad_ncc_mean'] - 1):+.1f} %)   "
                     f"mean_alpha {sa['mean_alpha'] - sb['mean_alpha']:+.4f}   "
                     f"MAE {sa['mae_full'] - sb['mae_full']:+.4f} "
                     f"({100 * (sa['mae_full'] / sb['mae_full'] - 1):+.1f} %)", h=34,
                  scale=0.62)], 0)
    canvas = np.concatenate([head] + rows, 0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), canvas[:, :, ::-1])
    info = {"out": a.out, "size": [int(canvas.shape[1]), int(canvas.shape[0])],
            "frames": frames, "run_dir": rep["run_dir"],
            "before": a.before, "after": a.after,
            "bytes": Path(a.out).stat().st_size}
    # verify by DECODING, never by exit code
    back = cv2.imread(a.out)
    info["decoded_ok"] = back is not None and back.shape[:2] == canvas.shape[:2]
    print(json.dumps(info, indent=1))
    if not info["decoded_ok"]:
        raise SystemExit("PNG VERIFICATION FAILED — the file does not decode")


if __name__ == "__main__":
    main()
