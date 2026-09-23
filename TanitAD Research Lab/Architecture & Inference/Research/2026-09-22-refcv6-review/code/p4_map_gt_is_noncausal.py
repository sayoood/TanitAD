"""P4 - the SAM3 MAP GT's ``seen`` is NOT the current frame's camera field.

MEASURED, from the artifact's own meta:

    "non_causal": "labels use every frame of the clip; inference must never
                   read this file"
    "source": {"map": "SAM3 front-camera map r ...", "ground": "ego path"}

so a cell is ``seen`` if the clip EVER saw it, not if THIS frame's 120 deg rig
camera sees it. That is legal for a LABEL (PI 2026-08-03: labels may use
anything; inference is vision-only) and it is exactly the condition
``refc_agents.filter_targets_to_visible`` refuses to train the BOX head on:
*"Without this filter a monocular head is trained to hallucinate on ~62 % of
its supervision"* (refc_agents.py:353-356).

This probe prices the map head's analogue of that number. For every cell of the
120 x 64 rig grid it computes the INSTANTANEOUS azimuth
``atan2(|y|, x)`` and splits the supervised (seen) cells into

  IN-FIELD   azimuth <= 60 deg   (refc_agents.FOV_HALF_ANGLE_RAD, the rig's own
                                  ``camera_front_wide_120fov``)
  OUT-FIELD  azimuth >  60 deg   -- supervised, and unobservable at t

ANALYTIC TARGET (the strongest discriminator available here): the geometry is
fixed, so the out-of-field cell COUNT is known in closed form before any file is
opened -- a cell (x, y) is out of field iff |y| > x*tan(60 deg). The probe
prints the analytic count beside the measured one; they must agree exactly.

DISCRIMINATING CONTROL: the same split computed on an ALL-TRUE mask. If the
seen mask were already the camera field, the two would agree; the gap between
them is what the mask does NOT remove.

Run (CPU, read-only):
  PYTHONPATH=D:/Projects/TanitAD/stack python p4_map_gt_is_noncausal.py \
      --root D:/Projects/TanitAD-artifacts/sam3-maps-eval
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np

from tanitad.data.semantic_map_gt import (CART_SHAPE, CHANNELS, FRACTION_SCALE,
                                          NOT_SEEN_CHANNEL)
from tanitad.refs.refc_agents import FOV_HALF_ANGLE_RAD

HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "raw" / "p4_map_gt_noncausal.json"
CELL_M = 0.5
Y_HALF = 16.0


def cell_azimuth_grid() -> np.ndarray:
    nx, ny = CART_SHAPE
    xc = (np.arange(nx) + 0.5) * CELL_M                       # 0.25 .. 59.75
    yc = -Y_HALF + (np.arange(ny) + 0.5) * CELL_M             # -15.75 .. 15.75
    X = xc[:, None]
    Y = yc[None, :]
    return np.arctan2(np.abs(Y), X)                           # [nx, ny]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    az = cell_azimuth_grid()
    infield = az <= float(FOV_HALF_ANGLE_RAD)
    nx, ny = CART_SHAPE

    # ---- the ANALYTIC target, before any file is opened ------------------- #
    tan60 = math.tan(FOV_HALF_ANGLE_RAD)
    n_out_analytic = 0
    for i in range(nx):
        x = (i + 0.5) * CELL_M
        for j in range(ny):
            y = -Y_HALF + (j + 0.5) * CELL_M
            if abs(y) > x * tan60:
                n_out_analytic += 1

    files = sorted(pathlib.Path(a.root).glob("*.sam3mapgt.npz"))
    if a.max_clips:
        files = files[: a.max_clips]

    t0 = time.time()
    n_seen_in = n_seen_out = 0
    n_frames = 0
    metas_noncausal = 0
    metas_read = 0
    mass_out = np.zeros(len(CHANNELS))
    mass_in = np.zeros(len(CHANNELS))
    for fp in files:
        with np.load(fp, allow_pickle=True) as z:
            u8 = z["cart_frac"][:: a.frame_stride]
            try:
                m = json.loads(str(z["meta_json"]))
                metas_read += 1
                if m.get("non_causal"):
                    metas_noncausal += 1
            except Exception:
                pass
        ns = u8[:, NOT_SEEN_CHANNEL].astype(np.int32)
        seen = 2 * (FRACTION_SCALE - ns) >= FRACTION_SCALE      # [T,nx,ny]
        n_frames += int(seen.shape[0])
        n_seen_in += int((seen & infield[None]).sum())
        n_seen_out += int((seen & (~infield)[None]).sum())
        p = u8.astype(np.float64) / FRACTION_SCALE
        mass_in += (p * (seen & infield[None])[:, None]).sum(axis=(0, 2, 3))
        mass_out += (p * (seen & (~infield)[None])[:, None]).sum(axis=(0, 2, 3))

    n_cells_total = n_frames * nx * ny
    n_seen = n_seen_in + n_seen_out
    mi, mo = float(mass_in.sum()) or 1.0, float(mass_out.sum()) or 1.0

    res = {
        "_evidence_class": "MEASURED (ours; artifact = this JSON + the code beside it)",
        "_line": "eval-139 (135 banked SAM3 GT clips) -- NOT the v7-B1 train corpus",
        "root": str(a.root),
        "n_clip_files": len(files), "frame_stride": a.frame_stride,
        "n_frames_read": n_frames,
        "meta_says_non_causal": {"n_metas_read": metas_read,
                                 "n_with_non_causal_field": metas_noncausal},
        "grid_geometry": {
            "shape": list(CART_SHAPE), "cell_m": CELL_M, "y_half_m": Y_HALF,
            "x_max_m": nx * CELL_M,
            "fov_half_deg": math.degrees(FOV_HALF_ANGLE_RAD),
            "fov_src": "refc_agents.FOV_HALF_ANGLE_RAD",
            "n_cells_per_frame": nx * ny,
            "n_cells_out_of_field_ANALYTIC": n_out_analytic,
            "n_cells_out_of_field_MEASURED": int((~infield).sum()),
            "analytic_matches_measured":
                n_out_analytic == int((~infield).sum()),
            "frac_cells_out_of_field": n_out_analytic / (nx * ny),
        },
        "supervised_cells": {
            "n_seen": n_seen,
            "frac_of_all_cells_seen": n_seen / max(n_cells_total, 1),
            "n_seen_in_field": n_seen_in,
            "n_seen_OUT_of_field": n_seen_out,
            "frac_of_SUPERVISION_out_of_field": n_seen_out / max(n_seen, 1),
            "control_all_true_mask_frac_out_of_field":
                n_out_analytic / (nx * ny),
            # ⭐ THE NEAR-ANALYTIC PROOF OF TEMPORAL ACCUMULATION. A cell with
            # azimuth > 60 deg is outside the rig's ONLY camera at the current
            # instant, at every instant -- so if it is labelled `seen`, the
            # label came from a DIFFERENT frame of the clip. The denominator is
            # exact (n_frames * 590); no model and no image is involved.
            "frac_of_OUT_OF_FIELD_cells_that_are_seen":
                n_seen_out / max(n_frames * n_out_analytic, 1),
            "frac_of_IN_FIELD_cells_that_are_seen":
                n_seen_in / max(n_frames * (nx * ny - n_out_analytic), 1),
            "reading": "if frac_of_SUPERVISION_out_of_field is close to the "
                       "all-true control, the seen mask removes essentially "
                       "NONE of the instantaneously-unobservable cells: it is a "
                       "clip-lifetime visibility mask, not a camera-field one",
        },
        "label_mass_shares": {
            CHANNELS[i]: {"in_field": float(mass_in[i] / mi),
                          "out_of_field": float(mass_out[i] / mo)}
            for i in range(len(CHANNELS))
        },
        "wall_s": round(time.time() - t0, 1),
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
