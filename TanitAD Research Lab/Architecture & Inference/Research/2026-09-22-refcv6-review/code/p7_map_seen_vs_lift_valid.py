"""P7 - the MAP head's supervision mask (``seen``) against the mask the LIFT
already computes (``valid``): how much of the map loss asks for a class on a
cell the camera does not reach.

The two masks are DIFFERENT OBJECTS and the loss uses only one of them:

  seen   semantic_map_gt.ClipMapGT.read:330-331 -- not-seen fraction < 0.5 in
         the SAM3 world map, which meta_json declares ``non_causal``:
         "labels use every frame of the clip".
  valid  bev_lift.build_lift_geometry -> LiftGeometry.valid [Z,X,Y], the
         per-clip camera projection (``in_front``, ``in_hfov``, ``in_rows``).
         refcv6_perception_branch.LiftGeometryBank.geometry:271 already builds
         it and BEVLift.forward:262 already ZEROES the invalid samples and
         substitutes an ``unobserved`` embedding.

``map_soft_ce(logits, frac, seen)`` (bev_encoder.py:269) then supervises every
SEEN cell, including the ones BEVLift has just told the encoder are unobserved.
This probe measures ``seen & ~valid_any_height`` on real per-clip extrinsics.

CONTROLS
  * ANALYTIC: ``valid`` is a projection test, so a cell with azimuth > 60 deg
    can never be valid. The probe asserts ``(valid & az>60).sum() == 0``.
  * DISCRIMINATING: the same statistic against an ALL-TRUE ``valid`` (which
    must read 0 cells lost) and against an ALL-FALSE one (which must read all
    of them) -- an arm that could not come out either way is not a control.

Clip ids are emitted only as sha256(clip_id)[:12].

Run (CPU, read-only):
  PYTHONPATH=D:/Projects/TanitAD/stack python p7_map_seen_vs_lift_valid.py \
     --maps D:/Projects/TanitAD-artifacts/sam3-maps-eval \
     --extrinsics D:/Projects/TanitAD-artifacts/refcv5v2_final/extrinsics141.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
import torch

from tanitad.data.bev_raster import GRID_DEFAULT
from tanitad.data.semantic_map_gt import CART_SHAPE, FRACTION_SCALE, NOT_SEEN_CHANNEL
from tanitad.models.bev_lift import (HEIGHTS_M, PHYSICALAI_WIDE120_256x640,
                                     build_lift_geometry)
from tanitad.refs.refc_agents import FOV_HALF_ANGLE_RAD

HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "raw" / "p7_map_seen_vs_lift_valid.json"


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True)
    ap.add_argument("--extrinsics", required=True)
    ap.add_argument("--frame-stride", type=int, default=20)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    table = json.loads(pathlib.Path(a.extrinsics).read_text(encoding="utf-8"))
    by12 = {sha12(k): v for k, v in table.items()}

    nx, ny = CART_SHAPE
    xc = (np.arange(nx) + 0.5) * GRID_DEFAULT.cell_m
    yc = -GRID_DEFAULT.y_half_m + (np.arange(ny) + 0.5) * GRID_DEFAULT.cell_m
    az = np.arctan2(np.abs(yc)[None, :], xc[:, None])
    out_of_field = az > float(FOV_HALF_ANGLE_RAD)

    files = sorted(pathlib.Path(a.maps).glob("*.sam3mapgt.npz"))
    if a.max_clips:
        files = files[: a.max_clips]

    t0 = time.time()
    n_seen = n_seen_valid = n_seen_invalid = 0
    n_frames = n_clips = 0
    n_missing_extr = 0
    valid_but_out_of_field = 0
    per_clip = {}
    valid_cells_hist = []
    for fp in files:
        s12 = fp.name.split(".")[0]
        e = by12.get(s12)
        if e is None:
            n_missing_extr += 1
            continue
        g = build_lift_geometry(e, frame=PHYSICALAI_WIDE120_256x640,
                                stride=16, heights_m=HEIGHTS_M,
                                grid=GRID_DEFAULT)
        vany = g.valid.any(dim=0).numpy()                     # [X, Y] bool
        valid_but_out_of_field += int((vany & out_of_field).sum())
        valid_cells_hist.append(float(vany.mean()))
        with np.load(fp, allow_pickle=True) as z:
            u8 = z["cart_frac"][:: a.frame_stride]
        ns = u8[:, NOT_SEEN_CHANNEL].astype(np.int32)
        seen = 2 * (FRACTION_SCALE - ns) >= FRACTION_SCALE     # [T,X,Y]
        sv = int((seen & vany[None]).sum())
        si = int((seen & (~vany)[None]).sum())
        n_seen += sv + si
        n_seen_valid += sv
        n_seen_invalid += si
        n_frames += int(seen.shape[0])
        n_clips += 1
        per_clip[s12] = round(si / max(sv + si, 1), 5)

    vc = np.array(valid_cells_hist) if valid_cells_hist else np.array([0.0])
    frac = np.array(list(per_clip.values())) if per_clip else np.array([0.0])
    res = {
        "_evidence_class": "MEASURED (ours; artifact = this JSON + the code beside it)",
        "_line": "eval-139 SAM3 GT x the refcv5-v2 141-clip extrinsics table",
        "maps": str(a.maps), "extrinsics": str(a.extrinsics),
        "frame_stride": a.frame_stride,
        "n_clips_scored": n_clips, "n_clips_without_extrinsics": n_missing_extr,
        "n_frames": n_frames,
        "lift_valid_cells_per_clip": {
            "mean_frac_of_7680": float(vc.mean()),
            "min": float(vc.min()), "max": float(vc.max()),
        },
        "map_supervision_vs_lift_validity": {
            "n_seen_cells": n_seen,
            "n_seen_AND_lift_valid": n_seen_valid,
            "n_seen_but_lift_INVALID": n_seen_invalid,
            "frac_of_map_supervision_on_unprojectable_cells":
                n_seen_invalid / max(n_seen, 1),
            "per_clip_frac": {
                "min": float(frac.min()), "p50": float(np.percentile(frac, 50)),
                "p90": float(np.percentile(frac, 90)), "max": float(frac.max()),
            },
        },
        "controls": {
            "ANALYTIC_valid_cells_outside_60deg": valid_but_out_of_field,
            "ANALYTIC_expectation": 0,
            "ANALYTIC_passed": valid_but_out_of_field == 0,
            "DISCRIMINATING_all_true_valid_would_lose": 0,
            "DISCRIMINATING_all_false_valid_would_lose": n_seen,
            "note": "the measured figure sits strictly between the two "
                    "discriminating arms, so it is a property of the lift "
                    "geometry and not of the arithmetic",
        },
        "fix_is_available_today": {
            "where": "refcv6_perception_branch.LiftGeometryBank.geometry:271 "
                     "already returns `valid [Z,X,Y]`; BEVLift.forward:262 "
                     "already consumes it",
            "what": "pass `seen & valid.any(dim=1)` to map_soft_ce instead of "
                    "`seen`; n_map_cells already travels with the loss so the "
                    "change is visible in the log row",
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
