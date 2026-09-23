"""EVALUATE D-3: how many map cells the lift mask removes, THROUGH THE FIXED CODE.

⛔ **THIS IS NOT A RE-DERIVATION OF THE REVIEW'S PREDICATE.** P7 measured ``seen & ~valid``
with its own arithmetic and got **11.048 %**. This pass runs the real
``refcv6_perception_branch.map_loss_row(..., lift_valid=map_valid_from_lift(valid))`` on
the real SAM3 GT and reads the COUNT THE LOSS ITSELF REPORTS -- ``n_map_cells`` and
``n_map_cells_unobserved`` -- so what is measured is what the trainer will compute.
Agreement with P7 is then an INDEPENDENT cross-check (different author, different code
path); a re-run of P7's own derivation would have measured determinism, not correctness.

CONTROLS, all four required:
  1. ANALYTIC. A cell whose azimuth exceeds 60 deg can never be lift-valid, so
     ``(valid & az>60).sum()`` must be EXACTLY 0 over every clip.
  2. DISCRIMINATING, both ends. An all-TRUE ``lift_valid`` must remove exactly 0 cells and
     an all-FALSE one must remove exactly all of them. A measurement that could not come
     out either way is not a measurement.
  3. IDENTITY vs the independent P7 artifact, to 3 decimal places.
  4. LOSS-SIDE. ``n_map_cells_seen - n_map_cells == n_map_cells_unobserved`` must hold on
     every batch, from the loss row's own fields -- the counts and the mask cannot disagree.

⛔ LINE: **eval-139** SAM3 GT (135 clips on this box) x the refcv5-v2 141-clip extrinsics
table. It is NOT a claim about the 4,572-clip v7-B1 train map release.
⛔ Clip ids appear only as sha256(clip_id)[:12] (the map files are already named that way).

Run (CPU, read-only, ~3 min):
  PYTHONPATH=D:/Projects/TanitAD/stack python eval_map_cells_recovered.py \
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
from tanitad.data.semantic_map_gt import (CART_SHAPE, FRACTION_SCALE,
                                          NOT_SEEN_CHANNEL)
from tanitad.models.bev_lift import (HEIGHTS_M, PHYSICALAI_WIDE120_256x640,
                                     build_lift_geometry)
from tanitad.models.refcv6_perception_branch import map_loss_row, map_valid_from_lift
from tanitad.refs.refc_agents import FOV_HALF_ANGLE_RAD

HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "raw" / "map_cells_recovered.json"
P7 = (HERE.parent / "2026-09-22-refcv6-review" / "raw"
      / "p7_map_seen_vs_lift_valid.json")


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
    out_of_field = torch.from_numpy(
        np.arctan2(np.abs(yc)[None, :], xc[:, None]) > float(FOV_HALF_ANGLE_RAD))

    files = sorted(pathlib.Path(a.maps).glob("*.sam3mapgt.npz"))
    if a.max_clips:
        files = files[: a.max_clips]

    t0 = time.time()
    n_clips = n_frames = n_missing = 0
    tot_seen = tot_kept = tot_removed = 0
    analytic_bad = 0
    disc_all_true_removed = disc_all_false_removed = 0
    loss_side_bad = 0
    per_clip: dict[str, float] = {}
    valid_frac = []

    for fp in files:
        s12 = fp.name.split(".")[0]
        e = by12.get(s12)
        if e is None:
            n_missing += 1
            continue
        g = build_lift_geometry(e, frame=PHYSICALAI_WIDE120_256x640, stride=16,
                                heights_m=HEIGHTS_M, grid=GRID_DEFAULT)
        # ⭐ THE FIX'S OWN FUNCTION, not a re-derivation.
        lv = map_valid_from_lift(g.valid.unsqueeze(0))[0]              # [X, Y] bool
        analytic_bad += int((lv & out_of_field).sum())
        valid_frac.append(float(lv.to(torch.float64).mean()))

        with np.load(fp, allow_pickle=True) as z:
            u8 = z["cart_frac"][:: a.frame_stride]
        ns = u8[:, NOT_SEEN_CHANNEL].astype(np.int32)
        seen = torch.from_numpy(2 * (FRACTION_SCALE - ns) >= FRACTION_SCALE)  # [T,X,Y]
        T, C = int(seen.shape[0]), int(u8.shape[1])
        frac = torch.from_numpy(u8.astype(np.float32) / float(FRACTION_SCALE))
        frac = frac / frac.sum(dim=1, keepdim=True).clamp_min(1e-6)
        logits = torch.zeros(T, C, nx, ny)

        lvb = lv.unsqueeze(0).expand(T, -1, -1)
        row = map_loss_row(logits, frac, seen, lift_valid=lvb)
        base = map_loss_row(logits, frac, seen)
        # control 2, both ends, through the same function
        t_all = map_loss_row(logits, frac, seen,
                             lift_valid=torch.ones_like(seen))
        f_all = map_loss_row(logits, frac, seen,
                             lift_valid=torch.zeros_like(seen))
        disc_all_true_removed += int(base["n_map_cells"] - t_all["n_map_cells"])
        disc_all_false_removed += int(base["n_map_cells"] - f_all["n_map_cells"])
        # control 4: the row's own three counts must be consistent
        loss_side_bad += int(row["n_map_cells_seen"] - row["n_map_cells"]
                             != row["n_map_cells_unobserved"])

        s, k = int(base["n_map_cells"]), int(row["n_map_cells"])
        tot_seen += s
        tot_kept += k
        tot_removed += s - k
        n_frames += T
        n_clips += 1
        per_clip[s12] = round((s - k) / max(s, 1), 5)

    p7 = None
    if P7.exists():
        try:
            d = json.loads(P7.read_text(encoding="utf-8"))
            p7 = d["map_supervision_vs_lift_validity"]
        except Exception:                                     # pragma: no cover
            p7 = None
    frac_removed = tot_removed / max(tot_seen, 1)
    pc = np.array(list(per_clip.values())) if per_clip else np.array([0.0])
    vf = np.array(valid_frac) if valid_frac else np.array([0.0])

    res = {
        "_evidence_class": "MEASURED (ours; artifact = this JSON + the code beside it)",
        "_line": "eval-139 SAM3 GT x the refcv5-v2 141-clip extrinsics table -- NOT a "
                 "claim about the 4,572-clip v7-B1 train map release",
        "_what": "the fix measured THROUGH map_loss_row's own reported counts",
        "maps": str(pathlib.Path(a.maps).name),
        "frame_stride": a.frame_stride,
        "n_clips_scored": n_clips, "n_clips_without_extrinsics": n_missing,
        "n_frames": n_frames,
        "cells": {
            "n_seen_supervised_BEFORE": tot_seen,
            "n_supervised_AFTER": tot_kept,
            "n_cells_REMOVED_as_unobserved": tot_removed,
            "frac_of_map_supervision_removed": frac_removed,
            "per_clip_frac_removed": {
                "min": float(pc.min()), "p50": float(np.percentile(pc, 50)),
                "p90": float(np.percentile(pc, 90)), "max": float(pc.max())},
            "lift_valid_cells_per_clip_frac_of_7680": {
                "mean": float(vf.mean()), "min": float(vf.min()),
                "max": float(vf.max())},
        },
        "controls": {
            "ANALYTIC_lift_valid_cells_outside_60deg": analytic_bad,
            "ANALYTIC_expectation": 0,
            "ANALYTIC_passed": analytic_bad == 0,
            "DISCRIMINATING_all_true_mask_removes": disc_all_true_removed,
            "DISCRIMINATING_all_true_expectation": 0,
            "DISCRIMINATING_all_false_mask_removes": disc_all_false_removed,
            "DISCRIMINATING_all_false_expectation": tot_seen,
            "DISCRIMINATING_passed": (disc_all_true_removed == 0
                                      and disc_all_false_removed == tot_seen),
            "LOSS_SIDE_rows_whose_three_counts_disagree": loss_side_bad,
            "LOSS_SIDE_passed": loss_side_bad == 0,
            "IDENTITY_vs_p7_independent_probe": (
                None if p7 is None else {
                    "p7_frac": p7["frac_of_map_supervision_on_unprojectable_cells"],
                    "ours_frac": frac_removed,
                    "p7_n_seen": p7["n_seen_cells"],
                    "ours_n_seen": tot_seen,
                    "agree_to_3dp": (round(frac_removed, 3) == round(
                        p7["frac_of_map_supervision_on_unprojectable_cells"], 3)),
                    "n_seen_identical": p7["n_seen_cells"] == tot_seen,
                }),
        },
        "wall_s": round(time.time() - t0, 1),
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    print(json.dumps({k: res[k] for k in
                      ("n_clips_scored", "n_frames", "cells", "controls", "wall_s")},
                     indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
