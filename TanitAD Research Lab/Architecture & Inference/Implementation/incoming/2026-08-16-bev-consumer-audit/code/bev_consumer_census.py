"""Per-CONSUMER camera-field census for `tanitad.data.bev_raster` (MEASURED, CPU, ~1 s).

The P8 v6 port established the WHOLE-GRID mismatch (590/7 680 = 7.682 % of P8's
Cartesian target lies outside a 120 deg camera; 2 126/7 680 = 27.682 % on the
legacy square frame). It explicitly did NOT audit the module's other consumers.

This reproducer answers the different question each consumer actually poses:

* ``lf0_bev_lead.py`` never scores the whole grid. It walks a HAND-DEFINED ego
  corridor (+-1.0/1.5/2.0 m, `CORRIDOR_M`) forward from ``--min-row`` (default
  2) and returns the range of the first cell >= tau. The number that decides
  whether LF0's banked verdict moves is therefore the out-of-field count
  **inside that corridor**, not over the grid — and it is a much smaller set
  (a few hundred cells, all of them near).
* ``p8_bev_reel.py`` DRAWS every cell and captions an all-cells IoU, so its
  exposure IS the whole-grid number — but as a *rendering* fact (pixels a
  viewer reads as scene) rather than a scoring one.

⛔ WHY THE CORRIDOR IS THE RIGHT DENOMINATOR FOR LF0. Out-of-field cells are all
NEAR (max x 8.75 m at 120 deg) and all off-axis: a cell at lateral |y| leaves a
half-angle ``th`` field when ``x < |y| / tan(th)``. The corridor is |y| <= 1.75 m
at its widest, so the corridor's own exposure is bounded by ``1.75 / tan(th)``
metres of range — 1.07 m at 117 deg but 3.64 m at the legacy 51.4 deg frame.
Quoting the grid-wide 8.151 % against LF0 would OVERSTATE its exposure by more
than an order of magnitude; quoting zero without checking would understate the
legacy case. Both failures are avoided by measuring the corridor directly.

Nothing here needs a checkpoint, a corpus or a GPU.

Usage (from anywhere):
    PYTHONUTF8=1 python "<this file>" --out <raw dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# stack/ root — this file lives in the hub, so the stack is located explicitly.
_STACK = Path(__file__).resolve().parents[6] / "stack"
for _p in (str(_STACK), str(_STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from tanitad.data.bev_raster import (GRID_DEFAULT, cell_centers_xy,  # noqa: E402
                                     fov_census, fov_mask)
from lf0_bev_lead import CORRIDOR_M, corridor_cols  # noqa: E402

#: The three real frames. `v5f_subframe` is DERIVED from its parent through the
#: same `centred_subframe` the trainer/evaluator seam uses (`resolve_v2_frames`,
#: train_flagship_v4.py:795-812) rather than hand-entered as "117 deg" — the
#: sub-frame's half-angle is a property of the parent's `f_ref`, and typing the
#: rounded number is exactly how a geometry drifts.
def _frames() -> dict:
    from tanitad.data.calib import (CANONICAL_256, CanonicalFrame,
                                    centred_subframe)
    v6f = CanonicalFrame.from_hfov(120.0, 256, 640, projection="cylindrical")
    return {
        # v6F's training frame (train_v6_staged catalog defaults)
        "v6f_256x640_cyl120": v6f,
        # the deployed v5 rig-clean sub-frame — the frame p8-occupancy-c and
        # LF0 actually ran on (recovered from p8c_chain.sh / lf0_chain.sh)
        "v5f_subframe_176x624": centred_subframe(v6f, 176, 624),
        # every pre-2026-07-27 published number
        "legacy_256x256_pinhole": CANONICAL_256,
    }


#: LF0's own defaults, imported where possible so this cannot drift from the
#: script. `min_row` is lf0_bev_lead's argparse default; `0` is included so the
#: guard's contribution is separable from the geometry's.
LF0_MIN_ROWS = (0, 2)


def corridor_report(frame, grid=GRID_DEFAULT) -> dict:
    """LF0's corridor cells vs ``frame``'s horizontal field.

    Reports, per corridor width and per ``min_row``: how many of the cells LF0
    actually SCANS are outside the field, the deepest such cell, and the first
    row from which the whole corridor is inside. A corridor whose scanned set
    is fully in-field cannot have its read changed by the mask — that is the
    fact that decides whether LF0's banked verdict moves.
    """
    nx, ny = grid.shape
    m = fov_mask(grid, frame.half_angle_x_rad())
    x, y = cell_centers_xy(grid)
    out = {"_half_angle_deg": round(math.degrees(frame.half_angle_x_rad()), 6),
           "_hfov_deg": round(frame.hfov_deg, 6),
           "_projection": frame.projection, "widths": {}}
    for w in CORRIDOR_M:
        cols = corridor_cols(ny, grid.y_half_m, grid.cell_m, w)
        sub = m[:, cols]                                  # [nx, |cols|]
        rows_all_in = [int(i) for i in range(nx) if bool(sub[i].all())]
        per_min_row = {}
        for mr in LF0_MIN_ROWS:
            scanned = sub[mr:]
            n_out = int((~scanned).sum())
            xs = x[:, cols][mr:][~scanned]
            per_min_row[str(mr)] = {
                "scanned_cells": int(scanned.size),
                "out_of_fov_cells": n_out,
                "out_of_fov_frac": round(n_out / max(scanned.size, 1), 6),
                "out_of_fov_max_x_m": (float(xs.max()) if n_out else None),
                "read_can_change": bool(n_out),
            }
        out["widths"][str(w)] = {
            "n_cols": int(len(cols)), "col_idx": [int(c) for c in cols],
            "max_abs_y_m": round(float(np.abs(y[0, cols]).max()), 6),
            "first_row_fully_in_fov": (rows_all_in[0] if rows_all_in else None),
            "first_row_fully_in_fov_x_m": (float(x[rows_all_in[0], 0])
                                           if rows_all_in else None),
            "per_min_row": per_min_row,
        }
    return out


def reel_report(frame, grid=GRID_DEFAULT) -> dict:
    """What ``p8_bev_reel.py`` DRAWS that the camera never observed.

    The reel renders the full ``[nx, ny]`` raster in three panes and captions a
    per-frame IoU over all cells. Its exposure is therefore the whole-grid
    number — reported here as a rendering fact: these are pixels a viewer reads
    as "the world model's belief about the scene".
    """
    m = fov_mask(grid, frame.half_angle_x_rad())
    nx, ny = grid.shape
    total = int(m.size)
    n_out = total - int(m.sum())
    return {
        "cells_drawn_per_pane": total,
        "cells_drawn_outside_field": n_out,
        "frac_of_pane_outside_field": round(n_out / total, 6),
        "panes_affected": ["decode(z_hat) — the WM's belief",
                           "belief ∩ truth overlay"],
        "caption_iou_cell_set": "ALL cells (unmasked) — p8_bev_reel.py:240-242",
        "_note": ("the GT pane is unaffected as a LABEL (the join's cuboids are "
                  "real wherever they are), but the BELIEF pane draws decoder "
                  "output on cells no camera in the rig can observe, and the "
                  "captioned IoU scores them"),
        "_render_aspect": {
            "grid_m": [grid.x_fwd_m, 2.0 * grid.y_half_m],
            "pane_px": [320, None],
            "_note": ("compose_frame resizes [nx, ny] to (pane_w, PANE_H) with "
                      "Image.NEAREST, so the BEV pane is NOT metric-square: at "
                      "width=1280 the pane is ~417x320 px for a 32 m x 60 m "
                      "grid = 13.0 px/m across vs 5.3 px/m forward, a 2.44x "
                      "horizontal stretch"),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("bev_consumer_census", description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "raw"))
    a = ap.parse_args(argv)
    frames = _frames()
    rep = {
        "_what": ("per-CONSUMER camera-field exposure of tanitad.data.bev_raster "
                  "— the audit the P8 v6 port deferred"),
        "_evidence_class": ("MEASURED (ours; pure geometry — no corpus, no "
                            "checkpoint, no GPU)"),
        "_generated_by": "code/bev_consumer_census.py",
        "frames": {n: {"h": f.height, "w": f.width,
                       "hfov_deg": round(f.hfov_deg, 6),
                       "half_angle_deg": round(
                           math.degrees(f.half_angle_x_rad()), 6),
                       "projection": f.projection,
                       "f_ref": round(f.f_ref, 6)}
                   for n, f in frames.items()},
        "whole_grid": {n: fov_census(GRID_DEFAULT, f.half_angle_x_rad(),
                                     projection=f.projection)
                       for n, f in frames.items()},
        "lf0_bev_lead": {
            "_scored_set": ("the ego corridor only — NOT the whole grid; "
                            "lf0_bev_lead.read_lead_range walks rows >= "
                            "min_row inside corridor_cols(width)"),
            "_defaults": {"min_row": 2, "tau": 0.7,
                          "headline_corridor_m": 1.5,
                          "corridors_m": list(CORRIDOR_M)},
            "_run_frame": ("v5f_subframe_176x624 — RECOVERED from "
                           "scripts/lf0_chain.sh:93-94 (--frame-h 256 "
                           "--frame-w 640 --frame-hfov 120 --projection "
                           "cylindrical --v2-subframe 176x624)"),
            "arms": {n: corridor_report(f) for n, f in frames.items()},
        },
        "p8_bev_reel": {
            "_scored_set": ("the WHOLE grid — every cell is rendered and the "
                            "caption IoU is unmasked"),
            "_run_frame": ("v5f_subframe_176x624 — the docstring's own "
                           "invocation (p8_bev_reel.py:24-25) and the run it "
                           "visualises (p8-occupancy-c, p8c_chain.sh)"),
            "arms": {n: reel_report(f) for n, f in frames.items()},
        },
    }
    outp = Path(a.out)
    outp.mkdir(parents=True, exist_ok=True)
    dest = outp / "bev_consumer_geometry.json"
    dest.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"[bev-audit] wrote {dest}")
    for n in frames:
        c = rep["lf0_bev_lead"]["arms"][n]["widths"]["1.5"]["per_min_row"]["2"]
        g = rep["whole_grid"][n]
        print(f"  {n:26s} grid {g['out_of_fov_cells']:5d}/"
              f"{g['total_cells']} ({100 * g['out_of_fov_frac']:6.3f} %) · "
              f"LF0 corridor@1.5/min_row2 {c['out_of_fov_cells']:4d}/"
              f"{c['scanned_cells']} ({100 * c['out_of_fov_frac']:6.3f} %) · "
              f"read_can_change={c['read_can_change']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
