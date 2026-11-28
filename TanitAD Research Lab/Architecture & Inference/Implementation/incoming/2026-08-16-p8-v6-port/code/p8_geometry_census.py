"""Reproduce the P8-on-v6 geometry census (MEASURED, pure geometry, CPU, ~1 s).

Writes ``raw/p8_v6_geometry.json`` — the statement of how much of P8's
pre-registered Cartesian target grid the v6 trunk's camera field can observe at
all, and how the target lines up against v6's own readout cells.

Nothing here needs a checkpoint, a corpus or a GPU: the whole question is
``BEVGrid`` vs ``CanonicalFrame.half_angle_x_rad()``, both of which are
declared. That is deliberate — the mismatch had to be establishable BEFORE any
GPU time was scheduled, so a probe pointed at the wrong projection could be
caught by arithmetic rather than by a wasted run.

Usage (from stack/):
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
if str(_STACK) not in sys.path:
    sys.path.insert(0, str(_STACK))

from tanitad.data.bev_raster import GRID_DEFAULT, fov_census  # noqa: E402

#: The arms the census is reported for. Each is a REAL geometry in the
#: programme, not a sweep: v6F's training frame, the deployed v5 rig-clean
#: sub-frame, and the legacy square frame every pre-2026-07-27 number used.
ARMS = (
    {"name": "v6f_256x640_cyl120", "h": 256, "w": 640, "hfov_deg": 120.0,
     "projection": "cylindrical", "patch": 16, "readout": (4, 4),
     "note": "the v6 catalog frame (train_v6_staged defaults)"},
    {"name": "v5_subframe_176x624_cyl117", "h": 176, "w": 624,
     "hfov_deg": 117.0, "projection": "cylindrical", "patch": 16,
     "readout": (4, 4),
     "note": "the rig-clean centred sub-frame; 39 token cols onto 4 readout "
             "cols does NOT tile, so the column mapping must refuse"},
    {"name": "legacy_256x256_pinhole", "h": 256, "w": 256, "hfov_deg": None,
     "f_ref": 266.0, "projection": "pinhole", "patch": 16, "readout": (4, 4),
     "note": "CANONICAL_256 — every pre-2026-07-27 published number"},
)


def half_angle_rad(arm) -> float:
    """Retained horizontal half-angle, by the same two formulas calib uses
    (calib.py:88-90): pinhole ``atan((W/2)/f_ref)``, cylindrical ``(W/2)/f_ref``.
    An explicit ``hfov_deg`` short-circuits both (it IS the answer)."""
    if arm.get("hfov_deg") is not None:
        return math.radians(0.5 * float(arm["hfov_deg"]))
    r = (arm["w"] / 2.0) / float(arm["f_ref"])
    return math.atan(r) if arm["projection"] == "pinhole" else r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("p8_geometry_census")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "raw"))
    a = ap.parse_args(argv)
    out = {"_what": "P8's Cartesian BEV target grid vs the trunk's camera "
                    "field and v6's image-plane readout cells",
           "_evidence_class": "MEASURED (ours; pure geometry — no corpus, no "
                              "checkpoint, no GPU)",
           "_generated_by": "code/p8_geometry_census.py",
           "target_grid": {"x_fwd_m": GRID_DEFAULT.x_fwd_m,
                           "y_half_m": GRID_DEFAULT.y_half_m,
                           "cell_m": GRID_DEFAULT.cell_m,
                           "shape": list(GRID_DEFAULT.shape),
                           "frame": "ego, +x forward, +y LEFT",
                           "row0": "ego origin (NEAREST)"},
           "arms": {}}
    for arm in ARMS:
        gh, gw = arm["readout"]
        token_w = arm["w"] // arm["patch"]
        rep = fov_census(grid=GRID_DEFAULT,
                         half_angle_rad=half_angle_rad(arm),
                         n_cols=gw, projection=arm["projection"],
                         token_w=token_w, readout_rows=gh)
        rep["arm"] = {k: v for k, v in arm.items()}
        rep["arm"]["token_grid"] = [arm["h"] // arm["patch"], token_w]
        out["arms"][arm["name"]] = rep
        print(f"[census] {arm['name']}: {rep['out_of_fov_cells']}/"
              f"{rep['total_cells']} cells out of field "
              f"({100 * rep['out_of_fov_frac']:.3f} %), first fully-visible "
              f"row {rep['first_fully_visible_row']}, columns exact="
              f"{rep['readout_columns']['exact']}")
    p = Path(a.out) / "p8_v6_geometry.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[census] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
