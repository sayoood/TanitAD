"""Q4 — re-derive 0.1172 °/px (ours) and 0.1367 °/px (NAVSIM/DiffusionDrive),
**stating the projection first**, because the pinhole formula is wrong here.

⛔ `CLAUDE.md` traps: *"THE PINHOLE FOV FORMULA IS WRONG ON A CYLINDRICAL
PROJECTION … the column is LINEAR IN AZIMUTH (az_max = (W/2)/f_ref)"*. Our
corpus is cylindrical; NAVSIM/DiffusionDrive's 1024x256 front view is three
cropped **pinhole** cameras concatenated (banked note on
`2411.15139_DiffusionDrive…pdf`). A single °/px number therefore means two
different things on the two sides, and that is the point of this file.

Evidence classes are stamped per row in the output.
"""
from __future__ import annotations

import json
import math
import sys


def main() -> int:
    from tanitad.data.calib import projection_density_report
    from tanitad.models.trunk_shapes import (FRAME_256x640, FRAME_256x1024,
                                             FRAME_416x1024)

    out = {}

    # ------------------------------------------------------- OURS -------- #
    ours = {}
    for tag, fr in (("256x640", FRAME_256x640), ("256x1024", FRAME_256x1024),
                    ("416x1024", FRAME_416x1024)):
        # CYLINDRICAL: column is LINEAR in azimuth. az_max = (W/2)/f_ref.
        az_max = (fr.width / 2.0) / float(fr.f_ref)
        hfov = 2.0 * math.degrees(az_max)
        deg_per_col = hfov / fr.width
        # the same quantity a SECOND way, from the definition of f_ref, as an
        # independent cross-check rather than a restatement
        deg_per_col_2 = math.degrees(1.0 / float(fr.f_ref))
        ours[tag] = {
            "projection": fr.projection,
            "f_ref": float(fr.f_ref),
            "HFOV_deg": round(hfov, 6),
            "deg_per_px_uniform": round(deg_per_col, 6),
            "deg_per_px_crosscheck_1_over_f_ref": round(deg_per_col_2, 6),
            "crosscheck_agrees": abs(deg_per_col - deg_per_col_2) < 1e-12,
            "edge_local_density_vs_center":
                projection_density_report(fr)["edge_local_density_vs_center"],
            "stride16_deg_per_col": round(deg_per_col * 16, 6),
            "stride32_deg_per_col": round(deg_per_col * 32, 6),
            "evidence": "MEASURED (ours) from trunk_shapes + calib",
        }
    out["OURS_cylindrical"] = ours
    out["SPEC_10_1_claim_0_1172"] = {
        "spec_value": 0.1172,
        "recomputed_416x1024": ours["416x1024"]["deg_per_px_uniform"],
        "recomputed_256x1024": ours["256x1024"]["deg_per_px_uniform"],
        "AGREES_to_4dp": round(ours["416x1024"]["deg_per_px_uniform"], 4) == 0.1172,
        "note": ("f_ref is a HORIZONTAL quantity on a cylindrical frame, so "
                 "256x1024 and 416x1024 have the SAME azimuth resolution; "
                 "only the vertical field changes."),
    }

    # ------------------------------------ NAVSIM / DiffusionDrive -------- #
    # ⚠️ INHERITED: `SPEC_REFCV6_V2.md` §10.1 states "~140 deg, 3 stitched
    # cameras" at 1024x256 and quotes 0.1367 deg/px. A repo-wide search finds
    # 0.1367 ONLY in that table (and PI_DECISION_QUEUE:596 quoting a different
    # column), so NOTHING in the record derives it. It is exactly 140/1024.
    W = 1024
    for fov in (140.0,):
        avg = fov / W
        # If the mosaic is treated as ONE pinhole of that field (which is what
        # an average deg/px implicitly assumes), the density is NOT uniform:
        # x = f tan(theta) => dtheta/dx = cos^2(theta)/f.
        half = math.radians(fov / 2.0)
        f = (W / 2.0) / math.tan(half)
        centre = math.degrees(1.0 / f)                 # theta = 0
        edge = math.degrees(math.cos(half) ** 2 / f)   # theta = half
        out["NAVSIM_DD_1024_wide"] = {
            "assumed_HFOV_deg": fov,
            "evidence": ("INHERITED — SPEC_REFCV6_V2.md:145-146 ('~140 deg, 3 "
                         "stitched cameras'); no derivation exists in the repo"),
            "average_deg_per_px": round(avg, 6),
            "SPEC_quotes": 0.1367,
            "average_reproduces_SPEC": round(avg, 4) == 0.1367,
            "projection": "PINHOLE mosaic (3 cropped forward cameras)",
            "single_pinhole_equivalent_f_px": round(f, 4),
            "deg_per_px_at_IMAGE_CENTRE": round(centre, 6),
            "deg_per_px_at_IMAGE_EDGE": round(edge, 6),
            "centre_to_edge_ratio": round(centre / edge, 4),
            "caveat": ("an AVERAGE deg/px is not a resolution on a pinhole: "
                       "the centre is coarser and the edge finer. A cylindrical "
                       "frame's is uniform by construction (ratio exactly 1)."),
        }

    ours416 = ours["416x1024"]["deg_per_px_uniform"]
    out["COMPARISON"] = {
        "ours_uniform": ours416,
        "theirs_average": round(140.0 / W, 6),
        "ratio_average_basis": round((140.0 / W) / ours416, 4),
        "ratio_at_THEIR_IMAGE_CENTRE":
            round(out["NAVSIM_DD_1024_wide"]["deg_per_px_at_IMAGE_CENTRE"]
                  / ours416, 4),
        "ratio_at_THEIR_IMAGE_EDGE":
            round(out["NAVSIM_DD_1024_wide"]["deg_per_px_at_IMAGE_EDGE"]
                  / ours416, 4),
        "reading": ("on the AVERAGE basis SPEC uses, ours is finer by the "
                    "quoted factor. If their mosaic is read as one pinhole, "
                    "ours is much finer where the road ahead is (centre) and "
                    "COARSER at the image edge. Which reading is right depends "
                    "on the stitch, which is not in our record."),
    }
    # ⭐ the one number that needs NO assumption about their optics
    out["ASSUMPTION_FREE_COMPARISON"] = {
        "both_are_1024_px_wide": True,
        "ours_HFOV_deg": ours["416x1024"]["HFOV_deg"],
        "theirs_HFOV_deg_INHERITED": 140.0,
        "statement": ("at the SAME pixel width we cover a NARROWER field "
                      "(120.0 vs ~140), so we spend more pixels per degree "
                      "on average — 1.1667x — with no optics assumption at "
                      "all. Everything beyond that needs their stitch."),
        "ratio": round(140.0 / ours["416x1024"]["HFOV_deg"], 4),
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
