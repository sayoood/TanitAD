"""Does a REAL v2 record survive its consumers unchanged? Run on the dev box, 0 GPU.

The unit tests pin the additive-schema promise against fakes. This runs the
promise against a record the T4 actually produced, through the real consumer:

 1. `ph1_fuse.build_tracks` must read it and must NOT turn a lane marking into
    an object track (the whole reason the scene channel is a separate key);
 2. `ph1_fuse.census_state` must produce the same shape of counts;
 3. ⭐ the RLE must REDRAW — this is the check that would have caught the v1
    flattening defect on day one, and it costs nothing: rebuild the mask from
    `rle_rows` and compare its area AND its bounding box against the banked
    `mask_area_px` and `box_xyxy`. A flattened v1 record fails it instantly
    (every run lands on row 0, so the redrawn bbox is one pixel tall).
 4. the contour must close and its area must match `contour_area_px`.

usage:  python v2_integration_check.py [record.json] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(PKG, "..", "..", "..", "..", ".."))
DEFAULT = os.path.join(PKG, "raw", "p2_sample_0089a096.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("v2_integration_check")
    ap.add_argument("record", nargs="?", default=DEFAULT)
    ap.add_argument("--out", default=os.path.join(PKG, "raw",
                                                  "v2_integration_check.json"))
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
    import numpy as np
    import ph0_sam3
    import ph1_fuse

    rec = json.load(open(a.record, encoding="utf-8"))
    W, H = rec["frame_wh"]
    scene = set(rec.get("concepts_scene") or [])

    tracks = ph1_fuse.build_tracks(rec["frames"])
    leaked = sorted({t["concept"] for t in tracks} & scene)
    counts = ph1_fuse.census_state(tracks).get("counts") or {}

    n_ok = n_bad = n_contour_ok = n_contour_bad = n_box_outside = 0
    worst = None
    for f in rec["frames"].values():
        for key in ("det", "scene"):
            for d in f.get(key) or []:
                runs = d.get("rle_rows")
                if not runs:
                    continue
                m = np.zeros((H, W), bool)
                for r, x0, x1 in runs:
                    m[r, x0:x1] = True
                ys, xs = np.nonzero(m)
                area_ok = int(m.sum()) == int(d["mask_area_px"])
                b = d.get("box_xyxy") or [0, 0, 0, 0]
                # ⛔ THE INVARIANT IS THE ONE THE DEFECT VIOLATES, NOT A
                # BOX-CONTAINMENT ONE. A first version of this check demanded
                # the redrawn mask sit inside `box_xyxy` and failed 16/163 —
                # but `box_xyxy` is an INDEPENDENT model output, not the mask's
                # bbox, and it can be tighter, looser, or (MEASURED here) run
                # past the frame edge at x1 = 450.7 on a 448-wide frame. That
                # disagreement is a property of SAM3, not of our serialisation,
                # and scoring it as a failure would have hidden the real check.
                # ⇒ A FLATTENED record collapses every run onto row 0 and emits
                # columns up to H*W. Those two are the signature.
                rows = int(ys.max() - ys.min()) if ys.size else -1
                cols_ok = bool(runs) and max(r[2] for r in runs) <= W
                not_collapsed = rows > 0 or (b[3] - b[1]) <= 1.5
                if bool(ys.size) and not (
                        ys.min() >= b[1] - 1 and ys.max() <= b[3] + 1
                        and xs.min() >= b[0] - 1 and xs.max() <= b[2] + 1):
                    n_box_outside += 1
                if area_ok and cols_ok and not_collapsed:
                    n_ok += 1
                else:
                    n_bad += 1
                    if worst is None:
                        worst = {"concept": d.get("concept"),
                                 "banked_area": d.get("mask_area_px"),
                                 "redrawn_area": int(m.sum()),
                                 "banked_box": b,
                                 "redrawn_bbox": [int(xs.min()), int(ys.min()),
                                                  int(xs.max() + 1),
                                                  int(ys.max() + 1)]
                                 if ys.size else None}
                xy = d.get("contour_xy")
                if xy:
                    pts = [(xy[i], xy[i + 1]) for i in range(0, len(xy) - 1, 2)]
                    ar = round(abs(ph0_sam3.shoelace2(pts)) / 2.0)
                    if abs(ar - d["contour_area_px"]) <= 1:
                        n_contour_ok += 1
                    else:
                        n_contour_bad += 1

    out = {"class": "MEASURED", "record": os.path.basename(a.record),
           "schema_version": rec.get("schema_version"),
           "engine": rec.get("engine"),
           "ph1_fuse_build_tracks": len(tracks),
           "track_counts": counts,
           "scene_concepts_leaked_into_object_tracks": leaked,
           "rle_redraw_ok": n_ok, "rle_redraw_bad": n_bad,
           "first_bad": worst,
           "mask_outside_its_own_box": n_box_outside,
           "_note_box": "box_xyxy is an INDEPENDENT SAM3 output, not the "
                        "mask's bbox — a mask reaching outside it is a model "
                        "property, reported and not scored as a failure",
           "contour_area_selfconsistent": n_contour_ok,
           "contour_area_mismatch": n_contour_bad}
    out["PASS"] = bool(not leaked and n_bad == 0 and n_contour_bad == 0
                       and len(tracks) > 0)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
