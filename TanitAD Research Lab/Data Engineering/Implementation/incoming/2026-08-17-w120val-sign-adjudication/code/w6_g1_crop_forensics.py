"""STEP 6 — ⭐ WHAT WERE G1'S CROPS ACTUALLY CROPS OF?

Arm C reproduced G1's *"no sign at all"* reading on G1's own pixels (48/54), so
the adjudicator is not the variable. Arm-C's own note recorded something odd
though: **six tiles are essentially the whole scene**, and many are exactly
`96 x 96`. `w5_maxarea_mechanism.py` then REFUTED the obvious explanation —
there are **zero** frame-spanning `traffic sign` boxes on any leg (largest box
anywhere = 7 364 px² = 9.2 % of the 448x179 frame).

⇒ If the boxes are small and the tiles are not, the tiles are **not tight crops
of the boxes**. This step establishes what they are, by matching each banked
`g1_evidence/crops/rowNN.jpg` against the top-2 largest `traffic sign` /
`traffic light` boxes in that clip's banked record — G1's own stated selection
rule (`G1_RESULT.md:6-7`).

⛔ **WHY THIS MATTERS MORE THAN THE PRECISION NUMBER.** The reliability study
tested *"is G1's tight 4x-LANCZOS RENDERING the cause?"* and REFUTED it — but it
tested a **reimplementation** (`r7_g1_reconcile.py` crops exactly the box, so the
object fills the tile). If G1's real renderer padded to a floor and drew **no box
outline**, then a genuine 8x9-px sign sits unmarked inside a 24x24 window of
scene, and *"I see foliage"* is the expected reading **whether or not the
detection is correct**. That is a defect in the INSTRUMENT, and it is invisible
to any re-implementation of what the instrument was believed to do.

Geometry only — no scores, no verdicts, no claim about correctness.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
SHEET = os.path.join(REPO, "Project Steering", "G1_SIGN_OCR_GRADING_SHEET.md")
SCR = (r"C:\Users\Admin\AppData\Local\Temp\claude"
       r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
       r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\w120sign\records")
NATIVE = (640, 256)        # the w120 cylindrical native frame
BRIDGE = (448, 179)        # the frame SAM3 scored


def rows_from_sheet() -> dict[int, str]:
    out = {}
    for m in re.finditer(r"^\|\s*(\d+)\s*\|\s*`([0-9a-f-]+)`",
                         open(SHEET, encoding="utf-8").read(), re.M):
        out[int(m.group(1))] = m.group(2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("w6_g1_crop_forensics")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rows = rows_from_sheet()
    recs = {}
    for leg in ("pilot50", "w120val600"):
        for r in json.load(open(os.path.join(SCR, f"{leg}.json"),
                                encoding="utf-8"))["clips"]:
            recs.setdefault(r["clip_id"], r)

    tiles = json.load(open(a.sample, encoding="utf-8"))["tiles"]
    by_row = {}
    for t in tiles:
        m = re.match(r"row(\d+)(b?)\.jpg$", t["row_file"])
        by_row[(int(m.group(1)), bool(m.group(2)))] = t

    per = []
    n_fullframe = n_floor96 = n_matched4x = 0
    for (rn, is_b), t in sorted(by_row.items()):
        pref = rows.get(rn)
        cid = next((c for c in recs if pref and c.startswith(pref)), None)
        tw, th = t["src_wh"]
        row = {"row": rn, "second_largest": is_b, "idx": t["idx"],
               "tile_wh": [tw, th], "clip_prefix": pref,
               "clip_resolved": bool(cid)}
        if tw == NATIVE[0] and th == NATIVE[1]:
            row["diagnosis"] = "THE WHOLE NATIVE FRAME (640x256) — not a crop"
            n_fullframe += 1
        if cid:
            boxes = []
            for f in (recs[cid].get("frames") or {}).values():
                for d in f.get("det", []) or []:
                    if d.get("concept") not in ("traffic sign",
                                                "traffic light"):
                        continue
                    b = d.get("box_xyxy")
                    if not b:
                        continue
                    w, h = max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
                    boxes.append({"concept": d["concept"], "w": round(w, 1),
                                  "h": round(h, 1), "area": round(w * h, 1)})
            boxes.sort(key=lambda x: -x["area"])
            pick = boxes[1] if (is_b and len(boxes) > 1) else \
                (boxes[0] if boxes else None)
            row["n_sign_or_light_boxes_in_clip"] = len(boxes)
            row["g1_rule_pick"] = pick
            if pick:
                # a pure 4x tight crop would give tile = 4*ceil(box)
                p4 = [round(pick["w"] * 4), round(pick["h"] * 4)]
                row["pure_4x_tight_would_be"] = p4
                row["tile_over_box_area_ratio"] = round(
                    (tw * th) / max(1.0, pick["area"] * 16), 2)
                if abs(tw - p4[0]) <= 4 and abs(th - p4[1]) <= 4:
                    n_matched4x += 1
                    row.setdefault("diagnosis",
                                   "consistent with a pure 4x tight crop")
                elif tw >= 96 and th >= 96 and (p4[0] < 96 or p4[1] < 96):
                    n_floor96 += 1
                    row.setdefault(
                        "diagnosis",
                        "PADDED TO A ~96 px FLOOR — the box is a MINORITY of "
                        "the tile and carries no outline")
                else:
                    row.setdefault("diagnosis", "neither a pure 4x tight crop "
                                                "nor an obvious 96 px floor")
        per.append(row)

    dims = collections.Counter()
    for t in tiles:
        dims[tuple(t["src_wh"])] += 1
    frac = [r["tile_over_box_area_ratio"] for r in per
            if r.get("tile_over_box_area_ratio")]
    frac.sort()
    out = {
      "question": "were G1's tiles tight crops of the SAM3 boxes they are "
                  "attributed to?",
      "n_tiles": len(tiles),
      "n_tiles_that_are_THE_WHOLE_NATIVE_FRAME_640x256": n_fullframe,
      "n_tiles_consistent_with_a_pure_4x_tight_crop": n_matched4x,
      "n_tiles_padded_to_a_96px_floor": n_floor96,
      "tile_dim_histogram_top": dims.most_common(8),
      "tile_area_over_4x_box_area_ratio": {
          "min": frac[0] if frac else None,
          "median": frac[len(frac) // 2] if frac else None,
          "max": frac[-1] if frac else None,
          "n": len(frac),
          "reading": "1.0 = the tile IS the box. >1 = the box is a MINORITY of "
                     "the tile, and G1's tiles carry NO box outline, so the "
                     "adjudicator is not told which pixels the detector "
                     "claimed."},
      "native_frame": list(NATIVE), "bridge_frame": list(BRIDGE),
      "rows": per}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"[forensics] {len(tiles)} tiles · WHOLE-FRAME {n_fullframe} · "
          f"pure-4x-tight {n_matched4x} · 96px-floor {n_floor96}")
    print(f"[forensics] tile-area / (4x box area): min "
          f"{out['tile_area_over_4x_box_area_ratio']['min']} · median "
          f"{out['tile_area_over_4x_box_area_ratio']['median']} · max "
          f"{out['tile_area_over_4x_box_area_ratio']['max']}")
    print(f"[forensics] tile dims: {dims.most_common(6)}")
    print("FORENSICS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
