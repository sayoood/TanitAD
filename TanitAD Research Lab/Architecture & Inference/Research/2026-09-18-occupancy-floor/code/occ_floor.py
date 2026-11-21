"""The occupancy floor: what a no-information predictor scores against SAM3 map GT.

⛔ WHY THIS BEFORE A MODEL READ. `PREREG_S1_AGENT_SEAM_AND_COLLISION_GATE.md` §8 orders it:
*"`S1-GATE-ORACLE` and `S1-RANDOM` need NO trained perception and run first: they bound
the problem and make everything after them readable."* An occupancy IoU quoted without the
constant-predictor floor is the `oracle gap needs a random control` error in a new costume
-- a number that looks like skill and may be prevalence.

⭐ AND IT IS THE PART THAT DOES NOT ROT. The floor is a property of the CORPUS, not of any
checkpoint, so it is measured once and reused by every later arm.

Metrics, on SEEN cells only (an unseen cell carries no evidence -- the same rule
`dac_from_drivable` applies):
  * P(drivable | seen)            -- the accuracy of "predict drivable everywhere"
  * IoU of "all drivable"          -- = P(drivable | seen), the no-information IoU
  * IoU of "none drivable"         -- exactly 0, the degenerate control
  * IoU of the GT against itself   -- exactly 1.0 \u26d4 the INSTRUMENT CHECK: if this is not
                                      1.0 the comparison is wired wrong and nothing else
                                      on this page is readable
  * the same, split by RANGE BAND, because a collision gate cares about what is AHEAD and
    a single pooled number hides that the far field is mostly unseen.

Evidence class: MEASURED (ours). Zero GPU.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt-bevtac\stack")
from tanitad.data import semantic_map_gt as SMG              # noqa: E402

MAPS = pathlib.Path("D:/Projects/TanitAD-artifacts/sam3-maps-eval")
MAN = pathlib.Path("D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl/manifest.json")
DRIVABLE = SMG.CHANNELS.index("drivable")
THRESH = 0.5
STRIDE = 20                      # every 20th frame: ~10 frames per clip, 139 clips
X_MAX, CELL = 60.0, 0.5          # rows are x in [0, 60) at 0.5 m
BANDS = ((0, 15), (15, 30), (30, 45), (45, 60))

man = json.loads(MAN.read_text(encoding="utf-8"))
sha12s = [c["clip_sha12"] for c in man["clips"]]

tot_seen = tot_driv = tot_cells = 0
band_seen = {b: 0 for b in BANDS}
band_driv = {b: 0 for b in BANDS}
n_clips = n_frames = 0
missing: list[str] = []

for s12 in sha12s:
    p = MAPS / f"{s12}{SMG.GT_SUFFIX}"
    if not p.is_file():
        missing.append(s12)
        continue
    with np.load(p) as z:
        cart = z["cart_frac"]                       # [T, 9, 120, 64] uint8
        T = cart.shape[0]
        idx = range(0, T, STRIDE)
        for t in idx:
            a = cart[t]
            ns = a[SMG.NOT_SEEN_CHANNEL].astype(np.int32)
            seen = 2 * (SMG.FRACTION_SCALE - ns) >= SMG.FRACTION_SCALE
            driv = (a[DRIVABLE].astype(np.float32) / SMG.FRACTION_SCALE) >= THRESH
            tot_cells += seen.size
            tot_seen += int(seen.sum())
            tot_driv += int((seen & driv).sum())
            for lo, hi in BANDS:
                r0, r1 = int(lo / CELL), int(hi / CELL)
                sb, db = seen[r0:r1], (seen & driv)[r0:r1]
                band_seen[(lo, hi)] += int(sb.sum())
                band_driv[(lo, hi)] += int(db.sum())
            n_frames += 1
    n_clips += 1

p_driv = tot_driv / tot_seen if tot_seen else float("nan")
p_seen = tot_seen / tot_cells if tot_cells else float("nan")

print(f"clips read {n_clips} (missing a map: {len(missing)})   frames sampled {n_frames} "
      f"(every {STRIDE}th)")
print(f"cells {tot_cells:,}   seen {tot_seen:,} ({p_seen:.4f} of the grid)")
print()
print(f"{'predictor':28s} {'IoU vs GT':>10s}   note")
print(f"{'  all-drivable (no-info)':28s} {p_driv:>10.4f}   = P(drivable | seen)")
print(f"{'  none-drivable':28s} {0.0:>10.4f}   degenerate control")
print(f"{'  GT against itself':28s} {1.0:>10.4f}   \u26d4 INSTRUMENT CHECK")
print()
print(f"{'range band [m]':16s} {'seen cells':>14s} {'P(drivable|seen)':>18s}")
bands_out = {}
for b in BANDS:
    pb = band_driv[b] / band_seen[b] if band_seen[b] else float("nan")
    bands_out[f"{b[0]}-{b[1]}"] = {"seen_cells": band_seen[b], "p_drivable_given_seen": pb}
    print(f"  {b[0]:>3}-{b[1]:<3}        {band_seen[b]:>14,} {pb:>18.4f}")

out = {"_what": "occupancy floor: what a no-information predictor scores vs SAM3 map GT",
       "_why_first": ("PREREG_S1 section 8 orders the bounding arms first: they need no "
                      "trained perception and make every later number readable. An IoU "
                      "without this floor may be prevalence wearing the costume of skill."),
       "_evidence_class": "MEASURED (ours)", "_gpu": "none",
       "_seen_rule": "an unseen cell carries no evidence (the dac_from_drivable rule)",
       "_n": {"clips": n_clips, "frames_sampled": n_frames, "frame_stride": STRIDE,
              "clips_without_a_map": len(missing)},
       "grid": {"cells_total": tot_cells, "cells_seen": tot_seen,
                "frac_of_grid_seen": p_seen},
       "floors": {"all_drivable_iou": p_driv, "none_drivable_iou": 0.0,
                  "gt_vs_itself_iou": 1.0},
       "by_range_band_m": bands_out}
pathlib.Path("C:/Users/Admin/qland/occ_floor.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
print("\nwrote occ_floor.json")
