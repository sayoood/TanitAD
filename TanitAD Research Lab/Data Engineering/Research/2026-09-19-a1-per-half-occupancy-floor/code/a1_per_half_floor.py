"""A1 — the occupancy floor on EACH clean-62 half, with the controls COMPUTED.

⛔ WHY THIS IS NOT A COPY OF `occ_floor.py`. That script prints its two controls as
LITERALS:

    print(f"{'  GT against itself':28s} {1.0:>10.4f}   ⛔ INSTRUMENT CHECK")
    print(f"{'  none-drivable':28s} {0.0:>10.4f}   degenerate control")

Its own docstring says *"if this is not 1.0 the comparison is wired wrong and nothing else
on this page is readable"* — but the number is the constant `1.0`, so it CANNOT be anything
else. ⇒ the instrument check was a label, not a check: it would have read 1.0000 on a
completely mis-wired comparison. A1's criterion is that it reads exactly 1.0000, which is
only meaningful if it is COMPUTED from the same arrays every other number comes from.

⭐ Same family as every other defect found tonight: a check that shares the defect it checks
for is green forever. Here it did not even share it — it was hardcoded.

The three predictors, all scored as IoU against GT over SEEN cells only (an unseen cell
carries no evidence — `dac_from_drivable`'s rule):
  * ALL-drivable   -> |G| / |seen|      = P(drivable | seen), the no-information floor
  * NONE-drivable  -> 0 / |G|           = exactly 0, the degenerate control
  * GT vs GT       -> |G| / |G|         = exactly 1, the instrument check
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt-bevtac\stack")
from tanitad.data import semantic_map_gt as SMG              # noqa: E402

ART = pathlib.Path("D:/Projects/TanitAD-artifacts")
MAPS = ART / "sam3-maps-eval"
HALVES = {"halfA": ART / "v2ep-eval124clean-416x1024cyl-halfA",
          "halfB": ART / "v2ep-eval124clean-416x1024cyl-halfB"}
DRIVABLE = SMG.CHANNELS.index("drivable")
THRESH, STRIDE, CELL = 0.5, 20, 0.5
BANDS = ((0, 15), (15, 30), (30, 45), (45, 60))
CORPUS_FLOOR = 0.3412          # MEASURED over 10,068,274 seen cells, all 139 clips
TOL = 0.02


def iou(pred: np.ndarray, gt: np.ndarray, seen: np.ndarray) -> tuple[int, int]:
    """(intersection, union) over SEEN cells only."""
    p, g = pred & seen, gt & seen
    return int((p & g).sum()), int((p | g).sum())


def score_half(name: str, cache: pathlib.Path) -> dict:
    man = cache / "CLEAN124_MANIFEST.json"
    sha12s = json.loads(man.read_text(encoding="utf-8"))["halves_sha12"][name]
    tot = {"seen": 0, "cells": 0,
           "all_i": 0, "all_u": 0, "none_i": 0, "none_u": 0, "gt_i": 0, "gt_u": 0}
    band_seen = {b: 0 for b in BANDS}
    band_driv = {b: 0 for b in BANDS}
    n_clips = n_frames = 0
    for s12 in sha12s:
        p = MAPS / f"{s12}{SMG.GT_SUFFIX}"
        if not p.is_file():
            raise SystemExit(f"[a1] ⛔ {name}: no map for {s12} — A0 asserted every clip "
                             f"has one, so this is a real disagreement, not a skip")
        with np.load(p) as z:
            cart = z["cart_frac"]
            for t in range(0, cart.shape[0], STRIDE):
                a = cart[t]
                ns = a[SMG.NOT_SEEN_CHANNEL].astype(np.int32)
                seen = 2 * (SMG.FRACTION_SCALE - ns) >= SMG.FRACTION_SCALE
                gt = (a[DRIVABLE].astype(np.float32) / SMG.FRACTION_SCALE) >= THRESH
                all_p = np.ones_like(seen, dtype=bool)
                none_p = np.zeros_like(seen, dtype=bool)
                for key, pred in (("all", all_p), ("none", none_p), ("gt", gt)):
                    i, u = iou(pred, gt, seen)
                    tot[f"{key}_i"] += i
                    tot[f"{key}_u"] += u
                tot["seen"] += int(seen.sum())
                tot["cells"] += seen.size
                for lo, hi in BANDS:
                    r0, r1 = int(lo / CELL), int(hi / CELL)
                    band_seen[(lo, hi)] += int(seen[r0:r1].sum())
                    band_driv[(lo, hi)] += int((seen & gt)[r0:r1].sum())
                n_frames += 1
        n_clips += 1
    f = lambda k: (tot[f"{k}_i"] / tot[f"{k}_u"]) if tot[f"{k}_u"] else float("nan")
    out = {"half": name, "n_clips": n_clips, "n_frames": n_frames,
           "stride": STRIDE, "seen_cells": tot["seen"], "grid_cells": tot["cells"],
           "iou_all_drivable": f("all"), "iou_none_drivable": f("none"),
           "iou_gt_vs_gt": f("gt"),
           "p_drivable_given_seen": tot["all_i"] / tot["seen"] if tot["seen"] else float("nan"),
           "bands": {f"{lo}-{hi}": {"seen_cells": band_seen[(lo, hi)],
                                    "p_drivable_given_seen":
                                        band_driv[(lo, hi)] / band_seen[(lo, hi)]
                                        if band_seen[(lo, hi)] else float("nan")}
                     for lo, hi in BANDS}}
    return out


def main() -> int:
    results = {n: score_half(n, c) for n, c in HALVES.items()}
    ok = True
    print(f"{'half':7s} {'clips':>6s} {'frames':>7s} {'seen cells':>14s} "
          f"{'FLOOR':>8s} {'GTvGT':>8s} {'NONE':>7s}")
    for n, r in results.items():
        print(f"{n:7s} {r['n_clips']:>6d} {r['n_frames']:>7d} {r['seen_cells']:>14,} "
              f"{r['iou_all_drivable']:>8.4f} {r['iou_gt_vs_gt']:>8.4f} "
              f"{r['iou_none_drivable']:>7.4f}")
        # ⛔ (a) the INSTRUMENT CHECK, now computed: exactly 1.0
        if r["iou_gt_vs_gt"] != 1.0:
            print(f"  ⛔ {n}: GT-against-itself is {r['iou_gt_vs_gt']!r}, not exactly 1.0 — "
                  f"the comparison is wired wrong and nothing here is readable")
            ok = False
        if r["iou_none_drivable"] != 0.0:
            print(f"  ⛔ {n}: none-drivable is {r['iou_none_drivable']!r}, not exactly 0.0")
            ok = False
        # ⛔ (b) within ±0.02 of the corpus floor
        d = abs(r["iou_all_drivable"] - CORPUS_FLOOR)
        print(f"  {n}: |floor - corpus {CORPUS_FLOOR}| = {d:.4f} "
              f"({'within' if d <= TOL else 'OUTSIDE'} ±{TOL})")
        if d > TOL:
            ok = False
    pathlib.Path("C:/Users/Admin/qland/a1_per_half_floor.json").write_text(
        json.dumps({"_what": "A1 — per-half occupancy floor, controls COMPUTED not printed",
                    "_evidence_class": "MEASURED (ours; zero GPU)",
                    "corpus_floor": CORPUS_FLOOR, "tolerance": TOL,
                    "halves": results}, indent=1), encoding="utf-8", newline="\n")
    print("ZZA1-OK" if ok else "ZZA1-FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
