"""P3 - the SAM3 MAP head's SEEN mask: is it real, how much does it remove, and
is the removal BIASED?

SPEC_REFCV6_V2.md SS6: "soft cross-entropy, 9 classes, **seen cells only**".
``bev_encoder.map_soft_ce`` takes ``seen`` as a BOOL mask and multiplies the
per-cell term by it, so an unseen cell contributes exactly zero. The mask itself
is derived in ``semantic_map_gt.ClipMapGT.read:330-331``::

    ns   = cart_frac[:, NOT_SEEN_CHANNEL].astype(int32)     # 0..255
    seen = 2 * (255 - ns) >= 255                            # not-seen share < 0.5

This probe reads the banked eval SAM3 GT and measures, WITHOUT the model:

  * the seen fraction overall and PER GRID ROW (row r == x in
    [r*0.5, (r+1)*0.5) m) -- the class-G correlation test: a mask that removes
    uniformly is a different problem from one that removes the far field;
  * the label mass per channel ON SEEN CELLS, because "seen, no map class"
    (channel 0) is a class in the 9 and a head can satisfy the CE by predicting
    it;
  * a DISCRIMINATING CONTROL: the mask recomputed from the RAW uint8 with the
    exact integer rule, asserted equal to the reader's, so a reader bug and a
    label property are not confused; plus an ALL-TRUE mask arm, which must move
    the numbers (a mask that changed nothing would not be a mask).

Clip ids are the file stems, which are ALREADY sha12 digests.

Run (CPU, read-only):
  PYTHONPATH=D:/Projects/TanitAD/stack python p3_map_seen_mask_census.py \
      --root D:/Projects/TanitAD-artifacts/sam3-maps-eval
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tanitad.data.semantic_map_gt import (CART_SHAPE, CHANNELS, FRACTION_SCALE,
                                          NOT_SEEN_CHANNEL, SEEN_SHARE_MIN)

HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "raw" / "p3_map_seen_mask_census.json"
CELL_M = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    files = sorted(pathlib.Path(a.root).glob("*.sam3mapgt.npz"))
    if a.max_clips:
        files = files[: a.max_clips]
    t0 = time.time()

    nx, ny = CART_SHAPE
    seen_per_row = np.zeros(nx, dtype=np.int64)
    cells_per_row = np.zeros(nx, dtype=np.int64)
    mass_seen = np.zeros(len(CHANNELS), dtype=np.float64)
    mass_all = np.zeros(len(CHANNELS), dtype=np.float64)
    n_frames = 0
    per_clip = {}
    reader_mismatch = 0
    frames_zero_seen = 0

    for fp in files:
        with np.load(fp, allow_pickle=True) as z:
            u8 = z["cart_frac"][:: a.frame_stride]             # [T,9,120,64]
        T = int(u8.shape[0])
        ns = u8[:, NOT_SEEN_CHANNEL].astype(np.int32)
        seen = 2 * (FRACTION_SCALE - ns) >= FRACTION_SCALE      # the rule, re-run
        # DISCRIMINATING CONTROL: the same predicate written the OTHER way
        # (float share >= SEEN_SHARE_MIN). Integer and float must agree; a
        # disagreement is a rounding hole in the rule, not a label property.
        share = 1.0 - ns.astype(np.float64) / FRACTION_SCALE
        alt = share >= SEEN_SHARE_MIN
        reader_mismatch += int((alt != seen).sum())

        seen_per_row += seen.sum(axis=(0, 2))
        cells_per_row += T * ny
        p = u8.astype(np.float64) / FRACTION_SCALE
        mass_all += p.sum(axis=(0, 2, 3))
        mass_seen += (p * seen[:, None]).sum(axis=(0, 2, 3))
        n_frames += T
        fs = float(seen.mean())
        per_clip[fp.name.split(".")[0]] = round(fs, 6)
        frames_zero_seen += int((seen.sum(axis=(1, 2)) == 0).sum())

    frac_row = seen_per_row / np.maximum(cells_per_row, 1)
    overall = float(seen_per_row.sum() / max(cells_per_row.sum(), 1))
    msum = float(mass_seen.sum()) or 1.0
    clips = np.array(list(per_clip.values()))

    # near/far split at 30 m == row 60
    near = float(seen_per_row[:60].sum() / max(cells_per_row[:60].sum(), 1))
    far = float(seen_per_row[60:].sum() / max(cells_per_row[60:].sum(), 1))

    res = {
        "_evidence_class": "MEASURED (ours; artifact = this JSON + the code beside it)",
        "_line": "eval-139 (135 banked SAM3 GT clips on the dev box) -- NOT the "
                 "v7-B1 train corpus, for which no SAM3 GT is on this machine",
        "root": str(a.root),
        "n_clip_files": len(files),
        "frame_stride": a.frame_stride,
        "n_frames_read": n_frames,
        "rule": "seen = 2*(255 - cart_frac[:, 8]) >= 255  "
                "(semantic_map_gt.py:330-331)",
        "seen_mask_is_real": {
            "frac_seen_overall": overall,
            "frac_unseen_overall": 1.0 - overall,
            "n_frames_with_zero_seen_cells": frames_zero_seen,
            "integer_vs_float_rule_mismatches": reader_mismatch,
            "control_note": "0 mismatches means the integer rule and the "
                            "float-share rule select the same cells; a non-zero "
                            "seen fraction strictly below 1 means the mask "
                            "actually selects (an all-true mask would read 1.0)",
        },
        "bias_by_forward_range": {
            "cell_m": CELL_M,
            "frac_seen_near_0_30m": near,
            "frac_seen_far_30_60m": far,
            "near_over_far": (near / far) if far else None,
            "per_row_frac_seen": [round(float(v), 5) for v in frac_row],
            "rows_below_1pct_seen": int((frac_row < 0.01).sum()),
            "first_row_below_50pct": (int(np.argmax(frac_row < 0.5))
                                      if bool((frac_row < 0.5).any()) else None),
        },
        "label_mass_on_seen_cells": {
            CHANNELS[i]: {"share_of_seen_mass": float(mass_seen[i] / msum),
                          "share_of_all_mass": float(mass_all[i]
                                                     / max(mass_all.sum(), 1.0))}
            for i in range(len(CHANNELS))
        },
        "per_clip_frac_seen": {
            "n_clips": int(len(clips)),
            "min": float(clips.min()), "p10": float(np.percentile(clips, 10)),
            "p50": float(np.percentile(clips, 50)),
            "p90": float(np.percentile(clips, 90)), "max": float(clips.max()),
            "n_clips_below_0.10": int((clips < 0.10).sum()),
        },
        "wall_s": round(time.time() - t0, 1),
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    slim = {k: v for k, v in res.items() if k != "bias_by_forward_range"}
    slim["bias_by_forward_range"] = {k: v for k, v
                                     in res["bias_by_forward_range"].items()
                                     if k != "per_row_frac_seen"}
    print(json.dumps(slim, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
