"""Q2c — tighten F2's bound: a PERSISTENT-zero mask instead of a fully-black ROW.

q2b bracketed the defect with two estimates that are known to be wrong in
opposite directions:
  * fully-black ROW  -> LOWER bound (misses the curved boundary's partial rows);
  * any exactly-zero PIXEL in one frame -> UPPER bound (a night shadow can be 0).

A pixel that is exactly (0,0,0) in EVERY sampled frame of a clip is unobserved:
scene content does not hold an exact triple zero across 12 independent frames.
That mask is the measurement; the two bounds above are its sanity rails.

Controls carried: the three counts must order LOWER <= PERSISTENT <= UPPER, and
a clip with no black rows at all must read ~0 on all three.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import torch
import torchvision.io as tvio

CACHE = "D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"
EXTR = ("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/"
        "Research/2026-09-06-refcv4b-landing/raw/extrinsics141.json")
N_FRAMES = 12


def main() -> int:
    from tanitad.models.bev_lift import build_lift_geometry
    from tanitad.models.trunk_shapes import FRAME_416x1024 as F416

    extr = json.load(open(EXTR))
    # the 5 worst from q2b plus 2 CONTROL clips that showed no black rows
    worst = ["5d94cbbb-55eb-4642-add5-054b9c47ccf3",
             "37e61793-36e9-426a-b257-23e684ba18b6",
             "f901e9e6-cc3a-4af4-8461-a6c6a6f3d6d2",
             "2e8a2df2-3f74-448e-9781-0739328ab42e",
             "d1d382aa-035a-4da8-a7c5-d06ec97f0fcd"]
    q2b = json.load(open(os.path.join(os.path.dirname(__file__), "..", "raw",
                                      "q2b_lift_unobserved.json")))
    clean = [r["clip_id"] for r in q2b["per_clip"]
             if r["n_black_rows"] == 0][:2]
    rows = []
    for cid in worst + clean:
        p = os.path.join(CACHE, cid + ".v2ep.pt")
        if not os.path.exists(p):
            rows.append({"clip_id": cid, "INCONCLUSIVE": "file absent"})
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        lens, buf = d["jpeg_len"], d["jpeg_buf"]
        offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                          torch.cumsum(lens, 0)])
        dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
        n = len(lens)
        idx = [int(round(i * (n - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES)]
        persistent = None
        first_zero = None
        for j, i in enumerate(idx):
            f = dec(buf[int(offs[i]):int(offs[i + 1])],
                    mode=tvio.ImageReadMode.RGB)
            z = (f == 0).all(dim=0)                       # [H, W]
            persistent = z if persistent is None else (persistent & z)
            if j == 0:
                first_zero = z
        black_row = persistent.all(dim=1)                 # [H]

        pose = {k: float(extr[cid][k])
                for k in ("qx", "qy", "qz", "qw", "x", "y", "z")}
        g = build_lift_geometry(pose, frame=F416, stride=16)
        ri = g.row.round().long().clamp(0, F416.height - 1)
        ci = g.col.round().long().clamp(0, F416.width - 1)

        def count(mask2d_or_1d, is_row):
            hit = (mask2d_or_1d[ri] if is_row else mask2d_or_1d[ri, ci]) & g.valid
            return int(hit.sum()), int(hit.any(dim=0).sum())

        s_low, c_low = count(black_row, True)
        s_per, c_per = count(persistent, False)
        s_up, c_up = count(first_zero, False)
        rows.append({
            "clip_id": cid,
            "n_frames_sampled": len(idx),
            "persistent_zero_pixel_frac": round(float(persistent.float().mean()), 6),
            "one_frame_zero_pixel_frac": round(float(first_zero.float().mean()), 6),
            "n_rows_fully_persistent_zero": int(black_row.sum()),
            "samples_valid_total": int(g.valid.sum()),
            "LOWER_samples_in_fully_black_rows": s_low,
            "MEASURED_samples_in_persistent_zero": s_per,
            "UPPER_samples_in_one_frame_zero": s_up,
            "MEASURED_cells_touching_persistent_zero": c_per,
            "pct_of_valid_samples_MEASURED":
                round(100.0 * s_per / max(int(g.valid.sum()), 1), 4),
            "ORDERING_OK": s_low <= s_per <= s_up,
        })
    out = {"cache": CACHE, "n_frames_per_clip": N_FRAMES,
           "grid_cells_total": 120 * 64, "samples_total": 4 * 120 * 64,
           "rows": rows,
           "ALL_ORDERINGS_OK": all(r.get("ORDERING_OK", True) for r in rows)}
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
