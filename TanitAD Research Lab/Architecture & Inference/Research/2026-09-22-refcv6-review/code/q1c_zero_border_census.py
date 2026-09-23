"""Q1c — 10.8 % of a real 416x1024 frame decoded EXACTLY zero (q1b). WHERE?

If the cylindrical rectify leaves an out-of-source-FOV border, that border is a
CONSTANT the ImageNet stem sees at -2.12 / -2.04 / -1.80 after normalisation, and
it also changes what "the field of view" means for the BEV lift (q2/q4). This
locates the zeros: per-row and per-column zero fraction, over several clips.

Also reads the payload's `image_size` field, which the 256-era builder wrote and
a consumer could still read as the geometry.
"""
from __future__ import annotations

import glob
import json
import sys

import torch
import torchvision.io as tvio

CACHE = "D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"


def main() -> int:
    out = {"clips": []}
    paths = sorted(glob.glob(CACHE + "/*.v2ep.pt"))[:5]
    agg_row = None
    agg_col = None
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        lens, buf = d["jpeg_len"], d["jpeg_buf"]
        offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                          torch.cumsum(lens, 0)])
        dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
        # 3 frames spread across the clip
        idx = [0, len(lens) // 2, len(lens) - 1]
        zr = torch.zeros(416)
        zc = torch.zeros(1024)
        n = 0
        mins = []
        for i in idx:
            f = dec(buf[int(offs[i]):int(offs[i + 1])],
                    mode=tvio.ImageReadMode.RGB)
            z = (f == 0).all(dim=0).float()      # zero in ALL THREE channels
            zr += z.mean(dim=1)
            zc += z.mean(dim=0)
            mins.append(int(f.min()))
            n += 1
        zr /= n
        zc /= n
        agg_row = zr if agg_row is None else agg_row + zr
        agg_col = zc if agg_col is None else agg_col + zc
        out["clips"].append({
            "clip_id": d.get("clip_id"),
            "image_size_field": d.get("image_size"),
            "image_h_w": [d.get("image_h"), d.get("image_w")],
            "zero_fraction_all3ch": round(float(zr.mean()), 6),
            "rows_fully_zero": int((zr > 0.99).sum()),
            "cols_fully_zero": int((zc > 0.99).sum()),
            "first_nonzero_row": int((zr < 0.99).nonzero()[0]) if (zr < 0.99).any() else -1,
            "last_nonzero_row": int((zr < 0.99).nonzero()[-1]) if (zr < 0.99).any() else -1,
            "first_nonzero_col": int((zc < 0.99).nonzero()[0]) if (zc < 0.99).any() else -1,
            "last_nonzero_col": int((zc < 0.99).nonzero()[-1]) if (zc < 0.99).any() else -1,
            "min_pixel": min(mins),
        })
    k = len(paths)
    agg_row /= k
    agg_col /= k
    out["aggregate_over_%d_clips" % k] = {
        "top_20_rows_zero_frac": [round(float(v), 4) for v in agg_row[:20]],
        "bottom_20_rows_zero_frac": [round(float(v), 4) for v in agg_row[-20:]],
        "left_10_cols_zero_frac": [round(float(v), 4) for v in agg_col[:10]],
        "right_10_cols_zero_frac": [round(float(v), 4) for v in agg_col[-10:]],
        "rows_over_50pct_zero": int((agg_row > 0.5).sum()),
        "cols_over_50pct_zero": int((agg_col > 0.5).sum()),
        "mean_zero_fraction": round(float(agg_row.mean()), 6),
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
