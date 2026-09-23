"""Q1d — census of the UNOBSERVED (black) region over ALL 139 clips of the
416x1024 cache refcv6 trains and validates on.

⛔ WHY THIS MATTERS AND IS NOT COSMETIC. `calib.py:1028-1036` records retraction
class **C26**: the front-wide has TWO rigs (cy~543 / cy~755); a frame one rig
fully observes and the other does not leaves a **rig-correlated black region**,
which is *"still a rig-correlated signal, and this model eats shortcuts"*. The
programme's answer was `PHYSICALAI_RIG_CLEAN_176x624` / `128x576` — fields BOTH
rigs fully observe. The 2026-09-17 move to 416 rows buys VFOV 46.09 deg, which
`SPEC_REFCV6_V2.md` §12.3 states *exceeds* the 256x640 reference (45.4556 deg).

This measures, on the CONSUMER'S OWN ARTIFACT (the decoded PNG frames):
  * per-clip fraction of pixels that are zero in all three channels;
  * how many whole rows are unobserved, and WHERE (top vs bottom);
  * whether the population is BIMODAL — the C26 signature.

Read-only. 2 frames per clip (first and middle).
"""
from __future__ import annotations

import glob
import json
import sys

import torch
import torchvision.io as tvio

CACHE = "D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"


def main() -> int:
    paths = sorted(glob.glob(CACHE + "/*.v2ep.pt"))
    rows = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        lens, buf = d["jpeg_len"], d["jpeg_buf"]
        offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                          torch.cumsum(lens, 0)])
        dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
        zr = torch.zeros(416)
        zf = 0.0
        idx = [0, len(lens) // 2]
        for i in idx:
            f = dec(buf[int(offs[i]):int(offs[i + 1])],
                    mode=tvio.ImageReadMode.RGB)
            z = (f == 0).all(dim=0).float()
            zr += z.mean(dim=1)
            zf += float(z.mean())
        zr /= len(idx)
        zf /= len(idx)
        full = (zr > 0.995)
        rows.append({
            "clip_id": d.get("clip_id"),
            "zero_frac": round(zf, 6),
            "n_rows_fully_black": int(full.sum()),
            "n_bottom_rows_black": int(full[208:].sum()),
            "n_top_rows_black": int(full[:208].sum()),
        })
    zfs = torch.tensor([r["zero_frac"] for r in rows])
    nb = torch.tensor([float(r["n_rows_fully_black"]) for r in rows])
    out = {
        "cache": CACHE, "n_clips": len(rows),
        "zero_frac": {
            "mean": round(float(zfs.mean()), 6),
            "min": round(float(zfs.min()), 6),
            "max": round(float(zfs.max()), 6),
            "median": round(float(zfs.median()), 6)},
        "clips_with_ANY_fully_black_row": int((nb > 0).sum()),
        "clips_with_zero_frac_over_5pct": int((zfs > 0.05).sum()),
        "clips_with_zero_frac_under_1pct": int((zfs < 0.01).sum()),
        "black_rows_when_present": {
            "mean": round(float(nb[nb > 0].mean()), 2) if (nb > 0).any() else 0,
            "min": int(nb[nb > 0].min()) if (nb > 0).any() else 0,
            "max": int(nb[nb > 0].max()) if (nb > 0).any() else 0},
        "bottom_vs_top": {
            "clips_black_at_BOTTOM": int(sum(
                1 for r in rows if r["n_bottom_rows_black"] > 0)),
            "clips_black_at_TOP": int(sum(
                1 for r in rows if r["n_top_rows_black"] > 0))},
        "per_clip": rows,
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
