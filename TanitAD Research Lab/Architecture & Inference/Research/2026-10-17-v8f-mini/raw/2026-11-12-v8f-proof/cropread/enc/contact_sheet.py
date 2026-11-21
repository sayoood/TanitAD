#!/usr/bin/env python3
"""Smoke only. ONE private contact sheet on the pod: 12 frames that have a lead in the label, each row = arm D W32 | arm D W16 | arm B W32 |
arm B W16 (the windows kept by window_encode --keep-windows). Camera pixels are private: the sheet stays on the pod, is not pulled,
opened or copied. Usage: contact_sheet.py --run <smokeN dir> --out <smokeN>/sheet_private.png"""
import argparse
import os
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--tile", type=int, default=160)
    a = ap.parse_args(argv)
    import torch
    import torch.nn.functional as F
    import torchvision
    wdir = os.path.join(a.run, "win")
    picks = []
    for f in sorted(os.listdir(wdir)):
        c = f[:-4]
        bx = os.path.join(a.run, "box", f"{c}.npz")
        if not os.path.exists(bx):
            continue
        has = np.load(bx)["has_lead"].astype(bool)
        idx = np.nonzero(has)[0]
        if idx.size:
            picks.append((c, idx))
    if not picks:
        print("no frame with a lead in the kept windows")
        return 2
    per = max(1, a.rows // len(picks))
    sel = []
    for c, idx in picks:
        step = max(1, len(idx) // per)
        sel += [(c, int(j)) for j in idx[::step][:per]]
    sel = sel[:a.rows]
    rows = []
    cache = {}
    for c, j in sel:
        if c not in cache:
            cache[c] = np.load(os.path.join(wdir, f"{c}.npy"), mmap_mode="r")
        w = torch.from_numpy(np.array(cache[c][j]))                  # [4,3,256,256]
        w = F.interpolate(w.float(), size=(a.tile, a.tile), mode="area").round().clamp(0, 255).to(torch.uint8)
        rows.append(torch.cat([w[i] for i in range(4)], dim=2))      # [3,tile,4*tile]
    sheet = torch.cat(rows, dim=1)
    torchvision.io.write_png(sheet, a.out)
    print(f"sheet written: {len(rows)} rows, {tuple(sheet.shape)}; PRIVATE, not pulled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
