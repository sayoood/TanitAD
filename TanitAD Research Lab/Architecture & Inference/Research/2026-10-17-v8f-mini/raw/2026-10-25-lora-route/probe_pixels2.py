"""Probe 3b: WHAT KIND of difference is Thor's pixel product (Aug-30 eval eps) from the dev-box rebuild (S4C = production eval frames)?  Clip dca157a0, every 10th cache step, decoded uint8.
Signed difference A30 - S4C per channel (a colour-matrix / range change shows as a signed, intensity-dependent offset; a rounding-mode change as a ~+-0.5 level constant; a resampler change as a spatial pattern),
the histogram of the difference, and its dependence on S4C intensity and on image row/column.  Diagnostic only: it narrows the unrecorded step, it does not identify it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torchvision

C = "dca157a0-20a9-464b-a74c-084b678e7be1"
A30 = Path("C:/Users/Admin/tanitad-data/refav1-eval141/eps") / f"{C}.v2ep.pt"
S4C = Path("C:/Users/Admin/tanitad-data/refav1-eval141-256rb") / f"{C}.v2ep.pt"


def frames(o, raw_idx):
    lens, buf = o["jpeg_len"], o["jpeg_buf"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    return torch.stack([torchvision.io.decode_png(buf[int(offs[j]):int(offs[j + 1])].clone()) for j in raw_idx])


def main() -> int:
    a, s = (torch.load(p, map_location="cpu", weights_only=False) for p in (A30, S4C))
    raw = [2 * g for g in range(0, 101, 10)]
    x, y = frames(a, raw).int(), frames(s, raw).int()
    d = (x - y)
    out = {"clip": C, "raw_frames": raw, "mean_signed_diff_per_channel_RGB": [float(d[:, c].float().mean()) for c in range(3)],
           "mean_abs_diff_per_channel_RGB": [float(d[:, c].abs().float().mean()) for c in range(3)],
           "hist_diff_-4..4": {str(k): float((d == k).float().mean()) for k in range(-4, 5)}, "frac_|diff|>3": float((d.abs() > 3).float().mean())}
    inten = y.float().mean(1, keepdim=True).expand_as(d)
    bins = [0, 32, 64, 96, 128, 160, 192, 224, 256]
    out["by_S4C_intensity_bin(mean_signed, mean_abs, frac_differing, n)"] = {f"{bins[i]}-{bins[i + 1]}": [round(float(d[(inten >= bins[i]) & (inten < bins[i + 1])].float().mean()), 4) if int(((inten >= bins[i]) & (inten < bins[i + 1])).sum()) else None,
                                                                                     round(float(d[(inten >= bins[i]) & (inten < bins[i + 1])].abs().float().mean()), 4) if int(((inten >= bins[i]) & (inten < bins[i + 1])).sum()) else None,
                                                                                     round(float((d[(inten >= bins[i]) & (inten < bins[i + 1])] != 0).float().mean()), 4) if int(((inten >= bins[i]) & (inten < bins[i + 1])).sum()) else None,
                                                                                     int(((inten >= bins[i]) & (inten < bins[i + 1])).sum())] for i in range(len(bins) - 1)}
    ad = d.abs().float().mean((0, 1))                              # [H, W]
    out["abs_diff_by_row_band(8 bands of 32 rows)"] = [round(float(ad[i * 32:(i + 1) * 32].mean()), 4) for i in range(8)]
    out["abs_diff_by_col_band(8 bands of 80 cols)"] = [round(float(ad[:, i * 80:(i + 1) * 80].mean()), 4) for i in range(8)]
    out["abs_diff_by_frame"] = [round(float(d[i].abs().float().mean()), 4) for i in range(d.shape[0])]
    json.dump(out, open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
