#!/usr/bin/env python3
"""K7 — my frame packing == the trainer's own window, on REAL PhysicalAI frames. TANITAD VENV.

The NavSim bridge builds refcv4b's input from raw per-time frames: 10 raw slots
(oldest first, t0 last) -> ``pack_frames`` -> [8 rows, 9 ch, H, W]. The model was
trained on ``LazyV2Episode.frames[t:t+8]`` (``v2_dataset._decode_stacked``). This
decodes the RAW PNG frames of a real v2ep clip INDEPENDENTLY (``torchvision
decode_png`` per frame, no provider code), packs raw frames t..t+9 with the
bridge's ``pack_frames``, and requires BYTE-EQUALITY with the provider rows
t..t+7. An off-by-one in the row<->frame alignment, a reversed channel order, or
a wrong "current frame" slot all fail it.

Cache: ``D:/Projects/TanitAD-artifacts/v2ep-eval6-256x640cyl-REF`` (256x640 cyl,
f_ref 305.5775, png, n_stack 3 — the model's frame; its MANIFEST.json).
Writes raw/K7_packer_vs_trainer.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG, "code"))
import tanitad_navsim_bridge as B  # noqa: E402

CACHE = "D:/Projects/TanitAD-artifacts/v2ep-eval6-256x640cyl-REF"


def main(n_clips: int = 2) -> dict:
    import torch
    import torchvision.io as tvio
    from tanitad.data.v2_dataset import _jpeg_offsets, build_v2_providers
    files = sorted(glob.glob(os.path.join(CACHE, "*.v2ep.pt")))[:n_clips]
    eps = build_v2_providers([CACHE], lru_size=2, verbose=False)[:n_clips]
    out = {"cache": CACHE, "checks": []}
    for path, ep in zip(files, eps):
        d = torch.load(path, map_location="cpu", weights_only=False)
        buf, offs, codec = d["jpeg_buf"], _jpeg_offsets(d["jpeg_len"]), str(d.get("codec"))
        dec = tvio.decode_png if codec == "png" else tvio.decode_jpeg
        T = int(ep.frames.shape[0])
        for t in (2, T // 2, T - 9):
            prov = ep.frames[t:t + 8]                                   # [8, 9, H, W] u8
            raw = [dec(buf[int(offs[i]):int(offs[i + 1])], mode=tvio.ImageReadMode.RGB)
                   for i in range(t, t + 10)]                          # raw t..t+9
            hwc = torch.stack(raw).permute(0, 2, 3, 1).numpy()          # [10, H, W, 3]
            mine = B.pack_frames(hwc, list(range(10)))
            eq = bool(torch.equal(mine, prov))
            # the discriminating mutation: a ONE-FRAME shift must be detected
            shifted = B.pack_frames(hwc, [0] + list(range(9)))
            out["checks"].append({"clip": os.path.basename(path), "t": t, "T_rows": T,
                                  "shape": list(prov.shape), "dtype": str(prov.dtype),
                                  "codec": codec, "byte_equal": eq,
                                  "shifted_by_one_detected": not bool(torch.equal(shifted, prov))})
    out["pass"] = all(c["byte_equal"] for c in out["checks"]) and \
        all(c["shifted_by_one_detected"] for c in out["checks"]) and len(out["checks"]) > 0
    return out


def test_packer_matches_trainer():
    assert main()["pass"]


if __name__ == "__main__":
    r = main()
    B.json_dump(r, os.path.join(PKG, "raw", "K7_packer_vs_trainer.json"))
    print(json.dumps(r, indent=1))
