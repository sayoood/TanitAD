"""Probe 3c: is the +0.75-level one-sided difference between Thor's Aug-30 eval eps (A30) and the dev-box rebuild (S4C = production) the float->uint8 CAST of the rectifier?

Hypothesis (from probe_pixels2): the production ``calib.cylindrical_rectify`` ends in ``out.clamp(0, 255).to(torch.uint8)`` (TRUNCATION toward zero); a rectifier that rounds (or adds a half level) gives a one-sided
positive offset of ~0.5 level.  TEST: rebuild the eval clip dca157a0 from the local mp4 with the production builder (v2_compressed._resampled, which this script does NOT edit) but with ``cylindrical_rectify`` replaced,
in this process only, by a copy whose last line is varied:  floor (production, the control: must equal S4C exactly), round, ceil.  Compare every frame (T=201 raw) against A30 and S4C.
Nothing on disk is edited; the monkey-patch lives in this process.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("PAI_DECODE_THREADS", "4")
import torch
import torch.nn.functional as F
import torchvision

HERE = Path(__file__).resolve()
REPO = next(p for p in HERE.parents if (p / "EXPORT_BASE.txt").is_file())
sys.path[:0] = [str(REPO / "stack"), str(REPO / "stack" / "scripts")]
import v2_compressed as V  # noqa: E402
from tanitad.data import calib  # noqa: E402
from tanitad.data.calib import CanonicalFrame  # noqa: E402

C = "dca157a0-20a9-464b-a74c-084b678e7be1"
ROOT = Path("R:/work/build_root")
A30 = Path("C:/Users/Admin/tanitad-data/refav1-eval141/eps") / f"{C}.v2ep.pt"
S4C = Path("C:/Users/Admin/tanitad-data/refav1-eval141-256rb") / f"{C}.v2ep.pt"


def make_rectify(cast: str):
    def rect(vid, intr, frame, padding_mode="zeros", *, require_per_clip=True):
        t = vid.shape[0]
        h, w = int(vid.shape[-2]), int(vid.shape[-1])
        grid, mask = calib.cylindrical_grid(intr, h, w, frame, device=vid.device)
        out = F.grid_sample(vid.float(), grid.expand(t, -1, -1, -1), mode="bilinear", padding_mode=padding_mode, align_corners=False)
        if padding_mode == "zeros":
            out = out * mask.to(out.dtype)
        out = out.clamp(0, 255)
        if cast == "floor":
            return out.to(torch.uint8)                       # == production's  out.clamp(0, 255).to(torch.uint8)
        if cast == "round":
            return out.round().to(torch.uint8)
        if cast == "ceil":
            return out.ceil().clamp(0, 255).to(torch.uint8)
        raise ValueError(cast)
    return rect


def frames_of(o):
    lens, buf = o["jpeg_len"], o["jpeg_buf"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    return torch.stack([torchvision.io.decode_png(buf[int(offs[j]):int(offs[j + 1])].clone()) for j in range(len(lens))])


def main() -> int:
    clip = {"clip_id": C, "mp4": str(ROOT / "r0/camera_front_wide" / f"{C}.mp4"), "timestamps": str(ROOT / "r0/camera_front_wide" / f"{C}.timestamps.parquet"),
            "ego_zip": str(ROOT / "labels/egomotion/egomotion_all.zip")}
    frame = CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical")
    a30, s4c = frames_of(torch.load(A30, map_location="cpu", weights_only=False)).int(), frames_of(torch.load(S4C, map_location="cpu", weights_only=False)).int()
    out = {"clip": C, "n_frames": int(a30.shape[0]), "A30_vs_S4C": {"mean_signed": float((a30 - s4c).float().mean()), "frac_equal": float((a30 == s4c).float().mean())}, "variants": {}}
    orig = V.cylindrical_rectify
    for cast in ("floor", "round", "ceil"):
        V.cylindrical_rectify = make_rectify(cast)
        try:
            vid, _, _ = V._resampled(clip, 256, frame, "cylindrical")      # the production decode + selection + rectify, only the cast differs
        finally:
            V.cylindrical_rectify = orig
        v = vid.int()
        assert v.shape == s4c.shape, (v.shape, s4c.shape)
        out["variants"][cast] = {"vs_S4C": {"frac_equal": float((v == s4c).float().mean()), "mean_signed": float((v - s4c).float().mean()), "max_abs": int((v - s4c).abs().max())},
                                 "vs_A30": {"frac_equal": float((v == a30).float().mean()), "mean_signed": float((v - a30).float().mean()), "max_abs": int((v - a30).abs().max()),
                                            "hist_-3..3": {str(k): float(((v - a30) == k).float().mean()) for k in range(-3, 4)}}}
        print(cast, json.dumps(out["variants"][cast]), flush=True)
    json.dump(out, open(HERE.with_suffix(".json"), "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
