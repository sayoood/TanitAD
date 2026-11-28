"""Q1b / Q3 — END-TO-END on a REAL 416x1024 episode, not a synthetic tensor.

⛔ The advisory's class A failed on REFe because nobody followed the frames from
the file to the stem. This opens an actual `.v2ep.pt` from the gated eval-139
cache, decodes a frame with the loader refcv6 uses, converts with the trainer's
OWN ingest function, and hooks the pretrained stem.

Also verifies (class B3, "correct intrinsics at the wrong resolution"):
  * the cache's stored `frame` f_ref == `trunk_shapes.FRAME_416x1024.f_ref`
  * the decoded frame's (H, W) == the declared (416, 1024)
  * the trainer's `_agent_cam_frames()[(416,1024)]` IS that same object

READ-ONLY on the cache.
"""
from __future__ import annotations

import glob
import json
import sys

import torch

CACHE = "D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"
OUT = {}


def main() -> int:
    paths = sorted(glob.glob(CACHE + "/*.v2ep.pt"))
    OUT["n_files_found"] = len(paths)
    if not paths:
        OUT["INCONCLUSIVE"] = "no .v2ep.pt in the cache — cannot verify"
        json.dump(OUT, sys.stdout, indent=2)
        return 1
    p = paths[0]
    d = torch.load(p, map_location="cpu", weights_only=False)
    OUT["file"] = p
    OUT["keys"] = sorted(k for k in d.keys())
    OUT["codec_field"] = d.get("codec")
    OUT["buffer_key_name"] = "jpeg_buf" if "jpeg_buf" in d else None
    OUT["stored_frame"] = d.get("frame")
    OUT["image_h_w"] = [d.get("image_h"), d.get("image_w")]
    OUT["n_stack"] = d.get("n_stack")

    # ⛔ READ THE CODEC FIELD, NEVER THE BUFFER'S NAME (the 2026-08-21 trap:
    # a buffer named `jpeg_buf` whose codec is `png`).
    buf = d["jpeg_buf"]
    OUT["magic_first4"] = [int(v) for v in buf[:4].tolist()]
    OUT["magic_is_png"] = OUT["magic_first4"][:2] == [0x89, 0x50]

    import torchvision.io as tvio
    lens = d["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
    dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
    f0 = dec(buf[0:int(offs[1])], mode=tvio.ImageReadMode.RGB)
    OUT["decoded_frame"] = {"shape": list(f0.shape), "dtype": str(f0.dtype),
                            "min": int(f0.min()), "max": int(f0.max()),
                            "per_channel_mean_u8":
                                [round(float(v), 3)
                                 for v in f0.float().mean(dim=(1, 2)).tolist()]}
    # ⭐ a CONTENT assertion, not a presence one (the all-zero-memmap trap)
    OUT["decoded_frame"]["nonzero_fraction"] = round(
        float((f0 != 0).float().mean()), 6)

    # ---- the trainer's OWN ingest -------------------------------------- #
    sys.path.insert(0, "D:/Projects/TanitAD/stack/scripts")
    import importlib
    rv3 = importlib.import_module("refc_v3_train")
    stack9 = torch.cat([f0, f0, f0], dim=0)[None]        # [1, 9, H, W] u8
    x = rv3.frames_to_device(stack9, torch.device("cpu"))
    OUT["after_frames_to_device"] = {
        "dtype": str(x.dtype), "min": round(float(x.min()), 6),
        "max": round(float(x.max()), 6),
        "per_channel_mean": [round(float(v), 6)
                             for v in x.mean(dim=(0, 2, 3)).tolist()[:3]]}

    # ---- geometry: does the trainer's table hold the SAME object? ------- #
    from tanitad.models import trunk_shapes as TS
    tbl = rv3._agent_cam_frames()
    f416 = TS.FRAME_416x1024
    OUT["geometry"] = {
        "trunk_shapes_FRAME_416x1024": {
            "h": f416.height, "w": f416.width, "f_ref": float(f416.f_ref),
            "projection": f416.projection},
        "trainer_table_has_416x1024": (416, 1024) in tbl,
        "trainer_table_IS_the_same_object": tbl.get((416, 1024)) is f416,
        "cache_f_ref": d["frame"]["f_ref"],
        "cache_f_ref_equals_trunk_shapes":
            abs(float(d["frame"]["f_ref"]) - float(f416.f_ref)) == 0.0,
        "cache_hw_equals_declared":
            (int(f0.shape[1]), int(f0.shape[2])) == (f416.height, f416.width),
    }

    # ---- the stem, on REAL pixels -------------------------------------- #
    from tanitad.models.timm_trunk import (TimmResNetTrunk, TimmTrunkConfig,
                                           _find_stem)
    t = TimmResNetTrunk(TimmTrunkConfig(
        model_name="resnet34.a1_in1k", frames=3, mode="shared",
        image_hw=(416, 1024), imagenet_norm=True, pretrained=True))
    rec = {}
    _n, stem = _find_stem(t.net)

    def hook(_m, inp, _o):
        z = inp[0].detach()
        rec["per_channel_mean"] = [round(float(v), 6)
                                   for v in z.mean(dim=(0, 2, 3)).tolist()]
        rec["per_channel_std"] = [round(float(v), 6)
                                  for v in z.std(dim=(0, 2, 3)).tolist()]
    h = stem.register_forward_hook(hook)
    # a 128x256 centre crop keeps this a seconds-long CPU forward; the stem
    # statistic is per-channel and does not depend on the crop size.
    with torch.no_grad():
        s16, s32, _p = t.forward_features(x[:, :, 144:272, 384:640])
    h.remove()
    OUT["real_pixels_stem_stats"] = rec
    OUT["real_pixels_feature_shapes"] = {"s16": list(s16.shape),
                                         "s32": list(s32.shape)}
    OUT["norm_calls"] = int(t.norm_calls)

    # the expected stem stats IF normalisation is applied, computed from the
    # measured raw statistics — an INDEPENDENT derivation, not a re-run of the
    # producer's own arithmetic.
    raw_m = x[:, :3].mean(dim=(0, 2, 3)).tolist()
    raw_s = x[:, :3].std(dim=(0, 2, 3)).tolist()
    im_m, im_s = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    OUT["expected_stem_if_normalised"] = {
        "mean": [round((raw_m[i] - im_m[i]) / im_s[i], 6) for i in range(3)],
        "std": [round(raw_s[i] / im_s[i], 6) for i in range(3)]}
    OUT["expected_stem_if_NOT_normalised"] = {
        "mean": [round(v, 6) for v in raw_m], "std": [round(v, 6) for v in raw_s]}
    # ⚠️ crop-sensitive: recompute on the SAME crop the trunk saw
    c = x[:, :3, 144:272, 384:640]
    cm, cs = c.mean(dim=(0, 2, 3)).tolist(), c.std(dim=(0, 2, 3)).tolist()
    OUT["expected_stem_if_normalised_SAME_CROP"] = {
        "mean": [round((cm[i] - im_m[i]) / im_s[i], 6) for i in range(3)],
        "std": [round(cs[i] / im_s[i], 6) for i in range(3)]}

    json.dump(OUT, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
