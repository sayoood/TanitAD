"""Q1 — is the ImageNet mean/std the checkpoint declares ACTUALLY applied to the
tensor the pretrained stem sees, at the right value range and channel order?

⛔ Advisory class A: *"A grep over every model file returned zero normalisation"*
on REFe. The check here is not a grep. A forward hook on the backbone's OWN stem
records the tensor the ImageNet weights actually multiply, and we compare its
per-channel statistics against `timm`'s declared `default_cfg['mean'/'std']`.

The controls, because a positive assertion can pass on the wrong thing:
  * `imagenet_norm=False` arm — the deliberate regression. If it reads the SAME
    stem statistics as the normalised arm, the probe is measuring nothing.
  * a [0,255] input — the value-range trap. Must move the stem input far.
  * a BGR input (channel-reversed) — the channel-order trap.

Run:
  PYTHONPATH=D:/Projects/TanitAD/stack python q1_normalisation_probe.py
"""
from __future__ import annotations

import json
import sys

import torch

from tanitad.models.timm_trunk import TimmResNetTrunk, TimmTrunkConfig
from tanitad.models.trunk_shapes import FRAME_416x1024

OUT = {}


def _stem_hook(trunk):
    """Record the tensor the BACKBONE'S OWN first conv receives."""
    rec = {}
    from tanitad.models.timm_trunk import _find_stem
    name, stem = _find_stem(trunk.net)
    rec["stem_name"] = name
    rec["stem_in_channels"] = int(stem.in_channels)

    def hook(_m, inp, _out):
        x = inp[0].detach()
        rec["shape"] = list(x.shape)
        # per-channel over the 3 (or 3K) stem channels
        rec["per_channel_mean"] = [round(float(v), 6)
                                   for v in x.mean(dim=(0, 2, 3)).tolist()]
        rec["per_channel_std"] = [round(float(v), 6)
                                  for v in x.std(dim=(0, 2, 3)).tolist()]
        rec["global_min"] = round(float(x.min()), 6)
        rec["global_max"] = round(float(x.max()), 6)
    h = stem.register_forward_hook(hook)
    return rec, h


def build(name: str, norm: bool, k: int = 3, hw=(416, 1024), pretrained=True):
    cfg = TimmTrunkConfig(model_name=name, frames=k, mode="shared",
                          image_hw=tuple(hw), imagenet_norm=norm,
                          pretrained=pretrained,
                          verify_imagenet_stats=bool(pretrained))
    return TimmResNetTrunk(cfg)


def main() -> int:
    torch.manual_seed(0)
    name = "resnet34.a1_in1k"
    hw = (416, 1024)
    # A SMALL spatial crop is enough for the statistics and keeps this on CPU;
    # shape correctness is measured separately at the full geometry (q3).
    small = (64, 128)

    import timm
    m = timm.create_model(name, pretrained=False, features_only=True,
                          out_indices=(3, 4))
    OUT["timm_default_cfg"] = {
        "mean": list(m.default_cfg.get("mean")),
        "std": list(m.default_cfg.get("std")),
        "input_size": list(m.default_cfg.get("input_size", [])),
        "crop_pct": m.default_cfg.get("crop_pct"),
        "interpolation": m.default_cfg.get("interpolation"),
    }
    from tanitad.models.timm_trunk import IMAGENET_MEAN, IMAGENET_STD
    OUT["module_constants"] = {"IMAGENET_MEAN": list(IMAGENET_MEAN),
                               "IMAGENET_STD": list(IMAGENET_STD)}
    OUT["declared_vs_module_identical"] = (
        list(m.default_cfg.get("mean")) == list(IMAGENET_MEAN)
        and list(m.default_cfg.get("std")) == list(IMAGENET_STD))

    # ---- the frames: [0,1] float, K=3 RGB stack, as the contract delivers -- #
    u8 = (torch.rand(2, 9, *small) * 255).to(torch.uint8)
    x01 = u8.float().div(255.0)          # == tanitad.data._contract.to_float_frames

    # ---- ARM 1: the shipped default ------------------------------------- #
    t = build(name, norm=True, hw=(small[0] // 1 * 1, small[1]))
    # image_hw only sets the declared grid; forward works on any /32 input
    rec, h = _stem_hook(t)
    t.norm_calls = 0
    with torch.no_grad():
        s16, s32, pooled = t.forward_features(x01)
    h.remove()
    OUT["arm_norm_on"] = {
        "norm_calls_for_one_forward": int(t.norm_calls),
        "stem": rec,
        "s16_shape": list(s16.shape), "s32_shape": list(s32.shape),
        "trunk_mean_buffer": [round(float(v), 6)
                              for v in t._mean.flatten().tolist()],
        "trunk_std_buffer": [round(float(v), 6)
                             for v in t._std.flatten().tolist()],
    }

    # ---- ARM 2: the DELIBERATE REGRESSION — no normalisation ------------- #
    t0 = build(name, norm=False, hw=(small[0], small[1]))
    rec0, h0 = _stem_hook(t0)
    t0.norm_calls = 0
    with torch.no_grad():
        s16_0, _s32_0, _p0 = t0.forward_features(x01)
    h0.remove()
    OUT["arm_norm_off_REGRESSION"] = {
        "norm_calls_for_one_forward": int(t0.norm_calls), "stem": rec0}

    # feature displacement — the advisory's own currency (rel L2 / cos)
    def disp(a, b):
        a, b = a.flatten(1).double(), b.flatten(1).double()
        rel = float(((a - b).norm(dim=1) / a.norm(dim=1)).mean())
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean())
        return {"rel_l2": round(rel, 4), "cos": round(cos, 4)}

    OUT["displacement_norm_off_vs_on"] = disp(s16, s16_0)

    # ---- ARM 3: the VALUE-RANGE trap — [0,255] fed to a [0,1] trunk ------ #
    t255 = build(name, norm=True, hw=(small[0], small[1]))
    rec255, h255 = _stem_hook(t255)
    with torch.no_grad():
        s16_255, _a, _b = t255.forward_features(u8.float())
    h255.remove()
    OUT["arm_value_range_0_255_TRAP"] = {"stem": rec255,
                                         "displacement_vs_correct":
                                             disp(s16, s16_255)}

    # ---- ARM 4: the CHANNEL-ORDER trap — BGR --------------------------- #
    bgr = x01.reshape(2, 3, 3, *small).flip(2).reshape(2, 9, *small)
    tb = build(name, norm=True, hw=(small[0], small[1]))
    with torch.no_grad():
        s16_bgr, _a, _b = tb.forward_features(bgr)
    OUT["arm_channel_order_BGR_TRAP"] = {"displacement_vs_correct":
                                         disp(s16, s16_bgr)}

    # ---- the ARITHMETIC IDENTITY: does normalise() do what it claims? ---- #
    man = torch.empty_like(x01)
    mean = torch.tensor(IMAGENET_MEAN * 3).reshape(1, 9, 1, 1)
    std = torch.tensor(IMAGENET_STD * 3).reshape(1, 9, 1, 1)
    man = (x01 - mean) / std
    got = t.normalise(x01)
    OUT["normalise_matches_hand_arithmetic"] = {
        "max_abs_diff": float((man - got).abs().max()),
        "tiled_per_frame": True}

    # ---- and per-frame tiling: frame 2's channels use the SAME triple ---- #
    OUT["mean_buffer_is_tiled_triple"] = (
        OUT["arm_norm_on"]["trunk_mean_buffer"]
        == [round(v, 6) for v in list(IMAGENET_MEAN) * 3])

    json.dump(OUT, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
