"""DrivoR-style encoder budget on OUR geometry — params + FLOPs, CPU only.

Question: what does DrivoR's perception recipe (DINOv2 ViT-S/14 + 16 per-camera
registers, arXiv 2601.05083) cost at TanitAD's input (one front camera,
256x640 cylindrical, 3-frame history), against our REF-C trunks?

Params are exact (random init; weights do not change counts or FLOPs).
FLOPs are counted with torch.utils.flop_counter (counts SDPA; fvcore, which the
paper used, does NOT). ⇒ CONTROL: the paper's own geometry (4 cams, 672x1148)
must reproduce its Table 11 backbone figure of ~350 GFLOPs when SDPA is
excluded, or the instrument is not trusted.

Latency is deliberately NOT measured here: the dev-box GPU was busy with a live
training run, and a CPU latency is not a GPU latency.

Usage: PYTHONPATH=<repo>/stack python drivor_encoder_budget.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.flop_counter import FlopCounterMode


def _vit_s(h: int, w: int, extra_reg: int):
    import timm
    # DINOv2 ViT-S/14 geometry; reg_tokens = DrivoR's per-camera registers
    return timm.create_model("vit_small_patch14_dinov2", pretrained=False,
                             img_size=(h, w), reg_tokens=extra_reg, num_classes=0)


def _flops(model, x) -> dict:
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model(x)
    per_op = {str(k): v for k, v in fc.get_flop_counts()["Global"].items()}
    tot = fc.get_total_flops()
    sdpa = sum(v for k, v in per_op.items() if "scaled_dot_product" in k or "_efficient_attention" in k
               or "_flash_attention" in k)
    # FlopCounter counts a multiply-add as 2 FLOPs; fvcore counts 1 "flop" per MAC.
    return {"flops_total": tot, "flops_sdpa": sdpa, "ops": sorted(per_op),
            "gmac_excl_sdpa": (tot - sdpa) / 2e9, "gmac_total": tot / 2e9}


def _params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)   # a live GPU training shares this host; stay small
    out = {}

    # --- CONTROL: DrivoR paper geometry, 4 cams x (672, 1148) -> must read ~350 G (fvcore, excl. SDPA)
    h, w = 672, 1148
    m = _vit_s(h, w, 16).eval()
    f = _flops(m, torch.zeros(1, 3, h, w))
    out["control_paper_geometry_4cam"] = {
        "params": _params(m), "tokens_per_cam": (h // 14) * (w // 14) + 1 + 16,
        "gmac_excl_sdpa_4cam": 4 * f["gmac_excl_sdpa"], "gmac_total_4cam": 4 * f["gmac_total"],
        "paper_table11_backbone_gflops_fvcore": 350, "counted_ops": f["ops"],
        "sdpa_gmac_4cam": 4 * f["flops_sdpa"] / 2e9}

    # --- OUR geometry: 1 front cam, 252x644 (divisible by 14; closest to 256x640), 3 frames shared weights
    h, w = 252, 644
    m = _vit_s(h, w, 16).eval()
    f = _flops(m, torch.zeros(1, 3, h, w))
    out["drivor_vits_ours_1cam_3frames"] = {
        "params": _params(m), "tokens_per_frame": (h // 14) * (w // 14) + 1 + 16,
        "gmac_excl_sdpa_3frames": 3 * f["gmac_excl_sdpa"], "gmac_total_3frames": 3 * f["gmac_total"],
        "scene_tokens_out": 16 * 3}

    # --- OUR REF-C in-repo trunk (pre-refcv6 default): base_width 88, 9-ch 3-frame stack, 256x640
    from tanitad.refs.refc import CNNEncoderConfig
    import tanitad.refs.refc as R
    enc_cls = getattr(R, "ResNetEncoder")  # the class that CNNEncoderConfig builds (refc.py)
    if True:
        cfg = CNNEncoderConfig(image_size=256, image_width=640)
        e = enc_cls(cfg).eval()
        f = _flops(e, torch.zeros(1, 9, 256, 640))
        out["refc_inrepo_trunk_base88"] = {"params": _params(e), "gmac_total": f["gmac_total"],
                                           "docstring_claim_params": 90_458_632}

    # --- refcv6 candidates: timm ResNet-101 / ResNet-34 ImageNet, shared weights over 3 frames
    import timm
    for name in ("resnet101", "resnet34"):
        r = timm.create_model(name, pretrained=False, num_classes=0, global_pool="").eval()
        f = _flops(r, torch.zeros(1, 3, 256, 640))
        out[f"timm_{name}_3frames_shared"] = {"params": _params(r), "gmac_total_3frames": 3 * f["gmac_total"]}

    print(json.dumps(out, indent=1))
    if a.json:
        open(a.json, "w", encoding="utf-8").write(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
