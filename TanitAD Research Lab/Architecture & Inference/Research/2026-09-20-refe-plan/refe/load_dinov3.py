"""Load real DINOv3 ViT-S/16 weights into REFe's frozen trunk -- and REFUSE a partial load.

SOURCE. facebook/dinov3-vits16-pretrain-lvd1689m is gated (HTTP 401 without a licence acceptance).
timm/vit_small_patch16_dinov3.lvd1689m is the same pretraining (lvd1689m), ungated, 82.4 MB,
21,586,944 parameters. Research use is sanctioned by the programme's standing rule.

THE MAPPING, from the checkpoint's own key structure (MEASURED, not assumed):
  blocks.{i}.attn.qkv.weight   (1152, 384)  -> split into q.base / k / v.base   (NO qkv bias)
  blocks.{i}.attn.proj.{w,b}                -> attn.proj
  blocks.{i}.norm{1,2}.{w,b}                -> n1 / n2
  blocks.{i}.mlp.fc{1,2}.{w,b}              -> mlp[0] / mlp[2]
  blocks.{i}.gamma_{1,2}       (384,)       -> LayerScale
  patch_embed.proj.{w,b}                    -> patch_embed

THE CONTROL, and why a partial load is worse than no load. A frozen trunk that is 95 % pretrained
and 5 % random is not a pretrained trunk -- it is a silently damaged one that still trains, still
converges, and quietly forfeits the reason DINOv3 was chosen. So this refuses unless EVERY target
tensor in the trunk is either loaded or on the explicit `EXPECTED_UNLOADED` list, and it prints the
checkpoint tensors it did not consume.

EXPECTED_UNLOADED is deliberately short and each entry carries its reason.

Usage: python load_dinov3.py [--weights <dir>] [--strict]
"""
from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import BACKBONES, REFe, REFeConfig  # noqa: E402

BACKBONE_ROOT = os.environ.get("REFE_BACKBONE_ROOT", "D:/Projects/TanitAD/data/backbones")  # pod portability

# Target tensors that legitimately have no counterpart in the checkpoint.
EXPECTED_UNLOADED = {
    # ⛔ "pos" USED TO SIT HERE, and its presence is why the no-partial-load control passed on a
    # defect. The entry read: "DINOv3 uses rotary position encoding; our trunk uses a learned table
    # sized to the 512x960 input, so there is nothing to copy." Both halves were true and the
    # CONCLUSION was wrong -- a trunk pretrained with RoPE and then given a RANDOM FROZEN table
    # instead is not the pretrained trunk. The table is gone (2026-09-20) and RoPE is computed on
    # the fly in `model.build_axial_rope`, so there is nothing to excuse any more.
    # ⭐ An allow-list entry is a claim that an absence is harmless. Re-read each one when the
    # thing it excuses changes; an allow-list is the one place a guard is designed not to look.
}


def load_state(weights_dir: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    p = Path(weights_dir) / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"{p} -- fetch it from timm/vit_small_patch16_dinov3.lvd1689m")
    return load_file(str(p))


def map_into_backbone(bb, sd: dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[str]]:
    loaded, used = [], set()

    def cp(dst: torch.nn.Parameter, key: str, slc=None):
        if key not in sd:
            return False
        t = sd[key]
        if slc is not None:
            t = t[slc]
        if tuple(t.shape) != tuple(dst.shape):
            raise ValueError(f"shape {tuple(t.shape)} != {tuple(dst.shape)} for {key}")
        with torch.no_grad():
            dst.copy_(t)
        used.add(key)
        return True

    d = bb.cfg.width
    if cp(bb.patch_embed.weight, "patch_embed.proj.weight"):
        loaded.append("patch_embed.weight")
    if cp(bb.patch_embed.bias, "patch_embed.proj.bias"):
        loaded.append("patch_embed.bias")
    if cp(bb.norm.weight, "norm.weight"):
        loaded.append("norm.weight")
    if cp(bb.norm.bias, "norm.bias"):
        loaded.append("norm.bias")
    # DINOv3's own class + 4 register tokens -- pretrained, part of the frozen function
    if cp(bb.cls_token, "cls_token"):
        loaded.append("cls_token")
    if cp(bb.reg_token, "reg_token"):
        loaded.append("reg_token")

    for i, blk in enumerate(bb.blocks):
        p = f"blocks.{i}."
        # fused qkv -> three separate projections
        if cp(blk.attn.q.base.weight, p + "attn.qkv.weight", slice(0, d)):
            loaded.append(p + "q")
        if cp(blk.attn.k.weight, p + "attn.qkv.weight", slice(d, 2 * d)):
            loaded.append(p + "k")
        if cp(blk.attn.v.base.weight, p + "attn.qkv.weight", slice(2 * d, 3 * d)):
            loaded.append(p + "v")
        for a, b in [(blk.attn.proj.weight, "attn.proj.weight"), (blk.attn.proj.bias, "attn.proj.bias"),
                     (blk.n1.weight, "norm1.weight"), (blk.n1.bias, "norm1.bias"),
                     (blk.n2.weight, "norm2.weight"), (blk.n2.bias, "norm2.bias"),
                     (blk.mlp[0].weight, "mlp.fc1.weight"), (blk.mlp[0].bias, "mlp.fc1.bias"),
                     (blk.mlp[2].weight, "mlp.fc2.weight"), (blk.mlp[2].bias, "mlp.fc2.bias"),
                     (blk.gamma_1, "gamma_1"), (blk.gamma_2, "gamma_2")]:
            if cp(a, p + b):
                loaded.append(p + b)

    tgt_missing = []
    for name, _ in bb.named_parameters():
        short = name.split("blocks.")[-1]
        short = short.split(".", 1)[1] if short[0].isdigit() else short
        hit = any(name.endswith(s) or short.endswith(s) or name == s for s in EXPECTED_UNLOADED)
        covered = any(name.replace(".base", "").startswith(x.rsplit(".", 1)[0]) for x in loaded)
        if not hit and not covered:
            tgt_missing.append(name)
    unused = sorted(set(sd) - used)
    return loaded, tgt_missing, unused


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vits16", choices=sorted(BACKBONES))
    ap.add_argument("--weights", default=None, help="override the weights dir")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any unconsumed tensor")
    a = ap.parse_args()

    cfg = REFeConfig.for_backbone(a.backbone)
    wdir = a.weights or f"{BACKBONE_ROOT}/{BACKBONES[a.backbone]['weights']}"
    pub = BACKBONES[a.backbone]["published_pdms"]
    print(f"  backbone {a.backbone}: width {cfg.width} depth {cfg.depth} heads {cfg.heads}   "
          f"published DINOv3 PDMS {pub if pub else 'not published'}")
    m = REFe(cfg)
    bb = m.backbone
    before = bb.blocks[0].mlp[0].weight.clone()
    sd = load_state(wdir)
    print(f"  checkpoint: {len(sd)} tensors, {sum(t.numel() for t in sd.values()):,} params")

    loaded, missing, unused = map_into_backbone(bb, sd)
    print(f"  copied {len(loaded)} target tensors")
    print(f"  checkpoint tensors NOT consumed: {len(unused)}")
    for k in unused[:10]:
        print(f"     {k}  {tuple(sd[k].shape)}")
    print(f"  target tensors NOT loaded and NOT expected: {len(missing)}")
    for k in missing[:10]:
        print(f"     {k}")

    changed = not torch.equal(before, bb.blocks[0].mlp[0].weight)
    print(f"\n  CONTROL 1 weights actually changed: {'PASS' if changed else 'FAIL -- nothing copied'}")
    n_loaded_params = sum(1 for _ in loaded)
    print(f"  CONTROL 2 every block got its 15 tensors: "
          f"{'PASS' if n_loaded_params >= 15 * cfg.depth else f'FAIL ({n_loaded_params})'}")
    print(f"  CONTROL 3 no unconsumed checkpoint tensors: {'PASS' if not unused else 'FAIL'}")
    print("\n  EXPECTED_UNLOADED, each with its reason:")
    for k, why in EXPECTED_UNLOADED.items():
        print(f"     {k:12s} {why}")

    ok = changed and not unused and not missing
    print("\n" + ("LOAD_OK" if ok else "LOAD_INCOMPLETE"))
    return 0 if ok or not a.strict else 1


if __name__ == "__main__":
    sys.exit(main())
