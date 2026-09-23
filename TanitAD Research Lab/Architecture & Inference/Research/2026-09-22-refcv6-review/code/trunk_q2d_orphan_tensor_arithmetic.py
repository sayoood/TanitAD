"""Q2d — class B1 by the ADVISORY'S OWN METHOD: parameter arithmetic against the
checkpoint, not an argument that "a ResNet has no positional table".

On REFe the orphan was found like this: *"our backbone totalled 305,045,504
against the checkpoint's 303,079,424, and the difference 1,966,080 = exactly
1920 x 1024 — the table was the ONLY trunk tensor with no checkpoint
counterpart."* The same arithmetic, here, for both refcv6 backbones:

  * every tensor of our backbone that has NO key in the downloaded checkpoint;
  * every checkpoint key our backbone does not load (`features_only=True` drops
    the classifier, which is expected and is the same-breath control that this
    comparison is actually reading the file);
  * the parameters the refcv6 trunk adds ON TOP of the backbone, named.

Reads the real `model.safetensors` from the HF cache. No download.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import torch

HUB = os.path.expanduser("~/.cache/huggingface/hub")


def ckpt_keys(name: str):
    repo = "models--timm--" + name.replace("/", "--")
    hits = glob.glob(os.path.join(HUB, repo, "snapshots", "*",
                                  "model.safetensors"))
    if not hits:
        hits = glob.glob(os.path.join(HUB, repo, "snapshots", "*", "*.bin"))
    if not hits:
        return None, None
    p = hits[0]
    from safetensors.torch import load_file
    sd = load_file(p) if p.endswith(".safetensors") else torch.load(
        p, map_location="cpu")
    return p, {k: tuple(v.shape) for k, v in sd.items()}


def main() -> int:
    import timm

    from tanitad.models.timm_trunk import TimmResNetTrunk, TimmTrunkConfig

    out = {}
    for name in ("resnet34.a1_in1k", "resnet101.a1_in1k"):
        path, ck = ckpt_keys(name)
        if ck is None:
            out[name] = {"INCONCLUSIVE": "checkpoint not in the HF cache"}
            continue
        t = TimmResNetTrunk(TimmTrunkConfig(
            model_name=name, frames=3, mode="shared", image_hw=(416, 1024),
            pretrained=True))
        bb = dict(t.net.named_parameters())
        bbuf = dict(t.net.named_buffers())
        # timm's features_only wrapper prefixes nothing for resnets, but be
        # explicit: strip a leading "model." if present.
        def norm(k):
            return k[6:] if k.startswith("model.") else k
        bb_all = {norm(k): tuple(v.shape) for k, v in
                  list(bb.items()) + list(bbuf.items())
                  if "num_batches_tracked" not in k}
        orphan = sorted(k for k in bb_all if k not in ck)
        unused = sorted(k for k in ck if k not in bb_all)
        n_bb = sum(p.numel() for p in t.net.parameters())
        n_ck = sum(int(torch.tensor(list(s)).prod()) if s else 1
                   for s in ck.values())
        extra = {n: int(p.numel()) for n, p in t.named_parameters()
                 if not n.startswith("net.")}
        out[name] = {
            "checkpoint_file": os.path.basename(path),
            "checkpoint_dir": os.path.dirname(path),
            "n_checkpoint_tensors": len(ck),
            "n_backbone_tensors_incl_buffers": len(bb_all),
            "BACKBONE_TENSORS_WITH_NO_CHECKPOINT_KEY": orphan,
            "n_orphan": len(orphan),
            "checkpoint_keys_not_in_backbone": unused,
            "backbone_param_count": n_bb,
            "checkpoint_total_numel": int(n_ck),
            "difference": int(n_ck) - n_bb,
            "refcv6_params_ON_TOP_of_backbone": extra,
            "n_extra_params": sum(extra.values()),
            "extra_are_identity_initialised": bool(t.cfg.fuse_identity_init),
        }
        # ⭐ the same-breath CONTROL: the comparison must find the classifier
        # head in `unused` — if it finds nothing at all, it read nothing.
        out[name]["CONTROL_found_expected_classifier_keys"] = any(
            k.startswith("fc.") for k in unused)
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
