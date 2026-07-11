"""REF-A stage 1: precompute frozen DINO features over episode caches.

Encodes the LATEST RGB frame of each timestep's stack once per episode
(stacks overlap, so per-frame features reconstruct any 3-frame window), and
stores fp16 token grids [T,256,768] + actions/poses per episode. The REF-A
predictor then trains from these files without ever touching images (the
stability-by-construction property, REFERENCE_ARCHITECTURES.md REF-A).

Tries DINOv3 (HF, gated) first; falls back to DINOv2-B/14 via torch.hub
(ungated, same 16x16 grid at 224 px) and RECORDS which encoder ran.

Usage (pod2):
  python scripts/dino_precompute.py --cache-root /opt/comma_epcache \
      --out /opt/dino_feats --train-n 400 --val-n 90
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tanitad.data.mixing import load_episode


def load_encoder(device):
    try:
        from transformers import AutoImageProcessor, AutoModel
        name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        model = AutoModel.from_pretrained(name).to(device).eval()
        return ("dinov3-b16", model, 256, None)
    except Exception as e:
        print(f"[dino] DINOv3 unavailable ({type(e).__name__}) -> DINOv2-B/14",
              flush=True)
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        return ("dinov2-b14", model.to(device).eval(), 224, None)


@torch.no_grad()
def encode_episode(tag, model, size, ep, device, batch=32):
    frames = ep.frames  # uint8 [T,9,S,S]
    latest = frames[:, -3:].float().div(255.0)               # [T,3,S,S]
    if size != latest.shape[-1]:
        latest = torch.nn.functional.interpolate(
            latest, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    latest = (latest - mean) / std
    toks = []
    for i in range(0, latest.shape[0], batch):
        x = latest[i:i + batch].to(device)
        if tag.startswith("dinov3"):
            out = model(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS
            # dinov3 may include register tokens; keep the last 256 (16x16)
            out = out[:, -256:]
        else:
            out = model.get_intermediate_layers(x, n=1)[0]        # [B,256,768]
        toks.append(out.half().cpu())
    return torch.cat(toks)                                        # [T,256,768]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-n", type=int, default=400)
    ap.add_argument("--val-n", type=int, default=90)
    args = ap.parse_args()
    device = "cuda"
    tag, model, size, _ = load_encoder(device)
    print(f"[dino] encoder: {tag} @ {size}px", flush=True)

    root = Path(args.cache_root)
    for pattern, n in (("*train*", args.train_n), ("*val*", args.val_n)):
        src = sorted(root.glob(pattern))[-1]
        dst = Path(args.out) / (src.name + f"-{tag}")
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("ep_*.pt"))[:n]
        for j, f in enumerate(files):
            o = dst / f.name
            if o.exists():
                continue
            ep = load_episode(str(f), mmap=True)
            feats = encode_episode(tag, model, size, ep, device)
            torch.save({"feats_fp16": feats, "actions": ep.actions,
                        "poses": ep.poses, "episode_id": ep.episode_id}, o)
            if j % 20 == 0:
                print(f"[dino] {src.name}: {j}/{len(files)}", flush=True)
        print(f"[dino] {src.name} done -> {dst}", flush=True)
    (Path(args.out) / "META.json").write_text(json.dumps(
        {"encoder": tag, "size": size, "grid": "16x16", "dim": 768,
         "note": "latest-frame features; 3-frame windows reconstructed at "
                 "train time from consecutive rows"}))
    print("PRECOMPUTE_DONE", flush=True)


if __name__ == "__main__":
    main()
