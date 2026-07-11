"""Rebuild the PhysicalAI epcache from origin — data-mix-as-recipe tool.

Replicates the trainer's `--data physicalai` branch EXACTLY (same
discover -> split(seed) -> build_episodes_cached chain, same params), so a
fresh pod regenerates the corpus pod1 trained on without touching pod1.
Prereqs on this pod: r0_selection.parquet + fetched camera clips
(scripts/physicalai_r0.py select / fetch-camera).

Identity check against the original: episode COUNT printed here must match
the source cache (pod1: 402 train), and a sampled-checksum comparison runs
post-30k. On mismatch: fall back to the direct pod1 copy — do NOT train a
comparison arm on unverified data silently.

Usage (pod3):
  python scripts/build_pai_cache.py --root /workspace/data/physicalai
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tanitad.config import base250cam_config
from tanitad.data.epcache import build_episodes_cached
from tanitad.data.physicalai import (build_episode, discover_r0_clips,
                                     split_clips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="PhysicalAI R0 root")
    ap.add_argument("--episodes", type=int, default=0,
                    help="clip bound before split (0 = all; must match the "
                         "original run's --episodes for identity)")
    args = ap.parse_args()

    cfg = base250cam_config()
    clips = discover_r0_clips(args.root)
    if args.episodes:
        clips = clips[:args.episodes]
    print(f"[build] {len(clips)} clips discovered", flush=True)
    tr, va = split_clips(clips, val_frac=0.2, seed=cfg.train.seed)   # I3
    print(f"[build] split: {len(tr)} train / {len(va)} val "
          f"(seed {cfg.train.seed})", flush=True)

    cache = Path(args.root) / "_epcache"
    params = {"size": cfg.encoder.image_size, "n_stack": 3, "hz": 10}
    for cs, split in ((tr, "train"), (va, "val")):
        eps = build_episodes_cached(
            cs, lambda c: build_episode(c, size=cfg.encoder.image_size),
            cache, f"physicalai-{split}", params)
        print(f"[build] physicalai-{split}: {len(eps)} episodes cached",
              flush=True)
    for d in sorted(cache.glob("physicalai-*")):
        n = len(list(d.glob("ep_*.pt")))
        print(f"[build] {d.name}: {n} files", flush=True)
    print("PAI_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
