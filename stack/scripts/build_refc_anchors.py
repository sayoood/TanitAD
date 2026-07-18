"""Build the REF-C anchor vocabulary via furthest-point sampling (FPS).

REF-C's anchored-diffusion decoder selects over a fixed vocabulary of ego-frame
future trajectories [n_horizons, 2]. This tool builds that vocabulary by FPS —
NOT k-means: comma2k19 is ~74 % straight, so k-means collapses nearly every
centroid onto the straight mode and starves the turns, whereas FPS deliberately
SPREADS coverage over the rare sharp-curve / hard-brake trajectories (the modes
that matter). Saved as a .pt dict consumed by ``refc_train.py --anchors``.

Two sources:
  - real data (--data-root): FPS over the ego-frame waypoint targets of EVERY
    window of the cached episodes (refb_labels.waypoint_targets — the exact
    targets the trainer regresses).
  - synthetic (--smoke, or no --data-root): FPS over a pool of random unicycle
    rollouts (refc.synth_anchor_pool) — the CPU-smoke / bootstrap path.

Usage:
  python scripts/build_refc_anchors.py --data-root /workspace/data \
      --out /workspace/experiments/refc_anchors.pt --n-anchors 64
  python scripts/build_refc_anchors.py --out /tmp/anchors.pt --smoke --n-anchors 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import refb_labels
from refc_train import load_cached_episodes
from tanitad.refs.refc import (furthest_point_sample, synth_anchor_pool)


def episode_traj_pool(episodes: list, horizons: tuple[int, ...]) -> torch.Tensor:
    """Ego-frame waypoint trajectories for EVERY window of the episodes.

    [sum_e (T_e - max_horizon), len(horizons), 2] — the exact targets the trainer
    regresses (refb_labels.waypoint_targets), vectorised per episode."""
    max_h = max(horizons)
    pool = []
    for ep in episodes:
        poses = ep.poses.float()                          # [T, 4]
        n = poses.shape[0] - max_h
        if n <= 0:
            continue
        idx = torch.arange(n)
        pose_last = poses[idx]                            # [n, 4]  (t = 0..n-1)
        fut = torch.stack([poses[idx + k] for k in range(1, max_h + 1)],
                          dim=1)                          # [n, max_h, 4]
        pool.append(refb_labels.waypoint_targets(pose_last, fut, horizons))
    if not pool:
        raise ValueError("no windows long enough to build an anchor pool")
    return torch.cat(pool, dim=0)                         # [M, S, 2]


def build_anchors(horizons: tuple[int, ...], n_anchors: int,
                  data_root: str | None = None, episodes: int = 0,
                  pool_size: int = 4096, max_pool: int = 200_000,
                  seed: int = 0) -> tuple[torch.Tensor, dict]:
    """Return (anchors [n_anchors, len(horizons), 2], metadata)."""
    if data_root:
        eps, src = load_cached_episodes(data_root, "*train*", episodes)
        pool = episode_traj_pool(eps, horizons)
        source = str(src)
        if pool.shape[0] > max_pool:                      # subsample for FPS
            g = torch.Generator().manual_seed(seed)
            sel = torch.randperm(pool.shape[0], generator=g)[:max_pool]
            pool = pool[sel]
    else:
        pool = synth_anchor_pool(horizons, pool_size, seed)
        source = "synthetic (random unicycle rollouts)"
    anchors = furthest_point_sample(pool, n_anchors, seed=seed).contiguous()
    meta = {"method": "fps", "horizons": list(horizons), "n_anchors": n_anchors,
            "pool_size": int(pool.shape[0]), "source": source, "seed": seed}
    return anchors, meta


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--data-root", default=None,
                    help="epcache root (*train* dirs); omit for synthetic")
    ap.add_argument("--n-anchors", type=int, default=64,
                    help="vocabulary size (64 default; 20 for smoke)")
    ap.add_argument("--horizons", default="5,10,15,20",
                    help="comma-separated future step horizons")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--pool-size", type=int, default=4096,
                    help="synthetic pool size (no --data-root)")
    ap.add_argument("--max-pool", type=int, default=200_000,
                    help="cap the real-data pool (random subsample) for FPS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="force the synthetic path with a 20-anchor default")
    args = ap.parse_args(argv)

    horizons = tuple(int(x) for x in args.horizons.split(","))
    data_root = None if args.smoke else args.data_root
    n_anchors = 20 if (args.smoke and args.n_anchors == 64) else args.n_anchors
    pool_size = 256 if args.smoke else args.pool_size
    anchors, meta = build_anchors(horizons, n_anchors, data_root=data_root,
                                  episodes=args.episodes, pool_size=pool_size,
                                  max_pool=args.max_pool, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"anchors": anchors, **meta}, out)
    print(json.dumps({"saved": str(out), "shape": list(anchors.shape), **meta}),
          flush=True)
    return str(out)


if __name__ == "__main__":
    main()
