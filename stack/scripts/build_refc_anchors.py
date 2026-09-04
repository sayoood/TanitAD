"""Build the REF-C anchor vocabulary via furthest-point sampling (FPS).

**FPS LOSES TO A SINGLE STRAIGHT LINE AT THE 6 s / 8-SLOT GRID. MEASURED 2026-09-04**
on 19,602 held-out windows over the 141 B1-v7.2 EVAL clips (oracle-in-vocabulary
ADE 0-2 s, lower is better):

    zero path (no information)      14.3264
    straight-line control            0.6843
    synthetic pool (refcv3 shipped)  1.0882
    best FPS variant                 0.7666   <- WORSE THAN A STRAIGHT LINE
    k-means, slot-normalised         0.3796   <- shipped in refcv4

The argument below - that k-means collapses onto the straight mode - is REAL, and was
right for the 2 s / 4-slot grid this tool still DEFAULTS to (--horizons 5,10,15,20,
--n-anchors 64). It does not survive at 6 s. FPS minimises worst-case COVERING RADIUS;
the gate that decides whether a model can plan is MEAN ADE to the nearest anchor, which
is Lloyd's objective, not FPS's. The fix for the straight-mode collapse is PER-SLOT
NORMALISATION, not a different sampler.

Why this matters more than a tuning note: refcv3 trained 40,284 steps on the synthetic
fallback because nobody passed --anchors, and a perfect chooser on that fan still lost
to a hold-action control. A VOCABULARY IS A CEILING - no selector, however good, can
pick a trajectory the fan does not contain. Rebuilding at 6 s with FPS reproduces that
failure with a different number.

This tool therefore REFUSES a >4-slot build unless --i-know-fps-loses-at-6s is passed.
The validated k-means/slot-norm builder and its held-out gate live at
"TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-refcv4-launch/".
Note it saves {"anchors": ..., **meta} - a DICT. refc_v3_train.py unwraps that as of
2026-09-04 (anc["anchors"] if isinstance(anc, dict)); older trainers do not.


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
                  seed: int = 0, v2_cache: list[str] | None = None
                  ) -> tuple[torch.Tensor, dict]:
    """Return (anchors [n_anchors, len(horizons), 2], metadata)."""
    if v2_cache:
        # v2 compressed cache (*.v2ep.pt, e.g. the B1 corpus): the pool needs
        # POSES ONLY, and build_v2_providers keeps poses resident from a
        # metadata-only mmap scan — no frame is ever decoded here, so anchors
        # off a multi-thousand-clip corpus cost minutes of CPU.
        from tanitad.data.v2_dataset import build_v2_providers
        eps = build_v2_providers(v2_cache, lru_size=1)
        if episodes:
            eps = eps[:episodes]
        pool = episode_traj_pool(eps, horizons)
        source = f"v2-cache {list(v2_cache)} ({len(eps)} clips)"
        if pool.shape[0] > max_pool:                      # subsample for FPS
            g = torch.Generator().manual_seed(seed)
            sel = torch.randperm(pool.shape[0], generator=g)[:max_pool]
            pool = pool[sel]
    elif data_root:
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
    ap.add_argument("--v2-cache", nargs="+", default=None,
                    help="v2 compressed cache dir(s) of *.v2ep.pt (e.g. the "
                         "B1 corpus). Poses-only scan — no frame decode. "
                         "Mutually exclusive with --data-root.")
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
    ap.add_argument("--i-know-fps-loses-at-6s", action="store_true",
                    help="build a >4-slot FPS vocabulary anyway; it is MEASURED to "
                         "lose to a straight line (0.7666 vs 0.6843)")
    ap.add_argument("--smoke", action="store_true",
                    help="force the synthetic path with a 20-anchor default")
    args = ap.parse_args(argv)

    if args.data_root and args.v2_cache:
        raise SystemExit("pass at most one of --data-root / --v2-cache")
    horizons = tuple(int(x) for x in args.horizons.split(","))
    # MEASURED 2026-09-04: FPS at the 6 s / 8-slot grid scores 0.7666 oracle-in-
    # vocabulary ADE against a straight line's 0.6843 - it is WORSE THAN DRAWING A
    # STRAIGHT LINE, and a vocabulary is a ceiling no selector can beat. refcv3 spent
    # 40,284 steps proving that. Refuse rather than hand back a fan that cannot plan.
    if len(horizons) > 4 and not args.i_know_fps_loses_at_6s:
        raise SystemExit(
            "REFUSED: %d slots requested, but FPS is MEASURED to lose to a single "
            "straight line beyond the 2 s / 4-slot grid (0.7666 vs 0.6843 oracle-in-"
            "vocabulary ADE, 19,602 held-out windows, 2026-09-04). Use the validated "
            "k-means/slot-normalised builder in 'TanitAD Research Lab/Architecture & "
            "Inference/Research/2026-09-04-refcv4-launch/' (it scores 0.3796), or pass "
            "--i-know-fps-loses-at-6s to build it anyway." % len(horizons))
    data_root = None if args.smoke else args.data_root
    v2_cache = None if args.smoke else args.v2_cache
    n_anchors = 20 if (args.smoke and args.n_anchors == 64) else args.n_anchors
    pool_size = 256 if args.smoke else args.pool_size
    anchors, meta = build_anchors(horizons, n_anchors, data_root=data_root,
                                  episodes=args.episodes, pool_size=pool_size,
                                  max_pool=args.max_pool, seed=args.seed,
                                  v2_cache=v2_cache)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"anchors": anchors, **meta}, out)
    print(json.dumps({"saved": str(out), "shape": list(anchors.shape), **meta}),
          flush=True)
    return str(out)


if __name__ == "__main__":
    main()
