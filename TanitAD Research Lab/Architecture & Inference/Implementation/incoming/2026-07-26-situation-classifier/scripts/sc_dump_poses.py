"""Situation classifier — STEP 0 (pod2, CPU): dump every cached episode's POSES.

Why this exists: `physicalai.build_episode` stores ``poses = [x, y, yaw, v]`` at 10 Hz, produced by
``signals_at(ego, t_query)`` with ``t_query = linspace(t_frames[0], t_frames[-1], n_target)`` and
then trimmed by ``k = n_stack - 1 = 2``.  **The ego trajectory the encoder's frames correspond to is
therefore already inside the episode file, on the camera clock, index-for-index with the frames.**

That removes H2's whole alignment problem for every label that is a function of the ego trajectory
alone (lane change, roundabout, the turn half of intersection): those labels are built on the
episode index directly and cannot be mis-aligned by construction.

Only the obstacle-derived quantities (cross traffic, per-camera visibility) live on the
egomotion/obstacle clock and need a mapping; `sc_build_labels.py` recovers it by fitting the two
linspace endpoints against these same poses and publishes the residual **in metres**.

Output: one compressed npz with `p{i}` -> float32 [T,4] plus a meta json. ~10 MB total.

usage (pod2):  python3 sc_dump_poses.py --out /workspace/sitclf/poses
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

CACHES = {
    "train": "/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894",
    "val": "/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    packs, meta = {}, []
    t0 = time.time()
    for cache, root in CACHES.items():
        files = sorted(f for f in os.listdir(root) if f.startswith("ep_") and f.endswith(".pt"))
        for f in files:
            ep = torch.load(os.path.join(root, f), map_location="cpu",
                            weights_only=True, mmap=True)
            P = ep["poses"].to(torch.float32).numpy()
            i = len(meta)
            packs[f"p{i}"] = P
            meta.append({"i": i, "cache": cache, "file": f,
                         "episode_id": int(ep["episode_id"]) if "episode_id" in ep else -1,
                         "T": int(P.shape[0]),
                         "T_frames": int(ep["frames"].shape[0]) if "frames" in ep
                         else int(ep["frames_u8"].shape[0])})
            if i % 250 == 0:
                print(f"[poses] {i} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(args.out, "poses.npz"), **packs)
    json.dump(meta, open(os.path.join(args.out, "poses_meta.json"), "w"))
    print(f"[poses] {len(meta)} episodes -> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
