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
    # Split-selective build (pod3, 2026-07-13). split_clips() still runs on the
    # FULL clip list, so the train cache key is byte-identical to a --only both
    # run — this only skips the build loop for the other split, letting a
    # disk/RAM-bound pod finish train first and backfill val later.
    ap.add_argument("--only", choices=["train", "val", "both"], default="both",
                    help="build only this split (default both; key unchanged)")
    args = ap.parse_args()

    cfg = base250cam_config()
    clips = discover_r0_clips(args.root)
    if args.episodes:
        clips = clips[:args.episodes]
    print(f"[build] {len(clips)} clips discovered", flush=True)

    # f-theta canonicalization self-check (D-016 fix, GEOMETRY_INTEGRITY_AUDIT):
    # PROVE the corrected crop lands f_eff ~= F_REF (266) before a 40-min decode,
    # so a bad intrinsics/poly change fails loudly instead of silently shipping
    # the wrong zoom the audit found (nominal path was ~434 px).
    from tanitad.data.calib import F_REF, ftheta_feff_report
    from tanitad.data.physicalai import intrinsics_for_clip
    if clips:
        cid = clips[0]["clip_id"]
        rep = ftheta_feff_report(intrinsics_for_clip(cid, args.root))
        print(f"[build] f-theta f_eff check (clip {cid}): "
              f"after={rep['f_eff_after']} before(nominal)={rep['f_eff_before_nominal']} "
              f"retained_hfov={rep['retained_hfov_after_deg']}deg", flush=True)
        assert abs(rep["f_eff_after"] - F_REF) < 8.0, (
            f"corrected f_eff {rep['f_eff_after']} != {F_REF} (D-016); ABORT")

    tr, va = split_clips(clips, val_frac=0.2, seed=cfg.train.seed)   # I3
    print(f"[build] split: {len(tr)} train / {len(va)} val "
          f"(seed {cfg.train.seed})", flush=True)

    cache = Path(args.root) / "_epcache"
    # "calib" tags the canonicalization generation: each f-theta fix makes a
    # FRESH cache key so a corrected cache can never collide with / silently
    # reload superseded episodes. v1: nominal->f-theta focal fix. v2: per-clip
    # (cx,cy) principal-point-centered crop (two-rig vertical fix) — different
    # pixels for rig-B clips, so it MUST invalidate any v1 (geometric-center) cache.
    params = {"size": cfg.encoder.image_size, "n_stack": 3, "hz": 10,
              "calib": "ftheta_v2"}
    for cs, split in ((tr, "train"), (va, "val")):
        if args.only != "both" and split != args.only:
            print(f"[build] --only={args.only} -> skipping {split} split",
                  flush=True)
            continue
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
