"""Download ONE real Cosmos-Drive-Dreams clip pair and run verify_real_clip.

The documented pod step from the cosmos-loader intake: settles the honest
limitations (pose glob correctness, axis order, A8 consequence-dominance,
plausible signal ranges) on real bytes before cosmos joins the D-010 mix.

Usage (pod):  python scripts/cosmos_verify.py --root /workspace/data/cosmos
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

REPO = "nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams"


def _explore(api, want_video: str = "front_wide", max_entries: int = 4000):
    """Bounded BFS over the repo tree: find one front-wide mp4 and the
    vehicle_pose entries sharing its clip id."""
    mp4 = None
    pose_dirs = []
    seen = 0
    q = deque([""])
    while q and seen < max_entries:
        path = q.popleft()
        try:
            entries = list(api.list_repo_tree(REPO, repo_type="dataset",
                                              path_in_repo=path,
                                              recursive=False))
        except Exception:
            continue
        for e in entries:
            seen += 1
            is_dir = type(e).__name__ == "RepoFolder" or getattr(
                e, "size", None) is None
            if is_dir:
                name = e.path.lower()
                if "vehicle_pose" in name:
                    pose_dirs.append(e.path)
                # descend into promising branches only
                if any(k in name for k in ("video", "pose", "cosmos", "hdmap",
                                           "sample", "rds", want_video, "v0",
                                           "data")) or path == "":
                    q.append(e.path)
            elif e.path.endswith(".mp4") and want_video in e.path.lower() \
                    and mp4 is None:
                mp4 = e.path
        if mp4 and pose_dirs:
            break
    return mp4, pose_dirs, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/data/cosmos")
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()

    mp4, pose_dirs, seen = _explore(api)
    print(f"[cosmos] explored {seen} entries; mp4={mp4}; "
          f"pose_dirs={pose_dirs[:3]}")
    if not mp4:
        raise SystemExit("[cosmos] no front-wide mp4 found — repo layout "
                         "differs from the loader's assumption; record in "
                         "the data card and adapt discover_clips.")
    clip_id = Path(mp4).stem.split(".")[0]

    # download the video + any pose files carrying the clip id
    got = [hf_hub_download(REPO, mp4, repo_type="dataset", local_dir=args.root)]
    for pd in pose_dirs[:3]:
        try:
            for e in api.list_repo_tree(REPO, repo_type="dataset",
                                        path_in_repo=pd, recursive=False):
                if clip_id[:12] in e.path and getattr(e, "size", 0):
                    got.append(hf_hub_download(REPO, e.path,
                                               repo_type="dataset",
                                               local_dir=args.root))
        except Exception as ex:
            print(f"[cosmos] pose dir {pd}: {type(ex).__name__}: {ex}")
    print(f"[cosmos] downloaded {len(got)} files")

    from tanitad.data.cosmos_drive import discover_clips, verify_real_clip
    clips = discover_clips(args.root)
    print(f"[cosmos] discover_clips found {len(clips)} clip(s)")
    if not clips:
        raise SystemExit("[cosmos] pairing failed — layout mismatch; adapt "
                         "discover_clips (record findings in the data card).")
    stats = verify_real_clip(clips[0])
    print(json.dumps(stats, indent=2))
    Path(args.root, "verify_real_clip.json").write_text(
        json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
