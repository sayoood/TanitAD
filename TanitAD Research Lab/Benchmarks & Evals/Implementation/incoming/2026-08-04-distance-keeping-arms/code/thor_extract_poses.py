#!/usr/bin/env python3
"""Extract a POSES-ONLY view of the canonical val40 cache on Thor.

Ships back ~130 KB instead of 4.7 GB. `score_val40_lead.py` reads only `ep.poses` and
`ep.episode_id`; `frames_u8` (117 MB/episode) is never touched by the registration or the
lead block. The poses bytes are hashed here so the dev-box view can be proved identical to
the committed manifest (`manifest_EVALPOD_val40.json`, poses_sha256 per episode).

Precedent: the S3 decision-grade package already used a poses-only view of this same cache.
Read-only: opens ep_*.pt, writes ONE npz to /tmp. Parity untouched.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/nvidia/valdata/physicalai-val-0c5f7dac3b11")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/val40_poses_view.npz")

eps = sorted(SRC.glob("ep_*.pt"))
print(f"{len(eps)} episodes under {SRC}", flush=True)
blob, meta = {}, []
for p in eps:
    d = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
    poses = d["poses"].to(torch.float32).contiguous()
    acts = d["actions"].to(torch.float32).contiguous()
    man = d.get("maneuvers")
    pn = poses.numpy()
    an = acts.numpy()
    blob[f"{p.stem}__poses"] = pn
    blob[f"{p.stem}__actions"] = an
    if man is not None:
        blob[f"{p.stem}__maneuvers"] = man.numpy()
    meta.append({
        "file": p.name,
        "episode_id": int(d["episode_id"]),
        "T": int(poses.shape[0]),
        # hash the tensor's own bytes, the same way the committed manifest did
        "poses_sha256": hashlib.sha256(poses.numpy().tobytes()).hexdigest(),
        "poses_shape": list(poses.shape),
        "actions_shape": list(acts.shape),
        "frames_u8_shape": list(d["frames_u8"].shape),
        "has_maneuvers": man is not None,
    })
    print(f"  {p.name} eid={meta[-1]['episode_id']} T={meta[-1]['T']}", flush=True)

blob["_meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
np.savez_compressed(OUT, **blob)
print(f"wrote {OUT} ({OUT.stat().st_size} B) for {len(meta)} episodes", flush=True)
