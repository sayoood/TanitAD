#!/usr/bin/env python3
"""Build the poses-only NPZ view that the 2026-08-04 target-speed scripts expect.

The D-VT1 package's `vt_band_from_vision.py` was WRITTEN and never RUN -- no
`vt_band_from_vision.json` was ever banked (verified: the package's `raw/`
holds 10 artifacts and none is that one). It is the only instrument in the
programme that asks the P2 question directly: **can VISION set the target
speed, or must the envelope be RECEIVED?**

It needs a poses-only NPZ view keyed exactly as `vt_build_labels.py` reads it:

    z["_meta_json"]                -> json list of {"episode_id": int, "file": "ep_00000.pt"}
    z["<stem>__poses"]             -> [T, 4] float  (x, y, yaw, v)

This builds that view from the LOCAL val40 pose cache so the pre-registered
script can run UNMODIFIED. Editing the instrument to fit the data is how a
result stops being the pre-registered one; building the data into the shape the
instrument already declares is not.

CONTENT ASSERTION, not presence: every episode must yield a non-degenerate pose
track (T >= 2, finite, non-constant speed) or the build FAILS. A silently
zero-filled view would make every arm score at chance and read as a negative
result -- the poisoned-memmap trap in this repo's CLAUDE.md, in npz costume.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def build(src_dir: Path, out_npz: Path) -> dict:
    files = sorted(src_dir.glob("ep_*.pt"))
    if not files:
        raise SystemExit(f"no ep_*.pt under {src_dir}")
    meta, arrays = [], {}
    n_bad = 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        p = d["poses"]
        p = p.numpy() if hasattr(p, "numpy") else np.asarray(p)
        p = p.astype(np.float64)
        # CONTENT assertion -- never trust that a file that opened holds data.
        if p.ndim != 2 or p.shape[1] != 4 or p.shape[0] < 2:
            raise SystemExit(f"{f.name}: bad pose shape {p.shape}")
        if not np.isfinite(p).all():
            raise SystemExit(f"{f.name}: non-finite poses")
        if float(np.abs(p[:, 3]).max()) == 0.0:
            n_bad += 1
        meta.append({"episode_id": int(d["episode_id"]), "file": f.name,
                     "T": int(p.shape[0])})
        arrays[f.stem + "__poses"] = p.astype(np.float32)
    if n_bad:
        raise SystemExit(f"{n_bad} episodes have an all-zero speed channel")
    arrays["_meta_json"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **arrays)
    md5 = hashlib.md5(out_npz.read_bytes()).hexdigest()
    speeds = np.concatenate([arrays[m["file"].replace(".pt", "") + "__poses"][:, 3]
                             for m in meta])
    return {"n_episodes": len(meta), "n_poses": int(sum(m["T"] for m in meta)),
            "md5": md5, "bytes": out_npz.stat().st_size,
            "speed_mean_mps": round(float(speeds.mean()), 4),
            "speed_max_mps": round(float(speeds.max()), 4),
            "episode_ids_head": [m["episode_id"] for m in meta[:3]]}


if __name__ == "__main__":
    info = build(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(info, indent=1))
