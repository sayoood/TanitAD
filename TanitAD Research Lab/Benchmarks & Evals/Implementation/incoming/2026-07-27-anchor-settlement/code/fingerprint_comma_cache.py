"""Content fingerprints for a comma2k19 episode cache — BY CONTENT, never by name.

WHY THIS EXISTS
---------------
`+0.3308` (the comma heading-repair anchor) was measured on
`comma2k19-val-76b6e94a97a1` (64 segs).  `idm_head_v1` ("A0") was TRAINED on
`cm_[0:40]` of `comma2k19-val-61c46fca8f7f` (90 eps).  The two caches have
different segment counts, so the `cm_*` tag indices are NOT comparable and the
overlap between the anchor's eval set and A0's training set was UNKNOWN.

⛔ This program has measured **600/600 filename overlap with 0/600 real
overlap**, and separately **4 of 36 val episodes that would have leaked (11 %)**
in a cache carrying no `episode_id`.  Therefore this script compares **raw
bytes**, and treats every name-derived key (`ep_XXXXX.pt`, `cm_XXXXX`,
`episode_id`) as a CROSS-CHECK ONLY, never as the answer.

`episode_id` is explicitly name-derived: `comma2k19.episode_id_of` is
`sha1(f"{route_folder}/{segment_name}")[:8]` — a hash OF A PATH, not of content.

WHAT IS HASHED (per episode)
----------------------------
  poses_sha256      sha256 of `poses` [T,4] float32 raw bytes  <- the brief's
                    named primary ("sha256/md5 of raw pose bytes")
  poses_xy_sha256   sha256 of the (x_east, y_north) columns only — robust to the
                    heading label protocol, which is the one column a repair or a
                    rebuild could legitimately change
  poses_v_sha256    sha256 of the speed column only
  actions_sha256    sha256 of `actions` [T,2] float32
  frames_sha256     sha256 of `frames_u8` [T,C,S,S] uint8 — the RAW SENSOR bytes,
                    independent of every label protocol
  frame_digests     per-frame sha1[:16] over all T frames — supports a
                    SHIFT-TOLERANT match (same segment, different start offset),
                    which an all-or-nothing whole-tensor hash cannot see
  poses_b64         the exact float32 pose bytes, base64 — so the intersection
                    step can do lag/near-duplicate analysis without re-reading
                    11 GB of cache

Frames are hashed incrementally frame-by-frame so the peak RSS stays at one
frame, not 2x176 MB.  sha256 over ordered chunks == sha256 over the contiguous
buffer, so `frames_sha256` is still the whole-tensor hash.

Usage:
    python3 fingerprint_comma_cache.py --cache <dir> --out <json> [--tags 0,3,6]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fingerprint(path: Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    poses = d["poses"]
    actions = d["actions"]
    frames = d["frames_u8"]

    assert poses.dtype == torch.float32, poses.dtype
    assert frames.dtype == torch.uint8, frames.dtype

    p = poses.contiguous().numpy()
    a = actions.contiguous().numpy()

    fh = hashlib.sha256()
    frame_digests = []
    T = int(frames.shape[0])
    for t in range(T):
        fb = frames[t].contiguous().numpy().tobytes()
        fh.update(fb)
        frame_digests.append(hashlib.sha1(fb).hexdigest()[:16])

    v = p[:, 3].astype(np.float64)
    xy = p[:, :2].astype(np.float64)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)

    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        # --- content hashes (the answer) ---------------------------------- #
        "poses_sha256": _sha256(p.tobytes()),
        "poses_xy_sha256": _sha256(np.ascontiguousarray(p[:, :2]).tobytes()),
        "poses_yaw_sha256": _sha256(np.ascontiguousarray(p[:, 2]).tobytes()),
        "poses_v_sha256": _sha256(np.ascontiguousarray(p[:, 3]).tobytes()),
        "actions_sha256": _sha256(a.tobytes()),
        "frames_sha256": fh.hexdigest(),
        "frame_digests": frame_digests,
        # --- name-derived, CROSS-CHECK ONLY -------------------------------- #
        "episode_id": int(d["episode_id"]),
        # --- shape + summary (near-duplicate screen if hashes are all 0) ---- #
        "T": T,
        "frame_shape": list(frames.shape[1:]),
        "v_min": float(v.min()),
        "v_max": float(v.max()),
        "v_mean": float(v.mean()),
        "path_len_m": float(step.sum()),
        "yaw_absmax": float(np.abs(p[:, 2]).max()),
        # --- exact bytes for the intersection step ------------------------- #
        "poses_b64": base64.b64encode(p.tobytes()).decode(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", default="", help="comma-separated ep indices; default = all")
    args = ap.parse_args()

    cache = Path(args.cache)
    files = sorted(p for p in cache.glob("ep_*.pt"))
    if args.tags:
        want = {int(x) for x in args.tags.split(",") if x.strip()}
        files = [p for p in files if int(re.findall(r"\d+", p.stem)[-1]) in want]
    assert files, f"no ep_*.pt under {cache}"

    log(f"{cache}  ->  {len(files)} episodes")
    eps = {}
    t0 = time.time()
    for i, f in enumerate(files):
        eps[f.stem] = fingerprint(f)
        if i % 10 == 0 or i == len(files) - 1:
            log(f"  {i+1}/{len(files)}  {f.stem}  ({time.time()-t0:.0f}s)")

    out = {
        "what": "content fingerprints of a comma2k19 episode cache (BY CONTENT)",
        "cache": str(cache),
        "cache_key": cache.name,
        "n_episodes_in_cache": len(list(cache.glob("ep_*.pt"))),
        "n_fingerprinted": len(eps),
        "host": __import__("socket").gethostname(),
        "torch": torch.__version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "episodes": eps,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    log(f"wrote {args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
