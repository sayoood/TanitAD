"""Content fingerprints for a PhysicalAI episode cache — BY CONTENT, never by name.

WHY THIS EXISTS
---------------
Every open-loop and closed-loop number this program has published is scored on
`physicalai-val-0c5f7dac3b11`.  `MODEL_REGISTRY.md` calls that split
"episode-disjoint from train" — but that is an `episode_id` CLAIM, and
`episode_id` here is a *name-derived* integer written by the cache builder, not
a function of the pixels.  Two facts make the claim untrustworthy until checked
by content:

  * a 78 % leak (62/79) has already been MEASURED on the sibling val cache
    `physicalai-val-f1b378f295ae` — **by episode-id intersection, never by
    content**;
  * a sibling has just settled that 2 of 22 comma evaluation episodes were
    bit-identical to TRAINING episodes, found ONLY by hashing raw bytes, and
    this program has separately measured **600/600 filename overlap with
    0/600 real overlap**.

⛔ Therefore: names, `episode_id`s and tag indices are recorded as CROSS-CHECKS
that could have disagreed.  They are never the evidence.

WHAT IS HASHED (per episode)
----------------------------
  poses_sha256      sha256 of `poses` [T,4] float32 raw bytes   <- primary
  poses_xy_sha256   sha256 of the (x, y) columns only
  poses_yaw_sha256  sha256 of the yaw column only
  poses_v_sha256    sha256 of the speed column only
  actions_sha256    sha256 of `actions` [T,2] float32
  maneuvers_sha256  sha256 of `maneuvers` [T] int64
  frames_sha256     sha256 of `frames_u8` [T,9,256,256] uint8 — the RAW SENSOR
                    bytes, independent of every label protocol
  frame_digests     per-frame sha1[:16] over all T frames — supports a
                    SHIFT-TOLERANT / PARTIAL match (a val episode that is a
                    sub-window of a train clip), which an all-or-nothing
                    whole-tensor hash cannot see
  poses_b64         the exact float32 pose bytes, base64 — lets the intersection
                    step assert bitwise equality without re-reading 260 GB

Frames are hashed frame-by-frame so peak RSS stays at one episode.  sha256 over
ordered chunks == sha256 over the contiguous buffer, so `frames_sha256` is still
the whole-tensor hash.

MODES
-----
  --mode full    load the whole episode (frames included).  ~0.6 s/ep at
                 189 MB/s single-threaded on pod3.
  --mode poses   torch.load(mmap=True), fault in ONLY the poses pages.  ~34
                 ms/ep.  Used for the poses-only VIEW directories, which carry
                 bit-identical poses but no frames.

Output is JSONL, one episode per line, flushed as it goes, so a killed run
still yields every episode it finished.

Usage:
    python3 fingerprint_pai_cache.py --cache <dir> --out <jsonl> \
        [--mode full|poses] [--workers 12] [--limit N]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fingerprint(path_mode) -> dict:
    path, mode = path_mode
    path = Path(path)
    d = torch.load(path, map_location="cpu", weights_only=False,
                   mmap=(mode == "poses"))
    poses = d["poses"]
    assert poses.dtype == torch.float32, poses.dtype
    p = np.ascontiguousarray(poses.numpy())

    v = p[:, 3].astype(np.float64)
    xy = p[:, :2].astype(np.float64)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)

    rec = {
        "file": path.name,
        "bytes": path.stat().st_size,
        # --- content hashes (the answer) ---------------------------------- #
        "poses_sha256": _sha256(p.tobytes()),
        "poses_xy_sha256": _sha256(np.ascontiguousarray(p[:, :2]).tobytes()),
        "poses_yaw_sha256": _sha256(np.ascontiguousarray(p[:, 2]).tobytes()),
        "poses_v_sha256": _sha256(np.ascontiguousarray(p[:, 3]).tobytes()),
        # --- name-derived, CROSS-CHECK ONLY ------------------------------- #
        "episode_id": int(d["episode_id"]) if "episode_id" in d else None,
        # --- shape + summary ---------------------------------------------- #
        "T": int(p.shape[0]),
        "v_min": float(v.min()),
        "v_max": float(v.max()),
        "v_mean": float(v.mean()),
        "path_len_m": float(step.sum()),
        "yaw_absmax": float(np.abs(p[:, 2]).max()),
        # --- exact bytes for the intersection step -------------------------- #
        "poses_b64": base64.b64encode(p.tobytes()).decode(),
        "mode": mode,
    }

    if mode == "full":
        a = np.ascontiguousarray(d["actions"].numpy())
        rec["actions_sha256"] = _sha256(a.tobytes())
        if "maneuvers" in d:
            mv = np.ascontiguousarray(d["maneuvers"].numpy())
            rec["maneuvers_sha256"] = _sha256(mv.tobytes())
        frames = d["frames_u8"]
        assert frames.dtype == torch.uint8, frames.dtype
        fh = hashlib.sha256()
        fd = []
        T = int(frames.shape[0])
        for t in range(T):
            fb = np.ascontiguousarray(frames[t].numpy()).tobytes()
            fh.update(fb)
            fd.append(hashlib.sha1(fb).hexdigest()[:16])
        rec["frames_sha256"] = fh.hexdigest()
        rec["frame_digests"] = fd
        rec["frame_shape"] = list(frames.shape[1:])
        rec["T_frames"] = T
    del d
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="full", choices=("full", "poses"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.cache)
    files = sorted(p for p in cache.glob("ep_*.pt"))
    if args.limit:
        files = files[:args.limit]
    assert files, f"no ep_*.pt under {cache}"

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    hdr = {
        "_header": True,
        "what": "content fingerprints of a PhysicalAI episode cache (BY CONTENT)",
        "cache": str(cache),
        "cache_key": cache.name,
        "n_episodes_in_cache": len(list(cache.glob("ep_*.pt"))),
        "n_to_fingerprint": len(files),
        "mode": args.mode,
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log(f"{cache} -> {len(files)} episodes  mode={args.mode} workers={args.workers}")
    t0 = time.time()
    tasks = [(str(f), args.mode) for f in files]
    with outp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(hdr) + "\n")
        fh.flush()
        if args.workers > 1:
            import multiprocessing as mp
            ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
            with ctx.Pool(args.workers) as pool:
                for i, rec in enumerate(pool.imap(fingerprint, tasks, chunksize=1)):
                    fh.write(json.dumps(rec) + "\n")
                    if i % 100 == 0:
                        fh.flush()
                        log(f"  {i+1}/{len(files)}  ({time.time()-t0:.0f}s)")
        else:
            for i, t in enumerate(tasks):
                fh.write(json.dumps(fingerprint(t)) + "\n")
                if i % 100 == 0:
                    fh.flush()
                    log(f"  {i+1}/{len(files)}  ({time.time()-t0:.0f}s)")
    log(f"wrote {outp}  ({outp.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
