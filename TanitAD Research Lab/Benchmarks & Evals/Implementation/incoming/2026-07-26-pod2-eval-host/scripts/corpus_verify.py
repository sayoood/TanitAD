"""Independent byte-level verification of an epcache split.

Emits, per episode: file, episode_id, T, sha256 over the RAW poses[T,4] bytes.
sha256(poses) is the key -- NOT episode_id, which COLLIDES (the parity train has
2,376 episodes and only 2,342 unique episode_ids) and over-reports overlap.

Reads with torch.load(mmap=True): only the poses pages fault in, so hashing a
2,376-episode cache costs ~2 min and a few MB of IO instead of 260 GB. This
script NEVER writes into the cache directory.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    src = Path(a.src)
    files = sorted(src.glob("ep_*.pt"))
    if a.limit:
        files = files[:a.limit]
    print(f"[verify] {a.label or src}: {len(files)} ep_*.pt", flush=True)

    rows, t0 = [], time.time()
    errs = []
    for i, f in enumerate(files):
        try:
            d = torch.load(f, weights_only=False, map_location="cpu", mmap=True)
            po = torch.as_tensor(d["poses"]).clone().contiguous()
            eid = d.get("episode_id")
            try:
                eid = int(eid)
            except Exception:
                eid = str(eid)
            h = hashlib.sha256(po.numpy().tobytes()).hexdigest()
            keys = sorted(d.keys()) if hasattr(d, "keys") else []
            rows.append({"file": f.name, "episode_id": eid,
                         "T": int(po.shape[0]),
                         "poses_sha256": h,
                         "keys": keys if i == 0 else None})
        except Exception as ex:  # noqa: BLE001
            errs.append({"file": f.name, "error": f"{type(ex).__name__}: {ex}"})
        if i and i % 400 == 0:
            print(f"  {i}/{len(files)} {time.time()-t0:.0f}s", flush=True)

    hashes = [r["poses_sha256"] for r in rows]
    eids = [r["episode_id"] for r in rows]
    out = {
        "label": a.label,
        "source_dir": src.as_posix(),
        "host": os.uname().nodename if hasattr(os, "uname") else "?",
        "n_files_listed": len(files),
        "n_episodes_read": len(rows),
        "n_errors": len(errs),
        "errors": errs[:20],
        "n_unique_poses_sha256": len(set(hashes)),
        "n_unique_episode_id": len(set(eids)),
        "T_min": min((r["T"] for r in rows), default=None),
        "T_max": max((r["T"] for r in rows), default=None),
        "seconds": round(time.time() - t0, 1),
        "episodes": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[verify] {a.label}: n={len(rows)} uniq_sha={len(set(hashes))} "
          f"uniq_eid={len(set(eids))} T=[{out['T_min']},{out['T_max']}] "
          f"{out['seconds']}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
