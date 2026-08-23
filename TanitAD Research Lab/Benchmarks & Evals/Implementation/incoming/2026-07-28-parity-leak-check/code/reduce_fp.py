"""Reduce a fingerprint JSONL to HASHES ONLY, so the evidence can be staged.

The full fingerprints carry `poses_b64` (raw float32 pose bytes of a GATED
corpus) and per-frame digests — 36 MB and actual corpus content, so they stay on
the pod.  What makes the verdict independently re-checkable is the hash set, and
that carries no corpus content at all.  This writes exactly that.

⛔ No clip UUIDs are emitted: `ep_XXXXX` is a cache-local tag and `episode_id` is
an integer written by the cache builder.
"""
import json, sys
from pathlib import Path

KEEP = ("file", "poses_sha256", "poses_xy_sha256", "poses_yaw_sha256",
        "poses_v_sha256", "actions_sha256", "maneuvers_sha256", "frames_sha256",
        "episode_id", "T", "T_frames", "frame_shape", "bytes", "mode")

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
hdr, eps = rows[0], rows[1:]
hdr["reduction"] = ("HASHES ONLY — `poses_b64` and `frame_digests` stripped: they "
                    "are raw bytes of a gated corpus. The hash set is the evidence "
                    "and carries no corpus content.")
hdr["n_episodes_reduced"] = len(eps)
out = {"_header": hdr,
       "episodes": [{k: e[k] for k in KEEP if k in e} for e in eps]}
dst.write_text(json.dumps(out, indent=0), encoding="utf-8")
print(f"{src.name}: {len(eps)} eps -> {dst} ({dst.stat().st_size/1e6:.2f} MB)")
