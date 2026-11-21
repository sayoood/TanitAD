"""Content-fingerprint every episode in a latent cache so two caches built on
DIFFERENT hosts (and different corpus keys) can be tested for episode overlap.

pod3's idm-proof latents store only {z, poses, actions} -- no episode_id, no src
-- so identity CANNOT be read off metadata. This computes fingerprints from the
tensor content instead.

Three fingerprints per episode, from strictest to loosest, because the two caches
may differ in dtype/precision without being different episodes:
  fp_exact : md5 of float32 poses bytes            (bit-identical build)
  fp_r3    : md5 of poses rounded to 1e-3          (tolerant of float noise)
  fp_stat  : coarse per-episode summary statistics (tolerant of frame offsets)

Also emits the raw stats so a near-miss can be diagnosed rather than guessed at.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def md5b(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def main(latdir: str, out: str) -> None:
    recs = []
    for p in sorted(Path(latdir).glob("*.pt")):
        o = torch.load(p, weights_only=False, map_location="cpu")
        po = o["poses"].float().numpy().astype(np.float32)
        ac = o["actions"].float().numpy().astype(np.float32)
        rec = {
            "tag": p.stem,
            "T": int(po.shape[0]),
            "pose_dim": int(po.shape[1]) if po.ndim == 2 else None,
            "act_dim": int(ac.shape[1]) if ac.ndim == 2 else None,
            "episode_id": int(o["episode_id"]) if "episode_id" in o else None,
            "fp_exact": md5b(po),
            "fp_r3": md5b(np.round(po, 3)),
            "fp_act_r3": md5b(np.round(ac, 3)),
            # coarse stats: survive small build differences, still highly discriminative
            "fp_stat": md5b(np.round(np.concatenate([
                po.mean(0), po.std(0), po.min(0), po.max(0)]), 2)),
            "pose_mean": [round(float(x), 5) for x in po.mean(0)],
            "pose_std": [round(float(x), 5) for x in po.std(0)],
            "act_mean": [round(float(x), 5) for x in ac.mean(0)],
        }
        recs.append(rec)
    Path(out).write_text(json.dumps({"latdir": latdir, "n": len(recs), "records": recs}, indent=1))
    print("WROTE", out, "n=", len(recs), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
