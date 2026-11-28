"""Video-motion vs pose-speed sync check, old cache vs new (chunk-pairing gate).

Per ep: Pearson corr between per-step frame change (mean |frames[t+1,-3:] -
frames[t,-3:]|) and cached speed v_t = 0.5*(v[t]+v[t+1]). A correctly paired
episode correlates positively (faster -> more pixel motion); the old cache's
chunk-1 eps were paired with pose[0:121] (a ~4 s desync), driving corr negative
or to noise (e12d35ed was the decisive case). Chunk-1 eps should now match the
chunk-0 distribution.

Usage: python3 check_cosmos_sync.py OLD_EPDIR NEW_EPDIR MANIFEST_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/TanitAD/stack")
from tanitad.data.mixing import load_episode  # noqa: E402


def motion_speed_corr(ep_path: Path) -> float:
    ep = load_episode(str(ep_path), mmap=True)
    fr = ep.frames[:, -3:].float()
    motion = (fr[1:] - fr[:-1]).abs().mean(dim=(1, 2, 3)).numpy()
    v = ep.poses[:, 3].float().numpy()
    v_mid = 0.5 * (v[1:] + v[:-1])
    if motion.std() < 1e-6 or v_mid.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(motion, v_mid)[0, 1])


def main():
    old_dir, new_dir, man_f = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    man = json.loads(Path(man_f).read_text())["eps"]
    rows = []
    for k in sorted(man, key=int):
        e = man[k]
        old_f = old_dir / f"ep_{int(k):05d}.pt"
        new_f = new_dir / f"ep_{int(k):05d}.pt"
        c_old = motion_speed_corr(old_f) if old_f.exists() else float("nan")
        c_new = motion_speed_corr(new_f)
        rows.append((int(k), e["chunk"], e["clip"][:8], c_old, c_new))
        print(f"ep{int(k):02d} ch{e['chunk']} {e['clip'][:8]} "
              f"old={c_old:+.3f} new={c_new:+.3f}", flush=True)
    for ch in (0, 1):
        o = np.array([r[3] for r in rows if r[1] == ch])
        n = np.array([r[4] for r in rows if r[1] == ch])
        print(f"chunk-{ch}: n={len(o)} old corr median {np.nanmedian(o):+.3f} "
              f"-> new {np.nanmedian(n):+.3f}")


if __name__ == "__main__":
    main()
