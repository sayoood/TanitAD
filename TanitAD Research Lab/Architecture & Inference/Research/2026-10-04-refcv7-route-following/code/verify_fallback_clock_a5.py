"""A5: verify, from the clips' own cached poses, that the 3 eval clips with no clock-sidecar row take the trainer's
`nominal_dt` fallback ((grid_start_s, dt_s) = (0.0, 0.1)), i.e. `clip_clock.pose_dt` returns None for them.

`pose_dt` rule (tanitad/data/clip_clock.py, re-implemented here, not imported): over steps whose BOTH ends move faster than
2.0 m/s, dt = sum|dxy| / sum(mean v); None when fewer than 10 such steps or dt outside [0.098, 0.104].
Prints sha12 only (the cache file stems are raw clip ids and are never printed).

Usage: python verify_fallback_clock_a5.py <eval cache dir> sha12 [sha12 ...]
"""
import hashlib
import os
import sys

import torch


def pose_dt(p, v_min=2.0, min_steps=10, band=(0.098, 0.104)):
    p = p.to(torch.float64)
    d = torch.hypot(p[1:, 0] - p[:-1, 0], p[1:, 1] - p[:-1, 1])
    vb = 0.5 * (p[1:, 3] + p[:-1, 3])
    m = (p[1:, 3] > v_min) & (p[:-1, 3] > v_min)
    n = int(m.sum())
    if n < min_steps:
        return None, n
    dt = float(d[m].sum() / vb[m].sum())
    return (dt if band[0] <= dt <= band[1] else None), n


def main():
    root, want = sys.argv[1], set(sys.argv[2:])
    found = set()
    for f in sorted(os.listdir(root)):
        if not f.endswith(".v2ep.pt"):
            continue
        s = hashlib.sha256(f[:-len(".v2ep.pt")].encode()).hexdigest()[:12]
        if s not in want:
            continue
        d = torch.load(os.path.join(root, f), map_location="cpu", weights_only=False)
        dt, n = pose_dt(d["poses"])
        found.add(s)
        print(s, "moving_steps", n, "pose_dt", dt, "->", "pose_dt branch" if dt is not None else "NOMINAL fallback (0.0, 0.1)")
    print("found", len(found), "of", len(want))


if __name__ == "__main__":
    main()
