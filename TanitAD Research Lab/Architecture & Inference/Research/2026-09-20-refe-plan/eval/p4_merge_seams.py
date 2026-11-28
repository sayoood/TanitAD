#!/usr/bin/env python3
"""Merge per-shard seam npz files (p4_tracker_comp.py --shard i/N) into ONE seam in W3 export order, refusing on a
duplicate or a missing token.  python eval/p4_merge_seams.py --out <seam.npz> --expect <reference seam.npz> <shard.npz> ...
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", required=True, help="the seam whose token set and order the merge must reproduce")
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    ref = np.load(a.expect)
    order = [str(t) for t in ref["token"]]
    got: dict = {}
    arm = None
    for p in a.shards:
        z = np.load(p)
        arm = str(z["arm"])
        for t, fp, ps in zip(z["token"], z["fingerprint"], z["poses"]):
            if str(t) in got:
                sys.exit(f"duplicate token {t} in {p}")
            got[str(t)] = (str(fp), ps)
    missing = [t for t in order if t not in got]
    extra = [t for t in got if t not in set(order)]
    if missing or extra:
        sys.exit(f"token-set mismatch: missing {len(missing)} extra {len(extra)}")
    fps = [got[t][0] for t in order]
    if fps != [str(f) for f in ref["fingerprint"]]:
        sys.exit("fingerprint mismatch against the reference seam")
    np.savez(a.out, token=np.array(order), fingerprint=np.array(fps), poses=np.stack([got[t][1] for t in order]),
             sampling=ref["sampling"], arm=np.array(arm))
    print(f"ZZMERGE_OK {len(order)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
