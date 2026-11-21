#!/usr/bin/env python
"""pod_dose_layout.py -- dose step of note PREREG_V8F_PROOF_ADD42_N2 section 3: the row layout and the ego kinematics of the training clips, from the pod token cache.

  python pod_dose_layout.py --tree /workspace/v8proof/tree_leadsup --cache /workspace/v8proof/data/cache_pod --split train --lo 0 --hi 4572 --out <layout.npz>

Runs ON THE POD, CPU only, one thread, ONE clip in memory at a time, no GPU, no label. Clips = positions lo .. hi-1 of the ascending sha12 order of the split manifest (the order of
extract_states.py and extract_probe_set.py). Per clip the file clip_id.pt is read as extract_probe_set.py reads it; only v, a, yaw_rate and the frame count T are kept.
Writes <out> (never over an existing file) with: clips (str [n_clips]), clip_idx (int32 [N]), t (int16 [N]), kin (float32 [N, 3] = v, a, yaw_rate), frames (int32 [n_clips]);
and <out>.json (split, lo, hi, n_manifest, counts, sha256 of the output). The label pack is built from this file on the dev box (build_dose_pack.py). Tier T0 input preparation.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_layout(clips, load_fn):
    """clips: list of ids in order. load_fn(id) -> (v, a, yaw_rate) as three 1-D arrays of the clip's T frames. Returns the arrays of the layout file."""
    Ci, Ts, Ks, fr = [], [], [], []
    for j, c in enumerate(clips):
        v, a, y = (np.asarray(x, np.float32).reshape(-1) for x in load_fn(c))
        if not (len(v) == len(a) == len(y)) or len(v) == 0:
            raise SystemExit("REFUSED: clip %s has kinematic arrays of lengths %d %d %d" % (c, len(v), len(a), len(y)))
        T = len(v)
        Ci.append(np.full(T, j, np.int32))
        Ts.append(np.arange(T, dtype=np.int16))
        Ks.append(np.stack([v, a, y], 1))
        fr.append(T)
    return {"clips": np.array([str(c) for c in clips]), "clip_idx": np.concatenate(Ci), "t": np.concatenate(Ts), "kin": np.concatenate(Ks).astype(np.float32),
            "frames": np.array(fr, np.int32)}


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", required=True, choices=("train",))
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out) or os.path.exists(a.out + ".json"):
        raise SystemExit("REFUSED: %s exists (a layout file is written once)" % a.out)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(a.tree, "stack"))
    from tanitad.data import v8f_mini_data as D
    man = D.read_manifest(a.cache, a.split)
    order = sorted([c for c, _ in man], key=D.sha12)
    if not (0 <= a.lo < a.hi <= len(order)):
        raise SystemExit("REFUSED: positions %d..%d are not inside the manifest of %d clips" % (a.lo, a.hi, len(order)))
    clips = order[a.lo:a.hi]
    t0 = time.time()

    def load_fn(c):
        d = torch.load(os.path.join(a.cache, a.split, "clips", c + ".pt"), map_location="cpu")
        T = int(d["pool4x10"].shape[0])
        v, ac, yw = d["v"].float().numpy(), d["a"].float().numpy(), d["yaw_rate"].float().numpy()
        if len(v) != T:
            raise SystemExit("REFUSED: clip %s has %d tokens frames and %d kinematic values" % (c, T, len(v)))
        del d
        return v, ac, yw

    L = build_layout(clips, load_fn)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **L)
    meta = {"split": a.split, "lo": a.lo, "hi": a.hi, "n_manifest": len(man), "n_clips": len(clips), "n_frames": int(len(L["t"])), "out_sha256": sha_file(a.out),
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1)}
    with open(a.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
