"""kin_ref.py -- control e4 of note N7 (S0E_DEFS E15): 30 random (clip, frame) rows read straight from the pod token cache, independent of extract_probe_set.py.

  python kin_ref.py --cache /workspace/v8proof/data/cache_pod --index /workspace/v8proof/probe/train_600_1800/INDEX.json --out /workspace/v8proof/probe/train_600_1800/KIN_REF.npz

Rows: numpy default_rng(4) draws 30 clip positions of the INDEX clip list (with replacement) and one frame each (uniform over the clip's frame count in INDEX).
Writes clip (str [30]), t (int32 [30]), kin (float32 [30, 3] = v, a, yaw_rate as the cache holds them). CPU only, one clip in memory at a time.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args(argv)
    with open(a.index, encoding="utf-8") as fh:
        ix = json.load(fh)
    rng = np.random.default_rng(4)
    pos = rng.integers(0, len(ix["clips"]), size=a.n)
    clips, ts, kin = [], [], []
    for p in pos:
        c = ix["clips"][int(p)]
        t = int(rng.integers(0, int(ix["frames"][int(p)])))
        d = torch.load(os.path.join(a.cache, ix["split"], "clips", c + ".pt"), map_location="cpu")
        kin.append([float(d["v"][t]), float(d["a"][t]), float(d["yaw_rate"][t])])
        clips.append(c)
        ts.append(t)
        del d
    np.savez(a.out, clip=np.array(clips), t=np.array(ts, np.int32), kin=np.array(kin, np.float32))
    print("rows", len(clips), "first", clips[0][:8], ts[0], kin[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
