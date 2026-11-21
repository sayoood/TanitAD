"""extract_probe_set.py -- streaming export of a compact probe set from the pod token cache (note PREREG_V8F_PROOF_ADD12_N7).

  python extract_probe_set.py --cache /workspace/v8proof/data/cache_pod --split train --lo 600 --hi 1800 --out-dir /workspace/v8proof/probe/train_600_1800

CPU only, one thread, ONE clip in memory at a time, no GPU. Clips = positions lo .. hi-1 of the ascending sha12 order of the split manifest (the trainer's order).
Per clip: pool4x10 [T, 40, 1024] fp8 -> float32 -> one fixed Gaussian projection to 64 numbers per cell (seed 0, scale 1/sqrt(64)) -> float16 [T, 40, 64]; v, a, yaw_rate [T] float32.
Writes shards of 100 clips: shard_XXXX.npz with X (float16 [N, 40, 64]), kin (float32 [N, 3] = v, a, yaw_rate), clip_idx (int32 [N]), t (int16 [N]) and clips (the shard's clip ids);
plus proj.npy (the matrix), INDEX.json (clip ids in order, frames per clip, shard of each clip, sha256 of every shard) and DONE.txt. Resumable per shard.
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", required=True, choices=("train", "eval"))
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", type=int, default=100)
    ap.add_argument("--dim", type=int, default=64)
    a = ap.parse_args(argv)
    from tanitad.data import v8f_mini_data as D
    man = D.read_manifest(a.cache, a.split)
    order = sorted([c for c, _ in man], key=D.sha12)
    clips = order[a.lo:a.hi]
    os.makedirs(a.out_dir, exist_ok=True)
    P = (np.random.default_rng(0).standard_normal((1024, a.dim)) / np.sqrt(a.dim)).astype(np.float32)
    np.save(os.path.join(a.out_dir, "proj.npy"), P)
    Pt = torch.from_numpy(P)
    index = {"split": a.split, "lo": a.lo, "hi": a.hi, "n_manifest": len(man), "clips": clips, "frames": [], "shard_of": [], "shards": {}, "dim": a.dim,
             "proj_sha256": hashlib.sha256(P.tobytes()).hexdigest()}
    t0 = time.time()
    for s0 in range(0, len(clips), a.shard):
        name = f"shard_{s0 // a.shard:04d}.npz"
        path = os.path.join(a.out_dir, name)
        part = clips[s0:s0 + a.shard]
        if os.path.exists(path) and os.path.exists(path + ".frames.json"):
            with open(path + ".frames.json", encoding="utf-8") as fh:
                fr = json.load(fh)
        else:
            Xs, Ks, Ci, Ts, fr = [], [], [], [], []
            for j, c in enumerate(part):
                d = torch.load(os.path.join(a.cache, a.split, "clips", c + ".pt"), map_location="cpu")
                x = d["pool4x10"].float()                                   # [T, 40, 1024]
                T = x.shape[0]
                Xs.append((x @ Pt).to(torch.float16).numpy())
                Ks.append(torch.stack([d["v"].float(), d["a"].float(), d["yaw_rate"].float()], 1).numpy().astype(np.float32))
                Ci.append(np.full(T, s0 + j, np.int32))
                Ts.append(np.arange(T, dtype=np.int16))
                fr.append(int(T))
                del d, x
            np.savez(path, X=np.concatenate(Xs), kin=np.concatenate(Ks), clip_idx=np.concatenate(Ci), t=np.concatenate(Ts), clips=np.array(part))
            with open(path + ".frames.json", "w", encoding="utf-8") as fh:
                json.dump(fr, fh)
            del Xs, Ks, Ci, Ts
        index["frames"] += fr
        index["shard_of"] += [name] * len(part)
        index["shards"][name] = sha_file(path)
        print(f"[probe {time.strftime('%H:%M:%S', time.gmtime())}] {name} clips {s0 + len(part)}/{len(clips)} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(a.out_dir, "INDEX.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    with open(os.path.join(a.out_dir, "DONE.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"done {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} clips {len(clips)} frames {sum(index['frames'])}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
