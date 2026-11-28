"""Builds SYNTHETIC state files in the state-file layout for the smoke of s1_read.py (first N clips of the probe set), inside spatial/s1/smoke_states/.
  python make_smoke_states.py noise --clips 300 --dim 128   -> noise_D128.npz (+ .json record): Gaussian noise [N, D]; needs only INDEX.json
  python make_smoke_states.py mean64 --clips 300            -> mean64.npz (+ .json): the 64-number mean over the 40 projected cells (control t2's state); READS THE SHARDS
Records are written like extract_states.py writes them (ckpt_sha256 is a dummy value of 64 'a'; wm_state mean/grid as declared).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
OUT = f"{HERE}/smoke_states"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("noise", "mean64"))
    ap.add_argument("--clips", type=int, default=300)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--mode", default="mean", choices=("mean", "grid"))
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    import s1_read as R
    I = R.E.load_index()
    clips = list(I["clips"])[:a.clips]
    frames = np.asarray(I["frames"], np.int64)[:a.clips]
    ci = np.repeat(np.arange(len(clips), dtype=np.int32), frames)
    t = np.concatenate([np.arange(n, dtype=np.int16) for n in frames])
    if a.kind == "noise":
        Z = np.random.default_rng(0).standard_normal((len(ci), a.dim)).astype(np.float16)
        name = f"noise_D{a.dim}"
    else:
        D = R.E.load_train(a.clips)
        Z = R.S0.mean_token(D["X"]).astype(np.float16)
        assert np.array_equal(D["epi"], ci) and np.array_equal(D["bg"], t.astype(np.int64))
        del D
        name = "mean64"
    p = f"{OUT}/{name}.npz"
    np.savez(p, Z=Z, clip_idx=ci, t=t, clips=np.array(clips))
    rec = {"ckpt": "synthetic", "ckpt_sha256": "a" * 64, "step": None, "wm_state": a.mode if a.kind == "noise" else "mean", "wm": True, "D": int(Z.shape[1]), "n_frames": int(len(Z)),
           "n_clips": len(clips), "split": "train", "lo": 600, "hi": 600 + len(clips), "n_manifest": None, "out_sha256": sha(p)}
    with open(p + ".json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(p, Z.shape, os.path.getsize(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
