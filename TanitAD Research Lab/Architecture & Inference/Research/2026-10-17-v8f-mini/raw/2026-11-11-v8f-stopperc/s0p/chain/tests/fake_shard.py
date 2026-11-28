"""fake_shard.py -- what the fake box 2 writes for one shard (same file names, dtypes, shapes, meta and manifest as s0p_extract_train.py --shard i/K; tiny N).  Test only."""
import argparse
import hashlib
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"), ap.add_argument("--i", type=int), ap.add_argument("--k", type=int), ap.add_argument("--n-total", type=int), ap.add_argument("--n-pos", type=int), ap.add_argument("--sample-sha")
    a = ap.parse_args()
    lo, hi = (a.i * a.n_total) // a.k, ((a.i + 1) * a.n_total) // a.k
    n = hi - lo
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(100 + a.i)
    y = (np.arange(lo, hi) < a.n_pos).astype(np.uint8)
    np.save(os.path.join(a.out, "E.npy"), rng.normal(0, 1, (n, 18)).astype(np.float32))
    np.save(os.path.join(a.out, "v40.npy"), rng.integers(0, 255, (n, 40, 1024), dtype=np.uint8))
    np.save(os.path.join(a.out, "v160.npy"), rng.integers(0, 255, (n, 160, 1024), dtype=np.uint8))
    np.save(os.path.join(a.out, "mean.npy"), rng.normal(0, 1, (n, 1024)).astype(np.float16))
    np.save(os.path.join(a.out, "y.npy"), y)
    np.save(os.path.join(a.out, "clip.npy"), np.asarray([f"clip{j // 5:03d}" for j in range(lo, hi)]).astype("U40"))
    np.save(os.path.join(a.out, "ws.npy"), np.arange(lo, hi, dtype=np.int64))
    np.save(os.path.join(a.out, "eid.npy"), np.arange(lo, hi, dtype=np.int64) // 5)
    meta = {"sample_sha256": a.sample_sha, "n_sample": a.n_total, "n_shard": n, "shard": {"i": a.i, "k": a.k, "lo": lo, "hi": hi, "n_total": a.n_total},
            "c2_second_route_max_abs": {"p10f": 0.0, "nav": 0.0, "limit": 0.0, "limit_ok": 0.0}, "causal_limit_ok_false_fraction": 0.0}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), sort_keys=True)
    json.dump({"clip": [], "ws": [], "y": y.tolist(), "sha256": a.sample_sha}, open(os.path.join(a.out, "sample.json"), "w", encoding="utf-8"))
    names = ["E.npy", "v40.npy", "v160.npy", "mean.npy", "y.npy", "clip.npy", "ws.npy", "eid.npy", "meta.json", "sample.json"]
    with open(os.path.join(a.out, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
        for nm in names:
            f.write(f"{hashlib.sha256(open(os.path.join(a.out, nm), 'rb').read()).hexdigest()}  {nm}\n")
    print("fake shard", a.i, n)


if __name__ == "__main__":
    main()
