"""fake_shard.py -- what the fake box 2 writes for one shard (or, with --sample-json, the dry run's full sample.json).  Same names, dtypes, shapes, meta and manifest as s0p_extract_train.py.  Test only."""
import argparse
import hashlib
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"), ap.add_argument("--i", type=int), ap.add_argument("--k", type=int), ap.add_argument("--n-total", type=int), ap.add_argument("--n-pos", type=int)
    ap.add_argument("--sample-sha"), ap.add_argument("--seed", type=int), ap.add_argument("--sample-json")
    a = ap.parse_args()
    y_all = np.zeros(a.n_total, np.uint8)
    y_all[:a.n_pos] = 1
    clip_all = [f"clip{j // 5:05d}" for j in range(a.n_total)]
    ws_all = list(range(a.n_total))
    if a.sample_json:
        os.makedirs(os.path.dirname(a.sample_json), exist_ok=True)
        json.dump({"clip": clip_all, "ws": ws_all, "y": y_all.tolist(), "sha256": a.sample_sha}, open(a.sample_json, "w", encoding="utf-8"))
        return
    lo, hi = (a.i * a.n_total) // a.k, ((a.i + 1) * a.n_total) // a.k
    n = hi - lo
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(100 + a.i + (a.seed % 7))
    np.save(os.path.join(a.out, "E.npy"), rng.normal(0, 1, (n, 18)).astype(np.float32))
    np.save(os.path.join(a.out, "v40.npy"), rng.integers(0, 255, (n, 40, 1024), dtype=np.uint8))
    np.save(os.path.join(a.out, "v160.npy"), rng.integers(0, 255, (n, 160, 1024), dtype=np.uint8))
    np.save(os.path.join(a.out, "mean.npy"), rng.normal(0, 1, (n, 1024)).astype(np.float16))
    np.save(os.path.join(a.out, "y.npy"), y_all[lo:hi])
    np.save(os.path.join(a.out, "clip.npy"), np.asarray(clip_all[lo:hi]).astype("U40"))
    np.save(os.path.join(a.out, "ws.npy"), np.asarray(ws_all[lo:hi], np.int64))
    np.save(os.path.join(a.out, "eid.npy"), np.arange(lo, hi, dtype=np.int64) // 5)
    meta = {"sample_sha256": a.sample_sha, "n_sample": a.n_total, "n_pos_sample": a.n_pos, "n_neg_sample": a.n_total - a.n_pos, "n_shard": n, "seed": a.seed, "pool_equals_banked_prevalence": True,
            "disjoint": {"overlap": 0}, "shard": {"i": a.i, "k": a.k, "lo": lo, "hi": hi, "n_total": a.n_total}, "elapsed_s": 100.0 + a.i,
            "c2_second_route_max_abs": {"p10f": 0.0, "nav": 0.0, "limit": 0.0, "limit_ok": 0.0}, "causal_limit_ok_false_fraction": 0.0}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1, sort_keys=True)
    json.dump({"clip": clip_all[lo:hi], "ws": ws_all[lo:hi], "y": y_all[lo:hi].tolist(), "sha256": a.sample_sha}, open(os.path.join(a.out, "sample.json"), "w", encoding="utf-8"))
    names = ["E.npy", "v40.npy", "v160.npy", "mean.npy", "y.npy", "clip.npy", "ws.npy", "eid.npy", "meta.json", "sample.json"]
    with open(os.path.join(a.out, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
        for nm in names:
            f.write(f"{hashlib.sha256(open(os.path.join(a.out, nm), 'rb').read()).hexdigest()}  {nm}\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
