"""WP-D P-BOX prep (CPU, no GPU): the registered TRAIN clip set as a symlink subset cache + its loader remap.

PREREG_WPD_PROBES.md (registered sha256 c054190b…): TRAIN = "600 train clips drawn with numpy.default_rng(0) from the
4,369, 5 windows each at evenly spaced t". Rule as implemented (fixed here, before any capture): the train cache's
clips sorted by sha12 = sha256(clip_id)[:12]; `default_rng(0).choice(4369, 600, replace=False)` over that order; the
chosen clips' REAL .v2ep.pt files are symlinked (os.path.realpath) into <out>/train_pbox600 -- the trained bytes, the
`prep_train_subset.py` pattern of the diagnostics package. Writes SUBSET_RECORD.json (sha12 list, rule) and
remap_train_pbox600.json for REFCV6_REMAP_OVERRIDES. ⛔ Prints sha12 only.
"""
import argparse
import hashlib
import json
import os
import sys


def sha12(c):
    return hashlib.sha256(str(c).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", default="/home/nvidia/data/refcv6-b1-416x1024-train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import numpy as np
    import torch
    man = torch.load(os.path.join(a.train_cache, "_v2manifest.pt"), map_location="cpu", weights_only=False)
    files, cids = list(man["files"]), [str(c) for c in man["clip_id"]]
    by_sha = {}
    for f, c in zip(files, cids):
        s = sha12(c)
        if s in by_sha:
            raise SystemExit(f"sha12 collision {s}")
        by_sha[s] = f
    order = sorted(by_sha)
    if len(order) != 4369:
        raise SystemExit(f"expected 4,369 train clips, the manifest has {len(order)}")
    pick = sorted(int(i) for i in np.random.default_rng(a.seed).choice(len(order), a.n, replace=False))
    shas = [order[i] for i in pick]
    d = os.path.join(a.out, "train_pbox600")
    os.makedirs(d, exist_ok=True)
    for s in shas:
        src = os.path.realpath(os.path.join(a.train_cache, by_sha[s]))
        dst = os.path.join(d, by_sha[s])
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    rec = {"name": "train_pbox600", "rule": f"train cache clips sorted by sha12; numpy default_rng({a.seed})"
           f".choice(4369, {a.n}, replace=False); symlinks to the real files", "n": len(shas), "sha12": shas,
           "sha12_list_sha256": hashlib.sha256(json.dumps(shas).encode()).hexdigest(), "train_cache": a.train_cache}
    with open(os.path.join(d, "SUBSET_RECORD.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    remap = {"--eval-cache": d,
             "--eval-labels": "/home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz",
             "--speed-max-sidecar-v6-eval": "/home/nvidia/data/refcv6_speed_max_v8_train.jsonl"}
    with open(os.path.join(a.out, "remap_train_pbox600.json"), "w") as fh:
        json.dump(remap, fh)
    print(f"[pbox-prep] {len(shas)} clips, sha12-list sha256 {rec['sha12_list_sha256'][:12]}, first {shas[0]}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
