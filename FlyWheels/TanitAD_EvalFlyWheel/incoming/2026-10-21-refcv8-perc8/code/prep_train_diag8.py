"""TRAIN-DIAG subset for refcv8 (adapted from 2026-10-04-refcv7-map-box-diagnostics/code/prep_train_subset.py, which needs a
calibration json this run has no use for): the refcv8 TRAIN cache's clips sorted by sha12 = sha256(clip_id)[:12], the first N (139)
whose 10 cm GT file <sha12>.sam3mapgt.npz exists; a symlink subset dir + SUBSET_RECORD.json.  CPU only.  Prints sha12 only."""
import argparse, hashlib, json, os, sys


def sha12(c):
    return hashlib.sha256(str(c).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", default="/home/nvidia/data/refcv8-b1-416x1024-train-4572")
    ap.add_argument("--gt-root", default="/home/nvidia/data/sam3_gt_v3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=139)
    a = ap.parse_args()
    import torch
    man = torch.load(os.path.join(a.train_cache, "_v2manifest.pt"), map_location="cpu", weights_only=False)
    files, cids = list(man["files"]), [str(c) for c in man["clip_id"]]
    assert len(files) == len(cids)
    by_sha = {}
    for f, c in zip(files, cids):
        s = sha12(c)
        if s in by_sha:
            raise SystemExit(f"sha12 collision {s}")
        by_sha[s] = f
    order = sorted(by_sha)
    gt = set(fn.split(".")[0] for fn in os.listdir(a.gt_root) if fn.endswith(".sam3mapgt.npz"))
    diag = [s for s in order if s in gt][: a.n]
    n_skipped = order.index(diag[-1]) + 1 - len(diag)
    d = os.path.join(a.out, "train_diag139")
    os.makedirs(d, exist_ok=True)
    for s in diag:
        src = os.path.realpath(os.path.join(a.train_cache, by_sha[s]))
        dst = os.path.join(d, by_sha[s])
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    rec = {"name": "train_diag139", "rule": f"refcv8 train cache clips sorted by sha12, first {a.n} with a GT file "
                                            f"({n_skipped} skipped for no GT before the cut)",
           "n": len(diag), "sha12": diag, "n_with_gt": sum(1 for s in diag if s in gt),
           "train_cache": a.train_cache, "gt_root": a.gt_root, "n_train_clips": len(order)}
    json.dump(rec, open(os.path.join(d, "SUBSET_RECORD.json"), "w"), indent=1)
    print(f"[prep] train_diag139: {len(diag)} clips of {len(order)}, {rec['n_with_gt']} with GT, first sha12 {diag[0]}")


if __name__ == "__main__":
    sys.exit(main())
