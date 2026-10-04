"""Build the TRAIN-DIAG and TRAIN-CALIB256 symlink subset caches (SPEC.md sec. 1). CPU only, no GPU.

TRAIN-DIAG  = the train cache's clips sorted by sha12 = sha256(clip_id)[:12], the first N (139) whose 10 cm GT
              file <sha12>.sam3mapgt.npz exists in the GT root.
TRAIN-CAL64 = the 64 clips of the banked A10 calibration artifact (box_calib_train256.json).

Each subset dir holds symlinks to the SAME real .v2ep.pt files the train cache points at (os.path.realpath), so
the v2 manifest built over it reads exactly the trained bytes. ⛔ Prints sha12 only, never a clip id.
Writes <out>/<name>/SUBSET_RECORD.json (sha12 list, selection rule, counts).
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
    ap.add_argument("--gt-root", default="/home/nvidia/data/sam3_gt_v3")
    ap.add_argument("--calib-json", required=True)
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
    cal = json.load(open(a.calib_json))
    cal_sha = sorted({str(w[0]) for w in cal["windows"]})
    missing_cal = [s for s in cal_sha if s not in by_sha]
    if missing_cal:
        raise SystemExit(f"{len(missing_cal)} calibration clips are not in the train cache")
    for name, shas, rule in (
            ("train_diag139", diag, f"train cache clips sorted by sha12, first {a.n} with a GT file "
                                    f"({n_skipped} skipped for no GT before the cut)"),
            ("train_cal64", cal_sha, "the 64 clips of box_calib_train256.json")):
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        for s in shas:
            src = os.path.realpath(os.path.join(a.train_cache, by_sha[s]))
            dst = os.path.join(d, by_sha[s])
            if not os.path.lexists(dst):
                os.symlink(src, dst)
        rec = {"name": name, "rule": rule, "n": len(shas), "sha12": shas,
               "n_with_gt": sum(1 for s in shas if s in gt),
               "train_cache": a.train_cache, "gt_root": a.gt_root}
        with open(os.path.join(d, "SUBSET_RECORD.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"[prep] {name}: {len(shas)} clips, {rec['n_with_gt']} with GT, first sha12 {shas[0]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
