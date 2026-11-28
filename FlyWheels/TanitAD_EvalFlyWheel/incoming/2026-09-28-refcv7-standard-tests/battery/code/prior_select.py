"""SPEC §3.4: the positional-prior FIT set = the 300 TRAIN clips with the smallest sha12 among Thor's
`sam3_gt_v3` files, excluding every eval clip. A fixed rule, applied before any fit or score.

    python prior_select.py --gt-list thor_gt_v3_list.txt --train-list thor_train_cache_list.txt \
        --eval-cache D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139 --n 300 --out prior_fit_set.json

Inputs are `ls` listings of Thor (read-only). The output carries sha12s only (never a raw clip id);
the raw train listing is deleted by the caller after this runs.
"""
import argparse
import hashlib
import json
import os


def sha12(c: str) -> str:
    return hashlib.sha256(str(c).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-list", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gt = {ln.strip().split(".")[0] for ln in open(a.gt_list, encoding="utf-8")
          if ln.strip().endswith(".sam3mapgt.npz")}
    train = {sha12(ln.strip()[:-len(".v2ep.pt")]) for ln in open(a.train_list, encoding="utf-8")
             if ln.strip().endswith(".v2ep.pt")}
    from tanitad.data.v2_dataset import load_or_build_manifest
    ev = {sha12(str(c)) for c in load_or_build_manifest(a.eval_cache, verbose=False)["clip_id"]}
    cand = sorted((gt & train) - ev)
    pick = cand[:a.n]
    rec = {"rule": "the N TRAIN clips with the smallest sha12 among Thor sam3_gt_v3 files, "
                   "excluding the 139 eval clips (SPEC.md §3.4)",
           "n_gt_files": len(gt), "n_train_cache_clips": len(train), "n_eval_clips": len(ev),
           "n_gt_and_train": len(gt & train), "n_eval_in_train": len(train & ev),
           "n_candidates": len(cand), "n_selected": len(pick), "sha12": pick,
           "sha12_list_sha256": hashlib.sha256("\n".join(pick).encode()).hexdigest()}
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "sha12"}))


if __name__ == "__main__":
    main()
