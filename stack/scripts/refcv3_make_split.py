#!/usr/bin/env python3
"""Split a built B1 epcache into the v7.2 TRAIN and EVAL corpora, by symlink.

⛔ WHY THIS EXISTS — A MEASURED LEAK (2026-09-02). The v7.2 release splits
4,572 train / 147 eval clips with ZERO intersection. The raw B1 corpus is
4,713 = 4,572 + **141 of those eval clips** (the other 6 are the val40 clips
the parity gate drops). So "train on all of B1, evaluate on the v7.2 eval
split" trains on 141 of the 147 eval clips, and every held-out number off them
would be optimistic and inadmissible — the leak would not announce itself,
because the eval would simply look good.

Symlinks, not copies: the built cache is ~160 GB and duplicating it to make a
split would be a second measured-disk decision for no benefit.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path


def clip_ids(path: Path) -> set[str]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        return {json.loads(l)["clip_id"] for l in fh if l.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="the built B1 epcache")
    ap.add_argument("--train-labels", required=True)
    ap.add_argument("--eval-labels", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-eval", required=True)
    a = ap.parse_args()

    cache = Path(a.cache)
    have = {p.name[:-len(".v2ep.pt")]: p for p in cache.glob("*.v2ep.pt")}
    tr, ev = clip_ids(Path(a.train_labels)), clip_ids(Path(a.eval_labels))
    if tr & ev:
        raise SystemExit(f"REFUSING: the label sets overlap on {len(tr & ev)} "
                         "clips — the split is not a split")

    made = {}
    for name, ids, out in (("train", tr, a.out_train), ("eval", ev, a.out_eval)):
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.v2ep.pt"):
            f.unlink()
        n = 0
        for cid in sorted(ids):
            src = have.get(cid)
            if src is None:
                continue                      # not built (e.g. parity-dropped)
            os.symlink(src, d / f"{cid}.v2ep.pt")
            n += 1
        geo = cache / "_geometry.json"
        if geo.exists() and not (d / "_geometry.json").exists():
            os.symlink(geo, d / "_geometry.json")
        made[name] = n
        print(f"[split] {name:5s}: {n} of {len(ids)} labelled clips present")

    # ⛔ VERIFY THE SPLIT IS A SPLIT — by CONTENT of the two directories, not by
    # trusting the label sets we just read.
    a_ids = {p.name for p in Path(a.out_train).glob("*.v2ep.pt")}
    b_ids = {p.name for p in Path(a.out_eval).glob("*.v2ep.pt")}
    inter = a_ids & b_ids
    if inter:
        raise SystemExit(f"REFUSING: {len(inter)} clips are in BOTH output "
                         f"dirs — that is not a held-out split")
    print(f"[split] disjoint verified: train {len(a_ids)} | eval {len(b_ids)} "
          f"| intersection 0")
    if made["eval"] == 0:
        raise SystemExit("REFUSING: the eval split is EMPTY — an eval that "
                         "runs on nothing reports a clean number for no work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
