"""R1 TRAIN sample (DESIGN.md sec. 4): choose the train clips, build a symlink subset cache + the remap the eval loader
reads. CPU only, no GPU, no model. Prints / writes sha12 only (never a clip id).

Rule (fixed before any R1 number, seed 0):
  * universe: the refcv7 train cache's clips (its own `_v2manifest.pt`) that HAVE a clip-clock sidecar row (so the
    label clock is measured, never the fallback) and a v8 label record;
  * per clip, every window's GT class by the route package's SPEC sec. 3 rule (r1_lib.gt_class) on the manifest poses
    with the V3Dataset window contract (W = 8);
  * TURN clips = clips with >= 1 GT-turn window; N_TURN = n_clips // 2 drawn from them, the rest from the other clips
    (rng = numpy default_rng(0), both pools sorted by sha12 first). Enriched on purpose: turns are ~10 % of windows.
The per-window choice inside each clip happens in r1_capture.py (it needs the dataset's own index).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r1_lib as L1  # noqa: E402

W = 8


def stable_sid(clip_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(str(clip_id).encode("utf-8"), digest_size=8).digest(), "big") >> 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", default="/home/nvidia/data/refcv6-b1-416x1024-train")
    ap.add_argument("--labels", default="/home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz")
    ap.add_argument("--sidecar", default="/home/nvidia/data/refcv6_clip_clock_sidecar.jsonl")
    ap.add_argument("--speed-sidecar", default="/home/nvidia/data/refcv6_speed_max_v8_train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clips", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    man = torch.load(os.path.join(a.train_cache, "_v2manifest.pt"), map_location="cpu", weights_only=False)
    files, cids, poses = list(man["files"]), [str(c) for c in man["clip_id"]], man["poses"]
    side = set()
    with open(a.sidecar, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                side.add(int(json.loads(ln)["sid"]))
    lab = set()
    with gzip.open(a.labels, "rt", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                lab.add(L1.sha12(json.loads(ln)["clip_id"]))
    rows = []
    n_noclock = n_nolab = 0
    for f, c, p in zip(files, cids, poses):
        s = L1.sha12(c)
        if stable_sid(c) not in side:
            n_noclock += 1
            continue
        if s not in lab:
            n_nolab += 1
            continue
        ts, gt, gv = L1.episode_window_gt(p.numpy(), W)
        cls, _ = L1.gt_class(gt, gv)
        n_turn = int(((cls == "turnL") | (cls == "turnR")).sum())
        rows.append((s, f, n_turn, len(ts)))
    rows.sort()
    turn = [r for r in rows if r[2] > 0]
    other = [r for r in rows if r[2] == 0]
    rng = np.random.default_rng(a.seed)
    n_t = min(a.n_clips // 2, len(turn))
    pick_t = [turn[i] for i in sorted(rng.choice(len(turn), n_t, replace=False))]
    pick_o = [other[i] for i in sorted(rng.choice(len(other), a.n_clips - n_t, replace=False))]
    chosen = sorted(pick_t + pick_o)
    d = Path(a.out) / "train_r1"
    d.mkdir(parents=True, exist_ok=True)
    for s, f, _n, _w in chosen:
        src = os.path.realpath(os.path.join(a.train_cache, f))
        dst = d / f
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    rec = {"name": "train_r1", "rule": __doc__.split("Rule")[1].strip(), "seed": a.seed,
           "n_universe": len(rows), "n_excluded_no_clock_row": n_noclock, "n_excluded_no_label": n_nolab,
           "n_turn_clips_universe": len(turn), "n_other_clips_universe": len(other),
           "n_chosen": len(chosen), "n_chosen_turn_clips": n_t,
           "gt_turn_windows_in_chosen": int(sum(r[2] for r in chosen)),
           "windows_in_chosen": int(sum(r[3] for r in chosen)),
           "sha12": [r[0] for r in chosen], "train_cache": a.train_cache}
    (d / "SUBSET_RECORD.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    remap = {"--eval-cache": str(d), "--eval-labels": a.labels, "--speed-max-sidecar-v6-eval": a.speed_sidecar}
    (Path(a.out) / "remap_train_r1.json").write_text(json.dumps(remap), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("sha12", "rule")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
