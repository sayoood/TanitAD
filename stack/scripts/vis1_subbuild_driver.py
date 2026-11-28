#!/usr/bin/env python3
"""vis1_subbuild_driver.py -- run ``precompute_vis1_sidecar.py --mode full`` over a SELECTED SUBSET of clips.

WHY (refcv8 +203 train clips, 2026-10-12): the VIS-1 sidecar was built once over the 4,369 train + 139 eval clips
(2,585 s pool wall on 12 workers, 859,497 frames). Adding 203 train clips needs rows for the 58 that are in the
agent joins and zero-frame entries for the 145 that are not; re-z-buffering the other 4,508 clips would cost
~8 CPU-hours to reproduce bytes we already hold. This driver lets the UNMODIFIED builder run over {the 203 ADD
clips} + {a control sample of OLD clips} so that

    * the 203 per-clip entries come out of the builder's own code path (``run_shard`` / ``camera_for`` /
      the trainer's own ``V3Dataset._agent_item``), and
    * the control sample is compared EXACTLY against the old sidecar (the builder + calib + extrinsics + joins used
      here are therefore proven to be the ones that built the old file).

The builder file is NOT modified. The only thing replaced is its ``load_split_clips`` (module-global lookup in
``main``): the replacement reads the cache's ``_v2manifest.pt`` DIRECTLY with ``torch.load`` -- it never calls
``load_or_build_manifest``, whose rebuild path would rewrite a pre-existing manifest on disk -- and keeps only the
clips whose sha12 is listed in ``--select-file`` (``{"train": [sha12, ...], "eval": [sha12, ...]}``). It REFUSES if a
listed sha12 is not in the manifest (a silent narrowing of the selection is the failure this guards).

Usage: identical to the builder plus ``--select-file``; every other argument is forwarded untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import precompute_vis1_sidecar as B  # noqa: E402  (the unmodified builder, same directory)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--select-file", required=True)
    ours, rest = ap.parse_known_args(argv)
    sel = json.load(open(ours.select_file, encoding="utf-8"))
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--train-cache")
    probe.add_argument("--eval-cache")
    pa, _ = probe.parse_known_args(rest)
    which = {str(pa.train_cache): "train", str(pa.eval_cache): "eval"}
    if len(which) != 2:
        raise SystemExit("[vis1-driver] --train-cache and --eval-cache must differ")

    def load_split_clips(cache: str):
        import torch
        split = which[str(cache)]
        mp = Path(cache) / "_v2manifest.pt"
        man = torch.load(mp, map_location="cpu", weights_only=False)
        cids = [str(c) for c in (man.get("clip_id") or [])]
        uids = [int(u) for u in (man.get("episode_uid") or [])]
        ns_all = [int(x) for x in (man.get("n_stack") or [])]
        if len(cids) != len(uids) or not cids:
            raise SystemExit(f"[vis1-driver] {mp}: {len(cids)} clip ids / {len(uids)} uids")
        want = set(sel[split])
        keep = [i for i, c in enumerate(cids) if B.sha12(c) in want]
        got = {B.sha12(cids[i]) for i in keep}
        if got != want:
            raise SystemExit(f"[vis1-driver] {split}: {len(want - got)} selected sha12 not in {mp} "
                             f"(e.g. {sorted(want - got)[:4]})")
        B.log(f"driver: {split} {len(keep)}/{len(cids)} clips selected from {mp}")
        return ([cids[i] for i in keep], [uids[i] for i in keep],
                sorted({ns_all[i] for i in keep}))

    B.load_split_clips = load_split_clips
    return B.main(rest)


if __name__ == "__main__":
    sys.exit(main())
