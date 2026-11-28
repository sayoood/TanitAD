#!/usr/bin/env python3
"""Merge the export shards into feats.npz + p2_score_head.npz + feats_manifest.json (sha256 of every shard, of the merged file, of the inputs and of the head).  CPU only.

    python merge_feats.py [--shards D:/.../l2_offprop/shards] [--out-dir D:/.../l2_offprop]
Refuses (exit 2) if a shard is missing, non-contiguous, or was made from a different bundle/perts than the run dir's.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2lib  # noqa: E402

RUN = "D:/Projects/TanitAD-artifacts/l1_stepA_run"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="D:/Projects/TanitAD-artifacts/l2_offprop/shards")
    ap.add_argument("--out-dir", default="D:/Projects/TanitAD-artifacts/l2_offprop")
    ap.add_argument("--run-dir", default=RUN)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.shards, "shard_*.npz")))
    if not files:
        print("ZZMERGE_FAIL no shards")
        return 2
    sha_bundle, sha_perts = l2lib.sha256_file(f"{a.run_dir}/bundle/bundle.npz"), l2lib.sha256_file(f"{a.run_dir}/perts.npz")
    parts = {k: [] for k in ("infan_tok", "infan_logit", "pert_tok", "pert_logit")}
    nxt, shard_sha = 0, {}
    for f in files:
        z = np.load(f, allow_pickle=False)
        lo, hi = int(z["lo"]), int(z["hi"])
        if lo != nxt:
            print(f"ZZMERGE_FAIL non-contiguous: expected shard starting at {nxt}, got {lo} ({f})")
            return 2
        if str(z["bundle_sha256"]) != sha_bundle or str(z["perts_sha256"]) != sha_perts:
            print(f"ZZMERGE_FAIL shard {f} was made from a different bundle/perts")
            return 2
        for k in parts:
            parts[k].append(z[k])
        shard_sha[os.path.basename(f)] = l2lib.sha256_file(f)
        nxt = hi
    if nxt != a.n:
        print(f"ZZMERGE_FAIL shards cover [0,{nxt}) not [0,{a.n})")
        return 2
    m = {k: np.concatenate(v, 0) for k, v in parts.items()}
    os.makedirs(a.out_dir, exist_ok=True)
    fp = os.path.join(a.out_dir, "feats.npz")
    np.savez(fp, **m, bundle_sha256=np.array(sha_bundle), perts_sha256=np.array(sha_perts))
    import torch
    sd = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=False)["model"]
    W, b = sd["score_head.weight"].float().numpy().copy(), sd["score_head.bias"].float().numpy().copy()
    hp = os.path.join(a.out_dir, "p2_score_head.npz")
    np.savez(hp, weight=W, bias=b, ckpt=np.array(a.ckpt))
    t0 = time.time()
    man = {"schema": "l2-offprop-feats-v1", "n_tokens": a.n, "shapes": {k: list(v.shape) for k, v in m.items()},
           "what": "infan_tok/pert_tok = the INPUT of P2's score_head (Linear(256,6); EP = row 2) recorded by a forward pre-hook while L1 Step A's own make_score_fn/score_replaced drove the landed REFe.score_trajectories on CPU fp32",
           "feats_npz_sha256": l2lib.sha256_file(fp), "p2_score_head_sha256": l2lib.sha256_file(hp), "shard_sha256": shard_sha,
           "inputs": {"bundle.npz": sha_bundle, "perts.npz": sha_perts, "scores.npz": l2lib.sha256_file(f"{a.run_dir}/scores.npz"), "labels.npz": l2lib.sha256_file(f"{a.run_dir}/labels.npz"),
                      "ctx.npy": json.load(open(f"{a.run_dir}/bundle/ctx.json"))["ctx_sha256"], "ckpt_model_final.pt": l2lib.sha256_file(a.ckpt)},
           "stepA_py_sha256": l2lib.sha256_file(os.path.join(l2lib.L1_CODE, "stepA.py")),
           "export_code_sha256": {fn: l2lib.sha256_file(os.path.join(HERE, fn)) for fn in ("export_feats.py", "merge_feats.py")}, "hash_seconds": round(time.time() - t0, 1)}
    json.dump(man, open(os.path.join(a.out_dir, "feats_manifest.json"), "w", encoding="utf-8"), indent=1)
    print("ZZMERGE_OK", {k: v.shape for k, v in m.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
