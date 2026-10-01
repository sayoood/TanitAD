#!/usr/bin/env python3
"""PROOF that the arms' micro-batch split (64 x accum 4) accumulates the SAME gradients as 32 x accum 8 -- the
coordinator's condition (a), 2026-09-27, before any M6 arm ran.

ONE real 256-sample optimiser batch (the first 256 samples of seed 0's SceneEpochSampler stream) from REAL cache frames
(the first 256-frame shard of cache_train_ep015, bf16 bits widened to fp32) with the REAL on-policy sets (ckpt_step <=
4933), through the REAL trainer `refe/proxy_train.py` (its `--dump-grads-after-accum` test hook saves the accumulated,
unclipped gradients of every trainable tensor after the first optimiser step's micro-batches), on CPU in fp32:
  CLEAN     32 x 8 vs 64 x 4 -> max per-tensor relative difference at fp32 rounding (bar 1e-5)
  MUTATION  the scorer loss normalised by the MICRO-batch's own covered count (a mean of ratios) instead of the
            trainer's GLOBAL constant cov_norm x micro-batch -> the two splits must DISAGREE (goes RED)
Also reported: the covered samples per micro-batch under each split (the mutation only bites when they differ).

    python refe/check_microbatch_invariance.py  -> ZZMICROBATCH_INVARIANT / ZZMICROBATCH_NOT_INVARIANT
    writes raw/2026-09-27-m6-proxy/microbatch_invariance.json
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
SRC = Path("D:/Projects/TanitAD/data/refe_proxy/cache_train_ep015")
SETS = Path("D:/Projects/TanitAD/data/refe_proxy/sets")
SNAP = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"
OUT = PKG / "raw" / "2026-09-27-m6-proxy" / "microbatch_invariance.json"
BAR = 1e-5


def child(argv, mutate: bool) -> int:
    sys.path.insert(0, str(HERE))
    import proxy_train as PT
    import torch.nn.functional as F
    if mutate:
        def score_loss_per_micro(sx, cg, batch, cov_norm, score_w, sco_params):
            if sx is None:
                return sum(p_.sum() for p_ in sco_params) * 0.0
            per = F.binary_cross_entropy_with_logits(sx.float(), cg, reduction="none").mean(-1)
            return per.sum() / max(float(per.numel()), 1.0) * score_w       # per-MICRO-batch covered count
        PT.score_loss = score_loss_per_micro
    sys.argv = ["proxy_train.py", *argv]
    return PT.main()


def main() -> int:
    import torch
    tmp = Path(tempfile.mkdtemp(prefix="mbinv_"))
    sub = tmp / "cache"
    sub.mkdir()
    frames = [json.loads(l) for l in open(SRC / "frames.jsonl", encoding="utf-8")]
    f0 = [f for f in frames if f["shard"] == 0]
    keys = {tuple(f["key"]) for f in f0}
    for suf in ("scene", "visual"):
        shutil.copy2(SRC / f"shard_0000.{suf}.bin", sub / f"shard_0000.{suf}.bin")
    with open(sub / "frames.jsonl", "w", encoding="utf-8", newline="\n") as fo:
        for f in f0:
            fo.write(json.dumps(f) + "\n")
    n_rows = 0
    with open(sub / "rows.jsonl", "w", encoding="utf-8", newline="\n") as fo:
        for l in open(SRC / "rows.jsonl", encoding="utf-8"):
            if tuple(json.loads(l)["key"]) in keys:
                fo.write(l if l.endswith("\n") else l + "\n")
                n_rows += 1
    json.dump({"version": 1, "split": "train", "dtype": "bf16", "frames": len(f0),
               "note": "SUB-CACHE for the micro-batch invariance proof: shard 0 of cache_train_ep015, verbatim"},
              open(sub / "manifest.json", "w"))
    runs = {}
    base = ["--cache", str(sub), "--sets", str(SETS), "--snapshot", SNAP, "--sets-work", str(tmp / "work"),
            "--device", "cpu", "--amp", "none", "--zero-lr-steps", "0", "--steps", "1", "--yaw-loss", "plain",
            "--seed", "0"]
    for name, b, acc, mut in (("clean_32x8", 32, 8, False), ("clean_64x4", 64, 4, False),
                              ("mut_32x8", 32, 8, True), ("mut_64x4", 64, 4, True)):
        dump = tmp / f"{name}.pt"
        argv = base + ["--out", str(tmp / name), "--batch", str(b), "--accum", str(acc),
                       "--dump-grads-after-accum", str(dump)]
        p = subprocess.run([sys.executable, __file__, "--child", "1" if mut else "0", "--", *argv],
                           capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="4"))
        if p.returncode != 0 or not dump.exists():
            print(p.stdout[-2000:], p.stderr[-2000:])
            print(f"ZZMICROBATCH_FAIL {name} did not dump")
            return 1
        runs[name] = torch.load(dump, weights_only=False)
        print(f"  {name}: covered per micro-batch {runs[name]['covered_per_micro_batch']}  cov_norm "
              f"{runs[name]['cov_norm']:.4f}  losses {runs[name]['losses']}", flush=True)

    def compare(a, b):
        ga, gb = runs[a]["grads"], runs[b]["grads"]
        assert set(ga) == set(gb) and not runs[a]["no_grad"] and not runs[b]["no_grad"]
        per = {}
        num = den = 0.0
        for k in ga:
            d = (ga[k] - gb[k]).abs().max().item()
            s = max(ga[k].abs().max().item(), 1e-30)
            per[k] = d / s
            num += float(((ga[k] - gb[k]) ** 2).sum())
            den += float((ga[k] ** 2).sum())
        worst = max(per, key=per.get)
        sco = {k: v for k, v in per.items() if k.startswith(("score_q_mlp.", "score_dec.", "score_head."))}
        return {"tensors": len(per), "max_rel_per_tensor": per[worst], "worst_tensor": worst,
                "max_rel_scorer_tensors": max(sco.values()) if sco else None,
                "global_rel_l2": (num / max(den, 1e-30)) ** 0.5}
    clean = compare("clean_32x8", "clean_64x4")
    mut = compare("mut_32x8", "mut_64x4")
    mut_vs_clean = compare("clean_32x8", "mut_32x8")
    ok_clean = clean["max_rel_per_tensor"] <= BAR
    red_mut = mut["max_rel_scorer_tensors"] is not None and mut["max_rel_scorer_tensors"] > 1e-3
    cov = {n: runs[n]["covered_per_micro_batch"] for n in runs}
    res = {"what": "accumulated unclipped gradients of every trainable tensor, one real 256-sample optimiser batch, "
                   "CPU fp32, refe/proxy_train.py via --dump-grads-after-accum",
           "bar_max_rel_per_tensor": BAR, "clean_32x8_vs_64x4": clean, "clean_invariant": ok_clean,
           "mutation_per_microbatch_norm_32x8_vs_64x4": mut, "mutation_goes_red": red_mut,
           "mutation_vs_clean_at_32x8": mut_vs_clean, "covered_per_micro_batch": cov,
           "covered_total": {n: sum(v) for n, v in cov.items()},
           "cov_norm_global_constant": runs["clean_32x8"]["cov_norm"], "sub_cache_frames": len(f0),
           "sub_cache_rows": n_rows,
           "tested_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
                             for f in ("proxy_train.py", "check_microbatch_invariance.py")}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: v for k, v in res.items() if k != "tested_sha256"}, indent=1))
    good = ok_clean and red_mut
    print("ZZMICROBATCH_INVARIANT" if good else "ZZMICROBATCH_NOT_INVARIANT")
    return 0 if good else 1


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        sys.exit(child(sys.argv[4:], sys.argv[2] == "1"))
    sys.exit(main())
