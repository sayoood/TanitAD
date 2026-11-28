#!/usr/bin/env python3
"""L2-OFFPROP step 1: export the EP head's INPUT TOKEN for every (token, anchor, perturbation) of L1 Step A and for every in-fan proposal.

"The EP head's input token" = the input of `score_head` (nn.Linear(256, 6); the EP head is row 2) when P2's scorer is driven by the LANDED
`REFe.score_trajectories`, exactly as L1 Step A's step 6 drove it (`real_scorer.make_score_fn` -> `stepA.run_scores` / `score_replaced`).
Nothing is re-implemented: we wrap `real_scorer.load_scorer` so a forward PRE-HOOK on `ns.score_head` records what the head consumed, and call
Step A's own `make_score_fn` / `score_replaced`.  The head's output on the recorded token is therefore the banked logit by construction; C-FEAT
(`gate_cfeat.py`) checks that against Step A's BANKED `scores.npz`.

Shards (resumable, atomic): shard_{start:04d}.npz holds tokens [start, start+SHARD) with
    infan_tok   [n,64,256] f32   the token of each of the 64 proposals of the UNPERTURBED fan
    infan_logit [n,64,6]   f32   the 6 logits the head produced from it
    pert_tok    [n,Q,256]  f32   the token at the ANCHOR SLOT of each perturbed COMPLETE 64-set (Q = 96 = 16 anchors x 6 perturbations)
    pert_logit  [n,Q,6]    f32
Merge with `merge_feats.py`.  CPU only (CUDA_VISIBLE_DEVICES=-1 is forced); the dev-box GPU lock is never touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

L1_CODE = os.environ.get("L1_CODE", "C:/Users/Admin/par18/l1run/code")
sys.path.insert(0, L1_CODE)
SHARD = 10


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def make_recording_score_fn(ckpt, ctx, tokens, refe_dir):
    """Step A's own `make_score_fn`, with the head-input recorder attached.  Returns (score_fn, store) where store['tok'] is the last head input [1,K,256]."""
    import real_scorer
    store = {"tok": None, "ns": None}
    orig = real_scorer.load_scorer

    def load_and_hook(*a, **k):
        ns, info = orig(*a, **k)
        ns.score_head.register_forward_pre_hook(lambda m, inp: store.__setitem__("tok", inp[0].detach().float().cpu().numpy().copy()))
        store["ns"] = ns
        return ns, info

    real_scorer.load_scorer = load_and_hook
    try:
        fn = real_scorer.make_score_fn(ckpt, ctx, tokens, device="cpu", refe_dir=refe_dir, autocast="none")
    finally:
        real_scorer.load_scorer = orig
    return fn, store


def export_range(a, lo, hi, out_path):
    import stepA
    b = stepA.load_npz(a.bundle)
    pe = stepA.load_npz(a.perts)
    fn, store = make_recording_score_fn(a.ckpt, a.ctx, np.asarray(b["token"]), a.refe_dir)
    n_tok = hi - lo
    Q = pe["anchor"].shape[1]
    infan_tok = np.zeros((n_tok, 64, 256), np.float32)
    infan_logit = np.zeros((n_tok, 64, 6), np.float32)
    pert_tok = np.zeros((n_tok, Q, 256), np.float32)
    pert_logit = np.zeros((n_tok, Q, 6), np.float32)
    for i, n in enumerate(range(lo, hi)):
        fn.bind(n)
        fan = np.asarray(b["traj_raw"][n], np.float32)
        lg = fn(np.array(fan, copy=True))                                       # the same call run_scores makes for a_logit
        infan_tok[i], infan_logit[i] = store["tok"][0], lg
        for q in range(Q):
            slot = int(pe["anchor"][n, q])
            lg = stepA.score_replaced(fn, fan, slot, pe["scorer_in"][n, q])      # the same call run_scores makes for p_logit
            pert_tok[i, q], pert_logit[i, q] = store["tok"][0, slot], lg[slot]
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, infan_tok=infan_tok, infan_logit=infan_logit, pert_tok=pert_tok, pert_logit=pert_logit, lo=np.array(lo), hi=np.array(hi),
             bundle_sha256=np.array(sha256_file(a.bundle)), perts_sha256=np.array(sha256_file(a.perts)))
    os.replace(tmp, out_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    D = "D:/Projects/TanitAD-artifacts/l1_stepA_run"
    ap.add_argument("--bundle", default=f"{D}/bundle/bundle.npz")
    ap.add_argument("--perts", default=f"{D}/perts.npz")
    ap.add_argument("--ctx", default=f"{D}/bundle/ctx.npy")
    ap.add_argument("--ckpt", default="D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt")
    ap.add_argument("--refe-dir", default=os.environ.get("L1_REFE_DIR", "C:/Users/Admin/par18/l1run/tree_hybrid/refe"))
    ap.add_argument("--out-dir", default="D:/Projects/TanitAD-artifacts/l2_offprop/shards")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=400)
    a = ap.parse_args(argv)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU only: this exporter never uses the GPU"
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    done = 0
    for lo in range(a.start, a.end, SHARD):
        hi = min(lo + SHARD, a.end)
        p = os.path.join(a.out_dir, f"shard_{lo:04d}.npz")
        if os.path.exists(p):
            print("ZZSKIP", lo, flush=True)
            continue
        t1 = time.time()
        export_range(a, lo, hi, p)
        done += hi - lo
        print(f"ZZSHARD {lo}-{hi} {time.time() - t1:.1f}s  total {time.time() - t0:.0f}s  {done} tokens", flush=True)
    print("ZZEXPORT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
