#!/usr/bin/env python3
"""Shared machinery of the H-ADAM instrument (PREREG_H_ADAM.md).  CPU only.

* The trainer's loss code is the trainer's OWN source: module-level functions are imported from `scorer_finetune.py` (blob af1e2a8), and the closures inside
  `main()` (lane_bce, sel_score, losses, losses6, epo_term, train_losses) are extracted from the file's AST VERBATIM and exec'd with the R0 argv namespace.
* The scorer is built with the model's own classes and `ScorerCopy` / `REFe.score_trajectories`, loaded STRICTLY from the base checkpoint.
* Adam is torch.optim.AdamW itself; the clip is torch.nn.utils.clip_grad_norm_ itself.

REFE_SRC (env) = directory holding the refe/ sources at blob af1e2a8 (recreate with `git archive af1e2a8 <PK>/refe | tar -x`).
"""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import math
import os
import sys
import tarfile
import textwrap
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

SCR = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/c216c849-5fe7-4d43-a3be-32429241f811/scratchpad"
REFE_SRC = os.environ.get("REFE_SRC", f"{SCR}/src/refe")
DL = os.environ.get("RLS1_DL", "C:/Users/Admin/rls1_adam_dl")
X = "C:/Users/Admin/rls1b_dl/data/x"                               # extracted, sha256-verified RL tarballs (the attribution package's tree)
CACHE = "E:/Projects/TanitAD/data/refe_trunk_cache/d7c59f4f"
BASE_CKPT = f"{DL}/rls1/base/model_sft4_B_u400.pt"
R0_PASS = "C:/Users/Admin/rls1b_dl/r0/final/PASS_a120418a870f.json"
POOL_DIR = f"{SCR}/adam_pool"
CLOSURES = ("lane_bce", "sel_score", "losses", "losses6", "epo_term", "train_losses")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_trainer():
    sys.path.insert(0, REFE_SRC)
    import scorer_finetune as SF
    return SF


def r0_args(SF):
    argv = json.load(open(R0_PASS, encoding="utf-8"))["binding"]["argv"]
    return SF.build_parser().parse_args(argv), argv


def build_scorer(SF, a):
    """-> (holder, cfg): the three scorer modules, built as REFe.__init__ builds them (model.py:551-563), sized by the checkpoint's own meta."""
    import model as M
    cfg = SF.ckpt_io.config_for_checkpoint(M.REFeConfig.for_backbone(a.backbone), BASE_CKPT)

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            sdd = cfg.dec_depth if cfg.score_dec_depth is None else cfg.score_dec_depth
            self.score_dec = nn.ModuleList([M.CrossBlock(cfg.dec_width, cfg.dec_heads, cfg.dec_width) for _ in range(sdd)])
            self.score_q_mlp = nn.Sequential(nn.Linear(cfg.horizon_steps * cfg.traj_dim, cfg.dec_width), nn.GELU(), nn.Linear(cfg.dec_width, cfg.dec_width))
            self.score_head = nn.Linear(cfg.dec_width, 7)

    h = Holder()
    ck = torch.load(BASE_CKPT, map_location="cpu", weights_only=False, mmap=True)["model"]
    for m in SF.SCORER:
        sd = {k[len(m) + 1:]: v.detach().clone().float() for k, v in ck.items() if k.startswith(m + ".")}
        getattr(h, m).load_state_dict(sd, strict=True)                    # C6: raises on any missing / unexpected key or shape mismatch
    return h, cfg


def extract_closures(SF, ns_extra: dict):
    """exec the trainer's own nested function sources (VERBATIM, from the AST of main()) in a namespace = the module's globals + `ns_extra`."""
    path = SF.__file__
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    ns = dict(vars(SF))
    ns.update(ns_extra)
    found = {}
    for node in main.body:
        if isinstance(node, ast.FunctionDef) and node.name in CLOSURES:
            seg = textwrap.dedent(ast.get_source_segment(src, node))
            found[node.name] = hashlib.sha256(seg.encode()).hexdigest()[:16]
            exec(compile(seg, f"{path}::{node.name}", "exec"), ns)
    missing = [n for n in CLOSURES if n not in found]
    if missing:
        raise RuntimeError(f"closures not found in main(): {missing}")
    return {n: ns[n] for n in CLOSURES}, found


# --------------------------------------------------------------------------------------------------------------------------- data pool
def cache_index():
    """{(log, token, step, rank): (shard_dir, row)} over the three splits of the trunk cache (visual_ctx present for every frame)."""
    idx = {}
    for sp in ("navtrain_train", "navtrain_val"):
        d = f"{CACHE}/{sp}"
        for sh in sorted(glob.glob(f"{d}/shard_*")):
            ks = np.load(f"{sh}/key.npy", allow_pickle=False)
            for i, k in enumerate(ks):
                lg, tk, st, rk = str(k).split("|")
                idx[(lg, tk, int(st), int(rk))] = (sh, i)
    return idx


def load_visual(shard, row):
    arr = np.load(f"{shard}/visual_ctx.npy", mmap_mode="r", allow_pickle=False)
    return np.array(arr[row])                                              # int16 bf16 bits [7680, 256]


def bf16bits_to_f32(x):
    return torch.from_numpy(x).view(torch.bfloat16).float()


def prepare_pool(SF, a, cfg, side="train", max_n=None):
    """Real OnPolicyBank (the trainer's own) over the sets that have a cached visual_ctx -> dict of arrays.  side 'train' = RL-train, 'val' = RL-val."""
    os.makedirs(POOL_DIR, exist_ok=True)
    ci = cache_index()
    if side == "train":
        tar = f"{DL}/rls1/data/rlfan_sets_train.tar.gz"
        sets_dir, pdm_dir = f"{POOL_DIR}/sets_train_full", f"{X}/pdm_train/pdm_train"
        tl_tar, tl_dir = f"{DL}/rls1/data/rlfan_tlane_train.tar.gz", f"{POOL_DIR}/tlane_train"
        terms_dir = f"{X}/terms_train/terms_train"
        if not glob.glob(f"{sets_dir}/*.jsonl"):
            os.makedirs(sets_dir, exist_ok=True)
            with tarfile.open(tar) as t:
                t.extractall(sets_dir)
        if not glob.glob(f"{tl_dir}/**/teacher_lane_r*.jsonl", recursive=True):
            os.makedirs(tl_dir, exist_ok=True)
            with tarfile.open(tl_tar) as t:
                t.extractall(tl_dir)
    else:
        sets_dir, pdm_dir, terms_dir = f"{X}/sets_val/sets_val", f"{X}/pdm_val/pdm_val", f"{X}/terms_val/terms_val"
        tl_dir = f"{X}/tlane_val/tlane_heldout"
    src_files = sorted(glob.glob(f"{sets_dir}/**/onpolicy_*.jsonl", recursive=True))
    pool_sets = f"{POOL_DIR}/pool_sets_{side}"
    os.makedirs(pool_sets, exist_ok=True)
    out_p = f"{pool_sets}/onpolicy_pool.jsonl"
    n_in = n_keep = 0
    with open(out_p, "w", encoding="utf-8") as fo:
        for fp in src_files:
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    n_in += 1
                    if '"kind": "onpolicy_set"' not in line[:200] and '"onpolicy_set"' not in line[:400]:
                        continue
                    r = json.loads(line)
                    k = (r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0)))
                    if k in ci:
                        fo.write(line if line.endswith("\n") else line + "\n")
                        n_keep += 1
    tl = sorted({os.path.dirname(p) for p in glob.glob(f"{tl_dir}/**/teacher_lane_r*.jsonl", recursive=True)})[0]
    op = SF.T.OnPolicyBank(pool_sets, cfg.n_proposals, cfg.horizon_steps, pdm_labels=pdm_dir, teacher_lane_labels=tl)
    keys = sorted(k for k in op.by if k in op.pdm_keys and k in ci and k in op.teacher_lane)
    terms, st_t = SF.load_rl_terms(terms_dir, op.by, cfg.n_proposals, a.epo_lam_d != 0.0, a.epo_k, a.epo_l)
    keys = [k for k in keys if k in terms]
    if max_n:
        keys = keys[:max_n]
    cxy = np.stack([op.get(*k)[0] for k in keys]).astype(np.float32)
    ctg = np.stack([op.get(*k)[1] for k in keys]).astype(np.float32)
    ckpt = [int(op.get(*k)[2]) for k in keys]
    info = {"side": side, "lines_scanned": n_in, "lines_with_cached_visual": n_keep, "n_pool": len(keys), "pdm_matched": op.n_pdm, "tlane_matched": op.n_tlane,
            "terms_stats": st_t, "n_cols": op.n_cols, "ckpt_steps": sorted(set(ckpt))}
    return {"keys": keys, "cxy": cxy, "ctg": ctg, "terms": {k: terms[k] for k in keys}, "info": info, "ci": ci}


class Pool:
    def __init__(self, d):
        self.keys, self.cxy, self.ctg, self.terms, self.info, self.ci = d["keys"], d["cxy"], d["ctg"], d["terms"], d["info"], d["ci"]
        self.rows = [{"log_name": k[0], "token": k[1], "step": k[2], "rank": k[3]} for k in self.keys]
        self.vis = None

    def load_visual_all(self):
        self.vis = np.stack([load_visual(*self.ci[k]) for k in self.keys])        # int16 [N, 7680, 256]; N*3.9 MB
        return self

    def micro(self, ids):
        ids = list(ids)
        return (torch.from_numpy(self.cxy[ids]), torch.from_numpy(self.ctg[ids]),
                torch.stack([bf16bits_to_f32(self.vis[i]) for i in ids]), ids)


def make_ns(SF, a):
    rcfg = SF.rl_reward.RewardCfg(lam_d=a.epo_lam_d, w_prog=a.epo_w_prog, guard=True, lam_k=a.epo_lam_k, use_k=a.epo_k, lam_l=a.epo_lam_l, use_l=a.epo_l)
    compw = torch.tensor([float(x) for x in a.b_compw.split(",")])
    return rcfg, compw


class Trainer:
    """Bundles the trainer's closures over one pool.  `fwd` is the trainer's fwd without the trunk: arm B / base scorers on (cxy, sctx)."""

    def __init__(self, SF, a, holder, pool: Pool):
        self.SF, self.a, self.pool = SF, a, pool
        rcfg, compw = make_ns(SF, a)
        self.epo_last = {}
        terms = pool.terms
        ds = SimpleNamespace(rows=pool.rows)
        self.fn, self.closure_hashes = extract_closures(SF, {"a": a, "compw": compw, "rcfg": rcfg, "terms_train": terms, "ds": ds, "epo_last": self.epo_last})
        self.base = SF.ScorerCopy(holder)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.armB = SF.ScorerCopy(holder)
        self.armA = SF.ScorerCopy(holder)
        self.params = [p for p in self.armB.parameters()]
        self.names = [n for n, _ in self.armB.named_parameters()]
        self._sx0 = {}

    def sx0(self, ids, cxy, sctx):
        out = []
        for j, i in enumerate(ids):
            if i not in self._sx0:
                with torch.no_grad():
                    self._sx0[i] = self.base(cxy[j:j + 1], sctx[j:j + 1])[0].float()
            out.append(self._sx0[i])
        return torch.stack(out)

    def forward(self, ids, cxy, sctx, arm=None, autocast=False):
        arm = self.armB if arm is None else arm
        if autocast:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                sx = arm(cxy, sctx)
            return sx.float()
        return arm(cxy, sctx)

    def micro_grads(self, ids, cxy, ctg, sctx):
        """-> (gB list, gE list, info): d(BCE+lane BCE)/dtheta and d(unit EPO)/dtheta for ONE micro-batch, each divided by accum like the trainer.
        BCE part = lB - ep at lam_pg 1.0 (the trainer's train_losses, called once); EPO part = ep at lam_pg 1.0."""
        a = self.a
        sx0 = self.sx0(ids, cxy, sctx)
        sxB = self.forward(ids, cxy, sctx)
        lA, lB, ln, ep = self.fn["train_losses"](sx0, sxB.detach(), sxB, ctg, ids, 0, lam_pg=1.0)
        bce = lB - ep
        gB = torch.autograd.grad(bce / a.accum, self.params, retain_graph=True)
        gE = torch.autograd.grad(ep / a.accum, self.params)
        return [g.detach().clone() for g in gB], [g.detach().clone() for g in gE], {"bce": float(bce), "epo_unit": float(ep), **{k: float(v) for k, v in self.epo_last.items()}}


def lr_at(u, a, total=1000):
    """the trainer's own lambda (scorer_finetune.py:1298), transcribed for a given (lr, warmup, total)"""
    return a.lr * (u + 1) / a.warmup if u < a.warmup else a.lr * 0.5 * (1 + math.cos(math.pi * (u - a.warmup) / max(total - a.warmup, 1)))


def lam_t(lam, u, a):
    return lam * (1.0 if a.epo_ramp <= 0 else min(1.0, u / a.epo_ramp))      # = a.epo_lam * epo_ramp(upd, a.epo_ramp), scorer_finetune.epo_ramp


def flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def unflat(v, like):
    out, o = [], 0
    for t in like:
        n = t.numel()
        out.append(v[o:o + n].view_as(t))
        o += n
    return out
