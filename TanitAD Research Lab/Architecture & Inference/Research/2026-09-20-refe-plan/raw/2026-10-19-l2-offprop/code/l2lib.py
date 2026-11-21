#!/usr/bin/env python3
"""L2-OFFPROP core: data, folds, the retrained EP head (arms H / H-null / planted), out-of-fold scoring, and the A1-A5 read (DRAFT_PREREG_L2.md).

WHAT THE "EP HEAD" IS (MEASURED from the checkpoint, not assumed): P2's scorer ends in `score_head = nn.Linear(256, 6)`; its row 2 is the EP head (component order NC, DAC, EP, TTC,
C, DDC; `stepA.py` docstring).  So the EP head has 257 parameters and reads a 256-d token: the INPUT of `score_head`, which `export_feats.py` recorded with a forward pre-hook while
driving the LANDED `REFe.score_trajectories` through L1 Step A's own `make_score_fn` / `score_replaced`.

NOTHING from L1 is re-implemented.  The pair assembly, the A1-A5 terms, the paired log-cluster bootstrap (24 logs, 10,000 resamples, seed 20260927) and the aggregate (V1_W = 5,5,2) are
`stepA.assemble_pairs`, `stepA.admission_terms`, `stepA.per_log_sums`, `stepA.ratio_ci`, `stepA.boot_indices`, `stepA.aggregate`.  This file only (a) substitutes the EP column of the
banked logits with an arm's out-of-fold EP score, (b) trains the head, (c) adds the in-fan and per-family reads the draft names.

TIER / STAMP: every number from here is SEEN-SCENE, SELECTION tier (draft section 1).  ONE training seed per arm cell; the bootstrap answers "another draw of LOGS" only (H-ESTIM-SEED-1).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _l1_code_dir() -> str:
    cands = [os.environ.get("L1_CODE"),
             os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "2026-10-18-ep-rl-lit", "l1_stepA")),     # the repo layout (landed at d8b77c1)
             "C:/Users/Admin/par18/l1run/code"]                                                                      # the run copy (byte-identical stepA.py, sha256 05d182c7...)
    for c in cands:
        if c and os.path.exists(os.path.join(c, "stepA.py")):
            return c
    raise ImportError("L1's stepA.py not found: set L1_CODE (no silent fallback)")


L1_CODE = _l1_code_dir()
if L1_CODE not in sys.path:
    sys.path.insert(0, L1_CODE)
import stepA  # noqa: E402  (L1's reference implementation, imported unchanged)

# ------------------------------------------------------------------------------------------------------------------------------ registered constants (draft section in brackets)
EP = 2                                   # component index of EP in a logit row [3.arms]
N_FOLDS = 5                              # [1] 5 folds by LOG, log rank mod 5 over the 24 logs sorted by name
THR_PAIR = stepA.THR_TRUE_PAIR           # 0.05, the A1 definition [3]
BOOT_B = stepA.N_BOOT                    # 10,000 [4]
BOOT_SEED = stepA.SEED                   # 20260927 [4]
V1_W = stepA.V1_W                        # (5, 5, 2) [4]
TRAIN_SEEDS = (0, 1)                     # [3] "each trained per seed in {0, 1}"
NULL_REPS = (100, 101, 102, 103, 104)    # [3] H-null replicates, seeds 100-104
INFAN_TOP = 8                            # the SELRANK instrument's contender set: the scorer's own top-8
INFAN_THR = 0.05                         # SELRANK: EP pairs need dEP > 0.05 (strict)  -- selrank_attr.py:461
N_ANCHOR_SLOTS = 64


# ------------------------------------------------------------------------------------------------------------------------------ recipe (the two numbers the draft leaves to the agent)
@dataclass(frozen=True)
class Recipe:
    """Fixed BEFORE sealing, justified on SYNTHETIC data only (DESIGN_CHOICES.md).  Everything else in the training is determined by the draft."""
    lr: float = 1e-3
    epochs: int = 30
    batch: int = 256                     # pairs per step; the in-fan BCE batch has the same size
    lam_bce: float = 0.1                 # weight of the pointwise in-fan BCE against the pairwise logistic (both are batch MEANS).  The MM's "lambda" (pairwise / in-fan) is 1 / lam_bce
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    planted_amp: float = 10.0            # C-REACH only: |planted feature| (token per-dim std is ~1.0, token norm ~30)
    prox: float = 0.0                    # proximal weight toward P2's head: + prox * (||w_EP - w_EP(P2)||^2 + (b_EP - b_EP(P2))^2) (MM addendum, E-recipe: "any proximal weight toward P2's head")

    def as_dict(self):
        d = dict(self.__dict__)
        d["betas"] = list(self.betas)
        return d

    @staticmethod
    def from_json(path: str) -> "Recipe":
        """the recipe as a DATA file (sha256-able, sealable), not as source defaults"""
        d = json.load(open(path))
        d = d.get("recipe", d)
        d["betas"] = tuple(d.get("betas", (0.9, 0.999)))
        known = set(Recipe.__dataclass_fields__)
        extra = sorted(set(d) - known)
        if extra:
            raise ValueError(f"recipe file has unknown fields {extra}: a silently ignored lever is the 'built, tested, unreachable' class")
        return Recipe(**d)


# ------------------------------------------------------------------------------------------------------------------------------ small helpers
def sha256_file(p: str) -> str:
    return stepA.sha256_file(p)


def sha256_arrays(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


GATE_STAMPED_CODE = ("l2lib.py", "gates.py", "synthetic.py")


def sha256_file_lf(p: str) -> str:
    """sha256 of a file with CRLF folded to LF: a git checkout with autocrlf must not make the SAME source look like different code (it would refuse every gate PASS)"""
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def gate_stamp(recipe_path: str | None) -> dict:
    """what a gate's PASS was earned on: the recipe FILE's sha256 and the code it ran.  `run_l2.check_seal` refuses a PASS whose stamp does not match the recipe and code the run is about to use
    (the chain's C-REACH FAIL was earned on a recipe that was then re-selected: a gate that is not bound to its recipe certifies nothing about the next one)."""
    return {"recipe_sha256": sha256_file_lf(recipe_path) if recipe_path and os.path.exists(recipe_path) else None,
            "code_sha256": {fn: sha256_file_lf(os.path.join(HERE, fn)) for fn in GATE_STAMPED_CODE}}


def fold_of_logs(log_names) -> np.ndarray:
    """[1] fold = rank mod 5 over the logs SORTED BY NAME.  Returns the fold of each name in the order given (the order is irrelevant: rank is taken in sorted order)."""
    names = [str(x) for x in log_names]
    if len(set(names)) != len(names):
        raise ValueError("log names must be unique")
    rank = {n: i for i, n in enumerate(sorted(names))}
    return np.array([rank[n] % N_FOLDS for n in names], dtype=np.int64)


def ep_apply(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The EP logit of a token: row 2 of the head applied in float64 (float32 inputs and weights are exact in float64, so the result does not depend on summation blocking to within 1 ulp-of-float64).
    X [..., D], W [6, D], b [6] -> [...]"""
    return np.asarray(X, np.float64) @ np.asarray(W[EP], np.float64) + float(b[EP])


# ------------------------------------------------------------------------------------------------------------------------------ the data
@dataclass
class Data:
    """Everything an arm needs.  Real data: `load_real`.  Synthetic: `synthetic.make_data` (same fields, same shapes)."""
    bundle: dict                 # token, log, traj_raw, logits, pick, sub[N,64,6], pdms, valid
    perts: dict                  # exec, scorer_in, anchor[N,Q], kind, param, is_top, profile, seed
    labels: dict                 # sub[N,Q,6], pdms[N,Q]
    scores: dict                 # a_logit[N,64,6], p_logit[N,Q,6]  (Step A's banked EP and all other heads)
    tok_in: np.ndarray           # [N,64,D] float32   EP-head input of every in-fan proposal
    tok_p: np.ndarray            # [N,Q,D]  float32   EP-head input at the anchor slot of every perturbed set
    W6: np.ndarray               # [6,D] P2 score_head weight
    b6: np.ndarray               # [6]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.N, self.Q = self.perts["anchor"].shape
        self.D = self.tok_in.shape[-1]
        if self.tok_in.shape[:2] != (self.N, N_ANCHOR_SLOTS) or self.tok_p.shape[:2] != (self.N, self.Q):
            raise ValueError(f"feature shapes {self.tok_in.shape} {self.tok_p.shape} do not match N={self.N} Q={self.Q}")
        self.log_names = sorted(set(str(x) for x in self.bundle["log"]))
        li = {l: i for i, l in enumerate(self.log_names)}
        self.n_logs = len(self.log_names)
        self.log_idx = np.array([li[str(x)] for x in self.bundle["log"]], dtype=np.int64)
        self.fold_of_log = fold_of_logs(self.log_names)
        self.fold = self.fold_of_log[self.log_idx]                       # fold of each token
        self.anchor = np.asarray(self.perts["anchor"], np.int64)         # [N,Q]
        self.kind = np.asarray(self.perts["kind"], np.int64)
        self.param = np.asarray(self.perts["param"], np.float64)
        self.t_in = np.asarray(self.bundle["sub"], np.float64)[..., EP]  # [N,64] true EP of the in-fan proposals
        self.t_p = np.asarray(self.labels["sub"], np.float64)[..., EP]   # [N,Q]  true EP of the perturbations
        self.t_a = np.take_along_axis(self.t_in, self.anchor, 1)         # [N,Q]  true EP of each pair's anchor
        # Step A's own pair assembly with the BANKED logits -> P0 (everything but the EP column is final); pair order is token-major, job-minor
        P, li_pairs, _ = stepA.assemble_pairs(self.bundle, self.perts, self.scores, self.labels, self.log_names)
        if len(P["kind"]) != self.N * self.Q:
            raise ValueError(f"assemble_pairs dropped pairs ({len(P['kind'])} != {self.N * self.Q}): tokens with non-finite labels are not supported by this arm")
        if not np.array_equal(P["tok"], np.repeat(np.arange(self.N), self.Q)):
            raise AssertionError("pair order is not token-major")
        self.P0, self.pair_log = P, li_pairs
        self.a_slot = self.anchor.reshape(-1)
        self.pair_tok = P["tok"]
        self.boot_idx = stepA.boot_indices(self.n_logs, BOOT_B, BOOT_SEED)
        # the SELRANK contender set: P2's own top-8 by its aggregate (banked logits), frozen for every arm
        agg = stepA.aggregate(np.asarray(self.scores["a_logit"], np.float64))
        self.top8 = np.argsort(-agg, axis=1, kind="stable")[:, :INFAN_TOP]


def load_real(feats_path: str, run_dir: str = "D:/Projects/TanitAD-artifacts/l1_stepA_run", head_path: str | None = None) -> Data:
    bundle = stepA.load_npz(f"{run_dir}/bundle/bundle.npz")
    perts = stepA.load_npz(f"{run_dir}/perts.npz")
    labels = stepA.load_npz(f"{run_dir}/labels.npz")
    scores = stepA.load_npz(f"{run_dir}/scores.npz")
    f = np.load(feats_path, allow_pickle=False)
    hp = head_path or os.path.join(os.path.dirname(feats_path), "p2_score_head.npz")
    h = np.load(hp, allow_pickle=False)
    return Data(bundle, perts, labels, scores, f["infan_tok"], f["pert_tok"], h["weight"], h["bias"],
                meta={"feats_path": feats_path, "run_dir": run_dir, "kind": "real"})


def shuffled_labels_data(D: Data, seed: int = 12345) -> Data:
    """C-LEAK on REAL features with SYNTHETIC labels: the EP column of BOTH the perturbation labels and the in-fan labels is shuffled globally, which kills every feature-label relation.  Evaluating this
    Data reads NO real label."""
    rng = np.random.default_rng(seed)
    labels = {k: np.array(v, copy=True) for k, v in D.labels.items()}
    bundle = {k: np.array(v, copy=True) for k, v in D.bundle.items()}
    ls = labels["sub"].reshape(-1, 6)
    ls[:, EP] = ls[rng.permutation(len(ls)), EP]
    labels["sub"] = ls.reshape(D.labels["sub"].shape)
    bs = bundle["sub"].reshape(-1, 6)
    bs[:, EP] = bs[rng.permutation(len(bs)), EP]
    bundle["sub"] = bs.reshape(D.bundle["sub"].shape)
    return Data(bundle, D.perts, labels, D.scores, D.tok_in, D.tok_p, D.W6, D.b6, meta={**D.meta, "kind": "shuffled-labels"})


# ------------------------------------------------------------------------------------------------------------------------------ the training set of one fold
@dataclass
class TrainSet:
    ia: np.ndarray       # [M] flat index into tok_in.reshape(N*64, D)   (the pair's anchor token)
    ip: np.ndarray       # [M] flat index into tok_p.reshape(N*Q, D)     (the pair's perturbation token)
    y: np.ndarray        # [M] +1 if the perturbation's TRUE EP is higher, -1 if lower
    fan_idx: np.ndarray  # [F] flat index into tok_in (every in-fan proposal of every training token)
    fan_t: np.ndarray    # [F] its true EP in [0,1]
    tokens: np.ndarray   # the training tokens


NULL_GROUP_DEFAULT = "global"            # MM ruling E1 (2026-10-19): the H-null is the GLOBAL shuffle; the draft's within-(token, anchor) permutation is WITHDRAWN


def permute_pert_labels(D: Data, rep_seed: int, group: str = NULL_GROUP_DEFAULT) -> np.ndarray:
    """[3] H-null (MM ruling E1): the perturbations' true EP is permuted ACROSS EVERY perturbation of the dataset (`group='global'`, the DEFAULT and the only definition the sealed prereg uses).
    The training-noise floor of THIS machinery: a head trained on labels with no information about the tokens must not raise A1-OOF.  The draft's within-(token, anchor) permutation (`group='anchor'`)
    is WITHDRAWN: it keeps the BETWEEN-group structure learnable (a group-level shift, e.g. the EP ceiling of a high-EP anchor, survives a within-group permutation) and read +0.09 above P2 on a planted
    world (`test_documented_caveat_*`), so the VOID rule could fire for a reason that is not an instrument fault.  `anchor` and `token` (across the 96 perturbations of a token) stay available as DIAGNOSTICS
    only: nothing in the outcome mapping reads them unless `--null-group` selects them explicitly.
    Returns the permuted [N,Q] array.  A pure function of (D's structure, rep_seed, group)."""
    rng = np.random.default_rng(int(rep_seed))
    out = np.array(D.t_p, copy=True)
    if group == "global":
        return D.t_p.reshape(-1)[rng.permutation(D.N * D.Q)].reshape(D.N, D.Q)
    for n in range(D.N):
        if group == "token":
            out[n] = D.t_p[n, rng.permutation(D.Q)]
            continue
        if group != "anchor":
            raise ValueError(f"unknown null group {group!r}")
        for a in np.unique(D.anchor[n]):
            q = np.nonzero(D.anchor[n] == a)[0]
            out[n, q] = D.t_p[n, q[rng.permutation(len(q))]]
    return out


def make_trainset(D: Data, tokens: np.ndarray, t_p_train: np.ndarray | None = None) -> TrainSet:
    """[3] pairs (anchor in-fan token, perturbation token) of `tokens` whose TRUE EP differs by >= 0.05 (the A1 definition) + every in-fan proposal for the pointwise BCE.
    `t_p_train` overrides the perturbation EP labels (H-null passes the permuted ones); the pair MEMBERSHIP and SIGN are then taken from them."""
    tokens = np.asarray(tokens, np.int64)
    tp = D.t_p if t_p_train is None else np.asarray(t_p_train, np.float64)
    diff = tp[tokens] - D.t_a[tokens]                                             # [T,Q]
    m = np.abs(diff) >= THR_PAIR - 1e-12
    n_idx = np.broadcast_to(tokens[:, None], diff.shape)
    q_idx = np.broadcast_to(np.arange(D.Q)[None, :], diff.shape)
    ia = (n_idx * N_ANCHOR_SLOTS + D.anchor[tokens])[m]
    ip = (n_idx * D.Q + q_idx)[m]
    y = np.sign(diff[m])
    fan_idx = (tokens[:, None] * N_ANCHOR_SLOTS + np.arange(N_ANCHOR_SLOTS)[None, :]).reshape(-1)
    fan_t = D.t_in[tokens].reshape(-1)
    return TrainSet(ia.astype(np.int64), ip.astype(np.int64), y.astype(np.float32), fan_idx.astype(np.int64), fan_t.astype(np.float32), tokens)


def assert_no_leak(D: Data, train_tokens, test_tokens) -> dict:
    """C-LEAK structural invariant: no TOKEN and no LOG of the held-out fold enters training.  Called by `oof_scores` on every fold; the RED mutation (`leak_fold`) must trip it."""
    tr, te = np.asarray(train_tokens, np.int64), np.asarray(test_tokens, np.int64)
    tok_overlap = np.intersect1d(tr, te)
    log_overlap = np.intersect1d(D.log_idx[tr], D.log_idx[te])
    fold_overlap = np.intersect1d(D.fold[tr], D.fold[te])
    if len(tok_overlap) or len(log_overlap) or len(fold_overlap):
        raise RuntimeError(f"C-LEAK: held-out data entered training: {len(tok_overlap)} tokens, {len(log_overlap)} logs, folds {fold_overlap.tolist()}")
    return {"tokens_overlap": 0, "logs_overlap": 0, "n_train": int(len(tr)), "n_test": int(len(te))}


def assert_trainset_clean(D: Data, ts: "TrainSet", train_tokens, test_tokens) -> dict:
    """C-LEAK reach check on the code path that FEEDS THE OPTIMISER: every token referenced by any index of the training set (the anchor token of a pair, the perturbation token of a pair, the in-fan
    proposal of the BCE) must be a training token of the fold, and none may be a held-out token or share a log with one.  `assert_no_leak` checks the token LISTS; this checks what `make_trainset`
    actually emitted, so a bug that routed a held-out row into the pair or in-fan index arrays is caught even when the lists are clean.  Fails closed (RuntimeError)."""
    tr, te = np.asarray(train_tokens, np.int64), np.asarray(test_tokens, np.int64)
    used = np.unique(np.concatenate([ts.ia // N_ANCHOR_SLOTS, ts.ip // D.Q, ts.fan_idx // N_ANCHOR_SLOTS]))
    stray = np.setdiff1d(used, tr)
    held = np.intersect1d(used, te)
    log_overlap = np.intersect1d(D.log_idx[used], D.log_idx[te])
    if len(stray) or len(held) or len(log_overlap):
        raise RuntimeError(f"C-LEAK: the training set references {len(stray)} tokens outside the fold's training tokens, {len(held)} held-out tokens, {len(log_overlap)} held-out logs")
    return {"tokens_used_by_trainset": int(len(used)), "held_out_tokens_referenced": 0, "held_out_logs_referenced": 0}


# ------------------------------------------------------------------------------------------------------------------------------ the head and its training
def _torch():
    import torch
    return torch


class HeadLoss:
    """The registered loss terms, as ONE place so G-LIVE and the training loop use the same code:  pairwise logistic over (anchor, perturbation) pairs of the batch
    + lam * pointwise BCE (soft EP target) over in-fan proposals of the batch.  Both are batch MEANS."""

    @staticmethod
    def terms(W, b, xa, xp, y, xf, tf, lam, w_pair=1.0, prox=0.0, W0=None, b0=None):
        """-> (pair, bce, total) with total = w_pair * pair + lam * bce + prox * (||W[EP]-W0[EP]||^2 + (b[EP]-b0[EP])^2).  `w_pair = 0` is arm H0 (in-fan BCE only); `prox` acts on the EP row only, so every
        frozen row's gradient stays exactly 0.0."""
        torch = _torch()
        F = torch.nn.functional
        za = xa @ W[EP] + b[EP]
        zp = xp @ W[EP] + b[EP]
        zf = xf @ W[EP] + b[EP]
        pair = F.softplus(-(y * (zp - za))).mean()
        bce = F.binary_cross_entropy_with_logits(zf, tf)
        tot = w_pair * pair + lam * bce
        if prox:
            if W0 is None or b0 is None:
                raise ValueError("prox > 0 needs the P2 head (W0, b0)")
            tot = tot + prox * (((W[EP] - W0[EP]) ** 2).sum() + (b[EP] - b0[EP]) ** 2)
        return pair, bce, tot


def _pad(W6, b6, extra):
    torch = _torch()
    W = torch.zeros(6, W6.shape[1] + extra, dtype=torch.float32)
    W[:, :W6.shape[1]] = torch.as_tensor(np.asarray(W6, np.float32))
    return W, torch.as_tensor(np.asarray(b6, np.float32)).clone()


def fit_head(Xin: np.ndarray, Xp: np.ndarray, ts: TrainSet, W6, b6, seed: int, rec: Recipe, steps: int | None = None, extra: int = 0,
             loss_mutation: str | None = None, check_frozen: bool = True, w_pair: float = 1.0) -> dict:
    """Train row 2 of the head (257 + extra parameters).  Xin [N*64,D'], Xp [N*Q,D'] flat float32 (D' = D + extra).  Adam, no weight decay, minibatches of `rec.batch` pairs shuffled each
    epoch by a generator seeded from (seed, n_pairs); the in-fan BCE batch is drawn with the same generator.  `steps` caps the number of optimiser steps (0 = C-ID: no step is taken).
    The full [6, D'] weight is the optimised leaf so that FROZEN rows are real parameters with real gradients, and `check_frozen` asserts their gradient is EXACTLY 0.0 at every step and
    that they are bit-identical to P2's at the end.  `loss_mutation='leak_other_row'` is the RED mutation of G-LIVE: it adds a term through row 3, which must trip that assertion.
    `w_pair` is the weight of the pairwise term: 1.0 for H, 0.0 for H0 (the SAME recipe, same init, same minibatch stream, in-fan BCE only); `rec.prox` is the proximal weight toward P2's EP row."""
    torch = _torch()
    W0, b0 = _pad(W6, b6, extra)
    W = torch.nn.Parameter(W0.clone())
    b = torch.nn.Parameter(b0.clone())
    opt = torch.optim.Adam([W, b], lr=rec.lr, betas=rec.betas, eps=rec.eps, weight_decay=0.0)
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(len(ts.ia)))
    Xi, Xq = torch.from_numpy(np.ascontiguousarray(Xin, np.float32)), torch.from_numpy(np.ascontiguousarray(Xp, np.float32))
    ia, ip = torch.from_numpy(ts.ia), torch.from_numpy(ts.ip)
    yy = torch.from_numpy(ts.y)
    fi, ft = torch.from_numpy(ts.fan_idx), torch.from_numpy(ts.fan_t)
    M, Fn = len(ia), len(fi)
    frozen = [r for r in range(6) if r != EP]
    done, losses, ep_grad_min = 0, [], float("inf")
    if M == 0:
        raise ValueError("no training pairs")
    for _ in range(rec.epochs):
        order = torch.randperm(M, generator=g)
        for s in range(0, M, rec.batch):
            if steps is not None and done >= steps:
                break
            bi = order[s:s + rec.batch]
            bf = torch.randint(0, Fn, (len(bi),), generator=g)
            pair, bce, tot = HeadLoss.terms(W, b, Xi[ia[bi]], Xq[ip[bi]], yy[bi], Xi[fi[bf]], ft[bf], rec.lam_bce, w_pair, rec.prox, W0, b0)
            if loss_mutation == "leak_other_row":
                tot = tot + 0.1 * (Xi[fi[bf]] @ W[3] + b[3]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            tot.backward()
            if check_frozen:
                fg = max(float(W.grad[frozen].abs().max()), float(b.grad[frozen].abs().max()))
                if fg != 0.0:
                    raise RuntimeError(f"G-LIVE: a FROZEN row received a non-zero gradient ({fg:.3e})")
            ep_grad_min = min(ep_grad_min, float(W.grad[EP].abs().sum()))
            if not (math.isfinite(float(pair.detach())) and math.isfinite(float(bce.detach()))):
                raise RuntimeError("G-LIVE: a loss term is not finite")
            opt.step()
            done += 1
            if done % 50 == 1:
                losses.append((done, float(pair.detach()), float(bce.detach())))
        if steps is not None and done >= steps:
            break
    Wn, bn = W.detach().numpy().copy(), b.detach().numpy().copy()
    if check_frozen:
        if not (np.array_equal(Wn[frozen], W0.numpy()[frozen]) and np.array_equal(bn[frozen], b0.numpy()[frozen])):
            raise RuntimeError("G-LIVE: a frozen row changed")
    return {"W": Wn, "b": bn, "steps": done, "loss_trace": losses, "min_ep_grad_abs_sum": (None if ep_grad_min == float("inf") else ep_grad_min)}


def g_live(Xin, Xp, ts: TrainSet, W6, b6, rec: Recipe, n_batches: int = 3, extra: int = 0) -> dict:
    """G-LIVE (draft section 5): every loss term finite; the EP head's parameters receive a NON-ZERO gradient; every frozen parameter's gradient is EXACTLY 0.0.  Plus its control: the mutated loss
    (a term through row 3) must make a frozen row's gradient non-zero, i.e. the probe can read a non-zero at all.
    REACH of the two levers added by the MM rulings (the 'built, tested, unreachable from its caller' class):
      H0   `w_pair = 0` -> the EP-row gradient equals the gradient of `lam * BCE` ALONE bit for bit (the pairwise term contributes exactly 0), and differs from H's (the pairwise term DOES reach the head in H);
      prox `rec.prox > 0` -> at the P2 init its gradient is exactly 0 (nothing to restore), off the init the EP-row gradient changes (it reaches the head) and a frozen row's stays exactly 0.0."""
    torch = _torch()
    W0, b0 = _pad(W6, b6, extra)
    g = torch.Generator().manual_seed(7)
    Xi, Xq = torch.from_numpy(np.ascontiguousarray(Xin, np.float32)), torch.from_numpy(np.ascontiguousarray(Xp, np.float32))
    ia, ip, yy, fi, ft = (torch.from_numpy(a) for a in (ts.ia, ts.ip, ts.y, ts.fan_idx, ts.fan_t))
    frozen = [r for r in range(6) if r != EP]
    out = {"batches": [], "frozen_rows": frozen, "ep_row": EP}
    for _ in range(n_batches):
        bi = torch.randint(0, len(ia), (256,), generator=g)
        bf = torch.randint(0, len(fi), (256,), generator=g)
        batch = (Xi[ia[bi]], Xq[ip[bi]], yy[bi], Xi[fi[bf]], ft[bf])
        for mut in (False, True):
            W, b = torch.nn.Parameter(W0.clone()), torch.nn.Parameter(b0.clone())
            pair, bce, tot = HeadLoss.terms(W, b, *batch, rec.lam_bce, 1.0, rec.prox, W0, b0)
            if mut:
                tot = tot + 0.1 * (batch[3] @ W[3] + b[3]).pow(2).mean()
            tot.backward()
            row = {"mutated": mut, "pair_loss": float(pair.detach()), "bce_loss": float(bce.detach()), "finite": bool(math.isfinite(float(pair.detach())) and math.isfinite(float(bce.detach()))),
                   "ep_W_grad_abs_sum": float(W.grad[EP].abs().sum()), "ep_b_grad_abs": float(b.grad[EP].abs()),
                   "frozen_W_grad_abs_max": float(W.grad[frozen].abs().max()), "frozen_b_grad_abs_max": float(b.grad[frozen].abs().max())}
            out["batches"].append(row)
        # ---- H0 reach: w_pair = 0 gives exactly the gradient of lam * BCE alone; w_pair = 1 does not
        Wa, ba = torch.nn.Parameter(W0.clone()), torch.nn.Parameter(b0.clone())
        _, _, t0 = HeadLoss.terms(Wa, ba, *batch, rec.lam_bce, 0.0, 0.0, W0, b0)
        t0.backward()
        Wb, bb = torch.nn.Parameter(W0.clone()), torch.nn.Parameter(b0.clone())
        F = torch.nn.functional
        (rec.lam_bce * F.binary_cross_entropy_with_logits(batch[3] @ Wb[EP] + bb[EP], batch[4])).backward()
        Wc, bc = torch.nn.Parameter(W0.clone()), torch.nn.Parameter(b0.clone())
        _, _, t1 = HeadLoss.terms(Wc, bc, *batch, rec.lam_bce, 1.0, 0.0, W0, b0)
        t1.backward()
        out.setdefault("h0", []).append({"w_pair0_grad_equals_bce_only_bitwise": bool(torch.equal(Wa.grad, Wb.grad) and torch.equal(ba.grad, bb.grad)),
                                         "w_pair1_grad_differs_from_w_pair0": bool(not torch.equal(Wc.grad[EP], Wa.grad[EP])),
                                         "w_pair0_frozen_grad_max": float(Wa.grad[frozen].abs().max())})
        # ---- prox reach (a nonzero test value, so it does not depend on this recipe's own prox)
        pv = 0.5
        Wp, bp = torch.nn.Parameter(W0.clone()), torch.nn.Parameter(b0.clone())
        _, _, tp0 = HeadLoss.terms(Wp, bp, *batch, rec.lam_bce, 1.0, pv, W0, b0)
        tp0.backward()
        Wq, bq = torch.nn.Parameter(W0.clone()), torch.nn.Parameter(b0.clone())
        _, _, tq0 = HeadLoss.terms(Wq, bq, *batch, rec.lam_bce, 1.0, 0.0, W0, b0)
        tq0.backward()
        off = torch.zeros_like(W0)
        off[EP, :W0.shape[1] - extra] = 0.05
        Wr, br = torch.nn.Parameter(W0.clone() + off), torch.nn.Parameter(b0.clone())
        _, _, tr1 = HeadLoss.terms(Wr, br, *batch, rec.lam_bce, 1.0, pv, W0, b0)
        tr1.backward()
        Ws, bs = torch.nn.Parameter(W0.clone() + off), torch.nn.Parameter(b0.clone())
        _, _, ts1 = HeadLoss.terms(Ws, bs, *batch, rec.lam_bce, 1.0, 0.0, W0, b0)
        ts1.backward()
        out.setdefault("prox", []).append({"zero_at_init": bool(torch.equal(Wp.grad, Wq.grad) and torch.equal(bp.grad, bq.grad)),
                                           "reaches_ep_row_off_init": bool(not torch.equal(Wr.grad[EP], Ws.grad[EP])),
                                           "frozen_grad_max_with_prox": float(Wr.grad[frozen].abs().max())})
    clean = [r for r in out["batches"] if not r["mutated"]]
    mut = [r for r in out["batches"] if r["mutated"]]
    out["pass"] = bool(all(r["finite"] for r in clean) and all(r["ep_W_grad_abs_sum"] > 0 and r["ep_b_grad_abs"] > 0 for r in clean)
                       and all(r["frozen_W_grad_abs_max"] == 0.0 and r["frozen_b_grad_abs_max"] == 0.0 for r in clean))
    out["control_mutation_reads_nonzero_on_frozen"] = bool(all(r["frozen_W_grad_abs_max"] > 0.0 for r in mut))
    out["h0_lever_reaches"] = bool(all(r["w_pair0_grad_equals_bce_only_bitwise"] and r["w_pair1_grad_differs_from_w_pair0"] and r["w_pair0_frozen_grad_max"] == 0.0 for r in out["h0"]))
    out["prox_lever_reaches"] = bool(all(r["zero_at_init"] and r["reaches_ep_row_off_init"] and r["frozen_grad_max_with_prox"] == 0.0 for r in out["prox"]))
    out["pass"] = bool(out["pass"] and out["h0_lever_reaches"] and out["prox_lever_reaches"])
    return out


# ------------------------------------------------------------------------------------------------------------------------------ an arm: out-of-fold EP scores
@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str = "H"                  # 'P2' (untouched) | 'H' | 'H0' (pairwise weight 0: the SAME recipe, in-fan BCE only) | 'Hnull' | 'planted'
    seed: int = 0                    # training seed
    null_rep: int | None = None      # permutation seed (H-null)
    null_group: str = NULL_GROUP_DEFAULT   # 'global' (MM ruling E1; the sealed null) | 'anchor' (the withdrawn draft definition) | 'token': diagnostics only
    steps: int | None = None         # None = full recipe; 0 = C-ID
    leak_fold: int | None = None     # RED mutation of C-LEAK: put held-out tokens of this fold into ITS OWN training set ...
    leak_n: int | None = None        # ... this many of them (None = the whole fold).  There is NO switch that turns the structural check off: it fails closed.


ARM_KINDS = ("P2", "H", "H0", "Hnull", "planted")


def planted_features(D: Data, amp: float):
    """C-REACH: append ONE column.  A perturbation token carries amp * sign(true EP difference to its anchor); every in-fan token carries 0.  Appended, so the head can be asked to use it."""
    sgn = np.sign(D.t_p - D.t_a)                                                         # [N,Q]
    Xp = np.concatenate([D.tok_p, (amp * sgn)[..., None].astype(np.float32)], -1)
    Xi = np.concatenate([D.tok_in, np.zeros(D.tok_in.shape[:-1] + (1,), np.float32)], -1)
    return Xi, Xp


def fold_split(D: Data, f: int, arm: ArmSpec, t_p_train: np.ndarray | None = None):
    """The training / held-out token split of fold `f` AND its structural C-LEAK checks (MM addendum 2026-10-19: C-LEAK is a structural reach check, because a 257-parameter linear head cannot memorise
    shuffled labels and a statistical test of leakage cannot fire on noise).  Returns (train tokens, held-out tokens, TrainSet, check record).  Raises RuntimeError('C-LEAK ...') - fails closed, never skippable -
    if (a) a held-out token or log is in the training token list, (b) the TrainSet's index arrays reference any token outside the training list / any held-out token or log.
    `arm.leak_fold == f` is the RED mutation: it injects held-out tokens into training BEFORE the checks, which must therefore fire."""
    te = np.nonzero(D.fold == f)[0]
    tr = np.nonzero(D.fold != f)[0]
    if len(te) == 0 or len(tr) == 0:
        raise RuntimeError(f"C-LEAK: fold {f} has an empty split ({len(tr)} train, {len(te)} held-out)")
    if arm.leak_fold == f:
        inj = te if arm.leak_n is None else te[:int(arm.leak_n)]
        tr = np.union1d(tr, inj)                                                       # <- the RED mutation: held-out data enters training
    chk = assert_no_leak(D, tr, te)
    ts = make_trainset(D, tr, t_p_train)
    chk["trainset"] = assert_trainset_clean(D, ts, tr, te)
    return tr, te, ts, chk


def oof_scores(D: Data, arm: ArmSpec, rec: Recipe, verbose: bool = False) -> dict:
    """The arm's OUT-OF-FOLD EP score of every in-fan proposal [N,64] and every perturbation token [N,Q]: a token's score comes from the model trained on the OTHER four folds.
    Every fold runs `fold_split` (structural C-LEAK, fails closed) before any optimiser step, and the folds must score every token exactly once."""
    t0 = time.time()
    if arm.kind not in ARM_KINDS:
        raise ValueError(f"unknown arm kind {arm.kind!r}")
    extra = 1 if arm.kind == "planted" else 0
    if arm.kind == "planted":
        Xi3, Xp3 = planted_features(D, rec.planted_amp)
    else:
        Xi3, Xp3 = D.tok_in, D.tok_p
    Xi_flat, Xp_flat = Xi3.reshape(-1, Xi3.shape[-1]), Xp3.reshape(-1, Xp3.shape[-1])
    W6, b6 = D.W6, D.b6
    if arm.kind == "P2":
        return {"ep_in": ep_apply(D.tok_in, W6, b6), "ep_p": ep_apply(D.tok_p, W6, b6), "diag": {"kind": "P2", "folds": []}, "seconds": time.time() - t0}
    t_p_train = None
    if arm.kind == "Hnull":
        if arm.null_rep is None:
            raise ValueError("H-null needs a permutation seed")
        t_p_train = permute_pert_labels(D, arm.null_rep, arm.null_group)
    w_pair = 0.0 if arm.kind == "H0" else 1.0
    ep_in = np.zeros((D.N, N_ANCHOR_SLOTS), np.float64)
    ep_p = np.zeros((D.N, D.Q), np.float64)
    scored = np.zeros(D.N, np.int64)
    folds = []
    for f in range(N_FOLDS):
        tr, te, ts, chk = fold_split(D, f, arm, t_p_train)
        fit = fit_head(Xi_flat, Xp_flat, ts, W6, b6, arm.seed, rec, steps=arm.steps, extra=extra, w_pair=w_pair)
        Wf, bf = fit["W"], fit["b"]
        ep_in[te] = ep_apply(Xi3[te], Wf, bf)
        ep_p[te] = ep_apply(Xp3[te], Wf, bf)
        scored[te] += 1
        folds.append({"fold": f, "n_train_tokens": int(len(tr)), "n_test_tokens": int(len(te)), "n_pairs": int(len(ts.ia)), "steps": fit["steps"], "leak_check": chk,
                      "min_ep_grad_abs_sum": fit["min_ep_grad_abs_sum"], "final_loss_trace": fit["loss_trace"][-1] if fit["loss_trace"] else None})
        if verbose:
            print(f"  [{arm.name}] fold {f}: {len(tr)} train tokens, {len(ts.ia)} pairs, {fit['steps']} steps", flush=True)
    if not (scored == 1).all():
        raise RuntimeError(f"C-LEAK: {int((scored != 1).sum())} tokens were not scored exactly once by a model that did not train on them")
    return {"ep_in": ep_in, "ep_p": ep_p, "diag": {"kind": arm.kind, "seed": arm.seed, "null_rep": arm.null_rep, "null_group": arm.null_group if arm.kind == "Hnull" else None, "folds": folds},
            "seconds": time.time() - t0}


# ------------------------------------------------------------------------------------------------------------------------------ the read: A1-A5 + the per-family table [5]
def _arm_pairs(D: Data, ep_in: np.ndarray, ep_p: np.ndarray) -> dict:
    P = dict(D.P0)
    a = np.array(D.P0["a_logit"], np.float64, copy=True)
    p = np.array(D.P0["p_logit"], np.float64, copy=True)
    a[:, EP] = ep_in[D.pair_tok, D.a_slot]
    p[:, EP] = ep_p.reshape(-1)
    P["a_logit"], P["p_logit"] = a, p
    return P


def infan_top8_terms(D: Data, ep_in: np.ndarray):
    """SELRANK instrument (selrank_attr.py:461-475): inside the scorer's own top-8 (P2's aggregate), ordered pairs whose true EP differs by MORE than 0.05; concordant if the head scores the better one
    higher, ties count half.  Per-token (num, den)."""
    num = np.zeros(D.N)
    den = np.zeros(D.N)
    for i in range(D.N):
        idx = D.top8[i]
        y, x = D.t_in[i, idx], ep_in[i, idx]
        dy = y[:, None] - y[None, :]
        m = dy > INFAN_THR
        dx = x[:, None] - x[None, :]
        den[i] = m.sum()
        num[i] = (m & (dx > 0)).sum() + 0.5 * (m & (dx == 0)).sum()
    return num, den


def paired_diff_ci(numA, denA, numB, denB, idx, alpha=0.05) -> dict:
    """A - B of two ratios-of-sums, on the SAME log resamples (paired).  Descriptive."""
    nA, dA, nB, dB = (np.asarray(x, np.float64) for x in (numA, denA, numB, denB))
    pt = float(nA.sum() / dA.sum() - nB.sum() / dB.sum())
    a_n, a_d, b_n, b_d = nA[idx].sum(1), dA[idx].sum(1), nB[idx].sum(1), dB[idx].sum(1)
    ok = (a_d > 0) & (b_d > 0)
    r = a_n[ok] / a_d[ok] - b_n[ok] / b_d[ok]
    return {"point": pt, "lo": float(np.quantile(r, alpha / 2)), "hi": float(np.quantile(r, 1 - alpha / 2)), "n_valid_resamples": int(ok.sum())}


def evaluate_arm(D: Data, ep_in: np.ndarray, ep_p: np.ndarray, ref_terms: dict | None = None) -> dict:
    """A1..A5(b) for all / lon / lat via L1's own `stepA.analyse`, with the arm's EP score in the aggregate; plus per-speed A1, the lateral DAC concordance, the in-fan top-8 EP concordance and the
    tactical share.  `ref_terms` (the P2 read's per-log sums) enables the paired A1 difference."""
    P = _arm_pairs(D, ep_in, ep_p)
    res = stepA.analyse(P, D.pair_log, D.n_logs, BOOT_B, BOOT_SEED)
    terms = stepA.admission_terms(P)
    kinds = np.asarray(P["kind"])
    out = {"n_pairs": res["n_pairs"], "all": res["all"], "lon": res["lon"], "lat": res["lat"], "tactical": res["tactical"], "strategic": res["strategic"],
           "stamp": "SEEN-SCENE, SELECTION tier; ONE training seed per cell; the CI answers 'another draw of LOGS' only (H-ESTIM-SEED-1)"}
    # LONGITUDINAL: speed-scale-wise A1
    ps = {}
    for s in stepA.SPEED_SCALES:
        m = (kinds == stepA.KIND_LON) & np.isclose(D.param.reshape(-1), s)
        sums = stepA.per_log_sums({"A1": terms["A1"]}, D.pair_log, D.n_logs, m)["A1"]
        ps[f"{s:g}"] = stepA.ratio_ci(sums[0], sums[1], D.boot_idx)
    out["lon_A1_by_speed_scale"] = ps
    # LATERAL: the +-0.3 m pairs' DAC concordance (the DAC head is frozen, so this is unchanged by the retrain; EP is out["lat"]["A1"])
    al, pl = np.asarray(P["a_logit"], np.float64), np.asarray(P["p_logit"], np.float64)
    asub, psub = np.asarray(P["a_sub"], np.float64), np.asarray(P["p_sub"], np.float64)
    dac = {"DAC_conc": stepA.concordance_counts(al[:, 1], pl[:, 1], asub[:, 1], psub[:, 1])}
    sums = stepA.per_log_sums(dac, D.pair_log, D.n_logs, kinds == stepA.KIND_LAT)["DAC_conc"]
    out["lat_DAC_concordance"] = stepA.ratio_ci(sums[0], sums[1], D.boot_idx)
    # in-fan top-8 EP concordance (the non-degradation term)
    num, den = infan_top8_terms(D, ep_in)
    nl, dl = np.bincount(D.log_idx, weights=num, minlength=D.n_logs), np.bincount(D.log_idx, weights=den, minlength=D.n_logs)
    out["infan_top8_ep_concordance"] = stepA.ratio_ci(nl, dl, D.boot_idx)
    out["_per_log"] = {"A1_all": [a.tolist() for a in stepA.per_log_sums({"A1": terms["A1"]}, D.pair_log, D.n_logs)["A1"]], "infan": [nl.tolist(), dl.tolist()]}
    if ref_terms is not None:
        out["A1_minus_P2_paired"] = paired_diff_ci(*out["_per_log"]["A1_all"], *ref_terms["A1_all"], D.boot_idx)
        out["infan_minus_P2_paired"] = paired_diff_ci(nl, dl, *ref_terms["infan"], D.boot_idx)
    return out


def summarise(ev: dict) -> dict:
    """the handful of numbers the outcome mapping reads"""
    return {"a1_point": ev["all"]["A1"]["point"], "a1_lo": ev["all"]["A1"]["lo"], "a1_hi": ev["all"]["A1"]["hi"], "a4_point": ev["all"]["A4"]["point"],
            "a5_point": ev["all"]["A5"]["point"], "infan": ev["infan_top8_ep_concordance"]["point"], "a1_lon": ev["lon"]["A1"]["point"], "a1_lat": ev["lat"]["A1"]["point"]}


def ref_terms_of(ev: dict) -> dict:
    return {"A1_all": [np.array(a) for a in ev["_per_log"]["A1_all"]], "infan": [np.array(a) for a in ev["_per_log"]["infan"]]}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
