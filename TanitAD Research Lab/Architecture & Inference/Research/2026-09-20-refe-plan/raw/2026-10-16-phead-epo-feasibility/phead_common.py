#!/usr/bin/env python3
"""PHEAD instrument (PREREG_PHEAD.md): shared machinery.  CPU ONLY.

REUSES the H-ADAM instrument (raw/2026-10-15-rls1-adam-dose/): the trainer's own closures extracted from the AST (adam_common.extract_closures), the scorer built by
adam_common.build_scorer, the gradient bank (grad_bank.py), torch.optim.AdamW and torch.nn.utils.clip_grad_norm_ themselves.  What is NEW here:

  * `group_slices`  -- the parameter groups ("which parameters may EPO move"), as (tensor index, rows) slices of the scorer's 78 tensors.
  * `Restricted`    -- the restricted optimiser: ONLY the group's parameters are handed to AdamW and to clip_grad_norm_, as a trainer that freezes the rest would
                       (requires_grad False => not in `paramsB`, scorer_finetune.py:1360-1362; clip over `paramsB`, :1391).  Weight decay therefore never touches a frozen entry.
  * `Evaluator`     -- J and the pick / policy sub-score proxies of a parameter state on a pool (per set, so a log-cluster bootstrap can be drawn).
  * `boot_ratio`    -- the paired log-cluster bootstrap (10,000 resamples, seed 20260927 = the programme's) of mean differences and of the ratio f.
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AD = os.path.normpath(os.path.join(HERE, "..", "2026-10-15-rls1-adam-dose"))
sys.path.insert(0, AD)
import adam_common as AC  # noqa: E402

SCR = AC.SCR
PH_SCR = f"{SCR}/phead"
os.makedirs(PH_SCR, exist_ok=True)
BANK_DIR = f"{SCR}/bank"
BOOT_N, BOOT_SEED = 10000, 20260927
COMP = ("nc", "dac", "ep", "ttc", "c", "ddc")            # train.py:502 PDM_ORDER = score_head rows 0..5 (model.py:563); row 6 = the lane output (scorer_finetune --lane-head)
EP_ROW = 2


# --------------------------------------------------------------------------------------------------------------------------- parameter groups
def group_slices(gname, names, shapes):
    """-> [(tensor_index, rows|None)] for a group.  rows None = the whole tensor; a list = those rows of dim 0 (score_head.weight [7,256] / bias [7])."""
    ix = {n: i for i, n in enumerate(names)}
    allt = [(i, None) for i in range(len(names))]
    head = [(ix["score_head.weight"], None), (ix["score_head.bias"], None)]
    def head_rows(rows):
        return [(ix["score_head.weight"], list(rows)), (ix["score_head.bias"], list(rows))]
    def blocks(ks):
        return [(i, None) for i, n in enumerate(names) if n.startswith("score_dec.") and int(n.split(".")[1]) in ks]
    if gname == "FULL":
        return allt
    if gname == "EMPTY":
        return []
    if gname == "PH_A_EPROW":              # the progress (EP) output's own parameters: row 2 of score_head
        return head_rows([EP_ROW])
    if gname == "PH_B_SIX":                # the six PDM component outputs (rows 0..5), without the lane row
        return head_rows(range(6))
    if gname == "PH_C_HEAD":               # the whole score_head, lane row included
        return head
    if gname == "LAD_1":                   # RUNG (not a candidate for 'progress heads'): head + the LAST scoring-decoder block
        return head + blocks({3})
    if gname == "LAD_2":                   # RUNG: head + all four scoring-decoder blocks (everything but score_q_mlp)
        return head + blocks({0, 1, 2, 3})
    raise KeyError(gname)


def group_numel(sl, shapes):
    n = 0
    for i, rows in sl:
        sh = shapes[i]
        per_row = int(np.prod(sh[1:])) if len(sh) > 1 else 1
        n += (len(rows) * per_row) if rows is not None else int(np.prod(sh))
    return n


def flat_mask(sl, shapes):
    """bool vector over the bank's flat layout: True = the entry may move."""
    offs, o = [], 0
    for sh in shapes:
        offs.append(o)
        o += int(np.prod(sh))
    m = torch.zeros(o, dtype=torch.bool)
    for i, rows in sl:
        sh = shapes[i]
        n = int(np.prod(sh))
        if rows is None:
            m[offs[i]:offs[i] + n] = True
        else:
            per_row = int(np.prod(sh[1:])) if len(sh) > 1 else 1
            for r in rows:
                m[offs[i] + r * per_row: offs[i] + (r + 1) * per_row] = True
    return m


class Restricted:
    """AdamW + clip over ONLY the group's entries.  Each slice becomes one Parameter (a whole tensor, or the listed rows of one); theta() returns the full per-tensor state.
    PHEAD_MUTATE_NO_RESTRICT=1 is the deliberate regression of control (c): the line that applies the restriction is skipped, so every tensor is trainable."""

    def __init__(self, theta0, sl, lr, wd=0.01, n_tensors=None):
        self.theta0 = [t.clone() for t in theta0]
        if os.environ.get("PHEAD_MUTATE_NO_RESTRICT") == "1":
            sl = [(i, None) for i in range(len(theta0))]                         # <- MUTATION: the restriction is not applied
        self.sl = sl                                                             # <- THE LINE THAT APPLIES THE RESTRICTION (the group's slices are the only params)
        self.params = [torch.nn.Parameter((self.theta0[i][rows] if rows is not None else self.theta0[i]).clone()) for i, rows in sl]
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=wd) if self.params else None

    def step(self, grads, lr):
        """grads = per-tensor full-size gradient list.  clip_grad_norm_(group params, 1.0) -> set lr -> AdamW step (the trainer's order, :1391-1395)."""
        if not self.params:
            return 0.0
        for p, (i, rows) in zip(self.params, self.sl):
            p.grad = (grads[i][rows] if rows is not None else grads[i]).clone()
        n = float(torch.nn.utils.clip_grad_norm_(self.params, 1.0))
        for gr in self.opt.param_groups:
            gr["lr"] = lr
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return n

    def theta(self):
        out = [t.clone() for t in self.theta0]
        for p, (i, rows) in zip(self.params, self.sl):
            if rows is None:
                out[i] = p.detach().clone()
            else:
                out[i][rows] = p.detach()
        return out


def outside_group_displacement(theta, theta0, sl):
    """max |theta - theta0| over every entry NOT in the group (must be exactly 0.0 for a restricted arm)."""
    inside = {}
    for i, rows in sl:
        inside.setdefault(i, set())
        inside[i] = None if (rows is None or inside[i] is None) else inside[i] | set(rows)
    worst = 0.0
    for i, (t, t0) in enumerate(zip(theta, theta0)):
        d = (t - t0).abs()
        if i in inside:
            if inside[i] is None:
                continue
            keep = torch.ones(d.shape[0], dtype=torch.bool)
            keep[list(inside[i])] = False
            d = d[keep]
        if d.numel():
            worst = max(worst, float(d.max()))
    return worst


def unflat_t(v, shapes):
    out, o = [], 0
    for sh in shapes:
        n = int(np.prod(sh))
        out.append(v[o:o + n].view(sh))
        o += n
    return out


# --------------------------------------------------------------------------------------------------------------------------- pools and the evaluator
def get_pool(SF, a, cfg, side):
    """the real OnPolicyBank pool of adam_common.prepare_pool, pickled once so parallel processes never race on its scratch files."""
    p = f"{PH_SCR}/pool_{side}.pkl"
    if os.path.exists(p):
        d = pickle.load(open(p, "rb"))
    else:
        d = AC.prepare_pool(SF, a, cfg, side)
        tmp = p + f".{os.getpid()}.tmp"
        pickle.dump(d, open(tmp, "wb"))
        os.replace(tmp, p)
    return AC.Pool(d).load_visual_all()


class Evaluator:
    """J (the trainer's EPO objective, exact over the 64 hypotheses) and sub-score proxies of a scorer state on one pool.  The functions are the trainer's own:
    SF.epo_policy / the extracted `sel_score` closure / SF.epo_reward / rl_reward.pdms_v1.  Per-set arrays are returned so a log-cluster bootstrap can be drawn."""

    def __init__(self, SF, a, tr, pool, name):
        self.SF, self.a, self.tr, self.pool, self.name = SF, a, tr, pool, name
        self.sel = tr.fn["sel_score"]
        rcfg, _ = AC.make_ns(SF, a)
        self.N = len(pool.keys)
        recs_all = [pool.terms[k] for k in pool.keys]
        self.R = torch.as_tensor(np.stack([SF.epo_reward(pool.ctg[i:i + 1, :, :6], [recs_all[i]], rcfg)[0] for i in range(self.N)]), dtype=torch.float32)       # [N, 64]
        self.tg6 = torch.from_numpy(pool.ctg[..., :6].astype(np.float32))                                                                                      # [N, 64, 6]
        self.logs = np.asarray([k[0] for k in pool.keys])
        self.batches = [list(range(i, min(i + 8, self.N))) for i in range(0, self.N, 8)]
        self.sx0 = None

    def _forward(self, arm, ids):
        cxy, ctg, sctx, _ = self.pool.micro(ids)
        with torch.no_grad():
            return arm(cxy, sctx).float()

    def set_theta(self, theta):
        with torch.no_grad():
            for p, t in zip(self.tr.armB.parameters(), theta):
                p.copy_(t)

    def evaluate(self, theta, max_sets=None):
        """-> dict of per-set numpy arrays (length N or max_sets) for the state `theta` (a per-tensor list)."""
        self.set_theta(theta)
        if self.sx0 is None:
            self.sx0 = torch.cat([self._forward(self.tr.base, ids) for ids in self.batches])
        out = {k: [] for k in ("J", "kl", "pi_comp", "pick_comp", "pick_R", "pick_pdms", "pick_idx", "pi_pdms")}
        for ids in self.batches:
            if max_sets and ids[0] >= max_sets:
                break
            sxB = self._forward(self.tr.armB, ids)
            logpi = self.SF.epo_policy(sxB, self.sel, self.a.epo_temp)
            logpi0 = self.SF.epo_policy(self.sx0[ids], self.sel, self.a.epo_temp)
            pi = logpi.exp()
            R, tg6 = self.R[ids], self.tg6[ids]
            J = (pi * R).sum(1)
            kl = (logpi0.exp() * (logpi0 - logpi)).sum(1)
            pick = self.sel(torch.sigmoid(sxB)).argmax(1)
            ar = torch.arange(len(ids))
            pdms_all = self.SF.agg_v1(tg6)
            out["J"].append(J)
            out["kl"].append(kl)
            out["pi_comp"].append((pi[..., None] * tg6).sum(1))
            out["pick_comp"].append(tg6[ar, pick])
            out["pick_R"].append(R[ar, pick])
            out["pick_pdms"].append(pdms_all[ar, pick])
            out["pick_idx"].append(pick.float())
            out["pi_pdms"].append((pi * pdms_all).sum(1))
        return {k: torch.cat(v).numpy().astype(np.float64) for k, v in out.items()}


# --------------------------------------------------------------------------------------------------------------------------- estimator
def _clusters(logs):
    u = {}
    for i, l in enumerate(logs):
        u.setdefault(l, []).append(i)
    return [np.asarray(v) for _k, v in sorted(u.items())]


def boot_ratio(logs, num, den=None, n=BOOT_N, seed=BOOT_SEED):
    """Paired log-cluster bootstrap on per-set arrays.  num, den: per-set difference arrays (arm - control).  Resamples the LOGS (clusters) with replacement.
    -> {mean_num, ci_num, [mean_den, ci_den, f, ci_f]}.  The ratio is a ratio of resampled means (same resample in numerator and denominator).
    Estimator, named: episode(log)-cluster bootstrap over the pool's logs, evaluation sets resampled, training state FIXED (answers 'another draw of EVALUATION sets', not 'another training run')."""
    cl = _clusters(logs)
    sn = np.asarray([num[c].sum() for c in cl])
    cnt = np.asarray([len(c) for c in cl], np.float64)
    sd = None if den is None else np.asarray([den[c].sum() for c in cl])
    rng = np.random.default_rng(seed)
    pk = rng.integers(0, len(cl), (n, len(cl)))
    cn = cnt[pk].sum(1)
    mn = sn[pk].sum(1) / cn
    res = {"mean_num": float(num.mean()), "ci_num": [float(np.percentile(mn, 2.5)), float(np.percentile(mn, 97.5))], "n_sets": int(len(num)), "n_logs": int(len(cl))}
    if den is not None:
        md = sd[pk].sum(1) / cn
        with np.errstate(divide="ignore", invalid="ignore"):
            f = mn / md
        f = f[np.isfinite(f)]
        res.update({"mean_den": float(den.mean()), "ci_den": [float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))],
                    "f": float(num.mean() / den.mean()) if den.mean() != 0 else float("nan"),
                    "ci_f": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))], "den_straddles_zero": bool(np.percentile(md, 2.5) <= 0 <= np.percentile(md, 97.5))})
    return res


def setup(threads=0, need_val=True):
    """-> SF, a, holder, cfg, tr (train Trainer), pools {train, val}, evaluators {train296, val}"""
    if threads:
        torch.set_num_threads(threads)
    SF = AC.load_trainer()
    a, argv = AC.r0_args(SF)
    holder, cfg = AC.build_scorer(SF, a)
    pool_t = get_pool(SF, a, cfg, "train")
    tr = AC.Trainer(SF, a, holder, pool_t)
    ev = {"train": Evaluator(SF, a, tr, pool_t, "train")}
    if need_val:
        pool_v = get_pool(SF, a, cfg, "val")
        ev["val"] = Evaluator(SF, a, tr, pool_v, "val")
    return SF, a, holder, cfg, tr, ev


def theta0_of(tr):
    return [p.detach().clone() for p in tr.params]
