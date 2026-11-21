#!/usr/bin/env python3
"""SFT-1: scorer-only fine-tune of REFe's final model on its on-policy sets -- two arms in ONE pass over the data.

WHY SCORER-ONLY: the scoring decoder reads the trajectories AND the visual context DETACHED (model.py:746, :751), so
the score loss reaches only `score_q_mlp`, `score_dec`, `score_head`. Freezing everything else changes nothing the
proposals depend on: the 64 hypotheses stay bit-identical, only the CHOICE among them moves. (The navtest evidence:
983 of 12,146 picks score 0, and on W3's 200 tokens 14 of 15 zero picks had a hypothesis scoring >= 80 -- selection,
not proposal, failures; raw/2026-10-04-navtest-final.)

ARMS (one trunk forward per batch; both scorers read the SAME captured context, so the comparison is paired by
construction):
  base   a frozen copy of the final scorer (the reference; must equal A and B exactly at step 0)
  A      the model's own scorer modules, trained with the TRAINING loss (per-component BCE on each labelled set) --
         the control: is it just more training at a low lr?
  B      a copy trained with that BCE + lam * a LISTWISE ranking loss over each 64-hypothesis set (ListNet: the
         predicted distribution p_i ~ the planner's own navsim_v1 aggregate of sigmoid scores, the target
         q_i ~ exp(true_v1_i / tau_t)) -- selection is a within-scene RANKING problem; BCE only calibrates.
DATA: train.py's own TargetBank + OnPolicyBank (the identical inputs and labels training used), restricted to samples
with a complete labelled set and to logs OUTSIDE the 24 held-out logs. HELD-OUT: the 3,137 v5-labelled sets of the
final model's own proposals (24 logs) -- selection metrics per arm: pick / random / best true navsim_v1, skill, picks
scoring 0, within-set AUC of NC and DAC, and the paired log-cluster bootstrap of pick(arm) - pick(base).
  python scorer_finetune.py --preflight ...      # the launch gate; prints ZZSFT_PREFLIGHT_PASS or _FAIL
  python scorer_finetune.py ...                  # the run (refuses without a preflight PASS file in --out)

RL-S1 (eval/PREREG_RL_S1.md, RL_POSTTRAIN_REVIEW 5.2) -- ALL DEFAULT OFF, the OFF path is bit-identical to the SFT-4 trainer (refe/test_rl_epo.py):
  --epo           arm B += lam_pg(t) * [ -J + beta * KL(pi_SFT || pi) ],  pi = softmax(log(A + eps) / T) under the BASE's own selection rule
                  (sel_score: navsim_v1 x the lane output), J = sum_i pi_i R_i EXACT over the 64 hypotheses, R from rl_reward (detached),
                  pi_SFT = the frozen base copy, lam_pg ramped 0 -> 1 over --epo-ramp updates.  Arm A stays the BCE-only control.
                  Requires --pdm-labels-train --require-pdm --b-mode listnet --lam 0 (arm B = BCE + EPO and nothing else) and either
                  --epo-terms-train (rl_terms_*.jsonl: D, K, lane-change inputs) or --epo-lam-d 0 NAMED; preflight adds gate G11.
  The checkpoint's own meta sizes the score head (a 7-output checkpoint such as SFT-4 arm B u400 loads and trains with --lane-head).
  --dump-scores DIR  OBSERVATIONAL, default off: after every held-out evaluation write the base / A / B scorers' per-proposal logits of every held-out set as
                  DIR/scores_<tag>_<arm>.npz -- the input of eval/rl_val_eval.py (log, token, step, rank, ckpt_step, scores [N, M, C], space="logit").
                  It changes no weight, no loss and no logged number (refe/test_dump_scores.py: the OFF path is AST-identical to the pre-edit evaluate()).
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt_io  # noqa: E402
import rl_reward  # noqa: E402  (RL-S1: pure numpy; the reward the EPO term maximises)
import train as T  # noqa: E402
from model import REFe, REFeConfig  # noqa: E402

SCORER = ("score_q_mlp", "score_dec", "score_head")
V1_W = (5.0, 5.0, 2.0)


def agg_v1(p):
    """planner.aggregate, rule navsim_v1, on [..., 6] probabilities OR labels: NC*DAC*(5EP+5TTC+2C)/12"""
    return p[..., 0] * p[..., 1] * (V1_W[0] * p[..., 2] + V1_W[1] * p[..., 3] + V1_W[2] * p[..., 4]) / sum(V1_W)


class ScorerCopy(nn.Module):
    """A detached copy of the three scorer modules; forward = REFe.score_trajectories itself (no re-implementation)."""

    def __init__(self, model):
        super().__init__()
        for m in SCORER:
            setattr(self, m, copy.deepcopy(getattr(model, m)))

    def forward(self, trajs, sctx):
        return REFe.score_trajectories(self, trajs, sctx)


def expected_reward_loss(sx, tg, tau_s, sel=None):
    """-E_{i ~ pi}[R_i], pi = softmax(log agg_v1(sigmoid(sx)) / tau_s) over each set, R_i = agg_v1(labels) (the true score
    of hypothesis i). A one-step choice whose outcome is KNOWN for every action: this is the policy-gradient objective in
    closed form -- no sampled action, so no sampling variance. As tau_s -> 0, pi -> the planner's argmax."""
    p = torch.sigmoid(sx)
    s = torch.log((agg_v1(p) if sel is None else sel(p)).clamp_min(1e-6))
    pi = torch.softmax(s / tau_s, dim=1)
    return -(pi * agg_v1(tg[..., :6])).sum(1).mean()


def key(r):
    return (r.get("log_name", ""), r.get("token", ""), int(r.get("step", 0)), int(r.get("rank", 0)))


def set_auc(pred, lab):
    """within-set AUC of `pred` ranking PASS (lab >= 0.5) above FAIL; None if the set has one class only"""
    pos, neg = pred[lab >= 0.5], pred[lab < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean()))


def boot_logs(diff: dict, logs: dict, n=10000, seed=20260927):
    toks = sorted(diff)
    by: dict = {}
    for i, t in enumerate(toks):
        by.setdefault(logs[t], []).append(i)
    groups = [np.asarray(v) for _k, v in sorted(by.items())]
    vals = np.asarray([diff[t] for t in toks], np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        means[b] = vals[idx].mean()
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ================================================================================================ RL-S1: the per-proposal score dump (default OFF)
class ScoreDump:
    """The held-out evaluation's per-proposal scores, in the format `eval/rl_val_eval.py` reads (`write_scores_npz`): arrays log, token (str), step, rank,
    ckpt_step (int64), scores float64 [N, M, C], scalar space = "logit".  Row n of `scores` belongs to key n, proposal m of row n is the m-th proposal of THAT set
    in the order the set stores it (the order the PDM labels use).  The evaluator pairs arms and joins labels by (key, ckpt_step), so a row that is not keyed to
    its own set would silently score another set's labels."""
    ARMS = ("base", "A", "B")

    def __init__(self):
        self.keys: list = []
        self.ckpts: list = []
        self.chunks = {arm: [] for arm in self.ARMS}

    def add(self, keys, ckpts, sx0, sxA, sxB):
        n = len(keys)
        if len(ckpts) != n:
            raise ValueError(f"{n} keys but {len(ckpts)} ckpt steps")
        for arm, sx in zip(self.ARMS, (sx0, sxA, sxB)):
            if sx.shape[0] != n:
                raise ValueError(f"arm {arm}: {sx.shape[0]} score rows for {n} keys")
            self.chunks[arm].append(sx.detach().float().cpu().numpy().astype(np.float64))
        self.keys += [tuple(k) for k in keys]
        self.ckpts += [int(c) for c in ckpts]

    def write(self, dir_, tag):
        """-> {arm: path}.  One file per arm: scores_<tag>_<arm>.npz (the tag is made file-name safe)."""
        os.makedirs(dir_, exist_ok=True)
        safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(tag))
        out = {}
        for arm in self.ARMS:
            path = os.path.join(dir_, f"scores_{safe}_{arm}.npz")
            np.savez(path, log=np.asarray([k[0] for k in self.keys]), token=np.asarray([k[1] for k in self.keys]),
                     step=np.asarray([k[2] for k in self.keys], np.int64), rank=np.asarray([k[3] for k in self.keys], np.int64),
                     ckpt_step=np.asarray(self.ckpts, np.int64), scores=np.concatenate(self.chunks[arm], 0), space=np.asarray("logit"))
            out[arm] = path
        return out


# ================================================================================================ RL-S1: the EPO scorer loss (all default OFF)
# eval/PREREG_RL_S1.md, RL_POSTTRAIN_REVIEW.md 5.2.  Arm B only; arm A stays the BCE-only control (= C), so A / B are paired by construction.
EPO_EPS = 1e-6                    # log(A + eps): the gradient stays defined as A -> 0 (a clamp would zero it exactly where a proposal needs raising)
EPO_DESTS = ("epo_temp", "epo_beta", "epo_ramp", "epo_lam", "epo_lam_d", "epo_w_prog", "epo_k", "epo_lam_k", "epo_l", "epo_lam_l",
             "epo_terms_train", "epo_terms_heldout")          # every EPO lever except the switch itself (named without --epo = refused)
RL_TERMS_GLOB = "rl_terms_*.jsonl"


def epo_ramp(upd, ramp):
    """lam_pg(t) ramp factor, review 5.2: 0 -> 1 linearly over `ramp` optimizer updates (update 0 trains BCE only); ramp <= 0 = no ramp"""
    return 1.0 if ramp <= 0 else min(1.0, upd / ramp)


def epo_policy(sx, rule_fn, T):
    """log pi_i = log_softmax_i( log(A_i + eps) / T ) with A = rule_fn(sigmoid(sx)): the selection policy under the BASE's OWN selection rule
    (`rule_fn` is main()'s sel_score = navsim_v1 x the lane output -- what u400 deploys -- never train._pdm_aggregate)"""
    return torch.log_softmax(torch.log(rule_fn(torch.sigmoid(sx)) + EPO_EPS) / T, dim=1)


def epo_loss(sxB, sx_ref, R, rule_fn, T, beta, lam_pg, parts=None):
    """lam_pg * mean_over_scenes[ -J + beta * KL(pi_SFT || pi) ],  J = sum_i pi_i R_i   (review 5.2) -- EXACT over the 64 hypotheses of a scene, no sampling.
    sxB [b, M, C] = arm B's logits (carries the gradient); sx_ref = the frozen base's logits (pi_SFT, no gradient); R [b, M] = the reward (rl_reward),
    DETACHED here; rule_fn maps probabilities [b, M, C] to the selection score [b, M].  `parts`, if given, receives the batch means J and kl."""
    if os.environ.get("REFE_RL_MUTATE_EPO_DETACH") == "1":          # deliberate regression for the gate (G11): the policy loses its graph
        sxB = sxB.detach()
    logpi = epo_policy(sxB, rule_fn, T)
    with torch.no_grad():
        logpi0 = epo_policy(sx_ref, rule_fn, T)
    J = (logpi.exp() * R.detach()).sum(1)
    kl = (logpi0.exp() * (logpi0 - logpi)).sum(1)
    if parts is not None:
        parts["J"], parts["kl"] = float(J.detach().mean()), float(kl.detach().mean())
    return lam_pg * (-J + beta * kl).mean()


def epo_reward(tg6, recs, rcfg):
    """R [b, M] (float64, numpy) for a micro-batch: rl_reward.reward on the six PDM sub-scores tg6 [b, M, 6] plus the per-set side terms in `recs`
    (one load_rl_terms record per set; None when the config needs none).  A term the config turns ON whose input is missing RAISES."""
    d = k = lane = None
    need = rcfg.lam_d != 0.0 or rcfg.use_k or rcfg.use_l
    if need and (recs is None or any(r is None for r in recs)):
        raise ValueError("EPO reward: a set in this batch has no rl_terms record (the training index must be restricted to sets that have one)")
    if rcfg.lam_d != 0.0:
        d = np.stack([r["d"] for r in recs])
    if rcfg.use_k:
        k = np.stack([r["k"] for r in recs])
    if rcfg.use_l:
        lane = {"lc_executed": np.stack([r["lc_executed"] for r in recs]).astype(bool),
                "label_fires": np.asarray([r["label_fires"] for r in recs], bool),
                "dither": np.stack([r["dither"] for r in recs])}
    return rl_reward.reward(tg6, rcfg, d=d, k=k, lane=lane)["R"]


def _rl_terms_record(d, n_prop, need_d, need_k, need_l):
    """one validated rl_terms line -> {'d': [M], 'k': [M], 'lc_executed': [M] bool, 'label_fires': bool, 'dither': [M]} (only what is needed), or None"""
    rec = {}
    try:
        if need_d:
            x = np.asarray(d["d"], np.float32)
            if x.shape != (n_prop,) or not np.isfinite(x).all() or x.min() < 0.0 or x.max() > 1.0 + 1e-6:
                return None
            rec["d"] = x
        if need_k:
            x = np.asarray(d["k"], np.float32)
            if x.shape != (n_prop,) or not np.isfinite(x).all() or x.min() < 0.0:
                return None
            rec["k"] = x
        if need_l:
            lc, dith = np.asarray(d["lc_executed"], bool), np.asarray(d["dither"], np.float32)
            if lc.shape != (n_prop,) or dith.shape != (n_prop,) or not np.isfinite(dith).all() or not isinstance(d["label_fires"], bool):
                return None
            rec["lc_executed"], rec["dither"], rec["label_fires"] = lc, dith, bool(d["label_fires"])
    except (KeyError, ValueError, TypeError):
        return None
    return rec


def load_rl_terms(path, bank_by, n_prop, need_d, need_k, need_l):
    """The part-2 labeller's side files: `<path>/rl_terms_*.jsonl`, one line per set,
    {"kind":"rl_terms","key":[log,token,step,rank],"ckpt_step":c,"d":[M],"k":[M],"lc_executed":[M],"label_fires":bool,"dither":[M]}.
    Staleness is OnPolicyBank's own rule: a line counts only for the set it was computed on (same key AND ckpt_step).
    Returns ({key: record}, stats); a set with no usable record is simply absent (the caller restricts the training index to the present ones)."""
    terms: dict = {}
    st = {"files": 0, "lines": 0, "matched": 0, "stale": 0, "superseded": 0, "incomplete": 0, "not_in_bank": 0, "bad_lines": 0, "duplicates": 0}
    for fp in sorted(glob.glob(os.path.join(path, RL_TERMS_GLOB))):
        st["files"] += 1
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    st["bad_lines"] += 1
                    continue
                if not isinstance(d, dict) or d.get("kind") != "rl_terms":
                    continue
                st["lines"] += 1
                k = tuple(d.get("key", ()))
                if len(k) != 4:
                    st["incomplete"] += 1
                    continue
                k = (k[0], k[1], int(k[2]), int(k[3]))
                e = bank_by.get(k)
                if e is None:
                    st["not_in_bank"] += 1
                    continue
                ck = int(d.get("ckpt_step", -1))
                if ck != int(e[0]):
                    st["superseded" if ck < int(e[0]) else "stale"] += 1
                    continue
                rec = _rl_terms_record(d, n_prop, need_d, need_k, need_l)
                if rec is None:
                    st["incomplete"] += 1
                    continue
                if k in terms:
                    st["duplicates"] += 1
                terms[k] = rec
    st["matched"] = len(terms)
    return terms, st


def epo_refusals(a, ap):
    """Every reason this command line must not start, as messages ([] = fine).  `ap` is build_parser() (its defaults say which levers were NAMED)."""
    bad = []
    named = [d for d in EPO_DESTS if getattr(a, d) != ap.get_default(d)]
    if not a.epo:
        if named:
            bad.append(f"EPO levers {named} are set but --epo is not: they would be silently ignored")
        return bad
    if not a.epo_terms_train and a.epo_lam_d != 0.0:
        bad.append("--epo needs --epo-terms-train (the drivable-area excess D of every set) unless --epo-lam-d 0 is NAMED")
    if (a.epo_k or a.epo_l) and not a.epo_terms_train:
        bad.append("--epo-k / --epo-l need --epo-terms-train (their per-proposal inputs live in the side files)")
    if not a.pdm_labels_train or not a.require_pdm:
        bad.append("--epo needs --pdm-labels-train and --require-pdm: R is computed from the PDM sub-scores, and no set may mix label sources")
    if a.b_mode != "listnet" or a.lam != 0.0:
        bad.append("--epo needs arm B's other loss to be plain BCE (--b-mode listnet --lam 0): R vs C must differ by the EPO term alone")
    if a.epo_temp <= 0.0:
        bad.append("--epo-temp must be > 0")
    for nm in ("epo_beta", "epo_ramp", "epo_lam", "epo_lam_d", "epo_w_prog", "epo_lam_k", "epo_lam_l"):
        if getattr(a, nm) < 0.0:
            bad.append(f"--{nm.replace('_', '-')} must be >= 0")
    return bad


class IndexedSubset(Subset):
    """A Subset whose samples also carry their DATASET ROW INDEX as a trailing element.  RL-S1 looks the D / K / L side files up by row key, and
    the stock batch carries none (TargetBank returns the per-proposal ckpt_step in that slot).  DataLoader's fetcher calls __getitems__, which
    Subset implements by reaching past __getitem__ -- so BOTH are overridden here."""

    def __getitem__(self, idx):
        return tuple(super().__getitem__(idx)) + (int(self.indices[idx]),)

    def __getitems__(self, indices):
        return [self[i] for i in indices]


def make_train_set(ds, idx, epo):
    """the training subset: the stock Subset unless --epo is on (then it also yields the row index)"""
    return IndexedSubset(ds, idx) if epo else Subset(ds, idx)


def split_rows(batch, epo):
    """(the 9-tuple fwd() expects, the row-index tensor or None); a no-op unless --epo is on"""
    if not epo:
        return batch, None
    return batch[:9], batch[9]


def module_grad_abs(arm, names=SCORER):
    """sum of |grad| per scorer module of `arm` (a parameter with no gradient reads 0.0)"""
    return {m: float(sum((p.grad.detach().abs().sum().double() if p.grad is not None else torch.zeros((), dtype=torch.float64))
                         for p in getattr(arm, m).parameters())) for m in names}


def g11_epo_live_and_identity(arm, probe, kl_tol, names=SCORER):
    """G11 (RL-S1 preflight).  probe(mode) -> (the EPO term alone from a FRESH forward, info) for mode 'on' (weight 1.0), 'lam0' (the term built with
    weight 0.0: graph live, factor 0) and 'off' (--epo off).  An EPO-ONLY backward must give a finite non-zero gradient on EVERY scorer module of
    arm B ('on'); the two controls must read EXACTLY 0.0 (they prove the probe can read zero at all -- a zero from a broken probe looks the same);
    and at step 0 the KL anchor must be ~0 (arm B == the base, so pi == pi_SFT)."""
    reads, vals, infos = {}, {}, {}
    for mode in ("on", "lam0", "off"):
        arm.zero_grad(set_to_none=True)
        term, info = probe(mode)
        vals[mode], infos[mode] = float(term.detach()), info
        if term.requires_grad:
            term.backward()
        reads[mode] = module_grad_abs(arm, names)
    arm.zero_grad(set_to_none=True)
    live = {m: bool(math.isfinite(reads["on"][m]) and reads["on"][m] > 0.0) for m in names}
    zero_ok = all(v == 0.0 for r in ("lam0", "off") for v in reads[r].values()) and vals["lam0"] == 0.0 and vals["off"] == 0.0
    kl0 = infos["on"].get("kl")
    kl_ok = kl0 is not None and math.isfinite(kl0) and abs(kl0) <= kl_tol
    return {"ok": bool(all(live.values()) and zero_ok and kl_ok), "per_module_on": reads["on"], "per_module_lam0": reads["lam0"],
            "per_module_off": reads["off"], "live": live, "controls_exactly_zero": bool(zero_ok), "value_on": vals["on"], "J_on": infos["on"].get("J"),
            "kl_step0": kl0, "kl_tol": kl_tol, "kl_ok": bool(kl_ok), "mutate_detach": os.environ.get("REFE_RL_MUTATE_EPO_DETACH") == "1"}


def extend_lane_head(model, dev):
    """SFT-4: grow the 6-row score head to 7 (the teacher-lane output): rows 0-5 copied, row 6 zero weight and bias 1.5.
    RL-S1: a head that ALREADY has 7 rows (u400, whose 7th row is trained) is left EXACTLY as it is -- re-extending it would copy
    7 rows into 6 and overwrite the trained lane output with the initialisation."""
    old = model.score_head
    if int(old.out_features) == 7:
        return old
    new = nn.Linear(old.in_features, 7).to(dev)
    with torch.no_grad():
        new.weight.zero_(); new.bias.zero_()
        new.weight[:6].copy_(old.weight); new.bias[:6].copy_(old.bias)
        new.bias[6] = 1.5                                   # ~0.82: the mean of the soft lane target on held-out
        if os.environ.get("REFE_SFT4_MUTATE_G3") == "1":   # deliberate regression: a mis-copied row 0-5 must turn G3 RED
            new.weight[2, 0] += 1e-3
    model.score_head = new
    for p in model.score_head.parameters():
        p.requires_grad_(True)
    return new


def build_parser() -> argparse.ArgumentParser:
    """The command line, extracted VERBATIM from main() (RL-S1) so a test and the launch gate can read the flags without running a job."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--onpolicy", required=True, help="the training on-policy sets dir")
    ap.add_argument("--heldout", required=True, help="the held-out on-policy sets dir")
    ap.add_argument("--heldout-logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.3, help="arm B: ListNet weight")
    ap.add_argument("--tau-t", type=float, default=0.1, help="arm B: target temperature on the true navsim_v1 score")
    ap.add_argument("--eval-every", type=int, default=400, help="optimizer updates between held-out evals")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--max-updates", type=int, default=0)
    # SFT-2 (LANE-1 follow-on); all default OFF = SFT-1 exactly
    ap.add_argument("--lane-labels-train", default="", help="side files of onpolicy_relabel_lane.py for --onpolicy")
    ap.add_argument("--lane-labels-heldout", default="", help="side files of onpolicy_relabel_lane.py for --heldout")
    ap.add_argument("--b-mode", default="listnet", choices=("listnet", "compw", "expreward"),
                    help="arm B: BCE + lam*ListNet (SFT-1) or per-component weighted BCE (--b-compw)")
    ap.add_argument("--b-compw", default="1,1,1,1,1,1", help="arm B component weights NC,DAC,EP,TTC,C,DDC (b-mode compw)")
    # SFT-4 (eval/PREREG_SFT4.md): the paper's PDM targets + the teacher's centre-line distance as a 7th output; default OFF
    ap.add_argument("--pdm-labels-train", default="", help="side files of onpolicy_relabel_pdm.py for --onpolicy")
    ap.add_argument("--pdm-labels-heldout", default="", help="side files of onpolicy_relabel_pdm.py for --heldout")
    ap.add_argument("--require-pdm", action="store_true", help="train and evaluate ONLY on sets that carry PDM targets")
    ap.add_argument("--teacher-lane-train", default="", help="side files of onpolicy_relabel_teacher_lane.py (train)")
    ap.add_argument("--teacher-lane-heldout", default="", help="side files of onpolicy_relabel_teacher_lane.py (held-out)")
    ap.add_argument("--lane-head", action="store_true", help="a 7th scorer output trained on the teacher lane label")
    ap.add_argument("--lane-weight", type=float, default=1.0)
    ap.add_argument("--tau-s", type=float, default=0.1,
                    help="b-mode expreward: temperature of the selection policy softmax(log agg_v1(p) / tau_s)")
    # RL-S1 (eval/PREREG_RL_S1.md, RL_POSTTRAIN_REVIEW 5.2): the EPO scorer loss on arm B; all default OFF = SFT-4 exactly
    ap.add_argument("--epo", action="store_true", help="arm B += lam_pg(t) * [-J + beta*KL(pi_SFT||pi)], J = E_pi[R] exact over the 64 proposals")
    ap.add_argument("--epo-temp", type=float, default=0.1, help="T of the selection policy pi = softmax(log(A + eps) / T), A = the BASE's own rule")
    ap.add_argument("--epo-beta", type=float, default=0.1, help="weight of the KL anchor KL(pi_SFT || pi) (pi_SFT = the frozen base)")
    ap.add_argument("--epo-ramp", type=int, default=200, help="optimizer updates over which lam_pg ramps 0 -> 1 (0 = no ramp)")
    ap.add_argument("--epo-lam", type=float, default=1.0, help="final weight lam_pg of the EPO term (0 reproduces arm A exactly)")
    ap.add_argument("--epo-lam-d", type=float, default=0.15, help="reward: drivable-area excess weight lam_d (0 must be NAMED to run without --epo-terms-train)")
    ap.add_argument("--epo-w-prog", type=float, default=0.5, help="reward: progress-guard weight (capped by rl_reward.guard_penalty)")
    ap.add_argument("--epo-k", action="store_true", help="reward += -lam_k*K (lane consistency; the R2 arm; needs --epo-terms-train)")
    ap.add_argument("--epo-lam-k", type=float, default=0.05)
    ap.add_argument("--epo-l", action="store_true", help="reward += +lam_l*L (lane-change shaping; the R2 arm; needs --epo-terms-train)")
    ap.add_argument("--epo-lam-l", type=float, default=0.05)
    ap.add_argument("--epo-terms-train", default="", help="dir of rl_terms_*.jsonl side files (D, K, lane-change inputs) for --onpolicy")
    ap.add_argument("--epo-terms-heldout", default="", help="dir of rl_terms_*.jsonl side files for --heldout (validated + logged; consumed by the RL-val lookup)")
    ap.add_argument("--dump-scores", default="", help="RL-S1: DIR; after every held-out evaluation write the base / A / B scorers' per-proposal logits as "
                    "<DIR>/scores_<tag>_<arm>.npz (eval/rl_val_eval.py's input).  Observational: no weight, loss or logged number changes.  '' = nothing is written")
    return ap


def main() -> int:
    ap = build_parser()
    a = ap.parse_args()
    bad_cli = epo_refusals(a, ap)                            # RL-S1: a command line that names EPO levers inconsistently never starts
    if bad_cli:
        for msg in bad_cli:
            print("REFUSED: " + msg)
        return 4
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(a.seed)
    log_p = os.path.join(a.out, "preflight.jsonl" if a.preflight else "sft.jsonl")

    def log(row):
        row["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(log_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    if not a.preflight and not os.path.exists(os.path.join(a.out, "PREFLIGHT_PASS.json")):
        print("REFUSED: no PREFLIGHT_PASS.json in --out -- run --preflight first (the launch gate)"); return 4

    # ---------------- model: built and loaded EXACTLY as the planner does
    # RL-S1: the config is sized from the checkpoint's own meta (ckpt_io.config_for_checkpoint, as planner.py does), so a 7-output
    # checkpoint (u400 = SFT-4 arm B) builds a 7-row head and loads STRICTLY; a checkpoint without that meta is sized as before.
    cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(a.backbone), a.ckpt)
    model = REFe(cfg).to(dev)
    meta: dict = {}
    fmt = ckpt_io.load_for_inference(model, a.ckpt, map_location=dev, backbone=a.backbone, meta_out=meta)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in SCORER:
        for p in getattr(model, m).parameters():
            p.requires_grad_(True)
    if os.environ.get("REFE_SFT_MUTATE_UNFREEZE") == "1":            # deliberate regression for the gate
        for p in model.traj_head.parameters():
            p.requires_grad_(True)
    native7 = int(model.score_head.out_features) == 7       # RL-S1: u400 ALREADY carries the trained 7th (lane) output
    if native7 and not a.lane_head:
        print("REFUSED: the checkpoint's score head has 7 outputs (the lane output) -- pass --lane-head and the teacher-lane labels"); return 4
    base = ScorerCopy(model).to(dev)
    for p in base.parameters():
        p.requires_grad_(False)
    if a.lane_head:
        extend_lane_head(model, dev)
    armB = ScorerCopy(model).to(dev)
    for p in armB.parameters():
        p.requires_grad_(True)
    cap: dict = {}
    model.score_dec[0].register_forward_pre_hook(lambda mod, args: cap.__setitem__("sctx", args[1]))

    # ---------------- data: train.py's own banks
    op = T.OnPolicyBank(a.onpolicy, cfg.n_proposals, cfg.horizon_steps, lane_labels=a.lane_labels_train or None,
                        pdm_labels=a.pdm_labels_train or None, teacher_lane_labels=a.teacher_lane_train or None)
    ho = T.OnPolicyBank(a.heldout, cfg.n_proposals, cfg.horizon_steps, lane_labels=a.lane_labels_heldout or None,
                        pdm_labels=a.pdm_labels_heldout or None, teacher_lane_labels=a.teacher_lane_heldout or None)
    if a.lane_head and (op.n_cols != 7 or ho.n_cols != 7):
        print("REFUSED: --lane-head needs --teacher-lane-train AND --teacher-lane-heldout"); return 4
    compw = torch.tensor([float(x) for x in a.b_compw.split(",")], device=dev)
    assert compw.numel() == 6 and (compw > 0).all(), "--b-compw needs 6 positive weights"
    ds = T.TargetBank(a.targets, a.images, cfg, calib=a.calib, onpolicy=op)
    ho_logs = set(open(a.heldout_logs, encoding="utf-8").read().split())
    tr_idx = [i for i, r in enumerate(ds.rows) if key(r) in op.by and r.get("log_name") not in ho_logs
              and (not a.require_pdm or key(r) in op.pdm_keys)]
    ds_ho = copy.copy(ds)
    ds_ho.onpolicy = ho
    ho_idx = [i for i, r in enumerate(ds.rows) if key(r) in ho.by and (not a.require_pdm or key(r) in ho.pdm_keys)]
    # RL-S1: the reward config and the D / K / L side terms.  The training index is restricted to sets that HAVE a usable record, so the reward
    # of every set in a batch is computable; nothing below runs unless --epo is on (the OFF path is untouched).
    rcfg = terms_train = terms_ho = st_tr = st_ho = None
    if a.epo:
        rcfg = rl_reward.RewardCfg(lam_d=a.epo_lam_d, w_prog=a.epo_w_prog, guard=True, lam_k=a.epo_lam_k, use_k=a.epo_k,
                                   lam_l=a.epo_lam_l, use_l=a.epo_l)
        if a.epo_terms_train:
            terms_train, st_tr = load_rl_terms(a.epo_terms_train, op.by, cfg.n_proposals, a.epo_lam_d != 0.0, a.epo_k, a.epo_l)
            tr_idx = [i for i in tr_idx if key(ds.rows[i]) in terms_train]
        if a.epo_terms_heldout:
            terms_ho, st_ho = load_rl_terms(a.epo_terms_heldout, ho.by, cfg.n_proposals, a.epo_lam_d != 0.0, a.epo_k, a.epo_l)
    lv: dict = {}
    for k_ in (key(ds.rows[i]) for i in tr_idx):
        v = op.by[k_][3][1]
        lv[v] = lv.get(v, 0) + 1
    info = {"event": "data", "train_sets": len(tr_idx), "train_sets_in_bank": len(op.by),
            "train_label_versions": lv, "navsim_dac_sets": op.n_navsim_dac,
            "heldout_sets": len(ho_idx), "heldout_in_bank": len(ho.by), "heldout_logs": len(ho_logs),
            "train_rows_in_heldout_logs": sum(1 for i in tr_idx if ds.rows[i].get("log_name") in ho_logs),
            "ckpt_format": fmt, "per_sample_calib": meta.get("per_sample_calib"),
            "lane_labels": {"train_sets": op.n_lane, "train_stale": op.n_lane_stale, "train_superseded": op.n_lane_superseded,
                            "heldout_sets": ho.n_lane,
                            "pdm_train": op.n_pdm, "pdm_train_stale": op.n_pdm_stale, "pdm_train_superseded": op.n_pdm_superseded,
                            "pdm_heldout": ho.n_pdm, "teacher_lane_train": op.n_tlane, "teacher_lane_heldout": ho.n_tlane,
                            "heldout_stale": ho.n_lane_stale}, "b_mode": a.b_mode, "b_compw": a.b_compw}
    if a.epo:
        info["epo"] = {"temp": a.epo_temp, "beta": a.epo_beta, "ramp": a.epo_ramp, "lam": a.epo_lam, "lam_d": a.epo_lam_d, "w_prog": a.epo_w_prog,
                       "k": a.epo_k, "lam_k": a.epo_lam_k, "l": a.epo_l, "lam_l": a.epo_lam_l, "eps": EPO_EPS,
                       "terms_train": st_tr, "terms_heldout": st_ho, "native7": native7}
    log(info)
    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(make_train_set(ds, tr_idx, a.epo), batch_size=a.batch, shuffle=True, generator=g, num_workers=a.workers,
                    drop_last=True, pin_memory=True, persistent_workers=a.workers > 0,
                    prefetch_factor=(4 if a.workers > 0 else None))

    def fwd(batch):
        img, ego, goal, _tgt, cxy, ctg, cm, _ci, cal = batch
        img, ego, goal, cxy, ctg = (x.to(dev, non_blocking=True) for x in (img, ego, goal, cxy, ctg))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _traj, _score, sxA = model(img, ego, goal, calib=cal if cal.numel() else None, score_extra=cxy)
            sctx = cap["sctx"]
            sxB = armB(cxy, sctx)
            with torch.no_grad():
                sx0 = base(cxy, sctx)
        return sx0.float(), sxA.float(), sxB.float(), ctg.float()

    def lane_bce(sx, tg):
        if not a.lane_head:
            return torch.zeros((), device=sx.device)
        m = tg[..., 6] >= 0
        if not bool(m.any()):
            return torch.zeros((), device=sx.device)
        return F.binary_cross_entropy_with_logits(sx[..., 6][m], tg[..., 6][m])

    def sel_score(p):
        """the selection score: navsim_v1 over the six components, x the lane output when there is one"""
        return agg_v1(p[..., :6]) * (p[..., 6] if (a.lane_head and p.shape[-1] == 7) else 1.0)

    def losses(sxA, sxB, tg, lam):
        if a.lane_head or tg.shape[-1] == 7:
            la, lb = a.lane_weight * lane_bce(sxA, tg), a.lane_weight * lane_bce(sxB, tg)
            sxA6, sxB6, tg6 = sxA[..., :6], sxB[..., :6], tg[..., :6]
            bA, bB, ln = losses6(sxA6, sxB6, tg6, lam, sxB)
            return bA + la, bB + lb, ln
        return losses6(sxA, sxB, tg, lam, sxB)

    def losses6(sxA, sxB, tg, lam, sxB_full):
        bceA = F.binary_cross_entropy_with_logits(sxA, tg)
        if a.b_mode == "compw":                              # SFT-2: weighted per-component BCE, no ranking term
            w = compw.view(1, 1, 6)
            bceBw = (F.binary_cross_entropy_with_logits(sxB, tg, reduction="none") * w).sum(-1).mean() / w.sum()
            return bceA, bceBw, torch.zeros((), device=sxB.device)
        if a.b_mode == "expreward":                          # SFT-3: the selection's EXPECTED true score, all 64 outcomes known
            bceB = F.binary_cross_entropy_with_logits(sxB, tg)
            er = expected_reward_loss(sxB_full, tg, a.tau_s, sel=sel_score)
            return bceA, bceB + lam * er, er
        bceB = F.binary_cross_entropy_with_logits(sxB, tg)
        s = torch.log(agg_v1(torch.sigmoid(sxB)).clamp_min(1e-6))
        q = torch.softmax(agg_v1(tg) / a.tau_t, dim=1)
        listnet = -(q * torch.log_softmax(s, dim=1)).sum(1).mean()
        return bceA, bceB + lam * listnet, listnet

    epo_last: dict = {}                                      # J and KL of the latest EPO call (logging; G11 reads kl at step 0)

    def epo_term(sx0, sxB, tg, rows, upd, lam_pg=None):
        """RL-S1: arm B's EPO contribution for one micro-batch = epo_loss(policy under sel_score, the base as pi_SFT, R from rl_reward, DETACHED).
        lam_pg=None -> the schedule a.epo_lam * ramp(upd); a scheduled 0 returns a constant with no graph (update 0, or --epo-lam 0 = arm A exactly).
        An EXPLICIT lam_pg (the G11 probe) is never short-circuited, so a weight of 0.0 still builds the graph and its gradient must read exactly 0.0."""
        if lam_pg is None:
            lam_pg = a.epo_lam * epo_ramp(upd, a.epo_ramp)
            if lam_pg == 0.0:
                return torch.zeros((), device=sxB.device)
        recs = [terms_train.get(key(ds.rows[int(i)])) for i in rows] if terms_train is not None else None
        R = epo_reward(tg[..., :6].detach().float().cpu().numpy(), recs, rcfg)
        return epo_loss(sxB, sx0, torch.as_tensor(R, dtype=torch.float32, device=sxB.device), sel_score, a.epo_temp, a.epo_beta, lam_pg, epo_last)

    def train_losses(sx0, sxA, sxB, tg, rows, upd, lam_pg=None):
        """THE loss assembly of the run: the training loop, the speed probe and the G11 probe all call it.  losses() exactly as before, plus
        (--epo only) the EPO term on arm B ONLY -- arm A stays the BCE-only control.  Returns (lossA, lossB incl. EPO, listnet/er, the EPO term)."""
        lA, lB, ln = losses(sxA, sxB, tg, a.lam)
        ep = torch.zeros((), device=sxB.device)
        if a.epo:
            ep = epo_term(sx0, sxB, tg, rows, upd, lam_pg)
            lB = lB + ep
        return lA, lB, ln, ep

    def epo_probe(batch, rows, mode):
        """G11: the EPO term ALONE from a FRESH forward.  'on' = weight 1.0; 'lam0' = built with weight 0.0 (graph live, factor 0); 'off' = --epo off"""
        s0, sA_, sB_, tg_ = fwd(batch)
        keep = a.epo
        try:
            a.epo = mode != "off"
            ep = train_losses(s0, sA_, sB_, tg_, rows, 0, lam_pg=(1.0 if mode == "on" else 0.0))[3]
        finally:
            a.epo = keep
        return ep, (dict(epo_last) if mode != "off" else {})

    @torch.no_grad()
    def evaluate(tag, limit=None):
        idx = ho_idx[:limit] if limit else ho_idx
        dlh = DataLoader(Subset(ds_ho, idx), batch_size=16, shuffle=False, num_workers=2)
        res = {k: {"pick": [], "rand": [], "best": [], "zero": [], "auc_nc": [], "auc_dac": [], "auc_ddc": [],
                   "onc": [], "pick_x": [], "onc_x": [], "zero_x": [], "lk": [], "pick_l": [], "zero_l": [], "onc_l": [],
                   "lk_l": [], "auc_lane": []} for k in ("base", "A", "B")}
        lk_rows = [ho.lane_keep.get(key(ds.rows[i])) for i in idx]       # NAVSIM lk10 yardstick (only with --lane-labels-heldout)
        toks, logs_ = [], []
        sdump = ScoreDump() if a.dump_scores else None
        j = 0
        for batch in dlh:
            sx0, sxA, sxB, tg = fwd(batch)
            if sdump is not None:                                            # RL-S1: THESE sets' scores, keyed to their own rows (j = rows already done)
                bkeys = [key(ds.rows[idx[j + bi]]) for bi in range(tg.shape[0])]
                sdump.add(bkeys, [ho.by[k_][0] for k_ in bkeys], sx0, sxA, sxB)
            tlane = tg[..., 6] if tg.shape[-1] == 7 else None                    # teacher lane target (-1 = none)
            tg = tg[..., :6]
            ta = agg_v1(tg)                                                      # [b, M] true navsim_v1
            lkb = [lk_rows[j + bi] for bi in range(tg.shape[0])]
            for k, sx in (("base", sx0), ("A", sxA), ("B", sxB)):
                p = torch.sigmoid(sx)
                ix = agg_v1(p).argmax(1, keepdim=True)
                pick = ta.gather(1, ix).squeeze(1)
                res[k]["pick"] += pick.tolist(); res[k]["rand"] += ta.mean(1).tolist()
                res[k]["best"] += ta.max(1).values.tolist(); res[k]["zero"] += (pick == 0).float().tolist()
                res[k]["onc"] += (tg[..., 5].gather(1, ix).squeeze(1) < 1).float().tolist()
                ixx = (agg_v1(p) * p[..., 5]).argmax(1, keepdim=True)          # LANE-1's L1: x the direction head
                pkx = ta.gather(1, ixx).squeeze(1)
                res[k]["pick_x"] += pkx.tolist(); res[k]["zero_x"] += (pkx == 0).float().tolist()
                res[k]["onc_x"] += (tg[..., 5].gather(1, ixx).squeeze(1) < 1).float().tolist()
                def lk_of(ixs):
                    return [float(lkb[bi][int(ixs[bi])]) if lkb[bi] is not None else float("nan") for bi in range(len(lkb))]
                res[k]["lk"] += lk_of(ix.squeeze(1).tolist())
                if p.shape[-1] == 7:                                             # the lane rule: navsim_v1 x p_lane
                    ixl = (agg_v1(p) * p[..., 6]).argmax(1, keepdim=True)
                    pkl = ta.gather(1, ixl).squeeze(1)
                    res[k]["pick_l"] += pkl.tolist()
                    res[k]["zero_l"] += (pkl == 0).float().tolist()
                    res[k]["onc_l"] += (tg[..., 5].gather(1, ixl).squeeze(1) < 1).float().tolist()
                    res[k]["lk_l"] += lk_of(ixl.squeeze(1).tolist())
                    if tlane is not None:
                        for bi in range(p.shape[0]):
                            if bool((tlane[bi] >= 0).all()):
                                v = set_auc(p[bi, :, 6].cpu(), tlane[bi].cpu())
                                if v is not None:
                                    res[k]["auc_lane"].append(v)
                for bi in range(p.shape[0]):
                    for comp, nm in ((0, "auc_nc"), (1, "auc_dac"), (5, "auc_ddc")):
                        v = set_auc(p[bi, :, comp].cpu(), tg[bi, :, comp].cpu())
                        if v is not None:
                            res[k][nm].append(v)
            for _ in range(tg.shape[0]):
                r = ds.rows[idx[j]]; j += 1
                toks.append(f"{r.get('log_name')}|{r.get('token')}|{r.get('step')}|{r.get('rank', 0)}"); logs_.append(r.get("log_name"))
        out = {"event": "eval", "tag": tag, "n": len(toks)}
        for k, v in res.items():
            pk, rd, bs = np.mean(v["pick"]), np.mean(v["rand"]), np.mean(v["best"])
            out[k] = {"pick": 100 * pk, "random": 100 * rd, "best": 100 * bs, "skill": (pk - rd) / max(bs - rd, 1e-9),
                      "zero_picks": int(sum(v["zero"])), "auc_nc": float(np.mean(v["auc_nc"])) if v["auc_nc"] else None,
                      "auc_dac": float(np.mean(v["auc_dac"])) if v["auc_dac"] else None,
                      "auc_ddc": float(np.mean(v["auc_ddc"])) if v["auc_ddc"] else None,
                      "onc_picks": int(sum(v["onc"])),
                      "x_ddc": {"pick": 100 * float(np.mean(v["pick_x"])), "zero_picks": int(sum(v["zero_x"])),
                                "onc_picks": int(sum(v["onc_x"]))},
                      "lk10_picks": int(np.nansum(v["lk"])) if any(x == x for x in v["lk"]) else None}
            if v["pick_l"]:
                out[k]["x_lane"] = {"pick": 100 * float(np.mean(v["pick_l"])), "zero_picks": int(sum(v["zero_l"])),
                                    "onc_picks": int(sum(v["onc_l"])),
                                    "lk10_picks": int(np.nansum(v["lk_l"])) if any(x == x for x in v["lk_l"]) else None,
                                    "auc_lane_vs_teacher": float(np.mean(v["auc_lane"])) if v["auc_lane"] else None}
        tl = tl_map = dict(zip(toks, logs_))
        for k in ("A", "B"):
            d = {t: 100 * (res[k]["pick"][i] - res["base"]["pick"][i]) for i, t in enumerate(toks)}
            mu, lo, hi = boot_logs(d, tl)
            out[k]["pick_minus_base"] = {"mean": mu, "ci95": [lo, hi], "estimator": "log-cluster bootstrap over the held-out logs, 10,000, seed 20260927"}
            dx = {t: 100 * (res[k]["pick_x"][i] - res["base"]["pick"][i]) for i, t in enumerate(toks)}
            mu, lo, hi = boot_logs(dx, tl)
            out[k]["x_ddc"]["pick_minus_base_v1"] = {"mean": mu, "ci95": [lo, hi]}
            do = {t: 100 * (res[k]["onc_x"][i] - res["base"]["onc"][i]) for i, t in enumerate(toks)}
            mu, lo, hi = boot_logs(do, tl)
            out[k]["x_ddc"]["onc_pp_minus_base_v1"] = {"mean": mu, "ci95": [lo, hi]}
        for k in ("A", "B"):
            if res[k]["pick_l"]:
                dl_ = {t: 100 * (res[k]["pick_l"][i] - res["base"]["pick"][i]) for i, t in enumerate(toks)}
                mu, lo, hi = boot_logs(dl_, tl_map)
                out[k]["x_lane"]["pick_minus_base_v1"] = {"mean": mu, "ci95": [lo, hi]}
                if all(x == x for x in res[k]["lk_l"]) and all(x == x for x in res["base"]["lk"]):
                    dk = {t: 100 * (res[k]["lk_l"][i] - res["base"]["lk"][i]) for i, t in enumerate(toks)}
                    mu, lo, hi = boot_logs(dk, tl_map)
                    out[k]["x_lane"]["lk10_pp_minus_base_v1"] = {"mean": mu, "ci95": [lo, hi]}
        dx = {t: 100 * (res["base"]["pick_x"][i] - res["base"]["pick"][i]) for i, t in enumerate(toks)}
        mu, lo, hi = boot_logs(dx, tl)
        out["base"]["x_ddc"]["pick_minus_base_v1"] = {"mean": mu, "ci95": [lo, hi]}
        do = {t: 100 * (res["base"]["onc_x"][i] - res["base"]["onc"][i]) for i, t in enumerate(toks)}
        mu, lo, hi = boot_logs(do, tl)
        out["base"]["x_ddc"]["onc_pp_minus_base_v1"] = {"mean": mu, "ci95": [lo, hi]}
        log(out)
        if sdump is not None:
            log({"event": "score_dump", "tag": tag, "n_sets": len(sdump.keys), "files": sdump.write(a.dump_scores, tag)})
        return out

    def save_full(arm_name, mods):
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        new = {}
        for m in SCORER:
            for k_, v in getattr(mods, m).state_dict().items():
                new[f"{m}.{k_}"] = v.detach().to("cpu", copy=True)
        assert set(new) <= set(state), "scorer keys missing from the checkpoint"
        n_out = int(new["score_head.weight"].shape[0])
        for k_, v in new.items():
            if not (n_out == 7 and k_.startswith("score_head.")):
                assert state[k_].shape == v.shape
            state[k_] = v.to(state[k_].dtype)
        meta = dict(sd.get("meta", {}) if isinstance(sd, dict) else {},
                    scorer_finetune={"arm": arm_name, "base_ckpt": a.ckpt, "args": vars(a)})
        meta["n_score_components"] = n_out                    # the planner sizes its score head from this (ckpt_io)
        out = {"format": ckpt_io.FORMAT_FULL, "model": state, "meta": meta}
        p = os.path.join(a.out, f"model_sft_{arm_name}.pt")
        ckpt_io.atomic_save(out, p)
        return p

    # ---------------- PREFLIGHT (the launch gate)
    if a.preflight:
        gates = {}
        tn = ckpt_io.trainable_names(model)
        want = {n for n, _ in model.named_parameters() if n.split(".")[0] in SCORER}
        gates["G1_trainable_is_exactly_the_scorer"] = {"ok": tn == want, "n_trainable": len(tn), "n_scorer": len(want),
                                                       "extra": sorted(tn - want)[:5], "missing": sorted(want - tn)[:5]}
        gates["G7_heldout_disjoint"] = {"ok": info["train_rows_in_heldout_logs"] == 0 and len(ho_idx) > 0,
                                        "heldout_sets": len(ho_idx), "train_rows_in_heldout_logs": info["train_rows_in_heldout_logs"]}
        batch, prow = split_rows(next(iter(DataLoader(make_train_set(ds, tr_idx[:a.batch], a.epo), batch_size=a.batch))), a.epo)
        hk: dict = {}                                        # score-head input/output of each copy (last call = the scored set)
        hooks = [mod.score_head.register_forward_hook(lambda m_, i_, o_, nm=nm: hk.__setitem__(nm, (i_[0].detach(), o_.detach())))
                 for nm, mod in (("A", model), ("B", armB), ("base", base))]
        sx0, sxA, sxB, tg = fwd(batch)
        for h_ in hooks:
            h_.remove()
        # RL-S1: [..., :sx0.shape[-1]] == [..., :6] for every 6-wide base (unchanged); a 7-wide NATIVE head (u400) compares all 7 columns
        d0A, dBA = float((sx0 - sxA[..., :sx0.shape[-1]]).abs().max()), float((sxB - sxA).abs().max())
        g3 = {"max_base_vs_A": d0A, "max_B_vs_A": dBA, "n_out": int(sxA.shape[-1])}
        if a.lane_head and not native7:
            # ⛔ A 7-row head is a different GEMM shape from the base's 6-row one, so under bf16 the shared outputs can differ by
            # one bf16 rounding step (MEASURED in the 2026-10-04 smoke: 0.0625 = 1 ulp at |logit| 8..16). "Identical at step
            # 0" is therefore asserted EXACTLY where it is exact by construction: the head's INPUT (same unchanged modules),
            # the copied parameters, and rows 0-5 applied as a 6-row GEMM, which is the base's own computation bit for bit.
            # The full-head residual must stay within 1 bf16 ulp. REFE_SFT4_MUTATE_G3=1 (a mis-copied row) must turn this RED.
            hA, h0, hB = hk["A"][0], hk["base"][0], hk["B"][0]
            W, bvec = model.score_head.weight, model.score_head.bias
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o6 = F.linear(hA, W[:6], bvec[:6])
            # bf16 residual of the full head (reported): the bias is added AFTER the GEMM's bf16 rounding, so an output near 0
            # carries rounding at the PRE-BIAS magnitude (MEASURED: 89 output-ulps at |diff| 0.0625). Scaled to the larger of
            # the output and the pre-bias product it is the real rounding step.
            W0, b0 = base.score_head.weight, base.score_head.bias
            pre = F.linear(h0.float(), W0.float()).abs()
            mag = torch.maximum(torch.maximum(sx0.abs(), sxA[..., :6].abs()), pre).clamp_min(2.0 ** -100)
            ulp = torch.exp2(torch.floor(torch.log2(mag)) - 7)          # bf16: 8 significand bits -> ulp = 2^(e - 7)
            # GATING residual, in fp32 on the identical captured input: two summation orders of the same dot products differ
            # by at most 2 * K * eps32 * sum_k |h_k w_k| (the standard forward-error bound gamma_K for ANY order), plus the
            # bias add. A real copy defect has no such bound (and is caught exactly by rows0to5_params_equal_base anyway).
            h32 = hA.float()
            tf32_was = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False          # the bound is for strict fp32, not TF32's 10-bit mantissa
            with torch.no_grad():
                o7 = F.linear(h32, W.float(), bvec.float())[..., :6]
                o6b = F.linear(h32, W0.float(), b0.float())
                eps32 = float(torch.finfo(torch.float32).eps)
                bound = 2 * h32.shape[-1] * eps32 * F.linear(h32.abs(), W0.float().abs()) + 2 * eps32 * o6b.abs()
            torch.backends.cuda.matmul.allow_tf32 = tf32_was
            g3.update({"max_head_input_A_vs_base": float((hA.float() - h0.float()).abs().max()),
                       "max_head_input_B_vs_A": float((hB.float() - hA.float()).abs().max()),
                       "max_rows0to5_as_6row_gemm_vs_base": float((o6.float() - sx0).abs().max()),
                       "rows0to5_params_equal_base": bool(torch.equal(W[:6], base.score_head.weight) and torch.equal(bvec[:6], base.score_head.bias)),
                       "other_scorer_params_equal_base": all(torch.equal(p, q) for m in ("score_q_mlp", "score_dec")
                                                             for p, q in zip(getattr(model, m).parameters(), getattr(base, m).parameters())),
                       "row6_init": bool(float(W[6].abs().max()) == 0.0 and float(bvec[6]) == 1.5),
                       "max_bf16_ulps_full_head_vs_base": float(((sxA[..., :6] - sx0).abs() / ulp).max()),
                       "max_fp32_full_head_vs_base": float((o7 - o6b).abs().max()),
                       "fp32_within_order_bound": bool(((o7 - o6b).abs() <= bound).all()),
                       "fp32_bound_max": float(bound.max())})
            g3["ok"] = (g3["max_head_input_A_vs_base"] == 0.0 and g3["max_head_input_B_vs_A"] == 0.0
                        and g3["max_rows0to5_as_6row_gemm_vs_base"] == 0.0 and g3["rows0to5_params_equal_base"]
                        and g3["other_scorer_params_equal_base"] and g3["row6_init"] and dBA == 0.0
                        and g3["fp32_within_order_bound"])
        else:
            g3["ok"] = d0A == 0.0 and dBA == 0.0
        gates["G3_copies_equal_model_at_step0"] = g3
        if a.b_mode == "compw":                              # SFT-2: unit weights must reproduce A; the declared ones must not
            keep_w = compw.clone()
            compw.fill_(1.0)
            bA, bB0, _ = losses(sxA, sxB, tg, 0.0)
            compw.copy_(keep_w)
            bA2, bB, ln = losses(sxA, sxB, tg, 0.0)
            differs = bool((keep_w != 1).any())
            gates["G4_compw_identity_and_live"] = {"ok": math.isclose(float(bA), float(bB0), rel_tol=1e-5) and
                                                   (float(bB) != float(bA) if differs else True) and math.isfinite(float(bB)),
                                                   "bceA": float(bA), "lossB_unit_w": float(bB0), "lossB": float(bB), "compw": keep_w.tolist()}
        else:
            bA, bB0, _ = losses(sxA, sxB, tg, 0.0)
            bA2, bB, ln = losses(sxA, sxB, tg, a.lam)
            # RL-S1: --lam 0 DECLARES arm B's non-EPO loss plain BCE (--epo requires it), so liveness of the ranking term is not demanded then
            # (the EPO term has its own live gate, G11); for every lam > 0 this is the unchanged identity-AND-live gate.
            gates["G4_listnet_identity_and_live"] = {"ok": float(bA) == float(bB0) and (float(bB) != float(bA) or a.lam == 0.0) and math.isfinite(float(ln)),
                                                     "bceA": float(bA), "lossB_lam0": float(bB0), "lossB": float(bB), "listnet": float(ln)}
        if a.lane_labels_train or a.lane_labels_heldout:      # G8: the NAVSIM direction labels really are in the banks
            def has_half(bank):
                return sum(1 for e in bank.by.values() if np.any(np.abs(e[2][:, 5] - 0.5) < 1e-6))
            g8 = {"train_lane_sets": op.n_lane, "train_stale": op.n_lane_stale, "train_superseded": op.n_lane_superseded,
                  "heldout_lane_sets": ho.n_lane, "heldout_stale": ho.n_lane_stale, "train_sets_with_ddc_0.5": has_half(op), "heldout_sets_with_ddc_0.5": has_half(ho)}
            g8["ok"] = all([(not a.lane_labels_train) or (op.n_lane > 0 and op.n_lane_stale <= 0.05 * max(op.n_lane, 1)
                                                          and g8["train_sets_with_ddc_0.5"] > 0),
                            (not a.lane_labels_heldout) or (ho.n_lane > 0 and ho.n_lane_stale <= 0.05 * max(ho.n_lane, 1)
                                                            and g8["heldout_sets_with_ddc_0.5"] > 0)])
            gates["G8_lane_labels_in_bank"] = g8
        if a.pdm_labels_train or a.pdm_labels_heldout:      # G9: the paper's PDM targets really are what is served
            def has_soft_ep(bank):                          # PDM EP is pairwise-normalised: values strictly inside (0, 1)
                return sum(1 for kk in bank.pdm_keys if np.any((bank.by[kk][2][:, 2] > 0.01) & (bank.by[kk][2][:, 2] < 0.99)))
            g9 = {"train_pdm": op.n_pdm, "train_stale": op.n_pdm_stale, "heldout_pdm": ho.n_pdm, "heldout_stale": ho.n_pdm_stale,
                  "train_sets_served": len(tr_idx), "heldout_sets_served": len(ho_idx),
                  "train_with_soft_ep": has_soft_ep(op), "heldout_with_soft_ep": has_soft_ep(ho), "require_pdm": a.require_pdm}
            g9["ok"] = (op.n_pdm > 0 and ho.n_pdm > 0 and op.n_pdm_stale <= 0.05 * max(op.n_pdm, 1)
                        and g9["train_with_soft_ep"] > 0 and (not a.require_pdm or (len(tr_idx) > 0 and len(ho_idx) > 0)))
            gates["G9_pdm_targets_in_bank"] = g9
        if a.lane_head:                                      # G10: the teacher lane label really reaches the 7th output
            soft = sum(1 for v in op.teacher_lane.values() if np.any((v > 0.01) & (v < 0.99)))
            g10 = {"train_lane_sets": op.n_tlane, "heldout_lane_sets": ho.n_tlane, "train_sets_with_soft_target": soft,
                   "batch_lane_labelled": int((tg[..., 6] >= 0).sum()) if tg.shape[-1] == 7 else 0}
            g10["ok"] = op.n_tlane > 0 and ho.n_tlane > 0 and soft > 0
            gates["G10_teacher_lane_in_bank"] = g10
        (bA2 + bB).backward()
        live = {}
        for m in SCORER:
            ga = [p.grad for p in getattr(model, m).parameters()]
            gb = [p.grad for p in getattr(armB, m).parameters()]
            live[m] = {"A_nonzero": all(x is not None and torch.isfinite(x).all() and float(x.abs().sum()) > 0 for x in ga),
                       "B_nonzero": all(x is not None and torch.isfinite(x).all() and float(x.abs().sum()) > 0 for x in gb)}
        frozen_grads = sum(1 for n, p in model.named_parameters() if n.split(".")[0] not in SCORER and p.grad is not None)
        base_grads = sum(1 for p in base.parameters() if p.grad is not None)
        gates["G2_live"] = {"ok": all(v["A_nonzero"] and v["B_nonzero"] for v in live.values()) and frozen_grads == 0 and base_grads == 0,
                            "per_module": live, "frozen_params_with_grad": frozen_grads, "base_params_with_grad": base_grads}
        if a.epo:                                            # G11 (RL-S1): the EPO term is live on every scorer module, its zero controls read EXACTLY 0.0
            g11 = g11_epo_live_and_identity(armB, lambda mode: epo_probe(batch, prow, mode), kl_tol=(1e-2 if (a.lane_head and not native7) else 0.0))
            g11["rows_have_terms"] = terms_train is None or all(key(ds.rows[int(i)]) in terms_train for i in prow)
            g11["heldout_terms"] = st_ho
            g11["ok"] = bool(g11["ok"] and g11["rows_have_terms"] and (not a.epo_terms_heldout or st_ho["matched"] > 0))
            gates["G11_epo_live_and_identity"] = g11
        # G5: a CHANGED scorer, saved, reloads through the PLANNER's loader and reproduces its logits (and differs from
        # base) -- saving an unchanged scorer would pass even if the save wrote the original weights back
        with torch.no_grad():
            gen = torch.Generator(device=dev).manual_seed(1)
            for p in (p for m in SCORER for p in getattr(model, m).parameters()):
                p.add_(1e-2 * torch.randn(p.shape, device=dev, generator=gen))
        pth = save_full("A_preflight", model)
        m2 = REFe(ckpt_io.config_for_checkpoint(cfg, pth)).to(dev)
        ckpt_io.load_for_inference(m2, pth, map_location=dev, backbone=a.backbone)
        m2.eval()
        cap2: dict = {}
        img, ego, goal, _t, cxy, _g, _m, _i, cal = batch
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, s_reload = m2(img.to(dev), ego.to(dev), goal.to(dev), calib=cal if cal.numel() else None, score_extra=cxy.to(dev))
            _, _, s_orig = model(img.to(dev), ego.to(dev), goal.to(dev), calib=cal if cal.numel() else None, score_extra=cxy.to(dev))
            s_base = base(cxy.to(dev), cap["sctx"])
        d_rt = float((s_reload.float() - s_orig.float()).abs().max())
        d_base = float((s_orig.float()[..., :s_base.shape[-1]] - s_base.float()).abs().max())      # == [..., :6] for a 6-wide base
        gates["G5_ckpt_roundtrip_planner_loader"] = {"ok": d_rt == 0.0 and d_base > 0.0, "max_abs_reload_vs_saved": d_rt,
                                                     "max_abs_saved_vs_base": d_base}
        os.remove(pth)
        # restore the true starting weights (the perturbation was for the gate only)
        sd0 = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        st0 = sd0["model"] if isinstance(sd0, dict) and "model" in sd0 else sd0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.split(".")[0] in SCORER:
                    if n.startswith("score_head.") and a.lane_head and not native7:
                        p[:6].copy_(st0[n].to(dev))
                        if n.endswith("weight"):
                            p[6].zero_()
                        else:
                            p[6] = 1.5
                    else:
                        p.copy_(st0[n].to(dev))
        del sd0, st0
        # speed: 3 timed micro-batches (fwd + both backwards)
        t0 = time.time()
        it = iter(dl)
        for _ in range(3):
            b_, r_ = split_rows(next(it), a.epo)
            s0, sA, sB, t_ = fwd(b_)
            l1, l2, _, _ = train_losses(s0, sA, sB, t_, r_, 0, lam_pg=(1.0 if a.epo else None))      # --epo: the timing includes the EPO term
            (l1 + l2).backward()
        torch.cuda.synchronize()
        sp = (time.time() - t0) / 3
        ok = all(v["ok"] for v in gates.values())
        res = {"event": "preflight", "verdict": "PASS" if ok else "FAIL", "gates": gates,
               "s_per_microbatch": round(sp, 2), "est_hours_per_epoch": round(len(tr_idx) / a.batch * sp / 3600, 2),
               "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        log(res)
        if ok and os.environ.get("REFE_SFT_MUTATE_UNFREEZE") != "1" and os.environ.get("REFE_RL_MUTATE_EPO_DETACH") != "1":
            json.dump(res, open(os.path.join(a.out, "PREFLIGHT_PASS.json"), "w"), indent=1, default=str)
        print("ZZSFT_PREFLIGHT_" + ("PASS" if ok else "FAIL"))
        return 0 if ok else 3

    # ---------------- the run
    paramsA = [p for m in SCORER for p in getattr(model, m).parameters()]
    paramsB = list(armB.parameters())
    optA = torch.optim.AdamW(paramsA, lr=a.lr, weight_decay=0.01)
    optB = torch.optim.AdamW(paramsB, lr=a.lr, weight_decay=0.01)
    total = int(math.ceil(len(tr_idx) / a.batch * a.epochs / a.accum))
    if a.max_updates:
        total = min(total, a.max_updates)
    lr_at = lambda u: a.lr * (u + 1) / a.warmup if u < a.warmup else a.lr * 0.5 * (1 + math.cos(math.pi * (u - a.warmup) / max(total - a.warmup, 1)))
    log({"event": "start", "updates": total, "micro_per_update": a.accum, "batch": a.batch, "train_sets": len(tr_idx)})
    evaluate("step0")
    upd, micro, t0 = 0, 0, time.time()
    accl = {"bceA": 0.0, "lossB": 0.0, "listnet": 0.0, "n": 0}
    epo_acc = {"epo": 0.0, "J": 0.0, "kl": 0.0, "n": 0}      # --epo only
    done = False
    while not done:
        for batch in dl:
            batch, rows = split_rows(batch, a.epo)
            _s0, sA, sB, tg = fwd(batch)
            lA, lB, ln, ep_t = train_losses(_s0, sA, sB, tg, rows, upd)
            ((lA + lB) / a.accum).backward()
            accl["bceA"] += float(lA); accl["lossB"] += float(lB); accl["listnet"] += float(ln); accl["n"] += 1
            if a.epo:
                epo_acc["epo"] += float(ep_t); epo_acc["J"] += epo_last.get("J", 0.0); epo_acc["kl"] += epo_last.get("kl", 0.0); epo_acc["n"] += 1
            micro += 1
            if micro % a.accum:
                continue
            for o, ps in ((optA, paramsA), (optB, paramsB)):
                torch.nn.utils.clip_grad_norm_(ps, 1.0)
                for gr in o.param_groups:
                    gr["lr"] = lr_at(upd)
                o.step()
                o.zero_grad(set_to_none=True)
            upd += 1
            if upd % 25 == 0:
                n = max(accl["n"], 1)
                row = {"event": "train", "update": upd, "of": total, "lr": lr_at(upd), "bceA": accl["bceA"] / n,
                       "lossB": accl["lossB"] / n, "listnet": accl["listnet"] / n, "s_per_update": (time.time() - t0) / upd}
                if a.epo:
                    ne = max(epo_acc["n"], 1)
                    row.update({"epo": epo_acc["epo"] / ne, "epo_J": epo_acc["J"] / ne, "epo_kl": epo_acc["kl"] / ne,
                                "epo_lam_pg": a.epo_lam * epo_ramp(upd, a.epo_ramp)})
                    epo_acc = {"epo": 0.0, "J": 0.0, "kl": 0.0, "n": 0}
                log(row)
                accl = {"bceA": 0.0, "lossB": 0.0, "listnet": 0.0, "n": 0}
            if upd % a.eval_every == 0 or upd == total:
                evaluate(f"update{upd}")
                torch.save({"A": {m: getattr(model, m).state_dict() for m in SCORER},
                            "B": {m: getattr(armB, m).state_dict() for m in SCORER}, "update": upd},
                           os.path.join(a.out, "scorer_states_last.pt"))
            if upd >= total:
                done = True
                break
    pA, pB = save_full("A", model), save_full("B", armB)
    log({"event": "done", "updates": upd, "model_A": pA, "model_B": pB, "hours": (time.time() - t0) / 3600})
    print("ZZSFT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
