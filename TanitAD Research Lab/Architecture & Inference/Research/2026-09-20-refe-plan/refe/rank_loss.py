#!/usr/bin/env python3
"""SCORER-FT's within-token ranking term L_rank (DESIGN_SCORER_FT.md section 2.2; PI go 2026-11-20).

WHAT.  The scorer's only term today is a pointwise BCE on the six heads.  L_rank adds PAIRWISE LOGISTIC losses inside each labelled 64-proposal set, weighted by the harness
(navsim_v1) PDMS gap, and rides the SAME deferred backward as l_score (train.py, online-labeller branch).  It trains the scorer decoder and, through `visual_ctx`
(--scorer-sees-trunk), the trunk.

  logits s [B, M, 6]  in the order NC, DAC, EP, TTC, C, DDC;  labels y [B, M, 6] (the lag-0 online labels);  msk [B] (1 = the set was labelled)
    safe_m = (y_NC > 0.5) & (y_DAC > 0.5)                                   (= NC*DAC == 1, head_auc.py:54)
    pdms_m = y_NC * y_DAC * (5 y_EP + 5 y_TTC + 2 y_C) / 12                 (navsim_v1, planner.py:558-559)
    topk   = the k largest of navsim_v1(sigmoid(s)) per set                 (no_grad: the planner's own pick score)
    EP   pairs (i, j): safe_i & safe_j & y_iEP - y_jEP >= gap               weight (5/12) (y_iEP - y_jEP)      z = s_EP
    gate pairs (i, j): safe_i & ~safe_j & j in topk & pdms_i > pdms_j       weight pdms_i - pdms_j             z = logsigmoid(s_NC) + logsigmoid(s_DAC)
    TTC  pairs (i, j): safe_i & safe_j & y_iTTC == 1 & y_jTTC == 0          weight 5/12                        z = s_TTC       (only if named)
    PICK pairs (i, k): k = pick_b = argmax_m navsim_v1(sigmoid(s_m)) (detached; the planner's own selection, REFePlanner.aggregate rule navsim_v1)
                       pdms_i > pdms_k (STRICT; pdms on the LABELS)      weight pdms_i - pdms_k             z = navsim_v1(sigmoid(s))   (the aggregate itself, with grad; O3 card, PREREG_SCORER_FT 9)
                       LABEL-CONSISTENT ROUTING: inside z, each head probability p_h(m) passes gradient only where it moves toward that proposal's OWN label y_h(m): on the pick (term lowers s_pick)
                       iff y_h < p_h, on a better proposal (term raises s_j) iff y_h > p_h; else p_h.detach() (forward value bit-identical; MM design decision 10).  No flag, no lever.
    L_h(b)  = sum a_ij softplus(-(z_i - z_j)) / sum a_ij        (0 for a set with no pair of head h; counted)
    L_rank  = (1 / batch) * sum_b msk_b * sum_{h in heads} L_h(b)     <- a CONSTANT denominator (the accumulation lesson of train.py:1858-1871)

EXACT ZEROS (analytic controls the launch gate reads): the term reads only the logits of NC, DAC, EP and TTC (the safe-gating is on LABELS), so rows C and DDC of
`score_head` get EXACTLY 0.0 (and TTC too unless named; head `pick` reads NC, DAC, EP, TTC and C through the aggregate, so only DDC is then EXACTLY 0.0); and the trainer feeds it the scorer output of the default forward, whose trajectory input is detached (model.py:779),
so the trajectory decoder gets EXACTLY 0.0.  The pair masks and weights are functions of the labels only and carry no gradient.

DELIBERATE-REGRESSION ARMS are environment mutations, never flags (the REFE_LAB_MUTATE convention): REFE_RANK_MUTATE =
  drop        (train.py) L_rank is computed and NOT added to the loss
  unsafe_ep   (here)     EP pairs admit proposals with NC*DAC < 1
  shuffle     (here)     the labels are permuted within the set (a fixed permutation) before pairing
  traj_leak   (train.py) L_rank is taken on a `detach_scorer=False` forward (model.py:711-712): the trajectory decoder then gets a gradient
  head_swap   (here)     the gate's z reads the TTC logit instead of the DAC logit
  pick_oracle (here)     the pick head anchors on the LABEL-best proposal instead of the scorer's own pick (a decision-level term that no longer follows the decision)
  pick_noroute (here)    the pick head's label-consistent gradient routing is removed (s~ = s, every factor live; forward value unchanged) -- CTRL-ROUTE's deliberate regression
The launch profile forbids the variable.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

HEADS = ("ep", "gate", "ttc", "pick")
MUTATE_ENV = "REFE_RANK_MUTATE"
MUTATIONS = ("drop", "unsafe_ep", "shuffle", "traj_leak", "head_swap", "pick_oracle", "pick_noroute")
V1_W = (5.0, 5.0, 2.0)                 # planner.py:520 -- progress (EP), time-to-collision, comfort; the sum is 12
NC, DAC, EP, TTC, C, DDC = range(6)


def mutation() -> str | None:
    """The deliberate-regression arm named by the environment, or None.  An unknown name is an error (a typo must not read as 'no mutation')."""
    m = os.environ.get(MUTATE_ENV) or None
    if m is not None and m not in MUTATIONS:
        raise ValueError(f"{MUTATE_ENV}={m!r} is not one of {MUTATIONS}")
    return m


def navsim_v1(p: torch.Tensor) -> torch.Tensor:
    """NC x DAC x (5 EP + 5 TTC + 2 C) / 12 on probabilities / labels [..., 6] (planner.py:558-559, rule navsim_v1)."""
    w1 = torch.tensor(V1_W, device=p.device, dtype=p.dtype)
    return p[..., NC] * p[..., DAC] * (p[..., EP] * w1[0] + p[..., TTC] * w1[1] + p[..., C] * w1[2]) / w1.sum()


def pick_index(score: torch.Tensor) -> torch.Tensor:
    """[B, M, 6] logits -> [B] long: the index the inference-time rule `navsim_v1` selects, argmax_m navsim_v1(sigmoid(score)), in the dtype the logits ARRIVE in (the trainer hands float32, as the planner's
    float32 argmax reads; float64 re-evaluation flips near-ties).  This is `REFePlanner.aggregate(rule=navsim_v1)` + `.argmax()` (planner.py:558-559, `_pick`:575-576): `navsim_v1` above is the same expression in
    the same operation order, and test_rank_loss.chk_pick_identity asserts bit-for-bit equality with the planner's own method on random logits.  No gradient."""
    with torch.no_grad():
        return navsim_v1(torch.sigmoid(score.detach())).argmax(1)


def parse_heads(s) -> tuple:
    """'ep,gate' -> ('ep', 'gate'); '' -> ().  Raises ValueError on an unknown, empty or duplicated entry."""
    if s is None or str(s).strip() == "":
        return ()
    parts = [x.strip() for x in str(s).split(",")]
    bad = [x for x in parts if x not in HEADS]
    if bad:
        raise ValueError(f"unknown rank head(s) {bad}: choose from {HEADS}")
    if len(set(parts)) != len(parts):
        raise ValueError(f"duplicate rank head in {parts}")
    return tuple(parts)


class RankLoss:
    def __init__(self, heads, ep_gap: float = 0.02, gate_topk: int = 8, mutate: str | None = None):
        self.heads = parse_heads(heads if isinstance(heads, str) else ",".join(heads))
        if not self.heads:
            raise ValueError("RankLoss needs at least one head")
        if not (0.0 < float(ep_gap) <= 1.0):
            raise ValueError(f"ep_gap {ep_gap} must be in (0, 1]")
        if int(gate_topk) != gate_topk or gate_topk < 1:
            raise ValueError(f"gate_topk {gate_topk} must be an int >= 1")
        self.ep_gap, self.gate_topk = float(ep_gap), int(gate_topk)
        self.mutate = mutate if mutate is not None else mutation()
        if self.mutate is not None and self.mutate not in MUTATIONS:
            raise ValueError(f"mutation {self.mutate!r} is not one of {MUTATIONS}")
        self.last: dict = {}

    def facts(self) -> dict:
        """What was BUILT (the `rank` start event; G-DVB reads it, never the namespace)."""
        return {"heads": list(self.heads), "ep_gap": self.ep_gap, "gate_topk": self.gate_topk, "aggregate": "navsim_v1", "reduce": "set-wmean/const-B",
                "mutate": self.mutate}

    def pairs(self, score: torch.Tensor, labels: torch.Tensor) -> dict:
        """The pair masks, weights and logit differences per head, each [B, M, M] (i = the better, j = the worse).  Float64 throughout: the pair test is on label values
        and the gate's numpy reference must agree with it at the boundary."""
        y = labels.detach().double()
        if self.mutate == "shuffle":
            perm = torch.randperm(y.shape[1], generator=torch.Generator().manual_seed(20261120))
            y = y[:, perm.to(y.device)]
        s = score.double()
        B, M, _ = s.shape
        safe = (y[..., NC] > 0.5) & (y[..., DAC] > 0.5)                                    # [B, M]
        pdms = navsim_v1(y)                                                                # the harness aggregate on the LABELS
        out = {"safe": safe, "pdms": pdms}
        if "ep" in self.heads:
            d = y[..., EP].unsqueeze(2) - y[..., EP].unsqueeze(1)                          # y_i - y_j
            both = torch.ones_like(d, dtype=torch.bool) if self.mutate == "unsafe_ep" else (safe.unsqueeze(2) & safe.unsqueeze(1))
            m = both & (d >= self.ep_gap)
            z = s[..., EP]
            out["ep"] = (m, (V1_W[0] / sum(V1_W)) * d, z.unsqueeze(2) - z.unsqueeze(1))
        if "gate" in self.heads:
            with torch.no_grad():
                agg = navsim_v1(torch.sigmoid(s))
                top = torch.zeros(B, M, dtype=torch.bool, device=s.device)
                top.scatter_(1, torch.topk(agg, min(self.gate_topk, M), dim=1).indices, True)
            d = pdms.unsqueeze(2) - pdms.unsqueeze(1)
            m = safe.unsqueeze(2) & (~safe).unsqueeze(1) & top.unsqueeze(1) & (d > 0)
            z_second = s[..., TTC] if self.mutate == "head_swap" else s[..., DAC]
            z = F.logsigmoid(s[..., NC]) + F.logsigmoid(z_second)
            out["gate"] = (m, d, z.unsqueeze(2) - z.unsqueeze(1))
        if "pick" in self.heads:
            with torch.no_grad():
                pk = pick_index(score)                                                       # the scorer's own pick under the planner's rule, in the logits' own dtype
                if self.mutate == "pick_oracle":
                    pk = pdms.argmax(1)
                at = torch.zeros(B, M, dtype=torch.bool, device=s.device)
                at.scatter_(1, pk.unsqueeze(1), True)                                        # at[b, k] = k is the pick
            prob = torch.sigmoid(s)                                                          # [B, M, 6] float64 WITH grad
            # LABEL-CONSISTENT ROUTING (MM design decision 10): a sub-score head's probability gets the term's gradient only in the direction of that proposal's OWN harness label.  The term
            # LOWERS s_pick, so on the pick p_h flows only if y_h < p_h; it RAISES s_j on a better proposal j, so p_h flows only if y_h > p_h.  Per-factor stop-gradient: the forward value
            # (so s, the pick and the loss value) is bit-identical to the unrouted form -- torch.where selects p or p.detach(), the SAME numbers.  y is the lag-0 online label that the BCE heads
            # train on (train.py:1978 lab_t -> BCE 1980 -> rank.term(labels=lab_t) 2011); all five heads inside the aggregate (NC, DAC, EP, TTC, C) are labelled there; DDC is not in the aggregate.
            if self.mutate != "pick_noroute":
                cons = torch.where(at.unsqueeze(2), y < prob.detach(), y > prob.detach())    # [B, M, 6]
                prob = torch.where(cons, prob, prob.detach())
            agg = navsim_v1(prob)                                                            # [B, M] float64 WITH grad: the aggregate the rule argmaxes
            d = pdms.unsqueeze(2) - pdms.unsqueeze(1)                                        # pdms_i - pdms_k
            m = at.unsqueeze(1) & (d > 0)                                                    # i better than the pick k, STRICT
            out["pick"] = (m, d, agg.unsqueeze(2) - agg.unsqueeze(1))
        if "ttc" in self.heads:
            m = safe.unsqueeze(2) & safe.unsqueeze(1) & (y[..., TTC] == 1.0).unsqueeze(2) & (y[..., TTC] == 0.0).unsqueeze(1)
            z = s[..., TTC]
            out["ttc"] = (m, torch.full(m.shape, V1_W[1] / sum(V1_W), dtype=torch.float64, device=s.device), z.unsqueeze(2) - z.unsqueeze(1))
        return out

    def term(self, score: torch.Tensor, labels: torch.Tensor, msk: torch.Tensor, batch: int) -> torch.Tensor:
        """L_rank (UNWEIGHTED by --rank-w) for one micro-batch.  score [B, M, 6] logits (with grad), labels [B, M, 6], msk [B], batch = the CONSTANT denominator (--batch).
        Pair / set counts of this call (over labelled sets only) are left in `self.last`."""
        P = self.pairs(score, labels)
        mw = msk.detach().double()
        has = mw > 0.0                                                                     # [B] labelled sets
        total = score.new_zeros((), dtype=torch.float64)
        stats = {"pairs": {}, "sets": {}}
        for h in self.heads:
            m, a, dz = P[h]
            w = a * m.double()                                                             # weights are label-only: no gradient
            den = w.sum((1, 2))                                                            # [B]
            num = (w * F.softplus(-dz)).sum((1, 2))
            Lh = torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(den))
            total = total + (Lh * mw).sum()
            stats["pairs"][h] = int(m.sum((1, 2))[has].sum())
            stats["sets"][h] = int(((den > 0) & has).sum())
        stats["sets_labelled"] = int(has.sum())
        self.last = stats
        return (total / float(batch)).to(score.dtype)
