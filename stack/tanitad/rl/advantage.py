"""Critic-free group-relative advantage estimators for planner post-training.

WHY A POLICY GRADIENT IS THE RIGHT TOOL *HERE* — AND NOT AT THE SELECTOR
------------------------------------------------------------------------
⚠️ `products/P4-training-pipelines/METHOD_LIBRARY.md` §2 concluded that GRPO/RLOO
is **dominated** for our stack. That conclusion stands, and this module does not
contradict it — it applies to a different site. Read the scope carefully, because
the two documents look contradictory if you do not:

| | METHOD_LIBRARY §2 (dominated) | HERE (policy gradient earns its keep) |
|---|---|---|
| site | the SELECTOR over the emitted fan | the DIFFUSION DECODER's sampled offsets |
| action set | ENUMERABLE (N anchors) — we take the expectation exactly | CONTINUOUS offsets under exploration noise — not enumerable |
| objective | metric L2 cost, DIFFERENTIABLE end-to-end (`unicycle_rollout` in-graph) | rule-based reward (collision / feasibility / headway) — **NOT differentiable** |
| ⇒ estimator | sampling only adds variance | the score-function estimator is the only route |

So: the score-function estimator exists because the environment is a black box.
At the *selector* we are the environment, so it is dominated. At the *decoder*
under a rule-based reward we are not, so it is not. ⇒ **This library targets the
generator, never the selector** — which is also exactly what the reward /
selector disjointness rule (`rewards.FORBIDDEN_REWARD_INPUTS`) requires, so the
two constraints agree rather than trade off.

THE TWO ESTIMATORS (DiffusionDriveV2, arXiv 2512.07745)
--------------------------------------------------------
DDv2 composes two advantages on top of a DiffusionDrive cold start:

1. **Intra-anchor GRPO** — for anchor *i* with a group of G sampled offsets,
   the advantage of sample *g* is its reward centred (and optionally scaled) on
   that anchor's OWN group::

       A_intra(i,g) = (r(i,g) - mean_g r(i,.)) / (std_g r(i,.) + eps)

   ⚠️ ``normalize="none"`` (centre only, no std division) is the **Dr. GRPO**
   correction — dividing by the group std reweights groups by the inverse of
   their reward spread, which is a bias, not a normalisation. We default to
   centre-only and make the std path opt-in and named.

2. **Inter-anchor truncated advantage** — across anchors, negatives are
   truncated to 0 and collisions pinned to -1::

       A_inter(i) = clamp(r_i - mean_i r, min=0)   , then  A_inter(i) = -1 where collided

   The truncation is the load-bearing half: it stops the update from actively
   pushing probability mass INTO the mass of low-quality modes (DDv2's stated
   defect — IL "neglects to impose any explicit constraints on trajectories
   sampled from the negative modes, which constitute the vast majority"),
   while still hard-penalising the one failure that is never acceptable.

⛔ A GROUP OF ONE HAS NO ADVANTAGE. With G=1 the centred advantage is
identically 0 and the update is a no-op — silently. `grpo_advantage` REFUSES
G < 2 rather than returning zeros, because a silent no-op that still exits 0 is
the false-green class this week's traps name explicitly.

Evidence class: the DDv2 mechanism is PUBLISHED (see the research RESULT.md);
this implementation is MEASURED (ours) and pinned by tests/test_rl_advantage.py.
"""

from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-8


def grpo_advantage(reward: Tensor, *, normalize: str = "none") -> Tensor:
    """Intra-group (per-anchor) centred advantage. ``reward`` is ``[..., G]``.

    ``normalize``:
      * ``"none"``  — centre only. **The default, and the Dr. GRPO-correct one.**
      * ``"std"``   — additionally divide by the group std. Opt-in and named so
        that choosing it is a decision someone made, not a default they inherited.

    Returns ``[..., G]``, mean-zero along the group axis.
    """
    if normalize not in ("none", "std"):
        raise ValueError(f"normalize must be 'none' or 'std', got {normalize!r}")
    g = reward.shape[-1]
    if g < 2:
        raise ValueError(
            f"grpo_advantage needs a group of >=2 to have any signal, got G={g}. "
            "A group of one yields an identically-zero advantage and a SILENT "
            "no-op update; refusing instead of returning zeros.")
    adv = reward - reward.mean(dim=-1, keepdim=True)
    if normalize == "std":
        adv = adv / (reward.std(dim=-1, keepdim=True) + EPS)
    return adv


def truncated_inter_anchor_advantage(reward: Tensor, *,
                                     veto: Tensor | None = None,
                                     veto_value: float = -1.0) -> Tensor:
    """Cross-anchor advantage with negatives truncated to 0. ``reward`` [..., N].

    Negatives -> 0 (do not push mass into the low-quality modes); VETOED
    candidates -> ``veto_value``.

    ⛔ ``veto`` IS THE CONSTRAINT CHANNEL, AND IT IS DELIBERATELY NOT A REWARD
    TERM. It carries collision AND TTC-imminent — failures that are never
    acceptable at any ranking. Applying it HERE, after the centring, is what
    makes it a constraint: a vetoed candidate is PINNED rather than merely
    ordered below its neighbours, so no amount of good behaviour elsewhere in
    the trajectory can buy it back.

    ⚠️ Why the separation is load-bearing: the first ``headway`` term tried to be
    a constraint and a ranking signal at once. It saturated at its bound and A0
    measured a median spread of **0.0000** across the fan — present, and
    carrying no signal. Constraints pin; ranking signals order. Fusing them
    produces something that does neither.
    """
    adv = (reward - reward.mean(dim=-1, keepdim=True)).clamp_min(0.0)
    if veto is not None:
        if veto.shape != reward.shape:
            raise ValueError(f"veto {tuple(veto.shape)} must match "
                             f"reward {tuple(reward.shape)}")
        adv = torch.where(veto, torch.full_like(adv, veto_value), adv)
    return adv


def composite_advantage(reward_ig: Tensor, *,
                        veto_ig: Tensor | None = None,
                        w_intra: float = 1.0, w_inter: float = 1.0,
                        normalize: str = "none") -> dict[str, Tensor]:
    """DDv2's composition on a ``[..., N, G]`` reward (N anchors x G samples).

    Returns the two parts and their weighted sum, so a caller can log and audit
    each independently — an aggregate that hides which half moved is how a
    +0.9/+0.6 ablation becomes unattributable.
    """
    if reward_ig.dim() < 2:
        raise ValueError(f"expected [..., N, G], got {tuple(reward_ig.shape)}")
    intra = grpo_advantage(reward_ig, normalize=normalize)          # [..., N, G]
    per_anchor = reward_ig.mean(dim=-1)                              # [..., N]
    veto_n = veto_ig.any(dim=-1) if veto_ig is not None else None
    inter = truncated_inter_anchor_advantage(per_anchor, veto=veto_n)
    total = w_intra * intra + w_inter * inter.unsqueeze(-1)
    return {"intra": intra, "inter": inter, "total": total,
            "per_anchor_reward": per_anchor}


def policy_gradient_loss(logp: Tensor, advantage: Tensor, *,
                         logp_ref: Tensor | None = None,
                         kl_coef: float = 0.0) -> dict[str, Tensor]:
    """REINFORCE-style surrogate ``-(A * logp)`` with an optional KL anchor.

    ⚠️ ``advantage`` is DETACHED here, always. An advantage that carries
    gradient turns the estimator into something that is not a policy gradient,
    and the failure is silent — the loss still decreases.

    The KL term is the k3 estimator (``exp(d) - d - 1``, d = logp_ref - logp),
    which is non-negative and lower-variance than the naive ``-d``. ⛔ It is
    OFF by default (``kl_coef=0``): DDv2 regularises toward the IL cold start
    with an imitation LOSS rather than a KL to a frozen reference, and adding
    both would double-count the anchor. Turn it on deliberately or not at all.
    """
    if logp.shape != advantage.shape:
        raise ValueError(f"logp {tuple(logp.shape)} vs advantage "
                         f"{tuple(advantage.shape)} must match")
    adv = advantage.detach()
    pg = -(adv * logp).mean()
    out = {"pg": pg, "loss": pg}
    if kl_coef:
        if logp_ref is None:
            raise ValueError("kl_coef != 0 requires logp_ref")
        d = (logp_ref.detach() - logp)
        kl = (torch.exp(d) - d - 1.0).mean()
        out["kl"] = kl
        out["loss"] = pg + kl_coef * kl
    return out
