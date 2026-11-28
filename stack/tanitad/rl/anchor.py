"""Reference-policy anchor — the TRUST REGION, not an imitation loss.

⭐ WHY THIS MODULE EXISTS (Master Mind ruling, 2026-08-29)
----------------------------------------------------------
Pilot P-RC21's P1 arm ran with NO policy anchor and the deployed trajectory
drifted while the rule-based reward rose — the classic unconstrained
policy-gradient failure (`RESULT_P_RC21.md` §2). I proposed the remedy as an
IMITATION loss against labels. That was the wrong object:

| | imitation loss | **reference-policy anchor (this module)** |
|---|---|---|
| pulls toward | LABELS (the logged expert) | the **frozen cold-start POLICY** |
| needs | a GT join, per corpus | **nothing** — label-free |
| transfers | must be redone per dataset | **unchanged**, any corpus/model |
| is what GRPO specifies | no | **yes** — the KL/trust region term |

⛔ **The two are NOT synonyms**, and treating them as such is how a design change
reads as a paraphrase. Secondary summaries of the RLHF/GRPO family use the names
interchangeably, so this is a common conflation rather than a private error — it
cost a measured +69.5 % sel-ADE drift here. `VOCABULARY.md` now requires that a
document saying "anchor" say WHICH.

CONSTRUCTION
------------
A deepcopy of the trainable submodule at cold start, ``requires_grad=False``,
never updated — reusing the proven `tanitad.models.v6._EmaCopy(src, decay=1.0)`
pattern from MM-E4's `--o5-target frozen` rather than re-deriving frozen-copy
machinery. Decay 1.0 means ``update()`` is a no-op even if someone wires it, and
we never wire it.

⚠️ **The anchor is a REGULARISER, not a reward term.** It lives OUTSIDE the
group-relative advantage — like the veto, and for the same reason: it constrains
the policy rather than ranking candidates. Putting it inside the advantage would
make it a fan-collapse objective (the mistake `rewards.DEFAULT_WEIGHTS` documents
for `gt_similarity`).
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn


class ReferencePolicy(nn.Module):
    """A frozen copy of a module, used only as a divergence target.

    ⛔ Every parameter is ``requires_grad=False`` and must stay out of every
    optimiser: the anchor is a target, never a second trainable path. Verified
    by ``assert_frozen``.
    """

    def __init__(self, src: nn.Module):
        super().__init__()
        self.module = copy.deepcopy(src)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()
        # same-config contract, the MM-E4 guard: a deepcopy cannot diverge in
        # shape today, but it can if construction is ever refactored.
        ds = [tuple(p.shape) for p in self.module.parameters()]
        ss = [tuple(p.shape) for p in src.parameters()]
        if ds != ss:
            raise RuntimeError("reference-policy shape mismatch — the "
                               "same-config contract is broken")

    def assert_frozen(self) -> int:
        n = 0
        for name, p in self.module.named_parameters():
            if p.requires_grad:
                raise RuntimeError(
                    f"reference-policy parameter {name} requires grad — the "
                    "anchor is a TARGET, never a second trainable path")
            n += p.numel()
        return n

    @torch.no_grad()
    def forward(self, *a, **kw):
        return self.module(*a, **kw)


def trajectory_divergence(traj: Tensor, ref_traj: Tensor, *,
                          form: str = "l2") -> Tensor:
    """Per-candidate divergence between the live fan and the reference fan.

    ``traj``/``ref_traj`` are ``[..., S, 2]`` on the SAME inputs and anchors, so
    the correspondence is positional — candidate *i* against reference *i*.

    * ``"l2"``   — mean squared displacement, in metres². Scale-honest and needs
      no density; the default because the emitted object IS a trajectory.
    * ``"l1"``   — mean absolute displacement; less dominated by one bad step.

    ⚠️ A KL form would need the decoder's own step density, which we do not have
    (the sampler is an explicit Gaussian surrogate — see ``refcv3_adapter``).
    Offering a "kl" name backed by the surrogate's density would be a claim we
    cannot support, so it is deliberately ABSENT rather than approximated.
    """
    if form not in ("l2", "l1"):
        raise ValueError(f"form must be 'l2' or 'l1', got {form!r} "
                         "(a KL form needs the true diffusion density, which "
                         "this library does not have — see the docstring)")
    if traj.shape != ref_traj.shape:
        raise ValueError(f"traj {tuple(traj.shape)} vs ref {tuple(ref_traj.shape)} "
                         "must match — the anchor compares candidate-to-candidate "
                         "on the same inputs")
    d = traj - ref_traj
    if form == "l1":
        return d.abs().sum(dim=-1).mean(dim=-1)
    return d.pow(2).sum(dim=-1).mean(dim=-1)


def anchor_penalty(traj: Tensor, ref_traj: Tensor, *, w: float,
                   form: str = "l2") -> dict[str, Tensor]:
    """``w * mean(divergence)`` plus the raw divergence for logging.

    Returned separately from the loss so an ablation can attribute movement to
    the anchor rather than to the reward — an aggregate that hides which half
    moved is how a sweep becomes uninterpretable.
    """
    div = trajectory_divergence(traj, ref_traj, form=form)
    return {"divergence": div, "penalty": w * div.mean()}
