"""Measuring whether a chain-of-thought INFLUENCES the plan and stays CONSISTENT.

The requirement (PI, 2026-09-05) is that TanitLang's reasoning must *change the
behaviour* and *agree with the trajectory finally selected*. Those pull in
opposite directions -- a CoT that describes the chosen plan is perfectly
consistent and completely inert; a CoT that overrides the plan is influential and
unconstrained -- so both halves have to be **separately measurable**, and each
needs a control that reads a value known in advance.

⭐ THE STRUCTURE THIS MODULE ASSUMES (design: `DIALOGUE_07_TANITLANG_ENERGY_BRIDGE`).
One learned scalar `E(z, tau)` is read in two directions:

    influence     S1 = S0 - beta * E(z, tau_k)     over the whole anchor fan
    consistency   E(z, tau_selected) must be the MINIMUM over that fan

so a CoT that is a caption makes `E` constant in `z`, the re-ranking vanishes and
the influence metrics read exactly zero; a CoT that confabulates leaves
`E(z, tau_sel)` above the minimum and the consistency metrics say so. Neither
failure can hide behind the other because both are readings of one function.

⛔ WHY THE CONTROLS ARE NOT OPTIONAL HERE. The programme has already shipped one
metric that measured an echo and called it skill: flagship v1's route head scored
**1.0000** because it was an exact bijection of the nav signal fed to it (369/369
and 81/81). A consistency metric with no `shuffled-z` control would score a
caption generator perfectly. So the control functions live in this module beside
the metrics, not in a caller that may forget them.

⚠️ SCOPE. Everything here is a *diagnostic over an already-computed fan*. None of
it is a driving-performance number: a capability claim needs T1 (the model
conditioned on its own actions) and the four metric families, per
`Project Steering/EVAL_DOCTRINE.md`.

Conventions
-----------
``s0``, ``s1``   selection logits ``[B, K]`` before / after the reasoner's ports
``fan``          candidate trajectories ``[B, K, S, 2]`` in the ego frame (metres)
``energy``       ``E(z, tau_k)`` as ``[B, K]``; LOWER means more compatible
``sel``          selected index ``[B]`` (``long``)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "InfluenceReport",
    "ConsistencyReport",
    "SemanticNullReport",
    "rerank_logits",
    "influence",
    "consistency",
    "roll_control",
    "constant_energy_floor",
    "norm_matched_null",
    "permuted_prior_null",
    "semantic_null_screen",
    "OutcomeNullReport",
    "outcome_null_screen",
    "echo_residual",
    "NoInfluenceViolation",
]


class NoInfluenceViolation(RuntimeError):
    """Raised when the beta=0 arm fails to reproduce the base policy exactly.

    This is an IDENTITY, not an approximation: with the guidance weight at zero
    the guided pass must return the base logits bit-for-bit. A programme that
    tolerates "close enough" here cannot later attribute a behaviour change to
    the reasoner, because the arm was never equal to its own baseline.
    """


@dataclass(frozen=True)
class InfluenceReport:
    """How much the reasoner moved the decision. All zero == no influence."""

    flip_rate: float          # fraction of windows whose argmax changed
    kl: float                 # mean KL(softmax s1 || softmax s0), nats
    geo_m: float              # mean ||tau_sel(s1) - tau_sel(s0)||, metres (ADE-style)
    n: int

    def is_inert(self, tol: float = 0.0) -> bool:
        return (self.flip_rate <= tol and self.kl <= tol and self.geo_m <= tol)


@dataclass(frozen=True)
class ConsistencyReport:
    """Whether the reasoning prefers the trajectory that was actually chosen."""

    mean_rank: float          # 1.0 == the selected candidate is the energy minimum
    top1_rate: float          # fraction of windows where it IS the minimum
    gap: float                # mean E(sel) - min_k E(k); 0.0 == perfect
    n: int
    k: int

    @property
    def chance_rank(self) -> float:
        """Mean rank a reasoner with NO preference would score: (K + 1) / 2."""
        return (self.k + 1) / 2.0


def rerank_logits(s0: Tensor, energy: Tensor, beta: float) -> Tensor:
    """``S1 = S0 - beta * E``.

    ⛔ ``beta`` is zero-initialised in the model for a reason: at step 0 the
    guided arm must be *bit-identical* to the base arm, so any later difference
    is attributable to the reasoner and to nothing else. A module that changes
    behaviour at initialisation cannot be compared against its own baseline.
    """
    if s0.shape != energy.shape:
        raise ValueError(f"s0 {tuple(s0.shape)} and energy {tuple(energy.shape)} "
                         f"must have the same [B, K] shape")
    if beta == 0.0:
        out = s0.clone()
        if not torch.equal(out, s0):
            raise NoInfluenceViolation("beta=0 did not reproduce s0 exactly")
        return out
    return s0 - beta * energy


def _selected_traj(fan: Tensor, s: Tensor) -> Tensor:
    idx = s.argmax(dim=-1)                                   # [B]
    b = torch.arange(fan.shape[0], device=fan.device)
    return fan[b, idx]                                       # [B, S, 2]


def influence(s0: Tensor, s1: Tensor, fan: Tensor | None = None) -> InfluenceReport:
    """Quantify the guidance delta ``s1 - s0``.

    ``geo_m`` is reported as **nan** when no fan is supplied, never as 0.0 --
    a metric that could not be computed must not be indistinguishable from a
    metric that was computed and came out at the no-effect value. That confusion
    is the same family as a search reporting "no matches" for a file it could
    not open.
    """
    if s0.shape != s1.shape:
        raise ValueError(f"s0 {tuple(s0.shape)} != s1 {tuple(s1.shape)}")
    if s0.ndim != 2:
        raise ValueError(f"expected [B, K] logits, got {tuple(s0.shape)}")
    flips = (s0.argmax(-1) != s1.argmax(-1)).float().mean().item()
    logp1 = torch.log_softmax(s1.float(), dim=-1)
    logp0 = torch.log_softmax(s0.float(), dim=-1)
    kl = (logp1.exp() * (logp1 - logp0)).sum(-1).mean().item()
    if fan is None:
        geo = float("nan")
    else:
        if fan.shape[:2] != s0.shape:
            raise ValueError(f"fan {tuple(fan.shape)} does not match logits "
                             f"{tuple(s0.shape)} on [B, K]")
        d = (_selected_traj(fan, s1) - _selected_traj(fan, s0)).norm(dim=-1)
        geo = d.mean().item()
    return InfluenceReport(flip_rate=flips, kl=kl, geo_m=geo, n=int(s0.shape[0]))


def consistency(energy: Tensor, sel: Tensor) -> ConsistencyReport:
    """Is the selected candidate the one the reasoning prefers?

    ``mean_rank`` is 1-based: 1.0 means the selected trajectory is the energy
    minimum in every window. Compare it against ``chance_rank`` == (K+1)/2 --
    a reasoner with no preference lands there, and reporting the rank without
    that reference invites reading K-dependent numbers as quality.

    Ties are resolved PESSIMISTICALLY (a tied candidate counts as ranked above
    the selected one), so a constant energy scores at the worst rank rather than
    at an accidental best. A metric must not flatter a degenerate model.
    """
    if energy.ndim != 2:
        raise ValueError(f"expected [B, K] energy, got {tuple(energy.shape)}")
    if sel.ndim != 1 or sel.shape[0] != energy.shape[0]:
        raise ValueError(f"sel {tuple(sel.shape)} does not match energy "
                         f"{tuple(energy.shape)} on B")
    b = torch.arange(energy.shape[0], device=energy.device)
    e_sel = energy[b, sel]                                    # [B]
    better = (energy <= e_sel.unsqueeze(1)).sum(dim=1).float()   # pessimistic ties
    gap = (e_sel - energy.min(dim=1).values)
    return ConsistencyReport(
        mean_rank=better.mean().item(),
        top1_rate=(better == 1).float().mean().item(),
        gap=gap.mean().item(),
        n=int(energy.shape[0]), k=int(energy.shape[1]))


def roll_control(z: Tensor, shift: int = 1) -> Tensor:
    """Pair each window's fan with ANOTHER window's reasoning state.

    ⛔ A ROLL, NOT A RANDOM PERMUTATION. A permutation leaves some elements at
    their own index with probability ~1/B, so a "shuffled" control can silently
    contain correctly-paired rows and drift toward the real arm. A roll by a
    non-zero shift guarantees **every** row is mispaired.

    ⛔ AND IT REFUSES B < 2. With one row a roll is the identity, so the control
    would return the real arm and read as a PASS. A control that cannot fail is
    not a control -- fail loudly instead.

    !!! AND IT IS THE WRONG CONTROL WHEN THE ROWS ARE GROUPED. A roll mispairs
    every row by INDEX, which is a meaningful control only when neighbouring rows
    are independent. MEASURED 2026-09-06 on a batch accumulated episode by
    episode: `roll_control(..., 1)` paired a window with the SAME EPISODE in
    **97.5 %** of rows -- usually the adjacent timestep, i.e. very nearly the
    CORRECT partner. It was ~97.5 % a no-op, and a conclusion drawn from it
    ("the port's effect is 89 % generic") had to be retracted.
    => When the batch has a grouping -- episode, clip, scene, subject -- the
    control must mispair ACROSS GROUPS, not across indices. Draw the partner from
    a different group and ASSERT that no same-group pair survives, so a silent
    failure to mispair cannot read as a pass.
    """
    b = z.shape[0]
    if b < 2:
        raise ValueError(f"roll_control needs batch >= 2 to mispair every row; "
                         f"got B={b}. With B=1 the control equals the real arm "
                         f"and would read as a pass.")
    if shift % b == 0:
        raise ValueError(f"shift {shift} is a multiple of B={b}: the roll would "
                         f"be the identity")
    return torch.roll(z, shifts=shift, dims=0)


def constant_energy_floor(energy: Tensor) -> Tensor:
    """Replace ``E(z, tau)`` by its per-window mean, killing all fan structure.

    The re-ranking this produces MUST be inert: ``rerank_logits`` with a constant
    energy adds the same scalar to every candidate, so the softmax and the argmax
    are unchanged. Asserting that identity is what proves the influence pathway
    really runs through the energy's *variation over candidates* and not through
    some incidental offset.
    """
    return energy.mean(dim=-1, keepdim=True).expand_as(energy).contiguous()


def echo_residual(z: Tensor, tau: Tensor, ridge: float = 1.0) -> float:
    """Fraction of ``z``'s variance NOT explained by a linear map from ``tau``.

    ⭐ THE FALSIFIABILITY TEST. A CoT that can be reconstructed from the answer
    is an echo of the answer, however fluent it reads. Fit ``tau -> z`` and
    report the residual: 1.0 means ``z`` carries information the trajectory does
    not, 0.0 means ``z`` is a re-encoding of ``tau``.

    ⚠️ THE IMPLICATION RUNS ONE WAY ONLY. A LOW residual proves echo. A HIGH
    residual proves only "not linearly reconstructible" -- it is not proof of
    reasoning content, and must be reported with the function class named.
    Establish a nonlinear probe with a time-shuffled control before making the
    stronger claim.

    Fitted in closed form on the SAME rows it scores, so this is an OPTIMISTIC
    bound on reconstructability, i.e. it is conservative in the direction that
    matters: it makes echo *easier* to detect, not harder.
    """
    if z.shape[0] != tau.shape[0]:
        raise ValueError(f"z has {z.shape[0]} rows, tau has {tau.shape[0]}")
    if z.shape[0] < 2:
        raise ValueError("echo_residual needs at least 2 rows to have variance")
    Z = z.reshape(z.shape[0], -1).double()
    A = tau.reshape(tau.shape[0], -1).double()
    A = torch.cat([A, torch.ones(A.shape[0], 1, dtype=A.dtype,
                                 device=A.device)], dim=1)
    W = torch.linalg.solve(
        A.T @ A + ridge * torch.eye(A.shape[1], dtype=A.dtype, device=A.device),
        A.T @ Z)
    resid = Z - A @ W
    denom = (Z - Z.mean(0, keepdim=True)).pow(2).sum()
    if denom <= 0:
        raise ValueError("z has zero variance; the residual is undefined")
    return float((resid.pow(2).sum() / denom).clamp(min=0.0))


# --------------------------------------------------------------------------
# THE SEMANTIC NULL — the control the published failure mode defeats
# --------------------------------------------------------------------------
# ⛔⛔ WHY THIS EXISTS, AND WHY `constant_energy_floor` ABOVE IS NOT ENOUGH.
# PUBLISHED (arXiv 2606.12706, VLADriveBench, PDF body read; banked): a driving
# VLA's chain-of-thought shortened its 3 s prediction on 81.9 % of steps by
# -0.437 m. Shuffling the CoT's WORDS reproduced that at 81.4 % / -0.429 m --
# indistinguishable. Replacing the CoT entirely with repeated copies of the
# token "depicts" produced a **STRONGER** effect: 90.4 % / -0.846 m, roughly 2x.
#
# ⇒ A LARGE GUIDANCE DELTA PROVES THE CHANNEL IS WIRED, NOT THAT THE REASONING
#   IS MEANINGFUL. Every metric in `influence()` above would have scored that
#   semantically empty prior as a success.
#
# ⚠️ And it says exactly why `constant_energy_floor` does not cover it: a mean
# energy is norm-REDUCED, so it under-states the intervention a content-free
# prior can make. The defeating null is norm-PRESERVED. An arm must beat a
# prior that intervenes just as hard as it does but says nothing.


@dataclass(frozen=True)
class SemanticNullReport:
    """Did the real reasoning beat priors that intervene as hard but mean nothing?"""

    real_kl: float
    norm_matched_kl: float
    permuted_kl: float
    n: int

    @property
    def beats_norm_matched(self) -> bool:
        return self.real_kl > self.norm_matched_kl

    @property
    def beats_permuted(self) -> bool:
        return self.real_kl > self.permuted_kl

    @property
    def passes(self) -> bool:
        """⛔ BOTH, not either. The published null beat the real CoT on one axis."""
        return self.beats_norm_matched and self.beats_permuted


def norm_matched_null(energy: Tensor, generator: torch.Generator | None = None
                      ) -> Tensor:
    """A content-free energy with the SAME per-window mean and spread.

    Random across candidates, so it re-ranks exactly as hard as the real energy,
    while carrying no information about the scene. Matching happens per window
    (per row), because the magnitude of the intervention is a per-window
    property and a globally-matched null would understate it in busy scenes and
    overstate it in empty ones.
    """
    if energy.ndim != 2:
        raise ValueError(f"expected [B, K] energy, got {tuple(energy.shape)}")
    if energy.shape[1] < 2:
        raise ValueError(f"a null over K={energy.shape[1]} candidates cannot "
                         f"re-rank anything; need K >= 2")
    z = torch.randn(energy.shape, generator=generator, dtype=energy.dtype,
                    device=energy.device)
    z = (z - z.mean(dim=1, keepdim=True)) / (z.std(dim=1, keepdim=True) + 1e-12)
    return z * energy.std(dim=1, keepdim=True) + energy.mean(dim=1, keepdim=True)


def permuted_prior_null(energy: Tensor, shift: int = 1) -> Tensor:
    """This window's fan scored by ANOTHER window's energy.

    Preserves the real energy's shape and scale exactly -- it IS a real energy,
    merely attached to the wrong scene -- so it is the sharper of the two nulls
    whenever the energy's structure matters more than its magnitude. Uses the
    same roll (never a random permutation) so every row is mispaired.
    """
    return roll_control(energy, shift=shift)


def semantic_null_screen(s0: Tensor, energy: Tensor, beta: float,
                         generator: torch.Generator | None = None,
                         shift: int = 1) -> SemanticNullReport:
    """Score the real energy against BOTH content-free priors, same beta.

    ⛔ This is a PRECONDITION for reading any influence number, not a follow-up
    check: an arm that fails it has demonstrated wiring and nothing else.

    ⚠️⚠️ AND IT MEASURES MAGNITUDE, WHICH IS NOT THE AXIS THAT DECIDES. This
    compares KL(S1 || S0) -- how far the distribution moved. The published
    failure mode is about the BEHAVIOURAL EFFECT, and a random direction
    generically diverges MORE than a structured one at matched spread, so a
    genuinely informative prior can LOSE this contest while being the only one
    that improves the outcome. MEASURED on REF-C's own trained
    `maneuver_to_anchor`: real KL 0.189 against a norm-matched null at 0.378 --
    the null "wins" by moving further in a random direction.
    ⇒ **Use :func:`outcome_null_screen` to decide.** Keep this one as the cheap
    screen it is: it can say a prior is inert, it cannot say a prior is empty.
    """
    real = influence(s0, rerank_logits(s0, energy, beta)).kl
    nm = influence(s0, rerank_logits(s0, norm_matched_null(energy, generator),
                                     beta)).kl
    pm = influence(s0, rerank_logits(s0, permuted_prior_null(energy, shift),
                                     beta)).kl
    return SemanticNullReport(real_kl=real, norm_matched_kl=nm, permuted_kl=pm,
                              n=int(s0.shape[0]))


@dataclass(frozen=True)
class OutcomeNullReport:
    """Did the real prior improve the OUTCOME more than content-free priors?"""

    real: float
    norm_matched: float
    permuted: float
    n: int
    lower_is_better: bool = True

    @property
    def passes(self) -> bool:
        """⛔ BOTH nulls, and on the metric -- not on the size of the nudge."""
        if self.lower_is_better:
            return self.real < self.norm_matched and self.real < self.permuted
        return self.real > self.norm_matched and self.real > self.permuted


def outcome_null_screen(s0: Tensor, energy: Tensor, beta: float,
                        outcome: Tensor, generator: torch.Generator | None = None,
                        shift: int = 1, lower_is_better: bool = True
                        ) -> OutcomeNullReport:
    """The screen that decides: real prior vs content-free priors, ON THE METRIC.

    ``outcome`` is ``[B, K]`` -- the value achieved if candidate ``k`` were
    selected (e.g. each anchor's ADE). Each arm re-ranks, selects its argmax and
    is scored by the outcome it actually chose, so all three arms are judged on
    the quantity the programme cares about rather than on how hard they pushed.

    ⭐ This is the honest form of the VLADriveBench test. There, a prior of
    repeated "depicts" tokens moved the plan ~2x as far as the real CoT; the
    question that matters is not which moved further but which moved it to a
    BETTER place.
    """
    if outcome.shape != energy.shape:
        raise ValueError(f"outcome {tuple(outcome.shape)} must match energy "
                         f"{tuple(energy.shape)} on [B, K]")
    b = torch.arange(s0.shape[0], device=s0.device)

    def _val(en):
        return float(outcome[b, rerank_logits(s0, en, beta).argmax(-1)].mean())

    return OutcomeNullReport(
        real=_val(energy),
        norm_matched=_val(norm_matched_null(energy, generator)),
        permuted=_val(permuted_prior_null(energy, shift)),
        n=int(s0.shape[0]), lower_is_better=lower_is_better)
