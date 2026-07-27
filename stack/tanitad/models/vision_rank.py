"""Vision enters at rank ~= 16 — the FLAT-state rank lever.

WHAT WAS MEASURED, AND WHAT IT DOES *NOT* SAY
----------------------------------------------
A linear probe on the **frozen 2048-d readout state** (``WorldModel.encode``'s
``SpatialGridReadout`` output), read alongside an ego window, shows a **monotone
swamping dose-response** in the number of principal components ``k`` admitted:

======================================  ==========  ====================
arm                                     AP / base   separated vs chance?
======================================  ==========  ====================
ego only (no image)                     3.659x      yes
ego + **k = 16**                        **3.685x**  yes  <- does not hurt
ego + k = 64                            3.000x      yes
ego + k = 256                           2.116x      yes
ego + k = 2048 (the raw concatenation)  1.59x       **NO**
======================================  ==========  ====================

Degradation begins at **k = 64**; **16 PCs carry 97.0 %** of the state's variance
(``cum16 = 0.96987``). **Replicated by a second, independent stream** on three
different targets with a different reader — all ten of its arms selected r=16
over r=64, and its raw-2048 concatenation arm was the worst of the ladder
(held-out ``concat - ego = -0.05567 [-0.08280, -0.03417]``).

⚠️ **THE CLAIM IS NARROW AND MUST NOT BE WIDENED.** k=16 is +0.026x over ego
alone, i.e. **indistinguishable from it**. The finding is *"at k = 16 vision
stops destroying the ego signal"*, **NOT** *"vision adds value"*. Anything built
here that is quoted as a vision *gain* is a misquote of the source.

⚠️ **SCOPE.** The measurement is a **linear probe on a frozen state**. It does
not by itself establish that a *trained non-linear* head with a learned 2048->d
projection degrades the same way. What it does establish is that the flat 2048-d
vector carries ~97 % of its variance in 16 directions and that every reader
measured so far did worse the more of the remaining 3 % it was handed. This
module therefore makes the rank an explicit, recorded choice — it does not claim
the reduction is free.

WHY THIS IS DECODE-SIDE
------------------------
v4 is at **2 of 2 encoder-touching levers** and ``encoder_touching_levers <= 2``
is a KILL secondary. A rank projection placed on the TRUNK would breach it. This
one sits between the trunk and the PLANNER's flat readers, so the encoder is
untouched and the lever count is unchanged.

WHAT "IMPOSSIBLE TO SELECT BY ACCIDENT" MEANS HERE
---------------------------------------------------
:func:`resolve_vision_rank` **raises** on the raw-2048 request unless the caller
passes BOTH ``allow_raw=True`` and a non-empty written ``reason``. A default
argument, a missing config key, a ``0``, a ``None`` or a rank at/above
``state_dim`` all land on the refusal — the accident modes, not just the
deliberate one.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["DEFAULT_VISION_RANK", "RAW_STATE_DIM", "DEGRADATION_ONSET_K",
           "VARIANCE_AT_16", "RawVisionRankRefused", "resolve_vision_rank",
           "VisionRankProjection", "LEGACY_RAW_REASON", "DOSE_RESPONSE"]

#: The shipped rank. 16 PCs, 97.0 % of the flat state's variance.
DEFAULT_VISION_RANK = 16
#: The flat readout state's width (4*4*128) — the raw concatenation's dim.
RAW_STATE_DIM = 2048
#: Where the MEASURED degradation begins. Ranks at or above this are warned about.
DEGRADATION_ONSET_K = 64
VARIANCE_AT_16 = 0.9699

#: The measured curve, carried WITH the lever so it can never be quoted bare.
DOSE_RESPONSE = {
    "metric": "AP / base-rate (lift over prevalence) of a linear probe on the "
              "frozen 2048-d readout state, read with an ego window",
    "ego_only": 3.659, "k16": 3.685, "k64": 3.000, "k256": 2.116, "k2048": 1.59,
    "k2048_separated_vs_chance": False,
    "explained_variance_cum16": VARIANCE_AT_16,
    "claim": "at k=16 vision stops DESTROYING the ego signal; k=16 is +0.026x "
             "over ego alone, i.e. indistinguishable from it. NOT 'vision adds "
             "value'.",
    "evidence_class": "MEASURED (ours; two independent streams, all ten arms of "
                      "the replication selecting r=16)",
}

LEGACY_RAW_REASON = (
    "legacy checkpoint: trained before the rank-16 lever existed, so its "
    "state_dict has no projection weights and its flat readers are 2048-wide. "
    "Loading it raw is a REPRODUCTION of an old arm, not a new design choice.")


class RawVisionRankRefused(ValueError):
    """A raw-2048 flat vision path was requested without an explicit override."""


def resolve_vision_rank(rank, state_dim: int = RAW_STATE_DIM, *,
                        allow_raw: bool = False, reason: str = "",
                        _what: str = "vision_rank") -> int:
    """Validate a requested rank and return it, or raise.

    Returns the effective rank ``r`` with ``1 <= r < state_dim``. ``rank in
    (None, 0)`` or ``rank >= state_dim`` all mean *"hand the reader the raw flat
    state"* — the arm MEASURED not separated from chance — and are refused unless
    ``allow_raw`` is set AND a written ``reason`` is supplied. Requiring the
    reason is the point: a boolean can be flipped absent-mindedly, a sentence
    cannot.
    """
    raw_requested = (rank is None or int(rank) <= 0 or int(rank) >= state_dim)
    if raw_requested:
        if allow_raw and reason.strip():
            return int(state_dim)
        raise RawVisionRankRefused(
            f"{_what}={rank!r} against state_dim={state_dim} requests the RAW "
            f"flat vision path. MEASURED: the raw-2048 arm is the ONLY rung of "
            f"the dose-response ladder NOT separated from chance "
            f"({DOSE_RESPONSE['k2048']}x vs {DOSE_RESPONSE['ego_only']}x for ego "
            f"alone), and an independent replication put its concat arm worst of "
            f"ten. The shipped rank is {DEFAULT_VISION_RANK} "
            f"({VARIANCE_AT_16:.1%} of variance). If you really want raw, pass "
            f"allow_raw=True AND a written reason — this cannot be reached by a "
            f"default, a missing config key, a 0 or a None.")
    r = int(rank)
    if r >= DEGRADATION_ONSET_K:
        # not an error — a measured warning, on the record in the run's own log
        print(f"[vision_rank] ⚠️ {_what}={r} is at or above the MEASURED "
              f"degradation onset k={DEGRADATION_ONSET_K} "
              f"({DOSE_RESPONSE['k64']}x vs {DOSE_RESPONSE['k16']}x at k=16). "
              f"This is admissible but it is a deliberate step down the ladder.",
              flush=True)
    return r


class VisionRankProjection(nn.Module):
    """Flat state ``[..., state_dim]`` -> ``[..., rank]``, or identity when raw.

    A plain bias-free linear map. It is TRAINED, not frozen: the measured basis
    is a PCA of one frozen encoder's state and this trunk moves during v4/v5
    training, so freezing a stale basis would pin the planner's view of vision to
    a checkpoint that no longer exists. :meth:`init_from_basis` lets a fitted PCA
    basis seed the map when one is available, which starts training at the
    measured operating point instead of at random.

    ``rank == state_dim`` (the explicitly-allowed raw path) makes this an exact
    identity with **zero parameters**, so an arm that opted into raw is
    bit-identical to having no projection at all — the reproduction claim.
    """

    def __init__(self, state_dim: int, rank: int):
        super().__init__()
        self.state_dim = int(state_dim)
        self.rank = int(rank)
        self.is_raw = self.rank >= self.state_dim
        self.out_dim = self.state_dim if self.is_raw else self.rank
        if not self.is_raw:
            self.proj = nn.Linear(self.state_dim, self.rank, bias=False)
            nn.init.orthogonal_(self.proj.weight)
        self.register_buffer("basis_loaded", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def init_from_basis(self, W: Tensor, mu: Tensor | None = None) -> None:
        """Seed from a fitted PCA basis ``W`` ``[state_dim, rank]``.

        The mean is folded into a buffer rather than a bias so the raw path stays
        parameter-free and the two paths cannot diverge in shape.
        """
        if self.is_raw:
            raise RawVisionRankRefused(
                "this projection is the explicitly-allowed RAW identity; a PCA "
                "basis has nothing to initialise.")
        W = torch.as_tensor(W, dtype=torch.float32)
        if tuple(W.shape) != (self.state_dim, self.rank):
            raise ValueError(f"basis {tuple(W.shape)} != "
                             f"({self.state_dim}, {self.rank})")
        self.proj.weight.copy_(W.t())
        if mu is not None:
            self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32)
                                 .reshape(-1)[: self.state_dim].clone())
        self.basis_loaded.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_raw:
            return x
        if hasattr(self, "mu"):
            x = x - self.mu
        return self.proj(x)

    def extra_repr(self) -> str:
        return (f"state_dim={self.state_dim}, rank={self.rank}, "
                f"raw={self.is_raw}, out_dim={self.out_dim}")
