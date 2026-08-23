"""SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA).

Constrains embeddings toward an isotropic Gaussian, provably preventing
representation collapse without EMA / stop-gradient / teacher-student
heuristics (Balestriero & LeCun, arXiv:2511.08544).

Mechanism (Cramer-Wold): a distribution is N(0, I) iff every 1-D projection is
N(0, 1). We draw M random unit directions (resampled every call — prevents
adversarial anisotropic collapse), project the batch, and score each projection
with the Epps-Pulley normality statistic, whose gradients are uniformly bounded.

Validated in ALPS-4B: apply to encoder embeddings AND predictor outputs at all
hierarchy levels; lambda = 0.1; slices = 512. Known historical bug to never
repeat: dividing the statistic by n cancels its built-in scale and silently
disables the loss (see ALPS-4B PROJECT_HANDOFF).

⛔ THE DIRECTION DRAW IS THE PROGRAMME'S ONLY UN-SEEDED RNG IN A LOSS PATH.
MEASURED 2026-08-16 (``LOSS_DETERMINISM.md``): ``_forward_fp32`` drew its M
directions from the GLOBAL RNG, so two ``v6_loss_step`` calls with the SAME
``generator`` disagreed — S-W total 3.9301 vs 3.9227, localised ENTIRELY to
``o6`` (0.046874 vs 0.039470, 18.7 %). A full training run is unaffected
(``train()`` seeds globally), but every IN-PROCESS A/B is — which is exactly
what an ablation harness is, so no ablation of any term was measurable above
its own noise. ``generator=`` (added 2026-08-16) fixes that.

⚠️ ``generator=None`` is the INCUMBENT and stays so: it draws from the global
RNG with the identical call, because v6F S-W is training from this code and its
per-step loss values must not move. Reproducibility is OPT-IN, per caller.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def epps_pulley(y: Tensor, beta: float = 1.0) -> Tensor:
    """Epps-Pulley test statistic of y (shape [n]) against N(0, 1).

    T = (1/n) * sum_{j,k} exp(-b^2 (y_j - y_k)^2 / 2)
        - 2 (1+b^2)^{-1/2} * sum_j exp(-b^2 y_j^2 / (2 (1+b^2)))
        + n (1+2 b^2)^{-1/2}

    Under H0 (standard normal) T is O(1); it grows with departure from
    normality. Differentiable with uniformly bounded gradients.
    """
    n = y.shape[0]
    b2 = beta * beta
    diff = y.unsqueeze(0) - y.unsqueeze(1)
    t1 = torch.exp(-0.5 * b2 * diff.pow(2)).sum() / n
    t2 = 2.0 / math.sqrt(1.0 + b2) * torch.exp(-0.5 * b2 * y.pow(2) / (1.0 + b2)).sum()
    t3 = n / math.sqrt(1.0 + 2.0 * b2)
    return t1 - t2 + t3


def sample_directions(d: int, n_slices: int, like: Tensor,
                      generator: torch.Generator | None = None) -> Tensor:
    """The M raw slice directions, ``[d, n_slices]``, on ``like``'s device/dtype.

    ⛔ ``generator is None`` MUST stay the literal incumbent call — same
    function, same argument order, same global stream — or a resumed v6F run
    stops reproducing the loss it was trained with. The branch below is
    deliberately not "unified" for tidiness.

    When a ``generator`` IS given and it lives on another device than ``like``
    (the normal case: a CPU ``torch.Generator`` feeding CUDA activations, since
    ``torch.randn(device='cuda', generator=<cpu gen>)`` raises), we draw on the
    GENERATOR's device and move. That is the same pattern
    :func:`tanitad.models.v6.sample_cell_block_mask` already uses, and it makes
    a run's O6 stream independent of which device it happens to land on.
    """
    if generator is None:
        return torch.randn(d, n_slices, device=like.device, dtype=like.dtype)
    if generator.device.type == like.device.type:
        return torch.randn(d, n_slices, device=like.device, dtype=like.dtype,
                           generator=generator)
    return torch.randn(d, n_slices, device=generator.device, dtype=like.dtype,
                       generator=generator).to(like.device)


class SigReg(torch.nn.Module):
    """Sliced Epps-Pulley loss over M freshly-sampled random directions."""

    def __init__(self, n_slices: int = 512, beta: float = 1.0):
        super().__init__()
        self.n_slices = n_slices
        self.beta = beta

    def forward(self, z: Tensor, *,
                generator: torch.Generator | None = None) -> Tensor:
        """z: [n, d] (flatten any leading dims before calling). Returns scalar.

        Always computed in fp32: the Epps-Pulley statistic is a difference of
        exponential sums — bf16 autocast would eat the signal. Gradients flow
        back to the (possibly lower-precision) embeddings unchanged.

        ``generator`` (default ``None`` = the incumbent global RNG) makes the
        direction draw reproducible, which is what an in-process A/B needs.
        """
        if z.ndim != 2:
            z = z.reshape(-1, z.shape[-1])
        if z.is_cuda:
            with torch.autocast("cuda", enabled=False):
                return self._forward_fp32(z.float(), generator)
        return self._forward_fp32(z.float(), generator)

    def _forward_fp32(self, z: Tensor,
                      generator: torch.Generator | None = None) -> Tensor:
        n, d = z.shape
        # Fresh random directions every call (never a fixed buffer).
        dirs = sample_directions(d, self.n_slices, z, generator)
        dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-8)
        proj = z @ dirs  # [n, M]
        # Vectorized Epps-Pulley across slices.
        b2 = self.beta * self.beta
        diff = proj.unsqueeze(0) - proj.unsqueeze(1)          # [n, n, M]
        t1 = torch.exp(-0.5 * b2 * diff.pow(2)).sum(dim=(0, 1)) / n   # [M]
        t2 = (2.0 / math.sqrt(1.0 + b2)
              * torch.exp(-0.5 * b2 * proj.pow(2) / (1.0 + b2)).sum(dim=0))  # [M]
        t3 = n / math.sqrt(1.0 + 2.0 * b2)
        stat = t1 - t2 + t3                                    # [M]
        # Do NOT normalize by n: the statistic's built-in batch-scale is part of
        # the validated (lambda=0.1, slices=512) operating point. Dividing by n
        # here was the historical ALPS-4B bug that silently disabled the loss.
        return stat.mean()


class SubspaceSigReg(torch.nn.Module):
    """Sub-JEPA's SUBSPACE Gaussian regularizer (arXiv 2605.09241, banked).

    ⛔ THE CRITIQUE IT IMPLEMENTS, verbatim from the paper: *"latent
    representations inherently lie on low-dimensional manifolds within a
    high-dimensional ambient space, and enforcing an isotropic Gaussian prior
    directly in this ambient space introduces an overly strong bias."* Sub-JEPA
    applies the Gaussian constraint in K random subspaces instead, which
    *"relaxes the global constraint while preserving its anti-collapse effect"*.

    THE METHOD (paper §Subspace Projection): K row-orthonormal projections
    ``P_k in R^{d_s x D}`` with ``d_s = round(D / K)`` — one hyperparameter.
    Each is built from a random Gaussian matrix via QR, and the projections are
    FROZEN: *"Freezing the projections prevents the regularizer itself from
    adapting to the evolving latent distribution."* LeWM's sliced Epps-Pulley
    then runs INDEPENDENTLY inside each subspace.

    ⭐ IMPLEMENTATION NOTE. With ``d_s = D/K`` and mutual orthogonality, the K
    projections together are exactly ONE ``D x D`` orthogonal matrix cut into K
    row-blocks — so one QR builds all of them, and the blocks are orthogonal to
    each other by construction, which is what the paper asks for.

    ⭐⭐ WHY THIS COMPOSES WITH OUR OWN FIX. MEASURED 2026-08-22: v6's O6 sees
    ``B*W = 24`` rows in ``D = 2048`` (n/d = 0.012), so its Epps-Pulley estimate
    is mostly sampling noise — the same ``n << d`` defect the O6 rank GATE
    refuses to rule on. ``SigRegRowBank`` raises n; this lowers d. Together,
    n=1536 rows in ``d_s=64`` is n/d_s = 24 — a ~2000x better-conditioned
    estimator than the incumbent, from two independent directions.

    ⚠️ Sub-JEPA is validated on four CONTINUOUS-CONTROL environments, not on
    driving video. Same scope caveat as LeWM itself.
    """

    def __init__(self, dim: int, n_subspaces: int, n_slices: int = 512,
                 beta: float = 1.0, seed: int = 0):
        super().__init__()
        if n_subspaces < 1:
            raise ValueError(f"n_subspaces must be >= 1, got {n_subspaces}")
        if n_subspaces > dim:
            raise ValueError(f"n_subspaces {n_subspaces} > dim {dim}")
        self.dim, self.n_subspaces = int(dim), int(n_subspaces)
        self.n_slices, self.beta = int(n_slices), float(beta)
        self.d_s = int(round(dim / n_subspaces))
        g = torch.Generator().manual_seed(int(seed))
        q, _r = torch.linalg.qr(torch.randn(dim, dim, generator=g))
        # [K, d_s, D] row-orthonormal blocks of one orthogonal basis
        blocks = q.t()[: self.n_subspaces * self.d_s]
        self.register_buffer("proj",
                             blocks.reshape(self.n_subspaces, self.d_s, dim))

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, K={self.n_subspaces}, d_s={self.d_s}, "
                f"slices={self.n_slices}")

    def forward(self, z: Tensor, *,
                generator: torch.Generator | None = None) -> Tensor:
        if z.ndim != 2:
            z = z.reshape(-1, z.shape[-1])
        if z.is_cuda:
            with torch.autocast("cuda", enabled=False):
                return self._forward_fp32(z.float(), generator)
        return self._forward_fp32(z.float(), generator)

    def _forward_fp32(self, z: Tensor,
                      generator: torch.Generator | None = None) -> Tensor:
        inner = SigReg(self.n_slices, self.beta)
        P = self.proj.to(device=z.device, dtype=z.dtype)
        # ⚠️ the Epps-Pulley statistic itself is REUSED, never reimplemented:
        # its no-divide-by-n behaviour is the validated operating point and was
        # once a silent-disable bug.
        stats = [inner._forward_fp32(z @ P[k].t(), generator)
                 for k in range(self.n_subspaces)]
        return torch.stack(stats).mean()


def position_relaxed(sigreg: "SigReg", z: Tensor, free_dims: int, *,
                     generator: torch.Generator | None = None) -> Tensor:
    """SIGReg on the COMPLEMENT of a fixed ego-motion subspace (§B.3 relaxation).

    Metric ego-position lives in a low-dimensional, structured (non-isotropic)
    subspace; plain SIGReg drives the WHOLE embedding toward an isotropic
    Gaussian and so actively whitens exactly that structure — the two objectives
    partially cancel (the diagnosed step-21k regression mechanism). The remedy:
    EXEMPT a fixed ``free_dims``-wide subspace (here the first ``free_dims``
    state columns, the reserved ego-motion channels) from SIGReg and apply the
    anti-collapse constraint only to the complement.

    The exempt columns receive EXACTLY zero SIGReg gradient (they are not passed
    to the statistic), so the grounding losses are free to route low-dimensional
    metric-position structure there without SIGReg fighting it, while the
    complement is still held against collapse. ``free_dims <= 0`` reproduces
    plain SIGReg on the full latent. Pure function (no params) — importable and
    unit-testable in isolation, and shared by the flagship and REF-A trainers.

    ``generator`` is forwarded to :meth:`SigReg.forward`; ``None`` (the default,
    and what every incumbent caller passes by omission) is the global RNG.
    """
    z = z.reshape(-1, z.shape[-1])
    d = z.shape[-1]
    if free_dims <= 0:
        return sigreg(z, generator=generator)
    if free_dims >= d:
        raise ValueError(f"sigreg free_dims={free_dims} must be < state dim {d} "
                         f"— the complement would be empty (no anti-collapse)")
    return sigreg(z[:, free_dims:], generator=generator)
