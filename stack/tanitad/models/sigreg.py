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


class SigReg(torch.nn.Module):
    """Sliced Epps-Pulley loss over M freshly-sampled random directions."""

    def __init__(self, n_slices: int = 512, beta: float = 1.0):
        super().__init__()
        self.n_slices = n_slices
        self.beta = beta

    def forward(self, z: Tensor) -> Tensor:
        """z: [n, d] (flatten any leading dims before calling). Returns scalar."""
        if z.ndim != 2:
            z = z.reshape(-1, z.shape[-1])
        n, d = z.shape
        # Fresh random directions every call (never a fixed buffer).
        dirs = torch.randn(d, self.n_slices, device=z.device, dtype=z.dtype)
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
