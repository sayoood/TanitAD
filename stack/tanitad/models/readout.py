"""Spatial grid readout and frozen calibrated probes.

A7: spatial readout >> global pooling for position-faithful compact state.
A3 (calibrated-decode doctrine): probes that read *imagination* must be fitted
on the predictor's own imagined latents, not on real-frame encodings — the
systematic manifold shift of imagination is absorbed into the probe weights
(measured 0.97 vs 0.66 direction accuracy on identical predictions).

Probes are fitted offline (closed-form ridge) and frozen. They are the minimal
form of the H13 extraction heads and carry zero training burden.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpatialGridReadout(nn.Module):
    """Token grid [B, N, D] -> compact state [B, G*G*d_r] preserving layout."""

    def __init__(self, n_tokens: int, d_model: int, grid: int = 4, d_readout: int = 32):
        super().__init__()
        hw = int(n_tokens ** 0.5)
        assert hw * hw == n_tokens, "readout expects a square token grid"
        assert hw % grid == 0, "token grid must be divisible by readout grid"
        self.hw, self.grid = hw, grid
        self.pool = nn.AvgPool2d(hw // grid)
        self.proj = nn.Linear(d_model, d_readout)
        self.out_dim = grid * grid * d_readout

    def forward(self, tokens: Tensor) -> Tensor:
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, self.hw, self.hw)
        x = self.pool(x)                                  # [B, D, G, G]
        x = x.flatten(2).transpose(1, 2)                  # [B, G*G, D]
        return self.proj(x).flatten(1)                    # [B, G*G*d_r]


class RidgeProbe:
    """Frozen linear readout fitted in closed form: W = (X'X + a I)^-1 X'Y.

    Fit either on real encodings (probe_real: reads current state) or on
    imagined latents (probe_imag: reads imagination — A3 calibration).
    Never trained in-loop; never receives gradients.
    """

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha
        self.W: Tensor | None = None
        self.b: Tensor | None = None

    def fit(self, feats: Tensor, targets: Tensor) -> "RidgeProbe":
        """feats [n, f], targets [n, t]. Solves ridge with intercept."""
        with torch.no_grad():
            x = feats.double()
            y = targets.double()
            x_mean, y_mean = x.mean(0), y.mean(0)
            xc, yc = x - x_mean, y - y_mean
            f = xc.shape[1]
            gram = xc.T @ xc + self.alpha * torch.eye(f, dtype=x.dtype, device=x.device)
            self.W = torch.linalg.solve(gram, xc.T @ yc)          # [f, t]
            self.b = y_mean - x_mean @ self.W
        return self

    def predict(self, feats: Tensor) -> Tensor:
        assert self.W is not None, "probe not fitted"
        with torch.no_grad():
            return (feats.double() @ self.W + self.b).to(feats.dtype)

    def r2(self, feats: Tensor, targets: Tensor) -> float:
        pred = self.predict(feats)
        ss_res = (pred - targets).pow(2).sum()
        ss_tot = (targets - targets.mean(0)).pow(2).sum().clamp_min(1e-12)
        return float(1.0 - ss_res / ss_tot)
