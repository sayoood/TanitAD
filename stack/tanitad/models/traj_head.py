"""Direct multi-horizon trajectory head (fix-B1 ABLATION, ``--mode head``).

This is the *contrast* arm, NOT the primary fix. It regresses ego-frame
waypoints at horizons {5,10,15,20} (0.5/1/1.5/2 s @10 Hz) directly from the
encoder state (optionally concatenated with the operative predictor's per-horizon
predicted states) in ONE shot. It leans toward supervised imitation and does not
reshape the action-conditioned dynamics — kept only so the dynamics-grounding
result (``--mode dynamics``, :mod:`tanitad.models.metric_dynamics`) can be
compared against a plain trained readout under identical fine-tune budget.

Per horizon: LayerNorm -> Linear -> GELU -> Linear -> 2 (small, independent MLP).
Standalone ``nn.Module`` (never registered on ``WorldModel``) so a checkpoint's
``["model"]`` stays a vanilla ``WorldModel`` state dict.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

DEFAULT_HORIZONS = (5, 10, 15, 20)          # 0.5/1/1.5/2 s @ 10 Hz


class TrajectoryHead(nn.Module):
    """Encoder state (+ optional predictor states) -> ego waypoints per horizon.

    Args:
        state_dim: dim S of the compact encoder/readout state.
        horizons: waypoint steps (1-based, latent-transition units).
        hidden: per-horizon MLP width.
        n_extra_states: number of additional per-horizon predictor states that
            will be concatenated to the encoder state at ``forward`` time. When
            > 0, gradients from the waypoint loss also reach the predictor. The
            input dim of each head is ``state_dim * (1 + n_extra_states)``.
    """

    def __init__(self, state_dim: int, horizons=DEFAULT_HORIZONS,
                 hidden: int = 256, n_extra_states: int = 0):
        super().__init__()
        self.horizons = tuple(horizons)
        self.n_extra_states = n_extra_states
        in_dim = state_dim * (1 + n_extra_states)
        self.in_dim = in_dim
        self.heads = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden), nn.GELU(),
                nn.Linear(hidden, 2),
            ) for k in self.horizons
        })

    def forward(self, state: Tensor,
                extra_states: list[Tensor] | None = None) -> Tensor:
        """``state`` ``[B, S]`` (+ optional list of ``[B, S]`` predictor states) ->
        waypoints ``[B, len(horizons), 2]`` in the ego frame."""
        if self.n_extra_states:
            assert extra_states is not None and len(extra_states) == self.n_extra_states, \
                f"expected {self.n_extra_states} extra states, got " \
                f"{0 if extra_states is None else len(extra_states)}"
            feat = torch.cat([state, *extra_states], dim=-1)
        else:
            feat = state
        return torch.stack([self.heads[str(k)](feat) for k in self.horizons], dim=1)
