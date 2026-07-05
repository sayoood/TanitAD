"""Inverse dynamics head: (z_t, z_{t+1}) -> action (A5).

Forces the controllable state into the compact latent — the cheapest and most
effective grounding signal we have (ego-motion is free proprioception:
CAN/IMU/odometry). Also the seed of the H7 inverse-dynamics model that will
pseudo-label action-free video in Phase 1.
"""

from __future__ import annotations

from torch import Tensor, nn
import torch


class InverseDynamicsHead(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, z_t: Tensor, z_next: Tensor) -> Tensor:
        return self.net(torch.cat([z_t, z_next], dim=-1))
