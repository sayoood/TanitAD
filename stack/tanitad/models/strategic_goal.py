"""Strategic goal-scalar head (§3.1 / §4.3 of V4_FLAGSHIP_DESIGN).

The strategic planner ① emits, alongside its route/exit choice, four per-window
goal SCALARS the operative planner can be conditioned on and that make the
strategic goal *time-varying* rather than a static class: time-to-maneuver,
curvature@3 s, curvature@5 s, target-speed@5 s (§7A.4). This module is the small
regression head that produces them.

⚠️ SCOPE. This is the head ONLY. The full strategic planner ① — ``E_strat``
(2048→128 compression), the 128-d strategic predictor, the discrete
strategic-action set and the imagined-rollout option evaluator — is P6 and lands
separately (V4_FLAGSHIP_DESIGN §7A.3–7A.5, build plan §15 P5d). In the finished
stack this head reads ``z_strat`` (``in_dim=d_strat=128``, ~17 k params, §3.1); it
is written with a configurable ``in_dim`` so P6 can point it at ``z_strat`` while
the label-wiring proof (train_flagship_v4.real_smoke) points it at the operative
readout state. What it exists to prove NOW: the minted goal-scalar labels are real
(non-IGNORE) and reach a trainable loss — the ``strategic_scalar_loss`` term
(``tanitad.train.v4_curriculum``) is non-zero and pushes gradient into this head.

The label side is ``scripts/v4_labels.py`` (kinematic mint from poses); the loss
side is ``v4_curriculum.strategic_scalar_loss``; the column order for both is
``v4_labels.STRAT_SCALAR_NAMES``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

N_STRAT_SCALARS = 4             # ttm · curvature@3 s · curvature@5 s · tspeed@5 s


@dataclass
class GoalScalarConfig:
    in_dim: int = 128           # d_strat in the finished stack (§3.1); the proof
                                # passes the operative state_dim (2048)
    hidden: int = 128
    n_out: int = N_STRAT_SCALARS


class GoalScalarHead(nn.Module):
    """``in_dim`` context vector -> the ``N_STRAT_SCALARS`` goal scalars, in the
    fixed physical-unit column order ``v4_labels.STRAT_SCALAR_NAMES``. A plain
    2-layer MLP (the §3.1 geometry): the head's job is regression, the reasoning
    lives in the strategic predictor upstream (P6)."""

    def __init__(self, cfg: GoalScalarConfig | None = None):
        super().__init__()
        self.cfg = cfg or GoalScalarConfig()
        self.net = nn.Sequential(
            nn.Linear(self.cfg.in_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.n_out),
        )

    def forward(self, z: Tensor) -> Tensor:
        """``z`` [B, in_dim] -> [B, N_STRAT_SCALARS] predicted goal scalars."""
        return self.net(z)


def param_count(head: GoalScalarHead) -> int:
    return sum(p.numel() for p in head.parameters())
