"""Differentiable kinematic bicycle layer (H14 Track 1).

The trajectory head predicts *controls*; this layer integrates them through
explicit vehicle kinematics, so every decoded trajectory is physically
realizable by construction. Also provides the Kamm-circle friction penalty
(|a_lat^2 + a_lon^2| <= (mu g)^2) used as a standing metric and loss barrier.
"""

from __future__ import annotations

import torch
from torch import Tensor


def rollout_bicycle(state0: Tensor, controls: Tensor, dt: float = 0.1,
                    wheelbase: float = 2.7) -> Tensor:
    """Integrate the kinematic bicycle model.

    state0: [B, 4] = (x, y, yaw, v); controls: [B, K, 2] = (accel, steer);
    returns states [B, K, 4] after each step. Differentiable throughout.
    """
    x, y, yaw, v = state0.unbind(-1)
    out = []
    for k in range(controls.shape[1]):
        accel, steer = controls[:, k, 0], controls[:, k, 1]
        x = x + v * torch.cos(yaw) * dt
        y = y + v * torch.sin(yaw) * dt
        yaw = yaw + v / wheelbase * torch.tan(steer) * dt
        v = (v + accel * dt).clamp_min(0.0)
        out.append(torch.stack([x, y, yaw, v], dim=-1))
    return torch.stack(out, dim=1)


def kamm_circle_violation(controls: Tensor, v: Tensor, wheelbase: float = 2.7,
                          mu: float = 0.8, g: float = 9.81) -> Tensor:
    """Mean rectified friction-circle violation of a control sequence.

    controls: [B, K, 2] = (accel, steer); v: [B] entry speed (approximation:
    constant per short horizon). Returns scalar >= 0; 0 means fully feasible.
    """
    a_lon = controls[..., 0]
    a_lat = v.unsqueeze(-1).pow(2) / wheelbase * torch.tan(controls[..., 1]).abs()
    # clamp_min inside sqrt: at total accel = 0, sqrt'(0)=inf and relu'=0 give
    # the 0*inf=NaN sqrt-relu trap on the backward pass — bound the magnitude so
    # a fully-feasible (near-zero accel) control never NaNs the gradient.
    total = (a_lon.pow(2) + a_lat.pow(2)).clamp_min(1e-12).sqrt()
    return torch.relu(total - mu * g).mean()
