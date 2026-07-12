"""Operative predictor: causal, action-conditioned, residual, multi-horizon.

Validated design (A4 bake-off): residual/delta prediction + change-weighted
latent loss beat plain MSE (0.97 vs 0.71) and flow sampling (0.44) for
action-conditioned short-horizon control. Multi-horizon heads (k in {1,2,4})
double as a training signal and an inference accelerator (MTP, H5).

Action conditioning is FiLM (scale/shift per layer) on continuous
(steer, accel) — cond_proj accepts arbitrary action_dim (H12 command
embeddings concatenate here later).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tanitad.config import PredictorConfig


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, d: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * d)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


class CausalBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.film = FiLM(cond_dim, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: Tensor, cond: Tensor, mask: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        x = x + self.mlp(self.film(self.norm2(x), cond))
        return x


class OperativePredictor(nn.Module):
    """Predicts future compact states from a causal window of (state, action).

    forward(states [B, W, D], actions [B, W, A], intent=None)
        -> {k: z_hat_{t+k} [B, D]}
    where t is the last window position. Residual: z_hat = z_t + delta_k.

    ``intent_dim`` (optional, D-030): when set, an ``intent`` token [B, intent_dim]
    from the tactical policy is projected and ADDED to the per-step action FiLM
    conditioning, so the tactical brain steers the operative dynamics (closing
    the hierarchy). ``intent=None`` reproduces the base behaviour exactly; base
    models build the predictor with ``intent_dim=None`` (no extra params), so a
    vanilla WorldModel checkpoint stays a strict subset of a 4b one.
    """

    def __init__(self, cfg: PredictorConfig, state_dim: int,
                 intent_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.in_proj = nn.Linear(state_dim, d)
        self.act_emb = nn.Sequential(nn.Linear(cfg.action_dim, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, cfg.window, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            CausalBlock(d, cfg.n_heads, cond_dim=d) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict(
            {str(k): nn.Linear(d, state_dim) for k in cfg.horizons})
        self.out_proj = nn.Linear(state_dim, d)  # reserved: feed predictions back
        # Tactical-intent conditioning (D-030). Projected into the FiLM cond
        # space and added to the action embedding. Non-zero init so a live FiLM
        # makes the intent steer the output; FiLM's own zero-init keeps the
        # identity start (intent has no effect until the FiLM weights train).
        self.intent_proj = (nn.Linear(intent_dim, d)
                            if intent_dim is not None else None)

    def forward(self, states: Tensor, actions: Tensor,
                intent: Tensor | None = None) -> dict[int, Tensor]:
        b, w, _ = states.shape
        assert w == self.cfg.window, f"window mismatch: {w} != {self.cfg.window}"
        x = self.in_proj(states) + self.pos[:, :w]
        cond = self.act_emb(actions)                        # [B, W, D]
        if intent is not None:
            if self.intent_proj is None:
                raise ValueError("predictor built without intent_dim cannot "
                                 "consume an intent token")
            cond = cond + self.intent_proj(intent).unsqueeze(1)   # broadcast W
        mask = torch.triu(torch.ones(w, w, device=states.device, dtype=torch.bool),
                          diagonal=1)
        for blk in self.blocks:
            x = blk(x, cond, mask)
        h_last = self.norm(x[:, -1])                        # [B, D]
        z_t = states[:, -1]
        out: dict[int, Tensor] = {}
        for k in self.cfg.horizons:
            delta = self.heads[str(k)](h_last)
            out[k] = z_t + delta if self.cfg.residual else delta
        return out


def change_weighted_mse(pred: Tensor, target: Tensor, prev: Tensor,
                        eps: float = 1e-6) -> Tensor:
    """MSE weighted by how much each latent dim actually changed (A4).

    Prevents static content from dominating the loss — the driving analog of
    ignoring parked background in favor of the consequence of the action.
    """
    w = (target - prev).abs()
    w = w / w.mean(dim=-1, keepdim=True).clamp_min(eps)
    return (w * (pred - target).pow(2)).mean()
