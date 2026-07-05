"""Vision encoder: small ViT with batch-free normalization.

Design constraints (Phase 0 Plan §2.1):
- Patch-token grid output (spatial readout downstream — never global-pool, A7).
- LayerNorm/RMSNorm only. BatchNorm is banned in the inference path: deployment
  is batch-1 streaming, and batch-statistic layers silently violate the I2
  batch-consistency instrument (the ALPS-4B "115 %" incident).
- 2-frame inputs (channel-stacked tubelets) so ego-motion consequence is
  visible to the encoder (consequence-dominance, A8).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tanitad.config import EncoderConfig


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)), nn.GELU(),
            nn.Linear(int(d * mlp_ratio), d),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Image/frame-stack -> token grid [B, N, D] with N = (H/P) * (W/P)."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.image_size % cfg.patch_size == 0
        self.grid_hw = cfg.image_size // cfg.patch_size
        self.n_tokens = self.grid_hw ** 2
        self.patch = nn.Conv2d(cfg.in_channels, cfg.d_model,
                               kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.n_heads) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, C, H, W] -> tokens [B, N, D]."""
        t = self.patch(x)                       # [B, D, H/P, W/P]
        t = t.flatten(2).transpose(1, 2)        # [B, N, D]
        t = t + self.pos
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)
