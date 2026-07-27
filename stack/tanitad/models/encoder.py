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


def resize_pos_embed(pos: Tensor, old_grid: tuple[int, int],
                     new_grid: tuple[int, int], mode: str = "bicubic") -> Tensor:
    """Resample a learned positional embedding ``[1, N_old, D]`` to a new token grid.

    The standard ViT resolution-transfer recipe, and the ONE part of a geometry
    change that is not free: ``pos`` is checkpoint-shaped, so a checkpoint trained
    at 16x16 cannot load into a 16x40 encoder without this.

    ⚠️ DECLARED BIAS, one-directional. Weights trained at one geometry are
    evaluated at another under a train/test shape shift that can only HURT the
    new shape. A wider/higher-res arm that WINS through a resample is therefore
    strong evidence; one that merely TIES is weak evidence and must NOT be read
    as "resolution does not help" — it may be reading the handicap. Any adopted
    geometry should be RETRAINED, not resampled. (Same caveat the FOV-audit
    stream's local shim declares; this is its in-repo counterpart.)
    """
    oh, ow = int(old_grid[0]), int(old_grid[1])
    nh, nw = int(new_grid[0]), int(new_grid[1])
    if pos.dim() != 3 or pos.shape[1] != oh * ow:
        raise ValueError(f"pos {tuple(pos.shape)} does not match old grid "
                         f"{oh}x{ow} (expected {oh * ow} tokens)")
    if (oh, ow) == (nh, nw):
        return pos
    d = pos.shape[-1]
    g = pos.reshape(1, oh, ow, d).permute(0, 3, 1, 2)         # [1, D, oh, ow]
    g = torch.nn.functional.interpolate(g.float(), size=(nh, nw), mode=mode,
                                        align_corners=False)
    return g.permute(0, 2, 3, 1).reshape(1, nh * nw, d).to(pos.dtype)


def adapt_pos_embed_(state: dict, encoder: "ViTEncoder", *,
                     key: str | None = None, mode: str = "bicubic") -> dict:
    """In-place: resample every positional embedding in ``state`` that does not
    match ``encoder``'s token grid. Returns the same dict for chaining.

    Use before ``load_state_dict(..., strict=True)`` when warm-starting weights
    trained at a DIFFERENT input geometry. Entries already the right size are
    untouched, so a same-geometry load is a no-op. The old grid is inferred as
    the square root of the checkpoint token count (every checkpoint before
    2026-07-27 is square); pass ``key`` to target one entry explicitly.
    """
    import math as _m
    keys = [key] if key else [k for k in state if k.endswith("pos")]
    for k in keys:
        v = state.get(k)
        if not isinstance(v, Tensor) or v.dim() != 3:
            continue
        n_old = int(v.shape[1])
        if n_old == encoder.n_tokens:
            continue
        side = int(round(_m.sqrt(n_old)))
        if side * side != n_old:
            raise ValueError(
                f"cannot infer the old token grid for {k!r}: {n_old} tokens is "
                f"not square. Pass the grid explicitly via resize_pos_embed().")
        state[k] = resize_pos_embed(v, (side, side), encoder.grid_shape, mode)
    return state


class ViTEncoder(nn.Module):
    """Image/frame-stack -> token grid [B, N, D] with N = (H/P) * (W/P).

    Height and width are INDEPENDENT (2026-07-27). ``cfg.image_size`` is the
    height; ``cfg.image_width`` (default ``None`` == square) overrides the width,
    so a wide driving frame such as 256 x 640 -> a 16 x 40 token grid is
    expressible. Every existing config is square and therefore byte-identical:
    same ``n_tokens``, same ``pos`` shape, same parameter count.

    ``grid_h`` / ``grid_w`` / ``grid_shape`` are the general accessors.
    ``grid_hw`` is kept for the square case only and RAISES on a non-square grid
    — a caller still reading one scalar has not been converted, and that must
    surface loudly rather than reshape a 16 x 40 grid as if it were square.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        h, w = cfg.image_hw()
        assert h % cfg.patch_size == 0, (
            f"image height {h} not divisible by patch {cfg.patch_size}")
        assert w % cfg.patch_size == 0, (
            f"image width {w} not divisible by patch {cfg.patch_size}")
        self.grid_h = h // cfg.patch_size
        self.grid_w = w // cfg.patch_size
        self.n_tokens = self.grid_h * self.grid_w
        self.patch = nn.Conv2d(cfg.in_channels, cfg.d_model,
                               kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.n_heads) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.d_model)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Token grid as ``(rows, cols)`` — the general form of ``grid_hw``."""
        return (self.grid_h, self.grid_w)

    @property
    def grid_hw(self) -> int:
        """Square-only token-grid side. Raises on a non-square grid ON PURPOSE.

        Downstream code that flattens/reshapes with one scalar (imagination
        advection, sector masks, spatial readout) is only correct for a square
        grid. Failing here names the unconverted call site; silently returning
        ``grid_h`` would corrupt a 16x40 grid into a 16x16 reshape.
        """
        if self.grid_h != self.grid_w:
            raise ValueError(
                f"token grid is {self.grid_h}x{self.grid_w} (non-square) — this "
                f"caller still reads the scalar `grid_hw`. Use `grid_shape` "
                f"(rows, cols); every consumer in tanitad/ accepts it.")
        return self.grid_h

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, C, H, W] -> tokens [B, N, D]."""
        if x.shape[-2:] != torch.Size(self.cfg.image_hw()):
            raise ValueError(
                f"encoder input is {tuple(x.shape[-2:])} but the config declares "
                f"{self.cfg.image_hw()} (image_size={self.cfg.image_size}, "
                f"image_width={self.cfg.image_width}). The positional embedding "
                f"is sized for the declared geometry — a mismatched input is the "
                f"stale-default failure this check exists to catch.")
        t = self.patch(x)                       # [B, D, H/P, W/P]
        t = t.flatten(2).transpose(1, 2)        # [B, N, D]
        t = t + self.pos
        use_ckpt = (self.cfg.grad_checkpoint and self.training
                    and t.requires_grad)
        for blk in self.blocks:
            if use_ckpt:                        # F-5: trade compute for memory
                t = torch.utils.checkpoint.checkpoint(
                    blk, t, use_reentrant=False)
            else:
                t = blk(t)
        return self.norm(t)
