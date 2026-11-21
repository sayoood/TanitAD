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


# ===========================================================================
# ViT-5 RECIPE ENCODER (PI 2026-08-13: "use an optimized encoder, leverage
# recent research work dealing with best encoder design suitable for our use
# case, type of architecture, inference efficiency and performance")
# ===========================================================================
# PUBLISHED, cited: "ViT-5: Vision Transformers for The Mid-2020s"
# (arXiv 2602.08071) — a component-wise re-derivation of the ViT backbone that
# keeps the canonical Attention–FFN skeleton and changes five things. ViT-5-Base
# reports 84.2 % ImageNet-1k top-1 vs DeiT-III-Base's 83.8 %.
#
#   1. RMSNorm everywhere (drops re-centering: cheaper AND slightly better)
#   2. LayerScale on every residual branch (stabilises deep ViTs)
#   3. QK-Norm inside attention (smooths optimisation, no loss spikes)
#   4. Register tokens, with their OWN 2D RoPE at a HIGHER frequency base
#   5. Joint learnable APE + 2D axial RoPE on patch and register tokens
#   ⛔ and one deliberate REJECTION: **NOT SwiGLU**. ViT-5 measured that
#      swapping GeLU->SwiGLU without extra stabilisation *drops* accuracy
#      because the gating interacts badly with LayerScale, and that compact
#      models are the most sensitive. Ours is a compact model. GeLU stays.
#
# ⭐ WHY 2D RoPE IS THE ONE THAT MATTERS FOR *US*, specifically. Our frames are
# 256x640 -> a 16x40 token grid: strongly non-square, and nothing like the
# square grids ViTs are usually trained on. A learnable APE is a table sized for
# exactly one geometry — which is why `ViTEncoder.forward` has to REFUSE a
# mismatched input, and why the first v6 S-W launch died on that guard when the
# documented command passed v5's 176x624 crop. Axial RoPE is a FUNCTION of (row,
# col), so it is defined at any grid, and it encodes RELATIVE geometry, which is
# what a driving scene actually has (a sign 3 tokens to the right of the ego
# corridor means the same thing wherever the corridor sits in frame).
# ⚠️ The APE is kept alongside it, per ViT-5 — this is "joint APE + RoPE", not
# a replacement — so the geometry guard STAYS. RoPE makes the encoder
# *extrapolatable*, it does not make a wrong-geometry input correct.
#
# INFERENCE EFFICIENCY (the third thing the PI asked for), on the Thor target:
#   · RMSNorm removes a mean-subtraction per norm, of which there are 2*depth+1.
#   · bias=False on QKV+proj removes 4*d params per block and a fused add.
#   · RoPE trades a [1, N, D] parameter table for two cached cos/sin buffers.
#   · No SwiGLU means the FFN stays 2 matmuls, not 3 — ~33 % fewer FFN GEMMs
#     than a gated variant at equal hidden width.
# ⚠️ NOT DONE, and deliberately: GQA/sliding-window attention. At 640 tokens
# attention is not the bottleneck, and both change the parameter count in ways
# that would confound the E-ENC comparison that is already banked.


class RMSNorm(nn.Module):
    """RMSNorm — normalise by RMS, no mean subtraction, no bias."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt) * self.weight)


def axial_rope_tables(grid_h: int, grid_w: int, head_dim: int,
                      theta: float, device=None, dtype=torch.float32):
    """cos/sin for 2D AXIAL RoPE over a (grid_h, grid_w) token grid.

    Half the head dimension encodes the ROW coordinate and half the COLUMN, so
    a non-square grid is handled natively — no interpolation, no reshape that
    assumes squareness (the `grid_hw` trap this file already guards).
    Returns (cos, sin), each [grid_h*grid_w, head_dim]."""
    if head_dim % 4 != 0:
        raise ValueError(
            f"axial RoPE needs head_dim divisible by 4 (two axes x complex "
            f"pairs); got {head_dim}. Choose n_heads so d_model/n_heads % 4 == 0.")
    quarter = head_dim // 4
    freqs = theta ** (-torch.arange(quarter, device=device,
                                    dtype=torch.float32) / quarter)
    rows = torch.arange(grid_h, device=device, dtype=torch.float32)
    cols = torch.arange(grid_w, device=device, dtype=torch.float32)
    ang_r = rows[:, None] * freqs[None, :]                  # [H, q]
    ang_c = cols[:, None] * freqs[None, :]                  # [W, q]
    ang_r = ang_r[:, None, :].expand(grid_h, grid_w, quarter)
    ang_c = ang_c[None, :, :].expand(grid_h, grid_w, quarter)
    ang = torch.cat([ang_r, ang_c], dim=-1).reshape(grid_h * grid_w, 2 * quarter)
    ang = torch.cat([ang, ang], dim=-1)                     # [N, head_dim]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x [B, heads, N, head_dim] with cos/sin [N, head_dim]."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class Attention5(nn.Module):
    """Attention with QK-Norm and 2D RoPE. No bias on qkv/proj (ViT-5)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d_model {d} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        # QK-Norm: RMSNorm over the HEAD dim, applied to q and k before the dot
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: Tensor, cos: Tensor | None, sin: Tensor | None,
                n_reg: int) -> Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each [B, h, N, hd]
        q, k = self.q_norm(q), self.k_norm(k)
        if cos is not None:
            # registers occupy the LEADING n_reg positions and carry their own
            # (higher-base) tables, already concatenated by the caller
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class Block5(nn.Module):
    """RMSNorm -> Attention(QK-norm, RoPE) -> LayerScale, then RMSNorm -> MLP
    (GeLU, NOT SwiGLU) -> LayerScale."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 4.0,
                 ls_init: float = 1e-5):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention5(d, n_heads)
        self.ls1 = nn.Parameter(ls_init * torch.ones(d))
        self.norm2 = RMSNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)), nn.GELU(),
            nn.Linear(int(d * mlp_ratio), d))
        self.ls2 = nn.Parameter(ls_init * torch.ones(d))

    def forward(self, x: Tensor, cos, sin, n_reg: int) -> Tensor:
        x = x + self.ls1 * self.attn(self.norm1(x), cos, sin, n_reg)
        return x + self.ls2 * self.mlp(self.norm2(x))


class ViT5Encoder(nn.Module):
    """The ViT-5-recipe encoder. Drop-in for :class:`ViTEncoder`.

    Same contract: ``[B, C, H, W] -> [B, N, D]`` PATCH tokens only, in row-major
    grid order, with ``grid_shape`` / ``n_tokens`` unchanged — so the spatial
    readout, the sector masks and every geometry consumer see exactly what they
    saw before. ⭐ Register tokens are consumed INTERNALLY and STRIPPED before
    the return: they exist to absorb the attention artifacts that would
    otherwise corrupt patch features, and letting them reach a 4x4 spatial
    readout would silently mix a non-spatial token into a spatial cell."""

    def __init__(self, cfg: EncoderConfig, n_registers: int = 4,
                 rope_theta: float = 100.0,
                 rope_theta_registers: float = 10000.0,
                 ls_init: float = 1e-5):
        super().__init__()
        self.cfg = cfg
        h, w = cfg.image_hw()
        if h % cfg.patch_size or w % cfg.patch_size:
            raise ValueError(f"image {h}x{w} not divisible by patch "
                             f"{cfg.patch_size}")
        self.grid_h, self.grid_w = h // cfg.patch_size, w // cfg.patch_size
        self.n_tokens = self.grid_h * self.grid_w
        self.n_registers = int(n_registers)
        self.patch = nn.Conv2d(cfg.in_channels, cfg.d_model,
                               kernel_size=cfg.patch_size,
                               stride=cfg.patch_size)
        # joint APE (ViT-5 keeps BOTH) — patches only; registers are learned
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if self.n_registers:
            self.registers = nn.Parameter(
                torch.zeros(1, self.n_registers, cfg.d_model))
            nn.init.trunc_normal_(self.registers, std=0.02)
        self.blocks = nn.ModuleList(
            Block5(cfg.d_model, cfg.n_heads, ls_init=ls_init)
            for _ in range(cfg.depth))
        self.norm = RMSNorm(cfg.d_model)

        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = axial_rope_tables(self.grid_h, self.grid_w, head_dim,
                                     rope_theta)
        if self.n_registers:
            # ⭐ registers get their OWN table at a MUCH higher frequency base
            # (ViT-5): they are not at a place in the image, so they must not
            # share the patches' low-frequency positional geometry.
            rcos, rsin = axial_rope_tables(1, self.n_registers, head_dim,
                                           rope_theta_registers)
            cos = torch.cat([rcos, cos], dim=0)
            sin = torch.cat([rsin, sin], dim=0)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.grid_h, self.grid_w)

    @property
    def grid_hw(self) -> int:
        """Square-only side; raises on a non-square grid, as ViTEncoder does."""
        if self.grid_h != self.grid_w:
            raise ValueError(
                f"token grid is {self.grid_h}x{self.grid_w} (non-square) — use "
                f"`grid_shape` (rows, cols).")
        return self.grid_h

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-2:] != torch.Size(self.cfg.image_hw()):
            raise ValueError(
                f"encoder input is {tuple(x.shape[-2:])} but the config declares "
                f"{self.cfg.image_hw()}. RoPE would tolerate another grid, but "
                f"the joint APE is sized for this one — the guard stays.")
        t = self.patch(x).flatten(2).transpose(1, 2)          # [B, N, D]
        t = t + self.pos
        if self.n_registers:
            t = torch.cat([self.registers.expand(t.shape[0], -1, -1), t], dim=1)
        cos = self.rope_cos.to(t.dtype)
        sin = self.rope_sin.to(t.dtype)
        use_ckpt = (self.cfg.grad_checkpoint and self.training
                    and t.requires_grad)
        for blk in self.blocks:
            if use_ckpt:
                t = torch.utils.checkpoint.checkpoint(
                    blk, t, cos, sin, self.n_registers, use_reentrant=False)
            else:
                t = blk(t, cos, sin, self.n_registers)
        t = self.norm(t)
        return t[:, self.n_registers:]      # ⛔ STRIP registers — see docstring
