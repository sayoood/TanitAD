"""ViT-5-recipe encoder (PI 2026-08-13: "use an optimized encoder").

PUBLISHED: "ViT-5: Vision Transformers for The Mid-2020s" (arXiv 2602.08071) —
RMSNorm + LayerScale + QK-Norm + register tokens + joint APE/2D-axial-RoPE,
GeLU MLP. 84.2 % IN-1k vs DeiT-III-Base 83.8 %.

These tests pin the parts that are easy to get subtly wrong and impossible to
notice: the register tokens leaking into the spatial readout, the axial split
on a NON-SQUARE grid, and the SwiGLU rejection.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.config import EncoderConfig
from tanitad.models.encoder import (RMSNorm, ViT5Encoder, apply_rope,
                                    axial_rope_tables)

CFG = EncoderConfig(in_channels=9, image_size=256, image_width=640,
                    patch_size=16, d_model=128, depth=2, n_heads=4)


def test_registers_are_STRIPPED_from_the_output():
    """⛔ THE ONE THAT WOULD BE SILENT. Registers exist to absorb attention
    artifacts; they are not at a place in the image. If they reached the 4x4
    SpatialGridReadout they would be mixed into a spatial cell and every
    geometry consumer downstream would be quietly wrong — with the right
    dtype, the right rank, and a plausible magnitude."""
    enc = ViT5Encoder(CFG, n_registers=4).eval()
    with torch.no_grad():
        out = enc(torch.randn(2, 9, 256, 640))
    assert out.shape == (2, 16 * 40, 128), "must return PATCH tokens only"
    assert enc.n_tokens == 640 and enc.grid_shape == (16, 40)


def test_register_count_does_not_change_the_output_shape():
    for n in (0, 2, 8):
        enc = ViT5Encoder(CFG, n_registers=n).eval()
        with torch.no_grad():
            assert enc(torch.randn(1, 9, 256, 640)).shape == (1, 640, 128)


def test_axial_rope_handles_a_NON_SQUARE_grid():
    """Our grid is 16x40. Half the head dim encodes the row and half the column,
    so a non-square grid needs no interpolation and no square reshape — the
    `grid_hw` trap this codebase already guards elsewhere."""
    cos, sin = axial_rope_tables(16, 40, 32, theta=100.0)
    assert cos.shape == (640, 32) and sin.shape == (640, 32)
    # tokens in the same COLUMN but different ROW must differ
    assert not torch.equal(cos[0], cos[40])
    # ...and tokens in the same ROW but different COLUMN must differ
    assert not torch.equal(cos[0], cos[1])


def test_axial_rope_refuses_a_head_dim_it_cannot_split():
    """head_dim must divide by 4 (two axes x complex pairs). Silently rounding
    would put the row/column split half a pair off."""
    with pytest.raises(ValueError, match="divisible by 4"):
        axial_rope_tables(4, 4, 30, theta=100.0)


def test_rope_is_norm_preserving():
    """A rotation must not change vector magnitude — if it does, the tables are
    not a rotation and QK-Norm is silently fighting it."""
    cos, sin = axial_rope_tables(4, 6, 16, theta=100.0)
    x = torch.randn(2, 3, 24, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_registers_get_a_different_rope_base_than_patches():
    """ViT-5: registers carry their OWN 2D RoPE at a MUCH higher frequency base,
    because they are not at a location in the image and must not share the
    patches' low-frequency positional geometry."""
    enc = ViT5Encoder(CFG, n_registers=4, rope_theta=100.0,
                      rope_theta_registers=10000.0)
    assert enc.rope_cos.shape[0] == 4 + 640
    reg, pat = enc.rope_cos[:4], enc.rope_cos[4:]
    # a higher base => slower angular progression across adjacent positions
    assert not torch.allclose(reg[1] - reg[0], pat[1] - pat[0], atol=1e-4)


def test_rmsnorm_does_not_re_centre():
    """RMSNorm divides by RMS and does NOT subtract the mean — that is the whole
    difference from LayerNorm, and a shifted input must stay shifted."""
    n = RMSNorm(8)
    x = torch.ones(1, 8) * 5.0
    y = n(x)
    assert y.mean().item() > 0.1, "RMSNorm must not zero-centre"
    assert torch.allclose(y.pow(2).mean().sqrt(), torch.tensor(1.0), atol=1e-3)


def test_no_bias_on_qkv_and_proj():
    """ViT-5 drops the QKV/proj biases: fewer params, one fewer fused add."""
    enc = ViT5Encoder(CFG, n_registers=2)
    for blk in enc.blocks:
        assert blk.attn.qkv.bias is None
        assert blk.attn.proj.bias is None


def test_mlp_is_GELU_not_SwiGLU():
    """⛔ A DELIBERATE REJECTION, not an omission. ViT-5 MEASURED that swapping
    GeLU->SwiGLU without extra stabilisation *drops* accuracy, because the
    gating interacts badly with LayerScale and compact models are the most
    sensitive. Ours is a compact model."""
    enc = ViT5Encoder(CFG, n_registers=0)
    acts = [type(m).__name__ for m in enc.blocks[0].mlp]
    assert "GELU" in acts and "SiLU" not in acts
    assert len(enc.blocks[0].mlp) == 3, "plain 2-matmul FFN, not a gated 3-matmul"


def test_layerscale_starts_small_so_blocks_begin_near_identity():
    enc = ViT5Encoder(CFG, n_registers=0, ls_init=1e-5)
    assert torch.allclose(enc.blocks[0].ls1,
                          torch.full_like(enc.blocks[0].ls1, 1e-5))


def test_geometry_guard_is_KEPT_despite_rope():
    """⚠️ RoPE would tolerate another grid, but the joint APE is sized for this
    one. RoPE makes the encoder extrapolatable; it does not make a
    wrong-geometry input correct — and a wrong geometry is exactly what killed
    the first v6 S-W launch."""
    enc = ViT5Encoder(CFG, n_registers=2).eval()
    with pytest.raises(ValueError, match="declares"):
        enc(torch.randn(1, 9, 176, 624))


def test_it_is_a_drop_in_for_ViTEncoder():
    """Same contract: [B,C,H,W] -> [B,N,D] patch tokens, same n_tokens, same
    grid_shape, and grid_hw still REFUSES a non-square grid."""
    from tanitad.models.encoder import ViTEncoder
    a, b = ViTEncoder(CFG).eval(), ViT5Encoder(CFG, n_registers=4).eval()
    x = torch.randn(1, 9, 256, 640)
    with torch.no_grad():
        assert a(x).shape == b(x).shape
    assert a.n_tokens == b.n_tokens and a.grid_shape == b.grid_shape
    with pytest.raises(ValueError, match="non-square"):
        b.grid_hw
