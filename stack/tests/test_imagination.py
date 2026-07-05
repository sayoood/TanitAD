"""H15 imagination: advection semantics, masking contract, NLL behavior."""

import torch

from tanitad.config import base250_config, smoke_config
from tanitad.models.fourbrain import WorldModel
from tanitad.models.imagination import (ImaginationField, advect, d9_rows,
                                        imagination_nll, sector_mask)


def test_sector_mask_contract():
    frames = torch.rand(4, 1, 64, 64)
    masked, vis = sector_mask(frames, grid_hw=8)
    assert masked.shape == frames.shape
    assert vis.shape == (4, 64)
    # Exactly half the tokens hidden per sample; hidden pixels are zeroed.
    assert torch.all(vis.sum(dim=1) == 32)
    assert (masked * 0 == 0).all() and float(masked.min()) == 0.0


def test_advect_identity_flow_is_identity():
    tokens = torch.randn(2, 64, 16)
    out = advect(tokens, torch.zeros(2, 64, 2), grid_hw=8)
    assert torch.allclose(out, tokens, atol=1e-5)


def test_advect_shifts_content():
    """Unit flow in x must move cell content by one cell (semi-Lagrangian)."""
    tokens = torch.zeros(1, 16, 4)
    tokens[0, 5] = 1.0                       # cell (row 1, col 1) in a 4x4 grid
    flow = torch.zeros(1, 16, 2)
    flow[..., 0] = 1.0                       # value'(x) = value(x - 1) in x
    out = advect(tokens, flow, grid_hw=4)
    assert out[0, 6].abs().sum() > 0.9       # content now at (row 1, col 2)
    assert out[0, 5].abs().sum() < 0.1


def test_nll_prefers_high_variance_where_wrong():
    pred = torch.zeros(2, 16, 8)
    target = torch.ones(2, 16, 8)            # everything wrong
    vis = torch.zeros(2, 16)                 # all hidden
    confident = imagination_nll(pred, target, torch.full((2, 16), -2.0), vis)
    humble = imagination_nll(pred, target, torch.zeros(2, 16), vis)
    assert humble < confident                # claiming certainty while wrong costs


def test_field_shapes_and_d9_rows():
    field = ImaginationField(d_model=32, grid_hw=4, depth=1, n_heads=2)
    tokens = torch.randn(3, 16, 32)
    vis = torch.ones(3, 16)
    vis[:, :8] = 0.0
    pred, logvar = field(tokens, vis)
    assert pred.shape == (3, 16, 32) and logvar.shape == (3, 16)
    rows = d9_rows(pred, torch.randn(3, 16, 32), logvar, vis)
    assert set(rows) == {"hidden_cosine", "shuffled_cosine", "calibration_gap"}


def test_base250_parameter_budget():
    """D-008: the main-track model must instantiate at >= 250 M params."""
    model = WorldModel(base250_config())
    n = sum(p.numel() for p in model.parameters())
    assert n >= 250_000_000, f"only {n/1e6:.1f} M params"
    assert n <= 320_000_000, f"{n/1e6:.1f} M params — budget creep, justify in DECISIONS"


def test_smoke_model_has_imagination():
    model = WorldModel(smoke_config())
    assert model.imagination is not None
