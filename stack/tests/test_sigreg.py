"""SIGReg sanity: penalizes collapse, tolerates Gaussians, finite gradients."""

import torch

from tanitad.models.sigreg import SigReg, epps_pulley


def test_gaussian_scores_lower_than_collapsed():
    torch.manual_seed(0)
    sig = SigReg(n_slices=128)
    z_gauss = torch.randn(256, 32)
    z_collapsed = torch.ones(256, 32) + 0.001 * torch.randn(256, 32)
    assert sig(z_gauss) < sig(z_collapsed)


def test_epps_pulley_detects_non_normality():
    torch.manual_seed(0)
    y_norm = torch.randn(512)
    y_bimodal = torch.cat([torch.randn(256) - 4, torch.randn(256) + 4])
    assert epps_pulley(y_norm) < epps_pulley(y_bimodal)


def test_gradients_finite_even_for_extreme_inputs():
    sig = SigReg(n_slices=32)
    z = (1000.0 * torch.randn(64, 16)).requires_grad_(True)
    loss = sig(z)
    loss.backward()
    assert torch.isfinite(z.grad).all()
