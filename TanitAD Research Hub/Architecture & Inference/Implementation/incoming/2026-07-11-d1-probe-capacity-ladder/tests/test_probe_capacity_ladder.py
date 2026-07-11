"""Ground-truth tests for the probe-capacity discriminator.

The discriminator's whole value is that it separates two cases:
  linear-recoverable target  -> small gap (best_linear ~= best_mlp)
  nonlinear-only target      -> large gap (best_mlp << best_linear)
and that the isotropy metric behaves (isotropic -> ~1; anisotropic -> low).
Everything here runs on synthetic data with a known answer — no checkpoint, no GPU.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_capacity_ladder import (  # noqa: E402
    ego_frame,
    isotropy_metrics,
    probe_ladder,
)


def _episodes(E_n=8, per=200):
    """Return an episode-index vector spanning both split parities."""
    return torch.repeat_interleave(torch.arange(E_n), per)


def test_ego_frame_rotation_zero_yaw_identity():
    dxy = torch.tensor([[1.0, 2.0], [-3.0, 0.5]])
    out = ego_frame(dxy, torch.zeros(2))
    assert torch.allclose(out, dxy, atol=1e-6)


def test_ego_frame_rotation_ninety_deg():
    # heading +90deg: a world +x displacement becomes ego -y.
    dxy = torch.tensor([[1.0, 0.0]])
    out = ego_frame(dxy, torch.tensor([torch.pi / 2]))
    assert torch.allclose(out, torch.tensor([[0.0, -1.0]]), atol=1e-5)


def test_isotropy_isotropic_is_high():
    torch.manual_seed(0)
    S = torch.randn(4000, 32)  # white
    m = isotropy_metrics(S)
    assert m["iso_ratio_active"] > 0.7
    assert m["cond_number_active"] < 4.0


def test_isotropy_anisotropic_is_low():
    torch.manual_seed(0)
    scale = torch.logspace(0, -2, 32)  # steep decay -> anisotropic
    S = torch.randn(4000, 32) * scale
    m = isotropy_metrics(S)
    assert m["iso_ratio_active"] < 0.5
    assert m["cond_number_active"] > 10.0


def test_ladder_linear_target_small_gap():
    """Y is an exact linear function of S -> linear probe already optimal -> gap ~ 0."""
    torch.manual_seed(0)
    N, D = 3200, 48
    S = torch.randn(N, D)
    W = torch.randn(D, 2)
    Y = S @ W + 0.01 * torch.randn(N, 2)
    E = _episodes(8, N // 8)
    out = probe_ladder(S, Y, E)
    assert out["best_linear"] < 0.2          # linear recovers it
    assert out["gap_rel_pct"] < 15.0          # MLP gives little extra


def test_ladder_nonlinear_target_large_gap():
    """Y depends nonlinearly on S -> MLP must beat the linear probe (positive gap)."""
    torch.manual_seed(0)
    N, D = 3200, 16
    S = torch.randn(N, D)
    # nonlinear, zero-linear-correlation target: products of independent coords
    Y = torch.stack([S[:, 0] * S[:, 1], (S[:, 2] ** 2 - 1.0)], dim=-1)
    E = _episodes(8, N // 8)
    out = probe_ladder(S, Y, E)
    assert out["best_mlp"] < out["best_linear"]     # MLP strictly better
    assert out["gap_abs"] > 0.1                       # and materially so


def test_pca_ladder_recovers_signal_in_low_variance_free_setting():
    """PCA-to-k on a target that lives in the top-k subspace keeps the linear probe strong,
    and fixes the D>>N overfit (raw OLS should be worse than the PCA-reduced probe)."""
    torch.manual_seed(0)
    N, D, k = 400, 512, 8  # D >> N: raw OLS overfits
    U = torch.linalg.svd(torch.randn(D, D), full_matrices=False)[0]
    latent_lowdim = torch.randn(N, k)
    scales = torch.logspace(0, -1, k)
    S = (latent_lowdim * scales) @ U[:k]  # signal in top-k dirs
    Y = latent_lowdim[:, :2].clone()
    E = _episodes(8, N // 8)
    raw = probe_ladder(S, Y, E, pca_k=0)
    red = probe_ladder(S, Y, E, pca_k=k)
    assert red["best_linear"] <= raw["linear_ols"]  # PCA fixes the OLS D>>N overfit


def test_ladder_reports_all_rungs_and_reference():
    torch.manual_seed(0)
    N, D = 1600, 16
    S = torch.randn(N, D)
    Y = S[:, :2].clone()
    E = _episodes(8, N // 8)
    out = probe_ladder(S, Y, E)
    for k in ("linear_ols", "ridge_a1", "ridge_a10", "ridge_a100",
              "mlp_256", "mlp_256x2", "ref_zero_ade", "gap_abs", "gap_rel_pct"):
        assert k in out
    assert out["n_train"] > 0 and out["n_val"] > 0
