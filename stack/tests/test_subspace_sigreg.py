"""Sub-JEPA's subspace regularizer — and the controls that make it readable.

⛔ EVERY PANEL CARRIES A CONTROL THAT MUST READ A KNOWN VALUE. Four estimator
failures on 2026-08-22 each produced a confident wrong number and each was
caught only by a control reading the same value as the thing being measured.
An anti-collapse loss that cannot DETECT collapse is the same class of defect,
and it would fail silently in exactly the way the incumbent n=24 SIGReg does.
"""
import math

import pytest
import torch

from tanitad.models.sigreg import SigReg, SubspaceSigReg


def test_projections_are_row_orthonormal_and_mutually_orthogonal():
    """The paper's construction: QR -> row-orthonormal, blocks mutually
    orthogonal so the subspaces are independent rather than redundant."""
    m = SubspaceSigReg(dim=64, n_subspaces=4, n_slices=32)
    assert m.d_s == 16
    P = m.proj                                    # [K, d_s, D]
    for k in range(m.n_subspaces):
        gram = P[k] @ P[k].t()
        assert torch.allclose(gram, torch.eye(m.d_s), atol=1e-5), \
            f"subspace {k} is not row-orthonormal"
    flat = P.reshape(-1, 64)
    cross = flat @ flat.t()
    assert torch.allclose(cross, torch.eye(flat.shape[0]), atol=1e-5), \
        "subspaces are not mutually orthogonal — they would be redundant"


def test_projections_are_frozen_not_parameters():
    """*"Freezing the projections prevents the regularizer itself from adapting
    to the evolving latent distribution."* A learnable projection could simply
    rotate to wherever the latent already is and report success."""
    m = SubspaceSigReg(dim=32, n_subspaces=4, n_slices=16)
    assert list(m.parameters()) == [], "projections must not be parameters"
    assert "proj" in dict(m.named_buffers()), "projections must be a buffer"
    before = m.proj.clone()
    z = torch.randn(64, 32, requires_grad=True)
    m(z).backward()
    assert torch.equal(m.proj, before), "projections moved during backward"
    assert z.grad is not None and float(z.grad.abs().sum()) > 0, \
        "no gradient reaches the embeddings"


def test_it_can_actually_detect_collapse():
    """⛔ THE GUARD-CAN-FAIL CONTROL. A COLLAPSED batch (rank 1) must score far
    worse than an isotropic Gaussian one. Without this, a regularizer that is
    silently disabled looks identical to one that is working."""
    torch.manual_seed(0)
    m = SubspaceSigReg(dim=64, n_subspaces=4, n_slices=64)
    healthy = torch.randn(256, 64)
    direction = torch.randn(1, 64)
    collapsed = torch.randn(256, 1) * direction        # rank 1
    s_ok, s_bad = float(m(healthy)), float(m(collapsed))
    assert s_bad > s_ok, (
        f"the regularizer does not penalise collapse: healthy {s_ok:.4f} vs "
        f"collapsed {s_bad:.4f} — it cannot do its job")


def test_k1_matches_full_space_sigreg():
    """K=1 must reduce to LeWM's full-space SIGReg up to the (orthogonal, hence
    distribution-preserving) rotation, so the new path is a strict superset."""
    torch.manual_seed(0)
    z = torch.randn(128, 32)
    sub = SubspaceSigReg(dim=32, n_subspaces=1, n_slices=256, seed=3)
    full = SigReg(n_slices=256)
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    a, b = float(sub(z, generator=g1)), float(full(z, generator=g2))
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a - b) / max(abs(b), 1e-6) < 0.25, (
        f"K=1 should track full-space SIGReg on isotropic input: {a} vs {b}")


@pytest.mark.parametrize("k", [1, 2, 8, 16])
def test_subspace_dim_partitions_the_ambient_dim(k):
    m = SubspaceSigReg(dim=64, n_subspaces=k, n_slices=8)
    assert m.d_s == round(64 / k)
    assert m.proj.shape == (k, m.d_s, 64)


def test_rejects_more_subspaces_than_dimensions():
    with pytest.raises(ValueError):
        SubspaceSigReg(dim=8, n_subspaces=16, n_slices=4)


def test_collapse_gate_uses_the_ENERGY_statistic_not_the_amplitude_one():
    """⛔ The two statistics in one spectrum_report disagree, and the gate was
    reading the wrong one.

    MEASURED 2026-08-22 on v7-tiny at n=1440:
        arm `fixed`     top-1 energy 0.551, effective_rank(σ) 130.91 -> PASSES 64
        arm `lewm-long` top-1 energy 0.339, effective_rank(σ)  24.14 -> FAILS 64
    A representation with 55 % of its variance in ONE direction passing a
    COLLAPSE gate is a contradiction. ``effective_rank`` uses p ∝ σ, which is
    dominated by the near-zero tail and rewards a noisy spectrum;
    ``participation_ratio`` uses p ∝ σ² (energy) and orders the arms coherently
    with top1_share and with the ego probe.
    """
    from tanitad.models.v6 import (O6_PARTICIPATION_FLOOR, o6_rank_verdict,
                                   spectrum_report)
    import torch

    # ⭐ the fixture reproduces the MEASURED inversion, not a caricature: ~55 %
    # of the energy in ONE direction (v7-tiny `fixed` read top-1 0.551) and the
    # remainder spread thinly over a broad tail. That tail is negligible in
    # ENERGY but not in AMPLITUDE, which is exactly what inflates p ∝ σ.
    torch.manual_seed(0)
    n, d = 1536, 512
    tail = ((1.0 - 0.55) / (d - 1)) ** 0.5
    z = torch.randn(n, d) * tail
    z[:, 0] = torch.randn(n) * (0.55 ** 0.5)
    rep = spectrum_report(z)
    assert rep["top_k_share"] > 0.5, "fixture is not concentrated enough"
    assert rep["effective_rank"] > rep["participation_ratio"] * 5, (
        f"the σ statistic should be inflated by the tail: got "
        f"effective_rank {rep['effective_rank']:.2f} vs participation "
        f"{rep['participation_ratio']:.2f} — re-derive this test before "
        f"trusting either number")

    v = o6_rank_verdict(rep)
    assert "participation_ratio" in v and "participation_pass" in v, \
        "the verdict carries no energy-based clause"
    assert v["participation_floor"] == O6_PARTICIPATION_FLOOR
    assert v["participation_pass"] is False, (
        "a population with >90 % of its energy in the top directions must FAIL "
        "the participation clause")
    assert "energy" in v["statistic_note"]


def test_participation_floor_is_calibrated_on_a_REAL_representation():
    """⛔ Not on a synthetic population. ``O6_RANK_FLOOR = 64`` came from an
    α=2 power-law (healthy 122.4 / collapsed 19.4) and no real representation
    measured in this programme approaches it. The participation floor is frozen
    DINOv3 measured on OUR frames — the only representation here with
    demonstrated decodability (speed R² +0.147 vs our trunk's +0.0025)."""
    from tanitad.models.v6 import O6_PARTICIPATION_FLOOR
    assert abs(O6_PARTICIPATION_FLOOR - 8.56) < 1e-9, (
        "the floor moved; it must stay anchored to a MEASURED reference "
        "representation, and the new anchor must be named where it is set")
