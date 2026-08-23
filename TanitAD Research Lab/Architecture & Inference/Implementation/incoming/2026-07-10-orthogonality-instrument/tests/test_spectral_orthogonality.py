"""Ground-truth tests for the orthogonality/isotropy instrument.

Each test constructs a latent with a KNOWN isotropy structure and asserts the
instrument reads it correctly — the same validation contract `spectral.py` uses
(recover a known rank). Pure torch, standalone: `pytest tests/`.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_orthogonality import (  # noqa: E402
    condition_number, covariance_eigs, isotropy_ratio, orthogonality_report,
    participation_ratio, rms_offdiag_correlation,
)


def _g(seed):
    g = torch.Generator().manual_seed(seed)
    return g


# --------------------------------------------------------------------------- #
# Primitive unit checks (closed-form answers)                                  #
# --------------------------------------------------------------------------- #
def test_isotropy_ratio_bounds_and_extremes():
    assert abs(isotropy_ratio(torch.ones(50)) - 1.0) < 1e-9         # flat -> 1
    steep = torch.tensor([1.0] + [1e-6] * 49)                        # one dominant
    r = isotropy_ratio(steep)
    assert 0.0 < r < 0.05                                            # far from isotropic
    # ratio is scale-invariant
    assert abs(isotropy_ratio(torch.ones(50)) - isotropy_ratio(7.3 * torch.ones(50))) < 1e-9


def test_participation_and_condition():
    assert abs(participation_ratio(torch.ones(10)) - 10.0) < 1e-6    # all equal -> = len
    rank1 = torch.tensor([1.0] + [0.0] * 9)
    assert abs(participation_ratio(rank1) - 1.0) < 1e-6              # one direction
    assert condition_number(torch.ones(10)) == 1.0
    assert condition_number(torch.tensor([10.0, 1.0])) == 10.0


def test_covariance_eigs_recovers_known_diagonal():
    # z with independent columns of prescribed variance -> eigs ~ those variances
    torch.manual_seed(0)
    var = torch.tensor([9.0, 4.0, 1.0, 0.25])
    z = torch.randn(40000, 4, generator=_g(1)) * var.sqrt()
    eigs = covariance_eigs(z)
    assert torch.allclose(eigs, var.double(), rtol=0.05), eigs
    assert torch.all(eigs[:-1] >= eigs[1:])                          # descending


# --------------------------------------------------------------------------- #
# End-to-end report on known structures                                       #
# --------------------------------------------------------------------------- #
def test_isotropic_gaussian_is_admissible():
    z = torch.randn(20000, 64, generator=_g(2))                     # N(0, I)
    rep = orthogonality_report(z)
    assert rep.iso_ratio_active > 0.85, rep.iso_ratio_active
    assert rep.iso_ratio_global > 0.85, rep.iso_ratio_global
    assert rep.rms_offdiag_corr < 0.05, rep.rms_offdiag_corr
    assert rep.verdict.startswith("ADMISSIBLE"), rep.verdict


def test_anisotropic_subspace_is_not_admissible():
    # steep geometric variance decay across ALL dims -> active subspace itself anisotropic
    scale = torch.logspace(0, -3, 64)                               # 1 .. 1e-3
    z = torch.randn(20000, 64, generator=_g(3)) * scale
    rep = orthogonality_report(z)
    assert rep.iso_ratio_active < 0.5, rep.iso_ratio_active
    assert rep.cond_number_active > 5.0, rep.cond_number_active
    assert rep.verdict.startswith("NOT-YET-ADMISSIBLE"), rep.verdict


def test_correlated_coordinates_flagged():
    # build coordinates with strong pairwise correlation but equal marginal variance
    base = torch.randn(20000, 32, generator=_g(4))
    shared = torch.randn(20000, 1, generator=_g(5))
    z = base + 2.0 * shared                                          # every coord shares a factor
    rms, m = rms_offdiag_correlation(z)
    assert rms > 0.3, rms                                            # heavily correlated
    rep = orthogonality_report(z)
    assert rep.rms_offdiag_corr > 0.1
    assert rep.verdict.startswith("NOT-YET-ADMISSIBLE"), rep.verdict


def test_over_provisioned_active_isotropic_is_admissible():
    # THE real-checkpoint pattern: r isotropic active dims + a near-dead tail.
    N, r, dead = 20000, 24, 2024
    active = torch.randn(N, r, generator=_g(6))                     # isotropic signal
    tail = 1e-3 * torch.randn(N, dead, generator=_g(7))             # near-dead
    z = torch.cat([active, tail], dim=1)                            # S = 2048
    rep = orthogonality_report(z)
    assert abs(rep.active_k - r) <= 6, rep.active_k                 # recovers the active dim
    assert rep.iso_ratio_active > 0.7, rep.iso_ratio_active        # active subspace ~isotropic
    assert rep.iso_ratio_global < 0.2, rep.iso_ratio_global        # global low by design
    assert rep.rms_offdiag_corr < 0.05, rep.rms_offdiag_corr       # coords independent
    assert rep.verdict.startswith("ADMISSIBLE"), rep.verdict


def test_all_metrics_finite_and_serializable():
    z = torch.randn(2000, 128, generator=_g(8))
    d = orthogonality_report(z).to_dict()
    import json
    json.dumps(d, default=str)                                      # must serialize
    for k, v in d.items():
        if isinstance(v, float):
            assert v == v and abs(v) != float("inf"), (k, v)        # no NaN/inf
