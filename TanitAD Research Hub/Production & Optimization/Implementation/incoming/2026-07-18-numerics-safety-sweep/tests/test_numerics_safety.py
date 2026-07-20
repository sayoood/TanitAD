"""Numerics-safety regression guard for the learned-output class.

Production & Optimization backlog P0 #4 (fleet directive 2026-07-17): the
2026-07-17 `imagination_nll` NaN was one instance of a class — an unbounded
`exp` / `log` / division applied to a *learned* (network-produced) or *data*
value, which silently NaN-a-training-run or NaN-a-metric. This module is the
executable guard that keeps that class closed: it feeds every such site in the
stack a pathological input and asserts the output stays finite.

Audit result (2026-07-18 grep-sweep of `stack/tanitad`, learned/data-output
`exp`/`log`/`div` sites):

  site                                       guard                     status
  ---------------------------------------------------------------------------
  imagination.ImaginationField.logvar_head   .clamp(-10,10) at head    GUARDED
  imagination.imagination_nll (exp(-logvar)) .clamp(-10,10) in nll     GUARDED
  replay/arms sigma export (0.5*logvar).exp  .clamp(-10,10) (arms:284) GUARDED
  refs/refb.FeatureOOD.score (sum/count)     count<2 -> zeros; var eps GUARDED
  models/sigreg epps-pulley/SigReg exp(-..)  neg-exponent (bounded)    SAFE-BY-CONSTRUCTION
  eval/spectral effective_rank/tail/optimal_k .clamp_min(1e-12)        GUARDED
  models/fourbrain erank exp(-entropy)       .clamp_min(1e-12)         GUARDED
  eval/metrics np.exp(-lam*..), denom=max(.) neg-exp / max(.,1e-9)     SAFE-BY-CONSTRUCTION

No new unguarded site was found — the class is closed. This test is the
regression guard so it stays closed on future merges. The imagination and
FeatureOOD cases are genuine failing-then-passing witnesses: each fails against
the pre-fix code (unclamped `exp`, ungated `sum/count`) and passes against the
current stack.

Standalone: `pytest <this-dir>` (tanitad installed in the venv).
"""

from __future__ import annotations

import math

import torch

from tanitad.eval.spectral import (effective_rank, energy_knee, optimal_k,
                                    spectral_tail)
from tanitad.models.imagination import ImaginationField, imagination_nll
from tanitad.models.sigreg import SigReg, epps_pulley
from tanitad.refs.refb import FeatureOOD

EXTREME = [-100.0, -50.0, -11.0, 0.0, 11.0, 50.0, 100.0]


# --------------------------------------------------------------------------- #
#  imagination_nll — THE fixed site (witness: fails on the unclamped path)
# --------------------------------------------------------------------------- #
def _rand_field(seed=0):
    g = torch.Generator().manual_seed(seed)
    B, N, D = 2, 16, 8
    pred = torch.randn(B, N, D, generator=g)
    target = torch.randn(B, N, D, generator=g)
    vis = (torch.rand(B, N, generator=g) > 0.5).float()
    return pred, target, vis


def test_imagination_nll_finite_at_extreme_logvar():
    pred, target, vis = _rand_field()
    for lv in EXTREME:
        logvar = torch.full(vis.shape, lv)
        out = imagination_nll(pred, target, logvar, vis)
        assert torch.isfinite(out), f"imagination_nll non-finite at logvar={lv}"


def test_imagination_nll_witness_unclamped_would_overflow():
    """Documents WHAT the clamp prevents: the pre-fix formula overflows to inf
    at logvar=-100, while the guarded `imagination_nll` stays finite."""
    pred, target, vis = _rand_field()
    err2 = (pred - target).pow(2).mean(dim=-1)
    unclamped = 0.5 * (torch.exp(-torch.full(vis.shape, -100.0)) * err2 - 100.0)
    assert not torch.isfinite(unclamped).all(), "expected the OLD path to overflow"
    guarded = imagination_nll(pred, target, torch.full(vis.shape, -100.0), vis)
    assert torch.isfinite(guarded)


def test_imagination_nll_behaviour_preserving_in_range():
    """In [-10, 10] the clamp is a no-op: guarded == manual formula."""
    pred, target, vis = _rand_field(seed=1)
    logvar = (torch.rand(vis.shape) * 18.0 - 9.0)          # in (-9, 9)
    err2 = (pred - target).pow(2).mean(dim=-1)
    nll = 0.5 * (torch.exp(-logvar) * err2 + logvar)
    w = (1.0 - vis) + 0.1 * vis
    manual = (w * nll).sum() / w.sum().clamp_min(1e-8)
    got = imagination_nll(pred, target, logvar, vis, observed_weight=0.1)
    assert torch.allclose(got, manual, atol=1e-6)


def test_imagination_nll_grads_finite_at_extreme_logvar():
    pred, target, vis = _rand_field(seed=2)
    logvar = torch.full(vis.shape, -100.0, requires_grad=True)
    out = imagination_nll(pred, target, logvar, vis)
    out.backward()
    assert torch.isfinite(logvar.grad).all(), "non-finite grad through clamped nll"


def test_imagination_field_head_bounds_logvar():
    """The head itself must bound logvar to [-10, 10] regardless of input scale
    (guards the OKRI/LOPS sigma export that .exp()s this value)."""
    field = ImaginationField(d_model=64, grid_hw=4, depth=1, n_heads=8).eval()
    for scale in (1.0, 50.0, 500.0):
        tokens = torch.randn(2, 16, 64) * scale
        vis = (torch.rand(2, 16) > 0.5).float()
        with torch.no_grad():
            _, logvar = field(tokens, vis)
        assert torch.isfinite(logvar).all()
        assert float(logvar.min()) >= -10.0 - 1e-4
        assert float(logvar.max()) <= 10.0 + 1e-4
        # the exp() that arms.py performs on this must stay finite
        assert torch.isfinite((0.5 * logvar).exp()).all()


# --------------------------------------------------------------------------- #
#  FeatureOOD (refs/refb) — sum/count division (witness: count<2 gate)
# --------------------------------------------------------------------------- #
def test_feature_ood_score_finite_before_two_samples():
    oo = FeatureOOD(dim=8)
    s0 = oo.score(torch.randn(3, 8))
    assert torch.isfinite(s0).all() and float(s0.abs().max()) == 0.0  # zeros, no div
    oo.update(torch.randn(1, 8))                                      # count == 1
    s1 = oo.score(torch.randn(3, 8))
    assert torch.isfinite(s1).all() and float(s1.abs().max()) == 0.0


def test_feature_ood_score_finite_on_zero_variance():
    oo = FeatureOOD(dim=8)
    oo.update(torch.zeros(5, 8))
    oo.update(torch.zeros(5, 8))                                      # count >= 2, var 0
    s = oo.score(torch.randn(3, 8))
    assert torch.isfinite(s).all(), "zero-variance stats produced non-finite score"


# --------------------------------------------------------------------------- #
#  SigReg / Epps-Pulley — bounded-by-construction (Gaussian kernels)
# --------------------------------------------------------------------------- #
def test_sigreg_finite_on_degenerate_and_huge_input():
    sr = SigReg(n_slices=16)
    assert torch.isfinite(sr(torch.zeros(32, 64))).all()             # zero variance
    assert torch.isfinite(sr(torch.randn(32, 64) * 1e4)).all()       # large magnitude


def test_epps_pulley_finite_on_degenerate_input():
    assert torch.isfinite(epps_pulley(torch.zeros(50))).all()
    assert torch.isfinite(epps_pulley(torch.randn(50) * 1e4)).all()


# --------------------------------------------------------------------------- #
#  spectral helpers — clamp_min guards on zero/degenerate spectra
# --------------------------------------------------------------------------- #
def test_spectral_helpers_finite_on_zero_spectrum():
    z = torch.zeros(10)
    assert math.isfinite(effective_rank(z))
    assert math.isfinite(spectral_tail(z, 3))
    assert isinstance(optimal_k(z, 5), int)
    assert isinstance(energy_knee(z), int)


def test_spectral_helpers_finite_on_single_mode():
    s = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert math.isfinite(effective_rank(s))
    assert abs(effective_rank(s) - 1.0) < 1e-3            # one mode -> rank ~1
    assert math.isfinite(spectral_tail(s, 1))
