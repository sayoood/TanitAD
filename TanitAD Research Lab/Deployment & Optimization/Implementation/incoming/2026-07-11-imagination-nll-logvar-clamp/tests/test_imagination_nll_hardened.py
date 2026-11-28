"""Standalone tests for the fail-safe imagination NLL (review #3, P1.7).

Run:
  pytest "TanitAD Research Hub/Production & Optimization/Implementation/incoming/2026-07-11-imagination-nll-logvar-clamp/tests" -q

Self-contained: torch only, no tanitad / CUDA / data. Each test documents the
CURRENT stack behaviour it corrects. The ORIGINAL (unclamped) NLL term is
redefined inline so the overflow/parity contrast is explicit.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imagination_nll_hardened import d9_rows, imagination_nll  # noqa: E402


# --- the CURRENT stack formula (unclamped), for contrast ------------------- #
def _nll_original(pred, target, logvar, vis, observed_weight=0.1):
    err2 = (pred - target).pow(2).mean(dim=-1)
    nll = 0.5 * (torch.exp(-logvar) * err2 + logvar)          # UNCLAMPED
    w = (1.0 - vis) + observed_weight * vis
    return (w * nll).sum() / w.sum().clamp_min(1e-8)


def _mk(B=2, N=5, D=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(B, N, D, generator=g)
    target = torch.randn(B, N, D, generator=g)
    vis = (torch.rand(B, N, generator=g) > 0.5).float()
    return pred, target, vis


# --------------------------------------------------------------------------- #
#  Defect 1 — overflow / NaN safety
# --------------------------------------------------------------------------- #
def test_in_band_parity_matches_original_exactly():
    """logvar inside [-8, 8]: hardened == original (clamp is identity in-band ->
    NO training-behaviour change where logvar is well-conditioned)."""
    pred, target, vis = _mk()
    for lv_val in (-8.0, -3.0, 0.0, 2.5, 8.0):
        logvar = torch.full(vis.shape, lv_val)
        a = _nll_original(pred, target, logvar, vis).double()
        b = imagination_nll(pred, target, logvar, vis).double()
        assert torch.allclose(a, b, atol=1e-12), f"parity broke at logvar={lv_val}"


@pytest.mark.parametrize("dtype,bad_logvar", [
    (torch.float16, -20.0),     # fp16 overflows below -11.09
    (torch.float32, -100.0),    # fp32 overflows below -88.72
])
def test_original_overflows_hardened_stays_finite(dtype, bad_logvar):
    """CURRENT stack: exp(-logvar) overflows -> non-finite loss -> NaN-corrupts
    the model via backward/opt.step. Fixed: hardened loss is finite."""
    pred, target, vis = _mk()
    pred, target, vis = pred.to(dtype), target.to(dtype), vis.to(dtype)
    logvar = torch.full(vis.shape, bad_logvar, dtype=dtype)
    orig = _nll_original(pred, target, logvar, vis)
    hard = imagination_nll(pred, target, logvar, vis)
    assert not torch.isfinite(orig), "expected the unclamped term to blow up"
    assert torch.isfinite(hard), "hardened loss must stay finite"


def test_hardened_gradient_finite_at_extreme_logvar():
    """The NaN propagates through backward in the stack; the hardened path must
    keep gradients finite so opt.step never writes NaN into the weights."""
    pred, target, vis = _mk()
    logvar = torch.full(vis.shape, -60.0, requires_grad=True)
    loss = imagination_nll(pred, target, logvar, vis)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logvar.grad).all()


def test_finite_over_full_logvar_range_both_dtypes():
    """Per-cell (N=1) sweep of logvar in [-160, 160] x a wide err2 range; the
    hardened loss must be finite in fp32 AND fp16 (deployment precision). N=1
    isolates the exp-overflow fix from the separate fp16 large-sum-reduction
    edge (autocast upcasts reductions in practice); the exp is the actual bug."""
    for dtype in (torch.float32, torch.float16):
        for lv_val in torch.linspace(-160, 160, 65).tolist():
            for e in (0.0, 1e-3, 1.0, 4.0, 20.0):
                pred = torch.zeros(1, 1, 1, dtype=dtype)
                target = torch.full((1, 1, 1), math.sqrt(e), dtype=dtype)
                vis = torch.zeros(1, 1, dtype=dtype)
                logvar = torch.full((1, 1), lv_val, dtype=dtype)
                out = imagination_nll(pred, target, logvar, vis)
                assert torch.isfinite(out), \
                    f"non-finite at dtype={dtype}, logvar={lv_val}, err2~{e}"


def test_clamp_parameter_respected():
    """A tighter clamp must bound exp(-logvar) more aggressively (monotone)."""
    pred, target, vis = _mk()
    logvar = torch.full(vis.shape, -50.0)
    loose = imagination_nll(pred, target, logvar, vis, logvar_clamp=8.0)
    tight = imagination_nll(pred, target, logvar, vis, logvar_clamp=4.0)
    # both finite; tighter clamp -> smaller exp(-logvar) -> smaller nll term
    assert torch.isfinite(loose) and torch.isfinite(tight)
    assert float(tight) < float(loose)


def test_weighting_semantics_unchanged():
    """Hidden cells (vis=0) carry full weight, visible get observed_weight — the
    clamp must not touch the weighting (checked via in-band parity with vis)."""
    pred, target, vis = _mk(seed=3)
    logvar = torch.zeros(vis.shape)                  # in-band
    a = _nll_original(pred, target, logvar, vis, observed_weight=0.25).double()
    b = imagination_nll(pred, target, logvar, vis, observed_weight=0.25).double()
    assert torch.allclose(a, b, atol=1e-12)


# --------------------------------------------------------------------------- #
#  Defect 2 — d9_rows determinism
# --------------------------------------------------------------------------- #
def test_d9_shuffled_cosine_reproducible_with_generator():
    """Same generator seed -> identical shuffled_cosine (reproducible D9 floor);
    the current stack draws an unseeded randperm -> non-reproducible."""
    torch.manual_seed(1)
    B, N, D = 2, 8, 4
    pred, target = torch.randn(B, N, D), torch.randn(B, N, D)
    logvar = torch.randn(B, N)
    # mix of hidden/visible so calibration_gap is finite (needs both)
    vis = torch.tensor([[0., 0., 0., 0., 1., 1., 1., 1.]] * B)
    r1 = d9_rows(pred, target, logvar, vis, generator=torch.Generator().manual_seed(7))
    r2 = d9_rows(pred, target, logvar, vis, generator=torch.Generator().manual_seed(7))
    assert r1["shuffled_cosine"] == r2["shuffled_cosine"]
    # hidden_cosine / calibration_gap never depend on the shuffle
    assert r1["hidden_cosine"] == r2["hidden_cosine"]
    assert r1["calibration_gap"] == r2["calibration_gap"]


def test_d9_default_generator_still_returns_finite_rows():
    """generator=None preserves the current call signature/behaviour."""
    pred, target = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
    logvar = torch.randn(2, 6)
    vis = torch.tensor([[0., 0., 0., 1., 1., 1.], [0., 1., 0., 1., 0., 1.]])
    r = d9_rows(pred, target, logvar, vis)
    assert set(r) == {"hidden_cosine", "shuffled_cosine", "calibration_gap"}
    assert all(math.isfinite(v) for v in r.values())


def test_d9_no_hidden_cells_returns_nan_rows():
    """All-visible -> the current guard returns NaN rows (unchanged)."""
    pred, target = torch.randn(1, 4, 3), torch.randn(1, 4, 3)
    logvar, vis = torch.randn(1, 4), torch.ones(1, 4)     # none hidden
    r = d9_rows(pred, target, logvar, vis)
    assert all(math.isnan(v) for v in r.values())
