"""Sanity tests for probe_r2_ci (G-B2: analytic / known-ground-truth cases).

Run standalone:  pytest -q "<pkg>/tests"
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_r2_ci import (  # noqa: E402
    BASE_GRID, N_HELD, _r2, _ridge_fit_predict, _standardize,
    gate_held_episodes, mask_for, probe_r2, random_splits, summarize,
)


def _primal_predict(Xtr, ytr, Xte, lam):
    """The literal estimator in taniteval/diag_v2mech.py:ridge_r2."""
    ym = ytr.mean()
    A = Xtr.T @ Xtr + lam * len(Xtr) * torch.eye(Xtr.shape[1]).double()
    w = torch.linalg.solve(A, Xtr.T @ (ytr - ym))
    return Xte @ w + ym


def _toy(n_ep=40, per_ep=22, dim=12, snr=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    eids = [f"ep{e:02d}" for e in range(n_ep) for _ in range(per_ep)]
    z = torch.randn(n_ep * per_ep, dim, generator=g, dtype=torch.double)
    w = torch.randn(dim, generator=g, dtype=torch.double)
    signal = z @ w
    if snr is None:
        y = torch.randn(len(eids), generator=g, dtype=torch.double)  # pure noise
    else:
        noise = torch.randn(len(eids), generator=g, dtype=torch.double)
        y = signal / signal.std() + noise / noise.std() / (snr ** 0.5)
    return z, y, eids


# --- 1. the dual form IS the gate's primal estimator ------------------------
def test_dual_matches_primal_ridge():
    z, y, eids = _toy(seed=1, snr=4.0)
    m = mask_for(eids, gate_held_episodes(eids))
    Xtr, Xte = _standardize(z[~m], z[m])
    for lam in BASE_GRID:
        dual = _ridge_fit_predict(Xtr, y[~m], Xte, lam)
        primal = _primal_predict(Xtr, y[~m], Xte, lam)
        assert torch.allclose(dual, primal, atol=1e-8), lam


# --- 2. known-R^2 recovery --------------------------------------------------
@pytest.mark.parametrize("snr,lo,hi", [(9.0, 0.75, 0.95), (1.0, 0.35, 0.62)])
def test_recovers_known_r2(snr, lo, hi):
    """y = signal + noise at a known SNR -> population R^2 = snr/(1+snr).
    A well-behaved probe lands near it (regularisation costs a little)."""
    z, y, eids = _toy(dim=8, snr=snr, seed=2)
    m = mask_for(eids, gate_held_episodes(eids))
    r2, _ = probe_r2(z, y, m, BASE_GRID)
    assert lo <= r2 <= hi, r2
    assert abs(r2 - snr / (1.0 + snr)) < 0.2


# --- 3. no signal -> R^2 at or below zero -----------------------------------
def test_pure_noise_gives_no_skill():
    z, y, eids = _toy(seed=3, snr=None)
    m = mask_for(eids, gate_held_episodes(eids))
    r2, _ = probe_r2(z, y, m, BASE_GRID)
    assert r2 < 0.15, r2


# --- 4. the gate's deterministic split is reproduced exactly ----------------
def test_gate_split_is_every_fifth_of_forty():
    _, _, eids = _toy(n_ep=40)
    held = gate_held_episodes(eids)
    uniq = sorted(set(eids))
    assert held == set(uniq[::5][:8])
    assert len(held) == N_HELD


# --- 5. random splits are episode-clustered and reproducible ---------------
def test_random_splits_are_clustered_and_seeded():
    _, _, eids = _toy(n_ep=40)
    a = random_splits(eids, 25, seed=7)
    b = random_splits(eids, 25, seed=7)
    assert a == b                                   # deterministic under seed
    assert random_splits(eids, 25, seed=8) != a     # and seed-sensitive
    for held in a:
        assert len(held) == N_HELD
        m = mask_for(eids, held)
        # no episode appears on both sides of the split
        te = {e for e, keep in zip(eids, m.tolist()) if keep}
        tr = {e for e, keep in zip(eids, m.tolist()) if not keep}
        assert te.isdisjoint(tr)
        assert int(m.sum()) == N_HELD * (len(eids) // 40)


# --- 6. summarize reports the quantiles it claims --------------------------
def test_summarize_quantiles_and_gate_fraction():
    vals = [i / 100.0 for i in range(101)]           # 0.00 .. 1.00
    s = summarize(vals)
    assert s["n"] == 101
    assert abs(s["median"] - 0.5) < 1e-9
    assert abs(s["mean"] - 0.5) < 1e-9
    assert abs(s["p2.5"] - 0.025) < 1e-3
    assert abs(s["p97.5"] - 0.975) < 1e-3
    # 0.55 .. 1.00 inclusive = 46 of 101 (summarize rounds to 4 decimals)
    assert abs(s["frac_ge_gate"] - 46 / 101) < 1e-4


# --- 7. _r2 is the textbook coefficient of determination -------------------
def test_r2_definition():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
    assert _r2(y, y) == pytest.approx(1.0)
    assert _r2(y, y.mean().expand_as(y)) == pytest.approx(0.0)
    # residual SS = 4 * 1.0, total SS = 5.0
    assert _r2(y, y + 1.0) == pytest.approx(1.0 - 4.0 / 5.0)


# --- 8. nested lambda never selects on the evaluation fold ------------------
def test_nested_lambda_is_not_selected_on_heldout():
    """With a deliberately bad lambda favoured by the held-out fold, the nested
    estimator must not be able to exceed the test-set-selected one."""
    z, y, eids = _toy(dim=8, snr=2.0, seed=5)
    m = mask_for(eids, gate_held_episodes(eids))
    r2_sel, _ = probe_r2(z, y, m, BASE_GRID, nested=False)
    r2_nest, _ = probe_r2(z, y, m, BASE_GRID, nested=True, seed=5)
    assert r2_nest <= r2_sel + 1e-12
