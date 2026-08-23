"""Standalone tests for the D1 bootstrap wrapper (G-B2 / G-E)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_d1_bootstrap import run_d1_bootstrap  # noqa: E402


def _synthetic(n_eps, w=30, spread=6.0, seed=0):
    """n_eps routes, each a constant ego-target offset -> heterogeneous per-route ADE."""
    g = torch.Generator().manual_seed(seed)
    S, Y, E = [], [], []
    offs = spread * torch.rand(n_eps, generator=g)
    for e in range(n_eps):
        S.append(torch.randn(w, 64, generator=g))
        Y.append(torch.full((w, 2), float(offs[e])))
        E += [e] * w
    return torch.cat(S), torch.cat(Y), E


def test_returns_expected_keys_and_forwards_seed0():
    S, Y, E = _synthetic(20)
    from tanitad.eval.gates import run_d1
    stats = run_d1_bootstrap(S, Y, E, seeds=6, val_frac=0.2)
    for k in ("ade@1s_mean", "ade@1s_ci95", "ade@1s_ci95_halfwidth",
              "seeds", "n_val_eps_approx", "single_seed0", "decision_grade"):
        assert k in stats
    # the wrapper must not alter the estimator: seed0 == plain run_d1 seed0
    r0 = run_d1(S, Y, E, unit="camera", alpha=1e-3, val_frac=0.2, seed=0)
    assert abs(stats["single_seed0"] - round(r0.metrics["ade@1s"], 3)) < 1e-6


def test_ci_halfwidth_shrinks_with_more_episodes():
    """More val episodes -> tighter CI (the audit's core claim, on the real estimator)."""
    S_s, Y_s, E_s = _synthetic(10, seed=1)     # ~2 val eps @ val_frac 0.2
    S_l, Y_l, E_l = _synthetic(40, seed=1)     # ~8 val eps
    hw_small = run_d1_bootstrap(S_s, Y_s, E_s, seeds=12)["ade@1s_ci95_halfwidth"]
    hw_large = run_d1_bootstrap(S_l, Y_l, E_l, seeds=12)["ade@1s_ci95_halfwidth"]
    assert hw_large < hw_small


def test_rejects_single_seed():
    S, Y, E = _synthetic(20)
    with pytest.raises(ValueError):
        run_d1_bootstrap(S, Y, E, seeds=1)


def test_decision_grade_flag_requires_enough_val_eps():
    # 10 eps @ val_frac 0.2 -> ~2 val eps -> can't be decision-grade regardless of CI
    S, Y, E = _synthetic(10, seed=3)
    stats = run_d1_bootstrap(S, Y, E, seeds=8, val_frac=0.2)
    assert stats["n_val_eps_approx"] < 20
    assert stats["decision_grade"] is False
