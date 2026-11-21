"""Unit tests for the blind-rollout diagnostic metric primitives.

These pin the metric behaviour so the measured curves in the research note are
trustworthy (a perfect prediction must read cosine 1, an unrelated one ~0, etc.).

Run:  <tanitad venv>/python -m pytest tests/ -q   (from the package dir)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blind_rollout import (  # noqa: E402
    _calib_corr, _center, _hidden_cosine, _inter_sample_cos, _rel_l2)


def _tokens(b=4, n=16, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, d, generator=g)


def test_center_removes_per_sample_mean():
    x = _tokens()
    c = _center(x)
    assert torch.allclose(c.mean(dim=1), torch.zeros(x.shape[0], x.shape[2]), atol=1e-5)


def test_perfect_prediction_cosine_one():
    true = _tokens()
    hidden = torch.ones(true.shape[0], true.shape[1], dtype=torch.bool)
    assert _hidden_cosine(true.clone(), true, hidden) > 0.999


def test_unrelated_prediction_cosine_near_zero():
    true = _tokens(seed=1)
    pred = _tokens(seed=2)
    hidden = torch.ones(true.shape[0], true.shape[1], dtype=torch.bool)
    assert abs(_hidden_cosine(pred, true, hidden)) < 0.25       # centered, high-D -> ~0


def test_hidden_mask_restricts_scope():
    true = _tokens()
    pred = true.clone()
    pred[:, :8] += 5.0                                          # corrupt first half
    hid_clean = torch.zeros(true.shape[0], true.shape[1], dtype=torch.bool)
    hid_clean[:, 8:] = True
    hid_dirty = torch.zeros_like(hid_clean)
    hid_dirty[:, :8] = True
    assert _hidden_cosine(pred, true, hid_clean) > _hidden_cosine(pred, true, hid_dirty)


def test_rel_l2_zero_for_perfect_and_positive_for_error():
    true = _tokens()
    hidden = torch.ones(true.shape[0], true.shape[1], dtype=torch.bool)
    assert _rel_l2(true.clone(), true, hidden) < 1e-6
    assert _rel_l2(true + 0.5, true, hidden) > 0.0


def test_rel_l2_mean_token_is_about_one():
    # predicting the per-sample mean token -> rel-L2 ~ 1 by construction.
    true = _tokens()
    hidden = torch.ones(true.shape[0], true.shape[1], dtype=torch.bool)
    pred = true.mean(dim=1, keepdim=True).expand_as(true).contiguous()
    assert 0.7 < _rel_l2(pred, true, hidden) < 1.4


def test_inter_sample_cos_high_when_beliefs_collapse():
    # The metric CENTERS per-sample (removes each sample's DC token), so "attractor
    # collapse" means the centered SPATIAL pattern converges across samples. Build a
    # shared spatial pattern + per-sample DC offset + small noise -> after centering the
    # DC is gone, the shared pattern remains -> inter-sample cosine ~ 1.
    # (Only HALF the cells hidden, as in the real sector mask -- the hidden-subset mean of
    # centered tokens is degenerate only if ALL cells are hidden.)
    b, n, d = 6, 16, 8
    g = torch.Generator().manual_seed(7)
    pattern = torch.randn(n, d, generator=g)                   # shared spatial structure
    dc = torch.randn(b, 1, d, generator=g)                     # per-sample DC (removed by centering)
    pred = pattern.unsqueeze(0) + dc + 0.02 * torch.randn(b, n, d, generator=g)
    hidden = torch.zeros(b, n, dtype=torch.bool)
    hidden[:, n // 2:] = True                                  # half-sector, like sector_mask
    assert _inter_sample_cos(pred, hidden) > 0.8


def test_inter_sample_cos_degenerate_when_all_hidden():
    # documents the metric's scope: with EVERY cell hidden the centered per-sample mean is
    # ~0, so the signal is only meaningful under a partial mask (the real usage).
    pred = _tokens(b=6, n=16, d=8, seed=9)
    allhid = torch.ones(6, 16, dtype=torch.bool)
    assert abs(_inter_sample_cos(pred, allhid)) < 0.5


def test_inter_sample_cos_low_when_beliefs_diverse():
    pred = _tokens(b=6, n=16, d=32, seed=3)
    hidden = torch.ones(6, 16, dtype=torch.bool)
    assert _inter_sample_cos(pred, hidden) < 0.5


def test_calib_corr_positive_when_var_tracks_error():
    # construct per-cell error^2 and a var that increases with it -> positive corr.
    b, n = 4, 16
    err2 = torch.rand(b, n)
    var = err2 * 2.0 + 0.1 * torch.rand(b, n)
    hidden = torch.ones(b, n, dtype=torch.bool)
    assert _calib_corr(err2, var, hidden) > 0.7
