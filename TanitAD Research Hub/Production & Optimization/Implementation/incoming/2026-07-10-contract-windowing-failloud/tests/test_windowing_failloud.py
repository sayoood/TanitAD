"""Standalone tests for the fail-loud episode windowing (review #3).

Run:
  pytest "TanitAD Research Hub/Production & Optimization/Implementation/incoming/2026-07-10-contract-windowing-failloud/tests" -q

Self-contained: builds tiny contract-shaped episodes (no tanitad, no real data,
no CUDA). Each test also documents the CURRENT stack behaviour it corrects.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windowing import (EpisodeWindowDataset, build_window_index,  # noqa: E402
                       min_episode_length)


@dataclass
class _Ep:
    """Minimal episode honouring the contract (frames [T,C,H,W], etc.)."""
    frames: torch.Tensor
    actions: torch.Tensor
    poses: torch.Tensor
    episode_id: int


def make_ep(T: int, eid: int = 0, C: int = 1, S: int = 2) -> _Ep:
    return _Ep(
        frames=torch.arange(T * C * S * S, dtype=torch.float32).reshape(T, C, S, S)
        / max(1, T * C * S * S),                       # in [0,1]
        actions=torch.zeros(T, 2),
        poses=torch.zeros(T, 4),
        episode_id=eid,
    )


W, H = 6, 4                         # window, max_horizon -> min length 11
MIN = W + H + 1


def test_min_length_formula():
    assert min_episode_length(W, H) == MIN == 11


def test_window_count_matches_old_convention():
    """A long episode yields EXACTLY range(T-window-max_horizon) windows —
    byte-identical to the current stack code (parity, no behaviour drift)."""
    T = 20
    idx, dropped = build_window_index([T], W, H)
    assert dropped == []
    assert len(idx) == T - W - H                        # == 10
    assert idx[0] == (0, 0) and idx[-1] == (0, T - W - H - 1)


def test_short_episode_silently_dropped_now_warns():
    """CURRENT stack: the T=5 episode's `range(5-6-4)=range(-5)` is empty, so it
    is dropped with NO signal. Fixed: dropped-count exposed + a warning fires."""
    eps = [make_ep(20, eid=100), make_ep(5, eid=101)]   # 2nd too short
    with pytest.warns(UserWarning, match="contribute 0"):
        ds = EpisodeWindowDataset(eps, window=W, max_horizon=H)
    assert ds.n_dropped_episodes == 1
    assert ds.dropped == [(1, 5)]
    assert len(ds) == 20 - W - H                         # only the long episode
    # every index points at the surviving (long) episode
    assert all(e_i == 0 for e_i, _ in ds.index)


def test_boundary_episode_exactly_min_length_kept():
    """T == window+max_horizon+1 must yield exactly one window (not dropped)."""
    idx, dropped = build_window_index([MIN], W, H)
    assert dropped == []
    assert len(idx) == 1 and idx == [(0, 0)]
    # one below the boundary is dropped. Pair it with a surviving (long) episode
    # so the index is non-empty and the drop is observable — a SOLE too-short
    # episode correctly *raises* (empty index), covered by the all-short tests.
    with pytest.warns(UserWarning, match="contribute 0"):
        idx2, dropped2 = build_window_index([MIN - 1, 20], W, H)
    assert dropped2 == [(0, MIN - 1)]
    assert len(idx2) == 20 - W - H and all(e_i == 1 for e_i, _ in idx2)


def test_all_short_raises_clear_error_not_silent_empty():
    """CURRENT stack: all-too-short -> len(ds)==0 -> trainer spins on
    StopIteration / crashes deep in training. Fixed: loud ValueError at build."""
    eps = [make_ep(5, eid=1), make_ep(8, eid=2), make_ep(3, eid=3)]
    with pytest.raises(ValueError, match="EMPTY"):
        EpisodeWindowDataset(eps, window=W, max_horizon=H)


def test_all_short_error_names_the_numbers():
    with pytest.raises(ValueError) as ei:
        build_window_index([5, 8], W, H)
    msg = str(ei.value)
    assert f"window={W}" in msg and f"max_horizon={H}" in msg
    assert str(MIN) in msg                               # required min length named


def test_no_warning_when_all_episodes_valid():
    """Regression: the happy path must be warning-free."""
    import warnings
    eps = [make_ep(20, eid=1), make_ep(15, eid=2)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # any warning -> failure
        ds = EpisodeWindowDataset(eps, window=W, max_horizon=H)
    assert ds.n_dropped_episodes == 0
    assert len(ds) == (20 - W - H) + (15 - W - H)


def test_getitem_returns_contract_shapes():
    """__getitem__ returns the byte-identical window dict (shapes + keys)."""
    ds = EpisodeWindowDataset([make_ep(20, eid=7, C=1, S=3)], window=W, max_horizon=H)
    item = ds[0]
    assert set(item) == {"frames", "actions", "future_frames", "future_actions",
                         "future_poses", "pose_last", "episode_id"}
    assert item["frames"].shape == (W, 1, 3, 3)
    assert item["actions"].shape == (W, 2)
    assert item["future_frames"].shape == (H, 1, 3, 3)
    assert item["future_poses"].shape == (H, 4)
    assert item["pose_last"].shape == (4,)
    assert item["episode_id"] == 7


def test_uint8_frames_converted_to_float01():
    ep = _Ep(frames=torch.randint(0, 256, (20, 1, 3, 3), dtype=torch.uint8),
             actions=torch.zeros(20, 2), poses=torch.zeros(20, 4), episode_id=0)
    ds = EpisodeWindowDataset([ep], window=W, max_horizon=H)
    fr = ds[0]["frames"]
    assert fr.dtype == torch.float32 and 0.0 <= float(fr.min()) and float(fr.max()) <= 1.0


def test_invalid_window_raises():
    with pytest.raises(ValueError, match="window"):
        build_window_index([20], 0, H)
