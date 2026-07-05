"""Consequence-dominance (A8) statistics harness (backlog item #4)."""

import numpy as np
import pytest
import torch

from tanitad.data.stats import (consequence_dominance_stats,
                                 episode_change_fraction, format_report,
                                 stats_by_label)
from tanitad.data.toy_driving import ToyEpisode, generate_episode


def _static_ep(episode_id: int, T: int = 20, C: int = 1) -> ToyEpisode:
    """A never-changing clip: A8 fraction must be exactly 0."""
    return ToyEpisode(frames=torch.zeros(T, C, 16, 16),
                      actions=torch.zeros(T, 2), poses=torch.zeros(T, 4),
                      episode_id=episode_id)


def test_static_episode_is_zero():
    assert episode_change_fraction(_static_ep(0), thresh=0.05) == 0.0


def test_toy_is_more_dominant_than_static():
    toy = [generate_episode(i, steps=40, size=64) for i in range(3)]
    st_toy = consequence_dominance_stats(toy, thresholds=(0.05,))
    st_static = consequence_dominance_stats([_static_ep(9)], thresholds=(0.05,))
    assert st_toy.per_threshold[0.05]["mean"] > st_static.per_threshold[0.05]["mean"]
    assert st_toy.per_threshold[0.05]["mean"] > 0.03          # toy clears A8 floor
    assert st_toy.n_episodes == 3 and st_toy.n_frames == 120


def test_percentile_keys_and_ordering():
    eps = [generate_episode(i, steps=30, size=48) for i in range(5)]
    st = consequence_dominance_stats(eps, (0.05, 0.10))
    s = st.per_threshold[0.05]
    for k in ("mean", "median", "p10", "p90", "min", "max"):
        assert k in s
    assert s["min"] <= s["p10"] <= s["median"] <= s["p90"] <= s["max"]
    # higher threshold detects less change
    assert st.per_threshold[0.10]["mean"] <= st.per_threshold[0.05]["mean"]


def test_channels_slice_matches_manual():
    frames = torch.rand(10, 6, 8, 8)
    ep = ToyEpisode(frames=frames, actions=torch.zeros(10, 2),
                    poses=torch.zeros(10, 4), episode_id=0)
    got = episode_change_fraction(ep, 0.05, channels=(3, 6))
    exp = ((frames[1:, 3:6] - frames[:-1, 3:6]).abs() > 0.05).float().mean()
    assert abs(got - float(exp)) < 1e-6


def test_stats_by_label_and_report():
    labelled = [("real", generate_episode(0, steps=30, size=48)),
                ("real", generate_episode(1, steps=30, size=48)),
                ("sim", _static_ep(2, T=30))]
    by = stats_by_label(labelled, thresholds=(0.05,))
    assert set(by) == {"real", "sim"}
    assert by["real"].n_episodes == 2 and by["sim"].n_episodes == 1
    assert by["real"].per_threshold[0.05]["mean"] > by["sim"].per_threshold[0.05]["mean"]
    report = format_report(by, thresh=0.05)
    assert "real" in report and "sim" in report and "A8@0.05" in report


def test_empty_raises():
    with pytest.raises(ValueError):
        consequence_dominance_stats([])
