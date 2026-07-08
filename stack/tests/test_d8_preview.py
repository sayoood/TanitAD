"""Tests for the SC-05/D8 preview harness (scripts/d8_preview.py)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from d8_preview import imag_rel_scores, rank_auroc  # noqa: E402

from tanitad.config import smoke_config
from tanitad.data.toy_driving import ToyEpisode
from tanitad.models.fourbrain import WorldModel


def test_rank_auroc_known_values():
    hi = torch.tensor([2.0, 3.0, 4.0])
    lo = torch.tensor([0.0, 0.5, 1.0])
    assert rank_auroc(hi, lo) == 1.0                    # perfect separation
    assert rank_auroc(lo, hi) == 0.0                    # perfectly inverted
    same = torch.tensor([1.0, 1.0])
    assert abs(rank_auroc(same, same) - 0.5) < 1e-9     # ties -> chance


def test_rank_auroc_partial_overlap():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([2.5, 3.5])
    # pairs where a > b: (3,2.5),(4,2.5),(4,3.5) = 3 of 8
    assert abs(rank_auroc(a, b) - 3 / 8) < 1e-9


def test_imag_rel_scores_smoke_model():
    cfg = smoke_config()
    world = WorldModel(cfg).eval()
    T, S = 20, cfg.encoder.image_size
    ep = ToyEpisode(frames=torch.randint(0, 255,
                                         (T, cfg.encoder.in_channels, S, S),
                                         dtype=torch.uint8),
                    actions=torch.randn(T, 2) * 0.1,
                    poses=torch.zeros(T, 4), episode_id=1)
    s = imag_rel_scores(world, [ep], "cpu", world.predictor.cfg.window,
                        stride=4, batch=2, max_windows=8)
    assert s.numel() > 0 and s.numel() <= 8
    assert torch.isfinite(s).all() and (s >= 0).all()
