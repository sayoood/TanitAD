"""Fail-fast save_episode regression tests (Production compliance review #1)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tanitad.data.mixing import save_episode  # noqa: E402

from tanitad.data.toy_driving import ToyEpisode


def _ep(T=5, C=1, H=8, W=8, act_T=None, pose_T=None):
    act_T = T if act_T is None else act_T
    pose_T = T if pose_T is None else pose_T
    return ToyEpisode(
        frames=torch.zeros(T, C, H, W, dtype=torch.uint8),
        actions=torch.zeros(act_T, 2),
        poses=torch.zeros(pose_T, 4),
        episode_id=0)


def test_valid_episode_saves(tmp_path):
    p = tmp_path / "ep.pt"
    save_episode(_ep(), str(p))
    assert p.exists()
    d = torch.load(str(p), weights_only=True)
    assert d["frames_u8"].shape[0] == 5


def test_actions_length_mismatch_raises(tmp_path):
    # The bug class: mis-shaped episode was persisted silently before this fix.
    with pytest.raises(ValueError, match="actions"):
        save_episode(_ep(T=5, act_T=4), str(tmp_path / "bad.pt"))


def test_poses_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="poses"):
        save_episode(_ep(T=5, pose_T=6), str(tmp_path / "bad.pt"))


def test_non_4d_frames_raises(tmp_path):
    ep = ToyEpisode(frames=torch.zeros(5, 8, 8, dtype=torch.uint8),
                    actions=torch.zeros(5, 2), poses=torch.zeros(5, 4),
                    episode_id=0)
    with pytest.raises(ValueError, match="frames"):
        save_episode(ep, str(tmp_path / "bad.pt"))


def test_float_frames_round_trip(tmp_path):
    ep = ToyEpisode(frames=torch.ones(3, 1, 4, 4) * 0.5,
                    actions=torch.zeros(3, 2), poses=torch.zeros(3, 4),
                    episode_id=7)
    p = tmp_path / "f.pt"
    save_episode(ep, str(p))
    d = torch.load(str(p), weights_only=True)
    assert d["frames_u8"].dtype == torch.uint8
    assert int(d["frames_u8"].max()) == 127  # 0.5*255 truncated
