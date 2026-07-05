"""MetaDrive adapter: contract-conversion helpers (no sim) + live rollout (skip).

The pure helpers are the load-bearing logic -- they guarantee a MetaDrive
episode is byte-for-byte contract-compatible with a toy episode. They run in CI
with zero simulator dependencies. The live rollout test skips when MetaDrive is
not importable (`pip install -e .[sim]`, supervised session).
"""

import math

import numpy as np
import pytest
import torch

from tanitad.data.metadrive_env import (MetaDriveDataset, assemble_episode,
                                        bev_frame_from_rgb, finite_diff_accel,
                                        frame_change_fraction, kmh_to_ms,
                                        pose_from_state, steering_to_rad)
from tanitad.data.toy_driving import ToyEpisode


def test_bev_frame_from_uint8_rgb():
    rgb = (np.random.default_rng(0).random((48, 80, 3)) * 255).astype(np.uint8)
    f = bev_frame_from_rgb(rgb, size=64)
    assert f.shape == (1, 64, 64)
    assert f.dtype == torch.float32
    assert 0.0 <= float(f.min()) and float(f.max()) <= 1.0


def test_bev_frame_accepts_gray_and_rgba():
    for arr in (np.zeros((32, 32), np.uint8),
                np.ones((32, 32, 4), np.uint8) * 255):
        f = bev_frame_from_rgb(arr, size=16)
        assert f.shape == (1, 16, 16)
        assert 0.0 <= float(f.min()) and float(f.max()) <= 1.0


def test_unit_conversions():
    assert kmh_to_ms(36.0) == pytest.approx(10.0)
    assert steering_to_rad(1.0, 40.0) == pytest.approx(math.radians(40.0))
    assert steering_to_rad(0.0) == 0.0


def test_finite_diff_accel_matches_definition():
    v = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)  # dt=0.5 -> a=[2,4,6,6]
    a = finite_diff_accel(v, dt=0.5)
    assert a.shape == v.shape
    np.testing.assert_allclose(a, [2.0, 4.0, 6.0, 6.0], rtol=1e-5)


def test_assemble_episode_satisfies_contract():
    T, H = 10, 64
    frames = [torch.rand(1, H, H) for _ in range(T)]
    poses = [pose_from_state(float(i), 0.0, 0.1 * i, 8.0 + 0.5 * i)
             for i in range(T)]
    steer = [0.01 * i for i in range(T)]
    ep = assemble_episode(frames, poses, steer, dt=0.1, episode_id=7)
    assert isinstance(ep, ToyEpisode)
    assert ep.frames.shape == (T, 1, H, H)
    assert ep.actions.shape == (T, 2)
    assert ep.poses.shape == (T, 4)
    assert ep.episode_id == 7
    # accel column equals finite-diff of the pose speeds
    np.testing.assert_allclose(
        ep.actions[:, 1].numpy(),
        finite_diff_accel(ep.poses[:, 3].numpy(), 0.1), rtol=1e-5)


def test_dataset_window_contract_matches_toy():
    """A MetaDrive episode fed to MetaDriveDataset yields the toy dict contract."""
    T, H, w, hz = 30, 64, 4, 2
    ep = assemble_episode(
        [torch.rand(1, H, H) for _ in range(T)],
        [pose_from_state(float(i), 0.0, 0.0, 8.0) for i in range(T)],
        [0.0] * T, dt=0.1, episode_id=0)
    ds = MetaDriveDataset([ep, ep], window=w, max_horizon=hz)
    item = ds[0]
    assert item["frames"].shape == (w, 1, H, H)
    assert item["actions"].shape == (w, 2)
    assert item["future_frames"].shape == (hz, 1, H, H)
    assert item["future_poses"].shape == (hz, 4)
    assert item["pose_last"].shape == (4,)
    assert len(ds) == 2 * (T - w - hz)


def test_frame_change_fraction_detects_motion():
    still = torch.zeros(5, 1, 16, 16)
    assert frame_change_fraction(still) == 0.0
    moving = torch.rand(5, 1, 16, 16)
    assert frame_change_fraction(moving) > 0.0


@pytest.mark.slow
def test_metadrive_live_episode():
    """Live smoke: only runs where MetaDrive is installed; skips otherwise."""
    pytest.importorskip("metadrive")
    from tanitad.data.metadrive_env import (MetaDriveEpisodeConfig,
                                            generate_metadrive_episode)
    ep = generate_metadrive_episode(0, MetaDriveEpisodeConfig(steps=20, size=64))
    assert ep.frames.shape[1:] == (1, 64, 64)
    assert ep.actions.shape == (ep.frames.shape[0], 2)
    assert ep.poses.shape == (ep.frames.shape[0], 4)
    # ego-centric top-down must be consequence-dominant (A8)
    assert frame_change_fraction(ep.frames) > 0.01
