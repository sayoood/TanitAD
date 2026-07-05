"""Toy driving env: contract shapes + consequence-dominance (A8)."""

from tanitad.data.toy_driving import (ToyDrivingDataset, frame_change_fraction,
                                      generate_episode)


def test_episode_contract():
    ep = generate_episode(0, steps=40, size=64)
    assert ep.frames.shape == (40, 1, 64, 64)
    assert ep.actions.shape == (40, 2)
    assert ep.poses.shape == (40, 4)
    assert 0.0 <= ep.frames.min() and ep.frames.max() <= 1.0


def test_consequence_dominance():
    """Ego-centric rendering must put the action's consequence in the frame:
    per-step change well above the Two-Rooms dot regime (~1 %)."""
    ep = generate_episode(1, steps=60, size=64)
    assert frame_change_fraction(ep) > 0.03


def test_dataset_windows():
    ds = ToyDrivingDataset([0, 1], window=4, max_horizon=2, size=64, steps=30)
    item = ds[0]
    assert item["frames"].shape == (4, 1, 64, 64)
    assert item["actions"].shape == (4, 2)
    assert item["future_frames"].shape == (2, 1, 64, 64)
    assert len(ds) == 2 * (30 - 4 - 2)
