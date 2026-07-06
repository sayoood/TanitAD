"""Mix mechanics (D-010) — validated on CI fixtures (allowed toy use)."""

import pytest
import torch

from tanitad.data.mixing import MixedWindowDataset, load_episode, save_episode
from tanitad.data.toy_driving import ToyDrivingDataset, generate_episode


def _ds(ids):
    return ToyDrivingDataset(ids, window=4, max_horizon=2, size=64, steps=30)


def test_mix_ratio_and_domain_tags():
    mixed = MixedWindowDataset([(_ds([0, 1]), 0.8), (_ds([2]), 0.2)],
                               length=500, seed=0)
    rep = mixed.mix_report()
    assert 0.72 < rep["domain_0_frac"] < 0.88          # ~80/20 within noise
    item = mixed[0]
    assert item["domain"] in (0, 1)
    assert item["frames"].shape == (4, 1, 64, 64)


def test_mix_is_deterministic():
    a = MixedWindowDataset([(_ds([0]), 0.5), (_ds([1]), 0.5)], length=50, seed=7)
    b = MixedWindowDataset([(_ds([0]), 0.5), (_ds([1]), 0.5)], length=50, seed=7)
    assert torch.equal(a._src, b._src) and torch.equal(a._item, b._item)


def test_contract_mismatch_fails_fast():
    small = ToyDrivingDataset([0], window=4, max_horizon=2, size=32, steps=30)
    with pytest.raises(AssertionError, match="contract mismatch"):
        MixedWindowDataset([(_ds([0]), 0.5), (small, 0.5)])


def test_episode_save_load_roundtrip(tmp_path):
    from tanitad.data._contract import to_float_frames
    ep = generate_episode(3, steps=20, size=64)
    p = str(tmp_path / "ep.pt")
    save_episode(ep, p)
    ep2 = load_episode(p)
    assert ep2.frames.shape == ep.frames.shape
    assert ep2.frames.dtype == torch.uint8          # memory layout (pod-OOM fix)
    assert torch.allclose(ep.frames, to_float_frames(ep2.frames),
                          atol=1 / 255 + 1e-6)
    assert torch.equal(ep.actions, ep2.actions)
    assert ep2.episode_id == ep.episode_id
