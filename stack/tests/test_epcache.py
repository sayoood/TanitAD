"""F-6 episode cache: build-once/load-after, fault tolerance, key invalidation."""

import torch

from tanitad.data.epcache import build_episodes_cached
from tanitad.data.toy_driving import generate_episode


def _builder(i):
    if i == 3:
        raise ValueError("corrupt clip")          # must be skipped, not fatal
    return generate_episode(int(i), steps=20, size=64)


def test_build_skip_cache_and_reload(tmp_path):
    srcs = list(range(6))
    eps1 = build_episodes_cached(srcs, _builder, tmp_path, "toy-test",
                                 {"size": 64})
    assert len(eps1) == 5                          # one skipped (fault tolerance)
    assert (tmp_path / [p.name for p in tmp_path.iterdir()][0] / "DONE").exists()

    calls = []
    eps2 = build_episodes_cached(srcs, lambda i: calls.append(i) or _builder(i),
                                 tmp_path, "toy-test", {"size": 64})
    assert not calls                               # loaded from cache, no rebuild
    assert len(eps2) == 5
    assert torch.equal(eps2[0].actions, eps1[0].actions)
    # F-7: cached episodes are disk-backed (mmap); slicing must still work and
    # window datasets consume them unchanged
    win = eps2[0].frames[2:6]
    assert win.shape[0] == 4 and win.dtype == torch.uint8
    from tanitad.data._contract import EpisodeWindowDataset
    ds = EpisodeWindowDataset(eps2, window=4, max_horizon=2)
    item = ds[0]
    assert item["frames"].dtype == torch.float32   # converted per window


def test_param_change_invalidates(tmp_path):
    srcs = list(range(2))
    build_episodes_cached(srcs, _builder, tmp_path, "toy-test", {"size": 64})
    calls = []
    build_episodes_cached(srcs, lambda i: calls.append(i) or _builder(i),
                          tmp_path, "toy-test", {"size": 32})
    assert calls                                   # different params -> rebuild
