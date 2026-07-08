"""Cache-key collision regression tests (Production compliance review #1).

Demonstrates failing-then-passing:
- `legacy_key` is a faithful copy of the CURRENT stack logic
  (epcache.build_episodes_cached lines 30-34). The two `*_legacy_collides`
  tests PROVE the bug exists today.
- `cache_key`/`_source_id` are the proposed fix; the matching `*_fixed_distinct`
  tests prove the fix resolves each collision.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tanitad.data.epcache import _source_id, cache_key  # noqa: E402

PARAMS = {"window": 8, "max_horizon": 4}


def legacy_key(sources, params):
    """Faithful copy of the CURRENT stack key logic (the bug under test)."""
    ids = [getattr(s, "name", None) or (s.get("clip_id") if isinstance(s, dict)
                                        else str(s)) for s in sources]
    return hashlib.sha1(json.dumps({"ids": ids, "params": params},
                                   sort_keys=True, default=str).encode()
                        ).hexdigest()[:12]


# --- Case 1: same basename, different chunk directory (cosmos chunk-pairing class) ---

CHUNK0 = [Path("data/cosmos/chunk_0/scene_000.hevc"),
          Path("data/cosmos/chunk_0/scene_001.hevc")]
CHUNK1 = [Path("data/cosmos/chunk_1/scene_000.hevc"),
          Path("data/cosmos/chunk_1/scene_001.hevc")]


def test_chunk_dirs_legacy_collides():
    # The bug: two DIFFERENT source sets share a cache key today.
    assert legacy_key(CHUNK0, PARAMS) == legacy_key(CHUNK1, PARAMS)


def test_chunk_dirs_fixed_distinct():
    assert cache_key(CHUNK0, PARAMS) != cache_key(CHUNK1, PARAMS)


# --- Case 2: dict sources missing 'clip_id' ---

DICTS_A = [{"path": "x"}, {"path": "y"}]
DICTS_B = [{"path": "p"}, {"path": "q"}]


def test_dicts_without_clip_id_legacy_collides():
    assert legacy_key(DICTS_A, PARAMS) == legacy_key(DICTS_B, PARAMS)


def test_dicts_without_clip_id_fixed_raises():
    # Fail-fast instead of a silent None id.
    with pytest.raises(ValueError):
        cache_key(DICTS_A, PARAMS)


# --- fix keeps legitimate identities stable / distinct ---

def test_dicts_with_clip_id_are_distinct_and_stable():
    a = [{"clip_id": "aaa"}, {"clip_id": "bbb"}]
    b = [{"clip_id": "aaa"}, {"clip_id": "ccc"}]
    assert cache_key(a, PARAMS) != cache_key(b, PARAMS)
    assert cache_key(a, PARAMS) == cache_key(list(a), PARAMS)  # deterministic


def test_params_change_invalidates_key():
    assert cache_key(CHUNK0, PARAMS) != cache_key(CHUNK0, {"window": 6})


def test_source_id_forms():
    assert _source_id(Path("a/b/c.hevc")) == "path:a/b/c.hevc"
    assert _source_id("plain") == "str:plain"
    assert _source_id({"clip_id": 42}) == "clip:42"


# --- integration addition: read-only legacy-dir fallback (no rebuilds) ---

def test_legacy_dir_fallback_reuses_old_cache(tmp_path):
    import torch
    from tanitad.data.epcache import (_legacy_cache_key, build_episodes_cached)
    from tanitad.data.mixing import save_episode
    from tanitad.data.toy_driving import ToyEpisode

    sources = [Path("chunk_0/scene.hevc"), Path("chunk_1/scene.hevc")]
    legacy = tmp_path / f"tag-{_legacy_cache_key(sources, PARAMS)}"
    legacy.mkdir(parents=True)
    for i in range(2):
        ep = ToyEpisode(frames=torch.zeros(4, 9, 8, 8, dtype=torch.uint8),
                        actions=torch.zeros(4, 2), poses=torch.zeros(4, 4),
                        episode_id=i)
        save_episode(ep, str(legacy / f"ep_{i:05d}.pt"))

    def boom(_):
        raise AssertionError("fallback failed - build_one was called")

    eps = build_episodes_cached(sources, boom, tmp_path, "tag", PARAMS)
    assert len(eps) == 2                       # loaded from the legacy dir
