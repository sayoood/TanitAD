"""Contract + memory tests for the v2 compressed-cache lazy dataloader
(``tanitad.data.v2_dataset``) and its ``--v2-cache`` wiring into the flagship
trainer.

The v2 corpus (physicalai-v2bal) is a JPEG-compressed on-disk cache too large to
hold decoded in RAM, so the trainer reads it through lazy, LRU-bounded
``LazyV2Episode`` providers instead of a materialised episode list. These tests
pin the property the whole integration rests on:

  (1) a lazy provider's PARTIAL frame decode is BYTE-IDENTICAL to a full decode
      (``decode_full_episode`` == ``scripts/v2_compressed.load_compressed`` on
      frames/poses/actions -- validated separately on real data on the eval pod);
  (2) feeding the providers to the UNCHANGED ``FlagshipWindowDataset`` yields
      windows that are VALUE-FOR-VALUE identical to feeding a materialised
      ``ToyEpisode`` of the same clip -- for both label regimes (v1 and v2) -- so
      the trainer cannot tell the v2 source from the raw source;
  (3) the emitted window matches the raw-epcache window SCHEMA (keys/shapes/
      dtypes the trainer consumes);
  (4) RAM stays bounded: the payload LRU never exceeds its cap across many
      window fetches spanning many clips;
  (5) the ``--v2-cache`` flag threads through the trainer's arg parser and its
      default (absent) leaves the raw path untouched.

CPU-only, synthetic ``*.v2ep.pt`` written in the exact ``build_compressed``
on-disk format (no pandas/pyav needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

tvio = pytest.importorskip("torchvision.io")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data.toy_driving import ToyEpisode  # noqa: E402
from tanitad.data.v2_dataset import (  # noqa: E402
    build_v2_providers, decode_full_episode, load_or_build_manifest)

WINDOW, MAXH, MANH, NSTACK, S = 4, 6, 3, 3, 32


def _write_v2ep(path: Path, n_raw: int, eid: int, seed: int,
                n_stack: int = NSTACK, size: int = S, quality: int = 90) -> None:
    """Write one synthetic clip in the exact ``build_compressed`` payload format
    (JPEG-encoded [3,S,S] frames + float32 poses/actions)."""
    g = torch.Generator().manual_seed(seed)
    vid = (torch.rand(n_raw, 3, size, size, generator=g) * 255).to(torch.uint8)
    jpegs = [tvio.encode_jpeg(vid[i].contiguous(), quality=quality)
             for i in range(n_raw)]
    lens = torch.tensor([int(j.numel()) for j in jpegs], dtype=torch.int64)
    buf = torch.cat(jpegs)
    poses = torch.randn(n_raw, 4, generator=g)
    poses[:, 3] = poses[:, 3].abs() * 5.0                 # non-negative speed
    actions = torch.randn(n_raw, 2, generator=g)
    torch.save({"jpeg_buf": buf, "jpeg_len": lens,
                "actions": actions, "poses": poses,
                "n_stack": n_stack, "image_size": size, "episode_id": eid,
                "clip_id": f"clip{eid:08d}", "quality": quality}, str(path))


def _fw_dataset(episodes, labels_v2: bool):
    """FlagshipWindowDataset built the SAME way ``_wrap`` builds it (the exact
    trainer path), with explicit dims (no heavy config/model needed)."""
    from train_flagship4b import FlagshipWindowDataset  # noqa: E402
    return FlagshipWindowDataset(episodes, window=WINDOW, max_horizon=MAXH,
                                 maneuver_h=MANH, channels=3 * NSTACK,
                                 labels_v2=labels_v2)


# --------------------------------------------------------------------------- #
def test_partial_decode_is_byte_identical(tmp_path):
    p = tmp_path / f"{'a'*8}.v2ep.pt"
    _write_v2ep(p, n_raw=24, eid=0x30313233, seed=1)
    full = decode_full_episode(str(p))               # reference (full decode)
    (prov,) = build_v2_providers(tmp_path, verbose=False)

    T = prov.frames.shape[0]
    assert tuple(prov.frames.shape) == (24 - (NSTACK - 1), 3 * NSTACK, S, S) == \
        tuple(full.frames.shape)
    assert prov.frames.dtype == torch.uint8 and prov.frames.ndim == 4
    # every contiguous slice the window path uses must equal the full decode
    for a, b in [(0, T), (0, WINDOW), (WINDOW, WINDOW + MAXH), (T - 1, T),
                 (5, 5 + WINDOW)]:
        assert torch.equal(prov.frames[a:b], full.frames[a:b]), (a, b)
    # poses / actions / id mirror load_compressed's [k:] slice, float32
    assert torch.equal(prov.poses, full.poses) and prov.poses.dtype == torch.float32
    assert torch.equal(prov.actions, full.actions) and prov.actions.dtype == torch.float32
    assert prov.episode_id == full.episode_id == 0x30313233


@pytest.mark.parametrize("labels_v2", [False, True])
def test_window_identical_to_materialised_episode(tmp_path, labels_v2):
    p = tmp_path / f"{'b'*8}.v2ep.pt"
    _write_v2ep(p, n_raw=40, eid=7, seed=2)
    full = decode_full_episode(str(p))
    materialised = ToyEpisode(frames=full.frames, actions=full.actions,
                              poses=full.poses, episode_id=full.episode_id)

    (prov,) = build_v2_providers(tmp_path, verbose=False)
    ds_v2 = _fw_dataset([prov], labels_v2)
    ds_raw = _fw_dataset([materialised], labels_v2)
    assert len(ds_v2) == len(ds_raw) > 0

    for i in range(len(ds_v2)):
        a, b = ds_v2[i], ds_raw[i]
        assert set(a) == set(b), (set(a) ^ set(b))
        for key in a:
            va, vb = a[key], b[key]
            if torch.is_tensor(vb):
                assert torch.is_tensor(va), key
                assert va.shape == vb.shape and va.dtype == vb.dtype, (key, va.shape,
                                                                       vb.shape)
                assert torch.equal(va, vb), key
            else:
                assert va == vb, key


def test_window_matches_raw_epcache_schema(tmp_path):
    """The v2 window schema == an INDEPENDENT raw 9-channel ToyEpisode window
    (the keys/shapes/dtypes the trainer's raw path emits)."""
    p = tmp_path / f"{'c'*8}.v2ep.pt"
    _write_v2ep(p, n_raw=40, eid=9, seed=3)
    (prov,) = build_v2_providers(tmp_path, verbose=False)
    T = prov.frames.shape[0]
    raw = ToyEpisode(
        frames=(torch.rand(T, 3 * NSTACK, S, S) * 255).to(torch.uint8),
        actions=torch.randn(T, 2), poses=torch.randn(T, 4), episode_id=1)

    wv = _fw_dataset([prov], labels_v2=False)[0]
    wr = _fw_dataset([raw], labels_v2=False)[0]
    assert set(wv) == set(wr)
    for key in wr:
        if torch.is_tensor(wr[key]):
            assert wv[key].shape == wr[key].shape, key
            assert wv[key].dtype == wr[key].dtype, key
    # frames are float32 in [0,1] after to_float_frames; 9-channel stack
    assert wv["frames"].dtype == torch.float32
    assert wv["frames"].shape == (WINDOW, 3 * NSTACK, S, S)
    assert float(wv["frames"].max()) <= 1.0 and float(wv["frames"].min()) >= 0.0


def test_lru_is_bounded_across_many_fetches(tmp_path):
    for j in range(4):
        _write_v2ep(tmp_path / f"clip{j:08d}.v2ep.pt", n_raw=30, eid=j, seed=10 + j)
    providers = build_v2_providers(tmp_path, lru_size=2, verbose=False)
    assert len(providers) == 4
    cache = providers[0].frames._cache            # shared per-dir cache
    ds = _fw_dataset(providers, labels_v2=False)
    # fetch windows spanning all clips in a shuffled-ish order
    order = list(range(len(ds)))
    order = order[::3] + order[1::3] + order[2::3]
    for i in order:
        _ = ds[i]["frames"]
        assert cache._lru is None or len(cache._lru) <= 2   # never exceeds cap
    assert cache._lru is not None and len(cache._lru) <= 2


def test_manifest_is_cached_and_reused(tmp_path):
    _write_v2ep(tmp_path / f"{'d'*8}.v2ep.pt", n_raw=20, eid=5, seed=4)
    m1 = load_or_build_manifest(tmp_path, verbose=False)
    assert (tmp_path / "_v2manifest.pt").exists()
    m2 = load_or_build_manifest(tmp_path, verbose=False)   # from sidecar
    assert m1["files"] == m2["files"] and m1["T_out"] == m2["T_out"]
    assert m2["version"] == 1 and m2["n_stack"][0] == NSTACK


def test_v2_cache_flag_threads_and_default_untouched(monkeypatch):
    """The trainer's REAL parser accepts --v2-cache / --v2-lru and reaches
    train(args); the absent flag leaves v2_cache None (raw path untouched)."""
    import train_flagship4b as T4  # noqa: E402
    captured: dict = {}

    def _stub_train(args):
        captured["args"] = args
        return {"stub": True}

    monkeypatch.setattr(T4, "train", _stub_train)
    T4.main(["--out", "x", "--v2-cache", "/a", "/b", "--v2-lru", "8"])
    assert captured["args"].v2_cache == ["/a", "/b"] and captured["args"].v2_lru == 8
    captured.clear()
    T4.main(["--out", "x"])                         # no v2 flag
    assert captured["args"].v2_cache is None        # default: raw path preserved
    assert captured["args"].v2_lru == 64
