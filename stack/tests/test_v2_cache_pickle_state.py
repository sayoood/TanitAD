"""`V2CompressedCache` must carry EVERY constructor attribute across a pickle -- the boundary a
SPAWN-started DataLoader worker crosses (refcv7 restart package item 4, 2026-09-28).

THE DEFECT this pins (step-cost package sec. 7): `__getstate__` omitted `newest_frame_only`, so a
spawned worker's copy had NO such attribute and died on its first `decode_stacked_range` with
`AttributeError` -- whatever the flag's value. Linux FORK workers (the live refcv7 run on Thor)
never pickle the dataset and were unaffected; Windows / macOS / `multiprocessing_context="spawn"`
were not.

Expectations are LITERALS (True / False, a 3-channel newest frame), and the attribute census is
derived from a FRESHLY CONSTRUCTED instance -- never from `__getstate__`, the code under test.
RED on the old code: tests 1, 2, 3 and 5 fail (AttributeError / a missing key); 4 pins the
backward-compatible read of an OLD pickle.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest
import torch

tvio = pytest.importorskip("torchvision.io")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data.v2_dataset import V2CompressedCache, build_v2_providers  # noqa: E402

S, NSTACK = 16, 3


def _write_v2ep(path: Path, n_raw: int, eid: int, seed: int) -> None:
    """One synthetic clip in the exact `build_compressed` payload format (as test_v2_dataset.py);
    PNG so the pixel value that encodes the frame index survives the codec."""
    bufs, lens = [], []
    for i in range(n_raw):
        img = torch.full((3, S, S), fill_value=(i * 7) % 256, dtype=torch.uint8)
        b = tvio.encode_png(img)
        bufs.append(b)
        lens.append(int(b.numel()))
    g = torch.Generator().manual_seed(seed)
    poses = torch.randn(n_raw, 4, generator=g)
    poses[:, 3] = poses[:, 3].abs() * 5.0
    torch.save({"jpeg_buf": torch.cat(bufs), "jpeg_len": torch.tensor(lens, dtype=torch.int64),
                "actions": torch.randn(n_raw, 2, generator=g), "poses": poses,
                "n_stack": NSTACK, "image_size": S, "episode_id": eid,
                "clip_id": f"clip{eid:08d}", "quality": 90, "codec": "png"}, str(path))


def _roundtrip(c):
    return pickle.loads(pickle.dumps(c))


def test_1_newest_frame_only_TRUE_survives_the_pickle(tmp_path):
    c = V2CompressedCache(tmp_path, lru_size=3, newest_frame_only=True)
    assert _roundtrip(c).newest_frame_only is True


def test_2_newest_frame_only_FALSE_survives_the_pickle_as_an_ATTRIBUTE(tmp_path):
    c2 = _roundtrip(V2CompressedCache(tmp_path, lru_size=3))
    assert c2.newest_frame_only is False            # present AND False, not merely absent


def test_3_EVERY_constructor_attribute_crosses_the_boundary(tmp_path):
    """The census is a FRESH instance's own __dict__ -- independent of __getstate__. The LRU is
    the one attribute that is reset by design (a populated cache is never serialised)."""
    fresh = V2CompressedCache(tmp_path, lru_size=5, allow_lossy=True, newest_frame_only=True)
    fresh.files = ["a.v2ep.pt", "b.v2ep.pt"]
    back = _roundtrip(fresh)
    assert set(vars(back)) == set(vars(fresh))
    for k in vars(fresh):
        if k == "_lru":
            assert back._lru is None
            continue
        assert getattr(back, k) == getattr(fresh, k), k


def test_4_an_OLD_pickle_without_the_key_reads_as_False(tmp_path):
    """A state dict written by the pre-fix __getstate__ (five keys) still loads."""
    c = V2CompressedCache.__new__(V2CompressedCache)
    c.__setstate__({"cache_dir": str(tmp_path), "lru_size": 2, "files": [], "frame": None,
                    "allow_lossy": False})
    assert c.newest_frame_only is False


def test_5_a_SPAWN_worker_decodes_the_newest_frame(tmp_path):
    """End to end through a real spawn-started DataLoader worker (the Windows / macOS / 3.14
    default). The old code dies in the worker with AttributeError."""
    _write_v2ep(tmp_path / f"clip{0:08d}.v2ep.pt", n_raw=12, eid=0, seed=3)
    frames = build_v2_providers(tmp_path, lru_size=2, verbose=False,
                                newest_frame_only=True)[0].frames
    dl = torch.utils.data.DataLoader(frames, batch_size=None, num_workers=1,
                                     multiprocessing_context="spawn")
    got = [x for _i, x in zip(range(2), dl)]
    assert [tuple(x.shape) for x in got] == [(3, S, S), (3, S, S)]
    # row j pairs with RAW frame j + n_stack - 1 (the manifest's alignment), pixel = 7 * index
    assert [int(x[0, 0, 0]) for x in got] == [14, 21]
