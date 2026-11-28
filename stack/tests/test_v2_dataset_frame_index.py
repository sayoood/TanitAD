"""NT-DECIDE R1: the optional ``frame_index`` key of a ``*.v2ep.pt`` payload (``tanitad/data/v2_dataset.py``).

A NavSim token-episode (E1, ``nt_build/INTERFACE.md`` sec. 4) stores each UNIQUE frame once and maps raw row ``i`` to
stored image ``frame_index[i]``. Every existing cache carries no ``frame_index`` and must decode exactly as before.

Expectations are LITERALS: each synthetic image is a constant raster whose channel ``c`` holds ``base + c``, so the
D-015 stack (oldest frame first, current frame in the LAST 3 channels) has a value per channel that is written out
below, never computed by the code under test.

* an OLD payload (no ``frame_index``) decodes to the literal stack -- the pre-patch behaviour, pinned;
* a ``frame_index`` payload maps every raw row to its stored image -- function level, the eager reader
  (``decode_full_episode``) and the lazy cache (``V2CompressedCache.decode_stacked_range``);
* a raw row outside ``frame_index`` raises ``IndexError`` (fail loud: the row has no frame by contract);
* ``frame_index = arange(n)`` reproduces the old path bit for bit (the two code paths agree);
* DELIBERATE REGRESSION: a loader that IGNORES ``frame_index`` (raw row i -> image i, the unpatched behaviour) must
  FAIL the literal check -- and on a real E1 file the unpatched loader's index runs past the stored images.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
tvio = pytest.importorskip("torchvision.io")

STACK = Path(__file__).resolve().parents[1]
if str(STACK) not in sys.path:
    sys.path.insert(0, str(STACK))

from tanitad.data import v2_dataset as V  # noqa: E402

H = W = 6


def _img(base: int) -> torch.Tensor:
    t = torch.empty(3, H, W, dtype=torch.uint8)
    for c in range(3):
        t[c].fill_(base + c)
    return t


def _payload(bases: list[int], frame_index: list[int] | None = None) -> dict:
    encs = [tvio.encode_png(_img(b)) for b in bases]
    d = {"jpeg_buf": torch.cat(encs), "jpeg_len": torch.tensor([len(e) for e in encs], dtype=torch.int64),
         "n_stack": 3, "codec": "png", "episode_id": 7}
    n_raw = len(bases) if frame_index is None else len(frame_index)
    d["poses"] = torch.zeros(n_raw, 4)
    d["actions"] = torch.zeros(n_raw, 2)
    if frame_index is not None:
        d["frame_index"] = torch.tensor(frame_index, dtype=torch.int64)
    return d


def _channels(stacked: torch.Tensor) -> list[list[int]]:
    """[N, 9, H, W] constant rasters -> per row the 9 channel values (asserting each channel IS constant)."""
    out = []
    for r in stacked:
        vals = []
        for ch in r:
            assert int(ch.min()) == int(ch.max())
            vals.append(int(ch[0, 0]))
        out.append(vals)
    return out


# ---- the OLD format: literal pre-patch behaviour -------------------------------------------------------------
OLD_BASES = [0, 10, 20, 30, 40]
OLD_WANT = [[0, 1, 2, 10, 11, 12, 20, 21, 22],
            [10, 11, 12, 20, 21, 22, 30, 31, 32],
            [20, 21, 22, 30, 31, 32, 40, 41, 42]]

# ---- the frame_index format (an ST-like episode: rows 0..2 one image, then 1, 2, 2, 3) ------------------------
FI_BASES = [100, 110, 120, 130]
FI_INDEX = [0, 0, 0, 1, 2, 2, 3]
FI_WANT = [[100, 101, 102, 100, 101, 102, 100, 101, 102],     # raw 0,1,2 -> images 0,0,0
           [100, 101, 102, 100, 101, 102, 110, 111, 112],     # raw 1,2,3 -> 0,0,1
           [100, 101, 102, 110, 111, 112, 120, 121, 122],     # raw 2,3,4 -> 0,1,2
           [110, 111, 112, 120, 121, 122, 120, 121, 122],     # raw 3,4,5 -> 1,2,2
           [120, 121, 122, 120, 121, 122, 130, 131, 132]]     # raw 4,5,6 -> 2,2,3


def _dec(d: dict, a: int, b: int, **kw) -> torch.Tensor:
    return V._decode_stacked(d["jpeg_buf"], V._jpeg_offsets(d["jpeg_len"]), 3, a, b, "png",
                             frame_index=d.get("frame_index"), **kw)


def test_old_payload_decodes_to_the_literal_stack():
    assert _channels(_dec(_payload(OLD_BASES), 0, 3)) == OLD_WANT


def test_frame_index_maps_every_raw_row():
    assert _channels(_dec(_payload(FI_BASES, FI_INDEX), 0, 5)) == FI_WANT


def test_frame_index_newest_only():
    out = _dec(_payload(FI_BASES, FI_INDEX), 0, 5, newest_only=True)        # raw rows 2..6
    assert [int(r[0, 0, 0]) for r in out] == [100, 110, 120, 120, 130]


def test_frame_index_row_outside_raises():
    with pytest.raises(IndexError, match="outside frame_index"):
        _dec(_payload(FI_BASES, FI_INDEX), 0, 6)                            # needs raw row 7 of 7


def test_identity_frame_index_equals_old_path():
    old = _dec(_payload(OLD_BASES), 0, 3)
    idx = _dec(_payload(OLD_BASES, list(range(len(OLD_BASES)))), 0, 3)
    assert torch.equal(old, idx)


def test_eager_and_lazy_readers(tmp_path):
    p = tmp_path / "navtrain-test.v2ep.pt"
    torch.save(_payload(FI_BASES, FI_INDEX), p)
    ep = V.decode_full_episode(str(p))                                      # eager: frames cover frame_index only
    assert _channels(ep.frames) == FI_WANT
    c = V.V2CompressedCache(str(tmp_path), lru_size=2)
    c.files = [p.name]
    assert _channels(c.decode_stacked_range(0, 0, 5)) == FI_WANT
    assert _channels(c.decode_stacked_range(0, 2, 4)) == FI_WANT[2:4]
    q = tmp_path / "old" / "clip.v2ep.pt"
    q.parent.mkdir()
    torch.save(_payload(OLD_BASES), q)
    assert _channels(V.decode_full_episode(str(q)).frames) == OLD_WANT
    c2 = V.V2CompressedCache(str(q.parent), lru_size=2)
    c2.files = [q.name]
    assert _channels(c2.decode_stacked_range(0, 0, 3)) == OLD_WANT


def test_mutation_loader_ignoring_frame_index_goes_red(monkeypatch):
    """DELIBERATE REGRESSION: the unpatched behaviour (raw row i -> stored image i) must FAIL the literal check."""
    def ignoring(jpeg_buf, offs, n_stack, a, b, codec, sl, newest_only, frame_index):
        dec = tvio.decode_png
        n_img = int(offs.shape[0]) - 1

        def raw(i):
            if i >= n_img:
                raise IndexError(f"raw row {i} past the {n_img} stored images")
            return dec(jpeg_buf[int(offs[i]):int(offs[i + 1])], mode=tvio.ImageReadMode.RGB)
        return V.stack_frames(torch.stack([raw(i) for i in range(a, b + n_stack - 1)]), n_stack)

    monkeypatch.setattr(V, "_decode_indexed", ignoring)
    d = _payload(FI_BASES, FI_INDEX)
    try:
        got = _channels(_dec(d, 0, 2))           # rows the unpatched loader can still reach
    except IndexError:
        got = None
    assert got != FI_WANT[:2]                    # RED: the literal stack is not reproduced
    with pytest.raises(IndexError):              # and a full window runs past the 4 stored images
        _dec(d, 0, 5)
