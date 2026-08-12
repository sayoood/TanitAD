"""CPU tests for the v2 -> PH0 pilot bridge.

The bug these exist to prevent is specific and was measured on pod4: the bridge
looked for a ``frames`` key that the compressed v2 cache does not have, skipped
every clip, and printed ``BRIDGE_DONE n=0`` while exiting 0 — so the chain read
it as success and launched a 9B VLM against an empty directory.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v2_to_pilot import main, pick_clips, stacked_to_rgb  # noqa: E402


def _write_clip(path: str, n_raw: int = 6, n_stack: int = 3, h: int = 16,
                w: int = 24) -> None:
    """A real compressed v2 episode: JPEG blobs + lens + poses + actions."""
    torchvision_io = pytest.importorskip("torchvision.io")
    blobs, lens = [], []
    for i in range(n_raw):
        img = torch.full((3, h, w), fill_value=(i * 30) % 255, dtype=torch.uint8)
        b = torchvision_io.encode_jpeg(img)
        blobs.append(b)
        lens.append(int(b.numel()))
    k = n_stack - 1
    torch.save({"jpeg_buf": torch.cat(blobs),
                "jpeg_len": torch.tensor(lens, dtype=torch.int64),
                "n_stack": n_stack, "codec": "jpeg",
                "poses": torch.randn(n_raw, 4),
                "actions": torch.randn(n_raw, 2),
                "episode_id": 7, "image_h": h, "image_w": w}, path)
    return n_raw - k


# --------------------------------------------------------------------------- #
# stacked_to_rgb — the channel-slice contract                                  #
# --------------------------------------------------------------------------- #
def test_takes_last_three_channels_not_first():
    """Row j's pose is raw frame j+k, which lives in the LAST 3 channels.

    Taking [:3] would offset the video from its own ego trace by k frames — a
    silent registration error. This pins the direction."""
    t, hh, ww = 4, 8, 10
    fr = torch.zeros(t, 9, hh, ww, dtype=torch.uint8)
    fr[:, :3] = 11          # first raw frame of each stack
    fr[:, 3:6] = 22
    fr[:, -3:] = 33         # the pose-aligned one
    out = stacked_to_rgb(fr)
    assert out.shape == (t, hh, ww, 3)
    assert (out == 33).all(), "must take the LAST raw frame of the stack"


def test_grayscale_is_broadcast_to_three():
    out = stacked_to_rgb(torch.full((2, 1, 4, 5), 7, dtype=torch.uint8))
    assert out.shape == (2, 4, 5, 3) and (out == 7).all()


def test_float_frames_are_scaled_not_truncated():
    out = stacked_to_rgb(torch.ones(1, 3, 2, 2) * 0.5)
    assert out.dtype == np.uint8 and int(out.max()) == 127


def test_rejects_channel_count_that_is_not_a_multiple_of_three():
    with pytest.raises(ValueError, match="neither 1 nor a multiple of 3"):
        stacked_to_rgb(torch.zeros(1, 5, 4, 4, dtype=torch.uint8))


def test_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"expected \[T,C,H,W\]"):
        stacked_to_rgb(torch.zeros(3, 4, 4, dtype=torch.uint8))


# --------------------------------------------------------------------------- #
# selection                                                                    #
# --------------------------------------------------------------------------- #
def test_no_overlap_refuses_rather_than_falling_back(tmp_path, monkeypatch):
    """A corpus-only fallback would produce a pilot whose Alpamayo column is
    empty and whose cross-engine comparison is meaningless."""
    (tmp_path / "aaa.v2ep.pt").write_bytes(b"x")
    rec = tmp_path / "r.parquet"
    pd = pytest.importorskip("pandas")
    pd.DataFrame({"clip_id": ["zzz"]}).to_parquet(rec)
    with pytest.raises(SystemExit, match="no clip overlap"):
        pick_clips(str(tmp_path), str(rec), 4, 0)


def test_empty_corpus_refuses(tmp_path):
    with pytest.raises(SystemExit, match="no \\*.v2ep.pt"):
        pick_clips(str(tmp_path), None, 4, 0)


def test_selection_is_seed_deterministic(tmp_path):
    for i in range(6):
        (tmp_path / f"c{i}.v2ep.pt").write_bytes(b"x")
    a, _ = pick_clips(str(tmp_path), None, 3, 0)
    b, _ = pick_clips(str(tmp_path), None, 3, 0)
    c, _ = pick_clips(str(tmp_path), None, 3, 1)
    assert a == b and len(a) == 3
    assert a != c or len(set(a) & set(c)) < 3


# --------------------------------------------------------------------------- #
# end-to-end over a REAL compressed episode                                    #
# --------------------------------------------------------------------------- #
def test_end_to_end_writes_video_and_ego(tmp_path):
    pytest.importorskip("torchvision.io")
    pytest.importorskip("imageio")
    corpus = tmp_path / "corp"
    corpus.mkdir()
    n_rows = _write_clip(str(corpus / "clipA.v2ep.pt"))
    out = tmp_path / "out"
    rc = main(["--corpus", str(corpus), "--out", str(out), "--n", "1"])
    assert rc == 0
    assert (out / "videos" / "clipA.mp4").exists()
    ego = np.load(out / "ego" / "clipA.npz")
    assert ego["poses"].shape == (n_rows, 4)
    assert ego["actions"].shape[0] == n_rows
    import json
    assert json.load(open(out / "clips.json")) == ["clipA"]


def test_zero_written_is_a_nonzero_exit(tmp_path):
    """THE regression: n=0 must not look like success."""
    corpus = tmp_path / "corp"
    corpus.mkdir()
    (corpus / "broken.v2ep.pt").write_bytes(b"not a torch file")
    out = tmp_path / "out"
    rc = main(["--corpus", str(corpus), "--out", str(out), "--n", "1"])
    assert rc == 3, "an empty bridge must exit non-zero"
    import json
    assert json.load(open(out / "clips.json")) == []
    assert (out / "failures.json").exists(), "failures must be recorded"


def test_one_bad_clip_does_not_kill_the_batch(tmp_path):
    pytest.importorskip("torchvision.io")
    pytest.importorskip("imageio")
    corpus = tmp_path / "corp"
    corpus.mkdir()
    _write_clip(str(corpus / "good.v2ep.pt"))
    (corpus / "bad.v2ep.pt").write_bytes(b"garbage")
    out = tmp_path / "out"
    rc = main(["--corpus", str(corpus), "--out", str(out), "--n", "2"])
    assert rc == 0
    import json
    assert json.load(open(out / "clips.json")) == ["good"]
    assert len(json.load(open(out / "failures.json"))) == 1
