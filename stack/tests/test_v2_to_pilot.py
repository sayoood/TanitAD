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
    assert json.load(open(out / "clips.json", encoding="utf-8")) == ["clipA"]


def test_zero_written_is_a_nonzero_exit(tmp_path):
    """THE regression: n=0 must not look like success."""
    corpus = tmp_path / "corp"
    corpus.mkdir()
    (corpus / "broken.v2ep.pt").write_bytes(b"not a torch file")
    out = tmp_path / "out"
    rc = main(["--corpus", str(corpus), "--out", str(out), "--n", "1"])
    assert rc == 3, "an empty bridge must exit non-zero"
    import json
    assert json.load(open(out / "clips.json", encoding="utf-8")) == []
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
    assert json.load(open(out / "clips.json", encoding="utf-8")) == ["good"]
    assert len(json.load(open(out / "failures.json", encoding="utf-8"))) == 1


# =========================================================================== #
# write_mp4 — the host-tolerant writer                                        #
# =========================================================================== #
def test_write_mp4_falls_back_to_the_bundled_ffmpeg_binary(tmp_path,
                                                           monkeypatch):
    """⛔ MEASURED 2026-08-13: pod5 — the pod HOLDING the 80 GB corpus — has
    imageio_ffmpeg (the ffmpeg BINARY) but not imageio, and no cv2/av. The
    bridge failed 2400/2400 clips on ModuleNotFoundError. Installing imageio was
    the WRONG fix: pod5 was training, and `uv pip install` has twice replaced
    torch with a wheel the driver cannot run. The binary is already there."""
    import builtins
    import sys as _sys
    import numpy as np
    from v2_to_pilot import write_mp4

    real_import = builtins.__import__

    def no_imageio(name, *args, **kw):
        if name == "imageio.v2" or name == "imageio":
            raise ImportError("simulated: imageio absent, as on pod5")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_imageio)
    monkeypatch.delitem(_sys.modules, "imageio", raising=False)
    monkeypatch.delitem(_sys.modules, "imageio.v2", raising=False)

    pytest.importorskip("imageio_ffmpeg")
    out = str(tmp_path / "f.mp4")
    frames = (np.random.default_rng(0).random((6, 32, 64, 3)) * 255
              ).astype(np.uint8)
    backend = write_mp4(out, frames, 10)
    assert backend == "imageio_ffmpeg"
    assert os.path.getsize(out) > 200


def test_write_mp4_rejects_a_wrong_shaped_stack(tmp_path):
    """A misshaped array must fail LOUDLY here, not produce a green-screen mp4
    that only reveals itself in a rendered overlay hours later."""
    import numpy as np
    from v2_to_pilot import write_mp4
    with pytest.raises(ValueError):
        write_mp4(str(tmp_path / "x.mp4"),
                  np.zeros((4, 8, 8), np.uint8), 10)      # missing channel dim
    with pytest.raises(ValueError):
        write_mp4(str(tmp_path / "y.mp4"),
                  np.zeros((4, 8, 8, 4), np.uint8), 10)   # RGBA, not RGB


def test_write_mp4_names_everything_it_tried_when_no_backend_exists(tmp_path,
                                                                    monkeypatch):
    """A failure must say WHICH backends were unavailable — the last error alone
    sent me chasing the wrong fix once already."""
    import builtins
    import numpy as np
    from v2_to_pilot import write_mp4
    real_import = builtins.__import__

    def none_available(name, *args, **kw):
        if name in ("imageio", "imageio.v2", "imageio_ffmpeg"):
            raise ImportError(f"simulated: {name} absent")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", none_available)
    with pytest.raises(RuntimeError) as ei:
        write_mp4(str(tmp_path / "z.mp4"),
                  np.zeros((2, 8, 8, 3), np.uint8), 10)
    msg = str(ei.value)
    assert "imageio" in msg and "imageio_ffmpeg" in msg


def test_module_structure_is_intact():
    """⛔ A COLUMN-0 `def` INSERTED INTO main()'s BODY SILENTLY ABSORBS THE REST
    OF main INTO THE HELPER. It is valid syntax, so `ast.parse` passes, the
    import works, and the only symptom is `main()` returning None instead of its
    exit code. MEASURED 2026-08-13 while adding write_mp4 — caught by
    test_zero_written_is_a_nonzero_exit going `assert None == 3`.

    This pins the shape directly so the next reader gets a named failure."""
    import ast
    import inspect
    import v2_to_pilot
    tree = ast.parse(inspect.getsource(v2_to_pilot))
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"stacked_to_rgb", "pick_clips", "write_mp4", "main"} <= set(fns)
    # main must still END in a return — an absorbed body loses it
    assert any(isinstance(n, ast.Return) for n in ast.walk(fns["main"]))
    # and write_mp4 must be small: if it swallowed main it would be huge
    assert len(fns["write_mp4"].body) < 12, "write_mp4 absorbed another function"
