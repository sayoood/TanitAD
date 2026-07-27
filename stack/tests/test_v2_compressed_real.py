"""`v2_compressed` must actually decode an mp4 — the SECOND instance of the bug.

**Why this test exists.** `fdc5b4f` added ``fr = as_frame(...)`` to two decode
functions on the same day and left both decode loops as ``for fr in ...``:

* ``physicalai._decode_mp4``                 — fixed by the `4cb37f4` hotfix
* ``v2_compressed._decode_cropped_selected`` — **survived that hotfix**

The second one is worse-hidden, because the rebound ``fr`` is captured by a
CLOSURE (``flush()``) rather than used inline, so reading the diff does not make
it obvious. It broke **every** v2 build path including the deployed 256 px JPEG
one, with::

    AttributeError: 'av.video.frame.VideoFrame' object has no attribute
                    'half_angle_x_rad'

MEASURED on pod2, 2026-07-27, by running the real builder against a real clip —
`pytest -q` was green with the bug present, exactly as it was for the first
instance. That is the point: **a unit test of the parts cannot catch a defect in
how the parts are wired**, and the v2 path is the only storage-viable route to a
wide-FOV corpus (112.9 GB PNG-lossless vs 697 GB raw epcache for the train split
alone), so a dead v2 builder blocks flagship v5.

The runtime tests write a genuine tiny mp4 with PyAV and drive the real
``_decode_cropped_selected`` for BOTH projection modes and BOTH frame shapes
(canonical square + non-square wide), since the shadowed name reached all of
them. Clip-keyed intrinsics are monkeypatched: they read a gated corpus absent
from CI, and they are not what is under test — the wiring is.

``test_frame_name_not_rebound_by_decode_loop`` deliberately imports NOTHING, so
it guards the defect class even on a box without torch/torchvision/PyAV (the dev
box has no torchvision — where the runtime tests skip, the guard still runs).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_V2_SRC = _SCRIPTS / "v2_compressed.py"

_W, _H, _N = 320, 180, 8


# --------------------------------------------------------------------------- #
# dependency-free guard on the defect CLASS                                    #
# --------------------------------------------------------------------------- #
def _loop_targets_named(path: Path, name: str) -> list[int]:
    """Line numbers of ``for <name> in ...`` loops — parsed, not grepped.

    A substring search matches this module's own prose (and the fix's
    explanatory comment), so the check is done on the AST: only a real ``For``
    node whose target actually BINDS ``name`` counts."""
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            tgt = node.target
            names = ([t.id for t in tgt.elts if isinstance(t, ast.Name)]
                     if isinstance(tgt, ast.Tuple)
                     else [tgt.id] if isinstance(tgt, ast.Name) else [])
            if name in names:
                hits.append(node.lineno)
    return hits


def test_frame_name_not_rebound_by_decode_loop():
    """Both known instances were ``for fr in <decoder>`` shadowing the
    CanonicalFrame bound as ``fr``. Guarding the source is crude, but the
    runtime tests only cover the paths they exercise and they SKIP wherever
    torchvision is absent — this one cannot be defeated by a new call site and
    never skips."""
    hits = _loop_targets_named(_V2_SRC, "fr")
    assert not hits, (
        f"{_V2_SRC.name} line(s) {hits}: a loop rebinds `fr`, the name reserved "
        f"for the CanonicalFrame — this is the fdc5b4f defect class (see this "
        f"module's docstring)")


def test_physicalai_decode_loop_also_clean():
    """The first instance, pinned here too so the pair cannot regress apart."""
    p = Path(__file__).resolve().parents[1] / "tanitad" / "data" / "physicalai.py"
    hits = _loop_targets_named(p, "fr")
    assert not hits, (
        f"physicalai.py line(s) {hits}: a loop rebinds `fr` again — the 4cb37f4 "
        f"hotfix has regressed")


# --------------------------------------------------------------------------- #
# runtime tests — real mp4 through the real builder                            #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def v2mod():
    pytest.importorskip("torch")
    pytest.importorskip("av")
    pytest.importorskip("torchvision")     # v2_compressed imports torchvision.io
    pytest.importorskip("pandas")
    sys.path.insert(0, str(_SCRIPTS))
    import v2_compressed
    return v2_compressed


def _write_tiny_mp4(path: Path) -> None:
    """A real, decodable mp4 — not a fixture stub."""
    import av
    with av.open(str(path), mode="w") as c:
        st = c.add_stream("libx264", rate=10)
        st.width, st.height, st.pix_fmt = _W, _H, "yuv420p"
        for i in range(_N):
            arr = np.full((_H, _W, 3), (i * 17) % 256, dtype=np.uint8)
            c.mux(st.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
        c.mux(st.encode(None))


@pytest.fixture()
def tiny_clip(tmp_path, monkeypatch, v2mod):
    from tanitad.data.calib import FThetaIntrinsics
    mp4 = tmp_path / "0123abcd-0000-0000-0000-000000000000.mp4"
    _write_tiny_mp4(mp4)
    # A plausible f-theta model for a 320x180 "sensor", principal point centred.
    intr = FThetaIntrinsics(poly=(0.0, 90.0, 0.0, 0.0, 0.0), cx=_W / 2.0,
                            cy=_H / 2.0, width=_W, height=_H, per_clip=True)
    monkeypatch.setattr(v2mod, "intrinsics_for_clip", lambda cid, root: intr)
    monkeypatch.setattr(v2mod, "_physicalai_root_of", lambda p: str(tmp_path))
    return mp4


@pytest.mark.parametrize("projection_mode", ["ftheta_crop", "cylindrical"])
def test_decode_cropped_selected_canonical_square(tiny_clip, v2mod,
                                                  projection_mode):
    """The DEPLOYED path. This is what was broken — not just the wide one."""
    import torch
    idx = torch.tensor([0, 2, 4])
    out = v2mod._decode_cropped_selected(tiny_clip, 64, idx,
                                         projection_mode=projection_mode)
    assert out.shape == (3, 3, 64, 64), out.shape
    assert out.dtype is torch.uint8


@pytest.mark.parametrize("projection_mode", ["ftheta_crop", "cylindrical"])
def test_decode_cropped_selected_wide_non_square(tiny_clip, v2mod,
                                                 projection_mode):
    """The wide-FOV frame v5 needs: non-square, explicit CanonicalFrame.

    ``size`` MUST stay at its 256 sentinel when a frame is passed: ``as_frame``
    refuses a frame combined with non-default legacy scalars, on purpose (two
    sources of truth for the same geometry is the bug that object removes). The
    frame carries the real 32x80 shape; ``size`` is inert here. This is exactly
    how ``build_compressed`` calls it in production."""
    import torch
    from tanitad.data.calib import CanonicalFrame
    frame = CanonicalFrame.from_hfov(
        60.0, 32, 80,
        "cylindrical" if projection_mode == "cylindrical" else "pinhole")
    idx = torch.tensor([1, 3])
    out = v2mod._decode_cropped_selected(tiny_clip, 256, idx, frame=frame,
                                         projection_mode=projection_mode)
    assert out.shape == (2, 3, 32, 80), out.shape
    assert out.dtype is torch.uint8


def test_batching_boundary_is_crossed(tiny_clip, v2mod, monkeypatch):
    """``flush()`` is called from INSIDE the decode loop once the batch fills —
    the exact call site where the rebound loop variable was read. Force >1 flush
    so the mid-loop path is exercised, not only the final flush."""
    import torch
    monkeypatch.setenv("PAI_DECODE_BATCH", "2")
    idx = torch.tensor([0, 1, 2, 3, 4, 5])
    out = v2mod._decode_cropped_selected(tiny_clip, 32, idx)
    assert out.shape == (6, 3, 32, 32), out.shape
