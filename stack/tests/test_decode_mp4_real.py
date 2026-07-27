"""`_decode_mp4` must actually decode an mp4 — end to end, on a real file.

**Why this test exists.** Commit ``fdc5b4f`` introduced ``fr = as_frame(...)``
to bind the ``CanonicalFrame``, and the decode loop below it was
``for fr in c.decode(stream)`` — which **rebound ``fr`` to a PyAV VideoFrame**.
Every downstream use (``_remap_batch``, ``fr.to_dict()``, ``fr.tag()``,
``ftheta_horizon_row``) then received a video frame where geometry was
expected, so ``_decode_mp4`` raised ``AttributeError`` **on every path,
including the deployed one**. HEAD was a dead corpus builder.

It survived because **no test in the suite decoded an mp4**. The targeted
geometry suite was 195-green *with the bug*, and the full suite was 1253-green
*with the bug*: every test either stopped at the pure-tensor resampler
(``ftheta_crop_resize`` / ``cylindrical_rectify``) or mocked the decode away.
A unit test of the parts cannot catch a defect in how the parts are wired.

So this test writes a genuine tiny mp4 with PyAV, decodes it through the real
``_decode_mp4``, and asserts the output shape and dtype — the cheapest thing
that exercises the wiring. It is driven for **both** projection modes, because
the shadowed name reached both.

The clip-keyed intrinsics/extrinsics lookups are monkeypatched: they read a
gated corpus that is not present in CI, and they are not what is under test
here — the wiring is.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
av = pytest.importorskip("av")

from tanitad.data import physicalai as pai  # noqa: E402
from tanitad.data.calib import FThetaIntrinsics  # noqa: E402

_W, _H, _N = 320, 180, 6


def _write_tiny_mp4(path) -> None:
    """A real, decodable mp4 — not a fixture stub."""
    with av.open(str(path), mode="w") as c:
        st = c.add_stream("libx264", rate=10)
        st.width, st.height, st.pix_fmt = _W, _H, "yuv420p"
        for i in range(_N):
            arr = np.full((_H, _W, 3), (i * 37) % 256, dtype=np.uint8)
            c.mux(st.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
        c.mux(st.encode())


@pytest.fixture()
def clip(tmp_path, monkeypatch):
    mp4 = tmp_path / "0123456789abcdef.mp4"
    _write_tiny_mp4(mp4)

    # A benign f-theta model centred on the frame; poly[1] is the paraxial focal.
    intr = FThetaIntrinsics(poly=(0.0, 140.0, 0.0, 0.0, 0.0),
                            cx=_W / 2.0, cy=_H / 2.0, width=_W, height=_H,
                            per_clip=True)
    monkeypatch.setattr(pai, "intrinsics_for_clip", lambda *a, **k: intr)
    monkeypatch.setattr(pai, "extrinsics_for_clip", lambda *a, **k: None)
    monkeypatch.setattr(pai, "_physicalai_root_of", lambda *a, **k: tmp_path)
    return mp4


@pytest.mark.parametrize("projection_mode", sorted(pai.PROJECTION_MODES))
def test_decode_mp4_returns_frames_not_an_attributeerror(clip, projection_mode):
    """THE REGRESSION. Before the fix this raised AttributeError on every mode."""
    out = pai._decode_mp4(clip, 64, projection_mode=projection_mode)
    assert out.dtype == torch.uint8
    assert out.ndim == 4 and out.shape[1] == 3, out.shape
    assert out.shape[0] == _N, (
        f"every decoded frame must survive: got {out.shape[0]} of {_N}")


def test_decode_mp4_honours_a_non_square_canonical_frame(clip):
    """The wide-FOV path — the one the shadowed name was introduced for.

    ``size`` stays at its default 256: ``as_frame`` deliberately REFUSES a frame
    plus non-default legacy scalars, because two sources of truth for the same
    geometry is the bug ``CanonicalFrame`` exists to remove. Passing both here
    was an error in this test, and the refusal caught it — which is the guard
    working, so it is left in place rather than relaxed.
    """
    from tanitad.data.calib import CanonicalFrame
    fr = CanonicalFrame(height=64, width=160, f_ref=pai.F_REF,
                        projection="cylindrical")
    assert fr.height != fr.width, "a square frame would make this test vacuous"
    out = pai._decode_mp4(clip, 256, frame=fr, projection_mode="cylindrical")
    assert out.shape[-2:] == (fr.height, fr.width), (
        "the output must match the CanonicalFrame, which is exactly what the "
        "shadowed `fr` destroyed")


def test_the_decode_loop_does_not_rebind_the_geometry_name():
    """Pin the defect itself, so a future edit cannot silently reintroduce it.

    Behavioural tests above are the real guard; this one names the mistake so
    the next person reaching for `fr` as a loop variable gets a red test with
    an explanation rather than a puzzling AttributeError in a build worker.
    """
    import inspect
    src = inspect.getsource(pai._decode_mp4)
    assert "for fr in" not in src, (
        "`fr` is the CanonicalFrame bound by as_frame(); rebinding it in the "
        "decode loop broke _decode_mp4 on every path (commit fdc5b4f)")
