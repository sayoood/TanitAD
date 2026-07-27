"""⭐ THE TRAINER ACTUALLY RECEIVES THE FRAME — and it FAILS when it does not.

THE DEFECT CLASS THIS FILE EXISTS FOR, stated plainly: a verified fix that
nothing calls. `load_compressed(path, frame=…)` was measured bit-exact on 1,206
real frames and **no trainer passed it** — and the trainer that v5 launches from
does not even go through `load_compressed`: it reads
`tanitad.data.v2_dataset.build_v2_providers`, a separate mirrored decode path.
So the fix was correct, tested, shipped, and structurally unreachable from the
run it was built for. That is the third instance in one week (`assert_ego_is_fed`
had six tests and had never been invoked; `launch_val_all.sh` was staged and
nothing scheduled it).

The tests below therefore assert the CALL, not the capability:

* :func:`test_the_seam_feeds_the_configured_subframe_and_it_is_the_exact_slice`
  — the trainer's own seam hands back 176x624 providers whose pixels are
  bit-identical to the parent's `[40:216, 8:632]`.
* ⛔ :func:`test_it_FAILS_LOUD_when_the_loader_is_never_told_the_frame` — the
  RED half. The loader is patched to DROP the frame argument, which is exactly
  the pre-fix state of the world, and the launch must REFUSE. A guard that
  cannot fail is not a guard (class C13).
* ⛔ :func:`test_preflight_REFUSES_a_wide_v2_run_that_forgot_the_flag` — and the
  gap cannot be recreated by simply not typing the flag either.

⚠️ ZERO SKIPS, on purpose and by the same argument. `tanitad.data.v2_dataset`
imports `torchvision`, which the dev box does not have. Rather than
`importorskip` — a guard that skips on the host where it is most likely to be
run — this file installs a LOSSLESS RAW CODEC in place of `torchvision.io`
**only when the real one is absent**, and removes it from `sys.modules`
immediately after the import so no other module's `importorskip` changes
behaviour. On a pod the real PNG codec is used and the identical assertions run.
The codec is not what is under test: the slice operates on decoded tensors.

The frames are 256x640 and the sub-frames 176x624 / 128x576 — the REAL rasters,
not scaled-down stand-ins, because the row/column margins (40/8 and 64/32) and
the even-margin rule are the load-bearing arithmetic.

🔒 Every clip id here is synthetic.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.config import flagship4b_config                      # noqa: E402
from tanitad.data import parity                                   # noqa: E402
from tanitad.data.calib import (CanonicalFrame,                    # noqa: E402
                                PHYSICALAI_RIG_CLEAN_128x576,
                                PHYSICALAI_RIG_CLEAN_176x624,
                                PHYSICALAI_WIDE120_256x640)

# --------------------------------------------------------------------------- #
# codec shim — installed ONLY if torchvision is genuinely absent               #
# --------------------------------------------------------------------------- #
HAVE_TORCHVISION = importlib.util.find_spec("torchvision") is not None


def _raw_encode(t: torch.Tensor) -> torch.Tensor:
    """[C,H,W] uint8 -> a self-describing 1-D uint8 buffer. LOSSLESS."""
    c, h, w = t.shape
    head = torch.tensor(list(struct.pack("<3I", c, h, w)), dtype=torch.uint8)
    return torch.cat([head, t.reshape(-1).contiguous()])


def _raw_decode(buf: torch.Tensor, mode=None) -> torch.Tensor:
    c, h, w = struct.unpack("<3I", bytes(buf[:12].tolist()))
    return buf[12:].reshape(c, h, w).clone()


def _install_shim() -> None:
    tv = types.ModuleType("torchvision")
    io = types.ModuleType("torchvision.io")
    io.decode_png = _raw_decode
    io.decode_jpeg = _raw_decode          # the LABEL is what the guard keys on
    io.encode_png = _raw_encode
    io.encode_jpeg = lambda t, quality=90: _raw_encode(t)
    io.ImageReadMode = types.SimpleNamespace(RGB="RGB")
    tv.io = io
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.io"] = io


if not HAVE_TORCHVISION:
    _install_shim()
import tanitad.data.v2_dataset as V2                              # noqa: E402
if not HAVE_TORCHVISION:                  # leave sys.modules exactly as found
    sys.modules.pop("torchvision.io", None)
    sys.modules.pop("torchvision", None)

import train_flagship_v4 as T                                     # noqa: E402

PARENT = PHYSICALAI_WIDE120_256x640                # 256x640, 120.000 deg
CLEAN = PHYSICALAI_RIG_CLEAN_176x624               # 176x624, 117.000 deg
TILED = PHYSICALAI_RIG_CLEAN_128x576               # 128x576, 108.000 deg
N_RAW, N_STACK = 6, 3
T_OUT = N_RAW - (N_STACK - 1)


# --------------------------------------------------------------------------- #
# a REAL v2 cache: payloads in the exact build_compressed on-disk format        #
# --------------------------------------------------------------------------- #
def _pattern(n: int, h: int, w: int, seed: int) -> torch.Tensor:
    """Deterministic, POSITION-DEPENDENT content: value = f(frame, ch, row, col).

    Random noise would also catch a wrong slice, but this catches it with a
    diagnosable value — a slice off by one row changes every byte by 7."""
    f = torch.arange(n).view(n, 1, 1, 1) * 29
    c = torch.arange(3).view(1, 3, 1, 1) * 3
    r = torch.arange(h).view(1, 1, h, 1) * 7
    x = torch.arange(w).view(1, 1, 1, w) * 13
    return ((f + c + r + x + seed) % 256).to(torch.uint8)


def _write_clip(path: Path, *, frame: CanonicalFrame, seed: int,
                codec: str = "png") -> torch.Tensor:
    enc = V2.tvio.encode_png if codec == "png" else (
        lambda t: V2.tvio.encode_jpeg(t, quality=90))
    vid = _pattern(N_RAW, frame.height, frame.width, seed)
    blobs = [enc(vid[i].contiguous()) for i in range(N_RAW)]
    lens = torch.tensor([int(b.numel()) for b in blobs], dtype=torch.int64)
    g = torch.Generator().manual_seed(seed)
    poses = torch.randn(N_RAW, 4, generator=g)
    poses[:, 3] = poses[:, 3].abs() * 5.0
    torch.save({"jpeg_buf": torch.cat(blobs), "jpeg_len": lens,
                "actions": torch.randn(N_RAW, 2, generator=g), "poses": poses,
                "n_stack": N_STACK, "image_size": frame.height,
                "image_h": frame.height, "image_w": frame.width,
                "episode_id": seed, "clip_id": f"synth{seed:08d}",
                "quality": 90, "frame": frame.to_dict(),
                "projection_mode": frame.projection, "codec": codec},
               str(path))
    return vid


def _cache(root: Path, name: str, seeds, *, frame: CanonicalFrame = PARENT,
           codec: str = "png") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        _write_clip(d / f"synth{s:08d}.v2ep.pt", frame=frame, seed=s,
                    codec=codec)
    return d


BASE_ARGV = ["--frame-h", "256", "--frame-w", "640", "--frame-hfov", "120",
             "--projection", "cylindrical", "--from-scratch",
             "--require-parity"]


def _args(tmp_path, subframe: str | None, *, extra=(), codec: str = "png"):
    tr = _cache(tmp_path, "train", [1, 2], codec=codec)
    va = _cache(tmp_path, "val", [3], codec=codec)
    argv = ["--v2-train-cache", str(tr), "--v2-val-cache", str(va), *BASE_ARGV]
    if subframe is not None:
        argv += ["--v2-subframe", subframe]
    argv += list(extra)
    return T.build_parser().parse_args(argv), tr, va


def _prov() -> dict:
    """The provenance skeleton `build_v2_data` writes its bindings into.

    Empty parity records on purpose: the DECLARATION binding is exercised
    separately; here the SHAPE binding is what must fire."""
    return {"train_parity": {}, "val_parity": {}}


# =========================================================================== #
# 1. ⭐ the seam delivers the sub-frame, and it is the EXACT slice              #
# =========================================================================== #
def test_the_seam_feeds_the_configured_subframe_and_it_is_the_exact_slice(tmp_path):
    a, tr, va = _args(tmp_path, "176x624")
    cfg = flagship4b_config()
    cache_frame, frame = T.resolve_v2_frames(a, cfg)

    assert cache_frame.hw == (256, 640) and frame.hw == (176, 624)
    assert frame == CLEAN, "the sub-frame must be the pinned rig-clean frame"
    # derived FROM the parent, so f_ref and projection are copied, not re-solved
    assert float(frame.f_ref) == float(cache_frame.f_ref)
    assert frame.projection == cache_frame.projection == "cylindrical"

    train_eps, val_eps = T.build_v2_data(
        a, (prov := _prov()), cache_frame=cache_frame, train_frame=frame,
        verbose=False)

    assert len(train_eps) == 2 and len(val_eps) == 1
    for ep in (*train_eps, *val_eps):
        assert tuple(ep.frames.shape) == (T_OUT, 3 * N_STACK, 176, 624)

    # ⭐ THE LOAD-BEARING ASSERTION: what the trainer gets is BIT-IDENTICAL to
    # the parent's [40:216, 8:632]. Built independently of the code under test —
    # the reference is the SAME loader with no frame, sliced by hand.
    (ref,) = V2.build_v2_providers(va, verbose=False)
    assert tuple(ref.frames.shape) == (T_OUT, 9, 256, 640)
    got = val_eps[0].frames[0:T_OUT]
    assert torch.equal(got, ref.frames[0:T_OUT][:, :, 40:216, 8:632])
    assert int((got.float() - ref.frames[0:T_OUT][:, :, 40:216, 8:632].float())
               .abs().max()) == 0

    # and the binding recorded WHICH slice, from WHICH parent
    sf = prov["geometry_binding"]["sliced_from"]
    assert sf["rows"] == [40, 216] and sf["cols"] == [8, 632]
    assert sf["parent_tag"] == "256x640f305.5775cyl"
    assert sf["sub_tag"] == "176x624f305.5775cyl"
    assert prov["geometry_binding"]["cache_frame_shapes"] == [[176, 624]]


def test_a_partial_window_read_is_the_same_slice_as_a_full_one(tmp_path):
    """The window path fetches CONTIGUOUS SUB-RANGES, not whole clips. The slice
    must survive the partial decode — that is where a per-clip cached slice
    could go stale."""
    a, tr, va = _args(tmp_path, "176x624")
    cache_frame, frame = T.resolve_v2_frames(a, flagship4b_config())
    train_eps, _ = T.build_v2_data(a, _prov(), cache_frame=cache_frame,
                                   train_frame=frame, verbose=False)
    ep = train_eps[0]
    full = ep.frames[0:T_OUT]
    for lo, hi in [(0, 1), (1, 3), (T_OUT - 1, T_OUT), (0, T_OUT)]:
        assert torch.equal(ep.frames[lo:hi], full[lo:hi]), (lo, hi)
    assert torch.equal(ep.frames[2], full[2])          # int index too
    assert tuple(ep.frames[T_OUT:T_OUT].shape) == (0, 9, 176, 624)  # empty


# =========================================================================== #
# 2. ⛔ THE RED HALF — it must FAIL when the loader is not told                 #
# =========================================================================== #
def test_it_FAILS_LOUD_when_the_loader_is_never_told_the_frame(tmp_path,
                                                               monkeypatch):
    """⛔ THE WHOLE POINT. Reproduce the pre-fix world exactly: the sub-frame is
    configured, the encoder is sized for it, and the loader is never given it.

    Before this test the run would have started and trained 30k steps on the
    rig-ASYMMETRIC frames while `config.json` claimed the clean ones."""
    a, tr, va = _args(tmp_path, "176x624")
    cache_frame, frame = T.resolve_v2_frames(a, flagship4b_config())

    real = V2.build_v2_providers
    seen: list = []

    def _drops_the_frame(dirs, **kw):
        seen.append(kw.pop("frame", "ABSENT"))      # <-- the defect, verbatim
        return real(dirs, **kw)

    monkeypatch.setattr(V2, "build_v2_providers", _drops_the_frame)

    with pytest.raises(parity.ParityViolation) as e:
        T.build_v2_data(a, _prov(), cache_frame=cache_frame, train_frame=frame,
                        verbose=False)

    msg = str(e.value)
    assert "SUB-FRAME WAS DECLARED BUT NEVER APPLIED" in msg
    assert "176x624" in msg and "256x640" in msg
    assert "build_v2_providers" in msg          # names the call that must change
    assert seen and seen[0] == CLEAN            # the frame WAS offered, and dropped


def test_the_same_guard_still_fires_for_an_ordinarily_wrong_cache(tmp_path):
    """⛔ C13 the other way: the shape binding must keep catching the plain
    mismatch it was built for, not only the new sub-frame case."""
    tr = _cache(tmp_path, "sq_train", [1], frame=CanonicalFrame())
    va = _cache(tmp_path, "sq_val", [2], frame=CanonicalFrame())
    a = T.build_parser().parse_args(
        ["--v2-train-cache", str(tr), "--v2-val-cache", str(va), *BASE_ARGV,
         "--v2-subframe", "none"])
    cache_frame, frame = T.resolve_v2_frames(a, flagship4b_config())
    with pytest.raises(parity.ParityViolation) as e:
        T.build_v2_data(a, _prov(), cache_frame=cache_frame, train_frame=frame,
                        verbose=False)
    msg = str(e.value)
    assert "MISMATCH" in msg and "SUB-FRAME WAS DECLARED" not in msg


# =========================================================================== #
# 3. ⛔ and the flag cannot be lost by omission                                 #
# =========================================================================== #
def test_preflight_REFUSES_a_wide_v2_run_that_forgot_the_flag(tmp_path):
    a, _, _ = _args(tmp_path, None)
    problems = T.preflight_asserts(a)
    hit = [p for p in problems if p.startswith("[RIG-CLEAN]")]
    assert hit, problems
    assert "8.897" in hit[0] and "176x624" in hit[0] and "none" in hit[0]


@pytest.mark.parametrize("spec", ["176x624", "128x576", "none"])
def test_preflight_is_SATISFIED_by_an_explicit_choice(tmp_path, spec):
    a, _, _ = _args(tmp_path, spec)
    assert not [p for p in T.preflight_asserts(a) if p.startswith("[RIG-CLEAN]")]


def test_preflight_does_NOT_fire_on_the_DEPLOYED_square_cache(tmp_path):
    """pod1's running `--v2-cache` arm is the deployed 256x256 square frame. Its
    pad defect is upstream of the cache and no slice fixes it, so demanding a
    sub-frame decision there would be noise — and would change a running arm."""
    tr = _cache(tmp_path, "d_train", [1], frame=CanonicalFrame())
    va = _cache(tmp_path, "d_val", [2], frame=CanonicalFrame())
    a = T.build_parser().parse_args(
        ["--v2-train-cache", str(tr), "--v2-val-cache", str(va),
         "--from-scratch", "--require-parity"])
    assert not [p for p in T.preflight_asserts(a) if p.startswith("[RIG-CLEAN]")]


@pytest.mark.parametrize("spec", ["176x624", "none"])
def test_the_staged_command_carries_the_subframe_in_BOTH_directions(tmp_path,
                                                                    spec):
    """A launch reconstructed from `--print-launch` must not be able to lose the
    decision — in either direction. 'none' is emitted too, so a rig-asymmetric
    arm stays a deliberate, visible act."""
    a, _, _ = _args(tmp_path, spec)
    assert f"--v2-subframe {spec}" in T._staged_command(a)


def test_the_staged_command_is_UNCHANGED_when_the_flag_is_absent(tmp_path):
    a, _, _ = _args(tmp_path, None)
    assert "--v2-subframe" not in T._staged_command(a)


# =========================================================================== #
# 4. the encoder moves with the frame (a half-applied change is the old bug)    #
# =========================================================================== #
def test_the_ENCODER_is_sized_for_the_subframe_not_the_cache(tmp_path):
    from tanitad.geometry import assert_geometry_consistent, tiling_report
    a, _, _ = _args(tmp_path, "176x624")
    cfg = flagship4b_config()
    cache_frame, frame = T.resolve_v2_frames(a, cfg)
    assert cfg.encoder.image_hw() == (176, 624)
    assert assert_geometry_consistent(cfg).hw == (176, 624)
    gh, gw = cfg.encoder.token_grid()
    assert (gh, gw) == (11, 39) and gh * gw == 429      # from 640: -33.0 %
    rep = tiling_report(cfg)
    # a KNOWN, measured consequence, pinned so it cannot be discovered as a
    # surprise mid-run: 176x624 does not tile the 4x4 readout evenly.
    assert rep["tiles_exactly"] is False
    assert rep["state_dim"] == 2048                     # unchanged


def test_the_TILED_alternative_is_the_same_one_flag_change(tmp_path):
    """The PI has not chosen between 176x624 and 128x576. Nothing may hard-code
    either — this is the same seam, one flag value apart."""
    from tanitad.geometry import tiling_report
    a, tr, va = _args(tmp_path, "128x576")
    cfg = flagship4b_config()
    cache_frame, frame = T.resolve_v2_frames(a, cfg)
    assert frame == TILED and cfg.encoder.image_hw() == (128, 576)
    gh, gw = cfg.encoder.token_grid()
    assert gh * gw == 288 and tiling_report(cfg)["tiles_exactly"] is True

    _, val_eps = T.build_v2_data(a, (prov := _prov()), cache_frame=cache_frame,
                                 train_frame=frame, verbose=False)
    assert tuple(val_eps[0].frames.shape) == (T_OUT, 9, 128, 576)
    (ref,) = V2.build_v2_providers(va, verbose=False)
    assert torch.equal(val_eps[0].frames[0:T_OUT],
                       ref.frames[0:T_OUT][:, :, 64:192, 32:608])
    assert prov["geometry_binding"]["sliced_from"]["rows"] == [64, 192]
    assert prov["geometry_binding"]["sliced_from"]["cols"] == [32, 608]


# =========================================================================== #
# 5. the default is byte-identical, and the refusals are real                   #
# =========================================================================== #
def test_no_subframe_is_BYTE_IDENTICAL_to_the_pre_fix_loader(tmp_path):
    """pod1 is mid-run on `build_v2_providers`. The new argument must be
    invisible when it is not passed."""
    va = _cache(tmp_path, "val", [3])
    (a_,) = V2.build_v2_providers(va, verbose=False)
    (b_,) = V2.build_v2_providers(va, verbose=False, frame=None, rebuild=True)
    assert tuple(a_.frames.shape) == tuple(b_.frames.shape) == (T_OUT, 9, 256, 640)
    assert torch.equal(a_.frames[0:T_OUT], b_.frames[0:T_OUT])
    assert a_.episode_id == b_.episode_id
    # and asking for the frame the cache ALREADY is stays a no-op
    (c_,) = V2.build_v2_providers(va, verbose=False, frame=PARENT)
    assert torch.equal(c_.frames[0:T_OUT], a_.frames[0:T_OUT])


def test_a_LOSSY_cache_is_REFUSED_and_the_override_is_explicit(tmp_path):
    """⚠️ THE LOAD-BEARING PRECONDITION. A slice equals a rebuild only for a
    LOSSLESS cache: re-encoding a JPEG at a new crop offset moves the 8x8
    blocks. Both v5 caches are `codec: "png"` (MEASURED on pod2), which is what
    makes this free — but nothing may assume it."""
    va = _cache(tmp_path, "jpg", [4], codec="jpeg")
    with pytest.raises(ValueError, match="LOSSY"):
        V2.build_v2_providers(va, verbose=False, frame=CLEAN)
    (p,) = V2.build_v2_providers(va, verbose=False, frame=CLEAN,
                                 allow_lossy=True, rebuild=True)
    assert tuple(p.frames.shape) == (T_OUT, 9, 176, 624)


@pytest.mark.parametrize("spec,match", [
    ("175x624", "EVEN margins"),        # odd margin => half-pixel boresight
    ("176x623", "EVEN margins"),
    ("300x640", "can only shrink"),     # a sub-frame cannot widen the field
])
def test_an_impossible_subframe_is_REFUSED_at_the_seam(tmp_path, spec, match):
    a, _, _ = _args(tmp_path, spec)
    with pytest.raises(ValueError, match=match):
        T.resolve_v2_frames(a, flagship4b_config())


@pytest.mark.parametrize("spec", ["176", "176*624", "abcxdef"])
def test_a_malformed_subframe_spec_is_REFUSED(tmp_path, spec):
    with pytest.raises(SystemExit, match="HxW"):
        T.parse_subframe(spec)


def test_a_RESAMPLE_is_not_a_slice(tmp_path):
    """A frame at a different focal is a rebuild, not a crop. The loader must
    refuse rather than silently produce pixels through the wrong camera model."""
    va = _cache(tmp_path, "val", [3])
    wrong = CanonicalFrame(height=176, width=624, f_ref=400.0,
                           projection="cylindrical")
    with pytest.raises(ValueError, match="not a slice"):
        V2.build_v2_providers(va, verbose=False, frame=wrong)


# =========================================================================== #
# 6. the trainer really routes through the seam                                 #
# =========================================================================== #
def test_train_delegates_to_the_seam_and_keeps_no_second_path():
    """The v2 branch of `train()` must call `build_v2_data`. A second, direct
    `build_v2_providers(...)` call in the trainer is how the frame would get
    lost again — so there must be exactly one, inside the seam."""
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "train_flagship_v4.py").read_text(encoding="utf-8")
    assert "train_eps, val_eps = build_v2_data(" in src
    assert "cache_frame=cache_frame, train_frame=frame" in src
    import re
    calls = re.findall(r"build_v2_providers\((.*?)\)\n", src, flags=re.S)
    assert len(calls) == 2, calls          # train + val, both inside the seam
    assert all("frame=slice_frame" in c for c in calls), calls
