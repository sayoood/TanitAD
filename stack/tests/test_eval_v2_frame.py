"""⭐ THE EVALUATOR CAN READ A V2 CACHE — and it REFUSES to score a checkpoint
on a frame it was not trained on.

THE BLOCKER THIS FILE CLOSES. v5 trains at 120 deg / 256x640, where the RAW
epcache is ~697 GB for one split and fits on no host in the fleet, so v5's
corpus can only be a v2 compressed cache. MEASURED at two probes before this
change: `eval_flagship_v4.py` took only `--val-cache` (raw epcache), and
`build_v2_providers` was *called* from exactly TWO files in the whole repo,
both trainers — none in `scripts/eval_*`, none in `taniteval/`.
⇒ **a v5 checkpoint was TRAINABLE BUT NOT EVALUABLE ON ITS OWN CORPUS**, so no
gate could be run on it and no CONTINUE/RESTART decision was reachable.

THE SECOND DEFECT, which is the one that is easy to miss. Making the evaluator
read v2 is not enough: the model is trained on a CENTRED SLICE of the cache
(the rig-clean fix), and an evaluator handed the same cache without
`--v2-subframe` reads the un-sliced parent. Same corpus, same clips, same
membership digest, different pixels, plausible ADE. That is the `ego=` failure
in geometry — *trained with a capability, scored without it* — and it is
reachable by pure OMISSION from a gate command.

So this file asserts BOTH directions, and the failing one is real:

* the seam DELIVERS the sub-frame and it is the exact bit-identical slice;
* ⛔ dropping `frame=` at the eval seam REFUSES (the pre-fix world, verbatim);
* ⛔ omitting `--v2-subframe` against a checkpoint whose own `config.json` says
  it trained on one REFUSES, and the refusal names the flag value;
* ⛔ and the refusal is on the **CLI path**, reached before the checkpoint is
  even opened — not merely on a function a gate command might not call.

⚠️ ZERO SKIPS, by the same argument the trainer's wiring suite used: a
`importorskip("torchvision")` here would skip on the dev box, which is exactly
where these guards are most likely to be run. A LOSSLESS RAW CODEC is installed
in place of `torchvision.io` **only when the real one is absent**, and removed
from `sys.modules` immediately after the import. On a pod the real PNG codec
runs the identical assertions; the codec is not what is under test.

🔒 Every clip id here is synthetic.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import types
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from tanitad.config import flagship4b_config                       # noqa: E402
from tanitad.data import parity                                    # noqa: E402
from tanitad.data.calib import (CanonicalFrame,                    # noqa: E402
                                CANONICAL_256,
                                PHYSICALAI_RIG_CLEAN_128x576,
                                PHYSICALAI_RIG_CLEAN_176x624,
                                PHYSICALAI_WIDE120_256x640)

# --------------------------------------------------------------------------- #
# codec shim — installed ONLY if torchvision is genuinely absent               #
# --------------------------------------------------------------------------- #
HAVE_TORCHVISION = importlib.util.find_spec("torchvision") is not None


def _raw_encode(t: torch.Tensor) -> torch.Tensor:
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
import tanitad.data.v2_dataset as V2                               # noqa: E402
if not HAVE_TORCHVISION:                  # leave sys.modules exactly as found
    sys.modules.pop("torchvision.io", None)
    sys.modules.pop("torchvision", None)

import eval_flagship_v4 as E                                       # noqa: E402
import train_flagship_v4 as T                                      # noqa: E402

PARENT = PHYSICALAI_WIDE120_256x640                # 256x640, 120.000 deg
CLEAN = PHYSICALAI_RIG_CLEAN_176x624               # 176x624, 117.000 deg
TILED = PHYSICALAI_RIG_CLEAN_128x576               # 128x576, 108.000 deg
N_RAW, N_STACK = 6, 3
T_OUT = N_RAW - (N_STACK - 1)


# --------------------------------------------------------------------------- #
# a REAL v2 cache, in the exact build_compressed on-disk format                 #
# --------------------------------------------------------------------------- #
def _pattern(n: int, h: int, w: int, seed: int) -> torch.Tensor:
    """POSITION-DEPENDENT content: a slice off by one row changes every byte."""
    f = torch.arange(n).view(n, 1, 1, 1) * 29
    c = torch.arange(3).view(1, 3, 1, 1) * 3
    r = torch.arange(h).view(1, 1, h, 1) * 7
    x = torch.arange(w).view(1, 1, 1, w) * 13
    return ((f + c + r + x + seed) % 256).to(torch.uint8)


def _write_clip(path: Path, *, frame: CanonicalFrame, seed: int,
                codec: str = "png") -> None:
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


def _cache(root: Path, name: str, seeds, *, frame: CanonicalFrame = PARENT,
           codec: str = "png") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        _write_clip(d / f"synth{s:08d}.v2ep.pt", frame=frame, seed=s,
                    codec=codec)
    return d


GEO_ARGV = ["--frame-h", "256", "--frame-w", "640", "--frame-hfov", "120",
            "--projection", "cylindrical"]


def _run_config(model_frame: CanonicalFrame,
                cache_frame: CanonicalFrame | None) -> dict:
    """The `config.json` the v5 trainer writes, in the two shapes it writes it.

    Built from the SAME `frame.report()` the trainer's `_geometry_report` emits,
    so this fixture cannot drift into testing a schema nobody produces."""
    return {"geometry": model_frame.report(),
            "geometry_cache": (None if cache_frame is None else
                               {**cache_frame.report(),
                                "subframe_of_cache": {"rows": [40, 216],
                                                      "cols": [8, 632]},
                                "note": "the cache is UNCHANGED on disk"}),
            "args": {}}


def _eval_args(tmp_path, *, v2=True, subframe: str | None = "176x624",
               extra=(), codec: str = "png", ckpt: str | None = None,
               run_cfg: dict | None = None, geometry=True):
    """A parsed `eval_flagship_v4` argv, plus the val dir it points at."""
    va = _cache(tmp_path, "val", [3], codec=codec) if v2 else (
        _raw_val(tmp_path))
    ckdir = tmp_path / "run"
    ckdir.mkdir(exist_ok=True)
    if run_cfg is not None:
        (ckdir / "config.json").write_text(json.dumps(run_cfg),
                                           encoding="utf-8")
    argv = ["--ckpt", ckpt or str(ckdir / "ckpt.pt"),
            "--key", "t", "--out", str(tmp_path / "o.json")]
    argv += (["--v2-val-cache", str(va)] if v2 else ["--val-cache", str(va)])
    if geometry and v2:
        argv += GEO_ARGV
    if subframe is not None:
        argv += ["--v2-subframe", subframe]
    argv += list(extra)
    return argv, va


def _raw_val(tmp_path) -> Path:
    d = tmp_path / parity.PARITY_VAL_KEY
    d.mkdir(parents=True, exist_ok=True)
    for i in range(40):
        (d / f"ep_{i:05d}.pt").touch()
    return d


def _parsed(argv):
    """Parse without running — mirrors what `main` does before any GPU work."""
    return _build_eval_parser().parse_args(argv)


def _build_eval_parser():
    """`eval_flagship_v4.main`'s own parser, reached without invoking main.

    Built by calling `main` with a deliberately unparseable argv and catching
    the SystemExit would be fragile, so instead the flags are asserted through
    `main` itself (below) and the light-weight parsing tests use argparse
    directly on the module's documented surface."""
    import argparse

    import goal_modes
    from tanitad.geometry import add_geometry_args
    ap = argparse.ArgumentParser("eval_flagship_v4")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-cache", default=None)
    ap.add_argument("--v2-val-cache", default=None, nargs="+")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None)
    ap.add_argument("--require-parity", action="store_true")
    add_geometry_args(ap)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-config", default=None)
    ap.add_argument("--goal-mode", choices=goal_modes.GOAL_MODES,
                    default="oracle")
    ap.add_argument("--select-rule", choices=("as-trained", "c2-wm-ref"),
                    default="as-trained")
    return ap


# =========================================================================== #
# 1. ⭐ THE BLOCKER: an evaluator that can read a v2 cache at all               #
# =========================================================================== #
def test_the_evaluator_reads_a_v2_cache_and_gets_the_exact_subframe(tmp_path):
    """⭐ THE HEADLINE. Before this, `build_v2_providers` had ZERO eval-side
    callers, so a v5 checkpoint could not be scored on its own corpus."""
    argv, va = _eval_args(tmp_path, subframe="176x624")
    a = _parsed(argv)
    cfg = flagship4b_config()
    cache_frame, model_frame = E.resolve_eval_frames(a, cfg)
    assert cache_frame.hw == (256, 640) and model_frame.hw == (176, 624)
    assert model_frame == CLEAN

    eps, prov = E.build_v2_val_episodes(a, cache_frame=cache_frame,
                                        train_frame=model_frame, verbose=False)
    assert len(eps) == 1
    assert tuple(eps[0].frames.shape) == (T_OUT, 3 * N_STACK, 176, 624)

    # ⭐ BIT-IDENTICAL to the parent's [40:216, 8:632], computed independently
    # by the SAME loader with NO frame.
    (ref,) = V2.build_v2_providers(va, verbose=False)
    assert tuple(ref.frames.shape) == (T_OUT, 9, 256, 640)
    got = eps[0].frames[0:T_OUT]
    want = ref.frames[0:T_OUT][:, :, 40:216, 8:632]
    assert torch.equal(got, want)
    assert int((got.float() - want.float()).abs().max()) == 0

    sf = prov["geometry_binding"]["sliced_from"]
    assert sf["rows"] == [40, 216] and sf["cols"] == [8, 632]
    assert prov["geometry_binding"]["cache_frame_shapes"] == [[176, 624]]


def test_the_other_frame_is_one_flag_value(tmp_path):
    """128x576 must work end-to-end too — the PI still owns the frame choice,
    and nothing in the evaluator may prefer one."""
    argv, va = _eval_args(tmp_path, subframe="128x576")
    a = _parsed(argv)
    cache_frame, model_frame = E.resolve_eval_frames(a, flagship4b_config())
    assert model_frame == TILED
    eps, _ = E.build_v2_val_episodes(a, cache_frame=cache_frame,
                                     train_frame=model_frame, verbose=False)
    assert tuple(eps[0].frames.shape) == (T_OUT, 9, 128, 576)
    (ref,) = V2.build_v2_providers(va, verbose=False)
    assert torch.equal(eps[0].frames[0:T_OUT],
                       ref.frames[0:T_OUT][:, :, 64:192, 32:608])


def test_none_reads_the_cache_exactly_as_built(tmp_path):
    argv, _ = _eval_args(tmp_path, subframe="none")
    a = _parsed(argv)
    cache_frame, model_frame = E.resolve_eval_frames(a, flagship4b_config())
    assert cache_frame == model_frame == PARENT
    eps, prov = E.build_v2_val_episodes(a, cache_frame=cache_frame,
                                        train_frame=model_frame, verbose=False)
    assert tuple(eps[0].frames.shape) == (T_OUT, 9, 256, 640)
    assert "sliced_from" not in prov["geometry_binding"]


# =========================================================================== #
# 2. ⛔ THE RED HALF — the loader that is never told the frame                  #
# =========================================================================== #
def test_it_FAILS_LOUD_when_the_eval_loader_is_never_told_the_frame(
        tmp_path, monkeypatch):
    """⛔ The pre-fix world reproduced verbatim on the EVAL side: the sub-frame
    is configured, and the loader never receives it. Without this the harness
    would publish an ADE off the rig-asymmetric parent frames."""
    argv, _ = _eval_args(tmp_path, subframe="176x624")
    a = _parsed(argv)
    cache_frame, model_frame = E.resolve_eval_frames(a, flagship4b_config())

    real = V2.build_v2_providers
    seen: list = []

    def _drops_the_frame(dirs, **kw):
        seen.append(kw.pop("frame", "ABSENT"))       # <-- the defect, verbatim
        return real(dirs, **kw)

    monkeypatch.setattr(V2, "build_v2_providers", _drops_the_frame)
    with pytest.raises(parity.ParityViolation) as e:
        E.build_v2_val_episodes(a, cache_frame=cache_frame,
                                train_frame=model_frame, verbose=False)
    msg = str(e.value)
    assert "SUB-FRAME WAS DECLARED BUT NEVER APPLIED" in msg
    assert "build_v2_providers" in msg          # names the call that must change
    assert seen and seen[0] is not None and seen[0] != "ABSENT"


def test_the_ordinary_wrong_cache_mismatch_still_fires_and_does_not_claim_the_subframe(
        tmp_path):
    """The two failures must stay distinguishable: a cache at the WRONG raster
    is not the same defect as a sub-frame that was never applied, and a message
    that conflates them sends the reader to the wrong file."""
    argv, _ = _eval_args(tmp_path, subframe=None, geometry=False)
    a = _parsed(argv)                    # declares 256x256, the cache is 256x640
    with pytest.raises(parity.ParityViolation) as e:
        E.build_v2_val_episodes(a, cache_frame=CANONICAL_256,
                                train_frame=CANONICAL_256, verbose=False)
    msg = str(e.value)
    assert "the cache is not the frame the run declares" in msg
    assert "SUB-FRAME WAS DECLARED BUT NEVER APPLIED" not in msg


# =========================================================================== #
# 3. ⛔ THE `ego=` SHAPE — scored on a frame it was not trained on              #
# =========================================================================== #
def test_scoring_a_SLICED_checkpoint_on_the_PARENT_is_refused(tmp_path):
    """⛔ THE FAILURE THAT IS REACHABLE BY OMISSION. The run trained on
    176x624; the gate command forgets `--v2-subframe`; every membership check
    passes and the ADE is computed off different pixels."""
    rc = _run_config(CLEAN, PARENT)
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_eval_frame_matches_run(rc, PARENT, label="t",
                                             cache_frame=PARENT)
    msg = str(e.value)
    assert "SCORING A CHECKPOINT ON A FRAME IT WAS NOT TRAINED ON" in msg
    assert "--v2-subframe 176x624" in msg       # names the value that fixes it
    assert "176x624" in msg and "256x640" in msg


def test_the_matching_frame_passes_and_records_both(tmp_path):
    rc = _run_config(CLEAN, PARENT)
    out = parity.assert_eval_frame_matches_run(rc, CLEAN, label="t",
                                               cache_frame=PARENT)
    assert out["checked"] is True
    assert out["trained_frame"] == CLEAN.to_dict()
    assert out["trained_cache_frame"] == PARENT.to_dict()
    assert out["cache_frame_matches_run"] is True


def test_a_same_size_slice_of_a_DIFFERENT_cache_is_refused():
    """The subtler twin: the MODEL frame agrees, so the encoder loads and the
    number looks fine — but the slice came out of a different field."""
    other = CanonicalFrame(height=256, width=640, f_ref=407.0,
                           projection="cylindrical")   # 90 deg, same pixels
    rc = _run_config(CLEAN, PARENT)
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_eval_frame_matches_run(rc, CLEAN, label="t",
                                             cache_frame=other)
    assert "the run sliced a DIFFERENT cache" in str(e.value)


def test_an_unsliced_run_scored_at_a_subframe_is_ALSO_refused():
    """Both directions. A run that trained on the whole cache and is scored on
    a slice is the same class of error with the sign flipped."""
    rc = _run_config(PARENT, None)              # geometry_cache is null
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_eval_frame_matches_run(rc, CLEAN, label="t")
    msg = str(e.value)
    assert "SCORING A CHECKPOINT ON A FRAME IT WAS NOT TRAINED ON" in msg
    # geometry_cache is null -> the fix is the --frame-* flags, not --v2-subframe
    assert "--frame-h 256 --frame-w 640" in msg
    assert "read the cache unsliced" in msg


def test_a_checkpoint_with_no_geometry_block_is_UNVERIFIED_not_silently_passed():
    """Every pre-2026-07-27 arm is in this state. Refusing would make the
    historical record unreproducible; passing silently would let a v5 number
    through unchecked. It is reported, loudly, as neither."""
    for rc in (None, {}, {"geometry": None}, {"geometry": {"tag": "x"}}):
        out = parity.assert_eval_frame_matches_run(rc, CLEAN, label="t")
        assert out["checked"] is False
        assert "UNVERIFIED" in out["note"]
        assert out["eval_frame"]["height"] == 176


def test_the_deployed_raw_path_still_passes_its_own_config(tmp_path):
    """A historical v4 run: 256x256 pinhole, no geometry_cache, eval declares
    nothing. This must stay a PASS or every existing eval command breaks."""
    rc = _run_config(CANONICAL_256, None)
    out = parity.assert_eval_frame_matches_run(rc, CANONICAL_256, label="t")
    assert out["checked"] is True and "cache_frame_matches_run" not in out


# =========================================================================== #
# 4. ⛔ ON THE CLI PATH — reached before the checkpoint is even opened          #
# =========================================================================== #
def test_the_frame_guard_fires_from_main_BEFORE_the_ckpt_is_loaded(tmp_path):
    """⛔ A guard on a function a gate command does not call is not a guard.
    This drives `main(argv)` itself with a checkpoint path that DOES NOT EXIST:
    the refusal must arrive first, which proves it is on the command path."""
    rc = _run_config(CLEAN, PARENT)
    argv, _ = _eval_args(tmp_path, subframe=None, run_cfg=rc)   # flag FORGOTTEN
    assert not Path(argv[argv.index("--ckpt") + 1]).exists()
    with pytest.raises(parity.ParityViolation) as e:
        E.main(argv)
    assert "--v2-subframe 176x624" in str(e.value)


def test_main_ACCEPTS_the_matching_frame_and_then_reaches_the_ckpt(tmp_path):
    """The GREEN twin of the test above: with the right flag the frame guard
    passes and `main` proceeds — failing later, on the missing checkpoint. The
    two tests together show the guard is neither absent nor unconditional."""
    rc = _run_config(CLEAN, PARENT)
    argv, _ = _eval_args(tmp_path, subframe="176x624", run_cfg=rc)
    with pytest.raises(Exception) as e:
        E.main(argv)
    assert not isinstance(e.value, parity.ParityViolation)
    assert "ckpt.pt" in str(e.value) or isinstance(e.value, (FileNotFoundError,
                                                             OSError))


# =========================================================================== #
# 5. corpus-format arguments                                                    #
# =========================================================================== #
def test_both_corpus_formats_at_once_is_refused(tmp_path):
    a = _parsed(["--ckpt", "c", "--key", "k", "--out", "o",
                 "--val-cache", str(tmp_path), "--v2-val-cache", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        E.assert_val_corpus_args(a)
    assert "two CORPUS FORMATS" in str(e.value)


def test_neither_corpus_format_is_refused():
    a = _parsed(["--ckpt", "c", "--key", "k", "--out", "o"])
    with pytest.raises(SystemExit) as e:
        E.assert_val_corpus_args(a)
    assert "--v2-val-cache" in str(e.value)


def test_v2_subframe_on_the_raw_path_is_refused(tmp_path):
    argv, _ = _eval_args(tmp_path, v2=False, subframe="176x624")
    with pytest.raises(SystemExit) as e:
        E.main(argv)
    assert "--v2-subframe applies to --v2-val-cache only" in str(e.value)


def test_c2_wm_ref_is_refused_on_a_non_deployed_frame(tmp_path):
    """The C2 scorer is a SEPARATE 256x256 v1 trunk that encodes the same batch
    as the arm. On a 176x624 frame it cannot, and a rule whose SIGN depends on
    its scorer must never be reached broken."""
    argv, _ = _eval_args(tmp_path, subframe="176x624",
                         extra=["--select-rule", "c2-wm-ref",
                                "--c2-scorer", "self"])
    with pytest.raises(SystemExit) as e:
        E.main(argv)
    assert "c2-wm-ref is not available on a non-deployed frame" in str(e.value)


def test_require_parity_refuses_an_unregistered_v2_val_cache(tmp_path):
    argv, _ = _eval_args(tmp_path, subframe="176x624",
                         extra=["--require-parity"])
    a = _parsed(argv)
    cache_frame, model_frame = E.resolve_eval_frames(a, flagship4b_config())
    with pytest.raises(parity.ParityViolation) as e:
        E.build_v2_val_episodes(a, cache_frame=cache_frame,
                                train_frame=model_frame, verbose=False)
    assert "unregistered v2 cache" in str(e.value)


def test_without_require_parity_an_unregistered_cache_only_warns(tmp_path,
                                                                 capsys):
    argv, _ = _eval_args(tmp_path, subframe="176x624")
    a = _parsed(argv)
    cache_frame, model_frame = E.resolve_eval_frames(a, flagship4b_config())
    eps, prov = E.build_v2_val_episodes(a, cache_frame=cache_frame,
                                        train_frame=model_frame, verbose=False)
    assert len(eps) == 1 and prov["val_parity"]["parity"] is False
    assert "NON-PARITY" in capsys.readouterr().out


# =========================================================================== #
# 6. the RAW path must be BYTE-IDENTICAL to before                              #
# =========================================================================== #
def test_the_raw_epcache_path_still_asserts_the_val_guard(tmp_path):
    """The evaluator's pre-existing val-integrity guard must survive the
    refactor: `load_val_episodes` still routes the raw path through
    `parity.assert_val_cache` BEFORE a single tensor is unpickled (the caches
    here are empty marker files)."""
    argv, va = _eval_args(tmp_path, v2=False, subframe=None)
    a = _parsed(argv)
    seen = {}
    real = parity.assert_val_cache

    def _spy(cache_dir, **kw):
        seen["called"] = str(cache_dir)
        return real(cache_dir, **kw)

    parity.assert_val_cache = _spy
    try:
        with pytest.raises(Exception):        # empty ep_*.pt -> load fails LATER
            E.load_val_episodes(a, cache_frame=CANONICAL_256,
                                train_frame=CANONICAL_256)
    finally:
        parity.assert_val_cache = real
    assert seen.get("called") == str(va)


def test_no_geometry_flags_leaves_the_config_at_the_deployed_frame(tmp_path):
    """`flagship4b_config()` IS the deployed frame, so the new geometry
    resolution must be a NO-OP on every existing eval command."""
    argv, _ = _eval_args(tmp_path, v2=False, subframe=None)
    a = _parsed(argv)
    before = flagship4b_config()
    cfg = flagship4b_config()
    cache_frame, model_frame = E.resolve_eval_frames(a, cfg)
    assert cache_frame == model_frame == CANONICAL_256
    assert cfg.encoder.image_hw() == before.encoder.image_hw() == (256, 256)
    assert cfg.geometry is None


def test_eval_cfg_sizes_the_encoder_for_the_frame():
    """`load_v1_from_ck` / `load_v4_from_ck` are STRICT loads: if the encoder is
    not sized for the eval frame the failure is a pos_embed shape error three
    files from its cause."""
    base = E._eval_cfg()
    assert base.encoder.image_hw() == (256, 256)
    sub = E._eval_cfg(CLEAN)
    assert sub.encoder.image_hw() == (176, 624)
    gh, gw = sub.encoder.token_grid()
    assert (gh, gw) == (11, 39) and gh * gw == 429       # the MEASURED v5 count
    assert E._eval_cfg(PARENT).encoder.token_grid() == (16, 40)


# =========================================================================== #
# 7. structural pins — no drift, and NO FOURTH DECODE IMPLEMENTATION            #
# =========================================================================== #
def test_the_evaluator_resolves_the_frame_THROUGH_THE_TRAINERS_OWN_FUNCTION():
    """If the two resolved `--v2-subframe` independently, the rig-clean fix
    would be exactly as easy to lose on the eval side as it was on the train
    side — which is the defect that produced this whole stream."""
    src = (SCRIPTS / "eval_flagship_v4.py").read_text(encoding="utf-8")
    assert "from train_flagship_v4 import resolve_v2_frames" in src
    a = _build_eval_parser().parse_args(
        ["--ckpt", "c", "--key", "k", "--out", "o", "--v2-val-cache", "d",
         *GEO_ARGV, "--v2-subframe", "176x624"])
    cfg_e, cfg_t = flagship4b_config(), flagship4b_config()
    assert E.resolve_eval_frames(a, cfg_e) == T.resolve_v2_frames(a, cfg_t)


#: Every place in the repo that turns v2 payload bytes back into pixels.
#: THREE, and this change adds NONE — the evaluator calls the loader the
#: trainers already read through. The count is pinned because the last fix had
#: to be applied to a second copy nobody knew existed: `load_compressed(frame=)`
#: was verified, shipped, and unreachable from the trainer, which decodes in
#: `v2_dataset._decode_stacked` instead.
DECODE_IMPLEMENTATIONS = {
    "scripts/v2_compressed.py",      # the BUILDER + its round-trip validator
    "scripts/slice_v2_cache.py",     # the RE-EMITTER (no rebuild)
    "tanitad/data/v2_dataset.py",    # the LOADER — trainers AND now the eval
}


def test_there_are_exactly_three_v2_decode_implementations_and_eval_adds_none():
    """⛔ A fourth copy is how a verified fix becomes unreachable. If this test
    fails, do not update the number: check whether the new file should have
    called `v2_dataset` instead."""
    root = SCRIPTS.parent
    found = set()
    for p in list(root.glob("scripts/*.py")) + list(root.glob("tanitad/**/*.py")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        code = "\n".join(ln for ln in txt.splitlines()
                         if not ln.lstrip().startswith("#"))
        if "decode_png" in code and "decode_jpeg" in code:
            found.add(p.relative_to(root).as_posix())
    assert found == DECODE_IMPLEMENTATIONS, (
        f"the set of v2 decode implementations changed: {sorted(found)}")
    ev = (SCRIPTS / "eval_flagship_v4.py").read_text(encoding="utf-8")
    assert "decode_png" not in ev and "decode_jpeg" not in ev
    assert "build_v2_providers" in ev


def test_the_evaluator_documents_the_two_corpus_formats():
    """The `--val-cache` docstring examples were the only record of how to run
    this harness; a v2 mode nobody can find is a v2 mode nobody runs."""
    doc = E.__doc__ or ""
    assert "--v2-val-cache" in doc and "--v2-subframe" in doc
    assert "697 GB" in doc            # WHY v5 can only be scored through v2
