"""t1_eval must roll a checkpoint in the frame that checkpoint was TRAINED in.

⛔ THE DEFECT. `t1_eval`'s geometry defaults reproduce the DEPLOYED 256x256 /
f_ref 266 / **pinhole** frame. Every v7 arm is 256x640 / hfov 120 /
**cylindrical**. A flagless v7 run therefore rolled in the wrong projection and
printed numbers that LOOK VALID — worse than a crash, and the same family as the
pinhole-FOV-on-a-cylindrical-corpus trap.

The previous mitigation was a COMMENT telling the operator to pass four flags.
These tests pin the guard that replaced it, INCLUDING the deliberate-regression
case: if the contradiction test does not FAIL an arm whose flags disagree with
the checkpoint, a pass on the adopting case means nothing.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))

import t1_eval                                                # noqa: E402


V7_ARGS = {"frame_h": 256, "frame_w": 640, "frame_hfov": 120.0,
           "projection": "cylindrical", "patch": 16}


def _ckpt_with_sidecar(tmp_path, args=None):
    """A ckpt path whose sibling config.json carries the trainer's own flags."""
    ck = tmp_path / "ckpt.pt"
    ck.write_bytes(b"not-a-real-checkpoint")      # never loaded: sidecar wins
    (tmp_path / "config.json").write_text(
        json.dumps({"args": dict(V7_ARGS if args is None else args)}),
        encoding="utf-8")
    return str(ck)


def _args(ckpt, **over):
    a = types.SimpleNamespace(ckpt=ckpt, frame_h=None, frame_w=None,
                              frame_hfov=None, f_ref=None, projection=None)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_geometry_is_read_from_the_sidecar_not_the_blob(tmp_path):
    """The sidecar is preferred, so a multi-GB ckpt is not loaded to read 4 ints."""
    geo = t1_eval.ckpt_geometry(_ckpt_with_sidecar(tmp_path))
    assert geo == {"frame_h": 256, "frame_w": 640, "frame_hfov": 120.0,
                   "projection": "cylindrical"}


def test_flagless_v7_run_adopts_the_checkpoint_frame(tmp_path):
    """⭐ THE FIX: no flags -> the v7 frame, NOT the deployed 256x256 pinhole."""
    a = _args(_ckpt_with_sidecar(tmp_path))
    adopted = t1_eval.adopt_ckpt_geometry(a)
    assert (a.frame_h, a.frame_w) == (256, 640)
    assert a.projection == "cylindrical"
    assert a.frame_hfov == 120.0
    assert adopted["projection"] == "cylindrical"
    # the defect, stated as the thing that must no longer happen
    assert not (a.frame_h == 256 and a.frame_w == 256 and
                a.projection == "pinhole")


@pytest.mark.parametrize("bad", [
    {"projection": "pinhole"},          # the exact silent failure, made loud
    {"frame_w": 256},                   # square crop against a 640-wide model
    {"frame_h": 512},
    {"frame_hfov": 92.6},               # the pinhole-formula answer on our corpus
])
def test_contradicting_flags_REFUSE(tmp_path, bad):
    """DELIBERATE REGRESSION. Each of these is a frame the model was not trained
    in. If any is silently accepted, the guard is worthless."""
    a = _args(_ckpt_with_sidecar(tmp_path), **bad)
    with pytest.raises(SystemExit) as ei:
        t1_eval.adopt_ckpt_geometry(a)
    msg = str(ei.value)
    assert "GEOMETRY CONTRADICTS THE CHECKPOINT" in msg
    # both sides must be named — a refusal that hides the values cannot be acted on
    key = next(iter(bad))
    assert key in msg


def test_matching_flags_are_accepted(tmp_path):
    """Passing the RIGHT geometry explicitly is not an error — only conflict is."""
    a = _args(_ckpt_with_sidecar(tmp_path), frame_h=256, frame_w=640,
              projection="cylindrical", frame_hfov=120.0)
    assert t1_eval.adopt_ckpt_geometry(a) == {}      # nothing to adopt, no exit


def test_f_ref_and_hfov_are_never_both_set(tmp_path):
    """hfov and f_ref are two spellings of one quantity and `frame_from_args`
    refuses both. Adopting the checkpoint's hfov over a passed f_ref would
    manufacture exactly that refusal."""
    a = _args(_ckpt_with_sidecar(tmp_path), f_ref=305.5774907364391)
    t1_eval.adopt_ckpt_geometry(a)
    assert a.frame_hfov is None
    assert a.f_ref == pytest.approx(305.5774907364391)


def test_flagship_checkpoint_is_untouched(tmp_path):
    """A ckpt with no recorded geometry returns None and changes nothing, so the
    v1/v4/v5f path is provably unaffected."""
    ck = tmp_path / "flagship.pt"
    ck.write_bytes(b"x")
    (tmp_path / "config.json").write_text(json.dumps({"args": {"batch": 8}}),
                                          encoding="utf-8")
    a = _args(str(ck))
    assert t1_eval.adopt_ckpt_geometry(a) is None
    assert a.frame_h is None and a.projection is None
