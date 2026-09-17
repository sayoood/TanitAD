"""The 416 x 1024 geometry (PI, 2026-09-17) is DECLARED everywhere it must be.

⚠️ WHY THIS FILE EXISTS. The same omission bit twice. A geometry is not finished
when its cache is built -- it is finished when every table that must declare it
does. MEASURED 2026-09-17 20:39Z: the first resnet101 run on the freshly built
416x1024 cache was refused by `refc_v3_train._agent_cam_frames`, exactly as the
256x1024 cache had been refused the day before, for the same reason.

⛔ AND THE REFUSAL IS RIGHT, so this file must never be "fixed" by loosening it.
A frame is NOT its pixel count: it carries `f_ref` AND the projection, and this
corpus is CYLINDRICAL (the column is linear in azimuth). Inventing an `f_ref`, or
applying the pinhole formula, reads 92.6 deg for a 120 deg camera and looks
entirely plausible.

The mutation arm below is the one that matters: it reintroduces the REAL
historical defect -- the rounded `f_ref` 488.92 the PI wrote by hand -- and
requires the 120.0000 deg check to go RED on it. A test that only asserts the
good value passes equally on a re-typed constant, which is the drift this whole
single-spelling discipline exists to prevent.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tanitad.data.calib import CanonicalFrame                # noqa: E402
from tanitad.models import trunk_shapes as TS                # noqa: E402

F416 = TS.FRAME_416x1024
F256 = TS.FRAME_256x1024
CACHE_MANIFEST = pathlib.Path(
    "D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl/manifest.json")


def hfov_deg(f: CanonicalFrame) -> float:
    """Cylindrical: x = f_ref * phi, so HFOV = 2 * (W/2) / f_ref -- LINEAR."""
    assert f.projection == "cylindrical"
    return math.degrees(2.0 * (f.width / 2.0) / f.f_ref)


def vfov_deg(f: CanonicalFrame) -> float:
    return math.degrees(2.0 * math.atan((f.height / 2.0) / f.f_ref))


# --------------------------------------------------------------------------- #
# the declaration itself                                                       #
# --------------------------------------------------------------------------- #

def test_416_keeps_the_field_at_exactly_120_deg():
    assert hfov_deg(F416) == pytest.approx(120.0, abs=1e-9)


def test_416_shares_f_ref_with_256x1024_EXACTLY_because_the_width_is_the_same():
    # a cylindrical f_ref is a HORIZONTAL quantity; changing only the height
    # cannot change it. If these ever differ, one of them was re-typed.
    assert F416.f_ref == F256.f_ref
    assert F416.width == F256.width == 1024
    assert (F416.height, F256.height) == (416, 256)


def test_416_is_stride_32_aligned_and_408_is_not():
    assert 416 % 32 == 0
    assert 408 % 32 == 24                     # the height first authorised
    assert TS.feature_hw(F416, 32) == (13, 32)
    assert TS.feature_hw(F416, 16) == (26, 64)
    with pytest.raises(ValueError):
        TS.feature_hw(CanonicalFrame(height=408, width=1024, f_ref=F416.f_ref,
                                     projection="cylindrical"), 32)


def test_416_exceeds_the_256x640_reference_vertical_field():
    # the point of choosing 416 over a smaller aligned height: no vertical
    # field is given up to gain the stride-32 alignment.
    assert vfov_deg(F416) > vfov_deg(TS.FRAME_256x640)
    assert vfov_deg(F416) == pytest.approx(46.0921, abs=1e-3)


# --------------------------------------------------------------------------- #
# every table that must declare it                                            #
# --------------------------------------------------------------------------- #

def test_the_trainers_rig_camera_table_declares_416_and_reuses_THE_OBJECT():
    import refc_v3_train as T                                # noqa: E402
    tbl = T._agent_cam_frames()
    assert (416, 1024) in tbl, (
        "--agent-rig-camera extrinsics refuses an undeclared geometry; without "
        "this row the 416x1024 cache cannot train the monocular terms")
    assert tbl[(416, 1024)] is F416, (
        "the table must REFERENCE trunk_shapes.FRAME_416x1024, not re-type it: "
        "a second spelling of a camera constant is how two files drift apart "
        "while both look right")


# --------------------------------------------------------------------------- #
# the cross-check: an INDEPENDENT producer must agree                          #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not CACHE_MANIFEST.is_file(),
                    reason="INCONCLUSIVE, never passing: the 416x1024 cache is "
                           "not on this box, so the independent producer that "
                           "would cross-check f_ref cannot be read")
def test_the_built_cache_agrees_with_the_declaration():
    fr = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))["frame"]
    assert (fr["height"], fr["width"]) == (416, 1024)
    assert fr["projection"] == "cylindrical"
    # the builder computed this from its own CanonicalFrame.from_hfov, a
    # different code path than frame_for_width -- agreement is evidence.
    assert fr["f_ref"] == pytest.approx(F416.f_ref, rel=1e-12)
    assert fr["vfov_deg"] == pytest.approx(vfov_deg(F416), abs=1e-9)


# --------------------------------------------------------------------------- #
# ⛔ THE MUTATION ARM -- the real historical defect, which must go RED          #
# --------------------------------------------------------------------------- #

def test_MUTATION_the_hand_rounded_f_ref_does_NOT_hold_the_field():
    """The PI wrote "f_ref 488.92". Re-typing it reads 120.0010 deg, not 120.

    If this assertion ever fails, the 120.0000 check above has been loosened to
    a tolerance that would ACCEPT a re-typed constant, and the single-spelling
    discipline is no longer enforced by anything.
    """
    rounded = CanonicalFrame(height=416, width=1024, f_ref=488.92,
                             projection="cylindrical")
    assert hfov_deg(rounded) != pytest.approx(120.0, abs=1e-9)
    assert hfov_deg(rounded) == pytest.approx(120.0010, abs=1e-4)
