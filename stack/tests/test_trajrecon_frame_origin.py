"""`CameraModel`'s vehicle frame is measured from the TRAJECTORY ORIGIN, not the lens.

WHY THIS EXISTS
---------------
MEASURED 2026-09-15 on the `14-19-54` recording, and it cost a full retraction
(`R-2026-09-15-longitudinal`) plus a published conclusion that turned out to be
exactly backwards.

`CameraModel.t_v = [longitudinal_m, lateral_m, height_m]` is *"the camera position
expressed in the vehicle frame"*, and `to_camera` subtracts it. So a point handed
to `project` as `x = 10` is **10 m ahead of the trajectory origin**, and with this
recording's `longitudinal_m = 2.1` that is only **7.9 m ahead of the lens**.

Every closed-form range inversion in the calibration probes is the OTHER frame::

    x_camera = f * h / (v - v_h)          <- range from the LENS

Feeding one into the other asks for the lateral scale at `x - 2.1` and applies it
at `x`. The error is **-27 % at 8 m, -21 % at 10 m, -12 % at 20 m, -7 % at 30 m**:

* it is large enough to move a measured lane boundary by 0.39 m, which is how it
  was finally caught — two probes disagreeing about the *same* painted line;
* and because it is **RANGE-DEPENDENT it manufactures a drift**, which a horizon
  estimator built on "a lane boundary must have the same lateral at 10 m and at
  30 m" then attributes to the horizon. That estimator spent a day measuring this
  bug.

Nothing type-checks a datum, and both frames call the variable `x`. These tests
pin the convention numerically so the next closed form is written against a
documented fact instead of an assumption.
"""
from __future__ import annotations

import importlib.util
import sys
import pathlib

import numpy as np
import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "tanitad" / "data" / "trajrecon"

# ⚠️ Loaded BY PATH, not as `tanitad.data.trajrecon.camera`. The package `__init__`
# pulls in `toy_driving`, which imports torch, and this convention is worth pinning
# in every environment — including a slim one with no training stack installed.
# `camera.py` itself needs only numpy, so the direct load is honest, not a dodge.
_spec = importlib.util.spec_from_file_location("_trajrecon_camera", PKG / "camera.py")
_cam_mod = importlib.util.module_from_spec(_spec)
# @dataclass resolves annotations through sys.modules[cls.__module__], so the module
# has to be registered BEFORE exec_module or the decorator dies on a None lookup.
sys.modules[_spec.name] = _cam_mod
_spec.loader.exec_module(_cam_mod)
CameraModel = _cam_mod.CameraModel

LON, HEIGHT, F = 2.10, 1.50, 1500.0


def _cam(**kw):
    """A pinhole with no mount rotation, so the projection is exact in closed form."""
    p = dict(width=1920, height=1080, fx=F, fy=F, cx=960.0, cy=540.0,
             height_m=HEIGHT, lateral_m=0.0, longitudinal_m=LON,
             yaw=0.0, pitch=0.0, roll=0.0)
    p.update(kw)
    return CameraModel(**p)


def test_the_x_argument_is_measured_from_the_trajectory_origin():
    """`project([[10, 0, 0]])` is 10 m from the ORIGIN — 7.9 m from the lens."""
    cam = _cam()
    uv, valid = cam.project(np.array([[10.0, 0.0, 0.0]]))
    assert bool(np.asarray(valid).ravel()[0])
    v = float(np.asarray(uv).reshape(-1, 2)[0, 1])

    from_lens = 540.0 + F * HEIGHT / (10.0 - LON)      # the convention that holds
    from_origin = 540.0 + F * HEIGHT / 10.0            # the one that looks right
    assert from_lens == pytest.approx(v, abs=0.5), (
        f"row {v:.1f} matches neither frame; the vehicle-frame origin moved")
    assert abs(from_origin - v) > 30.0, (
        "project() now measures x from the LENS. Every probe that pairs it with "
        "x = f*h/(v - v_h) was written for the origin convention and must be revisited")


def test_the_lateral_scale_uses_the_lens_range_not_the_argument():
    """Columns per metre of lateral is `f / (x - longitudinal)`. This is the bite."""
    cam = _cam()
    uv, valid = cam.project(np.array([[10.0, 0.0, 0.0], [10.0, 1.0, 0.0]]))
    assert np.asarray(valid).all()
    uv = np.asarray(uv).reshape(-1, 2)
    du = abs(float(uv[1, 0] - uv[0, 0]))
    assert du == pytest.approx(F / (10.0 - LON), rel=0.02)
    # 21 % at 10 m -- the size of the error that produced R-2026-09-15-longitudinal
    naive = F / 10.0
    assert du / naive == pytest.approx(10.0 / (10.0 - LON), rel=0.02)


@pytest.mark.parametrize("x_lens", [8.0, 10.0, 20.0, 30.0])
def test_the_error_is_range_dependent_so_it_looks_like_a_drift(x_lens):
    """A CONSTANT scale error is caught by any known-width check. This one is not.

    Because it shrinks with range it mimics exactly the signature a horizon
    estimator looks for, which is why it survived so long. The two ratios are
    easy to conflate and I conflated them while writing this test, so both are
    pinned explicitly:

        scale applied / scale required   =  x / (x - LON)      -> too BIG
        lateral recovered / lateral true =  1 - LON / x        -> too SMALL

    They are reciprocals, not the same number: at 8 m the scale is +36 % and the
    lateral is -26 %.
    """
    cam = _cam()

    def du(arg):
        uv, ok = cam.project(np.array([[arg, 0.0, 0.0], [arg, 1.0, 0.0]]))
        assert np.asarray(ok).all()
        uv = np.asarray(uv).reshape(-1, 2)
        return abs(float(uv[1, 0] - uv[0, 0]))

    du_required = du(x_lens + LON)       # the honest call: lens range x_lens
    du_applied = du(x_lens)              # the bug: a lens range fed as an origin range
    assert du_applied / du_required == pytest.approx(x_lens / (x_lens - LON), rel=1e-3)

    # what it does to a measured lateral, which is the quantity that misled us
    lateral_ratio = du_required / du_applied
    assert lateral_ratio == pytest.approx(1.0 - LON / x_lens, rel=1e-3)
    assert lateral_ratio < 1.0, "the bug must UNDER-report a lateral, never over"


def test_the_lateral_error_shrinks_with_range_by_more_than_a_tenth():
    """The drift is the damage. A flat bias would have been caught in a day."""
    cam = _cam()

    def lateral_ratio(x_lens):
        def du(arg):
            uv, _ = cam.project(np.array([[arg, 0.0, 0.0], [arg, 1.0, 0.0]]))
            uv = np.asarray(uv).reshape(-1, 2)
            return abs(float(uv[1, 0] - uv[0, 0]))
        return du(x_lens + LON) / du(x_lens)

    near, far = lateral_ratio(10.0), lateral_ratio(30.0)
    assert near == pytest.approx(0.79, abs=0.01)     # -21 % at 10 m
    assert far == pytest.approx(0.93, abs=0.01)      # -7 % at 30 m
    assert far - near > 0.10, (
        "the frame offset no longer produces a range-dependent lateral error; if "
        "longitudinal_m became 0 this is dead weight, but if the FRAME changed, "
        "every closed-form inversion in the calibration probes needs revisiting")


def test_the_convention_is_documented_where_it_is_defined():
    """A numeric test nobody reads is half a fix; the field itself must say it."""
    src = (PKG / "camera.py").read_text(encoding="utf-8").lower()
    i = src.index("longitudinal_m")
    assert "forward" in src[i:i + 160], (
        "longitudinal_m lost its comment naming what it is measured from")
    assert "camera position expressed in the vehicle frame" in src, (
        "t_v's docstring is what tells a reader which frame project() speaks; "
        "it must not be dropped")
