"""A time-varying attitude track must change ONLY the drawing camera, per frame.

The recording that motivated it had EIS on, so the effective camera yaw drifted
~3 deg through the clip and no constant yaw could keep the overlay in the lane.
These tests pin the contract: interpolation by session time, the original camera
never mutated, and a constant horizon row reproducing the --horizon-row override
exactly (so switching to a track cannot silently move the horizon).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

_PKG = pathlib.Path(__file__).resolve().parents[1] / "tanitad" / "data" / "trajrecon"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", _PKG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod              # @dataclass needs the module registered
    spec.loader.exec_module(mod)
    return mod


AT = _load("attitude_track")
CAM = _load("camera")


def _cam():
    return CAM.CameraModel(width=1920, height=1080, fx=1533.0, fy=1533.0,
                           cx=960.0, cy=540.0, yaw=np.deg2rad(-6.40),
                           pitch=np.arctan2(463.0 - 540.0, 1533.0))


def _track(**kw):
    d = dict(t_session_s=[0.0, 10.0, 20.0], yaw_deg=[-7.8, -6.4, -4.9])
    d.update(kw)
    return AT.attitude_track_from(d)


def test_yaw_is_interpolated_on_session_time():
    tr = _track()
    assert tr.yaw_at(5.0) == pytest.approx(-7.1)
    assert tr.yaw_at(15.0) == pytest.approx(-5.65)
    # clamped outside the track, never extrapolated
    assert tr.yaw_at(-3.0) == pytest.approx(-7.8)
    assert tr.yaw_at(99.0) == pytest.approx(-4.9)


def test_camera_for_time_sets_yaw_and_never_mutates_the_original():
    cam = _cam()
    yaw0, pitch0 = cam.yaw, cam.pitch
    c = AT.camera_for_time(cam, _track(), 10.0)
    assert np.rad2deg(c.yaw) == pytest.approx(-6.4)
    assert c is not cam
    assert cam.yaw == yaw0 and cam.pitch == pitch0
    # no horizon in the track => pitch untouched
    assert c.pitch == pitch0


def test_constant_horizon_row_matches_the_horizon_row_override():
    cam = _cam()
    tr = _track(horizon_row=[463.0, 463.0, 463.0])
    c = AT.camera_for_time(cam, tr, 7.0)
    override_pitch = float(np.arctan2(463.0 - cam.cy, cam.fy))
    assert c.pitch == pytest.approx(override_pitch, abs=1e-12)


def test_no_track_returns_the_camera_unchanged():
    cam = _cam()
    assert AT.camera_for_time(cam, None, 3.0) is cam


def test_the_drawn_vanishing_point_follows_the_track():
    """What the fix is FOR: the drawn straight-ahead direction must move with yaw.

    At 1533 px focal a 1 deg yaw change moves the far vanishing point ~26.8 px.
    """
    cam = _cam()
    tr = _track()
    far = np.array([[1e4, 0.0, 0.0]])
    u = []
    for t in (0.0, 20.0):
        c = AT.camera_for_time(cam, tr, t)
        uv, ok = c.project(far) if hasattr(c, "project") else (None, None)
        if uv is None:
            pytest.skip("CameraModel has no project(); geometry covered elsewhere")
        u.append(float(uv[0, 0]))
    assert abs(u[1] - u[0]) == pytest.approx(2.9 * 26.76, rel=0.08)


def test_unsorted_samples_are_sorted_and_duplicates_refused():
    tr = AT.attitude_track_from(dict(t_session_s=[20.0, 0.0, 10.0],
                                     yaw_deg=[-4.9, -7.8, -6.4]))
    assert list(tr.t) == [0.0, 10.0, 20.0]
    assert list(tr.yaw_deg) == [-7.8, -6.4, -4.9]
    with pytest.raises(ValueError):
        AT.attitude_track_from(dict(t_session_s=[0.0, 0.0], yaw_deg=[1.0, 2.0]))


@pytest.mark.parametrize("bad", [
    dict(t_session_s=[0.0], yaw_deg=[1.0]),
    dict(t_session_s=[0.0, 1.0], yaw_deg=[1.0]),
    dict(t_session_s=[0.0, 1.0], yaw_deg=[1.0, 2.0], horizon_row=[400.0]),
    dict(t_session_s=[0.0, 1.0], yaw_deg=[1.0, float("nan")]),
])
def test_malformed_tracks_are_refused(bad):
    with pytest.raises(ValueError):
        AT.attitude_track_from(bad)


def test_round_trips_through_a_file(tmp_path):
    p = tmp_path / "track.json"
    p.write_text(json.dumps(dict(t_session_s=[0.0, 1.0], yaw_deg=[-6.0, -5.0],
                                 horizon_row=[463.0, 470.0], source="unit test")))
    tr = AT.load_attitude_track(str(p))
    assert tr.source == "unit test"
    assert tr.horizon_at(0.5) == pytest.approx(466.5)
