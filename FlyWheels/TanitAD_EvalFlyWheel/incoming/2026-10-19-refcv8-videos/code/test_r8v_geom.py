"""test_r8v_geom.py -- controls for the cylindrical projection.  Expectations are LITERALS derived by hand, never from the code under test.

    PYTHONPATH=<launch tree>/stack python -m pytest -q test_r8v_geom.py     (the repo-agreement test needs torch + the launch tree; it skips otherwise)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8v_geom as G  # noqa: E402

F = 488.92398517830253
CX, CY = 511.5, 207.5


def test_nominal_straight_ahead_literal():
    # camera at (1.5, 0, 1.5), zero pitch.  Ground point (21.5, 0, 0): cam z = 20, y_cam = +1.5 (down) -> row = CY + F*1.5/20 ; col = CX
    cam = G.CylCam.nominal(x_m=1.5, h_m=1.5)
    col, row, ok = cam.project_ground([[21.5, 0.0]])
    assert ok[0]
    assert abs(col[0] - 511.5) < 1e-9
    assert abs(row[0] - (207.5 + F * 1.5 / 20.0)) < 1e-9          # 244.1 .. literal formula, hand-derived


def test_cylindrical_is_linear_in_azimuth_not_pinhole():
    # a ground point 30 deg to the LEFT of the camera at ground range 20 m.  Cylindrical column = CX - F * 30deg(rad) (left = smaller col).
    cam = G.CylCam.nominal(x_m=0.0, h_m=1.5)
    az = math.radians(30.0)
    pt = [[20.0 * math.cos(az), 20.0 * math.sin(az)]]            # x fwd, y left
    col, _, ok = cam.project_ground(pt)
    assert ok[0]
    assert abs(col[0] - (CX - F * az)) < 1e-6                     # linear in azimuth
    assert abs(col[0] - (CX - F * math.tan(az))) > 25.0           # hand-derived: F*(tan30deg - 30deg_rad) = 488.9*0.0538 = 26.3 px, so the pinhole answer is >25 px away: the two models are NOT interchangeable


def test_left_is_left_and_behind_is_invalid():
    cam = G.CylCam.nominal(x_m=0.0, h_m=1.5)
    col, _, ok = cam.project_ground([[20.0, 3.0], [20.0, -3.0], [-5.0, 0.0]])
    assert col[0] < CX < col[1] and ok[0] and ok[1]
    assert not ok[2]


def test_horizon_row_nominal_is_image_centre():
    assert abs(G.CylCam.nominal().horizon_row() - CY) < 1e-9


def test_densify_and_runs():
    d = G.densify([[10.0, 0.0]], step=2.0)
    assert d.shape == (6, 2) and abs(d[-1, 0] - 10.0) < 1e-12
    assert G.runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]


def test_agrees_with_repo_torch_implementation_on_a_real_extrinsic():
    torch = pytest.importorskip("torch")
    try:
        from tanitad.data.calib import CanonicalFrame
        from tanitad.data.rig_projection import RigCamera
    except Exception as e:                               # pragma: no cover
        pytest.skip(f"launch tree not on PYTHONPATH: {e}")

    class E:   # a real-looking extrinsic (quaternion from the banked table's first entry: ~ nominal rotation, pitch ~ -0.17 deg)
        qx, qy, qz, qw = -0.49736369042827033, 0.5011586471611009, -0.5003531383740136, 0.501114863817171
        x, y, z = 2.142972230911255, -0.031627677381038666, 1.61245858669281

        def rotation_cam_to_vehicle(self):
            return G.quat_to_R(self.qx, self.qy, self.qz, self.qw)
    fr = CanonicalFrame(height=416, width=1024, f_ref=F, projection="cylindrical")
    ref = RigCamera.from_extrinsics(E(), fr)
    mine = G.CylCam.from_extrinsics({"qx": E.qx, "qy": E.qy, "qz": E.qz, "qw": E.qw, "x": E.x, "y": E.y, "z": E.z})
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(4, 80, 400), rng.uniform(-25, 25, 400), np.zeros(400)], 1)
    rc, rr, rv = ref.project(torch.as_tensor(pts))
    c, r, v = mine.project(pts)
    assert np.array_equal(rv.numpy(), v)
    assert np.abs(rc.numpy()[v] - c[v]).max() < 1e-6 and np.abs(rr.numpy()[v] - r[v]).max() < 1e-6
    assert v.sum() > 100                                  # the comparison is not vacuous
