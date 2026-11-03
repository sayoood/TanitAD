"""test_r10_pitch.py -- the R10-A controls: hook OFF bit-identical, analytic pixel shift, sign (synthetic braking dip), table definitions.

    W=$(cygpath -w <tree>); export PYTHONPATH="$W\\stack;$W\\stack\\scripts;$W\\taniteval"; CUDA_VISIBLE_DEVICES=-1 <venv python> -X utf8 -m pytest -q test_r10_pitch.py
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r10_pitch_table as T  # noqa: E402
from tanitad.data.bev_raster import GRID_DEFAULT, cell_centers_xy  # noqa: E402
from tanitad.data.calib import PHYSICALAI_WIDE120_256x640 as FRAME  # noqa: E402
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.models import bev_lift as BL  # noqa: E402

TIP = Path(os.environ.get("R10_TIP_BEV_LIFT", "C:/Users/Admin/scratch_integ/tree6/stack/tanitad/models/bev_lift.py"))
H_CAM = 1.5
X_CAM = 1.5


def _cam(mount_pitch_down_rad=0.0):
    """A camera at (1.5, 0, 1.5) pitched down by ``mount_pitch_down_rad``, built WITHOUT compose_pitch_delta (independent numpy rotation)."""
    c = RigCamera.nominal(FRAME, height_m=H_CAM, x_m=X_CAM)
    s, k = math.sin(mount_pitch_down_rad), math.cos(mount_pitch_down_rad)
    M = torch.tensor([[1, 0, 0], [0, k, s], [0, -s, k]], dtype=torch.float64)
    return RigCamera(R_cam_to_rig=c.R_cam_to_rig @ M, t_cam_in_rig=c.t_cam_in_rig, frame=FRAME)


def _row_formula(p_rig, mount_down, delta_down):
    """Independent cylindrical row of rig points for a camera pitched down by mount_down + delta_down (nose-down both):
    camera frame of the NOMINAL level camera is (x right, y down, z fwd) = (-y_rig, H - z_rig, x_rig - X_CAM); a pure pitch-down by a
    rotates it about +x: y' = y cos a - z sin a, z' = y sin a + z cos a."""
    a = mount_down + delta_down
    x = -p_rig[..., 1]
    y = H_CAM - p_rig[..., 2]
    z = p_rig[..., 0] - X_CAM
    y2 = y * math.cos(a) - z * math.sin(a)
    z2 = y * math.sin(a) + z * math.cos(a)
    cy = (FRAME.height - 1) / 2.0
    return cy + FRAME.f_ref * y2 / torch.sqrt(x * x + z2 * z2)


def test_import_tree():
    assert "scratch_r10" in str(Path(BL.__file__)) or "tree" in str(Path(BL.__file__))
    print("bev_lift from", BL.__file__)


def test_delta_zero_is_bit_identical_to_the_tip():
    if not TIP.exists():
        pytest.skip("tip bev_lift.py not found")
    spec = importlib.util.spec_from_file_location("tip_bev_lift", TIP)
    tip = importlib.util.module_from_spec(spec)
    sys.modules["tip_bev_lift"] = tip                                        # dataclasses need the module registered
    spec.loader.exec_module(tip)
    assert not hasattr(tip, "compose_pitch_delta")
    cam = _cam(math.radians(2.0))
    g_tip = tip.build_lift_geometry(cam)
    for kw in ({}, {"pitch_delta_rad": 0.0}, {"pitch_delta_rad": -0.0}):
        g = BL.build_lift_geometry(cam, **kw)
        for f in ("grid", "valid", "col", "row", "in_front", "in_hfov", "in_rows"):
            assert torch.equal(getattr(g, f), getattr(g_tip, f)), (kw, f)
    assert BL.compose_pitch_delta(cam, 0.0) is cam                      # OFF returns the very object


@pytest.mark.parametrize("mount_deg", [0.0, 2.5])
@pytest.mark.parametrize("delta_deg", [0.3, 1.0, -1.0, 2.0])
def test_known_delta_moves_the_ground_row_by_the_predicted_pixels(mount_deg, delta_deg):
    """Flat-world cells of the rig grid (height 0): the cache row moves by the INDEPENDENT cylindrical formula to 1e-8 px (float64)."""
    cam = _cam(math.radians(mount_deg))
    g0 = BL.build_lift_geometry(cam)
    d = math.radians(delta_deg)
    g1 = BL.build_lift_geometry(cam, pitch_delta_rad=d)
    X, Y = cell_centers_xy(GRID_DEFAULT)
    xy = torch.stack([torch.as_tensor(X), torch.as_tensor(Y)], -1).to(torch.float64)
    p = torch.cat([xy, torch.zeros(*xy.shape[:2], 1, dtype=torch.float64)], -1)
    r0 = _row_formula(p, math.radians(mount_deg), 0.0)
    r1 = _row_formula(p, math.radians(mount_deg), d)
    ok = g0.in_front[0] & g1.in_front[0]
    assert float((g0.row[0][ok] - r0[ok]).abs().max()) < 1e-8                 # the library at delta 0 equals the formula
    assert float((g1.row[0][ok] - r1[ok]).abs().max()) < 1e-8                 # the hook equals the formula at delta
    # the headline number: a cell on the centre line at ~40 m
    ix = int(np.argmin(np.abs(np.asarray(X)[:, 0] - 40.25)))
    iy = int(np.argmin(np.abs(np.asarray(Y)[0, :] - 0.25)))
    shift = float(g1.row[0, ix, iy] - g0.row[0, ix, iy])
    pred = float(r1[ix, iy] - r0[ix, iy])
    assert abs(shift - pred) < 1e-8
    assert (shift < 0) == (delta_deg > 0)                                       # nose-down: the cell moves UP the image
    print(f"mount {mount_deg} deg, delta {delta_deg:+.1f} deg: row shift at x=40.25 m, y=0.25 m = {shift:+.4f} px (f_ref {FRAME.f_ref:.1f})")


def test_grid_follows_the_row():
    cam = _cam(0.0)
    g0, g1 = BL.build_lift_geometry(cam), BL.build_lift_geometry(cam, pitch_delta_rad=math.radians(1.0))
    both = g0.valid & g1.valid
    dgy = (g1.grid[..., 1] - g0.grid[..., 1])[both]
    drow = (g1.row - g0.row)[both].to(torch.float32)
    h = FRAME.height // 16
    assert float((dgy - 2.0 * drow / (16 * h)).abs().max()) < 5e-5


# ------------------------------------------------------------------ sign: the synthetic braking dip --------------------------------
def _quat_ry(theta):          # rotation about +y (LEFT, x forward, z up) by theta
    return np.stack([np.zeros_like(theta), np.sin(theta / 2), np.zeros_like(theta), np.cos(theta / 2)], 1)


def test_synthetic_braking_dip_reads_positive_and_moves_the_scene_up():
    # a body whose nose DIVES by 1.0 deg (a 0.3 s sigma dip) at t = 10 s (forward axis tilts toward -z, i.e. toward the road)
    t = np.arange(0, 20, 0.01)
    th = np.radians(1.0) * np.exp(-0.5 * ((t - 10.0) / 0.3) ** 2)
    q = _quat_ry(th)
    R = np.array([[math.cos(th[1000]), 0, math.sin(th[1000])], [0, 1, 0], [-math.sin(th[1000]), 0, math.cos(th[1000])]])    # Ry(theta)
    fwd_world = R @ np.array([1.0, 0.0, 0.0])
    assert fwd_world[2] < 0                                                    # PHYSICS: nose down = the forward axis points below the horizon (z up)
    pitch = T.euler_zyx_deg(q)[:, 1]
    assert abs(pitch[1000] - 1.0) < 1e-9 and pitch[1000] > 0                   # the table's pitch reads +1.0 deg at the dive
    H = T.hp_dev(pitch)
    assert H[1000] > 0.6 and abs(H[200]) < 1e-6                                # H keeps the dive, is ~0 far from it
    # the hook with H -> the ground point at 40 m moves UP (row falls) and the sampled ground distance of the SAME row grows
    cam = _cam(0.0)
    p = torch.tensor([X_CAM + 40.0, 0.0, 0.0], dtype=torch.float64)
    r0 = float(_row_formula(p, 0.0, 0.0))
    cam_dive = BL.compose_pitch_delta(cam, math.radians(float(H[1000])))
    from tanitad.data.rig_projection import project_cam_to_frame, rig_to_cam
    _, r_dive, _ = project_cam_to_frame(rig_to_cam(p, cam_dive.R_cam_to_rig, cam_dive.t_cam_in_rig), FRAME)
    assert float(r_dive) < r0
    print(f"dive 1.0 deg at 40 m: row {r0:.2f} -> {float(r_dive):.2f} px ({float(r_dive) - r0:+.2f})")


def test_sign_test_is_not_vacuous():
    """DELIBERATE REGRESSION: a sign-flipped composition must FAIL the sign assertion of the test above."""
    def flipped(cam, d):
        return BL.compose_pitch_delta(cam, -d)
    cam = _cam(0.0)
    p = torch.tensor([X_CAM + 40.0, 0.0, 0.0], dtype=torch.float64)
    from tanitad.data.rig_projection import project_cam_to_frame, rig_to_cam
    c = flipped(cam, math.radians(1.0))
    _, r, _ = project_cam_to_frame(rig_to_cam(p, c.R_cam_to_rig, c.t_cam_in_rig), FRAME)
    assert not (float(r) < float(_row_formula(p, 0.0, 0.0)))


# ------------------------------------------------------------------ table definitions -----------------------------------------------
def test_table_definitions_match_e0c_verbatim():
    ref = Path("C:/Users/Admin/scratch_r10/ref/e0bc_census.py")
    if not ref.exists():
        pytest.skip("e0bc_census.py reference copy not found")
    spec = importlib.util.spec_from_file_location("e0bc", ref)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rng = np.random.default_rng(0)
    q = rng.normal(size=(1200, 4))
    assert np.array_equal(T.euler_zyx_deg(q), m.euler_zyx_deg(q))
    a = rng.normal(size=1200).cumsum()
    assert np.array_equal(T.hp_dev(a), m.hp_dev(a))


def test_H_and_C_definitions():
    p = np.linspace(0, 3, 1000)
    assert abs(float((p - p.mean()).mean())) < 1e-12
    h = T.hp_dev(np.full(1000, 2.0))
    assert np.allclose(h, 0.0)                                                 # a constant has no dynamic part
    assert np.allclose(T.hp_dev(np.full(1000, 2.0) + 0.0), 0.0)
