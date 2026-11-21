"""refcv6's camera frame and BEV-lift geometry from NavSim — analytic targets + regression arms.
TANITAD VENV:  pytest -q tests/test_frames_lift6.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import frames416 as F4  # noqa: E402
import rig6  # noqa: E402
import torch  # noqa: E402
from tanitad.data.calib import CanonicalFrame, PHYSICALAI_WIDE120_256x640  # noqa: E402
from tanitad.models.trunk_shapes import FRAME_416x1024  # noqa: E402

DFW_BANK = "C:/Users/Admin/tanitad-wt/_s2build/navsim/corpus"
WARMUP_DATA = "C:/Users/Admin/navsim/data/openscene"


# --------------------------------------------------------------------------- #
# the frame                                                                     #
# --------------------------------------------------------------------------- #
def test_frame_is_the_models_literal():
    f = FRAME_416x1024
    assert (f.height, f.width, f.projection) == (416, 1024, "cylindrical")
    assert f.f_ref == pytest.approx(488.92398517830253, abs=1e-12)
    assert f.hfov_deg == pytest.approx(120.0, abs=1e-9)
    assert f.vfov_deg == pytest.approx(46.09213171161337, abs=1e-9)   # the builder manifest


def _col_azimuth_linear(ray: np.ndarray, frame) -> float:
    """max |atan2(x, z) - (u - (W-1)/2) / f_ref| along the middle row."""
    row = ray[frame.height // 2]
    az = np.arctan2(row[:, 0], row[:, 2])
    want = (np.arange(frame.width) - (frame.width - 1) / 2.0) / frame.f_ref
    return float(np.abs(az - want).max())


def test_KG416_configured_rays_are_cylindrical_416():
    F4.configure(FRAME_416x1024)
    assert F4.E2BF.RAY_CAN.shape == (416, 1024, 3)
    assert (F4.E2BF.H, F4.E2BF.W) == (416, 1024)
    # calib.cylindrical_rays builds its grid in float32 (the stitch's own table): the azimuth is
    # linear to ~1e-7 rad = 4e-5 px at f_ref 489 — MEASURED 8.5e-8; the pinhole arm reads > 0.05
    assert _col_azimuth_linear(F4.E2BF.RAY_CAN, FRAME_416x1024) < 1e-6
    # the edge column sits at +-60 deg (120 deg field), LITERAL
    az_edge = math.degrees(math.atan2(F4.E2BF.RAY_CAN[208, -1, 0], F4.E2BF.RAY_CAN[208, -1, 2]))
    assert az_edge == pytest.approx(60.0 * (1023 / 1024), abs=1e-6)


def test_REGRESSION_pinhole_frame_goes_red():
    """The cylindrical-FOV trap: the same (H, W, f_ref) as a PINHOLE frame is not linear in
    azimuth (and would read 92.6 deg of field)."""
    pin = CanonicalFrame(height=416, width=1024, f_ref=FRAME_416x1024.f_ref, projection="pinhole")
    F4.configure(pin)
    try:
        assert _col_azimuth_linear(F4.E2BF.RAY_CAN, pin) > 0.05
        assert pin.hfov_deg == pytest.approx(92.6, abs=0.1)
    finally:
        F4.configure(FRAME_416x1024)


def test_assert_frame_refuses_anything_else():
    with pytest.raises(SystemExit):
        F4.assert_frame(PHYSICALAI_WIDE120_256x640)


@pytest.mark.skipif(not os.path.isdir(DFW_BANK), reason="NO_TREE: DataFlyWheel 256x640 bank absent")
def test_KB256_repointing_reproduces_the_dataflywheel_bank(tmp_path):
    """Configured back to 256x640, the imported stitch must reproduce the DataFlyWheel bank
    BIT-EXACTLY — proof that re-pointing the frame changed nothing else."""
    rc = F4.main(["--split", "warmup_two_stage", "--stage", "2", "--frame", "256x640", "--keep", "4",
                  "--limit", "3", "--data-root", WARMUP_DATA, "--out", str(tmp_path),
                  "--verify-bank", DFW_BANK])
    import json
    rep = json.load(open(tmp_path / "BUILD_REPORT.json", encoding="utf-8"))
    F4.configure(FRAME_416x1024)
    assert rc == 0
    kb = rep["reproduction_control_KB256"]
    assert kb["n_compared"] == 3 and kb["n_bit_exact"] == 3 and kb["pass"] is True


# --------------------------------------------------------------------------- #
# the lift: analytic projection of a road point                                #
# --------------------------------------------------------------------------- #
def _level_rig(cam_z_ego=1.5, cam_x=1.7, pitch_down=0.0):
    """A synthetic rig record: the virtual camera looking straight ahead (optionally pitched
    down), centre at (cam_x, 0, cam_z_ego) in the NavSim ego frame."""
    c, s = math.cos(pitch_down), math.sin(pitch_down)
    fwd = np.array([c, 0.0, -s])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    m = np.stack([right, down, fwd], axis=1)
    return {"rig_key": "synthetic", "f0_sensor2lidar_translation": [cam_x, 0.0, cam_z_ego],
            "virtual_R_cam_to_lidar": m.tolist()}


def test_lift_road_point_projects_to_the_analytic_row():
    """Level camera at height h above the road, road point D m ahead: cam (x, y, z) =
    (0, h, D - cam_x) -> col = (W-1)/2, row = (H-1)/2 + f * h / (D - cam_x)."""
    road_z = -0.35
    rig = _level_rig(cam_z_ego=1.5)
    cam = rig6.rig_camera(rig, road_z)
    h = 1.5 - road_z
    for d in (8.0, 15.0, 30.0):
        col, row, ok = cam.project(torch.tensor([[d, 0.0, 0.0]], dtype=torch.float64))
        f = FRAME_416x1024.f_ref
        assert float(col[0]) == pytest.approx((1024 - 1) / 2.0, abs=1e-9)
        assert float(row[0]) == pytest.approx((416 - 1) / 2.0 + f * h / (d - 1.7), abs=1e-6)
        assert bool(ok[0])


def test_REGRESSION_road_plane_offset_ignored_goes_red():
    """Forgetting that NavSim's ego origin is ~0.35 m ABOVE the road (camera height 1.5 instead
    of 1.85 m) moves a 15 m road point by ~11 rows — the analytic row must not be met."""
    good = rig6.rig_camera(_level_rig(1.5), -0.35)
    bad = rig6.rig_camera(_level_rig(1.5), 0.0)
    p = torch.tensor([[15.0, 0.0, 0.0]], dtype=torch.float64)
    assert abs(float(good.project(p)[1][0]) - float(bad.project(p)[1][0])) > 5.0


def test_lift_geometry_shapes_and_mask():
    g, v, s = rig6.lift_geometry(_level_rig(1.5, pitch_down=math.radians(1.5)), -0.35, stride=16)
    assert tuple(g.shape[:1]) == (4,) and g.shape[-1] == 2 and tuple(v.shape) == tuple(g.shape[:-1])
    assert 0.3 < s["valid_frac"] < 1.0
    assert s["cam_height_m"] == pytest.approx(1.85, abs=1e-9)
    assert s["pitch_down_deg"] == pytest.approx(1.5, abs=1e-6)
    # the bottom 43 rows are UNOBSERVED in the lift (the trainer's mask): no valid sample may
    # land there
    cam = rig6.rig_camera(_level_rig(1.5, pitch_down=math.radians(1.5)), -0.35)
    o = rig6.observed_mask()
    assert int((~o).sum()) == 43 * 1024


def test_road_plane_from_synthetic_log():
    frames = []
    for i in range(10):
        boxes = np.array([[10.0 + i, 0.5, -0.35 + 1.0, 4.5, 1.9, 2.0, 0.0],
                          [20.0, -3.0, -0.35 + 0.8, 4.0, 1.8, 1.6, 0.0],
                          [50.0, 0.0, 5.0, 4.0, 1.8, 1.6, 0.0]])         # out of box: ignored
        frames.append({"lidar2ego_translation": [0, 0, 0], "lidar2ego_rotation": [1, 0, 0, 0],
                       "anns": {"gt_boxes": boxes, "gt_names": np.array(["vehicle"] * 3)}})
    r = rig6.road_plane_from_log(frames)
    assert r["status"] == "OK" and r["road_z_m"] == pytest.approx(-0.35, abs=1e-12)
    assert r["n_boxes"] == 20


def test_road_plane_refuses_non_identity_lidar2ego():
    fr = [{"lidar2ego_translation": [0, 0, 1.8], "lidar2ego_rotation": [1, 0, 0, 0], "anns": {}}]
    with pytest.raises(SystemExit):
        rig6.road_plane_from_log(fr)


def test_virtual_rotation_equals_build_map_M():
    """``frames416.virtual_rotation`` must be the SAME M E2's build_map returns (no second
    derivation drifting from the stitch)."""
    import glob
    pk = sorted(glob.glob(os.path.join(WARMUP_DATA, "warmup_two_stage",
                                       "synthetic_scene_pickles", "*.pkl")))
    if not pk:
        pytest.skip("NO_TREE: warmup synthetic scenes absent")
    sc = F4.E2BF.load(pk[0])
    cd = sc["scene_metadata"] and sc["frames"][-1]["camera_dict"]
    F4.configure(FRAME_416x1024)
    _, _, m = F4.E2BF.build_map(cd, (1080, 1920))
    np.testing.assert_allclose(F4.virtual_rotation(cd), m, atol=0.0)
