"""Tests for `tanitad.data.rig_projection` — the rig <-> canonical-image bridge.

⭐ THE TEST THE BRIEF ASKS FOR IS :func:`test_wrong_extrinsic_makes_a_parked_car_move`.
A parked car is world-STATIC by construction. Round-tripping it through the
projection with the CORRECT extrinsic must recover a constant world position to
machine precision; with a WRONG extrinsic the recovered position MOVES. That is
a control which must read a known value (zero motion), and it is the only thing
standing between "our agent tokens are grounded" and "our agent tokens are
grounded in the wrong place".
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tanitad.data.calib import CanonicalFrame                      # noqa: E402
from tanitad.data.rig_projection import (                          # noqa: E402
    CAM_HEIGHT_RANGE_M, ROAD_PLANE_Z_M, RigCamera, cam_to_rig,
    frame_to_cam_ray, ground_intersection, project_cam_to_frame, rig_to_cam,
)

# The corpus frame: physicalai-b1-w120-256x640cyl.
CYL = CanonicalFrame(height=256, width=640, f_ref=305.577,
                     projection="cylindrical")


# --------------------------------------------------------------------------- #
# The FOV trap, pinned                                                         #
# --------------------------------------------------------------------------- #

def test_corpus_frame_is_120_degrees_and_the_pinhole_formula_is_wrong():
    """CONTROL THAT MUST READ A KNOWN VALUE: the rig's own published FOV.

    `camera_front_wide_120fov` is the sensor's NAME, so 120.0 deg is a value we
    know independently of our own code. The cylindrical frame reproduces it; the
    pinhole formula on the same (W, f_ref) reads 92.6 deg and looks fine.
    """
    assert CYL.hfov_deg == pytest.approx(120.0, abs=0.05)
    pin = CanonicalFrame(height=256, width=640, f_ref=305.577,
                         projection="pinhole")
    assert pin.hfov_deg == pytest.approx(92.6, abs=0.2)
    assert abs(pin.hfov_deg - CYL.hfov_deg) > 25.0


def test_projection_matches_calib_cylindrical_rays_exactly():
    """The forward projection must invert `calib.cylindrical_rays` on the grid.

    Not a smoke test: `cylindrical_rays` is what BUILT the pixels (it makes the
    grid_sample grid in `cylindrical_rectify`), so agreeing with it is the whole
    correctness claim. Disagreement of even half a pixel means the detector's
    targets sit off the object.
    """
    from tanitad.data.calib import cylindrical_rays
    for frame in (CYL, CanonicalFrame(height=64, width=96, f_ref=50.0,
                                      projection="pinhole")):
        gy, gx = torch.meshgrid(
            torch.arange(frame.height, dtype=torch.float64),
            torch.arange(frame.width, dtype=torch.float64), indexing="ij")
        # (a) EXACTNESS, in the dtype the claim is about. `cylindrical_rays`
        #     builds its grid in float32; recomputing the identical expression
        #     in float64 isolates the GEOMETRY from the dtype.
        u = (gx - (frame.width - 1) / 2.0) / float(frame.f_ref)
        v = (gy - (frame.height - 1) / 2.0) / float(frame.f_ref)
        if frame.projection == "cylindrical":
            p64 = torch.stack([torch.sin(u), v, torch.cos(u)], dim=-1)
        else:
            p64 = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        col, row, valid = project_cam_to_frame(p64, frame)
        assert valid.all(), f"{frame.tag()}: own grid must be fully in-frame"
        assert float((col - gx).abs().max()) < 1e-12, frame.tag()
        assert float((row - gy).abs().max()) < 1e-12, frame.tag()

        # (b) AGREEMENT with the SHIPPED float32 helper, and its MEASURED
        #     residual stated rather than hidden: 1.83e-05 px on the corpus
        #     frame (max over all 256x640 pixels), i.e. ~1/50000 of a pixel.
        #     A regression that silently switched a formula would blow past
        #     this by orders of magnitude.
        x, y, z = cylindrical_rays(frame)
        c32, r32, _ = project_cam_to_frame(
            torch.stack([x, y, z], dim=-1).to(torch.float64), frame)
        assert float((c32 - gx).abs().max()) < 1e-4, frame.tag()
        assert float((r32 - gy).abs().max()) < 1e-4, frame.tag()


def test_frame_to_cam_ray_round_trips():
    cols = torch.tensor([0.0, 123.5, 319.5, 639.0], dtype=torch.float64)
    rows = torch.tensor([0.0, 60.25, 127.5, 255.0], dtype=torch.float64)
    d = frame_to_cam_ray(cols, rows, CYL)
    c2, r2, _ = project_cam_to_frame(d, CYL)
    assert torch.allclose(c2, cols, atol=1e-9)
    assert torch.allclose(r2, rows, atol=1e-9)


# --------------------------------------------------------------------------- #
# Hand-computable geometry                                                     #
# --------------------------------------------------------------------------- #

def test_straight_ahead_lands_on_the_boresight_column():
    cam = RigCamera.nominal(CYL, height_m=1.5, x_m=1.5, y_m=0.0)
    p = torch.tensor([[20.0, 0.0, 1.5]], dtype=torch.float64)   # at cam height
    col, row, valid = cam.project(p)
    assert bool(valid[0])
    assert float(col[0]) == pytest.approx((CYL.width - 1) / 2.0, abs=1e-9)
    assert float(row[0]) == pytest.approx((CYL.height - 1) / 2.0, abs=1e-9)


def test_azimuth_is_linear_in_column_on_a_cylinder():
    """A 60 deg bearing lands exactly at the frame edge; 30 deg exactly halfway.

    Linearity in azimuth IS the cylindrical projection; on a pinhole frame the
    30 deg point would sit at 0.577/1.732 = 33.3 % of the half-width, not 50 %.
    """
    cam = RigCamera.nominal(CYL, height_m=1.5, x_m=0.0, y_m=0.0)
    cx = (CYL.width - 1) / 2.0
    for deg, frac in ((60.0, 1.0), (30.0, 0.5), (15.0, 0.25)):
        a = math.radians(deg)
        # bearing +a to the LEFT (+y) => negative camera x => column left of cx
        p = torch.tensor([[100.0 * math.cos(a), 100.0 * math.sin(a), 1.5]],
                         dtype=torch.float64)
        col, _, _ = cam.project(p)
        assert float(col[0]) == pytest.approx(cx - CYL.f_ref * a, abs=1e-6), deg
        del frac
        # ...and the half-width fraction is exactly `deg / 60`
        assert (cx - float(col[0])) / (CYL.width / 2.0) == pytest.approx(
            deg / 60.0, rel=2e-3)


def test_rig_to_cam_round_trips_and_uses_the_measured_axis_convention():
    cam = RigCamera.nominal(CYL, height_m=1.5, x_m=1.5, y_m=0.0)
    p = torch.tensor([[10.0, 3.0, 0.0], [-4.0, -2.0, 2.0]], dtype=torch.float64)
    c = rig_to_cam(p, cam.R_cam_to_rig, cam.t_cam_in_rig)
    # +y rig is LEFT => negative camera x (camera x is RIGHT)
    assert float(c[0, 0]) == pytest.approx(-3.0)
    # +z rig is UP => negative camera y (camera y is DOWN); minus mount height
    assert float(c[0, 1]) == pytest.approx(1.5)
    # +x rig is FORWARD => camera z; minus the mount's forward offset
    assert float(c[0, 2]) == pytest.approx(8.5)
    back = cam_to_rig(c, cam.R_cam_to_rig, cam.t_cam_in_rig)
    assert torch.allclose(back, p, atol=1e-12)


def test_ground_intersection_recovers_a_known_road_point():
    """rig z = 0 IS the road plane (MEASURED over 87,481 cuboids)."""
    cam = RigCamera.nominal(CYL, height_m=1.5, x_m=0.0, y_m=0.0)
    truth = torch.tensor([[25.0, -2.0, ROAD_PLANE_Z_M],
                          [8.0, 1.0, ROAD_PLANE_Z_M]], dtype=torch.float64)
    col, row, valid = cam.project(truth)
    assert bool(valid.all())
    got, hits = ground_intersection(col, row, cam)
    assert bool(hits.all())
    assert torch.allclose(got, truth, atol=1e-9)


def test_a_ray_above_the_horizon_has_no_ground_intersection():
    """A control that must read a KNOWN NEGATIVE: no silent 400 m agents."""
    cam = RigCamera.nominal(CYL, height_m=1.5)
    col = torch.tensor([(CYL.width - 1) / 2.0], dtype=torch.float64)
    row = torch.tensor([10.0], dtype=torch.float64)          # well above centre
    _, hits = ground_intersection(col, row, cam)
    assert not bool(hits.any())


def test_measured_camera_height_bracket_is_carried_not_guessed():
    lo, hi = CAM_HEIGHT_RANGE_M
    assert lo < 1.5 < hi


# --------------------------------------------------------------------------- #
# ⭐ THE PARKED-CAR CONTROL                                                     #
# --------------------------------------------------------------------------- #

def _ego_poses(n: int = 5, v: float = 10.0, dt: float = 0.5, yaw_rate: float = 0.0):
    """Ego (x, y, yaw) in the WORLD frame along a constant-speed arc."""
    out = []
    x = y = yaw = 0.0
    for _ in range(n):
        out.append((x, y, yaw))
        x += v * dt * math.cos(yaw)
        y += v * dt * math.sin(yaw)
        yaw += yaw_rate * dt
    return out


def _world_to_rig(px: float, py: float, ego) -> torch.Tensor:
    """World point -> rig frame, exactly `refb_labels.ego_frame` (+x fwd, +y left)."""
    ex, ey, eyaw = ego
    dx, dy = px - ex, py - ey
    c, s = math.cos(-eyaw), math.sin(-eyaw)
    return torch.tensor([dx * c - dy * s, dx * s + dy * c, 0.0],
                        dtype=torch.float64)


def _recover_world_positions(cam_used: RigCamera, cam_truth: RigCamera,
                             px: float, py: float, poses):
    """Project a parked car with ``cam_truth``, back-project with ``cam_used``.

    Returns the WORLD (x, y) recovered at each pose. When the two cameras agree
    the answer is the parked car's true, CONSTANT position; when they disagree
    the car appears to move — which is the whole point.
    """
    got = []
    for ego in poses:
        p_rig = _world_to_rig(px, py, ego).unsqueeze(0)
        col, row, valid = cam_truth.project(p_rig)
        assert bool(valid[0]), "the parked car must be in frame for the test"
        rec, hits = ground_intersection(col, row, cam_used)
        assert bool(hits[0])
        ex, ey, eyaw = ego
        c, s = math.cos(eyaw), math.sin(eyaw)
        rx, ry = float(rec[0, 0]), float(rec[0, 1])
        got.append((ex + rx * c - ry * s, ey + rx * s + ry * c))
    return got


def test_correct_extrinsic_keeps_a_parked_car_parked():
    frame = CYL
    cam = RigCamera.from_extrinsics(_Extr(0.0, 0.0, 0.0), frame)
    poses = _ego_poses(n=5, v=10.0, dt=0.5)
    got = _recover_world_positions(cam, cam, 45.0, 3.0, poses)
    xs = [g[0] for g in got]
    ys = [g[1] for g in got]
    assert max(xs) - min(xs) < 1e-6, xs
    assert max(ys) - min(ys) < 1e-6, ys
    assert xs[0] == pytest.approx(45.0, abs=1e-6)
    assert ys[0] == pytest.approx(3.0, abs=1e-6)


@pytest.mark.parametrize("kind,val,floor_m", [
    ("yaw", math.radians(3.0), 1.0),
    ("pitch", math.radians(2.0), 1.0),
    ("height", 0.25, 1.0),
])
def test_wrong_extrinsic_makes_a_parked_car_move(kind, val, floor_m):
    """⭐ THE DELIBERATE REGRESSION. A wrong extrinsic MUST be visible.

    Without this the projection could be silently wrong and every downstream
    number would still look plausible — the failure mode that produced a route
    metric of 1.0000 measuring nothing. The car is world-static; if the recovered
    world position wanders by more than a metre, the instrument can see a wrong
    extrinsic, which is what makes a null result interpretable.
    """
    frame = CYL
    truth = RigCamera.from_extrinsics(_Extr(0.0, 0.0, 0.0), frame)
    if kind == "height":
        used = RigCamera.from_extrinsics(_Extr(0.0, 0.0, 0.0, z=1.5 + val), frame)
    else:
        used = RigCamera.from_extrinsics(
            _Extr(val if kind == "yaw" else 0.0,
                  val if kind == "pitch" else 0.0, 0.0), frame)
    poses = _ego_poses(n=5, v=10.0, dt=0.5)
    got = _recover_world_positions(used, truth, 45.0, 3.0, poses)
    xs = [g[0] for g in got]
    ys = [g[1] for g in got]
    drift = max(max(xs) - min(xs), max(ys) - min(ys))
    assert drift > floor_m, (
        f"a {kind} error of {val} moved the parked car only {drift:.3f} m — "
        f"the instrument cannot see a wrong extrinsic")


class _Extr:
    """Minimal `physicalai.FrontWideExtrinsics` stand-in (same duck type).

    Built from mount (yaw, pitch, roll) about the RIG axes so a test can state
    an error in degrees. ``rotation_cam_to_vehicle`` returns R_cam->rig.
    """

    def __init__(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                 x: float = 1.5, y: float = 0.0, z: float = 1.5):
        self.yaw, self.pitch, self.roll = yaw, pitch, roll
        self.x, self.y, self.z = x, y, z

    def rotation_cam_to_vehicle(self):
        import numpy as np
        base = np.array([[0.0, -1.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [1.0, 0.0, 0.0]]).T          # nominal cam -> rig
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        return Rz @ Ry @ Rx @ base


def test_refuses_a_crop_frame():
    """An `ftheta_crop` build keeps the radial warp; refusing beats guessing."""
    bad = CanonicalFrame(height=256, width=256, f_ref=266.0, projection="pinhole")
    object.__setattr__(bad, "projection", "ftheta_crop")
    with pytest.raises(ValueError, match="ftheta"):
        project_cam_to_frame(torch.zeros(1, 3, dtype=torch.float64), bad)


def test_refuses_a_non_rotation():
    with pytest.raises(ValueError, match="not a rotation"):
        RigCamera(R_cam_to_rig=torch.eye(3, dtype=torch.float64) * 2.0,
                  t_cam_in_rig=torch.zeros(3, dtype=torch.float64), frame=CYL)
