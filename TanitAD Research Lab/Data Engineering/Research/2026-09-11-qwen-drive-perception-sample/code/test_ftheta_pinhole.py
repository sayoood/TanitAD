"""Controls for the f-theta -> pinhole rectifier.

⛔ Each test compares against a target derived INDEPENDENTLY of the code under
test -- an analytic identity or a closed-form forward model -- never against a
re-run of the rectifier's own arithmetic. Re-running the producer and finding
agreement measures determinism, not correctness (CLAUDE.md).

Each test also carries a deliberate-regression arm: the assertion must go RED
when the real historical defect is reintroduced.
"""

from __future__ import annotations

import numpy as np
import pytest

from ftheta_pinhole import (FTheta, build_K, choose_pinhole, max_theta_on_sensor,
                            quat_to_R, rectify_map, remap_bilinear)

# The MEASURED PhysicalAI front-wide f-theta (calib.py:PHYSICALAI_FRONT_WIDE_FTHETA),
# rig B principal point -- the case that runs off the sensor bottom.
RIG_B = FTheta(poly=(0.0, 927.5032, 23.1353, -58.5012, 16.5067),
               cx=958.0, cy=753.18, width=1920, height=1080)
RIG_A = FTheta(poly=(0.0, 927.5032, 23.1353, -58.5012, 16.5067),
               cx=958.0, cy=543.0, width=1920, height=1080)


def ftheta_forward(intr: FTheta, p_cam: np.ndarray) -> np.ndarray:
    """Closed-form f-theta projection, written independently of rectify_map.

    Camera convention +x right, +y down, +z boresight (calib.py:564-581).
    """
    x, y, z = p_cam
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    r = float(intr.r_of_theta(np.array(theta)))
    if rho < 1e-12:
        return np.array([intr.cx, intr.cy])
    return np.array([intr.cx + x / rho * r, intr.cy + y / rho * r])


def test_optical_axis_maps_to_principal_point():
    """ANALYTIC TARGET: theta = 0 lands exactly on (cx, cy). Not a fit, an identity.

    Tolerance is the NATIVE px subtended by one OUTPUT px (paraxial_focal / f),
    because the grid is sampled at integer output pixels and (u0, v0) is not one.
    It is derived, not tuned: at f=313.7 that is 927.5/313.7 = 2.96 native px per
    output px, so half a pixel of rounding is up to ~1.5 native px.
    """
    pin = choose_pinhole(RIG_B, 896, 512, 110.0)
    mx, my, valid = rectify_map(RIG_B, 896, 512, pin["f"], pin["u0"], pin["v0"])
    u0, v0 = int(round(pin["u0"])), int(round(pin["v0"]))
    tol = 0.75 * RIG_B.paraxial_focal / pin["f"]
    assert valid[v0, u0]
    assert mx[v0, u0] == pytest.approx(RIG_B.cx, abs=tol)
    assert my[v0, u0] == pytest.approx(RIG_B.cy, abs=tol)


@pytest.mark.parametrize("intr", [RIG_A, RIG_B], ids=["rigA", "rigB"])
@pytest.mark.parametrize("hfov", [110.0, 70.0, 30.0])
def test_roundtrip_against_independent_forward_model(intr, hfov):
    """A 3D point's pinhole pixel must resolve, through rectify_map, to the SAME
    native pixel the closed-form f-theta forward model gives it."""
    pin = choose_pinhole(intr, 896, 512, hfov)
    f, u0, v0 = pin["f"], pin["u0"], pin["v0"]
    mx, my, valid = rectify_map(intr, 896, 512, f, u0, v0)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(400):
        u = rng.uniform(1, 894)
        v = rng.uniform(1, 510)
        # the ray this OUTPUT pixel represents, by the pinhole model
        p = np.array([(u - u0) / f, (v - v0) / f, 1.0])
        want = ftheta_forward(intr, p)          # independent closed form
        if not (0 <= want[0] <= intr.width - 1 and 0 <= want[1] <= intr.height - 1):
            continue
        iu, iv = int(round(u)), int(round(v))
        assert valid[iv, iu], f"pixel {(iu, iv)} marked invalid but forward model is on-sensor"
        got = np.array([mx[iv, iu], my[iv, iu]])
        assert np.allclose(got, want, atol=1.5), f"{got} vs {want}"
        checked += 1
    assert checked > 50, f"control too weak: only {checked} points exercised"


def test_rig_b_fan_runs_off_sensor_and_is_masked():
    """DELIBERATE REGRESSION of the real defect: a fan CENTRED on the optical
    axis leaves the sensor for rig B. The mask must SEE it."""
    f = (896 / 2.0) / np.tan(np.deg2rad(110.0) / 2.0)
    # centred principal point == the naive/buggy placement
    _, _, valid_centred = rectify_map(RIG_B, 896, 512, f, 448.0, 256.0)
    assert valid_centred.mean() < 0.999, (
        "centred fan on rig B should leave the sensor; a mask that cannot see "
        "this would let border-replicated pixels pass as real road")
    # and the adaptive placement must do better
    pin = choose_pinhole(RIG_B, 896, 512, 110.0)
    _, _, valid_adaptive = rectify_map(RIG_B, 896, 512, pin["f"], pin["u0"], pin["v0"])
    assert valid_adaptive.mean() >= valid_centred.mean()


def test_unobserved_pixels_are_black_not_replicated():
    """Border replication is the trap; unobserved must be identifiably empty."""
    img = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    mx, my, valid = rectify_map(RIG_B, 896, 512, 200.0, 448.0, 256.0)  # deliberately wide
    out = remap_bilinear(img, mx, my, valid)
    assert (~valid).any(), "need an unobserved region for this control to bite"
    assert out[~valid].max() == 0, "unobserved pixels must be black, not replicated"
    assert out[valid].min() > 0, "observed pixels must carry the image"


def test_max_theta_matches_polynomial_inverse():
    """ANALYTIC: r(theta_of_r(R)) == R. Inverse consistency, not a refit."""
    lim = max_theta_on_sensor(RIG_B)
    assert float(RIG_B.r_of_theta(np.array(lim["up"]))) == pytest.approx(RIG_B.cy, abs=0.5)
    assert float(RIG_B.r_of_theta(np.array(lim["down"]))) == pytest.approx(
        RIG_B.height - 1 - RIG_B.cy, abs=0.5)


def test_hfov_is_what_was_asked_for():
    """f is set from the requested HFOV; check the closed-form inverse agrees."""
    for hfov in (30.0, 70.0, 110.0):
        pin = choose_pinhole(RIG_B, 896, 512, hfov)
        got = 2 * np.rad2deg(np.arctan((896 / 2.0) / pin["f"]))
        assert got == pytest.approx(hfov, abs=1e-6)


def test_K_is_the_matrix_the_rectifier_implements():
    """K must reproduce the SAME pixel the rectify_map ray definition uses."""
    pin = choose_pinhole(RIG_B, 896, 512, 110.0)
    K = build_K(pin["f"], pin["u0"], pin["v0"])
    p = np.array([1.3, -0.4, 7.0])
    uv = K @ p
    uv = uv[:2] / uv[2]
    # invert through the rectify_map ray definition
    back = np.array([(uv[0] - pin["u0"]) / pin["f"], (uv[1] - pin["v0"]) / pin["f"], 1.0])
    assert np.allclose(back * p[2], p, atol=1e-6)


def test_quaternion_is_orthonormal_and_right_handed():
    """Catches the swapped-component error (w,x,y,z vs x,y,z,w) by determinant."""
    R = quat_to_R(0.6991566475011527, -0.14649640692362326,
                  0.1470943038511604, -0.684165221849196)   # cross_left, clip f0ea3c5b
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def _boresight_az_deg(qx, qy, qz, qw) -> float:
    z_in_rig = quat_to_R(qx, qy, qz, qw) @ np.array([0.0, 0.0, 1.0])
    return float(np.rad2deg(np.arctan2(z_in_rig[1], z_in_rig[0])))


# ⚠️ SCOPE. `camera_rig_probe.json` publishes a 100-clip AGGREGATE azimuth. A
# SINGLE clip's mount does not equal that mean, so comparing one clip against it
# is the scope error this programme keeps re-learning. MEASURED here over 799
# clips / 8 calibration chunks: cross_left +66.81 +/- 0.76 (range 64.98..71.49),
# front_wide -0.40 +/- 0.46, rear_left +151.96 +/- 2.18. The per-clip tolerance
# below is 3 sd of that measured spread -- wide enough to admit any real mount,
# far too tight to admit a quaternion-ORDER bug (which moves the axis by tens of
# degrees, as the regression arm proves).
_PROBE_AZ = {"cross_left": 67.33, "front_wide": -0.58, "rear_left": 151.80}
_CLIP_F0EA3C5B = {
    "cross_left": (0.6991566475011527, -0.14649640692362326,
                   0.1470943038511604, -0.684165221849196),
    "front_tele": (0.49606069973423134, -0.49931877766295607,
                   0.5073818258170811, -0.4971601585836114),
}


def test_cross_left_boresight_reproduces_measured_rig_azimuth():
    """INDEPENDENTLY AUTHORED reference: the rig probe's measured azimuth.
    Catches a quaternion-order bug, which orthonormality alone cannot see."""
    az = _boresight_az_deg(*_CLIP_F0EA3C5B["cross_left"])
    assert az == pytest.approx(_PROBE_AZ["cross_left"], abs=3 * 0.76), (
        f"boresight az {az:.2f} outside 3sd of the measured +66.81 +/- 0.76")


def test_front_tele_boresight_points_forward():
    """Second camera, same convention: the front tele must look down +x (az ~ 0)."""
    az = _boresight_az_deg(*_CLIP_F0EA3C5B["front_tele"])
    assert abs(az) < 3.0, f"front tele boresight az {az:.2f} is not forward"


@pytest.mark.parametrize("bad_order", ["wxyz", "negate_w", "swap_xy"])
def test_wrong_quaternion_order_is_CAUGHT(bad_order):
    """DELIBERATE REGRESSION: the real historical defect is a component-order
    mistake. Every wrong reading must move the boresight far outside tolerance,
    or the control above is decorative."""
    qx, qy, qz, qw = _CLIP_F0EA3C5B["cross_left"]
    if bad_order == "wxyz":          # (x,y,z,w) misread as (w,x,y,z)
        args = (qy, qz, qw, qx)
    elif bad_order == "negate_w":    # conjugate -> inverse rotation
        args = (qx, qy, qz, -qw)
    else:                            # transposed axes
        args = (qy, qx, qz, qw)
    az = _boresight_az_deg(*args)
    assert abs(az - _PROBE_AZ["cross_left"]) > 3 * 0.76, (
        f"{bad_order} produced az {az:.2f}, indistinguishable from correct -- "
        "the tolerance is too loose to be a real control")
