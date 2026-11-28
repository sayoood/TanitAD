"""The camera projection is checked against ANALYTIC identities and against the frame bank's OWN ray
model — never against itself.

1. **boresight identity** — a point on the virtual camera's boresight lands EXACTLY on the frame
   centre ((W−1)/2, (H−1)/2). An analytic target, not a fit.
2. **round trip against `tanitad.data.calib.cylindrical_rays`** — the ray the BANK BUILDER used for
   pixel (u, v), pushed out into the world and projected back, returns (u, v). The reference is the
   builder's own function (`build_navsim_eval.py` line-for-line), so this is a cross-check against an
   independently authored derivation, not a re-run of ours.
3. **deliberate regression** — the pinhole formula (the trap CLAUDE.md records: it reads 92.6° on a
   120° cylindrical rig) and a mis-ordered basis must BOTH break identity 1 or 2. A check that cannot
   fail is not a check.
4. **FOV** — the frame's own half-FOV is 60° (120° rig), which is what makes the column linear in azimuth.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from taniteval.benchreport import camproj

H, W, F = 256, 640, 305.5774907364391          # PHYSICALAI_WIDE120_256x640 (the bank's BUILD.json)
# cam_f0 of a real warmup rig (E2's scene 4df69dee983b5725e; OpenScene lidar2ego is the identity)
R_F0 = np.array([[0.0332, -0.0402, 0.9986], [-0.9994, 0.0008, 0.0333], [-0.0021, -0.9992, -0.0401]])
T_F0 = np.array([1.6557, -0.0111, 1.5203])


def test_the_frame_is_120_degrees_because_the_column_is_linear_in_azimuth():
    half_fov = (W - 1) / 2.0 / F
    assert math.degrees(2 * half_fov) == pytest.approx(119.8, abs=0.3)      # the rig's own 120 deg
    pinhole = math.degrees(2 * math.atan((W / 2) / F))                       # the documented trap
    assert pinhole == pytest.approx(92.6, abs=0.3)


def test_boresight_projects_to_the_exact_frame_centre():
    M = camproj.canonical_basis(R_F0)
    fwd = M[:, 2]
    for dist in (5.0, 20.0, 80.0):
        p = T_F0 + fwd * dist
        u, v, ok = camproj.project(p.reshape(1, 3), M, T_F0, H, W, F)
        assert bool(ok[0])
        assert u[0] == pytest.approx((W - 1) / 2.0, abs=1e-9)
        assert v[0] == pytest.approx((H - 1) / 2.0, abs=1e-9)


def _bank_rays(us, vs):
    """The BANK BUILDER's own ray model (tanitad.data.calib.cylindrical_rays), evaluated at (u, v)."""
    from tanitad.data.calib import PHYSICALAI_WIDE120_256x640 as FRAME, cylindrical_rays
    assert (FRAME.height, FRAME.width) == (H, W) and FRAME.f_ref == pytest.approx(F)
    assert FRAME.projection == "cylindrical"
    xc, yc, zc = (t.double().numpy() for t in cylindrical_rays(FRAME))
    return np.stack([xc[vs, us], yc[vs, us], zc[vs, us]], axis=-1)


def test_round_trip_against_the_bank_builders_own_ray_model():
    """⚠️ ``cylindrical_rays`` returns FLOAT32 (that is what the bank was built with), so the residual
    here is the reference's own dtype: MEASURED 6.3e-6 px. The same identity in float64 closes to
    2.8e-14 px (next test) — i.e. the model is exact and this bound is the reference's precision."""
    M = camproj.canonical_basis(R_F0)
    us = np.array([40, 160, 319, 320, 480, 600])
    vs = np.array([60, 100, 127, 128, 170, 200])
    rays_can = _bank_rays(us, vs)
    rays_can /= np.linalg.norm(rays_can, axis=-1, keepdims=True)
    for t in (3.0, 25.0):
        pts = T_F0.reshape(1, 3) + (rays_can @ M.T) * t
        u2, v2, ok = camproj.project(pts, M, T_F0, H, W, F)
        assert ok.all()
        assert np.abs(u2 - us).max() < 1e-4
        assert np.abs(v2 - vs).max() < 1e-4


def test_the_round_trip_is_exact_in_float64():
    M = camproj.canonical_basis(R_F0)
    us = np.array([40, 160, 319, 480, 600])
    vs = np.array([60, 100, 127, 170, 200])
    phi = (us - (W - 1) / 2.0) / F                       # the same formula, in double precision
    yn = (vs - (H - 1) / 2.0) / F
    rays = np.stack([np.sin(phi), yn, np.cos(phi)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    pts = T_F0.reshape(1, 3) + (rays @ M.T) * 25.0
    u2, v2, ok = camproj.project(pts, M, T_F0, H, W, F)
    assert ok.all() and np.abs(u2 - us).max() < 1e-9 and np.abs(v2 - vs).max() < 1e-9


def test_ray_of_pixel_is_the_inverse_of_project():
    M = camproj.canonical_basis(R_F0)
    us, vs = np.array([12.0, 320.0, 610.0]), np.array([30.0, 128.0, 240.0])
    pts = T_F0.reshape(1, 3) + camproj.ray_of_pixel(us, vs, M, H, W, F) * 17.0
    u2, v2, ok = camproj.project(pts, M, T_F0, H, W, F)
    assert ok.all() and np.abs(u2 - us).max() < 1e-9 and np.abs(v2 - vs).max() < 1e-9


def test_a_pinhole_projection_FAILS_the_same_identities():
    """Deliberate regression: the trap formula must not pass the round trip."""
    M = camproj.canonical_basis(R_F0)
    us = np.array([40, 600])
    vs = np.array([60, 200])
    rays = _bank_rays(us, vs)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    pts = T_F0.reshape(1, 3) + (rays @ M.T) * 20.0
    can = (pts - T_F0.reshape(1, 3)) @ M
    u_pin = can[:, 0] / can[:, 2] * F + (W - 1) / 2.0        # pinhole instead of cylindrical
    assert np.abs(u_pin - us).max() > 5.0, "the pinhole formula must disagree — it reads a 92.6 deg FOV"


def test_a_mis_ordered_basis_moves_an_OFF_AXIS_point():
    """⚠️ The regression arm must be OFF-AXIS: a point ON the boresight has canonical x = y = 0, so
    swapping right <-> down leaves it at the centre — a check that cannot fail (MEASURED while writing
    this test: the first version asserted exactly that and was green against the broken basis)."""
    M = camproj.canonical_basis(R_F0)
    on_axis = (T_F0 + M[:, 2] * 20.0).reshape(1, 3)
    u_ok, v_ok, _ = camproj.project(on_axis, M, T_F0, H, W, F)
    u_bad, v_bad, _ = camproj.project(on_axis, M[:, [1, 0, 2]], T_F0, H, W, F)
    assert (u_ok[0], v_ok[0]) == (u_bad[0], v_bad[0])        # the blind spot, stated
    off_axis = (T_F0 + M[:, 2] * 20.0 + M[:, 0] * 5.0).reshape(1, 3)
    u1, v1, _ = camproj.project(off_axis, M, T_F0, H, W, F)
    u2, v2, _ = camproj.project(off_axis, M[:, [1, 0, 2]], T_F0, H, W, F)
    assert abs(u1[0] - u2[0]) > 50.0 and abs(v1[0] - v2[0]) > 50.0


def test_points_outside_the_frames_own_fov_are_marked_invalid():
    M = camproj.canonical_basis(R_F0)
    behind = (T_F0 - M[:, 2] * 10.0).reshape(1, 3)
    _u, _v, ok = camproj.project(behind, M, T_F0, H, W, F)
    assert not bool(ok[0])
    side = (T_F0 + M[:, 0] * 10.0).reshape(1, 3)             # 90 deg to the right: outside 60 deg
    _u, _v, ok2 = camproj.project(side, M, T_F0, H, W, F)
    assert not bool(ok2[0])


def test_densify_keeps_the_endpoints_and_adds_the_origin():
    poses = np.array([[5.0, 0.1], [11.0, -0.2]])
    d = camproj.densify(poses, n_per_segment=8)
    assert d.shape[1] == 3 and (d[:, 2] == 0).all()
    assert np.allclose(d[0], [0.0, 0.0, 0.0])
    assert np.allclose(d[-1], [11.0, -0.2, 0.0])
    assert len(d) == 8 * 2 + 1
