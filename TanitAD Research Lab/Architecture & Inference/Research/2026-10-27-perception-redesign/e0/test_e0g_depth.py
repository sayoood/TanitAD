"""Analytic controls + deliberate-regression arm for e0g_depth_targets.py (REFCV9_DESIGN s2.3 / s4.1 E0g).

Run (dev box):  W=$(cygpath -w <tree>); export PYTHONPATH="$W\\stack;$W\\stack\\scripts;$W\\taniteval"; CUDA_VISIBLE_DEVICES=-1 \
                python -m pytest test_e0g_depth.py -q        ($tree must contain stack/tanitad of the tip; the first test asserts it)

The controls (all analytic -- no fitted number anywhere):
  C1  a cuboid at a known range returns that range at its centre pixel                         (tol 1e-3 m; analytic dev 1e-6 * D)
  C2  OFF-CENTRE, 45 deg azimuth: rho == D to 1e-9 and range == D sqrt(1 + v^2) to 1e-9; across the face rho = D / cos(dphi)
  C3  a frame with no box (or only invisible boxes) has ZERO foreground pixels, cells, target mass
  C4  the nearer of two overlapping cuboids wins, in either input order
  C5  behind the camera / outside the 120 deg field: none
  RED the PINHOLE formula in place of the cylindrical one must FAIL C2 (a miss for a narrow box, a finite WRONG range for a wide wall)
"""
import dataclasses
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e0g_depth_targets as E  # noqa: E402

H, W = E.FRAME_H, E.FRAME_W
# one real front-wide extrinsic row (numbers only): small mount pitch, camera 1.25 m up, 1.78 m ahead of the rear axle
REAL_EXTR = dict(qx=0.49568088958923384, qy=-0.5005550637768956, qz=0.5045287735465595, qw=-0.4991951527077742,
                 x=1.779527187347412, y=-0.05956505239009857, z=1.2531490325927734)


def box(cx, cy, cz, l, w, h, yaw=0.0):
    return [cx, cy, cz, l, w, h, yaw]


def known_range_ok(cam, boxes, row, col, expected, tol, kind):
    """The analytic control, as a function: does the cast depth at (row, col) equal `expected` within `tol`?"""
    got = float(E.raycast(boxes, cam, depth_kind=kind).depth[row, col])
    return (np.isfinite(got) and abs(got - expected) <= tol), got


def face_box_at_column(cam, col, D, l=4.0, w=6.0, h=4.0):
    """A box whose NEAR face is a vertical plane perpendicular to the azimuth of pixel column `col` (CYLINDRICAL
    azimuth phi = (col - (W-1)/2) / f), at horizontal distance D from the camera.  Level nominal camera at the rig origin."""
    phi = (col - (W - 1) / 2.0) / cam.f_ref
    dir_rig = np.array([math.cos(phi), -math.sin(phi)])        # camera +x (right) is rig -y
    c = (D + l / 2.0) * dir_rig
    return box(c[0], c[1], float(cam.t_cam_in_rig[2]), l, w, h, math.atan2(dir_rig[1], dir_rig[0])), phi


# --------------------------------------------------------------------------------------------------------------- #
def test_tanitad_is_from_the_tree_under_test_and_frame_is_the_programmes():
    import tanitad
    from tanitad.data import vis1_zbuffer as VZ
    p = Path(tanitad.__file__).resolve()
    assert "scratch_e0g" in str(p).lower() or "stack" in p.parts, p      # printed so the run log shows which tree
    print("tanitad from", p)
    cam = E.nominal_camera()
    assert abs(cam.f_ref - VZ.F) < 1e-9 and (VZ.H, VZ.W) == (H, W)
    # the cylindrical relation gives 120 deg; the pinhole formula applied to the same (W, f_ref) gives 92.6 deg (TRAPS.md)
    assert abs(math.degrees(2 * (W / 2) / cam.f_ref) - 120.0) < 1e-9
    assert abs(math.degrees(2 * math.atan((W / 2) / cam.f_ref)) - 92.6) < 0.1


def test_module_import_is_torch_free():
    code = ("import sys; sys.path.insert(0, %r); import e0g_depth_targets as E; "
            "assert 'torch' not in sys.modules, 'torch imported at module level'; print('ok')" % str(HERE))
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# ---- C1 -------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("D", [5.0, 20.0, 50.0, 99.0])
def test_c1_known_range_at_centre_pixel(D):
    cam = E.nominal_camera(height_m=1.5, x_m=1.5)
    b = [box(1.5 + D + 2.0, 0.0, 1.5, 4.0, 6.0, 4.0)]        # near face at horizontal distance D, on the boresight
    for r, c in ((207, 511), (208, 512)):                     # the 4 pixels around the half-pixel centre (207.5, 511.5)
        ok, got = known_range_ok(cam, b, r, c, D, 1e-3, "range")
        assert ok, (D, r, c, got)
        assert abs(got - D) < 2e-6 * D + 1e-9                 # analytic deviation: D (sqrt(1+v^2) / cos u - 1) ~ 1.0e-6 D


# ---- C2 + RED -------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("col,D", [(895, 20.0), (130, 35.0), (700, 12.0)])
def test_c2_known_range_off_centre_cylindrical(col, D):
    cam = E.nominal_camera(height_m=1.5, x_m=0.0)
    b, phi = face_box_at_column(cam, col, D)
    row = 207
    v = (row - (H - 1) / 2.0) / cam.f_ref
    ok_rho, got_rho = known_range_ok(cam, [b], row, col, D, 1e-9, "rho")
    assert ok_rho, (got_rho, D)                                # rho == D EXACTLY on a face perpendicular to the azimuth
    ok_rng, got_rng = known_range_ok(cam, [b], row, col, D * math.sqrt(1 + v * v), 1e-9, "range")
    assert ok_rng, (got_rng, D)
    # across the face (cylindrical: column linear in azimuth) rho = D / cos(dphi), for every column the face covers
    r = E.raycast([b], cam, depth_kind="rho")
    cols = np.nonzero(np.isfinite(r.depth[row]))[0]
    assert cols.size > 20
    dphi = (cols - col) / cam.f_ref
    assert np.abs(r.depth[row, cols] - D / np.cos(dphi)).max() < 1e-9 * D


def test_red_arm_pinhole_formula_fails_the_off_centre_control():
    """THE DELIBERATE REGRESSION: same pixel grid, same f_ref, pinhole rays in place of the cylindrical ones."""
    cam = E.nominal_camera(height_m=1.5, x_m=0.0)
    pin = dataclasses.replace(cam, projection="pinhole")
    assert pin.f_ref == cam.f_ref and pin.projection == "pinhole"
    # (a) at the CENTRE the two formulas agree -- the control is specific to off-centre
    bc = [box(20.0 + 2.0, 0.0, 1.5, 4.0, 6.0, 4.0)]                # camera at x = 0: near face at 20 m
    assert known_range_ok(cam, bc, 207, 511, 20.0, 1e-3, "range")[0]
    assert known_range_ok(pin, bc, 207, 511, 20.0, 1e-3, "range")[0]
    # (b) off-centre, a narrow box built through the CYLINDRICAL model: the pinhole ray at that column points 38 deg, not 45 deg -> MISS
    b, _ = face_box_at_column(cam, 895, 20.0, w=2.0)
    assert known_range_ok(cam, [b], 207, 895, 20.0, 1e-9, "rho")[0]
    ok_p, got_p = known_range_ok(pin, [b], 207, 895, 20.0, 1e-3, "rho")
    assert not ok_p and not np.isfinite(got_p)
    # (c) off-centre, a wide wall: the pinhole ray HITS but at the wrong range (D / cos(6.8 deg) = 1.007 D)
    bw, _ = face_box_at_column(cam, 895, 30.0, w=80.0)
    assert known_range_ok(cam, [bw], 207, 895, 30.0, 1e-9, "rho")[0]
    ok_w, got_w = known_range_ok(pin, [bw], 207, 895, 30.0, 1e-3, "rho")
    assert not ok_w and np.isfinite(got_w) and abs(got_w - 30.0) > 0.15, got_w


# ---- C3 -------------------------------------------------------------------------------------------------------- #
def test_c3_no_box_gives_zero_foreground():
    cam = E.nominal_camera()
    for boxes in (np.zeros((0, 7)), [box(-20.0, 0.0, 1.5, 4.0, 2.0, 1.6)], [box(10.0, 80.0, 1.5, 4.0, 2.0, 1.6)]):
        dt = E.build_depth_targets(boxes, cam)
        assert (dt.pixel_state == E.PIX_FG).sum() == 0 and not np.isfinite(dt.depth).any() and (dt.box_id == -1).all()
        for s, g in dt.grids.items():
            assert not g.fg.any() and g.target.sum() == 0.0 and g.fg_frac.max() == 0.0
            assert np.all(g.weight == np.float32(E.ALPHA_BG))     # all BG: nothing to ignore, nothing foreground
        assert E.frame_stats(dt)["fg_pixel_share"] == 0.0


# ---- C4 -------------------------------------------------------------------------------------------------------- #
def test_c4_nearer_of_two_overlapping_cuboids_wins():
    cam = E.nominal_camera(height_m=1.5, x_m=0.0)
    near = box(22.0, 0.0, 1.5, 4.0, 3.0, 3.0)                  # near face at 20 m, 3 m wide
    far = box(37.0, 0.0, 1.5, 4.0, 12.0, 3.0)                  # near face at 35 m, 12 m wide (wider than `near`)
    d1 = E.raycast([near, far], cam)
    d2 = E.raycast([far, near], cam)
    assert abs(d1.depth[207, 511] - 20.0) < 1e-3 and d1.box_id[207, 511] == 0
    assert abs(d2.depth[207, 511] - 20.0) < 1e-3 and d2.box_id[207, 511] == 1
    assert np.array_equal(d1.depth, d2.depth)                  # input order does not matter
    # outside `near`'s silhouette but inside `far`'s, the far wall shows
    row = 207
    cols = np.nonzero(np.isfinite(d1.depth[row]))[0]
    vals = d1.depth[row, cols]
    assert vals.min() < 20.01 and vals.max() > 34.9 and cols.size > 0
    assert set(np.unique(d1.box_id[row][cols])) == {0, 1}


# ---- C5 -------------------------------------------------------------------------------------------------------- #
def test_c5_behind_camera_and_outside_the_120_degree_field_give_none():
    cam = E.nominal_camera(height_m=1.5, x_m=0.0)
    az = lambda deg, D: (D * math.cos(math.radians(deg)), -D * math.sin(math.radians(deg)))   # noqa: E731
    behind = box(-25.0, 0.0, 1.5, 4.0, 2.0, 1.6)
    side = box(*az(75.0, 20.0), 1.5, 2.0, 2.0, 1.6)             # 75 deg: outside +-60
    just_out = box(*az(66.0, 25.0), 1.5, 1.0, 1.0, 1.6)          # entirely beyond the 60 deg edge
    for b in (behind, side, just_out):
        r = E.raycast([b], cam)
        assert not np.isfinite(r.depth).any(), b
    # straddling the field edge: only the inside part is cast, and it is cast
    edge = box(*az(60.0, 20.0), 1.5, 2.0, 12.0, 2.0)
    r = E.raycast([edge], cam)
    cols = np.nonzero(np.isfinite(r.depth).any(axis=0))[0]
    assert cols.size > 0 and cols.max() == W - 1               # right frame edge reached (camera +x = rig -y)


# ---- LID / soft ------------------------------------------------------------------------------------------------ #
def test_lid_literals_and_roundtrip():
    bs = 2 * 100.0 / (80 * 81)
    e = E.lid_edges()
    assert abs(E.lid_bin_size() - bs) < 1e-15 and e[0] == 0.0 and abs(e[-1] - 100.0) < 1e-9
    assert abs(e[1] - bs) < 1e-12 and abs(e[2] - 3 * bs) < 1e-12 and abs(e[3] - 6 * bs) < 1e-12      # i (i + 1) / 2 literal
    assert np.allclose(np.diff(e), bs * (np.arange(80) + 1), atol=1e-12)                              # width grows by bs per bin
    d = np.linspace(0.0, 99.999, 5001)
    assert np.abs(E.lid_depth(E.lid_tau(d)) - d).max() < 1e-9
    assert np.allclose(E.lid_tau(e), np.arange(81), atol=1e-9)
    # hard bin = the bin that contains d
    i0, _, w0, _ = E.soft_two_hot(d, soft=False)
    assert (e[i0] <= d + 1e-12).all() and (d < e[i0 + 1] + 1e-12).all() and (w0 == 1.0).all()


def test_soft_two_hot_is_a_distribution_and_preserves_the_expected_bin_coordinate():
    d = np.linspace(0.5, 99.0, 777)
    i0, i1, w0, w1 = E.soft_two_hot(d)
    assert np.allclose(w0 + w1, 1.0) and (w0 >= 0).all() and (w1 >= 0).all() and (i1 == i0 + 1).all()
    tgt = np.zeros((d.size, 80))
    tgt[np.arange(d.size), i0] += w0
    tgt[np.arange(d.size), i1] += w1
    interior = (E.lid_tau(d) > 0.5) & (E.lid_tau(d) < 79.5)
    assert np.abs(E.expected_depth(tgt)[interior] - d[interior]).max() < 1e-9
    assert (tgt.argmax(1) == np.floor(E.lid_tau(d)).astype(int)).mean() > 0.999     # argmax of the two-hot is the containing bin


# ---- targets / weights / grids --------------------------------------------------------------------------------- #
def test_targets_weights_and_grid_shapes():
    cam = E.nominal_camera(height_m=1.5, x_m=0.0)
    dt = E.build_depth_targets([box(22.0, 0.0, 1.5, 4.0, 3.0, 2.0, 0.3)], cam)
    assert dt.grids[8].target.shape == (80, 52, 128) and dt.grids[16].target.shape == (80, 26, 64)
    for s, g in dt.grids.items():
        assert g.fg.any()
        assert set(np.unique(g.weight).tolist()) == {0.0, 0.25, 3.25}      # FG / BG / IGNORE all present
        assert np.all(g.weight[g.fg] == np.float32(3.25))
        assert np.allclose(g.target[:, g.fg].sum(0), 1.0, atol=1e-5)
        assert g.target[:, ~g.fg].sum() == 0.0
        assert np.all(g.fg_frac[g.fg] >= 0.5) and np.all(g.weight[(g.fg_frac > 0) & ~g.fg] == 0.0)
        assert np.abs(E.expected_depth(np.moveaxis(g.target[:, g.fg], 0, -1)) - g.cell_depth[g.fg]).max() < 0.05
    with pytest.raises(ValueError):
        E.build_depth_targets([], cam, strides=(7,))


def test_unobserved_pixels_and_out_of_range_are_ignored_not_background():
    cam = E.nominal_camera(height_m=1.5, x_m=0.0)
    obs = np.ones((H, W), bool)
    obs[:, :300] = False
    dt = E.build_depth_targets([box(22.0, 0.0, 1.5, 4.0, 3.0, 2.0)], cam, observed=obs)
    assert (dt.pixel_state[:, :300] == E.PIX_UNOBS).all()
    assert (dt.grids[8].weight[:, :37] == 0.0).all()                       # fully unobserved cells carry no weight
    far = E.build_depth_targets([box(122.0, 0.0, 1.5, 4.0, 3.0, 2.0)], cam)   # near face at 120 m > d_max
    assert (far.pixel_state == E.PIX_OOR).sum() > 0 and (far.pixel_state == E.PIX_FG).sum() == 0
    assert not far.grids[8].fg.any()


def test_unusable_cuboids_are_counted_not_dropped_silently():
    cam = E.nominal_camera(height_m=1.5, x_m=0.0)
    bad = [box(10, 0, 1.5, 0.0, 2, 2), box(10, np.nan, 1.5, 4, 2, 2), box(0.0, 0.0, 1.5, 4, 4, 4), box(22, 0, 1.5, 4, 3, 2)]
    dt = E.build_depth_targets(bad, cam)
    assert dt.meta["skipped"] == {"invalid": 2, "camera_inside": 1} and (dt.pixel_state == E.PIX_FG).sum() > 0


def test_rho_and_range_agree_on_ordering_and_differ_by_the_ray_tilt():
    cam = E.nominal_camera(height_m=1.5, x_m=0.0)
    b = [box(22.0, 0.0, 1.5, 4.0, 3.0, 4.0), box(40.0, 3.0, 1.5, 4.0, 3.0, 4.0, 0.2)]
    a, r = E.raycast(b, cam, depth_kind="range"), E.raycast(b, cam, depth_kind="rho")
    assert np.array_equal(a.box_id, r.box_id)
    m = np.isfinite(a.depth)
    rows = np.nonzero(m)[0]
    v = (rows - (H - 1) / 2.0) / cam.f_ref
    assert np.allclose(r.depth[m], a.depth[m] / np.sqrt(1 + v * v), rtol=1e-12)


# ---- the prefilter is conservative; agreement with the programme's own z-buffer; geometry round trip ------------- #
def random_scene(seed, n=40):
    rng = np.random.default_rng(seed)
    b = np.c_[rng.uniform(-35, 90, n), rng.uniform(-40, 40, n), rng.uniform(0.5, 1.5, n),
              rng.uniform(1.0, 12.0, n), rng.uniform(0.5, 3.0, n), rng.uniform(1.0, 3.5, n), rng.uniform(-math.pi, math.pi, n)]
    b[0] = box(-8.0, 0.0, 1.0, 4.5, 1.9, 1.6)                    # a car right behind the ego (azimuth ~180 deg: the seam)
    b[1] = box(6.0, 2.0, 1.0, 4.5, 1.9, 1.6, 0.4)                # a close one, far to the side
    b[2] = box(30.0, -50.0, 1.0, 4.0, 2.0, 1.6)                  # beyond the 60 deg edge
    return b


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prefilter_is_conservative_exactly(seed):
    cam = E.camera_from_quat(**REAL_EXTR)
    b = random_scene(seed)
    a, c = E.raycast(b, cam, prefilter=True), E.raycast(b, cam, prefilter=False)
    assert np.array_equal(a.depth, c.depth) and np.array_equal(a.box_id, c.box_id)
    assert np.isfinite(a.depth).sum() > 1000


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nearest_box_identity_agrees_with_the_programmes_vis1_zbuffer(seed):
    from tanitad.data import vis1_zbuffer as VZ
    cam = E.camera_from_quat(**REAL_EXTR)
    b = random_scene(seed)
    R, t = cam.R_cam_to_rig, cam.t_cam_in_rig
    _, _, _, _, idb = VZ.zbuffer(b, R, t, np.ones((H, W), bool))
    ref = idb[VZ.PAD_V:VZ.PAD_V + H, VZ.PAD_U:VZ.PAD_U + W]
    mine = E.raycast(b, cam).box_id
    agree = float((ref == mine).mean())
    both = (ref >= 0) | (mine >= 0)
    jacc = float(((ref == mine) & both).sum() / both.sum())
    print(f"seed {seed}: pixel agreement {agree:.6f}, over object pixels {jacc:.6f}, n obj {int(both.sum())}")
    assert agree > 0.9999 and jacc > 0.999


def test_hit_points_lie_on_the_surface_and_project_back_to_their_own_pixel():
    import torch
    from tanitad.data.rig_projection import project_cam_to_frame
    cam = E.camera_from_quat(**REAL_EXTR)
    b = random_scene(5)
    r = E.raycast(b, cam)
    rr, cc = np.nonzero(np.isfinite(r.depth))
    sel = np.random.default_rng(0).choice(rr.size, size=min(4000, rr.size), replace=False)
    rr, cc = rr[sel], cc[sel]
    d_rig = E.pixel_rays(cam)[rr, cc] @ cam.R_cam_to_rig.T
    p = cam.t_cam_in_rig + r.depth[rr, cc][:, None] * d_rig
    # (1) back through the programme's own projection (the inverse of the ray generator): same pixel
    col, row, valid = project_cam_to_frame(torch.from_numpy((p - cam.t_cam_in_rig) @ cam.R_cam_to_rig), cam.frame)
    assert valid.all() and np.abs(col.numpy() - cc).max() < 1e-6 and np.abs(row.numpy() - rr).max() < 1e-6
    # (2) on the surface of the box the id buffer names: max |local / half| == 1 and never above
    worst, touch = 0.0, []
    for i in np.unique(r.box_id[rr, cc]):
        cx, cy, cz, l, w, h, yaw = b[i]
        q = p[r.box_id[rr, cc] == i] - np.array([cx, cy, cz])
        loc = np.c_[math.cos(yaw) * q[:, 0] + math.sin(yaw) * q[:, 1], -math.sin(yaw) * q[:, 0] + math.cos(yaw) * q[:, 1], q[:, 2]]
        m = np.abs(loc) / np.array([l / 2, w / 2, h / 2])
        worst = max(worst, float(m.max()))
        touch.append(float(np.abs(m.max(1) - 1.0).max()))
    assert worst < 1 + 1e-9 and max(touch) < 1e-9
