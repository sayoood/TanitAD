"""Tests for ``tanitad.models.bev_lift`` -- the refcv6-grounded BEV lift.

(b) geometry against ANALYTIC targets written as literals (hand-computed from the
    cylindrical formula, not from the module): straight-ahead ground point -> centre
    column 319.5; a LEFT point (+y) -> col < W/2; behind the camera -> invalid; far
    ground approaches the horizon row monotonically (level and 2-deg-pitched mounts);
(c) MUTATION: a mirrored lift (y -> -y) must fail (b);
    the feature-coordinate convention pinned to REF-C's real ``ResNetEncoder`` by a
    forward impulse; the lift pixel reproduces the cache BUILDER's own resampling
    (``calib.cylindrical_grid``) native pixel; ``BEVLift`` samples the index it claims;
(d) cross-check of the validity mask against the LiDAR BEV GT's ``cart_cam_vis`` on
    >= 5 real clips (skips, naming what is missing, off the dev box).

Clip ids appear nowhere in this file: real clips are joined by sha12 at runtime.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from tanitad.data.calib import (PHYSICALAI_FRONT_WIDE_FTHETA,  # noqa: E402
                                PHYSICALAI_WIDE120_256x640, cylindrical_grid,
                                ftheta_project_rays)
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.models import bev_lift as L  # noqa: E402

FR = PHYSICALAI_WIDE120_256x640
LIDAR_GT_DIR = Path(os.environ.get("TANITAD_LIDAR_BEV_GT_DIR",
                                   "D:/Projects/TanitAD-artifacts/bev-lidar-gt-b1eval-20260913"))
CALIB_DIR = Path(os.environ.get(
    "TANITAD_CALIB_DIR", "D:/Projects/TanitAD-artifacts/hf-corpus-aug-20260915/stage/calibration"))


def _nominal() -> RigCamera:
    """Level, boresight-forward camera 1.5 m high at x = 1.5 m, y = 0."""
    return RigCamera.nominal(FR, height_m=1.5, x_m=1.5, y_m=0.0)


def _pitched(deg: float = 2.0) -> RigCamera:
    """The nominal camera pitched DOWN by ``deg``: columns of R_cam_to_rig are the
    camera right / down / boresight axes in the rig frame."""
    th = math.radians(deg)
    R = torch.tensor([[0.0, -math.sin(th), math.cos(th)],
                      [-1.0, 0.0, 0.0],
                      [0.0, -math.cos(th), -math.sin(th)]], dtype=torch.float64)
    return RigCamera(R_cam_to_rig=R, t_cam_in_rig=torch.tensor([1.5, 0.0, 1.5],
                                                                dtype=torch.float64), frame=FR)


def _pts(*xyz) -> torch.Tensor:
    return torch.tensor(xyz, dtype=torch.float64)


def _project_mirrored(p, cam):
    return L.project_rig_points(p * torch.tensor([1.0, -1.0, 1.0], dtype=p.dtype), cam)


def _geometry_failures(project) -> list[str]:
    """(b) as a list of failed checks, so the mirror mutation can be run through it."""
    fails = []
    cam = _nominal()
    ahead = project(_pts([21.5, 0.0, 0.0]), cam)
    if not (bool(ahead["valid"][0]) and abs(float(ahead["col"][0]) - 319.5) < 1e-9
            and abs(float(ahead["row"][0]) - 150.41831180523292) < 1e-9):
        fails.append("ahead->centre")
    lr = project(_pts([21.5, 3.0, 0.0], [21.5, -3.0, 0.0]), cam)
    if not abs(float(lr["col"][0]) - 274.00258341360995) < 1e-9:
        fails.append("LEFT literal")
    if not float(lr["col"][0]) < 320.0:
        fails.append("LEFT->u<W/2")
    if not abs(float(lr["col"][1]) - 364.99741658639005) < 1e-9:
        fails.append("RIGHT literal")
    if not abs(float(lr["row"][0]) - 150.1647516905655) < 1e-9:
        fails.append("LEFT row literal")
    behind = project(_pts([0.5, 0.0, 0.5], [-5.0, 2.0, 0.0], [1.4, 0.0, 0.0]), cam)
    if bool(behind["valid"].any()) or bool(behind["in_front"].any()):
        fails.append("behind->invalid")
    d = torch.tensor([4.5, 10.0, 20.0, 40.0, 80.0, 160.0, 1000.0], dtype=torch.float64)
    far = project(torch.stack([d + 1.5, torch.zeros_like(d), torch.zeros_like(d)], -1), cam)
    lit = [229.35916357881302, 173.33662361046586, 150.41831180523292, 138.95915590261646,
           133.22957795130824, 130.3647889756541, 127.95836623610465]
    rows = far["row"]
    if not (torch.allclose(rows, torch.tensor(lit, dtype=torch.float64), atol=1e-9, rtol=0)
            and bool((rows.diff() < 0).all()) and bool((rows > 127.5).all())):
        fails.append("far ground -> level horizon 127.5, monotone")
    D = torch.tensor([10.0, 20.0, 40.0, 80.0, 160.0, 1000.0, 1e5], dtype=torch.float64)
    pit = project(torch.stack([D + 1.5, torch.zeros_like(D), torch.zeros_like(D)], -1), _pitched())
    lit_p = [162.48238074644414, 139.71531806637424, 128.28712402511852, 122.56181018646284,
             119.69634263840447, 117.2879000418138, 116.83358813367683]
    rp = pit["row"]
    if not (torch.allclose(rp, torch.tensor(lit_p, dtype=torch.float64), atol=1e-8, rtol=0)
            and bool((rp.diff() < 0).all()) and bool((rp > 116.82899888412614).all())
            and float(rp[-1]) - 116.82899888412614 < 0.01):
        fails.append("far ground -> pitched horizon 116.829, monotone")
    return fails


def test_b_geometry_analytic_literals():
    assert _geometry_failures(L.project_rig_points) == []


def test_c_MUTATION_mirrored_lift_fails_b():
    fails = _geometry_failures(_project_mirrored)
    assert "LEFT->u<W/2" in fails and "LEFT literal" in fails and "RIGHT literal" in fails
    # the straight-ahead / far-field checks cannot see a mirror -- and must not claim to
    assert "ahead->centre" not in fails


def _grid_side_failures(geo) -> list[str]:
    fails = []
    # cell (40, 48): x 20.25 m, y +8.25 m LEFT; cell (40, 15): y -8.25 m RIGHT
    if not (bool(geo.valid[0, 40, 48]) and float(geo.col[0, 40, 48]) < 319.5
            and float(geo.grid[0, 40, 48, 0]) < 0):
        fails.append("LEFT cell -> left half")
    if not (bool(geo.valid[0, 40, 15]) and float(geo.col[0, 40, 15]) > 319.5
            and float(geo.grid[0, 40, 15, 0]) > 0):
        fails.append("RIGHT cell -> right half")
    return fails


def test_b_grid_orientation_and_validity(monkeypatch):
    geo = L.build_lift_geometry(_nominal())
    assert geo.grid.shape == (4, 120, 64, 2) and geo.valid.shape == (4, 120, 64)
    assert geo.feat_hw == (16, 40) and geo.heights_m == (0.0, 0.5, 1.5, 2.5)
    assert _grid_side_failures(geo) == []
    # mirror symmetry of a y = 0 camera: col(j) + col(63 - j) == 2 * 319.5
    v = geo.valid[:, 10:, :]
    s = (geo.col[:, 10:, :] + geo.col[:, 10:, :].flip(-1))[v & v.flip(-1)]
    assert float((s - 639.0).abs().max()) < 1e-9
    # rows 0..2 (x < 1.5 m) are behind the camera at every height
    assert not bool(geo.valid[:, :3].any()) and not bool(geo.in_front[:, :3].any())
    # a cell at 60 deg azimuth edge: y = x * tan(60) is outside for y beyond it
    assert bool(geo.valid[0, 100, 32]) and not bool(geo.in_hfov[0, 6, 0])
    assert torch.isfinite(geo.grid).all()
    # MUTATION at grid level: mirrored cell centres must fail the side checks
    real = L.cell_centers_xy
    monkeypatch.setattr(L, "cell_centers_xy", lambda g: (real(g)[0], -real(g)[1]))
    bad = _grid_side_failures(L.build_lift_geometry(_nominal()))
    assert bad == ["LEFT cell -> left half", "RIGHT cell -> right half"]


def test_feature_grid_literals_and_stride_refusal():
    col = torch.tensor([0.0, 112.0, 319.5, 624.0], dtype=torch.float64)
    row = torch.tensor([0.0, 127.5, 127.5, 255.0], dtype=torch.float64)
    g = L.pixel_to_feature_grid(col, row, FR, 16)
    torch.testing.assert_close(g[:, 0], torch.tensor([-0.975, -0.625, 0.0234375, 0.975],
                                                     dtype=torch.float64))
    torch.testing.assert_close(g[:, 1], torch.tensor([-0.9375, 0.05859375, 0.05859375,
                                                      1.0546875], dtype=torch.float64))
    sym = L.pixel_to_feature_grid(col, row, FR, 16, centre_offset_px=7.5)
    assert float(sym[2, 0]) == 0.0                      # the convention REF-C does NOT have
    with pytest.raises(ValueError):
        L.pixel_to_feature_grid(col, row, FR, 12)


def _binomial_(enc):
    with torch.no_grad():
        for m in enc.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0]
                c = torch.tensor([math.comb(k - 1, i) for i in range(k)], dtype=torch.float64)
                ker = (c[:, None] * c[None, :]) / c.sum() ** 2
                m.weight.copy_(ker.to(m.weight.dtype).expand_as(m.weight) / m.in_channels)


def test_refc_feature_centre_is_stride_times_index():
    """Feature column k of REF-C's stride-16 map is centred on image pixel 16 k (offset
    0), from the module's own layers AND from a forward impulse through the real class."""
    from tanitad.refs.refc import CNNEncoderConfig, ResNetEncoder
    # the refcv5-v2 trunk topology (blocks (3, 6, 16, 6), 256x640); width shrunk to
    # 2 because kernel / stride / padding do not depend on it
    full = ResNetEncoder(CNNEncoderConfig(in_channels=9, image_size=256, image_width=640,
                                          base_width=2, blocks=(3, 6, 16, 6)))
    for n, s in ((3, 16), (4, 32)):
        assert L.receptive_field_centre(L.refc_trunk_path(full, n)) == (s, 0.0)
        assert L.receptive_field_centre(L.refc_trunk_path(full, n, shortcut=True)) == (s, 0.0)
    assert L.REFC_FEATURE_CENTRE_OFFSET_PX == 0.0
    with torch.no_grad():                                # the map the lift is sized for
        h = full.stem(torch.zeros(1, 9, 256, 640))
        for st in full.stages[:3]:
            h = st(h)
        assert tuple(h.shape[-2:]) == L.build_lift_geometry(_nominal()).feat_hw == (16, 40)
        assert h.shape[1] == 4 * 2                       # 4 * base_width (352 at width 88)
    enc = ResNetEncoder(CNNEncoderConfig(in_channels=1, image_size=128, image_width=512,
                                         base_width=2, blocks=(1, 1, 1, 1))).eval()
    _binomial_(enc)
    enc = enc.double()

    def profile(x0):
        img = torch.zeros(1, 1, 128, 512, dtype=torch.float64)
        img[0, 0, 64, x0] = 1.0
        with torch.no_grad():
            h = enc.stem(img)
            for st in enc.stages[:3]:
                h = st(h)
        return h[0].sum(0)[4]                            # row 4 <-> pixel row 64

    p = profile(112)                                     # 16 * 7
    assert int(p.argmax()) == 7 and float((p[6] - p[8]).abs()) <= 1e-12 * float(p[7])
    p = profile(120)                                     # 16 * 7.5: tie between 7 and 8
    assert float((p[7] - p[8]).abs()) <= 1e-12 * float(p[7])
    p = profile(119)                                     # symmetric tiling's centre of 7
    assert float(p[7] - p[8]) > 0.1 * float(p[7])       # ... is NOT a symmetry point


def test_lift_pixel_matches_the_cache_builder_resample():
    """Rig point -> lift (col, row) -> the BUILDER's grid (``calib.cylindrical_grid``,
    what ``cylindrical_rectify`` sampled the fisheye with) -> native pixel must equal
    the direct f-theta projection of the same point."""
    intr = PHYSICALAI_FRONT_WIDE_FTHETA                  # a real f-theta polynomial
    cam = _pitched(1.3)
    g = torch.Generator().manual_seed(0)
    n = 4000
    p = torch.stack([torch.rand(n, generator=g, dtype=torch.float64) * 55 + 4,
                     torch.rand(n, generator=g, dtype=torch.float64) * 32 - 16,
                     torch.rand(n, generator=g, dtype=torch.float64) * 2.5], -1)
    pr = L.project_rig_points(p, cam)
    keep = pr["valid"] & (pr["col"] > 1) & (pr["col"] < 638) & (pr["row"] > 1) & (pr["row"] < 254)
    assert int(keep.sum()) > 2000
    grid, _mask = cylindrical_grid(intr, intr.height, intr.width, FR)   # [1,256,640,2]
    uv_n = (grid[0].permute(2, 0, 1).double() + 1.0) / 2.0 * torch.tensor(
        [intr.width - 1.0, intr.height - 1.0], dtype=torch.float64).view(2, 1, 1)
    col, row = pr["col"][keep], pr["row"][keep]
    s = torch.stack([2 * col / (FR.width - 1) - 1, 2 * row / (FR.height - 1) - 1], -1)
    u_b = torch.nn.functional.grid_sample(uv_n[None], s.view(1, 1, -1, 2), mode="bilinear",
                                          align_corners=True)[0, :, 0]       # [2, n]
    pc = (p[keep] - cam.t_cam_in_rig) @ cam.R_cam_to_rig
    u_d, v_d = ftheta_project_rays(intr, pc[:, 0], pc[:, 1], pc[:, 2])
    err = torch.hypot(u_b[0] - u_d, u_b[1] - v_d)
    assert float(err.max()) < 0.05, float(err.max())    # native px (float32 builder grid)
    # the mirrored lift lands tens of native pixels away
    mr = _project_mirrored(p[keep], cam)
    sm = torch.stack([2 * mr["col"] / (FR.width - 1) - 1, 2 * mr["row"] / (FR.height - 1) - 1], -1)
    u_m = torch.nn.functional.grid_sample(uv_n[None], sm.view(1, 1, -1, 2), mode="bilinear",
                                          align_corners=True)[0, :, 0]
    assert float(torch.median(torch.hypot(u_m[0] - u_d, u_m[1] - v_d))) > 20.0


def test_bevlift_module_params_sampling_and_refusals():
    lift = L.BEVLift(d_in=352, d_out=128)
    print(f"\nBEVLift(d_in=352, d_out=128, Z=4) parameters: {lift.n_params:,}")
    assert lift.n_params == 352 * 4 * 128 + 128 + 128 == 180_480
    geo = L.build_lift_geometry(_pitched(1.0))
    B = 2
    grid = geo.grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
    valid = geo.valid.unsqueeze(0).expand(B, -1, -1, -1)
    out = lift(torch.randn(B, 352, 16, 40), grid, valid)
    assert out.shape == (B, 128, 120, 64) and torch.isfinite(out).all()
    # sampling identity: a map holding its own (column, row) feature index returns, at
    # every valid sample strictly inside the centre lattice, the geometry's index
    probe = L.BEVLift(d_in=2, d_out=2, feat_hw=(16, 40))
    with torch.no_grad():
        probe.proj.weight.zero_()
        probe.proj.bias.zero_()
        for z in range(4):
            probe.proj.weight[:, 2 * z] = 0.0
        probe.proj.weight[0, 0, 0, 0] = 1.0              # height 0, channel "column index"
        probe.proj.weight[1, 1, 0, 0] = 1.0              # height 0, channel "row index"
        probe.unobserved.fill_(-100.0)
    kk = torch.arange(40, dtype=torch.float32).view(1, 1, 40).expand(1, 16, 40)
    rr = torch.arange(16, dtype=torch.float32).view(1, 16, 1).expand(1, 16, 40)
    fmap = torch.cat([kk, rr], 0).unsqueeze(0)
    with torch.no_grad():
        got = probe(fmap, geo.grid[:1].unsqueeze(0).expand(1, 4, -1, -1, -1).contiguous(),
                    geo.valid.unsqueeze(0))[0]
    kx, ky = geo.col[0] / 16.0, geo.row[0] / 16.0
    inner = geo.valid[0] & (kx >= 0) & (kx <= 39) & (ky >= 0) & (ky <= 15)
    assert int(inner.sum()) > 3000
    assert float((got[0][inner] - kx[inner].float()).abs().max()) < 1e-3
    assert float((got[1][inner] - ky[inner].float()).abs().max()) < 1e-3
    none = ~geo.valid.any(0)
    assert bool(none.any()) and bool((got[0][none] == -100.0).all())
    # refusals: wrong stride map, wrong channels, wrong height count
    with pytest.raises(ValueError):
        lift(torch.randn(B, 352, 8, 20), grid, valid)
    with pytest.raises(ValueError):
        lift(torch.randn(B, 704, 16, 40), grid, valid)
    with pytest.raises(ValueError):
        lift(torch.randn(B, 352, 16, 40), grid[:, :3], valid[:, :3])


# --------------------------------------------------------------------------- #
# (d) cross-check against the LiDAR BEV GT camera-field grids                  #
# --------------------------------------------------------------------------- #
def _real_join(n_min: int = 5):
    ext_p = CALIB_DIR / "sensor_extrinsics.parquet"
    if not LIDAR_GT_DIR.is_dir() or not ext_p.is_file():
        pytest.skip(f"needs $TANITAD_LIDAR_BEV_GT_DIR (*.bevgt.npz) and "
                    f"$TANITAD_CALIB_DIR/sensor_extrinsics.parquet")
    from tanitad.data.physicalai import _load_chunk_extrinsics
    ex = _load_chunk_extrinsics(str(ext_p))
    by12 = {hashlib.sha256(c.encode()).hexdigest()[:12]: e for c, e in ex.items()}
    files = sorted(LIDAR_GT_DIR.glob("*.bevgt.npz"))
    pairs = [(f.name[:12], by12[f.name[:12]], f) for f in files if f.name[:12] in by12]
    if len(pairs) < n_min:
        pytest.skip(f"only {len(pairs)} LiDAR GT clips with calibration (need {n_min})")
    return pairs


def _emulate_cam_vis(cam: RigCamera, obs: torch.Tensor) -> np.ndarray:
    """``p2_build_corpus.camera_visibility`` (cart grid: 4x4 sub-samples x heights
    .3/1/2/3 m, in-frame pixel-centre bounds, nearest pixel observed, uint8 share)
    evaluated THROUGH THE LIFT: its cell centres (``L.cell_centers_xy``) and its
    projection (``L.project_rig_points``), so a defect in either breaks exactness."""
    Xc, Yc = L.cell_centers_xy(L.GRID_DEFAULT)
    cell = L.GRID_DEFAULT.cell_m
    ix, iy = Xc / cell - 0.5, (Yc + L.GRID_DEFAULT.y_half_m) / cell - 0.5   # exact cell indices
    sub = (np.arange(4) + 0.5) / 4
    X = np.broadcast_to((ix[:, :, None, None] + sub[None, None, :, None]) * cell, (120, 64, 4, 4))
    Y = np.broadcast_to((iy[:, :, None, None] + sub[None, None, None, :]) * cell
                        - L.GRID_DEFAULT.y_half_m, (120, 64, 4, 4))
    vis = np.zeros((120, 64))
    eps = 1e-6
    for zh in (0.3, 1.0, 2.0, 3.0):
        P = torch.as_tensor(np.stack([X, Y, np.full(X.shape, zh)], -1).reshape(-1, 3), dtype=torch.float64)
        pr = L.project_rig_points(P, cam)
        col, row = pr["col"], pr["row"]
        valid = (pr["in_front"] & (col >= -eps) & (col <= FR.width - 1 + eps)
                 & (row >= -eps) & (row <= FR.height - 1 + eps))
        ok = valid & obs[row.round().long().clamp(0, 255), col.round().long().clamp(0, 639)]
        vis += ok.numpy().reshape(120, 64, -1).mean(-1)
    return np.round(vis / 4 * 255.0).astype(np.uint8)


def test_d_validity_mask_vs_lidar_cam_vis_on_real_clips():
    """Pre-declared reading: disagreement is allowed only inside the 1-cell FOV-edge
    band. Our mask differs from ``cam_vis`` BY DEFINITION (heights 0/.5/1.5/2.5 at the
    cell centre vs .3/1/2/3 at 4x4 sub-samples; field [-0.5, W-0.5] vs [0, W-1]; no
    observed-pixel term), so it is compared as "valid at >= 2 of 4 heights".

    ⚠️ The reference is NOT independent in projection code (the LiDAR builder also
    projects through ``rig_projection.RigCamera``), and the IoU is blind to a mirror or
    a 2 deg yaw/pitch (MEASURED, raw/cam_vis_crosscheck.json). So on rig-A clips
    (no black band, whose observed mask the builder read from decoded frames) the
    lift's camera must also re-emulate ``cam_vis`` EXACTLY -- which a mirrored grid,
    0.5 deg of yaw/pitch or 5 cm of height all break."""
    from scipy import ndimage
    from tanitad.data.calib import cylindrical_grid as _cg
    pairs = _real_join()
    intr = None
    ip = CALIB_DIR / "camera_intrinsics.parquet"
    if ip.is_file():
        from tanitad.data.physicalai import _load_chunk_intrinsics
        intr = {hashlib.sha256(c.encode()).hexdigest()[:12]: i
                for c, i in _load_chunk_intrinsics(str(ip)).items()}
    step = max(1, len(pairs) // 12)
    table, n_emulated = [], 0
    k3 = np.ones((3, 3), bool)
    for s12, extr, f in pairs[::step]:
        with np.load(f) as z:
            cv = z["cart_cam_vis"]
        ref = cv >= 128
        mine = (L.build_lift_geometry(extr).valid.sum(0) >= 2).numpy()
        iou = int((mine & ref).sum()) / int((mine | ref).sum())
        dis = mine != ref
        edge = ref & ~ndimage.binary_erosion(ref, k3, border_value=1)
        edge |= ~ref & ~ndimage.binary_erosion(~ref, k3, border_value=1)
        dist = ndimage.distance_transform_cdt(~edge, metric="chessboard")
        worst = int(dist[dis].max()) if dis.any() else 0
        emu = "-"
        if intr is not None and s12 in intr and intr[s12].cy < 650:        # rig A
            ii = intr[s12]
            obs = _cg(ii, int(ii.height), int(ii.width), FR)[1]
            e = _emulate_cam_vis(L.camera_from_extrinsics(extr), obs)
            assert np.array_equal(e, cv), f"{s12}: lift camera does not re-emulate cam_vis"
            assert not np.array_equal(e[:, ::-1], cv), f"{s12}: mirror control passed"
            emu, n_emulated = "exact", n_emulated + 1
        table.append((s12, round(iou, 5), int(dis.sum()), worst, emu))
        assert worst <= 1, f"{s12}: a disagreeing cell {worst} cells inside the field edge"
        assert iou > 0.99, (s12, iou)
    print("\nsha12         IoU    n_disagree max_dist_to_edge emulation(rig A)")
    for row in table:
        print("%s  %.5f  %5d  %d  %s" % row)
    assert len(table) >= 5
    if intr is not None:
        assert n_emulated >= 2
