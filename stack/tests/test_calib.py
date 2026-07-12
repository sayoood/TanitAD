"""D-016 focal canonicalization: pinhole formula (comma reference) + the
f-theta FISHEYE fix for PhysicalAI (GEOMETRY_INTEGRITY_AUDIT.md).

The front-wide is an f-theta fisheye (real paraxial focal ~928 px @1920). The
old path fed its NOMINAL 120-deg pinhole focal (554 px) into the pinhole crop
and silently produced f_eff ~434 px (1.6x over-zoomed vs comma's 266). These
tests pin the corrected f-theta canonicalization to f_eff ~= 266 and guard the
regression.
"""

import math

import torch

from tanitad.data.calib import (COMMA2K19_FOCAL_PX, F_REF,
                                PHYSICALAI_FRONT_WIDE_FTHETA,
                                PHYSICALAI_FRONT_WIDE_HFOV_DEG,
                                canonical_halfangle_rad, focal_crop_resize,
                                focal_crop_size, ftheta_crop_resize,
                                ftheta_crop_size, ftheta_feff_report,
                                ftheta_undistort, ftheta_undistort_grid,
                                nominal_focal_from_hfov)


# --------------------------------------------------------------------------- #
# comma2k19 reference (UNCHANGED by the fix) + the pinhole formula in isolation #
# --------------------------------------------------------------------------- #
def test_comma_reference_is_nearly_uncropped():
    """F_REF was chosen so the comma camera keeps ~its full frame height."""
    c = focal_crop_size(COMMA2K19_FOCAL_PX, 874, 1164, 256)
    assert c == 874                       # clamped to frame height ~ full view
    f_eff = COMMA2K19_FOCAL_PX * 256 / c
    assert abs(f_eff - F_REF) / F_REF < 0.02


def test_pinhole_crop_formula_self_consistent():
    """The pinhole crop formula maps a nominal pinhole focal back to F_REF. This
    is the pinhole path (correct for comma's ~rectilinear lens) — NOT how the
    f-theta front-wide is handled (see the ftheta tests): feeding the nominal
    120-deg focal here is exactly the assumption the audit proved wrong."""
    w = 1920
    f = nominal_focal_from_hfov(w, PHYSICALAI_FRONT_WIDE_HFOV_DEG)
    assert abs(f - w / (2 * math.tan(math.radians(60)))) < 1e-6
    c = focal_crop_size(f, 1080, w, 256)
    assert c < 1080                       # genuinely cropped, not clamped
    f_eff = f * 256 / c
    assert abs(f_eff - F_REF) / F_REF < 0.02


def test_focal_crop_resize_shapes_and_feff():
    vid = torch.randint(0, 255, (4, 3, 1080, 1920), dtype=torch.uint8)
    f = nominal_focal_from_hfov(1920, 120.0)
    out = focal_crop_resize(vid, f, 256)
    assert out.shape == (4, 3, 256, 256) and out.dtype == torch.uint8
    assert abs(focal_crop_resize.last_f_eff - F_REF) / F_REF < 0.02


# --------------------------------------------------------------------------- #
# f-theta fix: PhysicalAI front-wide canonicalizes to the SHARED f_eff (~266)   #
# --------------------------------------------------------------------------- #
def test_ftheta_physicalai_achieves_shared_focal():
    """Real f-theta front-wide (paraxial ~928 px) crops to f_eff ~= 266 px,
    matching comma — the whole point of D-016."""
    intr = PHYSICALAI_FRONT_WIDE_FTHETA
    assert 920 < intr.paraxial_focal < 935            # measured real focal
    vid = torch.randint(0, 255, (3, 3, 1080, 1920), dtype=torch.uint8)
    out = ftheta_crop_resize(vid, intr, 256)
    assert out.shape == (3, 3, 256, 256) and out.dtype == torch.uint8
    assert abs(ftheta_crop_resize.last_f_eff - F_REF) / F_REF < 0.01
    # retained field ~ comma's ~51 deg (NOT 120, and NOT the buggy ~33)
    rep = ftheta_feff_report(intr)
    assert 49.0 < rep["retained_hfov_after_deg"] < 53.0


def test_ftheta_regression_guard_old_nominal_path_was_434():
    """Guard the CONFIRMED bug: the nominal-554-pinhole path over-zoomed the
    fisheye to ~434 px f_eff, not 266 — a ~1.6x error."""
    rep = ftheta_feff_report(PHYSICALAI_FRONT_WIDE_FTHETA)
    assert 425.0 < rep["f_eff_before_nominal"] < 445.0     # ~434 measured
    assert abs(rep["f_eff_after"] - F_REF) < 3.0           # fix lands 266
    assert 1.55 < rep["f_eff_before_nominal"] / rep["f_eff_after"] < 1.72


def test_ftheta_crop_is_not_the_pinhole_crop():
    """The f-theta crop inverts the REAL radial map; plugging the paraxial focal
    into the pinhole crop formula over-crops and misses 266."""
    intr = PHYSICALAI_FRONT_WIDE_FTHETA
    c_ftheta = ftheta_crop_size(intr, 256)
    c_pinhole = focal_crop_size(intr.paraxial_focal, 1080, 1920, 256)
    assert c_ftheta != c_pinhole
    f_pinhole_plug = 128.0 / math.tan(intr.theta_of_r(c_pinhole / 2.0))
    assert abs(f_pinhole_plug - F_REF) > 10.0             # wrong (~245)
    f_ftheta = 128.0 / math.tan(intr.theta_of_r(c_ftheta / 2.0))
    assert abs(f_ftheta - F_REF) < 3.0                    # right (~266)


def test_canonical_halfangle_matches_comma():
    """The shared canonical half-angle (~25.65 deg) is comma's retained field."""
    th = canonical_halfangle_rad(256, F_REF)
    comma_c = min(round(COMMA2K19_FOCAL_PX * 256 / F_REF), 874)
    comma_th = math.atan((comma_c / 2) / COMMA2K19_FOCAL_PX)
    assert abs(math.degrees(th) - math.degrees(comma_th)) < 0.1
    assert 25.0 < math.degrees(th) < 26.0


def test_ftheta_forward_inverse_roundtrip():
    intr = PHYSICALAI_FRONT_WIDE_FTHETA
    for deg in (1.0, 10.0, 25.0, 40.0):
        th = math.radians(deg)
        r = float(intr.r_of_theta(th))
        assert abs(intr.theta_of_r(r) - th) < 1e-4
    # near axis r ~= f_paraxial * theta
    assert abs(float(intr.r_of_theta(0.01)) - intr.paraxial_focal * 0.01) < 0.5


# --------------------------------------------------------------------------- #
# undistort maths (R1 option) on a synthetic f-theta pattern                    #
# --------------------------------------------------------------------------- #
def test_ftheta_undistort_grid_math():
    """The undistort grid maps each pinhole output pixel to native pp + r(theta)
    along its azimuth — recompute the map and check the grid matches exactly."""
    intr = PHYSICALAI_FRONT_WIDE_FTHETA
    size = 128
    g = ftheta_undistort_grid(intr, size, F_REF)
    assert g.shape == (1, size, size, 2)
    c = (size - 1) / 2.0
    for (i, j) in [(20, 63), (40, 63), (63, 40), (30, 90)]:
        x, y = j - c, i - c
        rho = math.hypot(x, y)
        theta = math.atan2(rho, F_REF)
        r = float(intr.r_of_theta(theta))
        u_exp = intr.cx + (x / rho) * r
        v_exp = intr.cy + (y / rho) * r
        u = (float(g[0, i, j, 0]) + 1) / 2 * (intr.width - 1)
        v = (float(g[0, i, j, 1]) + 1) / 2 * (intr.height - 1)
        assert abs(u - u_exp) < 0.5 and abs(v - v_exp) < 0.5


def test_ftheta_undistort_synthetic_pattern():
    """A feature at real incidence theta lands at rectilinear radius
    F_REF*tan(theta) after undistort (fisheye distortion removed)."""
    intr = PHYSICALAI_FRONT_WIDE_FTHETA
    H, W = intr.height, intr.width
    theta_test = math.radians(15.0)
    r = float(intr.r_of_theta(theta_test))
    img = torch.zeros(1, 3, H, W)
    cy, cx = int(round(intr.cy)), int(round(intr.cx))
    yc, xc = cy - int(round(r)), cx                   # straight above the axis
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    disk = ((yy - yc) ** 2 + (xx - xc) ** 2) <= 18 ** 2
    img[:, :, disk] = 255.0
    out = ftheta_undistort(img, intr, 256)
    assert out.shape == (1, 3, 256, 256) and out.dtype == torch.uint8
    assert ftheta_undistort.last_f_eff == F_REF
    ys, xs = torch.where(out[0].float().mean(0) > 50)
    assert len(ys) > 0
    rad = math.hypot(float(ys.float().mean()) - 127.5,
                     float(xs.float().mean()) - 127.5)
    expected = F_REF * math.tan(theta_test)
    assert abs(rad - expected) / expected < 0.08


# --------------------------------------------------------------------------- #
# per-clip intrinsics loaded from the LOCAL data dir (gated -> not committed)    #
# --------------------------------------------------------------------------- #
def test_intrinsics_per_clip_local_table(tmp_path, monkeypatch):
    """Per-clip table is read from a local file (env override), covering BOTH
    camera rigs (cy~540 and cy~754 — the audit sampled only rig A). Every rig
    still canonicalizes to ~266."""
    from tanitad.data import physicalai as P
    csv = tmp_path / P._INTR_BASENAME
    csv.write_text(
        "clip_id,width,height,cx,cy,fw_poly_0,fw_poly_1,fw_poly_2,fw_poly_3,fw_poly_4\n"
        "rigA-uuid,1920,1080,959,540.0,0.0,928.0,23.0,-58.0,16.0\n"
        "rigB-uuid,1920,1080,958,754.0,0.0,934.0,23.0,-60.0,18.0\n")
    monkeypatch.setenv(P._INTR_ENV, str(csv))
    P._load_intrinsics_csv.cache_clear()
    a = P.intrinsics_for_clip("rigA-uuid")
    b = P.intrinsics_for_clip("rigB-uuid")
    assert abs(a.cy - 540) < 1 and abs(b.cy - 754) < 1        # both rigs load
    assert 900 < a.paraxial_focal < 960
    for intr in (a, b):
        f = 128.0 / math.tan(intr.theta_of_r(ftheta_crop_size(intr, 256) / 2.0))
        assert abs(f - F_REF) < 3.0                           # both -> 266
    assert P.intrinsics_for_clip("unknown") is PHYSICALAI_FRONT_WIDE_FTHETA


def test_intrinsics_fallback_when_no_local_table(monkeypatch):
    """No local table -> measured corpus-median fallback (still lands ~266)."""
    from tanitad.data import physicalai as P
    monkeypatch.delenv(P._INTR_ENV, raising=False)
    P._load_intrinsics_csv.cache_clear()
    intr = P.intrinsics_for_clip("any", root=None)
    assert intr is PHYSICALAI_FRONT_WIDE_FTHETA
    f = 128.0 / math.tan(intr.theta_of_r(ftheta_crop_size(intr, 256) / 2.0))
    assert abs(f - F_REF) < 3.0
