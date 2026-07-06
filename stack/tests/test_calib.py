"""D-016 focal canonicalization: formula, clamping, cross-corpus consistency."""

import math

import torch

from tanitad.data.calib import (COMMA2K19_FOCAL_PX, F_REF,
                                PHYSICALAI_FRONT_WIDE_HFOV_DEG,
                                focal_crop_resize, focal_crop_size,
                                nominal_focal_from_hfov)


def test_comma_reference_is_nearly_uncropped():
    """F_REF was chosen so the comma camera keeps ~its full frame height."""
    c = focal_crop_size(COMMA2K19_FOCAL_PX, 874, 1164, 256)
    assert c == 874                       # clamped to frame height ~ full view
    f_eff = COMMA2K19_FOCAL_PX * 256 / c
    assert abs(f_eff - F_REF) / F_REF < 0.02


def test_wide_camera_gets_zoom_crop_to_same_focal():
    w = 1920
    f = nominal_focal_from_hfov(w, PHYSICALAI_FRONT_WIDE_HFOV_DEG)
    assert abs(f - w / (2 * math.tan(math.radians(60)))) < 1e-6
    c = focal_crop_size(f, 1080, w, 256)
    assert c < 1080                       # genuinely cropped, not clamped
    f_eff = f * 256 / c
    assert abs(f_eff - F_REF) / F_REF < 0.02
    # retained horizontal FOV ~ comma-like (~50 deg), not 120
    hfov_kept = 2 * math.degrees(math.atan(c / (2 * f)))
    assert 45 < hfov_kept < 58


def test_focal_crop_resize_shapes_and_feff():
    vid = torch.randint(0, 255, (4, 3, 1080, 1920), dtype=torch.uint8)
    f = nominal_focal_from_hfov(1920, 120.0)
    out = focal_crop_resize(vid, f, 256)
    assert out.shape == (4, 3, 256, 256) and out.dtype == torch.uint8
    assert abs(focal_crop_resize.last_f_eff - F_REF) / F_REF < 0.02
