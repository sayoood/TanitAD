"""Standalone tests for the focal-invariance validation module (intake gate-E).

No network, no real bytes, no encoder: synthetic tensors exercise the geometry,
the fail-loud guard, the pixel-level canonicalization claim, and the episode
contract (G-D2). Run:
  pytest "TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-11-focal-invariance-validation/tests" -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import focal_invariance as fi  # noqa: E402

from tanitad.data.calib import (COMMA2K19_FOCAL_PX, F_REF, focal_crop_resize,  # noqa: E402
                                nominal_focal_from_hfov)
from tanitad.data._contract import assert_contract  # noqa: E402
from tanitad.data.toy_driving import ToyEpisode  # noqa: E402


# --------------------------------------------------------------------------- #
# virtual_focal_zoom geometry                                                  #
# --------------------------------------------------------------------------- #
def test_virtual_focal_zoom_magnifies_center():
    """A centered bright square occupies MORE of the frame after zoom-in."""
    img = torch.zeros(1, 1, 100, 100)
    img[..., 40:60, 40:60] = 1.0                     # central 20x20 = 4 % bright
    zoomed = fi.virtual_focal_zoom(img, z=2.0)
    assert zoomed.shape == img.shape
    frac_before = float((img > 0.5).float().mean())
    frac_after = float((zoomed > 0.5).float().mean())
    assert frac_after > 3.0 * frac_before            # ~z^2 magnification


def test_virtual_focal_zoom_identity_at_z1():
    img = torch.rand(2, 3, 48, 64)
    out = fi.virtual_focal_zoom(img, z=1.0)
    assert torch.allclose(out, img, atol=1e-5)


def test_virtual_focal_zoom_rejects_widening():
    with pytest.raises(ValueError):
        fi.virtual_focal_zoom(torch.rand(1, 3, 32, 32), z=0.8)


# --------------------------------------------------------------------------- #
# assert_effective_focal — the fail-loud data-card guard                       #
# --------------------------------------------------------------------------- #
def test_guard_passes_for_reference_and_wide_corpora():
    # comma2k19 (near-reference, no headroom) lands right at F_REF
    f_eff = fi.assert_effective_focal(COMMA2K19_FOCAL_PX, 874, 1164, 256)
    assert abs(f_eff - F_REF) / F_REF < 0.03
    # 120-deg wide corpora (Cosmos / PhysicalAI front_wide) have headroom
    for w in (1280, 1920):
        f = nominal_focal_from_hfov(w, 120.0)
        f_eff = fi.assert_effective_focal(f, int(w * 9 / 16), w, 256)
        assert abs(f_eff - F_REF) / F_REF < 0.03


def test_guard_raises_for_narrow_camera_without_headroom():
    # A telephoto-ish 20-deg camera cannot be cropped down to F_REF.
    f_narrow = nominal_focal_from_hfov(800, 20.0)
    with pytest.raises(ValueError, match="narrower than"):
        fi.assert_effective_focal(f_narrow, 800, 800, 256)


def test_achieved_focal_matches_calib_side_channel():
    f = nominal_focal_from_hfov(1280, 120.0)
    vid = torch.zeros(1, 3, 720, 1280, dtype=torch.uint8)
    focal_crop_resize(vid, f, 256)                   # sets .last_f_eff
    assert abs(fi.achieved_focal(f, 720, 1280, 256)
               - focal_crop_resize.last_f_eff) < 1e-6


# --------------------------------------------------------------------------- #
# The core claim, at pixel level (deterministic, no encoder)                   #
# --------------------------------------------------------------------------- #
def _radial_texture(h: int, w: int) -> torch.Tensor:
    """A scale-sensitive structured image (concentric rings)."""
    ys = torch.linspace(-1, 1, h).view(h, 1)
    xs = torch.linspace(-1, 1, w).view(1, w)
    r = (ys ** 2 + xs ** 2).sqrt()
    rings = (torch.sin(r * 25.0) * 0.5 + 0.5)        # [h,w] in [0,1]
    return rings.expand(3, h, w).unsqueeze(0).mul(255).to(torch.uint8)


def test_correct_intrinsics_beats_wrong_intrinsics_pixel():
    """Using the perturbed camera's TRUE focal recovers the base image; using
    the wrong (base) focal does not. This is D-016's claim without a model."""
    h, w, z, size = 720, 1280, 1.5, 128
    img = _radial_texture(h, w)                      # [1,3,h,w] uint8
    f0 = nominal_focal_from_hfov(w, 120.0)
    fi.assert_effective_focal(f0, h, w, size)        # base has headroom
    zoomed = fi.virtual_focal_zoom(img, z)

    a = focal_crop_resize(img, f0, size).float()
    b_correct = focal_crop_resize(zoomed, z * f0, size).float()
    b_wrong = focal_crop_resize(zoomed, f0, size).float()

    mae_correct = (a - b_correct).abs().mean()
    mae_wrong = (a - b_wrong).abs().mean()
    assert mae_correct < mae_wrong                   # correct intrinsics wins
    # and by a clear margin (not a coin-flip)
    assert mae_correct < 0.6 * mae_wrong


# --------------------------------------------------------------------------- #
# Episode contract through the focal pipeline (G-D2)                           #
# --------------------------------------------------------------------------- #
def test_focal_pipeline_preserves_episode_contract():
    """Frames run through focal_crop_resize + 3-frame stacking still satisfy the
    9-channel D-015 episode contract (channels=9, [T,9,S,S], aligned shapes)."""
    T, h, w, size = 6, 720, 1280, 64
    raw = torch.randint(0, 256, (T, 3, h, w), dtype=torch.uint8)
    f = nominal_focal_from_hfov(w, 120.0)
    canon = focal_crop_resize(raw, f, size)          # [T,3,size,size] uint8
    parts = [canon[i:T - 2 + i] for i in range(3)]   # D-015 3-frame stack
    frames9 = torch.cat(parts, dim=1)                # [T-2, 9, size, size]
    Tp = frames9.shape[0]
    ep = ToyEpisode(
        frames=frames9,
        actions=torch.zeros(Tp, 2),
        poses=torch.zeros(Tp, 4),
        episode_id=7,
    )
    assert_contract(ep, channels=9)                  # raises on any drift
    assert frames9.shape[1] == 9 and frames9.dtype == torch.uint8
