"""Tests for the D-016 R1 pinhole rectify primitive.

Covers: f_eff exactness, the PandaSet unblock (naive 467 -> rectify 266), comma2k19
regression (reference corpus unchanged), Brown-Conrady undistort correctness
(independent iterative inverse round-trip + a rendered checkerboard recovery), the
observed-mask honesty, and the episode contract (G-D2: rectified frames are a
drop-in [T,9,256,256] u8 stack).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calib_r1 import (  # noqa: E402
    COMMA2K19_INTR, PANDASET_FRONT_INTR, PinholeIntrinsics,
    brown_conrady_distort, pinhole_geometry_report, pinhole_rectify,
    pinhole_rectify_grid, square_crop_feff)

from tanitad.data._contract import assert_contract, finite_diff_accel  # noqa: E402
from tanitad.data.calib import F_REF  # noqa: E402
from tanitad.data.comma2k19 import stack_frames  # noqa: E402
from tanitad.data.toy_driving import ToyEpisode  # noqa: E402


# --------------------------------------------------------------------------- #
# f_eff exactness + shapes                                                     #
# --------------------------------------------------------------------------- #
def test_feff_exact_by_construction():
    vid = torch.randint(0, 256, (2, 3, 1080, 1920), dtype=torch.uint8)
    out = pinhole_rectify(vid, PANDASET_FRONT_INTR, size=256)
    assert out.shape == (2, 3, 256, 256)
    assert out.dtype == torch.uint8
    assert pinhole_rectify.last_f_eff == F_REF          # exact, not ~
    assert 0.0 < pinhole_rectify.last_observed_frac <= 1.0


def test_center_ray_maps_to_principal_point():
    """The center output pixel must sample the native principal point."""
    intr = PinholeIntrinsics(fx=1000.0, fy=1000.0, cx=800.0, cy=500.0,
                             width=1600, height=1000)
    grid, _ = pinhole_rectify_grid(intr, 1000, 1600, size=257)  # odd -> exact center
    cx_norm = grid[0, 128, 128, 0].item()
    cy_norm = grid[0, 128, 128, 1].item()
    # denormalize back to native px (same convention as the grid builder)
    u = (cx_norm + 1.0) / 2.0 * (1600 - 1)
    v = (cy_norm + 1.0) / 2.0 * (1000 - 1)
    assert abs(u - 800.0) < 1e-3 and abs(v - 500.0) < 1e-3


# --------------------------------------------------------------------------- #
# PandaSet unblock — the headline result                                      #
# --------------------------------------------------------------------------- #
def test_pandaset_naive_crop_is_height_bound():
    """Regression anchor: reproduce the 0715 blocker (square crop lands ~467)."""
    naive = square_crop_feff(PANDASET_FRONT_INTR.fx, 1080, 1920)
    assert naive["height_clamped"] is True
    assert naive["drop_in"] is False
    assert 460 < naive["achieved_feff_px"] < 475       # ~467, the 1.75x zoom


def test_pandaset_rectify_lands_canonical():
    """The rectify path lands f_eff == 266 EXACTLY where the crop lands 467."""
    rep = pinhole_geometry_report(PANDASET_FRONT_INTR)
    assert rep["rectify_feff_px"] == F_REF
    assert rep["rectify_drop_in"] is True
    assert rep["naive_square_crop"]["drop_in"] is False
    # partial observation: horizontally fine (~52 deg native), vertically clipped
    assert 0.4 < rep["rectify_observed_frac"] < 1.0
    # k1=-0.589 is a large barrel term -> tens of px displacement at the edge
    assert rep["max_distort_px_at_edge"] > 5.0


# --------------------------------------------------------------------------- #
# comma2k19 regression — the reference corpus must be untouched                #
# --------------------------------------------------------------------------- #
def test_comma_reference_near_full_observation():
    rep = pinhole_geometry_report(COMMA2K19_INTR)
    assert rep["rectify_feff_px"] == F_REF
    # comma is the F_REF reference: the canonical field is (nearly) all in-frame
    assert rep["rectify_observed_frac"] > 0.98
    # no distortion coeffs -> rectify is a pure pad-crop here
    assert rep["max_distort_px_at_edge"] < 1e-6


# --------------------------------------------------------------------------- #
# Brown-Conrady correctness — independent iterative inverse round-trip         #
# --------------------------------------------------------------------------- #
def _iterative_undistort(xd, yd, dist, iters=30):
    """Independent inverse of brown_conrady_distort (fixed-point), for validation
    only. Recovers the ideal ray from a distorted one."""
    x, y = xd.clone(), yd.clone()
    k1, k2, p1, p2, k3 = dist
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


def test_distortion_forward_inverse_roundtrip():
    """Forward-distort a grid of ideal rays, then recover them with an INDEPENDENT
    iterative inverse -> sub-pixel agreement proves the forward model is correct."""
    dist = PANDASET_FRONT_INTR.dist
    xs = torch.linspace(-0.4, 0.4, 25)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    xd, yd = brown_conrady_distort(gx, gy, dist)
    xr, yr = _iterative_undistort(xd, yd, dist)
    err = torch.sqrt((xr - gx) ** 2 + (yr - gy) ** 2).max().item()
    assert err < 1e-4, f"round-trip residual {err}"


def test_distortion_correction_recovers_checkerboard():
    """End-to-end: warp an ideal checkerboard onto a distorted native sensor, then
    rectify -> high correlation with the original ideal pattern; skipping the
    undistort (k=0) recovers it markedly worse. Exercises the real grid_sample path.
    """
    size = 128
    dist = (-0.5894, 0.66, 0.0011, -0.001, -1.0088)
    intr = PinholeIntrinsics(fx=600.0, fy=600.0, cx=400.0, cy=300.0,
                             width=800, height=600, dist=dist)
    f_ref = 300.0                                       # keep the field in-frame

    # ideal rectilinear checkerboard on the F_REF canvas
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    ideal = (((xs // 12) + (ys // 12)) % 2).float() * 255.0
    ideal = ideal[None, None]                           # [1,1,size,size]

    # render the NATIVE distorted capture: for each native px, find its ideal ray
    # (iterative inverse) and sample the ideal pattern there.
    ny, nx = torch.meshgrid(torch.arange(600.0), torch.arange(800.0), indexing="ij")
    xn = (nx - intr.cx) / intr.fx
    yn = (ny - intr.cy) / intr.fy
    xi, yi = _iterative_undistort(xn, yn, dist)         # ideal ray for each native px
    # ideal ray -> ideal-canvas px (focal f_ref, centered)
    ux = xi * f_ref + (size - 1) / 2.0
    uy = yi * f_ref + (size - 1) / 2.0
    g = torch.stack([ux / (size - 1) * 2 - 1, uy / (size - 1) * 2 - 1], -1)[None]
    native = torch.nn.functional.grid_sample(
        ideal, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    native3 = native.expand(1, 3, 600, 800)

    rect = pinhole_rectify(native3, intr, size=size, f_ref=f_ref)[:, :1].float()
    mask = pinhole_rectify.last_mask
    wrong = pinhole_rectify(
        native3, PinholeIntrinsics(fx=600, fy=600, cx=400, cy=300,
                                   width=800, height=600),   # k=0: no undistort
        size=size, f_ref=f_ref)[:, :1].float()

    def corr(a):
        a, b = a[0, 0][mask], ideal[0, 0][mask]
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))

    c_rect, c_wrong = corr(rect), corr(wrong)
    assert c_rect > 0.9, f"rectified corr {c_rect}"
    assert c_rect > c_wrong + 0.05, f"undistort added nothing: {c_rect} vs {c_wrong}"


# --------------------------------------------------------------------------- #
# Observed-mask honesty                                                        #
# --------------------------------------------------------------------------- #
def test_zeros_padding_blanks_unobserved():
    """With padding_mode='zeros' the unobserved periphery is exactly 0."""
    vid = torch.full((1, 3, 1080, 1920), 200, dtype=torch.uint8)
    out = pinhole_rectify(vid, PANDASET_FRONT_INTR, size=256, padding_mode="zeros")
    mask = pinhole_rectify.last_mask
    assert (out[0, :, ~mask] == 0).all()               # outside native -> black
    assert (out[0, :, mask] > 0).any()                 # inside -> real content


# --------------------------------------------------------------------------- #
# Episode contract (G-D2): rectified frames are a drop-in 9-ch stack           #
# --------------------------------------------------------------------------- #
def _rectified_pandaset_episode(n=8, size=256):
    native = torch.randint(0, 256, (n, 3, 1080, 1920), dtype=torch.uint8)
    rect = pinhole_rectify(native, PANDASET_FRONT_INTR, size=size)   # [n,3,S,S] u8
    stacked = stack_frames(rect, 3)                                  # [n-2,9,S,S] u8
    m = stacked.shape[0]
    v = np.linspace(10.0, 12.0, m + 2)[2:]
    accel = finite_diff_accel(np.linspace(10.0, 12.0, m + 2), 0.1)[2:]
    poses = np.stack([np.arange(m) * 1.1, np.zeros(m), np.zeros(m), v], 1)
    actions = np.stack([np.zeros(m), accel], 1)
    return ToyEpisode(frames=stacked,
                      actions=torch.from_numpy(actions).float(),
                      poses=torch.from_numpy(poses).float(),
                      episode_id=7)


def test_rectified_frames_pass_episode_contract():
    ep = _rectified_pandaset_episode()
    assert_contract(ep, channels=9)                    # G-D2: drop-in contract
    assert ep.frames.shape[1] == 9 and ep.frames.shape[2] == 256
    assert ep.frames.dtype == torch.uint8
    assert torch.isfinite(ep.frames.float()).all()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
