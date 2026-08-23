"""D-016 R1 — pinhole rectify-to-canvas (undistort + pad), the owned-real-urban unblock.

WHY THIS EXISTS (Data-Eng 2026-07-17, continues the 2026-07-15 PandaSet blocker)
--------------------------------------------------------------------------------
The 2026-07-15 run shipped the PandaSet loader but found a HARD, MEASURED geometry
blocker that gates the ENTIRE owned real-urban tier (OWN_DATASET_PLAN §7 #1-#4):

    D-016 canonicalizes every camera to a shared effective focal F_REF=266 px by
    cropping a centered SQUARE of side ``c = fx*size/F_REF`` then resizing to
    ``size`` (``calib.focal_crop_resize``). On a 16:9 frame the square is bounded
    by the HEIGHT. PandaSet's real front camera (fx=1970.01 on 1920x1080) needs
    c=1896 px >> 1080 -> the crop clamps to 1080 and lands f_eff ~= 467 px
    (~1.75x more zoomed than comma2k19) -- a REAL cross-corpus action->pixel scale
    mismatch -- AND the pinhole crop ignores the barrel distortion (k1=-0.589).
    General rule the 0715 run proved: any pinhole with fx>1122 px on a 1080-tall
    frame is not square-croppable to 266. Udacity (narrow FOV) hits the same wall.

The existing calib.py has a fisheye rectify (``ftheta_undistort`` / ``_grid``) built
on ``grid_sample`` but NO pinhole equivalent. This module adds it, mirroring that
design exactly so it composes with the codebase:

    build an ideal PINHOLE canvas of focal F_REF at the output size, forward-map
    each ideal ray through the Brown-Conrady distortion model onto the native
    (distorted) sensor, and ``grid_sample`` it back. f_eff == F_REF holds BY
    CONSTRUCTION (the canvas is defined at F_REF), the barrel distortion is
    removed, and rays that fall OUTSIDE the native frame become an explicit,
    MEASURED unobserved mask instead of a silent zoom-in.

This turns "height-bound -> zoom (wrong)" into "height-bound -> masked periphery
(honest)". The unobserved band is exactly the H17 masked-periphery philosophy
(``Architecture & Inference/Research/UNIFIED_FOV_FOVEATED_PATCHING.md``) and a free
H15 imagination target; the ``observed_frac`` is the data-card number that decides
whether a source is worth ingesting at the canonical scale.

With zero distortion coeffs the same primitive degrades to a pure pad-crop (the
"letterbox" path the 0715 INTAKE asked for), so ONE function covers both the
"undistort" and the "pad-crop" halves of the D-016 R1 request.

SCOPE. Pure geometry, no I/O, no new deps (torch only). Reuses ``calib.F_REF`` and
``nominal_focal_from_hfov``. Proposed target: fold into ``stack/tanitad/data/calib.py``
and flip PandaSet ``_canonicalize`` (and future Udacity) from ``focal_crop_resize``
to ``pinhole_rectify``. Fisheye corpora (ZOD Kannala-Brandt, PhysicalAI/Cosmos
f-theta) keep the existing ``ftheta_*`` path -- see INTAKE for the coverage map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.data.calib import F_REF, canonical_halfangle_rad


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Brown-Conrady pinhole model on a native ``width x height`` sensor.

    ``dist = (k1, k2, p1, p2, k3)`` are the OpenCV-order radial/tangential coeffs
    (radial k1,k2,k3; tangential p1,p2). Defaults to zero -> a pure pinhole, in
    which case :func:`pinhole_rectify` is an undistort-free pad-crop.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 1920
    height: int = 1080
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))


def brown_conrady_distort(x: Tensor, y: Tensor,
                          dist: tuple[float, float, float, float, float]
                          ) -> tuple[Tensor, Tensor]:
    """Forward Brown-Conrady: ideal normalized ray ``(x, y)=(X/Z, Y/Z)`` ->
    DISTORTED normalized coords (still focal-normalized, pre-``fx`` scaling).

    ``x_d = x*(1+k1 r^2+k2 r^4+k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)`` and the
    symmetric ``y_d``. This is the map a real lens applies; rectification samples
    the native frame at ``fx*x_d+cx`` so the OUTPUT is the ideal rectilinear ray.
    """
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))          # Horner in r^2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_d, y_d


def pinhole_rectify_grid(intr: PinholeIntrinsics, h: int, w: int,
                         size: int = 256, f_ref: float = F_REF,
                         device: str | torch.device = "cpu"
                         ) -> tuple[Tensor, Tensor]:
    """``grid_sample`` grid + observed mask mapping an ideal pinhole (focal
    ``f_ref``, centered on the optical axis) onto the native ``h x w`` frame.

    Intrinsics are defined on ``intr.width x intr.height``; they are scaled to the
    DECODED ``(h, w)`` so a frame decoded at a non-native resolution still maps
    correctly. Returns ``(grid [1,size,size,2] in [-1,1], mask [size,size] bool)``
    where ``mask`` is True for output pixels whose ideal ray lands INSIDE the
    native frame (the observed region). ``f_eff == f_ref`` holds by construction:
    output pixel at ``d`` px from center is the ray ``atan(d / f_ref)``.
    """
    sx = w / float(intr.width)
    sy = h / float(intr.height)
    fx, fy = intr.fx * sx, intr.fy * sy
    cx, cy = intr.cx * sx, intr.cy * sy

    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32, device=device),
        torch.arange(size, dtype=torch.float32, device=device), indexing="ij")
    x = (xs - (size - 1) / 2.0) / float(f_ref)             # ideal normalized ray
    y = (ys - (size - 1) / 2.0) / float(f_ref)
    x_d, y_d = brown_conrady_distort(x, y, intr.dist)
    u = fx * x_d + cx                                      # native px
    v = fy * y_d + cy
    mask = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    # normalization convention matches calib.ftheta_undistort_grid (codebase-consistent)
    gx = u / (w - 1) * 2.0 - 1.0
    gy = v / (h - 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    return grid, mask


def pinhole_rectify(vid: Tensor, intr: PinholeIntrinsics, size: int = 256,
                    f_ref: float = F_REF, padding_mode: str = "zeros") -> Tensor:
    """[T,3,H,W] uint8/float -> [T,3,size,size] uint8 rectilinear pinhole at
    ``f_eff == f_ref``, barrel distortion removed, out-of-frame periphery masked.

    Unobserved pixels are filled per ``padding_mode`` (``"zeros"`` = honest black
    periphery, the default; ``"border"`` = edge-extended). The achieved focal and
    the observed fraction are exposed for the data card:
        ``pinhole_rectify.last_f_eff``        == f_ref (exact by construction)
        ``pinhole_rectify.last_observed_frac`` fraction of pixels inside native
        ``pinhole_rectify.last_mask``          [size,size] bool observed mask
    """
    t, _, h, w = vid.shape
    grid, mask = pinhole_rectify_grid(intr, h, w, size, f_ref, device=vid.device)
    out = F.grid_sample(vid.float(), grid.expand(t, -1, -1, -1),
                        mode="bilinear", padding_mode=padding_mode,
                        align_corners=False)
    if padding_mode == "zeros":
        out = out * mask.to(out.dtype)                     # crisp unobserved band
    pinhole_rectify.last_f_eff = float(f_ref)
    pinhole_rectify.last_mask = mask
    pinhole_rectify.last_observed_frac = float(mask.float().mean())
    return out.clamp(0, 255).to(torch.uint8)


def square_crop_feff(fx: float, h: int, w: int, size: int = 256,
                     f_ref: float = F_REF) -> dict:
    """What the OLD centered-square crop (``calib.focal_crop_size``) actually lands
    for this pinhole -- reproduces the 0715 blocker number as a regression anchor.
    """
    c_ideal = int(round(fx * size / f_ref))
    c_used = max(32, min(c_ideal, min(h, w)))
    achieved = fx * size / c_used
    return {"ideal_crop_px": c_ideal, "used_crop_px": c_used,
            "achieved_feff_px": round(achieved, 2),
            "height_clamped": c_ideal > min(h, w),
            "drop_in": abs(achieved - f_ref) / f_ref <= 0.05}


def pinhole_geometry_report(intr: PinholeIntrinsics, size: int = 256,
                            f_ref: float = F_REF) -> dict:
    """Naive square-crop vs pinhole-rectify for one camera. Data-card + regression
    provenance: proves the rectify path lands f_eff==f_ref where the crop can't,
    and quantifies the unobserved cost (``observed_frac``) and the distortion it
    corrects (``max_distort_px`` at the canonical corner)."""
    naive = square_crop_feff(intr.fx, intr.height, intr.width, size, f_ref)
    # rectify on a native-resolution dummy frame -> real observed mask
    dummy = torch.zeros(1, 3, intr.height, intr.width)
    pinhole_rectify(dummy, intr, size, f_ref)
    obs = pinhole_rectify.last_observed_frac

    # distortion magnitude at the canonical corner ray (px on the native sensor)
    corner = canonical_halfangle_rad(size, f_ref)          # edge half-angle
    xr = math.tan(corner)                                   # ideal normalized @ edge
    xd, _ = brown_conrady_distort(torch.tensor(xr), torch.tensor(0.0), intr.dist)
    max_distort_px = abs(float(xd) - xr) * intr.fx

    return {
        "camera_hfov_deg": round(intr.hfov_deg, 2),
        "camera_vfov_deg": round(intr.vfov_deg, 2),
        "canonical_hfov_deg": round(math.degrees(2 * corner), 2),
        "naive_square_crop": naive,
        "rectify_feff_px": float(f_ref),
        "rectify_observed_frac": round(obs, 4),
        "rectify_drop_in": True,
        "max_distort_px_at_edge": round(max_distort_px, 2),
        "k1": intr.dist[0],
    }


# The REAL PandaSet front calibration (arXiv 2112.12610), grounded 2026-07-15.
# Kept here so the rectify path has a first-class, tested owned-real-urban target.
PANDASET_FRONT_INTR = PinholeIntrinsics(
    fx=1970.0131, fy=1970.0091, cx=970.0002, cy=483.2988,
    width=1920, height=1080,
    dist=(-0.5894, 0.66, 0.0011, -0.001, -1.0088))

# comma2k19 EON road camera (near-pinhole reference; f~910 px on 1164x874, ~0 dist)
COMMA2K19_INTR = PinholeIntrinsics(
    fx=910.0, fy=910.0, cx=582.0, cy=437.0, width=1164, height=874)
