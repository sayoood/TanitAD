"""Camera intrinsic canonicalization (D-016; the VLM3/H7 principle applied now).

Problem: comma2k19's road camera (~50 deg HFOV, f~910 px @ 1164x874) and
PhysicalAI-AV's front-wide (120 deg HFOV) have wildly different pixel<->metric
scales. A naive center-crop+resize feeds the world model inconsistent
action->pixel-motion geometry across corpora, corrupting exactly the dynamics
it must learn (and any metric probe on top).

Fix: crop each camera so the EFFECTIVE focal length at the model input size is
one shared constant, then resize. After cropping a centered square of side c
from an image with focal f_px and resizing to `size`:
    f_eff = f_px * size / c        =>       c = f_px * size / F_REF
F_REF is chosen so the reference camera (comma2k19) is (nearly) uncropped:
910 * 256 / 874 ~= 266.5 -> F_REF = 266. PhysicalAI front-wide then gets a
tighter central crop (~51 deg retained) — angularly consistent with comma;
the sacrificed wide periphery is precisely what H2 modality steering
re-introduces later as dedicated side views.

Extrinsics (mount height/pitch/roll) are NOT yet normalized — recorded as a
known limitation in D-016; horizon-alignment homography is the R1 follow-up
(Deep Think 8). Per-clip intrinsics from PhysicalAI `calibration/` replace the
nominal-FOV focal when the DataEng agent lands them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

F_REF = 266.0                     # effective focal [px] at the 256-px input
COMMA2K19_FOCAL_PX = 910.0        # EON road camera, 1164x874
PHYSICALAI_FRONT_WIDE_HFOV_DEG = 120.0


def nominal_focal_from_hfov(width_px: int, hfov_deg: float) -> float:
    """Pinhole focal from horizontal FOV: f = W / (2 tan(HFOV/2))."""
    return width_px / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def focal_crop_size(f_px: float, h: int, w: int, size: int,
                    f_ref: float = F_REF) -> int:
    """Centered-square crop side that yields f_eff == f_ref (clamped)."""
    c = int(round(f_px * size / f_ref))
    return max(32, min(c, min(h, w)))


def focal_crop_resize(vid: Tensor, f_px: float, size: int,
                      f_ref: float = F_REF) -> Tensor:
    """[T, 3, H, W] (uint8 or float) -> [T, 3, size, size] uint8, canonical focal.

    Center crop of side focal_crop_size(...), then bilinear resize. Returns the
    achieved effective focal in `focal_crop_resize.last_f_eff` for data cards.
    """
    t, _, h, w = vid.shape
    c = focal_crop_size(f_px, h, w, size, f_ref)
    top, left = (h - c) // 2, (w - c) // 2
    out = vid[..., top:top + c, left:left + c].float()
    out = F.interpolate(out, size=(size, size), mode="bilinear",
                        align_corners=False)
    focal_crop_resize.last_f_eff = f_px * size / c
    return out.clamp(0, 255).to(torch.uint8)


# --------------------------------------------------------------------------- #
# f-theta (fisheye) canonicalization — D-016 fix (GEOMETRY_INTEGRITY_AUDIT.md) #
# --------------------------------------------------------------------------- #
# The PhysicalAI front-wide is an f-theta FISHEYE, not a rectilinear pinhole.
# The old path fed its NOMINAL 120-deg pinhole focal (554 px) into the pinhole
# `focal_crop_resize`, cropping a 533-px square that — under the REAL fisheye
# radial map r(theta) — retains only ~16.4 deg half-angle -> canonical f_eff
# ~434 px, i.e. 1.63x more zoomed than the f_eff=266 the pipeline claims (and
# than comma2k19). Fix: crop against the REAL radial map so the retained
# half-angle equals the shared canonical one (comma's), making f_eff == F_REF.
#
# The canonical FOV is defined as the half-angle a pinhole of focal F_REF
# subtends over size/2 px (exactly what comma2k19 achieves). Cropping the
# fisheye to that SAME half-angle equalizes the action->pixel scale across
# corpora — the whole point of D-016.


def canonical_halfangle_rad(size: int = 256, f_ref: float = F_REF) -> float:
    """Retained half-angle of the shared canonical frame: atan((size/2)/f_ref).

    A pinhole of focal ``f_ref`` maps a ray at this angle to the frame edge
    (size/2). comma2k19 achieves exactly this (f_eff ~= 266); every other corpus
    is cropped to retain the same half-angle so f_eff and angular field match.
    """
    return math.atan((size / 2.0) / f_ref)


@dataclass(frozen=True)
class FThetaIntrinsics:
    """f-theta fisheye model: radius ``r(theta) = sum poly[i]*theta**i`` [native
    px] from the principal point (cx, cy), on a native ``width x height`` sensor.

    PhysicalAI-AV ships this per clip in ``calibration/camera_intrinsics``
    (columns ``fw_poly_0..4``, ``cx``, ``cy``, ``width``, ``height``);
    ``fw_poly_1`` is the paraxial focal dr/dtheta|_0. Poly is evaluated with
    Horner so it works on floats and on tensors (undistort grid) alike.
    """

    poly: tuple[float, ...]
    cx: float
    cy: float
    width: int = 1920
    height: int = 1080

    def r_of_theta(self, theta):
        """Fisheye radius [native px] for incidence angle ``theta`` [rad]."""
        r = 0.0 if isinstance(theta, float) else torch.zeros_like(theta)
        for c in reversed(self.poly):
            r = r * theta + c
        return r

    def theta_of_r(self, r_target: float, hi: float = 1.6) -> float:
        """Inverse map (scalar): incidence angle whose radius is ``r_target``.

        The map is monotone over a fisheye's field, so plain bisection is exact
        and dependency-free.
        """
        lo, high = 0.0, hi
        for _ in range(80):
            mid = 0.5 * (lo + high)
            if float(self.r_of_theta(mid)) < r_target:
                lo = mid
            else:
                high = mid
        return 0.5 * (lo + high)

    @property
    def paraxial_focal(self) -> float:
        return float(self.poly[1])


# Corpus-median fallback (MEASURED 2026-07-12 over the 500 R0-selected clips /
# 30 calibration chunks; per-clip focal sigma 0.47%). USED ONLY when a per-clip
# entry is unavailable. NOTE: the vertical principal point is BIMODAL across two
# rigs (cy~543 for 23% of clips, cy~754 for 77%) — so principal-point-dependent
# processing MUST use per-clip cy, never this median. The default crop-based
# canonicalization below is geometric-centered and does NOT use (cx, cy), so it
# is robust to the rig split; (cx, cy) here serve only the undistort helper.
PHYSICALAI_FRONT_WIDE_FTHETA = FThetaIntrinsics(
    poly=(0.0, 927.5032, 23.1353, -58.5012, 16.5067),
    cx=958.0, cy=753.18, width=1920, height=1080)


def ftheta_crop_size(intr: FThetaIntrinsics, size: int = 256,
                     f_ref: float = F_REF) -> int:
    """Centered-square crop side that retains the shared canonical half-angle
    under the REAL f-theta radial map (-> edge f_eff == f_ref).

    Unlike `focal_crop_size` (which assumes a pinhole, ``c = f*size/f_ref``), a
    fisheye's radius is sub-linear in angle, so we invert the real polynomial at
    the canonical half-angle. Plugging the paraxial focal into the pinhole
    formula would over-crop ~8% and land f_eff ~245, not 266.
    """
    r = float(intr.r_of_theta(canonical_halfangle_rad(size, f_ref)))
    c = int(round(2.0 * r))
    return max(32, min(c, min(intr.height, intr.width)))


def ftheta_crop_resize(vid: Tensor, intr: FThetaIntrinsics, size: int = 256,
                       f_ref: float = F_REF) -> Tensor:
    """[T,3,H,W] uint8/float -> [T,3,size,size] uint8, f-theta-correct canonical.

    Geometric-center square crop (side `ftheta_crop_size`) retaining the shared
    canonical half-angle, then bilinear resize. Geometric-center (not principal-
    point) matches comma2k19's convention and stays in-frame for BOTH PhysicalAI
    rigs; horizon/principal-point (extrinsic) normalization is the deferred
    D-016 R1 step. Achieved edge-referenced f_eff is stored in `.last_f_eff`
    (measured by round-tripping the integer crop through the real poly, not
    assumed), so a data card / build check reports the TRUE value.
    """
    t, _, h, w = vid.shape
    # intrinsics are defined on the native sensor; scale the crop if the decoded
    # frame differs in resolution (front_wide mp4 decodes at native 1920x1080).
    sx = w / float(intr.width)
    sy = h / float(intr.height)
    c_native = ftheta_crop_size(intr, size, f_ref)
    c = int(round(c_native * min(sx, sy)))
    c = max(32, min(c, min(h, w)))
    top, left = (h - c) // 2, (w - c) // 2
    out = vid[..., top:top + c, left:left + c].float()
    out = F.interpolate(out, size=(size, size), mode="bilinear",
                        align_corners=False)
    theta_edge = intr.theta_of_r((c / min(sx, sy)) / 2.0)
    ftheta_crop_resize.last_f_eff = (size / 2.0) / math.tan(theta_edge)
    return out.clamp(0, 255).to(torch.uint8)


def ftheta_feff_report(intr: FThetaIntrinsics, size: int = 256,
                       f_ref: float = F_REF) -> dict:
    """Achieved edge f_eff of the corrected crop vs the f_eff the OLD nominal-
    pinhole path silently produced. Regression guard + data-card provenance."""
    c_after = ftheta_crop_size(intr, size, f_ref)
    th_after = intr.theta_of_r(c_after / 2.0)
    f_after = (size / 2.0) / math.tan(th_after)

    f_nom = nominal_focal_from_hfov(intr.width, PHYSICALAI_FRONT_WIDE_HFOV_DEG)
    c_before = focal_crop_size(f_nom, intr.height, intr.width, size, f_ref)
    th_before = intr.theta_of_r(c_before / 2.0)
    f_before = (size / 2.0) / math.tan(th_before)
    return {
        "f_eff_after": round(f_after, 2),
        "f_eff_before_nominal": round(f_before, 2),
        "crop_side_after": c_after, "crop_side_before": c_before,
        "retained_hfov_after_deg": round(math.degrees(2 * th_after), 2),
        "retained_hfov_before_deg": round(math.degrees(2 * th_before), 2),
        "paraxial_focal_px": round(intr.paraxial_focal, 2),
        "nominal_pinhole_focal_px": round(f_nom, 2),
    }


def ftheta_undistort_grid(intr: FThetaIntrinsics, size: int = 256,
                          f_ref: float = F_REF,
                          device: str | torch.device = "cpu") -> Tensor:
    """`grid_sample` grid [1,size,size,2] mapping an ideal pinhole (focal f_ref,
    centered on the optical axis) back onto the f-theta native frame.

    This is the fully-rectilinear option for the deferred D-016 R1 step. It is
    NOT the default cache build: 77% of PhysicalAI clips (rig B) have the
    principal point low in the frame (cy~754), so a pinhole ray fan centered on
    the optical axis samples below the native bottom edge (near-field road out of
    frame). The default `ftheta_crop_resize` avoids that. Kept + tested so the
    f-theta forward map is executable and R1 can adopt it with per-rig extrinsics.
    """
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32), indexing="ij")
    x = xs - (size - 1) / 2.0
    y = ys - (size - 1) / 2.0
    rho = torch.sqrt(x * x + y * y)                       # rectilinear radius px
    theta = torch.atan2(rho, torch.full_like(rho, float(f_ref)))
    r = intr.r_of_theta(theta)                            # native fisheye radius
    scale = torch.where(rho > 1e-6, r / rho, torch.zeros_like(rho))
    u = intr.cx + x * scale
    v = intr.cy + y * scale
    gx = u / (intr.width - 1) * 2.0 - 1.0
    gy = v / (intr.height - 1) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1).unsqueeze(0).to(device)


def ftheta_undistort(vid: Tensor, intr: FThetaIntrinsics, size: int = 256,
                     f_ref: float = F_REF) -> Tensor:
    """[T,3,H,W] -> [T,3,size,size] uint8 true rectilinear pinhole (f_eff=f_ref),
    fisheye distortion removed. See `ftheta_undistort_grid` for why this is the
    R1 option, not the default. f_eff == f_ref exactly by construction."""
    grid = ftheta_undistort_grid(intr, size, f_ref, device=vid.device)
    out = F.grid_sample(vid.float(), grid.expand(vid.shape[0], -1, -1, -1),
                        mode="bilinear", padding_mode="border",
                        align_corners=False)
    ftheta_undistort.last_f_eff = float(f_ref)
    return out.clamp(0, 255).to(torch.uint8)
