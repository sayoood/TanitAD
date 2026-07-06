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
