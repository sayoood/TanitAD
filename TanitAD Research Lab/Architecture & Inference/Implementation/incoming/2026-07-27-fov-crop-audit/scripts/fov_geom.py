"""Rectangular f-theta crop at an ARBITRARY horizontal field — the generalisation of
`calib.ftheta_crop_resize`, plus the two degradation controls the sweep needs.

`calib.ftheta_crop_resize` crops a SQUARE of side `2*r(canonical_half_angle)` about the per-clip
principal point. Two things have to generalise for the FOV x resolution x aspect question:

  1. the half-angle becomes a free parameter (`hfov_deg`), and
  2. the crop becomes a RECTANGLE whose aspect matches the OUTPUT aspect, so a 256x640 letterbox
     keeps the same angular scale on both axes instead of squashing the image.

Because the f-theta radial map is isotropic in the image plane, a rectangle of half-width
`Wpx = r(hfov/2)` and half-height `Hpx = Wpx * out_h/out_w` about `(cx, cy)` retains a horizontal
half-angle of exactly `hfov/2` and a vertical half-angle of `theta_of_r(Hpx)` — which is REPORTED,
never assumed, because at these widths the fisheye is far from paraxial and the pinhole formula
`2*atan((h/2)/f_eff)` is simply wrong.

⭐ **The padding cost is a first-class output.** The sensor is 1920x1080; a crop wide enough to
reach the periphery MUST spill past the top/bottom edge, and those rows are replicate-padded —
pixels the sensor never captured. `crop_geometry` returns the exact padded fraction per clip, and
the rig split makes it asymmetric (rig B's principal point sits ~215 px lower).

The two controls:
  `blur_like`  — render the CANONICAL 51.4 deg field at another arm's native-pixel sampling density
                 and resample back, so the resolution loss is imposed with ZERO change of field.
  `pad_like`   — impose another arm's replicate-padded row fractions on the canonical field.
Their composition is the matched-degradation control the primary contrast is measured against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def poly_r(poly, th):
    r = 0.0 if isinstance(th, float) else torch.zeros_like(th)
    for c in reversed(poly):
        r = r * th + c
    return r


def theta_of_r(poly, r_target: float, hi: float = 2.2) -> float:
    lo, high = 0.0, hi
    for _ in range(80):
        mid = 0.5 * (lo + high)
        if float(poly_r(poly, mid)) < r_target:
            lo = mid
        else:
            high = mid
    return 0.5 * (lo + high)


@dataclass(frozen=True)
class CropGeom:
    out_h: int
    out_w: int
    hfov_deg: float
    vfov_deg: float           # MEASURED through the real radial map, not a pinhole formula
    half_w_px: float          # native half-width of the crop rectangle
    half_h_px: float
    top: int
    left: int
    pad_top: int
    pad_bot: int
    pad_left: int
    pad_right: int
    native_px_per_out_px: float
    px_per_deg: float
    pad_frac_rows: float
    pad_frac_cols: float


def crop_geometry(poly, cx: float, cy: float, h: int, w: int,
                  out_h: int, out_w: int, hfov_deg: float) -> CropGeom:
    """The exact crop box + padding ledger for one clip at one arm (decoded frame is h x w)."""
    half_w = float(poly_r(poly, math.radians(hfov_deg / 2.0)))
    half_h = half_w * (out_h / out_w)
    vfov = 2.0 * math.degrees(theta_of_r(poly, half_h))
    # ⚠️ The offset MUST be derived from the ROUNDED box size, not from the real half-side.
    # `calib.ftheta_crop_box` does `c = round(2r)` then `top = round(cy - c/2)`; deriving `top`
    # from the unrounded `r` instead disagrees by ONE PIXEL whenever `cy - r` sits near a .5
    # boundary, which C-FID caught on a real clip. Matching the convention exactly is what makes
    # the baseline arm bit-identical to the deployed crop.
    bh, bw = int(round(2 * half_h)), int(round(2 * half_w))
    top = int(round(cy - bh / 2.0))
    left = int(round(cx - bw / 2.0))
    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bot = max(0, top + bh - h)
    pad_right = max(0, left + bw - w)
    return CropGeom(out_h=out_h, out_w=out_w, hfov_deg=hfov_deg, vfov_deg=vfov,
                    half_w_px=half_w, half_h_px=half_h, top=top, left=left,
                    pad_top=pad_top, pad_bot=pad_bot, pad_left=pad_left, pad_right=pad_right,
                    native_px_per_out_px=(2 * half_w) / out_w,
                    px_per_deg=out_w / hfov_deg,
                    pad_frac_rows=(pad_top + pad_bot) / max(bh, 1),
                    pad_frac_cols=(pad_left + pad_right) / max(bw, 1))


def crop_resize(vid: Tensor, g: CropGeom) -> Tensor:
    """[T,3,H,W] uint8/float -> [T,3,out_h,out_w] uint8. Replicate-pads any spill past the edge,
    exactly as `calib.ftheta_crop_resize` does, so the principal point stays at the crop centre."""
    _t, _c, h, w = vid.shape
    bh, bw = int(round(2 * g.half_h_px)), int(round(2 * g.half_w_px))
    y0, y1 = max(0, g.top), min(h, g.top + bh)
    x0, x1 = max(0, g.left), min(w, g.left + bw)
    out = vid[..., y0:y1, x0:x1].float()
    pt, pb = y0 - g.top, (g.top + bh) - y1
    pl, pr = x0 - g.left, (g.left + bw) - x1
    if pt or pb or pl or pr:
        out = F.pad(out, (pl, pr, pt, pb), mode="replicate")
    out = F.interpolate(out, size=(g.out_h, g.out_w), mode="bilinear", align_corners=False)
    return out.clamp(0, 255).to(torch.uint8)


def blur_like(img: Tensor, ref_native_px_per_out_px: float, own: float) -> Tensor:
    """Impose ANOTHER arm's sampling density on this arm's image, field unchanged.

    `own`/`ref` are native-sensor px per output px. If the reference samples `k = ref/own` times
    more coarsely, this renders at `1/k` of the output size and resamples back — the same
    information loss with the same interpolation kernel, and no change of field whatsoever.
    """
    k = ref_native_px_per_out_px / own
    if k <= 1.0:
        return img
    h, w = img.shape[-2:]
    sh, sw = max(8, int(round(h / k))), max(8, int(round(w / k)))
    x = img.float()
    x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False, antialias=True)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.clamp(0, 255).to(torch.uint8)


def pad_like(img: Tensor, frac_top: float, frac_bot: float) -> Tensor:
    """Impose another arm's replicate-padded row extents on this image (field unchanged)."""
    h = img.shape[-2]
    nt, nb = int(round(frac_top * h)), int(round(frac_bot * h))
    if nt <= 0 and nb <= 0:
        return img
    x = img.clone()
    if nt > 0:
        nt = min(nt, h - 1)
        x[..., :nt, :] = x[..., nt:nt + 1, :]
    if nb > 0:
        nb = min(nb, h - 1 - max(nt, 0))
        if nb > 0:
            x[..., h - nb:, :] = x[..., h - nb - 1:h - nb, :]
    return x
