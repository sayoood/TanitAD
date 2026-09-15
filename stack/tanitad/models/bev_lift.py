"""BEV lift for refcv6-grounded (``Project Steering/REFCV6_DESIGN_GROUNDED.md`` §2.2).

Samples REF-C's stride-16 image feature map at every 120 x 64 @ 0.5 m rig-grid cell
centre, at 4 heights above the road plane, through the clip's own camera pose, into
the 256 x 640 CYLINDRICAL frame the trunk reads. The geometry is exact, per clip,
label-free and parameter-free; the only learned pieces are a 1x1 conv and an
"unobserved" embedding.

⭐ NOTHING GEOMETRIC IS RE-DERIVED HERE. Every crossing reuses the code that built
the pixels or already owns the convention:

* rig -> camera: ``tanitad/data/rig_projection.py:254-260`` (``rig_to_cam``,
  ``p_cam = R^T (p_rig - t)``) with ``R, t`` from the dataset's
  ``calibration/sensor_extrinsics`` quaternion via
  ``physicalai.FrontWideExtrinsics.rotation_cam_to_vehicle``
  (``tanitad/data/physicalai.py:258-265``) and ``RigCamera.from_extrinsics``
  (``rig_projection.py:218-232``). Camera frame: +x right, +y down, +z boresight.
* camera -> cache pixel: ``rig_projection.project_cam_to_frame``
  (``rig_projection.py:272-314``), ``col = (W-1)/2 + f_ref*atan2(x, z)``,
  ``row = (H-1)/2 + f_ref*y/hypot(x, z)``, the exact inverse of
  ``calib.cylindrical_rays`` (``tanitad/data/calib.py:906-924``), which is the ray
  fan ``calib.cylindrical_rectify`` (``calib.py:937-998``) resampled the fisheye
  with when ``v2_compressed.build_compressed`` built the cache
  (``stack/scripts/v2_compressed.py:45-50``). The boresight sits at the frame
  centre for BOTH rigs (cy ~543 / ~755): the per-clip cy is inside the frame by
  construction, the per-clip pitch comes from the extrinsic.
  In continuous pixel coordinates (pixel i spans [i, i+1)) this is the design's
  ``az = (u - W/2)/f_ref``; in index coordinates it is ``(col - (W-1)/2)/f_ref``.
* the grid: ``tanitad.data.bev_raster.cell_centers_xy(GRID_DEFAULT)``, the same
  cells as the SAM3 GT (row 0 = x in [0, 0.5 m), col 0 = y = -16 m RIGHT, +y LEFT).

Validity (per height): IN FRONT of the camera (z_cam > 0), inside the 120° field
(|azimuth| <= W/(2 f_ref), i.e. col in [-0.5, W-0.5]), inside the image rows (row
in [-0.5, H-0.5]); optionally AND an observed-pixel mask of the cache frame (rig B
leaves ~8.9 % of the 256 x 640 frame black, ``calib.py:1245-1248``).

Feature-map coordinates. REF-C's ``ResNetEncoder`` (``tanitad/refs/refc.py``:
stem 7x7/2 pad 3 + maxpool 3x3/2 pad 1 at 1240-1243, BasicBlock 3x3/stride pad 1
+ 1x1/stride shortcut at 1204-1213) maps feature index ``k`` of every stride to
image pixel index ``stride * k``: every downsampling layer has
``padding == (kernel-1)/2``, so its output i is centred on input 2i
(:func:`receptive_field_centre` recomputes this from the module objects, and the
test pins it with a forward impulse). Hence the fractional feature index of pixel
``col`` is ``(col - offset)/stride`` with ``offset = 0`` for REF-C -- NOT the
half-cell-shifted ``(col + 0.5)/stride - 0.5`` a symmetric tiling would assume
(7.5 px = 1.41° at stride 16, ~0.5 m lateral at 20 m, one BEV cell). The grid is
normalised for ``F.grid_sample(..., align_corners=False)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.data.bev_raster import GRID_DEFAULT, BEVGrid, cell_centers_xy
from tanitad.data.calib import PHYSICALAI_WIDE120_256x640, CanonicalFrame
from tanitad.data.rig_projection import RigCamera, project_cam_to_frame, rig_to_cam

__all__ = [
    "HEIGHTS_M", "REFC_FEATURE_CENTRE_OFFSET_PX", "LiftGeometry", "BEVLift",
    "camera_from_extrinsics", "project_rig_points", "pixel_to_feature_grid",
    "build_lift_geometry", "receptive_field_centre", "refc_trunk_path",
]

#: heights above the road plane (rig z = 0 is the road, rig_projection.py:40-47)
HEIGHTS_M: tuple[float, ...] = (0.0, 0.5, 1.5, 2.5)
#: image pixel index of REF-C feature index 0, for every stride (see module doc)
REFC_FEATURE_CENTRE_OFFSET_PX: float = 0.0


def camera_from_extrinsics(extr, frame: CanonicalFrame = PHYSICALAI_WIDE120_256x640
                           ) -> RigCamera:
    """``RigCamera`` for a clip from anything ``RigCamera.from_extrinsics`` takes
    (``physicalai.FrontWideExtrinsics``), or a mapping with qx qy qz qw x y z."""
    if isinstance(extr, RigCamera):
        if extr.frame != frame:
            raise ValueError(f"RigCamera renders into {extr.frame.tag()}, the lift "
                             f"was asked for {frame.tag()}")
        return extr
    if isinstance(extr, dict) or hasattr(extr, "keys"):
        from tanitad.data.physicalai import FrontWideExtrinsics
        extr = FrontWideExtrinsics(**{k: float(extr[k]) for k in
                                      ("qx", "qy", "qz", "qw", "x", "y", "z")})
    return RigCamera.from_extrinsics(extr, frame)


def project_rig_points(p_rig: Tensor, cam: RigCamera) -> dict:
    """Rig points ``[..., 3]`` -> cache-frame pixel ``col``/``row`` (index
    coordinates) and the three validity conditions, all ``[...]``."""
    frame = cam.frame
    if frame.projection != "cylindrical":
        raise ValueError(f"the lift is defined on the cylindrical cache frame, got "
                         f"{frame.projection!r}")
    p_cam = rig_to_cam(p_rig, cam.R_cam_to_rig, cam.t_cam_in_rig)
    col, row, _ = project_cam_to_frame(p_cam, frame)
    in_front = p_cam[..., 2] > 0
    in_hfov = (col >= -0.5) & (col <= frame.width - 0.5)
    in_rows = (row >= -0.5) & (row <= frame.height - 0.5)
    finite = torch.isfinite(col) & torch.isfinite(row)
    return {"col": col, "row": row, "in_front": in_front, "in_hfov": in_hfov & finite,
            "in_rows": in_rows & finite,
            "valid": in_front & in_hfov & in_rows & finite}


def pixel_to_feature_grid(col: Tensor, row: Tensor, frame: CanonicalFrame,
                          stride: int,
                          centre_offset_px: float = REFC_FEATURE_CENTRE_OFFSET_PX
                          ) -> Tensor:
    """Cache pixel (index coords) -> ``grid_sample`` coordinates on the stride-
    ``stride`` feature map, ``align_corners=False``; returns ``[..., 2]`` = (x, y)."""
    if frame.width % stride or frame.height % stride:
        raise ValueError(f"{frame.height}x{frame.width} is not divisible by stride "
                         f"{stride}: the feature map would not tile the frame")
    h, w = frame.height // stride, frame.width // stride
    kx = (col - centre_offset_px) / stride
    ky = (row - centre_offset_px) / stride
    return torch.stack([(2.0 * kx + 1.0) / w - 1.0, (2.0 * ky + 1.0) / h - 1.0], dim=-1)


@dataclass(frozen=True)
class LiftGeometry:
    """Per-clip lift geometry. ``Z`` heights x the ``X x Y`` rig grid."""

    grid: Tensor              # [Z, X, Y, 2] float32, grid_sample (x, y); 0 where invalid
    valid: Tensor             # [Z, X, Y] bool
    col: Tensor               # [Z, X, Y] float64 cache pixel column (index coords)
    row: Tensor               # [Z, X, Y] float64 cache pixel row
    in_front: Tensor          # [Z, X, Y] bool
    in_hfov: Tensor           # [Z, X, Y] bool
    in_rows: Tensor           # [Z, X, Y] bool
    heights_m: tuple
    stride: int
    feat_hw: tuple
    frame: CanonicalFrame
    centre_offset_px: float

    @property
    def cell_valid(self) -> Tensor:
        """[X, Y] bool: at least one height projects validly."""
        return self.valid.any(dim=0)


def build_lift_geometry(extr, *, frame: CanonicalFrame = PHYSICALAI_WIDE120_256x640,
                        stride: int = 16, heights_m=HEIGHTS_M,
                        grid: BEVGrid = GRID_DEFAULT,
                        centre_offset_px: float = REFC_FEATURE_CENTRE_OFFSET_PX,
                        observed: Tensor | None = None) -> LiftGeometry:
    """The lift geometry for ONE clip.

    ``extr``: the clip's front-wide extrinsics (``FrontWideExtrinsics``, a mapping
    with qx qy qz qw x y z, or a ``RigCamera`` on ``frame``). ``observed``: optional
    ``[H, W]`` bool observed-pixel mask of the cache frame (e.g.
    ``calib.cylindrical_grid(intr, intr.height, intr.width, frame)[1]``); a sample
    is then valid only if its nearest pixel is observed."""
    cam = camera_from_extrinsics(extr, frame)
    hs = tuple(float(h) for h in heights_m)
    X, Y = cell_centers_xy(grid)                                   # [nx, ny] m
    xy = torch.stack([torch.as_tensor(X), torch.as_tensor(Y)], dim=-1).to(torch.float64)
    nz = len(hs)
    pts = torch.cat([xy.unsqueeze(0).expand(nz, -1, -1, -1),
                     torch.tensor(hs, dtype=torch.float64).view(nz, 1, 1, 1)
                     .expand(nz, xy.shape[0], xy.shape[1], 1)], dim=-1)
    pr = project_rig_points(pts, cam)
    valid = pr["valid"]
    if observed is not None:
        obs = torch.as_tensor(observed, dtype=torch.bool)
        if tuple(obs.shape) != (frame.height, frame.width):
            raise ValueError(f"observed mask {tuple(obs.shape)} != frame "
                             f"{(frame.height, frame.width)}")
        ci = torch.where(valid, pr["col"], torch.zeros_like(pr["col"])).round().long()
        ri = torch.where(valid, pr["row"], torch.zeros_like(pr["row"])).round().long()
        ci = ci.clamp(0, frame.width - 1)
        ri = ri.clamp(0, frame.height - 1)
        valid = valid & obs[ri, ci]
    g = pixel_to_feature_grid(pr["col"], pr["row"], frame, stride, centre_offset_px)
    g = torch.where(valid.unsqueeze(-1), g, torch.zeros_like(g)).to(torch.float32)
    return LiftGeometry(grid=g, valid=valid, col=pr["col"], row=pr["row"],
                        in_front=pr["in_front"], in_hfov=pr["in_hfov"],
                        in_rows=pr["in_rows"], heights_m=hs, stride=int(stride),
                        feat_hw=(frame.height // stride, frame.width // stride),
                        frame=frame, centre_offset_px=float(centre_offset_px))


# --------------------------------------------------------------------------- #
# receptive-field centre of a conv path (pins the feature-coordinate convention)
# --------------------------------------------------------------------------- #
def receptive_field_centre(layers) -> tuple[int, float]:
    """``(jump, start)``: output index ``k`` of the path is centred on input pixel
    index ``start + jump * k``. ``layers`` = Conv2d / MaxPool2d / AvgPool2d in
    forward order along ONE path (horizontal axis)."""
    jump, start = 1, 0.0
    for m in layers:
        if not isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            raise TypeError(f"unsupported layer {type(m).__name__}")
        k = m.kernel_size[1] if isinstance(m.kernel_size, tuple) else m.kernel_size
        s = m.stride[1] if isinstance(m.stride, tuple) else m.stride
        p = m.padding[1] if isinstance(m.padding, tuple) else m.padding
        d = getattr(m, "dilation", 1)
        d = d[1] if isinstance(d, tuple) else d
        if isinstance(p, str):
            raise ValueError(f"string padding {p!r} not supported")
        start = start + ((k - 1) * d / 2.0 - p) * jump
        jump = jump * s
    return int(jump), float(start)


def refc_trunk_path(encoder, n_stages: int, shortcut: bool = False) -> list:
    """The conv/pool layers of REF-C ``ResNetEncoder`` along the main path (or the
    1x1 shortcut path of the first block of each stage) up to ``n_stages``."""
    layers = [m for m in encoder.stem if isinstance(m, (nn.Conv2d, nn.MaxPool2d))]
    for stage in list(encoder.stages)[:n_stages]:
        for bi, blk in enumerate(stage):
            if shortcut and bi == 0 and blk.down is not None:
                layers += [m for m in blk.down if isinstance(m, nn.Conv2d)]
            else:
                layers += [blk.conv1, blk.conv2]
    return layers


# --------------------------------------------------------------------------- #
# the module                                                                   #
# --------------------------------------------------------------------------- #
class BEVLift(nn.Module):
    """Multi-height bilinear sampling of an image feature map into the BEV grid.

    ``forward(fmap [B,C,h,w], grid [B,Z,X,Y,2], valid [B,Z,X,Y]) -> [B,d_out,X,Y]``:
    sample every height (invalid samples zeroed), concatenate over heights
    (height-major channel order), 1x1 conv to ``d_out``, and add a learned
    ``unobserved`` embedding on cells where NO height is valid. Parameters:
    ``d_in * Z * d_out + d_out`` (conv) ``+ d_out`` (embedding)."""

    def __init__(self, d_in: int = 352, d_out: int = 128, n_heights: int = len(HEIGHTS_M),
                 feat_hw: tuple | None = (16, 40)):
        super().__init__()
        self.d_in, self.d_out, self.n_heights = int(d_in), int(d_out), int(n_heights)
        self.feat_hw = None if feat_hw is None else (int(feat_hw[0]), int(feat_hw[1]))
        self.proj = nn.Conv2d(self.d_in * self.n_heights, self.d_out, kernel_size=1)
        self.unobserved = nn.Parameter(torch.empty(self.d_out))
        nn.init.normal_(self.unobserved, std=0.02)

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def forward(self, fmap: Tensor, grid: Tensor, valid: Tensor) -> Tensor:
        B, C, h, w = fmap.shape
        if C != self.d_in:
            raise ValueError(f"feature map has {C} channels, BEVLift expects {self.d_in}")
        if self.feat_hw is not None and (h, w) != self.feat_hw:
            raise ValueError(f"feature map is {h}x{w}, the lift geometry was built for "
                             f"{self.feat_hw[0]}x{self.feat_hw[1]} (wrong stride?)")
        if grid.dim() != 5 or grid.shape[0] != B or grid.shape[1] != self.n_heights \
                or grid.shape[-1] != 2:
            raise ValueError(f"grid must be [B={B}, Z={self.n_heights}, X, Y, 2], got "
                             f"{tuple(grid.shape)}")
        _, Z, X, Y, _ = grid.shape
        if tuple(valid.shape) != (B, Z, X, Y):
            raise ValueError(f"valid must be {(B, Z, X, Y)}, got {tuple(valid.shape)}")
        vm = valid.to(fmap.dtype)
        g = torch.where(valid.unsqueeze(-1), grid.to(fmap.dtype), torch.zeros_like(grid, dtype=fmap.dtype))
        s = F.grid_sample(fmap, g.reshape(B, Z * X, Y, 2), mode="bilinear",
                          padding_mode="border", align_corners=False)   # [B,C,Z*X,Y]
        s = s.reshape(B, C, Z, X, Y) * vm.unsqueeze(1)
        s = s.permute(0, 2, 1, 3, 4).reshape(B, Z * C, X, Y)
        out = self.proj(s)
        unobs = (~valid.any(dim=1)).to(out.dtype).unsqueeze(1)          # [B,1,X,Y]
        return out + unobs * self.unobserved.view(1, -1, 1, 1)


if __name__ == "__main__":
    lift = BEVLift()
    print(f"BEVLift(d_in=352, d_out=128, heights={HEIGHTS_M}): "
          f"{lift.n_params:,} parameters")
