"""Rig-frame <-> canonical-image geometry for the refcv5 environment head.

WHY THIS MODULE EXISTS
----------------------
refcv5 grounds its decoder on AGENTS, and the agents (``obstacle.offline`` 3D
cuboids) live in the **rig frame** while the model sees a **canonical image
frame**. Every crossing between the two must go through ONE implementation, or
the detector and its evaluation drift apart and the gate measures the drift.

⛔ THE TRAP THIS MODULE IS BUILT AROUND (CLAUDE.md, MEASURED 2026-08-21).
Our corpus is ``256 x 640`` **CYLINDRICAL** with ``f_ref = 305.577``
(``physicalai-b1-w120-256x640cyl``). On a cylindrical projection the image
column is **LINEAR IN AZIMUTH**, so::

    HFOV = 2 * (W/2) / f_ref = 2 * 320 / 305.577 = 2.0944 rad = 120.0 deg

matching the rig's own name ``camera_front_wide_120fov``. The pinhole formula
``2*atan((W/2)/f)`` yields **92.6 deg** and is entirely plausible-looking. Any
projection here therefore reads ``frame.projection`` and never assumes; the
formula lives in exactly one place, keyed on the frame.

FRAMES AND CONVENTIONS (both MEASURED, both pinned by tests)
------------------------------------------------------------
* **RIG / vehicle frame** — ``+x forward, +y LEFT, +z UP``, origin on the rear
  axle at the road plane. This is the frame ``obstacle.offline`` cuboids are
  expressed in (``reference_frame == "rig"`` on every row) and the frame
  ``tanitad.data.bev_raster`` rasterises. The axis convention was MEASURED by
  the parked-car experiment (world-static under xf_yl for 1,756/2,778 tracks,
  7.4x over the nearest alternative).
* **CAMERA frame** — ``+x RIGHT, +y DOWN, +z boresight``. This is the
  convention of :func:`tanitad.data.calib.cylindrical_rays` and of
  :meth:`tanitad.data.physicalai.FrontWideExtrinsics.vehicle_forward_in_cam`,
  so the dataset's ``sensor_extrinsics`` quaternion (cam -> vehicle) plugs in
  with no extra permutation.
* **IMAGE** — ``(row, col)`` in the canonical frame's pixels, with the
  boresight at ``((H-1)/2, (W-1)/2)``. ⚠️ ``(W-1)/2``, not ``W/2``: that is
  what ``calib.cylindrical_rays`` uses to build the resampling grid, so any
  other centre would be off by half a pixel against the actual pixels.

⭐ THE METRIC-SCALE ANCHOR IS SETTLED AND FREE. Over 87,481 ``obstacle.offline``
cuboids the ground-standing classes' bottom faces read **-0.05 to -0.13 m**
while ``protruding_object`` reads **+1.68 m** => **rig z = 0 IS the road
plane**, and the front-wide camera sits at **1.43-1.56 m** above it. A
monocular detector therefore does not have to invent scale: a pixel plus the
road-plane assumption already fixes a metric range
(:func:`ground_intersection`).

WHAT THIS MODULE REFUSES
------------------------
``physicalai.PROJECTION_MODE_CROP`` (``ftheta_crop``) frames keep the sensor's
own f-theta radial warp in the pixels, so a ``CanonicalFrame`` alone does NOT
determine where a ray lands. :func:`project_cam_to_frame` therefore takes only
``"pinhole"`` / ``"cylindrical"`` frames, and a caller holding a crop build must
project through :func:`tanitad.data.calib.ftheta_project_rays` with the clip's
own intrinsics instead. Refusing beats silently returning a plausible column.

Torch throughout (CPU or CUDA): the same functions serve the data-side target
builder and the model-side waypoint sampler, and one implementation is the
point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from tanitad.data.calib import CanonicalFrame

__all__ = [
    "RIG_TO_CAM", "ROAD_PLANE_Z_M", "CAM_HEIGHT_RANGE_M",
    "RigCamera", "rig_to_cam", "cam_to_rig",
    "project_cam_to_frame", "project_rig_to_frame",
    "frame_to_cam_ray", "ground_intersection",
]

#: Nominal rig -> camera rotation: the axis permutation ALONE, i.e. a camera
#: whose boresight is exactly vehicle-forward and whose up is exactly vehicle-up.
#: ``p_cam = RIG_TO_CAM @ p_rig`` with p_rig = (fwd, left, up):
#:     cam_x (right)     = -left
#:     cam_y (down)      = -up
#:     cam_z (boresight) = +fwd
#: ⚠️ This is the NOMINAL mount only. Real clips carry a mount rotation in
#: ``sensor_extrinsics`` and :meth:`RigCamera.from_extrinsics` uses it; this
#: constant exists for tests and for rigs with no extrinsic on file, where the
#: caller must record that the mount pose is nominal.
RIG_TO_CAM: tuple[tuple[float, ...], ...] = (
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
)

#: The road plane in the rig frame, MEASURED (see the module docstring).
ROAD_PLANE_Z_M: float = 0.0

#: Front-wide camera height above the road plane, MEASURED range over the corpus.
CAM_HEIGHT_RANGE_M: tuple[float, float] = (1.43, 1.56)

_EPS = 1e-9

#: Sub-pixel slack on the in-frame test. A ray reconstructed from the frame's
#: OWN corner pixel round-trips to ``-1.1e-13`` rather than exactly ``0.0``
#: (atan2(sin u, cos u) is not bit-exactly u), so a strict ``col >= 0`` marks
#: the corpus frame's border column INVALID — MEASURED here, and it would
#: silently drop the periphery where the interesting agents are. This is float
#: slack, not a geometry decision: at 1e-6 px it can never admit a point that
#: is actually outside.
_PIX_EPS = 1e-6


def _as_t(x, ref: Tensor | None = None) -> Tensor:
    t = torch.as_tensor(x)
    if ref is not None:
        t = t.to(device=ref.device, dtype=ref.dtype)
    elif not t.is_floating_point():
        t = t.to(torch.float64)
    return t


@dataclass(frozen=True)
class RigCamera:
    """A camera's pose in the rig frame + the canonical frame it renders into.

    ``R_cam_to_rig`` [3, 3] maps a CAMERA-frame vector to the RIG frame (this is
    exactly the dataset's ``sensor_extrinsics`` quaternion, whose own name in
    ``physicalai.FrontWideExtrinsics`` is ``rotation_cam_to_vehicle``).
    ``t_cam_in_rig`` [3] is the camera's position in the rig frame — for the
    front-wide, ``z`` is the 1.43-1.56 m mount height above the road plane.

    ``frame`` is the CanonicalFrame the pixels were resampled into. It carries
    the projection, so no call site here decides between the pinhole and the
    cylindrical formula.
    """

    R_cam_to_rig: Tensor
    t_cam_in_rig: Tensor
    frame: CanonicalFrame

    def __post_init__(self):
        R = _as_t(self.R_cam_to_rig)
        t = _as_t(self.t_cam_in_rig, R)
        if R.shape != (3, 3):
            raise ValueError(f"R_cam_to_rig must be [3, 3], got {tuple(R.shape)}")
        if t.reshape(-1).shape != (3,):
            raise ValueError(f"t_cam_in_rig must be [3], got {tuple(t.shape)}")
        # Orthonormality is asserted, not assumed: a quaternion read from a
        # parquet with a swapped column order still normalises to something
        # unit-length, and the only thing that catches it is R R^T == I.
        err = float((R @ R.transpose(-1, -2)
                     - torch.eye(3, dtype=R.dtype, device=R.device)).abs().max())
        if err > 1e-4:
            raise ValueError(f"R_cam_to_rig is not a rotation (|RR^T - I|_max "
                             f"= {err:.2e}); check the quaternion column order")
        object.__setattr__(self, "R_cam_to_rig", R)
        object.__setattr__(self, "t_cam_in_rig", t.reshape(3))

    # -- constructors ------------------------------------------------------ #
    @classmethod
    def nominal(cls, frame: CanonicalFrame, height_m: float = 1.5,
                x_m: float = 1.5, y_m: float = 0.0,
                dtype: torch.dtype = torch.float64) -> "RigCamera":
        """A boresight-forward camera at ``(x_m, y_m, height_m)`` in the rig.

        ⚠️ NOMINAL — no mount pitch. Real clips have one; use
        :meth:`from_extrinsics`. This exists so a test (and only a test) has a
        camera whose answers are computable by hand.
        """
        R_rig_to_cam = torch.tensor(RIG_TO_CAM, dtype=dtype)
        return cls(R_cam_to_rig=R_rig_to_cam.transpose(0, 1),
                   t_cam_in_rig=torch.tensor([x_m, y_m, height_m], dtype=dtype),
                   frame=frame)

    @classmethod
    def from_extrinsics(cls, extr, frame: CanonicalFrame,
                        dtype: torch.dtype = torch.float64) -> "RigCamera":
        """From a :class:`physicalai.FrontWideExtrinsics` (or anything exposing
        ``rotation_cam_to_vehicle()`` and ``x/y/z``).

        This is the ONLY admissible constructor for corpus work: the mount pitch
        is what puts the horizon on the right row, and the two-rig fix
        (``cy ~ 543`` vs ``cy ~ 755``) is already inside the frame the pixels
        were resampled into.
        """
        R = torch.as_tensor(extr.rotation_cam_to_vehicle(), dtype=dtype)
        t = torch.tensor([float(extr.x), float(extr.y), float(extr.z)],
                         dtype=dtype)
        return cls(R_cam_to_rig=R, t_cam_in_rig=t, frame=frame)

    # -- geometry ---------------------------------------------------------- #
    def to_cam(self, p_rig: Tensor) -> Tensor:
        """Rig-frame points ``[..., 3]`` -> camera frame ``[..., 3]``."""
        return rig_to_cam(p_rig, self.R_cam_to_rig, self.t_cam_in_rig)

    def to_rig(self, p_cam: Tensor) -> Tensor:
        """Camera-frame points ``[..., 3]`` -> rig frame ``[..., 3]``."""
        return cam_to_rig(p_cam, self.R_cam_to_rig, self.t_cam_in_rig)

    def project(self, p_rig: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Rig points ``[..., 3]`` -> ``(col, row, valid)`` in ``self.frame``."""
        return project_cam_to_frame(self.to_cam(p_rig), self.frame)

    def ground_intersection(self, col: Tensor, row: Tensor,
                            z_plane: float = ROAD_PLANE_Z_M
                            ) -> tuple[Tensor, Tensor]:
        """Back-project pixels onto the rig-frame plane ``z = z_plane``."""
        return ground_intersection(col, row, self, z_plane=z_plane)


def rig_to_cam(p_rig: Tensor, R_cam_to_rig: Tensor, t_cam_in_rig: Tensor
               ) -> Tensor:
    """``p_cam = R^T (p_rig - t)`` — rig-frame points into the camera frame."""
    p = _as_t(p_rig)
    R = _as_t(R_cam_to_rig, p)
    t = _as_t(t_cam_in_rig, p).reshape(3)
    return (p - t) @ R                      # (p - t) @ R  ==  (R^T (p - t))^T


def cam_to_rig(p_cam: Tensor, R_cam_to_rig: Tensor, t_cam_in_rig: Tensor
               ) -> Tensor:
    """``p_rig = R p_cam + t`` — the exact inverse of :func:`rig_to_cam`."""
    p = _as_t(p_cam)
    R = _as_t(R_cam_to_rig, p)
    t = _as_t(t_cam_in_rig, p).reshape(3)
    return p @ R.transpose(-1, -2) + t


def project_cam_to_frame(p_cam: Tensor, frame: CanonicalFrame
                         ) -> tuple[Tensor, Tensor, Tensor]:
    """Camera-frame points ``[..., 3]`` -> ``(col, row, valid)``.

    ``col`` / ``row`` are float pixel coordinates in ``frame`` (bilinear-ready);
    ``valid`` is True only where the point is in FRONT and lands inside the
    frame. The formula is selected by ``frame.projection`` — see the module
    docstring for why hard-coding either one is a measured error.

    * ``cylindrical`` (ours): ``phi = atan2(x, z)``, ``col = (W-1)/2 + f*phi``,
      ``row = (H-1)/2 + f*y/hypot(x, z)`` — the exact inverse of
      ``calib.cylindrical_rays``.
    * ``pinhole``: ``col = (W-1)/2 + f*x/z``, ``row = (H-1)/2 + f*y/z``.
    """
    p = _as_t(p_cam)
    if p.shape[-1] != 3:
        raise ValueError(f"p_cam must be [..., 3], got {tuple(p.shape)}")
    if frame.projection not in ("pinhole", "cylindrical"):
        raise ValueError(
            f"project_cam_to_frame cannot handle projection "
            f"{frame.projection!r}. An `ftheta_crop` build keeps the sensor's "
            f"radial warp in the pixels, so a CanonicalFrame alone does not "
            f"determine the column — project through "
            f"`calib.ftheta_project_rays` with the clip's own intrinsics.")
    f = float(frame.f_ref)
    cx = (frame.width - 1) / 2.0
    cy = (frame.height - 1) / 2.0
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    if frame.projection == "cylindrical":
        rho = torch.sqrt(x * x + z * z)
        phi = torch.atan2(x, z)
        col = cx + f * phi
        row = cy + f * y / rho.clamp_min(_EPS)
        ahead = rho > _EPS
    else:
        zc = z.clamp_min(_EPS)
        col = cx + f * x / zc
        row = cy + f * y / zc
        ahead = z > _EPS
    valid = (ahead
             & (col >= -_PIX_EPS) & (col <= frame.width - 1 + _PIX_EPS)
             & (row >= -_PIX_EPS) & (row <= frame.height - 1 + _PIX_EPS))
    return col, row, valid


def project_rig_to_frame(p_rig: Tensor, cam: RigCamera
                         ) -> tuple[Tensor, Tensor, Tensor]:
    """Convenience: rig points ``[..., 3]`` -> ``(col, row, valid)``."""
    return cam.project(p_rig)


def frame_to_cam_ray(col: Tensor, row: Tensor, frame: CanonicalFrame) -> Tensor:
    """``(col, row)`` -> a camera-frame direction ``[..., 3]`` (unnormalised).

    The exact inverse of :func:`project_cam_to_frame`, and the tensor twin of
    ``calib.cylindrical_rays`` evaluated at arbitrary (possibly fractional)
    pixels rather than on the full grid.
    """
    c = _as_t(col)
    r = _as_t(row, c)
    f = float(frame.f_ref)
    u = (c - (frame.width - 1) / 2.0) / f
    v = (r - (frame.height - 1) / 2.0) / f
    if frame.projection == "cylindrical":
        return torch.stack([torch.sin(u), v, torch.cos(u)], dim=-1)
    if frame.projection == "pinhole":
        return torch.stack([u, v, torch.ones_like(u)], dim=-1)
    raise ValueError(f"frame_to_cam_ray cannot handle projection "
                     f"{frame.projection!r} (see project_cam_to_frame)")


def ground_intersection(col: Tensor, row: Tensor, cam: RigCamera,
                        z_plane: float = ROAD_PLANE_Z_M
                        ) -> tuple[Tensor, Tensor]:
    """Where a pixel's ray meets the rig-frame plane ``z = z_plane``.

    Returns ``(p_rig [..., 3], hits [...])``. ``hits`` is False where the ray is
    parallel to the plane or points away from it (above the horizon) — a
    horizon-crossing ray has NO ground intersection and returning a huge number
    silently is how a detector learns to place cars at 400 m.

    ⭐ This is the free metric-scale anchor: rig ``z = 0`` IS the road plane
    (MEASURED over 87,481 cuboids), so a ground-standing agent's image FOOT
    already fixes its range without any learned depth.
    """
    d_cam = frame_to_cam_ray(col, row, cam.frame)              # [..., 3]
    R = cam.R_cam_to_rig.to(d_cam.dtype)
    d_rig = d_cam @ R.transpose(-1, -2)                        # [..., 3]
    t = cam.t_cam_in_rig.to(d_cam.dtype)
    denom = d_rig[..., 2]
    # The camera is ABOVE the plane, so a ray that reaches it must go DOWN.
    s = (float(z_plane) - t[2]) / torch.where(denom.abs() > _EPS, denom,
                                              torch.full_like(denom, _EPS))
    hits = (denom.abs() > 1e-6) & (s > 0)
    p = t + s.unsqueeze(-1) * d_rig
    return p, hits
