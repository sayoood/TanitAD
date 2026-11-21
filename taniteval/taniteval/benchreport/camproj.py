"""Project ego-frame ground points onto the programme's stitched 256x640 CYLINDRICAL frame.

⚠️ THE PROJECTION IS CYLINDRICAL, NOT PINHOLE. The column is LINEAR IN AZIMUTH
(``u = f·φ + (W−1)/2``); the pinhole formula applied here reads ~92.6° for a 120° rig (CLAUDE.md,
cylindrical-FOV trap). The model here is the EXACT inverse of the ray model the frame bank was built
with — ``tanitad.data.calib.cylindrical_rays``: the ray of pixel (u, v) is ``(sin φ, y_n, cos φ)`` with
``φ = (u − (W−1)/2)/f_ref`` and ``y_n = (v − (H−1)/2)/f_ref``, in the canonical basis
``M = [right, down, fwd]`` built from cam_f0's boresight (``build_navsim_eval.py::build_map``).

Scope, stated because a projection quoted outside it is wrong:
* the bank's stitch is ROTATION-ONLY (one panorama from three cameras' rays, no parallax correction),
  so the exact optical centre is cam_f0's (``sensor2lidar_translation``); side-camera regions carry the
  parallax of their own centre. The plan sits mostly in the front camera's cone.
* NavSim/OpenScene ``lidar2ego`` is the IDENTITY (MEASURED in the log metadata: translation [0,0,0],
  rotation [1,0,0,0]), so ego-frame points ARE lidar-frame points, and z = 0 is the ground plane
  (CAM_F0 sits at z = 1.52 m, matching a ground-level rear-axle origin).

Pure numpy + stdlib, Python 3.9 compatible: the NavSim-venv renderer imports this file directly.
"""
from __future__ import annotations

import numpy as np


def canonical_basis(sensor2lidar_rotation) -> np.ndarray:
    """cam_f0's rotation -> ``M`` (columns right, down, fwd): canonical -> lidar/ego."""
    R = np.asarray(sensor2lidar_rotation, dtype=np.float64)
    fwd = R @ np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    down /= np.linalg.norm(down)
    return np.stack([right, down, fwd / np.linalg.norm(fwd)], axis=1)


def project(points_ego, M, cam_t, height: int, width: int, f_ref: float):
    """[N,3] ego-frame points -> (u, v, valid) on the stitched cylindrical frame."""
    p = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
    d = p - np.asarray(cam_t, dtype=np.float64).reshape(1, 3)
    can = d @ M                                    # == M^T·d per row
    x, y, z = can[:, 0], can[:, 1], can[:, 2]
    rho = np.hypot(x, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.arctan2(x, z)
        u = phi * f_ref + (width - 1) / 2.0
        v = np.where(rho > 1e-9, y / np.where(rho > 1e-9, rho, 1.0), 0.0) * f_ref + (height - 1) / 2.0
    phi_max = (width - 1) / 2.0 / f_ref            # the frame's own half-FOV in azimuth
    valid = (rho > 1e-9) & (np.abs(phi) <= phi_max) & (v >= 0) & (v <= height - 1)
    return u, v, valid


def ray_of_pixel(u, v, M, height: int, width: int, f_ref: float) -> np.ndarray:
    """The inverse of :func:`project` (the frame bank's own ray model), in ego/lidar coordinates."""
    phi = (np.asarray(u, dtype=np.float64) - (width - 1) / 2.0) / f_ref
    yn = (np.asarray(v, dtype=np.float64) - (height - 1) / 2.0) / f_ref
    ray_can = np.stack([np.sin(phi), yn, np.cos(phi)], axis=-1)
    return ray_can @ M.T


def densify(poses_xy, n_per_segment: int = 24, with_origin: bool = True) -> np.ndarray:
    """A plan's [K,2] waypoints -> a dense [N,3] ground polyline (z = 0) from the ego origin.

    Dense because a straight line in the world is a CURVE on a cylindrical image."""
    pts = np.asarray(poses_xy, dtype=np.float64)[:, :2]
    if with_origin:
        pts = np.concatenate([np.zeros((1, 2)), pts], axis=0)
    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        t = np.linspace(0.0, 1.0, n_per_segment, endpoint=False).reshape(-1, 1)
        segs.append(a.reshape(1, 2) * (1 - t) + b.reshape(1, 2) * t)
    segs.append(pts[-1:].reshape(1, 2))
    xy = np.concatenate(segs, axis=0)
    return np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)


def polyline_pixels(poses_xy, M, cam_t, height: int, width: int, f_ref: float,
                    n_per_segment: int = 24) -> list:
    """-> a list of [M,2] pixel polylines (one per visible run of the plan; empty when nothing is in view)."""
    pts = densify(poses_xy, n_per_segment)
    u, v, ok = project(pts, M, cam_t, height, width, f_ref)
    out, cur = [], []
    for i in range(len(u)):
        if ok[i]:
            cur.append((float(u[i]), float(v[i])))
        elif cur:
            out.append(np.asarray(cur))
            cur = []
    if cur:
        out.append(np.asarray(cur))
    return [seg for seg in out if len(seg) >= 2]
