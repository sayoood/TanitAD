"""r8v_geom.py -- the CYLINDRICAL rig->image projection used by the refcv8 videos (numpy twin of tanitad/data/rig_projection.py).

PROJECTION USED: cylindrical (NOT pinhole).  The eval frames are 416x1024, `projection: cylindrical`, f_ref 488.924 (== 305.577 x 1.6 of the 256x640 rig
frame; HFOV = 2 * (W/2) / f_ref = 120 deg): the COLUMN IS LINEAR IN AZIMUTH.  A pinhole `col = cx + f*x/z` would put every off-axis point in the wrong
column (corpus_overlay.py's FlatProjector is the pinhole one, F_EFF 266, and is therefore NOT used here).

    p_cam = R_cam_to_rig^T (p_rig - t_cam_in_rig)         camera frame: +x right, +y DOWN, +z boresight; rig: +x fwd, +y LEFT, +z UP, origin rear axle on the road
    phi   = atan2(x_cam, z_cam)             col = (W-1)/2 + f * phi
    row   = (H-1)/2 + f * y_cam / hypot(x_cam, z_cam)

Per-clip extrinsics (quaternion cam->vehicle + mount xyz) come from refcv8_train4572_eval139_extrinsics.json: the mount pitch is what puts the horizon on the
right row, and the quaternion convention is physicalai.FrontWideExtrinsics.rotation_cam_to_vehicle.  Independent controls live in test_r8v_geom.py
(analytic nominal-camera targets as LITERALS, and an agreement check against the repo's own torch implementation).
"""
from __future__ import annotations

import math

import numpy as np

PIX_EPS = 1e-6


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


# nominal mount (no pitch): p_cam = RIG_TO_CAM @ p_rig  (cam_x = -left, cam_y = -up, cam_z = +fwd)  -- rig_projection.RIG_TO_CAM
RIG_TO_CAM = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


class CylCam:
    def __init__(self, R_cam_to_rig, t_cam_in_rig, height=416, width=1024, f_ref=488.92398517830253):
        self.R = np.asarray(R_cam_to_rig, np.float64)
        self.t = np.asarray(t_cam_in_rig, np.float64).reshape(3)
        err = float(np.abs(self.R @ self.R.T - np.eye(3)).max())
        if err > 1e-4:
            raise ValueError(f"R_cam_to_rig is not a rotation (|RR^T-I|max={err:.2e})")
        self.H, self.W, self.f = int(height), int(width), float(f_ref)
        self.cx, self.cy = (self.W - 1) / 2.0, (self.H - 1) / 2.0

    @classmethod
    def from_extrinsics(cls, e, **kw):
        return cls(quat_to_R(float(e["qx"]), float(e["qy"]), float(e["qz"]), float(e["qw"])),
                   [float(e["x"]), float(e["y"]), float(e["z"])], **kw)

    @classmethod
    def nominal(cls, x_m=1.5, y_m=0.0, h_m=1.5, **kw):
        return cls(RIG_TO_CAM.T, [x_m, y_m, h_m], **kw)

    def project(self, p_rig):
        """rig points [N,3] -> (col, row, valid) float arrays; valid = in front AND inside the frame."""
        p = np.asarray(p_rig, np.float64).reshape(-1, 3)
        pc = (p - self.t) @ self.R                    # == R^T (p - t)
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        rho = np.sqrt(x * x + z * z)
        phi = np.arctan2(x, z)
        col = self.cx + self.f * phi
        row = self.cy + self.f * y / np.maximum(rho, 1e-9)
        valid = (rho > 1e-9) & (z > 0) & (col >= -PIX_EPS) & (col <= self.W - 1 + PIX_EPS) & (row >= -PIX_EPS) & (row <= self.H - 1 + PIX_EPS)
        return col, row, valid

    def project_ground(self, xy, z=0.0):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        return self.project(np.concatenate([xy, np.full((len(xy), 1), z)], axis=1))

    def horizon_row(self):
        """Row of the straight-ahead horizon ray (vehicle +x at infinity)."""
        d = self.R.T @ np.array([1.0, 0.0, 0.0])         # vehicle-forward in the camera frame
        return self.cy + self.f * d[1] / max(math.hypot(d[0], d[2]), 1e-9)


def densify(xy, step=0.5, include_origin=True):
    """Polyline through (0,0) -> waypoints, resampled every `step` metres of arclength (so a clipped ground path loses only its near-field tail)."""
    P = np.asarray(xy, np.float64).reshape(-1, 2)
    if include_origin:
        P = np.vstack([[0.0, 0.0], P])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return P[:1]
    q = np.arange(0.0, s[-1], step)
    q = np.concatenate([q, [s[-1]]])
    return np.stack([np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])], axis=1)


def runs(valid):
    """[(i0, i1)] maximal runs of True in a bool array (inclusive end) -- polyline segments to draw."""
    out, i, n = [], 0, len(valid)
    while i < n:
        if valid[i]:
            j = i
            while j + 1 < n and valid[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out
