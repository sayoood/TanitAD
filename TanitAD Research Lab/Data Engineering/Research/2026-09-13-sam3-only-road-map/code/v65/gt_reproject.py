"""The ground-truth WORLD MAP drawn back into a camera image: the display that shows whether the map lies on the road.

Every pixel (stride s of the full image) casts its ray (camera_model, f-theta or pinhole) onto the smooth LiDAR ground with
the same 3 fixed-point iterations as Camera.lift; the world cell under the hit gives the class (0.1 m, nearest). A pixel is
NOT painted when (a) its ray does not reach the ground in front of the camera within MAX_RANGE, (b) it is on the ego body
mask, (c) a tracked agent box covers it (obstacle.offline via gt.npz, label-time evidence), or (d) an obstacle LiDAR return
(0.3-4 m above the ground, ego body excluded) projects into the same or a neighbouring 8 px block at a range >= OCC_M shorter
than the ground hit -- something stands in front of that ground. Display only: edge cells are widened by one cell.
"""
import numpy as np
import cv2
import ground_surface as GS

MAX_RANGE, OCC_M, BLOCK = 40.0, 0.7, 4          # metres, metres, working pixels (x stride = full-resolution pixels)


def display_world(cls):
    out = cls.copy()
    e = cv2.dilate((cls == 5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out[e & ~np.isin(cls, (2, 3, 4, 6))] = 5
    return out


def ground_hits(C, sg, stride):
    """(xy rig [h*w, 2], 3-D range [h*w], valid [h*w]) for the pixel centres of the stride grid."""
    h, w = C.H // stride, C.W // stride
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * stride, (np.arange(h) + 0.5) * stride)
    d = C.rays_rig(uu.ravel(), vv.ravel())
    n = np.linalg.norm(d, axis=1)
    good = np.isfinite(d).all(axis=1) & (d[:, 2] < -1e-3 * np.where(np.isfinite(n), n, 1.0))
    dg = d[good] / n[good, None]
    z = np.full(len(dg), float(np.median(sg)))
    for _ in range(3):
        s = (z - C.t[2]) / dg[:, 2]
        p = C.t[:2] + dg[:, :2] * s[:, None]
        z = GS.height(p, sg)
    xy = np.zeros((len(d), 2)); rng = np.full(len(d), np.inf)
    xy[good] = p; rng[good] = np.where(s > 0, s, np.inf)
    valid = np.isfinite(rng) & (np.hypot(xy[:, 0] - C.t[0], xy[:, 1] - C.t[1]) <= MAX_RANGE)
    return xy, rng, valid, (h, w)


def lidar_occluded(C, lidar, sg, rng, hw, stride):
    """bool [h*w]: an obstacle return lies in front of the ground hit within the pixel's 3x3 block neighbourhood."""
    P = lidar[:, :3].astype(np.float64)
    P = P[np.hypot(P[:, 0], P[:, 1]) <= MAX_RANGE + 20.0]
    hz = GS.height(P[:, :2], sg)
    ego = (P[:, 0] > -1.2) & (P[:, 0] < 4.2) & (np.abs(P[:, 1]) < 1.2)
    P = P[(P[:, 2] > hz + 0.3) & (P[:, 2] < hz + 4.0) & ~ego]
    h, w = hw
    bh, bw = h // BLOCK + 1, w // BLOCK + 1
    depth = np.full((bh, bw), 1e6, np.float32)
    if len(P):
        u, v, ok = C.project_rig(P)
        r = np.linalg.norm(P - C.t, axis=1)
        bi = (v[ok] / (stride * BLOCK)).astype(np.int64); bj = (u[ok] / (stride * BLOCK)).astype(np.int64)
        inb = (bi >= 0) & (bi < bh) & (bj >= 0) & (bj < bw)
        np.minimum.at(depth, (bi[inb], bj[inb]), r[ok][inb].astype(np.float32))
        depth = cv2.erode(depth, np.ones((3, 3), np.uint8))
    rows = (np.arange(h) // BLOCK)[:, None].repeat(w, 1).ravel(); cols = (np.arange(w) // BLOCK)[None, :].repeat(h, 0).ravel()
    return depth[rows, cols] < rng - OCC_M


def camera_ground(C, sg, T, stride, lidar=None, ego_small=None, occ_small=None):
    """World xy of every visible ground pixel of camera C at rig pose T: (world xy [h*w, 2], paintable [h*w], (h, w)).
    Computed once per camera and frame; any number of world maps are then looked up with lookup()."""
    xy, rng, valid, (h, w) = ground_hits(C, sg, stride)
    keep = valid.copy()
    if lidar is not None:
        keep &= ~lidar_occluded(C, lidar, sg, rng, (h, w), stride)
    for m in (ego_small, occ_small):                    # 540 x 960 masks (the extraction raster)
        if m is not None:
            keep &= cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).ravel() == 0
    return xy @ T[:2, :2].T + T[:2, 3], keep, (h, w)


def lookup(wxy, keep, hw, wm_cls, origin, res):
    """uint8 [h, w] class raster of the world map at the camera's ground pixels (0 = not painted)."""
    i = np.floor((wxy[:, 0] - origin[0]) / res).astype(np.int64); j = np.floor((wxy[:, 1] - origin[1]) / res).astype(np.int64)
    ok = keep & (i >= 0) & (i < wm_cls.shape[0]) & (j >= 0) & (j < wm_cls.shape[1])
    out = np.zeros(len(wxy), np.uint8)
    code = wm_cls[i[ok], j[ok]]
    out[ok] = np.where(code == 255, 0, code)
    return out.reshape(hw)
