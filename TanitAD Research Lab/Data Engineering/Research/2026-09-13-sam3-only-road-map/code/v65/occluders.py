"""Image masks of things standing in front of the ground, v2 (PI 2026-09-13: "why are there uncolored areas near to the objects?").

MEASURED (occluder_check.py, day clip, 10 vehicle boxes within 20 m): the tracked boxes' z lies 0.43-1.28 m BELOW the smooth LiDAR
ground under them (median 0.99 m) while z + h matches the LiDAR roof (+0.19 m median). The v1 hull (sam3map_render_v5*.agent_occluders)
spanned z - h/2 - 0.2 .. z + h + 0.2, i.e. down to ~2 m under the road, and blanked the asphalt in front of every vehicle -- in the
reprojection display AND in the map arms that mask agents (e2-e6).
v2 hull: bottom = the smooth LiDAR ground under the box centre, top = z + h + 0.15 (at least bottom + 0.5 m); footprint +0.1 m.
LiDAR occluders: every obstacle return (0.3-4 m above the smooth ground, ego body excluded) is splatted as a disc of 0.12 m at its
own range (1-8 working pixels) into a per-pixel depth image -- a post masks a post-shaped strip, not a 24 px square.
Ego body for DISPLAY: only the connected components of the extraction's ego mask that touch the image border (the hood); small
interior patches (e.g. a reflection patch zone) are not the car body and stay paintable.
"""
import numpy as np
import cv2
import ground_surface as GS

OCC_M = 0.7


def agent_hulls(C, boxes, sg, shape=(540, 960), r_max=45.0):
    """bool [h, w] (the extraction raster, 960x540 by default): convex hulls of the tracked boxes, bottom on the LiDAR ground."""
    m = np.zeros(shape, np.uint8)
    sx, sy = shape[1] / C.W, shape[0] / C.H
    for b in boxes:
        x, y, z, l, w, h, yaw = [float(v) for v in b[:7]]
        if np.hypot(x - C.t[0], y - C.t[1]) > r_max:
            continue
        g = float(GS.height(np.array([[x, y]]), sg)[0])
        bottom = g; top = max(z + h + 0.15, g + 0.5)
        c_, s_ = np.cos(yaw), np.sin(yaw)
        dx = np.array([1, 1, -1, -1]) * (l / 2 + 0.1); dy = np.array([1, -1, -1, 1]) * (w / 2 + 0.1)
        cx = x + dx * c_ - dy * s_; cy = y + dx * s_ + dy * c_
        Pb = np.c_[np.r_[cx, cx], np.r_[cy, cy], np.r_[np.full(4, bottom), np.full(4, top)]]
        Pc = (Pb - C.t) @ C.R
        ok = Pc[:, 2] > 0.1
        if C.ft is not None:
            ok &= np.arctan2(np.hypot(Pc[:, 0], Pc[:, 1]), Pc[:, 2]) <= C._th_tab[-1]
        if ok.sum() < 4:
            continue
        u, v, _ = C.project_cam(Pc[ok])
        pts = np.c_[u * sx, v * sy]
        if not np.isfinite(pts).all() or pts[:, 0].max() < 0 or pts[:, 0].min() > shape[1] or pts[:, 1].max() < 0 or pts[:, 1].min() > shape[0]:
            continue
        cv2.fillConvexPoly(m, cv2.convexHull(np.clip(pts, -5000, 5000).astype(np.int32)), 1)
    return m.astype(bool)


def lidar_depth(C, lidar, sg, hw, stride, max_range=60.0):
    """float32 [h, w] nearest obstacle-return range per working pixel (1e6 where none): each return splatted as a 0.12 m disc."""
    h, w = hw
    P = lidar[:, :3].astype(np.float64)
    P = P[np.hypot(P[:, 0], P[:, 1]) <= max_range]
    hz = GS.height(P[:, :2], sg)
    ego = (P[:, 0] > -1.2) & (P[:, 0] < 4.2) & (np.abs(P[:, 1]) < 1.2)
    P = P[(P[:, 2] > hz + 0.3) & (P[:, 2] < hz + 4.0) & ~ego]
    depth = np.full((h, w), 1e6, np.float32)
    if not len(P):
        return depth
    u, v, ok = C.project_rig(P)
    r = np.linalg.norm(P - C.t, axis=1)
    f = float(C.ft["poly"][1]) if C.ft is not None else float(C.K[0, 0])
    ui = (u[ok] / stride).astype(np.int64); vi = (v[ok] / stride).astype(np.int64); rr = r[ok].astype(np.float32)
    inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, rr = ui[inb], vi[inb], rr[inb]
    rad = np.clip(np.round(0.12 * f / (np.maximum(rr, 0.5) * stride)), 1, 8).astype(np.int64)
    for k in np.unique(rad):
        sel = rad == k
        d = np.full((h, w), 1e6, np.float32)
        np.minimum.at(d, (vi[sel], ui[sel]), rr[sel])
        d = cv2.erode(d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(k) + 1, 2 * int(k) + 1)))
        depth = np.minimum(depth, d)
    return depth


def hood_only(ego_small):
    """The ego mask's components that touch the image border (the car body); interior patches dropped."""
    if ego_small is None:
        return None
    m = ego_small.astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    keep = np.zeros(m.shape, bool)
    border = np.unique(np.r_[lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    for k in border:
        if k > 0:
            keep |= lab == k
    return keep
