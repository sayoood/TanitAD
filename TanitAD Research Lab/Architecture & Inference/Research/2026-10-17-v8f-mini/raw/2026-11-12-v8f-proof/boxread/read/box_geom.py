"""box_geom.py -- window geometry of card PREREG_V8F_PROOF_ADD127 (BOXREAD-T0). numpy only (the render test needs torch and the stack through enc_common).

The fixed rectilinear-in-azimuth window of card ADD99 (enc_common.window_frame / rotation_az_el / window_grid):
  win = (size_deg, px, az_left_deg, el_up_deg); the frame is square, px x px, cylindrical: azimuth is linear in the column,
  f = (px / 2) / radians(size_deg / 2); the ray of pixel INDEX (i = column, j = row) is p = (sin u, v, cos u) with
  u = (i - (px - 1) / 2) / f and v = (j - (px - 1) / 2) / f (camera axes +x right, +y down, +z boresight); the camera-frame ray is Q = R p
  with R = Ry(-az0) Rx(el0); the direction is az_left = atan2(-Qx, Qz) and el_up = atan2(-Qy, hypot(Qx, Qz)).
Box coordinates are CONTINUOUS window pixels, origin top left, a pixel i covering [i, i + 1): continuous c = index + 0.5, so the window
centre is (px / 2, px / 2). A detector box x1 y1 x2 y2 is in this convention; a rendered mask whose pixel indices span i0..i1 is the box [i0, i1 + 1].
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

DEG = math.pi / 180.0


def f_ref(win):
    return (float(win[1]) / 2.0) / math.radians(float(win[0]) / 2.0)


def rot_az_el(az_left_deg, el_up_deg):
    """R = Ry(-az) @ Rx(el) (enc_common.rotation_az_el), float64 3 x 3."""
    az, el = float(az_left_deg) * DEG, float(el_up_deg) * DEG
    ca, sa, ce, se = math.cos(-az), math.sin(-az), math.cos(el), math.sin(el)
    ry = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
    return ry @ rx


def vec_of(az_left_deg, el_up_deg):
    """unit camera-frame ray of a direction (azimuth left positive, elevation up positive): (-cos el sin az, -sin el, cos el cos az)."""
    az, el = np.asarray(az_left_deg, np.float64) * DEG, np.asarray(el_up_deg, np.float64) * DEG
    return np.stack([-np.cos(el) * np.sin(az), -np.sin(el) * np.ones_like(az), np.cos(el) * np.cos(az)], -1)


def dir_of(q):
    q = np.asarray(q, np.float64)
    return (np.degrees(np.arctan2(-q[..., 0], q[..., 2])), np.degrees(np.arctan2(-q[..., 1], np.hypot(q[..., 0], q[..., 2]))))


def pix_to_dir(col, row, win):
    """continuous window pixel (column, row) -> (az_left, el_up) in degrees of the camera the window is rendered from."""
    px, f = float(win[1]), f_ref(win)
    u = (np.asarray(col, np.float64) - 0.5 - (px - 1.0) / 2.0) / f
    v = (np.asarray(row, np.float64) - 0.5 - (px - 1.0) / 2.0) / f
    p = np.stack([np.sin(u), v, np.cos(u)], -1)
    q = p @ rot_az_el(win[2], win[3]).T
    return dir_of(q)


def dir_to_pix(az_left_deg, el_up_deg, win):
    """(az_left, el_up) -> (column, row, ahead) continuous window pixels; ahead is False when the ray is behind the window axis (z <= 0)."""
    px, f = float(win[1]), f_ref(win)
    d = vec_of(az_left_deg, el_up_deg)
    p = d @ rot_az_el(win[2], win[3])
    u = np.arctan2(p[..., 0], p[..., 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        v = p[..., 1] / np.hypot(p[..., 0], p[..., 2])
    return (px - 1.0) / 2.0 + f * u + 0.5, (px - 1.0) / 2.0 + f * v + 0.5, p[..., 2] > 0


def box_angles_many(boxes, win):
    """boxes [K, 4] (x1 y1 x2 y2) -> (az_c, el_c, width_deg, height_deg), each [K]. Centre = the direction of the box centre pixel; width and height =
    the range of azimuth and of elevation over the four corners and the four edge midpoints (the label's own definition: the range of the
    azimuths and elevations of the box's corners, lead_box.py box_geometry)."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    xs = np.stack([x1, x2, x1, x2, mx, mx, x1, x2], 1)
    ys = np.stack([y1, y1, y2, y2, y1, y2, my, my], 1)
    az, el = pix_to_dir(xs, ys, win)
    azc, elc = pix_to_dir(mx, my, win)
    return azc, elc, az.max(1) - az.min(1), el.max(1) - el.min(1)


def box_angles(box, win):
    a = box_angles_many(np.asarray(box, np.float64).reshape(1, 4), win)
    return tuple(float(x[0]) for x in a)


def contains_many(boxes, az, el, win):
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    c, r, ahead = dir_to_pix(az, el, win)
    return ahead & (c >= b[:, 0]) & (c <= b[:, 2]) & (r >= b[:, 1]) & (r <= b[:, 3])


def contains(box, az, el, win):
    return bool(contains_many(np.asarray(box, np.float64).reshape(1, 4), az, el, win)[0])


def edge_distance_many(boxes, az, el, win):
    """angular distance in degrees from the direction to the nearest point of each box (0 inside); inf when the ray is behind the window axis."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    c, r, ahead = dir_to_pix(az, el, win)
    qc, qr = np.clip(c, b[:, 0], b[:, 2]), np.clip(r, b[:, 1], b[:, 3])
    inside = (c >= b[:, 0]) & (c <= b[:, 2]) & (r >= b[:, 1]) & (r <= b[:, 3])
    qa, qe = pix_to_dir(qc, qr, win)
    v0, v1 = vec_of(az, el), vec_of(qa, qe)
    cr = np.linalg.norm(np.cross(v0, v1), axis=-1)
    dt = (v0 * v1).sum(-1)
    d = np.degrees(np.arctan2(cr, dt))
    d = np.where(inside, 0.0, d)
    return np.where(ahead, d, np.inf)


def edge_distance_deg(box, az, el, win):
    return float(edge_distance_many(np.asarray(box, np.float64).reshape(1, 4), az, el, win)[0])


def pick_one(boxes, az, el, win, edge_tol=1.5):
    """card section 3, Pick: among the boxes the one that contains the direction (the smallest by pixel area when several do), else the one whose
    edge is nearest within `edge_tol` degrees, else -1. Ties go to the lower index."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    if len(b) == 0 or not (np.isfinite(az) and np.isfinite(el)):
        return -1
    ins = np.flatnonzero(contains_many(b, az, el, win))
    if ins.size:
        area = (b[ins, 2] - b[ins, 0]) * (b[ins, 3] - b[ins, 1])
        return int(ins[int(np.argmin(area))])
    d = edge_distance_many(b, az, el, win)
    j = int(np.argmin(d))
    return j if np.isfinite(d[j]) and d[j] <= edge_tol else -1


def label_inside(az, el, win):
    """True where the direction lies inside the window's pixel square (and ahead of the axis); False for a non-finite direction."""
    az, el = np.asarray(az, np.float64), np.asarray(el, np.float64)
    ok = np.isfinite(az) & np.isfinite(el)
    c, r, ahead = dir_to_pix(np.where(ok, az, 0.0), np.where(ok, el, 0.0), win)
    px = float(win[1])
    return ok & ahead & (c >= 0) & (c <= px) & (r >= 0) & (r <= px)


# ------------------------------------------------------------------------------------------------ wide camera direction -> tele camera
def rot_cam_to_vehicle(qx, qy, qz, qw):
    """cam -> vehicle rotation of a unit quaternion (the stack's FrontWideExtrinsics.rotation_cam_to_vehicle)."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def point_in_other_cam(R_src_to_veh, R_dst_to_veh, az_left_deg, el_up_deg):
    """The routine of tele_encode.point_in_other_cam (telegap/enc), vectorised: a direction (az left+, el up+) of the SOURCE camera expressed in the
    DESTINATION camera's own (az, el). Both rotations are cam -> vehicle. A roll between the cameras is carried too (the full rotation is applied)."""
    d = vec_of(az_left_deg, el_up_deg)
    e = d @ (np.asarray(R_dst_to_veh, np.float64).T @ np.asarray(R_src_to_veh, np.float64)).T
    return dir_of(e)


# ------------------------------------------------------------------------------------------------ render test (tool test of control b1)
def render_roundtrip(enc, win, az_c, el_c, w_deg, h_deg, native=1600, focal=1200.0):
    """Draw a rectangle of angular size (w_deg, h_deg) at (az_c, el_c) into a synthetic f-theta camera frame (r = focal * theta), render the window
    `win` with enc_common.window_grid / window_sample, read the box of the lit region (threshold 127) and return (az_c, el_c, w, h) from box_angles.
    `enc` is the imported enc_common module. The native image is an anti-aliased membership function (a linear ramp of one native pixel), so the
    edges are placed to a small fraction of a native pixel."""
    import torch
    enc.setup_paths()
    from tanitad.data.calib import FThetaIntrinsics
    intr = FThetaIntrinsics(poly=(0.0, float(focal), 0.0, 0.0, 0.0), cx=(native - 1) / 2.0, cy=(native - 1) / 2.0, width=native, height=native, per_clip=True)
    jj, ii = np.meshgrid(np.arange(native, dtype=np.float64), np.arange(native, dtype=np.float64), indexing="ij")
    dx, dy = ii - intr.cx, jj - intr.cy
    r = np.hypot(dx, dy)
    th = r / focal
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = np.where(r > 0, np.sin(th) / np.where(r > 0, r, 1.0), 1.0 / focal)
    q = np.stack([dx * sc, dy * sc, np.cos(th)], -1)
    az, el = dir_of(q)
    pix_deg = math.degrees(1.0 / focal)
    mem = np.minimum(np.clip((w_deg / 2.0 - np.abs(az - az_c)) / pix_deg + 0.5, 0, 1), np.clip((h_deg / 2.0 - np.abs(el - el_c)) / pix_deg + 0.5, 0, 1))
    img = torch.from_numpy((mem * 255.0).round().astype(np.uint8))[None, None].expand(1, 3, native, native).contiguous()
    frame = enc.window_frame(float(win[0]), int(win[1]))
    grid, mask = enc.window_grid(intr, native, native, frame, float(win[2]), float(win[3]))
    out = enc.window_sample(img, grid, mask)[0, 0].numpy()
    lit = out > 127
    if not lit.any():
        return None
    cols, rows = np.flatnonzero(lit.any(0)), np.flatnonzero(lit.any(1))
    box = [cols.min(), rows.min(), cols.max() + 1.0, rows.max() + 1.0]
    return box_angles(box, win)


def load_enc_common(path=None):
    """import enc_common.py of card ADD99 by path (env BOXREAD_ENC_COMMON; default the dev-box and pod copies)."""
    import importlib.util
    cands = [path, os.environ.get("BOXREAD_ENC_COMMON"), "/workspace/v8proof/cropread/kit/enc/enc_common.py", "F:/TanitAD-scratch/v8proof/cropread/enc/enc_common.py"]
    for p in cands:
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("enc_common", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["enc_common"] = m
            spec.loader.exec_module(m)
            return m
    raise OSError("enc_common.py not found (set BOXREAD_ENC_COMMON)")


RENDER_CASES = (((32.0, 768, 0.0, -6.0), 2.0, -4.5, 4.0, 3.0), ((32.0, 768, 0.0, -6.0), -3.0, -7.5, 5.5, 2.5), ((16.0, 1024, 0.0, 0.0), 1.5, 0.5, 2.4, 1.6))
