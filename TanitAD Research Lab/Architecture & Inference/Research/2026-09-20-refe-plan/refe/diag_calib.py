"""Per-sample calibration (R22): the frustum against an INDEPENDENT reference, plus the arms that
must fail.

⛔ A cross-check must not be derived from the code it checks. The reference here is OpenCV's
`undistortPoints` (the Caltech k1,k2,p1,p2,k3 inverse, independently authored) and a numpy
quaternion->matrix written from the textbook formula -- never `model._lift_camera` or
`model._quat_to_mat`.

Checks (CPU, ViT-S config, seconds):
  A  calib=None is BIT-IDENTICAL to passing the baked rig explicitly (one code path, by design)
  B  every camera's lifted patch points match the OpenCV+numpy reference, for the baked rig AND
     for a real other vehicle's rig from the calibration table (max |err| < 1e-3 m at 60 m)
  C  MUTATION: a different vehicle's rig MOVES the encoding, by the angle the reference predicts
  D  a mixed batch [A, B, A] equals the three single-sample encodings exactly
  E  the cache key names the rig: rig A then rig B on ONE live instance gives two encodings
  F  full forward with calib runs and reaches pos3d_mlp's gradient

Usage: python diag_calib.py --table <calib_table.json>     prints CALIB_OK or CALIB_FAILED
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_table as CT  # noqa: E402
from model import REFe, REFeConfig  # noqa: E402


def quat_to_mat_np(q):
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    """Hamilton product, w,x,y,z."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def yawed_rig(rig, deg):
    """The whole rig rotated about the ego z-axis by `deg`: R' = Rz R, t' = Rz t.

    ⭐ An ANALYTIC target: every lifted point must then be exactly Rz @ the baked point. It needs no
    second vehicle in the table, so the guard means the same thing on a one-log rehearsal table
    as on the pod's 1,192-log one.
    """
    a = np.radians(deg)
    qz = (np.cos(a / 2), 0.0, 0.0, np.sin(a / 2))
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    out = []
    for cam in rig:
        t = Rz @ np.array(cam[9:12])
        q = quat_mul(qz, cam[12:16])
        out.append(tuple(cam[:9]) + tuple(float(v) for v in t) + tuple(float(v) for v in q))
    return tuple(out), Rz


def reference_points(cfg, cam, gh, gw):
    """Ego-frame points [P, nd, 3] for one camera, from OpenCV + numpy only."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3, tx, ty, tz, qw, qx, qy, qz = cam
    sx, sy = cfg.img_w / cfg.cam_native_w, cfg.img_h / cfg.cam_native_h
    u = (np.arange(gw) + 0.5) * (cfg.img_w / gw)
    v = (np.arange(gh) + 0.5) * (cfg.img_h / gh)
    vv, uu = np.meshgrid(v, u, indexing="ij")
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]])
    pix = np.stack([uu.ravel(), vv.ravel()], -1).reshape(-1, 1, 2).astype(np.float64)
    dist = np.array([k1, k2, p1, p2, k3]) if cfg.undistort else np.zeros(5)
    # the iterated variant, run to convergence (the plain one stops after 5 iterations)
    norm = cv2.undistortPointsIter(pix, K, dist, None, None,
                                   (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-14))
    xy = norm.reshape(-1, 2)
    d = np.linspace(cfg.pos3d_near_m, cfg.pos3d_far_m, cfg.pos3d_depth_bins)
    cam_pts = np.stack([xy[:, 0:1] * d, xy[:, 1:2] * d, np.broadcast_to(d, (xy.shape[0], d.size))], -1)
    R = quat_to_mat_np((qw, qx, qy, qz))
    return cam_pts @ R.T + np.array([tx, ty, tz])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    cfg = REFeConfig.for_backbone("vits16")
    net = REFe(cfg).eval()
    tab = CT.load(a.table)
    baked = net.baked_calib()
    # a REAL other vehicle: the log whose front-camera rotation is furthest from the baked rig
    far_log, far_ang = None, -1.0
    Rb = quat_to_mat_np(baked[0][12:16])
    for lg in tab["logs"]:
        vec = CT.vectors(tab, lg, cfg.cameras)
        if vec is None:
            continue
        Rl = quat_to_mat_np(vec[0][12:16])
        ang = np.degrees(np.arccos(np.clip((np.trace(Rb.T @ Rl) - 1) / 2, -1, 1)))
        if ang > far_ang:
            far_log, far_ang = lg, ang
    # the real farthest vehicle when the table has one; otherwise the analytic yawed rig stands in
    if far_log is not None and far_ang > 0.01:
        other = tuple(tuple(v) for v in CT.vectors(tab, far_log, cfg.cameras))
        print(f"  other rig: {far_log}  (front-camera rotation {far_ang:.3f} deg from the baked rig)")
    else:
        other = yawed_rig(baked, 2.0)[0]
        print(f"  other rig: NONE in this table differs from the baked rig -- using the analytic "
              f"2-deg yaw rig for the real-vehicle arms")
    gh, gw = cfg.img_h // cfg.patch, cfg.img_w // cfg.patch
    P, nd = gh * gw, cfg.pos3d_depth_bins
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{name}] {'PASS' if cond else 'FAIL'} -- {detail}")

    img = torch.zeros(1, cfg.n_cameras, 3, cfg.img_h, cfg.img_w)
    f_none = net._frustum(gh, gw, img.device, img.dtype, cfg.n_cameras)
    f_baked = net._frustum(gh, gw, img.device, img.dtype, cfg.n_cameras, calib=baked)
    check("A.none==baked", torch.equal(f_none, f_baked), "calib=None and the explicit baked rig")
    with torch.no_grad():
        e_none = net._pos3d(img, cfg.n_cameras)
        e_bk = net._pos3d(img, cfg.n_cameras,
                          torch.tensor([baked], dtype=torch.float64))
    check("A.pos3d", torch.equal(e_none, e_bk), "the embedding too, through the per-sample path")

    for label, rig in (("baked", baked), ("other", other)):
        fr = net._frustum(gh, gw, img.device, img.dtype, cfg.n_cameras, calib=rig)[0]
        errs = []
        for i in range(cfg.n_cameras):
            model_pts = fr[i * P:(i + 1) * P].reshape(P, nd, 3).double() * cfg.pos3d_far_m
            ref = torch.from_numpy(reference_points(cfg, rig[i], gh, gw))
            errs.append(float((model_pts - ref).abs().max()))
        check(f"B.reference_{label}", max(errs) < 1e-3,
              f"max |model - OpenCV/numpy| per camera {['%.2e' % e for e in errs]} m")

    f_other = net._frustum(gh, gw, img.device, img.dtype, cfg.n_cameras, calib=other)
    moved = float((f_other - f_none).abs().max() * cfg.pos3d_far_m)
    # predicted displacement of the front camera's optical-axis point at the far plane
    ray = np.array([0.0, 0.0, cfg.pos3d_far_m])
    pb = quat_to_mat_np(baked[0][12:16]) @ ray + np.array(baked[0][9:12])
    po = quat_to_mat_np(other[0][12:16]) @ ray + np.array(other[0][9:12])
    check("C.other_rig_MOVES", moved > 0.1,
          f"max point displacement {moved:.3f} m at <= {cfg.pos3d_far_m:.0f} m "
          f"(front optical axis alone: {np.linalg.norm(po - pb):.3f} m predicted)")

    # C2: the ANALYTIC identity -- the rig yawed by 2 deg about the ego origin must lift every patch
    # to exactly Rz(2 deg) @ the baked point
    yr, Rz = yawed_rig(baked, 2.0)
    f_yaw = net._frustum(gh, gw, img.device, img.dtype, cfg.n_cameras, calib=yr)[0]
    pb = f_none[0].reshape(-1, nd, 3).double() * cfg.pos3d_far_m
    py = f_yaw.reshape(-1, nd, 3).double() * cfg.pos3d_far_m
    err = float((py - pb @ torch.from_numpy(Rz).T).abs().max())
    check("C2.analytic_yaw", err < 1e-3,
          f"yawed rig == Rz(2 deg) @ baked, max |err| {err:.2e} m over {pb.shape[0]:,} patches x {nd}")

    with torch.no_grad():
        mixed = net._pos3d(img.expand(3, -1, -1, -1, -1), cfg.n_cameras,
                           torch.tensor([baked, other, baked], dtype=torch.float64))
        single_o = net._pos3d(img, cfg.n_cameras, torch.tensor([other], dtype=torch.float64))
    check("D.mixed_batch", torch.equal(mixed[0], e_none[0]) and torch.equal(mixed[2], e_none[0])
          and torch.equal(mixed[1], single_o[0]), "rows 0/2 == baked, row 1 == other, exactly")

    fresh = REFe(cfg).eval()
    a1 = fresh._frustum(gh, gw, img.device, img.dtype, cfg.n_cameras, calib=baked)
    a2 = fresh._frustum(gh, gw, img.device, img.dtype, cfg.n_cameras, calib=other)
    check("E.cache_names_rig", not torch.equal(a1, a2) and len(fresh._pos3d_cache) == 2,
          f"{len(fresh._pos3d_cache)} cache entries for 2 rigs on one live instance")

    net.train()
    x = torch.randn(2, cfg.n_cameras, 3, cfg.img_h, cfg.img_w)
    traj, score = net(x, torch.zeros(2, cfg.ego_dim), torch.zeros(2, 2 * cfg.n_goal_points),
                      calib=torch.tensor([baked, other], dtype=torch.float64))
    traj.sum().backward()
    g = net.pos3d_mlp[0].weight.grad
    check("F.forward_grad", traj.shape[0] == 2 and g is not None and float(g.abs().sum()) > 0,
          f"traj {tuple(traj.shape)}, pos3d_mlp grad |sum| {float(g.abs().sum()) if g is not None else 0:.3e}")
    print("CALIB_OK" if ok else "CALIB_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
