"""VIS-1's per-pixel ray/cuboid z-buffer -- VENDORED VERBATIM from the box-head audit.

SPEC_REFCV7 §14 (A9) R3: *"vis_frac comes from the audit's per-pixel ray/cuboid z-buffer: the clip's own
416x1024 cylinder and extrinsics, with every GT cuboid as an occluder. Reuse the audit's code; do not
re-derive it."*

⛔ NOTHING BELOW THE MARKER IS NEW CODE. The constants and the four functions between the markers are a
byte-for-byte copy of ``TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit/
code/vis_zbuf.py`` lines 39-144 (file md5 ``3721a83f6bf8b1072c7b1f987c908c0f``). Each function's source
segment is pinned by sha256 in :data:`AUDIT_FUNCTION_SHA256` and re-hashed from THIS file by
``tests/test_refcv7_vis1.py``, which also re-hashes the audit file itself when it is reachable, so a silent
"improvement" here goes RED instead of shipping a second visibility definition.

What the z-buffer computes (the audit's docstring, restated): per GT box, on the EXTENDED cylinder
(azimuth to +-120 deg, rows -300..H+300): ``n_full`` pixels of its whole silhouette; ``n_img`` those inside
the real 416x1024 frame AND on a pixel the clip's own f-theta sensor observes; ``n_vis`` those where THIS
box is the nearest cuboid. ``vis_frac = n_vis / n_full``; ``vis_rows`` = the vertical extent of its visible
pixels. Occluders = every valid GT row with a 3-D label. ⚠️ An UPPER bound on visibility: walls, trees,
poles, the ego hood and unlabelled objects are not in the GT.

The camera for a clip is :func:`clip_camera`: the audit's ``camera()`` closure (vis_zbuf.py:208-224)
lifted to take its inputs explicitly (the extrinsics-table entry and the intrinsics parquet paths), with
the same search order and the same fallback semantics. It is the ONLY non-verbatim function here and it
performs no geometry of its own.
"""
from __future__ import annotations

import ast as _ast
import glob as _glob
import hashlib  # the vendored block's sha12 reads the bare name (the audit file imported it at its top)
import hashlib as _hashlib
import math

import numpy as np

__all__ = ["W", "H", "F", "PAD_U", "PAD_V", "EDGES", "R_from_quat", "observed_mask", "corners", "zbuffer",
           "clip_camera", "AUDIT_SOURCE", "AUDIT_FUNCTION_SHA256", "function_sha256"]

#: where the verbatim block came from (the audit package; not part of the stack).
AUDIT_SOURCE = {
    "path": ("TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit/"
             "code/vis_zbuf.py"),
    "md5": "3721a83f6bf8b1072c7b1f987c908c0f",
    "lines": "39-144",
}

#: sha256 (first 16 hex) of each vendored function's ``ast.get_source_segment`` -- computed on the AUDIT
#: file at vendoring time (2026-09-27) and re-checked against THIS file by the tests.
AUDIT_FUNCTION_SHA256 = {
    "R_from_quat": "e30c2bb4bccdc2df",
    "observed_mask": "bff155873d8ba014",
    "corners": "523c9d642557c085",
    "zbuffer": "0f6b41fbb32320d6",
}

# ---- BEGIN VERBATIM (vis_zbuf.py:39-144) -------------------------------------------------------------- #
W, H, F = 1024, 416, 488.92398517830253
PAD_U, PAD_V = 512, 300
CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer", "person", "rider", "stroller",
           "animal", "protruding_object")


def sha12(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def R_from_quat(qx, qy, qz, qw):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]])


def observed_mask(poly, icx, icy, iw, ih):
    """[H, W] bool: canonical pixels whose ray lands inside the native f-theta frame."""
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    phi = (uu - (W - 1) / 2.0) / F
    yn = (vv - (H - 1) / 2.0) / F
    x, y, z = np.sin(phi), yn, np.cos(phi)
    rho = np.hypot(x, y)
    th = np.arctan2(rho, z)
    r = np.zeros_like(th)
    for c in reversed(poly):
        r = r * th + c
    k = np.where(rho > 1e-12, r / np.maximum(rho, 1e-12), 0.0)
    u, v = icx + x * k, icy + y * k
    return (u >= 0) & (u <= iw - 1) & (v >= 0) & (v <= ih - 1) & (z > 0)


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def corners(cx, cy, cz, l, w, h, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    xy = loc @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
    return np.concatenate([np.c_[xy, np.full(4, cz - h / 2)], np.c_[xy, np.full(4, cz + h / 2)]])


def zbuffer(boxes, R, t, obs):
    """boxes: [M, 7] (cx, cy, cz, l, w, h, yaw) rig frame. Returns per-box n_full, n_img, n_vis, vis_rows."""
    M = len(boxes)
    Hx, Wx = H + 2 * PAD_V, W + 2 * PAD_U
    zb = np.full((Hx, Wx), np.inf)
    idb = np.full((Hx, Wx), -1, np.int64)
    n_full = np.zeros(M, np.int64)
    n_img = np.zeros(M, np.int64)
    inimg = np.zeros((Hx, Wx), bool)
    inimg[PAD_V:PAD_V + H, PAD_U:PAD_U + W] = obs
    for i in range(M):
        cx, cy, cz, l, w, h, yaw = [float(v) for v in boxes[i]]
        if not (l > 0 and w > 0 and h > 0):
            continue
        cor = corners(cx, cy, cz, l, w, h, yaw)
        samp = np.concatenate([cor[a][None] + (cor[b] - cor[a])[None] * np.linspace(0, 1, 17)[:, None]
                               for a, b in EDGES])
        pc = (samp - t) @ R
        rho = np.hypot(pc[:, 0], pc[:, 2])
        phi = np.arctan2(pc[:, 0], pc[:, 2])
        cen = (np.array([cx, cy, cz]) - t) @ R
        if abs(math.atan2(cen[0], cen[2])) > math.radians(150) or (np.abs(phi) > math.radians(170)).any():
            continue                                    # behind the camera: not in any front image
        u = (W - 1) / 2.0 + F * phi + PAD_U
        v = (H - 1) / 2.0 + F * pc[:, 1] / np.maximum(rho, 1e-6) + PAD_V
        u0, u1 = int(max(0, math.floor(u.min()) - 1)), int(min(Wx - 1, math.ceil(u.max()) + 1))
        v0, v1 = int(max(0, math.floor(v.min()) - 1)), int(min(Hx - 1, math.ceil(v.max()) + 1))
        if u1 < u0 or v1 < v0:
            continue
        uu, vv = np.meshgrid(np.arange(u0, u1 + 1, dtype=np.float64), np.arange(v0, v1 + 1, dtype=np.float64))
        ph = (uu - PAD_U - (W - 1) / 2.0) / F
        yn = (vv - PAD_V - (H - 1) / 2.0) / F
        d = np.stack([np.sin(ph), yn, np.cos(ph)], -1) @ R.T          # rig-frame ray directions
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        o = t - np.array([cx, cy, cz])
        ob = np.array([cyaw * o[0] + syaw * o[1], -syaw * o[0] + cyaw * o[1], o[2]])
        db = np.stack([cyaw * d[..., 0] + syaw * d[..., 1], -syaw * d[..., 0] + cyaw * d[..., 1], d[..., 2]], -1)
        hs = np.array([l / 2, w / 2, h / 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-hs - ob) / db
            t2 = (hs - ob) / db
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = tmax >= np.maximum(tmin, 0.0)
        dep = np.maximum(tmin, 0.0) * np.linalg.norm(d, axis=-1)
        n_full[i] = int(hit.sum())
        sub_in = inimg[v0:v1 + 1, u0:u1 + 1]
        n_img[i] = int((hit & sub_in).sum())
        zs = zb[v0:v1 + 1, u0:u1 + 1]
        ids = idb[v0:v1 + 1, u0:u1 + 1]
        upd = hit & (dep < zs)
        zs[upd] = dep[upd]
        ids[upd] = i
    own = idb[inimg]
    n_vis = np.bincount(own[own >= 0], minlength=M)[:M] if M else np.zeros(0, np.int64)
    rows = np.zeros(M, np.int64)
    ys = np.nonzero(inimg)[0]
    for i in np.unique(own[own >= 0]):
        yy = ys[own == i]
        rows[i] = int(yy.max() - yy.min() + 1)
    return n_full, n_img, n_vis, rows, idb
# ---- END VERBATIM --------------------------------------------------------------------------------------- #


def function_sha256(source: str, name: str) -> str:
    """sha256 (first 16 hex) of the top-level function ``name``'s source segment in ``source``."""
    tree = _ast.parse(source)
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == name:
            seg = _ast.get_source_segment(source, node)
            return _hashlib.sha256(seg.encode("utf-8")).hexdigest()[:16]
    raise KeyError(f"no top-level function {name!r}")


#: The two intrinsics-file search roots the audit's ``camera()`` used, in its order. A caller on another
#: host passes its own ``calib_dirs``; the ORDER (first hit wins) is the audit's.
INTRINSICS_GLOBS = ("{calib_dir}/camera_intrinsics/camera_intrinsics.chunk_{ch:04d}.parquet",
                    "{calib_dir}/camera_intrinsics.chunk_{ch:04d}.parquet")
CAMERA_NAME = "camera_front_wide_120fov"
MASK_SOURCE_FTHETA = "f-theta observed mask"
MASK_SOURCE_BOUNDS = "no intrinsics file: image bounds only"


def clip_camera(clip_id: str, entry: dict, calib_dirs, *, read_parquet=None):
    """``(R, t, obs, why)`` for one clip -- the audit's ``camera()`` (vis_zbuf.py:208-224), inputs explicit.

    ``entry`` is the clip's row of the per-clip extrinsics table (``qx qy qz qw x y z chunk``), the SAME
    table the trainer's ``--agent-rig-extrinsics`` reads. ``calib_dirs`` are searched in order; the first
    ``camera_intrinsics.chunk_<chunk>.parquet`` found supplies the f-theta polynomial, and a clip whose chunk
    has no file falls back to image bounds only -- the audit's fallback, REPORTED in ``why`` so a caller can
    refuse it (``precompute_vis1_sidecar.py`` does, by default).
    """
    if read_parquet is None:
        import pandas as pd
        read_parquet = pd.read_parquet
    R = R_from_quat(float(entry["qx"]), float(entry["qy"]), float(entry["qz"]), float(entry["qw"]))
    t = np.array([float(entry["x"]), float(entry["y"]), float(entry["z"])])
    ch = int(entry.get("chunk", -1))
    ip = []
    for d in calib_dirs:
        for pat in INTRINSICS_GLOBS:
            ip = _glob.glob(pat.format(calib_dir=str(d), ch=ch))
            if ip:
                break
        if ip:
            break
    obs, why, src = np.ones((H, W), bool), MASK_SOURCE_BOUNDS, None
    if ip:
        it = read_parquet(ip[0]).reset_index()
        it = it[(it["clip_id"] == clip_id) & (it["camera_name"] == CAMERA_NAME)]
        if len(it):
            r = it.iloc[0]
            obs = observed_mask(tuple(float(r[f"fw_poly_{i}"]) for i in range(5)), float(r.cx), float(r.cy),
                                int(r.width), int(r.height))
            why = MASK_SOURCE_FTHETA
            src = ip[0]
    return R, t, obs, why, src
