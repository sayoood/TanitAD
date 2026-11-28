"""e0g_depth_targets.py -- E0g (REFCV9_DESIGN.md s2.3 "Box-derived foreground depth (B1)", s4.1 row E0g, s6b D4):
the box-derived depth-target builder.  Cuboids -> per-pixel nearest-surface depth on object pixels of the 416 x 1024
frame, through the CYLINDRICAL model -> LID bins over 0-100 m, CaDDN weights alpha_fg 3.25 / alpha_bg 0.25, soft
(categorical) targets on the s8 and s16 grids.

PURE NUMPY.  No torch import at module level (a fresh interpreter importing this module must not have `torch` in
`sys.modules`; `test_e0g_depth.py::test_module_import_is_torch_free` pins it).  The cylindrical geometry itself is the
programme's, imported LAZILY inside the functions that need it -- this file contains NO second projection formula:

* pixel -> camera ray ...... `tanitad.data.rig_projection.frame_to_cam_ray`  (keyed on `frame.projection`; the exact
                             inverse of `project_cam_to_frame` and the tensor twin of `calib.cylindrical_rays`)
* point -> pixel ........... `tanitad.data.rig_projection.project_cam_to_frame` (column prefilter only; the tests also
                             use it as an independent round trip)
* the canonical frame ...... `tanitad.data.calib.CanonicalFrame.from_hfov(120, 416, 1024, "cylindrical")`
                             (f_ref 488.92398517830253 == `vis1_zbuffer.F`; HFOV = 2 (W/2) / f_ref, NOT the pinhole formula)
* box layout / camera ...... `[cx, cy, cz, l, w, h, yaw]` rig frame and `(R_cam_to_rig, t_cam_in_rig)` -- exactly
                             `tanitad.data.vis1_zbuffer.zbuffer`'s arguments, so the two can be run on one scene.
* cuboid reader ............ `tanitad.data.agent_cuboid_gt.cuboids_at_time` (used by `e0g_content.py`, not here).

What is NEW here is only what no existing function returns: the DEPTH VALUE (`vis1_zbuffer.zbuffer` keeps its depth
buffer local and returns ids only), the LID discretisation, and the s8 / s16 cell aggregation.  The nearest-box identity
is cross-checked against `vis1_zbuffer.zbuffer`'s id buffer in the tests.

DEFINITIONS (every one a free parameter with its default stated; the design pins only the range, the bins' form, the two
alphas, "soft (categorical)", the two strides and the cylinder):
* depth_kind="range" : Euclidean distance along the pixel's ray (what a LiDAR measures).  "rho" : hypot(x_cam, z_cam),
  the distance in the camera's x-z plane == the cylinder's analogue of pinhole z-depth.  They differ by 1/sqrt(1+v^2)
  per pixel (<= 9 % at the frame's top/bottom edge), identically for every box, so the NEAREST-surface ordering is the
  same for both.  Default "range".  C3 must pick one and state it (E0g card, decision 1).
* LID = CaDDN eq. 2 (banked 2103.01100, sha12 3970e26ca935): d_c = d_min + (d_max - d_min) / (D (D + 1)) * i (i + 1), i.e.
  edge_i = d_min + bin_size * i (i + 1) / 2 with bin_size = 2 (d_max - d_min) / (D (D + 1)).  D = 80 is FREE: neither the
  design nor the CaDDN text fixes it (the public config's 80 is recalled, UNVERIFIED).  Continuous bin coordinate
  tau(d) = (sqrt(1 + 8 (d - d_min) / bin_size) - 1) / 2, bin i covers tau in [i, i + 1).
* soft target = TWO-HOT in the bin-centre coordinate (c = tau - 0.5; weight 1 - frac on floor(c), frac on floor(c) + 1;
  clamped at both ends): sums to 1 and preserves E[tau] exactly.  The design says "soft (categorical)"; CaDDN itself
  supervises with the ONE-HOT of the containing bin (banked 2103.01100, s1) -- soft=False reproduces that.
* a pixel is FOREGROUND iff the nearest surface along its ray is a cuboid whose depth is in [d_min, d_max) and the pixel
  is observed.  A hit beyond d_max is OUT-OF-RANGE (ignored), an unobserved pixel is IGNORED.
* a CELL (stride s) is FG iff >= fg_min_frac (0.5) of its s x s pixels are foreground; its target is the MEAN of its
  foreground pixels' soft targets (a mixture where two depths meet).  A cell with NO object, no out-of-range and no
  unobserved pixel is BG; every other cell (a boundary) is IGNORE.  weight = alpha_fg / alpha_bg / 0.

THE alpha_bg QUESTION (decision 2 on the card; in CaDDN these are the FOCAL-LOSS alpha for foreground / background pixels,
both of which have LiDAR depth -- banked 2103.01100 s3.5): with box-only depth a BG cell carries NO depth label -- `target` is all
zeros there and `fg` is False.  alpha_bg therefore weights nothing unless C3 adds a background label (the road plane is the
free candidate; it is NOT in the pinned design and D4 says dense ground depth is forgone).  The weight map is emitted in
the CaDDN shape so C3 can decide; the builder does not invent a background label.

Cuboids that are not usable are SKIPPED AND COUNTED, never silently dropped: non-finite, a non-positive dimension, or the
camera inside the box (`meta["skipped"]`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

__all__ = [
    "FRAME_H", "FRAME_W", "HFOV_DEG", "D_MIN_M", "D_MAX_M", "N_BINS", "ALPHA_FG", "ALPHA_BG", "FG_MIN_FRAC",
    "STRIDES", "DEPTH_KINDS", "PIX_BG", "PIX_FG", "PIX_OOR", "PIX_UNOBS",
    "Camera", "nominal_camera", "camera_from_quat", "pixel_rays", "raycast", "RayCast",
    "lid_bin_size", "lid_edges", "lid_tau", "lid_depth", "lid_centres", "soft_two_hot", "expected_depth",
    "GridTarget", "DepthTargets", "build_depth_targets", "frame_stats",
]

FRAME_H, FRAME_W, HFOV_DEG = 416, 1024, 120.0
D_MIN_M, D_MAX_M = 0.0, 100.0
N_BINS = 80
ALPHA_FG, ALPHA_BG = 3.25, 0.25
FG_MIN_FRAC = 0.5
STRIDES = (8, 16)
DEPTH_KINDS = ("range", "rho")

#: per-pixel states of `DepthTargets.pixel_state`
PIX_BG, PIX_FG, PIX_OOR, PIX_UNOBS = 0, 1, 2, 3


# --------------------------------------------------------------------------------------------------------------- #
# camera                                                                                                          #
# --------------------------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Camera:
    """A camera in the rig frame + the canonical frame its pixels live in.

    `R_cam_to_rig` [3, 3] maps a CAMERA-frame vector (+x right, +y down, +z boresight) to the RIG frame (+x forward,
    +y left, +z up) -- the dataset's `sensor_extrinsics` quaternion, the `R` of `vis1_zbuffer.zbuffer`.  `t_cam_in_rig`
    [3] is the camera position in the rig frame.  `projection` is "cylindrical" for the corpus (416 x 1024, 120 deg,
    f_ref 488.924); "pinhole" exists ONLY for the deliberate-regression arm of the tests (`dataclasses.replace(cam,
    projection="pinhole")` keeps the same f_ref and pixel grid, i.e. the pinhole formula used in place of the cylindrical
    one).
    """

    R_cam_to_rig: np.ndarray
    t_cam_in_rig: np.ndarray
    height: int = FRAME_H
    width: int = FRAME_W
    f_ref: Optional[float] = None
    projection: str = "cylindrical"

    def __post_init__(self):
        R = np.asarray(self.R_cam_to_rig, dtype=np.float64)
        t = np.asarray(self.t_cam_in_rig, dtype=np.float64).reshape(-1)
        if R.shape != (3, 3) or t.shape != (3,):
            raise ValueError(f"R must be [3, 3] and t [3], got {R.shape} / {t.shape}")
        err = float(np.abs(R @ R.T - np.eye(3)).max())
        if err > 1e-4:
            raise ValueError(f"R_cam_to_rig is not a rotation (|R R^T - I|_max = {err:.2e}); check the quaternion order")
        object.__setattr__(self, "R_cam_to_rig", R)
        object.__setattr__(self, "t_cam_in_rig", t)
        if self.f_ref is None:
            if self.projection != "cylindrical":
                raise ValueError("f_ref must be given explicitly for a non-cylindrical projection")
            # the programme's own constructor: HFOV 120 deg through the CYLINDRICAL relation (not the pinhole one)
            from tanitad.data.calib import CanonicalFrame
            f = CanonicalFrame.from_hfov(HFOV_DEG, int(self.height), int(self.width), "cylindrical").f_ref
            object.__setattr__(self, "f_ref", float(f))

    @property
    def frame(self):
        from tanitad.data.calib import CanonicalFrame
        return CanonicalFrame(height=int(self.height), width=int(self.width), f_ref=float(self.f_ref),
                              projection=self.projection)


def nominal_camera(height_m: float = 1.5, x_m: float = 1.5, y_m: float = 0.0, **kw) -> Camera:
    """A level, boresight-forward camera at (x_m, y_m, height_m) in the rig.  For tests (answers computable by hand)."""
    from tanitad.data.rig_projection import RIG_TO_CAM
    R = np.asarray(RIG_TO_CAM, dtype=np.float64).T          # rig->cam permutation, transposed = cam->rig
    return Camera(R, np.array([x_m, y_m, height_m], dtype=np.float64), **kw)


def camera_from_quat(qx, qy, qz, qw, x, y, z, **kw) -> Camera:
    """Camera from a `sensor_extrinsics` row (quaternion cam->vehicle + position), via the vendored `R_from_quat`."""
    from tanitad.data.vis1_zbuffer import R_from_quat
    return Camera(R_from_quat(float(qx), float(qy), float(qz), float(qw)),
                  np.array([float(x), float(y), float(z)], dtype=np.float64), **kw)


_RAY_CACHE: dict = {}


def pixel_rays(cam: Camera) -> np.ndarray:
    """[H, W, 3] float64 UNIT camera-frame rays, one per pixel, from the programme's `frame_to_cam_ray`."""
    key = (int(cam.height), int(cam.width), float(cam.f_ref), cam.projection)
    got = _RAY_CACHE.get(key)
    if got is None:
        import torch
        from tanitad.data.rig_projection import frame_to_cam_ray
        rr, cc = np.meshgrid(np.arange(cam.height, dtype=np.float64), np.arange(cam.width, dtype=np.float64),
                             indexing="ij")
        d = frame_to_cam_ray(torch.from_numpy(cc), torch.from_numpy(rr), cam.frame).numpy()
        got = d / np.linalg.norm(d, axis=-1, keepdims=True)
        if len(_RAY_CACHE) > 8:
            _RAY_CACHE.clear()
        _RAY_CACHE[key] = got
    return got


# --------------------------------------------------------------------------------------------------------------- #
# ray cast                                                                                                        #
# --------------------------------------------------------------------------------------------------------------- #
@dataclass
class RayCast:
    depth: np.ndarray        # [H, W] float64, +inf where no cuboid is hit; in `depth_kind` units
    box_id: np.ndarray       # [H, W] int32, index of the NEAREST cuboid in the input array, -1 where none
    skipped: dict            # {"invalid": n, "camera_inside": n}


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _column_window(corners_rig: np.ndarray, cam: Camera, c_ctr_rig: np.ndarray):
    """Conservative [c0, c1] column interval a box can occupy, or None = cannot prefilter (use every column), or
    (1, 0) = provably outside the frame.  The azimuth extent of a convex body not containing the camera is attained at
    its vertices, so the corner columns bound it; boxes whose corners span >= ~pi (they wrap the camera) are not
    prefiltered."""
    import torch
    from tanitad.data.rig_projection import project_cam_to_frame
    pts = np.vstack([corners_rig, c_ctr_rig[None]])
    pc = (pts - cam.t_cam_in_rig) @ cam.R_cam_to_rig            # R^T (p - t)
    col, _, _ = project_cam_to_frame(torch.from_numpy(pc), cam.frame)
    col = col.numpy()
    cx = (cam.width - 1) / 2.0
    f = float(cam.f_ref)
    phi = (col - cx) / f                                         # azimuth of each point, (-pi, pi]
    dphi = _wrap(phi[:8] - phi[8])
    if dphi.max() - dphi.min() >= math.pi - 0.05:
        return None
    a, b = phi[8] + dphi.min(), phi[8] + dphi.max()
    fov = cx / f + 2.0 / f                                       # half field in rad, +2 px slack
    best = None
    for k in (-1, 0, 1):
        lo, hi = max(a + 2 * math.pi * k, -fov), min(b + 2 * math.pi * k, fov)
        if lo <= hi:
            best = (lo, hi)
    if best is None:
        return (1, 0)
    c0 = max(0, int(math.floor(cx + f * best[0])) - 2)
    c1 = min(cam.width - 1, int(math.ceil(cx + f * best[1])) + 2)
    return (c0, c1)


def _corners(cx, cy, cz, l, w, h, yaw) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    xy = loc @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
    return np.concatenate([np.c_[xy, np.full(4, cz - h / 2)], np.c_[xy, np.full(4, cz + h / 2)]])


def raycast(boxes, cam: Camera, *, depth_kind: str = "range", prefilter: bool = True) -> RayCast:
    """Per-pixel nearest-cuboid depth.  `boxes` [M, 7] = (cx, cy, cz, l, w, h, yaw), rig frame (yaw about +z, l along
    the heading, w across, h up; cz is the CENTRE).  Nearest surface wins; a pixel whose ray hits nothing keeps +inf / -1.
    """
    if depth_kind not in DEPTH_KINDS:
        raise ValueError(f"depth_kind must be one of {DEPTH_KINDS}, got {depth_kind!r}")
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    H, W = int(cam.height), int(cam.width)
    depth = np.full((H, W), np.inf)
    bid = np.full((H, W), -1, np.int32)
    skipped = {"invalid": 0, "camera_inside": 0}
    if boxes.shape[0] == 0:
        return RayCast(depth, bid, skipped)
    d_cam = pixel_rays(cam)                                       # [H, W, 3] unit
    d_rig = d_cam @ cam.R_cam_to_rig.T                            # [H, W, 3]
    rho_scale = np.hypot(d_cam[..., 0], d_cam[..., 2]) if depth_kind == "rho" else None
    t0 = cam.t_cam_in_rig
    for i in range(boxes.shape[0]):
        cx, cy, cz, l, w, h, yaw = boxes[i]
        if not (np.all(np.isfinite(boxes[i])) and l > 0 and w > 0 and h > 0):
            skipped["invalid"] += 1
            continue
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        o = t0 - np.array([cx, cy, cz])
        ob = np.array([cyaw * o[0] + syaw * o[1], -syaw * o[0] + cyaw * o[1], o[2]])
        hs = np.array([l / 2, w / 2, h / 2])
        if np.all(np.abs(ob) < hs):
            skipped["camera_inside"] += 1
            continue
        c0, c1 = 0, W - 1
        if prefilter:
            win = _column_window(_corners(cx, cy, cz, l, w, h, yaw), cam, np.array([cx, cy, cz]))
            if win is not None:
                c0, c1 = win
                if c0 > c1:
                    continue
        dr = d_rig[:, c0:c1 + 1, :]
        db = np.stack([cyaw * dr[..., 0] + syaw * dr[..., 1], -syaw * dr[..., 0] + cyaw * dr[..., 1], dr[..., 2]], -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-hs - ob) / db
            t2 = (hs - ob) / db
        lo = np.where(np.isnan(t1) | np.isnan(t2), -np.inf, np.minimum(t1, t2))
        hi = np.where(np.isnan(t1) | np.isnan(t2), np.inf, np.maximum(t1, t2))
        tmin = lo.max(axis=-1)
        tmax = hi.min(axis=-1)
        hit = (tmax >= tmin) & (tmin > 0.0)                       # camera is outside the box, so a hit has tmin > 0
        dep = tmin if rho_scale is None else tmin * rho_scale[:, c0:c1 + 1]
        cur = depth[:, c0:c1 + 1]
        upd = hit & (dep < cur)
        cur[upd] = dep[upd]
        bid[:, c0:c1 + 1][upd] = i
    return RayCast(depth, bid, skipped)


# --------------------------------------------------------------------------------------------------------------- #
# LID discretisation, soft targets                                                                                #
# --------------------------------------------------------------------------------------------------------------- #
def lid_bin_size(d_min: float = D_MIN_M, d_max: float = D_MAX_M, n_bins: int = N_BINS) -> float:
    return 2.0 * (d_max - d_min) / (n_bins * (n_bins + 1))


def lid_edges(d_min: float = D_MIN_M, d_max: float = D_MAX_M, n_bins: int = N_BINS) -> np.ndarray:
    """[n_bins + 1] bin edges, edge_i = d_min + bin_size * i (i + 1) / 2; edge_0 = d_min, edge_D = d_max."""
    i = np.arange(n_bins + 1, dtype=np.float64)
    return d_min + lid_bin_size(d_min, d_max, n_bins) * i * (i + 1) / 2.0


def lid_tau(d, d_min: float = D_MIN_M, d_max: float = D_MAX_M, n_bins: int = N_BINS):
    """Continuous bin coordinate: tau in [i, i + 1) <=> d in bin i; tau(d_min) = 0, tau(d_max) = n_bins."""
    bs = lid_bin_size(d_min, d_max, n_bins)
    return (np.sqrt(1.0 + 8.0 * (np.asarray(d, dtype=np.float64) - d_min) / bs) - 1.0) / 2.0


def lid_depth(tau, d_min: float = D_MIN_M, d_max: float = D_MAX_M, n_bins: int = N_BINS):
    """Inverse of `lid_tau`."""
    tau = np.asarray(tau, dtype=np.float64)
    return d_min + lid_bin_size(d_min, d_max, n_bins) * tau * (tau + 1.0) / 2.0


def lid_centres(d_min: float = D_MIN_M, d_max: float = D_MAX_M, n_bins: int = N_BINS) -> np.ndarray:
    """[n_bins] depth at each bin's centre in tau (tau = i + 0.5) -- the abscissae of the two-hot interpolation."""
    return lid_depth(np.arange(n_bins) + 0.5, d_min, d_max, n_bins)


def soft_two_hot(d, d_min: float = D_MIN_M, d_max: float = D_MAX_M, n_bins: int = N_BINS, soft: bool = True):
    """Depths [N] (all within [d_min, d_max)) -> (idx0 [N], idx1 [N], w0 [N], w1 [N]); idx1 = idx0 + 1 (clamped)."""
    tau = lid_tau(d, d_min, d_max, n_bins)
    if not soft:
        i0 = np.clip(np.floor(tau).astype(np.int64), 0, n_bins - 1)
        return i0, i0, np.ones_like(tau), np.zeros_like(tau)
    c = np.clip(tau - 0.5, 0.0, n_bins - 1.0)
    i0 = np.minimum(np.floor(c).astype(np.int64), n_bins - 2)
    fr = c - i0
    return i0, i0 + 1, 1.0 - fr, fr


def expected_depth(target, d_min: float = D_MIN_M, d_max: float = D_MAX_M, n_bins: int = N_BINS) -> np.ndarray:
    """[..., D] distribution -> depth at its expected bin coordinate (exact for a two-hot away from the ends)."""
    target = np.asarray(target, dtype=np.float64)
    tau = (target * (np.arange(target.shape[-1]) + 0.5)).sum(-1) / np.maximum(target.sum(-1), 1e-12)
    return lid_depth(tau, d_min, d_max, n_bins)


# --------------------------------------------------------------------------------------------------------------- #
# targets                                                                                                         #
# --------------------------------------------------------------------------------------------------------------- #
@dataclass
class GridTarget:
    stride: int
    target: np.ndarray       # [D, Hs, Ws] float32, soft categorical; sums to 1 on FG cells, exactly 0 elsewhere
    fg: np.ndarray           # [Hs, Ws] bool
    weight: np.ndarray       # [Hs, Ws] float32: alpha_fg on FG, alpha_bg on BG, 0 on IGNORE
    fg_frac: np.ndarray      # [Hs, Ws] float32: share of the cell's pixels that are foreground
    cell_depth: np.ndarray   # [Hs, Ws] float32: mean depth of the cell's foreground pixels, nan elsewhere


@dataclass
class DepthTargets:
    depth: np.ndarray        # [H, W] float32, +inf where no cuboid is hit (in depth_kind units)
    box_id: np.ndarray       # [H, W] int32
    pixel_state: np.ndarray  # [H, W] uint8: PIX_BG / PIX_FG / PIX_OOR / PIX_UNOBS
    grids: dict = field(default_factory=dict)     # stride -> GridTarget
    meta: dict = field(default_factory=dict)


def build_depth_targets(boxes, cam: Camera, *, strides=STRIDES, depth_kind: str = "range", observed=None,
                        n_bins: int = N_BINS, d_min: float = D_MIN_M, d_max: float = D_MAX_M,
                        alpha_fg: float = ALPHA_FG, alpha_bg: float = ALPHA_BG, fg_min_frac: float = FG_MIN_FRAC,
                        soft: bool = True, prefilter: bool = True) -> DepthTargets:
    """boxes [M, 7] + camera -> per-pixel depth, LID soft targets and CaDDN weights on each stride's grid."""
    H, W = int(cam.height), int(cam.width)
    rc = raycast(boxes, cam, depth_kind=depth_kind, prefilter=prefilter)
    hit = np.isfinite(rc.depth)
    obs = np.ones((H, W), bool) if observed is None else np.asarray(observed, bool)
    if obs.shape != (H, W):
        raise ValueError(f"observed must be [{H}, {W}], got {obs.shape}")
    state = np.full((H, W), PIX_BG, np.uint8)
    in_rng = hit & (rc.depth >= d_min) & (rc.depth < d_max)
    state[hit & ~in_rng] = PIX_OOR
    state[in_rng] = PIX_FG
    state[~obs] = PIX_UNOBS
    out = DepthTargets(depth=rc.depth.astype(np.float32), box_id=rc.box_id, pixel_state=state,
                       meta={"skipped": rc.skipped, "depth_kind": depth_kind, "n_bins": n_bins, "d_min": d_min,
                             "d_max": d_max, "alpha_fg": alpha_fg, "alpha_bg": alpha_bg, "fg_min_frac": fg_min_frac,
                             "soft": bool(soft), "projection": cam.projection, "f_ref": float(cam.f_ref),
                             "n_boxes": int(np.asarray(boxes).reshape(-1, 7).shape[0])})
    ys, xs = np.nonzero(state == PIX_FG)
    dv = rc.depth[ys, xs]
    i0, i1, w0, w1 = soft_two_hot(dv, d_min, d_max, n_bins, soft)
    for s in strides:
        if H % s or W % s:
            raise ValueError(f"stride {s} does not divide the {H} x {W} frame")
        Hs, Ws = H // s, W // s
        cell = (ys // s) * Ws + (xs // s)
        n_cells = Hs * Ws
        cnt = lambda m: m.reshape(Hs, s, Ws, s).sum(axis=(1, 3)).astype(np.float64)       # noqa: E731
        n_fg, n_oor, n_un = cnt(state == PIX_FG), cnt(state == PIX_OOR), cnt(state == PIX_UNOBS)
        frac = n_fg / float(s * s)
        fg = frac >= fg_min_frac
        bg = (n_fg == 0) & (n_oor == 0) & (n_un == 0)
        weight = np.where(fg, alpha_fg, np.where(bg, alpha_bg, 0.0)).astype(np.float32)
        acc = np.bincount(cell * n_bins + i0, weights=w0, minlength=n_cells * n_bins)
        acc += np.bincount(cell * n_bins + i1, weights=w1, minlength=n_cells * n_bins)
        acc = acc.reshape(n_cells, n_bins)
        nf = n_fg.reshape(-1)
        tgt = np.zeros_like(acc)
        m = fg.reshape(-1)
        tgt[m] = acc[m] / nf[m, None]
        dsum = np.bincount(cell, weights=dv, minlength=n_cells)
        cdep = np.where(m, dsum / np.maximum(nf, 1.0), np.nan).reshape(Hs, Ws)
        out.grids[s] = GridTarget(stride=s, target=np.ascontiguousarray(tgt.T.reshape(n_bins, Hs, Ws), np.float32),
                                  fg=fg, weight=weight, fg_frac=frac.astype(np.float32),
                                  cell_depth=cdep.astype(np.float32))
    return out


def frame_stats(dt: DepthTargets) -> dict:
    """The per-frame numbers the content assertion prints: foreground pixel share, per-stride FG cell share, box count."""
    st = dt.pixel_state
    n = float(st.size)
    r = {"fg_pixel_share": float((st == PIX_FG).sum() / n), "oor_pixel_share": float((st == PIX_OOR).sum() / n),
         "unobs_pixel_share": float((st == PIX_UNOBS).sum() / n),
         "boxes_visible": int(np.unique(dt.box_id[st == PIX_FG]).size)}
    for s, g in dt.grids.items():
        r[f"fg_cell_share_s{s}"] = float(g.fg.mean())
    return r
