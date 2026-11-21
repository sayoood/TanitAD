"""refcv9 C4 -- arm B2 (the TRAINING-ONLY perspective 3D auxiliary head) and its B2c control (a second, parameter-matched BEV head).
REFCV9_DESIGN 2.4 row "perspective 3D aux head (B2)" and 4.3 rows B2 / B2c.  The flag contract with the trainer lives in
``tanitad/train/persp3d_aux_flag.py``.

B2 (``Persp3DAuxHead``).  An FCOS3D-style head on the perception branch's stride-8 and stride-16 maps (the same two maps C3's depth head
reads).  Per cell: projected-centre HEATMAP (1 channel, class-agnostic, CenterNet focal loss on Gaussian targets), DEPTH (80 LID bins over
0-100 m -- C3's bins, the builder's ``soft_two_hot``), SIZE (log l, w, h), YAW (sin, cos of the rig-frame yaw) and CENTRE-NESS (FCOS: sqrt of
the min/max ratios of the cell's distances to the projected 2-D extent, BCE).
B2c (``Bev2AuxHead``).  A second BEV head on the SHARED BEV (``perception.bev_feats``, [B, 96, 120, 64], 0.5 m cells) with the SAME box
targets (the batch's own join3d cuboids): centre heatmap, the 3-D position (sub-cell offset + centre z -- the BEV's analogue of the
perspective head's depth), size, yaw, centre-ness.  Its width is chosen so its parameter count matches B2's within 10 % (BEVFormer v2's
BEV&BEV control: if B2c gains as much as B2 the gain was capacity, not the perspective view).

TARGETS: built per ITEM in the DataLoader worker (``persp3d_aux_flag.item_targets``), compactly -- a Gaussian heat map, an OWNER map (the agent
SLOT index owning each cell), a centre-ness map and, per slot, the depth two-hot -- so a batch costs ~1.5 MB, not C3's 43 MB.  The size / yaw
(/ cz / centre) targets are gathered ON THE DEVICE from the batch's own ``agent_*`` fields by the owner map.  Geometry goes through the
CYLINDRICAL model only: the cuboid corners / centre are projected with ``rig_projection.project_cam_to_frame`` and the visibility ray-cast is
E0g's ``box_depth_targets.raycast`` -- there is NO second projector here (the builder is a pinned byte copy and is not edited).

DEFINITIONS THE DESIGN DOES NOT PIN (all mine, HYPOTHESIS, stated in the stamp):
* a cuboid is a target iff >= ``VIS_MIN_PIX`` (32) of its pixels are the nearest surface along their ray (so an occluded object is not a
  positive); it is a heat positive iff its projected centre is inside the frame and in the observed rows, and a regression positive on (a) the
  cells where >= 50 % of the pixels belong to it (C3's FG rule) plus (b) its centre cell;
* depth target = the Euclidean range of the cuboid CENTRE (not the surface), one value for every cell of the object (FCOS3D's centre depth);
* both strides see every object; Gaussian sigma = max(0.8, 0.54 w / 6) cells per axis (CenterNet / TTFNet style);
* every part is weighted 1.0; strides averaged; each regression part is the mean over the positive cells (0.0 with the graph kept if none).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.data import box_depth_targets as BD

N_BINS = BD.N_BINS                          # 80 -- C3's LID bins (the SAME bins)
STRIDES = BD.STRIDES                        # (8, 16)
MID = 184                                   # persp branch width: ~1.53 M parameters with the resnet101 widths (512, 1024) -- design "~1.5 M"
HEAT_PRIOR = 0.01
SIGMA_MIN, SIGMA_ALPHA = 0.8, 0.54          # Gaussian radius (cells): max(SIGMA_MIN, SIGMA_ALPHA * extent_cells / 6)
VIS_MIN_PIX = 32                            # a cuboid is a target iff this many pixels are its nearest surface
FG_MIN_FRAC = BD.FG_MIN_FRAC                # C3's cell rule
MIN_HALF_CELLS = 1.0                        # extents for the centre-ness are at least +-1 cell about the centre
PART_W = {"heat": 1.0, "depth": 1.0, "size": 1.0, "yaw": 1.0, "ctr": 1.0}
BEV2_PART_W = {"heat": 1.0, "pos": 1.0, "size": 1.0, "yaw": 1.0, "ctr": 1.0}
PERSP_CH = {"heat": 1, "depth": N_BINS, "size": 3, "yaw": 2, "ctr": 1}
BEV2_CH = {"heat": 1, "off": 2, "cz": 1, "size": 3, "yaw": 2, "ctr": 1}
SLOT_KEYS = ("agent_box", "agent_yaw", "agent_valid", "agent_cz", "agent_h", "agent_zh_mask")
MATCH_TOL = 0.10                            # B2c's parameter count must be within +-10 % of B2's (brief)


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


# --------------------------------------------------------------------------------------------------------------- #
# the heads                                                                                                       #
# --------------------------------------------------------------------------------------------------------------- #
class _Branch(nn.Module):
    """1x1 reduce -> two 3x3 -> five 1x1 prediction convs (no normalisation: no running stats, no batch coupling)."""

    def __init__(self, c_in: int, mid: int):
        super().__init__()
        self.tower = nn.Sequential(nn.Conv2d(c_in, mid, 1), nn.ReLU(inplace=False), nn.Conv2d(mid, mid, 3, padding=1),
                                   nn.ReLU(inplace=False), nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(inplace=False))
        self.heat, self.depth = nn.Conv2d(mid, 1, 1), nn.Conv2d(mid, N_BINS, 1)
        self.size, self.yaw, self.ctr = nn.Conv2d(mid, 3, 1), nn.Conv2d(mid, 2, 1), nn.Conv2d(mid, 1, 1)
        with torch.no_grad():
            self.heat.bias.fill_(math.log(HEAT_PRIOR / (1.0 - HEAT_PRIOR)))

    def forward(self, x: Tensor) -> dict:
        t = self.tower(x)
        return {"heat": self.heat(t), "depth": self.depth(t), "size": self.size(t), "yaw": self.yaw(t), "ctr": self.ctr(t)}


class Persp3DAuxHead(nn.Module):
    """FCOS3D-style outputs at s8 and s16.  Called ONLY by the trainer's loss -- never by ``RefCV3Model.forward`` (training-only)."""

    def __init__(self, c8: int, c16: int, *, mid: int = MID):
        super().__init__()
        self.c8, self.c16, self.mid = int(c8), int(c16), int(mid)
        self.s8, self.s16 = _Branch(self.c8, self.mid), _Branch(self.c16, self.mid)

    def forward(self, fmap_s8: Tensor, fmap_s16: Tensor) -> dict:
        return {8: self.s8(fmap_s8), 16: self.s16(fmap_s16)}


class Bev2AuxHead(nn.Module):
    """B2c: three 3x3 convs on the shared BEV -> six 1x1 prediction convs.  ``mid`` is matched to B2's parameter count by ``matched_mid``."""

    def __init__(self, d_bev: int, *, mid: int):
        super().__init__()
        self.d_bev, self.mid = int(d_bev), int(mid)
        self.tower = nn.Sequential(nn.Conv2d(self.d_bev, self.mid, 3, padding=1), nn.ReLU(inplace=False),
                                   nn.Conv2d(self.mid, self.mid, 3, padding=1), nn.ReLU(inplace=False),
                                   nn.Conv2d(self.mid, self.mid, 3, padding=1), nn.ReLU(inplace=False))
        self.heat, self.off, self.cz = nn.Conv2d(self.mid, 1, 1), nn.Conv2d(self.mid, 2, 1), nn.Conv2d(self.mid, 1, 1)
        self.size, self.yaw, self.ctr = nn.Conv2d(self.mid, 3, 1), nn.Conv2d(self.mid, 2, 1), nn.Conv2d(self.mid, 1, 1)
        with torch.no_grad():
            self.heat.bias.fill_(math.log(HEAT_PRIOR / (1.0 - HEAT_PRIOR)))

    def forward(self, bev: Tensor) -> dict:
        t = self.tower(bev)
        return {"heat": self.heat(t), "off": self.off(t), "cz": self.cz(t), "size": self.size(t), "yaw": self.yaw(t), "ctr": self.ctr(t)}


def matched_mid(d_bev: int, n_target: int) -> int:
    """The B2c width whose parameter count is closest to ``n_target`` (deterministic search; the build asserts the +-10 % band)."""
    best = None
    for mid in range(16, 1025):
        n = n_params(Bev2AuxHead(d_bev, mid=mid))
        if best is None or abs(n - n_target) < abs(best[1] - n_target):
            best = (mid, n)
        if n > 1.5 * n_target:
            break
    return best[0]


def count_matched(n_b2c: int, n_b2: int, tol: float = MATCH_TOL) -> bool:
    return abs(int(n_b2c) - int(n_b2)) <= tol * int(n_b2)


# --------------------------------------------------------------------------------------------------------------- #
# targets (numpy; run in the DataLoader worker)                                                                   #
# --------------------------------------------------------------------------------------------------------------- #
def selected_slots(item_like: dict):
    """-> (slot indices [M] of the usable cuboids, boxes [M, 7] (cx, cy, cz, l, w, h, yaw)).  A cuboid is usable iff ``agent_valid &
    agent_zh_mask`` on a labelled row (NO_LABEL is not 'clear'); the slot index is kept so the owner maps address the batch's own ``agent_*``."""
    valid = np.asarray(item_like["agent_valid"]).astype(bool).reshape(-1)
    zh = np.asarray(item_like["agent_zh_mask"]).astype(bool).reshape(-1)
    lab = item_like.get("agent_label")
    lab = True if lab is None else bool(np.asarray(lab).reshape(-1)[0])
    box = np.asarray(item_like["agent_box"], dtype=np.float64).reshape(-1, 4)
    yaw = np.asarray(item_like["agent_yaw"], dtype=np.float64).reshape(-1)
    cz = np.asarray(item_like["agent_cz"], dtype=np.float64).reshape(-1)
    hh = np.asarray(item_like["agent_h"], dtype=np.float64).reshape(-1)
    sel = np.nonzero(valid & zh)[0] if lab else np.zeros(0, np.int64)
    n_nozh = int((valid & ~zh).sum()) if lab else 0
    boxes = np.stack([box[sel, 0], box[sel, 1], cz[sel], box[sel, 2], box[sel, 3], hh[sel], yaw[sel]], axis=-1) if sel.size else np.zeros((0, 7))
    return sel.astype(np.int64), boxes, n_nozh


def _splat(heat: np.ndarray, i: int, j: int, su: float, sv: float) -> None:
    """Elliptical Gaussian peaking at exactly 1.0 in the INTEGER cell (i, j); max-combined into ``heat`` [Hs, Ws]."""
    Hs, Ws = heat.shape
    ru, rv = int(math.ceil(3.0 * su)), int(math.ceil(3.0 * sv))
    a0, a1, b0, b1 = max(0, i - rv), min(Hs, i + rv + 1), max(0, j - ru), min(Ws, j + ru + 1)
    ii = np.arange(a0, a1, dtype=np.float64)[:, None]
    jj = np.arange(b0, b1, dtype=np.float64)[None, :]
    g = np.exp(-((ii - i) ** 2 / (2.0 * sv * sv) + (jj - j) ** 2 / (2.0 * su * su)))
    heat[a0:a1, b0:b1] = np.maximum(heat[a0:a1, b0:b1], g.astype(np.float32))


def _sigma(extent_cells: float) -> float:
    return max(SIGMA_MIN, SIGMA_ALPHA * float(extent_cells) / 6.0)


def _fcos_ctr(jj, ii, u0, u1, v0, v1):
    """FCOS centre-ness of cell centres (jj column, ii row; cell coordinates) against a 2-D extent [u0, u1] x [v0, v1] (>= 1 cell half-size)."""
    l, r, t, b = jj - u0, u1 - jj, ii - v0, v1 - ii
    num = np.clip(np.minimum(l, r), 0.0, None) * np.clip(np.minimum(t, b), 0.0, None)
    den = np.maximum(np.maximum(l, r), 1e-9) * np.maximum(np.maximum(t, b), 1e-9)
    return np.sqrt(num / den).astype(np.float32)


def cuboid_projection(box, cam: BD.Camera):
    """One cuboid (cx, cy, cz, l, w, h, yaw) -> dict(col, row, range, u, v extents in PIXELS, ok) through the programme's cylindrical
    ``project_cam_to_frame``.  Corner columns are unwrapped about the centre's azimuth (the cylinder's seam is behind the camera).
    ``ok`` is False when the box cannot be projected (non-finite, non-positive dimension, camera inside it, a body that wraps the camera)."""
    from tanitad.data.rig_projection import project_cam_to_frame
    b = np.asarray(box, dtype=np.float64)
    bad = {"ok": False}
    if not np.all(np.isfinite(b)) or b[3] <= 0 or b[4] <= 0 or b[5] <= 0:
        return bad
    corners = BD._corners(*[float(x) for x in b])                                    # the builder's own corner layout
    pts = np.vstack([corners, b[None, :3]])
    pc = (pts - cam.t_cam_in_rig) @ cam.R_cam_to_rig                                  # R^T (p - t), the builder's convention
    col, row, _ = project_cam_to_frame(torch.from_numpy(pc), cam.frame)
    col, row = col.numpy(), row.numpy()
    rho = np.hypot(pc[:, 0], pc[:, 2])
    if np.any(rho < 1e-6):
        return bad
    cx0 = (cam.width - 1) / 2.0
    f = float(cam.f_ref)
    dphi = BD._wrap((col[:8] - cx0) / f - (col[8] - cx0) / f)
    if dphi.max() - dphi.min() >= math.pi - 0.05:
        return bad
    u = col[8] + f * dphi
    v = row[:8]
    rng = float(np.linalg.norm(pc[8]))
    inside = bool(0.0 <= col[8] <= cam.width - 1 and 0.0 <= row[8] <= cam.height - 1)
    return {"ok": True, "col": float(col[8]), "row": float(row[8]), "range": rng, "u0": float(u.min()), "u1": float(u.max()),
            "v0": float(v.min()), "v1": float(v.max()), "centre_in_frame": bool(inside)}


def persp_targets(boxes: np.ndarray, slots: np.ndarray, n_slots: int, cam: BD.Camera, *, equalize_bottom_rows: int = 0,
                  strides=STRIDES) -> dict:
    """The compact B2 targets of ONE frame.  ``boxes`` [M, 7] (cx, cy, cz, l, w, h, yaw), ``slots`` [M] their indices into the item's agent
    arrays.  -> {"heat{s}": [Hs, Ws] f32, "own{s}": [Hs, Ws] i16 (slot index or -1), "ctr{s}": [Hs, Ws] f32, "ign{s}": [Hs, Ws] bool,
    "i0": [A] i64 (LID bin of the centre range, -1 = no depth label), "w1": [A] f32, "census": dict}."""
    H, W = int(cam.height), int(cam.width)
    M = int(boxes.shape[0])
    out: dict = {"i0": np.full(n_slots, -1, np.int64), "w1": np.zeros(n_slots, np.float32)}
    cen = {"n_boxes": M, "n_unprojectable": 0, "n_visible": 0, "n_heat_pos": 0, "n_centre_unobserved": 0, "n_out_of_range": 0}
    eq = int(equalize_bottom_rows)
    obs_row = np.ones(H, bool)
    if eq > 0:
        obs_row[H - eq:] = False
    proj = [cuboid_projection(boxes[k], cam) for k in range(M)]
    ok = np.array([p["ok"] for p in proj], bool) if M else np.zeros(0, bool)
    cen["n_unprojectable"] = int((~ok).sum())
    if M:
        rc = BD.raycast(boxes, cam, depth_kind="range")
        fgpix = np.isfinite(rc.depth) & (rc.depth >= BD.D_MIN_M) & (rc.depth < BD.D_MAX_M) & obs_row[:, None]
        ys, xs = np.nonzero(fgpix)
        bid = rc.box_id[ys, xs].astype(np.int64)
        nvis = np.bincount(bid, minlength=M)
        usable = ok & (nvis >= VIS_MIN_PIX)
    else:
        ys = xs = bid = np.zeros(0, np.int64)
        nvis = np.zeros(0, np.int64)
        usable = np.zeros(0, bool)
    cen["n_visible"] = int(usable.sum())
    slots = np.asarray(slots, np.int64)
    for k in np.nonzero(usable)[0]:                                                  # the depth two-hot of the CENTRE range, per slot
        r = proj[k]["range"]
        if BD.D_MIN_M <= r < BD.D_MAX_M:
            i0, _i1, _w0, w1 = BD.soft_two_hot(np.array([r]))
            out["i0"][slots[k]], out["w1"][slots[k]] = int(i0[0]), float(w1[0])
        else:
            cen["n_out_of_range"] += 1
    order = np.argsort([-proj[k]["range"] if usable[k] else 0.0 for k in range(M)], kind="stable")       # far -> near: nearer wins ties
    for s in strides:
        Hs, Ws = H // s, W // s
        heat = np.zeros((Hs, Ws), np.float32)
        own = np.full((Hs, Ws), -1, np.int64)
        ctr = np.zeros((Hs, Ws), np.float32)
        ign = np.zeros((Hs, Ws), bool)
        if eq > 0:
            ign[(np.arange(Hs) * s + s - 1) >= (H - eq), :] = True                  # a cell touching a zeroed row is IGNORE, never a target
        if ys.size:
            cell = (ys // s) * Ws + (xs // s)
            cnt = np.bincount(cell * M + bid, minlength=Hs * Ws * M).reshape(Hs * Ws, M)
            fg = cnt.sum(1) >= s * s * FG_MIN_FRAC
            ow = np.where(fg, cnt.argmax(1), -1)
            ow = np.where((ow >= 0) & usable[np.clip(ow, 0, M - 1)], ow, -1)
            own = ow.reshape(Hs, Ws)
        fg_own = own.copy()                                                           # the FG (visibility) owners: a centre claim never steals one
        jj, ii = np.meshgrid(np.arange(Ws, dtype=np.float64), np.arange(Hs, dtype=np.float64))
        for k in order:
            if not usable[k]:
                continue
            p = proj[k]
            uc, vc = (p["col"] + 0.5) / s - 0.5, (p["row"] + 0.5) / s - 0.5          # cell coordinates (integer = a cell centre)
            u0, u1 = (p["u0"] + 0.5) / s - 0.5, (p["u1"] + 0.5) / s - 0.5
            v0, v1 = (p["v0"] + 0.5) / s - 0.5, (p["v1"] + 0.5) / s - 0.5
            ci, cj = int(math.floor(vc + 0.5)), int(math.floor(uc + 0.5))
            centre_ok = p["centre_in_frame"] and 0 <= ci < Hs and 0 <= cj < Ws and not ign[ci, cj]
            e_u0, e_u1 = min(u0, uc - MIN_HALF_CELLS), max(u1, uc + MIN_HALF_CELLS)
            e_v0, e_v1 = min(v0, vc - MIN_HALF_CELLS), max(v1, vc + MIN_HALF_CELLS)
            mine = own == k
            if centre_ok:
                if fg_own[ci, cj] < 0 or fg_own[ci, cj] == k:                         # far -> near: the nearer centre claim wins a shared cell
                    own[ci, cj] = k
                    mine[ci, cj] = True
                _splat(heat, ci, cj, _sigma((u1 - u0)), _sigma((v1 - v0)))
                heat[ci, cj] = 1.0
                if s == strides[0]:
                    cen["n_heat_pos"] += 1
            elif s == strides[0]:
                cen["n_centre_unobserved"] += 1
            if mine.any():
                c = _fcos_ctr(jj, ii, e_u0, e_u1, e_v0, e_v1)
                ctr[mine] = c[mine]
        own = np.where(own >= 0, slots[np.clip(own, 0, max(M - 1, 0))] if M else -1, -1)
        out[f"heat{s}"], out[f"own{s}"], out[f"ctr{s}"], out[f"ign{s}"] = heat, own.astype(np.int16), ctr, ign
    out["census"] = cen
    return out


def bev_targets(boxes: np.ndarray, slots: np.ndarray, grid) -> dict:
    """The compact B2c targets of ONE frame on the BEV grid (x forward 0..60 m, y left +-16 m, 0.5 m): the same cuboids, the same compact
    forms.  -> {"heat": [nx, ny] f32, "own": [nx, ny] i16 (slot or -1), "ctr": [nx, ny] f32, "census"}.  A cell is owned by the cuboid whose
    centre-ness there is largest (inside its oriented footprint, plus its centre cell); a cuboid whose CENTRE is off the grid is not a target
    (GT beyond the decode range is IGNORE, counted)."""
    nx, ny = grid.shape
    cell, yh = float(grid.cell_m), float(grid.y_half_m)
    xc = (np.arange(nx, dtype=np.float64) + 0.5) * cell
    yc = -yh + (np.arange(ny, dtype=np.float64) + 0.5) * cell
    heat = np.zeros((nx, ny), np.float32)
    own = np.full((nx, ny), -1, np.int64)
    ctr = np.zeros((nx, ny), np.float32)
    cen = {"n_boxes": int(boxes.shape[0]), "n_off_grid": 0, "n_heat_pos": 0, "n_unusable": 0}
    for k in range(boxes.shape[0]):
        cx, cy, _cz, l, w, _h, yaw = [float(x) for x in boxes[k]]
        if not (np.isfinite([cx, cy, l, w, yaw]).all() and l > 0 and w > 0):
            cen["n_unusable"] += 1
            continue
        i, j = int(math.floor(cx / cell)), int(math.floor((cy + yh) / cell))
        if not (0 <= i < nx and 0 <= j < ny):
            cen["n_off_grid"] += 1
            continue
        c, s_ = math.cos(yaw), math.sin(yaw)
        ex, ey = 0.5 * (abs(l * c) + abs(w * s_)), 0.5 * (abs(l * s_) + abs(w * c))
        _splat(heat, i, j, _sigma(2.0 * ey / cell), _sigma(2.0 * ex / cell))
        heat[i, j] = 1.0
        cen["n_heat_pos"] += 1
        a0, a1 = max(0, int((cx - ex) / cell) - 2), min(nx, int((cx + ex) / cell) + 3)
        b0, b1 = max(0, int((cy + yh - ey) / cell) - 2), min(ny, int((cy + yh + ey) / cell) + 3)
        dx, dy = xc[a0:a1, None] - cx, yc[None, b0:b1] - cy
        lon, lat = dx * c + dy * s_, -dx * s_ + dy * c                                 # world -> box frame (bev_raster.rasterize's rotation)
        hl, hw = max(l / 2.0, MIN_HALF_CELLS * cell), max(w / 2.0, MIN_HALF_CELLS * cell)
        inside = (np.abs(lon) <= hl) & (np.abs(lat) <= hw)
        ck = np.sqrt(np.clip((hl - np.abs(lon)) / (hl + np.abs(lon)), 0, None) * np.clip((hw - np.abs(lat)) / (hw + np.abs(lat)), 0, None))
        win_c = ctr[a0:a1, b0:b1]
        better = inside & (ck > win_c)
        win_c[better] = ck[better].astype(np.float32)
        own[a0:a1, b0:b1][better] = int(slots[k])
    return {"heat": heat, "own": own.astype(np.int16), "ctr": ctr, "census": cen}


# --------------------------------------------------------------------------------------------------------------- #
# losses (torch; the step)                                                                                        #
# --------------------------------------------------------------------------------------------------------------- #
def slot_tables(batch: dict, device) -> dict:
    """The per-slot regression targets, gathered later by the owner maps.  ``size`` log(l, w, h), ``yaw`` (sin, cos) -- all from the batch's
    own ``agent_*`` fields (the very cuboids the box3d loss reads)."""
    box = batch["agent_box"].to(device).float()
    h = batch["agent_h"].to(device).float()
    yaw = batch["agent_yaw"].to(device).float()
    size = torch.stack([box[..., 2], box[..., 3], h], dim=-1).clamp_min(1e-3).log()
    tab = {"size": size, "yaw": torch.stack([yaw.sin(), yaw.cos()], dim=-1), "xy": box[..., :2], "cz": batch["agent_cz"].to(device).float()}
    # a padded / invalid slot may hold NaN; the owner maps never gather it into a positive cell, but a NaN gathered at a masked cell would still
    # poison `NaN * 0` -- so the tables are finite by construction (0.0 at an invalid slot)
    return {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in tab.items()}


def take(table: Tensor, idx: Tensor) -> Tensor:
    """table [B, A] or [B, A, k] gathered by idx [B, Hs, Ws] (slot index, already clamped >= 0) -> [B, Hs, Ws] / [B, Hs, Ws, k]."""
    B = int(idx.shape[0])
    flat = idx.reshape(B, -1)
    if table.dim() == 2:
        return table.gather(1, flat).reshape(idx.shape)
    k = int(table.shape[-1])
    return table.gather(1, flat.unsqueeze(-1).expand(-1, -1, k)).reshape(*idx.shape, k)


def focal_heat(logit: Tensor, heat: Tensor, ign: Tensor | None = None):
    """CenterNet penalty-reduced focal loss on Gaussian targets: positives are the cells with target == 1 exactly.  -> (loss, n_pos)."""
    x = logit.float().reshape(heat.shape)
    pos = (heat >= 1.0 - 1e-6)
    neg = ~pos if ign is None else (~pos & ~ign)
    lp, ln = F.logsigmoid(x), F.logsigmoid(-x)
    p = torch.sigmoid(x)
    pl = -((1.0 - p) ** 2 * lp * pos)
    nl = -(((1.0 - heat) ** 4) * (p ** 2) * ln * neg)
    n = pos.sum()
    return (pl.sum() + nl.sum()) / n.clamp_min(1).to(pl.dtype), n


def _mean_pos(per_cell: Tensor, mask: Tensor) -> Tensor:
    return (per_cell * mask).sum() / mask.sum().clamp_min(1).to(per_cell.dtype)


def persp_loss(pred: dict, tg: dict, slots: dict):
    """``pred`` {stride: head dict}, ``tg`` {stride: {heat, own, ctr, ign}} + {"i0","w1"}: per stride the five parts, strides averaged.
    -> (loss, parts {name: tensor}, metrics)."""
    parts = {k: [] for k in PART_W}
    met: dict = {}
    for s, pr in pred.items():
        t = tg[s]
        own = t["own"].long()
        pos = own >= 0
        idx = own.clamp_min(0)
        l_heat, n_heat = focal_heat(pr["heat"], t["heat"], t["ign"])
        i0g, w1g = take(tg["i0"], idx), take(tg["w1"], idx)
        has = pos & (i0g >= 0)
        logp = F.log_softmax(pr["depth"].float(), dim=1)
        a = logp.gather(1, i0g.clamp(0, N_BINS - 1).unsqueeze(1))[:, 0]
        b = logp.gather(1, (i0g + 1).clamp(0, N_BINS - 1).unsqueeze(1))[:, 0]
        ce = -((1.0 - w1g) * a + w1g * b)
        sz = take(slots["size"], idx).permute(0, 3, 1, 2)
        yw = take(slots["yaw"], idx).permute(0, 3, 1, 2)
        posf = pos.to(ce.dtype)
        parts["heat"].append(l_heat)
        parts["depth"].append(_mean_pos(ce, has.to(ce.dtype)))
        parts["size"].append(_mean_pos(F.smooth_l1_loss(pr["size"].float(), sz, beta=0.1, reduction="none").sum(1), posf))
        parts["yaw"].append(_mean_pos((pr["yaw"].float() - yw).abs().sum(1), posf))
        parts["ctr"].append(_mean_pos(F.binary_cross_entropy_with_logits(pr["ctr"].float()[:, 0], t["ctr"], reduction="none"), posf))
        met[f"n_pos_s{s}"] = float(pos.sum())
        met[f"n_heat_pos_s{s}"] = float(n_heat)
    red = {k: sum(v) / float(len(v)) for k, v in parts.items()}
    return sum(PART_W[k] * red[k] for k in red), red, met


def bev2_cell_centres(grid, device):
    nx, ny = grid.shape
    cell, yh = float(grid.cell_m), float(grid.y_half_m)
    xc = ((torch.arange(nx, device=device, dtype=torch.float32) + 0.5) * cell)[:, None].expand(nx, ny)
    yc = (-yh + (torch.arange(ny, device=device, dtype=torch.float32) + 0.5) * cell)[None, :].expand(nx, ny)
    return xc, yc


def bev2_loss(pred: dict, tg: dict, slots: dict, grid):
    """B2c: heat focal; the 3-D position = sub-cell offset (L1, cell units) + centre z (smooth-L1, m); size; yaw; centre-ness."""
    own = tg["own"].long()
    pos = own >= 0
    idx = own.clamp_min(0)
    l_heat, n_heat = focal_heat(pred["heat"], tg["heat"])
    xc, yc = bev2_cell_centres(grid, own.device)
    xy = take(slots["xy"], idx)
    off_t = torch.stack([(xy[..., 0] - xc) / float(grid.cell_m), (xy[..., 1] - yc) / float(grid.cell_m)], dim=1)
    posf = pos.float()
    l_off = _mean_pos((pred["off"].float() - off_t).abs().sum(1), posf)
    l_cz = _mean_pos(F.smooth_l1_loss(pred["cz"].float()[:, 0], take(slots["cz"], idx), beta=1.0, reduction="none"), posf)
    sz = take(slots["size"], idx).permute(0, 3, 1, 2)
    yw = take(slots["yaw"], idx).permute(0, 3, 1, 2)
    red = {"heat": l_heat, "pos": l_off + l_cz,
           "size": _mean_pos(F.smooth_l1_loss(pred["size"].float(), sz, beta=0.1, reduction="none").sum(1), posf),
           "yaw": _mean_pos((pred["yaw"].float() - yw).abs().sum(1), posf),
           "ctr": _mean_pos(F.binary_cross_entropy_with_logits(pred["ctr"].float()[:, 0], tg["ctr"], reduction="none"), posf)}
    met = {"n_pos": float(pos.sum()), "n_heat_pos": float(n_heat), "pos_off": float(l_off.detach()), "pos_cz": float(l_cz.detach())}
    return sum(BEV2_PART_W[k] * red[k] for k in red), red, met
