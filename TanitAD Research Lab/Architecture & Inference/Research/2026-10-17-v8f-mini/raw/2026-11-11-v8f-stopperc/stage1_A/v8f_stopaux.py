"""PREREG_V8F_STOPPERC arm A: the stop-LOCATION label of the vision-only stop auxiliary (``--stop-aux on``).

LABELS MAY USE EGO (CLAUDE.md, 2026-08-03): the label is the cell of the 4x10 token grid into which the ego's REST POINT projects. ``stop_x_m`` / ``stop_y_m`` are the ego's
future position where it comes to rest, in the NOW frame (``build_v9_labels.py:937-942``; rig frame: +x forward, +y LEFT, z = 0 the road plane), carried in
``batch["lab"]["cons"]`` (cells ``CONS_FIELDS['stop_x_m' / 'stop_y_m']``, NaN = no stop within the horizon). The head that learns it reads vision only.

GEOMETRY (one implementation, ``rig_projection.project_cam_to_frame``): the frame is the corpus' CYLINDRICAL 256x640 (``calib.PHYSICALAI_WIDE120_256x640``,
f_ref 305.5775, HFOV 120 deg): column = (W-1)/2 + f x atan2(x_cam, z_cam) (LINEAR IN AZIMUTH; the pinhole 2 atan((W/2)/f) = 92.6 deg is the TRAPS.md error and is never used here),
row = (H-1)/2 + f x y_cam / hypot(x_cam, z_cam). The camera pose is the clip's own ``sensor_extrinsics`` row (quaternion cam -> vehicle + position), NEVER a constant height
(554 distinct heights in 2,400 clips). The token grid is ``pool4x10``: 4 rows x 10 columns of 64 x 64 px cells, row-major token = r x 10 + c (builder
``pull_encode_pooled.py:13``, ``pool_field``).
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch

from tanitad.data.calib import PHYSICALAI_WIDE120_256x640
from tanitad.data.rig_projection import project_cam_to_frame

GRID_ROWS, GRID_COLS = 4, 10
N_CELLS = GRID_ROWS * GRID_COLS
CELL_PX = 64                                   # 256 / 4 = 640 / 10
STOPAUX_FRAME = PHYSICALAI_WIDE120_256x640
#: below this share of the TRAIN clips with an extrinsics row, ``--stop-aux on`` is REFUSED (the term would train on a biased subset)
MIN_EXTR_COVERAGE = 0.99
CAM_FIELDS = ("qx", "qy", "qz", "qw", "x", "y", "z")


def load_extrinsics(path) -> dict:
    """``{clip_id: {qx, qy, qz, qw, x, y, z}}`` (the ``rig_extrinsics_train2400.json`` format of ``build_rig_extrinsics_table.py``)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def clip_cam_table(clip_ids, table: dict) -> tuple:
    """-> (cam ``[C, 7]`` float64 in CAM_FIELDS order, ok ``[C]`` bool): one row per clip, in the order of ``clip_ids`` (the join's ``J["clip"]``). A clip with no row (or a non-finite one) is ok = False."""
    cam = np.zeros((len(clip_ids), 7), np.float64)
    ok = np.zeros(len(clip_ids), bool)
    for i, c in enumerate(clip_ids):
        r = table.get(str(c))
        if r is None:
            continue
        v = np.array([float(r[k]) for k in CAM_FIELDS], np.float64)
        if np.isfinite(v).all() and float(np.linalg.norm(v[:4])) > 0.0:
            cam[i], ok[i] = v, True
    return cam, ok


def _rot_cam_to_vehicle(q: torch.Tensor) -> torch.Tensor:
    """[B, 4] (qx, qy, qz, qw) -> [B, 3, 3] camera -> vehicle: the batched twin of ``physicalai.FrontWideExtrinsics.rotation_cam_to_vehicle`` (same normalisation, same formula)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
                     2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
                     2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)], -1)
    return R.reshape(-1, 3, 3)


def stop_cell(stop_xy: torch.Tensor, cam: torch.Tensor, frame=STOPAUX_FRAME) -> tuple:
    """Project the rest point (x, y, z = 0 in the rig frame) ``stop_xy [B, 2]`` through the camera ``cam [B, 7]`` (CAM_FIELDS order).

    Returns ``(cell [B] long, valid [B] bool)``: ``cell`` = r x 10 + c of the 4x10 grid (-1 where not valid); ``valid`` is True only where the point is finite, in FRONT of the camera and inside the
    frame (``project_cam_to_frame``: behind the camera, past +-60 deg of azimuth, above row 0 or below row 255 are all INVALID = presence 0)."""
    p = stop_xy.to(torch.float64)
    cam = cam.to(torch.float64)
    fin = torch.isfinite(p).all(-1)
    rig = torch.cat([torch.nan_to_num(p), torch.zeros_like(p[:, :1])], -1)               # [B, 3], z = 0: the road plane (rig z = 0 IS the road, MEASURED over 87,481 cuboids)
    R = _rot_cam_to_vehicle(cam[:, :4])
    p_cam = torch.einsum("bi,bij->bj", rig - cam[:, 4:7], R)                               # p_cam = R^T (p_rig - t), row-vector form of ``rig_projection.rig_to_cam``
    col, row, valid = project_cam_to_frame(p_cam, frame)
    valid = valid & fin
    c = (col.clamp(0.0, frame.width - 1) / CELL_PX).floor().long().clamp(0, GRID_COLS - 1)
    r = (row.clamp(0.0, frame.height - 1) / CELL_PX).floor().long().clamp(0, GRID_ROWS - 1)
    return torch.where(valid, r * GRID_COLS + c, torch.full_like(c, -1)), valid


def make_stop_aux_arrays(J: dict, table: dict) -> dict:
    """The per-clip camera arrays aligned to the label join (``J["clip"]`` order), plus the coverage record the launch gate reads."""
    cam, ok = clip_cam_table(list(J["clip"]), table)
    return {"cam": cam, "ok": ok, "n_clips": int(len(ok)), "n_clips_with_extrinsics": int(ok.sum())}


#: v9 lon ids (``build_v9_labels.py``: 0 HOLD, 1 CREEP, 2 STOP, 3..6 the moving classes). A window with no rest point is a presence NEGATIVE only when the ego is
#: MOVING with no stop in the horizon (ids 3..6); HOLD / CREEP (already at rest or crawling) and an unknown class (IGNORE) carry no single-frame target and are left out.
NEG_LON_IDS = (3, 4, 5, 6)


def stop_aux_labels(lab_cons: torch.Tensor, lab_has: torch.Tensor, ci: torch.Tensor, SA: dict, ix_x: int, ix_y: int, lab_lon_cls: torch.Tensor) -> tuple:
    """-> ``(stop_cell [B] long (-1 = none), stop_pos [B] bool, stop_has [B] bool)``.

    * ``stop_pos``: the window carries a v9 label (``lab["has"]``), its clip has an extrinsics row, AND the rest point exists (finite ``stop_x_m`` / ``stop_y_m``) and projects into the
      frame: presence 1, and ``stop_cell`` is its 4x10 cell (the per-token term).
    * ``stop_has``: the rows of the presence BCE = ``stop_pos`` (presence 1) plus the windows with NO rest point whose lon class is a moving one (:data:`NEG_LON_IDS`, presence 0).
      LEFT OUT of the presence BCE (MM reading, AMENDMENT_3_A_READINGS.md, before any arm-A data): a rest point that exists but projects outside the frame (behind the camera, past
      +-60 deg, or closer than the ground is visible: the last metres of every stop), and a window with no rest point whose ego is at HOLD / CREEP or whose class is unknown.
      From ONE frame those scenes look like an approach to a stop; labelling them presence 0 would ask the head to separate what the frame cannot show."""
    ci = ci.long().cpu()
    cam = torch.as_tensor(SA["cam"])[ci]
    lab_ok = torch.as_tensor(SA["ok"])[ci] & lab_has.cpu().bool()
    xy = torch.stack([lab_cons[:, ix_x], lab_cons[:, ix_y]], -1).cpu()
    cell, valid = stop_cell(xy, cam)
    pos = valid & lab_ok
    lon = lab_lon_cls.cpu().long()
    neg = lab_ok & ~torch.isfinite(xy).any(-1) & (lon >= NEG_LON_IDS[0]) & (lon <= NEG_LON_IDS[-1])
    return torch.where(pos, cell, torch.full_like(cell, -1)), pos, pos | neg
