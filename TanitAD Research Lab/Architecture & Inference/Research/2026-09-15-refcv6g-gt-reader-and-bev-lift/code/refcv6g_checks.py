#!/usr/bin/env python3
"""Real-data checks for the refcv6-grounded GT reader and BEV lift (dev box, CPU only).

Three measurements, each written to ``../raw/<name>.json`` (sha12 only, no clip ids):

  camvis   the lift's validity mask vs the LiDAR BEV GT ``cart_cam_vis`` on every local
           B1-eval LiDAR GT clip; an exact re-emulation of ``cam_vis``; the disagreement
           band; what the comparison can and cannot detect (perturbation arms).
  pixreg   pixel registration against the SAM3 GT, whose camera model is independent of
           ours: luminance AUC of lane-line vs plain-drivable cells, sampled at the lift's
           ground (z = 0) pixels on real cache frames, under column / row shifts and a
           mirror. The lift is registered if the peak sits at zero shift.
  align    time + pose alignment of the SAM3 GT against the v2ep episodes (reader API).

Inputs (dev-box paths, overridable by env): see ``PATHS``. Usage:
    PYTHONPATH=<worktree>/stack python refcv6g_checks.py camvis pixreg align
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tvio
from scipy import ndimage
from scipy.stats import rankdata

from tanitad.data import semantic_map_gt as G
from tanitad.data.calib import PHYSICALAI_WIDE120_256x640 as FR, cylindrical_grid
from tanitad.data.physicalai import (FrontWideExtrinsics, _calib_chunk_path,
                                     _chunk_of_clip, _load_chunk_extrinsics,
                                     _load_chunk_intrinsics)
from tanitad.data.rig_projection import RigCamera
from tanitad.models import bev_lift as L

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
PATHS = {
    "calib": os.environ.get("TANITAD_CALIB_DIR",
                            "D:/Projects/TanitAD-artifacts/hf-corpus-aug-20260915/stage/calibration"),
    "lidar_gt": os.environ.get("TANITAD_LIDAR_BEV_GT_DIR",
                               "D:/Projects/TanitAD-artifacts/bev-lidar-gt-b1eval-20260913"),
    "b1_root": os.environ.get("TANITAD_B1_ROOT", "C:/Users/Admin/tanitad-data/physicalai-b1"),
    "v2ep": os.environ.get("TANITAD_V2EP_EVAL_DIR", "D:/Projects/TanitAD-artifacts/refcv5cmp/data/eval"),
    "sam3_root": os.environ.get("TANITAD_SAM3_GT_ROOT", "C:/Users/Admin/tanitad-caches/refcv6g-20260915"),
    "cam_ts": os.environ.get("TANITAD_CAM_TS_DIR",
                             "C:/Users/Admin/tanitad-data/physicalai-b1/r0/camera_front_wide_120fov"),
}
CAM_VIS_Z_M = (0.30, 1.00, 2.00, 3.00)          # p2_build_corpus.py:103-104
SUB = (np.arange(4) + 0.5) / 4


def s12(cid: str) -> str:
    return hashlib.sha256(cid.encode()).hexdigest()[:12]


def _file_sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def provenance() -> dict:
    stack = Path(G.__file__).resolve().parents[2]
    return {"semantic_map_gt.py_sha256": _file_sha256(G.__file__),
            "bev_lift.py_sha256": _file_sha256(L.__file__),
            "checks_sha256": _file_sha256(__file__),
            "stack": str(stack), "torch": torch.__version__, "numpy": np.__version__,
            "t_unix": round(time.time())}


def v2ep_payload(cid: str) -> dict:
    return torch.load(Path(PATHS["v2ep"]) / f"{cid}.v2ep.pt", map_location="cpu", weights_only=False)


def decode_raw(payload: dict, i: int) -> torch.Tensor:
    lens = payload["jpeg_len"].to(torch.int64)
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
    dec = tvio.decode_png if payload.get("codec") == "png" else tvio.decode_jpeg
    return dec(payload["jpeg_buf"][int(offs[i]):int(offs[i + 1])], mode=tvio.ImageReadMode.RGB)


def frames_observed(payload: dict) -> torch.Tensor:
    """The LiDAR builder's observed-pixel mask: max over 6 frames > 0 (p2_build_corpus.py:206-213)."""
    T = len(payload["jpeg_len"])
    obs = torch.zeros(FR.height, FR.width, dtype=torch.bool)
    for i in np.linspace(0, T - 1, 6).round().astype(int):
        obs |= decode_raw(payload, int(i)).amax(dim=0) > 0
    return obs


def emulate_cam_vis(cam: RigCamera, obs: torch.Tensor) -> np.ndarray:
    """``p2_build_corpus.camera_visibility`` for the cart grid (p2_build_corpus.py:216-246),
    evaluated THROUGH THE LIFT: its cell centres and ``project_rig_points``."""
    Xc, Yc = L.cell_centers_xy(L.GRID_DEFAULT)
    cell, yh = L.GRID_DEFAULT.cell_m, L.GRID_DEFAULT.y_half_m
    ix, iy = Xc / cell - 0.5, (Yc + yh) / cell - 0.5                 # exact cell indices
    X = np.broadcast_to((ix[:, :, None, None] + SUB[None, None, :, None]) * cell, (120, 64, 4, 4))
    Y = np.broadcast_to((iy[:, :, None, None] + SUB[None, None, None, :]) * cell - yh, (120, 64, 4, 4))
    vis = np.zeros((120, 64))
    eps = 1e-6                                                        # rig_projection._PIX_EPS
    for z in CAM_VIS_Z_M:
        P = torch.as_tensor(np.stack([X, Y, np.full(X.shape, z)], -1).reshape(-1, 3), dtype=torch.float64)
        pr = L.project_rig_points(P, cam)
        col, row = pr["col"], pr["row"]
        valid = (pr["in_front"] & (col >= -eps) & (col <= FR.width - 1 + eps)
                 & (row >= -eps) & (row <= FR.height - 1 + eps))
        ci = col.round().long().clamp(0, FR.width - 1)
        ri = row.round().long().clamp(0, FR.height - 1)
        vis += (valid & obs[ri, ci]).numpy().reshape(120, 64, -1).mean(-1)
    return np.round(vis / len(CAM_VIS_Z_M) * 255.0).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(int((a | b).sum()), 1))


def band_distance(ref: np.ndarray, mine: np.ndarray) -> np.ndarray:
    """Chessboard distance (cells) of every disagreeing cell to the in/out boundary of
    ``ref``; the grid frame is NOT a boundary (border_value=1 in both erosions)."""
    k = np.ones((3, 3), bool)
    edge = ref & ~ndimage.binary_erosion(ref, k, border_value=1)
    edge |= ~ref & ~ndimage.binary_erosion(~ref, k, border_value=1)
    if not edge.any():
        return np.full(int((ref != mine).sum()), 10 ** 6)
    d = ndimage.distance_transform_cdt(~edge, metric="chessboard")
    return d[ref != mine]


def rotate_extr(e: FrontWideExtrinsics, yaw_deg=0.0, pitch_deg=0.0, dz=0.0) -> RigCamera:
    """Perturbation arm: rotate the camera about rig z (yaw, +left) / rig y (pitch, +down)."""
    R = torch.as_tensor(e.rotation_cam_to_vehicle(), dtype=torch.float64)
    a, b = math.radians(yaw_deg), math.radians(pitch_deg)
    Rz = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], dtype=torch.float64)
    Ry = torch.tensor([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]], dtype=torch.float64)
    return RigCamera(R_cam_to_rig=Rz @ Ry @ R, t_cam_in_rig=torch.tensor([e.x, e.y, e.z + dz], dtype=torch.float64), frame=FR)


# --------------------------------------------------------------------------- #
def run_camvis() -> dict:
    calib = Path(PATHS["calib"])
    ex = _load_chunk_extrinsics(str(calib / "sensor_extrinsics.parquet"))
    intr = _load_chunk_intrinsics(str(calib / "camera_intrinsics.parquet"))
    by12 = {s12(c): c for c in ex}
    chunk_of = _chunk_of_clip(PATHS["b1_root"])
    files = sorted(glob.glob(str(Path(PATHS["lidar_gt"]) / "*.bevgt.npz")))
    rows, dist_all, loc = [], [], {"near_field_rows": 0, "lateral_fov_edge": 0, "other": 0}
    for f in files:
        k = Path(f).name[:12]
        cid = by12.get(k)
        if cid is None:
            rows.append({"sha12": k, "error": "no calibration row"})
            continue
        with np.load(f) as z:
            cv = z["cart_cam_vis"]
        ref = cv >= 128
        e, ii = ex[cid], intr[cid]
        geo = L.build_lift_geometry(e)
        nv = geo.valid.sum(0).numpy()
        obs_i = cylindrical_grid(ii, int(ii.height), int(ii.width), FR)[1]
        nvo = L.build_lift_geometry(e, observed=obs_i).valid.sum(0).numpy()
        cam = L.camera_from_extrinsics(e)
        emu_i = emulate_cam_vis(cam, obs_i)
        r = {"sha12": k, "rig": "B" if ii.cy >= 650 else "A", "cy": round(float(ii.cy), 1),
             "pitch_deg": round(math.degrees(e.optical_axis_pitch_rad()), 3),
             "cam_height_m": round(float(e.z), 4), "ref_in_field_cells": int(ref.sum()),
             "iou_ge2of4": iou(nv >= 2, ref), "iou_any": iou(nv >= 1, ref), "iou_all4": iou(nv == 4, ref),
             "iou_ge2of4_observed": iou(nvo >= 2, ref),
             "n_disagree_ge2of4": int(((nv >= 2) != ref).sum()),
             "n_partial_cells": int(((nv > 0) & (nv < 4)).sum()),
             "n_cells_no_valid_height": int((nv == 0).sum()),
             "emulation_exact_intrinsics_obs": bool((emu_i == cv).all()),
             "emulation_maxdiff_intrinsics_obs": int(np.abs(emu_i.astype(int) - cv.astype(int)).max())}
        if not r["emulation_exact_intrinsics_obs"]:
            obs_f = frames_observed(v2ep_payload(cid))
            emu_f = emulate_cam_vis(cam, obs_f)
            r["emulation_exact_frames_obs"] = bool((emu_f == cv).all())
            r["obs_pixels_intrinsics_only"] = int((obs_i & ~obs_f).sum())
            r["obs_pixels_frames_only"] = int((~obs_i & obs_f).sum())
        # which calibration file the LiDAR builder read (physicalai-b1 per-chunk) == HF consolidated?
        ch = chunk_of.get(cid)
        pp = _calib_chunk_path(PATHS["b1_root"], "sensor_extrinsics", int(ch), download=False) if ch is not None else None
        if pp is not None:
            eb = _load_chunk_extrinsics(str(pp)).get(cid)
            r["extrinsics_equal_b1_chunk_vs_hf"] = bool(eb == e) if eb is not None else None
        d = band_distance(ref, nv >= 2)
        r["max_band_distance_cells"] = int(d.max()) if d.size else 0
        dist_all += d.tolist()
        dis = np.argwhere((nv >= 2) != ref)
        for i, j in dis:
            xc, yc = (i + 0.5) * 0.5, (j + 0.5) * 0.5 - 16.0
            az = math.degrees(math.atan2(abs(yc - e.y), xc - e.x)) if xc > e.x else 90.0
            if xc < 8.0 and az < 50.0:
                loc["near_field_rows"] += 1
            elif az >= 50.0:
                loc["lateral_fov_edge"] += 1
            else:
                loc["other"] += 1
        rows.append(r)
    ok = [r for r in rows if "error" not in r]
    dist_all = np.asarray(dist_all)

    # sensitivity: what the cam_vis comparison can and cannot see (first 20 clips)
    arms = {"mirror_y": None, "yaw_+2deg": dict(yaw_deg=2.0), "pitch_+2deg": dict(pitch_deg=2.0),
            "height_+0.30m": dict(dz=0.30), "nominal_camera_no_extrinsics": "nominal"}
    sens = {}
    for name, arm in arms.items():
        ious, maxd = [], []
        for r in ok[:20]:
            cid = by12[r["sha12"]]
            with np.load(Path(PATHS["lidar_gt"]) / f"{r['sha12']}.bevgt.npz") as z:
                ref = z["cart_cam_vis"] >= 128
            e = ex[cid]
            if arm is None:
                v = L.build_lift_geometry(e).valid.sum(0).numpy()[:, ::-1] >= 2
            elif arm == "nominal":
                v = L.build_lift_geometry(RigCamera.nominal(FR, height_m=1.3, x_m=1.9)).valid.sum(0).numpy() >= 2
            else:
                v = L.build_lift_geometry(rotate_extr(e, **arm)).valid.sum(0).numpy() >= 2
            ious.append(iou(v, ref))
            dd = band_distance(ref, v)
            maxd.append(int(dd.max()) if dd.size else 0)
        sens[name] = {"n_clips": len(ious), "iou_min": min(ious), "iou_median": float(np.median(ious)),
                      "max_band_distance_cells": max(maxd),
                      "clips_outside_1cell_band": int(sum(m > 1 for m in maxd))}

    # the exact uint8 re-emulation IS sensitive: cells that change under small errors (rig A)
    emu_sens = {"mirror_y": [], "yaw_+0.5deg": [], "pitch_+0.5deg": [], "height_+0.05m": []}
    for r in [r for r in ok if r["rig"] == "A"][:12]:
        cid = by12[r["sha12"]]
        e, ii = ex[cid], intr[cid]
        obs = cylindrical_grid(ii, int(ii.height), int(ii.width), FR)[1]
        with np.load(Path(PATHS["lidar_gt"]) / f"{r['sha12']}.bevgt.npz") as z:
            cv = z["cart_cam_vis"]
        base = emulate_cam_vis(L.camera_from_extrinsics(e), obs)
        emu_sens["mirror_y"].append(int((base[:, ::-1] != cv).sum()))
        emu_sens["yaw_+0.5deg"].append(int((emulate_cam_vis(rotate_extr(e, yaw_deg=0.5), obs) != cv).sum()))
        emu_sens["pitch_+0.5deg"].append(int((emulate_cam_vis(rotate_extr(e, pitch_deg=0.5), obs) != cv).sum()))
        emu_sens["height_+0.05m"].append(int((emulate_cam_vis(rotate_extr(e, dz=0.05), obs) != cv).sum()))
    emu_sens = {k: {"n_clips": len(v), "cells_changed_min": min(v), "cells_changed_max": max(v)}
                for k, v in emu_sens.items()}

    def summ(key, sel=ok):
        v = np.array([r[key] for r in sel], dtype=float)
        return {"n": int(v.size), "min": float(v.min()), "median": float(np.median(v)),
                "mean": float(v.mean()), "max": float(v.max())}

    out = {
        "what": "lift validity mask (valid at >= 2 of 4 heights) vs LiDAR BEV GT cart_cam_vis >= 128",
        "reference_independence": ("NOT independent in projection code: p2_build_corpus.camera_visibility uses "
                                   "tanitad.data.rig_projection.RigCamera.from_extrinsics(physicalai.FrontWideExtrinsics) "
                                   "(p2_build_corpus.py:197-199,215,240) -- the same projection the lift reuses. "
                                   "Independent parts: sample heights, sub-sampling, bounds, observed-pixel term, "
                                   "extrinsics file."),
        "n_files": len(files), "n_compared": len(ok), "n_errors": len(rows) - len(ok),
        "by_rig": {g: sum(r["rig"] == g for r in ok) for g in ("A", "B")},
        "iou_ge2of4": summ("iou_ge2of4"), "iou_any": summ("iou_any"), "iou_all4": summ("iou_all4"),
        "iou_ge2of4_observed": summ("iou_ge2of4_observed"),
        "iou_ge2of4_rigA": summ("iou_ge2of4", [r for r in ok if r["rig"] == "A"]),
        "iou_ge2of4_rigB": summ("iou_ge2of4", [r for r in ok if r["rig"] == "B"]),
        "disagreeing_cells_total": int(dist_all.size),
        "disagreeing_share_within_1_cell_of_boundary": float((dist_all <= 1).mean()) if dist_all.size else None,
        "disagreeing_max_distance_cells": int(dist_all.max()) if dist_all.size else 0,
        "disagreeing_location": loc,
        "partial_cells_per_clip": summ("n_partial_cells"),
        "emulation": {
            "exact_with_intrinsics_observed_mask": int(sum(r["emulation_exact_intrinsics_obs"] for r in ok)),
            "inexact_with_intrinsics_mask": int(sum(not r["emulation_exact_intrinsics_obs"] for r in ok)),
            "of_those_exact_with_frames_observed_mask": int(sum(r.get("emulation_exact_frames_obs", False) for r in ok)),
        },
        "extrinsics_b1_chunk_equal_hf_consolidated": {
            "n_checked": int(sum(r.get("extrinsics_equal_b1_chunk_vs_hf") is not None for r in ok)),
            "n_equal": int(sum(bool(r.get("extrinsics_equal_b1_chunk_vs_hf")) for r in ok))},
        "sensitivity_first20": sens,
        "emulation_sensitivity_rigA_first12": emu_sens,
        "per_clip": rows, "provenance": provenance(), "paths": {k: PATHS[k] for k in ("calib", "lidar_gt", "b1_root", "v2ep")},
    }
    return out


# --------------------------------------------------------------------------- #
def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _sample(Y: torch.Tensor, col: torch.Tensor, row: torch.Tensor) -> np.ndarray:
    g = torch.stack([2 * col / (FR.width - 1) - 1, 2 * row / (FR.height - 1) - 1], -1).view(1, 1, -1, 2)
    return F.grid_sample(Y.view(1, 1, FR.height, FR.width), g.to(Y.dtype), mode="bilinear",
                         align_corners=True).view(-1).numpy()


def run_pixreg() -> dict:
    ex = _load_chunk_extrinsics(str(Path(PATHS["calib"]) / "sensor_extrinsics.parquet"))
    root = PATHS["sam3_root"]
    ids = sorted(Path(p).name[:-len(G.GT_SUFFIX)] for p in glob.glob(str(Path(root).joinpath(*G.GT_SUBDIR) / f"*{G.GT_SUFFIX}")))
    shifts = list(range(-8, 9))
    out = {"what": ("luminance AUC of SAM3 lane-line cells (ch2 >= 0.5) vs plain drivable cells (ch1 >= 0.9), seen, "
                    "x in [cam_x + 4, 30] m, |y| <= 12 m, sampled at the lift's z = 0 pixels on the v2ep frame the "
                    "label belongs to (every 10th raw frame); columns / rows shifted by s px; mirror = cell (i, j) "
                    "sampled at the projection of (x, -y)"),
           "sam3_camera_model": ("independent: Thor-side sam3_paint.lift with the qwendrive camera parameters "
                                 "(Data Engineering 2026-09-13-sam3-only-road-map/code/sam3map_extract_v31.py:81-86,194-202)"),
           "clips": {}, "provenance": provenance()}
    for cid in ids:
        pay_path = Path(PATHS["v2ep"]) / f"{cid}.v2ep.pt"
        if not pay_path.is_file() or cid not in ex:
            out["clips"][s12(cid)] = {"skipped": "no v2ep payload or calibration"}
            continue
        gt = G.open_clip(root, cid)
        pay = v2ep_payload(cid)
        e = ex[cid]
        geo = L.build_lift_geometry(e, heights_m=(0.0,))
        col, row, val = geo.col[0], geo.row[0], geo.valid[0].numpy()
        colm, rowm, valm = col.flip(1), row.flip(1), val[:, ::-1]
        frames = list(range(10, gt.n_frames, 10))
        m = gt.read(np.array(frames))
        Xc = (np.arange(120) + 0.5) * 0.5
        Yc = (np.arange(64) + 0.5) * 0.5 - 16.0
        rng = (Xc[:, None] >= e.x + 4.0) & (Xc[:, None] <= 30.0) & (np.abs(Yc[None, :]) <= 12.0)
        acc = {("c", s): ([], []) for s in shifts} | {("r", s): ([], []) for s in shifts} | {("m", 0): ([], [])}
        for k, fi in enumerate(frames):
            img = decode_raw(pay, fi).double()
            Y = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            lane = (m.cart[k, 2] >= 0.5) & m.seen[k] & rng
            plain = (m.cart[k, 1] >= 0.9) & m.seen[k] & rng
            for s in shifts:
                for ax in ("c", "r"):
                    v = _sample(Y, col + (s if ax == "c" else 0), row + (s if ax == "r" else 0)).reshape(120, 64)
                    acc[(ax, s)][0].append(v[lane & val])
                    acc[(ax, s)][1].append(v[plain & val])
            v = _sample(Y, colm, rowm).reshape(120, 64)
            acc[("m", 0)][0].append(v[lane & valm])
            acc[("m", 0)][1].append(v[plain & valm])
        auc = {key: _auc(np.concatenate(p), np.concatenate(n)) for key, (p, n) in acc.items()}
        col_curve = {s: auc[("c", s)] for s in shifts}
        row_curve = {s: auc[("r", s)] for s in shifts}
        out["clips"][s12(cid)] = {
            "n_frames_used": len(frames),
            "n_lane_samples": int(sum(a.size for a in acc[("c", 0)][0])),
            "n_plain_samples": int(sum(a.size for a in acc[("c", 0)][1])),
            "auc_aligned": auc[("c", 0)], "auc_mirror": auc[("m", 0)],
            "col_shift_px_argmax": int(max(col_curve, key=col_curve.get)),
            "row_shift_px_argmax": int(max(row_curve, key=row_curve.get)),
            "col_curve": col_curve, "row_curve": row_curve,
            "cam_boresight_yaw_deg": round(math.degrees(math.atan2(
                float(e.rotation_cam_to_vehicle()[1, 2]), float(e.rotation_cam_to_vehicle()[0, 2]))), 3),
        }
    return out


# --------------------------------------------------------------------------- #
def run_align() -> dict:
    import pandas as pd
    root = PATHS["sam3_root"]
    ids = sorted(Path(p).name[:-len(G.GT_SUFFIX)] for p in glob.glob(str(Path(root).joinpath(*G.GT_SUBDIR) / f"*{G.GT_SUFFIX}")))
    out = {"clips": {}, "provenance": provenance()}
    for cid in ids:
        rec = {}
        gt = G.open_clip(root, cid)
        pay_path = Path(PATHS["v2ep"]) / f"{cid}.v2ep.pt"
        ts_path = Path(PATHS["cam_ts"]) / f"{cid}.timestamps.parquet"
        rec["has_v2ep"], rec["has_camera_timestamps"] = pay_path.is_file(), ts_path.is_file()
        if not (rec["has_v2ep"] and rec["has_camera_timestamps"]):
            out["clips"][gt.clip_sha12] = rec
            continue
        pay = torch.load(pay_path, map_location="cpu", weights_only=False, mmap=True)
        ts = pd.read_parquet(ts_path)
        grid = G.episode_frame_times_us(ts[next(k for k in ts.columns if "time" in k.lower())].to_numpy())
        T = len(pay["jpeg_len"])
        rec.update({"T_gt": gt.n_frames, "T_v2ep": T, "T_grid": int(len(grid["t_img_us"])),
                    "n_stack": int(pay["n_stack"]), "frame": pay["frame"], "codec": pay.get("codec"),
                    "grid_equal_gt_t_img_us": bool(np.array_equal(grid["t_img_us"], gt.t_img_us)),
                    "grid_equal_gt_cam_frame_idx": bool(np.array_equal(grid["cam_frame_idx"], gt.cam_frame_idx)),
                    "grid_equal_gt_t_query_us": bool(np.array_equal(grid["t_query_us"], gt.t_query_us)),
                    "check_times_worst_us": gt.check_times(np.arange(T), grid["t_img_us"])})
        refused = {}
        for name, fn in (("time_shift_1_frame", lambda: gt.check_times(np.arange(T - 1), grid["t_img_us"][1:])),
                         ("time_plus_1001us", lambda: gt.check_times(np.arange(T), grid["t_img_us"] + 1001.0)),
                         ("pose_shift_1_frame", lambda: gt.check_pose_alignment(np.roll(pay["poses"].numpy(), 1, 0)))):
            try:
                fn()
                refused[name] = False
            except G.TimeMisalignment:
                refused[name] = True
        rec["refused"] = refused
        rec["time_plus_999us_passes"] = gt.check_times(np.arange(T), grid["t_img_us"] + 999.0) == 999.0
        rep = gt.check_pose_alignment(pay["poses"].numpy())
        rec["pose_alignment"] = rep
        lg = Path(PATHS["lidar_gt"]) / f"{gt.clip_sha12}.bevgt.npz"
        if lg.is_file():
            with np.load(lg) as z:
                rec["lidar_gt_t_img_us_equal"] = bool(np.array_equal(z["t_img_us"], gt.t_img_us))
        out["clips"][gt.clip_sha12] = rec
    return out


def main(argv) -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    todo = argv or ["camvis", "pixreg", "align"]
    fns = {"camvis": ("cam_vis_crosscheck.json", run_camvis), "pixreg": ("pixel_registration.json", run_pixreg),
           "align": ("real_alignment.json", run_align)}
    for name in todo:
        fname, fn = fns[name]
        t0 = time.time()
        res = fn()
        res["wall_s"] = round(time.time() - t0, 1)
        txt = json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        (RAW / fname).write_text(txt, encoding="utf-8")
        print(f"[{name}] wrote {RAW / fname} in {res['wall_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
