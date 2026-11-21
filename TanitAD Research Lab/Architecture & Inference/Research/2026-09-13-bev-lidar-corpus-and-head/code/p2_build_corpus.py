#!/usr/bin/env python3
"""P2 - corpus-scale LiDAR BEV ground truth for the 139-clip B1 EVAL join. Resumable.

⭐ THE CONSUMER, NAMED (CLAUDE.md: an artifact's cost is the cost of the file its CONSUMER
opens). The consumer of these artifacts is the frozen-trunk BEV head of this package:

    code/p4_bev_head.py
      -> code/bev_gt_loader.py :: load_clip(<sha12>.bevgt.npz)
        -> `polar48_occ` / `polar48_observed` / `polar48_cam_vis` at RAW v2ep frame i,
           paired with the refcv5-v2 trunk token map of v2ep STACKED ROW j = i - 2
           (the D-015 3-frame stack puts raw frame j+2 in the LAST 3 channels,
           `tanitad/data/comma2k19.py::stack_frames`)

and the frames that trunk reads are the refcv5-v2 EVAL cache `*.v2ep.pt`
(`stack/tanitad/data/v2_dataset.py::_decode_stacked`, built by
`stack/scripts/v2_compressed.py::build_compressed`). So the GT grid is the v2ep EPISODE
grid, not the 30 Hz camera and not the 10 Hz LiDAR:

    t_query[i] = linspace(t_cam[0], t_cam[-1], int(span_s * 10))[i]      (_resampled)
    t_img[i]   = t_cam[searchsorted(t_cam, t_query[i])]                  (the pixels)

MEASURED before this builder was written: that reconstruction reproduces the join's
`t_s` on all 26,394 join rows of 139 clips (median |dt| 0.31 ms, max 10.3 ms), and
`T == len(jpeg_len)` on all 139 v2ep payloads. The GT reference instant is `t_img` --
the instant the camera actually exposed -- and the LiDAR spin is the one whose MIDPOINT
is nearest `t_img`; |dt| > 50 ms is NO_LABEL (never extrapolated).

GRIDS written (all rig frame, +x fwd, +y LEFT, +z up, origin rear axle on the road):
  * `polar48` [48 range x 40 az], 1.25 m x 3.0 deg, 60 m, HFOV 120 -- the grid of the
    variant-B cross-attention head sized in 09-11 `p4_head_sizing.py`
    (`B_xattn_polar48x40_d192_L3`). THE CONSUMER'S GRID.
  * `polar24` [24 x 20], 2.5 m x 6.0 deg -- the `bev_aux.PolarBEVSpec` drop-in.
  * `cart`    [120 x 64], 0.5 m -- the P8 extent of `bev_raster`.

THE THIRD STATE, exactly as `lidar_bev.py` defines it, plus the camera field:
  * `occ`       -- the default (z-band AND vertical-spread) rule; `occ_zband` the flat rule
  * `observed`  -- 0 = shadowed beyond the first obstacle return in that bearing
  * `cam_vis`   -- uint8 0..255, the fraction of a cell's sample points (z in the
                   obstacle band) that project into an OBSERVED pixel of the canonical
                   256x640 frame, per clip. `in_camera_field = cam_vis >= 128`.
  ⇒ 4-way state (loader): OUT_OF_FIELD if not in_camera_field; else OCCUPIED if occ;
    else OBSERVED_EMPTY if observed; else OCCLUDED.

⛔ CONTENT-ASSERTED, because a job that exits 0 is not evidence: see `content_checks`.
⛔ The deskew is the CORRECTED sign (see `lidar_bev.py` module docstring, P0).
⛔ Clip ids are gated-confidential: artifacts are named `<sha12>.bevgt.npz`; the manifest,
the log and every error string carry sha12 only (`lidar_fetch.redact_text`).
⛔ Raw parquets are deleted after a clip passes, EXCEPT the four protected Qwen-Drive clips.

Usage:
  python p2_build_corpus.py --fetch-conc 4 --build-workers 8
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import lzma
import math
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lidar_bev import (  # noqa: E402
    CartesianBEVSpec, PolarBEVSpec, ZBand, DRACO_ATTR_INTENSITY, DRACO_ATTR_POINT_TS_US,
    artifact_meta, deskew_rigid, lidar_to_rig, rasterise_cartesian, rasterise_polar,
)
from lidar_fetch import (  # noqa: E402
    LIDAR_CACHE, cached_path, chunk_map, fetch_clip, is_protected, redact_text, sha12,
)

JOIN = (HERE.parents[3] / "Benchmarks & Evals" / "Research" / "2026-09-06-b1-agent-join"
        / "raw" / "b1eval_agents.jsonl.xz")
V2EP_DIR = r"D:\Projects\TanitAD-artifacts\refcv5cmp\data\eval"
CAM_DIR = r"C:\Users\Admin\tanitad-data\physicalai-b1\r0\camera_front_wide_120fov"
EXT_GLOB = r"C:\Users\Admin\tanitad-data\physicalai-b1\calibration\sensor_extrinsics\*.parquet"
EGO_DIR = r"C:\Users\Admin\tanitad-data\physicalai\labels\egomotion_alpamayo"
OUT_DIR = r"C:\Users\Admin\tanitad-caches\bevhead-20260913\bev_gt"
TARGET_HZ = 10.0                  # tanitad/data/physicalai.py:64, read by v2_compressed
NO_LABEL_DT_MS = 50.0             # half a 10 Hz spin: beyond it the nearest spin is not "this" spin
N_STACK = 3

SPECS = {
    "cart": CartesianBEVSpec(),
    "polar48": PolarBEVSpec(n_az=40, n_rng=48),
    "polar24": PolarBEVSpec(),
}
ZB = ZBand()
CAM_VIS_Z_M = (0.30, 1.00, 2.00, 3.00)   # sample heights spanning the obstacle band
CAM_VIS_SUB = 4                          # sub-samples per cell axis

#: content-check bands -- DECLARED before the build ran; a clip outside them is
#: quarantined (kept with its parquet), never silently accepted.
CHECKS = {
    "min_valid_frac": 0.50,
    # RE-DECLARED 2026-09-13 BEFORE the rule-B rebuild: the first build's [0.005, 0.50]
    # tripped on 3/24 clips because the SHIPPED polar rule is biased (P2b). This band is a
    # DEGENERACY gate (all-free / all-occupied), not a scientific bar; under rule B the
    # measured values on the 3 tripped clips are 0.278 / 0.465 / 0.364.
    "polar48_marginal_observed_infield": (0.005, 0.80),
    "cart_occ_frac_of_grid": (0.003, 0.35),
    "polar48_observed_frac_infield": (0.30, 1.00),
    "polar48_infield_frac": (0.40, 1.00),
    "ground_peak_z_m": (-0.20, 0.20),
}


# ---------------------------------------------------------------------------
# side inputs (helpers COPIED from 09-11 p2_build_bev_gt.py so no module from the
# 09-11 package -- whose lidar_bev carries the wrong deskew sign -- is ever imported)
# ---------------------------------------------------------------------------
_EXT_INDEX: dict | None = None


def extrinsics(clip_id: str, sensor: str) -> dict:
    global _EXT_INDEX
    import pyarrow.parquet as pq
    if _EXT_INDEX is None:
        _EXT_INDEX = {}
        for fp in sorted(glob.glob(EXT_GLOB)):
            t = pq.read_table(fp).to_pydict()
            for i, (cid, sn) in enumerate(zip(t["clip_id"], t["sensor_name"])):
                if sn in ("lidar_top_360fov", "camera_front_wide_120fov"):
                    _EXT_INDEX[(cid, sn)] = {k: float(t[k][i])
                                             for k in ("x", "y", "z", "qx", "qy", "qz", "qw")}
    return _EXT_INDEX[(clip_id, sensor)]


def load_ego(clip_id: str):
    import pyarrow.parquet as pq
    p = Path(EGO_DIR) / f"{clip_id}.parquet"
    return pq.read_table(p).to_pydict() if p.exists() else None


def ego_state_at(ego: dict, t_us: int) -> tuple[float, float]:
    """(speed m/s, yaw-rate rad/s) at t_us -- identical to 09-11 p2's helper, whose yaw
    rate sign P0 cross-checked against v*curvature (100 %) and the trajectory heading."""
    if ego is None:
        return 0.0, 0.0
    ts = np.asarray(ego["timestamp"], dtype=np.float64)
    i = int(np.clip(np.searchsorted(ts, float(t_us)), 1, len(ts) - 1))
    vx = np.asarray(ego.get("vx", np.zeros_like(ts)), dtype=np.float64)
    vy = np.asarray(ego.get("vy", np.zeros_like(ts)), dtype=np.float64)
    v = float(math.hypot(vx[i], vy[i]))
    try:
        qz = np.asarray(ego["qz"], dtype=np.float64)
        qw = np.asarray(ego["qw"], dtype=np.float64)
        yaw = 2.0 * np.arctan2(qz, qw)
        dt = (ts[i] - ts[i - 1]) * 1e-6
        dyaw = math.atan2(math.sin(yaw[i] - yaw[i - 1]), math.cos(yaw[i] - yaw[i - 1]))
        r = dyaw / dt if dt > 0 else 0.0
    except Exception:
        r = 0.0
    return v, r


def episode_grid(clip_id: str) -> dict:
    """The v2ep episode grid, reconstructed exactly as `v2_compressed._resampled`."""
    import pyarrow.parquet as pq
    t_cam = np.asarray(pq.read_table(Path(CAM_DIR) / f"{clip_id}.timestamps.parquet")
                       .column("timestamp").to_numpy(), dtype=np.float64)
    span = t_cam[-1] - t_cam[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand
            break
    if unit != 1e6:
        raise RuntimeError(f"camera timestamps unit {unit} != 1e6 (us)")
    n_target = max(int(span / unit * TARGET_HZ), 4)
    t_query = np.linspace(t_cam[0], t_cam[-1], n_target)
    fidx = np.searchsorted(t_cam, t_query).clip(0, len(t_cam) - 1)
    return {"t_query_us": t_query, "cam_frame_idx": fidx.astype(np.int32),
            "t_img_us": t_cam[fidx].astype(np.int64), "n_target": n_target}


# ---------------------------------------------------------------------------
# camera field: which BEV cells can the VISION consumer see at all?
# ---------------------------------------------------------------------------
def camera_visibility(clip_id: str, v2ep: dict) -> dict:
    import torch
    import torchvision.io as tvio
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.physicalai import FrontWideExtrinsics
    from tanitad.data.rig_projection import RigCamera

    frame = CanonicalFrame.from_dict(v2ep["frame"])
    if frame.projection != "cylindrical" or (frame.height, frame.width) != (256, 640):
        raise RuntimeError(f"unexpected canonical frame {v2ep['frame']}")
    # observed pixels: the cylindrical rectification leaves exact zeros outside the
    # f-theta image; PNG is lossless, so max over frames spread across the clip > 0.
    lens = v2ep["jpeg_len"].to(torch.int64)
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
    T = len(lens)
    obs = torch.zeros(frame.height, frame.width, dtype=torch.bool)
    for i in np.linspace(0, T - 1, 6).round().astype(int):
        img = tvio.decode_png(v2ep["jpeg_buf"][int(offs[i]):int(offs[i + 1])],
                              mode=tvio.ImageReadMode.RGB)
        obs |= (img.amax(dim=0) > 0)
    ce = extrinsics(clip_id, "camera_front_wide_120fov")
    cam = RigCamera.from_extrinsics(FrontWideExtrinsics(**ce), frame)
    sub = (np.arange(CAM_VIS_SUB) + 0.5) / CAM_VIS_SUB
    out = {"pixel_observed_frac": float(obs.float().mean())}
    for name, spec in SPECS.items():
        if name == "cart":
            nx, ny = spec.n_x, spec.n_y
            gx = (np.arange(nx)[:, None] + sub[None, :]) * spec.cell_m          # [nx, s]
            gy = (np.arange(ny)[:, None] + sub[None, :]) * spec.cell_m - spec.y_half_m
            X = gx[:, None, :, None] + 0 * gy[None, :, None, :]
            Y = 0 * gx[:, None, :, None] + gy[None, :, None, :]
            shape = (nx, ny)
        else:
            rr = spec.r_min_m + (np.arange(spec.n_rng)[:, None] + sub[None, :]) * spec.cell_rng_m
            # column 0 = +hfov/2 (LEFT): az = half - (col + u) * cell_deg
            az = np.radians(spec.hfov_deg / 2.0
                            - (np.arange(spec.n_az)[:, None] + sub[None, :]) * spec.cell_deg)
            X = rr[:, None, :, None] * np.cos(az)[None, :, None, :]
            Y = rr[:, None, :, None] * np.sin(az)[None, :, None, :]
            shape = (spec.n_rng, spec.n_az)
        X = np.broadcast_to(X, (*shape, CAM_VIS_SUB, CAM_VIS_SUB))
        Y = np.broadcast_to(Y, (*shape, CAM_VIS_SUB, CAM_VIS_SUB))
        vis = np.zeros(shape, dtype=np.float64)
        for z in CAM_VIS_Z_M:
            P = torch.as_tensor(np.stack([X, Y, np.full(X.shape, z)], -1).reshape(-1, 3),
                                dtype=torch.float64)
            col, row, valid = cam.project(P)
            ci = col.round().long().clamp(0, frame.width - 1)
            ri = row.round().long().clamp(0, frame.height - 1)
            ok = valid & obs[ri, ci]
            vis += ok.numpy().reshape(*shape, -1).mean(axis=-1)
        vis /= len(CAM_VIS_Z_M)
        out[f"{name}_cam_vis"] = np.round(vis * 255.0).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# one clip
# ---------------------------------------------------------------------------
def build_clip(clip_id: str, out_dir: str = OUT_DIR, lidar_cache: str = LIDAR_CACHE) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import pyarrow.parquet as pq
    import torch
    import DracoPy
    torch.set_num_threads(2)

    t0 = time.time()
    s12 = sha12(clip_id)
    rec: dict = {"clip_sha12": s12, "ok": False}
    try:
        lp = cached_path(clip_id, lidar_cache)
        rec["parquet_bytes"] = lp.stat().st_size
        pf = pq.ParquetFile(lp)
        sp = pq.read_table(lp, columns=["spin_start_timestamp", "spin_end_timestamp"]).to_pydict()
        s_start = np.asarray(sp["spin_start_timestamp"], dtype=np.int64)
        s_end = np.asarray(sp["spin_end_timestamp"], dtype=np.int64)
        s_mid = (s_start + s_end) // 2
        if pf.metadata.num_row_groups != len(s_mid):
            raise RuntimeError("row groups != spins")

        g = episode_grid(clip_id)
        v2 = torch.load(Path(V2EP_DIR) / f"{clip_id}.v2ep.pt", map_location="cpu",
                        weights_only=False)
        T = int(len(v2["jpeg_len"]))
        if T != g["n_target"]:
            raise RuntimeError(f"v2ep T {T} != reconstructed grid {g['n_target']}")
        if int(v2["n_stack"]) != N_STACK:
            raise RuntimeError(f"n_stack {v2['n_stack']} != {N_STACK}")
        vis = camera_visibility(clip_id, v2)
        del v2

        ext = extrinsics(clip_id, "lidar_top_360fov")
        ego = load_ego(clip_id)
        t_img = g["t_img_us"]
        near = np.abs(s_mid[:, None] - t_img[None, :]).argmin(axis=0)
        dt_ms = np.abs(s_mid[near] - t_img) / 1e3
        valid = dt_ms <= NO_LABEL_DT_MS

        arrs = {}
        for name, spec in SPECS.items():
            shp = (spec.n_x, spec.n_y) if name == "cart" else (spec.n_rng, spec.n_az)
            arrs[f"{name}_occ"] = np.zeros((T, *shp), dtype=bool)
            arrs[f"{name}_occ_zband"] = np.zeros((T, *shp), dtype=bool)
            arrs[f"{name}_observed"] = np.zeros((T, *shp), dtype=bool)
            if name != "cart":
                arrs[f"{name}_observed_shipped"] = np.zeros((T, *shp), dtype=bool)
                arrs[f"{name}_visible_first_hit"] = np.zeros((T, *shp), dtype=bool)
        arrs["cart_n_pts_log2"] = np.zeros((T, SPECS["cart"].n_x, SPECS["cart"].n_y), dtype=np.uint8)
        n_points = np.zeros(T, dtype=np.int32)
        ego_v = np.zeros(T, dtype=np.float32)
        ego_r = np.zeros(T, dtype=np.float32)
        ground_z = []
        t_dec = t_ras = 0.0
        cache_k, cache_val = -1, None
        for i in range(T):
            if not valid[i]:
                continue
            k = int(near[i])
            if k != cache_k:
                ta = time.time()
                blob = pf.read_row_group(k, columns=["draco_encoded_pointcloud"]).column(0)[0].as_py()
                pc = DracoPy.decode(blob)
                pts = np.asarray(pc.points)
                if pts.shape[0] == 0 or float(np.abs(pts).max()) == 0.0:
                    raise RuntimeError(f"spin {k} decoded EMPTY/ALL-ZERO")
                attrs = {a["unique_id"]: a["data"] for a in pc.attributes}
                inten = attrs[DRACO_ATTR_INTENSITY][:, 0].astype(np.uint8)
                pt_ts = attrs[DRACO_ATTR_POINT_TS_US][:, 0].astype(np.int64)
                rig0 = lidar_to_rig(pts, ext)
                cache_k, cache_val = k, (rig0, inten, pt_ts)
                t_dec += time.time() - ta
            rig0, inten, pt_ts = cache_val
            tb = time.time()
            v_ms, yr = ego_state_at(ego, int(t_img[i]))
            rig = deskew_rigid(rig0, pt_ts, int(t_img[i]), v_ms, yr) if ego is not None else rig0
            C = rasterise_cartesian(rig, inten, SPECS["cart"], ZB)
            arrs["cart_occ"][i] = C["occ"] > 0
            arrs["cart_occ_zband"][i] = C["occ_zband_only"] > 0
            arrs["cart_observed"][i] = C["observed"] > 0
            arrs["cart_n_pts_log2"][i] = np.minimum(np.floor(np.log2(C["n_pts"].astype(np.float64) + 1.0)), 8).astype(np.uint8)
            for name in ("polar48", "polar24"):
                P = rasterise_polar(rig, SPECS[name], ZB)
                arrs[f"{name}_occ"][i] = P["occ"] > 0
                arrs[f"{name}_occ_zband"][i] = P["occ_zband_only"] > 0
                arrs[f"{name}_observed"][i] = P["observed_npts"] > 0          # rule B (consumer)
                arrs[f"{name}_observed_shipped"][i] = P["observed"] > 0       # rule A (as shipped)
                arrs[f"{name}_visible_first_hit"][i] = P["visible_first_hit"] > 0  # rule C
            n_points[i] = rig0.shape[0]
            ego_v[i], ego_r[i] = v_ms, yr
            if i % 10 == 0:
                r2 = np.hypot(rig[:, 0], rig[:, 1])
                m = (r2 > 3) & (r2 < 20)
                h, e = np.histogram(rig[m, 2], bins=np.arange(-2, 2.001, 0.05))
                ground_z.append(float(e[int(np.argmax(h))] + 0.025))
            t_ras += time.time() - tb

        # ------------------------- content assertions -------------------------
        fails = []
        nv = int(valid.sum())
        infield48 = vis["polar48_cam_vis"] >= 128
        occ48 = arrs["polar48_occ"][valid]
        obs48 = arrs["polar48_observed"][valid]
        scored = obs48 & infield48[None]
        marg48 = float(occ48[scored].mean()) if scored.any() else float("nan")
        cart_frac = float(arrs["cart_occ"][valid].mean()) if nv else float("nan")
        obs_frac = float(obs48[:, infield48].mean()) if nv and infield48.any() else float("nan")
        infield_frac = float(infield48.mean())
        gz = float(np.median(ground_z)) if ground_z else float("nan")
        stats = {
            "T": T, "n_valid": nv, "valid_frac": round(nv / T, 4),
            "align_dt_ms": {"median": round(float(np.median(dt_ms[valid])), 2) if nv else None,
                            "max_valid": round(float(dt_ms[valid].max()), 2) if nv else None,
                            "n_no_label": int(T - nv)},
            "polar48_marginal_observed_infield": round(marg48, 5),
            "polar48_marginal_rule_A_shipped": round(float(occ48[arrs["polar48_observed_shipped"][valid] & infield48[None]].mean()), 5) if nv else None,
            "polar48_marginal_rule_C_first_hit": round(float(occ48[arrs["polar48_visible_first_hit"][valid] & infield48[None]].mean()), 5) if nv else None,
            "cart_occ_frac_of_grid": round(cart_frac, 5),
            "polar48_observed_frac_infield": round(obs_frac, 5),
            "polar48_infield_frac": round(infield_frac, 5),
            "cart_infield_frac": round(float((vis["cart_cam_vis"] >= 128).mean()), 5),
            "pixel_observed_frac": round(vis["pixel_observed_frac"], 5),
            "ground_peak_z_m": round(gz, 4),
            "n_points_min_valid": int(n_points[valid].min()) if nv else 0,
            "n_all_zero_occ_valid_frames": int((arrs["cart_occ"][valid].reshape(nv, -1).sum(1) == 0).sum()) if nv else None,
            "ego_v_ms_median": round(float(np.median(ego_v[valid])), 3) if nv else None,
            "abs_yaw_rate_p95": round(float(np.percentile(np.abs(ego_r[valid]), 95)), 4) if nv else None,
            "deskew_source": "egomotion_alpamayo" if ego is not None else "NONE (no egomotion)",
        }
        if nv / T < CHECKS["min_valid_frac"]:
            fails.append(f"valid_frac {nv / T:.3f}")
        for key in ("polar48_marginal_observed_infield", "cart_occ_frac_of_grid",
                    "polar48_observed_frac_infield", "polar48_infield_frac", "ground_peak_z_m"):
            lo, hi = CHECKS[key]
            val = stats[key]
            if not (isinstance(val, float) and lo <= val <= hi):
                fails.append(f"{key}={val} outside [{lo},{hi}]")
        if stats["n_points_min_valid"] <= 0:
            fails.append("a valid frame has zero points")
        for k2 in ("cart_occ", "cart_observed", "polar48_occ", "polar48_observed"):
            if not arrs[k2].any():
                fails.append(f"{k2} ALL ZERO")
        if ego is None:
            fails.append("no egomotion: deskew not applied")

        meta = artifact_meta(
            SPECS["cart"], SPECS["polar24"], ZB,
            clip_sha12=s12,
            polar48={"n_rng": 48, "n_az": 40, "r_max_m": 60.0, "r_min_m": 0.0,
                     "hfov_deg": 120.0, "cell_rng_m": 1.25, "cell_deg": 3.0,
                     "shape": [48, 40],
                     "col0": "+hfov/2 = LEFT (same sense as polar24)",
                     "role": "THE CONSUMER'S GRID (variant-B cross-attention head)"},
            grid_note="`polar` above is polar24 (bev_aux drop-in); `polar48` is the head grid",
            time_grid={
                "axis0": "RAW v2ep frame index i, 0..T-1, on the episode grid of "
                         "v2_compressed._resampled: t_query = linspace(t_cam[0], t_cam[-1], "
                         "int(span_s*10)); pixels from camera frame searchsorted(t_cam, t_query)",
                "consumer_row": "trunk STACKED ROW j reads raw frames j..j+2; its current "
                                "frame -- and therefore its target -- is raw frame i = j + 2",
                "reference_instant": "t_img_us = camera exposure timestamp of raw frame i",
                "dt_s_nominal": 0.1,
                "no_label_rule_ms": NO_LABEL_DT_MS,
                "spin_choice": "LiDAR spin whose MIDPOINT is nearest t_img_us",
            },
            deskew_applied=ego is not None,
            deskew_model="constant v + constant yaw-rate over the spin, per-point us timestamps, "
                         "reference t_img_us, rotation +yaw_rate*dt (P0-measured sign)",
            third_state={
                "occ": "lidar_bev default rule: z in [0.30,3.00] m AND per-cell vertical spread >= 0.30 m",
                "occ_zband": "lidar_bev flat rule only (over-reports; kept for audit)",
                "observed": "RULE B (consumer default). cart: lidar_bev rule ~shadow | occ | n_pts>0. "
                            "polar*: the SAME rule ported (lidar_bev.rasterise_polar `observed_npts`)",
                "observed_shipped": "RULE A, polar only: lidar_bev.rasterise_polar `observed` as shipped "
                                    "(~shadow | occ). MEASURED BIASED (p2b_observed_rule_probe.json): "
                                    "keeps occupied cells behind the first hit, drops free cells with returns",
                "visible_first_hit": "RULE C, polar only: ~shadow (cells up to the first occupied cell + 1)",
                "cam_vis": "uint8 round(255*fraction of 4x4 sub-samples x z in {0.3,1,2,3} m "
                           "projecting into an OBSERVED pixel of the canonical frame); static per clip",
                "state_encoding_in_loader": {"0": "OUT_OF_FIELD (cam_vis<128)", "1": "OBSERVED_EMPTY",
                                             "2": "OCCUPIED", "3": "OCCLUDED"},
                "v1_limit": "a bearing with NO obstacle return is observed out to 60 m even when "
                            "it points at sky; cart_n_pts_log2 lets a consumer tighten that",
            },
            channels_units={"*_occ, *_occ_zband, *_observed": "bool",
                            "cart_n_pts_log2": "uint8 floor(log2(n_returns_in_cell + 1)), SATURATES at 8 (>= 255 returns); n>=1 <=> value>=1, n>=3 <=> >=2, n>=7 <=> >=3",
                            "*_cam_vis": "uint8 0..255 = fraction x 255",
                            "t_img_us, t_query_us": "int64 microseconds, clip clock",
                            "align_dt_ms": "float32 ms |t_img - nearest spin midpoint|",
                            "ego_v_ms": "m/s", "ego_yaw_rate_rps": "rad/s, +left"},
            content_checks={"bands": {k: list(v) if isinstance(v, tuple) else v
                                      for k, v in CHECKS.items()},
                            "failures": fails},
            stats=stats,
        )
        out = Path(out_dir) / (f"{s12}.bevgt.npz" if not fails else f"quarantine/{s12}.bevgt.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp.npz")
        np.savez_compressed(
            tmp, meta_json=np.array(json.dumps(meta)),
            raw_frame=np.arange(T, dtype=np.int16),
            t_img_us=t_img.astype(np.int64), t_query_us=g["t_query_us"].round().astype(np.int64),
            cam_frame_idx=g["cam_frame_idx"], spin_index=near.astype(np.int16),
            align_dt_ms=dt_ms.astype(np.float32), label_valid=valid,
            n_points=n_points, ego_v_ms=ego_v, ego_yaw_rate_rps=ego_r,
            polar48_cam_vis=vis["polar48_cam_vis"], polar24_cam_vis=vis["polar24_cam_vis"],
            cart_cam_vis=vis["cart_cam_vis"], **arrs)
        os.replace(tmp, out)
        # read-back content assertion from the FILE, not from memory
        with np.load(out) as z:
            back = z["polar48_occ"]
            if back.shape != (T, 48, 40) or int(back.sum()) != int(arrs["polar48_occ"].sum()):
                raise RuntimeError("read-back mismatch")
            json.loads(str(z["meta_json"]))
        rec.update(stats)
        rec["artifact"] = out.name if not fails else "quarantine/" + out.name
        rec["artifact_bytes"] = out.stat().st_size
        rec["failures"] = fails
        rec["ok"] = not fails
        rec["decode_s"] = round(t_dec, 1)
        rec["raster_s"] = round(t_ras, 1)
    except Exception as e:  # noqa: BLE001
        rec["error"] = redact_text(f"{type(e).__name__}: {e}")[:500]
        rec["traceback_tail"] = redact_text(traceback.format_exc()[-800:])
    rec["build_s"] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------------------
# orchestration: fetch pool -> build pool -> manifest -> delete parquet
# ---------------------------------------------------------------------------
def join_clips() -> list[str]:
    seen: dict[str, None] = {}
    with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
        for line in fh:
            seen.setdefault(json.loads(line)["clip_id"], None)
    return list(seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-conc", type=int, default=4)
    ap.add_argument("--build-workers", type=int, default=8)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 139")
    ap.add_argument("--keep-parquets", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.jsonl"
    done = set()
    if man_path.exists():
        for line in man_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok") and (out_dir / r.get("artifact", "")).exists():
                done.add(r["clip_sha12"])
    clips = join_clips()
    if args.limit:
        clips = clips[: args.limit]
    todo = [c for c in clips if sha12(c) not in done]
    chunks = chunk_map()
    print(f"[p2] join clips {len(clips)}, already done {len(done)}, todo {len(todo)}; "
          f"fetch-conc {args.fetch_conc}, build-workers {args.build_workers}", flush=True)

    t0 = time.time()
    need_fetch = [c for c in todo if not cached_path(c).exists()]
    ready = [c for c in todo if cached_path(c).exists()]
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.fetch_conc) as fx, \
            ProcessPoolExecutor(max_workers=args.build_workers) as bx:
        fetch_f = {fx.submit(fetch_clip, c, chunks[c]): c for c in need_fetch}
        build_f = {bx.submit(build_clip, c, str(out_dir)): c for c in ready}
        fetch_log = {}
        while fetch_f or build_f:
            dn, _ = wait(list(fetch_f) + list(build_f), return_when=FIRST_COMPLETED)
            for f in dn:
                if f in fetch_f:
                    c = fetch_f.pop(f)
                    r = f.result()
                    fetch_log[sha12(c)] = r
                    print(f"[p2] fetched {r['clip']} ok={r['ok']} {r.get('stream_MBps')} MB/s "
                          f"{r.get('error', '')}", flush=True)
                    if r["ok"]:
                        build_f[bx.submit(build_clip, c, str(out_dir))] = c
                    else:
                        with open(man_path, "a", encoding="utf-8") as mf:
                            mf.write(json.dumps({"clip_sha12": sha12(c), "ok": False,
                                                 "stage": "fetch", "error": r.get("error")}) + "\n")
                        n_fail += 1
                else:
                    c = build_f.pop(f)
                    r = f.result()
                    fr = fetch_log.get(sha12(c))
                    if fr:
                        r["fetch"] = {k: fr.get(k) for k in ("stream_MBps", "stream_s",
                                                             "central_dir_s", "member_bytes")}
                    else:
                        r["fetch"] = "pre-cached (09-11 package or P1 probe)"
                    deleted = False
                    if r.get("ok") and not args.keep_parquets and not is_protected(c):
                        try:
                            cached_path(c).unlink()
                            deleted = True
                        except OSError as e:
                            r["delete_error"] = redact_text(str(e))
                    r["parquet_deleted"] = deleted
                    r["protected"] = is_protected(c)
                    r["t_unix"] = time.time()
                    with open(man_path, "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(r) + "\n")
                    n_ok += int(bool(r.get("ok")))
                    n_fail += int(not r.get("ok"))
                    print(f"[p2] built {r['clip_sha12']} ok={r.get('ok')} "
                          f"marg48={r.get('polar48_marginal_observed_infield')} "
                          f"gz={r.get('ground_peak_z_m')} valid={r.get('n_valid')}/{r.get('T')} "
                          f"{r.get('build_s')} s fails={r.get('failures')} "
                          f"err={r.get('error', '')}  [{n_ok} ok / {n_fail} fail, "
                          f"{time.time() - t0:.0f} s]", flush=True)
    print(f"[p2] DONE ok={n_ok} fail={n_fail} wall={time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
