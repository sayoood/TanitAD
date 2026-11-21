#!/usr/bin/env python3
"""nt_e1.py -- H-NT-DECIDE-1 E1: the token-episode ``*.v2ep.pt`` writer library (TANITAD VENV on Thor; CPU only).

Contract: ``C:/Users/Admin/refcv8_pf/nt_build/INTERFACE.md`` (v1).  Every number this module writes is defined there.

IMPORTED, never copied (the brief's rule):
  * the stitch: the r8ns agent's ``build_navtrain_frames.load_stitcher`` -> E2's ``build_scene`` configured for
    ``FRAME_416x1024`` through refcv6's ``frames416.py`` -- the exact path that built the navtrain selection frame bank;
  * the ego history: ``refcv8_bridge.declare8`` + ``ego_history_poses`` + ``run_bridge7.times_rel_t0`` -- the R8_LEGAL
    inference feed, called, so ``pose_hist`` == the bridge input by construction (K1);
  * the slot construction: ``refcv8_bridge.slot_sources8(times, "ST")`` (asserted all-t0);
  * the frame constant: ``tanitad.models.trunk_shapes.FRAME_416x1024``;
  * the join key: ``tanitad.data.v2_dataset.stable_episode_id``;
  * log pickles: ``navtrain_common.load_log_pickle`` (allow-listed unpickler).
What E1 implements itself (and K2 / E1-INT check independently): the FUTURE poses from the OpenScene log pickle
(SE(2)-relative to t0, nominal 0.5 s knots, not-a-knot cubic spline).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np

C = "/home/nvidia/r8ns/code"
ENV = {"TANITAD_REPO": f"{C}/tree", "R8_LAUNCH_TREE": f"{C}/tree", "R8_EVAL_KIT": "/home/nvidia",
       "R8_BATTERY_CODE": f"{C}/batt", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
for _k, _v in ENV.items():
    os.environ.setdefault(_k, _v)
for _p in (f"{C}/tree/stack", f"{C}/tree/taniteval", "/home/nvidia/r8ns/overlay_scipy", f"{C}/navtrain", f"{C}/pkg/code"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True

import torch  # noqa: E402
import torchvision.io as tvio  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

import navtrain_common as N  # noqa: E402  (r8ns agent)
import build_navtrain_frames as BNF  # noqa: E402  (r8ns agent: load_stitcher, _tar_readers)
import refcv8_bridge as B  # noqa: E402  (r8ns agent: the refcv8 NavSim seam)
import run_bridge7 as RB7  # noqa: E402  (times_rel_t0, as run_bridge8 imports it)
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402
from tanitad.models.trunk_shapes import FRAME_416x1024  # noqa: E402

FRAMES416 = (f"{C}/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/frames416.py")
SCHEMA = "nt-e1/1.2"
H, W = 416, 1024
N_STACK = 3
RAW_NOW = 9                      # raw row of t0 (provider row 7 = window - 1, + raw offset 2)
N_FRAME_ROWS = 30                # raw rows 0..29 = window t=0's frames (8 rows) + future_frames (20 rows), n_stack 3
LAW_FIRST_RAW = 10               # raw rows >= 10 (tau > 0) carry the LAW frame
MAX_FUT_TICKS = 60               # 6.0 s at 0.1 s
MIN_FUT_TICKS = 21               # T_out >= window + max_horizon + 1 = 29 -> raw >= 31 -> ticks >= 21
DT = 0.1
KNOT_DT = 0.5
GAP_TOL_S = 0.05                 # |dt - 0.5| above this ends the usable future (a dropped 2 Hz frame)
WHEELBASE_LEGACY = 2.9           # physicalai.signals_at legacy constant (actions are not consumed by refcv8)
KAPPA_V_FLOOR = 1.0
LEGACY_EID = int.from_bytes("navtrain-".encode()[:4].ljust(4, b"\0"), "big")   # build_compressed's formula (= 1851881076)
ARM = "R8_LEGAL"

_E2 = None


def stitcher():
    """E2's stitch configured for FRAME_416x1024, loaded exactly as the r8ns bank builder loads it."""
    global _E2
    if _E2 is None:
        _E2 = BNF.load_stitcher(FRAMES416)
        if _E2.FRAME != FRAME_416x1024:
            raise SystemExit(f"stitcher frame {_E2.FRAME} != FRAME_416x1024")
    return _E2


def clip_id_of(token: str) -> str:
    return f"navtrain-{token}"


def sha16(hwc: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(hwc).tobytes()).hexdigest()[:16]


def stitch_frame(cams: dict, sensor_root: str, rig_cache: dict):
    """One frame's CAM_L0/F0/R0 -> (u8 [H,W,3], src int8 [H,W], rig_key, observed_frac, mean_px).  build_navtrain_frames'
    own call (``e2.build_scene([cd], scratch, rig_cache)``) with its own checks (shape/dtype, KB2 mean_px >= 1)."""
    import pathlib
    e2 = stitcher()
    one, src, rk, obs = e2.build_scene([cams], pathlib.Path(sensor_root), rig_cache)
    if one.shape != (1, H, W, 3) or one.dtype != np.uint8:
        raise ValueError(f"stitch returned {one.dtype}{one.shape}")
    mpx = float(one.mean())
    if mpx < 1.0:
        raise ValueError(f"mean_px {mpx:.3f} < 1.0 (an all-black frame, KB2)")
    return one[0], src, rk, float(obs), mpx


def encode_png(hwc: np.ndarray) -> torch.Tensor:
    """The trainer's codec (``v2_compressed.py:143``: ``tvio.encode_png`` on CHW u8)."""
    return tvio.encode_png(torch.from_numpy(np.ascontiguousarray(hwc)).permute(2, 0, 1).contiguous())


def decode_png_hwc(buf: torch.Tensor) -> np.ndarray:
    return tvio.decode_png(buf, mode=tvio.ImageReadMode.RGB).permute(1, 2, 0).contiguous().numpy()


# --------------------------------------------------------------------------------------------------------------------------
# poses
# --------------------------------------------------------------------------------------------------------------------------
def quat_yaw(q) -> float:
    """yaw of a (w, x, y, z) ``ego2global_rotation`` EXACTLY as the NavSim v1.1 devkit reads it:
    ``pyquaternion.Quaternion(*q).yaw_pitch_roll[0]`` (``navsim/common/dataclasses.py:171-173, 389-391``) =
    ``atan2(2(wz - xy), 1 - 2(y^2 + z^2))`` on the normalised quaternion (pyquaternion's z-y'-x'' form; MEASURED identical on
    Thor 2026-10-21).  ⛔ NOT the textbook ZYX ``+xy`` form: that one differs by ~1e-4 rad and put the future 2-20 mm off
    the devkit's own GT and off the export's history poses (caught by E1-K2pre on 200 tokens; v1.2)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n > 0 and abs(n - 1.0) > 1e-14:
        w, x, y, z = w / n, x / n, y / n, z / n
    return math.atan2(2.0 * (w * z - x * y), 1.0 - 2.0 * (y * y + z * z))


def rel_se2(frames: list, pos: int, ks: list) -> np.ndarray:
    """``[len(ks), 3]`` float64 (x, y, yaw_unwrapped) of log frames ``pos + k`` in the t0 (= frames[pos]) ego frame."""
    f0 = frames[pos]
    X0, Y0 = (float(v) for v in np.asarray(f0["ego2global_translation"], np.float64)[:2])
    T0 = quat_yaw(f0["ego2global_rotation"])
    c, s = math.cos(T0), math.sin(T0)
    out = []
    for k in ks:
        f = frames[pos + k]
        X, Y = (float(v) for v in np.asarray(f["ego2global_translation"], np.float64)[:2])
        dx, dy = X - X0, Y - Y0
        out.append((c * dx + s * dy, -s * dx + c * dy, quat_yaw(f["ego2global_rotation"]) - T0))
    a = np.asarray(out, np.float64)
    a[:, 2] = np.unwrap(a[:, 2])
    return a


def future_knot_range(frames: list, pos: int) -> tuple:
    """``(K, gap_truncated)``: the number of usable future 2 Hz frames (<= 12), ending before a dropped frame or the log end."""
    K = 0
    gap = False
    for k in range(1, MAX_FUT_TICKS // 5 + 1):
        if pos + k >= len(frames):
            break
        dt = (int(frames[pos + k]["timestamp"]) - int(frames[pos + k - 1]["timestamp"])) / 1e6
        if abs(dt - KNOT_DT) > GAP_TOL_S:
            gap = True
            break
        K = k
    return K, gap


def history_knot_start(frames: list, pos: int) -> int:
    """The earliest history offset j in {-3..0} such that frames pos+j..pos are gap-free at 2 Hz (spline knots stay nominal)."""
    j = 0
    for k in (-1, -2, -3):
        dt = (int(frames[pos + k + 1]["timestamp"]) - int(frames[pos + k]["timestamp"])) / 1e6
        if abs(dt - KNOT_DT) > GAP_TOL_S:
            break
        j = k
    return j


def spline_track(knot_t: np.ndarray, xyy: np.ndarray, v_knots: np.ndarray, ax_knots: np.ndarray, taus: np.ndarray) -> tuple:
    """(x, y, yaw, v) + (steer, accel) at ``taus`` from 2 Hz knots.  x/y/yaw: not-a-knot C2 cubic spline (exact at knots: the
    knot values are assigned verbatim where tau hits a knot); v and accel: linear (the bridge's rule for v)."""
    cs = CubicSpline(knot_t, xyy, axis=0, bc_type="not-a-knot")
    p = cs(taus)
    d = cs(taus, 1)
    hit = np.isclose(taus[:, None], knot_t[None, :], rtol=0, atol=1e-9)
    for i, j in zip(*np.nonzero(hit)):
        p[i] = xyy[j]
    v = np.interp(taus, knot_t, v_knots)
    vs = np.hypot(d[:, 0], d[:, 1])
    kappa = np.where(vs >= KAPPA_V_FLOOR, d[:, 2] / np.maximum(vs, KAPPA_V_FLOOR), 0.0)
    steer = np.arctan(WHEELBASE_LEGACY * kappa)
    acc = np.interp(taus, knot_t, ax_knots)
    yaw = np.arctan2(np.sin(p[:, 2]), np.cos(p[:, 2]))
    return np.stack([p[:, 0], p[:, 1], yaw, v], -1), np.stack([steer, acc], -1)


def token_poses(rec: dict, frames: list, pos: int) -> dict:
    """``poses [T_raw,4]`` / ``actions [T_raw,2]`` float32 on the raw timeline tau_r = (r - 9) * 0.1 s (INTERFACE section 3)."""
    if frames[pos]["token"] != rec["frame_tokens"][-1]:
        raise ValueError("log position does not hold the token's t0 frame")
    if [frames[pos - 3 + i]["token"] for i in range(4)] != list(rec["frame_tokens"]):
        raise ValueError("log window != the export's frame tokens")
    times = RB7.times_rel_t0(rec)
    decl = B.declare8(rec["ego_statuses"], ARM)
    hist = B.ego_history_poses(decl, times)                       # [8,4] f32, tau -0.7..0 (the R8_LEGAL feed)
    if hist.shape != (8, 4) or hist.dtype != np.float32:
        raise ValueError(f"bridge history {hist.dtype}{hist.shape}")
    src = B.slot_sources8(times, "ST")
    if list(src) != [3] * 10:
        raise ValueError(f"ST slot sources {src} are not all-t0")
    # rows 0,1 (tau -0.9, -0.8): stack padding, never read by a window -- the bridge's own linear rule at those times
    ts = [float(times[i]) for i in B.R6.HIST_SRC_IDX]
    xy = np.array([decl[f"ego_pose[{i}]"][:2] for i in B.R6.HIST_SRC_IDX], np.float64)
    yw = np.unwrap(np.array([decl[f"ego_pose[{i}]"][2] for i in B.R6.HIST_SRC_IDX], np.float64))
    vv = np.array([B.speed_of(decl[f"ego_velocity[{i}]"]) for i in B.R6.HIST_SRC_IDX], np.float64)
    q = np.array([-0.9, -0.8] + list(B.R6.HIST_T_S), np.float64)
    pad = np.stack([np.interp(q, ts, xy[:, 0]), np.interp(q, ts, xy[:, 1]), np.interp(q, ts, yw), np.interp(q, ts, vv)], -1)
    pad = pad.astype(np.float32)
    if not np.array_equal(pad[2:], hist):
        raise ValueError("own linear rule != ego_history_poses on the window rows (the padding rule drifted)")
    # future from the log pickle
    K, gap = future_knot_range(frames, pos)
    n_fut = min(5 * K, MAX_FUT_TICKS)
    if n_fut < MIN_FUT_TICKS:
        raise ValueError(f"only {n_fut} usable future ticks (K={K}, gap={gap}); one window needs >= {MIN_FUT_TICKS}")
    j0 = history_knot_start(frames, pos)
    ks = list(range(j0, K + 1))
    rel = rel_se2(frames, pos, ks)
    knot_t = np.array([KNOT_DT * k for k in ks], np.float64)
    dyn = [np.asarray(frames[pos + k]["ego_dynamic_state"], np.float64) for k in ks]
    v_kn = np.array([math.hypot(d[0], d[1]) for d in dyn])
    ax_kn = np.array([d[2] for d in dyn])
    taus_f = np.arange(1, n_fut + 1, dtype=np.float64) / 10.0
    fut, fut_act = spline_track(knot_t, rel, v_kn, ax_kn, taus_f)
    taus_h = np.clip(np.arange(-9, 1, dtype=np.float64) / 10.0, knot_t[0], None)   # actions only (not consumed)
    _, hist_act = spline_track(knot_t, rel, v_kn, ax_kn, taus_h)
    poses = np.concatenate([pad, fut.astype(np.float32)], 0)
    actions = np.concatenate([hist_act, fut_act], 0).astype(np.float32)
    if poses.shape[0] != 10 + n_fut or not np.isfinite(poses).all() or not np.isfinite(actions).all():
        raise ValueError("non-finite or mis-sized track")
    if not np.array_equal(poses[RAW_NOW, :3], np.zeros(3, np.float32)):
        raise ValueError(f"NOW pose {poses[RAW_NOW]} is not the t0 origin")
    return {"poses": poses, "actions": actions, "n_future_ticks": int(n_fut), "future_gap_truncated": bool(gap), "K": int(K),
            "history_knot_start": int(j0),
            "v0": float(B.v0_of(decl)), "times_rel_t0": [float(t) for t in times], "hist": hist,
            "knots_rel_xy_yaw": rel, "knot_t": knot_t}


# --------------------------------------------------------------------------------------------------------------------------
# the payload
# --------------------------------------------------------------------------------------------------------------------------
PAYLOAD_KEYS = ("jpeg_buf", "jpeg_len", "actions", "poses", "n_stack", "image_size", "episode_id", "clip_id", "quality",
                "image_h", "image_w", "frame", "projection_mode", "codec", "frame_index", "nt", "nt_frame")


def build_payload(token: str, t0_png: torch.Tensor, law_png: "torch.Tensor | None", tr: dict, meta: dict) -> dict:
    pngs = [t0_png] if law_png is None else [t0_png, law_png]
    fi = torch.zeros(N_FRAME_ROWS, dtype=torch.int64)
    if law_png is not None:
        fi[LAW_FIRST_RAW:] = 1
    return {"jpeg_buf": torch.cat(pngs), "jpeg_len": torch.tensor([int(p.numel()) for p in pngs], dtype=torch.int64),
            "actions": torch.from_numpy(tr["actions"]), "poses": torch.from_numpy(tr["poses"]),
            "n_stack": N_STACK, "image_size": 256, "episode_id": LEGACY_EID, "clip_id": clip_id_of(token), "quality": 90,
            "image_h": H, "image_w": W, "frame": FRAME_416x1024.to_dict(), "projection_mode": "cylindrical", "codec": "png",
            "frame_index": fi, "nt": {"schema": SCHEMA, **meta}, "nt_frame": "ST"}


def save_atomic(payload: dict, path: str) -> int:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return os.path.getsize(path)


def verify_file(path: str, want_t0_sha: str, want_law_sha: "str | None", tr: dict) -> dict:
    """Re-read the file as ``torch.load`` does, decode EVERY stored PNG, re-derive the HWC sha256[:16], assert the schema."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    probs = []
    if tuple(sorted(d)) != tuple(sorted(PAYLOAD_KEYS)):
        probs.append(f"keys {sorted(d)}")
    if d["jpeg_buf"].dtype != torch.uint8 or d["jpeg_len"].dtype != torch.int64 or d["frame_index"].dtype != torch.int64:
        probs.append("dtypes")
    if d["poses"].dtype != torch.float32 or d["actions"].dtype != torch.float32:
        probs.append("pose/action dtype")
    T = int(d["poses"].shape[0])
    if tuple(d["poses"].shape) != (T, 4) or tuple(d["actions"].shape) != (T, 2) or not (10 + MIN_FUT_TICKS <= T <= 70):
        probs.append(f"pose shape {tuple(d['poses'].shape)}")
    if not torch.equal(d["poses"], torch.from_numpy(tr["poses"])):
        probs.append("poses differ from what was written")
    if tuple(d["frame_index"].shape) != (N_FRAME_ROWS,) or int(d["frame_index"].max()) >= len(d["jpeg_len"]):
        probs.append("frame_index")
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(d["jpeg_len"], 0)])
    if int(offs[-1]) != int(d["jpeg_buf"].numel()):
        probs.append("jpeg_len does not sum to the buffer")
    shas = []
    for j in range(len(d["jpeg_len"])):
        hwc = decode_png_hwc(d["jpeg_buf"][int(offs[j]):int(offs[j + 1])])
        if hwc.shape != (H, W, 3):
            probs.append(f"png {j} shape {hwc.shape}")
        shas.append(sha16(hwc))
    if shas[0] != want_t0_sha:
        probs.append("t0 PNG does not round-trip bit-exact")
    if want_law_sha is not None and (len(shas) < 2 or shas[1] != want_law_sha):
        probs.append("LAW PNG does not round-trip bit-exact")
    if want_law_sha is None and len(shas) != 1:
        probs.append("fallback file stores a second PNG")
    return {"problems": probs, "T_raw": T, "pngs": len(shas)}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def code_sha256() -> str:
    return file_sha256(os.path.abspath(__file__))


def manifest_row(token: str, path: str, shard: str, meta: dict, tr: dict, nbytes: int) -> dict:
    cid = clip_id_of(token)
    return {"token": token, "clip_id": cid, "sid": stable_episode_id(cid), "log_name": meta["log_name"], "shard": shard,
            "file": os.path.basename(path), "bytes": nbytes, "file_sha256": file_sha256(path),
            "t0_frame_sha16": meta["t0_frame_sha16"], "law_frame_sha16": meta["law_frame_sha16"], "law_src": meta["law_src"],
            "law_token": meta["law_token"], "T_raw": int(tr["poses"].shape[0]), "n_future_ticks": tr["n_future_ticks"],
            "future_gap_truncated": tr["future_gap_truncated"], "rig_key": meta["rig_key"], "observed_frac": meta["observed_frac"],
            "mean_px": meta["mean_px"], "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def token_meta(token: str, rec: dict, pos: int, shard: str, fm: dict, law: dict, tr: dict) -> dict:
    return {"token": token, "log_name": rec["log_name"], "map_name": rec.get("map_name"), "log_pos": int(pos),
            "t0_timestamp_us": int(rec["timestamps_us"][-1]), "shard": shard,
            "driving_command_t0": [int(round(x)) for x in rec["ego_statuses"][-1]["driving_command"]],
            "law_src": law["src"], "law_token": law["token"], "t0_frame_sha16": fm["sha16"], "law_frame_sha16": law["sha16"],
            "rig_key": fm["rig_key"], "observed_frac": fm["observed_frac"], "mean_px": fm["mean_px"],
            "n_future_ticks": tr["n_future_ticks"], "future_gap_truncated": tr["future_gap_truncated"], "v0": tr["v0"],
            "history_knot_start": tr["history_knot_start"],
            "pose_frame": "t0_ego", "raw_now_row": RAW_NOW, "dt_s": DT, "frame_rows": N_FRAME_ROWS,
            "hist_source": "refcv8_bridge.ego_history_poses(declare8(ego_statuses,'R8_LEGAL'), times_rel_t0) [raw 2..9]",
            "future_source": "OpenScene log pickle ego2global, SE2-rel t0, not-a-knot CubicSpline on nominal 0.5 s knots",
            "writer": {"tool": "nt_e1.py", "sha256": code_sha256()}}
