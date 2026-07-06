"""PhysicalAI-AV R0 -> TanitAD episode contract (D-012, D-015; tag data:physicalai).

Consumes the Stage-R0 output of `scripts/physicalai_r0.py`:
    <root>/r0/r0_selection.parquet
    <root>/r0/camera_front_wide/**/{clip_id}.camera_front_wide_120fov.mp4
                                  + .timestamps.parquet
    <root>/labels/egomotion/egomotion.chunk_{chunk:04d}.zip   ({clip_id}.egomotion.parquet)

Signal derivation (egomotion schema: timestamp, q*, x/y/z, vx/vy/vz, ax/ay/az, curvature):
    v      = ||(vx, vy)||                       [m/s]
    steer  = atan(WHEELBASE * curvature)        road-wheel angle proxy [rad]
    accel  = finite difference of v             [m/s^2]  (contract semantics)
    pose   = (x, y, yaw=atan2(vy, vx), v)       clip-local frame
Video frames are resampled to 10 Hz by nearest-timestamp alignment against the
egomotion clock, then D-015 stacking (3 frames @100 ms -> 9 channels, actions/
poses aligned to the latest frame).

Splits: CLIP-level (each 20 s clip is an independent recording -> the I3 unit
here is the clip; never split windows of one clip across train/val).
"""

from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.data._contract import finite_diff_accel
from tanitad.data.comma2k19 import stack_frames
from tanitad.data.toy_driving import ToyEpisode

WHEELBASE = 2.9          # Hyperion platform class, sedan/SUV proxy
TARGET_HZ = 10.0


def discover_r0_clips(root: str | Path) -> list[dict]:
    """Selected clips that have both camera mp4 and egomotion available."""
    root = Path(root)
    sel = pd.read_parquet(root / "r0" / "r0_selection.parquet")
    chunk_of = dict(zip(sel["clip_id"].astype(str), sel["chunk"].astype(int)))
    out = []
    for mp4 in sorted((root / "r0" / "camera_front_wide").rglob("*.mp4")):
        clip_id = mp4.name.split(".")[0]
        if clip_id not in chunk_of:
            continue
        ts = mp4.with_name(mp4.name.replace(".mp4", ".timestamps.parquet"))
        ego_zip = root / "labels" / "egomotion" / \
            f"egomotion.chunk_{chunk_of[clip_id]:04d}.zip"
        if ts.exists() and ego_zip.exists():
            out.append({"clip_id": clip_id, "mp4": mp4, "timestamps": ts,
                        "ego_zip": ego_zip})
    return out


def load_egomotion(ego_zip: Path, clip_id: str) -> pd.DataFrame:
    with zipfile.ZipFile(ego_zip) as z:
        name = next(n for n in z.namelist()
                    if n.endswith(f"{clip_id}.egomotion.parquet"))
        return pd.read_parquet(io.BytesIO(z.read(name)))


def _decode_mp4(mp4: Path, size: int) -> Tensor:
    """All frames of the clip -> uint8 [N, 3, size, size] (crop-square+resize)."""
    import av
    frames = []
    with av.open(str(mp4)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for fr in c.decode(stream):
            rgb = torch.from_numpy(fr.to_ndarray(format="rgb24")).permute(2, 0, 1)
            frames.append(rgb)
    vid = torch.stack(frames)
    # D-016: canonical effective focal. Nominal focal from the 120-deg HFOV
    # spec until per-clip intrinsics (calibration/ feature) are ingested —
    # the crop keeps the central ~51 deg, angularly consistent with comma2k19;
    # the wide periphery returns later as H2 side-view modalities.
    from tanitad.data.calib import (PHYSICALAI_FRONT_WIDE_HFOV_DEG,
                                    focal_crop_resize, nominal_focal_from_hfov)
    f_px = nominal_focal_from_hfov(vid.shape[-1],
                                   PHYSICALAI_FRONT_WIDE_HFOV_DEG)
    return focal_crop_resize(vid, f_px, size)


def signals_at(ego: pd.DataFrame, t_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate egomotion signals at query timestamps (same clock).

    Returns (actions [n,2], poses [n,4]) per the contract."""
    t = ego["timestamp"].to_numpy(np.float64)
    order = np.argsort(t)
    t = t[order]

    def col(c):
        return np.interp(t_query, t, ego[c].to_numpy(np.float64)[order])

    vx, vy = col("vx"), col("vy")
    v = np.hypot(vx, vy)
    curv = col("curvature")
    steer = np.arctan(WHEELBASE * curv)
    dt = float(np.median(np.diff(t_query))) if len(t_query) > 1 else 0.1
    # normalize dt to seconds if the clock is in micro/nanoseconds
    for scale in (1e9, 1e6, 1e3):
        if dt > 50.0:
            dt /= scale if dt / scale >= 1e-3 else 1.0
    dt = dt if 1e-3 < dt < 10.0 else 0.1
    accel = finite_diff_accel(v.astype(np.float32), dt)
    actions = np.column_stack([steer, accel]).astype(np.float32)
    yaw = np.arctan2(vy, vx)
    poses = np.column_stack([col("x"), col("y"), yaw, v]).astype(np.float32)
    return actions, poses


def build_episode(clip: dict, size: int = 256, n_stack: int = 3,
                  decode_fn=_decode_mp4) -> ToyEpisode:
    """One R0 clip -> contract episode at 10 Hz with D-015 stacking."""
    ts = pd.read_parquet(clip["timestamps"])
    tcol = next(c for c in ts.columns if "time" in c.lower())
    t_frames = ts[tcol].to_numpy(np.float64)
    ego = load_egomotion(clip["ego_zip"], clip["clip_id"])

    # resample the video timeline to 10 Hz on the shared clock
    span = t_frames[-1] - t_frames[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):                 # ns/us/ms clocks -> seconds
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * TARGET_HZ), n_stack + 1)
    t_query = np.linspace(t_frames[0], t_frames[-1], n_target)
    frame_idx = np.searchsorted(t_frames, t_query).clip(0, len(t_frames) - 1)

    vid = decode_fn(clip["mp4"], size)[frame_idx]              # [n,3,S,S] u8
    actions, poses = signals_at(ego, t_query)
    n = min(vid.shape[0], actions.shape[0])
    stacked = stack_frames(vid[:n], n_stack)
    k = n_stack - 1
    ep_id = int.from_bytes(clip["clip_id"].encode()[:4].ljust(4, b"\0"), "big")
    # uint8 in memory; datasets convert per window (500 float32 clips ~ 236 GB).
    return ToyEpisode(frames=stacked,
                      actions=torch.from_numpy(actions[k:n]),
                      poses=torch.from_numpy(poses[k:n]),
                      episode_id=ep_id)


def split_clips(clips: list[dict], val_frac: float = 0.2,
                seed: int = 0) -> tuple[list[dict], list[dict]]:
    """CLIP-level split (the I3 unit for independent 20 s recordings)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * val_frac))
    val_i = set(perm[:n_val])
    return ([c for i, c in enumerate(clips) if i not in val_i],
            [c for i, c in enumerate(clips) if i in val_i])
