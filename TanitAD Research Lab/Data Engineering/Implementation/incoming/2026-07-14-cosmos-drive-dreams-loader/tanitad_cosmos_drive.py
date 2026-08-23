"""Cosmos-Drive-Dreams -> TanitAD episode contract (D-014 sim arm; CC-BY-4.0).

WHY THIS EXISTS
---------------
The license review (2026-07-07, D-002) excluded the *real* NVIDIA PhysicalAI-AV
sets from all public claims (internal-dev-only / confidential / 12-mo expiry).
D-014 then split the sim arm and named the two ungated NVIDIA synthetic corpora
as the training-mix data source, of which **Cosmos-Drive-Dreams is CC-BY-4.0**
-> the one AV asset we may render, train on, AND cite publicly. comma2k19 (MIT)
stays the real anchor; Cosmos-Drive-Dreams is the *publicly-claimable* long-tail
(rain/snow/fog/night, intersections, pedestrians) that comma2k19's highway
commute lacks. This module is the first loader for it.

CORPUS (nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams, RDS-HQ format)
-------------------------------------------------------------------------------
Per clip, on disk (the fields this loader needs):
    videos/<cam_subdir>/<clip_id>_<chunk>[_<weather>].mp4    30 fps synthetic RGB
    vehicle_pose/<frame:06d>.vehicle_pose.npy               per-frame 4x4 ego pose
    pinhole_intrinsic.<cam>.npy  [fx, fy, cx, cy, w, h]     (H2/R1 follow-up)
Front camera = ``front_wide_120fov`` -> the SAME 120-deg HFOV as PhysicalAI-AV
front-wide, so D-016 focal canonicalization reuses the identical nominal focal
and the crop is angularly consistent with comma2k19 (F_REF=266).

Signal derivation (from the 4x4 vehicle-to-world sequence; no CAN here):
    x, y   = translation[:2]                    world meters, re-origined to frame 0
    yaw    = atan2(R[1,0], R[0,0])              heading of the vehicle x-axis (FLU)
    v      = ||d/dt (x, y)||                    [m/s]
    kappa  = yaw_rate / v                       path curvature
    steer  = atan(WHEELBASE * kappa)            road-wheel angle proxy [rad]
    accel  = finite difference of v             [m/s^2]  (contract semantics)
Frames are resampled 30 Hz -> 10 Hz (stride 3) and D-015 stacked (3 frames @
100 ms -> 9 channels, actions/poses aligned to the latest frame).

Splits: CLIP-level (I3) — each Cosmos clip is an independent synthetic recording.

HONEST LIMITATIONS (P8), verified on the pod before any trained claim:
  * The 4x4 convention (vehicle-to-world, FLU x-forward, metres) is taken from
    the RDS-HQ toolkit docs; the exact ``vehicle_pose`` filename glob and axis
    order are pod-verified via ``verify_real_clip`` (below) — the pure signal /
    contract code is unit-tested on synthetic fixtures regardless.
  * Synthetic pixels are NOT off-expert action-consequence rollouts (the max_a
    JEPA argument) — that job stays with CARLA-on-pod (D-014). This corpus buys
    long-tail *scene* diversity, not counterfactual dynamics.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tanitad.data._contract import (EpisodeWindowDataset, assert_contract,
                                    finite_diff_accel)
from tanitad.data.calib import (PHYSICALAI_FRONT_WIDE_HFOV_DEG,
                                focal_crop_resize, nominal_focal_from_hfov)
from tanitad.data.comma2k19 import stack_frames
from tanitad.data.toy_driving import ToyEpisode

WHEELBASE = 2.9          # Hyperion platform class, shared with PhysicalAI-AV
TARGET_HZ = 10.0
SRC_FPS = 30.0           # Cosmos-Drive-Dreams synthetic camera rate
FRONT_CAM = "front_wide_120fov"
MAX_STEER_RAD = 0.7      # ~40 deg road-wheel clip (guards low-speed 1/v blow-up)

DATA_TAG = "data:cosmos_dd"      # every consuming experiment carries this tag
LICENSE = "CC-BY-4.0"            # publicly-claimable (unlike the real PhysicalAI sets)

# I7 task-identity fingerprint (D-017) — DELIBERATELY identical to comma2k19 /
# physicalai. D-015 stacking + D-016 canonicalization make the three corpora one
# task; the i7_task_identity check PROVES it, so probes fit on one are admissible
# on the others and MixedWindowDataset admits this corpus into the real+sim mix.
CORPUS_META = {
    "channels": 9, "image_size": 256, "f_eff_px": 266.0, "hz": 10.0,
    "actions": ("steer_road_rad", "accel_mps2"),
    "poses": ("x_east_m", "y_north_m", "yaw_rad", "v_mps"),
}


# --------------------------------------------------------------------------- #
# Poses -> actions & poses (pure numpy, unit-tested without real data)         #
# --------------------------------------------------------------------------- #
def poses_to_signals(veh_to_world: np.ndarray, dt: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """4x4 vehicle-to-world sequence [N,4,4] -> (actions [N,2], poses [N,4]).

    ``dt`` is the (already-strided) inter-sample spacing in seconds. Yaw is read
    from the rotation block (stable at any speed); speed/accel from the smoothed
    position derivative. Steering is the bicycle-model road-wheel angle from the
    path curvature, clipped so a near-stationary frame cannot emit a spurious
    hard-lock (kappa = yaw_rate / v diverges as v -> 0).
    """
    M = np.asarray(veh_to_world, dtype=np.float64)
    assert M.ndim == 3 and M.shape[1:] == (4, 4), M.shape
    x = M[:, 0, 3] - M[0, 0, 3]
    y = M[:, 1, 3] - M[0, 1, 3]
    yaw = np.arctan2(M[:, 1, 0], M[:, 0, 0])                 # FLU x-axis heading

    vx = np.gradient(x, dt, edge_order=1)
    vy = np.gradient(y, dt, edge_order=1)
    v = np.hypot(vx, vy)
    yaw_rate = np.gradient(np.unwrap(yaw), dt, edge_order=1)
    kappa = yaw_rate / np.where(v > 0.5, v, np.inf)          # 0 curvature at rest
    steer = np.clip(np.arctan(WHEELBASE * kappa), -MAX_STEER_RAD, MAX_STEER_RAD)
    accel = finite_diff_accel(v.astype(np.float32), dt)

    actions = np.column_stack([steer, accel]).astype(np.float32)
    poses = np.column_stack([x, y, yaw, v]).astype(np.float32)
    return actions, poses


# --------------------------------------------------------------------------- #
# Discovery, pose IO, video decode                                             #
# --------------------------------------------------------------------------- #
def _clip_id_of(mp4: Path) -> str:
    """`<clip>_<chunk>[_<weather>].mp4` -> `<clip>` (drop trailing chunk/weather)."""
    stem = mp4.stem
    stem = re.sub(r"_(Foggy|Golden_hour|Morning|Night|Rainy|Snowy|Sunny)$", "",
                  stem, flags=re.IGNORECASE)
    return re.sub(r"_\d+$", "", stem)                        # drop `_<chunk>`


def load_vehicle_pose(pose_ref: str | Path) -> np.ndarray:
    """Ego trajectory as [N,4,4] vehicle-to-world.

    Accepts either a single stacked ``.npy`` ([N,4,4]) or a directory of
    per-frame ``*.vehicle_pose.npy`` files (RDS-HQ layout), sorted by frame
    index. Injectable via ``pose_fn`` for tests.
    """
    p = Path(pose_ref)
    if p.is_file():
        arr = np.load(p)
        return arr.reshape(-1, 4, 4).astype(np.float64)
    files = sorted(p.glob("*.vehicle_pose.npy")) or sorted(p.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no vehicle_pose npy under {p}")
    return np.stack([np.load(f).reshape(4, 4) for f in files]).astype(np.float64)


def _decode_mp4(mp4: Path, size: int) -> Tensor:
    """All frames -> uint8 [N,3,size,size], D-016 focal-canonicalized (120 HFOV)."""
    import av                                                # lazy: .[real]
    frames = []
    with av.open(str(mp4)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for fr in c.decode(stream):
            rgb = torch.from_numpy(fr.to_ndarray(format="rgb24")).permute(2, 0, 1)
            frames.append(rgb)
    vid = torch.stack(frames)                                # [N,3,H,W] u8
    f_px = nominal_focal_from_hfov(vid.shape[-1], PHYSICALAI_FRONT_WIDE_HFOV_DEG)
    return focal_crop_resize(vid, f_px, size)


def discover_clips(root: str | Path, camera_subdir: str | None = None,
                   weather: str | None = None) -> list[dict]:
    """Clips that have both a front-camera mp4 and a vehicle_pose source.

    ``camera_subdir`` overrides the video folder (default: first match of
    ``videos/*front_wide_120fov``). ``weather`` filters the generated variant.
    Each clip pairs its mp4 with ``vehicle_pose/`` (per-frame dir) if present,
    else ``vehicle_pose/<clip_id>.npy``.
    """
    root = Path(root)
    if camera_subdir is not None:
        vid_dir = root / camera_subdir
    else:
        cands = [d for d in (root / "videos").glob(f"*{FRONT_CAM}")
                 if d.is_dir()] if (root / "videos").exists() else []
        vid_dir = cands[0] if cands else root / "videos" / f"pinhole_{FRONT_CAM}"
    pose_root = root / "vehicle_pose"
    out = []
    for mp4 in sorted(vid_dir.glob("*.mp4")):
        if weather and weather.lower() not in mp4.stem.lower():
            continue
        cid = _clip_id_of(mp4)
        pose_dir = pose_root / cid
        pose_file = pose_root / f"{cid}.npy"
        pose = pose_dir if pose_dir.is_dir() else (
            pose_file if pose_file.exists() else pose_root)
        if Path(pose).exists():
            out.append({"clip_id": cid, "mp4": mp4, "pose": pose,
                        "weather": _weather_of(mp4)})
    return out


def _weather_of(mp4: Path) -> str:
    m = re.search(r"_(Foggy|Golden_hour|Morning|Night|Rainy|Snowy|Sunny)$",
                  mp4.stem, flags=re.IGNORECASE)
    return m.group(1).lower() if m else "unknown"


def _episode_id(clip_id: str, weather: str) -> int:
    return int(hashlib.sha1(f"cosmos/{clip_id}/{weather}".encode())
               .hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# Build one episode                                                            #
# --------------------------------------------------------------------------- #
def build_episode(clip: dict, size: int = 256, n_stack: int = 3,
                  src_fps: float = SRC_FPS, decode_fn=_decode_mp4,
                  pose_fn=load_vehicle_pose) -> ToyEpisode:
    """One Cosmos clip -> contract episode at 10 Hz with D-015 stacking.

    Video and per-frame poses are both at ``src_fps`` on a shared clock; both are
    strided to TARGET_HZ (dt = stride/src_fps ~ 0.1 s), signals derived on the
    strided timeline, then ``n_stack`` frames are channel-stacked with
    actions/poses aligned to the LATEST frame of each stack (drop the first k).
    """
    stride = max(1, int(round(src_fps / TARGET_HZ)))
    dt = stride / src_fps

    vid = decode_fn(clip["mp4"], size)                       # [M,3,S,S] u8
    poses4 = pose_fn(clip["pose"])                           # [N,4,4]
    n = min(vid.shape[0], poses4.shape[0])
    vid = vid[:n][::stride]                                  # [n',3,S,S]
    actions, poses = poses_to_signals(poses4[:n][::stride], dt)

    m = min(vid.shape[0], actions.shape[0])
    stacked = stack_frames(vid[:m], n_stack)                 # [m-k,9,S,S] u8
    k = n_stack - 1
    # uint8 in memory; window datasets convert per window (the pod-OOM lesson).
    return ToyEpisode(frames=stacked,
                      actions=torch.from_numpy(actions[k:m]),
                      poses=torch.from_numpy(poses[k:m]),
                      episode_id=_episode_id(clip["clip_id"], clip["weather"]))


def build_episodes(clips: list[dict], **kw) -> list[ToyEpisode]:
    """Build many clips, skipping any that fail to decode (corrupt/partial)."""
    eps = []
    for i, clip in enumerate(clips):
        if i % 20 == 0:
            print(f"[cosmos] building {i}/{len(clips)} (decode is the slow part)",
                  flush=True)
        try:
            eps.append(build_episode(clip, **kw))
        except Exception as e:                               # skip, keep building
            print(f"[cosmos] skip {clip.get('clip_id')}: {type(e).__name__}: {e}")
    return eps


def split_clips(clips: list[dict], val_frac: float = 0.2,
                seed: int = 0) -> tuple[list[dict], list[dict]]:
    """CLIP-level split (the I3 unit for independent synthetic recordings)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * val_frac))
    val_i = set(perm[:n_val])
    return ([c for i, c in enumerate(clips) if i not in val_i],
            [c for i, c in enumerate(clips) if i in val_i])


def CosmosDriveDataset(clips: list[dict], window: int = 8, max_horizon: int = 16,
                       size: int = 256, n_stack: int = 3,
                       **build_kw) -> EpisodeWindowDataset:
    """Window dataset over Cosmos clips — item contract byte-identical to every
    other adapter (reuses the shared EpisodeWindowDataset). Pass an already-split
    clip list (I3); never split the windows of one clip across train/val."""
    eps = build_episodes(clips, size=size, n_stack=n_stack, **build_kw)
    return EpisodeWindowDataset(eps, window=window, max_horizon=max_horizon)


# --------------------------------------------------------------------------- #
# Pod-only real-clip sanity (not run in CI — documents the verification step)  #
# --------------------------------------------------------------------------- #
def verify_real_clip(clip: dict, size: int = 256) -> dict:
    """Decode+derive ONE real clip and return sanity stats for the data card.

    Run on the pod after the first download to settle the honest limitations:
    A8 consequence, plausible speed/steer ranges, and that the pose glob found
    frames. Not a test (needs real bytes)."""
    from tanitad.data._contract import frame_change_fraction, to_float_frames
    ep = build_episode(clip, size=size)
    assert_contract(ep, channels=9)
    a8 = frame_change_fraction(to_float_frames(ep.frames)[:, -3:])
    return {"clip_id": clip["clip_id"], "weather": clip["weather"],
            "T": int(ep.frames.shape[0]),
            "speed_mps_mean": float(ep.poses[:, 3].mean()),
            "steer_rad_absmax": float(ep.actions[:, 0].abs().max()),
            "accel_mps2_absmax": float(ep.actions[:, 1].abs().max()),
            "a8_frame_change_fraction": a8}
