"""comma2k19 -> TanitAD episode contract (D-009: real camera data first).

comma2k19: 33 h of commute driving (comma.ai), 20 fps front camera + CAN
(speed, steering-wheel angle) + global pose — real actions and real
trajectory targets with zero annotation. Source used here: the ungated
HuggingFace mirror ``commaai/comma2k19`` (``raw_data/Chunk_*.zip``), official
layout preserved:

    Chunk_X/<dongle_id|date--time>/<segment>/            (1-minute segments)
        video.hevc                                       20 fps, 1164x874
        processed_log/CAN/speed/{t,value}                m/s
        processed_log/CAN/steering_angle/{t,value}       deg (steering wheel)
        global_pose/frame_times                          s, one per frame
        global_pose/frame_positions                      ECEF meters [T,3]
        global_pose/frame_velocities                     ECEF m/s   [T,3]

Contract produced (camera variant of the toy contract):
    frames  [T, 6, S, S]   2 consecutive RGB frames channel-stacked (t-1, t),
                           float32 in [0,1] — consequence-dominance needs the
                           motion visible inside one input (A8)
    actions [T, 2]         (road-wheel steer rad, longitudinal accel m/s^2)
    poses   [T, 4]         (x_east, y_north, yaw, v) in a segment-local ENU frame
    episode_id             stable int hash of the segment path

Splits are ROUTE-level (I3): split_by_route() groups segments by their
<dongle|date> route folder — never split by frame or by segment of the same
drive. v0 notes: geocentric-latitude ENU approximation (sub-0.3 % tilt over a
1-min segment); constant steering ratio 15.3 (Civic/Corolla-class) for
wheel->road angle; both documented for the D1 gate report.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tanitad.data.toy_driving import ToyEpisode

FPS = 20
STEER_RATIO = 15.3           # steering wheel -> road wheel, v0 constant
WHEEL_TO_RAD = math.pi / 180.0


# --------------------------------------------------------------------------- #
# Discovery & splits                                                           #
# --------------------------------------------------------------------------- #
def discover_segments(root: str | Path) -> list[Path]:
    """All segment dirs under root that carry video + pose logs."""
    root = Path(root)
    segs = []
    for video in sorted(root.rglob("video.hevc")):
        seg = video.parent
        if (seg / "global_pose" / "frame_times").exists():
            segs.append(seg)
    return segs


def route_of(segment: Path) -> str:
    """Route id = the <dongle|date--time> folder name above the segment."""
    return segment.parent.name


def sample_segments_across_routes(segments: list[Path], n: int,
                                  seed: int = 0) -> list[Path]:
    """Cap to n segments while spanning as many ROUTES as possible (round-robin).

    Plain `segments[:n]` returns n minutes of the SAME drive (segments of one
    route sort together), which a route-level split then puts wholly into one
    side — the p0-sB00 '0 train segments' failure. Round-robin over routes
    keeps small caps split-able and diverse.
    """
    by_route: dict[str, list[Path]] = {}
    for s in segments:
        by_route.setdefault(route_of(s), []).append(s)
    order = sorted(by_route)
    g = torch.Generator().manual_seed(seed)
    order = [order[i] for i in torch.randperm(len(order), generator=g).tolist()]
    out: list[Path] = []
    depth = 0
    while len(out) < min(n, len(segments)):
        added = False
        for r in order:
            if depth < len(by_route[r]):
                out.append(by_route[r][depth])
                added = True
                if len(out) == n:
                    break
        if not added:
            break
        depth += 1
    return out


def split_by_route(segments: list[Path], val_frac: float = 0.2,
                   seed: int = 0) -> tuple[list[Path], list[Path]]:
    """I3: disjoint ROUTES, not frames, not segments."""
    routes = sorted({route_of(s) for s in segments})
    assert len(routes) >= 2, (
        f"route-level split needs >= 2 routes, got {len(routes)} "
        f"({routes}) — cap segments with sample_segments_across_routes(), "
        "not list slicing")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(routes), generator=g).tolist()
    n_val = max(1, int(len(routes) * val_frac))
    val_routes = {routes[i] for i in perm[:n_val]}
    train = [s for s in segments if route_of(s) not in val_routes]
    val = [s for s in segments if route_of(s) in val_routes]
    assert not {route_of(s) for s in train} & {route_of(s) for s in val}
    return train, val


def episode_id_of(segment: Path) -> int:
    key = f"{route_of(segment)}/{segment.name}"
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# Logs -> actions & poses (pure numpy, unit-tested without real data)          #
# --------------------------------------------------------------------------- #
def _load_tv(seg: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    d = seg / "processed_log" / "CAN" / name
    return np.load(d / "t"), np.load(d / "value").astype(np.float64).squeeze()


def ecef_to_enu(positions: np.ndarray, velocities: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Segment-local ENU (east, north) positions & velocities.

    Reference = first frame. Geocentric-latitude approximation — fine for a
    1-minute local frame; the derived (x, y) are relative displacements.
    """
    ref = positions[0]
    x, y, z = ref
    lon = math.atan2(y, x)
    lat = math.atan2(z, math.hypot(x, y))
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    r = np.array([[-so, co, 0.0],
                  [-sl * co, -sl * so, cl],
                  [cl * co, cl * so, sl]])
    enu_p = (positions - ref) @ r.T
    enu_v = velocities @ r.T
    return enu_p[:, :2], enu_v[:, :2]


def actions_and_poses(frame_times: np.ndarray, positions: np.ndarray,
                      velocities: np.ndarray, can_speed: tuple[np.ndarray, np.ndarray],
                      can_steer_deg: tuple[np.ndarray, np.ndarray],
                      stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Contract actions [T,2] and poses [T,4] at frame_times[::stride]."""
    t = frame_times[::stride]
    speed = np.interp(t, *can_speed)
    steer_wheel_deg = np.interp(t, *can_steer_deg)
    steer_road_rad = steer_wheel_deg * WHEEL_TO_RAD / STEER_RATIO
    accel = np.gradient(speed, t, edge_order=1)
    actions = np.column_stack([steer_road_rad, accel]).astype(np.float32)

    enu_p, enu_v = ecef_to_enu(positions[::stride], velocities[::stride])
    yaw = np.arctan2(enu_v[:, 1], enu_v[:, 0])                 # heading in ENU
    v = np.linalg.norm(enu_v, axis=1)
    poses = np.column_stack([enu_p[:, 0], enu_p[:, 1], yaw, v]).astype(np.float32)
    return actions, poses


# --------------------------------------------------------------------------- #
# Video decode (lazy av import) and frame preprocessing                        #
# --------------------------------------------------------------------------- #
def _decode_video(seg: Path, stride: int, size: int,
                  max_frames: int | None) -> torch.Tensor:
    """video.hevc -> uint8 [T, 3, size, size], every stride-th frame,
    center-cropped square then bilinearly resized."""
    import av                                                   # lazy: .[real]
    out = []
    with av.open(str(seg / "video.hevc")) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i % stride:
                continue
            rgb = torch.from_numpy(frame.to_ndarray(format="rgb24"))  # H W 3
            out.append(rgb.permute(2, 0, 1))
            if max_frames is not None and len(out) >= max_frames:
                break
    vid = torch.stack(out)                                      # T 3 H W uint8
    h, w = vid.shape[-2:]
    c = min(h, w)
    top, left = (h - c) // 2, (w - c) // 2
    vid = vid[..., top:top + c, left:left + c].float()
    vid = F.interpolate(vid, size=(size, size), mode="bilinear",
                        align_corners=False)
    return vid.clamp(0, 255).to(torch.uint8)


def stack_frames(vid_u8: torch.Tensor, n_stack: int = 3) -> torch.Tensor:
    """[T,3,S,S] uint8 -> [T-(n-1), 3n, S, S]: frames t-(n-1)..t channel-stacked.

    D-015: n_stack=3 at 10 Hz -> the encoder sees [t-200ms, t-100ms, t] in one
    9-channel input, making acceleration/curvature observable per input.
    Oldest frame first, current frame in the LAST 3 channels.
    """
    parts = [vid_u8[i:vid_u8.shape[0] - (n_stack - 1) + i] for i in range(n_stack)]
    return torch.cat(parts, dim=1)


def stack_two_frames(vid_u8: torch.Tensor) -> torch.Tensor:
    """Legacy 2-frame stack (pre-D-015). Kept for tests/compat."""
    return stack_frames(vid_u8, n_stack=2)


def build_episode(segment: Path, size: int = 256, stride: int = 2,
                  max_steps: int | None = 300, n_stack: int = 3,
                  decode_fn=_decode_video) -> ToyEpisode:
    """One comma2k19 segment -> contract episode at FPS/stride Hz.

    D-015: n_stack consecutive strided frames (100 ms apart at stride 2) are
    channel-stacked per step -> [T, 3*n_stack, S, S]; actions/poses aligned to
    the LATEST frame of each stack. max_steps caps memory. decode_fn is
    injectable for tests (no real video needed in CI).
    """
    segment = Path(segment)
    ft = np.load(segment / "global_pose" / "frame_times")
    pos = np.load(segment / "global_pose" / "frame_positions")
    vel = np.load(segment / "global_pose" / "frame_velocities")
    n_avail = (len(ft) + stride - 1) // stride
    n = n_avail if max_steps is None else min(max_steps + n_stack - 1, n_avail)

    actions, poses = actions_and_poses(
        ft, pos, vel, _load_tv(segment, "speed"),
        _load_tv(segment, "steering_angle"), stride)
    vid = decode_fn(segment, stride, size, n)                   # [n,3,S,S] u8
    n = min(n, vid.shape[0], actions.shape[0])
    stacked = stack_frames(vid[:n], n_stack)                    # [n-k+1,3k,S,S]
    k = n_stack - 1
    return ToyEpisode(
        frames=stacked.float().div_(255.0),
        actions=torch.from_numpy(actions[k:n]),
        poses=torch.from_numpy(poses[k:n]),
        episode_id=episode_id_of(segment),
    )


# --------------------------------------------------------------------------- #
# Dataset — window contract identical to the toy/MetaDrive datasets            #
# --------------------------------------------------------------------------- #
class Comma2k19Dataset(torch.utils.data.Dataset):
    """Windows over comma2k19 segments. Build train/val from split_by_route
    (I3) — constructor takes an explicit segment list on purpose."""

    def __init__(self, segments: list[Path], window: int = 8,
                 max_horizon: int = 16, size: int = 256, stride: int = 2,
                 max_steps: int | None = 300, decode_fn=_decode_video):
        self.window, self.max_horizon = window, max_horizon
        self.episodes: list[ToyEpisode] = []
        for seg in segments:
            try:
                self.episodes.append(build_episode(
                    seg, size=size, stride=stride, max_steps=max_steps,
                    decode_fn=decode_fn))
            except Exception as e:                # corrupt segment: skip, log
                print(f"[comma2k19] skipping {seg}: {type(e).__name__}: {e}")
        self.index: list[tuple[int, int]] = []
        for e_i, ep in enumerate(self.episodes):
            t_max = ep.frames.shape[0] - window - max_horizon
            self.index.extend((e_i, t) for t in range(max(0, t_max)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        e_i, t = self.index[i]
        ep = self.episodes[e_i]
        w = self.window
        return {
            "frames": ep.frames[t:t + w],
            "actions": ep.actions[t:t + w],
            "future_frames": ep.frames[t + w:t + w + self.max_horizon],
            "future_poses": ep.poses[t + w:t + w + self.max_horizon],
            "pose_last": ep.poses[t + w - 1],
            "episode_id": ep.episode_id,
        }
