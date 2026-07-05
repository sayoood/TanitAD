"""Synthetic ego-centric BEV driving toy — zero external dependencies.

Purpose: CI smoke tests and gate rehearsals on the RTX 4060 before MetaDrive.
An ego vehicle follows a kinematic bicycle model on a procedurally generated
road; the observation is an ego-centric BEV rendering (road + lane marks +
one moving obstacle). Because the frame is ego-centric, every action moves
every pixel — the consequence-dominant regime (A8) that egocentric driving
satisfies by construction.

Episode contract (mirrors the ALPS-4B Two-Rooms contract so downstream code
ports unchanged): frames [T, 1, H, W] float32 in [0,1]; actions [T, 2]
(steer rad, accel m/s^2); poses [T, 4] (x, y, yaw, v); episode_id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class ToyEpisode:
    frames: Tensor        # [T, 1, H, W]
    actions: Tensor       # [T, 2] action taken between t and t+1
    poses: Tensor         # [T, 4] (x, y, yaw, v)
    episode_id: int


def _road_centerline(rng: np.random.Generator, n_pts: int = 400,
                     seg_len: float = 2.0) -> np.ndarray:
    """Procedural centerline: straights and arcs (curvature piecewise const)."""
    pts = [np.zeros(2)]
    heading = 0.0
    curv = 0.0
    for i in range(n_pts - 1):
        if i % 40 == 0:
            curv = float(rng.choice([0.0, 0.02, -0.02, 0.04, -0.04]))
        heading += curv * seg_len
        pts.append(pts[-1] + seg_len * np.array([math.cos(heading), math.sin(heading)]))
    return np.stack(pts)


def _render_bev(pose: np.ndarray, center: np.ndarray, obstacle: np.ndarray,
                size: int, scale: float, road_half_width: float) -> np.ndarray:
    """Ego-centric BEV: road pixels bright, lane center brighter, obstacle brightest."""
    x, y, yaw, _ = pose
    c, s = math.cos(-yaw), math.sin(-yaw)
    rel = center - np.array([x, y])
    rel = rel @ np.array([[c, -s], [s, c]]).T          # rotate into ego frame
    img = np.zeros((size, size), dtype=np.float32)
    half = size // 2
    # Rasterize road band around visible centerline points.
    vis = rel[(np.abs(rel[:, 0]) < size / scale) & (np.abs(rel[:, 1]) < size / scale)]
    for px, py in vis:
        u = int(half + px * scale * 0.5)               # ego at center-left
        v = int(half - py * scale)
        if 0 <= u < size and 0 <= v < size:
            w = max(1, int(road_half_width * scale))
            img[max(0, v - w):v + w, max(0, u - 1):u + 2] = \
                np.maximum(img[max(0, v - w):v + w, max(0, u - 1):u + 2], 0.4)
            img[v, u] = 0.7                            # centerline
    # Obstacle in ego frame.
    orel = (obstacle - np.array([x, y])) @ np.array([[c, -s], [s, c]]).T
    ou = int(half + orel[0] * scale * 0.5)
    ov = int(half - orel[1] * scale)
    if 0 <= ou < size - 2 and 2 <= ov < size:
        img[ov - 2:ov + 2, ou - 1:ou + 3] = 1.0
    # Ego marker (fixed position — its *surroundings* move with actions).
    img[half - 1:half + 2, half - 2:half + 2] = 0.9
    return img


def generate_episode(episode_id: int, steps: int = 80, size: int = 64,
                     dt: float = 0.1, seed: int | None = None) -> ToyEpisode:
    rng = np.random.default_rng(episode_id if seed is None else seed)
    center = _road_centerline(rng)
    pose = np.array([0.0, 0.0, 0.0, 8.0])             # start on road, 8 m/s
    obstacle = center[60].copy()
    obs_speed = 6.0
    obs_idx = 60.0
    frames, actions, poses = [], [], []
    wheelbase = 2.7
    scale = size / 40.0                                # ~40 m field of view
    for t in range(steps):
        frames.append(_render_bev(pose, center, obstacle, size, scale, 3.5))
        poses.append(pose.copy())
        # Driver policy: steer toward lookahead centerline point + noise.
        look = center[min(len(center) - 1, int(t * 1.2) + 15)]
        dx, dy = look - pose[:2]
        target_yaw = math.atan2(dy, dx)
        err = (target_yaw - pose[2] + math.pi) % (2 * math.pi) - math.pi
        steer = float(np.clip(1.5 * err + rng.normal(0, 0.05), -0.5, 0.5))
        accel = float(np.clip(rng.normal(0.2, 0.8), -3.0, 2.0))
        actions.append([steer, accel])
        # Bicycle-model step.
        x, y, yaw, v = pose
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += v / wheelbase * math.tan(steer) * dt
        v = max(0.0, v + accel * dt)
        pose = np.array([x, y, yaw, v])
        # Obstacle advances along the centerline.
        obs_idx = min(len(center) - 1.0, obs_idx + obs_speed * dt / 2.0)
        obstacle = center[int(obs_idx)]
    return ToyEpisode(
        frames=torch.from_numpy(np.stack(frames)).unsqueeze(1),
        actions=torch.tensor(actions, dtype=torch.float32),
        poses=torch.from_numpy(np.stack(poses)).float(),
        episode_id=episode_id,
    )


class ToyDrivingDataset(torch.utils.data.Dataset):
    """Windows of (frames, actions, target future states) over N episodes.

    Splits are EPISODE-level (I3): pass disjoint episode id ranges for
    train/val — never split by frame.
    """

    def __init__(self, episode_ids: list[int], window: int = 6,
                 max_horizon: int = 4, size: int = 64, steps: int = 80):
        self.window, self.max_horizon = window, max_horizon
        self.episodes = [generate_episode(i, steps=steps, size=size)
                         for i in episode_ids]
        self.index: list[tuple[int, int]] = []
        for e_i, ep in enumerate(self.episodes):
            t_max = ep.frames.shape[0] - window - max_horizon
            self.index.extend((e_i, t) for t in range(t_max))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        e_i, t = self.index[i]
        ep = self.episodes[e_i]
        w = self.window
        return {
            "frames": ep.frames[t:t + w],                        # [W, 1, H, W]
            "actions": ep.actions[t:t + w],                      # [W, 2]
            "future_frames": ep.frames[t + w:t + w + self.max_horizon],
            "future_poses": ep.poses[t + w:t + w + self.max_horizon],
            "pose_last": ep.poses[t + w - 1],
            "episode_id": ep.episode_id,
        }


def frame_change_fraction(ep: ToyEpisode) -> float:
    """Consequence-dominance probe (A8): mean fraction of pixels that change
    per step. Egocentric driving should be in the tens of percent."""
    diffs = (ep.frames[1:] - ep.frames[:-1]).abs() > 0.05
    return float(diffs.float().mean())
