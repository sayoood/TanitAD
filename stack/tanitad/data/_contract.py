"""Shared TanitAD episode-contract assembly -- one home for the contract guarantee.

Every data adapter (toy, MetaDrive, comma2k19, ...) emits a
:class:`~tanitad.data.toy_driving.ToyEpisode` with the SAME contract so that all
downstream consumers (dataset windows, world-model training, the D1-D3 gate
runner) are source-agnostic:

    frames  [T, 1, H, W]  float32 in [0, 1]     (single-channel ego BEV)
    actions [T, 2]        (steer rad, accel m/s^2), taken between t and t+1
    poses   [T, 4]        (x, y, yaw, v)
    episode_id : int

The finite-difference accel semantics, the contract assertion, and the windowed
``Dataset`` are defined ONCE here and reused by every adapter, so a loader cannot
silently drift from the contract. ``toy_driving`` predates this module and keeps
its own (episode-level) ``frame_change_fraction``; this module's variant operates
on a raw frames tensor, matching the adapter loaders.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from tanitad.data.toy_driving import ToyEpisode


def finite_diff_accel(v: np.ndarray, dt: float) -> np.ndarray:
    """Longitudinal accel (m/s^2) as the forward finite difference of speed.

    ``accel[t]`` is the acceleration applied between ``t`` and ``t+1`` (matching
    the contract's action-at-t semantics). The final step repeats the previous
    value so the length equals ``len(v)``.
    """
    v = np.asarray(v, dtype=np.float32)
    if v.shape[0] < 2:
        return np.zeros_like(v)
    a = np.empty_like(v)
    a[:-1] = (v[1:] - v[:-1]) / float(dt)
    a[-1] = a[-2]
    return a


def frame_change_fraction(frames: Tensor, thresh: float = 0.05) -> float:
    """Consequence-dominance probe (A8): mean fraction of pixels changing/step."""
    diffs = (frames[1:] - frames[:-1]).abs() > thresh
    return float(diffs.float().mean())


def assert_contract(ep: ToyEpisode, channels: int | None = 1) -> None:
    """Validate a :class:`ToyEpisode` against the episode contract (raises).

    ``channels`` is the required frame channel count: ``1`` for the single-channel
    BEV contract (toy / MetaDrive), ``6`` for the D-009 ``base250cam`` 2-frame RGB
    stack (comma2k19), or ``None`` to accept any channel count. The action/pose/
    range invariants are identical across every adapter regardless of channels.
    """
    T = ep.frames.shape[0]
    assert ep.frames.ndim == 4, ep.frames.shape
    if channels is not None:
        assert ep.frames.shape[1] == channels, ep.frames.shape
    assert ep.actions.shape == (T, 2), ep.actions.shape
    assert ep.poses.shape == (T, 4), ep.poses.shape
    assert float(ep.frames.min()) >= 0.0 and float(ep.frames.max()) <= 1.0


def assemble_episode(frames: list[Tensor], poses: list[np.ndarray],
                     steer_rad: list[float], dt: float,
                     episode_id: int) -> ToyEpisode:
    """Stack per-step buffers into a contract-compliant :class:`ToyEpisode`.

    ``actions`` = column-stack of ``steer_rad`` and the finite-difference accel
    of the pose speeds (``poses[:, 3]``). Shapes are validated before return, so
    a mis-shaped adapter fails here rather than deep in training.
    """
    frames_t = torch.stack(list(frames), dim=0).to(torch.float32)   # [T, 1, H, W]
    poses_arr = np.stack(poses).astype(np.float32)                  # [T, 4]
    accel = finite_diff_accel(poses_arr[:, 3], dt)                  # [T]
    actions = np.column_stack([np.asarray(steer_rad, np.float32), accel])

    ep = ToyEpisode(
        frames=frames_t,
        actions=torch.from_numpy(actions).to(torch.float32),
        poses=torch.from_numpy(poses_arr).to(torch.float32),
        episode_id=episode_id,
    )
    assert_contract(ep)
    return ep


class EpisodeWindowDataset(torch.utils.data.Dataset):
    """Windows of (frames, actions, future states) over a list of episodes.

    The window/return contract is byte-for-byte identical to
    :class:`~tanitad.data.toy_driving.ToyDrivingDataset`, so a model trained on
    one adapter's episodes consumes any other's without a code change. Splits are
    EPISODE-level (I3): build train/val from disjoint episode lists -- never split
    the windows of one episode across sets.
    """

    def __init__(self, episodes: list[ToyEpisode], window: int = 6,
                 max_horizon: int = 4):
        self.window, self.max_horizon = window, max_horizon
        self.episodes = episodes
        self.index: list[tuple[int, int]] = []
        for e_i, ep in enumerate(episodes):
            t_max = ep.frames.shape[0] - window - max_horizon
            self.index.extend((e_i, t) for t in range(t_max))

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
