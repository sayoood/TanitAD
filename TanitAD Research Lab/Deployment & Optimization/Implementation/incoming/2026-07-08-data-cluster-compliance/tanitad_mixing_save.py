"""Fail-fast `save_episode` — write-boundary shape validation.

Compliance review #1 (Production & Optimization, 2026-07-08). PROPOSED edit to the
`save_episode` function in `stack/tanitad/data/mixing.py` (the rest of that module
is unchanged).

The current `save_episode` writes whatever it is handed. A mis-shaped episode
(actions/poses length != frames T, or non-4D frames) is persisted silently and
only blows up much later inside a training window — the exact failure mode
`_contract.assert_contract` was written to prevent "here rather than deep in
training". This adds the same cheap shape check at the persistence boundary, so a
bad build item fails at write time with a clear message. Range/[0,1] is NOT
re-checked here (save already clamps float frames before uint8 quantization).
"""

from __future__ import annotations

import torch

from tanitad.data.toy_driving import ToyEpisode


def save_episode(ep: ToyEpisode, path: str) -> None:
    """Persist an episode — frames stored uint8 to keep files small
    (accepts uint8 [0,255] or float [0,1] frames).

    Fail-fast: raises ValueError if the episode is mis-shaped, so a broken adapter
    item is caught at the write boundary, not deep inside a training window.
    """
    T = ep.frames.shape[0]
    if ep.frames.ndim != 4:
        raise ValueError(f"save_episode: frames must be [T,C,H,W], got "
                         f"{tuple(ep.frames.shape)}")
    if ep.actions.shape[0] != T or ep.actions.ndim != 2 or ep.actions.shape[1] != 2:
        raise ValueError(f"save_episode: actions must be [T,2] with T={T}, got "
                         f"{tuple(ep.actions.shape)}")
    if ep.poses.shape[0] != T or ep.poses.ndim != 2 or ep.poses.shape[1] != 4:
        raise ValueError(f"save_episode: poses must be [T,4] with T={T}, got "
                         f"{tuple(ep.poses.shape)}")

    if ep.frames.dtype == torch.uint8:
        u8 = ep.frames
    else:
        u8 = (ep.frames.clamp(0, 1) * 255).to(torch.uint8)
    torch.save({
        "frames_u8": u8,
        "actions": ep.actions,
        "poses": ep.poses,
        "episode_id": ep.episode_id,
    }, path)
