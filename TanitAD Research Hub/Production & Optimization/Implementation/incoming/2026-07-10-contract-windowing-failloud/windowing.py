"""Fail-loud episode windowing — Production & Optimization compliance review #3.

Self-contained drop-in for the window-index construction shared by
``stack/tanitad/data/_contract.py::EpisodeWindowDataset`` and the identical loop
in ``stack/tanitad/data/toy_driving.py::ToyDrivingDataset``. Imports only torch +
stdlib, so the intake tests need no ``tanitad`` install.

Defect (silent-wrong/no-data class, F-ops-fragility). Both datasets build their
window index with:

    t_max = ep.frames.shape[0] - window - max_horizon
    self.index.extend((e_i, t) for t in range(t_max))

If an episode has ``T < window + max_horizon + 1`` then ``t_max <= 0``,
``range(t_max)`` is empty, and the episode contributes **zero windows and is
silently dropped** — no counter, no warning, no log. Consequences observed in
this program's ops history (F-5/F-6/F-7 class):

* A data/config change (larger ``window``/``max_horizon``, or a corpus of
  shorter clips) silently shrinks the training set with **no signal at all** —
  the biased-toward-long-episodes sampling is invisible.
* If **every** episode is too short the dataset becomes ``len == 0``. The trainer
  (`train_worldmodel.train`) then loops ``while step < steps`` over an empty
  ``DataLoader``: ``next(data_iter)`` raises ``StopIteration``, the loop rebuilds
  the iterator, it raises again — either an infinite no-progress spin (with
  ``drop_last=True``, an empty loader yields nothing) or a crash deep inside
  training whose message hides the real cause (episodes too short for the
  window+horizon).

Fix: count the dropped episodes, **warn** (observable) whenever any are dropped,
and **raise a clear ``ValueError`` at construction time** if the resulting index
is empty — naming ``window``, ``max_horizon``, the required minimum episode
length, and the lengths actually seen. Turns a silent spin/late crash into a
loud, actionable failure at dataset-build time.

Non-dropped behaviour is byte-for-byte identical to the current code: the index
for an episode of length ``T >= window + max_horizon + 1`` is exactly
``range(T - window - max_horizon)`` (the shared, deliberately-conservative
boundary convention is preserved — do NOT change it, it keeps parity with
``ToyDrivingDataset``).
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor


def min_episode_length(window: int, max_horizon: int) -> int:
    """Smallest episode length ``T`` that yields at least one window.

    A window exists iff ``t_max = T - window - max_horizon >= 1``  ->
    ``T >= window + max_horizon + 1``.
    """
    return window + max_horizon + 1


def build_window_index(
    episode_lengths: list[int], window: int, max_horizon: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Build the ``(episode_index, start_t)`` list, failing loud on empties.

    Parameters
    ----------
    episode_lengths : list[int]
        Per-episode frame count ``T`` (``ep.frames.shape[0]``), in order.
    window, max_horizon : int
        The window and max-horizon sizes.

    Returns
    -------
    index : list[(e_i, t)]
        Identical to ``[(e_i, t) for e_i,T in ... for t in range(T-window-
        max_horizon)]`` — the current convention, unchanged for valid episodes.
    dropped : list[(e_i, T)]
        Episodes that contribute zero windows (too short).

    Raises
    ------
    ValueError
        If ``window < 1`` or ``max_horizon < 0``, or if **no** episode yields a
        window (the dataset would be empty).

    Warns
    -----
    UserWarning
        If some (but not all) episodes are dropped for being too short.
    """
    if window < 1 or max_horizon < 0:
        raise ValueError(
            f"invalid window/max_horizon: window={window} (need >=1), "
            f"max_horizon={max_horizon} (need >=0)")

    min_len = min_episode_length(window, max_horizon)
    index: list[tuple[int, int]] = []
    dropped: list[tuple[int, int]] = []
    for e_i, T in enumerate(episode_lengths):
        t_max = int(T) - window - max_horizon
        if t_max <= 0:
            dropped.append((e_i, int(T)))
            continue
        index.extend((e_i, t) for t in range(t_max))

    if dropped:
        shown = [T for _, T in dropped][:8]
        warnings.warn(
            f"EpisodeWindowDataset: {len(dropped)} of {len(episode_lengths)} "
            f"episode(s) are shorter than window+max_horizon+1={min_len} "
            f"(window={window}, max_horizon={max_horizon}) and contribute 0 "
            f"windows — dropped. Lengths seen: {shown}"
            + (" ..." if len(dropped) > 8 else "")
            + ". A shrinking, long-episode-biased train set is otherwise silent.",
            stacklevel=2,
        )

    if not index:
        raise ValueError(
            f"EpisodeWindowDataset is EMPTY: all {len(episode_lengths)} "
            f"episode(s) are shorter than the required "
            f"window+max_horizon+1={min_len} (window={window}, "
            f"max_horizon={max_horizon}); episode lengths="
            f"{[int(t) for t in episode_lengths][:16]}"
            + (" ..." if len(episode_lengths) > 16 else "")
            + ". A length-0 dataset makes the trainer spin on StopIteration or "
            "crash deep in training — fix the window/horizon or the data source.")

    return index, dropped


def _to_float_frames(x: Tensor) -> Tensor:
    """uint8 [0,255] -> float32 [0,1]; float passes through (contract layout)."""
    return x.float().div(255.0) if x.dtype == torch.uint8 else x


class EpisodeWindowDataset(torch.utils.data.Dataset):
    """Fail-loud variant of ``stack/tanitad/data/_contract.EpisodeWindowDataset``.

    Duck-typed on the episode contract (``.frames [T,C,H,W]``, ``.actions
    [T,2]``, ``.poses [T,4]``, ``.episode_id``) so it needs no ``tanitad``
    import. ``__getitem__`` returns the byte-for-byte identical window dict.

    Extra attributes vs the current class:
      ``n_dropped_episodes`` : int
      ``dropped``            : list[(e_i, T)]
    """

    def __init__(self, episodes: list, window: int = 6, max_horizon: int = 4):
        self.window, self.max_horizon = window, max_horizon
        self.episodes = episodes
        lengths = [ep.frames.shape[0] for ep in episodes]
        self.index, self.dropped = build_window_index(lengths, window, max_horizon)
        self.n_dropped_episodes = len(self.dropped)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        e_i, t = self.index[i]
        ep = self.episodes[e_i]
        w = self.window
        return {
            "frames": _to_float_frames(ep.frames[t:t + w]),
            "actions": ep.actions[t:t + w],
            "future_frames": _to_float_frames(
                ep.frames[t + w:t + w + self.max_horizon]),
            "future_actions": ep.actions[t + w:t + w + self.max_horizon],
            "future_poses": ep.poses[t + w:t + w + self.max_horizon],
            "pose_last": ep.poses[t + w - 1],
            "episode_id": ep.episode_id,
        }
