"""Consequence-dominance (A8) statistics harness — backlog item #4.

Measures how much of each frame *changes per step* across a corpus. Egocentric
driving is consequence-dominant (A8): the consequence of an action lives in the
frame. This harness turns that from an assumption into a measured, per-corpus
number that two decisions consume:

  * **Change-weighted loss (W2 bake-off, H3/A8):** on a real highway camera the
    A8 fraction is low (~0.05, measured 2026-07-07) because sky/road are
    low-texture — flat MSE is then dominated by trivially-predictable background.
    The measured per-corpus fraction sets the change-weight schedule from data.
  * **Real+sim mix (D-010):** compare the consequence-signal of each domain (real
    vs sim) so the sim share is justified by coverage it adds, not assumed.

The metric is :func:`tanitad.data._contract.frame_change_fraction` (mean fraction
of pixels whose abs change exceeds ``thresh``). It is **representation-dependent**
— a 1-channel BEV frame and a 6-channel 2-frame-stack are not directly comparable
in absolute terms (the stack's overlapping channels dampen the per-step diff).
Compare *within* a representation, or pass ``channels`` to score a common slice
(e.g. the last 3 channels = the current RGB frame of a 2-frame stack).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tanitad.data._contract import frame_change_fraction
from tanitad.data.toy_driving import ToyEpisode


def episode_change_fraction(ep: ToyEpisode, thresh: float = 0.05,
                            channels: tuple[int, int] | None = None) -> float:
    """A8 fraction for one episode. ``channels=(a, b)`` scores frames[:, a:b]
    only (e.g. ``(3, 6)`` = the current RGB frame of a 2-frame stack), so
    differently-stacked corpora can be compared on a common slice."""
    frames = ep.frames if channels is None else ep.frames[:, channels[0]:channels[1]]
    return frame_change_fraction(frames, thresh=thresh)


@dataclass
class A8Stats:
    n_episodes: int
    n_frames: int
    # thresh -> {mean, median, p10, p90, min, max} of the per-episode A8 fraction
    per_threshold: dict[float, dict[str, float]]

    def as_row(self, thresh: float = 0.05) -> str:
        s = self.per_threshold[thresh]
        return (f"n={self.n_episodes:>4} eps / {self.n_frames:>6} frames | "
                f"A8@{thresh:.2f}: mean={s['mean']:.3f} median={s['median']:.3f} "
                f"p10={s['p10']:.3f} p90={s['p90']:.3f}")


def consequence_dominance_stats(episodes: list[ToyEpisode],
                                thresholds: tuple[float, ...] = (0.05, 0.10),
                                channels: tuple[int, int] | None = None
                                ) -> A8Stats:
    """Aggregate the per-episode A8 fraction over a corpus, per threshold.

    Percentiles are over the per-episode values (each episode is one sample), so a
    long tail of static clips is visible rather than washed out by a frame-weighted
    mean. ``channels`` is forwarded to :func:`episode_change_fraction`.
    """
    if not episodes:
        raise ValueError("no episodes to summarize")
    n_frames = int(sum(ep.frames.shape[0] for ep in episodes))
    per_threshold: dict[float, dict[str, float]] = {}
    for thr in thresholds:
        vals = np.array([episode_change_fraction(ep, thr, channels)
                         for ep in episodes], dtype=np.float64)
        per_threshold[float(thr)] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return A8Stats(len(episodes), n_frames, per_threshold)


def stats_by_label(labelled: list[tuple[str, ToyEpisode]],
                   thresholds: tuple[float, ...] = (0.05, 0.10),
                   channels: tuple[int, int] | None = None
                   ) -> dict[str, A8Stats]:
    """Per-corpus / per-domain A8 stats. Feed ``[(domain_name, episode), ...]``
    (e.g. tagging real vs sim) to get the D-010 side-by-side comparison."""
    groups: dict[str, list[ToyEpisode]] = {}
    for label, ep in labelled:
        groups.setdefault(label, []).append(ep)
    return {label: consequence_dominance_stats(eps, thresholds, channels)
            for label, eps in groups.items()}


def format_report(by_label: dict[str, A8Stats], thresh: float = 0.05) -> str:
    """One line per corpus, widest label aligned — drop into an experiment REPORT."""
    if not by_label:
        return "(no corpora)"
    w = max(len(k) for k in by_label)
    return "\n".join(f"{label:<{w}}  {st.as_row(thresh)}"
                     for label, st in by_label.items())
