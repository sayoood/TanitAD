"""Curve-rebalance analysis for the TanitAD training mix (FLEET_REVIEW P0#3).

WHY THIS EXISTS (data-side attack on the #1 program risk, 2026-07-18)
--------------------------------------------------------------------
The 2026-07-17 fleet review named the top risk as the single-camera driving-
capability gap, and named its ENABLING CONDITION as a corpus that is
**highway-heavy / mostly-straight** (the review cites ~74% straight; nuScenes
73.9%). A mix dominated by straight driving lets a model satisfy the loss with
the ego-status shortcut (predict "keep going straight at v0") instead of
learning action->consequence on the turns that actually exercise steering. The
loss-side half of the REF-B curve failure shipped in `refbpatch`; THIS is the
data-side half: measure the straight/gentle/sharp distribution per source on
REAL episode bytes, then derive a turn-weighted sampling recipe that moves the
combined mix from its measured straight-fraction toward ~55-60% straight.

STRATUM DEFINITION — copied EXACTLY from `stack/scripts/driving_diagnostic.py`
-----------------------------------------------------------------------------
The eval owner (Benchmarks & Eval) defines the D1 curvature strata by the
**absolute net heading (yaw) change over the 2 s / 20-step prediction horizon**:
    straight : |dyaw@2s| <  5 deg
    gentle   : 5 <= |dyaw@2s| <= 20 deg
    sharp    : |dyaw@2s| >  20 deg
Measuring the CORPUS with the identical convention makes the data-side
straight-fraction directly comparable to the D1 per-stratum ADE numbers
(`net_heading_change_deg` / `curvature_bucket` reproduced here verbatim; a test
asserts the constants still match the eval script so they can never drift).

The anchor set matches the diagnostic: every window whose LAST history frame is
`last` in [1, T-1-horizon] (needs `last-1` for the velocity/yaw-rate baselines
and `last+horizon` for the future). One count per valid anchor => the same
window population the D1 gate strata are computed over.

PURE + STANDALONE: the stratum math is numpy/torch only; episode IO is injected
so the unit tests run with zero real bytes. Run on real epcache with `analyze`.
"""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

# --- constants: MUST equal stack/scripts/driving_diagnostic.py (test-guarded) -
CURV_STRAIGHT_DEG = 5.0          # |net heading change @2s| < 5 deg -> straight
CURV_GENTLE_DEG = 20.0           # 5-20 gentle; >20 sharp
HORIZON = 20                     # K_MAX = max(WP_STEPS) = 20 steps = 2 s @10 Hz
STRATA = ("straight", "gentle", "sharp")


def _wrap(a):
    """Wrap angle(s) to (-pi, pi]. Works on float or Tensor."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def net_heading_change_deg(poses: Tensor, last, horizon: int = HORIZON) -> Tensor:
    """|net heading change| over the future ``horizon`` steps, degrees.

    Verbatim from driving_diagnostic.py: ``poses[:,2]`` is yaw [rad]. ``last`` may
    be a scalar or a 1-D index tensor of anchors -> returns [b]."""
    return _wrap(poses[last + horizon, 2] - poses[last, 2]).abs() * (180.0 / math.pi)


def curvature_bucket(deg: float) -> str:
    """Scalar stratum label (verbatim from driving_diagnostic.py)."""
    if deg < CURV_STRAIGHT_DEG:
        return "straight"
    if deg <= CURV_GENTLE_DEG:
        return "gentle"
    return "sharp"


def episode_stratum_counts(poses: Tensor, horizon: int = HORIZON) -> dict[str, int]:
    """Window counts per stratum for ONE episode ``poses`` [T,4].

    Anchors = ``last`` in [1, T-1-horizon] (the diagnostic's window population:
    needs last-1 for the velocity baseline and last+horizon for the future).
    Vectorized over all valid anchors; returns {stratum: count}."""
    T = poses.shape[0]
    hi = T - 1 - horizon                       # inclusive upper bound on `last`
    counts = {s: 0 for s in STRATA}
    if hi < 1:
        return counts
    anchors = torch.arange(1, hi + 1, dtype=torch.long)
    deg = net_heading_change_deg(poses, anchors, horizon)          # [n]
    straight = (deg < CURV_STRAIGHT_DEG)
    sharp = (deg > CURV_GENTLE_DEG)
    gentle = (~straight) & (~sharp)
    counts["straight"] = int(straight.sum())
    counts["gentle"] = int(gentle.sum())
    counts["sharp"] = int(sharp.sum())
    return counts


@dataclass
class SourceDist:
    """Aggregate stratum distribution for one corpus source."""
    name: str
    n_episodes: int = 0
    counts: dict[str, int] = field(default_factory=lambda: {s: 0 for s in STRATA})

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def fractions(self) -> dict[str, float]:
        tot = self.total or 1
        return {s: self.counts[s] / tot for s in STRATA}

    def add(self, poses: Tensor, horizon: int = HORIZON) -> None:
        c = episode_stratum_counts(poses, horizon)
        for s in STRATA:
            self.counts[s] += c[s]
        self.n_episodes += 1


def corpus_distribution(name: str, poses_iter, horizon: int = HORIZON) -> SourceDist:
    """Aggregate an iterable of episode ``poses`` [T,4] into a SourceDist."""
    d = SourceDist(name=name)
    for poses in poses_iter:
        d.add(poses, horizon)
    return d


# --------------------------------------------------------------------------- #
# Turn-weighted sampling recipe                                               #
# --------------------------------------------------------------------------- #
def turn_upweight_beta(straight_frac: float, target_straight: float) -> float:
    """Single-knob turn-upweight ``beta`` (weight on EVERY non-straight window,
    straight windows keep weight 1) that moves the sampled straight-fraction from
    the measured ``straight_frac`` to ``target_straight``.

    Sampled straight-fraction after weighting =
        s / (s + beta*(1-s))            (s = straight_frac)
    Set equal to target t and solve:
        beta = s*(1-t) / (t*(1-s))
    beta > 1 upweights turns (t < s, the intended direction). Returns 1.0 if the
    mix is already at/below the target (no upweighting needed)."""
    s = float(straight_frac)
    t = float(target_straight)
    if not (0.0 < t < 1.0) or not (0.0 < s < 1.0):
        raise ValueError(f"fractions must be in (0,1): s={s}, t={t}")
    if s <= t:
        return 1.0
    return (s * (1.0 - t)) / (t * (1.0 - s))


def sampled_straight_fraction(straight_frac: float, beta: float) -> float:
    """Straight-fraction after applying turn-upweight ``beta`` (inverse of
    :func:`turn_upweight_beta`; used to verify the recipe by construction)."""
    s = float(straight_frac)
    return s / (s + beta * (1.0 - s))


def per_stratum_weights(straight_frac: float, target_straight: float) -> dict[str, float]:
    """Per-window sampling weights {straight:1, gentle:beta, sharp:beta} for a
    ``WeightedRandomSampler`` keyed by each window's stratum label."""
    beta = turn_upweight_beta(straight_frac, target_straight)
    return {"straight": 1.0, "gentle": beta, "sharp": beta}


def combine_sources(dists: list[SourceDist],
                    source_weights: dict[str, float] | None = None
                    ) -> dict[str, float]:
    """Combined stratum fractions across sources.

    ``source_weights`` (optional) reweights each source's CONTRIBUTION to the mix
    (e.g. the training mix ratio); default = each source contributes in
    proportion to its own window count (natural pooled distribution)."""
    agg = {s: 0.0 for s in STRATA}
    for d in dists:
        w = 1.0 if source_weights is None else source_weights.get(d.name, 0.0)
        if d.total == 0:
            continue
        # weight scales the source's per-window mass so the source's share of the
        # mix equals w (normalized below); within-source shape preserved.
        scale = w / d.total if source_weights is not None else 1.0
        for s in STRATA:
            agg[s] += d.counts[s] * scale
    tot = sum(agg.values()) or 1.0
    return {s: agg[s] / tot for s in STRATA}


# --------------------------------------------------------------------------- #
# Real-bytes IO (epcache) — not exercised by the unit tests                    #
# --------------------------------------------------------------------------- #
def iter_epcache_poses(root: str, limit: int | None = None):
    """Yield ``poses`` [T,4] float32 from each ``ep_*.pt`` under ``root``.

    epcache episodes are ``torch.save({"frames_u8","actions","poses",
    "episode_id"})``; only poses are loaded (frames are skipped for speed via a
    full load then discard — epcache has no partial-tensor read)."""
    eps = sorted(glob.glob(os.path.join(root, "ep_*.pt")))
    if limit is not None:
        eps = eps[:limit]
    for p in eps:
        d = torch.load(p, map_location="cpu", weights_only=False)
        poses = d["poses"] if isinstance(d, dict) else None
        if poses is not None:
            yield poses.float()


def analyze(sources: dict[str, list[str]], target_straight: float = 0.575,
            horizon: int = HORIZON, limit: int | None = None) -> dict:
    """Measure the per-source + combined stratum distribution over real epcache
    roots and derive the turn-weight recipe. ``sources`` maps a source name to a
    list of epcache dirs (train+val shards pooled per source).

    Returns a JSON-able report: per-source fractions/counts, the natural pooled
    combined fractions, the turn-upweight beta to reach ``target_straight``, and
    the verified post-weight straight fraction."""
    dists = []
    for name, roots in sources.items():
        d = SourceDist(name=name)
        for r in roots:
            for poses in iter_epcache_poses(r, limit):
                d.add(poses, horizon)
        dists.append(d)
    combined = combine_sources(dists)                 # natural pooled
    s = combined["straight"]
    beta = turn_upweight_beta(s, target_straight)
    return {
        "horizon_steps": horizon,
        "strata_deg": {"straight": f"<{CURV_STRAIGHT_DEG}",
                       "gentle": f"{CURV_STRAIGHT_DEG}-{CURV_GENTLE_DEG}",
                       "sharp": f">{CURV_GENTLE_DEG}"},
        "per_source": {d.name: {"n_episodes": d.n_episodes, "n_windows": d.total,
                                "counts": d.counts,
                                "fractions": {k: round(v, 4)
                                              for k, v in d.fractions.items()}}
                       for d in dists},
        "combined_natural": {k: round(v, 4) for k, v in combined.items()},
        "target_straight": target_straight,
        "turn_upweight_beta": round(beta, 3),
        "post_weight_straight": round(sampled_straight_fraction(s, beta), 4),
        "per_stratum_weights": {k: round(v, 3)
                                for k, v in per_stratum_weights(s, target_straight).items()},
    }
