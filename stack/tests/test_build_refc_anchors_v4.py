"""flagship v4 P3 (anchor vocabularies) tests — scripts/build_refc_anchors.py.

The existing FPS builder already takes arbitrary ``--horizons``, so v4 needs no new
builder — only the P3 contract pinned (V4_FLAGSHIP_DESIGN §15 P3, §3.3, O-19):

* the DENSE operative vocabulary (256 FPS over horizons 1..20) is a 10,240-float
  buffer — the +24,608-param dense plan is the precondition for every smoothness
  term (§7), and a 4-point head admits exactly one third difference;
* the COARSE tactical vocabulary (256 FPS over 5,10,..,50) is [256, 10, 2];
* base-128 is a BIT-EXACT prefix of the 256 vocabulary (greedy FPS, same seed), so
  the base/XL vocabularies are nested and cross-arm comparable;
* PARITY (O-19): the pool is built from WINDOWS, never re-selected episodes, and a
  window shorter than ``max_horizon`` contributes NOTHING rather than being padded
  — which makes the "74.3 % have 5 s of future" subset automatic for the 5 s
  vocabulary, with no episode dropped from the corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_refc_anchors import build_anchors, episode_traj_pool  # noqa: E402

from tanitad.models.flagship_v4 import (DENSE_HORIZONS,  # noqa: E402
                                        TACTICAL_HORIZONS)
from tanitad.refs.refc import (furthest_point_sample,  # noqa: E402
                               synth_anchor_pool)


class _Ep:
    """Minimal cached-episode stub: episode_traj_pool only reads ``.poses``."""

    def __init__(self, poses: torch.Tensor):
        self.poses = poses


def _straight_ep(t_len: int, v: float = 10.0, dt: float = 0.1) -> _Ep:
    x = torch.arange(t_len, dtype=torch.float32) * v * dt
    poses = torch.zeros(t_len, 4)
    poses[:, 0] = x
    poses[:, 3] = v
    return _Ep(poses)


# ------------------------------------------------------ buffer / geometry ---
def test_dense_operative_vocabulary_is_10240_floats():
    dense, meta = build_anchors(DENSE_HORIZONS, 256, pool_size=1024, seed=0)
    assert tuple(dense.shape) == (256, 20, 2)
    assert dense.numel() == 10_240
    assert meta["method"] == "fps" and meta["n_anchors"] == 256


def test_coarse_tactical_vocabulary_is_256x10():
    coarse, _ = build_anchors(TACTICAL_HORIZONS, 256, pool_size=1024, seed=0)
    assert tuple(coarse.shape) == (256, 10, 2)
    assert TACTICAL_HORIZONS[-1] == 50 and len(TACTICAL_HORIZONS) == 10


# --------------------------------------------------- the nested-prefix pin ---
def test_base128_is_a_bit_exact_prefix_of_the_256_vocabulary():
    """Greedy FPS from a fixed seed picks the same first 128 points whether asked
    for 128 or 256, so the base vocabulary is the XL vocabulary's prefix — the
    property that keeps a K-sweep (§2.7) bit-comparable across anchor counts."""
    pool = synth_anchor_pool(DENSE_HORIZONS, pool_size=1024, seed=0)
    a256 = furthest_point_sample(pool, 256, seed=0)
    a128 = furthest_point_sample(pool, 128, seed=0)
    assert float((a256[:128] - a128).abs().max()) == 0.0


def test_builder_is_deterministic_given_seed():
    a, _ = build_anchors(DENSE_HORIZONS, 64, pool_size=512, seed=0)
    b, _ = build_anchors(DENSE_HORIZONS, 64, pool_size=512, seed=0)
    assert torch.equal(a, b)


# --------------------------------------------------------- parity (O-19) ----
def test_pool_masks_short_windows_without_re_selecting_episodes():
    """A window shorter than max_horizon contributes ZERO rows; it is never padded
    and its episode is never dropped. For the 5 s (max_h 50) vocabulary this makes
    the '74.3 % have 5 s of future' subset automatic — a shorter clip just adds
    fewer trajectories, so parity (no episode re-selection) is preserved."""
    long_ep = _straight_ep(80)        # 80 - 50 = 30 windows at max_h 50
    short_ep = _straight_ep(40)       # 40 - 50 < 0 -> 0 windows at max_h 50
    pool_both = episode_traj_pool([long_ep, short_ep], TACTICAL_HORIZONS)
    pool_long = episode_traj_pool([long_ep], TACTICAL_HORIZONS)
    # the short clip contributes nothing at the 5 s horizon, but is not an error
    assert pool_both.shape[0] == pool_long.shape[0] == 30
    # at a short horizon the SAME short clip does contribute — it was masked, not
    # dropped from the corpus
    pool_short_h = episode_traj_pool([short_ep], (1, 2, 3, 4))
    assert pool_short_h.shape[0] == 40 - 4
