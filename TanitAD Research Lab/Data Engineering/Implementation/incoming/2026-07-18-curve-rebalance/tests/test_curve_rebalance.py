"""Standalone tests for the curve-rebalance analyzer (FLEET_REVIEW P0#3).

Zero real bytes: episode poses are synthesized as constant-yaw-rate arcs, so the
per-stratum window counts are analytically known. `tanitad` need NOT be
importable for the core tests; one test additionally guards that the stratum
constants still match the eval script (skipped if the script can't be imported).

    pytest "TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-18-curve-rebalance/tests" -q
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import curve_rebalance as cr                                       # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic constant-yaw-rate arc: yaw advances omega*dt per step.             #
# net heading change over `horizon` steps = omega*dt*horizon (constant for     #
# every interior anchor) -> a whole episode lands in ONE known stratum.        #
# --------------------------------------------------------------------------- #
def arc_poses(T: int, deg_per_2s: float, horizon: int = cr.HORIZON) -> torch.Tensor:
    """[T,4] poses whose |net heading change| over `horizon` steps == deg_per_2s
    for every anchor (yaw linear in step)."""
    dyaw_per_step = math.radians(deg_per_2s) / horizon
    yaw = torch.arange(T, dtype=torch.float32) * dyaw_per_step
    poses = torch.zeros(T, 4)
    poses[:, 2] = yaw
    poses[:, 3] = 10.0                                            # v
    # fill x,y so it is a plausible arc (not used by the stratum math)
    poses[:, 0] = torch.cumsum(torch.cos(yaw) * 1.0, 0)
    poses[:, 1] = torch.cumsum(torch.sin(yaw) * 1.0, 0)
    return poses


def test_stratum_buckets_match_thresholds():
    assert cr.curvature_bucket(0.0) == "straight"
    assert cr.curvature_bucket(4.99) == "straight"
    assert cr.curvature_bucket(5.0) == "gentle"        # boundary: >=5 gentle
    assert cr.curvature_bucket(20.0) == "gentle"       # boundary: <=20 gentle
    assert cr.curvature_bucket(20.01) == "sharp"


def test_episode_counts_straight_arc():
    # 2 deg over 2 s -> straight; every valid anchor counts, none elsewhere.
    poses = arc_poses(200, deg_per_2s=2.0)
    c = cr.episode_stratum_counts(poses)
    assert c["gentle"] == 0 and c["sharp"] == 0
    # anchors: last in [1, T-1-horizon] = [1, 179] -> 179 windows
    assert c["straight"] == 200 - 1 - cr.HORIZON


def test_episode_counts_gentle_and_sharp():
    assert cr.episode_stratum_counts(arc_poses(120, 10.0))["gentle"] > 0
    assert cr.episode_stratum_counts(arc_poses(120, 10.0))["straight"] == 0
    sharp = cr.episode_stratum_counts(arc_poses(120, 40.0))
    assert sharp["sharp"] > 0 and sharp["gentle"] == 0 and sharp["straight"] == 0


def test_short_episode_yields_no_windows():
    # T <= horizon+1 -> no valid anchor
    assert sum(cr.episode_stratum_counts(arc_poses(cr.HORIZON, 5.0)).values()) == 0


def test_source_dist_fractions_sum_to_one():
    d = cr.SourceDist("mix")
    for _ in range(3):
        d.add(arc_poses(120, 2.0))     # straight
    for _ in range(1):
        d.add(arc_poses(120, 40.0))    # sharp
    fr = d.fractions
    assert abs(sum(fr.values()) - 1.0) < 1e-9
    assert fr["straight"] > fr["sharp"] > 0.0
    assert d.n_episodes == 4


def test_turn_upweight_beta_hits_target_by_construction():
    # A measured 74%-straight mix, target 57.5% -> beta upweights turns; applying
    # it must reproduce the target exactly (round-trip).
    s = 0.74
    beta = cr.turn_upweight_beta(s, 0.575)
    assert beta > 1.0                                  # turns upweighted
    assert abs(cr.sampled_straight_fraction(s, beta) - 0.575) < 1e-9


def test_beta_is_identity_when_already_below_target():
    # already less straight than target -> no upweight (beta == 1)
    assert cr.turn_upweight_beta(0.40, 0.575) == 1.0
    assert abs(cr.sampled_straight_fraction(0.40, 1.0) - 0.40) < 1e-12


def test_per_stratum_weights_shape():
    w = cr.per_stratum_weights(0.74, 0.575)
    assert w["straight"] == 1.0
    assert w["gentle"] == w["sharp"] > 1.0


def test_combine_sources_natural_pool():
    a = cr.SourceDist("a"); a.counts.update(straight=80, gentle=15, sharp=5)
    b = cr.SourceDist("b"); b.counts.update(straight=40, gentle=40, sharp=20)
    comb = cr.combine_sources([a, b])                  # natural pool = raw sum
    assert abs(comb["straight"] - (120 / 200)) < 1e-9
    assert abs(sum(comb.values()) - 1.0) < 1e-9


def test_combine_sources_with_mix_weights():
    # equal 50/50 source weights average the two shapes regardless of raw counts
    a = cr.SourceDist("a"); a.counts.update(straight=90, gentle=8, sharp=2)   # tot 100
    b = cr.SourceDist("b"); b.counts.update(straight=200, gentle=600, sharp=200)  # tot 1000
    comb = cr.combine_sources([a, b], source_weights={"a": 0.5, "b": 0.5})
    # a is 0.90 straight, b is 0.20 straight -> 50/50 -> 0.55
    assert abs(comb["straight"] - 0.55) < 1e-9


def test_constants_match_eval_script():
    """Guard against drift: the stratum thresholds + horizon MUST equal
    stack/scripts/driving_diagnostic.py (the eval owner's definition). Skipped
    only if the eval script cannot be imported in this environment."""
    repo = Path(__file__).resolve().parents[6]
    scripts = repo / "stack" / "scripts"
    if not (scripts / "driving_diagnostic.py").exists():
        pytest.skip("driving_diagnostic.py not locatable from intake dir")
    sys.path.insert(0, str(scripts))
    try:
        import driving_diagnostic as dd                # noqa: E402
    except Exception as e:                              # heavy deps missing
        pytest.skip(f"cannot import driving_diagnostic: {e}")
    assert cr.CURV_STRAIGHT_DEG == dd.CURV_STRAIGHT_DEG
    assert cr.CURV_GENTLE_DEG == dd.CURV_GENTLE_DEG
    assert cr.HORIZON == dd.K_MAX
    # and the bucket function agrees on a sweep
    for deg in (0.0, 3.0, 5.0, 12.0, 20.0, 35.0):
        assert cr.curvature_bucket(deg) == dd.curvature_bucket(deg)


def test_net_heading_change_matches_manual():
    poses = arc_poses(120, deg_per_2s=12.0)
    d = float(cr.net_heading_change_deg(poses, torch.tensor([50]))[0])
    assert abs(d - 12.0) < 1e-3
