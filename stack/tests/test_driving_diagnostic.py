"""Tests for scripts/driving_diagnostic.py.

CPU synthetic, no real data / no checkpoint:
  - trivial baselines EXACT on hand-built straight + constant-curvature paths
    (constant_velocity ~0 on constant-velocity data; go_straight ~0 on straight
    data; constant_yaw_rate ~0 on a forward-Euler constant-yaw-rate arc);
  - curvature bucketing + net-heading-change correct on synthetic straight/curve;
  - scalar metric + stratum aggregation values correct by hand;
  - the whole collect -> 4 sections pipeline runs end-to-end through a real
    (untrained) smoke WorldModel + toy episodes and is well-formed.

Poses are built in float64 so the "exact" assertions hold to ~machine epsilon.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import driving_diagnostic as dd  # noqa: E402

from tanitad.config import smoke_config  # noqa: E402
from tanitad.data.toy_driving import generate_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402


# --------------------------------------------------------------------------- #
# synthetic trajectory builders (poses [T,4] = x, y, yaw, v)                   #
# --------------------------------------------------------------------------- #
def _straight_poses(T=40, theta=0.3, s=2.0):
    """Constant-velocity straight line at heading ``theta``, speed ``s``/step."""
    t = torch.arange(T, dtype=torch.float64)
    d = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float64)
    xy = t[:, None] * s * d[None, :]
    yaw = torch.full((T,), theta, dtype=torch.float64)
    v = torch.full((T,), s, dtype=torch.float64)
    return torch.cat([xy, yaw[:, None], v[:, None]], dim=1)


def _curve_poses(T=40, phi0=0.1, omega=0.02, s=2.0):
    """Constant speed + constant yaw-rate, forward-Euler (matches the baseline).

    Step k (k>=1) advances along heading phi[k-1]; yaw[k] = phi[k] so a
    one-step yaw difference recovers ``omega`` exactly.
    """
    phi = phi0 + omega * torch.arange(T, dtype=torch.float64)
    xy = torch.zeros(T, 2, dtype=torch.float64)
    for k in range(1, T):
        step = s * torch.tensor([math.cos(float(phi[k - 1])),
                                 math.sin(float(phi[k - 1]))], dtype=torch.float64)
        xy[k] = xy[k - 1] + step
    v = torch.full((T,), s, dtype=torch.float64)
    return torch.cat([xy, phi[:, None], v[:, None]], dim=1)


# --------------------------------------------------------------------------- #
# baseline exactness                                                          #
# --------------------------------------------------------------------------- #
def test_constant_velocity_exact_on_cv_data():
    poses = _straight_poses()
    last = torch.tensor([1, 5, 10])
    gt = dd.gt_ego_waypoints(poses, last)
    bp = dd.baseline_waypoints(poses, last)
    de = dd.de_of(bp["constant_velocity"], gt)
    assert float(de.max()) < 1e-9


def test_all_baselines_exact_on_straight_data():
    """On a straight constant-speed line, all three trivial predictors are exact
    (go_straight and constant_yaw_rate included: zero lateral, omega==0)."""
    poses = _straight_poses(theta=-0.4, s=3.0)
    last = torch.tensor([2, 7, 15])
    gt = dd.gt_ego_waypoints(poses, last)
    bp = dd.baseline_waypoints(poses, last)
    for name in dd.BASELINES:
        assert float(dd.de_of(bp[name], gt).max()) < 1e-9, name


def test_constant_yaw_rate_exact_on_curve_and_cv_is_not():
    poses = _curve_poses(T=45, omega=0.02, s=2.0)
    last = torch.tensor([2, 6, 12])
    gt = dd.gt_ego_waypoints(poses, last)
    bp = dd.baseline_waypoints(poses, last)
    # CYR reconstructs the arc to machine precision...
    assert float(dd.de_of(bp["constant_yaw_rate"], gt).max()) < 1e-9
    # ...while constant-velocity / go-straight genuinely miss the curve.
    assert float(dd.de_of(bp["constant_velocity"], gt).max()) > 1e-2
    assert float(dd.de_of(bp["go_straight"], gt).max()) > 1e-2


# --------------------------------------------------------------------------- #
# curvature bucketing + net heading change                                    #
# --------------------------------------------------------------------------- #
def test_curvature_bucket_thresholds():
    assert dd.curvature_bucket(0.0) == "straight"
    assert dd.curvature_bucket(4.99) == "straight"
    assert dd.curvature_bucket(5.0) == "gentle"      # 5 deg -> gentle (boundary)
    assert dd.curvature_bucket(20.0) == "gentle"     # 20 deg -> gentle (boundary)
    assert dd.curvature_bucket(20.01) == "sharp"
    assert dd.curvature_bucket(45.0) == "sharp"


def test_net_heading_change_and_bucket_on_synthetic():
    # straight -> ~0 deg -> "straight"
    st = _straight_poses()
    deg0 = float(dd.net_heading_change_deg(st, torch.tensor([2]), 20)[0])
    assert deg0 < 1e-6 and dd.curvature_bucket(deg0) == "straight"
    # sharp curve: 20 steps * 0.02 rad = 0.4 rad = 22.918 deg -> "sharp"
    sharp = _curve_poses(T=45, omega=0.02)
    deg_s = float(dd.net_heading_change_deg(sharp, torch.tensor([2]), 20)[0])
    assert abs(deg_s - math.degrees(0.4)) < 1e-4
    assert dd.curvature_bucket(deg_s) == "sharp"
    # gentle curve: 20 * radians(0.5) = 10 deg -> "gentle"
    gentle = _curve_poses(T=45, omega=math.radians(0.5))
    deg_g = float(dd.net_heading_change_deg(gentle, torch.tensor([2]), 20)[0])
    assert abs(deg_g - 10.0) < 1e-4 and dd.curvature_bucket(deg_g) == "gentle"


# --------------------------------------------------------------------------- #
# metric + aggregation values                                                 #
# --------------------------------------------------------------------------- #
def test_scalar_metrics_values():
    de = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    m = dd.scalar_metrics(de)
    assert m["de@0.5s"] == pytest.approx(1.0)
    assert m["de@1s"] == pytest.approx(2.0)
    assert m["de@2s"] == pytest.approx(4.0)
    assert m["ade@1s"] == pytest.approx(1.5)          # mean(1,2)
    assert m["ade@2s"] == pytest.approx(2.5)          # mean(1,2,3,4)
    assert m["ade_0_2s"] == pytest.approx(2.5)


def test_stratum_aggregation_values():
    labels = ["a", "a", "b"]
    md = torch.tensor([[1.0, 1, 1, 1], [3.0, 3, 3, 3], [5.0, 5, 5, 5]])
    cd = md.clone()
    out = dd._strat(labels, md, cd)
    assert out["a"]["n"] == 2 and out["b"]["n"] == 1
    assert out["a"]["model_ade@2s"] == pytest.approx(2.0)   # mean of rows 0,1
    assert out["a"]["model_ade@1s"] == pytest.approx(2.0)
    assert out["a"]["model_de@1s"] == pytest.approx(2.0)
    assert out["b"]["model_ade@2s"] == pytest.approx(5.0)
    assert out["a"]["low_confidence"] is True              # n < 30
    assert out["b"]["low_confidence"] is True


def test_mean_ci_basic():
    ci = dd.mean_ci([1.0, 1.0, 1.0])
    assert ci["mean"] == pytest.approx(1.0) and ci["ci95"] == pytest.approx(0.0)
    assert ci["n_splits"] == 3


# --------------------------------------------------------------------------- #
# end-to-end pipeline through a real (untrained) smoke WorldModel             #
# --------------------------------------------------------------------------- #
def test_pipeline_end_to_end_cpu():
    torch.manual_seed(0)
    world = WorldModel(smoke_config()).eval()
    eps = [generate_episode(i, steps=80, size=64) for i in range(4)]
    corpora = ["comma2k19", "comma2k19", "physicalai", "physicalai"]
    window = world.predictor.cfg.window
    with strict_numerics():
        data = dd.collect(world, eps, corpora, "cpu", window, stride=8, batch=4)
        assert data["states"].shape[0] > 0 and data["gt"].shape[1:] == (4, 2)
        splits = [dd.split_by_episode(data["eid"], 0.5, s) for s in range(3)]
        s1 = dd.section1_baselines(data, splits)
        s2 = dd.section2_decode_ladder(data, splits, mlp_epochs=4)
        s3 = dd.section3_localization(
            data, splits, s2["best_probe_by_heldout_ade_0_2s"], mlp_epochs=4)
        data["_localization"] = s3
        s4 = dd.section4_instruments(world, eps, data, splits, s2, "cpu")

    for n in dd.BASELINES:
        assert math.isfinite(s1[n]["ade@1s"]["mean"])
        assert math.isfinite(s1[n]["ade@2s"]["ci95"])
    assert s2["model_trajectory_head"] is None            # no native head
    assert s2["best_probe_by_heldout_ade_0_2s"] in {
        "ridge_a1", "ridge_a10", "ridge_a100", "mlp"}
    # oracle in-distribution ceiling must not be worse than held-out on ridge_a1
    assert (s2["oracle_ceiling"]["ridge_a1"]["ade_0_2s"]["mean"]
            <= s2["held_out"]["ridge_a1"]["ade_0_2s"]["mean"] + 1e-6)
    for dim in ("by_curvature", "by_speed", "by_corpus"):
        assert s3[dim], dim
        for lab, v in s3[dim].items():
            assert v["n"] >= 1 and "model_ade@1s" in v and "cv_ade@1s" in v
    assert s4["I2_batch_consistency_max_rel"] >= 0.0
    assert s4["I2_pass"] is True                           # batch-free smoke encoder
    assert math.isfinite(s4["I1_oracle_probe_fit_r2"]["mean"])
    assert len(s4["episodes_per_split"]) == 3
