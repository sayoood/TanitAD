"""flagship v4 label mint + wiring — the pre-launch blocker that makes a v4 run
actually test v4 (V4_FLAGSHIP_DESIGN §6.2 factorised tactical, §4.3/§7A.4 strategic
scalars).

Pins, each with the failure it guards:
(a) VOCAB WIDTHS — the label→index maps are sized off refb_labels' kinematic
    token tuples and MATCH the model widths flagship_v4.N_LAT/N_LON/N_DIST. A drift
    here trains the wrong number of logits.
(b) MASKING — an unknown/out-of-horizon slot reaches the loss as IGNORE_INDEX, a
    class it never trains, never a wrong class (§6.5). Every valid index is in range.
(c) KINEMATICS — a stop trajectory mints stop_at_point; a cruise mints free_cruise
    /lane_keep; a junction turn mints a route token + ttm. The labels mean what
    they say.
(d) LOSS — factorised_ce and strategic_scalar_loss are non-zero and push gradient
    on real minted targets, and are exactly 0 (no grad) when every row is masked.
(e) DATASET — FlagshipV4Dataset adds the batch keys train_flagship_v4 reads and
    leaves every v1/v2.1 key byte-identical (additive).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import v4_labels as VL  # noqa: E402
from tanitad.models.flagship_v4 import N_DIST, N_LAT, N_LON  # noqa: E402
from tanitad.models.strategic_goal import (GoalScalarConfig,  # noqa: E402
                                           GoalScalarHead, param_count)
from tanitad.train.v4_curriculum import (IGNORE_INDEX,  # noqa: E402
                                         factorised_ce, strategic_scalar_loss)


# --------------------------------------------------------- synthetic poses ---
def _integrate(v, yaw_rate, T=200, dt=0.1):
    """poses [T, 4] = x, y, yaw, v from a speed + yaw-rate profile (scalars or
    length-T arrays), unicycle-integrated in the repo's +y=left convention."""
    def at(x, k):
        return x[k] if hasattr(x, "__len__") else x
    x = y = yaw = 0.0
    rows = []
    for k in range(T):
        vv = at(v, k)
        rows.append([x, y, yaw, vv])
        x += vv * math.cos(yaw) * dt
        y += vv * math.sin(yaw) * dt
        yaw += at(yaw_rate, k) * dt
    return torch.tensor(rows, dtype=torch.float32)


def _cruise(T=140, v=10.0):
    return _integrate(v, 0.0, T)


def _stop(T=140, v0=10.0, decel=1.5, dt=0.1):
    v = np.maximum(v0 - decel * dt * np.arange(T), 0.0)
    return _integrate(v, 0.0, T)


def _left_turn(T=200, v=8.0, R=15.0, t0=80, dt=0.1):
    yr = np.zeros(T)
    n_turn = int(round((math.pi / 2) / (v / R * dt)))          # sweep ~90 deg
    yr[t0:t0 + n_turn] = v / R
    return _integrate(v, yr, T)


# ----------------------------------------------- (a) vocab widths pinned -----
def test_index_maps_match_the_model_widths():
    assert len(VL.LAT_TOKENS) + 1 == N_LAT          # +1 masked sentinel
    assert len(VL.LON_TOKENS) + 1 == N_LON
    assert len(VL.DIST_TOKENS) == N_DIST            # d_unknown IS a token, masked
    assert max(VL.LAT_IX.values()) < N_LAT and max(VL.LON_IX.values()) < N_LON
    assert set(VL.STRAT_SCALAR_NAMES) == {"ttm", "curv_3s", "curv_5s", "tspeed_5s"}
    assert VL.LON_IX["free_cruise"] == 0            # a real, trainable class
    assert VL.LAT_IX["lane_keep"] == 0


# ------------------------------------------------- (b/c) kinematic labels ----
def test_cruise_mints_lanekeep_freecruise_in_range():
    ep = VL.mint_episode(_cruise())
    n = ep["lat_target"].numel()
    assert n == 140 - VL.WINDOW - VL.MAX_HORIZON
    for name, hi in (("lat_target", N_LAT), ("lon_target", N_LON),
                     ("dist_target", N_DIST)):
        t = ep[name]
        valid = t[t != IGNORE_INDEX]
        assert valid.numel() > 0, name
        assert int(valid.min()) >= 0 and int(valid.max()) < hi, name
    # a constant cruise is lane_keep + free_cruise on every judged window
    lat_v = ep["lat_target"][ep["lat_target"] != IGNORE_INDEX]
    lon_v = ep["lon_target"][ep["lon_target"] != IGNORE_INDEX]
    assert (lat_v == VL.LAT_IX["lane_keep"]).all()
    assert (lon_v == VL.LON_IX["free_cruise"]).all()


def test_stop_trajectory_mints_stop_at_point():
    ep = VL.mint_episode(_stop())
    lon = ep["lon_target"]
    assert (lon == VL.LON_IX["stop_at_point"]).any(), "braking-to-stop must mint it"
    # and the stop distance band is a real (non-unknown) band on those windows
    sd = ep["stop_dist_target"]
    assert (sd != IGNORE_INDEX).any()


def test_junction_turn_mints_route_token_and_ttm():
    ep = VL.mint_episode(_left_turn())
    # a tight transient left sweep is a route event -> a v3 token beyond `follow`
    tok = ep["route_token"]
    minted = tok[tok != IGNORE_INDEX]
    assert minted.numel() > 0
    assert (minted == VL.ROUTE_V3_IX["turn_left"]).any()
    # time-to-maneuver is valid on the windows that have the turn ahead
    ttm_ok = ep["strat_scalar_mask"][:, 0]
    assert bool(ttm_ok.any()), "ttm must be valid where a maneuver is in range"
    assert float(ep["strat_scalars"][ttm_ok, 0].min()) > 0.0


def test_masking_is_by_horizon_not_a_wrong_class():
    # the 5 s scalars must be MASKED near the clip end (no 5 s of future), never
    # regressed against a fabricated value — this is the whole IGNORE discipline
    ep = VL.mint_episode(_cruise(T=140))
    m = ep["strat_scalar_mask"]
    curv5, tspeed = m[:, 2], m[:, 3]
    assert bool(curv5.any()) and not bool(curv5.all()), "curv@5s must be partly masked"
    assert bool(tspeed.any()) and not bool(tspeed.all())


def test_mint_window_matches_mint_episode_row():
    poses = _left_turn()
    ep = VL.mint_episode(poses)
    j = 20
    last = VL.WINDOW - 1 + j
    w = VL.mint_window(poses, last)
    for k in ("lat_target", "lon_target", "dist_target", "route", "route_token"):
        assert int(w[k]) == int(ep[k][j]), k
    assert torch.equal(w["strat_scalar_mask"], ep["strat_scalar_mask"][j])
    assert torch.allclose(w["strat_scalars"], ep["strat_scalars"][j], atol=1e-5)


def test_empty_episode_returns_aligned_empties():
    short = _cruise(T=VL.WINDOW + VL.MAX_HORIZON)      # n = 0
    ep = VL.mint_episode(short)
    assert ep["lat_target"].numel() == 0
    assert ep["strat_scalars"].shape == (0, 4)


# ----------------------------------------------------------- (d) losses ------
def test_factorised_ce_nonzero_and_grads_on_real_targets():
    ep = VL.mint_episode(_stop())
    b = 16
    lat = ep["lat_target"][:b]
    lon = ep["lon_target"][:b]
    dist = ep["dist_target"][:b]
    logits = {"lat_logits": torch.randn(b, N_LAT, requires_grad=True),
              "lon_logits": torch.randn(b, N_LON, requires_grad=True),
              "dist_logits": torch.randn(b, N_DIST, requires_grad=True)}
    loss, log = factorised_ce(logits, lat, lon, dist)
    assert loss.item() > 0.0
    assert log["lat_ce"] > 0.0 and log["lon_ce"] > 0.0
    loss.backward()
    assert logits["lat_logits"].grad.abs().sum() > 0


def test_factorised_ce_all_masked_is_zero_no_grad():
    b = 8
    ig = torch.full((b,), IGNORE_INDEX)
    logits = {"lat_logits": torch.randn(b, N_LAT, requires_grad=True),
              "lon_logits": torch.randn(b, N_LON, requires_grad=True),
              "dist_logits": torch.randn(b, N_DIST, requires_grad=True)}
    loss, log = factorised_ce(logits, ig, ig, ig)
    assert loss.item() == 0.0
    assert log["lat_ce"] == 0.0
    loss.backward()                                  # stays in the graph, 0 grad
    assert float(logits["lat_logits"].grad.abs().sum()) == 0.0


def test_strategic_scalar_loss_masks_and_trains():
    b = 12
    pred = torch.randn(b, 4, requires_grad=True)
    target = torch.randn(b, 4)
    mask = torch.zeros(b, 4, dtype=torch.bool)
    mask[:6, 0] = True                               # only ttm, only half the rows
    loss, log = strategic_scalar_loss(pred, target, mask)
    assert loss.item() > 0.0
    assert log["strat_scalar_cov"][0] == 0.5 and log["strat_scalar_cov"][1] == 0.0
    loss.backward()
    g = pred.grad.abs().sum(dim=0)
    assert float(g[0]) > 0.0 and float(g[1]) == 0.0  # grad only where supervised

    pred2 = torch.randn(b, 4, requires_grad=True)
    loss0, _ = strategic_scalar_loss(pred2, target, torch.zeros_like(mask))
    assert loss0.item() == 0.0


# ----------------------------------------------------------- head ------------
def test_goal_scalar_head_shape_and_size():
    head = GoalScalarHead(GoalScalarConfig(in_dim=128))
    out = head(torch.randn(5, 128))
    assert out.shape == (5, 4)
    assert 12_000 <= param_count(head) <= 25_000     # ~17 k, §3.1 order


# ----------------------------------------------------- (e) dataset wiring ----
def _toy_eps(n=3):
    from tanitad.data.toy_driving import generate_episode
    return [generate_episode(i, steps=80, size=32) for i in range(n)]


def test_dataset_emits_v4_keys_and_is_additive():
    from flagship_v4_data import FlagshipV4Dataset
    from train_flagship4b import FlagshipWindowDataset

    eps = _toy_eps()
    kw = dict(window=8, max_horizon=VL.MAX_HORIZON, maneuver_h=20, channels=1)
    v4 = FlagshipV4Dataset(eps, **kw)
    base = FlagshipWindowDataset(eps, **kw)
    it, b0 = v4[3], base[3]
    # new keys present with the right dtype/shape
    for k in ("lat_target", "lon_target", "dist_target", "route", "vt_band"):
        assert it[k].dtype == torch.long and it[k].ndim == 0, k
    assert it["strat_scalars"].shape == (4,)
    assert it["strat_scalar_mask"].shape == (4,) and it["strat_scalar_mask"].dtype == torch.bool
    assert it["route_graded"].dtype == torch.float32
    # additive: every shared key is byte-identical to the base dataset
    for k, v in b0.items():
        assert k in it
        if torch.is_tensor(v):
            assert torch.equal(it[k], v), k


def test_dataset_collates_into_a_batch():
    from torch.utils.data import default_collate
    from flagship_v4_data import FlagshipV4Dataset
    eps = _toy_eps()
    ds = FlagshipV4Dataset(eps, window=8, max_horizon=VL.MAX_HORIZON,
                           maneuver_h=20, channels=1)
    batch = default_collate([ds[i] for i in range(4)])
    assert batch["lat_target"].shape == (4,)
    assert batch["strat_scalars"].shape == (4, 4)
    assert batch["strat_scalar_mask"].shape == (4, 4)


def test_mintability_report_names_the_gaps():
    r = VL.mintability_report()
    gaps = r["not_mintable_needs_data"]
    assert any("lead_state" in v for v in gaps.values())
    assert any("MAP" in v or "map" in v for v in gaps.values())
