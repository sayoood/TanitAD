"""Tests for the S3 target / miner / metric / firewall.

The two that matter most and are the reason this file exists:

  * ``test_target_never_reads_the_observed_window`` -- the NON-CIRCULARITY
    proof, asserted in code. Mutate everything the model can see; every S3
    target must be bit-identical. A refactor that lets the observed window into
    the target fails here, not in an audit months later.
  * ``test_synthetic_echo_label_is_REFUSED`` -- the firewall's own regression
    test (spec §7 item 3): a label that IS a conditioning channel must be
    refused. Without it the firewall could silently degrade into a no-op.

Run:  python -m pytest test_s3_labels.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s3_labels as S3                                          # noqa: E402
from s3_blind_baseline import (blind_conditioning_baseline,     # noqa: E402
                               refusal_verdict)


# ---------------------------------------------------------------------------
# synthetic tracks
# ---------------------------------------------------------------------------
def straight_track(T=200, v=10.0):
    """Constant-speed straight line -> no lateral and no longitudinal event."""
    p = torch.zeros(T, 4)
    p[:, 0] = torch.arange(T, dtype=torch.float32) * v * S3.DT
    p[:, 3] = v
    return p


def turn_at(T=200, v=10.0, t_turn=80, sweep_deg=90.0, n_turn=40):
    """Straight, then a junction-scale turn starting at index ``t_turn``."""
    dyaw = np.deg2rad(sweep_deg) / n_turn
    yaw, x, y = 0.0, 0.0, 0.0
    p = torch.zeros(T, 4)
    for i in range(T):
        p[i] = torch.tensor([x, y, yaw, v])
        if t_turn <= i < t_turn + n_turn:
            yaw += dyaw
        x += v * S3.DT * float(np.cos(yaw))
        y += v * S3.DT * float(np.sin(yaw))
    return p


def brake_at(T=200, v0=14.0, t_brake=80, a=-1.5, n=40):
    """Constant speed, then a sustained deceleration starting at ``t_brake``."""
    v = np.full(T, v0, dtype=np.float64)
    for i in range(t_brake, min(T, t_brake + n)):
        v[i] = max(2.0, v[i - 1] + a * S3.DT)
    v[min(T, t_brake + n):] = v[min(T - 1, t_brake + n - 1)]
    p = torch.zeros(T, 4)
    p[:, 3] = torch.tensor(v, dtype=torch.float32)
    p[:, 0] = torch.tensor(np.cumsum(v * S3.DT), dtype=torch.float32)
    return p


# ===========================================================================
# 1. NON-CIRCULARITY -- the load-bearing test
# ===========================================================================
def test_target_never_reads_the_observed_window():
    """Mutating poses[:L+1] must not move any S3 target."""
    p = turn_at()
    rng = np.random.default_rng(0)
    for L in (7, 30, 60):
        t_lat0, ok0 = S3.ttm_lateral(p, L)
        t_lon0, ok1 = S3.ttm_longitudinal(p, L)
        q = p.clone()
        q[:L + 1] = torch.tensor(rng.normal(0, 50, size=(L + 1, 4)),
                                 dtype=torch.float32)
        t_lat1, ok2 = S3.ttm_lateral(q, L)
        t_lon1, ok3 = S3.ttm_longitudinal(q, L)
        assert (ok0, ok1) == (ok2, ok3)
        assert np.allclose([t_lat0], [t_lat1], equal_nan=True)
        assert np.allclose([t_lon0], [t_lon1], equal_nan=True)


def test_target_DOES_move_when_the_future_moves():
    """The complement: the target must be sensitive to the future, or the
    disjointness test above would pass on a constant."""
    a, b = turn_at(t_turn=60), turn_at(t_turn=110)
    ta, oka = S3.ttm_lateral(a, 20)
    tb, okb = S3.ttm_lateral(b, 20)
    assert oka and not np.isnan(ta)
    assert (not okb) or tb > ta      # later turn -> later (or out of horizon)


# ===========================================================================
# 2. the target itself
# ===========================================================================
def test_lateral_target_finds_the_turn_and_times_it():
    p = turn_at(t_turn=80, n_turn=40)
    t, ok = S3.ttm_lateral(p, L=20, horizon_s=12.0)
    assert ok
    assert 4.0 < t < 8.0, t          # turn starts ~6 s after L=20


def test_longitudinal_target_finds_the_brake_and_times_it():
    p = brake_at(t_brake=80, n=40)
    t, ok = S3.ttm_longitudinal(p, L=20, horizon_s=12.0)
    assert ok
    assert 3.0 < t < 9.0, t


def test_straight_constant_speed_has_NO_event_on_either_axis():
    p = straight_track()
    assert not S3.ttm_lateral(p, 20)[1]
    assert not S3.ttm_longitudinal(p, 20)[1]
    assert S3.band_of(*S3.ttm_lateral(p, 20)) == S3.IX_NONE
    assert S3.band_of(*S3.ttm_longitudinal(p, 20)) == S3.IX_NONE


def test_axes_are_independent_a_turn_is_not_a_brake():
    """The single-axis rule, asserted: a pure turn must NOT mint a longitudinal
    event and a pure brake must NOT mint a lateral one. The v1 5-way softmax's
    failure was exactly this conflation."""
    turn, brake = turn_at(t_turn=80), brake_at(t_brake=80)
    assert S3.ttm_lateral(turn, 20)[1] and not S3.ttm_longitudinal(turn, 20)[1]
    assert S3.ttm_longitudinal(brake, 20)[1] and not S3.ttm_lateral(brake, 20)[1]


# ===========================================================================
# 3. the miner
# ===========================================================================
def test_M1_refuses_windows_that_cannot_observe_the_horizon():
    p = turn_at(T=200)
    rows = S3.mine_episode(p, "ep", horizon_s=12.0)
    h = int(12.0 / S3.DT)
    for r in rows:
        assert r["m1"] == (r["obs_h_steps"] >= h)
        if not r["m1"]:
            assert not r["lat_admissible"] and not r["lon_admissible"]


def test_M2_excludes_a_manoeuvre_already_under_way():
    """A window sitting INSIDE the turn must not be an admissible lateral
    decision point -- the answer would be visible in the observation."""
    p = turn_at(t_turn=40, n_turn=60)
    rows = {r["L"]: r for r in S3.mine_episode(p, "ep", horizon_s=8.0)}
    inside = rows.get(55)
    assert inside is not None and inside["m1"]
    assert not inside["lat_admissible"], "window inside the turn was admitted"


def test_M4_excludes_standstill():
    p = straight_track(v=0.05)
    rows = S3.mine_episode(p, "ep", horizon_s=8.0)
    assert all(not r["m4"] for r in rows)
    assert all(not r["lat_admissible"] for r in rows)


def test_no_censoring_every_admissible_window_resolves_the_full_horizon():
    p = turn_at()
    for r in S3.mine_episode(p, "ep", horizon_s=12.0):
        if r["lat_admissible"]:
            assert r["obs_h_steps"] >= 120


# ===========================================================================
# 4. the metric -- the route_acc=1.0 / route_skill=0.0 shape must be impossible
# ===========================================================================
def test_majority_class_predictor_scores_EXACTLY_zero_qwk():
    rng = np.random.default_rng(0)
    y = rng.choice(S3.N_BANDS, size=500, p=[.4, .25, .15, .1, .1])
    maj = int(np.bincount(y, minlength=S3.N_BANDS).argmax())
    k = S3.quadratic_weighted_kappa(y, np.full_like(y, maj))
    assert abs(k) < 1e-9, k


def test_perfect_predictor_scores_one():
    y = np.array([0, 1, 2, 3, 4, 0, 2, 4] * 10)
    assert S3.quadratic_weighted_kappa(y, y) == pytest.approx(1.0)


def test_qwk_is_ORDINAL_a_far_miss_costs_more_than_a_near_miss():
    y = np.array([0, 1, 2, 3, 4] * 40)
    near = (y + 1) % S3.N_BANDS
    far = (y + 2) % S3.N_BANDS
    assert S3.quadratic_weighted_kappa(y, near) > \
        S3.quadratic_weighted_kappa(y, far)


def test_band_metrics_always_ships_the_baseline_and_per_band_recall():
    y = np.array([0, 0, 0, 1, 2, 3, 4] * 20)
    m = S3.band_metrics(y, np.zeros_like(y))
    assert m["band_acc"] > 0.4 and m["qwk"] == pytest.approx(0.0, abs=1e-9)
    assert "majority_rate" in m and "per_band_recall" in m
    dead = [k for k, v in m["per_band_recall"].items() if v == 0.0]
    assert len(dead) == 4, "a dead band must be visible in per_band_recall"


def test_mae_skill_is_zero_for_the_median_constant():
    t = np.array([1.5, 2.0, 3.0, 7.0, 9.0])
    r = S3.mae_skill_s(t, np.full_like(t, np.median(t)))
    assert r["mae_skill_s"] == pytest.approx(0.0)


def test_early_late_bias_has_a_sign():
    t = np.array([2.0, 4.0, 6.0])
    assert S3.mae_skill_s(t, t + 1.0)["early_late_bias_s"] == pytest.approx(1.0)
    assert S3.mae_skill_s(t, t - 1.0)["early_late_bias_s"] == pytest.approx(-1.0)


def test_qwk_bootstrap_names_its_estimator_and_never_the_deprecated_one():
    rng = np.random.default_rng(1)
    y = rng.choice(S3.N_BANDS, size=300)
    p = np.where(rng.random(300) < 0.6, y, rng.choice(S3.N_BANDS, size=300))
    eid = [f"ep{i // 30}" for i in range(300)]
    out = S3.qwk_bootstrap(y, p, eid, n_boot=200)
    assert out["estimator"] == "episode_cluster_bootstrap"
    assert out["n_episodes"] == 10 and out["lo"] <= out["mean"] <= out["hi"]


# ===========================================================================
# 5. THE FIREWALL's own regression test (spec §7 item 3)
# ===========================================================================
def test_synthetic_echo_label_is_REFUSED():
    """A label that IS a conditioning channel must be refused.

    This is the ``route_target(nav_cmd) = _NAV_TO_ROUTE[nav_cmd]`` shape that
    cost months and produced ``route_skill = 0.0000`` by construction. If this
    test ever passes with ``REFUSED=False`` the firewall has become a no-op.
    """
    rng = np.random.default_rng(0)
    n = 1500
    ch = rng.integers(0, S3.N_BANDS, size=n)           # a conditioning channel
    X = np.stack([ch.astype(float), rng.normal(size=n)], 1)
    y = ch.copy()                                      # the target IS the input
    eid = [f"ep{i // 50}" for i in range(n)]
    r = blind_conditioning_baseline(X[:1000], y[:1000], X[1000:], y[1000:],
                                    eid[1000:], S3.N_BANDS,
                                    lambda a, b: S3.band_metrics(a, b),
                                    label="echo", epochs=600)
    v = refusal_verdict(r["blind"]["qwk"])
    assert v["REFUSED"], f"echo label was ADMITTED: {r['blind']}"


def test_pure_noise_label_is_NOT_refused():
    """The complement -- the firewall must not refuse a label a blind head
    cannot predict, or it would reject every admissible problem."""
    rng = np.random.default_rng(1)
    n = 1500
    X = rng.normal(size=(n, 4))
    y = rng.integers(0, S3.N_BANDS, size=n)
    eid = [f"ep{i // 50}" for i in range(n)]
    r = blind_conditioning_baseline(X[:1000], y[:1000], X[1000:], y[1000:],
                                    eid[1000:], S3.N_BANDS,
                                    lambda a, b: S3.band_metrics(a, b),
                                    label="noise", epochs=300)
    assert not refusal_verdict(r["blind"]["qwk"])["REFUSED"]


def test_blind_baseline_test_set_is_episode_disjoint_by_construction():
    """The helper carries the test eids so the caller can bootstrap them; a
    blind score on windows from a training episode is not a blind score."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    y = rng.integers(0, S3.N_BANDS, size=200)
    eid = [f"ep{i // 20}" for i in range(200)]
    r = blind_conditioning_baseline(X[:100], y[:100], X[100:], y[100:],
                                    eid[100:], S3.N_BANDS,
                                    lambda a, b: S3.band_metrics(a, b),
                                    epochs=50)
    assert r["n_test_episodes"] == 5
