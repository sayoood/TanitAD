"""T3 — the ``blind_conditioning_baseline`` firewall.

The test that matters is the FIRST one: fed the real ``refb_labels`` derivation
(``route_target = _NAV_TO_ROUTE[nav_cmd]``) with ``nav_cmd`` as the only
context, the firewall must return ``CIRCULAR``. That is the check which would
have caught the strategic route head before it reached a shipped checkpoint and
was reported as ``load_bearing: true`` for months.

The rest pin that the check DISCRIMINATES — a clean target must come back
``CLEAN``, a partially-informative one ``LEAKY`` — and that the registry cannot
be talked into accepting a circular problem.
"""
from __future__ import annotations

import numpy as np
import pytest

import refb_labels as rl

from taniteval import blind_baseline as BB


def _eids(n, n_ep=12):
    return [f"ep{i % n_ep:03d}" for i in range(n)]


def _nav(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([rl.NAV_FOLLOW, rl.NAV_LEFT, rl.NAV_RIGHT],
                      size=n, p=[0.6, 0.22, 0.18])


# ========================================================================== #
# 1. THE POST-MORTEM — the real defect, caught in CPU-seconds                  #
# ========================================================================== #
def test_the_real_route_target_is_caught_as_CIRCULAR():
    """``refb_labels.route_target(nav_cmd)`` fed its own input. Blind = 1.0."""
    n = 900
    nav = _nav(n)
    y = np.array([rl.route_target(int(c)) for c in nav])       # the shipped map
    fw = BB.blind_conditioning_baseline(
        {"nav_cmd": nav}, y, _eids(n), problem="v1_route_target", n_boot=200)
    assert fw["verdict"] == "CIRCULAR"
    assert fw["admissible"] is False
    assert fw["target_is_deterministic_in_context"] is True
    assert fw["blind_accuracy"]["mean"] >= 1.0 - BB.DETERMINISTIC_EPS
    # and the linear probe alone already sees it, because context is one-hotted
    assert fw["blind_accuracy_linear_probe"]["mean"] >= 0.95
    assert "route_target = _NAV_TO_ROUTE[nav_cmd]" in fw["_read"]


def test_v21_route_target_is_ALSO_circular_given_the_fed_command():
    """The correction the 4-Brain brief needed: swapping the LABELER does not
    break the echo. ``nav_command_v21`` and ``route_target_v21`` come from the
    same ``route_from_future_v21`` call, and ``_ROUTE_TO_NAV`` is a bijection,
    so on every CE-eligible window the target is still a lookup of the input.
    MEASURED on 17 100 real PhysicalAI windows: echo 1.0000 for v1, v2 AND
    v2.1."""
    n = 600
    rng = np.random.default_rng(3)
    route = rng.choice([rl.ROUTE_LEFT, rl.ROUTE_STRAIGHT, rl.ROUTE_RIGHT],
                       size=n, p=[0.2, 0.6, 0.2])
    nav = np.array([rl._ROUTE_TO_NAV[int(r)] for r in route])   # v2.1 wiring
    fw = BB.blind_conditioning_baseline(
        {"nav_cmd": nav}, route, _eids(n), problem="v21_route_target",
        n_boot=200)
    assert fw["verdict"] == "CIRCULAR"


def test_vision_buys_nothing_is_also_CIRCULAR():
    """Second arm: the target is not deterministic in the context, but the
    image-using model does no better than the blind head."""
    n = 800
    rng = np.random.default_rng(1)
    nav = _nav(n, seed=1)
    y = np.array([c if rng.random() > 0.15 else rng.integers(0, 3)
                  for c in nav])
    real = nav.copy()                       # the "real" model just echoes too
    fw = BB.blind_conditioning_baseline(
        {"nav_cmd": nav}, y, _eids(n), real_pred=real,
        problem="echo_with_noise", n_boot=200)
    assert fw["vision_buys_nothing"] is True
    assert fw["verdict"] == "CIRCULAR"
    assert "vision_gain_over_blind" in fw


# ========================================================================== #
# 2. it DISCRIMINATES — a clean problem passes                                 #
# ========================================================================== #
def test_a_target_independent_of_context_is_CLEAN():
    n = 900
    # NOTE the DISTINCT seeds. Drawing `nav` and `y` from generators seeded
    # alike makes them deterministic functions of the SAME uniform stream, and
    # the firewall correctly calls that LEAKY — which is how this fixture was
    # first written, and the check caught it. Independence has to be built, not
    # assumed.
    rng = np.random.default_rng(9002)
    nav = _nav(n, seed=2)
    y = rng.choice([0, 1, 2], size=n, p=[0.2, 0.6, 0.2])    # independent of nav
    fw = BB.blind_conditioning_baseline(
        {"nav_cmd": nav}, y, _eids(n), problem="clean", n_boot=200)
    assert fw["verdict"] == "CLEAN"
    assert fw["admissible"] is True
    assert fw["blind_skill_over_majority"] < BB.SKILL_EPS


def test_a_partially_informative_context_is_LEAKY_not_circular():
    n = 1200
    rng = np.random.default_rng(4)
    nav = _nav(n, seed=4)
    # the context explains ~half the label; the rest is genuinely unpredictable
    y = np.where(rng.random(n) < 0.5, nav,
                 rng.choice([0, 1, 2], size=n, p=[0.2, 0.6, 0.2]))
    fw = BB.blind_conditioning_baseline(
        {"nav_cmd": nav}, y, _eids(n), problem="leaky", n_boot=200)
    assert fw["verdict"] == "LEAKY"
    assert fw["admissible"] is True          # usable, but only as a SKILL delta
    assert fw["context_leaks"] is True


def test_multiple_context_fields_and_continuous_are_accepted():
    n = 600
    rng = np.random.default_rng(5)
    nav = _nav(n, seed=5)
    fw = BB.blind_conditioning_baseline(
        {"nav_cmd": nav,
         "maneuver": rng.integers(0, 5, size=n),
         "speed": rng.normal(12.0, 3.0, size=n)},
        np.array([rl.route_target(int(c)) for c in nav]), _eids(n),
        problem="multi", n_boot=200)
    assert fw["verdict"] == "CIRCULAR"       # the leak survives extra features
    assert set(fw["context_fields"]) == {"nav_cmd", "maneuver", "speed"}


# ========================================================================== #
# 3. methodology guards                                                        #
# ========================================================================== #
def test_split_is_episode_clustered():
    n = 400
    fw = BB.blind_conditioning_baseline(
        {"nav_cmd": _nav(n, 6)}, np.zeros(n, dtype=int), _eids(n),
        problem="x", n_boot=100)
    assert fw["split"]["unit"] == "episode"
    assert fw["estimator"]["interval"] == "episode_cluster_bootstrap"
    assert fw["estimator"]["resampling_unit"] == "episode"


def test_single_episode_is_refused():
    with pytest.raises(ValueError, match="within-episode split"):
        BB.blind_conditioning_baseline({"nav_cmd": _nav(50)},
                                       np.zeros(50, dtype=int), ["e"] * 50)


def test_empty_context_is_refused():
    with pytest.raises(ValueError, match="at least one context field"):
        BB.blind_conditioning_baseline({}, np.zeros(50, dtype=int), _eids(50))


def test_categorical_context_is_one_hot_not_integer():
    """An integer-coded lookup leak is invisible to a linear probe on the raw
    integer — the encoding IS the check."""
    m, names = BB.build_context_matrix({"nav_cmd": np.array([0, 1, 2, 0])})
    assert m.shape == (4, 3) and names == ["nav_cmd"] * 3
    m2, _ = BB.build_context_matrix({"v": np.array([1.5, 2.5])})
    assert m2.shape == (2, 1)                       # continuous stays scalar


# ========================================================================== #
# 4. the registry cannot be talked into a circular problem                     #
# ========================================================================== #
def test_registry_refuses_a_circular_problem():
    n = 600
    nav = _nav(n, 7)
    fw = BB.blind_conditioning_baseline(
        {"nav_cmd": nav}, np.array([rl.route_target(int(c)) for c in nav]),
        _eids(n), problem="v1_route", n_boot=200)
    with pytest.raises(BB.CircularTarget, match="CIRCULAR"):
        BB.register_decision_problem("v1_route", target="route_target",
                                     conditioning=["nav_cmd"], firewall=fw)
    assert "v1_route" not in BB.DECISION_PROBLEMS


def test_registry_refuses_a_problem_with_no_firewall_at_all():
    for bad in (None, {}, {"verdict": "CLEAN"}, "passed"):
        with pytest.raises(BB.CircularTarget, match="blind_conditioning"):
            BB.register_decision_problem("nofw", target="t",
                                         conditioning=["c"], firewall=bad)


def test_registry_accepts_a_clean_problem_and_flags_a_leaky_one():
    n = 900
    rng = np.random.default_rng(9008)          # distinct stream from _nav(n, 8)
    nav = _nav(n, 8)
    clean = rng.choice([0, 1, 2], size=n, p=[0.2, 0.6, 0.2])
    fw = BB.blind_conditioning_baseline({"nav_cmd": nav}, clean, _eids(n),
                                        problem="ok", n_boot=200)
    rec = BB.register_decision_problem("ok", target="t", conditioning=["nav"],
                                       firewall=fw, owner="tester")
    assert rec["verdict"] == "CLEAN"
    assert rec["must_report_skill_over_blind"] is False
    assert BB.assert_registered("ok") is rec

    leaky_y = np.where(rng.random(n) < 0.5, nav, clean)
    fw2 = BB.blind_conditioning_baseline({"nav_cmd": nav}, leaky_y, _eids(n),
                                         problem="leaky2", n_boot=200)
    rec2 = BB.register_decision_problem("leaky2", target="t",
                                        conditioning=["nav"], firewall=fw2)
    assert rec2["must_report_skill_over_blind"] is True


def test_assert_registered_fails_for_an_unfirewalled_problem():
    with pytest.raises(BB.CircularTarget, match="not registered"):
        BB.assert_registered("never_checked")
