"""Offline tests for the Traffic-Light / signalized-intersection scenario (SC-14, W-03 family).

These validate the *scenario generator* and its wiring into the real TLC metric (no simulator).
Asserted:

1. **Telemetry contract** — the emitted log has exactly the ScenarioTelemetry field names, correct
   lengths/shapes, per-timestep signal state, and the scenario-specific compliance signals.
2. **Fairness guard** — the geometry is not a dilemma zone: a comfortable stop is always feasible,
   so a policy that enters on red did so by choice, not necessity.
3. **Discriminative structure** — through the real ``traffic_light_metrics``: a rule-barrier scores
   TLC=1 on both red (clean stop) and green (smooth proceed); a soft prior scores 0 on red (runs it)
   and < the barrier on green (phantom braking).
4. **The barrier property** — the rule barrier's red-run rate is invariant to the cross-clearance
   temptation (stays 0), while the soft prior's line-crossing speed grows with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# stack/ is importable as the package root in this test suite (see conftest / existing tests).
from tanitad.eval.metrics import (
    SIGNAL_GREEN,
    SIGNAL_RED,
    SIGNAL_YELLOW,
    ScenarioTelemetry,
    traffic_light_metrics,
)
from tanitad.eval.scenarios.traffic_light import (
    POLICIES,
    SIGNAL_PLANS,
    TELEMETRY_KEYS,
    TrafficLightScenario,
    red_run_rate,
    simulate_policy,
)


def _telemetry_from(log: dict) -> ScenarioTelemetry:
    return ScenarioTelemetry(**{k: v for k, v in log.items() if not k.startswith("_")})


def _score(log: dict, model_name="m") -> dict:
    ex = log["_extra"]
    return traffic_light_metrics(_telemetry_from(log), ex["ego_s"], ex["signal_state"],
                                 ex["stopline_s"], model_name=model_name)


# --------------------------------------------------------------------------- #
# 1. Telemetry contract                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("plan", SIGNAL_PLANS)
@pytest.mark.parametrize("policy", POLICIES)
def test_telemetry_contract(policy, plan):
    sc = TrafficLightScenario(signal_plan=plan)
    tel = simulate_policy(sc, policy=policy)
    for key in TELEMETRY_KEYS:
        assert key in tel, f"missing telemetry key {key!r}"
    T = sc.steps
    for key in ("ego_v", "ego_jerk", "steer_rate", "latency_ms", "hazard_los_flag",
                "dist_to_blind_spot", "is_occluded_flag"):
        assert len(tel[key]) == T, f"{key} wrong length"
    assert np.asarray(tel["wm_hazard_xy"]).shape == (T, 2)
    assert np.asarray(tel["gt_hazard_xy"]).shape == (T, 2)
    ex = tel["_extra"]
    for k in ("signal_state", "ego_s", "stopline_s", "signal_plan", "entered_on_red",
              "stop_margin_m", "min_speed_at_stopline", "phantom_brake", "cross_clearance_m"):
        assert k in ex, f"missing _extra key {k!r}"
    assert len(ex["signal_state"]) == T
    assert len(ex["ego_s"]) == T


def test_carla_recipe_shape():
    rec = TrafficLightScenario().carla_recipe()
    assert rec["camera"] == {"channels": 6, "size": 256, "stack": 2}   # base250cam contract
    prop_types = {p["type"] for p in rec["props"]}
    assert any("traffic_light" in p for p in prop_types)
    assert any("stop_line" in p for p in prop_types)
    assert rec["signal_plan"]["plan"] == "red"
    assert rec["success"]["no_red_entry"] is True


def test_signal_state_phases():
    # red plan: green then yellow then red, in that order; green plan: stale green throughout.
    sig_r = TrafficLightScenario(signal_plan="red").signal_state()
    assert sig_r[0] == SIGNAL_GREEN
    assert (sig_r == SIGNAL_YELLOW).any()
    assert sig_r[-1] == SIGNAL_RED
    # the phases are contiguous and ordered G < Y < R by first-occurrence index
    g0 = int(np.argmax(sig_r == SIGNAL_GREEN))
    y0 = int(np.argmax(sig_r == SIGNAL_YELLOW))
    r0 = int(np.argmax(sig_r == SIGNAL_RED))
    assert g0 < y0 < r0
    sig_g = TrafficLightScenario(signal_plan="green").signal_state()
    assert (sig_g == SIGNAL_GREEN).all()


def test_invalid_signal_plan_raises():
    with pytest.raises(ValueError):
        TrafficLightScenario(signal_plan="flashing")


# --------------------------------------------------------------------------- #
# 2. Fairness guard: not a dilemma zone                                        #
# --------------------------------------------------------------------------- #
def test_comfortable_stop_is_feasible():
    assert TrafficLightScenario(signal_plan="red").comfortable_stop_feasible() is True


# --------------------------------------------------------------------------- #
# 3. Discriminative structure through the real TLC metric                      #
# --------------------------------------------------------------------------- #
def test_red_barrier_stops_soft_prior_runs_it():
    sc = TrafficLightScenario(signal_plan="red")
    barrier = simulate_policy(sc, "rule_barrier")
    soft = simulate_policy(sc, "soft_prior")
    # barrier: never enters on red, halts before the line at a comfortable margin
    assert barrier["_extra"]["entered_on_red"] is False
    assert barrier["_extra"]["stop_margin_m"] >= 0.0
    assert float(np.min(barrier["ego_v"])) <= 1e-3          # comes to a full stop
    # soft prior: rolls through the red at speed
    assert soft["_extra"]["entered_on_red"] is True
    assert soft["_extra"]["min_speed_at_stopline"] > 0.5


def test_tlc_barrier_perfect_soft_prior_fails_on_red():
    sc = TrafficLightScenario(signal_plan="red")
    barrier = _score(simulate_policy(sc, "rule_barrier"))
    soft = _score(simulate_policy(sc, "soft_prior"))
    assert barrier["TLC"] == pytest.approx(1.0)
    assert soft["TLC"] == 0.0                                # running a red zeroes TLC
    assert barrier["TLC"] > soft["TLC"]


def test_tlc_barrier_smooth_soft_prior_phantom_brakes_on_green():
    sc = TrafficLightScenario(signal_plan="green")
    barrier = _score(simulate_policy(sc, "rule_barrier"))
    soft = _score(simulate_policy(sc, "soft_prior"))
    assert barrier["TLC"] == pytest.approx(1.0)              # smooth proceed, no phantom brake
    assert soft["TLC"] < barrier["TLC"]                      # phantom braking penalized
    assert simulate_policy(sc, "soft_prior")["_extra"]["phantom_brake"] is True
    assert simulate_policy(sc, "rule_barrier")["_extra"]["phantom_brake"] is False


def test_barrier_carries_less_energy_toward_the_line_on_red():
    sc = TrafficLightScenario(signal_plan="red")
    # OKRI is "lower safer": a stopped barrier carries less kinetic energy toward the stop line
    # than a policy that rolls the red at speed.
    assert _score(simulate_policy(sc, "rule_barrier"))["OKRI"] < _score(
        simulate_policy(sc, "soft_prior"))["OKRI"]


def test_no_occlusion_object_permanence_na():
    # traffic-light is a rule-compliance scenario, not an occluder: no hidden agent is claimed.
    for policy in POLICIES:
        tel = simulate_policy(TrafficLightScenario(), policy)
        assert not np.asarray(tel["is_occluded_flag"]).any()
        assert not np.isfinite(np.asarray(tel["wm_hazard_xy"])).any()


# --------------------------------------------------------------------------- #
# 4. The barrier property: red-run rate vs the cross-clearance temptation       #
# --------------------------------------------------------------------------- #
def test_red_run_rate_barrier_zero_soft_prior_positive():
    assert red_run_rate("rule_barrier") == 0.0
    assert red_run_rate("soft_prior") > 0.0


def test_barrier_invariant_to_temptation_soft_prior_is_not():
    low = TrafficLightScenario(signal_plan="red", cross_clearance_m=0.0)
    high = TrafficLightScenario(signal_plan="red", cross_clearance_m=12.0)
    # barrier: never enters on red at either extreme
    assert simulate_policy(low, "rule_barrier")["_extra"]["entered_on_red"] is False
    assert simulate_policy(high, "rule_barrier")["_extra"]["entered_on_red"] is False
    # soft prior: rolls through the red faster as the crossing looks clearer
    s_low = simulate_policy(low, "soft_prior")["_extra"]["min_speed_at_stopline"]
    s_high = simulate_policy(high, "soft_prior")["_extra"]["min_speed_at_stopline"]
    assert s_high > s_low


def test_deterministic():
    sc = TrafficLightScenario(signal_plan="red")
    a = simulate_policy(sc, "rule_barrier")
    b = simulate_policy(sc, "rule_barrier")
    assert np.allclose(a["ego_v"], b["ego_v"])
    assert np.allclose(a["_extra"]["ego_s"], b["_extra"]["ego_s"])
