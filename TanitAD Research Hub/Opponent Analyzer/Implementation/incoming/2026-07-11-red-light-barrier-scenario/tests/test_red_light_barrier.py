"""Offline tests for the Red-light barrier scenario feed (W-03 / SC-14).

These validate the *scenario generator*, not the metrics (those live in the Benchmarks & Eval
suite). Asserted:

1. **Telemetry contract** — the emitted log has exactly the ScenarioTelemetry field names, correct
   lengths/shapes, and a scenario-specific ``red_light_violation`` compliance signal.
2. **Discriminative structure** — the scenario separates a soft-prior baseline from a rule-barrier
   policy along the axis the Waymo-Dallas and Tesla EA26002 red-light dockets expose: a hard,
   discrete signal-phase rule must be obeyed regardless of the apparent clearance.
3. **The barrier property (the point of the scenario)** — the rule barrier's violation rate is
   **invariant to the apparent clearance** (stays 0), while the soft prior's violation rate **grows
   with it**. That is the mechanistic difference between a barrier term (H9) and a soft learned prior.

Deliberately mirrors ``test_stop_arm_gate.py`` (SC-14 reuses the SC-04 barrier oracle). No simulator,
no cross-package import — numpy + pytest only. Run:
    pytest "TanitAD Research Hub/Opponent Analyzer/Implementation/incoming/2026-07-11-red-light-barrier-scenario/tests"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from red_light_barrier import (  # noqa: E402
    POLICIES,
    TELEMETRY_KEYS,
    RedLightBarrierScenario,
    simulate_policy,
    violation_rate,
)

APPROACH_M = 30.0          # matches OKRI_APPROACH_M in the metric suite


def _okri_like(tel: dict) -> float:
    """Crude OKRI proxy (same shape as compute_okri) for a relative comparison."""
    d = np.asarray(tel["dist_to_blind_spot"])
    v = np.asarray(tel["ego_v"])
    mask = d < APPROACH_M
    ke = 0.5 * tel["ego_mass_kg"] * v ** 2
    risk = np.where(mask, ke / (d + 0.1), 0.0)
    return float(np.sum(risk) * tel["dt"])


# --------------------------------------------------------------------------- #
# 1. Telemetry contract                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("policy", POLICIES)
def test_telemetry_contract(policy):
    sc = RedLightBarrierScenario()
    tel = simulate_policy(sc, policy=policy)
    for key in TELEMETRY_KEYS:
        assert key in tel, f"missing telemetry key {key!r}"
    T = sc.steps
    for key in ("ego_v", "ego_jerk", "steer_rate", "latency_ms", "hazard_los_flag",
                "dist_to_blind_spot", "is_occluded_flag"):
        assert len(tel[key]) == T, f"{key} wrong length"
    assert np.asarray(tel["wm_hazard_xy"]).shape == (T, 2)
    assert np.asarray(tel["gt_hazard_xy"]).shape == (T, 2)
    for k in ("red_light_violation", "stop_distance_m", "passed_stop_line",
              "min_speed_at_stopline", "cross_clearance_m"):
        assert k in tel["_extra"], f"missing _extra key {k!r}"


def test_carla_recipe_shape():
    rec = RedLightBarrierScenario().carla_recipe()
    assert rec["camera"] == {"channels": 6, "size": 256, "stack": 2}  # base250cam contract
    prop_types = {p["type"] for p in rec["props"]}
    assert any("traffic_light" in p for p in prop_types)
    assert any("stop_line" in p for p in prop_types)
    assert rec["props"][0]["phase"] == "red"
    assert rec["actors"][0]["occluded_until_s"] == RedLightBarrierScenario().crosser_los_s
    assert rec["success"]["full_stop_before_line"] is True


def test_scenario_has_a_stopline_and_occluded_crosser():
    tel = simulate_policy(RedLightBarrierScenario(), policy="soft_prior")
    assert np.asarray(tel["is_occluded_flag"]).any(), "no occluded phase"
    assert np.asarray(tel["hazard_los_flag"]).any(), "crosser never reaches LoS"
    assert float(np.min(tel["dist_to_blind_spot"])) < APPROACH_M, "ego never nears the stop line"


# --------------------------------------------------------------------------- #
# 2. Discriminative structure (rule_barrier obeys, soft_prior violates)        #
# --------------------------------------------------------------------------- #
def test_rule_barrier_stops_soft_prior_runs_red():
    sc = RedLightBarrierScenario()
    barrier = simulate_policy(sc, "rule_barrier")
    soft = simulate_policy(sc, "soft_prior")
    assert barrier["_extra"]["red_light_violation"] is False
    assert barrier["_extra"]["min_speed_at_stopline"] < 0.5
    assert barrier["_extra"]["stop_distance_m"] >= 0.0
    assert soft["_extra"]["red_light_violation"] is True
    assert soft["_extra"]["min_speed_at_stopline"] > 0.5
    assert soft["_extra"]["passed_stop_line"] is True


def test_rule_barrier_comes_to_full_stop():
    tel = simulate_policy(RedLightBarrierScenario(), "rule_barrier")
    assert float(np.min(tel["ego_v"])) <= 1e-3, "rule barrier never reaches a full stop"


def test_barrier_carries_less_energy_toward_the_crosser():
    sc = RedLightBarrierScenario()
    assert _okri_like(simulate_policy(sc, "rule_barrier")) < _okri_like(
        simulate_policy(sc, "soft_prior"))


def test_object_permanence_only_for_barrier():
    sc = RedLightBarrierScenario()
    barrier = simulate_policy(sc, "rule_barrier")
    soft = simulate_policy(sc, "soft_prior")
    occ = np.asarray(barrier["is_occluded_flag"])
    assert np.isfinite(np.asarray(barrier["wm_hazard_xy"])[occ]).all()
    assert not np.isfinite(np.asarray(soft["wm_hazard_xy"])[occ]).any()


# --------------------------------------------------------------------------- #
# 3. The barrier property: violation-rate vs the apparent clearance            #
# --------------------------------------------------------------------------- #
def test_violation_rate_barrier_is_zero_soft_prior_positive():
    assert violation_rate("rule_barrier") == 0.0
    assert violation_rate("soft_prior") > 0.0


def test_barrier_invariant_to_clearance_soft_prior_is_not():
    """A hard phase barrier stops regardless of how clear the intersection looks; a soft prior is
    temptable."""
    low = RedLightBarrierScenario(cross_clearance_m=0.0)
    high = RedLightBarrierScenario(cross_clearance_m=12.0)
    assert simulate_policy(low, "rule_barrier")["_extra"]["red_light_violation"] is False
    assert simulate_policy(high, "rule_barrier")["_extra"]["red_light_violation"] is False
    b_low = simulate_policy(low, "rule_barrier")["_extra"]["min_speed_at_stopline"]
    b_high = simulate_policy(high, "rule_barrier")["_extra"]["min_speed_at_stopline"]
    assert abs(b_high - b_low) < 1e-6, "barrier speed-at-line must not depend on clearance"
    s_low = simulate_policy(low, "soft_prior")["_extra"]["min_speed_at_stopline"]
    s_high = simulate_policy(high, "soft_prior")["_extra"]["min_speed_at_stopline"]
    assert s_high > s_low, "soft prior should be more tempted by a clearer intersection"


def test_deterministic():
    sc = RedLightBarrierScenario()
    a = simulate_policy(sc, "rule_barrier")
    b = simulate_policy(sc, "rule_barrier")
    assert np.allclose(a["ego_v"], b["ego_v"])
    assert np.allclose(np.nan_to_num(a["wm_hazard_xy"]), np.nan_to_num(b["wm_hazard_xy"]))
