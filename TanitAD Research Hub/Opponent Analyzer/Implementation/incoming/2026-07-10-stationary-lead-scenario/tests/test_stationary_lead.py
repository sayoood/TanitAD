"""Offline tests for the Stationary-object / same-lane lead scenario feed (W-08 / SC-13).

These validate the *scenario generator*, not the metrics (those live in the Benchmarks & Eval
suite). Asserted:

1. **Telemetry contract** — the emitted log has exactly the ScenarioTelemetry field names, correct
   lengths/shapes, and the scenario-specific ``_extra`` compliance/safety signals.
2. **Discriminative structure** — the scenario separates a detection-then-react baseline from an
   imagination-forward policy along the axis the Avride ODI exposes: braking for a stopped/slow
   lead must not wait on a class label.
3. **The mechanistic property (the point of the scenario)** — imagination's advantage is
   *structural* (acts on TTC before classification): its collision rate is 0 across the
   approach-speed sweep while the classifier-react baseline collides at higher speed; and the
   anticipation lead **vanishes when the competitor classifies early** (the honest falsifier).

No simulator, no cross-package import — numpy + pytest only. Run:
    pytest "TanitAD Research Hub/Opponent Analyzer/Implementation/incoming/2026-07-10-stationary-lead-scenario/tests"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stationary_lead import (  # noqa: E402
    APPROACH_M,
    POLICIES,
    TELEMETRY_KEYS,
    StationaryLeadScenario,
    brake_lead_time,
    collision_rate,
    simulate_policy,
)


def _okri_like(tel: dict) -> float:
    """Crude OKRI proxy (same shape as compute_okri) for a relative comparison."""
    d = np.asarray(tel["dist_to_blind_spot"])
    v = np.asarray(tel["ego_v"])
    mask = (d < APPROACH_M) & (d > -1.0)
    ke = 0.5 * tel["ego_mass_kg"] * v ** 2
    risk = np.where(mask, ke / (np.abs(d) + 0.1), 0.0)
    return float(np.sum(risk) * tel["dt"])


# --------------------------------------------------------------------------- #
# 1. Telemetry contract                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("policy", POLICIES)
def test_telemetry_contract(policy):
    sc = StationaryLeadScenario()
    tel = simulate_policy(sc, policy=policy)
    for key in TELEMETRY_KEYS:
        assert key in tel, f"missing telemetry key {key!r}"
    T = sc.steps
    for key in ("ego_v", "ego_jerk", "steer_rate", "latency_ms", "hazard_los_flag",
                "dist_to_blind_spot", "is_occluded_flag"):
        assert len(tel[key]) == T, f"{key} wrong length"
    assert np.asarray(tel["wm_hazard_xy"]).shape == (T, 2)
    assert np.asarray(tel["gt_hazard_xy"]).shape == (T, 2)
    for k in ("collision", "brake_onset_s", "min_ttc_s", "min_range_m",
              "okri_proxy", "peak_jerk", "v_cruise", "detect_range_m"):
        assert k in tel["_extra"], f"missing _extra key {k!r}"


def test_carla_recipe_shape():
    rec = StationaryLeadScenario().carla_recipe()
    assert rec["camera"] == {"channels": 6, "size": 256, "stack": 2}  # base250cam contract
    assert rec["props"][0]["partial_block"] is True
    assert rec["props"][0]["lane"] == "ego"
    assert rec["success"]["no_collision"] is True
    assert rec["perception"]["classify_within_m"] == StationaryLeadScenario().detect_range_m


def test_scenario_has_ambiguity_phase_and_closing_lead():
    tel = simulate_policy(StationaryLeadScenario(), policy="classifier_react")
    assert np.asarray(tel["hazard_los_flag"]).any(), "lead never enters line-of-sight"
    assert np.asarray(tel["is_occluded_flag"]).any(), "no pre-classification (ambiguous) phase"
    assert float(np.min(tel["dist_to_blind_spot"])) < APPROACH_M, "ego never nears the lead"


# --------------------------------------------------------------------------- #
# 2. Discriminative structure (imagination anticipates, classifier reacts late)#
# --------------------------------------------------------------------------- #
def test_imagination_brakes_earlier_than_classifier():
    sc = StationaryLeadScenario()
    imag = simulate_policy(sc, "imagination_forward")["_extra"]["brake_onset_s"]
    react = simulate_policy(sc, "classifier_react")["_extra"]["brake_onset_s"]
    assert imag < react, "imagination must begin braking before the classifier-react baseline"
    assert brake_lead_time(sc) > 1.0, "anticipation lead should be > 1 s at default speed"


def test_imagination_keeps_more_ttc_and_range():
    sc = StationaryLeadScenario()
    imag = simulate_policy(sc, "imagination_forward")["_extra"]
    react = simulate_policy(sc, "classifier_react")["_extra"]
    assert imag["min_ttc_s"] > react["min_ttc_s"], "imagination should preserve more min-TTC"
    assert imag["min_range_m"] > react["min_range_m"], "imagination should keep a larger gap"


def test_imagination_is_smoother():
    sc = StationaryLeadScenario()
    imag = simulate_policy(sc, "imagination_forward")["_extra"]["peak_jerk"]
    react = simulate_policy(sc, "classifier_react")["_extra"]["peak_jerk"]
    assert imag < react, "comfort-bounded deceleration should have lower peak jerk than a hard stop"


def test_barrier_carries_less_energy_toward_the_lead():
    sc = StationaryLeadScenario()
    # OKRI is "lower safer": slowing early carries less KE into the closing gap
    assert _okri_like(simulate_policy(sc, "imagination_forward")) < _okri_like(
        simulate_policy(sc, "classifier_react"))


def test_latent_estimate_held_earlier_by_imagination():
    """H15 holds a latent lead estimate through the pre-classification phase; react does not.

    Each policy is checked against *its own* ambiguity phase (the two trajectories differ), plus the
    cross-policy claim that imagination begins holding the lead earlier than the classifier does.
    """
    sc = StationaryLeadScenario()
    imag = simulate_policy(sc, "imagination_forward")
    react = simulate_policy(sc, "classifier_react")
    # imagination holds the lead through its whole ambiguous (pre-classification) phase
    imag_amb = np.asarray(imag["is_occluded_flag"])
    assert np.isfinite(np.asarray(imag["wm_hazard_xy"])[imag_amb]).all(), \
        "imagination must hold the lead through the ambiguous phase"
    # classifier-react holds NOTHING during its own ambiguous phase (it waits for the class label)
    react_amb = np.asarray(react["is_occluded_flag"])
    assert not np.isfinite(np.asarray(react["wm_hazard_xy"])[react_amb]).any(), \
        "classifier-react must NOT hold a latent estimate before it classifies"
    # and imagination begins holding the lead strictly earlier (at a larger range / earlier step)
    imag_first = int(np.argmax(np.isfinite(np.asarray(imag["wm_hazard_xy"])[:, 0])))
    react_first = int(np.argmax(np.isfinite(np.asarray(react["wm_hazard_xy"])[:, 0])))
    assert imag_first < react_first, "imagination must begin holding the lead earlier than react"


# --------------------------------------------------------------------------- #
# 3. The mechanistic property: collision rate vs approach speed + falsifier     #
# --------------------------------------------------------------------------- #
def test_collision_rate_imagination_zero_classifier_positive():
    assert collision_rate("imagination_forward") == 0.0
    assert collision_rate("classifier_react") > 0.0


def test_classifier_collides_at_high_speed_not_low():
    """The classifier-react baseline is safe at low approach speed and collides at high speed
    (fixed detection range, braking distance ∝ v²)."""
    low = StationaryLeadScenario(v_cruise=10.0)
    high = StationaryLeadScenario(v_cruise=25.0)
    assert simulate_policy(low, "classifier_react")["_extra"]["collision"] is False
    assert simulate_policy(high, "classifier_react")["_extra"]["collision"] is True
    # imagination survives both
    assert simulate_policy(low, "imagination_forward")["_extra"]["collision"] is False
    assert simulate_policy(high, "imagination_forward")["_extra"]["collision"] is False


def test_advantage_vanishes_with_early_classification():
    """Honest falsifier (P8): the anticipation lead is *specifically* about acting-before-class.
    If the competitor classifies early (large detect_range_m), the safety lead collapses."""
    late = StationaryLeadScenario(detect_range_m=20.0)     # realistic late classification
    early = StationaryLeadScenario(detect_range_m=120.0)   # near-perfect early classification
    assert brake_lead_time(late) > brake_lead_time(early), \
        "advantage must shrink when the competitor classifies early"
    # with early classification the react baseline no longer collides either
    assert collision_rate("classifier_react", base=early) < collision_rate(
        "classifier_react", base=late)


def test_deterministic():
    sc = StationaryLeadScenario()
    a = simulate_policy(sc, "imagination_forward")
    b = simulate_policy(sc, "imagination_forward")
    assert np.allclose(a["ego_v"], b["ego_v"])
    assert np.allclose(np.nan_to_num(a["wm_hazard_xy"]), np.nan_to_num(b["wm_hazard_xy"]))
