"""Offline tests for the Stationary-Lead scenario feed (W-08 / SC-13).

These validate the *scenario generator*, not the metrics (those live in the Benchmarks & Eval
suite). Asserted:

1. **Telemetry contract** — the emitted log has exactly the ScenarioTelemetry field names, correct
   lengths/shapes, and the scenario-specific outcome signals in ``_extra``.
2. **Discriminative structure** — the scenario separates a detection-reactive baseline from a
   forward-model policy along the axis the Avride ODI + Tesla EA26002 findings expose: the
   *consequence* of a closing gap on a stationary lead is knowable before the object is classified.
3. **The invariance property (the point of the scenario)** — the forward-model policy's braking-onset
   lead time, min-TTC and collision outcome are **invariant to ``classification_ambiguity``**, while
   the detection-reactive policy **degrades monotonically** with it and eventually collides (drops the
   lead). That is the mechanistic difference between forward-modelling the gap (H15) and waiting for a
   detection/classification prior.

No simulator, no cross-package import — numpy + pytest only. Run:
    pytest "TanitAD Research Hub/Opponent Analyzer/Implementation/incoming/2026-07-31-stationary-lead-scenario/tests"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stationary_lead import (  # noqa: E402
    POLICIES,
    TELEMETRY_KEYS,
    StationaryLeadScenario,
    collision_rate,
    mean_lead_time,
    simulate_policy,
)

APPROACH_M = 30.0          # matches OKRI_APPROACH_M in the metric suite
SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]


def _okri_like(tel: dict) -> float:
    """Crude OKRI proxy (same shape as compute_okri) for a relative comparison toward the lead."""
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
    for k in ("brake_onset_lead_time_s", "min_ttc_s", "min_gap_m", "collided",
              "lead_dropped", "classification_ambiguity"):
        assert k in tel["_extra"], f"missing _extra key {k!r}"


def test_invalid_policy_raises():
    with pytest.raises(ValueError):
        simulate_policy(StationaryLeadScenario(), policy="wishful")


def test_carla_recipe_shape():
    rec = StationaryLeadScenario().carla_recipe()
    assert rec["camera"] == {"channels": 6, "size": 256, "stack": 2}  # base250cam contract
    prop_types = {p["type"] for p in rec["props"]}
    assert any("stalled" in p for p in prop_types)
    assert rec["props"][0]["lane"] == "ego"
    assert rec["success"]["no_contact"] is True


def test_object_is_visible_but_in_lane():
    tel = simulate_policy(StationaryLeadScenario(), policy="detection_reactive")
    assert np.asarray(tel["hazard_los_flag"]).any(), "lead never visible"
    assert float(np.min(tel["dist_to_blind_spot"])) < APPROACH_M, "ego never nears the lead"


# --------------------------------------------------------------------------- #
# 2. Discriminative structure (forward-model anticipates, reactive is late)    #
# --------------------------------------------------------------------------- #
def test_forward_model_anticipates_reactive_is_late():
    sc = StationaryLeadScenario(classification_ambiguity=0.0)
    imf = simulate_policy(sc, "imagination_forward")["_extra"]
    det = simulate_policy(sc, "detection_reactive")["_extra"]
    # forward model brakes BEFORE the anticipation reference (positive lead time); reactive after it
    assert imf["brake_onset_lead_time_s"] > 0.0
    assert det["brake_onset_lead_time_s"] <= 0.0
    assert imf["brake_onset_lead_time_s"] > det["brake_onset_lead_time_s"]


def test_forward_model_keeps_more_ttc_margin():
    sc = StationaryLeadScenario(classification_ambiguity=0.5)
    imf = simulate_policy(sc, "imagination_forward")["_extra"]
    det = simulate_policy(sc, "detection_reactive")["_extra"]
    assert imf["min_ttc_s"] > det["min_ttc_s"]
    assert imf["min_gap_m"] > det["min_gap_m"]


def test_forward_model_never_contacts_lead():
    for a in SWEEP:
        tel = simulate_policy(StationaryLeadScenario(classification_ambiguity=a), "imagination_forward")
        assert tel["_extra"]["collided"] is False, f"forward model collided at ambiguity {a}"
        assert tel["collisions"] == 0


def test_forward_model_carries_less_energy_toward_lead():
    sc = StationaryLeadScenario(classification_ambiguity=0.5)
    # OKRI is "lower safer": braking early carries less KE toward the stationary lead
    assert _okri_like(simulate_policy(sc, "imagination_forward")) < _okri_like(
        simulate_policy(sc, "detection_reactive"))


def test_object_permanence_only_survives_ambiguity_for_forward_model():
    # under high ambiguity the reactive stack drops the lead (wm NaN in-range); the forward model holds it
    sc = StationaryLeadScenario(classification_ambiguity=1.0)
    imf = simulate_policy(sc, "imagination_forward")
    det = simulate_policy(sc, "detection_reactive")
    in_range = np.asarray(imf["hazard_los_flag"])
    assert np.isfinite(np.asarray(imf["wm_hazard_xy"])[in_range]).all()
    assert det["_extra"]["lead_dropped"] is True
    assert not np.isfinite(np.asarray(det["wm_hazard_xy"])[in_range]).any()


# --------------------------------------------------------------------------- #
# 3. The invariance property: outcome vs classification ambiguity              #
# --------------------------------------------------------------------------- #
def test_collision_rate_forward_model_zero_reactive_positive():
    assert collision_rate("imagination_forward") == 0.0
    assert collision_rate("detection_reactive") > 0.0


def test_forward_model_is_invariant_to_ambiguity():
    """The consequence of the closing gap does not depend on how classifiable the object is."""
    base = [simulate_policy(StationaryLeadScenario(classification_ambiguity=a),
                            "imagination_forward")["_extra"] for a in SWEEP]
    lts = [e["brake_onset_lead_time_s"] for e in base]
    ttcs = [e["min_ttc_s"] for e in base]
    assert max(lts) - min(lts) < 1e-6, "forward-model lead time must not depend on ambiguity"
    assert max(ttcs) - min(ttcs) < 1e-6, "forward-model min-TTC must not depend on ambiguity"


def test_reactive_degrades_monotonically_with_ambiguity():
    ex = [simulate_policy(StationaryLeadScenario(classification_ambiguity=a),
                          "detection_reactive")["_extra"] for a in SWEEP]
    lts = [e["brake_onset_lead_time_s"] for e in ex]
    ttcs = [e["min_ttc_s"] for e in ex]
    # both the anticipation lead time and the min-TTC are non-increasing as ambiguity rises
    assert all(lts[i + 1] <= lts[i] + 1e-9 for i in range(len(lts) - 1)), lts
    assert all(ttcs[i + 1] <= ttcs[i] + 1e-9 for i in range(len(ttcs) - 1)), ttcs
    # and the forward model beats the reactive lead time at every ambiguity level
    assert mean_lead_time("imagination_forward") > mean_lead_time("detection_reactive")


def test_deterministic():
    sc = StationaryLeadScenario(classification_ambiguity=0.5)
    a = simulate_policy(sc, "imagination_forward")
    b = simulate_policy(sc, "imagination_forward")
    assert np.allclose(a["ego_v"], b["ego_v"])
    assert np.allclose(np.nan_to_num(a["wm_hazard_xy"]), np.nan_to_num(b["wm_hazard_xy"]))
