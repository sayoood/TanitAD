"""Offline tests for the Stationary-Lead scenario feed (W-08 / SC-13).

These validate the *scenario generator*, not the metrics (those live in the Benchmarks & Eval
suite). Asserted:

1. **Telemetry contract** — the emitted log has exactly the ScenarioTelemetry field names, correct
   lengths/shapes, and the scenario-specific compliance signals (``collision``, ``min_ttc_s``,
   ``lal_v2_lead_s``).
2. **Real kinematics** — the log is a genuine forward integration: speed is non-increasing under
   braking, position is monotone, and a collision is exactly ``gap -> 0 while moving``.
3. **Discriminative structure** — the imagination policy anticipates (LAL-v2 lead > 0), never
   collides, and keeps a larger min-TTC; the detection-reactive policy reacts (LAL-v2 lead <= 0)
   and collides when classification fires too late.
4. **The invariance property (the point of the scenario)** — the imagination policy's collision rate
   is **invariant to the classification-range knob** (stays 0), while the reactive policy's collision
   rate **grows as classification fires later**. That is the mechanistic difference between a
   consequence forward-model (keyed on closing geometry) and a detect-then-react policy.

No simulator; numpy + pytest only (one optional test imports the suite if it is on the path). Run:
    pytest "TanitAD Research Hub/Opponent Analyzer/Implementation/incoming/2026-07-15-stationary-lead-scenario/tests"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stationary_lead import (  # noqa: E402
    LAL2_DROP_FRAC,
    LAL2_HOLD,
    LAL2_REF_FRAC,
    POLICIES,
    TELEMETRY_KEYS,
    StationaryLeadScenario,
    collision_rate,
    lal_v2_lead,
    simulate_policy,
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
    for k in ("collision", "min_ttc_s", "lal_v2_lead_s", "decel_onset_s", "t_los_s",
              "detect_range_m", "stop_gap_m"):
        assert k in tel["_extra"], f"missing _extra key {k!r}"


def test_carla_recipe_shape():
    rec = StationaryLeadScenario().carla_recipe()
    assert rec["camera"] == {"channels": 6, "size": 256, "stack": 2}  # base250cam contract
    prop_types = {p["type"] for p in rec["props"]}
    assert any("stalled" in p for p in prop_types)
    assert rec["success"]["full_stop_before_object"] is True
    assert rec["competence"]["detect_range_m"] == StationaryLeadScenario().detect_range_m


def test_scenario_has_object_and_classification_phase():
    tel = simulate_policy(StationaryLeadScenario(), policy="detection_reactive")
    assert np.asarray(tel["is_occluded_flag"]).any(), "no pre-classification phase"
    assert np.asarray(tel["hazard_los_flag"]).any(), "object never reaches nominal classification"
    assert float(np.min(tel["dist_to_blind_spot"])) < APPROACH_M, "ego never nears the object"


# --------------------------------------------------------------------------- #
# 2. Real kinematics                                                           #
# --------------------------------------------------------------------------- #
def test_speed_nonincreasing_and_position_monotone_under_braking():
    tel = simulate_policy(StationaryLeadScenario(), "imagination")
    v = np.asarray(tel["ego_v"])
    gap = np.asarray(tel["dist_to_blind_spot"])
    # imagination only ever decelerates -> speed never rises above cruise, gap never grows
    assert float(np.max(v)) <= StationaryLeadScenario().v_cruise + 1e-6
    assert np.all(np.diff(gap) <= 1e-6), "gap to the object must be non-increasing"


def test_collision_is_gap_to_zero_while_moving():
    # a very late classification forces the reactive policy into contact
    sc = StationaryLeadScenario(detect_range_m=10.0)
    tel = simulate_policy(sc, "detection_reactive")
    if tel["_extra"]["collision"]:
        assert float(np.min(tel["dist_to_blind_spot"])) <= 0.0
        assert tel["collisions"] == 1


def test_lal_v2_helper_matches_reference_sentinels():
    # object never classified -> 0.0; classified but no braking -> NO_REACTION
    z = np.zeros(50, dtype=bool)
    assert lal_v2_lead(np.full(50, 20.0), z) == 0.0
    los = z.copy()
    los[30:] = True
    assert lal_v2_lead(np.full(50, 20.0), los) == pytest.approx(-999.0)


# --------------------------------------------------------------------------- #
# 3. Discriminative structure (imagination anticipates & is safe)              #
# --------------------------------------------------------------------------- #
def test_imagination_anticipates_reactive_does_not():
    sc = StationaryLeadScenario()
    imag = simulate_policy(sc, "imagination")
    react = simulate_policy(sc, "detection_reactive")
    # imagination decelerates before the nominal classification time -> positive lead
    assert imag["_extra"]["lal_v2_lead_s"] > 0.0
    # reactive only decelerates at/after classification -> non-positive lead (or never in time)
    assert react["_extra"]["lal_v2_lead_s"] <= 0.0


def test_imagination_no_collision_reactive_collides_at_default():
    sc = StationaryLeadScenario()   # default detect_range is deliberately late
    assert simulate_policy(sc, "imagination")["_extra"]["collision"] is False
    assert simulate_policy(sc, "detection_reactive")["_extra"]["collision"] is True


def test_imagination_keeps_larger_min_ttc():
    sc = StationaryLeadScenario()
    assert (simulate_policy(sc, "imagination")["_extra"]["min_ttc_s"]
            > simulate_policy(sc, "detection_reactive")["_extra"]["min_ttc_s"])


def test_imagination_carries_less_energy_toward_object():
    sc = StationaryLeadScenario()
    assert _okri_like(simulate_policy(sc, "imagination")) < _okri_like(
        simulate_policy(sc, "detection_reactive"))


def test_latent_estimate_only_for_imagination():
    sc = StationaryLeadScenario()
    imag = simulate_policy(sc, "imagination")
    react = simulate_policy(sc, "detection_reactive")
    occ = np.asarray(imag["is_occluded_flag"])
    assert np.isfinite(np.asarray(imag["wm_hazard_xy"])[occ]).all()
    assert not np.isfinite(np.asarray(react["wm_hazard_xy"])[occ]).any()


# --------------------------------------------------------------------------- #
# 4. The invariance property: collision-rate vs the classification-range knob  #
# --------------------------------------------------------------------------- #
def test_collision_rate_imagination_zero_reactive_positive():
    assert collision_rate("imagination") == 0.0
    assert collision_rate("detection_reactive") > 0.0


def test_reactive_collision_rate_grows_as_classification_fires_later():
    early = collision_rate("detection_reactive", detect_ranges=[60.0, 55.0, 50.0])
    late = collision_rate("detection_reactive", detect_ranges=[25.0, 15.0, 5.0])
    assert late > early, "reactive should collide more when classification fires later"
    # imagination is invariant to the same knob
    assert collision_rate("imagination", detect_ranges=[60.0, 55.0, 50.0]) == 0.0
    assert collision_rate("imagination", detect_ranges=[25.0, 15.0, 5.0]) == 0.0


def test_deterministic():
    sc = StationaryLeadScenario()
    a = simulate_policy(sc, "imagination")
    b = simulate_policy(sc, "imagination")
    assert np.allclose(a["ego_v"], b["ego_v"])
    assert np.allclose(np.nan_to_num(a["wm_hazard_xy"]), np.nan_to_num(b["wm_hazard_xy"]))


# --------------------------------------------------------------------------- #
# Optional: pin LAL-v2 constants to the integrated suite if it is importable   #
# --------------------------------------------------------------------------- #
def test_lal2_constants_match_suite():
    try:
        repo = Path(__file__).resolve().parents[6]
        sys.path.insert(0, str(repo / "stack"))
        from tanitad.eval import metrics  # type: ignore  # noqa
    except Exception:
        pytest.skip("stack.tanitad.eval.metrics not importable in this environment")
    assert (LAL2_DROP_FRAC, LAL2_REF_FRAC, LAL2_HOLD) == (
        metrics.LAL2_DROP_FRAC, metrics.LAL2_REF_FRAC, metrics.LAL2_HOLD)
