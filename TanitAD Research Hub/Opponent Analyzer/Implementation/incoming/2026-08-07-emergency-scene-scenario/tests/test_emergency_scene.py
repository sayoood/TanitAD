"""Offline tests for the Emergency-Scene scenario feed (W-09 / SC-06).

These validate the *scenario generator*, not the metrics (those live in the Benchmarks & Eval
suite). Asserted:

1. **Telemetry contract** — the emitted log has exactly the ScenarioTelemetry field names, correct
   lengths/shapes, and the scenario-specific outcome signals in ``_extra``.
2. **Discriminative structure** — the scenario separates a rule-literal baseline from a yielding
   policy along the axis the NHTSA ADS-developers letter and the Zoox smoke recall expose: an
   emergency scene is flaggable at the **scene** level before any object in it is classifiable.
3. **The obscurant-collapse property (the point of the scenario)** — as ``scene_ambiguity`` rises,
   the rule-literal policy's usable reaction range collapses, so it enters the responders' corridor
   and **halts inside it** (the documented failure: responders physically moved the vehicle), while
   the yielding policy's corridor blockage stays exactly 0 s at every ambiguity level.

No simulator, no cross-package import — numpy + pytest only. Run:
    pytest "TanitAD Research Hub/Opponent Analyzer/Implementation/incoming/2026-08-07-emergency-scene-scenario/tests"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emergency_scene import (  # noqa: E402
    POLICIES,
    TELEMETRY_KEYS,
    EmergencySceneScenario,
    incursion_rate,
    mean_blockage_s,
    mean_detect_lead_time,
    simulate_policy,
)

APPROACH_M = 30.0          # matches OKRI_APPROACH_M in the metric suite
SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]


def _okri_like(tel: dict) -> float:
    """Crude OKRI proxy (same shape as compute_okri) for a relative comparison toward the scene."""
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
    sc = EmergencySceneScenario()
    tel = simulate_policy(sc, policy=policy)
    for key in TELEMETRY_KEYS:
        assert key in tel, f"missing telemetry key {key!r}"
    T = sc.steps
    for key in ("ego_v", "ego_jerk", "steer_rate", "latency_ms", "hazard_los_flag",
                "dist_to_blind_spot", "is_occluded_flag"):
        assert len(tel[key]) == T, f"{key} wrong length"
    assert np.asarray(tel["wm_hazard_xy"]).shape == (T, 2)
    assert np.asarray(tel["gt_hazard_xy"]).shape == (T, 2)
    for k in ("blockage_duration_s", "scene_incursion_m", "detect_lead_time_s",
              "non_nominal_detected", "halted_in_corridor", "min_gap_m", "scene_ambiguity"):
        assert k in tel["_extra"], f"missing _extra key {k!r}"


def test_invalid_policy_raises():
    with pytest.raises(ValueError):
        simulate_policy(EmergencySceneScenario(), policy="wishful")


def test_carla_recipe_shape():
    rec = EmergencySceneScenario(scene_ambiguity=0.5).carla_recipe()
    assert rec["camera"] == {"channels": 6, "size": 256, "stack": 2}  # base250cam contract
    prop_types = {p["type"] for p in rec["props"]}
    assert any("emergency" in p for p in prop_types), "no responder vehicle in the recipe"
    assert any("cone" in p for p in prop_types) and any("flare" in p for p in prop_types)
    assert any("smoke" in p for p in prop_types)
    # the corridor is an explicit region, and the success bars are the zero-bars
    assert rec["corridor"]["s1"] > rec["corridor"]["s0"]
    assert rec["success"]["scene_incursion_m"] == 0.0
    assert rec["success"]["blockage_duration_s"] == 0.0
    # honest Phase-0 limit is recorded in the recipe, not hidden
    assert rec["limits"]["audio"] == "out-of-scope-phase0"


def test_scene_is_in_line_of_sight_the_whole_approach():
    """The damning point: the scene is geometrically visible; only its *legibility* differs."""
    tel = simulate_policy(EmergencySceneScenario(scene_ambiguity=1.0), "rule_literal")
    assert np.asarray(tel["hazard_los_flag"]).any(), "scene never in line of sight"
    # visible-but-illegible band is non-empty exactly when the obscurant is thick
    assert np.asarray(tel["is_occluded_flag"]).any()
    clear = simulate_policy(EmergencySceneScenario(scene_ambiguity=0.0), "rule_literal")
    assert not np.asarray(clear["is_occluded_flag"]).any(), "clear air should be fully legible"


# --------------------------------------------------------------------------- #
# 2. Discriminative structure (scene-level flag beats object classification)   #
# --------------------------------------------------------------------------- #
def test_obscurant_collapses_object_range_but_not_scene_range():
    """THE hypothesis under test, made explicit and asserted as the scenario's design."""
    clear, thick = EmergencySceneScenario(scene_ambiguity=0.0), EmergencySceneScenario(scene_ambiguity=1.0)
    assert thick.obj_range_m() < 0.25 * clear.obj_range_m(), "object range must collapse"
    assert thick.ood_range_m() > 0.8 * clear.ood_range_m(), "scene range must survive"
    assert thick.ood_range_m() > thick.obj_range_m()


def test_yielding_policy_flags_the_scene_earlier():
    sc = EmergencySceneScenario(scene_ambiguity=0.5)
    yld = simulate_policy(sc, "imagine_and_yield")["_extra"]
    lit = simulate_policy(sc, "rule_literal")["_extra"]
    # yielding flags with distance to spare; rule-literal reacts only at/after the boundary
    assert yld["detect_lead_time_s"] > 0.0
    assert yld["detect_lead_time_s"] > lit["detect_lead_time_s"]
    assert yld["non_nominal_detected"] is True


def test_yielding_policy_stops_before_the_corridor():
    for a in SWEEP:
        tel = simulate_policy(EmergencySceneScenario(scene_ambiguity=a), "imagine_and_yield")
        ex = tel["_extra"]
        assert ex["scene_incursion_m"] == 0.0, f"yielding policy entered the corridor at {a}"
        assert ex["blockage_duration_s"] == 0.0
        assert ex["halted_in_corridor"] is False
        assert tel["collisions"] == 0


def test_yielding_policy_carries_less_energy_toward_the_scene():
    sc = EmergencySceneScenario(scene_ambiguity=0.5)
    # OKRI is "lower safer": flagging early carries less KE toward the emergency scene
    assert _okri_like(simulate_policy(sc, "imagine_and_yield")) < _okri_like(
        simulate_policy(sc, "rule_literal"))


def test_rule_literal_reproduces_the_zoox_signature_under_thick_smoke():
    """Drove in -> failed to recognize in time -> panic brake -> came to rest inside the scene."""
    ex = simulate_policy(EmergencySceneScenario(scene_ambiguity=1.0), "rule_literal")["_extra"]
    assert ex["scene_incursion_m"] > 0.0, "should have entered the corridor"
    assert ex["halted_in_corridor"] is True, "should have come to rest inside the corridor"
    assert ex["detect_lead_time_s"] < 0.0, "should only have reacted after the boundary"


def test_scene_estimate_is_only_held_after_reacting():
    sc = EmergencySceneScenario(scene_ambiguity=1.0)
    yld, lit = simulate_policy(sc, "imagine_and_yield"), simulate_policy(sc, "rule_literal")
    n_held = lambda t: int(np.isfinite(np.asarray(t["wm_hazard_xy"])[:, 0]).sum())
    # the yielding policy holds an estimate of the responder for strictly longer
    assert n_held(yld) > n_held(lit)
    assert n_held(lit) >= 0


# --------------------------------------------------------------------------- #
# 3. Outcome vs obscurant density                                              #
# --------------------------------------------------------------------------- #
def test_incursion_rate_yielding_zero_rule_literal_positive():
    assert incursion_rate("imagine_and_yield") == 0.0
    assert incursion_rate("rule_literal") > 0.0


def test_blockage_duration_bar():
    assert mean_blockage_s("imagine_and_yield") == 0.0
    assert mean_blockage_s("rule_literal") > 0.0
    assert mean_detect_lead_time("imagine_and_yield") > mean_detect_lead_time("rule_literal")


def test_rule_literal_degrades_monotonically_with_obscurant():
    ex = [simulate_policy(EmergencySceneScenario(scene_ambiguity=a), "rule_literal")["_extra"]
          for a in SWEEP]
    leads = [e["detect_lead_time_s"] for e in ex]
    incur = [e["scene_incursion_m"] for e in ex]
    assert all(leads[i + 1] <= leads[i] + 1e-9 for i in range(len(leads) - 1)), leads
    assert all(incur[i + 1] >= incur[i] - 1e-9 for i in range(len(incur) - 1)), incur


def test_yielding_policy_is_near_invariant_to_obscurant():
    ex = [simulate_policy(EmergencySceneScenario(scene_ambiguity=a), "imagine_and_yield")["_extra"]
          for a in SWEEP]
    leads = [e["detect_lead_time_s"] for e in ex]
    # scene-level detection degrades only weakly (by design) — never enough to enter the corridor
    assert max(leads) - min(leads) < 1.5, leads
    assert all(e["scene_incursion_m"] == 0.0 for e in ex)


def test_deterministic():
    sc = EmergencySceneScenario(scene_ambiguity=0.5)
    a, b = simulate_policy(sc, "imagine_and_yield"), simulate_policy(sc, "imagine_and_yield")
    assert np.allclose(a["ego_v"], b["ego_v"])
    assert np.allclose(np.nan_to_num(a["wm_hazard_xy"]), np.nan_to_num(b["wm_hazard_xy"]))
