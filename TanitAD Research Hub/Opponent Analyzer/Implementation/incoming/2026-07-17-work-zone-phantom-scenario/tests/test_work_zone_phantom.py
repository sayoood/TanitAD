"""Offline tests for the Work-Zone Phantom scenario feed (W-01).

These validate the *scenario generator*, not the metrics (those live in the Benchmarks & Eval
suite). Two things are asserted:

1. **Telemetry contract** — the emitted log has exactly the ScenarioTelemetry field names,
   correct lengths/shapes, and a scenario-specific ``closure_incursion_m`` compliance signal.
2. **Discriminative structure** — the scenario actually separates a pixel-reactive baseline
   from a world-model policy along the axes the Waymo work-zone recall exposes: anticipation
   (brake before line-of-sight), occluded-region caution (lower kinetic energy into the blind
   edge), latent object permanence (a hidden-actor estimate under occlusion), and rule
   compliance (no incursion into the closed lane).

No simulator, no cross-package import — numpy + pytest only. Run:
    pytest "TanitAD Research Hub/Opponent Analyzer/Implementation/incoming/2026-07-17-work-zone-phantom-scenario/tests"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from work_zone_phantom import (  # noqa: E402
    POLICIES,
    TELEMETRY_KEYS,
    WorkZonePhantomScenario,
    simulate_policy,
)

APPROACH_M = 30.0          # matches OKRI_APPROACH_M in the metric suite
BRAKE_JERK = -0.5          # braking-onset threshold (negative jerk)


def _first_brake_idx(tel: dict) -> int:
    idx = np.flatnonzero(np.asarray(tel["ego_jerk"]) < BRAKE_JERK)
    return int(idx[0]) if idx.size else len(tel["ego_v"])


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
    sc = WorkZonePhantomScenario()
    tel = simulate_policy(sc, policy=policy)
    # every ScenarioTelemetry field is present
    for key in TELEMETRY_KEYS:
        assert key in tel, f"missing telemetry key {key!r}"
    # time series are length T
    T = sc.steps
    for key in ("ego_v", "ego_jerk", "steer_rate", "latency_ms", "hazard_los_flag",
                "dist_to_blind_spot", "is_occluded_flag"):
        assert len(tel[key]) == T, f"{key} wrong length"
    # hidden-actor arrays are [T, 2]
    assert np.asarray(tel["wm_hazard_xy"]).shape == (T, 2)
    assert np.asarray(tel["gt_hazard_xy"]).shape == (T, 2)
    # scenario-specific compliance signal present and non-negative
    assert "closure_incursion_m" in tel["_extra"]
    assert tel["_extra"]["closure_incursion_m"] >= 0.0


def test_carla_recipe_shape():
    rec = WorkZonePhantomScenario().carla_recipe()
    assert rec["camera"] == {"channels": 6, "size": 256, "stack": 2}  # base250cam contract
    prop_types = {p["type"] for p in rec["props"]}
    assert any("ramp_closure" in p for p in prop_types)
    assert any("cone" in p or "taper" in p for p in prop_types)
    assert rec["actors"][0]["occluded_until_s"] == WorkZonePhantomScenario().los_s
    assert rec["success"]["no_closure_incursion"] is True


def test_scenario_has_a_blind_edge_and_occlusion():
    """The geometry must actually produce an occluded phase and a blind approach."""
    tel = simulate_policy(WorkZonePhantomScenario(), policy="reactive")
    assert np.asarray(tel["is_occluded_flag"]).any(), "no occluded phase"
    assert np.asarray(tel["hazard_los_flag"]).any(), "hazard never reaches LoS"
    assert float(np.min(tel["dist_to_blind_spot"])) < APPROACH_M, "ego never nears blind edge"


# --------------------------------------------------------------------------- #
# 2. Discriminative structure (world_model should beat reactive)               #
# --------------------------------------------------------------------------- #
def test_world_model_anticipates_before_line_of_sight():
    sc = WorkZonePhantomScenario()
    wm = simulate_policy(sc, "world_model")
    rx = simulate_policy(sc, "reactive")
    los_idx = int(np.flatnonzero(np.asarray(wm["hazard_los_flag"]))[0])
    # LAL > 0 for the world model: it brakes strictly before line-of-sight
    assert _first_brake_idx(wm) < los_idx
    # LAL <= 0 for reactive: it does not brake before line-of-sight
    assert _first_brake_idx(rx) >= los_idx


def test_world_model_carries_less_energy_into_the_blind_region():
    sc = WorkZonePhantomScenario()
    # OKRI is "lower safer": the world model must score lower than the reactive baseline
    assert _okri_like(simulate_policy(sc, "world_model")) < _okri_like(
        simulate_policy(sc, "reactive"))


def test_object_permanence_only_for_world_model():
    sc = WorkZonePhantomScenario()
    wm = simulate_policy(sc, "world_model")
    rx = simulate_policy(sc, "reactive")
    occ = np.asarray(wm["is_occluded_flag"])
    # world model holds a finite hidden-actor estimate while occluded -> LOPS > 0
    assert np.isfinite(np.asarray(wm["wm_hazard_xy"])[occ]).all()
    # reactive holds no estimate at all -> LOPS 0.0 baseline
    assert not np.isfinite(np.asarray(rx["wm_hazard_xy"])[occ]).any()


def test_reactive_violates_the_closure_world_model_does_not():
    sc = WorkZonePhantomScenario()
    assert simulate_policy(sc, "world_model")["_extra"]["closure_incursion_m"] == 0.0
    assert simulate_policy(sc, "reactive")["_extra"]["closure_incursion_m"] > 0.0


def test_deterministic():
    sc = WorkZonePhantomScenario()
    a = simulate_policy(sc, "world_model")
    b = simulate_policy(sc, "world_model")
    assert np.allclose(a["ego_v"], b["ego_v"])
    assert np.allclose(np.nan_to_num(a["wm_hazard_xy"]), np.nan_to_num(b["wm_hazard_xy"]))
