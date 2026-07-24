"""Integration guard: scenario oracle -> ScenarioTelemetry -> metric suite.

Keeps the wiring proven by scripts/scenario_suite_dryrun.py from regressing:
the oracle's field contract must construct a ScenarioTelemetry, and the
design-oracle discriminative structure must survive the real metrics. Also guards
the scenario REGISTRY (registry.py) that the dryrun and future runners iterate,
including the traffic-light (SC-14) TLC path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scenario_suite_dryrun import telemetry_from_oracle  # noqa: E402

from tanitad.eval.metrics import run_scenario_suite
from tanitad.eval.scenarios.registry import (SCENARIO_REGISTRY,
                                             run_registered_suite)
from tanitad.eval.scenarios.work_zone_phantom import (WorkZonePhantomScenario,
                                                      simulate_policy)


def test_oracle_contract_constructs_telemetry():
    sc = WorkZonePhantomScenario()
    for policy in ("reactive", "world_model"):
        tel = telemetry_from_oracle(simulate_policy(sc, policy))
        suite = run_scenario_suite(tel, model_name=policy)
        assert set(suite) >= {"LAL_s", "TMS", "OKRI", "CNCE", "LOPS"}


def test_discriminative_structure_survives_metrics():
    sc = WorkZonePhantomScenario()
    wm = run_scenario_suite(telemetry_from_oracle(
        simulate_policy(sc, "world_model")))
    re_ = run_scenario_suite(telemetry_from_oracle(
        simulate_policy(sc, "reactive")))
    assert wm["LAL_s"] > re_["LAL_s"]          # proactive vs reactive braking
    assert wm["OKRI"] < re_["OKRI"]            # less energy into the blind spot
    assert wm["LOPS"] > 0.5 > re_["LOPS"]      # latent tracking vs none
    assert wm["CNCE"] > re_["CNCE"]            # efficiency edge


# --------------------------------------------------------------------------- #
# Registry wiring: the runner picks up work-zone AND traffic-light (SC-14).     #
# --------------------------------------------------------------------------- #
def test_registry_lists_traffic_light_and_work_zone():
    assert "work_zone_phantom" in SCENARIO_REGISTRY
    assert "traffic_light_red" in SCENARIO_REGISTRY
    assert "traffic_light_green" in SCENARIO_REGISTRY
    assert SCENARIO_REGISTRY["traffic_light_red"].headline == "TLC"


def test_registered_suite_scores_every_scenario_end_to_end():
    out = run_registered_suite()
    # every registered scenario yields a metric dict per archetypal policy
    for key, entry in SCENARIO_REGISTRY.items():
        assert set(out[key]) == set(entry.policies)
        for pol in entry.policies:
            assert entry.headline in out[key][pol]


def test_traffic_light_tlc_discriminates_through_registry():
    out = run_registered_suite(["traffic_light_red", "traffic_light_green"])
    red, green = out["traffic_light_red"], out["traffic_light_green"]
    # red: a rule barrier stops cleanly (TLC=1); a soft prior runs the red (TLC=0)
    assert red["rule_barrier"]["TLC"] == 1.0
    assert red["soft_prior"]["TLC"] == 0.0
    # green: the barrier proceeds smoothly; the soft prior phantom-brakes -> penalized
    assert green["rule_barrier"]["TLC"] > green["soft_prior"]["TLC"]
    # OKRI (lower safer) ranks the barrier safer on the red approach
    assert red["rule_barrier"]["OKRI"] < red["soft_prior"]["OKRI"]
