"""Integration guard: scenario oracle -> ScenarioTelemetry -> metric suite.

Keeps the wiring proven by scripts/scenario_suite_dryrun.py from regressing:
the oracle's field contract must construct a ScenarioTelemetry, and the
design-oracle discriminative structure must survive the real metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scenario_suite_dryrun import telemetry_from_oracle  # noqa: E402

from tanitad.eval.metrics import run_scenario_suite
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
