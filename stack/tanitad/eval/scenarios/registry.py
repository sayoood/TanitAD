"""Scenario-suite registry — the single seam a runner iterates to score every eval scenario.

Each registered scenario knows how to (a) build its default geometry, (b) enumerate its archetypal
oracle policies, (c) emit a ScenarioTelemetry-shaped oracle log, and (d) score that log into a metric
dict with the correct entry point. This decouples the runner (scripts/scenario_suite_dryrun.py, the
gate, a future closed-loop harness) from the per-scenario details: adding a scenario = adding one
``ScenarioEntry`` here, and every runner picks it up.

Two scoring entry points are needed because the scenarios live in two metric domains:
  * occluder / rule-compliance scenarios (work_zone_phantom)  -> ``run_scenario_suite`` (LAL/TMS/OKRI/CNCE/LOPS)
  * signalized-intersection (traffic_light, SC-14)            -> ``traffic_light_metrics`` (those five + TLC)

P8: the registered oracle logs are DESIGN ORACLES (archetypal policies), not our checkpoint — no
model claim. The same entries score a real rollout once its ``simulate`` is replaced by a
checkpoint-driven ego.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tanitad.eval.metrics import (ScenarioTelemetry, run_scenario_suite,
                                  traffic_light_metrics)
from tanitad.eval.scenarios import traffic_light as _tl
from tanitad.eval.scenarios import work_zone_phantom as _wz


def _telemetry_from_oracle(log: dict) -> ScenarioTelemetry:
    """Oracle dict -> ScenarioTelemetry (drops the scenario-specific ``_extra``)."""
    return ScenarioTelemetry(**{k: v for k, v in log.items() if not k.startswith("_")})


def _score_base(log: dict, model_name: str) -> dict:
    """Score a telemetry-only scenario with the five-metric suite."""
    return run_scenario_suite(_telemetry_from_oracle(log), model_name=model_name)


def _score_traffic_light(log: dict, model_name: str) -> dict:
    """Score a signalized-intersection scenario: the five-metric suite + TLC."""
    ex = log["_extra"]
    return traffic_light_metrics(_telemetry_from_oracle(log), ex["ego_s"],
                                 ex["signal_state"], ex["stopline_s"], model_name=model_name)


@dataclass(frozen=True)
class ScenarioEntry:
    name: str
    make: Callable            # () -> scenario dataclass instance (default geometry)
    policies: tuple           # archetypal oracle policy names (worst-first, best-last)
    simulate: Callable        # (scenario, policy) -> oracle log dict
    score: Callable           # (log, model_name) -> metric dict
    headline: str             # the primary metric key for this scenario


# The registry. Traffic-light is registered on both its canonical plans (red = must-stop barrier;
# green = must-proceed / no-phantom-brake) so a runner exercises both halves of TLC.
SCENARIO_REGISTRY: dict[str, ScenarioEntry] = {
    "work_zone_phantom": ScenarioEntry(
        name="work_zone_phantom",
        make=_wz.WorkZonePhantomScenario,
        policies=_wz.POLICIES,                         # ("reactive", "world_model")
        simulate=_wz.simulate_policy,
        score=_score_base,
        headline="OKRI",
    ),
    "traffic_light_red": ScenarioEntry(
        name="traffic_light_red",
        make=lambda: _tl.TrafficLightScenario(signal_plan="red"),
        policies=_tl.POLICIES,                         # ("soft_prior", "rule_barrier")
        simulate=_tl.simulate_policy,
        score=_score_traffic_light,
        headline="TLC",
    ),
    "traffic_light_green": ScenarioEntry(
        name="traffic_light_green",
        make=lambda: _tl.TrafficLightScenario(signal_plan="green"),
        policies=_tl.POLICIES,
        simulate=_tl.simulate_policy,
        score=_score_traffic_light,
        headline="TLC",
    ),
}


def run_registered_suite(names=None) -> dict:
    """Score every registered scenario's archetypal policies. Returns {scenario: {policy: metrics}}.

    ``names`` optionally restricts to a subset of registry keys. Design-oracle telemetry (P8).
    """
    keys = names if names is not None else list(SCENARIO_REGISTRY)
    out: dict[str, dict] = {}
    for key in keys:
        entry = SCENARIO_REGISTRY[key]
        out[key] = {pol: entry.score(entry.simulate(entry.make(), pol), f"oracle:{pol}")
                    for pol in entry.policies}
    return out
