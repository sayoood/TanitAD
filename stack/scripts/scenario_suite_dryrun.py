"""Wiring dry-run: work-zone-phantom oracle -> ScenarioTelemetry -> metric suite.

Closes the "scenario-to-telemetry-to-metric wiring is partial" gap flagged in the
hub deep-screen (2026-07-08): proves the full path scenario module -> telemetry
contract -> run_scenario_suite executes end-to-end and that the design oracle's
discriminative structure survives the real metric implementations (world_model
must beat reactive on LAL, OKRI, LOPS and on the H9 closure-incursion signal).

P8: the telemetry is a DESIGN ORACLE (archetypal policies), not our checkpoint —
no model claim. The same path scores real CARLA rollouts once W31-32 lands; only
`simulate_policy` gets replaced.

Usage:  python stack/scripts/scenario_suite_dryrun.py [--out report.json]
"""

from __future__ import annotations

import argparse
import json

from tanitad.eval.metrics import ScenarioTelemetry, run_scenario_suite
from tanitad.eval.scenarios.registry import run_registered_suite
from tanitad.eval.scenarios.work_zone_phantom import (POLICIES,
                                                      WorkZonePhantomScenario,
                                                      simulate_policy)


def telemetry_from_oracle(log: dict) -> ScenarioTelemetry:
    """Oracle dict -> ScenarioTelemetry (drops the scenario-specific _extra)."""
    return ScenarioTelemetry(**{k: v for k, v in log.items()
                                if not k.startswith("_")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sc = WorkZonePhantomScenario()
    rows = {}
    for policy in POLICIES:
        log = simulate_policy(sc, policy)
        suite = run_scenario_suite(telemetry_from_oracle(log),
                                   model_name=f"oracle:{policy}")
        suite["closure_incursion_m"] = log["_extra"]["closure_incursion_m"]
        rows[policy] = suite

    wm, re_ = rows["world_model"], rows["reactive"]
    checks = {
        "LAL_wm_proactive": wm["LAL_s"] > 0 > re_["LAL_s"] or
                            wm["LAL_s"] > re_["LAL_s"],
        "OKRI_wm_safer": wm["OKRI"] < re_["OKRI"],
        "LOPS_wm_tracks": wm["LOPS"] > 0.5 > re_["LOPS"],
        "CNCE_wm_efficient": wm["CNCE"] > re_["CNCE"],
        "closure_wm_compliant": wm["closure_incursion_m"] == 0.0
                                < re_["closure_incursion_m"],
    }
    # --- traffic light (SC-14) picked up via the scenario registry ---------------------------- #
    tl_rows = run_registered_suite(["traffic_light_red", "traffic_light_green"])
    tl_red, tl_green = tl_rows["traffic_light_red"], tl_rows["traffic_light_green"]
    checks.update({
        # a rule barrier scores a perfect TLC on a clean red stop; a soft prior runs the red -> 0
        "TLC_red_barrier_compliant": tl_red["rule_barrier"]["TLC"] == 1.0,
        "TLC_red_soft_prior_runs_red": tl_red["soft_prior"]["TLC"] == 0.0,
        # on green the barrier proceeds smoothly; the soft prior phantom-brakes -> penalized
        "TLC_green_barrier_beats_soft": (tl_green["rule_barrier"]["TLC"]
                                         > tl_green["soft_prior"]["TLC"]),
    })

    report = {"exp": "scenario-suite-wiring-dryrun",
              "scenario": sc.name,
              "scenarios": [sc.name, "traffic_light_red", "traffic_light_green"],
              "rows": rows, "traffic_light": tl_rows, "checks": checks,
              "all_pass": all(checks.values()),
              "note": "design-oracle telemetry, NOT a model claim (P8)"}
    print(json.dumps(report, indent=2))
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(report, indent=2))
    if not report["all_pass"]:
        raise SystemExit("wiring dry-run FAILED — discriminative structure "
                         "did not survive the metric implementations")


if __name__ == "__main__":
    main()
