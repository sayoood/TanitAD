# INTAKE — Red-light barrier scenario (SC-14 / W-03)

- **Discipline:** Opponent Analyzer (run #4)
- **Date:** 2026-07-11 (real wall-clock).
- **Orchestrator verdict:** _pending_

## ⚠ Reconciliation note (READ FIRST — concurrent opponent runs)
This run (#4) discovered that a **prior run #3 exists on an unmerged branch**
`worktree-agent-opponent-20260710` (commit **874f78e**, pushed to origin, **not in `main`**). Run #3
already: (a) authored **SC-13 stationary-lead** (`2026-07-10-stationary-lead-scenario/`, collision
rate imagination 0.000 / classifier_react 0.429, +3.10 s brake lead), and (b) led with the
**FMVSS-135 NPRM H11 tailwind**. To avoid a duplicate/conflicting second SC-13, this run **defers
SC-13 entirely to run #3** and instead ships the **next** backlog scenario (**SC-14 red-light
barrier**) plus the second-major-operator evidence run #3 did not have (Tesla Houston fatality, the
distinct **EA26002** traffic-violation docket, **Zoox** as a new opponent, GigaWorld-Policy).
**Orchestrator: merge run #3 (874f78e) first, then this run #4; they are additive, not competing.**

## What
A pure, sim-agnostic **scenario specification + synthetic-telemetry generator** for the
**red-light / signal-phase** rule-compliance failure (`red_light_barrier.py`), plus its offline test
suite (`tests/test_red_light_barrier.py`, **11/11 passing**). **Deliberately reuses the accepted
SC-04 `stop_arm_gate.py` barrier-vs-soft-prior oracle** — the signal phase replaces the stop-arm,
cross-traffic replaces the bus — so one `violation_rate` reducer serves both SC-04 and SC-14.

## Why
Weakness **W-03** / **SC-14** in `SCENARIO_DATABASE.md`. The signal-phase rule-barrier now has **two
independent major-operator FACT sources** (gathered this run): a **Waymo** ran a red light in Dallas
(Irving Blvd/Inwood Rd, 2026-07, primary dashcam), and NHTSA **EA26002** (~2.88 M Tesla FSD vehicles)
documents **80 traffic-violation incidents** (from 58) incl. **red-light running / illegal turns**,
14 crashes / 23 injuries. Two sources across two operators is exactly what turns a scenario into a
credible public excellence claim.

## Evidence (design-oracle, P8 — NOT a claim about our model)
`soft_prior` (treats red as a soft cost; enters on red when the intersection looks clear) vs
`rule_barrier` (H9 hard phase barrier; full stop at/before the line regardless of clearance).
Measured (`red_light_barrier_result.json`, local, CPU numpy, 0.2 s, $0):

- **Violation rate over the apparent-clearance sweep {0…12} m:** `rule_barrier` **0.0** /
  `soft_prior` **1.0**. The barrier is **invariant to the temptation**; the soft prior's
  line-crossing speed grows monotonically **3.2 → 10.4 m/s** as the intersection looks clearer.
- **Stop margin:** barrier comes to a full stop **1.1 m before** the line.
- **OKRI toward the occluded crosser:** **12,387 vs 63,765 → −82%** at 4 B vs 15 B params.
- **Object permanence:** only the barrier holds a latent estimate of the pedestrian occluded in the
  conflict zone (H15); the soft prior holds none.

## Tests run
`pytest .../2026-07-11-red-light-barrier-scenario/tests` → **11 passed in 0.20 s**
(`venvs/tanitad`, numpy 2.5.1, pytest 9.1.1). numpy + pytest only; no simulator, no cross-package
import. Telemetry-contract test pins the `ScenarioTelemetry` field names.

## Proposed target location in stack/
`stack/tanitad/eval/scenarios/red_light_barrier.py` (alongside `stop_arm_gate` once that integrates),
consumed by `tanitad/eval/tanitad_metrics.py`. **Handoff to Benchmarks & Eval (Thu):** the SC-04
`violation_rate` reducer (over `_extra`) applies **unchanged** to `_extra.red_light_violation` — one
reducer, two scenarios.

## Risk
Low. Self-contained, numpy-only, no stack imports, no training-path effect. Design oracle only — not
a measurement of the TanitAD checkpoint (P8). Real numbers require a checkpoint rollout on
CARLA-on-pod (signalized junction + phase control via `carla_recipe()`).

## Rollback
Delete the package directory. Nothing in `stack/` depends on it until an explicit integration.
