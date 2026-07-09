# INTAKE — LAL-v2 anticipatory-deceleration metric + SC-01 metric audit

- **Discipline:** Benchmarks & Eval
- **Date:** 2026-07-09
- **Slug:** `2026-07-09-lal-v2-anticipation`
- **Author:** Benchmarks & Eval weekly agent (Thursday)

## What

1. **`lal_v2.py`** — a new custom metric, **LAL-v2 (anticipatory deceleration lead)**,
   fixing a discrimination failure in the shipped LAL-v1 (`stack/tanitad/eval/metrics.py::compute_lal`).
   LAL-v1 detects "braking onset" via a hard longitudinal-jerk threshold (`ego_jerk < -1.5 m/s^3`);
   LAL-v2 detects the onset of *sustained deceleration by speed drop* relative to the free-cruise
   reference, so a gentle (comfort-bounded) anticipatory slowdown registers. numpy-only, plain-array
   API (no `ScenarioTelemetry` import) so it tests standalone.
2. **`tests/test_lal_v2.py`** — 7 sanity tests, every case analytic ground truth (gate G-B2).
3. **`audit_sc01.py` + `audit_results.json`** — the measured experiment (G-H): an independent-test
   audit of the 2026-07-08 first-live-CARLA SC-01 run. Not part of the standalone test; a research
   artifact that imports the stack it audits.

## Why

The **first live CARLA measurement of SC-01** (2026-07-08, `stack/experiments/p0-carla-workzone/`)
scored **LAL-v1 = -0.7 s for BOTH** the reactive and the anticipating policy — zero discrimination on
real physics. The run author flagged it in commit `2d87acb`: *"LAL (jerk-threshold never fires on the
gentle slowdown — both read -0.7)"*. LAL-v1 measures reaction *hardness*, not anticipation, and is
blind to exactly the smooth pre-line-of-sight slowing TanitAD's world model (H15) is meant to produce.

The audit reproduces the failure and quantifies the fix (see the research note for full numbers):

- **LAL-v1 collapse cliff at the -1.5 m/s^3 threshold.** An anticipatory ease-off (16 -> 9 m/s) that is
  *smoother than ~4 s* keeps its peak jerk above -1.5 m/s^3 -> LAL-v1 never fires (sentinel) -> no
  separation. Realistic comfort-bounded braking (ISO-2631-scale |jerk| < ~2 m/s^3) lands in this zone.
- **LAL-v2 separates across the whole smoothness range**: anticipation lead +0.3 .. +3.1 s vs the
  reactive baseline's -0.3 s.

LAL-v2 is the latent, pre-line-of-sight generalization of the recognized **Time-To-Brake (TTB)** metric
family (TTB/TTC require the hazard to be detectable; LAL-v2 credits braking before line-of-sight).

## Evidence / tests run

- `pytest tests/` → **7 passed** (venv `C:\Users\Admin\venvs\tanitad`, numpy 2.5.1, pytest 9.1.1).
- `python audit_sc01.py` → `audit_results.json` (reproducible; RNG seed 12345 for the LOPS MC).
- Hardware: local (CPU, numpy only); wall-clock < 3 s; cost $0.

## Proposed target location in `stack/`

Add to **`stack/tanitad/eval/metrics.py`** (same module as LAL-v1), next to `compute_lal`:
- `LAL2_DROP_FRAC / LAL2_REF_FRAC / LAL2_HOLD` constants,
- `decel_onset_index(...)`, `compute_lal_v2(tel, ...)` (a thin wrapper reading `tel.ego_v`,
  `tel.hazard_los_flag`, `tel.timestamp_s`/`tel.dt`),
- extend `run_scenario_suite` to emit **both** `LAL_s` (v1, relabeled "reaction-onset latency,
  hard-brake") and `LAL_v2_s` ("anticipation lead, deceleration-onset") with directions.
Keep LAL-v1 (non-breaking; it is a valid *reaction-hardness* instrument, just not an anticipation one).
Add the 7 tests to `stack/tests/test_metrics.py`.

## Risk / rollback

- **Risk:** low. Pure additive numpy function + tests; no change to existing metrics, gate runner, or
  training path. `drop_frac`/`ref_frac`/`hold` are documented defaults; the free-cruise reference uses
  `max` over the first 30 % of the clip, which assumes the clip *starts* in free cruise (true for the
  SC-01 / Work-Zone-Phantom contract; document the assumption at the call site).
- **Rollback:** delete the three symbols + tests; LAL-v1 behaviour is untouched.

## Verdict (orchestrator writes here)

_pending triage_
