# INTAKE — Work-Zone Phantom weak-spot eval scenario (W-01)

- **Package:** `Opponent Analyzer/Implementation/incoming/2026-07-17-work-zone-phantom-scenario/`
- **Author agent / date:** Opponent Analyzer, 2026-07-17
- **Proposed target:** the weak-spot eval scenario set (CARLA-on-pod harness, D-014) — pairs with
  `Benchmarks & Eval/.../2026-07-16-eval-metric-suite/tanitad_metrics.py` (`scenario_metrics`). Likely
  `stack/tanitad/eval/scenarios/work_zone_phantom.py` once the eval-scenario home is created; Thursday
  (Benchmarks & Eval) owns the wiring.
- **Hypothesis / WP served:** H6 (weak-spot corpus) → H15 (imagine changed/unobserved area), H9
  (rule/closure compliance), H1 (fallback). Gate: D9 (hidden-sector) + a future H9 violation-rate metric.

## What & why (≤10 lines)
Converts Opponent Analyzer weakness **W-01** into a repeatable eval scenario. On 2026-06-18 Waymo
recalled 3,871 robotaxis for driving into freeway **construction zones** (unrecognized ramp-closure
signs; drove between lane-closure cones). This package specifies a **construction-zone scenario** —
ramp-closure sign + cone taper + closed lane + an **occluded merging actor** behind the taper — and a
**synthetic-telemetry design oracle** that emits the exact `ScenarioTelemetry` contract of the metric
suite for two archetypal policies (pixel-reactive vs world-model). It also adds a scenario-specific
`closure_incursion_m` signal (metres driven into the closed lane) as the seed of an H9 rule-compliance /
violation-rate metric. This is the Opponent Analyzer **monthly scenario feed** (agent-file duty #3).
Research: `../../Research/2026-07-17-opponent-sweep-w2.md`; catalog entry W-01 in
`../../Research/WEAKNESS_CATALOG.md`.

## Evidence & tests
- Tests: `tests/test_work_zone_phantom.py` — **9 passed in 0.15 s** on the author machine
  (`C:\Users\Admin\venvs\tanitad`, py3.13). numpy + pytest only; **no simulator, no cross-package import**.
- What the tests prove: (1) the emitted telemetry matches the `ScenarioTelemetry` field contract
  (names/shapes) + carries the `closure_incursion_m` compliance signal; (2) the scenario is
  **discriminative** — the world-model policy brakes before line-of-sight (LAL>0), carries less kinetic
  energy into the blind edge (OKRI lower), holds a latent hidden-actor estimate under occlusion (LOPS>0),
  and does not enter the closed lane, while the reactive baseline fails each. These are **design-oracle**
  assertions (P8: NOT a claim about our real model — real numbers come from the pod rollout).
- Not validated here: the live CARLA build (`carla_recipe()` → CARLA blueprints/triggers) — the explicit
  next step, gated on the CARLA-on-pod harness (Tools&DevEnv W31–32).

## Risk & rollback
- Blast radius: additive; a new self-contained scenario module + test. No change to existing stack or
  metric code. When integrated, it imports nothing from the suite — it *produces* telemetry the suite
  *consumes*, so the two evolve independently against the documented field contract.
- Rollback: delete the package / target file; nothing depends on it upstream.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)
- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
