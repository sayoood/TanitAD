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

> ⚠️ **RE-CONFIRMED STILL GATED 2026-08-16 — but the GATE ITSELF MOVED, so do not wait on W31–32.**
> The scenario module IS integrated (`stack/tanitad/eval/scenarios/work_zone_phantom.py`, registered
> as `SCENARIO_REGISTRY["work_zone_phantom"]` at `registry.py:59`), and it is still the only
> non-traffic-light scenario in the registry. What has NOT happened is the live render — and the
> substrate named here was never fired:
> - `Project Steering/ROADMAP.md:79` still lists CARLA-on-pod as X4's substrate and marks it
>   *"Prototype … camera-driven ego needs the graphics-pod recipe, **not fired yet**"*.
> - The re-perception line moved to **AlpaSim/NuRec** —
>   `…/Architecture & Inference/…/2026-08-06-mpc-planner-design/MPC_WM_DESIGN.md:99`: *"re-perception
>   closed loop remains the AlpaSim/NuRec work item"* — and AlpaSim has real banked renders at
>   `stack/experiments/alpasim-gsplat/results/` (`closedloop-hq-render/`, `cutin/`,
>   `2026-08-03-rolling-shutter/`).
> - Meanwhile the **action-closed-loop tier T1 became available and BINDING** without any renderer:
>   `taniteval/tools/t1_eval.py`, `Project Steering/EVAL_DOCTRINE.md` (2026-08-09).
> ⇒ Anyone picking this up should re-target `carla_recipe()` at the AlpaSim/NuRec path (or score what
> it can under T1) rather than waiting for a CARLA-on-pod harness that has not been provisioned in
> ~30 days. ⛔ Roadmap text is Project-Steering-owned and was deliberately not edited by this sweep.
> Swept by the 2026-08-16 stale-blocker sweep.

## Risk & rollback
- Blast radius: additive; a new self-contained scenario module + test. No change to existing stack or
  metric code. When integrated, it imports nothing from the suite — it *produces* telemetry the suite
  *consumes*, so the two evolve independently against the documented field contract.
- Rollback: delete the package / target file; nothing depends on it upstream.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)
- **Verdict:** integrate
- **Date / by:** 2026-07-08 (overnight), MVP orchestrator (autonomous loop iteration 4)
- **Reason & notes:** The H6 pipeline working end-to-end: real competitor failure (Waymo W-01
  construction-zone recall) → catalog entry → repeatable scenario + telemetry oracle + the
  `closure_incursion_m` seed of the H9 violation metric. Design-oracle honesty preserved (no claims
  about our model). Created the `eval/scenarios/` home; test import de-hacked (from-import variant).
  Full suite 137 passed / 1 sim-skip. Live CARLA build stays gated on the W31–32 harness as documented.
- **Integrated as:** `stack/tanitad/eval/scenarios/work_zone_phantom.py` + test
  (see `intake(opponent)` commit, 2026-07-08)
