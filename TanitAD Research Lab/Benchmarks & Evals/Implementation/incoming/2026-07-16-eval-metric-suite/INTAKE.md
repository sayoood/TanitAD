# INTAKE — TanitAD custom evaluation metric suite (LAL/TMS/OKRI/CNCE/LOPS + trajectory seam)

- **Package:** `Benchmarks & Eval/Implementation/incoming/2026-07-16-eval-metric-suite/`
- **Author agent / date:** Benchmarks & Eval (Thursday), 2026-07-16 (base commit `ff89194`)
- **Proposed target:** `stack/tanitad/eval/metrics.py` (+ `stack/tests/test_metrics.py`) — the **same**
  `stack/tanitad/eval/` package the D1–D3 gate runner (intake 2026-07-14) proposes. On integration,
  `eval/__init__.py` should additionally export `run_scenario_suite`, `ScenarioTelemetry`,
  `trajectory_extra_metrics`, and the five `compute_*` functions.
- **Hypothesis / WP served:** H15 (imagination self-monitoring → LOPS/OKRI), H1/H5 (efficiency moat → CNCE),
  WP6 (eval suite) · Phase 0 Plan §4 exit item G0.6 ("custom metric suite live") · gate G-B2.

## What & why (≤10 lines)

The five "Deep Think 14" custom metrics that cover edges the recognized KPIs miss — plus a trajectory
seam that plugs into the gate runner. **LAL** (anticipation before line-of-sight), **TMS** (control
smoothness under partial observability), **OKRI** (kinetic energy into blind spots), **CNCE**
(safe-progress per compute), **LOPS** (latent tracking of a fully-occluded agent). Motivation is
external and current: the 2026 cross-benchmark study (arXiv 2605.00066) finds displacement metrics have
*no reliable correlation* with closed-loop driving score and NAVSIM EPDMS correlates *non-monotonically*
with Bench2Drive DS — so recognizable open-loop numbers cannot, alone, prove our edge; these metrics
target the edge directly. Each formula is reproduced in its docstring from
`Ressources/Deep Think Analysis/Deep Think 14.md`. See research note
`Benchmarks & Eval/Research/2026-07-16-benchmark-ecosystem-and-metric-suite.md`.

**Composition, not collision.** `trajectory_extra_metrics()` returns the `{name: callable}` dict the
gate runner's `extra_metrics=` hook expects (each `(pred_xy, true_xy) -> float`), so the custom suite
merges into D1–D3 reports **without either module importing the other**. Verified live against the
Wednesday `tanitad_gates.run_d1` (see below). The five headline metrics operate on *scenario telemetry*
(a different domain than trajectory tensors) and are exposed via `run_scenario_suite` / `ScenarioTelemetry`.

## Evidence & tests

- Tests included: `tests/test_metrics.py` — **22 passed / 1.88 s** on the author venv (py3.13 + torch
  cu128), no simulator, no trained model. Standalone: `pytest tests/` needs only numpy (+ torch for the
  one seam test, `importorskip`-guarded).
- **G-B2 — every metric checked against analytically-known ground truth** (derivation in each test's
  comment): LAL +0.4 s (LoS@5.0, brake@4.6) / −0.3 s reactive / −999 sentinel / 0 no-hazard;
  TMS = 1.0 for zero-jerk-zero-steer and 1/11 for a known integral, monotone in jerk; OKRI = 29.4118
  for KE=75000/d_blind=5.1 over 2 s, 0 with no blind spot, lower when slower; CNCE = 100.0 (dist 20 /
  (0.05 s · 4B)), ×e⁻² per collision, lower for a 15B model; LOPS = 1.0 perfect / e⁻¹ at 2 m error /
  0.0 E2E baseline / ignores unoccluded+NaN rows.
- **Seam verified live (not just asserted):** `run_d1(..., extra_metrics=trajectory_extra_metrics(...))`
  on the runner's own controlled-linear fixture merged `ade`/`rmse`/`miss_rate` into the D1 `metrics`
  block (rmse ≈ 7.7e-5 on decodable data). Status came back BLOCKED because I2 was deliberately withheld
  — i.e. the doctrine held (no instrument row → no claim) *and* the custom metrics still merged. Merge is
  independent of PASS/BLOCKED, as intended.
- **Honest scope (P8):** LAL/OKRI/LOPS need closed-loop occluder-scenario telemetry (Ghost Cut-Through /
  Blind Creep / Choke Weave). Per **D-014** (landed mid-run; MetaDrive retired) the closed-loop substrate
  is now **CARLA-on-pod (W31–32)**; the ungated synthetic corpora (`PhysicalAI-WorldModel-Synthetic`,
  `Cosmos-Drive-Dreams`) can drive a cheaper pre-rendered first pass now. The suite is **sim-agnostic** —
  it consumes `ScenarioTelemetry` columns, not any simulator API, so D-014 does not touch this package.
  **No metric is claimed on a real TanitAD run here** — only synthetic fixtures with known answers.

> ⏹ **PARTIALLY CLOSED 2026-08-16 — 2 of the 5 metrics HAVE since been measured for real; 3 are
> still renderer-gated; and the named substrate has moved on. Check this before commissioning any
> "measure the custom suite" work.**
>
> **CLEARED — TMS and CNCE are no longer synthetic-only.** Evidence (MEASURED):
> `…/Benchmarks & Eval/…/incoming/2026-07-24-traffic-light-scenario-metric/real_tms_cnce.json`,
> `evidence_class: "MEASURED (real comma2k19 val telemetry + real base250cam architecture)"`,
> 30 episodes, `params_billions 0.2628`, `decision_tick_p50_ms 14.331` → `TMS_expert_log` median
> **0.0435** and `CNCE` median **210 551**. ⚠️ Read its own two caveats with the numbers: TMS there
> scores the **expert log's** smoothness (a reference band, P8 — not our policy), and CNCE's
> `collisions = 0` holds **by log-replay construction**, so it is an *architecture* efficiency number,
> not a driving one.
>
> ✅ **STILL TRUE — LAL / OKRI / LOPS remain un-measured and renderer-gated.** The same artifact says
> it verbatim: *"LAL/OKRI/LOPS/TLC need rendered occlusion/signal geometry … renderer-gated, NOT
> computed here (no telemetry, no number)."* Independently corroborated by the scenario side: the only
> runner is still `stack/scripts/scenario_suite_dryrun.py` (a **dryrun**), and
> `stack/tanitad/eval/scenarios/registry.py` registers only `work_zone_phantom` +
> `traffic_light_red/green`.
>
> ⚠️ **SUPERSEDED — "the closed-loop substrate is now CARLA-on-pod (W31–32)" is no longer the whole
> picture, and CARLA-on-pod was never fired.** What actually exists today:
> - **T1, the action-closed-loop tier, IS provisioned and is now the PRIMARY tier for any capability
>   claim** — `taniteval/tools/t1_eval.py`, per `Project Steering/EVAL_DOCTRINE.md` (**BINDING
>   2026-08-09**). It needs no simulator: the predictor consumes the planner's **own** actions with
>   perception fixed at t0.
> - **Re-perception (T2) moved from CARLA to AlpaSim/NuRec.**
>   `…/Architecture & Inference/…/incoming/2026-08-06-mpc-planner-design/MPC_WM_DESIGN.md:99` —
>   *"re-perception closed loop remains the AlpaSim/NuRec work item"* — and AlpaSim has real banked
>   output at `stack/experiments/alpasim-gsplat/results/` (`closedloop-hq-render`, `cutin`,
>   `2026-08-03-rolling-shutter`, `metrics_flagship_obj_vs_empty.json`).
> - `Project Steering/ROADMAP.md:79` still names CARLA-on-pod as X4's substrate and marks it
>   *"Prototype … camera-driven ego needs the graphics-pod recipe, **not fired yet**"*. ⛔ Roadmap text
>   is Project-Steering-owned and was deliberately not edited by this sweep — flagged, not resolved.
>
> ⇒ The package's own design claim survives all of this untouched: the suite is **sim-agnostic**
> (it consumes `ScenarioTelemetry` columns, not a simulator API), so the substrate churn costs it
> nothing. Swept by the 2026-08-16 stale-blocker sweep.

## Risk & rollback

- Blast radius if integrated: additive only — one module + one test file in the `eval/` package. No
  existing module changes. Dependencies: numpy only (torch imported lazily solely to accept tensors on
  the seam). Local `_trapz` avoids the numpy-2.0 `trapz→trapezoid` rename.
- Coordination with the 2026-07-14 gate-runner package: both target `stack/tanitad/eval/`. No import cycle
  (the seam passes callables at call-time). Recommend integrating the gate runner first, then this module,
  then have `eval/__init__.py` export both surfaces. Thresholds/weights are named constants at module top
  (`JERK_BRAKE_THRESHOLD`, `TMS_ALPHA/BETA`, `OKRI_APPROACH_M`, `CNCE_LAMBDA`, `LOPS_GAMMA`, …) sourced
  from Deep Think 14 — tune in one place.
- Rollback: delete `stack/tanitad/eval/metrics.py` and `stack/tests/test_metrics.py`; no other file touched.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate
- **Date / by:** 2026-07-08 (overnight), MVP orchestrator (autonomous loop iteration 3)
- **Reason & notes:** Exemplary G-B2 discipline — every metric validated against analytically-derived
  ground truth; the gate-runner seam verified live including the doctrine case (metrics merge while the
  gate stays BLOCKED); sim-agnostic telemetry consistent with D-014; honest no-claims scope. Only
  change: test import de-hacked. Integrated AFTER the gate runner per the package's own coordination
  note; `eval/__init__.py` exports both surfaces. Full suite 119 passed / 1 sim-skip. G0.6 exit item
  ("custom metric suite live") is now satisfied on the tooling side — first real numbers come with the
  checkpoint evaluation and, for LAL/OKRI/LOPS, the CARLA-on-pod scenario telemetry (W31–32).
- **Integrated as:** `stack/tanitad/eval/metrics.py` + `stack/tests/test_metrics.py`
  (see `intake(bench-eval)` commit, 2026-07-08)
