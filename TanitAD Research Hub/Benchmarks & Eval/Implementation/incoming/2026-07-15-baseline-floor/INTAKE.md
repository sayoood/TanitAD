# INTAKE — Stratified trivial-baseline floor (the honest denominator)

- **Discipline:** Benchmarks & Eval
- **Date:** 2026-07-15
- **Slug:** `2026-07-15-baseline-floor`
- **Author:** Benchmarks & Eval agent (Thursday, scheduled; dev-box CPU, $0)
- **Status:** metric + first numbers — **awaiting orchestrator triage** (read-only consumer; does not touch the running contract)

## What

A tested kinematic-baseline floor for open-loop ego-motion eval — the *denominator*
the driving-capability diagnostic divides the model's ADE by.

1. **`baseline_predictors.py`** — numpy-only module: three causal kinematic nulls
   (constant-velocity, go-straight, **CTRV** constant-turn-rate-velocity), each
   predicting future waypoints in the ego frame at the anchor; `skill_score` =
   model_ADE ÷ best-of-3 baseline; curvature/speed stratification with a **speed
   gate** (v < 2 m/s → `standstill`, because κ = yaw_rate/v is singular as v → 0).
2. **`test_baseline_predictors.py`** — **8 analytic-ground-truth tests** (straight →
   all exact; arc → CTRV exact, CV fails; ego-frame transform; crab-motion; standstill
   yaw-jitter gated; skill_score arithmetic; record contract).
3. **`run_baseline_floor.py`** + **`results_baseline_floor.json`** — cross-corpus run
   on 13 Cosmos clips (1 022 anchors) + comma2k19-val 90 eps (25 110 anchors).

Full write-up: `Research/2026-07-15-baseline-floor-honest-denominator.md`.

## Why

The diagnostic framework's headline ("model is 10–15× worse than constant-velocity
everywhere") uses **one** trivial baseline (CV) on comma highway. CV is the weakest
kinematic null on curves; the community (NAVSIM v2, arXiv 2506.04218) uses CV as a
*stratum-sensitive* triviality filter. Measured result: the honest best-of-3 floor is
**0.056–0.06 m @1s** (CTRV-dominated), 1.4–4.6× tighter than CV-only → the model gap
is ~115× the floor, not 10–15×. Direction of the framework verdict is reinforced;
the denominator is corrected and standardized.

## Evidence / tests run

- `pytest test_baseline_predictors.py -q` → **8 passed / 0.16 s** (venv
  `C:\Users\Admin\venvs\tanitad`, numpy 2.5.1). No model, no simulator, no torch.
- End-to-end run on **real local data**: comma2k19-val (`eval/comma2k19-val-…`) +
  Cosmos `cosmos_bench3` — 26 132 anchors total; results in `results_baseline_floor.json`.
- Hardware: local CPU, **$0**, no pod / no CARLA / no training cache touched.

## Proposed target location in `stack/`

`stack/tanitad/eval/baselines.py` (sibling of the integrated `metrics.py` and
`gates.py`). `skill_score` is designed to plug into the gate runner's `extra_metrics`
seam (Architecture 2026-07-14) — the D1 gate divides model_ADE by the per-stratum
best-of-3 floor instead of a single CV scalar. If a tests home is wanted, the analytic
tests port as-is (no fixtures / no real bytes required).

## Risk / rollback

- **Risk:** low. Read-only consumer; no metric/gate/training change until the
  orchestrator wires `skill_score` into the gate runner. The baselines use privileged
  GT ego-state (denominator, not a model competitor — see note §Caveats in the write-up).
- The hardcoded per-stratum denominators (straight 0.056 / gentle 0.059 / sharp 0.164 @1s,
  comma-hwy) are corpus-specific — recompute per eval corpus, do not treat as universal.
- **Rollback:** delete this folder; nothing depends on it.

## Verdict (orchestrator writes here)

- **Verdict:** _pending_
- **Date / by:**
- **Reason:**
