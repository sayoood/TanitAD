# INTAKE — Robustness metric suite, first pass on ungated synthetic corpora (backlog #3)

- **Discipline:** Benchmarks & Eval
- **Date:** 2026-07-13
- **Slug:** `2026-07-13-cosmos-robustness-first-pass`
- **Author:** Benchmarks & Eval (Sayed-directed pod-independent task; dev-box / 4060, $0)
- **Status:** glue + first numbers — **awaiting orchestrator triage** (does NOT touch the running contract)

## What

The "available NOW, no simulator" first pass named in STATE backlog #3: exercise the custom
robustness suite (LAL / TMS / OKRI / CNCE / LOPS, `stack/tanitad/eval/metrics.py`) on the **ungated
NVIDIA synthetic corpora**, before the CARLA-on-pod closed loop lands (D-014).

1. **`cosmos_telemetry.py`** — glue that derives the `ScenarioTelemetry` contract from
   Cosmos-Drive-Dreams per-clip RDS-HQ annotations (ego `vehicle_pose` + `all_object_info` 3D boxes +
   `pinhole_intrinsic`). Ego kinematics **reuse** `stack/tanitad/data/cosmos_drive.poses_to_signals`;
   occlusion geometry (blind-spot distance, occlusion / line-of-sight flags, hidden-agent GT track) is
   a bird's-eye ray test against the 3D boxes. Pixel-free → **no 43 GB video shard needed**.
2. **`acquire_cosmos_sample.py`** — pulls a small annotation-only sample over the HF tree API
   (ungated; TLS via `tanitad.keys.enable_tls`).
3. **`results.json` / `diagnostics.json`** — the 13-clip first-pass numbers + the TMS/jerk-noise and
   LOPS-oracle diagnostics.
4. **`worldmodel_structure.json`** — evidence that `PhysicalAI-WorldModel-Synthetic-Autonomous-
   Driving-Scenarios` is **video + VLM-caption only** (no pose / no boxes) → cannot feed the geometric
   suite data-only (the second corpus in backlog #3; documented gap, not a loader to write yet).

Full write-up: `Research/2026-07-13-backlog3-synthetic-corpora-first-pass.md`.

## Why

Backlog #3 asks for a *cheaper first pass* that (a) validates the metric pipeline end-to-end on real
downloaded data and (b) produces first robustness numbers without a pod or CARLA. Prefer data-only
metrics; label anything model-dependent preliminary.

## Evidence / tests run

- Existing contract intact: `pytest tests/test_metrics.py test_metric_dynamics.py
  test_scenario_suite_wiring.py test_work_zone_phantom.py` → **40 passed** (venv
  `C:\Users\Admin\venvs\tanitad`, numpy 2.5.1).
- Pipeline end-to-end on **13 real Cosmos-Drive-Dreams clips** (annotation tars, ~200 MB total, no
  video): every clip yields a hazard + occlusion track; occlusion detector fires on **13/13**.
- **LOPS path validated**: perfect-perception oracle (wm = gt + N(0,0.3)) → mean **0.844** vs the
  analytic E[exp(-0.5·|N(0,0.3)|₂)] = **0.8325** (|diff| 0.011), the same constant the 2026-07-09
  SC-01 audit recomputed.
- Hardware: local (CPU, numpy + tarfile), $0, no pod / no CARLA / no training cache touched.

## Proposed target location in `stack/`

`stack/scripts/cosmos_telemetry.py` (sibling of `cosmos_pairs.py`, `scenario_suite_dryrun.py`).
**Nothing to change in `stack/tanitad/eval/metrics.py`** — this is a *consumer* of the contract. If a
tests home is wanted, add a fixture-based test that mocks 3-frame pose+object tars (the geometry is
unit-testable without real bytes, like `work_zone_phantom`).

## Risk / rollback

- **Risk:** low. Read-only consumer; no metric/gate/training change. The occlusion geometry is a
  documented bird's-eye approximation (center-bearing occlusion, half-diagonal footprint) — good
  enough to *exercise* the suite, not a perception-grade occlusion oracle.
- The `latency_ms` / `params_billions` inputs are a **labelled TanitAD-4B stub** (CNCE only); `LOPS`
  data-only is the honest 0.0 baseline. Neither is a model claim.

> ⏹ **PARTIALLY CLOSED 2026-08-16 — the `latency_ms` / `params_billions` STUB CAN NOW BE REPLACED
> WITH MEASURED VALUES. Do not re-derive them; they exist.**
> Evidence (MEASURED): `…/Benchmarks & Eval/…/incoming/2026-07-24-traffic-light-scenario-metric/real_tms_cnce.json`,
> `evidence_class: "MEASURED (real comma2k19 val telemetry + real base250cam architecture)"` —
> `params_billions` **0.2628** (not a labelled 4B) and a real decision tick:
> `encode_1frame_p50_ms` 9.273 + `select_K9_p50_ms` 5.058 → `decision_tick_p50_ms` **14.331**, on an
> RTX 4060 over 30 episodes. That artifact's own note explains why the pair is admissible here:
> *"latency+params are weight-independent … -> a real ARCHITECTURE efficiency number."*
> ⚠️ **Carry the hardware stamp with them.** These are **RTX 4060** numbers; every committed
> `taniteval.efficiency` artifact records `env.gpu = "NVIDIA A40"`, and a separate NVIDIA Thor row
> exists. Substituting across hosts is the exact defect
> `…/Data Engineering/…/2026-08-04-instrument-durability/INTAKE.md` §3.2 refuses to commit.
> ✅ **STILL TRUE:** `LOPS` remains the honest 0.0 data-only baseline, and the proposed target
> `stack/scripts/cosmos_telemetry.py` is **ABSENT** (two probes: file not present; `grep -rn
> "cosmos_telemetry" --include="*.py" stack` → zero hits) — this package is still unintegrated and its
> ORCHESTRATOR VERDICT is still `_pending_`. Swept by the 2026-08-16 stale-blocker sweep.
- **Rollback:** delete this folder; nothing depends on it.

## Verdict (orchestrator writes here)

- **Verdict:** _pending_
- **Date / by:**
- **Reason:**
