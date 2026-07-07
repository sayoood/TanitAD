# INTAKE — D1–D3 gate runner with instrument-doctrine gating

- **Package:** `Architecture & Inference/Implementation/incoming/2026-07-14-gate-runner-d1-d3/`
- **Author agent / date:** Architecture & Inference (Wednesday), 2026-07-14 (base commit `a731481`)
- **Proposed target:** `stack/tanitad/eval/gates.py` (+ `stack/tests/test_gates.py`); create the
  `stack/tanitad/eval/` package (`__init__.py` exporting `run_d1/run_d2/run_d3/gates_metrics_json`).
- **Hypothesis / WP served:** H1/H3/H13 · WP6 (eval suite / gate runner) · gates D1, D2, D3 · D-004 instrument doctrine

## What & why (≤10 lines)

`tanitad_gates.py` is the falsification harness for the three Phase-0 **decode** gates (Phase 0 Plan §4).
Each gate assembles its I1–I4 instrument rows **first** and can only be marked `passed` if it is first
`admissible` (instruments clear their bars) — a gate whose instruments fail is `BLOCKED`, never `FAIL`.
This is D-004 ("no architecture change may be motivated by a gate that has not passed its instrument
rows") made mechanical, and it is the concrete substrate for G-AI1 (no gate, no change). The runner
**composes** the existing `tanitad.instruments.checks` (I2/I3/I4) and the frozen `RidgeProbe` — it does
not reimplement them. It ships `ade_fde`, an I3-correct `split_by_episode`, the *vs global-pool* (D1) and
*probe_real vs probe_imag* (D3, A3) ablations, and an `extra_metrics` seam so **Thursday's** custom suite
(LAL/TMS/OKRI/CNCE/LOPS) plugs in without editing the runner. Motivating research + a first-class caveat:
the JEPA-WM planning ablation (arXiv 2512.24497) shows decode quality does *not* reliably predict planning
success → D1–D3 are labelled **necessary-not-sufficient**; closed-loop D4–D6 remain the arbiters. See
`Architecture & Inference/Research/2026-07-14-gate-runner-and-jepa-wm-deltas.md`.

## Evidence & tests

- Tests included: `tests/test_gates.py` — **13 passed / 1.58 s** on the author venv (py3.13 + torch cu128),
  no simulator, no trained model. Full stack suite after adding the package to path: **65 passed, 1 skipped**
  (unchanged — nothing written into `stack/`).
- Coverage proves BOTH branches of the doctrine, not just the happy path:
  - PASS path on controlled-linear data — D1 ADE@1s ≈ 1.6e-4 m < 1.0 (camera); grid readout beats a
    layout-destroying global pool (0.0002 m vs 3.97 m, A7); D2 dir-acc = 1.0 > 0.7 with imag-rel < 0.8;
    D3 imagined/oracle ADE@2s ratio ≈ 1.0 ≤ 1.5.
  - BLOCKED path — a batch-dependent `encode_fn` (the ALPS-4B BatchNorm incident) fails I2 → gate BLOCKED;
    missing I2 → BLOCKED; a garbage predictor (imag-rel ≫ 0.8) → D2 BLOCKED on I4.
  - Assembly: `gates_metrics_json` emits `instruments` before `gates` (protocol §6) and a PASS/FAIL/BLOCKED
    summary; `extra_metrics` hook merges.
  - End-to-end: a real `WorldModel(smoke_config)` runs through `run_d1`; its batch-free-norm encoder
    genuinely passes I2 (status ∈ {PASS, FAIL}, never BLOCKED) — the harness is exercised against the
    actual model, not only synthetic arrays.
- Instrument rows are real, not stubs: I2 = `i2_batch_consistency`, I3 = `i3_episode_split`, I4 =
  `i4_imag_relative` (all from `stack/tanitad/instruments/checks.py`). I1 = probe fit-R² sanity floor (0.9).

## Risk & rollback

- Blast radius if integrated: additive only — new `stack/tanitad/eval/` package + one test file. No existing
  module changes. Import surface: `tanitad.instruments.checks`, `tanitad.models.readout` (both stable).
- Thresholds are named constants (`D1_ADE_MAX`, `D2_DIR_ACC_MIN`, `D2_IMAG_REL_MAX`, `D3_RATIO_MAX`, …)
  sourced from Phase 0 Plan §4 — adjust in one place if the plan revises a bar.
- Coordination note for the orchestrator: Master Plan §3 assigns the *gate harness* to Benchmarks & Eval
  (Thursday) and the *stack/gates wiring* to Architecture. This package is deliberately the Architecture
  half (standard ADE/FDE + instrument gating + model wiring) with an explicit hook for Thursday's custom
  metrics; if Thursday has a parallel `eval/` skeleton in flight, integrate as `eval/gates.py` and let the
  custom-metric module import `run_d*`.
- Rollback: delete `stack/tanitad/eval/` and `stack/tests/test_gates.py`; no other file touched.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate-with-changes
- **Date / by:** 2026-07-08, MVP orchestrator (autonomous loop iteration 1)
- **Reason & notes:** Excellent instrument-doctrine mechanics (BLOCKED≠FAIL, instruments-first,
  both doctrine branches tested). Package predates D-017, so integration applied the conditioned
  changes: (1) **P4 forward-dynamics readout** added to run_d2 ([low-D state ⊕ action] → displacement;
  gate passes via P1 OR P4); (2) **imag-rel demoted** from D2 admissibility row to diagnostic metric
  (A13 — new test proves the "usable at imag-rel>1" case); (3) **I7 task-identity row** wired
  (fit/run fingerprint mismatch blocks); (4) I1 sanity floor applies to the ACTIVE readout path
  (either probe family). D3 keeps its I4 row (multi-step decode is a different claim). Import
  de-hacked. Standalone 13/13 pre-rework; full suite 89 passed / 1 sim-skip post-rework.
- **Integrated as:** `stack/tanitad/eval/gates.py` + `stack/tanitad/eval/__init__.py` +
  `stack/tests/test_gates.py` (see `intake(arch-inf)` commit, 2026-07-08)
