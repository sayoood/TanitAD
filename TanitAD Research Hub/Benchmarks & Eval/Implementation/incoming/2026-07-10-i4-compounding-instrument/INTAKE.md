# INTAKE — I4/D3 compounding instrument (hardened)

- **Discipline:** Benchmarks & Eval
- **Date:** 2026-07-10
- **Slug:** i4-compounding-instrument
- **Author:** Benchmarks & Eval agent (Thursday), independent-test role (Mission Plan; agent duty #5)

## What
`i4_compounding.py` — three artifact-immune companions to the D3/I4 relative-error readout used in
`stack/scripts/d3_decompose.py`:
1. `rel_triplet(pred, true, z_t)` → `{rel_k, abs_err_median, drift_median}` — the SAME rel_k, but
   numerator (absolute error) and denominator (persistence drift) are always exposed.
2. `compounding_ratio()` / `cr_from_predictions()` / `cr_from_rel()` → the **Compounding Ratio CR**
   (rollout error ÷ direct/teacher-forced error at a shared horizon). Denominator-free: the drift
   cancels because rollout and direct share the target and `z_t`.
3. `compounds(abs_err_by_k)` → a compounding verdict + log-log growth exponent taken from the
   **absolute-error curve**, never from the rel_k-vs-k slope.

Pure numpy, no torch dependency, standalone-testable. 7 analytic-ground-truth tests, all green.

## Why
Independent audit (`../../i4_horizon_normalization_audit/`, seed 20260710) of the D3 decomposition
(commits `9bbf4ca`/`c0b22b7`, ckpt step-14k). The audit found, with synthetic ground truth:
- **The "rel error FALLS with k ⇒ direct heads don't compound" read is a normalization artifact.**
  A model whose absolute error compounds *superlinearly* (err ∝ k¹·³) still produces a **falling**
  rel_k when drift ∝ k¹·⁵ (the highway near-constant-velocity prior) — reproduced exactly. The
  rel_k-vs-k slope is therefore invalid as a compounding readout. (The Wednesday note already
  flagged "highway normalization artifact"; this proves and quantifies it.)
- **The "recursion 2–4× worse than direct" finding is REAL** — it is denominator-free (shared
  `‖true₄−z_t‖`), i.e. exactly the accepted **Compounding Ratio** (SkyJEPA arXiv 2606.23444;
  Robotic World Model arXiv 2501.10100). Reproduced: CR_comma 4.00, CR_physicalai 3.72 (step-14k).
- **Cross-MODEL rel_k comparison is confounded.** Halving encoder drift inflates rel_k ~1.94× at
  identical absolute error → the K-step-arm headline "imag_rel 8.13→1.03 (1-step)" (base vs K-step,
  different encoders) can be off by up to ~2× from geometry alone. The honest, artifact-immune
  statement is the **within-arm CR**: base 3.90 → K-step 0.385 (<1 = the trained-for rollout path is
  now the strong one).

Net effect: no D3/gate decision should rest on the rel_k slope; CR + absolute-error curve are the
decision-grade instruments. This closes a silent-measurement-error class exactly like D-004's I2/I4
doctrine intends.

## Tests run
`pytest test_i4_compounding.py -q` → **7 passed in 0.25s** (venv `C:/Users/Admin/venvs/tanitad`).
Covers: triplet exposes num/den; falling-rel-despite-superlinear-compounding regression;
CR = abs-error ratio (denominator-free); reported step-14k / arm numbers reproduce as CR;
collapse-masquerade ~2× inflation; CR≈1 when rollout tracks direct; flat error not superlinear.

## Proposed target location
`stack/tanitad/eval/` (same package as `gates.py`, alongside the metric suite). Two integration hooks:
1. Have `scripts/d3_decompose.py:analyze()` emit `{rel_k, abs_err_k, drift_k}` per horizon and a top
   level `CR` (= `recursive_1step_x4 / direct_k4`, denominator-free) — one-line change per horizon.
2. Add CR to the D3 gate runner's `extra_metrics` so the gate REPORT carries the artifact-immune
   compounding number next to I4.

## Risk / rollback
Additive only — a new module + reporting fields; changes no training code, no existing metric value,
no gate threshold. Rollback = delete the module and revert the two `d3_decompose` reporting lines.
Risk that `d3_decompose` currently discards per-window absolute errors: the triplet needs
`analyze()` to keep `num`/`den` (currently only their ratio is medianed) — a trivial refactor
included as the proposed hook, not yet applied to `stack/` (hub boundary D-011).

## Orchestrator verdict
_(to be filled by MVP orchestrator triage)_
