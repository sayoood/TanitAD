# INTAKE — operative-predictor fail-fast input validation (models review #2)

- **Package:** `Production & Optimization/Implementation/incoming/2026-07-09-models-predictor-failfast/`
- **Author agent / date:** production-optimization-agent (Saturday run #2), 2026-07-09
- **Proposed target:** `stack/tanitad/models/predictor.py` — replace the `assert`
  at `OperativePredictor.forward` (line 73) with a call to
  `validate_operative_inputs(...)`; add `validate.py` as
  `stack/tanitad/models/_validate.py` (or fold the function into `predictor.py`).
- **Hypothesis / WP served:** production hardening (P3 / D-020 §3); no hypothesis
  claim. Compliance review #2 of the `tanitad/models/` cluster.

## What & why (≤10 lines)

The operative predictor is the batch-1 streaming hot path (runs every decision
tick, ×K=9 in imagine-and-select). Its only input guard is
`assert w == self.cfg.window`. Two measured production failure modes:
1. **`assert` is stripped under `python -O`/`-OO`.** Measured: a states/actions
   pair of window **W-1** re-aligns on every axis (pos slice, causal mask, FiLM
   cond) and runs SILENTLY — the predictor emits an output for a window it was
   never configured for (silent-wrong-data class, same as the epcache/cosmos
   bugs this stream already fixed).
2. **Wrong `state_dim`/`action_dim` raise a cryptic** `RuntimeError: mat1 and
   mat2 shapes cannot be multiplied (8x2047 and 2048x768)` from inside a Linear
   — no signal that the module *contract* was violated.
`validate_operative_inputs` gives named-axis, `-O`-proof `ValueError`s. Pure
`tensor.shape[i]` int comparisons → constant-fold on static-shape ONNX export
(same as the assert they replace, which exported clean at opset 17/18 on
2026-07-08) → **export unaffected, proven by `test_export_safe`.**

## Evidence & tests

- Tests: `tests/test_predictor_failfast.py` — **8 passed** on the author machine
  (RTX 4060 dev box, py3.13, torch 2.11). Covers: valid no-op; wrong window
  (was AssertionError→now ValueError); wrong state_dim / action_dim (was cryptic
  matmul RuntimeError→now named ValueError); states/actions window disagreement;
  non-3-D; an `ast`-based tripwire that the guard uses **no `assert` statement**;
  and `test_export_safe` — a toy predictor calling the validator still exports to
  ONNX opset 17 (shape checks constant-fold; TracerWarnings are the expected
  fold, not errors).
- Measured failure modes (current stack behavior, this run):
  `wrong_window→AssertionError`, `wrong_state_dim/action_dim→RuntimeError(matmul)`,
  valid→ok. See research note `2026-07-09-half-precision-and-models-failfast.md`.

## Risk & rollback

- Blast radius: one module (`predictor.py`), one new tiny helper. The guard is a
  strict superset of the current assert (still rejects wrong window; now also
  rejects wrong dims and survives `-O`). No behavior change on valid input
  (no-op) → training/eval/export paths unchanged; full stack suite should stay
  green. Apply the SAME guard to `tactical_pred` (same class) if desired.
- Rollback: revert the one-line call site to the original `assert`; delete
  `_validate.py`. No state, no data, no config touched.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:** <...>
- **Reason & notes:** <...>
- **Integrated as:** <commit hash / stack path> (if applicable)
