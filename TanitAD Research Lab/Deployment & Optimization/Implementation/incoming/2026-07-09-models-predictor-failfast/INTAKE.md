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

- **Verdict:** **integrate-with-changes**
- **Date / by:** 2026-08-03 · orchestrator daily sweep (age at adjudication: **25 days**)
- **Reason & notes:**
  - **Re-verified here, not inherited:** `tests/` re-run on this box → **8 passed / 1.62 s**
    (`venvs/tanitad`), including `test_export_safe` (ONNX opset 17 still clean; the
    TracerWarnings are the expected constant-fold, not errors).
  - **The defect is still present at tip and still un-fixed after 25 days.** The bare
    `assert w == self.cfg.window` is live — it has merely MOVED from line 73 to **line 101**
    (`grep` on tip). The package's line reference was stale; the code it describes was not.
    This is the silent-wrong-data class, on the batch-1 streaming hot path, ×K=9 in
    imagine-and-select.
  - **Changes made on integration** (all forced by the call site, none by the validator):
    1. `state_dim` is a **ctor argument, not a `cfg` field** (`__init__(cfg, state_dim, …)`), so
       there was nothing to pass the validator. Stored as `self.state_dim = state_dim` — a plain
       int, **not** a buffer or Parameter, so **`state_dict` is byte-identical** and every banked
       checkpoint still loads. (This matters: the ctor's own comment already guards
       "byte-identical state_dict" for the ReZero lever.)
    2. Import added at module scope; `assert` replaced by the validator call, which now runs
       **before** the `b, w, _ = states.shape` unpack rather than after it.
    3. The `-O` tripwire was strengthened. As shipped it parsed only `validate.py` — but the
       failure mode being guarded is *an `assert` in the predictor*, and the shipped test could
       not have caught the assert coming back. It now ALSO walks
       `predictor.py`'s `forward()` bodies and fails if any `assert` statement reappears there.
       (Resolved via `_validate.__file__` / `predictor.__file__` rather than a relative path, so
       the tripwire cannot silently pass by reading the wrong file.)
  - **Scope deliberately NOT widened.** The INTAKE offers "apply the SAME guard to `tactical_pred`
    if desired" — declined for this sweep. Same class, different module, and a 25-day-old package
    should land as its author measured it; the tactical call site is filed as its own item.
- **Integrated as:**
  - `stack/tanitad/models/_validate.py` (as proposed)
  - `stack/tanitad/models/predictor.py` — `self.state_dim`, import, call site (assert removed)
  - `stack/tests/test_predictor_failfast.py` (8 tests + the strengthened call-site tripwire)
