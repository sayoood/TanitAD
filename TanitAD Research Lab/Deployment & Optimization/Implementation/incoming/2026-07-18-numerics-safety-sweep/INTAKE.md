# INTAKE — numerics-safety regression guard (learned-output exp/log/div class)

- **Package:** `Production & Optimization/Implementation/incoming/2026-07-18-numerics-safety-sweep/`
- **Author agent / date:** Production & Optimization (Saturday), 2026-07-18
- **Proposed target:** `stack/tests/test_numerics_safety.py` (test-only; NO source change)
- **Hypothesis / WP served:** production-hardening (F-5/6/7 silent-NaN class); guards H15 (imagination NLL / OKRI-LOPS sigma export)

## What & why (≤10 lines)

The 2026-07-17 `imagination_nll` NaN (unclamped `exp(-logvar)` → inf grads, live in the
flagship trainer) was one instance of a **class**: an unbounded `exp`/`log`/division on a
*learned* or *data* value that silently NaN-a-training-run or NaN-a-metric. Fleet directive
P0 #4 asked to "sweep the stack for other unbounded exp/log/div on learned outputs — one
sweep closes the class." I did the grep-sweep of `stack/tanitad` (see the audit table in the
test docstring and the research note): **every such site is already guarded** — clamp at the
head, `clamp_min`, a count-gate, or bounded-by-construction (negative-exponent Gaussian
kernels). No new unguarded site. The class is closed. This package converts that manual audit
into the **executable regression guard** that keeps it closed on future merges. Research note:
`Production & Optimization/Research/2026-07-18-predictor-cudagraph-and-numerics-sweep.md`.

## Evidence & tests

- Tests included: `tests/test_numerics_safety.py` — **11 passed in 1.47 s** (venv `tanitad`, RTX 4060 box).
- Genuine failing-then-passing witnesses (fail on pre-fix code, pass on current stack):
  - `test_imagination_nll_finite_at_extreme_logvar` / `..._witness_unclamped_would_overflow`:
    the OLD path `0.5*(exp(-(-100))*err2 + (-100))` → **inf** (asserted); the guarded
    `imagination_nll` → finite. `..._grads_finite...` guards the inf-grad mode.
  - `test_imagination_field_head_bounds_logvar`: head clamps logvar ∈ [-10,10] at input scale
    up to 500× → the `.exp()` in the OKRI/LOPS sigma export (`replay/arms.py:284`) stays finite.
  - `test_feature_ood_score_finite_before_two_samples` / `..._on_zero_variance`: the `sum/count`
    division in `refs/refb.py:366` is gated by `count<2 → zeros` and `var.clamp_min(eps)`.
  - Bounded-by-construction sites still guarded as a tripwire: `sigreg`/`epps_pulley` (degenerate
    + 1e4-scale input finite), `spectral.effective_rank/spectral_tail/optimal_k/energy_knee`
    (zero/single-mode spectrum finite), behaviour-preserving check that the clamp is a no-op in-range.

## Risk & rollback

- Blast radius if integrated: **test-only** — adds `stack/tests/test_numerics_safety.py`, imports
  existing public symbols (`imagination_nll`, `ImaginationField`, `FeatureOOD`, `SigReg`,
  `epps_pulley`, spectral helpers). No production code path touched; cannot change model behaviour.
- Rollback: delete the test file. Nothing else references it.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
