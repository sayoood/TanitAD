# INTAKE — imagination-field logvar clamp (models review #3, numerics)

- **Package:** `Production & Optimization/Implementation/incoming/2026-07-17-imagination-logvar-clamp/`
- **Author agent / date:** production-optimization-agent (Saturday run #3), 2026-07-17
- **Proposed target:** `stack/tanitad/models/imagination.py` — two minimal edits:
  1. `ImaginationField.forward` (line 123): bound the head output —
     `return x, clamp_logvar(self.logvar_head(x).squeeze(-1))`.
  2. `imagination_nll` (line 135): clamp defensively before `exp(-logvar)` —
     `logvar = clamp_logvar(logvar)` as the first line.
  Add `clamp_logvar` + `LOGVAR_MIN/MAX` from `bounded_logvar.py` into
  `imagination.py` (they are 4 lines; no new module needed on merge).
- **Hypothesis / WP served:** production hardening (P3 / D-020 §3), H15 numerics.
  No hypothesis claim. Compliance review #3 of `tanitad/models/imagination.py`.

## What & why (≤12 lines)

The H15 imagination field's per-cell `logvar_head` is an **unbounded** `nn.Linear`
whose output hits two un-clamped `exp()` sites:
1. **`imagination_nll` (`imagination.py:135`)** — `exp(-logvar)`. When the head
   goes over-confident (logvar strongly negative — routine early in training and
   on OOD/occluded sectors) it **overflows fp32 at logvar < -88.7 → +inf loss →
   NaN gradients**. MEASURED this run: `logvar=-100 → loss=inf` (witness test).
   This is a LIVE path — `imagination_nll` is called in `train_worldmodel.py:338`
   (the flagship arm), `train_flagship4b.py:164`, `finetune_traj.py:217` — so one
   pathological cell can NaN-kill a training run (the F-5/F-6/F-7 ops-fragility
   class this stream tracks; the exact silent-death mode the pod monitor keeps
   catching).
2. **The uncertainty EXPORT path `replay/arms.py:284`** — `(0.5*logvar).exp()`
   feeds OKRI / LOPS / the H2 modality-steering trigger; an unbounded **positive**
   logvar overflows the std → NaN metric / NaN safety trigger.

Fix: clamp logvar to `[-10, 10]` at the head output (protects all three consumers
uniformly) plus a defensive clamp inside `imagination_nll`. `exp(10)≈2.2e4` and
`exp(5)≈148` are far inside fp32 range. Behaviour-preserving in the healthy range
(a converged head sits well within the bounds) → a trained checkpoint is
numerically unchanged; only pathological values are trimmed. `clamp` passes
gradient in-range and zeroes it at the bound = "stop pushing logvar to ±inf". A
tanh/softplus reparam would perturb every value and require retraining — wrong
tool for a post-hoc guard.

## Evidence & tests

- Tests: `tests/test_imagination_logvar_clamp.py` — **17 passed** on the author
  machine (RTX 4060 dev box, py3.13, torch 2.11). Covers:
  - `test_safe_nll_finite_on_pathological_logvar` (logvar ∈ {-100, -1e4, -88.8,
    50, 1e4}) — the safe NLL is finite where the stack formula overflows;
  - `test_safe_matches_reference_in_healthy_range` (logvar ∈ [-3, 3]) — allclose
    to the exact stack formula → **no change to a trained checkpoint**;
  - `test_clamp_bounds_and_noop_in_range`, `test_clamp_gradient_zeroed_at_bound`,
    `test_safe_nll_is_differentiable`;
  - `test_std_export_path_finite_after_clamp` — reproduces the `replay/arms.py`
    std overflow and shows the clamp fixes it;
  - `test_field_forward_would_be_bounded_by_head_clamp` — an over-confident
    `ImaginationField`-style head is bounded by the proposed forward clamp;
  - **witness (needs `PYTHONPATH=stack`):** `test_stack_imagination_nll_overflows_witness`
    asserts the UNPATCHED stack `imagination_nll` returns non-finite at
    logvar=-100 (documents the live bug; will flip to a failure signalling the
    fix is already merged), and `test_stack_matches_safe_in_healthy_range_witness`.
- Baseline unaffected: `pytest stack/tests -k "imag or h15 or d9"` = **11 passed**
  (the fix is proposed, not yet applied; confirms the H15 suite is green pre-merge).

## Risk & rollback

- Blast radius: one module (`imagination.py`), one 4-line helper + two call-site
  edits. Strict superset of current behaviour: identical output for any logvar in
  `[-10, 10]` (the healthy range), only clips pathological values that would
  otherwise NaN. No training/eval/export contract change; export path is
  element-wise `clamp` (ONNX `Clip`, trivially supported). Full stack suite should
  stay green.
- Rollback: revert the two call-site edits and delete the helper. No state, data,
  or config touched. If a future arm legitimately needs logvar outside `[-10,10]`,
  widen the bounds (the values are named module constants).

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:** <...>
- **Reason & notes:** <...>
- **Integrated as:** <commit hash / stack path> (if applicable)
