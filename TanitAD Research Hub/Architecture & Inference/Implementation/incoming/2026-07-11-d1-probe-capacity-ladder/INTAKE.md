# INTAKE — D1 probe-capacity ladder + isotropy linkage (well-powered discriminator)

- **Package:** `Architecture & Inference/Implementation/incoming/2026-07-11-d1-probe-capacity-ladder/`
- **Author agent / date:** Architecture & Inference (Wednesday), 2026-07-11
- **Proposed target:** `stack/scripts/probe_capacity_ladder.py` — OR fold the PCA-reduce
  fix into the existing `stack/scripts/d1_probe_capacity.py` (`0284a5c`) and keep this as the
  ladder+isotropy superset. (Orchestrator's call; see risk note.)
- **Hypothesis / WP served:** D1 investigation (info-lost vs less-linear) / H3 (isotropy) / H5 (readout) / D-004

## What & why (≤10 lines)

The loop opened a live D1 discriminator (`d1_probe_capacity.py`): is trajectory info LOST or
just less LINEAR? This package (a) turns it into a **capacity ladder** (OLS → ridge sweep →
MLP-1h → MLP-2h), (b) measures the latent **isotropy in the same run** (cross-checks the
2026-07-10 orthogonality instrument: `iso_active` 0.25↔0.27, `active_k` 21↔19), and (c) fixes a
confound: with 12 eps the probe is **underdetermined (D=2048 ≫ N=204)** so `linear_ols` overfits
(24.4 m) vs ridge (10.3 m) — a 2.4× swing from regularisation, not capacity. **Fix: PCA-reduce to
`active_k` (train-only basis)** → N>D, clean read. Result at step-6500: **no nonlinear advantage**
(gap −15 %/−39 %), i.e. the "less-linear" branch is not supported once properly powered — but this
is directional (MLP data-starved at n=204), decision-grade needs ≥50 eps + the 14k-vs-21k pair.
Research note: `Research/2026-07-11-d1-probe-capacity-ladder-and-isotropy-linkage.md`.

## Evidence & tests

- Tests included: `tests/test_probe_capacity_ladder.py` — **8 passed** on author machine
  (RTX 4060, ~4.3 s). Ground-truth: isotropic→high iso, anisotropic→low; **linear-recoverable
  target → small gap; nonlinear-only target → large positive gap** (the discriminator's core
  behaviour); PCA fixes the D≫N OLS overfit; ego-frame rotation on closed forms.
- Measured numbers (step-6500 `ckpt_full.pt`, 12 comma2k19 val eps, 408 latents, dim 2048):
  isotropy `active_k=19 / iso_active=0.269 / cond_active=230.6`; PCA-19 @1s best-linear 8.84 vs
  best-MLP 10.18 (gap −15.1 %), zero-motion ref 18.36. Full JSON:
  `results/2026-07-11-probe_ladder_step6500.json` (+ copy in `Research/`).
- Stack suite unaffected by this package: **189 passed / 1 skipped** (nothing in `stack/` touched).

## Risk & rollback

- Blast radius if integrated: script-only (`stack/scripts/`), no library/model/training-path
  change; zero new deps (pure `torch`). If folded into `d1_probe_capacity.py`, that script's CLI
  changes (add `--pca-k`/`--episodes`) — additive, defaulted off.
- Rollback: delete the script (or revert the `d1_probe_capacity.py` edit). No state migration.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
