# INTAKE — Orthogonality / isotropy admissibility instrument for the readout latent

- **Package:** `Architecture & Inference/Implementation/incoming/2026-07-10-orthogonality-instrument/`
- **Author agent / date:** Architecture & Inference (Wednesday), 2026-07-10
- **Proposed target:** extend `stack/tanitad/eval/spectral.py` (fold `_effective_rank`/`_energy_knee`
  into the identical existing primitives; add the isotropy functions + `OrthogonalityReport` +
  `orthogonality_report`); add `stack/scripts/run_orthogonality.py`; add a test module under
  `stack/tests/` (mirrors `test_spectral*`).
- **Hypothesis / WP served:** H3 (SIGReg-only anti-collapse / identifiability) · D-021 (latent-dim
  sizing) · gate admissibility for D1–D3 sizing claims (instrument doctrine, D-004 / G-AI1)

## What & why (≤10 lines)

`spectral.py` sizes the latent by fitting a **linear** action-conditioned transition operator and
reading its spectrum (D-021: "the 2048 readout is over-provisioned; size to the ~tens-dim knee").
That sizing only licenses an *optimal-planning* claim when a precondition holds: *When Does LeJEPA
Learn a World Model?* (arXiv **2605.26379**, Klindt/LeCun/Balestriero, Lean-4-verified) proves
LeJEPA/SIGReg gives **linear + orthogonal** identifiability **iff** the SIGReg-regularized marginal
has reached its **isotropic-Gaussian** target — and only then is latent-space planning optimal (for
rotation-invariant cost). This package makes that precondition **falsifiable on our own checkpoint**:
it measures the readout covariance's isotropy (global + within the active subspace), condition
number, participation ratio, and coordinate decorrelation, and returns an ADMISSIBLE /
NOT-YET-ADMISSIBLE verdict for the sizing claim. It is an **instrument row, not an architecture
change** (no gate, no change — G-AI1). Research note:
`Architecture & Inference/Research/2026-07-10-orthogonality-instrument-and-isotropy-theory.md`.

## Evidence & tests

- Tests included: `tests/test_spectral_orthogonality.py` — **8 passed** (venv, 2.9 s). Ground-truth
  cases: isotropic-Gaussian → ADMISSIBLE; steep-spectrum → NOT-YET; correlated-coords → flagged;
  **over-provisioned (r isotropic dims + dead tail) → recovers active_k≈r, active isotropic,
  global-iso low → ADMISSIBLE** (the real-checkpoint pattern); primitives checked on closed forms.
- Standalone stack suite unaffected (touches no `stack/` file): **189 passed, 1 skipped**.
- **Measured run (G-H), step-6500 trained ckpt** (`ckpt_full.pt`, 24 comma2k19 val eps, 7 200 readout
  latents, RTX 4060, 72 s, $0): `active_k=21`, `cov_effective_rank=24.93`
  (**reproduces** the independent spectral run's `repr_effective_rank=24.93` / `optimal_k=21` — a
  cross-instrument consistency check), `iso_ratio_global=2.0e-8` (dead-tail dominated, expected),
  **`iso_ratio_active=0.250`** (cond 246), **`rms_offdiag_corr=0.428`** → **VERDICT: NOT-YET-ADMISSIBLE**
  at step-6500. Honest read (P8): the D-021 over-provisioning finding stays *descriptive*, but its
  *optimal-planning* interpretation is **not licensed** until SIGReg isotropy converges — re-measure at
  15k/30k (iso_active should climb toward 1). Artifact: `Research/2026-07-10-orth_step6500.json`.
- DIAGNOSTIC caveat: step-6500 is undertrained (target 30k) → this is a convergence tripwire, not a
  decision-grade admissibility verdict.

## Risk & rollback

- Blast radius if integrated: additive — new functions + `OrthogonalityReport` in `spectral.py`, one
  new script, one new test file. No change to existing spectral APIs, models, training, or configs.
  Pure `torch`; 0 new deps.
- Rollback: delete the added functions/script/test; `spectral.py`'s existing API is untouched.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
