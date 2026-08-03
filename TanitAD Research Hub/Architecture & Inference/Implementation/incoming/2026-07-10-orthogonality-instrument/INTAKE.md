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

- **Verdict:** **integrate-with-changes**
- **Date / by:** 2026-08-03 · orchestrator daily sweep (age at adjudication: **24 days**)
- **Reason & notes:**
  - **This is the package CLAUDE.md names by description as a programme failure** — "an
    orthogonality instrument sat unmerged for **10 days** because the request lived in a README
    nobody re-read". The true figure at adjudication is **24 days**. The rule that was written to
    stop this from happening did not stop it; a weekly orchestrator slot was the mechanism, which
    is precisely what the daily rotation exists to fix. Landing it is the point.
  - **Re-verified here, not inherited:** `tests/` re-run on this box → **8 passed / 2.90 s**
    (`venvs/tanitad`). Confirmed still-absent at tip before integrating (two probes per the
    absence rule): `grep` on `stack/tanitad/eval/spectral.py` shows no `orthogonality_report` /
    `OrthogonalityReport` / `covariance_eigs`, and `stack/scripts/run_orthogonality.py` does not
    exist.
  - **Instrument row, not an architecture change** (G-AI1) — nothing it computes gates anything.
    It makes a **precondition falsifiable**: `spectral.py` sizes the latent from a transition
    spectrum, and that sizing only licenses the *optimal-planning* reading if the SIGReg marginal
    has actually reached isotropy. Today the programme quotes D-021's over-provisioning finding
    with no way to check that precondition on its own checkpoint. After this, it can.
  - **Changes made on integration:**
    1. `_effective_rank` / `_energy_knee` were **inlined duplicates** of `spectral.py`'s public
       `effective_rank` / `energy_knee` (the package says so itself: "identical to spectral.py;
       inlined for standalone tests"). Folded onto the existing primitives, as the INTAKE
       proposed — two definitions of an effective rank in one module is exactly how a metric
       silently forks. The isotropy section (`covariance_eigs` → `orthogonality_report`) is
       appended to `spectral.py` unchanged.
    2. `run_orthogonality.py` resolved the stack via `parents[6] / "stack"` — a **worktree-shaped
       path** that is wrong from `stack/scripts/` and would have imported nothing (or, worse,
       a different checkout). Repointed to `parents[1]`. Verified: `--help` renders from the
       installed location.
    3. Test imports repointed from the standalone `spectral_orthogonality` to
       `tanitad.eval.spectral`.
  - Verified green AFTER integration: `test_spectral_orthogonality.py` + the pre-existing
    `test_spectral_sizing.py` → **16 passed**, i.e. folding the primitives did not move the
    sizing numbers.
  - **The step-6500 verdict is NOT promoted by this integration.** `NOT-YET-ADMISSIBLE` at 6500
    is a convergence tripwire on an undertrained checkpoint, as the package itself flags. The
    re-measure at a trained checkpoint is a follow-up work item, filed — not assumed done because
    the tool landed.
- **Integrated as:**
  - `stack/tanitad/eval/spectral.py` — isotropy primitives + `OrthogonalityReport` +
    `orthogonality_report`, folded onto the existing `effective_rank` / `energy_knee`
  - `stack/scripts/run_orthogonality.py`
  - `stack/tests/test_spectral_orthogonality.py` (8 tests)
