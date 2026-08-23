# INTAKE — p0-spectral-sizing: latent-dim sizing from the transition spectrum

- **Package:** `Architecture & Inference/Implementation/incoming/2026-07-14-spectral-sizing-p0/`
- **Author agent / date:** Architecture & Inference (Wednesday), 2026-07-14 (base commit `121177a`)
- **Proposed target:** `stack/tanitad/eval/spectral.py` (+ `stack/tests/test_spectral_sizing.py`)
- **Hypothesis / WP served:** H3 (data-efficiency / latent trade-off) · WP3 · leverage action **L2**
  (`p0-spectral-sizing`) from DECISIONS D-013 theory-watch + arXiv 2606.27014

## What & why (≤10 lines)

Backlog **#0** (new top item, added to the agent file mid-session by D-013). The JEPA generalization
theory (arXiv 2606.27014, analyzed in `Research/2026-07-06-jepa-generalization-theory-and-hit-jepa.md`)
turns the latent dimension into a *measurable* decision: approximation error is the **spectral tail**
Σ_{i>k}σᵢ² of the action-conditioned transition operator (↓k), sample error grows ~O(k²) (↑k), and the
optimal k sits at the spectral **knee** (Thms 4.3–4.6). `tanitad_spectral_sizing.py` estimates that
spectrum — fits a linear operator `(z_t, a_t) → z_{t+1}` with the tested closed-form `RidgeProbe`, SVDs
the state-transition block, and reports σ decay, entropy effective-rank (the offline twin of the live
`erank` collapse row), the 99%-energy knee, a trade-off-optimal `k*`, the spectral tail at candidate dims,
and a recommendation **vs the current 2048-dim readout** (D-008). It flags OVER-/UNDER-provisioning.

## Evidence & tests

- Tests included: `tests/test_spectral_sizing.py` — **8 passed / 1.52 s** on the author venv (py3.13 +
  torch cu128), no simulator. Full stack suite unaffected: **65 passed, 1 skipped**.
- Load-bearing validation — on synthetic latents with a KNOWN rank r=5 embedded in S=32 (orthonormal-row
  embedding, controlled operator singular values), the estimator recovers `energy_knee_k == 5`,
  `operator_effective_rank ≈ 5`, spectral tail beyond r ≈ 0, and a sharp σ gap at r. Also validated: the
  OVER-PROVISIONED flag fires when the 2048 readout dwarfs a rank-4 spectrum; UNDER-PROVISIONED fires when
  the readout is below the knee; `k*` does not shrink as N grows (weaker O(k²)/N penalty); `pairs_from_states`
  temporal alignment; and an end-to-end run of a real `WorldModel(smoke_config)` latent path.
- Instrument honesty: `fit_r2` is reported as a sanity floor — a near-zero R² means the linear-operator
  proxy is inappropriate for those latents and no sizing should be read from its spectrum.

## Risk & rollback

- Blast radius: additive only — new `stack/tanitad/eval/spectral.py` + one test. Reuses stable
  `tanitad.models.readout.RidgeProbe`; no existing module changes.
- **Scope caveat (P8, must not be lost on integration):** a *decision-grade* comma2k19 spectrum needs a
  **trained** checkpoint. Untrained/collapsed latents (e.g. the p0-sB00 pipe-proof) are near-isotropic and
  give a degenerate spectrum — the tool runs but its recommendation is meaningless there. The real sizing
  run is queued behind the A40 Stage-0 checkpoint (`stack/RUNPOD_RUNBOOK.md`); this package is the tool it
  calls. No sizing claim is made on untrained latents in this intake.
- `complexity_weight` in `optimal_k` is an exposed knob, not a derived constant (the theory's constants do
  not transfer — L5); the robust, validated output is the spectrum + knee, not an absolute `k*`.
- Rollback: delete `stack/tanitad/eval/spectral.py` and its test; no other file touched.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate
- **Date / by:** 2026-07-08, MVP orchestrator (autonomous loop iteration 2)
- **Reason & notes:** Additive, honest (fit-R² sanity floor; explicit no-claims-on-untrained-latents
  caveat preserved), rank-recovery validation on known-rank synthetics is exactly right. Only change:
  test import de-hacked to `tanitad.eval.spectral`. Full suite 97 passed / 1 sim-skip. The
  decision-grade sizing run stays queued behind the trained checkpoint per the package's own caveat;
  a diagnostic (non-sizing) spectrum preview on the live ckpt is being taken to inform the step-10k
  erank decision — clearly labeled diagnostic, not a sizing claim.
- **Integrated as:** `stack/tanitad/eval/spectral.py` + `stack/tests/test_spectral_sizing.py`
  (see `intake(arch-inf)` commit, 2026-07-08)
