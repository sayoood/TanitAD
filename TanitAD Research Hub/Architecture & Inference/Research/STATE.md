# STATE — Architecture & Inference

LAST_RUN: 2026-07-10 (Wednesday weekly agent — orthogonality/isotropy admissibility instrument (backlog P1 #3b) built + measured on step-6500 ckpt (NOT-YET-ADMISSIBLE) + isotropy-theory watch HamJEPA 2605.20107 / PGSA 2606.12471). Worktree branch: `worktree-agent-arch-inf-20260710`.
QUALITY: full (G-A…G-H, G-AI1, G-AI2 met; 3 searches + 2 fetches + 1 measured experiment / ~2.2 h — under caps. G-E: intake pkg 8✓ standalone, stack suite 189✓/1s unchanged (touched no stack file). G-H: orthogonality instrument on step-6500 → iso_ratio_active 0.250 / rms_offdiag 0.428 → NOT-YET-ADMISSIBLE; cross-checked against spectral (cov_eff_rank 24.93 = repr_eff_rank 24.93). No gate passed → no claim; blocks a premature D-021 optimal-planning reading, D-004.)
(Calendar: wall-clock 2026-07-10; hub notes forward-dated by the autonomous loop. Dating by wall
clock per the Data-Eng precedent — see the run note's calendar footnote.)

## HANDOFF

No half-done work. One build + one measured experiment + theory-watch this run:

1. **G-E + G-H — orthogonality/isotropy admissibility instrument** (backlog P1 #3b, from 2605.26379).
   Intake pkg `Implementation/incoming/2026-07-10-orthogonality-instrument/` (`spectral_orthogonality.py`
   + `run_orthogonality.py` + `tests/` **8✓**, pure torch, 0 deps; proposed target = extend
   `stack/tanitad/eval/spectral.py`). Measures whether the SIGReg readout reached the isotropic-Gaussian
   target the optimal-planning theorem needs. **Measured on step-6500** (24 val eps, 7 200 latents, 4060,
   72 s, $0): `active_k=21` / `cov_effective_rank=24.93` **exactly reproduce** the independent spectral
   numbers (cross-instrument sanity); `iso_ratio_active=0.250` / `cond_active=246` / `rms_offdiag_corr=0.428`
   → **NOT-YET-ADMISSIBLE**. **Honest tempering (P8):** the D-021 over-provisioning finding stays
   *descriptive*, but its *optimal-latent-planning* reading is **not licensed** at step-6500 — the
   instrument *blocks* a premature claim (D-004/G-AI1). Convergence tripwire for 15k/30k. Artifact
   `Research/2026-07-10-orth_step6500.json`. No stack file touched (suite 189✓/1s). No trained-config change (D-018).
2. **Theory-watch (D-013):** **HamJEPA (2605.20107)** — "no geometry-independent fixed marginal target is
   canonical"; isotropy optimal only for rotation-invariant cost (= 2605.26379's condition) → validates
   the instrument's scoping + seeds a Phase-1 symplectic-predictor lever; beats SIGReg +3.5/+7.5/+10.6
   probe pts on structured tasks. **PGSA (2606.12471)** — identifiability without Gaussianity via symbolic
   grounding; statistical-WM error "grows monotonically with time" under non-Gaussian dynamics = the
   mechanism behind our K-step horizon-degradation → reinforces H1 hierarchy. Both Lean-verified.

### Exact next steps (next Wednesday run, in priority order)
- **P0 #1 + P1 #3b re-run — orthogonality + spectral at the FINAL Stage-0 ckpt** (15k preview / 30k
  decision-grade). The orthogonality instrument is now the **admissibility gate** on any D-021 resize:
  no resize is admissible until `iso_ratio_active` converges (falsifier: stalls low → withhold + escalate
  SigReg weight, D-018). Turnkey: `run_orthogonality.py --ckpt <final> --cache-dir <val cache DIR>` (pass
  the cache DIR, not the eval root — the `*val*` glob otherwise grabs `comma_val.tgz`; the script now
  filters to dirs). Pair with `run_spectral.py` (rank was still climbing 35→43 at 6.5k).
- **P0 #2b — decision-grade K∈{1,2,4} sweep at OPERATIVE scale** from the pod2 step-8k `ckpt_full.pt`
  (Phase C). Primary metric **`imag_rel` per horizon** (NOT dir-acc — proven to saturate); reuse
  `Implementation/kstep_bakeoff_probe/kstep_bakeoff_probe.py`. **D-018 Tactic → escalate before it touches the trained config.**
- **P0 #2c — extend imagination horizons past 0.4 s** (predictor imagines k∈{1,2,4}; D3 wants 2 s). Couples
  with 2b (K must cover the horizon). Escalate before trained-config change.
- **NEW build — live `iso_ratio_active` training row** (from this run's rec #3): add active-subspace
  isotropy to the trainer's collapse-health log next to `erank`, so SIGReg-isotropy convergence is
  watchable in-flight; feeds the "raise SigReg if it stalls" decision with a *direct* signal. Cheap
  (reuses `spectral_orthogonality` on the readout batch). Ship as intake.
- **P1 #3 (build) — AdaLN `CondBlock` + RoPE** in `OperativePredictor` so those `planned` levers become
  runnable; smoke-first (expect small Δ per 2605.08567). Ship each as an intake with the harness sweep
  pre-wired. **D-018: escalate before either touches the trained config.**
- **Standing duties (D-013):** theory-watch (Balestriero/LeCun spectral-SSL IEEE SPMag 2026, Klindt +
  `github.com/klindtlab/lejepa-identifiability`, HaoChen, PKU Yisen Wang 2606.27014); citation-walk set now
  includes Delta-JEPA / FF-JEPA / OmniDreams / LeJEPA-identifiability (2605.26379) / **HamJEPA
  (2605.20107)** / **PGSA (2606.12471)**; `Ressources/` inbox re-check newest-mtime each run.

## Open coordination
- Master Plan §3 puts the *gate harness* under Benchmarks & Eval (Thu). The gate runner is deliberately the
  Architecture half (standard ADE/FDE + instrument gating + model wiring) with an `extra_metrics` seam for
  Thursday's custom suite (LAL/TMS/OKRI/CNCE/LOPS). Thursday: import `run_d1/run_d2/run_d3` and plug the
  custom metrics through the hook rather than forking a parallel runner.
