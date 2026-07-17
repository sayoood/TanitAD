# STATE — Architecture & Inference

LAST_RUN: 2026-07-17 (Wednesday weekly agent — 2 measured experiments on real comma2k19. E1 (NOVEL): the 1-step ImaginationField DISSIPATES epistemic σ + collapses to an attractor under blind K-step rollout (the H11/D8 trigger risk flagged 07-15, now measured, matching "Biased Dreams" 2604.25416); cause = the recursion (freeze-1 is flat-safe); shipped as a prototype with 10✓ metric tests. E2 (INTEGRITY): found a theoretically-superior orthogonality instrument already built 2026-07-10 but unmerged → WITHDREW my duplicate, VERIFIED the prior one (iso_ratio_active 0.254 NOT-YET-ADMISSIBLE) → D-021 = subspace ID not "optimal planning"; flagged the stranded pkg for merge. Created GOALS.md.)
QUALITY: full (G-A…G-H, G-AI1, G-AI2 met; 3 searches + 2 fetches + 2 measured GPU experiments (2 seeds each) + 1 verifiable increment (blind-rollout diagnostic, 10✓ tests) / ~2.8 h — under caps. G-H #1 (H15/H11/D8, top-risk-adjacent): blind-rollout diagnostic on the trained field, real comma2k19, 4060, $0 — σ log-var −7.79→−8.55 (dissipation), fidelity 0.357→0.011 by k4, attractor cos 0.21→0.57, freeze-1 flat ~0.25 → cap operative self-monitor at 1-step (D-018 escalate, no config change). G-H #2 (H3/D-021): VERIFIED the stranded 07-10 orthogonality instrument, n=2600>S — iso_ratio_active 0.254 NOT-YET-ADMISSIBLE; global isotropy ~0 = over-provisioning by design (not failure). Both instrument-only, G-AI1 honored. No competing intake shipped (duplicate withdrawn, P8). Stack suite 343✓/2s in a fresh worktree, untouched by me.)
(Calendar: wall-clock 2026-07-17. Dating by wall clock per the Data-Eng precedent.)

## HANDOFF

No half-done work. Two measured experiments + one intake + bounded sweep this run:

1. **E1 (G-H, top-risk-adjacent) — blind K-step belief-rollout diagnostic** on the trained 1-step
   ImaginationField (step-6500 base250cam ckpt, real comma2k19 val, 4060, 2 seeds, $0). Rolled fully
   blind: fidelity **0.357(k1)→0.011(k4=chance)→neg**; **σ dissipates** (hidden log-var −7.79→−8.55 =
   *more* confident as it decays); **attractor collapse** (inter-sample cos 0.21→0.57, belief energy −11×;
   true energy flat). **Cause is the recursion:** freezing the k=1 imagination holds **~0.25 flat across 8
   horizons** and beats persistence. Backlog-0b falsifier MET. **Constraint: cap the operative H15/D8
   self-monitor at 1-step / parallel-horizon until a multi-step σ is validated** (D-018 escalate; no
   config change executed). Artifacts: `Implementation/belief_rollout_diagnostic/blind_rollout.py` +
   `results/2026-07-17-blind_rollout-seed{0,1}.json`. Note (P8): pre-reset directional ckpt.
2. **E2 (H3/D-021) — VERIFIED a stranded instrument instead of duplicating (P8/D-026 integrity).** Found a
   theoretically-superior orthogonality instrument already built **2026-07-10** but **unmerged** (branch
   `worktree-agent-arch-inf-20260710`, intake `incoming/2026-07-10-orthogonality-instrument/`). **Withdrew
   my duplicate draft**, ran the prior `orthogonality_report` unchanged on step-6500 (n=2600>S): reproduces
   exactly — active_k 23, **iso_ratio_active 0.254 < 0.5 → NOT-YET-ADMISSIBLE**, cond_active 218, rms_offdiag
   0.424. Correction: global isotropy ~0 is over-provisioning **by design**, the active-subspace read is the
   admissibility number. **D-021 = "identifies a low-dim subspace," NOT "optimal planning."** Artifact +
   merge-recommendation: `Implementation/orthogonality_verification/`. **ORCHESTRATOR: merge the 07-10 pkg.**
   The G-E increment this run is the E1 blind-rollout diagnostic (`belief_rollout_diagnostic/`, 10✓ tests).
3. **Bounded sweep (D-013/D-028):** new anchor **"Biased Dreams" (2604.25416)** (latent attractor behaviour
   → E1's exact prediction); UWM-JEPA / VJEPA / Var-JEPA (σ-grounding routes); JEPA generalization theory
   (2606.27014); recency scan (cs.RO) → BadWAM (2607.15207) Phase-1 watch. No `Ressources/` folder present.
4. **Created `GOALS.md` (D-029):** G1 flagship-beats-CV (top risk), G2 safe multi-step H15 σ, G3 FLOPs ledger.

### FLAGGED for orchestrator (not mine to fix)
- `stack/tests/test_physicalai_rig.py` (untracked PhysicalAI-rig work) still fails **collection** on a bare
  `pytest` (`ImportError: ftheta_horizon_row` from `tanitad.data.calib`) — BUT it is untracked, so it does
  NOT come into agent worktrees; a fresh-worktree `pytest` is green (343✓/2s). Owner must still add the
  symbol / drop the import for the main tree. My work touches no `stack/` files.

### Exact next steps (next Wednesday run, in priority order)
- **P0 0b-B (cheap, recommended) — parallel-horizon operative imagination.** Wire a non-autoregressive
  imagination path (predict each horizon from the last real obs, not fed back) + measure D8 AUROC on
  degraded-visibility episodes. freeze-1 already showed it recovers ~0.25 flat fidelity. **D-018 escalate**
  before it becomes the operative default (changes self-monitor semantics).
- **P0 0b-A (build) — multi-step belief-rollout TRAINING.** NLL at k∈{1,2,4} on the *recursive* path +
  anti-attractor term (penalise belief-energy collapse / inter-sample-cosine growth). Target: σ grows with
  horizon, rolled fidelity ≥ freeze-1. Falsifier: σ still dissipates after training → architecture ceiling,
  adopt 0b-B permanently. Reuse `Implementation/belief_rollout_diagnostic/`. **D-018 escalate.**
- **P0 (decision-grade re-runs the moment the flagship @30k lands) — re-run E1 (blind rollout) + E2
  (orthogonality) + spectral on the OPERATIVE flagship ckpt.** All three are pre-reset-directional now;
  they become decision-grade on the operative ckpt and couple to the flagship verdict (G1). Turnkey:
  `blind_rollout.py` + `run_orthogonality.py` + `run_spectral.py` (stage the val cache so `*val*` doesn't
  grab `comma_val.tgz`).
- **P0 #2b — decision-grade K∈{1,2,4} sweep at OPERATIVE scale** from the pod2 step-8k `ckpt_full.pt`
  (Phase C). Primary metric **`imag_rel` per horizon** (NOT dir-acc — proven to saturate); reuse
  `Implementation/kstep_bakeoff_probe/kstep_bakeoff_probe.py`. **D-018 Tactic → escalate before trained-config.**
- **P1 #3b-follow — readout-whitening / orthogonality-penalty bake-off lever** (from E2): one-lever smoke
  first; falsifier = Δ within noise on D2/D1. Restores the LeJEPA orthogonality condition if we want the
  optimal-planning corollary back. **D-018 escalate.**
- **P1 #3 (build) — AdaLN `CondBlock` + RoPE** in `OperativePredictor` so those `planned` levers become
  runnable; smoke-first (expect small Δ per 2605.08567). Ship each as an intake with the harness sweep
  pre-wired. **D-018: escalate before either touches the trained config.**
- **Standing duties (D-013):** theory-watch (Balestriero/LeCun spectral-SSL IEEE SPMag 2026, Klindt +
  `github.com/klindtlab/lejepa-identifiability`, HaoChen, PKU Yisen Wang 2606.27014); citation-walk set now
  includes Delta-JEPA / FF-JEPA / OmniDreams / LeJEPA-identifiability (2605.26379) / **UWM-JEPA
  (2605.25313)** / **"Biased Dreams" (2604.25416, attractor/UQ limits)** / **JEPA generalization theory
  (2606.27014)**; no `Ressources/` folder present (re-check newest-mtime each run).

## Open coordination
- Master Plan §3 puts the *gate harness* under Benchmarks & Eval (Thu). The gate runner is deliberately the
  Architecture half (standard ADE/FDE + instrument gating + model wiring) with an `extra_metrics` seam for
  Thursday's custom suite (LAL/TMS/OKRI/CNCE/LOPS). Thursday: import `run_d1/run_d2/run_d3` and plug the
  custom metrics through the hook rather than forking a parallel runner.
