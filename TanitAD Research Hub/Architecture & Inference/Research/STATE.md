# STATE — Architecture & Inference

LAST_RUN: 2026-07-11 (Wednesday weekly agent, worktree `worktree-arch-inf-20260711` — D1 probe-capacity
ladder + isotropy linkage: advanced the loop's live D1 discriminator (`0284a5c`); found the raw-2048 probe
underdetermined (D≫N), PCA-to-active_k fix, NO nonlinear advantage at step-6500 (gap −15 %/−39 %),
anisotropy-taxes-linear-probe prediction REFUTED; theory-watch Sub-JEPA 2605.09241 + layerwise-probing 2606.09646)
QUALITY: full (G-A…G-H, G-AI1, G-AI2 met; 4 searches + 1 fetch + 1 measured experiment on the 4060 / ~1.4 h —
under caps. G-H: probe-capacity ladder step-6500, measured gap −15.1 % @1s PCA-19, falsifier verdict delivered
(≥50-ep + 14k/21k pair for decision-grade). 8 intake tests + stack 189✓/1s unaffected.)
(Calendar: wall-clock 2026-07-11 (fired Saturday clock — narrative runs ahead). Dating by wall clock per the
Data-Eng precedent.)

## HANDOFF

No half-done work. One measured experiment + theory-watch this run; all outputs committed.

1. **G-H measured experiment — D1 probe-capacity ladder** (advances the loop's live `d1_probe_capacity.py`,
   `0284a5c`). Ladder (OLS→ridge sweep→MLP-1h→MLP-2h) + in-run isotropy on step-6500. **F-1:** raw-2048 probe
   underdetermined (D≫N=204) → `linear_ols` 24.35 m vs ridge 10.31 m @1s (regularisation artifact); **fix =
   PCA-reduce to `active_k` (train-only basis).** **F-2:** NO nonlinear advantage — gap negative everywhere
   (−15 %/−39 %); best probe = linear ridge (8.84 @1s vs 18.36 zero-motion) → "less-linear" branch disfavoured,
   but directional (MLP starved at n=204). **F-3 (P8 negative):** my pre-registered "anisotropy taxes the
   linear probe" prediction REFUTED — ridge absorbs the anisotropy; 2605.26379 orthogonality governs PLANNING
   regret (D4–D6), not D1 probe recoverability. Intake `Implementation/incoming/2026-07-11-d1-probe-capacity-ladder/`
   (8 tests) + `Research/2026-07-11-probe_ladder_step6500.json`. No stack change (189✓/1s).
2. **Theory-watch (D-013 + D-028 recency scan):** Sub-JEPA (2605.09241) subspace-SIGReg = candidate remedy for
   the `iso_active` 0.27 shortfall (isotropy gate, not D1); layerwise-probing (2606.09646) grounds the
   linear-vs-MLP protocol. D-027 (rollout_k=4) settled my K-step thread → P0 #2b/#2c retire.

### Cross-run dependency (flag for orchestrator)
- The **2026-07-10 orthogonality branch `worktree-arch-inf-20260710` is still UNMERGED in main** (W29 report
  flagged it). This run cross-validates its numbers (iso_active 0.25↔0.27, active_k 21↔19) and depends on the
  isotropy instrument conceptually. **Recommend merge** (orchestrator triage). My STATE/KB/BACKLOG edits are
  against main, so a small reconcile with that branch's same-file deltas is expected.

### Exact next steps (next Wednesday run, in priority order)
- **P0 #A — well-powered D1 discriminator on the pod:** PCA-to-active_k + ≥50 eps, run on the actual
  **14k-vs-21k (or 30k) pair** for the decision-grade info-lost-vs-less-linear verdict. Turnkey: this run's
  `probe_capacity_ladder.py --episodes 50` (already generalises to any ckpt). If no positive gap → D1
  regression is NOT a decode-capacity artifact → escalate toward coordinate-frame/normalisation.
- **P1 #B — characterise Sub-JEPA subspace-SIGReg (2605.09241)** as the `iso_active` remedy; design-note
  first (which active-k dirs; loss form), smoke on 4060, D-018 escalate before any trained-config change.
- **P1 #1 — re-run spectral + orthogonality at the FINAL Stage-0 ckpt** (30k) — rank/iso were both still
  moving; decision-grade D-021 + admissibility input. Turnkey: `run_spectral.py`/`run_orthogonality.py`.
- **P1 #4 — frozen-DINOv3 WM arm (H4 arm-B):** the loop shipped `dino_precompute.py` (`cda93df`) — WATCH;
  pick up predictor-on-frozen-features training if the loop doesn't reach it.
- **Standing duties (D-013 + D-028):** theory-watch (Balestriero/LeCun/Klindt/PKU-Yisen-Wang; citation-walk
  now incl. Sub-JEPA / layerwise-probing) + **recency-first arXiv listing scan** (cs.CV/cs.RO/cs.AI, 14 d,
  world-model/E2E = my seam); `Ressources/` inbox re-check newest-mtime each run.

## Open coordination
- Master Plan §3 puts the *gate harness* under Benchmarks & Eval (Thu). The gate runner is the Architecture
  half (ADE/FDE + instrument gating + model wiring) with an `extra_metrics` seam for Thursday's suite
  (LAL/TMS/OKRI/CNCE/LOPS). Thursday: import `run_d1/run_d2/run_d3` and plug through the hook.
- **D1 probe hygiene (this run):** flagged to the loop — `d1_probe_capacity.py` should PCA-reduce to
  active_k + raise episodes before its ridge-vs-MLP read is trustworthy (D≫N confound).
