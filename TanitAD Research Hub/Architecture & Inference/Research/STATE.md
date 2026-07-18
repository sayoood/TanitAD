# STATE — Architecture & Inference

LAST_RUN: 2026-07-18 (Wednesday weekly agent — executed backlog P0.1: re-ran E1+E2 on the OPERATIVE flagship-speed @19k on the eval-pod A40, dropping the 07-17 pre-reset caveat. E1 (blind K-step rollout, 320 windows, 2 seeds): the σ-dissipation + attractor-collapse pathology REPRODUCES on the operative model — falsifier "speed+jerk recipe fixed it" NOT met (cos_rollout→chance by k3; σ_hidden −9.461→−9.564, *lower* absolute σ = worse temporal calibration; attractor 0.219→0.805, sharper). freeze-1 holds 0.213–0.232 flat across 8 horizons (7× persistence) → parallel-horizon confirmed safe on the shipping model. NEW refinement: σ is spatially calibrated (hidden>visible +0.37; err↔var corr +0.29–0.43) but temporally anti-calibrated → target narrows to a horizon-aware σ. E2 (orthogonality, 7,964 latents): iso_ratio_active 0.254→0.546 (SIGReg converging as predicted; crossed 0.5), cond 218→61, but still NOT-YET-ADMISSIBLE (rms_offdiag 0.32>0.1); active_k≈19, cov_eff_rank≈30 ≪ 2048 → readout not the D1 bottleneck (G1) reaffirmed. New branch agent/arch-inf-20260718.)
QUALITY: full (G-A…G-H, G-AI1, G-AI2, G-I met; RECALL from KB + 2 measured GPU experiments (2 seeds on E1) on the OPERATIVE model + 1 verifiable increment (blind_rollout_flagship.py + run_orthogonality_flagship.py, package 15✓ tests incl. 5 new parity tests) / ~2.5 h — under caps. G-H (both experiments target the TOP program risk directly): E1 blind-rollout on flagship-speed @19k, eval-pod A40, PhysicalAI val, $0 — falsifier REPRODUCES (speed recipe did NOT fix σ-dissipation); E2 orthogonality same ckpt — iso converging 0.254→0.546, still NOT-YET-ADMISSIBLE. Both instrument-only, no config change (D-018), G-AI1 honored (each recommendation names gate+falsifier). Val = PhysicalAI (pod canonical), differs from 07-17 comma — noted (P8); qualitative pathology transfers. Touched zero stack/ files.)
RESOURCE (G-I): eval pod tanitad-eval (A40 48GB, idle on entry) — 2× blind-rollout seeds (36.4s+34.2s) + 1 orthogonality pass (~40s) + model loads, ~2 min GPU, cost $0 (standing pod, LOCK.arch-inf held). Why not bigger: this IS the pod-scale eval the mandate reserves for the A40 — the 263M flagship ckpt lives on the pod and the 4060 can't hold model+PhysicalAI-val comfortably; Colab unnecessary (job < 2 min). No 4060/Colab needed this run.
(Calendar: wall-clock 2026-07-18. Dating by wall clock per the Data-Eng precedent.)

## HANDOFF

No half-done work. Backlog P0.1 executed: **E1+E2 re-run on the OPERATIVE flagship-speed @19k**, dropping
the 07-17 pre-reset caveat. Both are turnkey to re-run at flagship @30k (the last step to decision-grade).

1. **E1 (G-H, TOP RISK) — blind K-step belief rollout on `flagship-speed` @19k** (WorldModel flagship4b
   action_dim=3, eval-pod A40, PhysicalAI val, 320 windows, 2 seeds, $0). **Falsifier NOT met — the
   σ-dissipation + attractor collapse REPRODUCE on the operative model** (the speed+jerk recipe did not fix
   the recursion). cos_rollout 0.232→chance by **k3**; σ_hidden **−9.461→−9.564** (*lower* absolute σ than
   the −7.8 pre-reset ckpt = worse temporal calibration); attractor **0.219→0.805** (sharper than 0.57).
   **freeze-1 holds 0.213–0.232 flat across 8 horizons (7× persistence)** → parallel-horizon confirmed safe
   on the shipping model. **NEW refinement:** σ is *spatially* calibrated (calib_gap +0.37 hidden>visible;
   per-cell err↔var corr +0.29–0.43) but *temporally* anti-calibrated → the design target narrows to a
   **horizon-aware** σ, not a spatial rebuild. Constraint stands: cap operative H15/D8 self-monitor at
   1-step / parallel-horizon until a multi-step σ is validated (D-018 escalate; no config change executed).
   Artifacts: `Implementation/belief_rollout_diagnostic/blind_rollout_flagship.py` +
   `results/2026-07-18-blind_rollout-flagship-speed-seed{0,1}.json`.
2. **E2 (H3/D-021) — orthogonality on `flagship-speed` @19k** (same ckpt, 7,964 latents). **iso_ratio_active
   0.254→0.546** (crossed 0.5 — SIGReg converging exactly as the 07-17 note predicted), cond_active 218→61,
   rms_offdiag 0.42→0.32 — but still **NOT-YET-ADMISSIBLE** (offdiag > 0.1 → LeJEPA optimal-planning corollary
   still withheld). active_k≈19, cov_eff_rank≈30 ≪ 2048 → **readout capacity is NOT the D1 bottleneck (G1),
   reaffirmed on the operative model.** Artifact: `Implementation/orthogonality_verification/run_orthogonality_flagship.py`
   + `2026-07-18-orth-flagship-speed.json`. **The 2026-07-10 orthogonality instrument is STILL UNMERGED
   (now 3rd+ week) — ORCHESTRATOR: merge `incoming/2026-07-10-orthogonality-instrument/` into stack/tanitad/eval/.**
3. **G-E increment:** `blind_rollout_flagship.py` + `run_orthogonality_flagship.py` shipped into the existing
   Implementation packages; added `tests/test_flagship_parity.py` (5 tests pinning the flagship variant's
   metric primitives bit-for-bit against `blind_rollout`) → **package 15/15 green** (tanitad venv). Touched
   zero stack/ files.
4. **GOALS updated (D-029):** G1 movement=yes (readout-not-bottleneck reaffirmed on operative model + iso
   converging), G2 movement=yes (target sharpened to horizon-aware σ). G3 still carried (not yet 2 runs stale).

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
- **P0 (decision-grade re-run the moment the flagship @30k lands) — re-run E1+E2 (+spectral) on flagship
  @30k.** E1+E2 are now done at the OPERATIVE @19k (2026-07-18): σ-dissipation reproduces (validated), iso
  converging 0.254→0.546. The @30k re-run is the ONLY remaining step from validated→decision-grade and
  couples to the flagship-vs-CV verdict (G1). Turnkey on the eval pod (~2 min): the two staged scripts
  `Implementation/belief_rollout_diagnostic/blind_rollout_flagship.py` +
  `Implementation/orthogonality_verification/run_orthogonality_flagship.py` (just bump the ckpt if the path
  changes) + `run_spectral.py`. Expected @30k: σ-dissipation persists (architecture property); iso rises
  further toward but likely not past admissibility (falsifier for building the whitening lever: iso≥0.7 &
  offdiag<0.1 → SIGReg gets there alone, drop 3b).
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
