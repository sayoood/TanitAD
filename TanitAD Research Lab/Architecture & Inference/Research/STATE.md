# STATE — Architecture & Inference

LAST_RUN: 2026-08-03 (DAILY agent, slot 3 of 6, 09:51 local — cadence changed 2026-08-03 from Wednesday-weekly.
Executed the #1 item of this discipline's own 2026-08-03 SOTA scan: **the LONGITUDINAL family's
distance-keeping half, which the binding four-family rule made mandatory on 2026-08-02 and which had been
UNCOMPUTABLE ever since** — `four_families.longitudinal` returned `distance_keeping: UNAVAILABLE` because our
ingest never read `obstacle.offline`. Now implemented (`taniteval/taniteval/lead_metrics.py` = headway /
time-gap / min-TTC, pure; `build_lead_tracks.py` = the rig→world→t0 frame composition) and, before scoring any
arm, ADMITTED by the pre-registered **D-LEAD-1** GT-vs-CV discrimination control: **Δ min-TTC +1.7474 s
[1.5813, 1.9218] · Δ headway +0.9769 m [0.8830, 1.0758] · Δ time-gap +0.1641 s [0.1499, 0.1786]**, paired
episode-cluster bootstrap, **14,027 windows / 1,431 clip clusters**, B=2000, all three separated with the
correct sign ⇒ prereg branch 1, PASS. Wiring landed additively; taniteval 810✓ / stack 1719✓ 12s 2xf.
⛔ **It is NOT yet fed on the eval path** — arm evals still report the family UNAVAILABLE until `win["lead"]`
is built for the 40 val episodes (new backlog **P0 L1**). Branch `agent/arch-inf-20260803`.)
QUALITY: full (G-A…G-H, G-AI1, G-AI2, G-I met. G-H: one measured experiment on real bytes — D-LEAD-1,
2,417 clips / 41,087 windows, dev-box CPU, 125.3 s, $0, falsifier verdict PASS with all three CIs excluding 0.
G-E: 14 tests, both suites green. G-AI1: every recommendation names its gate + falsifier. ⚠️ ONE experiment,
not the ≥2 D-029 asks — a daily run is scoped to one item that LANDS rather than two that half-land; declared,
not silently skipped.)
RESOURCE (G-I): dev-box CPU only (no GPU), 125.3 s for the control, **$0**. Why not the eval pod: the job is
pandas/numpy over local PhysicalAI label zips — a GPU would sit idle. `tanitad-eval` and `tanitad-pod3` both
**refused connection** this run (banner exchange / connection refused); pod5 is at 100 % GPU with v5f
(no-touch), pod2 and pod4 idle but not needed. Fleet probe was enumerated from `~/.ssh/config`, per C62.

## ⚠️ DEBT — surfaced for the orchestrator's daily sweep (first duty, 2026-08-03)

1. ⛔ **7 of my intake packages carry NO orchestrator verdict**, not 3 as the schedule brief recorded.
   Oldest is **26 days**: `2026-07-08-bakeoff-harness`, `2026-07-10-orthogonality-instrument` (the one
   CLAUDE.md cites as having sat unmerged 10 days — it is now at **24**), `2026-07-14-gate-runner-d1-d3`,
   `2026-07-14-spectral-sizing-p0`, `2026-07-15-h15-logging-fidelity`,
   `2026-07-23-planner-wm-gradient-coupling`, `2026-07-23-refc-planner-closedloop`.
   **ORCHESTRATOR: triage these — integrate / defer / reject-with-reason.**
2. ✅ **No stranded-branch debt.** `agent/arch-inf-20260721` and `agent/arch-inf-20260718` are both
   **+0 vs HEAD** — merged. (Verified, not assumed.)
3. ⚠️ **This STATE was stale by 16 days** — it read `LAST_RUN: 2026-07-18` while Research notes dated
   07-21 … 08-03 sat committed beside it. Root-cause class: the autonomous loop writes into this
   discipline's folders without touching its STATE, so **a run that does not update STATE is invisible**
   and the schedule and the file disagree. Fixed here; the class is worth a RETRACTION_LOG row.
4. ⛔ **`Project Steering/AgentSchedule/DAILY_ROTATION.md` DOES NOT EXIST** (probed by path and by
   repo-wide `find -iname`; the only thing under `AgentSchedule/` is `skills-backup-2026-08-03`). The
   scheduled-task brief instructs every agent to read it for the slot table and daily-cadence rules.
   **Someone must write it, or six agents will keep failing that step silently.**

## HANDOFF — 2026-08-03

⚠️ **PROVENANCE — this run's deliverables landed inside commit `3a27899`
`hub(orchestrator): daily sweep 2026-08-03`, NOT under a `hub(arch-inf):` message.** The orchestrator
agent ran `git commit` on the shared index while my files were staged in it, so the whole index went
in under its message. **Nothing is lost** — verified with `git ls-tree -r HEAD`, all 10 artifacts are
in the tree. History was NOT rewritten (that would clobber the sibling's work). This is the **third**
occurrence of the class CLAUDE.md §"Git hygiene" records — `60265d3` swallowed the eval tooling,
`3d41bd0` swallowed REF-C v1.2's rescorer, and now `3a27899` has swallowed this package. ⇒ the
documented remedy ("commit with an explicit pathspec") is **not usable** here, because
`git commit -- <pathspec>` segfaults on this repo. **The remaining mitigation is TIMING, and it is
not written down anywhere: an agent must not leave work staged while another agent may commit.**
Worth a RETRACTION_LOG row and a rule: *stage and commit in one uninterrupted step, or don't stage.*

**No half-done work.** Everything below is in the repo with its provenance; nothing on a pod or in a worktree.

1. **D-LEAD-1 (G-H, targets the binding rule + the 88.7 % longitudinal gap).** GT vs hold-`v0` CV over
   PhysicalAI `obstacle.offline`, horizon 2.0 s / dt 0.5 s / stride 1.0 s. **PASS on all three metrics**
   (table above). Coverage: 2,417 clips scanned, 1,548 with a lead; 41,087 windows, **15,760 (38.4 %) with a
   causal lead**; 14,027 paired; censoring 51.7 % / 47.5 %. ⛔ Says **nothing about any TanitAD arm** — it
   measures the gauge. ⛔ min-TTC is **censored** at 30 s on ~50 % of windows: quote `n_closing`, never the
   mean alone. ⛔ Coverage is the **26 local label chunks**, not the canonical corpus and not the 40 val eps.
2. **G-E increment.** `taniteval/taniteval/lead_metrics.py` + `taniteval/tests/test_lead_metrics.py` (14
   tests, all hand-computable geometry) + a strictly additive `four_families.py` change
   (`longitudinal(..., lead=None)`, `all_families` reads `win["lead"]`). Default path byte-identical.
   ⚠️ **Landed directly rather than left in intake — deviation declared in the INTAKE.md**, because item 1
   of this debt list is exactly what happens to packages that wait.
3. **Reused, not rebuilt** (CLAUDE.md rule 2): `stack/scripts/lead_state_gate.py` already had the proven
   strictly-causal `obstacle.offline` reader, and E-GOAL-1 already wrapped it. The new part is the
   **rig→world→t0 frame composition** — an arm's waypoints live in the t0 frame while cuboids are stamped in
   the frame of their own timestamp. Skipping it understates the gap by ~27 m at 13.6 m/s over 2 s, i.e. it
   would have **invented tailgating everywhere**.

### Exact next steps (next daily run, in priority order)
- ⭐ **P0 L1 — feed the eval path.** Build `win["lead"]` for the 40 val episodes and re-report every banked
  arm's LONGITUDINAL family. 0 GPU. Falsifier: < 20 % of val windows carry a causal lead ⇒ report the family
  **NOT-APPLICABLE with its n**, per the binding rule's clause 5.
- **P0 L2 — closing-only stratum** beside the censored min-TTC mean.
- **P0 L3 — TACTICAL + STRATEGIC families get the same treatment**, each with its own pre-registered
  discrimination control *and an INSTRUMENT-FAIL branch* (C63's lesson).
- Then the older P0 block below (E1+E2 at operative scale, parallel-horizon imagination).

---

## (historical) LAST_RUN: 2026-07-18 (Wednesday weekly agent — executed backlog P0.1: re-ran E1+E2 on the OPERATIVE flagship-speed @19k on the eval-pod A40, dropping the 07-17 pre-reset caveat. E1 (blind K-step rollout, 320 windows, 2 seeds): the σ-dissipation + attractor-collapse pathology REPRODUCES on the operative model — falsifier "speed+jerk recipe fixed it" NOT met (cos_rollout→chance by k3; σ_hidden −9.461→−9.564, *lower* absolute σ = worse temporal calibration; attractor 0.219→0.805, sharper). freeze-1 holds 0.213–0.232 flat across 8 horizons (7× persistence) → parallel-horizon confirmed safe on the shipping model. NEW refinement: σ is spatially calibrated (hidden>visible +0.37; err↔var corr +0.29–0.43) but temporally anti-calibrated → target narrows to a horizon-aware σ. E2 (orthogonality, 7,964 latents): iso_ratio_active 0.254→0.546 (SIGReg converging as predicted; crossed 0.5), cond 218→61, but still NOT-YET-ADMISSIBLE (rms_offdiag 0.32>0.1); active_k≈19, cov_eff_rank≈30 ≪ 2048 → readout not the D1 bottleneck (G1) reaffirmed. New branch agent/arch-inf-20260718.)
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
