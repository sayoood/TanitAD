# STATE — Benchmarks & Eval

LAST_RUN: 2026-07-10 (Thursday weekly agent) — base commit `859caa8`, branch
`worktree-agent-bench-eval-20260710` (D-026 worktree isolation)
QUALITY: full (all gates G-A…G-F, G-B1, G-B2, **G-H measured experiment** met; loop iteration 1 of 4,
well under budget: 3 web searches, ≈1.3 h)

## Latest run (2026-07-10) — D3 decomposition independent audit + Compounding Ratio adoption

Executed the **independent-test role** (agent duty #5) on the step-14k D3 decomposition that a
gate/arch decision could rest on (Wed commits `9bbf4ca`/`c0b22b7`). Measured, local CPU, numpy-only,
deterministic (seed 20260710), <2 s, **$0** (G-H):
- **Falsified the naive read** "rel-error falls with k ⇒ direct heads don't compound": a synthetic
  model with *superlinear* absolute-error compounding (err∝k¹·³) on drift∝k¹·⁵ reproduces the exact
  **falling** rel_k → the slope is a normalization artifact of the persistence-drift denominator, not a
  compounding readout.
- **Confirmed the recursion 2–4× claim is REAL** — it is denominator-free, i.e. the accepted
  **Compounding Ratio** CR (SkyJEPA 2606.23444 / Robotic-WM 2501.10100): CR **4.00 comma / 3.72
  physicalai** at step-14k. The K-step arm's win, honestly stated, is the within-arm CR **3.90→0.385**
  (<1 = trained rollout path is the strong one) — cleaner than the ≤2×-drift-confounded cross-model
  "imag_rel 8.13→1.03" headline.
- **Quantified the cross-model confound:** halving encoder drift inflates rel_k ×1.94 at identical
  absolute error → never compare rel_k across arms; use within-model CR.
- **Shipped:** intake `Implementation/incoming/2026-07-10-i4-compounding-instrument/`
  (`i4_compounding.py` = CR + abs/drift companions + `compounds()`; **7 analytic-GT tests, green**);
  audit proof `Implementation/i4_horizon_normalization_audit/` (+ result JSON); LEADERBOARD D3
  decomposition row + measurement-doctrine footnote; KB + HYPOTHESIS_LEDGER (H1/H5 evidence, no status
  change, P8); research note `Research/2026-07-10-d3-compounding-instrument-audit.md`; BACKLOG
  re-prioritized (CR-integration now P0).

## Prior run (2026-07-09) — SC-01 live-metric audit + LAL-v2

The first **live CARLA** SC-01 run (committed 2026-07-08) ran my metric suite on real physics and flagged
two instruments as broken. I executed my **independent-test role** on it (measured experiment, local CPU,
$0):
- **LAL-v2 shipped** (intake `Implementation/incoming/2026-07-09-lal-v2-anticipation/`, 7 analytic tests,
  standalone-green): LAL-v1's −1.5 m/s³ jerk trigger is blind to smooth comfort-bounded anticipation
  (reproduced the live −0.7/−0.7 null; cliff located exactly at −1.5 m/s³). LAL-v2 (deceleration-onset by
  speed drop; TTB generalization) gives +0.3…+3.1 s lead vs −0.3 s reactive across the whole smoothness
  sweep.
- **SC-01 LOPS 0.834 independently recomputed** = analytic E 0.8325 of the injected σ=0.3 noise (inside
  95% CI, N=5000) → reproducible, but reactive 0.0 is *structural* (presence not quality; P8).
- **OKRI ≥3-seed CI-separation rule made numeric** (gap 19.5; ~2 seeds at SD≈5; SD must be measured).
- **LEADERBOARD**: NAVSIM-v2 navhard refresh (DriveFuture 55.5 / DrivoR 56.3, Apr-2026) + first **Live
  SC-01 block** (flagged scripted/single-seed/LAL-v1-superseded) + **competitor efficiency block** (W-05:
  32 B/15 B vs 261 M). REGULATION_TRACE: ISMR anticipation-evidence row. Ledger: H15 instrument-hardening
  (no upgrade, P8).

## Where this discipline stands (prior, 2026-07-16 run)

- **Metric suite shipped** as intake pkg `Implementation/incoming/2026-07-16-eval-metric-suite/`
  (`tanitad_metrics.py` + 22 tests + INTAKE) — LAL/TMS/OKRI/CNCE/LOPS (Deep Think 14) + trajectory
  `extra_metrics` seam. Every metric has an analytic-ground-truth sanity test (G-B2). Seam **verified live**
  against Wednesday's `tanitad_gates.run_d1`. Proposed target `stack/tanitad/eval/metrics.py` (same package
  as the gate runner). **Awaiting orchestrator triage.**
- **LEADERBOARD.md**: added a separate closed-loop Bench2Drive block (TF++ 86.97/71.97, ADT 77.90/55.0) +
  NAVSIM-v2 PDM-Closed EPDMS=51.3 (navhard) + a standing open-loop⊥closed-loop footnote (2605.00066). Our
  own rows still "—" (no run yet — honest).
- **REGULATION_TRACE.md**: ISMR + DSSAD rows enriched with WP.29 June-2026 sub-asks (DSSAD standard format
  / retrievability / tamper protection; virtual-toolchain acceptance under credible-testing).
- **KNOWLEDGE_BASE / HYPOTHESIS_LEDGER**: updated (deltas newest-first; H15/H5/H11/H9 evidence-of-need, no
  unearned status change).

## Adopted rules (this discipline owns)

- **Open-loop numbers are weak claims** (2605.00066): never rank a TanitAD checkpoint on ADE/FDE alone;
  open- and closed-loop leaderboard blocks stay separate with the non-correlation footnote (G-B1).
- **Closed-loop gate claims** report **mean ± CI over ≥3 seeds**; "beats baseline" needs separated CIs
  (CARLA ~5 DS seed variance).

## Next actions (backlog, priority order)

- [ ] After orchestrator integrates the metric suite: G0.6 ("custom metric suite live") is code-complete;
      only live scenario telemetry remains.
- [ ] Backlog #3 (with Friday/Opponent Analyzer): author Ghost Cut-Through / Blind Creep / Choke Weave
      scenarios emitting the exact `ScenarioTelemetry` columns → wire LAL+LOPS / OKRI+TMS / CNCE. **Retargeted
      per D-014 (MetaDrive retired):** substrate is now **CARLA-on-pod (Docker, W31–32)** for the closed-loop
      occluder-LOPS path — the scenario/occluder/perturbation logic is sim-agnostic and ports to the CARLA
      adapter. **Available NOW, no simulator:** the ungated synthetic corpora
      `PhysicalAI-WorldModel-Synthetic` (pedestrian/emergency/nudging/weather long-tail) + `Cosmos-Drive-Dreams`
      can exercise LOPS/OKRI/LAL on pre-rendered occlusion clips before CARLA lands — a cheaper first pass.
- [ ] Backlog #4: full paragraph-level extraction of `ECE-TRANS-WP.29-2026-139e.pdf` into REGULATION_TRACE.
- [ ] Backlog #5 (gate-result audit): once a Wednesday D-gate has a real number, recompute one independently
      (fresh seed) — the Mission-Plan independent-test role.
- [ ] Populate LEADERBOARD TanitAD rows after the A40 Stage-0 run + D1–D3 through the gate runner.

## Mid-session repo advances (P8 — "re-check git" memory earned its keep, twice)

During this run Sayed's auto-commit swept my in-progress artifacts into **two of his commits**:
`5940129` absorbed the metric suite (pkg + tests + INTAKE), the research note, LEADERBOARD and
KNOWLEDGE_BASE; `47a89c4` absorbed REGULATION_TRACE, this STATE, and the HYPOTHESIS_LEDGER entry. All are
on `origin/main` (verified 0/0 vs origin). The follow-up `hub(bench-eval)` commit carries only the D-014
reconciliation below — my deliverables landed under Sayed's messages, not mine, which is why the provenance
note lives here. Also observed: `core.fsmonitor=true` hides tool-written changes on this Google-Drive repo
(git can't see them until fsmonitor is toggled off / index refreshed) — a real gotcha for every hub agent.

`47a89c4` also brought **D-014** (MetaDrive retired; sim arm = synthetic corpora + CARLA-on-pod). Consumed
and reconciled this run: the metric suite is sim-agnostic (math on telemetry columns — unaffected); the
scenario backlog re-targets CARLA-on-pod + the ungated synthetic corpora (see backlog #3). The Bench2Drive
closed-loop block I added to the LEADERBOARD is now doubly relevant (D-014 names CARLA/Bench2Drive as the
Phase-1 path).

## HANDOFF

None — 2026-07-10 run completed cleanly (worktree `worktree-agent-bench-eval-20260710`, branched from
`859caa8`). Deliverables: `i4_compounding.py` intake (7 tests green) + audit proof
(`i4_horizon_normalization_audit/`) + research note + LEADERBOARD/KB/HYPOTHESIS_LEDGER/STATE/BACKLOG.
**Open for next run / orchestrator:** (1) triage the `2026-07-10-i4-compounding-instrument` intake →
fold CR into `stack/tanitad/eval/` + `d3_decompose.analyze()` emits `{rel_k, abs_err_k, drift_k}` + CR;
(2) the still-pending `2026-07-09-lal-v2-anticipation` intake (integrate into metrics.py); (3) re-run D3
at 30k with abs_err/drift **exported** so CR is measured, not reconstructed; (4) SC-01 CARLA re-run
(≥3 seeds, LAL-v2) still blocked on the CARLA-on-pod camera path; (5) closure-incursion detector still
reads 0 (backlog P0#5). **Note for orchestrator:** the "rel-error falls with k" phrasing in D3 commit
messages should not be read as "no compounding" — see the LEADERBOARD D3 measurement-doctrine footnote.
