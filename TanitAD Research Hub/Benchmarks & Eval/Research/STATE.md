# STATE — Benchmarks & Eval

LAST_RUN: 2026-07-16 (Thursday weekly agent, W29) — base commit `ff89194`
QUALITY: full (all gates G-A…G-F, G-B1, G-B2 met; loop iteration 1 of 3, well under budget)

## Where this discipline stands

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
- [ ] Backlog #3 (with Friday/Opponent Analyzer): author Ghost Cut-Through / Blind Creep / Choke Weave as
      MetaDrive occluder configs emitting the exact `ScenarioTelemetry` columns → wire LAL+LOPS / OKRI+TMS
      / CNCE. Substrate = Monday's front-cam occluder path (2026-07-13 Tools&DevEnv). Blocked on supervised
      MetaDrive source install (PROJECT_STATE W2).
- [ ] Backlog #4: full paragraph-level extraction of `ECE-TRANS-WP.29-2026-139e.pdf` into REGULATION_TRACE.
- [ ] Backlog #5 (gate-result audit): once a Wednesday D-gate has a real number, recompute one independently
      (fresh seed) — the Mission-Plan independent-test role.
- [ ] Populate LEADERBOARD TanitAD rows after the A40 Stage-0 run + D1–D3 through the gate runner.

## HANDOFF

None — run completed cleanly (session-end ritual done, committed + pushed). No half-done work in flight.
