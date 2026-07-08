# Benchmarks & Eval — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

1. **Metric-suite live dry-run on training telemetry** — LAL/TMS/OKRI/CNCE/LOPS have only seen
   design-oracle fixtures. Method: build a `ScenarioTelemetry` stream from a real evaluation
   episode (evaluate_checkpoint output + work_zone_phantom oracle), run the full suite, publish
   first real numbers to LEADERBOARD (diagnostic block, weak-claim footnote). Expected: suite
   runs end-to-end; CNCE computable from step-latest checkpoint + 4060 watt draw. Falsifier:
   contract mismatch ⇒ intake fix package.
2. **Competitor efficiency block in LEADERBOARD** — param counts + published compute of GAIA-3
   (15B, offline), Alpamayo-2 (32B), UniAD-class, vs TanitAD 261M live; each row sourced.
   This operationalizes W-05 (CNCE differentiation).

## P1

3. **WP.29 UN ADS regulation paragraph-level extraction** — close-read
   ECE-TRANS-WP.29-2026-139e.pdf 1–2 h per run; map each requirement (ISMR, DSSAD, MRM…) to a
   TanitAD artifact / ledger row in REGULATION_TRACE.md. Non-blocking Phase 0; Phase-1 safety case.
4. **Scenario-metric wiring dry-run** — full path scenario → telemetry → metric suite → leaderboard
   for work_zone_phantom (oracle mode), so W-02/W-03 scenario authoring inherits a proven contract.
5. **Metis (arXiv 2606.15869) deep-read** — nearest CNCE head-to-head academic; extract their
   normalization, decide comparability, add to watch-list dossier.

## P2

6. **NAVSIM/Bench2Drive entry requirements audit** — exact submission formats, licenses, compute
   budgets; readiness checklist for the post-D4–D6 entry (open-loop ⊥ closed-loop caveat stays
   attached to any open-loop number, arXiv 2605.00066).
7. **Per-scenario excellence leaderboard section** — schema for SCENARIO_DATABASE-linked rows
   (one row per W-scenario: our metric vs opponent's documented failure), per D-020 §5.

## Done / retired
- (2026-07-16-run) Custom metric suite shipped via intake; integrated (22 tests).
- (2026-07-08) First TanitAD gate rows on LEADERBOARD (D1 FAIL 10.94m / D2 PASS 0.872/0.940 /
  D3 BLOCKED, step-5000 preview).
