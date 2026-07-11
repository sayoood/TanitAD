# Benchmarks & Eval — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

1. **D1/D3 bootstrap re-read (R3, this-run finding)** — re-run the step-14k **and** step-21k D1 comparison
   with (a) a shared fixed val set of **≥20 routes** across both checkpoints and (b) the bootstrap wrapper
   (`incoming/2026-07-11-d1-gate-bootstrap/`) → report `ade@1s_mean ± CI95`. Needs the 14k+21k checkpoints
   (pod/loop). Goal: settle whether D1 moved at all. Expected: CIs overlap → **no regression**. Falsifier:
   14k and 21k CIs separate → real movement (revisit). Blocks any D1 trend line in the gate narrative.
2. **run_d1_bootstrap integration** (intake `2026-07-11-d1-gate-bootstrap`, orchestrator triage): add the
   helper to `stack/tanitad/eval/gates.py` + `--d1-seeds N` in `evaluate_checkpoint.py`; monitor previews
   then emit mean±CI, not single-seed points. 4 tests green.
3. **LAL-v2 integration** (intake `2026-07-09-lal-v2-anticipation`, orchestrator triage): merge into
   `stack/tanitad/eval/metrics.py`, relabel LAL-v1 "reaction-onset latency", report both. Unblocks G0.6.
4. **≥3-seed SC-01 CARLA re-run** — the first live run (2026-07-08) was single-seed + scripted policy.
   Next run: ≥3 seeds, OKRI/LOPS/TMS as mean±CI, emit LAL-v2, and **measure OKRI per-seed SD** to size
   future runs (audit Result C: gap 19.5; ~2 seeds at SD≈5, up to 17 at SD≈20). Falsifier: CIs overlap → no
   "beats baseline" claim.
5. **Closure-incursion detector fix** (H9, Friday co-own) — reads 0 on the reactive run; needs a
   lane-polygon check + collision sensor on the CARLA side. Flagged in commit `2d87acb`, NOT fixed this run.

## Done this run (2026-07-11)
- **D1 ADE statistical-power audit** (G-H): measured the estimator's sampling variance on the real
  step-6500 ckpt (4060, $0) → step-21k D1 "regression" is inside the ±4.5 m noise band, NOT decision-grade.
  Shipped `d1_power_audit/` (diagnostic, 4 tests) + `incoming/2026-07-11-d1-gate-bootstrap/` (wrapper, 4
  tests). Adopted rule R1 (D1/D3 mean±CI over ≥5 seeds). Audited the loop's `d1_probe_capacity.py` (same
  fragility). LEADERBOARD D1 row + statistical-power footnote.

## Done earlier (2026-07-09)
- **Competitor efficiency block** (was P0 #2, W-05): shipped to LEADERBOARD — Alpamayo-2 32B / GAIA-3 15B
  / DriveFuture vs TanitAD 261M, sourced to Opponent profiles.
- ~~Metric-suite live dry-run~~ (done 2026-07-08 MVP loop): TMS/CNCE on real telemetry + wiring dry-run.

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
