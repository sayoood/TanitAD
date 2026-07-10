# Benchmarks & Eval — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

1. **CR into `d3_decompose` + gate `extra_metrics`** (intake `2026-07-10-i4-compounding-instrument`,
   orchestrator triage): have `analyze()` emit `{rel_k, abs_err_k, drift_k}` per horizon + a top-level
   CR (= recursive/direct_k4, denominator-free); surface CR in the D3 gate REPORT. Additive, no
   threshold change. *(Instrument shipped this run; integration is the orchestrator step.)*
   Expected number: CR replaces the rel-slope as the compounding readout; falsifier: if exported
   per-window abs errors give CR ≠ the reconstructed 4.00/3.72, the shared-denominator identity is wrong.
2. **Re-run D3 with abs_err/drift exported** on the next checkpoint (30k) so CR is *measured directly*,
   not reconstructed from medianed ratios. Report CR per corpus + within-arm CR for base vs K-step.
3. **LAL-v2 integration** (intake `2026-07-09-lal-v2-anticipation`, orchestrator triage): merge into
   `stack/tanitad/eval/metrics.py`, relabel LAL-v1 "reaction-onset latency", report both. Unblocks G0.6.
4. **≥3-seed SC-01 CARLA re-run** — the first live run (2026-07-08) was single-seed + scripted policy.
   Next run: ≥3 seeds, OKRI/LOPS/TMS as mean±CI, emit LAL-v2, and **measure OKRI per-seed SD** to size
   future runs (audit Result C: gap 19.5; ~2 seeds at SD≈5, up to 17 at SD≈20). Blocked on the
   CARLA-on-pod camera path (Tools&DevEnv graphics-pod recipe). Falsifier: CIs overlap → no claim.
5. **Closure-incursion detector fix** (H9, Friday co-own) — reads 0 on the reactive run; needs a
   lane-polygon check + collision sensor on the CARLA side. Flagged in commit `2d87acb`.

## Done this run (2026-07-10)
- **D3 decomposition independent audit** (G-H, $0, seed 20260710): proved "rel falls with k" is a
  normalization artifact (superlinear-compounding synthetic reproduces it); confirmed recursion
  compounding is REAL (CR 4.00/3.72); quantified cross-model rel_k confound (≤2×). Shipped hardened
  instrument `i4_compounding.py` (7 analytic tests green) + LEADERBOARD D3 doctrine footnote + KB/ledger.

## Done prior run (2026-07-09)
- **Competitor efficiency block** (W-05): shipped to LEADERBOARD — Alpamayo-2 32B / GAIA-3 15B
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
