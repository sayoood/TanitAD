# Benchmarks & Eval — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

1. **LAL-v2 integration** (intake `2026-07-09-lal-v2-anticipation`, orchestrator triage): merge into
   `stack/tanitad/eval/metrics.py`, relabel LAL-v1 "reaction-onset latency", report both. Unblocks G0.6.
   *(Metric shipped this run; integration is the orchestrator step.)*
2. **≥3-seed SC-01 CARLA re-run** — the first live run (2026-07-08) was single-seed + scripted policy.
   Next run: ≥3 seeds, OKRI/LOPS/TMS as mean±CI, emit LAL-v2, and **measure OKRI per-seed SD** to size
   future runs (audit Result C: gap 19.5; ~2 seeds at SD≈5, up to 17 at SD≈20). Goal: first decision-grade
   live SC-01 rows. Falsifier: CIs overlap → no "beats baseline" claim.
3. **Closure-incursion detector fix** (H9, Friday co-own) — reads 0 on the reactive run; needs a
   lane-polygon check + collision sensor on the CARLA side. Flagged in commit `2d87acb`, NOT fixed this run.

## Done this run (2026-07-09)
- **Competitor efficiency block** (was P0 #2, W-05): shipped to LEADERBOARD — Alpamayo-2 32B / GAIA-3 15B
  / DriveFuture vs TanitAD 261M, sourced to Opponent profiles.
- ~~Metric-suite live dry-run~~ (done 2026-07-08 MVP loop): TMS/CNCE on real telemetry + wiring dry-run.

## P1

0. **AUTOPILOT-VQA probe-transfer (arXiv 2607.08745, Sayed-delivered 2026-07-11, D-028 seam: ours).**
   Step 1: locate the Kaggle competition dataset, record license verdict (eval-use? research-use?)
   in the ranked-dataset list. Step 2 (if license OK): fit A3-style calibrated probes from OUR 30k
   latents onto their 9 safety-category labels (weather, road surface, entities, impact location…)
   on a ≤500-clip subset — external-validity number for the inherent-safety edge on never-trained
   real incident data. Falsifier: probe AUC ≈ chance → latents do NOT carry safety-relevant scene
   variables → H-safety evidence weakened (report honestly). Resource: 4060, hours.
   See `../2026-07-11-sayed-papers-screening.md`.

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
