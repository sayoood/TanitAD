# Benchmarks & Eval — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

1. **`skill_score` on a real checkpoint (G1, top program risk).** The 2026-07-15 run shipped the honest
   best-of-3 floor (≈0.056 m@1s); the next step is the first *model-relative* number: run the flagship /
   refa / refb-speed held-out ADE per stratum ÷ the per-stratum floor. **Blocked on a pullable checkpoint**
   (pod running refb-speed-30k). Falsifier: skill_score ≤ 3 on straights → model near-trivial-competitive,
   "fundamental failure" reading softens. Resource: 4060/CPU, hours.
2. **`skill_score` + best-of-3 floor into the gate runner** (orchestrator triage of intake
   `2026-07-15-baseline-floor`): wire into the D1 `extra_metrics` seam so every D1 number reports the
   stratified skill denominator, not raw ADE. Speed-gate the curvature strata.
3. **LAL-v2 integration** (intake `2026-07-09-lal-v2-anticipation`, orchestrator triage): merge into
   `stack/tanitad/eval/metrics.py`, relabel LAL-v1 "reaction-onset latency", report both. Unblocks G0.6.
4. **≥3-seed SC-01 CARLA re-run** (G2) — the first live run (2026-07-08) was single-seed + scripted policy.
   Next run: ≥3 seeds, OKRI/LOPS/TMS as mean±CI, emit LAL-v2, and **measure OKRI per-seed SD** to size
   future runs (audit Result C: gap 19.5; ~2 seeds at SD≈5, up to 17 at SD≈20). Blocked on CARLA-on-pod
   (D-014, Tools&DevEnv). Falsifier: CIs overlap → no "beats baseline" claim.
5. **Closure-incursion detector fix** (H9, Friday co-own) — reads 0 on the reactive run; needs a
   lane-polygon check + collision sensor on the CARLA side. Flagged in commit `2d87acb`, NOT fixed this run.

## Done this run (2026-07-15) — the honest denominator (G1 advanced)
- **Tested best-of-3 kinematic-baseline floor shipped** (CV/go-straight/CTRV + `skill_score`, speed-gated
  strata; 8 analytic tests). Measured on 26 132 real anchors (comma-val + Cosmos-DD): honest floor
  **≈0.056 m@1s** (CTRV-dominated), correcting the framework's single-CV 0.28 m → model gap ~115× floor.
  Found+fixed the standstill yaw-noise stratification artifact (speed-gate). Cosmos-DD confirmed a poor
  maneuver source (0 % genuine sharp). Intake `2026-07-15-baseline-floor` + note. Framework §Results, KB,
  ledger, LEADERBOARD (floor block), GOALS updated.

## Done this run (2026-07-13) — backlog #3 "available NOW, no simulator" first pass
- **Robustness suite exercised on the ungated synthetic corpora** (Sayed-directed, pod-independent,
  dev-box/4060, $0). Intake `2026-07-13-cosmos-robustness-first-pass` + note
  `Research/2026-07-13-backlog3-synthetic-corpora-first-pass.md`. First data-only numbers on 13
  Cosmos-Drive-Dreams clips (annotation tars only — the suite is pixel-free, no 43 GB shard):
  **OKRI median 21.1 (0.06–268), headline**; LOPS path validated via oracle (0.844 ≈ analytic 0.8325);
  TMS/LAL-v1/LAL-v2 characterized with failure modes quantified. **WorldModel-Synthetic-Scenarios
  ruled out as a data-only geometric source** (video + VLM-caption only, no pose/boxes — documented).
  40 contract tests green. Gaps: pose-jerk noise (TMS/LAL-v1), LAL-v2 free-cruise assumption, CNCE
  latency/params + LOPS/collisions still model-dependent → CARLA-on-pod.

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
