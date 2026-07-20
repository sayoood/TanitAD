# Benchmarks & Eval — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## v2.1 / v3 LEVER — flagship high-speed longitudinal fix (Sayed 2026-07-18, deferred from v2)
The `taniteval/pathspeed.py` decoupled long/lat panel found flagship-30k's high-speed loss to CTRV is
**LONGITUDINAL, not lateral**: 89% of the 2s sq-error is along-track; at high speed the model
**over-predicts speed by +0.66 m/s** (long-RMSE 1.38m vs CTRV 0.077m; lateral only 0.63m); error
compounds 0.07→0.91m over the horizon. It plans the PATH well, the SPEED poorly. **Lever (deferred —
"launch v2 now, measure, add if needed"):** up-weight the along-track/speed-profile error term +
speed-stratified/balanced sampling + possibly a speed-calibration/anti-overshoot term. GATE: read
pathspeed longitudinal RMSE + speed-bias at the v2 5k/15k milestones FIRST (the rebalance +
anti-shortcut levers may already reduce it); implement as v2.1 only if it persists. See
`memory/flagship-longitudinal-lever.md`.


## P0 — FLEET DIRECTIVE 2026-07-17 (Sayed; supersedes prior P0 ordering; resource-mandated G-I)

Context: `Project Steering/FLEET_REVIEW_2026-07-17.md`. **TanitEval is now the canonical eval
substrate** — a deployed app on the eval pod (`/root/taniteval`: fresh grounded-rollout benchmarks,
CTRV baseline, paired-bootstrap A/B, BEV+camera-overlay viz, imagination panel, tactical/strategic
planning panel, regression goldens, report generator). Your 2026-07-15/17 deliverables (best-of-3
floor, Ridge ego-status ceiling, both-convention L2) were graded A in review and are CONVERGENT
with TanitEval's findings (your CTRV 0.545 ≈ its 0.544 on different corpora). Your job now is to
make TanitEval bulletproof, not to build a parallel harness (compare_arms.py's pod-pull machinery
is superseded; its behavior-decodability block and `eval_behavior --config` fix are lifted).

1. **Port your floors/ceilings INTO TanitEval and recompute on the canonical val (EVAL POD).**
   `baseline_predictors.py` best-of-3 (CV/go-straight/CTRV, speed-gated strata) + the learned
   `RidgeTrajectoryHead` ego-status ceiling + both-convention L2 + per-stratum `skill_score` →
   `taniteval/bench.py` rows beside CTRV. Deliverable: leaderboard shows floor/ceiling/skill_score
   for all 4 arms + flagship@30k when it lands. Falsifier: ridge ceiling ≤ CTRV on the canonical
   val → the learned shortcut adds nothing here (report; keep CTRV as the bar).
2. **Route base-rate + behavior-decodability into the planning panel (EVAL POD, hours).** The
   flagship's route-from-vision 67.5% is UNJUDGED without the majority-class base rate — compute
   it; add your `_behavior_probe` (linear decodability vs chance) to `taniteval/planning.py`.
3. **Closed-loop bring-up = your #1 multi-run arc (CARLA-on-pod, W31-32, with Opponent).** The
   open-loop⊥closed-loop rule you adopted makes everything above "weak claims" — the arbiter is
   LAL/OKRI/LOPS/CNCE on live scenarios. Wire Opponent's SC-13/W-09/SC-06 scripts to emit
   `ScenarioTelemetry`; your metric suite is sim-agnostic and ready. Escalate the pod/ledger row
   via M-3 if blocked.
4. **Nightly TanitEval ownership:** the regression goldens + `nightly.sh` are yours now — arm the
   detached scheduler, watch for regressions when flagship@30k / refbpatch ckpts rotate in.

## P0 — next run

1. **`skill_score` on a real checkpoint (G1, top program risk).** The floor (≈0.056 m@1s ADE, 07-15) and
   the metric-BEV **open-loop L2 shortcut ceiling (0.66 m avg, 07-17)** are both shipped; the missing half
   is the first *model-relative* number. **New concrete step:** add a metric-BEV ego-frame trajectory decode
   to `driving_diagnostic.py` (so the model number lives in the SAME space as the 0.66 m shortcut, not
   camera-frame) → emit `skill_score = model_L2 ÷ shortcut` per stratum. **Blocked on a pullable post-reset
   checkpoint** (pods training; gated HF `Sayood/*`). Local `ckpt_full.pt` is pre-reset camera-frame → NOT
   comparable (do not use, G-B1). Falsifier: skill_score ≤ 3 on straights → model near-trivial-competitive.
2. **`skill_score` + floor + open-loop-L2 into the gate runner** (orchestrator triage of intakes
   `2026-07-15-baseline-floor` + `2026-07-17-openloop-l2-egostatus-shortcut`): wire into the D1
   `extra_metrics` seam so every D1 number reports the stratified skill denominator AND a metric-BEV L2 vs
   the ego-status shortcut, not raw camera-frame ADE. Speed-gate the curvature strata.
3. **LAL-v2 integration** (intake `2026-07-09-lal-v2-anticipation`, orchestrator triage): merge into
   `stack/tanitad/eval/metrics.py`, relabel LAL-v1 "reaction-onset latency", report both. Unblocks G0.6.
4. **≥3-seed SC-01 CARLA re-run** (G2) — the first live run (2026-07-08) was single-seed + scripted policy.
   Next run: ≥3 seeds, OKRI/LOPS/TMS as mean±CI, emit LAL-v2, and **measure OKRI per-seed SD** to size
   future runs (audit Result C: gap 19.5; ~2 seeds at SD≈5, up to 17 at SD≈20). Blocked on CARLA-on-pod
   (D-014, Tools&DevEnv). Falsifier: CIs overlap → no "beats baseline" claim.
5. **Closure-incursion detector fix** (H9, Friday co-own) — reads 0 on the reactive run; needs a
   lane-polygon check + collision sensor on the CARLA side. Flagged in commit `2d87acb`, NOT fixed this run.

## Done this run (2026-07-17) — leaderboard-comparable denominator (G1 advanced)
- **nuScenes-style open-loop L2 protocol + no-vision ego-status shortcut shipped** (intake
  `2026-07-17-openloop-l2-egostatus-shortcut`, 8 analytic tests). Metric-BEV ego frame, both averaging
  conventions, `collision_rate` proxy, `RidgeTrajectoryHead` (AD-MLP repro). Measured on 7 920 comma-hwy
  val anchors (held out by clip): **shortcut ceiling avg L2 0.66 m** ≈ CTRV; **comma is 73.9 % straight =
  nuScenes' 73.9 %** → open-loop L2 is a weak capability test. `skill_score = model_L2 ÷ 0.66 m` now in
  community units. cosmos-urban: learned shortcut (1.19 m) beats the kinematic floor (1.34). LEADERBOARD
  (open-loop-L2 block), KB, GOALS, ledger updated. Blocker: model-relative number needs a post-reset ckpt.

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
