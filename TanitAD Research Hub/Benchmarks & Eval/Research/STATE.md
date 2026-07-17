# STATE — Benchmarks & Eval

LAST_RUN: 2026-07-17 (Thursday scheduled) — open-loop L2 + ego-status shortcut ceiling; branch `agent/benchmarks-eval-20260717` (worktree off bench tip e8fca8e, D-026, off-Drive)
QUALITY: full (1 measured experiment with held-out numbers; new metric ships with 8 analytic tests; $0, dev-box CPU, no pod touched)

## Latest run (2026-07-17) — the denominator in leaderboard-comparable units (G1 advanced)

Continued the #1 program-risk work (single-camera driving gap) — took the denominator from our internal
camera-frame/floor units into **community-comparable nuScenes-style open-loop L2** (metric-BEV ego frame).
Data-only, $0, local CPU, 7 920 comma-hwy + 318 cosmos-urban held-out (clip-level) val anchors.
Deliverable: intake `Implementation/incoming/2026-07-17-openloop-l2-egostatus-shortcut/` (module + 8 tests
+ run + results) + note `Research/2026-07-17-openloop-l2-egostatus-shortcut.md`.
- **Shipped the open-loop L2 protocol** (metric-BEV metres, both UniAD-`pointwise` / ST-P3-`cumulative`
  conventions, collision proxy) + a **no-vision learned ego-status shortcut** (`RidgeTrajectoryHead`,
  AD-MLP repro, arXiv 2312.03031). Proposed target `stack/tanitad/eval/openloop_l2.py`.
- **F1 (measured):** the no-vision shortcut ceiling on comma-hwy = **avg L2 0.66 m** (0.144/0.552/1.256
  @1/2/3s), statistically tied with CTRV (0.656) → **`skill_score = model_L2 ÷ 0.66 m` now defined in
  leaderboard units.** cosmos-urban: the *learned* shortcut (1.19 m) beats the fixed kinematic floor (1.34).
- **F2 (methodology):** **comma highway is 73.9 % straight — identical to nuScenes' 73.9 %** → our open-loop
  val inherits the exact ego-status-shortcut pathology; aggregate open-loop L2 is a **weak capability test**.
  The driving-capability verdict must be per-stratum `skill_score` + closed-loop, never an aggregate L2.
- **F3 (hygiene):** nuScenes L2 has two undisclosed averaging conventions differing ~2× — every TanitAD/
  competitor L2 row must state which (G-B1). Added a pre-registered open-loop **reporting protocol** to
  LEADERBOARD (shortcut ceiling + skill_score + open⊥closed footnote + unit disclosure beside every number).
- **Caveat/blocker (P8):** the model-relative number needs a post-reset ckpt decoded in metric-BEV ego
  frame; local `ckpt_full.pt` is pre-reset **camera-frame** (not comparable → NOT used, G-B1); post-reset
  ckpts on pods (training, off-limits) / gated HF. Queued: add a metric-BEV decode to `driving_diagnostic.py`.

## Prior run (2026-07-15) — the honest denominator for the driving-capability gap

Targeted the #1 program risk (single-camera driving gap, `DRIVING_DIAGNOSTIC_FRAMEWORK.md`). Data-only,
$0, local CPU, 26 132 anchors (comma2k19-val + Cosmos-DD). Deliverable: intake
`Implementation/incoming/2026-07-15-baseline-floor/` (metric + 8 tests + run + results) + note
`Research/2026-07-15-baseline-floor-honest-denominator.md`.
- **Shipped a tested best-of-3 kinematic-baseline floor** (CV / go-straight / **CTRV** + `skill_score`,
  speed-gated curvature strata; 8 analytic-ground-truth tests green). Proposed target
  `stack/tanitad/eval/baselines.py`, plugs into the D1 gate `extra_metrics` seam.
- **F1 (measured):** the diagnostic's single-CV floor (≈0.28 m@1s) is not honest — the best-of-3 floor is
  **≈0.056 m@1s** (CTRV wins 55–58 %; CV overstates the floor 4.6× on curves). → flagship held-out 6.44 m =
  **~115× the trivial floor** (framework verdict *reinforced*, not overturned). **D1 should divide by the
  per-stratum best-of-3 floor.**
- **F2 (protocol fix):** curvature stratification must be speed-gated (v≥2 m/s) — 12.4 % of comma anchors
  are standstill yaw-noise mislabeled "sharp" (κ=yaw_rate/v singular at v→0). Appended to the framework §Results.
- **F3 (data):** the ungated Cosmos-DD sample is a poor maneuver source (0 % genuine sharp, 95.8 % straight);
  comma-highway carries more real curve content. Refines 2026-07-13 note + framework §D2 → Data-Eng flag.
- **Ecosystem (D-028 seam):** logged Occluded-nuScenes (2510.18552) + Bench2Drive-Robust (2605.18059,
  occlusion×latency = our edge pair) + IDOL (2605.31476, supports the diagnostic root-cause remedy).
- **Caveat (P8):** baselines use privileged GT ego-state → this is the *denominator*, not a model
  competitor; no model ADE computed (pod busy with refb-speed-30k). Follow-up: `skill_score` on a checkpoint.
- Created `GOALS.md` (D-029, was missing) — G1 driving-gap denominator advanced.

## Prior run (2026-07-13) — backlog #3 first pass on the ungated synthetic corpora
_(historical — LAST_RUN before 2026-07-15)_
LAST_RUN: 2026-07-13 (Sayed-directed pod-independent task) — backlog #3 first pass
QUALITY: full (data-only first pass on real ungated corpora; pipeline validated end-to-end; 40 tests
green; $0, dev-box/4060, no pod touched)

## Latest run (2026-07-13) — backlog #3 first pass on the ungated synthetic corpora

Exercised the robustness suite on the ungated synthetic corpora (the "available NOW, no simulator"
half of backlog #3), dev-box/4060 only, $0. Deliverable: intake
`Implementation/incoming/2026-07-13-cosmos-robustness-first-pass/` + note
`Research/2026-07-13-backlog3-synthetic-corpora-first-pass.md`.
- **First data-only numbers on 13 Cosmos-Drive-Dreams clips.** The suite is **pixel-free** → needs only
  the small per-clip RDS-HQ annotation tars (ego `vehicle_pose` + `all_object_info` 3D boxes +
  intrinsics, ~15 MB/clip), **not** the 43 GB video shards. **OKRI** median **21.1** (0.06–268),
  non-trivial on 12/13 — the headline data-only robustness number, scaling v²×occlusion as designed.
- **LOPS path validated on real occlusion geometry**: data-only LOPS = 0.0 (honest, no model estimate);
  perfect-perception oracle (wm=gt+N(0,0.3)) → mean **0.844** vs analytic **0.8325** (the 2026-07-09
  σ=0.3 constant). Occlusion detector fires on 13/13 clips.
- **WorldModel-Synthetic-Scenarios is video + VLM-caption ONLY** (no ego pose / no 3D boxes; probe in
  `worldmodel_structure.json`) → cannot feed the geometric suite data-only; needs a perception/pose
  model (same dependency as the closed loop). Stays a scene-diversity source.
- **Gaps quantified:** pose-derived jerk is noise-amplified → TMS collapses (median 0.039; 0.117 with
  5-tap smoothing) and LAL-v1 positivity is a jerk-noise artifact (re-confirms H15); LAL-v2 free-cruise
  assumption violated on 7/13 logged clips; CNCE (latency/params) + LOPS/collisions still model-dependent.
- Nothing touches the running contract — glue is a consumer awaiting orchestrator triage (proposed
  target `stack/scripts/`).

## Prior run (2026-07-09) — SC-01 live-metric audit + LAL-v2 (base commit `c4375f8`)

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
      adapter. **Available NOW, no simulator — FIRST PASS DONE 2026-07-13**
      (`Research/2026-07-13-backlog3-synthetic-corpora-first-pass.md`): `Cosmos-Drive-Dreams` feeds the suite
      data-only (ego pose + 3D boxes → OKRI/TMS/LAL numbers, occlusion geometry validated; LOPS oracle ≈
      analytic). **`PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios` is video + VLM-caption
      only** (no pose/boxes) → NOT a data-only geometric source. Remaining here = the scripted occluder
      scenarios (Ghost Cut-Through / Blind Creep / Choke Weave) on CARLA-on-pod for real LOPS/CNCE.
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

None — 2026-07-09 run completed cleanly. Deliverables: LAL-v2 intake pkg (7 tests green),
`audit_sc01.py`+`audit_results.json`, research note, LEADERBOARD/REGULATION_TRACE/KNOWLEDGE_BASE/
HYPOTHESIS_LEDGER/STATE/BACKLOG updates. Committed + pushed as `hub(bench-eval)`. **Open for next run /
orchestrator:** (1) triage the LAL-v2 intake → integrate into `stack/tanitad/eval/metrics.py`; (2)
closure-incursion detector still reads 0 (not fixed this run — backlog #3); (3) next SC-01 CARLA run must
be ≥3 seeds + emit LAL-v2.
