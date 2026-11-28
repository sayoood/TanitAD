# Benchmarks & Eval — 2026-07-15 — The honest denominator: stratified trivial-baseline floor for the driving diagnostic

**Run:** weekly Benchmarks & Eval agent (Thursday, scheduled). **Loop:** iteration 1.
**Budget:** 5 web searches, ≈1.5 h wall-clock, **$0** (dev-box CPU, local data only, no pod / no CARLA / no training touched). Well under caps (25 searches / 4 h / 4 iters).
**Consumed this week:** Architecture 2026-07-14 (gate runner + JEPA-WM deltas — the `extra_metrics` seam my suite plugs into; decode≠planning), Data-Eng 2026-07-14 (Cosmos-Drive-Dreams loader + `poses_to_signals` I reuse), the Sayed-directed `Benchmarks & Eval/DRIVING_DIAGNOSTIC_FRAMEWORK.md` (top program risk), my own 2026-07-13 Cosmos first-pass. Every claim carries a source link or repo-path (G-A).

**Mid-session git note (P8):** the repo is mid-week in a multi-worktree state — `main` is 7 commits behind the bench branch and neither is a superset. Worked in an isolated worktree `agent/benchmarks-eval-20260715` branched off the latest bench tip `e17435e` (D-026). Orchestrator reconciles at Friday triage.

---

## 0. Why this experiment (targets the #1 program risk directly)

The top program risk is the single-camera driving-capability gap
(`DRIVING_DIAGNOSTIC_FRAMEWORK.md`). Its decisive 2026-07-12 result reads: *"the model
is ~10–15× worse than constant-velocity everywhere, including straight highway
(2.75 vs 0.18 m)."* That verdict rests entirely on **one** trivial baseline —
constant-velocity — measured on comma highway (framework §0: CV ADE@1s ≈ 0.28 m,
straight 0.18 m). Two problems with a single-CV denominator:

1. **CV is the *weakest* of the standard kinematic nulls on any non-straight
   motion.** The trajectory-prediction literature uses CV / CA / **CTRV**
   (constant-turn-rate-velocity) precisely because CV cannot follow a curve
   ([arXiv 2503.03262](https://arxiv.org/pdf/2503.03262); CTRV traces a circular arc,
   "suitable for predicting paths on curved roads").
2. **The community already treats the CV floor as load-bearing** and *stratum-sensitive*:
   NAVSIM v2 uses a constant-velocity agent as a **triviality filter**, removing frames a
   CV agent solves with PDMS > 0.8 ([arXiv 2506.04218](https://arxiv.org/html/2506.04218v1)).
   If CV-triviality depends on curvature, that filter (and our diagnostic denominator) must be
   stratified.

So the eval question, pre-registered: **is the trivial-baseline floor corpus- and
curvature-dependent, and is CV the right denominator?** Data-only, $0, CPU — this
establishes the *denominator* the model's ADE is divided by; it computes **no** model output.

## 1. Method (shipped as a tested metric — G-B2)

Intake `Implementation/incoming/2026-07-15-baseline-floor/` — `baseline_predictors.py`
(numpy-only) + `test_baseline_predictors.py` (**8 analytic-ground-truth tests, all green,
0.16 s**) + `run_baseline_floor.py` + `results_baseline_floor.json`. Proposed target
`stack/tanitad/eval/baselines.py` (sibling of the integrated `metrics.py`/`gates.py`).

Three causal kinematic baselines, each predicting future waypoints in the **ego frame at the
anchor** (FLU, metres), from samples ≤ t only:
- **CV** — world velocity vector held constant (captures lateral drift);
- **go-straight** — constant heading + speed (pure forward);
- **CTRV** — constant turn-rate + velocity (circular arc).

`skill_score(model_ade, baselines) = model_ade / min(baseline_ade)` — the honest
denominator is the **best-of-3**, per anchor. Anchors are stratified by 1 s future heading
change (straight/gentle/sharp) **and** speed-gated: anchors below 2 m/s are tagged
`standstill` because curvature κ = yaw_rate / v is singular as v → 0 (the same singularity
`poses_to_signals` already clips for steering).

**Sanity tests (analytic):** straight line → all three baselines exact (<1e-6); constant arc
→ CTRV exact, CV/go-straight fail predictably (chord < arc, error grows with horizon);
ego-frame transform vs hand computation; crab-motion → CV models lateral drift, go-straight
does not; **standstill yaw-jitter is gated to `standstill`, not `sharp`**; skill_score arithmetic.

**Data (local, 10 Hz, pose convention `[x,y,yaw,v]`):** 13 Cosmos-Drive-Dreams clips
(`cosmos_bench3`, 30 fps → stride-3, `poses_to_signals`) = **1 022 anchors**; comma2k19-val 90
episodes (`eval/comma2k19-val-…`, native 10 Hz) = **25 110 anchors**.

## 2. Results (measured)

**Per-stratum best-of-3 floor and the CV-vs-CTRV gap (median ADE, m):**

| corpus | stratum | n | **floor@1s** | CV@1s | CTRV@1s | CV/CTRV | floor@2s |
|---|---|---:|---:|---:|---:|---:|---:|
| comma-hwy (25 m/s) | straight | 18 785 | **0.056** | 0.088 | 0.062 | 1.4× | 0.206 |
| | gentle | 3 008 | **0.059** | 0.275 | 0.060 | **4.6×** | 0.228 |
| | sharp (genuine) | 212 | **0.164** | 0.404 | 0.167 | 2.4× | 0.608 |
| | standstill *(artifact)* | 3 105 | 0.003 | 0.004 | 0.005 | — | 0.009 |
| cosmos-urban (12.9 m/s) | straight | 979 | **0.086** | 0.102 | 0.087 | 1.2× | 0.259 |
| | gentle | 18 | 0.156 | 0.171 | 0.156 | 1.1× | 0.530 |

Overall best-null wins: **CTRV 55–58 %**, CV 20–30 %, go-straight 18–23 % (both corpora).
Curvature population (speed-gated): comma-hwy = 74.8 % straight / 12.0 % gentle / **0.8 %
genuine sharp** / 12.4 % standstill; cosmos-urban = 95.8 % straight / 1.8 % gentle / **0 %
sharp** / 2.4 % standstill.

### Findings (honest, P8)

**F1 — The honest floor is ≈ 0.056–0.06 m @1s (best-of-3, CTRV-dominated), not the single-CV
0.28 m the diagnostic used.** The diagnostic's denominator is essentially *CV's error on
curves* (gentle CV = 0.275 m). Using CV-only overstates the floor 1.4× on straights and **4.6×
on gentle curves**, which *understates* how far the vision model is from trivial. The best-of-3
floor is flat (~0.056–0.059 m) across straight AND gentle — a matched kinematic null tracks
steady curves as easily as straights.

**F2 — Consequence for the diagnostic verdict (reinforces, does not overturn).** Against the
corrected floor (~0.056 m), the model's held-out D1 ADE@1s = 6.44 m is **~115× the trivial
floor** (not "10–15× worse than CV"); even the vision oracle-ceiling (1.65 m, framework §B) is
**~29× the floor**. The gap is *larger* than stated, but the framework's direction —
representation **and** readout both broken, no explicit metric-ego-trajectory target — is
unchanged and reinforced. **Recommendation:** the D1 gate should report `skill_score` =
model_ADE ÷ per-stratum best-of-3 floor, not divide by a single CV scalar. (The residual gap
between my 0.056 m and the framework's 0.28 m is estimator/protocol-dependent — velocity-estimation
window, anchor selection, horizon averaging — which is exactly why the denominator must be a
*specified, standardized* best-of-3, speed-gated, per-stratum floor.)

**F3 — Curvature stratification MUST be speed-gated.** 12.4 % of comma "anchors" are
near-standstill (median 0.01 m/s); ungated, GNSS/INS heading jitter mislabels them as "sharp"
with a spurious 0.003 m floor (trivially-predictable no-motion). Speed-gating recovers the
*genuine* sharp stratum (n=212, floor 0.164 m@1s — the hardest, as expected). The framework's
own diagnostic uses curvature strata → **it should adopt this gate** or its "sharp" bucket is
polluted.

**F4 — The ungated Cosmos-Drive-Dreams sample is a poor maneuver source.** 95.8 % straight,
1.8 % gentle, **0 % genuine sharp**, median 12.9 m/s. comma-highway ironically carries **more**
real curve content (12 % gentle + 0.8 % highway-speed sharp). This refines my 2026-07-13 note
(Cosmos-DD = scene-diversity source) *and* the framework §D2 curve-scarcity remedy: **this
Cosmos sample does not supply the missing curve/maneuver events** — the semantic-label dataset
survey (nuPlan/DriveLM/CoVLA, Data-Eng backlog) remains the path.

**F5 — CV is the wrong default null.** CTRV wins the majority of anchors on both corpora; CV
wins only 20–30 %. NAVSIM v2's CV-triviality filter would therefore *under-filter* curve frames
that CTRV solves trivially but CV does not — a subtle over-crediting of curve performance
(modest claim; their filter is closed-loop-PDMS-based, not open-loop ADE).

### Caveats (P8, load-bearing)
- Baselines use **privileged GT ego-state** (velocity/yaw-rate from the pose stream) the
  vision model never sees → the floor is a lower bound on *trivial predictability of the smooth
  derived trajectory*, **the denominator, not a fair model competitor**. The fair model question
  stays the vision oracle-ceiling (framework §B, 1.65 m).
- The derived pose stream (integrated `vehicle_pose` / GNSS-INS) is smooth → baselines look very
  strong; a perception-noise floor would be higher (separate question).
- Cosmos sample = 13 clips → small-sample characterization, not a corpus verdict.
- Strata (future-1s heading change, speed-gated) differ from the framework's (path curvature) →
  population fractions are not directly comparable; the qualitative ordering holds.
- **No model ADE computed this run** (no local checkpoint; pod is running `refb-speed-30k`).
  This delivers the corrected denominator; wiring the model's ADE through `skill_score` is the
  follow-up when a checkpoint is pullable.

## 3. Benchmark-ecosystem deltas (D-028 seam: benchmark/dataset releases → we own)

- **Occluded nuScenes** ([arXiv 2510.18552](https://arxiv.org/abs/2510.18552)) — multi-sensor
  occlusion-robustness benchmark (4 camera + parameterised radar/LiDAR occlusion types). Directly
  relevant to our OKRI/LOPS occlusion suite — a public, citable occlusion stressor. → LEADERBOARD
  watch + Data-Eng flag (candidate real-occlusion eval set once our perception path exists).
- **Bench2Drive-Robust** ([arXiv 2605.18059](https://arxiv.org/html/2605.18059)) — closed-loop AD
  under deployment perturbations (occlusion **and inference latency**); SimLingo degrades sharply
  under severe occlusion + latency. This is our exact edge pair (OKRI/LOPS × CNCE); the strongest
  clean-baseline-degrades-under-stress datapoint yet → competitor context for the closed-loop block.
- **NAVSIM v2** — `navhard_two_stage` + **Scaling Self-Play** ([arXiv 2606.19641](https://arxiv.org/html/2606.19641v1))
  a new entrant on navhard; PDM-Closed EPDMS 51.3 stands. No change to our posted number.
- **"Creating Impactful AD Datasets"** ([arXiv 2607.00710](https://arxiv.org/abs/2607.00710),
  Jul-1-2026) — research-gap→benchmark methodology guide; feeds `Data Engineering/OWN_DATASET_PLAN.md`.
- **IDOL** ([arXiv 2605.31476](https://arxiv.org/pdf/2605.31476)) "Inverse-Dynamics-Guided Future
  Prediction" — external support for the framework's **#1 root-cause lever** (inverse-dynamics /
  ego-motion supervision grounds the latent). → pointer to Architecture; strengthens the
  diagnostic's remedy, no status change.

## 4. Actionable recommendations (each names its gate — G-B)

1. **[D1 gate / WP6 — ready]** Adopt `skill_score` = model_ADE ÷ **per-stratum best-of-3,
   speed-gated** floor in the gate runner's `extra_metrics`. Denominators to hardcode from this
   run (comma-hwy): straight 0.056, gentle 0.059, sharp 0.164 (@1s). Falsifier: if a future
   corpus shows CV ≤ CTRV everywhere, the best-of-3 collapses to CV and the change is a no-op.
2. **[Diagnostic framework — protocol fix]** Speed-gate the curvature strata (v ≥ 2 m/s) before
   the next `driving_diagnostic` run; its "sharp" bucket is otherwise standstill-polluted.
   Appended to `DRIVING_DIAGNOSTIC_FRAMEWORK.md` §Results this run.
3. **[Data-Eng hand-off]** This Cosmos-DD sample is not a curve/maneuver source (0 % sharp) —
   the curve-scarcity remedy needs the semantic-label survey, not more Cosmos-DD. Flag on their
   backlog.
4. **[Follow-up, gated on a checkpoint]** When a post-reset checkpoint (flagship/refa/refb-speed)
   is pullable, run `skill_score` on its held-out ADE per stratum — first *model-relative*
   driving number with a defensible denominator. Falsifier: skill_score ≤ 3 on straights →
   the model is near-trivial-competitive and the "fundamental failure" reading softens.

## 5. Self-critique (quality gates)

- **G-A** every claim links a source or repo-path. ✅  **G-B** 4 actionable recs, each gated. ✅
  **G-C** KB updated (deltas, newest-first). ✅  **G-D** ledger evidence row (instrument/denominator
  hardening; **no** hypothesis status change — honest). ✅  **G-E** 8-test passing standalone metric
  + explicit next step (skill_score on a checkpoint). ✅  **G-H** ≥2 measured experiments with
  numbers (cross-corpus floor **and** curvature/speed population), hardware/wall-clock/cost recorded,
  falsifier verdicts stated; BACKLOG re-prioritized. ✅  **G-B1** LEADERBOARD floor block carries
  method + date + privileged-ego-state caveat. ✅  **G-B2** every baseline ships an analytic test. ✅
- **Honesty (P8):** the result makes the model look *worse* vs the floor than the framework said,
  and it corrects a number in a Sayed-directed doc — recorded as first-class, with the estimator
  caveat, not buried. The standstill artifact was caught and gated, not hidden.
- **Readiness (D-029):** the metric is **validated** (analytic tests + two real corpora); the gap
  to **production** is (i) orchestrator integration into `stack/tanitad/eval/`, (ii) one model-relative
  skill_score run on a checkpoint.
