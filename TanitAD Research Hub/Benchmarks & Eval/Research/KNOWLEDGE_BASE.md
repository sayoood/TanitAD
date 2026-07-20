# KNOWLEDGE_BASE — Benchmarks & Eval

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

- [2026-07-17] [this run / measured] **The ego-status shortcut ceiling on OUR data = avg L2 0.66 m
  (comma-hwy, metric-BEV, held-out by clip).** A no-vision ~20-param ridge from ego-status history scores
  0.144/0.552/1.256 m @1/2/3s — statistically tied with CTRV (0.656) — the AD-MLP shortcut (2312.03031)
  reproduced on comma. **`skill_score = model_L2 ÷ 0.66 m` now defined in leaderboard-comparable units.**
  cosmos-urban: the *learned* shortcut (1.19 m) beats the fixed kinematic floor (1.34) — impact: G1 /
  validation strategy / D1 — `../Implementation/incoming/2026-07-17-openloop-l2-egostatus-shortcut/`
- [2026-07-17] [arXiv 2312.03031, CVPR'24 / this run] **comma highway is 73.9 % straight — identical to
  nuScenes' 73.9 %** (the ego-status-critique figure). Our open-loop val inherits the *exact* shortcut
  pathology: aggregate open-loop L2 is dominated by trivial straight cruising → a **weak capability test**
  (community-unit restatement of "10–15× worse than CV" + 2605.00066). Verdict must be per-stratum
  `skill_score` + closed-loop, never an aggregate open-loop L2 — impact: G1 / DIAGNOSTIC §A/C — https://arxiv.org/abs/2312.03031
- [2026-07-17] [protocol] **nuScenes L2 has two undisclosed averaging conventions** — `pointwise` (UniAD:
  L2 at exactly t) vs `cumulative` (ST-P3/VAD: mean up to t); they differ ~2×. Any TanitAD L2 row (and any
  competitor row we cite) must state which — impact: G-B1 leaderboard hygiene — `openloop_l2.py`
- [2026-07-15] [this run / measured] **The honest trivial-baseline floor is CTRV best-of-3 ≈ 0.056–0.06 m@1s,
  not the single-CV 0.28 m** the driving diagnostic used. 26 132 anchors (comma-val + Cosmos-DD), 10 Hz. CV
  is the weakest kinematic null on curves (gentle CV 0.275 vs CTRV 0.060 = 4.6×); CTRV wins 55–58 % of anchors.
  → model held-out 6.44 m = **~115× floor** (not 10–15×), verdict direction reinforced; **D1 gate should use
  `skill_score` = model_ADE ÷ per-stratum best-of-3 floor** — impact: D1 gate / DRIVING_DIAGNOSTIC §A / WP6 —
  `../Implementation/incoming/2026-07-15-baseline-floor/`
- [2026-07-15] [this run / measured] **Curvature stratification must be speed-gated** (v ≥ 2 m/s): 12.4 % of
  comma anchors are near-standstill (median 0.01 m/s) where κ=yaw_rate/v is singular → GNSS yaw-jitter
  mislabels them "sharp" with a spurious 0.003 m floor. The framework's `driving_diagnostic` §C strata are
  standstill-polluted without the gate — impact: diagnostic protocol / P8 — same intake.
- [2026-07-15] [this run / measured] **The ungated Cosmos-Drive-Dreams sample is a poor maneuver source:**
  95.8 % straight, 1.8 % gentle, **0 % genuine sharp**, median 12.9 m/s. comma-highway carries MORE real curve
  content (12 % gentle + 0.8 % highway-speed sharp). Refines the 2026-07-13 note (Cosmos-DD = scene-diversity
  only) + framework §D2 (curve-scarcity remedy needs semantic-label survey, not more Cosmos-DD) — impact:
  Data-Eng curve-scarcity / backlog #3 — same intake.
- [2026-07-15] [arXiv 2506.04218 / NAVSIM v2] **NAVSIM v2 uses a constant-velocity agent as a triviality
  FILTER** (removes frames a CV agent solves with PDMS>0.8). Community precedent that the CV floor is
  load-bearing AND stratum-sensitive → validates our per-stratum best-of-3 skill denominator — impact:
  validation strategy / D1 gate — https://arxiv.org/html/2506.04218v1
- [2026-07-15] [arXiv 2510.18552 / 2605.18059] **New occlusion-robustness benchmarks (D-028 seam, ours):**
  **Occluded-nuScenes** (multi-sensor: 4 camera + parameterised radar/LiDAR occlusion types) — public,
  citable stressor for our OKRI/LOPS suite; **Bench2Drive-Robust** (closed-loop AD under occlusion **and
  inference latency**; SimLingo degrades sharply) = our exact edge pair OKRI/LOPS × CNCE — impact:
  LEADERBOARD watch / occlusion suite — https://arxiv.org/abs/2510.18552 · https://arxiv.org/html/2605.18059
- [2026-07-15] [arXiv 2605.31476] **IDOL — Inverse-Dynamics-Guided Future Prediction** — external support for
  the diagnostic's #1 root-cause lever (inverse-dynamics / ego-motion supervision grounds the latent).
  Pointer to Architecture; no status change (P8) — impact: DRIVING_DIAGNOSTIC root cause / H1 — https://arxiv.org/pdf/2605.31476
- [2026-07-11] [sweep / OpenReview nG35q8pNL9] *"What Truly Matters in Trajectory Prediction for AD?"* —
  reinforces that displacement-error (ADE/FDE) on curated sets does not track what matters for driving;
  external support for our decode-gates-are-weak-claims stance and the R1 mean±CI discipline. Bootstrap-CI
  on ADE/FDE is still **rare in the field** → our power-audit rigor is a differentiator, not overhead —
  impact: validation strategy / D1 — https://openreview.net/forum?id=nG35q8pNL9
- [2026-07-11] [sweep / NAVSIM GH+2506.04218] **No NAVSIM-v2 leaderboard delta since 2026-07-09** (PDM-Closed
  EPDMS still 51.3 navhard; EPDMS extended-comfort compares subsequent-frame trajectories = our TMS analogue).
  No LEADERBOARD competitor-row refresh due this run — impact: LEADERBOARD currency check — https://github.com/autonomousvision/navsim
- [2026-07-11] [this run / power audit] **D1 ADE@1s is NOT decision-grade at the val sizes we run.** Measured
  the estimator's sampling variance on the real step-6500 ckpt + comma2k19 val (RTX 4060, $0): per-route
  ADE@1s spans **2.31–18.75 m** (CoV 0.58); the shipped single-seed `run_d1` swings **7.28 m across split
  seeds** at 4 val eps (5.46 m at 8); fixed-probe bootstrap 95 % CI half-width ±4.51 m (n=4) / ±3.13 (n=9) /
  ±2.11 (n=20). Falsifier band (½ the reported 5.18→11.52 swing) = 3.17 m → **the step-21k D1 "regression"
  is inside the estimator's own noise band** (11.52 m sits in the n=4 CI upper bound 13.55 m). Even the n=9
  step-14k read is marginal. → **Rule R1: D1/D3 open-loop gates report mean±CI over ≥5 seeds; single-seed
  points deprecated for "gate movement". Decision-grade D1 needs ≥20 val eps.** — impact: validation
  strategy / D1 / D3 integrity — `../Implementation/d1_power_audit/`, `2026-07-11-d1-ade-statistical-power-audit.md`
- [2026-07-11] [this run / audit] **`d1_probe_capacity.py` (loop's D1 discriminator, `0284a5c`) shares the
  small-sample fragility** — uses ~6 val eps/corpus, single-split, compares ckpt-to-ckpt ADE deltas that at
  n≈6 are <3 m CI-noise; also mixes corpora (comma direct_k1 12.11 vs physicalai 6.88 m) in the split. Its
  "info-lost vs less-linear" verdict is not decision-grade as written → recommend bootstrap + per-corpus +
  MLP-convergence check (feedback to loop; no stack edit) — impact: D1 methodology — `../Implementation/incoming/2026-07-11-d1-gate-bootstrap/`
- [2026-07-09] [this run / audit] **LAL-v1 is blind to smooth anticipation** — first-live SC-01 CARLA run
  scored LAL-v1 −0.7 for BOTH policies; reproduced the cliff exactly at the −1.5 m/s³ jerk trigger (a
  comfort-bounded ease-off, |jerk|<~2, never fires it). Shipped **LAL-v2** (deceleration-onset by speed
  drop; the pre-line-of-sight generalization of TTB/TTC) → +0.3…+3.1 s anticipation lead vs −0.3 s
  reactive, 7 analytic tests — impact: WP6 / G0.6 / H15 — `../Implementation/incoming/2026-07-09-lal-v2-anticipation/`
- [2026-07-09] [this run / audit] **SC-01 LOPS 0.834 recompute:** matches analytic E=0.8325 of the injected
  σ=0.3 noise model (inside 95% CI, N=5000, all n_occ) → reproducible, NOT seed-luck; but reactive's 0.0 is
  *structural* → proves latent-track presence not quality; reflects injected noise, not our model (P8) —
  impact: LEADERBOARD SC-01 block / H15 honesty — `audit_results.json`
- [2026-07-09] [arXiv 2605.09701 / 2606.07170] **NAVSIM-v2 navhard leaderboard moved (Apr 2026):**
  DriveFuture **55.5 EPDMS** (#1 learned, future-aware latent WM); DrivoR **56.3** (test-time trajectory
  opt) — both above PDM-Closed 51.3. EPDMS adds compliance sub-metrics DDC/TLC/LK + HC/EC comfort split
  (= our H9 analogue) — impact: LEADERBOARD open-loop refresh / H9 — https://arxiv.org/html/2605.09701v1
- [2026-07-09] [Euro-NCAP AEB / S0001457522002329] **TTB/TTC require a detectable hazard**; occlusion-AEB
  studies recommend *longer* activation thresholds under occlusion. LAL(-v2) credits braking *before*
  line-of-sight — the gap TTB structurally cannot score → grounds our anticipation metric in accepted
  metrology — impact: LAL-v2 justification / metric-gap thesis — https://www.sciencedirect.com/science/article/abs/pii/S0001457522002329
- [2026-07-16] [arXiv 2605.00066] Cross-benchmark study (15 methods): ADE/FDE have **no reliable
  correlation** with closed-loop Driving Score; NAVSIM PDMS correlates positively but **non-monotonically**
  with Bench2Drive DS (ranking inversions); fully-paired subset only n=8 — impact: validation strategy /
  D1–D6 (closed-loop arbitrates) / justifies custom suite — https://arxiv.org/abs/2605.00066
- [2026-07-16] [arXiv 2506.04218] NAVSIM v2 pseudo-sim (3DGS-augmented) R²≈0.8 vs closed-loop (0.7 pure
  open-loop); PDM-Closed **EPDMS=51.3** navhard (Mar-2026 snapshot); criticized as non-reactive, short-
  horizon, PDMS over-weights progress/comfort/TTC (thin on safety-critical occlusion = our OKRI/LOPS niche)
  — impact: LEADERBOARD context / metric-gap — https://arxiv.org/abs/2506.04218
- [2026-07-16] [Bench2Drive] Closed-loop CARLA: 220 short routes, **one safety-critical scenario each**, 44
  categories×23 weathers×12 towns; SOTA ctx TF++/VLAAD-MIL DS 86.97/SR 71.97, ADT 77.90/55.0 — impact:
  closed-loop competitor rows / weak-spot scenario template — https://github.com/Thinklab-SJTU/Bench2Drive
- [2026-07-16] [multi-source] CARLA closed-loop **~5 DS run-to-run seed variance** for the same model →
  gate claims need mean±CI over ≥3 seeds, CIs must separate to claim "beats baseline" — impact: G-B /
  validation rigor
- [2026-07-16] [UNECE WP.29, June-2026] Global ADS GTR adopted: SMS + credible-testing/safety-case (incl.
  validated virtual toolchains) + ISMR + **DSSAD** (standard format / retrievable via electronic interface
  / tamper-evident) — impact: REGULATION_TRACE / H10–H12 — Ressources/ECE-TRANS-WP.29-2026-139e.pdf
- [2026-07-16] [Deep Think 14 / this run] Custom metric suite implemented (LAL/TMS/OKRI/CNCE/LOPS +
  trajectory seam), 22 tests on analytic ground truth; plugs into the D1–D3 gate runner's `extra_metrics`
  seam (verified live) — impact: WP6 / G0.6 — `../Implementation/incoming/2026-07-16-eval-metric-suite/`
- [2026-07-05] [kickoff] Initial research baseline for all hypotheses established; discipline agenda
  seeds defined — impact: all — see `../../INITIAL_RESEARCH_SYNTHESIS.md`
