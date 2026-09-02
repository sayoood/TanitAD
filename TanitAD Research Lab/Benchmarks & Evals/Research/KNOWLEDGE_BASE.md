# KNOWLEDGE_BASE — Benchmarks & Eval

> **Curated, deduplicated, newest first.** Format:
> `- [YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`
>
> This is the **Benchmarks & Eval** agent's findings log. The router across all areas is
> [`../../KNOWLEDGE_BASE.md`](../../KNOWLEDGE_BASE.md).
>
> ⛔ **Three layers, and they are not interchangeable.**
> **PAPER** (`Paper/TANITAD_PAPER.md`) = the scientific account of the frontier work — derivations,
> argument, results in narrative form.
> **KNOWLEDGE_BASE** (this file) = what we learned, written for an agent about to make a decision.
> **LIBRARY** (`../../Library/`) = the evidence. Every `[PUBLISHED]` entry cites a **library key**,
> not only a URL — bank it with `python tools/kb_add.py <arxiv-id> --tag <topic> --cited-by <report>`.

- [2026-08-31] [PUBLISHED/navsim-official-README, verified today] ⭐⭐ **NAVSIM'S SCORING BASIS MOVED TWICE AND THE
  VERSION STRING HIDES BOTH — one of the changes NAMES OUR PINNED SPLIT.** (a) **2025-04-28 (v2.2)**: *"Fixed bug
  in `openscene_meta_datas` for `navhard` and `warmup`—If you used `navhard_two_stage/openscene_meta_datas` … please
  re-download and use the new data"*; (b) **2025-09-29**: *"Fixed a bug in metric filtering where
  `multiplicative_metrics_prod` and `weighted_metrics` were not correctly excluded by the human filter"* — the
  reference-forgiveness machinery itself. **Latest release is STILL v2.2** ⇒ two numbers both labelled "navhard
  EPDMS" on opposite sides of 2025-09-29 are not the same quantity, and nothing in the version reveals it. ⇒ ⭐
  **RULE: every EPDMS/PDMS we quote carries the NAVSIM version AND changelog date, not just the split.** ⚠️ The
  2026-08-29 navhard-two-stage pin is NOT challenged — it needs its date basis stated — impact: benchmark
  portfolio, any comparability table — `../../Opponent Analysis/Research/2026-08-31-frontier-action-provenance-sweep/RESULT.md` F4
- [2026-08-31] [PUBLISHED lib `2608.04896` abstract-only] ⭐⭐ **THE EXTERNAL PRECEDENT FOR OUR FLOOR-ARM DOCTRINE:
  a NAVSIM audit where BLIND probes outrank human replay.** *"reference-conditioned forgiveness, under which an
  agent receives credit when the logged human reference fails a compliance channel"* ⇒ *"the route-blind Ignore-All
  probe and a route-aware actor-blind probe outrank human replay and PDM-Closed over the complete 12,146-token
  navtest split."* **A score a policy ignoring the input can achieve is not measuring the named capability** — the
  eval-side twin of the model-side failure we measure. ⛔ **FIVE scope conditions must travel with any citation:**
  *NAVSIM v2.2 · original scene · **single-stage** · "the affected documented-stack condition" · "the audited
  numerical backend"* ⇒ it does **NOT** automatically reach our navhard-**two-stage** target. ⛔ The circulating
  79.6/74.0 figures are NOT in the abstract — UNVERIFIED, do not quote — impact: paper §methodology, portfolio
  reading — same RESULT.md F4
- [2026-08-31] [PUBLISHED lib `2606.12987`] ⭐ **A DISTORTION METRIC THAT REWARDS THE BLURRY MEAN IS THE VIDEO-DOMAIN
  TWIN OF OUR REGRESSION-TO-MEAN ADE.** Verified: *"distortion metrics (cosine similarity, SSIM) favor the blurry
  mean, masking that the diffusion model is far closer to the real frame distribution"* — **KID 0.078 vs 0.375
  (4.8×)** between a diffusion and a regression model that the distortion metrics rank the other way. ⇒ an
  independent, published statement of the failure our four-family rule exists to prevent, in a different metric
  family — impact: metric-suite argument in the paper — same RESULT.md F2
- [2026-08-30] [PUBLISHED/library] ⭐⭐ **OUR ANTI-ECHO SUITE IS STRICTER THAN THE FIELD'S — but the claim must be
  ACTION-SIDE only.** ⛔ *"open-loop is not driving performance"* is NOT novel (`1809.04843` 2018 · `2306.07962`
  2023 · `2505.05638` 2025 · `2605.00066` 2026) — claiming it invites a correct reviewer objection. ⭐ What IS
  unprecedented at five probes: a **hold-action floor a model must beat** (none found in driving WMs) and a
  **GT-calibrated echo METRIC** (`echo_index` 0.0000 vs GT 0.2113). hold-v0 is WELL precedented (NAVSIM's
  `ConstantVelocityAgent`, `2406.15349`) — claim no novelty there — impact: paper §methodology —
  `2026-08-30-anti-echo-control-precedent/RESULT.md` VERDICT + F1
- [2026-08-30] [PUBLISHED lib `2412.05337`,`2512.10958`] ⛔ **BOTH purpose-built action-fidelity benchmarks were
  verified at source to run NO CONTROL ARM.** ACT-Bench (ICLR'25) reports IEC + ADE/FDE with no frozen-action, no
  shuffled-action, no trivial floor, no oracle ceiling; WorldLens (CVPR'26 Oral) scores 24 dimensions incl.
  Action-Following with no null/adversarial baseline. `2511.20325` (AD-R1) documents the DUAL defect — WMs
  hallucinate a safe future when conditioned on an UNSAFE trajectory — narratively, with no metric — impact:
  the open lane for our contribution — same RESULT.md F2
- [2026-08-30] [PUBLISHED lib `2406.03877` Tab.3] ⭐ **The field's own strongest warrant for our doctrine:
  AD-MLP — ego-state-only, no camera, no LiDAR, near-SOTA OPEN-loop — scores DS 18.05 / Success Rate 0.00 % in
  Bench2Drive CLOSED loop.** A better citation for "T0 is not capability" than anything we have measured. The
  presentation template for our echo result is `2312.03031`'s ego-status ablation ladder (UniAD 1.03 m with no
  ego status → 0.66 → 0.46) — impact: paper framing — same RESULT.md F3
- [2026-08-30] [PUBLISHED lib `2605.00066`] ⚠️ **ESTIMATOR GUARD: open-loop vs closed-loop is ρ = −0.36 with
  n = 8 and p = 0.43** — that establishes **ABSENCE of correlation, NOT negative correlation**. Quoting ρ bare,
  as though open-loop anti-predicts closed-loop, is the slip our own standard forbids. NAVSIM PDMS ρ = 0.90
  (p = 0.002) but with ranking inversions — impact: any correlation claim we make — same RESULT.md F5
- [2026-08-30] [PUBLISHED/measured] ⛔ **COMPARABILITY GUARD: no table may place a TanitAD ADE/FDE beside a
  PDMS / DS / EPDMS score.** Our T1 numbers are metres on our own 40-episode PhysicalAI split; camera-only
  closed-loop SOTA is SimLingo 85.07 DS / 67.27 % SR on Bench2Drive (`2503.09594` Tab.2) — different corpus,
  metric family and simulator regime. ⚠️ Also: NAVSIM v2 EPDMS is quoted in the wild as 87.1 / 88.6 / 36.9 —
  these CANNOT be one quantity (navhard vs navtest); name the split or do not quote — impact: TanitEval
  portfolio, paper tables — same RESULT.md F4
- [2026-08-02] [this run / measured] ⭐⭐ **CROSS-TRACK SAYS THE MODEL WINS; CURVATURE SAYS IT IS 3×
  WORSE THAN THE TRIVIAL FLOOR.** Four-family panel (CLAUDE.md binding rule, landed mid-run) applied to
  the arm **and every floor**, n=881: flagship-v1 `cross_mae_m` **0.1152** vs CTRV 0.1604 (+0.0452
  [0.008, 0.084] separated, favours model) but `curvature_mae_1pm` **0.026969** vs CTRV **0.008967**
  (3.0×) and vs the straight-line floors 0.012221 (2.2×). **The path hits roughly the right points with
  the wrong SHAPE** — invisible to ADE and to cross-track. **REF-C-XL's curvature is 2.2× better than
  the flagship's while its ADE is worse (0.4714 vs 0.4271) — ADE inverts the ordering.** Also
  longitudinal: flagship `speed_bias` **+0.1911 m/s (too fast)** vs REF-C +0.0209; `along_final_bias`
  **+0.3375 m** vs floors ≈−0.11 (separated, favours floor) — impact: the "lateral only" narrative,
  arm selection, v2.1 lever — `../Implementation/incoming/2026-08-02-ctrv-floor/raw/four_families_vs_floors.json`
- [2026-08-02] [this run / instrument] **TACTICAL and STRATEGIC are UNAVAILABLE on the canonical eval
  surface, and that is a WORK ITEM.** `rollout.collect` is a world-model *fidelity* pass under the
  expert's true future actions (`pc2_pass=False`, `actions_source="expert_future"`), so no manoeuvre or
  route decision is decoded — `maneuver_pred/gt`, `route_pred/gt` are absent. Same for LONGITUDINAL
  distance-keeping (headway/TTC): no lead-agent track is read, though `obstacle.offline` exists on
  97.44 % of the corpus. ⇒ **two of the four binding families cannot be reported from any current eval**
  — impact: H1b hierarchy claim, binding-rule compliance — same intake.
- [2026-08-02] [this run / instrument] **A per-window reimplementation of a pooled-mean metric is NOT
  the same statistic.** `four_families` reduces heading/yaw-rate/curvature as a pooled mean over valid
  steps; a per-window form (needed for a paired bootstrap) is a mean-of-per-window-means and differs by
  7.6e-01 / 5.9e-01 / 3.6e-02 on flagship-30k. The driver measures the disagreement per metric and
  **refuses the interval above 1e-3** rather than bootstrap a different statistic than the one
  published. *(Its first cut returned heading in RADIANS and silently dropped curvature and yaw-rate by
  mis-keying `_seq_geometry` — the C63 failure mode; the agreement check is what caught it.)* — impact:
  estimator hygiene / G-B1 — same intake.
- [2026-08-02] [this run / measured] ⭐ **THE CANONICAL DRIVING GATE'S FLOOR CANNOT TURN.**
  `taniteval/driving.py:304` is `FLOORS = ("cv", "holdv0")` — both straight lines — while CTRV, same
  information budget, is **already computed on every window by `driving_diagnostic.baseline_waypoints`
  and discarded by `rollout.collect`**. On the canonical 881-window/40-ep val, CTRV is the DOMINANT
  floor: ADE **0.5265** (gated) vs CV 0.8377 / hold-v0 0.7876; paired **+0.3113 [0.167, 0.484]
  separated**; wins **423/881** windows (CV 156, hold-v0 302) — impact: every lateral/turn verdict,
  D1 gate, LEADERBOARD §0/§1b — `../Implementation/incoming/2026-08-02-ctrv-floor/`
- [2026-08-02] [this run / measured] **16 of 25 banked arms' headline verdicts move when CTRV is the
  floor.** 12 arms beat the floor under CV; **6** under CTRV, best surviving margin **+0.0890 m**.
  ⭐ **flagship-v1 @30k: vs CV +0.4106 separated → vs CTRV +0.0993 [−0.026, +0.220] NOT separated** ⇒
  "the FIRST arm below EVERY trivial bar" is a **point-estimate** claim, a **tie** under the program's
  own paired estimator. Escalated (PROJECT_STATE/MODEL_REGISTRY not agent-editable) — impact: the
  program's central capability claim — same intake.
- [2026-08-02] [this run / measured] **The lateral win is REAL but ~5× smaller than published.**
  Pre-registered criteria all held for flagship-v1: `sustained_turn` ADE +1.8063 → **+0.3398 [0.153,
  0.550] still separated, favours model**; sharp-curvature heading +24.93° → **+7.69°**; overall
  \|cross\| +0.7720 → **+0.1372**. ⇒ `where_the_win_lives = "lateral only"` survives as a DIRECTION and
  may never again be quoted with a CV-derived magnitude — impact: H15 / v4-v5 gate narrative — same intake.
- [2026-08-02] [this run / measured] **At the top speed decile CTRV beats the model LATERALLY, not just
  longitudinally** (n=89: CTRV ADE **0.0986** vs model **0.7159**, 7.3×; \|cross\|, heading and
  crosstrack all flip tie→floor; `speed_high` n=294 ADE flips tie→floor −0.2154 [−0.386, −0.030]). The
  high-speed weakness was framed as purely longitudinal because a straight-line floor cannot expose a
  lateral one on a locally-arc road — impact: v2.1/v3 high-speed lever, `memory/flagship-longitudinal-lever` — same intake.
- [2026-08-02] [nuScenes devkit / published] **The community physics floor is a best-of-FOUR including
  two yaw-rate models** — `PhysicsOracle` = {const-velocity+yaw, **const-velocity+yaw-rate (CTRV)**,
  const-accel+yaw, **const-accel+yaw-rate**}, best-per-sample vs GT. Ours was a best-of-two with both
  yaw-rate members removed. Also fixes our own labelling: **best-of-N is an ORACLE (privileged), never a
  competitor** — impact: G-B1 leaderboard hygiene / floor design —
  https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/README.md
- [2026-08-02] [this run / measured] **Three banked window dumps use a DIFFERENT `eid` encoding.**
  `windows_flagship-v4.1-10k / v4.2-step4000 / v16-ab-ft` label the same 40 episodes with the packed
  string uid (`808464434`, …) where all others use `0..39`, with **bit-identical** `gt`/`cv`/`speed`.
  Any cross-arm join keyed on `eid` mis-joins these three; a literal-equality alignment check refuses
  three provably-aligned arms (compare the PARTITION instead) — impact: harness hygiene — same intake.
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
- [2026-07-05] [kickoff] Initial research baseline for all hypotheses established; discipline agenda
  seeds defined — impact: all — see `../../INITIAL_RESEARCH_SYNTHESIS.md`

- [2026-09-01] [MEASURED ours, census over `taniteval/results/`] **THE BINDING LONGITUDINAL DISTANCE-KEEPING FAMILY IS FED ON ZERO ARTIFACTS** - **0 of 71** readable JSONs carry a numeric `distance_keeping.*` value; **28** carry the key with one identical stale reason, *"no lead-agent state exists (lead_state is a None stub)"*; 43 never mention it; **4 are UNREADABLE from the dev box** (two independent APIs, 3 runs) and are declared UNKNOWN, not absent.
  impact: **the instrument is present and ADMITTED and the refusal was retired 2026-08-18 - the gap is SUPPLY (`win["lead"]` never attached), a wiring task with no new science. ⚠️ Second gap the row missed: the 08-18 correction reached the PROSE and none of the 28 artifacts, which still assert what the source calls a STALE ABSENCE-CLAIM. Pin it with a test, as the "2 of 36" count had to be** - `Benchmarks & Evals/Research/2026-09-01-distance-keeping-fed-check/RESULT.md`

- [2026-09-02] [MEASURED ours, re-analysis of 16 banked latentmotion reads; constant control exactly 0.0 on all 16] **BACKLOG ROW FS-1's PREMISE IS WRONG ABOUT OUR INSTRUMENT, AND DRIFT IS FLAT ACROSS A 15x HORIZON CHANGE** - our "drift" is NOT start-vs-end: `mm-e19-probes/latentmotion.py` (E-DEC-59) defines it as the `r` of a k-fold-fit linear probe predicting `dz = z_{t+k} - z_t` from `z_t`, a per-window predictability statistic at ONE fixed k. Across six (arm, PCA-band) pairs, k 4 -> 60 moves drift **-2.39% to +0.38%**.
  impact: **porting WorldRoamBench's segment-based metric would not correct the numbers carrying P3 and P5/L3 - re-scope FS-1 to a k-SWEEP of the probe we already have. And a transporting latent should get HARDER to predict further ahead; it does not, at all, over 15x - independent support for P5/L3 on an axis P5 does not use** - `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-drift-metric-axis-audit/RESULT.md`
- [2026-09-02] [MEASURED ours, three source sites] **"DRIFT" NAMES AT LEAST THREE DISTINCT QUANTITIES WITH DIFFERENT UNITS AND NULLS** - latent drift (`r` of `dz` on `z_t`), `lat_drift` (`taniteval/taniteval/control.py:959` - lateral control error, m per m, null 0.0), and `max_drift` (`stack/tanitad/eval/goal_provenance.py:310` - a determinism check).
  impact: **an unqualified "drift" in a report, `GOALS_AND_CLAIMS.md` or the paper is ambiguous; every drift number must name which one it is** - same package
