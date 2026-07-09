# KNOWLEDGE_BASE — Benchmarks & Eval

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

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
