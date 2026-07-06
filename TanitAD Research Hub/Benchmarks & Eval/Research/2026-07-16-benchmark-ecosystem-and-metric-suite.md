# Benchmarks & Eval — 2026-07-16 — Benchmark ecosystem deltas, open/closed-loop correlation, custom metric suite

> Note on dating: first substantive Benchmarks & Eval weekly run (STATE seeded `never` at kickoff).
> Dated 2026-07-16 (Thursday of W29) to stay chronological after Monday's Tools&DevEnv (07-13),
> Tuesday's DataEng (07-07), and Wednesday's Architecture note (07-14) — same convention those notes use.

**Run:** weekly Benchmarks & Eval agent (Thursday). **Loop:** iteration 1 of 3.
**Budget used:** 5 web searches, ≈1.2 h wall-clock — well under caps (25 searches / 2 h / 3 iters).
**Consumed this week:** Monday Tools&DevEnv note (2026-07-13, MetaDrive front-cam RGB + occluder/blocked-route
scenario configs — the substrate our LOPS/OKRI scenarios need); Tuesday DataEng (2026-07-07, comma2k19 +
A8 change-dominance); **Wednesday Architecture note (2026-07-14, D1–D3 gate runner + `extra_metrics` seam
explicitly reserved for this suite)**; DECISIONS D-004/D-007/D-011; Phase 0 Plan §3–§4; Deep Think 14
metric definitions; existing `LEADERBOARD.md` / `REGULATION_TRACE.md`. Every claim carries a source link or
repo path (G-A).

---

## 1. Implementation increment — custom metric suite (backlog #1)

Delivered as an intake package (D-011): `Benchmarks & Eval/Implementation/incoming/2026-07-16-eval-metric-suite/`
(`tanitad_metrics.py` + `tests/test_metrics.py` + `INTAKE.md`). Proposed target `stack/tanitad/eval/metrics.py`
— the **same `eval/` package** the Wednesday gate runner lands in.

**What.** The five Deep Think 14 custom metrics, each formula reproduced in its docstring:

| Metric | Definition (source: Deep Think 14) | Direction | Edge it isolates |
|---|---|---|---|
| **LAL** | `t_LoS − t_anticipation` (s) | >0 proactive | acting before line-of-sight (object permanence) |
| **TMS** | `1/(1 + α∫|jerk| + β∫|steer_rate|)` | →1 smooth | no control flicker under partial observability |
| **OKRI** | `∫ (½mv²)/(d_blind+ε) dt / 1000` | lower safer | throttling kinetic energy into blind spots |
| **CNCE** | `D_progress /(τ̄·P_B)·e^{−λ·collisions}` | higher efficient | safe metres per compute (the 4B moat) |
| **LOPS** | `mean_occ exp(−γ‖p_wm−p_gt‖)` | →1 tracks hidden | latent tracking of a 100%-occluded agent |

Plus a **trajectory seam** (`trajectory_extra_metrics()`) returning the `{name: (pred_xy,true_xy)→float}`
dict the gate runner's `extra_metrics=` hook expects (ade/fde/rmse/miss-rate), so the custom suite plugs
into D1–D3 **without either module importing the other**.

**Tests.** 22 passed / 1.88 s; **every metric checked against analytically-known ground truth** (gate
G-B2) — e.g. LAL=+0.4 s for LoS@5.0 s vs brake-onset@4.6 s; OKRI=29.4118 for KE=75000 / (d_blind 5+0.1)
over 2 s; CNCE=100.0 then ×e⁻² per collision; LOPS=1.0 perfect, e⁻¹ at 2 m error, 0.0 E2E baseline. The
seam is **verified live** against Wednesday's `tanitad_gates.run_d1` (metrics merged; report BLOCKED only
because I2 was deliberately withheld — doctrine intact). — impact: WP6 / G0.6 / H15 / H5 — repo-path.

**Honest scope (P8):** LAL/OKRI/LOPS consume closed-loop occluder-scenario telemetry (Ghost
Cut-Through / Blind Creep / Choke Weave), still gated on the supervised MetaDrive source install
(PROJECT_STATE W2). No metric is claimed on a real TanitAD run this week — synthetic fixtures only. This
is the computation those scenarios will call the moment live logs exist.

## 2. Research delta — the benchmark ecosystem and what it means for our validation strategy

- **The finding that anchors our whole eval thesis (arXiv [2605.00066](https://arxiv.org/abs/2605.00066),
  Apr 2026, "Do Open-Loop Metrics Predict Closed-Loop Driving? A Cross-Benchmark Correlation Study of
  NAVSIM and Bench2Drive").** Across 15 SOTA methods: **ADE/FDE show no reliable correlation with
  closed-loop Driving Score**; even the aggregate NAVSIM PDM score correlates **positively but
  non-monotonically** with Bench2Drive DS — *with clear ranking inversions*. Caveat they flag honestly:
  the fully-paired subset is only **n=8** methods (significant at p<0.01 but small). **Three consequences
  for TanitAD:** (i) validates the Wednesday doctrine baked into the gate runner — D1–D3 (decode) are
  necessary-not-sufficient; **closed-loop D4–D6 arbitrate**; (ii) validates *why we built the custom
  suite* — recognizable open-loop numbers cannot, alone, prove the edge, so we measure the edge directly
  (LAL/OKRI/LOPS are closed-loop-native); (iii) an open-loop-only leaderboard row is a **weak claim** and
  must be footnoted as such (G-B1). — impact: validation strategy / D1–D6 / H1.
- **NAVSIM v2 / pseudo-simulation (arXiv [2506.04218](https://arxiv.org/abs/2506.04218), CoRL '25).**
  EPDMS on the *navhard* split; 3D-Gaussian-Splatting-augmented synthetic observations give **R²≈0.8**
  correlation with true closed-loop vs **≈0.7** for the best pure open-loop — better, still not a
  substitute. **PDM-Closed baseline EPDMS = 51.3** on navhard (leaderboard snapshot Mar 2026, 450 Stage-1
  / 5462 Stage-2). Documented criticisms: **non-reactive** (no error compounding), **short horizon**, and
  PDMS **over-weights progress/comfort/TTC** — thin on genuinely safety-critical occlusion. That thinness
  is precisely our OKRI/LOPS/LAL niche. — impact: LEADERBOARD context row / metric-gap framing.
- **Bench2Drive (closed-loop, CARLA).** 220 short routes (~150 m), **one safety-critical scenario each**,
  44 interactive categories × 23 weathers × 12 towns; metrics = Success Rate, Driving Score, Route
  Completion, efficiency, comfort, per-skill Skill Score. Current SOTA context: TF++ (VLAAD-MIL) **DS
  86.97 / SR 71.97**; ADT **77.90 / 55.0**. Its one-scenario-per-route structure is the template for our
  weak-spot scenario clips (and it *justifies* LAL's clip-global "first braking onset" attribution). —
  impact: closed-loop competitor rows / scenario-spec design (Friday hand-off).
- **Statistical power — a hard number to design against.** CARLA closed-loop shows **~5 DS run-to-run
  variance across seeds** for the *same* model (multiple 2026 sources). Direct mean comparison cannot
  separate a real shift from simulation noise; resampling (jackknife) across metrics is the recommended
  rigor. **Actionable rule for our gate claims:** any closed-loop D4–D6 number is reported as
  **mean ± CI over ≥3 seeds**, and a gate "beats baseline" only if the CIs separate — cheap to state now,
  saves a false-positive edge claim later. — impact: G-B / validation strategy / resource-frugality (P5).

## 3. Regulation delta — WP.29 ADS GTR (June 2026)

- The UNECE **global framework on Automated Driving Systems** was adopted at the WP.29 23–26 June 2026
  session ([UNECE press](https://unece.org/sustainable-development/press/unece-adopts-first-ever-global-rules-allowing-fully-autonomous),
  our stored copy `Ressources/ECE-TRANS-WP.29-2026-139e.pdf`). Confirmed pillars, now mapped in
  `REGULATION_TRACE.md`: certified **Safety Management System**; **credible testing / safety case**
  (incl. validated virtual toolchains — our sim-eval + instrument doctrine is exactly this);
  **in-service monitoring & reporting (ISMR)**; and a **DSSAD** data-storage requirement with three
  testable sub-asks: **standard output format, practical retrievability via an electronic interface, and
  tamper protection**. — impact: REGULATION_TRACE rows / H10–H12 monitoring & reporting.
- **Actionable:** the DSSAD "standard format + retrievable + tamper-evident" triple is a concrete
  Phase-1 spec for the H10 latent event log (write-on-surprise). Recorded as a requirement, not built
  this week (Phase 0 scope). The virtual-toolchain acceptance means our MetaDrive/NAVSIM evidence is
  regulation-relevant, not just academic — worth stating in the eventual safety-case artifact.

## 4. Ledger / leaderboard / regulation-trace updates

- **LEADERBOARD.md:** added a closed-loop Bench2Drive context block (TF++/ADT rows, cited+dated) and a
  NAVSIM-v2 PDM-Closed EPDMS=51.3 navhard row; added the correlation-study caveat as a standing footnote
  (open-loop rows are weak claims). All rows carry source+date+eval-condition (G-B1). Our own rows remain
  "—" (no run yet — honest).
- **REGULATION_TRACE.md:** enriched the ISMR and DSSAD rows with the June-2026 confirmed sub-requirements
  (standard format / retrievability / tamper protection) and the virtual-toolchain acceptance under
  credible-testing. Full paragraph-level extraction from the PDF remains the open analysis task.
- **HYPOTHESIS_LEDGER:** H15/H5 gain an *evidence-of-need* note (custom suite implemented + external
  motivation from 2605.00066) — **no status upgrade** (nothing measured on our stack yet; honest).

## 5. Actionable recommendations (each tied to a hypothesis / gate — G-B)

1. **[WP6, G0.6 — ready for triage]** Integrate the metric suite alongside the gate runner into
   `stack/tanitad/eval/`; then G0.6 ("custom metric suite live") is code-complete and only awaits live
   scenario telemetry. Owner: MVP orchestrator.
2. **[Validation strategy, D4–D6 — adopt now, zero cost]** Encode the **≥3-seed mean±CI, CIs-must-separate**
   rule for every closed-loop gate claim (CARLA ~5 DS seed variance). Falsifier: a gate that "passes" on a
   single seed but whose CI overlaps the baseline is not a pass.
3. **[Scenario specs, Friday hand-off — backlog #3]** Author Ghost Cut-Through / Blind Creep / Choke Weave
   scenarios wired to LAL+LOPS / OKRI+TMS / CNCE hooks respectively — the exact telemetry columns
   `ScenarioTelemetry` expects. **Retargeted per D-014 (see §7): substrate = CARLA-on-pod (W31–32)** for the
   closed-loop occluder path; the ungated synthetic corpora give a pre-rendered first pass now. Co-own with
   the Opponent Analyzer (Friday).
4. **[Leaderboard hygiene, G-B1 — standing]** Keep open-loop and closed-loop rows in separate blocks with
   the 2605.00066 non-correlation footnote; never rank a TanitAD checkpoint on an open-loop number alone.
5. **[Regulation, Phase-1 — record]** Spec the H10 DSSAD event log to the June-2026 triple (standard
   format / retrievable via electronic interface / tamper-evident). No build this week (Phase 0 scope).

## 6. Self-critique (quality gates)

- **G-A** every claim carries a link or repo path. ✅  **G-B** 5 actionable recs, each tied to a gate/WP/H. ✅
  **G-C** KNOWLEDGE_BASE updated (deltas, newest first). ✅  **G-D** HYPOTHESIS_LEDGER touched (evidence-of-need,
  no unearned status change). ✅  **G-E** verifiable increment: 22-test suite + live seam check + measurable
  next step (integrate → G0.6). ✅  **G-F** session-end ritual below.
- **G-B1** every new leaderboard number carries source+date+eval-condition; open vs closed-loop kept in
  separate blocks with the non-correlation footnote — no apples-to-oranges row. ✅
- **G-B2** every custom metric ships a synthetic-ground-truth sanity test with the answer derived in-comment. ✅
- **Honesty (P8):** the headline research finding (open-loop ⊥ closed-loop) *undercuts* the temptation to
  publish an early ADE leaderboard win — recorded as first-class and turned into a hygiene rule, not buried.
  No metric claimed on a real run; the n=8 weakness of the anchor study is stated, not hidden.
- **Gap (recorded):** the five headline metrics are unexercised on live telemetry; the LEADERBOARD's
  TanitAD rows stay empty until the A40 Stage-0 run. Both are upstream blockers, not eval-side debt.

## 7. Mid-session repo advance (P8) — D-014 consumed and reconciled

Sayed's auto-commit swept this run's artifacts into **his** commits mid-session (`5940129`: metric suite +
note + LEADERBOARD + KNOWLEDGE_BASE; `47a89c4`: REGULATION_TRACE + STATE + ledger) and `47a89c4` brought
**D-014 — MetaDrive retired; sim arm = synthetic corpora + CARLA-on-pod**. Reconciliation: (i) the metric
suite is **sim-agnostic** (consumes `ScenarioTelemetry` columns, no simulator API) → **unaffected**; (ii)
the closed-loop occluder-LOPS path re-targets **CARLA-on-pod (W31–32)** — which is also the Bench2Drive
path in the LEADERBOARD's new closed-loop block, so D-014 makes that block *more* relevant, not less; (iii)
the ungated synthetic corpora (`PhysicalAI-WorldModel-Synthetic`, `Cosmos-Drive-Dreams`) enable a
pre-rendered LOPS/OKRI/LAL first pass **now**, before CARLA lands. STATE backlog #3 and rec #3 updated
accordingly. Operational note for hub agents: `core.fsmonitor=true` on this Google-Drive repo hides
tool-written changes from git until fsmonitor is toggled off and the index refreshed.
