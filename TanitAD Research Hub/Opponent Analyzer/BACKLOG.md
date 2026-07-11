# Opponent Analyzer — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.
Joint duty (D-020 §5): you own `SCENARIO_DATABASE.md` scenario entries + opponent evidence;
DataEng owns the data-source rows; Benchmarks & Eval owns the metric hooks + excellence rows.

## P0 — next run

1. **SC-11 "Oncoming-lane barrier" spec** — now the cheapest high-value item (**2 FACT sources**:
   Waymo ODI + **Zoox 332-veh recall**, run #4). Reuses the barrier oracle (directional/lane barrier
   instead of stop-line); H9 contra-flow excursion bounded by H15 imagined oncoming occupancy.
   Deliverable: scenario spec + telemetry oracle + intake pkg with passing tests; SCENARIO_DATABASE
   SC-11 → `spec-drafted`. Falsifier: a directional barrier that still commits to the oncoming lane
   on an "ambiguous" marking ⇒ oracle mis-specified.
2. **W-04 degraded-visibility D8 stressor — REVISED after first measurement** (2026-07-08,
   step 6500: naive relative imag-error AUROC 0.34 inverted / 0.54 weather axis — falsifier
   fired, recorded in `stack/experiments/p0-d8-preview/NOTE.md`). Next experiment:
   **matched-pairs weather test** — pair cosmos clips by base clip id (same scene, different
   weather), score with per-corpus z-normalized ABSOLUTE error + latent Mahalanobis, add
   physicalai-val in-domain control; re-run at 15k and 30k. Expected: weather-axis AUROC on
   matched pairs > 0.6 by 30k. Falsifier: still ~0.5 on matched pairs at 30k ⇒ raw predictor
   error is not the D8 signal — escalate to the H15 σ-head (heteroscedastic variance) as the
   detector.

## P1

3. **Scenario database expansion sweep** — each run: mine fresh recalls/NTSB dockets/DMV reports
   (last 7 days) for NEW documented failures; add W-entries with FACT/CLAIM/INFER labels;
   keep SCENARIO_DATABASE.md the single source of truth.
4. **Ghost Cut-Through + Blind Creep scenario specs** (W-02 occlusion amnesia, NTSB HWY26FH008)
   — telemetry oracles for D9/H15 (LOPS/OKRI hooks); CARLA live build waits on W31–32 harness.
5. **Watch-list deep-reads** — Autobrains "Liquid AI" efficiency claims (**escalated 2026-07-24: now an
   Uber L4 Munich pilot** — pre-empt any compute-normalized publication); **Metis deep-read DONE
   2026-07-24** → next: monitor github.com/LogosRoboticsGroup/Metis for a param count, then run a real
   CNCE comparability pass (with Benchmarks & Eval). New: read **AlpaSim** (NVIDIA open sim) with
   Tools & DevEnv; adjacent-domain read of **SkyJEPA** (2606.23444) for sim-to-real transfer.

## P2

6. **W-06 unit-economics dossier** — quarterly financials sweep (Pony.ai, WeRide, Waymo
   expansion costs) feeding the H0 narrative; numbers with sources into OPPONENT_PROFILES.
7. **Per-opponent counter-scenario coverage matrix** — for each profile, which W-entries apply,
   which TanitAD gates cover them, where coverage is zero (those become new scenario items).

## Done / retired
- (run #4, 2026-07-11) **SC-14 red-light barrier** shipped via intake (**11 tests**; violation rate
  rule_barrier 0.0 / soft_prior 1.0; soft line-cross 3.2→10.4 m/s; OKRI −82%); **Zoox** profiled (new
  opponent, SC-11 2nd source); **Tesla** Houston fatality + EA26002 (SC-14 2nd source); GigaWorld-Policy
  screened. Deferred SC-13 to run #3 (874f78e) on discovering the duplicate.
- (run #3, 874f78e branch, pending merge) **SC-13 stationary-lead scenario** shipped via intake
  (collision rate imagination 0.000 / classifier_react 0.429; +3.10 s brake lead); **FMVSS-135 H11
  tailwind**; Avride ODI enrich.
- (2026-07-24-run) **Stop-Arm Gate scenario (SC-04, W-03)** shipped via intake (11 tests; violation
  rate rule_barrier 0.0 / soft_prior 1.0); **Metis deep-read** done; **Avride** profiled (W-08/SC-13);
  SC-14 (red-light) catalogued.
- (2026-07-17-run) WEAKNESS_CATALOG v1 (W-01…W-07); work-zone-phantom scenario shipped via
  intake; integrated (9 tests).
