# Opponent Analyzer — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.
Joint duty (D-020 §5): you own `SCENARIO_DATABASE.md` scenario entries + opponent evidence;
DataEng owns the data-source rows; Benchmarks & Eval owns the metric hooks + excellence rows.

## P0 — next run

1. **W-03 "Stop-Arm Gate" scenario authoring** — school-bus stop-arm passing (NTSB/NHTSA probe
   on Waymo). Deliverable: scenario spec + telemetry oracle module (mirror work_zone_phantom
   structure) + SCENARIO_DATABASE row upgrade to `spec-drafted`; intake package with passing
   tests. Metric hook: H9 violation-rate (hard-barrier compliance), not soft prior.
2. **W-04 degraded-visibility D8 stressor spec** — glare/rain/obscurant (NHTSA Tesla FSD
   engineering analysis). Method: define the D8 OOD stressor protocol using Cosmos weather
   variants (Rainy/Night/Foggy already in shard part-000, verified on real bytes); measure
   imagination-error separation healthy-vs-degraded on the step-latest checkpoint (4060) —
   first real AUROC preview. Falsifier: AUROC < 0.6 at 17% training ⇒ record honestly, re-test
   at 30k.

## P1

3. **Scenario database expansion sweep** — each run: mine fresh recalls/NTSB dockets/DMV reports
   (last 7 days) for NEW documented failures; add W-entries with FACT/CLAIM/INFER labels;
   keep SCENARIO_DATABASE.md the single source of truth.
4. **Ghost Cut-Through + Blind Creep scenario specs** (W-02 occlusion amnesia, NTSB HWY26FH008)
   — telemetry oracles for D9/H15 (LOPS/OKRI hooks); CARLA live build waits on W31–32 harness.
5. **Watch-list deep-reads** — Autobrains "Liquid AI" efficiency claims (pre-empt any
   compute-normalized publication); Metis CNCE comparability (with Benchmarks & Eval).

## P2

6. **W-06 unit-economics dossier** — quarterly financials sweep (Pony.ai, WeRide, Waymo
   expansion costs) feeding the H0 narrative; numbers with sources into OPPONENT_PROFILES.
7. **Per-opponent counter-scenario coverage matrix** — for each profile, which W-entries apply,
   which TanitAD gates cover them, where coverage is zero (those become new scenario items).

## Done / retired
- (2026-07-17-run) WEAKNESS_CATALOG v1 (W-01…W-07); work-zone-phantom scenario shipped via
  intake; integrated (9 tests).
