# Opponent Analyzer — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — FLEET DIRECTIVE 2026-07-17 (Sayed; supersedes prior P0 ordering; resource-mandated G-I)

Context: `Project Steering/FLEET_REVIEW_2026-07-17.md`. Your SC-13 (collision 0.0 vs 0.4) + W-09
first-responder work is on the tip. The gap: ALL your numbers are design-oracle — nothing yet
measured on OUR checkpoints. That changes now; the eval pod + TanitEval give you the substrate.

1. **Stationary-Lead ON OUR FLAGSHIP (first non-oracle number, EVAL POD).** Port SC-13's
   consequence-forward-model contrast onto TanitEval windows: select val windows matching the
   stationary-lead signature (lead Δv → 0, closing), run flagship-speed's imagination panel
   machinery (real vs mean vision, actions withheld) on exactly those windows → does OUR
   world model's imagined future distinguish the stalled lead earlier than a reactive proxy?
   Deliverable: the first SC-13 row with a MEASURED TanitAD number. Falsifier: no separation →
   the consequence-forward-model thesis needs the closed loop (say so; it sharpens the CARLA ask).
2. **Executable scenario scripts (with Benchmarks): SC-13 / W-09 / SC-06 → CARLA-ready** —
   parametrized spawns + trigger logic emitting `ScenarioTelemetry` columns verbatim, tested
   against the oracle path on the 4060. These are the first closed-loop excellence rows when the
   CARLA pod lands (W31-32).
3. **W-09 first-responder scenario spec deepened to NHTSA's directive taxonomy** (flashing lights,
   flares, smoke, cones as separate perturbation axes) — each axis maps to an H15-imagination or
   H2-steering claim we can test, not just a narrative.
4. **Keep the weekly sweep** (your existing cadence) but every FACT row now carries a "testable on
   our stack? Y/N + how" column — the review found your evidence quality high but under-connected
   to our checkpoints.
Joint duty (D-020 §5): you own `SCENARIO_DATABASE.md` scenario entries + opponent evidence;
DataEng owns the data-source rows; Benchmarks & Eval owns the metric hooks + excellence rows.

## P0 — next run

1. **SC-06 / W-09 "Emergency-scene interference" scenario authoring** — NEW top item after the
   **NHTSA 2026-07-08 first-responder directive** (all-operator, end-July deadline; "emergency scenes are
   not edge cases"). Deliverable: scenario spec + telemetry oracle (mirror stop_arm_gate/stationary_lead)
   + intake pkg with passing tests; SC-06 → `spec-drafted`. Geometry: ego approaches an emergency scene
   (stopped responder vehicle + flashing-light/cone/flare tableau + a partially-blocked corridor);
   archetypes `rule_literal` (drives in / blocks corridor — the documented failure) vs `imagine_and_yield`
   (H15 imagines scene actors + H11 OOD-flags the non-nominal scene → A9 yields/clears). Metrics:
   corridor-clear time, blockage duration, **non-nominal-scene-detected flag** (OOD proxy). Falsifier:
   imagine_and_yield corridor-clear ≤ rule_literal ⇒ no anticipation advantage here. Reuses the W-01
   changed-drivable-area machinery for cone/flare recognition.
2. **SC-14 "Red-light barrier" spec** — near-free once SC-04 integrates (reuses the stop-line barrier
   oracle; H9 violation-rate=0). Author after SC-06 if budget allows. *(Note: SC-14 already authored
   speculatively on the unmerged `agent/opponent-20260715` branch — check the orchestrator's dedup verdict
   before re-authoring.)*
3. **W-04 degraded-visibility D8 stressor — REVISED after first measurement** (2026-07-08,
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
5. **Watch-list deep-reads** — **SGDrive (2601.05640) — NEW top read**: "scene-to-goal *hierarchical*
   world cognition" — is the hierarchy at planning time (our H1 claim) or representation-only? First
   tracked academic line to make hierarchy explicit; pre-empt with a CNCE + in-loop-imagination contrast
   (Architecture handoff). Autobrains "Liquid AI" (Uber L4 Munich pilot — pre-empt any compute-normalized
   publication); **Metis** — monitor github.com/LogosRoboticsGroup/Metis for a param count, then a real
   CNCE comparability pass (with Benchmarks & Eval); read **AlpaSim** (NVIDIA open sim) with Tools &
   DevEnv; adjacent-domain read of **SkyJEPA** (2606.23444) for sim-to-real transfer.

## P2

6. **W-06 unit-economics dossier** — quarterly financials sweep (Pony.ai, WeRide, Waymo
   expansion costs) feeding the H0 narrative; numbers with sources into OPPONENT_PROFILES.
7. **Per-opponent counter-scenario coverage matrix** — for each profile, which W-entries apply,
   which TanitAD gates cover them, where coverage is zero (those become new scenario items).

## Done / retired
- (run #3, narrative 2026-07-31 / real 2026-07-17) **Stationary-Lead scenario (SC-13, W-08)** shipped via
  intake (**14 tests**; collision rate imagination 0.0 / detection-reactive 0.4; lead time +1.20 s vs
  −1.26 s; invariant to ambiguity). **New W-09** (first-responder/emergency-scene) from the NHTSA 07-08
  directive → **SC-06 elevated**. Deltas sweep (Waymo ultimatum / Wayve +$60 M / Pony Q2 / Zoox/WeRide/
  Nuro) + field-scan (SGDrive hierarchy signal).
- (2026-07-24-run) **Stop-Arm Gate scenario (SC-04, W-03)** shipped via intake (11 tests; violation
  rate rule_barrier 0.0 / soft_prior 1.0); **Metis deep-read** done; **Avride** profiled (W-08/SC-13);
  SC-14 (red-light) catalogued.
- (2026-07-17-run) WEAKNESS_CATALOG v1 (W-01…W-07); work-zone-phantom scenario shipped via
  intake; integrated (9 tests).
