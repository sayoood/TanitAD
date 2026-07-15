# Opponent Analyzer — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.
Joint duty (D-020 §5): you own `SCENARIO_DATABASE.md` scenario entries + opponent evidence;
DataEng owns the data-source rows; Benchmarks & Eval owns the metric hooks + excellence rows.

## P0 — next run

1. **SC-13 real open-loop probe (comma2k19 stopped-lead) — the trained-model step.** The oracle is
   shipped (spec-drafted); the falsifier can only be answered on real data. Once DataEng tags stopped/
   slow-lead comma2k19 segments (Tue handoff), score our checkpoint's decel-onset/imagination-error
   lead vs a detection-only baseline on matched segments. Deliverable: a measured lead-time table +
   min-TTC distribution. Falsifier: lead ≤ baseline ⇒ H15-vs-detection advantage unproven → escalate to
   the H15 σ-head. **Gated on the DataEng tagging handoff** — if not ready, advance SC-14 instead.
2. **SC-14 "Red-light barrier" spec** — near-free once SC-04 integrates (reuses the stop-line barrier
   oracle; H9 violation-rate=0). **Now top authoring item** (SC-13 spec done). FACT evidence in hand
   (Dallas red-light, W-03 family). Deliverable: `red_light_barrier.py` oracle + tests + intake.
3. **SC-15 "Emergency-scene interference" spec (W-09, NEW)** — Waymo+Tesla NHTSA first-responder letter
   (2026-07-08). Distinct from SC-06 (moving EV): non-nominal-scene recognition → H1 fallback / stop
   placement. Needs the CARLA emergency-scene + stall-injection harness (Tools & DevEnv) → author the
   oracle now (OOD-recognition + corridor-clear + TMS hooks), live-measure when the harness lands.
4. **W-04 degraded-visibility D8 stressor — REVISED after first measurement** (2026-07-08,
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
- (2026-07-15-run #3) **SC-13 Stationary-Lead scenario** shipped via intake (**16 tests**, forward-
  simulated oracle with real kinematics): `imagination` anticipates (LAL-v2 +2.30 s, no collision) vs
  `detection_reactive` collides (min-TTC 0.09 s); collision rate 0.00 vs 0.60, invariant to the
  detection-competence knob. **W-09** (first-responder interference, Waymo+Tesla) + **SC-15** catalogued;
  **GOALS.md** created (D-029 gap); W-06 honesty delta (Pony break-even) logged.
- (2026-07-24-run) **Stop-Arm Gate scenario (SC-04, W-03)** shipped via intake (11 tests; violation
  rate rule_barrier 0.0 / soft_prior 1.0); **Metis deep-read** done; **Avride** profiled (W-08/SC-13);
  SC-14 (red-light) catalogued.
- (2026-07-17-run) WEAKNESS_CATALOG v1 (W-01…W-07); work-zone-phantom scenario shipped via
  intake; integrated (9 tests).
