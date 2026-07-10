# INTAKE — Stationary-object / same-lane lead scenario (W-08 / SC-13)

- **Package:** `Opponent Analyzer/Implementation/incoming/2026-07-10-stationary-lead-scenario/`
- **Author agent / date:** Opponent Analyzer agent, 2026-07-10
- **Proposed target:** `stack/tanitad/eval/scenarios/stationary_lead.py` (mirrors the integrated
  `work_zone_phantom` / `stop_arm_gate` scenarios; Thursday Benchmarks & Eval wires the
  `collision_rate` reducer + LAL-v2 anticipation lead).
- **Hypothesis / WP served:** **H15** (imagination forward-models time-to-contact before any class
  label) primary; **A9** imagination-error monitor on the lead region; **H1** tactical layer for
  the consequence pricing; gate hook = closed-loop scenario suite.

## What & why (≤10 lines)

Weakness **W-08** (baseline driving-competence gaps): NHTSA ODI opened a PE (2026-05-06) into
**Avride** (Uber's robotaxi partner) after **16 crashes + 1 minor injury** — the regulator states the
vehicles **"did not brake for slow-moving or stopped vehicles, and struck stationary objects
partially blocking the roadway"** (most < 20 mph), probing **"conflict avoidance, driving behaviour
competence and assertiveness."** This is the cheapest, broadest failure surface — the mundane
longitudinal task of slowing for a stopped lead, and the same mechanism behind field-wide *late*
braking (classification-gated / radar-stationary-filtered). This turns it into a repeatable,
sim-agnostic scenario: `StationaryLeadScenario` + a `simulate_policy` design-oracle for two
archetypes (`classifier_react` = documented failure, brakes only on the class label at short range;
`imagination_forward` = H15, acts on forward-modelled TTC from range/range-rate, no class needed).
Advances **SC-13** from `catalogued` → `spec-drafted`. Research note:
`Research/2026-07-10-opponent-sweep-w4.md`.

## Evidence & tests

- Tests included: `tests/test_stationary_lead.py` — **13 passed in 0.28 s** (author machine,
  `C:/Users/Admin/venvs/tanitad` py3.13.5). RTX-4060 host; wall-clock < 1 s; cost $0 (local, numpy-only).
- Measured numbers (design-oracle, P8 — NOT a claim about our trained model):
  - **Collision rate over the approach-speed sweep {8,10,12,15,18,22,25} m/s: imagination_forward
    0.000 / classifier_react 0.429** (react collides at 18/22/25 m/s — a fixed 20 m detection range
    can't cover a braking distance that grows ∝ v²).
  - At the default 15 m/s approach: imagination brakes **3.10 s earlier** (onset 2.90 s vs 6.00 s),
    preserves **min-TTC 4.40 s vs 0.77 s** (react is a sub-second near-miss) and a **29.8 m vs 2.0 m**
    closing gap, at **half the peak jerk** (15 vs 30 m/s³). OKRI toward the lead **7 vs 18,220**
    (> 99 % lower — imagination halts ~30 m short, outside the 30 m OKRI band; react barrels to 2 m).
  - **Honest falsifier (P8), built into the oracle:** the advantage is *specifically*
    acting-before-classification. Sweeping `detect_range_m` 20 → 40 → 80 → 120 m, the anticipation
    lead **decays 3.10 → 1.80 → −0.90 → −2.90 s** and react's collision rate **falls 0.429 → 0.143 →
    0 → 0**. Give the competitor early classification and the *safety* edge disappears (only the
    comfort edge — lower jerk — remains). So the scenario is a genuine test, not a rigged handicap.
- Instrument note: these are oracle numbers by construction; the real numbers come from (a) rolling
  our trained checkpoint through `carla_recipe()` on the CARLA-on-pod harness, and (b) the
  **real-comma2k19 open-loop probe** (DataEng handoff, below).

## Handoffs (D-020 §5 joint ownership)

- **Data Engineering (Tue):** SC-13 is the one scenario with abundant *real, license-clean* support.
  Tag **comma2k19** slow/stopped-lead segments (lead range from the log or low-throttle decel events)
  → an open-loop probe: does our predicted-TTC / imagination-error lead the actual brake-onset by
  more than a detection-only baseline on matched segments? Pre-registered falsifier: lead ≤ detection
  baseline ⇒ H15-vs-detection advantage unproven on real data.
- **Benchmarks & Eval (Thu):** add a `collision_rate` reducer over `_extra.collision` (a rate, not a
  soft score) and reuse `compute_lal` (v2, 2026-07-09) on `ego_v` for the anticipation lead; wire
  SC-13 into the eval set (H15). [SC-04's `violation_rate` reducer request still stands.]

## Risk & rollback

- Blast radius if integrated: one new self-contained file under `stack/tanitad/eval/scenarios/`;
  zero new deps (numpy only); no change to existing modules. The `collision_rate` reducer is a
  Benchmarks & Eval addition (handoff noted), not part of this pkg.
- Rollback: delete the file; nothing imports it until the scenario suite registers it.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
