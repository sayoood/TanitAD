# INTAKE — Stationary-Lead / Same-Lane weak-spot eval scenario (W-08 / SC-13)

- **Package:** `Opponent Analyzer/Implementation/incoming/2026-07-15-stationary-lead-scenario/`
- **Author agent / date:** Opponent Analyzer agent, 2026-07-15 (run #3; note the narrative-clock date
  gap — STATE `LAST_RUN` read run #2 = 2026-07-24; this run is dated to wall-clock 2026-07-15).
- **Proposed target:** `stack/tanitad/eval/scenarios/stationary_lead.py` (mirrors the integrated
  `work_zone_phantom` / proposed `stop_arm_gate`; Thursday Benchmarks & Eval wires the collision-rate
  reducer + `min_ttc_s` scenario metric).
- **Hypothesis / WP served:** **H15** (consequence forward-model of the closing gap — no detection/
  class prior to be wrong about) primary; **A9** imagination-error monitor + **H1** tactical layer
  secondary; gate hook = closed-loop scenario suite + real open-loop comma2k19 probe.

## What & why (≤10 lines)

Weakness **W-08**: NHTSA ODI (PE26003, opened 2026-05-08) into **Avride** (Uber partner, Yandex SDG
lineage) — **16 crashes + 1 minor injury**, all tied to **"the competence of"** the system:
lane-changing, same-lane vehicle response, and **responding to stationary objects partially
obstructing the lane ahead** (NHTSA: *"excessive assertiveness and insufficient capability … may
constitute traffic safety violations"*, all under a safety monitor). The failure class is generic —
a **detect-then-react** stack keeps cruise speed until a stalled object is *confidently classified*,
which for a stationary object is exactly where classification is weakest, so it brakes too late. This
turns that failure into a repeatable, sim-agnostic scenario: `StationaryLeadScenario` (ego cruising
toward a stationary object; `detect_range_m` = the classification-competence knob) + a
`simulate_policy` **forward-simulated** oracle (real position/speed/TTC/collision) for two archetypes
(`detection_reactive` = documented failure; `imagination` = H15). Advances **SC-13** `catalogued →
spec-drafted`. Research note: `Research/2026-07-15-opponent-sweep-run3.md`.

## Evidence & tests

- Tests included: `tests/test_stationary_lead.py` — **16 passed in 1.48 s** (author machine,
  `C:/Users/Admin/venvs/tanitad` py3.13). Includes a test pinning the mirrored LAL2 constants to the
  integrated `stack/tanitad/eval/metrics.py` (passes — they match).
- Measured numbers (design-oracle, P8 — NOT a claim about our trained model;
  `2026-07-15-stationary_lead_result.json`):
  - Default scene (object 110 m, cruise 20 m/s, classification range 30 m — the late/competence regime):
    **`detection_reactive` collides** (min-TTC **0.09 s**, LAL-v2 lead **−0.50 s**, contact);
    **`imagination` does not** (LAL-v2 lead **+2.30 s**, min-TTC **1.83 s**, stops **3.0 m short**,
    OKRI **−73 %**: 8 800 vs 32 941).
  - **Collision rate** over the classification-range sweep {50,40,30,20,10} m:
    **`detection_reactive` 0.60 / `imagination` 0.00.**
  - **Invariance:** reactive collides once classification range < its ~33 m emergency stopping distance;
    min-TTC collapses 2.99 → 0.04 s as classification fires later, while imagination's min-TTC is
    invariant at 1.83 s and its lead grows +0.80 → +3.30 s.
- Honest bound (P8): at *early* classification (≥ 40 m) the reactive stack is also safe; the failure is
  the **late-classification** regime the ODI documents. Real numbers come from mining comma2k19
  stopped-lead segments and rolling our checkpoint through `carla_recipe()`.

## Risk & rollback

- Blast radius if integrated: one new self-contained file under `stack/tanitad/eval/scenarios/`; zero
  new deps (numpy only); no change to existing modules. The **collision-rate reducer** + `min_ttc_s`
  scenario metric are Benchmarks & Eval additions (handoff noted), not part of this pkg.
- Rollback: delete the file; nothing imports it until the scenario suite registers it.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
