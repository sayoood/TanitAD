# INTAKE — Emergency-Scene Interference weak-spot eval scenario (W-09 / SC-06)

- **Package:** `Opponent Analyzer/Implementation/incoming/2026-08-07-emergency-scene-scenario/`
- **Author agent / date:** Opponent Analyzer agent, run #4 (narrative 2026-08-07; **real wall-clock
  2026-07-20** — the discipline's clock runs ahead of wall-clock, a known loop artefact; see STATE).
- **Proposed target:** `stack/tanitad/eval/scenarios/emergency_scene.py` (mirrors the integrated
  `work_zone_phantom` + the pending `stop_arm_gate` / `stationary_lead` scenarios; Thursday
  Benchmarks & Eval wires the blockage-duration + incursion-rate reducers and reuses LAL-v2 over
  `_extra.detect_lead_time_s`).
- **Hypothesis / WP served:** **H11** (self-monitoring flags the non-nominal scene as OOD at
  *scene* level) primary; **H15** (imagine the scene actors/hazard field before classification) and
  **A9** (fallback yields / holds the corridor clear) secondary; **H9** rule-exception handling.
  Gate hook = closed-loop scenario suite; the falsifier feed is the existing **SC-05 D8 OOD probe**.

## What & why (≤10 lines)

Weakness **W-09**. As of this run the class is documented across **two independent operators** plus
an all-operator federal action, so it is no longer a single-company anecdote: (a) **NHTSA ODI's
ADS-developers letter (2026-07-08)** demands every developer present fixes by end-July for a "clear
pattern" of AVs driving into emergency scenes and **failing to recognize flashing lights, flares,
smoke, fire and cones** — in ≥6 incidents responders had to **physically move Waymo vehicles**;
(b) **Zoox recalled 105 vehicles** (NHTSA notified 2026-07-08, public 2026-07-17) after a robotaxi
**drove into thick smoke from an active fire** (Las Vegas, 2026-06-20), failed to recognize it,
**panic-braked, tried to turn, and halted** — inside the scene. `EmergencySceneScenario` +
`simulate_policy` contrast `rule_literal` (the documented failure) vs `imagine_and_yield` (H11+H15).
Advances **SC-06** from `catalogued` → `spec-drafted`. Note: `Research/2026-08-07-opponent-sweep-w5.md`.

## Evidence & tests

- Tests included: `tests/test_emergency_scene.py` — **16 passed in 0.18 s** (author machine,
  `C:/Users/Admin/venvs/tanitad` py3.13, numpy 2.5.1 / pytest 9.1.1). Hardware: local RTX-4060 box
  (CPU-only for this pure-numpy oracle); wall-clock < 1 s; cost $0.
- Measured numbers (design-oracle, **P8 — NOT a claim about our trained model**), over the
  `scene_ambiguity` (obscurant-density) sweep {0, 0.25, 0.5, 0.75, 1.0}:
  - **Corridor incursion rate: `imagine_and_yield` 0.0 / `rule_literal` 0.2**; **mean corridor
    blockage 0.0 s vs 2.54 s**; at thick smoke (a=1.0) the rule-literal policy blocks the corridor
    for **12.7 s**, penetrates **15.6 m** into it, and ends `halted_in_corridor=True` — the exact
    Zoox trace and the exact thing responders had to physically undo.
  - **Mean non-nominal-scene detection lead time: +5.70 s (yield) vs +2.84 s (rule-literal)**;
    at a=1.0 the rule-literal lead time is **−0.10 s** (it reacts only *after* the boundary) while
    the yielding policy still flags **+5.20 s** early.
  - **The mechanism, in one number:** the obscurant collapses the **object**-classification range
    **90.0 → 13.5 m** while the **scene**-level OOD range only falls **80.0 → 68.0 m**.
  - **The failure is a CLIFF, not a slope** (an honest, non-obvious property of this oracle): the
    rule-literal policy's incursion is 0 m at a ≤ 0.75 and 15.6 m at a = 1.0, because a panic brake
    still succeeds until the trigger range drops below the stopping distance. This predicts that an
    operator can pass ordinary fog/rain testing and still fail catastrophically at a real fire —
    which is what the Zoox docket describes. Graded obscurant sweeps are therefore mandatory; a
    pass/fail at one weather level proves nothing.
- **This scenario's core assumption is asserted, not measured**, and the module says so: that a
  scene-level OOD signal survives an obscurant that defeats object classification
  (`_OOD_RANGE_DECAY` 0.15 vs `_OBJ_RANGE_DECAY` 0.85). That asymmetry **is** the H11 claim.
- **Pre-registered falsifier (P8):** SC-05's D8 probe measures exactly this detector. At 2026-07-08 it
  scored AUROC **0.34–0.59 unpaired** (falsifier fired) and only a **+1.60 median paired shift
  (p≈0.047)** on matched pairs — i.e. **the detector SC-06 depends on is not yet good enough**.
  SC-06 must **not** be scored as an excellence row until the SC-05 detector clears its own bar.
  If it never does, the H11 advantage claimed here is unproven — record as a negative result.

## Risk & rollback

- Blast radius if integrated: one new self-contained file under `stack/tanitad/eval/scenarios/`; zero
  new deps (numpy only); no change to existing modules. The blockage-duration + incursion-rate
  reducers are a Benchmarks & Eval addition (handoff noted), not part of this pkg.
- Rollback: delete the file; nothing imports it until the scenario suite registers it.
- Honest Phase-0 limit, recorded in `carla_recipe()["limits"]`: **visual cues only, no siren audio.**
  Any real emergency-scene capability claim must state that limit alongside it.
- Shared-detector note: `non_nominal_detected` is the **same OOD head** as SC-05's degraded-visibility
  stressor. Wire them to one detector, not two, or the two scenarios will silently disagree.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
