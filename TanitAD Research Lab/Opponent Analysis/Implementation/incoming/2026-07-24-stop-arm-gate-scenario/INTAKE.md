# INTAKE — Stop-Arm Gate weak-spot eval scenario (W-03 / SC-04)

- **Package:** `Opponent Analyzer/Implementation/incoming/2026-07-24-stop-arm-gate-scenario/`
- **Author agent / date:** Opponent Analyzer agent, 2026-07-24
- **Proposed target:** `stack/tanitad/eval/scenarios/stop_arm_gate.py` (mirrors the integrated
  `work_zone_phantom` scenario; Thursday Benchmarks & Eval wires the violation-rate reducer).
- **Hypothesis / WP served:** **H9** (inherent rule compliance — hard barrier, not soft prior)
  primary; **H15** (occluded-child latent estimate) secondary; gate hook = closed-loop scenario suite.

## What & why (≤10 lines)

Weakness **W-03**: a separate NTSB/NHTSA probe covers a Waymo robotaxi illegally passing a stopped
school bus with its stop-arm deployed (Austin ISD; one case reportedly attributed to human error,
CLAIM). The failure class is generic — a hard, discrete, legally-binding rule with thin training
support is treated as a **soft prior**, so when the adjacent/ego lane *looks* free the policy trades
the rule away and passes the bus. This turns that failure into a repeatable, sim-agnostic scenario:
`StopArmGateScenario` (stopped bus + deployed stop-arm + legal stop line + occluded child crossing
in front of the bus + a *tempting* free path) and a `simulate_policy` design-oracle for two
archetypes (`soft_prior` = documented failure; `rule_barrier` = H9). Advances **SC-04** from
`catalogued` → `spec-drafted`. Research note: `Research/2026-07-24-opponent-sweep-w3.md`.

## Evidence & tests

- Tests included: `tests/test_stop_arm_gate.py` — **11 passed in 0.33 s** (author machine, `venvs/tanitad` py3.13).
- Measured numbers (design-oracle, P8 — NOT a claim about our trained model):
  - **H9 violation rate over the free-path temptation sweep {0,2,4,6,8,10,12} m: rule_barrier 0.0 / soft_prior 1.0.**
  - At 8 m clearance: rule_barrier stops **0.4 m before the line at v=0** (no violation); soft_prior
    rolls through at **8.28 m/s** (violation, never halts).
  - **Barrier property:** rule_barrier speed-at-line is invariant to the temptation; soft_prior's
    grows monotonically **3.0 → 9.6 m/s** as the free path opens — the mechanistic difference between
    a barrier term and a learned soft prior.
  - OKRI toward the occluded child **80% lower** (10.5k vs 51.4k) at **4 B vs 15 B** params.
- Instrument note: these are oracle numbers by construction; the real numbers come from rolling our
  trained checkpoint through `carla_recipe()` on the CARLA-on-pod harness (next step in the module).

## Risk & rollback

- Blast radius if integrated: one new self-contained file under
  `stack/tanitad/eval/scenarios/`; zero new deps (numpy only); no change to existing modules. The
  new **violation-rate reducer** is a Benchmarks & Eval addition (handoff noted), not part of this pkg.
- Rollback: delete the file; nothing imports it until the scenario suite registers it.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
