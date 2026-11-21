# INTAKE — Stationary-Lead weak-spot eval scenario (W-08 / SC-13)

- **Package:** `Opponent Analyzer/Implementation/incoming/2026-07-31-stationary-lead-scenario/`
- **Author agent / date:** Opponent Analyzer agent, run #3 (narrative 2026-07-31; **real wall-clock
  2026-07-17** — the discipline's clock runs ahead of wall-clock, a known loop artefact; see STATE).
- **Proposed target:** `stack/tanitad/eval/scenarios/stationary_lead.py` (mirrors the integrated
  `work_zone_phantom` + `stop_arm_gate` scenarios; Thursday Benchmarks & Eval wires the min-TTC +
  collision-rate reducers and reuses the LAL-v2 lead-time metric).
- **Hypothesis / WP served:** **H15** (imagination forward-models the closing-gap consequence *before*
  classification) primary; **A9** (imagination-error monitor / latent lead permanence) secondary; gate
  hook = closed-loop scenario suite + open-loop comma2k19 stopped-lead probe.

## What & why (≤10 lines)

Weakness **W-08** (baseline driving-competence gaps). Two FACT-grade opponent failures name this exact
axis: **Avride** (Uber robotaxi partner) is under NHTSA ODI investigation (opened 2026-05-06) for **16
crashes** whose common thread is **"the competence of"** the system — same-lane following and
**stationary-object response** (NHTSA video: "failing to avoid slow-moving vehicles ahead, and striking
stationary objects partially blocking the roadway"; **only 1 of 16** safety monitors even attempted to
intervene). And **Tesla FSD**'s EA26002 found that in the degraded-visibility crashes it reviewed FSD
"also lost track of or never detected a lead vehicle in its path." The failure class is generic and
**mundane** — a detection-then-react stack brakes late on a stopped/slow lead because it waits for the
object to be *classified*; under ambiguity the reaction slips later, and in the limit the lead is
dropped. `StationaryLeadScenario` + `simulate_policy` design-oracle contrast two archetypes:
`detection_reactive` (the documented failure) vs `imagination_forward` (H15). Advances **SC-13** from
`catalogued` → `spec-drafted`. Research note: `Research/2026-07-31-opponent-sweep-w4.md`.

## Evidence & tests

- Tests included: `tests/test_stationary_lead.py` — **14 passed in 0.34 s** (author machine,
  `C:/Users/Admin/venvs/tanitad` py3.13, numpy 2.5.1 / pytest 9.1.1). Hardware: local RTX-4060 box
  (CPU-only for this pure-numpy oracle); wall-clock < 1 s; cost $0.
- Measured numbers (design-oracle, **P8 — NOT a claim about our trained model**), over the
  classification-ambiguity sweep {0, 0.25, 0.5, 0.75, 1.0}:
  - **Collision rate: `imagination_forward` 0.0 / `detection_reactive` 0.4** — the reactive policy
    contacts the stationary lead at ambiguity ≥ 0.75 (it drops the lead: `wm_hazard_xy` → NaN in-range).
  - **Mean braking-onset lead time (LAL-v2): +1.20 s (imagination) vs −1.26 s (reactive)** — the
    forward model brakes *before* the anticipation reference; the reactive one after it.
  - **Invariance property (the point):** the forward model's lead time (+1.20 s), min-TTC (2.88 s) and
    min-gap (10.75 m) are **invariant to ambiguity**; the reactive policy **degrades monotonically** —
    min-TTC 1.91 → 1.48 → 1.11 → 0.00 s and min-gap 11.5 → 7.0 → 4.0 → 0.4 m as ambiguity rises.
  - **OKRI toward the lead @ ambiguity 0.5: 4,612 (imagination) vs 14,570 (reactive)** — the reactive
    policy carries **~3.2×** the kinetic energy into the closing gap. Latency/params tag 18 ms / 4 B vs
    40 ms / 15 B.
- Instrument note: oracle numbers by construction. Real numbers come from (a) an **open-loop lead-time
  probe on comma2k19 stopped/slow-lead segments** (DataEng handoff — mine + tag the segments, license-
  clean) and (b) rolling our trained checkpoint through `carla_recipe()` (stalled-vehicle / debris prop
  in-lane) on the CARLA-on-pod harness. The module's `simulate_policy` is then replaced by the real
  rollout; the geometry and `_extra` contract stay.
- **Pre-registered falsifier (P8):** if, on matched real stopped-lead segments with our checkpoint, the
  imagination-error braking-onset lead time is **≤** a detection-only baseline, the H15-vs-detection
  advantage is unproven here — record as a negative result, do not claim SC-13 excellence.

## Risk & rollback

- Blast radius if integrated: one new self-contained file under `stack/tanitad/eval/scenarios/`; zero
  new deps (numpy only); no change to existing modules. The **min-TTC + collision-rate reducers** are a
  Benchmarks & Eval addition (handoff noted), not part of this pkg.
- Rollback: delete the file; nothing imports it until the scenario suite registers it.
- **Orchestrator dedup note:** an unmerged parallel branch (`agent/opponent-20260715`, off-schedule loop
  over-run) also authored an SC-13 stationary-lead scenario (commit `787671a`, "collision 0.00 vs 0.60").
  This package is the canonical scheduled-run (`agent/opponent-20260717`) version. **Pick one at merge;
  do not integrate both.**

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
