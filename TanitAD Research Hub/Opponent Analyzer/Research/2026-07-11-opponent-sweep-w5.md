# Opponent Analyzer — sweep run #4 (w5)

**Real wall-clock at authoring: 2026-07-11 (Sat).** `QUALITY: complete` — gates G-A…G-F + G-O1/G-O2
+ G-H met. Loop: 8 web searches + 2 fetches (budget 25), 1 iteration.

## ⚠ Reconciliation — concurrent opponent runs (P8 honesty)
Mid-session I found a **prior run #3 on an unmerged branch** `worktree-agent-opponent-20260710`
(commit **874f78e**, pushed to origin, **not in `main`**). Run #3 already: (a) shipped **SC-13
stationary-lead** (`2026-07-10-stationary-lead-scenario/`; design-oracle collision rate
imagination 0.000 / classifier_react 0.429; +3.10 s brake lead; min-TTC 4.40 vs 0.77; OKRI 7 vs
18,220; with an honest built-in falsifier — the edge decays as the competitor's `detect_range`
grows), and (b) led with the **FMVSS-135 NPRM (H11 regulatory tailwind)**. My session had
independently begun re-authoring SC-13; on discovering 874f78e I **deleted the duplicate** and
pivoted this run to be **additive**:
- ship the *next* scenario (**SC-14 red-light barrier**, §3), which run #3 did not, and
- capture the **second-major-operator evidence run #3 lacked** (Tesla Houston fatality, the distinct
  **EA26002** rule-compliance docket, **Zoox** as a new opponent, GigaWorld-Policy).

**Order = run number.** Runs #1 (date-string 07-17) and #2 (07-24) are narrative-clock-ahead; run #3
(874f78e, 07-10) and this run #4 (07-11) are the real-wall-clock runs, both currently **unmerged** —
the orchestrator should merge **874f78e first, then this branch** (additive, not competing).

Labels per G-O1: **FACT** / **CLAIM** / **INFER**.

---

## 1. Net-new deltas this run (items run #3 did not have)

### Tesla — a fatality and a *second, distinct* NHTSA docket (rule-compliance) ★ headline
- **FACT — Houston fatality (2026-06-21).** A Tesla Model 3 reported under automated driving crossed
  a front lawn at high speed and rammed a brick house near Houston, **killing 76-year-old Martha
  Avila as she stood in her living room**. NHTSA opened a **special crash investigation 2026-06-23**
  (context: **46** Tesla ADS/ADAS special crash investigations over the decade, "more than a dozen"
  with ≥1 fatality). Mechanism undisclosed.
  — https://fortune.com/2026/06/23/tesla-autopilot-nhtsa-investigation-houston-crash-robotaxi/
- **FACT — traffic-violation Engineering Analysis (EA26002), distinct from the visibility EA that
  run #3 cited.** Opened **2025-10-07**, ~**2.88 M** FSD vehicles; incidents grew **58 → 80 by
  Dec-2025** (62 driver complaints / 14 field reports / 4 media): **red-light running, illegal turns,
  driving into oncoming traffic**; **14 crashes / 23 injuries** (no fatalities in *this* docket);
  data deadline 2026-03-09; fines ≤ ~**$139.4 M**.
  — https://electrek.co/2026/02/23/tesla-nhtsa-fsd-traffic-violation-investigation-second-extension/
- **Why it matters:** Tesla now supplies **FACT evidence for the rule-barrier family (SC-14
  red-light / W-03)** — a *second major operator* after the Waymo-Dallas red-light. Tesla's two open
  dockets are separate surfaces: **visibility** (EA, 3.2 M, degradation-detection → W-04) and **rule
  compliance** (EA26002, 2.88 M, red-light/illegal-turn → W-03/SC-14). This is the direct motivation
  for the SC-14 scenario shipped in §3.

### Zoox (Amazon) — NEW tracked opponent; oncoming-lane recall = 2nd FACT source for SC-11
- **FACT.** Zoox recalled **332 robotaxis (2026-12-23 filing)** — software could **cross the yellow
  centre line / stop in front of oncoming traffic near intersections** (bug surfaced 2025-08-26 on a
  wide right turn into the opposing lane; **63 crossing instances** by Dec-5). **Third software
  recall in ~8 months.** Still needs an **FMVSS exemption** (petition ≤2,500 vehicles, under NHTSA
  review late-June 2026) before paid driverless operation.
  — https://techcrunch.com/2025/12/23/zoox-issues-software-recall-over-lane-crossings/
- **Why it matters:** wrong-side / oncoming-lane entry + bad intersection stop-placement → a **second
  FACT source for SC-11** (previously Waymo-ODI only) and SC-08 family. New profile added
  (Amazon-funded, purpose-built vehicle → distinct FMVSS-exemption regulatory-risk surface).

### Literature (D-028 recency scan, seam-routed)
- **GigaWorld-Policy (arXiv 2603.17240, open-gigaai)** — FACT/INFER: **open-source** efficient
  action-centered WAM (**9× faster than Motus, +7%** on real robots; +95% vs π-0.5 on RoboTwin 2.0).
  Same skip-generative-rollout efficiency lever as Metis but **robotics, not AD** — watch for an AD
  port. No hierarchy / no self-monitoring. — https://arxiv.org/abs/2603.17240
- **"Diagnosing Dynamic Consistency in World-Action Models" (2605.07514)** — INFER: a WAM-consistency
  *diagnostic* → external support for our self-monitoring (H11) thesis; candidate read for Benchmarks
  & Eval. WM surveys 2606.00133 / 2603.09086 consolidate the field around world-action models — none
  couples hierarchy + in-loop imagination + a compute-normalized metric.
- **Benchmarks (→ Benchmarks & Eval owns per D-028 seam):** DrivingGen, HERMES, AD-L-JEPA (CVPR-2026)
  — routed, not deep-read here.

*(Waymo / Wayve / Pony / Momenta / Autobrains / NVIDIA deltas were covered by run #3 —
FMVSS-135, Waymo 6th recall, Wayve→Stellantis L4, Pony 4 cities, AlpaGym. Not re-litigated here;
I add only the cross-check that Wayve's L4 push (Stellantis+Uber+Nissan on NVIDIA DRIVE Hyperion)
and Pony's fleet-3,500 target are consistent with run #3's account.)*

---

## 2. Two hard classes now have TWO major-operator FACT sources each
This is the run's strategic point: opponent failures are converging onto exactly the scenarios our
database targets, and doing so **across independent operators** — which is what upgrades a scenario
from "one company's bug" to a *class* worth an excellence claim:

| Scenario class | FACT source A | FACT source B (this run) |
|---|---|---|
| **SC-14** red-light / signal-phase rule-barrier | Waymo-Dallas red-light | **Tesla EA26002** (80 traffic-violation incidents) |
| **SC-11** wrong-side / oncoming-lane entry | Waymo ODI (May-2024) | **Zoox** 332-vehicle recall |
| **SC-13** stationary-object / same-lane (run #3) | Avride ODI PE26003 | (Zoox/Tesla broaden the competence surface) |

---

## 3. Measured experiment (G-H) — SC-14 authored: red-light barrier

Backlog **P0** (promoted this run after the 2nd red-light FACT source landed). Shipped intake pkg
`Implementation/incoming/2026-07-11-red-light-barrier-scenario/` (`red_light_barrier.py` + telemetry
oracle + tests). Advances **SC-14 catalogued → spec-drafted.**

**Hardware/cost:** local (CPU numpy design-oracle), wall-clock **0.2 s**, **$0**. **11/11 offline
tests pass** (`venvs/tanitad`, numpy 2.5.1, pytest 9.1.1). Record: `red_light_barrier_result.json`.

**Design (P8 — design oracle, NOT a measurement of our checkpoint):** SC-14 deliberately **reuses the
accepted SC-04 `stop_arm_gate` barrier-vs-soft-prior oracle** (signal phase replaces the stop-arm;
cross-traffic replaces the bus) so **one violation-rate reducer serves both**. `soft_prior` treats
the red as a soft cost and enters on red when the intersection looks clear; `rule_barrier` (H9) is a
hard phase barrier — full stop at/before the line regardless of clearance, holding an H15 latent
estimate of the occluded crosser.

**Measured:**

| Metric | soft_prior (15 B) | rule_barrier (4 B) |
|---|---|---|
| Violation rate, clearance sweep {0…12} m | **1.0** | **0.0** |
| Line-crossing speed (0 m → 12 m clearance) | **3.2 → 10.4 m/s** (grows) | 0 (invariant) |
| Stop margin before the line | passes (−) | **+1.1 m** |
| OKRI toward the occluded crosser | 63,765 | **12,387 (−82%)** |
| Latent estimate of occluded crosser (H15) | none | held |

**Falsifier:** a hard phase-barrier that still crosses on a "clear" intersection ⇒ oracle
mis-specified. Not observed — the barrier's violation rate and stop margin are invariant to the
clearance temptation while the soft prior's line-crossing speed grows monotonically with it. Real
numbers require a checkpoint rollout on CARLA-on-pod (signalized junction + phase control).

---

## 4. Recommendations logged for other disciplines (no cross-boundary writes)
- **Benchmarks & Eval (Thu):** wire **SC-14** into the eval set — the **SC-04 `violation_rate`
  reducer applies unchanged** to `_extra.red_light_violation` (one reducer, two scenarios). [prior
  SC-04 violation-rate + degraded-visibility D8 + competitor-CNCE recs stand.] Skim **2605.07514**
  (WAM dynamic-consistency diagnostic) for the H11 self-monitor metric family.
- **Data Eng (Tue):** screen the **SC-11 oncoming-lane** class (now Zoox-sourced) in comma2k19
  intersection segments; source a signalized-junction phase-controlled recipe for SC-14. [run #3's
  stopped-lead comma2k19 tagging for SC-13 stands.]
- **Tools & DevEnv (Mon):** read **GigaWorld-Policy** (open-source, Metis-like no-rollout lever)
  alongside run #3's **AlpaGym/AlpaSim** flag as efficiency references.
- **Orchestrator:** **merge 874f78e (run #3) before this branch**; log **Zoox** as a new tracked
  opponent and the **Tesla Houston fatality + EA26002** as strategy signals (rule-compliance surface
  now spans Waymo + Tesla).

## 5. Ledger
`HYPOTHESIS_LEDGER.md`: change-log evidence row on **H9 / H15 / H6** (SC-14 spec-drafted + two-source
convergence). **No status upgrade** — nothing measured on *our* stack (P8); SC-14 numbers are a
design oracle.
