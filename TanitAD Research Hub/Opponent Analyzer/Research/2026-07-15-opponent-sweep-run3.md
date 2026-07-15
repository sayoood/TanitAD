# Opponent Analyzer — sweep run #3 (2026-07-15)

> **Date note (narrative-clock, per repo convention):** this run is stamped to the **wall-clock date
> 2026-07-15**. STATE `LAST_RUN` read **2026-07-24 (run #2)** and the previous two notes are dated
> 2026-07-17 / 2026-07-24 — i.e. the hub's narrative clock runs *ahead* of wall-clock (a known loop
> artefact). To keep the record honest I date by wall-clock and identify runs by **run number**, not
> by newest-note ordering. So this is **run #3**, chronologically the third Opponent sweep, and it may
> sort *before* the run-#1/#2 filenames. Flagged for the orchestrator.
>
> **Labels (G-O1):** FACT = recall/NHTSA/NTSB/DMV record or primary footage · CLAIM = press/unverified
> attribution · INFER = our own inference.

## TL;DR

- **Measured experiment (G-H): SC-13 "Stationary-Lead / Same-Lane" scenario shipped** as an intake
  package with a **genuine forward-simulated** telemetry oracle (real kinematics: position, speed,
  time-to-contact, collision) — **16/16 offline tests pass**. Advances **SC-13 catalogued →
  spec-drafted**. Design-oracle headline (P8, *not* a trained-model claim): in the documented
  late-classification regime the **detection-reactive** archetype **collides** (min-TTC **0.09 s**,
  LAL-v2 lead **−0.50 s**), while the **imagination** archetype **anticipates** (LAL-v2 lead
  **+2.30 s**, min-TTC **1.83 s**, stops **3.0 m short**, **no collision**, **73 % lower OKRI**).
  Over a classification-range sweep the **collision rate is 0.60 (reactive) vs 0.00 (imagination)**.
- **Fresh evidence (FACT):** the NHTSA ODI language on **Avride** is sharper than we had catalogued —
  crashes involve *"responding to **stationary objects partially obstructing the lane ahead**"* and
  NHTSA flags *"**excessive assertiveness and insufficient capability** … may also constitute traffic
  safety violations,"* all **under a safety monitor** in the driver's seat. This directly validates
  SC-13's geometry (a *partial* in-lane obstruction, not an exotic edge case).
- **New weakness family (FACT): first-responder / emergency-scene interference.** NHTSA sent Waymo
  **and** Tesla a letter (2026-07-08) over a *"clear pattern"* of first-responder interference and
  gave AV developers ~a month to fix emergency-scene detection — triggered in part by Waymo's
  **July-4 San Francisco stall** (robotaxis stuck in gridlock, batteries died, towed). New **W-09**;
  enriches SC-06 / SC-08.
- **Honesty delta (FACT, P8):** **Pony.ai** reports **break-even operations in a single city
  (Guangzhou)** and raised its goal to **10,000+ vehicles** (+ a **Bolt** EU ride-hail partnership).
  This *partially blunts* our W-06 "thin unit economics" narrative — recorded straight, not spun.
- **Tesla delta (FACT):** a **fatal Model-3 Autopilot crash** (Katy, TX; 76-yo killed) is under a new
  federal probe — added to the Tesla profile / W-04.
- **Lit (FACT/INFER):** the latent-WM surge continues (Latent-WAM 2603.24581, DriveWorld-VLA
  2602.06521, DriveFuture 2605.09701, survey 2606.00133) — reinforces "latent WM is table stakes";
  none reports hierarchy + in-loop imagination + self-monitoring + a compute-normalized metric
  *together*. Moat read unchanged.

Loop: **5/25 searches, 1 iteration.** Compute for the experiment: local (numpy only), wall-clock
< 2 s, **cost $0**.

---

## 1. Measured experiment (G-H) — SC-13 Stationary-Lead / Same-Lane

**Backlog item:** P0 #1 (top). **Weakness:** W-08. **Hypotheses:** H15 (primary), A9 monitor, H1
tactical (secondary). **Package:**
`Implementation/incoming/2026-07-15-stationary-lead-scenario/` (`stationary_lead.py` + tests +
`2026-07-15-stationary_lead_result.json`).

### Why this scenario
Avride's ODI list is a **basic-competence** indictment: stationary objects and same-lane vehicles.
The mechanistic root (INFER, grounded in the FACT ODI language): for a **stationary** object,
appearance-classification is exactly where a detect-then-react stack is weakest (no motion cue,
cluttered background), so line-of-sight *classification* fires late — and by then the closing
geometry has already made a comfortable stop infeasible. A **consequence forward-model (H15)** does
not need the class label; it prices the *closing gap's time-to-contact* and eases off before the
object is classifiable at all. SC-13 turns that into a repeatable, sim-agnostic eval.

### What shipped
Unlike the SC-04 stop-arm oracle (a hand-set kinematic heuristic), SC-13 **forward-integrates real
longitudinal kinematics** (position, speed, TTC, collision) so min-TTC and collision are genuine
*consequences* of each policy, not asserted. Two archetypes on the exact `ScenarioTelemetry` field
contract of the metric suite:
- **`detection_reactive`** — holds cruise until the object is within `detect_range_m` (classification),
  then emergency-brakes; no latent estimate before classification. *The documented failure.*
- **`imagination`** — eases off (comfortable decel, ≤ 2.5 m/s²) as soon as the gap's TTC drops below
  a threshold, **before** classification, and holds a latent estimate of the un-classified object.

The LAL-v2 lead uses the **integrated** `compute_lal_v2` definition (`t_LoS − t_decel_onset`), with the
`t_LoS` reference set to a *nominal constant-cruise* ego so the lead credits braking that begins before
a non-anticipating stack would have classified the object. A test pins the mirrored LAL2 constants to
`stack/tanitad/eval/metrics.py` (passes — they match).

### Measured numbers (design oracle, P8 — NOT a claim about our trained model)
Default scene: stationary object at **110 m**, cruise **20 m/s** (~72 km/h), classification range
**30 m** (late — the competence regime):

| Policy | collision | min-TTC (s) | LAL-v2 lead (s) | decel onset (s) | OKRI (toward object) | final gap |
|---|---|---|---|---|---|---|
| `detection_reactive` | **True** | **0.09** | **−0.50** | 4.5 (at/after LoS 4.0) | 32 941 | 0.0 m (contact) |
| `imagination` | **False** | **1.83** | **+2.30** | 1.7 (before LoS 4.0) | 8 800 | **3.0 m** |

- **Collision rate** over the classification-range sweep {50, 40, 30, 20, 10} m:
  **`detection_reactive` 0.60 / `imagination` 0.00.**
- **Invariance property (the point of the scenario):** the reactive policy collides once the
  classification range drops below its emergency stopping distance (**≈ 33 m** from 20 m/s at 6 m/s²);
  its min-TTC collapses **2.99 → 0.04 s** as classification fires later. The imagination policy's
  min-TTC is **invariant at 1.83 s** and its anticipation lead *grows* **+0.80 → +3.30 s** — the
  mechanistic analogue of the SC-04 barrier's invariance to the free-path temptation, here invariance
  to the *detection-competence knob*.
- **OKRI −73 %** (8 800 vs 32 941): imagination carries far less kinetic energy into the < 30 m zone.

**Honest bound (P8):** at *early* classification (≥ 40 m) the reactive stack is also safe (it brakes
hard and early), so its min-TTC there is *higher* — the failure is specifically the **late-classification
regime** the ODI documents, and that is where the fair (default) comparison sits. These are oracle
numbers by construction; the real numbers come from (a) mining comma2k19 stopped-lead segments for a
real open-loop lead-time probe, and (b) rolling our checkpoint through `carla_recipe()` on
CARLA-on-pod.

**Falsifier (pre-registered):** if our checkpoint's imagination-error/decel-onset lead ≤ a
detection-only baseline on matched stopped-lead comma2k19 segments, the H15-vs-detection advantage is
**unproven here** → escalate to the H15 σ-head as the trigger.

**Handoff — Benchmarks & Eval (Thu):** add a `collision_rate` reducer over `_extra.collision`
(a rate, not a soft score) and expose `min_ttc_s` as a scenario metric; wire SC-13 into the eval set
(H15). **DataEng (Tue):** tag stopped/slow-lead comma2k19 segments (license-clean) for the real probe.

---

## 2. Opponent deltas (this run)

### Avride (emerging player) — FACT, ODI language sharpened
The NHTSA ODI (PE26003, opened 2026-05-08) covers **16 crashes + 1 minor injury** in **Dallas and
Austin, Jan–Mar 2026**, all **under a safety monitor**. New precision vs our prior catalogue: the
crashes involve **lane changes into other vehicles**, **failing to respond to vehicles in/entering the
lane ahead**, and **responding to stationary objects partially obstructing the lane ahead**; NHTSA
characterises the behaviour as **"excessive assertiveness and insufficient capability"** that **"may
also constitute traffic safety violations."** → strengthens **W-08 / SC-13** (a *partial* in-lane
obstruction is exactly SC-13's geometry).
Sources: https://www.cnbc.com/2026/05/08/us-opens-probe-into-startup-avride-self-driving-crashes-in-texas.html
· https://thenextweb.com/news/avride-uber-robotaxi-crashes-nhtsa-investigation

### First-responder / emergency-scene interference (Waymo + Tesla) — FACT, NEW → W-09
NHTSA issued a letter (2026-07-08) warning AV developers over a **"clear pattern"** of first-responder
interference and gave them ~a month to fix emergency-scene detection. Trigger context: Waymo robotaxis
**stalled in San Francisco's July-4 gridlock**, some **towed after batteries died**; Axios/press frame
it as accountability for emergency-scene responses. This is a **cross-opponent, regulator-driven**
weakness with a deadline → new **W-09**; it enriches **SC-06** (emergency-vehicle interaction) and
**SC-08** (fleet stall / frozen-vehicle blocking).
Sources: https://www.benzinga.com/markets/tech/26/07/60351872/nhtsa-warns-autonomous-vehicle-companies-over-clear-pattern-of-first-responder-interference
· https://easternherald.com/2026/07/13/nhtsa-waymo-robotaxi-emergency-responder-deadline/
· https://www.axios.com/2026/07/15/waymo-accountability-emergencies-nhtsa

### Pony.ai — FACT, honesty delta on W-06
Pony.ai says it reached **break-even operations in a single city (Guangzhou)**, raised its fleet goal
to **10,000+ vehicles**, and signed a **Bolt** ride-hail partnership (EU + non-EU from 2026). City-level
break-even ≠ company-level profitability, but it is a real datapoint that **partially blunts** our W-06
"thin unit economics / no data-efficiency story" narrative — recorded straight (P8). Our counter shifts
from "they don't make money" to "**compute-normalized** cost-per-safe-mile (H3/H7, CNCE) and a
data-efficiency slope they still have no answer to."
Sources: https://thenextweb.com/news/pony-ai-lifts-3500-robotaxi-fleet-target-2026 · https://finance.biggo.com/news/638af7df-2312-4277-971d-e14618593700

### Tesla — FACT, new fatal probe
A **fatal Model-3 crash** (Katy, TX; **76-yo killed**) is under a new federal probe tied to Autopilot;
Tesla also appears in the first-responder-interference pattern (W-09). Adds field weight to W-04.
Source: https://www.cnbc.com/2026/06/22/tesla-nhtsa-model-3-crash-autopilot-katy-texas.html

### Rear-ended-while-stopped pattern — INFER, explicitly NOT a weakness
Press notes a large share of AV crashes are the AV being **rear-ended while fully stopped** (red
lights, stop signs). This is **human-fault** and actually *validates* correct stopping behaviour — the
inverse of SC-13 (here the AV is the stationary object). Recorded so we don't mis-file it as an
opponent competence weakness.

## 3. Literature (FACT/INFER) — latent-WM surge continues
Recency sweep surfaced **Latent-WAM** (2603.24581), **DriveWorld-VLA** (2602.06521), **DriveFuture**
(2605.09701), and a broad **World Models survey** (2606.00133), alongside the already-tracked Drive-JEPA
/ Metis / GraphWorld. All predict latent features instead of pixels for efficiency — reinforcing that
**"latent world model" is table stakes**, not a differentiator. None reports **hierarchy + in-loop
imagination for planning + self-monitoring-with-guarantees + a compute-normalized causal-efficacy
number together.** Next deep-read candidates: **Latent-WAM** and **DriveWorld-VLA** (closest to our
latent-planning path); continue watching **Metis** github for a param count → then a real CNCE pass.
Sources: https://arxiv.org/abs/2603.24581 · https://arxiv.org/pdf/2602.06521 · https://arxiv.org/pdf/2606.00133

---

## 4. Recommendations logged for other disciplines (no cross-boundary writes)
- **Benchmarks & Eval (Thu):** add the `collision_rate` reducer + `min_ttc_s` scenario metric for
  SC-13; wire SC-13 into the eval set (H15). [prior recs still stand: SC-04 violation-rate reducer;
  degraded-visibility D8 stressor; competitor-param CNCE block.]
- **Data Eng (Tue):** **cheap, high-value** — tag stopped/slow-lead comma2k19 segments (license-clean)
  for the SC-13 real open-loop lead-time probe; source a stalled-vehicle CARLA asset (blocked_route
  family). [SC-04 school-bus asset still pending.]
- **Tools & DevEnv (Mon):** SC-06/SC-08 (W-09) need the emergency-scene + connectivity-loss/stall
  injection on the CARLA-on-pod harness; AlpaSim eval still worth a look.
- **Orchestrator:** triage the SC-13 intake; log **W-09 first-responder interference** (Waymo+Tesla,
  regulator deadline) and the **Pony break-even** honesty delta as strategy signals; note the
  narrative-clock date gap (STATE said run #2 = 07-24; this is run #3 dated to wall-clock 07-15).

## 5. Gates self-check
- G-A source/repo ref per claim ✓ · G-B actionable recs tied to H15/H9/H6 ✓ · G-C KB updated ✓ ·
  G-D ledger evidence row (H15/H9/H0/H6, **no status upgrade — P8**, nothing measured on our stack) ✓ ·
  G-E/G-H measured experiment with numbers + passing tests (16/16) + backlog re-prioritised ✓ ·
  G-O1 FACT/CLAIM/INFER labelled ✓ · G-O2 every weakness names its H (W-09 → H1 fallback + strategic
  re-route/stop memory) ✓ · G-F session-end (STATE + commit + push) — see STATE.
- **New this run (D-029):** `GOALS.md` created (was missing — a standing-goal gap).
