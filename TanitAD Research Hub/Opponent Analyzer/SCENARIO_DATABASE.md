# Opponent-Weakness Scenario Database

**Created 2026-07-08 (D-020 §5). Ultimate goal: prove TanitAD excels at every scenario in this
database.** Each entry is a documented "dumb situation" of an opponent system, expressed as a
reproducible scenario description plus sourced training/validation data, wired directly into the
main MVP stream through gate/metric hooks and a per-scenario excellence row in
`Benchmarks & Eval/LEADERBOARD.md`.

**Joint ownership (D-020 §5):** Opponent Analyzer authors entries + opponent evidence (Friday);
Data Engineering fills/verifies data-source rows (Tuesday); Benchmarks & Eval owns metric hooks +
excellence rows (Thursday); the MVP orchestrator wires accepted scenario modules into
`stack/tanitad/eval/scenarios/` via intake.

**Evidence labels (P8):** FACT = recall/NTSB/NHTSA/DMV record or primary footage; CLAIM = press or
unverified attribution; INFER = our inference from adjacent evidence. No excellence claim is ever
made against CLAIM/INFER-only evidence.

**Lifecycle:** `catalogued → spec-drafted → data-sourced → oracle-tested → live-measured →
excellence-proven`. *Excellence-proven* = TanitAD metric beats the entry's bar on the live scenario
(closed-loop where applicable) AND the opponent failure is FACT-documented — then the row is a
public claim candidate.

**Scenario-data doctrine (Sayed directive 2026-07-09 — two strictly separated lanes):**
- **TRAINING lane:** scenario *classes* trained via synthetic corpora (WorldModel-Synthetic-Scenarios
  once D-022 lands; Cosmos pre-rendered variants) and — Phase 1 — **targeted generation**: the
  Cosmos toolkit conditioned on OUR scenario geometries (weather × density × layout sweeps of
  opponent-failure classes) and **NuRec neural reconstructions** of real drives perturbed with
  scenario elements (real geometry + synthetic hazard — closes CARLA's visual gap). PhysicalAI-family
  license firewall applies (internal use; public claims stay comma+Cosmos).
- **EVAL lane (never trained):** held-out CARLA closed-loop builds, Cosmos matched pairs (SC-05),
  nuScenes OOD probe. Separation enforced by I3 splits + I7 fingerprints + domain disjointness —
  class-level generalization is the claim; clip-level leakage is structurally impossible.

---

## SC-01 — Work-zone phantom lane / cone-taper entry  [W-01] ★★★ headline
- **Opponent evidence (FACT):** Waymo recall of 3,871 vehicles (2026-06-18) for ramp-closure /
  cone-taper entries; 6 Phoenix + 7 SF Bay incidents.
- **Description:** signed/coned closure reconfigures drivable space; correct behavior is early
  taper-merge and never entering the closed area, including when cones contradict lane markings.
- **TanitAD mechanism:** H15 (imagine changed/unobserved drivable area) + H9 closure compliance
  (`closure_incursion_m`) + fallback epistemic σ-gate.
- **Data sources:** CARLA cone/work-zone assets (live build, W31–32); Cosmos-Drive-Dreams — DataEng
  to screen captions for construction (open); comma2k19 work-zone segments (unlabeled — mine via
  high imagination-error stretches, INFER-quality only).
- **Metric hooks:** `closure_incursion_m`, LAL, OKRI, LOPS via `ScenarioTelemetry`.
- **Status:** **live-measured (partial), 2026-07-08 night** — first CARLA build on pod2
  (`stack/scripts/carla_work_zone.py`, nullrhi, real physics + raycast occlusion + measured tick
  latency; policies still scripted archetypes, honestly labeled). Live rows: **OKRI 32.4 vs 12.8**
  (reactive vs anticipating), **LOPS 0.0 vs 0.83**, TMS 0.006 vs 0.023 —
  `stack/experiments/p0-carla-workzone/suite_results_v1.json`. Instruments needing v2 before
  decision-grade: LAL (jerk-threshold never fires on the gentle slowdown — both read −0.7),
  closure-incursion detector (reads 0 for the reactive run; needs lane-polygon check + collision
  sensor). Checkpoint-driven ego waits on camera rendering (pod recreation decision).

## SC-02 — Ghost cut-through (occluded pedestrian, school zone)  [W-02] ★★★
- **Opponent evidence (FACT):** NTSB HWY26FH008 — Waymo I-Pace struck a 9-year-old near a school;
  anticipation deficit under examination.
- **Description:** pedestrian emerges from full occlusion (parked bus/van) in a low-speed zone;
  correct behavior is speed shaped by *imagined* hidden-actor risk before any detection.
- **TanitAD mechanism:** H15 latent object permanence + D9 hidden-sector gates; speed policy from
  epistemic variance, not detection confidence.
- **Data sources:** CARLA scripted-occluder (config exists in metadrive/CARLA scenario set);
  PhysicalAI-WorldModel-Synthetic-Scenarios pedestrian split (license check = DataEng backlog P1.4).
- **Metric hooks:** LOPS, OKRI, time-to-hidden-actor margin.
- **Status:** catalogued → spec next (Opponent Analyzer backlog P1.4).

## SC-03 — Blind creep (occluded intersection entry)  [W-02]
- **Opponent evidence (FACT-family):** same NTSB anticipation-deficit class; multiple robotaxi
  videos entering occluded intersections at nominal speed (CLAIM until per-incident sourcing).
- **Description:** view into cross-traffic blocked; correct behavior is creep-and-peek with
  velocity proportional to revealed sector area.
- **TanitAD mechanism:** H15 sector variance directly throttles entry speed.
- **Data sources:** CARLA occluded-junction recipes; Cosmos urban clips with parked occluders.
- **Metric hooks:** revealed-sector-area vs speed correlation; incursion events.
- **Status:** catalogued.

## SC-04 — Stop-arm gate (school-bus stop-arm passing)  [W-03] ★★
- **Opponent evidence (CLAIM):** NTSB/NHTSA probe into a Waymo passing a school bus with deployed
  stop-arm (Austin ISD; one incident attributed to human error, unverified).
- **Description:** stopped school bus, stop-arm out, occluded child crossing in front of the bus;
  correct behavior is a hard stop at/before the stop line regardless of apparent free path — a rule
  barrier, not a cost trade-off.
- **TanitAD mechanism:** H9 inherent rule compliance (hard barrier terms; violation-rate metric) +
  H15 latent estimate of the child occluded by the bus body.
- **Data sources:** CARLA `Town10HD` bus asset + scripted stop-arm + walker.child; US dashcam corpora
  screening (DataEng handoff).
- **Metric hooks:** **violation rate (must be exactly 0)**, stop-distance margin distribution, OKRI/LOPS
  toward the occluded child. **Handoff to Benchmarks & Eval (Thu):** add a `violation_rate` reducer over
  the scenario `_extra.stop_arm_violation` field (a rate, not a soft score) alongside `scenario_metrics`.
- **Status:** **spec-drafted, 2026-07-24** — intake pkg
  `Implementation/incoming/2026-07-24-stop-arm-gate-scenario/` (`stop_arm_gate.py` + telemetry oracle,
  **11/11 offline tests**, awaiting orchestrator triage). Design-oracle numbers (P8, not our model):
  **violation rate rule_barrier 0.0 / soft_prior 1.0** over the free-path sweep {0…12} m; barrier stops
  0.4 m before the line at v=0; soft prior rolls through at 8.3 m/s and its line-crossing speed grows
  monotonically 3.0→9.6 m/s with the temptation while the barrier is invariant; OKRI toward the child
  80% lower at 4 B vs 15 B params. **Next:** DataEng sources the bus asset / real footage; Benchmarks &
  Eval wires the violation-rate reducer; then live-measure our checkpoint on CARLA-on-pod.

## SC-05 — Degraded-visibility overdriving (glare / rain / obscurant)  [W-04] ★★
- **Opponent evidence (FACT):** NHTSA engineering analysis (Mar 2026) on Tesla FSD — "fails to
  detect/warn under glare, airborne obscurants"; pre-recall step.
- **Description:** visibility collapses (sun glare, heavy rain, fog, dust); correct behavior is
  self-known degradation → speed reduction / modality shift / MRM, never silent continuation.
- **TanitAD mechanism:** H11 self-monitoring (D8 OOD AUROC), H15 epistemic σ throttle, H2 modality
  steering.
- **Data sources:** **verified available** — Cosmos-Drive-Dreams weather variants
  (Rainy/Night/Foggy in shard part-000, real-bytes verified 2026-07-08); nuScenes-rain as
  held-out probe (never trained).
- **Metric hooks:** D8 AUROC healthy-vs-degraded; speed-vs-σ correlation.
- **Status:** **data-sourced + first paired measurement** (2026-07-08, step 6500, 4060).
  Naive unpaired scores don't separate (rel AUROC 0.34 inverted vs comma; weather axis
  0.54–0.59; diagonal Mahalanobis ~chance — within-comma route shift swamps it). **But the
  matched-pairs test (23 same-scene clear-vs-degraded pairs from shard part-000) shows the
  first directional positive: 16/23 scenes (69.6%) score higher imagination error under
  degraded weather, median paired shift +1.60, sign-test p≈0.047.** Pre-registered: paired
  effect should grow at 15k/30k; falsifier: ~chance at 30k ⇒ switch detector to the H15
  σ-head. Full record: `stack/experiments/p0-d8-preview/NOTE.md` + result JSONs. Honest
  status stays **data-sourced** (no oracle module yet; paired protocol now exists).

## SC-06 — Emergency-vehicle interaction failure
- **Opponent evidence (FACT):** Cruise–fire-truck collision (SF, Aug 2023); SF fire department's
  documented obstruction complaints (2023).
- **Description:** active emergency vehicle (siren/lights, contra-flow or intersection takeover);
  correct behavior is early yield + clearing the corridor, including rule-exceptions (mount curb
  line, pass red).
- **TanitAD mechanism:** strategic re-route + H9 exception handling + OOD familiarity signal
  (sirens/light patterns are rare states → fallback awareness).
- **Data sources:** thin publicly — CARLA emergency-vehicle scenarios; synthetic audio out of
  scope Phase 0 (visual-only proxy: light patterns). Honest gap recorded.
- **Metric hooks:** corridor-clear time; blockage duration.
- **Status:** catalogued (Phase-1 candidate; data gap flagged).

## SC-07 — Post-incident wrong response (MRM incorrectness)
- **Opponent evidence (FACT):** Cruise pedestrian-dragging (SF, 2023-10-02) — vehicle initiated a
  pullover *with a person under it*; permit suspension followed.
- **Description:** after any collision/anomaly, the correct MRM depends on state the nominal stack
  no longer trusts; "complete the maneuver" is exactly wrong.
- **TanitAD mechanism:** fallback brain owns post-anomaly behavior; imagination-error spike ⇒
  freeze-in-place class MRM unless corridor verified clear (A9 monitor → MRC hook).
- **Data sources:** not reproducible from public data; scenario is simulation-only by design
  (CARLA collision-inject + post-state).
- **Metric hooks:** MRM-selection correctness on injected anomalies.
- **Status:** catalogued (Phase 1; needs closed-loop harness + anomaly injection).

## SC-08 — Fleet stall / frozen vehicle blocking traffic
- **Opponent evidence (FACT):** Cruise mass stall (SF, June 2022, ~20 vehicles); repeated
  single-vehicle intersection freezes 2023 (DMV/city records).
- **Description:** uncertainty spike (connectivity loss, OOD scene) must degrade to a *well-placed*
  stop, not an in-lane freeze; recovery without remote operator where safe.
- **TanitAD mechanism:** MRM with stop-placement optimization via imagine-and-select over safe-stop
  candidates; strategic graph provides shoulder/pull-out memory.
- **Data sources:** CARLA blocked-route + connectivity-loss injection (blocked_route config
  exists in the MetaDrive/CARLA scenario set).
- **Metric hooks:** blockage time; stop-placement quality score; TMS.
- **Status:** catalogued.

## SC-09 — Changed surface / fresh-concrete entry  [W-01 family]
- **Opponent evidence (FACT):** Waymo drove into a fresh-concrete construction area (Phoenix,
  Aug 2023, city-confirmed).
- **Description:** visually-subtle surface change (wet concrete, flooded stretch, gravel) inside a
  marked work area; correct behavior is treating surface-state uncertainty as closure.
- **TanitAD mechanism:** same H15/H9 machinery as SC-01 with surface-anomaly emphasis; imagination
  error on texture-anomalous drivable area.
- **Data sources:** Cosmos weather/surface variants; CARLA custom material patches.
- **Metric hooks:** `closure_incursion_m` variant; surface-anomaly OOD rows.
- **Status:** catalogued (shares SC-01 oracle machinery — cheap spec).

## SC-10 — Atypical-vehicle misprediction (towed/oversized/articulated)
- **Opponent evidence (FACT):** Waymo recall (Feb 2024) after two vehicles hit the *same*
  backwards-towed pickup within minutes — misprediction of an atypical vehicle's motion.
- **Description:** object whose kinematics contradict its class prior (towed backwards, oversize
  load, articulated); correct behavior is widening predictive uncertainty when observed dynamics
  disagree with imagination.
- **TanitAD mechanism:** imagination-error monitor per-object region (A9) — disagreement between
  imagined and observed latent flow is exactly this signal; no class prior to be wrong about.
- **Data sources:** rare in public corpora; synthesize in CARLA (trailer/towing assets); mine
  comma2k19 for high-imag-error vehicle interactions (INFER-quality).
- **Metric hooks:** OKRI on the anomalous object; imag-error-vs-TTC lead time.
- **Status:** catalogued.

## SC-11 — Wrong-side / oncoming-lane entry
- **Opponent evidence (FACT):** NHTSA ODI investigation (opened May 2024) into Waymo incidents
  including wrong-side-of-road driving; multiple primary-footage events.
- **Description:** faced with obstruction/ambiguous markings, system commits to the oncoming lane
  without clearance; correct behavior bounds any contra-flow excursion by imagined oncoming risk.
- **TanitAD mechanism:** H9 directional barrier + H15 imagined oncoming occupancy before any
  contra-flow plan is selectable.
- **Data sources:** CARLA contra-flow overtake recipes; Cosmos urban obstruction clips.
- **Metric hooks:** contra-flow time-in-lane × imagined-clearance margin; violation rate.
- **Status:** catalogued.

## SC-12 — Gesture / traffic-officer blindness  [W-03 family]
- **Opponent evidence (INFER):** recurring class in regulator probes and press (officers waving
  robotaxis through dead signals; vehicles obeying the dead signal instead); per-incident FACT
  sourcing still open.
- **Description:** human directs traffic contrary to static rules; correct behavior is recognizing
  authority-override context (even without full gesture parsing: scene-level "intersection is
  being directed" state) and degrading gracefully if unsure.
- **TanitAD mechanism:** OOD/familiarity signal flags the non-nominal intersection state → tactical
  layer defers to creep protocol / MRM rather than rule-literal continuation. Full gesture
  understanding is out of scope Phase 0/1 (honest limit — H12 language/VLM bridge, Phase 2).
- **Data sources:** none adequate publicly for training; validation-only via CARLA walker-agent
  scripts. Recorded as the database's hardest data gap.
- **Metric hooks:** non-nominal-intersection detection rate (proxy), deferral correctness.
- **Status:** catalogued (Phase-2 horizon; kept for coverage honesty).

## SC-13 — Stationary-object / same-lane lead response  [W-08]  ★★ (new 2026-07-24)
- **Opponent evidence (FACT):** NHTSA ODI opened a PE (**2026-05-06**, published 05-08) into **Avride**
  (Uber's robotaxi partner, Yandex SDG lineage) after **16 crashes + 1 minor injury** (Dallas/Austin).
  The regulator's wording is verbatim this scenario: the vehicles **"did not brake for slow-moving or
  stopped vehicles, and struck stationary objects partially blocking the roadway"** (most **< 20 mph**;
  the one injury = clipping a **parked pickup's open door**, Dec 2025); a safety operator was aboard all
  16 but **intervened in only one**. Probe focus: **"conflict avoidance, driving behaviour competence
  and assertiveness."**
  — https://techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-for-self-driving-crashes/
  , https://www.dallasobserver.com/news/robotaxi-crashes-in-dallas-under-scrutiny-with-nhtsa-investigation-40674744/
- **Description:** a stopped/slow lead vehicle or a stationary object (stalled car, debris, disabled
  vehicle) in the ego lane; correct behavior is early, smooth deceleration to a safe following
  distance — the classic "stationary-object braking" failure that also underlies phantom/late braking.
- **TanitAD mechanism:** H15 imagination predicts the *consequence* of the closing gap (forward-model
  time-to-contact) before the object is classified — no dependence on a detection/classification prior,
  which is exactly where competence-limited stacks fail; A9 imagination-error monitor on the lead region.
- **Data sources:** comma2k19 has abundant real lead-vehicle following (mine slow/stopped-lead segments,
  license-clean); CARLA stationary-object + cut-in recipes (blocked_route family). DataEng handoff:
  tag stopped-lead / stationary-object segments in comma2k19 for a real open-loop probe.
- **Metric hooks:** OKRI on the lead object; LAL (braking-onset lead time, LAL-v2 per 2026-07-09);
  min-TTC distribution; collisions=0 bar.
- **Status:** **spec-drafted, 2026-07-10** — intake pkg
  `Implementation/incoming/2026-07-10-stationary-lead-scenario/` (`stationary_lead.py` + telemetry
  oracle, **13/13 offline tests**, numpy-only, awaiting orchestrator triage). Design-oracle numbers
  (P8, NOT our model): over the approach-speed sweep {8…25} m/s **collision rate imagination_forward
  0.000 / classifier_react 0.429**; at 15 m/s imagination brakes **3.10 s earlier** (onset 2.90 vs
  6.00 s), keeps **min-TTC 4.40 s vs 0.77 s** and a **29.8 m vs 2.0 m** gap at half the peak jerk;
  OKRI toward the lead 7 vs 18,220. **Honest falsifier built in:** the lead decays 3.10→−2.90 s and
  react's collisions vanish as the competitor's `detect_range_m` grows 20→120 m — the edge is
  *specifically* acting-before-classification. **Next:** DataEng tags comma2k19 stopped-lead segments
  for the real open-loop probe; Benchmarks & Eval wires a `collision_rate` reducer + LAL-v2 lead;
  then live-measure our checkpoint on CARLA-on-pod. Real-experiment falsifier: our predicted-TTC lead
  ≤ a detection-only baseline on matched real segments ⇒ H15-vs-detection advantage unproven there.

## SC-14 — Signal-phase compliance (red-light running)  [W-03 family]  ★ (new 2026-07-24)
- **Opponent evidence (FACT/CLAIM):** a Waymo in **Dallas** was recorded running a red light at Irving
  Blvd / Inwood Rd (2026-07, primary dashcam footage; per-incident causation CLAIM); coincides with a
  new federal investigation + recall activity in the Dallas market.
  — https://www.dallasobserver.com/news/robotaxi-crashes-in-dallas-under-scrutiny-with-nhtsa-investigation-40674744/
- **Description:** a red or newly-red signal on the ego approach; correct behavior is a hard stop at the
  line — a discrete rule barrier, not a soft trade-off against an apparently clear intersection.
- **TanitAD mechanism:** H9 directional/phase barrier term (same hard-barrier machinery as SC-04);
  violation-rate metric = 0. Shares the Stop-Arm Gate oracle structure (stop line + barrier vs soft
  prior) — a near-free spec once SC-04 integrates.
- **Data sources:** CARLA signalized-junction recipes with phase control; comma2k19 intersection
  segments (INFER-quality for real red-light approaches).
- **Metric hooks:** violation rate (0 bar), stop-distance margin at the line.
- **Status:** **catalogued** (2026-07-24; reuses SC-04 machinery — flagged as the cheapest next spec).

---

## Coverage matrix (mechanism × scenario)

| Mechanism | Scenarios |
|---|---|
| H15 imagination (unobserved/changed area) | SC-01, SC-02, SC-03, SC-05, SC-09, SC-11, SC-13 |
| H9 rule/closure barriers | SC-01, SC-04, SC-09, SC-11, SC-14 |
| A9/D8 self-monitoring + fallback | SC-05, SC-06, SC-07, SC-08, SC-10, SC-12, SC-13 |
| Strategic graph (re-route/stop memory) | SC-06, SC-08 |

Non-scenario weaknesses W-05 (compute), W-06 (unit economics), W-07 (metric fragility) live in
`WEAKNESS_CATALOG.md` and feed the CNCE/leaderboard and narrative streams instead.

## Excellence scoreboard (mirrors to LEADERBOARD)

| ID | Stage | Excellence bar | Proven? |
|---|---|---|---|
| SC-01 | oracle-tested | 0 closure incursions over scenario suite (closed-loop) | — |
| SC-04 | spec-drafted | violation rate exactly 0 (closed-loop) + full stop before line | — |
| SC-05 | data-sourced | D8 AUROC > 0.85 + monotone speed-vs-σ | — |
| SC-13 | spec-drafted | 0 collisions (closed-loop) + predicted-TTC lead > detection baseline (real) | — |
| SC-02…SC-03, SC-06…SC-12, SC-14 | catalogued | per-entry bars set at spec time | — |
