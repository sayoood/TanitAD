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
  **WorldModel-Synthetic-Scenarios pedestrian split** (ped 12.4% + veh-ped 21.1% of 264k; OpenMDW ungated)
  — **verified 2026-07-15: POSE-LESS** (7-cam video + Qwen captions + weather/tod/region, no actions) → an
  **EVAL-lane / scenario-mining** source (caption-searchable) and an H7-IDM training target, **not** an
  action-labelled training source until the IDM head exists.
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
  (Rainy/Night/Foggy in shard part-000, real-bytes verified 2026-07-08); **WorldModel-Synthetic-Scenarios
  `weather_degradation` family (9.2% of 264k ≈ 24k clips, OpenMDW ungated; pose-less video+caption, verified
  2026-07-15)** as a second degraded-half source for the paired set; nuScenes-rain as held-out probe (never
  trained). Note (DataEng): an **own photometric-degradation augmentation** (fog/rain/glare filters on
  clean-source frames) can make the D8 paired set arbitrarily large — OWN_DATASET_PLAN §5.2.
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

## SC-06 — Emergency-vehicle / emergency-scene interference  [W-09]  ★★ (evidence upgraded run #3)
- **Opponent evidence (FACT — upgraded 2026-07-08):** NHTSA ODI issued a formal **ADS-developers letter**
  demanding every AV developer present fixes **by end of July 2026** for a **"clear pattern"** of
  robotaxis interfering with first responders — driving into emergency scenes, blocking ambulances/fire
  crews, **failing to recognize flashing lights, flares, smoke, fire, cones.** ≥6 incidents through Mar
  2026 required responders to **physically move Waymo vehicles**; a June 2026 natural-gas-explosion case.
  Administrator Morrison: a **"functional insufficiency"**; **"Emergency scenes are not rare or extreme
  edge cases."** (Prior evidence: Cruise–fire-truck collision, SF Aug 2023; SFFD obstruction complaints.)
  — https://techcrunch.com/2026/07/08/feds-demand-autonomous-vehicle-companies-stop-interfering-with-first-responders/
- **Description:** active emergency vehicle / scene (siren/lights, flares, cones, personnel, contra-flow
  or intersection takeover); correct behavior is early yield + **clearing the corridor**, including
  rule-exceptions (mount curb line, pass red, hold clear of the scene).
- **TanitAD mechanism:** **H15** imagines the scene actors + hazard field before classification; **H11**
  self-monitoring flags the non-nominal scene OOD → **A9 fallback** yields / clears the corridor; **H9**
  exception handling; strategic re-route for corridor memory. (Light/flare/cone recognition shares W-01
  changed-drivable-area machinery.)
- **Data sources:** CARLA emergency-vehicle + light-pattern + cone/flare assets (visual-only proxy —
  synthetic audio out of scope Phase 0, honest limit); screen dashcam corpora for flashing-light events
  (DataEng handoff). 
- **Metric hooks:** corridor-clear time; blockage duration; **non-nominal-scene-detected flag** (OOD
  proxy, shared with SC-05). Benchmarks & Eval handoff: define the corridor-clear + detection reducers.
- **Opponent evidence (FACT — SECOND OPERATOR, run #4 2026-08-07 / real 2026-07-20):** **Zoox issued a
  software recall for 105 vehicles** (NHTSA notified **2026-07-08**, public **2026-07-17**) after a
  robotaxi **drove into thick smoke from an active fire** (Las Vegas, **2026-06-20**), **failed to
  recognize the smoke**, then **suddenly braked and tried to turn**, and **halted** — inside the scene.
  The class is now cross-operator, not a Waymo anecdote. Smoke is simultaneously an obscurant (SC-05)
  and an emergency-scene cue (SC-06) → **one shared OOD head**, not two.
  — https://www.cnbc.com/2026/07/17/amazon-zoox-recalls-robotaxi-smoke.html
- **Status:** **spec-drafted, run #4 (narrative 2026-08-07 / real 2026-07-20)** — intake pkg
  `Implementation/incoming/2026-08-07-emergency-scene-scenario/` (`emergency_scene.py` + telemetry
  oracle, **16/16 offline tests**, awaiting orchestrator triage). Design-oracle numbers (P8, not our
  model), over the obscurant sweep {0…1}: **corridor incursion rate `imagine_and_yield` 0.0 /
  `rule_literal` 0.2**; **mean corridor blockage 0.0 s vs 2.54 s** (**12.7 s** at thick smoke);
  **mean non-nominal-scene detection lead time +5.70 s vs +2.84 s** (rule-literal falls to **−0.10 s**
  at thick smoke, i.e. it reacts only *after* the boundary, penetrates **15.6 m** and ends
  `halted_in_corridor=True` — the exact Zoox trace). Mechanism in one number: the obscurant collapses
  **object**-classification range **90.0 → 13.5 m** while the **scene**-level OOD range falls only
  **80.0 → 68.0 m**. **The failure is a CLIFF, not a slope** (incursion 0 m at ambiguity ≤ 0.75, 15.6 m
  at 1.0) → an operator can pass ordinary fog/rain testing and still fail at a real fire; **graded
  obscurant sweeps are mandatory**.
  **BLOCKING CONDITION (P8):** this scenario's core assumption — that a scene-level OOD signal survives
  an obscurant that defeats object classification — **is asserted, not measured**, and its falsifier is
  **SC-05's D8 probe, which is currently failing** (AUROC 0.34–0.59 unpaired; matched-pairs shift
  +1.60, p≈0.047). **SC-06 must not be scored as an excellence row until the SC-05 detector clears its
  bar.** **Next:** Benchmarks & Eval defines the blockage-duration + incursion-rate reducers and
  **unifies `non_nominal_detected` with the SC-05 OOD head**; Tools & DevEnv sources CARLA emergency /
  flare / cone assets + a smoke overlay; DataEng screens corpora for smoke/flashing-light events.

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

## SC-08 — Fleet stall / frozen vehicle blocking traffic  [W-10]  ★★ (evidence upgraded run #4)
- **Opponent evidence (FACT — upgraded 2026-07-04, fresh and large-N):** **Waymo, San Francisco,
  2026-07-04.** Dozens of vehicles stalled in post-fireworks gridlock around the **Presidio**;
  **64 vehicles** had to be retrieved by staff or tow truck, several with **depleted batteries**;
  **unplanned road closures** around the Golden Gate Bridge show were a named contributor; one
  **occupied** vehicle **drove over a lit firework**. This is the proximate trigger for NHTSA framing
  its first-responder deadline as urgent, and it maps to the new **W-10** (fleet-scale mission/energy/
  network-disruption blindness). — https://sfstandard.com/2026/07/05/waymo-sf-gridlock-fourth-of-july-2026/
  , https://abc7news.com/post/waymo-fleet-clogs-presidio-july-4-fireworks-leaving-vehicles-stranded-towed/
- **Opponent evidence (FACT, prior):** Cruise mass stall (SF, June 2022, ~20 vehicles); repeated
  single-vehicle intersection freezes 2023 (DMV/city records).
- **Description:** uncertainty spike (connectivity loss, OOD scene) must degrade to a *well-placed*
  stop, not an in-lane freeze; recovery without remote operator where safe.
- **TanitAD mechanism:** MRM with stop-placement optimization via imagine-and-select over safe-stop
  candidates; strategic graph provides shoulder/pull-out memory.
- **Data sources:** CARLA blocked-route + connectivity-loss injection (blocked_route config
  exists in the MetaDrive/CARLA scenario set).
- **Metric hooks:** blockage time; stop-placement quality score; TMS; **(new)** energy/feasibility
  margin at stop-decision time.
- **Status:** **catalogued, evidence FACT-upgraded (run #4).** Honest scope note: the *fleet* dimension
  (dozens of vehicles jointly creating the gridlock they then cannot escape) is **out of reach of our
  single-vehicle harness** and our counter is **`no-counter-yet`** (W-10). The tractable Phase-0/1 slice
  is single-vehicle: **a degrading energy/feasibility margin in a congesting corridor — does the
  strategic layer choose a well-placed stop *before* it has no choice?** Orchestrator decision needed
  on whether mission-feasibility is in Phase-0 scope or an explicit deferral.

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
- **Opponent evidence (FACT):** NHTSA ODI opened an investigation (2026-05-08) into **Avride** (Uber's
  robotaxi partner, Yandex SDG lineage) after identifying **16 crashes + 1 minor injury**; ODI states
  all relate to **"the competence of"** the driving system — specifically **changing lanes, responding
  to other vehicles in the same lane, and responding to stationary objects.**
  — https://techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-for-self-driving-crashes/
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
- **Status:** **live-measured — FALSIFIER FIRED, run #4 (narrative 2026-08-07 / real 2026-07-20)** —
  the first entry in this database with a number from our own checkpoint rather than a design oracle,
  and the number does **not** support the entry's claim.
  Protocol: `sc13_real_probe.py` on the eval pod (A40), **flagship-30k** (step 29999), canonical 40-ep
  held-out PhysicalAI val, window 8 / stride 2 → **3,241 anchors**. No object labels exist on this val,
  so the *observable consequence* is labelled instead — a sustained ego deceleration (scenario identity
  = **INFER**, behaviour = measured). Signal `D = CV_forward(2 s) − pred_forward(2 s)` (imagined
  slowdown); arms = **informed** (true future actions — LEAKS, upper bound only), **held** (last
  observed action repeated — the real test), **blind** (held + vision replaced by a mean frame),
  **reactive** (−Δv/0.5 s kinematic floor).
  **Result on `BRAKE_FAR`** (braking starts 2–3 s out, i.e. **outside** the 2 s rollout; n=23 events vs
  1,283 cruise), after controlling a real speed confound (events at 8.94 m/s vs cruise at 17.34 m/s)
  by both per-event ±1 m/s matching and v0-stratification:
  **held 0.723 / 0.740 (raw 0.821, boot-CI [0.702, 0.917]) · blind 0.654 / 0.685 · gt-oracle 0.633 /
  0.668 · reactive 0.434 / 0.450.** On `BRAKE_NEAR` (0–2 s, n=157): held 0.963 / blind 0.955 /
  reactive 0.956.
  **CROSS-CORPUS REPLICATION — AND THE FALSIFIER FIRES.** The same probe on the **comma2k19** held-out
  val (64 eps, **8,384 anchors**, n=45 BRAKE_FAR) **contradicts** the above: speed-matched **held 0.538
  / stratified 0.605** vs **blind 0.608 / 0.549** vs **reactive 0.588 / 0.549** — all three mutually
  indistinguishable. **Verdict: the consequence-forward-model advantage is NOT established.** The
  corpus where it appeared is the corpus the model was trained on.
  **Two confounds keep this from being a clean refutation (INFER):** (1) **out-of-domain** — on
  comma2k19 **constant velocity beats the model outright** (CV 1.302 m vs held 1.874 m ADE), and a
  "deficit vs CV" signal is unreliable by construction on a corpus where the model loses to CV;
  (2) **regime** — comma2k19 cruise anchors sit at 29.1 m/s (vs PhysicalAI's 17.3), the highway regime
  where CV is near-unbeatable. Both corpora are under-powered (n=23 / 45): the negative is as noisy as
  the positive was.
  **What is settled: we may not claim a measured consequence-forward-model advantage**, and the oracle
  contrast below is now explicitly **unsupported** by the one real-data test we have run — it must not
  appear in any external narrative. **Next experiment changed shape:** not "10× more events" but
  (a) more **in-domain** events (PhysicalAI, stride 1) and (b) the probe on an arm whose ADE **beats
  CV** on the target corpus — if anticipation appears exactly when the model beats CV, the signal is a
  competence artefact, not a capability. Full protocol/caveats:
  `Research/2026-08-07-opponent-sweep-w5.md` §1; archived at `Implementation/sc13-real-probe/`.
  The oracle contrast below stands as authored and remains **oracle-only, now unsupported**:
- **Design-oracle status (run #3, narrative 2026-07-31 / real 2026-07-17)** — intake pkg
  `Implementation/incoming/2026-07-31-stationary-lead-scenario/` (`stationary_lead.py` + telemetry oracle,
  **14/14 offline tests**, awaiting orchestrator triage; dedup vs the unmerged `agent/opponent-20260715`
  SC-13). Design-oracle numbers (P8, not our model), over classification-ambiguity sweep {0…1}:
  **collision rate imagination_forward 0.0 / detection_reactive 0.4**; **braking-onset lead time (LAL-v2)
  +1.20 s vs −1.26 s**; the forward model is **invariant to ambiguity** (min-TTC 2.88 s, min-gap 10.75 m
  at every level) while the reactive policy degrades monotonically (min-TTC 1.91→0.00 s; drops the lead,
  wm→NaN, at ambiguity ≥ 0.75); OKRI toward the lead ~3.2× lower (4.6k vs 14.6k). **Next:** DataEng tags
  comma2k19 stopped/slow-lead segments for the open-loop lead-time probe; Benchmarks & Eval adds min-TTC +
  collision-rate reducers; then live-measure our checkpoint on CARLA-on-pod. Falsifier: imagination-error
  lead time ≤ a detection-only baseline on matched real segments ⇒ H15-vs-detection advantage unproven here.

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
| H15 imagination (unobserved/changed area) | SC-01, SC-02, SC-03, SC-05, SC-06, SC-09, SC-11, SC-13 |
| H9 rule/closure barriers | SC-01, SC-04, SC-06, SC-09, SC-11, SC-14 |
| A9/D8 self-monitoring + fallback | SC-05, SC-06, SC-07, SC-08, SC-10, SC-12, SC-13 |
| Strategic graph (re-route/stop memory) | SC-06, SC-08 |

Weakness→scenario map: W-01→SC-01/09, W-02→SC-02/03, W-03→SC-04/12/14, W-04→SC-05**+SC-06**,
W-08→SC-13, **W-09→SC-06**, **W-10→SC-08 (partial — fleet dimension `no-counter-yet`)**.
Non-scenario weaknesses W-05 (compute), W-06 (unit economics), W-07 (metric fragility) live in
`WEAKNESS_CATALOG.md` and feed the CNCE/leaderboard and narrative streams instead.

**Shared-detector note (run #4):** SC-05 (degraded visibility) and SC-06 (emergency scene) both key on
a **non-nominal-scene OOD flag** — the Zoox smoke case is literally both at once. They must be wired to
**one** detector; two would silently disagree. SC-05's D8 bar therefore **gates** SC-06 scoring.

## Excellence scoreboard (mirrors to LEADERBOARD)

| ID | Stage | Excellence bar | Proven? |
|---|---|---|---|
| SC-01 | oracle-tested | 0 closure incursions over scenario suite (closed-loop) | — |
| SC-04 | spec-drafted | violation rate exactly 0 (closed-loop) + full stop before line | — |
| SC-05 | data-sourced | D8 AUROC > 0.85 + monotone speed-vs-σ | — |
| SC-13 | **live-measured — falsifier fired** | collisions == 0 + braking-onset lead time > detection baseline (real segments) | **NO** — in-domain positive (0.72 vs 0.43 reactive, n=23) **did not replicate** on comma2k19 (0.54–0.61, indistinguishable from vision-blind and reactive, n=45); collision-rate bar oracle-only and now unsupported |
| SC-06 | **spec-drafted** | 0 corridor incursions + 0 s blockage + non-nominal-scene detected before the boundary | — **blocked**: depends on the SC-05 OOD detector, which has not cleared its own bar |
| SC-08 | catalogued (FACT-upgraded) | well-placed stop before feasibility is lost; blockage time | — (fleet dimension `no-counter-yet`, W-10) |
| SC-02…SC-03, SC-07, SC-09…SC-12, SC-14 | catalogued | per-entry bars set at spec time | — |
