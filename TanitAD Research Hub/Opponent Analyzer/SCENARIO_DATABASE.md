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
- **Status:** **oracle-tested** (`stack/tanitad/eval/scenarios/work_zone_phantom.py`, 9 tests);
  live build gated on CARLA harness.

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
  stop-arm (one incident attributed to human error, unverified).
- **Description:** stopped school bus, stop-arm out; correct behavior is a hard stop regardless of
  apparent free path — a rule barrier, not a cost trade-off.
- **TanitAD mechanism:** H9 inherent rule compliance (hard barrier terms; violation-rate metric).
- **Data sources:** CARLA bus asset + scripted stop-arm; US dashcam corpora screening (DataEng).
- **Metric hooks:** violation rate (must be exactly 0), stop distance distribution.
- **Status:** catalogued → **spec scheduled** (Opponent Analyzer backlog P0.1, August feed).

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
- **Status:** **data-sourced**; first AUROC preview = Opponent Analyzer backlog P0.2.

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

---

## Coverage matrix (mechanism × scenario)

| Mechanism | Scenarios |
|---|---|
| H15 imagination (unobserved/changed area) | SC-01, SC-02, SC-03, SC-05, SC-09, SC-11 |
| H9 rule/closure barriers | SC-01, SC-04, SC-09, SC-11 |
| A9/D8 self-monitoring + fallback | SC-05, SC-06, SC-07, SC-08, SC-10, SC-12 |
| Strategic graph (re-route/stop memory) | SC-06, SC-08 |

Non-scenario weaknesses W-05 (compute), W-06 (unit economics), W-07 (metric fragility) live in
`WEAKNESS_CATALOG.md` and feed the CNCE/leaderboard and narrative streams instead.

## Excellence scoreboard (mirrors to LEADERBOARD)

| ID | Stage | Excellence bar | Proven? |
|---|---|---|---|
| SC-01 | oracle-tested | 0 closure incursions over scenario suite (closed-loop) | — |
| SC-05 | data-sourced | D8 AUROC > 0.85 + monotone speed-vs-σ | — |
| SC-02…SC-04, SC-06…SC-12 | catalogued | per-entry bars set at spec time | — |
