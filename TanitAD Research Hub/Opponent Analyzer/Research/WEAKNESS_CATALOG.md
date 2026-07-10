# WEAKNESS CATALOG — opponent failure modes → TanitAD attack surface

> Maintained by the Opponent Analyzer agent. Each entry: **mechanism hypothesis**, **evidence**
> (labeled FACT/CLAIM/INFER per G-O1), **TanitAD counter** (which H exploits it), **scenario-spec
> status** (with the Thursday Benchmarks&Eval agent), **training-data recipe** (H6 pipeline).
> An entry with no counter is marked `no-counter-yet` (a strategy gap for the Orchestrator).
> Created 2026-07-17 (v1). Newest / highest-priority first.

---

## W-01 — Work-zone / construction-zone brittleness  ★ headline

- **Mechanism (INFER):** the operator's *prior* road topology (map + expected lane graph) diverges from
  the *posterior* reality — closed ramp, cone taper, shifted lane, active work zone. A perception-reactive
  stack either fails to recognize the novel static configuration (ramp-closure sign, cone line) or cannot
  reason about the *changed drivable area*, so it drives the pre-planned path into the closure.
- **Evidence (FACT):** Waymo **recalled 3,871 robotaxis (2026-06-18, NHTSA campaign 26E035)** after
  **13 freeway construction-zone incidents** — 6 Phoenix Apr'26 (drove past **ramp-closure signs** into
  pre-planned freeway work zones), 7 SF Bay May'26 (entered lanes with active construction). Waymo's own
  NHTSA filing names the mechanism: the AV **"inappropriately prioritiz[es] the avoidance of other
  freeway hazards and/or fail[s] to recognize the construction zone."** Waymo **pulled all robotaxis from
  highways on 2026-05-19**; this is its **second recall in ~one month**; a fix is "under development."
  Freeway autonomy suspended; 20+-city 2026 expansion (incl. London/Tokyo) freeway-constrained.
  — https://www.cnbc.com/2026/06/18/waymo-nhtsa-voluntary-recall-robotaxis-entered-freeway-construction-zones.html
  , https://techcrunch.com/2026/06/18/waymo-recalls-nearly-4000-robotaxis-to-stop-them-driving-into-highway-construction-zones/
- **TanitAD counter:** **H15** (imagine changed/unobserved drivable area from partial cues) + **H9**
  (inherent compliance with sign/cone/closure semantics via barrier terms) + **H1** fallback layer
  (stop/hand-back when the imagined map's epistemic σ is high). A latent world model that predicts the
  *consequence* of entering the taper — and prices the σ of an unseen closed region — should refuse the
  path a map-follower takes.
- **Scenario-spec status:** **DRAFTED + intake pkg this run** →
  `Implementation/incoming/2026-07-17-work-zone-phantom-scenario/` (CARLA-on-pod per D-014; emits the
  `ScenarioTelemetry` contract that drives OKRI/LOPS/LAL + a blocked-lane/rule-compliance signal). Awaiting
  orchestrator triage → Thursday agent wiring into the eval set.
- **Training-data recipe (H6):** (a) real: mine comma2k19 / Cosmos-Drive-Dreams for cone/work-zone frames
  (rare → oversample); (b) sim: CARLA work-zone props (cone taper, ramp-closure sign, arrow board) with an
  **occluded merging actor** behind the taper; (c) off-expert perturbation rollouts that *enter* the
  closure so the WM learns the negative-consequence manifold (D-010 sim role).

## W-02 — Occlusion amnesia / hidden-actor anticipation

- **Mechanism (INFER):** standard E2E policies have **no persistent latent estimate of a fully-occluded
  agent**; when a pedestrian/vehicle is hidden then emerges, the policy reacts late because it never held
  the object in state (no object permanence). Late braking, not proactive slowing.
- **Evidence (FACT):** NTSB **HWY26FH008** (2026-01-23, Santa Monica) — Waymo I-Pace **struck a 9-yo
  pedestrian** crossing midblock in a school zone; ADS **detected + braked heavily but late** (minor
  injuries); NTSB is examining **anticipation of sudden pedestrian movement**.
  — https://www.ntsb.gov/investigations/Pages/HWY26FH008.aspx
- **TanitAD counter:** **H15** (imagination holds a latent estimate of the occluded agent) measured by
  **LOPS** (latent object permanence) + **OKRI** (kinetic energy carried into the blind region) + **LAL**
  (anticipation latency). This is the D9 hidden-sector gate.
- **Scenario-spec status:** **Ghost Cut-Through** + **Blind Creep** already seeded (Phase-0 3-scenario set,
  H6). Metric hooks exist in the Benchmarks&Eval suite (`compute_lops/okri/lal`, telemetry contract
  `ScenarioTelemetry`). Live scenario build still gated on CARLA-on-pod (Tools&DevEnv W31–32).
- **Training-data recipe (H6):** CARLA scripted occluder (lead vehicle / parked van) + emerging VRU;
  perturbation rollouts that keep speed past the blind edge to populate the negative manifold.

## W-03 — Rule-compliance edge cases (school-bus stop-arm, gestures)

- **Mechanism (INFER):** hard, discrete, legally-binding rules with rare training support (stopped
  school-bus stop-arm, traffic-cop gestures) are under-represented in log data; a reward/imitation policy
  treats them as soft priors and violates them under distribution shift.
- **Evidence (CLAIM/FACT):** separate **NTSB/NHTSA probe into Waymo illegal school-bus stop-arm passing**
  (Austin ISD); one case reportedly attributed to human error (CLAIM). Distinct from W-02.
  — https://techcrunch.com/2026/01/23/waymo-probed-by-national-transportation-safety-board-over-illegal-school-bus-behavior/
  **New family evidence (FACT/CLAIM, 2026-07):** a Waymo was recorded **running a red light** in Dallas
  (Irving Blvd/Inwood Rd) amid a new federal investigation there — the same hard-discrete-rule failure
  class (signal-phase compliance). — https://www.dallasobserver.com/news/robotaxi-crashes-in-dallas-under-scrutiny-with-nhtsa-investigation-40674744/
- **TanitAD counter:** **H9** (inherent rule compliance via RMFM / hard barrier terms — a violation-rate
  metric, not a learned soft prior). Owner of the violation-rate metric home: Benchmarks&Eval.
- **Scenario-spec status:** **DRAFTED + intake pkg 2026-07-24** →
  `Implementation/incoming/2026-07-24-stop-arm-gate-scenario/` (**SC-04**; stopped bus + deployed
  stop-arm + occluded child + tempting free path; **11/11 offline tests**). Design-oracle result: H9
  **violation rate rule_barrier 0.0 vs soft_prior 1.0**, barrier invariant to the free-path temptation
  while the soft prior's line-crossing speed grows 3.0→9.6 m/s. Red-light running catalogued as **SC-14**
  (reuses the same stop-line barrier oracle). Handoff to Benchmarks&Eval: add a `violation_rate` reducer.
- **Training-data recipe (H6):** synthetic stop-arm + red-signal events (rare in real logs); barrier-term
  supervision rather than pure imitation.

## W-04 — Camera-only degraded-visibility (glare, rain, airborne obscurant)

- **Mechanism (INFER):** a camera-only stack with **no calibrated epistemic uncertainty** stays confident
  when the sensing channel degrades (sun glare, heavy rain, dust/smoke), so it neither slows nor hands
  back — it drives blind at speed.
- **Evidence (FACT):** NHTSA **upgraded** the FSD probe to an **Engineering Analysis (2026-03-18)**
  covering **~3.2 M vehicles** (MY2016–2026 S/X/3/Y/Cybertruck) — the final investigative phase before a
  recall. It found FSD's **"degradation-detection" feature** (meant to recognize when cameras can't see
  and alert the driver) **"did not detect common roadway conditions that impaired camera visibility
  and/or provide alerts … until immediately before the crash."** **9 crashes** flagged incl. **1 fatality
  + 2 injuries** under glare/fog/dust. Miami robotaxi launched into exactly this regime (rain,
  2026-07-03). — https://electrek.co/2026/03/19/nhtsa-upgrades-tesla-fsd-visibility-investigation-3-2-million-vehicles/
- **TanitAD counter:** **H11** (self-monitoring w/ guarantees — degraded-visibility as a D8 OOD stressor,
  AUROC>0.85) + **H15** (epistemic σ throttles on uncertainty) + **H2** (attention-based modality steering
  to radar when the camera degrades).
- **Scenario-spec status:** `no-scenario-yet`. **Recommendation to Benchmarks&Eval:** add a
  degraded-visibility D8 stressor using the Cosmos-Drive-Dreams weather corpus (already in the D-014 mix).
- **Training-data recipe (H6):** Cosmos weather variants + glare augmentation; label the *sensing-degraded*
  regime for the self-monitor's OOD head.

## W-05 — Compute-hungry world/reasoning models (scale-first)

- **Mechanism (INFER):** competitors buy capability with parameters — 32 B on-car VLA (Alpamayo 2),
  15 B offline generative WM (GAIA-3), monolithic E2E foundation nets (Wayve/Tesla). High cost/vehicle,
  hard on Orin/Thor-class compute, poor compute-normalized efficacy.
- **Evidence (FACT):** Alpamayo 2 Super = **32 B VLA**; Wayve GAIA-3 = **15 B** (offline eval); both
  scale-first. — https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development , https://wayve.ai/thinking/gaia-3/
- **TanitAD counter:** **H1/H3/H5** (hierarchical ~261 M model, data-efficient, real-time on Orin) proven
  by **CNCE** (compute-normalized causal efficacy). Our ~261 M-on-Orin vs 32 B-on-car is the wedge.
- **Scenario-spec status:** metric-only (CNCE) — no scenario needed. **Recommendation:** competitor param
  counts into the `LEADERBOARD.md` efficiency block.
- **Training-data recipe (H6):** n/a (efficiency is architectural, not data).

## W-06 — Thin unit economics / no data-efficiency story

- **Mechanism (INFER):** robotaxi revenue is tiny vs fleet size; the stacks are data-hungry (huge fleets
  to cover the tail). Nobody has a credible **data-efficiency-from-action-free-video** story.
- **Evidence (FACT):** Pony.ai Q1'26 robotaxi revenue **$8.6 M** against a **3,500-vehicle** 2026 target.
  — https://mlq.ai/news/v2/pony-ai-q1-revenue-more-than-doubles-to-343m-as-robotaxi-sales-surge-nearly-fivefold/
- **TanitAD counter:** **H3** (latent WM data efficiency) + **H7** (1000× leverage via IDM + focal
  canonicalization). The data-efficiency slope experiment is the headline proof (Data Eng, Phase 1).
- **Scenario-spec status:** n/a (data-slope experiment, not a scenario). Owner: Data Eng.
- **Training-data recipe (H6):** the H7 pipeline itself is the artifact.

## W-07 — Metric fragility (disengagement deprecation) — narrative asset

- **Mechanism (INFER):** the industry's headline safety metric (miles/disengagement) is gameable and not
  decision-relevant — the regulator itself is abandoning it, echoing the open-loop⊥closed-loop finding.
- **Evidence (FACT):** **CA DMV proposes to replace disengagement reporting in 2026** with
  "safety-relevant event" metrics. — https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/disengagement-reports/
- **TanitAD counter:** **H0 / narrative** + our closed-loop, regulation-native custom metrics
  (LAL/TMS/OKRI/CNCE/LOPS) designed from day one (Benchmarks&Eval, D-007).
- **Scenario-spec status:** n/a. **Story beat** for the vision deck / Orchestrator.
- **Training-data recipe (H6):** n/a.

## W-08 — Baseline driving competence gaps (lane-change / same-lane / stationary object)  ★ (new 2026-07-24)

- **Mechanism (INFER):** stacks that lean on **detection-then-react** rather than a forward model of
  consequence fail the most basic longitudinal/lateral tasks under distribution shift — they brake late
  on a stationary or slow lead because classification is uncertain, mishandle same-lane vehicles, and
  botch lane changes. The failure is *competence*, not an exotic edge case, which makes it a broad and
  damning surface.
- **Evidence (FACT):** NHTSA ODI opened a PE (**2026-05-06**, published 05-08) into **Avride** (Uber's
  robotaxi partner, Yandex SDG lineage) after **16 crashes + 1 minor injury** (Dallas/Austin). The
  regulator's wording is verbatim the failure: the vehicles **"did not brake for slow-moving or stopped
  vehicles, and struck stationary objects partially blocking the roadway"** (most **< 20 mph**; the one
  injury = clipping a parked pickup's open door, Dec 2025); a safety operator was aboard all 16 but
  **intervened in only one**. Probe focus: **"conflict avoidance, driving behaviour competence and
  assertiveness."** — https://techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-for-self-driving-crashes/
- **TanitAD counter:** **H15** imagination forward-models time-to-contact on a stopped/slow lead
  *before* the object is classified (no detection/class prior to be wrong about) + **A9** imagination-
  error monitor on the lead region; **H1** tactical layer for lane-change consequence pricing.
- **Scenario-spec status:** **DRAFTED + intake pkg 2026-07-10** →
  `Implementation/incoming/2026-07-10-stationary-lead-scenario/` (**SC-13**; `stationary_lead.py` +
  telemetry oracle, **13/13 offline tests**). Design-oracle: over the {8…25} m/s approach sweep
  **collision rate imagination_forward 0.0 / classifier_react 0.43**; at 15 m/s imagination brakes
  **3.1 s earlier**, holds **min-TTC 4.4 s vs 0.77 s** and a **29.8 m vs 2.0 m** gap. Honest falsifier
  built in: the edge decays to 0 as the competitor classifies early (detect_range 20→120 m) — it is
  *specifically* acting-before-classification. Real open-loop probe on comma2k19 = DataEng handoff;
  `collision_rate` reducer = Benchmarks & Eval handoff.
- **Training-data recipe (H6):** comma2k19 slow/stopped-lead following (real, license-clean, oversample
  the rare stopped-lead tail); CARLA stationary-object + cut-in perturbation rollouts (negative manifold).

---

### Watch-list (not yet weaknesses — competitive-narrative risks)
- **Autobrains "Liquid AI"** — the one competitor whose *efficiency pitch* overlaps ours. **UPDATE
  2026-07-24: moving UP from ADAS toward L4** — Uber + Autobrains (+ NVIDIA) announced a **Munich robotaxi
  pilot (2026-06-02)**, apparently displacing/paralleling Uber's earlier Momenta-Munich plan. Still no
  public hierarchical latent WM / in-loop imagination / self-monitoring-with-guarantees; but the "less
  compute, standard sensors" message now attaches to an L4 pilot → **priority pre-empt** with
  compute-normalized (CNCE) proof on L4-grade edge cases. (H1/H3/H5) — https://www.electrive.com/2026/06/02/uber-and-autobrains-to-partner-on-munich-robotaxi-pilot-project/
- **Metis (arXiv 2606.15869, Fudan/HKU/Tongji/Li Auto, subm. 2026-06-14)** — **deep-read done this run.**
  An "efficient world-action model": Mixture-of-Transformers with separate video-generation and
  action-prediction experts + an **asymmetric attention mask that lets the action head skip generative
  rollout at inference** (its efficiency lever — conceptually our latent/no-pixel path). SOTA on NAVSIM
  navhard/navtest + CityWalker. **But:** flat MoT (**no hierarchy**), **no in-loop imagination for
  planning**, **no self-monitoring/OOD guarantee**, and it reports **no parameter count / no
  compute-normalized metric** → *not* a true CNCE competitor yet. Gap to exploit: publish params + a
  compute-normalized causal-efficacy number Metis doesn't. Nearest academic head-to-head — track its
  code (github.com/LogosRoboticsGroup/Metis) for a param disclosure. (H1/H3/H5/H15/H11)
