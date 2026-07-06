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
- **Evidence (FACT):** Waymo **recalled 3,871 robotaxis (2026-06-18)** after **13 freeway construction-zone
  incidents** — 6 Phoenix (failed to recognize **ramp-closure signs**), 7 SF Bay (drove **between
  lane-closure cones**). Freeway autonomy suspended; 20+-city 2026 expansion frozen.
  — https://www.cnbc.com/2026/06/18/waymo-nhtsa-voluntary-recall-robotaxis-entered-freeway-construction-zones.html
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
- **TanitAD counter:** **H9** (inherent rule compliance via RMFM / hard barrier terms — a violation-rate
  metric, not a learned soft prior). Owner of the violation-rate metric home: Benchmarks&Eval.
- **Scenario-spec status:** `no-scenario-yet` — **candidate for the August scenario feed** ("Stop-Arm
  Gate": stopped school bus with extended stop-arm across a multi-lane road). Flag to Thursday agent.
- **Training-data recipe (H6):** synthetic stop-arm events (rare in real logs); barrier-term supervision
  rather than pure imitation.

## W-04 — Camera-only degraded-visibility (glare, rain, airborne obscurant)

- **Mechanism (INFER):** a camera-only stack with **no calibrated epistemic uncertainty** stays confident
  when the sensing channel degrades (sun glare, heavy rain, dust/smoke), so it neither slows nor hands
  back — it drives blind at speed.
- **Evidence (FACT):** **NHTSA engineering analysis (Mar 2026)** on Tesla — camera-only FSD **"fails to
  detect and/or warn … under degraded visibility such as glare and airborne obscurants."** Final step
  before a possible recall. — https://www.automotiveworld.com/news/tesla-robotaxi-fleet-hits-25-as-musk-defers-scale-to-fsd-v15/
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

---

### Watch-list (not yet weaknesses — competitive-narrative risks)
- **Autobrains "Liquid AI"** — the one competitor whose *efficiency pitch* overlaps ours; L2+/ADAS only,
  no L4 WM/imagination/self-monitoring. Pre-empt any compute-normalized claim they publish. (H1/H3/H5)
- **Metis (arXiv 2606.15869)** — academic "efficient world-action model"; nearest CNCE head-to-head.
  Deep-read next run.
