# STATE — Opponent Analyzer

LAST_RUN: 2026-07-15 (run #3 — SC-13 stationary-lead scenario + first-responder weakness W-09 + deltas)
QUALITY: complete (all gates G-A…G-F + G-O1/G-O2 + G-H met; loop used 5/25 searches, 1 iteration)

> **Narrative-clock note:** run #3 is dated to **wall-clock 2026-07-15**; the prior runs' stamps
> (07-17, 07-24) run ahead of wall-clock (known loop artefact). Order runs by **run number**, not date.
> Branch: `agent/opponent-20260715` (worktree `C:/Users/Admin/wt-opp`, from HEAD ae67b5c — origin/main
> was 11 commits stale).

## This run (2026-07-15, run #3)
- **Measured experiment (G-H): SC-13 "Stationary-Lead / Same-Lane" scenario** shipped as intake pkg
  `Implementation/incoming/2026-07-15-stationary-lead-scenario/` — **16/16 offline tests**, a **forward-
  simulated** oracle (real kinematics: position/speed/TTC/collision, unlike SC-04's heuristic).
  Design-oracle (P8): default late-classification regime — **`detection_reactive` COLLIDES** (min-TTC
  0.09 s, LAL-v2 lead **−0.50 s**); **`imagination` anticipates** (LAL-v2 **+2.30 s**, min-TTC 1.83 s,
  stops 3.0 m short, no collision, **OKRI −73 %**). **Collision rate 0.60 (reactive) / 0.00
  (imagination)** over the classification-range sweep, invariant to the detection-competence knob.
  **SC-13 → spec-drafted.**
- **New weakness W-09 (FACT): first-responder / emergency-scene interference** — NHTSA letter
  (2026-07-08) to Waymo **and** Tesla over a "clear pattern," ~1-month fix deadline; Waymo's July-4 SF
  stall (dead batteries, towed) is the trigger. New **SC-15**; enriches SC-06/SC-08.
- **Sharper Avride evidence (FACT):** ODI = *"stationary objects partially obstructing the lane ahead"*
  + *"excessive assertiveness / insufficient capability … may constitute traffic safety violations,"*
  under a safety monitor → directly validates SC-13.
- **Honesty delta (FACT, P8):** Pony.ai reports Guangzhou city-level **break-even** + 10k-vehicle goal
  + Bolt EU deal → partially blunts W-06; shift the argument to CNCE + data-efficiency slope.
- **Tesla (FACT):** new fatal Model-3 Autopilot probe (Katy TX). **Lit:** latent-WM surge continues
  (Latent-WAM/DriveWorld-VLA/DriveFuture/survey) — moat read unchanged.
- **GOALS.md created** (D-029 gap — was missing). Ledger H15/A9/H0/H6 change-log row (no upgrade, P8).
  KB: 6 new dated findings. Research note: `2026-07-15-opponent-sweep-run3.md`.

## Recommendations logged for other disciplines (no cross-boundary writes)
- **Benchmarks & Eval (Thu):** add a `collision_rate` reducer over `_extra.collision` + expose
  `min_ttc_s` as a scenario metric; wire **SC-13** into the eval set (H15). [SC-04 `violation_rate`
  reducer + degraded-visibility D8 stressor + competitor-param CNCE recs still stand.]
- **Data Eng (Tue):** **cheap, high-value** — tag stopped/slow-lead comma2k19 segments (license-clean)
  for the **SC-13 real open-loop lead-time probe**; source a stalled-vehicle CARLA asset (blocked_route).
  [SC-04 school-bus asset still pending.]
- **Tools & DevEnv (Mon):** SC-15 (W-09) needs the CARLA emergency-scene + connectivity-loss/stall
  injection harness; AlpaSim eval still worth a look.
- **Orchestrator:** triage the SC-13 intake; log **W-09 first-responder interference** (Waymo+Tesla,
  regulator deadline) + the **Pony break-even** honesty delta as strategy signals; note the
  narrative-clock date gap (STATE said run #2 = 07-24; this run #3 dated to wall-clock 07-15).

## HANDOFF / next run (run #4)
- Deltas-only. Priorities: (1) **SC-13 real open-loop probe** on comma2k19 stopped-lead segments once
  DataEng tags them — the trained-model falsifier step (else advance SC-14); (2) **SC-14 red-light**
  spec (near-free off SC-04 oracle); (3) **SC-15 emergency-scene** oracle (W-09) — author now,
  live-measure when the CARLA harness lands; (4) check orchestrator verdicts on the SC-04 + SC-13
  intakes and adapt; (5) refresh NHTSA SGO / any new Waymo–Tesla–Avride recall; watch Metis github for
  a param count → real CNCE head-to-head; (6) advance a GOALS.md goal with a measured step.
- Anchors (citation-graph walk): Wayve GAIA line; NVIDIA Alpamayo/Cosmos/AlpaSim; Momenta R7; Metis
  2606.15869; Latent-WAM 2603.24581; DriveWorld-VLA 2602.06521; latent-WM survey 2606.00133; hierarchy
  anchor 2604.03208; adjacent-domain SkyJEPA 2606.23444.
