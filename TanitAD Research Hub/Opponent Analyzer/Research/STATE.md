# STATE — Opponent Analyzer

LAST_RUN: run #3 — narrative 2026-07-31, **real wall-clock 2026-07-17** (Fri scheduled run).
  Branch: `agent/opponent-20260717`. Deltas sweep + Stationary-Lead scenario (SC-13) + new W-09.
QUALITY: complete (all gates G-A…G-F + G-O1/G-O2 + G-H met; loop used 11/25 searches, 1 iteration).

> **Clock note (honesty, memory `narrative-clock-ahead-of-wallclock`):** the discipline's narrative clock
> runs ~2 weeks ahead of real time (a loop artefact). Runs are keyed by **run number** (this = run #3),
> not by note date. Notes/packages are dated **2026-07-31** to keep the folder names, module docstrings
> and this STATE internally consistent; the real calendar date is **2026-07-17**.
>
> **Orchestrator dedup flag (D-026):** an unmerged off-schedule branch `agent/opponent-20260715` already
> authored SC-13 (commit `787671a`, "collision 0.00 vs 0.60") and SC-14 speculatively, ahead of the
> schedule. This canonical scheduled branch re-authored **SC-13 cleanly** (14/14 tests). **At merge, pick
> ONE SC-13; do not integrate both.** SC-14 on that branch is likewise not on my line.

## This run (run #3)
- **Measured experiment (G-H): Stationary-Lead scenario** (SC-13, W-08) shipped as intake pkg
  `Implementation/incoming/2026-07-31-stationary-lead-scenario/` — **14/14 offline tests** (venvs/tanitad
  py3.13, numpy 2.5.1; RTX-4060 CPU-only; <1 s; $0). Design-oracle over classification-ambiguity {0…1}:
  **collision rate imagination 0.0 / detection-reactive 0.4**; **braking-onset lead time +1.20 s vs
  −1.26 s**; forward model **invariant to ambiguity** (min-TTC 2.88 s, min-gap 10.75 m) while reactive
  degrades to a collision (drops the lead ≥ 0.75); OKRI ~3.2× lower. **SC-13 → spec-drafted.**
- **New weakness W-09 (first-responder / emergency-scene interference)** from the **NHTSA 2026-07-08
  ADS-developers letter** (all-operator, end-July deadline; ≥6 Waymo physical-takeover incidents;
  Morrison: "functional insufficiency", "emergency scenes are not rare or extreme edge cases"). **SC-06
  elevated** catalogued→(priority ↑), now backed by a live federal action + linked to W-09 (H15/H11/A9/H9).
- **Deltas:** Waymo "robotaxi ultimatum" (fighting NHTSA + Uber at once; W-01/W-03/W-09 all live); Wayve
  **+$60 M** AMD/Arm/Qualcomm + Tokyo pilot; Pony **Q2 200+ Gen-7 / rev +76%** (net loss $50.4 M — W-06
  intact); Momenta→Autobrains Munich = **EU resistance to Chinese key-tech**; NVIDIA CLA = MB.Drive
  Assist Pro (no Nano-tier CNCE number); richer **Avride** FACT detail (16 crashes, only 1/16 monitor
  intervened) reinforcing SC-13. Emerging: **Zoox** (NHTSA-gated ≤2,500), **WeRide** (Dubai driverless),
  **Nuro+Lucid+Uber** (≥35 k vehicles). Field-scan: **hierarchy surfacing** (SGDrive 2601.05640).
- Ledger: H0/H6/H11 + H15/A9 change-log row (no status upgrade — nothing measured on our stack, P8).
- KB: 6 new dated findings. Research note: `2026-07-31-opponent-sweep-w4.md`.

## Recommendations logged for other disciplines (no cross-boundary writes)
- **Benchmarks & Eval (Thu):** (a) add **min_ttc + collision_rate reducers** over SC-13 `_extra`, reuse
  LAL-v2 over `_extra.brake_onset_lead_time_s`, wire SC-13 into the eval set (H15); (b) define a **W-09/
  SC-06 emergency-scene metric** (corridor-clear time + non-nominal-scene-detected flag). [SC-04
  violation-rate reducer + SC-05 degraded-visibility D8 stressor still stand.]
- **Data Eng (Tue):** **top cheap win** — tag stopped/slow-lead + stationary-object segments in comma2k19
  for the SC-13 open-loop probe; screen dashcam corpora for flashing-light/emergency-scene events (W-09).
- **Tools & DevEnv (Mon):** for W-09/SC-06 CARLA build use emergency-vehicle + light-pattern assets
  (visual-only proxy); evaluate **AlpaSim** as closed-loop harness (still open from run #2).
- **Architecture & Inference (Wed):** deep-read **SGDrive (2601.05640)** — hierarchy at planning time or
  representation-only? Feeds the H1 differentiation story.
- **Orchestrator:** (a) triage the Stationary-Lead intake + **dedup SC-13 vs `agent/opponent-20260715`**;
  (b) log the **NHTSA first-responder directive** as a strategy-grade signal for the vision deck / weekly
  report ("emergency scenes are not edge cases" = strongest external endorsement of the scenario-DB
  thesis); (c) note the Uber multi-vendor L4 marketplace (distribution moat is Uber's → technical-moat
  premium rises).

## HANDOFF / next run (run #4)
- Deltas-only. Priorities: (1) **author SC-06 / W-09 emergency-scene spec** as the next scenario feed
  (backlog P0.1) — mirror stationary_lead/stop_arm_gate; reuse W-01 changed-area machinery + a
  corridor-clear/OOD-flag oracle; (2) check the orchestrator verdict on the Stationary-Lead intake (and
  the SC-13 dedup) and adapt; (3) **SGDrive deep-read** (hierarchy planning-time vs representation) →
  sharpen the H1 differentiation; (4) refresh NHTSA SGO / any new Waymo–Tesla–Avride–Zoox recall or the
  end-July first-responder-fix outcomes; (5) SC-14 red-light spec only after the dedup verdict (already
  authored on the unmerged branch); watch whether NVIDIA publishes a Nano-tier CNCE number.
- Anchors (citation-graph walk): Wayve GAIA line; NVIDIA Alpamayo/Cosmos/AlpaSim; Momenta R7; Metis
  2606.15869; **SGDrive 2601.05640 (hierarchy)**; DriveFuture 2605.09701; latent-WM taxonomy 2603.09086;
  adjacent-domain SkyJEPA 2606.23444.
