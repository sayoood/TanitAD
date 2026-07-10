# STATE — Opponent Analyzer

LAST_RUN: 2026-07-10 (run #3 — wall-clock; branch `worktree-agent-opponent-20260710`) — deltas sweep +
SC-13 Stationary-lead scenario shipped + FMVSS-135 regulatory finding
QUALITY: complete (all gates G-A…G-F + G-O1/G-O2 + G-H met; loop 1 iteration, 7/25 searches, < 1 h)

> **Timeline note (P8):** the system wall-clock for this scheduled run is 2026-07-10 (Friday). The two
> prior notes carry narrative week-labels (07-17 run #1, 07-24 run #2) that run ahead of the wall clock
> (autonomous-loop artefact). This run is dated to the real clock and continues as run #3; all events
> are cross-checked against live July-2026 web results (see the run-3 note's timeline block).

## This run (2026-07-10)
- **Measured experiment (G-H): SC-13 Stationary-lead scenario** shipped as intake pkg
  `Implementation/incoming/2026-07-10-stationary-lead-scenario/` (`stationary_lead.py` + telemetry
  oracle, **13/13 offline tests**, numpy-only, RTX-4060, $0). Design-oracle (P8, not our model):
  H15 `imagination_forward` vs `classifier_react` — **collision rate 0.000 vs 0.429** over the
  {8…25} m/s approach sweep; at 15 m/s imagination brakes **3.10 s earlier** (onset 2.90 vs 6.00 s),
  keeps **min-TTC 4.40 vs 0.77 s** and a **29.8 vs 2.0 m** gap at half the peak jerk; OKRI 7 vs 18,220.
  **Honest falsifier built into the oracle:** the lead decays 3.10→−2.90 s and react's collisions
  vanish as `detect_range_m` grows 20→120 m → the edge is *specifically* acting-before-classification.
  **SC-13 → spec-drafted.**
- **Regulatory finding (FACT): NHTSA FMVSS-135 NPRM (2026-06-26, comments to 07-27)** — proposes that
  an ADS "be aware of the operational status of each safety-critical vehicle system … and respond to
  degradations/failures"; withdraws AV STEP → **regulation-native tailwind for H11**, adjacent W-04/W-07.
- Deltas: Avride PE wording is **verbatim** the SC-13 failure ("did not brake for slow/stopped
  vehicles, struck stationary objects"; operator intervened in 1/16); Waymo confirmed **sixth recall**;
  NVIDIA **AlpaGym** closed-loop RL on AlpaSim+NuRec (usable asset + closed-loop now table stakes);
  Wayve **Stellantis-on-Uber** (CLAIM $2.8 B); Pony 4 driver-out cities + Singapore; Tesla eyeing
  ~5,000 Las Vegas slots; arXiv WM substrate trending to expensive **explicit 4D-occupancy** (our latent
  consequence-forward path is the efficiency counter).
- Ledger: H11/H15/A9 change-log row (no status upgrade — nothing measured on our stack, P8).
- KB: 6 new dated findings. Research note: `2026-07-10-opponent-sweep-w4.md`.

## Recommendations logged for other disciplines (no cross-boundary writes)
- **Benchmarks & Eval (Thu):** (a) add a `collision_rate` reducer over `_extra.collision` + reuse
  `compute_lal` v2 on `ego_v` for the SC-13 anticipation lead; wire SC-13 into the eval set (H15).
  (b) Add the **FMVSS-135 NPRM** row to `REGULATION_TRACE.md` (deadline 07-27; maps to D8/H11).
  (c) SC-04 `violation_rate` reducer request still stands.
- **Data Eng (Tue):** **cheapest high-value item** — tag comma2k19 slow/stopped-lead segments and build
  the SC-13 **real open-loop probe** (predicted-TTC lead vs detection baseline on matched segments).
- **Tools & DevEnv (Mon):** evaluate **NVIDIA AlpaGym/AlpaSim + Omniverse NuRec** as an open
  closed-loop asset (matches the Phase-1 real-geometry+synthetic-hazard doctrine) alongside CARLA-on-pod.
- **Orchestrator:** triage the SC-13 intake; log FMVSS-135 as an H11 tailwind (deck beat) + AlpaGym as
  a competitive signal (closed-loop table stakes).

## HANDOFF / next run
- Deltas-only. Priorities: (1) **SC-14 red-light** spec (near-free off the SC-04 barrier oracle) —
  cheapest next item; (2) check orchestrator verdict on the SC-13 (and SC-04) intakes and adapt;
  (3) once DataEng tags comma2k19 stopped-lead segments, drive the **SC-13 real open-loop probe** to a
  measured lead-time-vs-detection number (turns SC-13 from design-oracle → real evidence);
  (4) reconcile Wayve's $2.8 B vs $1.2 B/$8.6 B; primary-source the **UNECE global driverless rulebook**
  (CLAIM) for REGULATION_TRACE; watch **Metis** repo for a param count → real CNCE head-to-head; watch
  for any Nano-tier (10 B) CNCE number.
- Anchors (citation-graph walk): Wayve GAIA; NVIDIA Alpamayo/Cosmos/AlpaSim/AlpaGym; Momenta R7; Metis
  2606.15869; occupancy-WM DriveFuture 2605.09701 / GenieDrive 2512.12751; latent-WM survey 2603.09086.
