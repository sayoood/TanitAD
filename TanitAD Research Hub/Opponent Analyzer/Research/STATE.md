# STATE — Opponent Analyzer

LAST_RUN: **run #4, real wall-clock 2026-07-11** (additive delta sweep + SC-14 authored).
Worktree branch: `worktree-opponent-20260711` (D-026 isolation; orchestrator merges to main).
QUALITY: complete (gates G-A…G-F + G-O1/G-O2 + G-H met; loop 8 searches + 2 fetches, 1 iteration)

## ⚠ RECONCILIATION — two unmerged opponent runs (read first)
Run ordering is by **run number**, not date string (narrative-clock artefact; runs #1/#2 are
date-stringed 07-17/07-24 ahead of the real clock).
- **Run #3** = branch `worktree-agent-opponent-20260710` (commit **874f78e**, pushed, **NOT in main**):
  authored **SC-13 stationary-lead** (`2026-07-10-stationary-lead-scenario/`, collision 0.000 vs
  0.429) + led with the **FMVSS-135 NPRM (H11 tailwind)**.
- **Run #4** = this branch: discovered 874f78e mid-session, **deleted my duplicate SC-13**, and
  pivoted to be additive — ships **SC-14** + the 2nd-source evidence run #3 lacked.
- **Orchestrator: merge 874f78e (run #3) FIRST, then this branch.** They are additive, not competing.

## This run (run #4, 2026-07-11)
- **Measured experiment (G-H): SC-14 red-light barrier** shipped as intake pkg
  `Implementation/incoming/2026-07-11-red-light-barrier-scenario/` (`red_light_barrier.py` + telemetry
  oracle, **11/11 offline tests**). **Reuses the SC-04 barrier-vs-soft-prior oracle** → one
  violation-rate reducer serves both. Design-oracle (P8): **violation rate rule_barrier 0.0 /
  soft_prior 1.0** over the apparent-clearance sweep; barrier stops **1.1 m before** the line and is
  invariant to the temptation while the soft prior's line-crossing speed grows **3.2→10.4 m/s**; OKRI
  toward the occluded crosser **−82%** (12,387 vs 63,765) at 4 B vs 15 B. **SC-14 catalogued →
  spec-drafted.**
- **New tracked opponent: Zoox (Amazon)** — 332-vehicle recall (oncoming-lane crossing / stop in
  oncoming lane), 3rd recall in ~8 months, FMVSS-exemption-dependent → new profile + **2nd FACT
  source for SC-11** (was Waymo-only).
- **Tesla:** Houston **fatality** (2026-06-21, Martha Avila) + special investigation; the **EA26002**
  traffic-violation docket (2.88 M veh, 80 incidents, red-light/illegal-turn, 14 crashes/23 injuries)
  = **2nd major-operator FACT source for SC-14/W-03** (distinct from the 3.2 M visibility EA).
- **Two hard classes now have 2 major-operator FACT sources each:** SC-14 red-light (Waymo + Tesla),
  SC-11 oncoming-lane (Waymo + Zoox) → both upgraded from "one company's bug" to a class.
- **D-028 literature (seam-routed):** GigaWorld-Policy (2603.17240, open-source efficient WAM,
  robotics not AD); WAM dynamic-consistency diagnostic 2605.07514 (→ H11); benchmarks routed to Bench.
- Ledger: H9/H15/H6 change-log evidence row (no status upgrade — design oracle, P8).
- KB: 5 new run-#4 findings. Research note: `2026-07-11-opponent-sweep-w5.md`.

## Recommendations logged for other disciplines (no cross-boundary writes)
- **Benchmarks & Eval (Thu):** wire **SC-14** into the eval set — the **SC-04 `violation_rate`
  reducer applies unchanged** to `_extra.red_light_violation` (one reducer, two scenarios). Skim
  **2605.07514** for the H11 self-monitor metric. [prior SC-04 violation-rate + degraded-visibility
  D8 + competitor-CNCE recs stand; run #3's SC-13 reducer rec also stands.]
- **Data Eng (Tue):** screen the **SC-11 oncoming-lane** class (now Zoox-sourced) in comma2k19
  intersection segments; source a signalized-junction phase recipe for SC-14. [run #3's stopped-lead
  comma2k19 tagging for SC-13 stands.]
- **Tools & DevEnv (Mon):** read **GigaWorld-Policy** (open-source, Metis-like no-rollout lever)
  alongside run #3's AlpaGym/AlpaSim flag.
- **Orchestrator:** **merge 874f78e (run #3) before this branch**; triage the SC-14 intake; log
  **Zoox** + **Tesla Houston fatality / EA26002** as strategy signals.

## HANDOFF / next run (run #5)
- Deltas-only. Priorities: (1) **author SC-11 oncoming-lane spec** — now 2 FACT sources (Waymo + Zoox);
  reuses the barrier oracle (H9 directional barrier + H15 imagined oncoming occupancy); (2) check
  orchestrator verdicts on the **SC-14** (07-11), **SC-13** (874f78e), **Stop-Arm Gate** (07-24)
  intakes and adapt; (3) refresh: Waymo construction-zone remedy status, Zoox FMVSS-exemption
  decision, Tesla EA26002 March-deadline outcome, FMVSS-135 comment deadline (2026-07-27); (4) watch
  **Metis** + **GigaWorld-Policy** repos for a param count → CNCE head-to-head (with Benchmarks & Eval).
- Anchors: Wayve GAIA line; NVIDIA Alpamayo/Cosmos/AlpaSim/AlpaGym; Momenta R7; Metis 2606.15869;
  GigaWorld-Policy 2603.17240; WAM-consistency 2605.07514; latent-WM survey 2603.09086/2606.00133.
