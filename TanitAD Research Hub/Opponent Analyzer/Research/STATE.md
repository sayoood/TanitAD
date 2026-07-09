# STATE — Opponent Analyzer

LAST_RUN: 2026-07-24 (run #2 — deltas sweep + Stop-Arm Gate scenario + Metis deep-read)
QUALITY: complete (all gates G-A…G-F + G-O1/G-O2 + G-H met; loop used 9/25 searches, 1 iteration)

## This run (2026-07-24)
- **New emerging player: Avride** (Uber partner, Yandex SDG) — NHTSA ODI investigation (16 crashes,
  competence: lane-change / same-lane / stationary-object) → new profile + **W-08** + **SC-13**.
- **Measured experiment (G-H): Stop-Arm Gate scenario** (SC-04, W-03) shipped as intake pkg
  `Implementation/incoming/2026-07-24-stop-arm-gate-scenario/` — **11/11 offline tests**. Design-oracle:
  H9 **violation rate rule_barrier 0.0 / soft_prior 1.0** over the free-path sweep; barrier invariant to
  temptation, soft prior's line-crossing speed grows 3.0→9.6 m/s; OKRI toward the occluded child 80%
  lower at 4 B vs 15 B params. **SC-04 → spec-drafted.**
- **Metis deep-read** (arXiv 2606.15869): efficient WAM via MoT + asymmetric mask (action head skips
  video rollout at inference). Nearest CNCE competitor but flat/no-hierarchy/no-imagination/no-monitoring
  and reports no params → not a true CNCE rival yet. Watch its code for a param disclosure.
- Deltas: Waymo 2nd recall + highways pulled + Dallas red-light (**SC-14**); Tesla EA 3.2 M / 9 crashes /
  1 fatality naming the failed "degradation-detection" feature; Momenta listed + R7 RL WM + X7 chip;
  Autobrains→L4 (Uber Munich pilot); NVIDIA Mercedes CLA ships Alpamayo + AlpaSim open-sourced (10 B Nano
  tier). "World model" now table stakes → moat = hierarchy+CNCE+imagination+self-monitoring.
- Ledger: H0/H6/H9 change-log row (no status upgrade — nothing measured on our stack, P8).
- KB: 6 new dated findings. Research note: `2026-07-24-opponent-sweep-w3.md`.

## Recommendations logged for other disciplines (no cross-boundary writes)
- **Benchmarks & Eval (Thu):** add a `violation_rate` reducer over `_extra.stop_arm_violation` (a rate,
  not a soft score) + wire SC-04 into the eval set (H9). [prior run's degraded-visibility D8 stressor +
  competitor-param CNCE recs still stand.]
- **Data Eng (Tue):** source school-bus/stop-arm asset + screen dashcam corpora (SC-04/SC-14); **cheap
  win** — tag stopped/slow-lead comma2k19 segments for SC-13 (Avride competence, license-clean).
- **Tools & DevEnv (Mon):** evaluate **AlpaSim** (open-source closed-loop sim, NVIDIA/GitHub) as a
  closed-loop asset alongside CARLA-on-pod (D5/D6, G0.5).
- **Orchestrator:** triage Stop-Arm Gate intake; log Autobrains ADAS→L4 + Avride entrant as strategy signals.

## HANDOFF / next run (2026-07-31)
- Deltas-only. Priorities: (1) **author SC-13 (stationary-object / same-lead) spec** as the next scenario
  feed — reuses comma2k19 + LAL-v2/OKRI, cheapest high-value item now (Avride/W-08); (2) check orchestrator
  verdict on the Stop-Arm Gate intake and adapt; (3) **SC-14 red-light** spec (near-free off SC-04 oracle)
  if SC-13 lands; (4) watch **Metis** code repo for a param count → then a real CNCE head-to-head;
  (5) refresh NHTSA SGO / any new Waymo–Tesla–Avride recall; watch whether NVIDIA publishes a Nano-tier
  CNCE number.
- Anchors (citation-graph walk): Wayve GAIA line; NVIDIA Alpamayo/Cosmos/AlpaSim; Momenta R7; Metis
  2606.15869; latent-WM survey 2603.09086; hierarchy anchor 2604.03208; adjacent-domain SkyJEPA 2606.23444.
