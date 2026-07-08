# Data Engineering — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

1. **Cosmos T=39 temporal semantics** — episodes decode to 39 frames, shorter than nominal
   20 s @ 10 Hz; determine true fps + pose-stream rate before cosmos enters training windows.
   Method: ffprobe 5 extracted clips on the pod (`/workspace/data/cosmos/extracted`), cross-check
   pose timestamps vs frame count, verify dt against vehicle_pose deltas. Expected: a definite
   dt (e.g. 2 Hz keyframes or clipped 4 s segments). Falsifier: inconsistent dt across clips ⇒
   cosmos held out of training mix until resolved (P8). Wall-clock ~30 min pod.
2. **SCENARIO_DATABASE data sourcing (joint duty, D-020 §5)** — for the top-3 unsourced scenarios
   in `Opponent Analyzer/SCENARIO_DATABASE.md`, find concrete training/validation data (public
   corpora slices, Cosmos weather variants, CARLA recipes). Deliverable: filled data-source rows
   + one downloaded/verified sample each where public.

## P1

3. **Zenseact ZOD pilot loader** — real-CAN corpus #2 (EU/night distribution comma lacks);
   ~3–4 h loader cost, contract tests on 5 drives; feeds H4 arm-B diversity. Expected: CORPUS_META
   fingerprint byte-compatible; A8 consequence stat reported. License: research/NC — tag like
   physicalai until reviewed.
4. **PhysicalAI-WorldModel-Synthetic-Scenarios license + pilot** — verify HF card; if claimable,
   mirror the cosmos loader (shares pose/contract code, near-zero cost). This is H6/H15/D9
   long-tail material (emergency/lanechange/pedestrian/weather_degradation).
5. **`data:physicalai` tag audit** — grep ledger/leaderboard/paper for untagged PhysicalAI-AV
   numbers; one-command audit script committed to Implementation/. Expected: 0 violations.
6. **comma2k19 Chunk_2–10 streaming plan** — measure per-chunk ingest cost (curl + sanitizing
   extractor + epcache) on one additional chunk; extrapolate to full 33 h for the Phase-0 close.

## P2

7. **BDD100K + IDM pseudo-labels** (H7 proof-of-concept, Phase 1) — needs trained inv-dyn head;
   design note + 100-clip pilot once 30k checkpoint lands.
8. **Monthly HF `datasets` sweep** for new 2026 AV video releases (standing duty, D-012).
9. **OpenDV-2K heterogeneous-focal canonicalization** (H7 flywheel, Phase 1) — depends on D-016
   robustness beyond two corpora.

## Done / retired
- (2026-07-08) Cosmos loader integrated; verify_real_clip PASSED on real bytes (A8=0.109,
  60/60 pose pairing after base-id fix). Layout finding documented.
