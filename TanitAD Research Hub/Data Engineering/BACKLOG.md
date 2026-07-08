# Data Engineering — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

0. **PhysicalAI-AV R1 expansion (Sayed directive 2026-07-08: leverage all high-quality data;
   license solved later or corpus replaced):** scale the R0 urban scorer 500 → **2,000 clips** on
   pod1 (disk OK; run scorer + fetch during trainer idle-CPU windows; epcache build AFTER the 30k
   run finishes — 62 GB cgroup). Deliverable: R1 clip list + fetch plan + measured ingest cost.
   Everything tagged `data:physicalai` as always. Ranked queue: `DATASET_LANDSCAPE.md` (new section).
1. **PhysicalAI-WorldModel-Synthetic-Scenarios license check + 50-clip pilot** (promoted from P1.4;
   rank #2 in the acquisition queue — H6/H15/D9 long-tail + scenario-DB data rows).
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
- (2026-07-08 loop) **Cosmos T=39 RESOLVED + chunk-pairing bug fixed**: videos = 121-frame
  chunks @30 Hz (container fps is a mux artifact), clips = 10 s/300 poses; chunk-1 videos
  (~half the extract) were getting chunk-0 actions → `_chunk_of` + pose offset + episode-id
  fix, 141 tests green. Cosmos cleared for the D-010 mix. See layout-finding note UPDATE 2.
