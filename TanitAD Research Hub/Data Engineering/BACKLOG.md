# Data Engineering — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

0. **Land per-clip intrinsics (PROMOTED 2026-07-11 — measured: wrong focal = ~10–15× encoder drift).**
   Replace the nominal 120° focal in `physicalai.py` with PhysicalAI `calibration/` per-clip focal; add
   per-video focal recovery for YouTube (Y1). Wire `assert_effective_focal` (shipped intake
   `2026-07-11-focal-invariance-validation/`) into each loader's data-card path so a no-headroom camera
   fails at ingest. Expected: per-corpus f_eff within ±3 % of 266, guard green on comma/cosmos/physicalai.
1. **R1 top-up to 2,000 (BLOCKED on the 30k pod run finishing; measured 2026-07-09: 1,926 reachable,
   74 short).** When the pod frees: fetch **2 more egomotion chunks** (~4 GB), re-run
   `physicalai_r1.py --target 2000` → clears 2,000; then camera-fetch (≤32 chunks, ~64 GB, extract ALL
   gate-passing clips per chunk → 3.85× clips free); epcache AFTER the trainer frees the cgroup.
   ~~WorldModel-Synthetic pose probe~~ **DONE on branch `agent/data-engineering-20260710` (`96d85eb`):
   NO-POSE → video-only loader; awaiting orchestrator merge.**
2. **SCENARIO_DATABASE data sourcing (joint duty, D-020 §5)** — SC-02/05/06 (2026-07-09) merged; **SC-13
   advanced to data-sourced (partial) 2026-07-11** (comma Chunk_1 stopped-lead subset measured THIN).
   Next: **SC-13 full 10-chunk stop-go mine** (or CARLA lead-brake recipe) for the matched hard-braking
   set; then fill SC-04 (stop-arm) + SC-11 (wrong-side) with CARLA recipes / Cosmos obstruction clips.

## P1

2e. **L2D loader — Phase-1 strategic-supervision ingest (PROMOTED from the 2026-07-11 survey; the
   recommended fix for REF-B's `follow`-starvation).** Build `stack/tanitad/data/l2d.py` mirroring the
   Cosmos loader + D-016: `front_left`→`focal_crop_resize` (fixed KIA rig), stream a **filtered slice**
   (non-`follow` maneuver tail, ~2,000 eps balanced over the 4,219 task classes), `CORPUS_META`≡comma
   (D-017 I7), and the `l2d_contract_map.py` action/instruction helper already shipped (intake
   `2026-07-11-semantic-label-survey/`, 10✓). **First cheap experiment (pre-loader):** decode 200 eps →
   (a) confirm `action.continuous[3]→[steer,accel]` is physically sane (sign/unit — the recommendation's
   falsifier); (b) `front_left` f_eff after D-016 within ±25 % (cos≥.92); (c) measure the label-entropy
   gap (comma ~1 class vs L2D effective-#classes) to quantify REF-B starvation numerically. Falsifier
   fail → L2D drops to EVAL-only, nuPlan becomes the train pick. Resource: HF stream + 4060, no pod.
   ~~2d Semantic/strategic-label survey (Sayed directive)~~ **DONE 2026-07-11** → L2D #1 recommendation
   (Apache-2.0, 4,219 nav cmds + real actions), ranked note `2026-07-11-semantic-strategic-label-dataset-survey.md`.

2c. **Y-pilot-50 (Sayed directive 2026-07-09): YouTube dashcam pilot** — 50 diverse videos through
   the full Y0-Y2 pipeline (self-calibration -> canonicalize -> filter -> pseudo-actions); measure
   focal-recovery spread, action agreement r vs VO, A8, probe-fit vs comma. Strategy + falsifiers:
   `Research/YOUTUBE_DASHCAM_STRATEGY.md`. Go/no-go for the Phase-1 OpenDV-2K ingest.


2b. **NuRec feasibility probe (Sayed suggestion 2026-07-09, scenario-data doctrine):** what NuRec
   (neural-reconstruction) assets exist in the PhysicalAI-AV HF family (NuRec/NCore variants),
   their format (Gaussian-splat? USD?), license class, and whether reconstructed scenes can be
   perturbed (insert cones/occluders) with available tooling. Deliverable: yes/no adoption note +
   one downloaded sample inspected. Feeds the Phase-1 targeted-generation WP (Cosmos-conditioned
   scenario sweeps + NuRec real-geometry eval scenes).

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
