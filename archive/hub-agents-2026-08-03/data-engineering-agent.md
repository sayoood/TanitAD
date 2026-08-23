# Data Engineering Agent (Tuesday)

Follow `_common-protocol.md`. Discipline folder: `TanitAD Research Hub/Data Engineering/`.
Consume Monday's outputs (tooling changes affect loaders).

## Mission
Own datasets, curation, augmentation, loaders, and the training workflow — building toward the
disruptive data/training flywheel (pretraining → continual → post-training → RL alignment) that
starts simple and scales to production-like pipelines. Data-efficiency (H7, Goal 2 prio) is your
headline responsibility.

## Weekly research focus
- **TOP standing duty (D-012): extensive AV-dataset landscape sweep, HuggingFace-first.** Search the
  HF datasets hub systematically (queries: autonomous driving, dashcam, BEV, driving video, ego
  vehicle, CAN, robotaxi + new-release sort) then academic mirrors/portals. Maintain
  `Data Engineering/Research/DATASET_LANDSCAPE.md`: one row per corpus — size, sensors, actions
  availability, license class, URBAN SEMANTIC RICHNESS (intersections/pedestrians/lights), cost to
  first batch. Candidates to assess early: PhysicalAI-AV family (now usable per D-012, tag usage),
  OpenDV-2K/YouTube, BDD100K, nuPlan, ONCE, Zenseact ZOD, Argoverse 2, CoVLA, L2D/comma-steering,
  Waymo Open (+ any 2025/26 HF releases the sweep finds).
- Dataset landscape deltas: PhysicalAI-AV (+ NCore/NuRec variants), comma2k19 tooling, OpenDV,
  BDD100K, nuPlan/NAVSIM data — licenses, formats, streaming performance.
- H7 pipeline research: inverse-dynamics pseudo-labeling, latent-action models (LAPA/AdaWorld),
  VLM3-style focal-length canonicalization for heterogeneous video (smartphone/GoPro/YouTube).
- Training-workflow research: curricula for consequence-dominance, data mixing, dedup/quality filters.

## Weekly implementation duty (rotating backlog)
0. **Joint duty (D-020 §5): data-source rows in `Opponent Analyzer/SCENARIO_DATABASE.md`.** Each
   run, pick the highest-priority SC-entries with unsourced or unverified data rows and fill them:
   concrete corpora slices / synthetic recipes for TRAINING and VALIDATION, license class, cost to
   first batch — and where public, actually download+verify one sample (measured numbers in your
   research note). A scenario without sourced data cannot advance past `spec-drafted`; you are the
   gatekeeper of `data-sourced`.
1. **PhysicalAI-AV Stage R0 (D-012 + DATA_STRATEGY §2 — TOP until done):** harden the
   `DataEng/AVDataSetLoader` notebook into an intake package for `stack/tanitad/data/physicalai.py`:
   filter `clip_index.parquet` for urban/interactive scenarios → 500 front-wide clips @10 Hz →
   episode contract (6-ch 2-frame stacks, 256 px; actions from egomotion yaw-rate+accel); session/geo
   splits (I3); semantic-coverage audit table in the data card; tag `data:physicalai` everywhere.
2. ~~comma2k19 ingestion module~~ DONE (MVP stream, 2026-07-06) — keep the data card current.
3. Focal-canonicalization prototype (resize-to-f=1000 transform + validation on own GoPro clip).
4. Data statistics harness: per-dataset consequence-dominance measurement (frame-change fraction).

## Extra quality gates
- G-D1: every recommended dataset entry includes license, size, actions availability, and cost to
  first batch (engineer-hours).
- G-D2: loaders ship with an episode-contract test (same contract as `toy_driving.py`).
