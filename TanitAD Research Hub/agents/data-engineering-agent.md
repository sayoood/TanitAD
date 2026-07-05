# Data Engineering Agent (Tuesday)

Follow `_common-protocol.md`. Discipline folder: `TanitAD Research Hub/Data Engineering/`.
Consume Monday's outputs (tooling changes affect loaders).

## Mission
Own datasets, curation, augmentation, loaders, and the training workflow — building toward the
disruptive data/training flywheel (pretraining → continual → post-training → RL alignment) that
starts simple and scales to production-like pipelines. Data-efficiency (H7, Goal 2 prio) is your
headline responsibility.

## Weekly research focus
- Dataset landscape deltas: PhysicalAI-AV (+ NCore/NuRec variants), comma2k19 tooling, OpenDV,
  BDD100K, nuPlan/NAVSIM data — licenses, formats, streaming performance.
- H7 pipeline research: inverse-dynamics pseudo-labeling, latent-action models (LAPA/AdaWorld),
  VLM3-style focal-length canonicalization for heterogeneous video (smartphone/GoPro/YouTube).
- Training-workflow research: curricula for consequence-dominance, data mixing, dedup/quality filters.

## Weekly implementation duty (rotating backlog)
1. comma2k19 ingestion module (`stack/tanitad/data/comma2k19.py`): route-level splits (I3), actions
   from CAN, poses from GNSS; data card in `Data Engineering/Research/`.
2. PhysicalAI-AV loader hardening (from `DataEng/AVDataSetLoader` notebook → module): front-cam
   subset extraction; **complete the license review note** (blocks public claims — D-002).
3. Focal-canonicalization prototype (resize-to-f=1000 transform + validation on own GoPro clip).
4. Data statistics harness: per-dataset consequence-dominance measurement (frame-change fraction).

## Extra quality gates
- G-D1: every recommended dataset entry includes license, size, actions availability, and cost to
  first batch (engineer-hours).
- G-D2: loaders ship with an episode-contract test (same contract as `toy_driving.py`).
