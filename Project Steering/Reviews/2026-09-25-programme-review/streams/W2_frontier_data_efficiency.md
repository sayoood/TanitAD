# Stream W2 — Frontier research: massively reducing labelled-data needs for driving; knowledge injection; continual/fleet learning

**Status:** IN PROGRESS (banking incrementally). **Date:** 2026-09-25. **Model tier:** Sonnet.
**Method:** web research (WebSearch/WebFetch), primary sources + arXiv ids, evidence-class discipline per
`CLAUDE.md` rule "Operating standard §1". Search/fetch budget: ≤45 searches, ≤30 fetches.

**Delta skim performed against** (headers/conclusions only, 2026-09-25):
`TanitAD Research Hub/2026-07-08-screening-digest.md`, `INITIAL_RESEARCH_SYNTHESIS.md` (v1.1,
2026-07-06), `Data Engineering/{DATA_STRATEGY_FOR_HIERARCHY,OWN_DATASET_PLAN}.md`,
`Architecture & Inference/IDM_VIDEO_PRETRAIN_DESIGN.md` (v0, 2026-07-22),
`Project Steering/PREREG_deep_research_2026-07-29.md`.

**What is already banked internally (do NOT re-report as new; this stream hunts the DELTA since):**
- H7 (data-leverage-via-IDM) already a Phase-1-core recommendation as of 2026-07-06/07-22, citing
  VPT (arXiv:2206.11795), Genie (2402.15391), LAPA (2410.11758), BCO (1805.01954), Seer (ICLR'25,
  2412.15109), DriveWAM (2605.28544), IDM-vs-BC (2602.02762). Design: non-causal predictive IDM head
  reusing the flagship WM predictor trunk, trained on CAN-labeled corpus, applied to YouTube.
  **Per the orchestrator fact sheet this line has since FAILED held-out-camera-rig transfer** — this
  is the single most important open question for §C below, and the reason §C is not "redo H7" but
  "why did it fail and what specifically fixes it."
- L2D (Apache-2.0) identified as the strategic/horizon/indicator unblocker (2026-07-21); obstacle.offline
  lead-state ingest recommendation **falsified** by gate (2026-07-21): lead state does not improve
  ego-only longitudinal ADE at the FULL-population level (+1.16% [-0.92,+3.19], inside FAIL band);
  survives as a lead-conditioned-specialist hypothesis on the 38.5% lead-present subpopulation, not yet
  pre-registered.
- Data-scaling-laws for E2E driving and Waymo's motion-forecasting scaling work are NOT yet in the
  hub docs skimmed — open territory for §A below.
- Own-dataset plan already scoped: comma2k19, Cosmos-Drive-Dreams (CC-BY-4.0), PhysicalAI-WorldModel-
  Synthetic-Scenarios (OpenMDW-1.1), PandaSet, Udacity, CARLA self-gen as the license-clean core;
  ZOD (CC-BY-SA) as flagship new real-urban ingest. This stream does not re-litigate licensing (that's
  DataEng's job) but flags where new 2026 sources/methods bear on it.
- RMFM (Reward-Modulated Flow Matching) already the Phase-1 flagship for rule-injection (H9) —
  relevant context for §H (knowledge injection) below, not re-derived here.
- σ-gated tactical MoE, K-step rollout loss, RoPE-in-FiLM already triaged into BACKLOG — not repeated.

---

## A. Data scaling laws for driving/planning

*(section pending)*

## B. Self-supervised & video pretraining for driving policies

*(section pending)*

## C. Pseudo-labelling unlabelled video with an inverse-dynamics model — camera-rig transfer

*(section pending)*

## D. Synthetic & generative data

*(section pending)*

## E. Curation & active learning

*(section pending)*

## F. Distillation & privileged teachers

*(section pending)*

## G. Structural priors that substitute for data

*(section pending)*

## H. Knowledge injection & continual/fleet learning

*(section pending)*

## Ranked TOP-10 for TanitAD

*(pending)*

## A concrete "10x less labelled data" programme

*(pending)*

## What NOT to do

*(pending)*

## Deliverable manifest

*(pending)*
