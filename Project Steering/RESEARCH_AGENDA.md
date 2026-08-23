# RESEARCH AGENDA — mandatory research lists per Lab field

`Created 2026-08-22. PROPOSALS FOR PI REVIEW AND AUGMENTATION (the PI will
review and extend). Each item is phrased as a RESEARCH QUESTION with the process
it must follow: literature survey → ideation (own hypotheses) → experiments →
synthesis → paper-grade writeup. Items marked ⭐ are the PI's own examples or
derive directly from measured programme needs.`

## Field 1 — Data Engineering for AI & autonomous driving

1. ⭐ **Data amount & distribution science.** Research the latest approaches —
   and autonomously create NEW ones — for the proven methodology of defining the
   REQUIRED amount of data and its distribution to achieve best results in
   training physical AI, especially self-driving cars and robots. Deliverables:
   scaling-law fits on OUR corpora (with fit window, R², n per the exponent
   rule), a data-value estimator, and a "minimum data for target metric" recipe.
2. **Curation that provably beats random.** Active-learning / coreset /
   difficulty-scoring methods; the moat criterion is a curated set beating a
   random set of equal size on a fixed model.
3. **Auto-labeling & pseudo-labeling at scale.** VLM-ensemble labeling with
   hindsight-geometry arbitration (our Engine A/B pattern), confidence
   calibration, and cost-per-verified-label as the metric.
4. **The IDM frontier** (P3, owned by DataFlyWheel; Lab feeds it): action
   reconstruction from observation-only video — latent-action models (Genie,
   LAPO), BCO/ILPO lineage, V-JEPA-AC-style action heads; monocular scale
   resolution (camera height, speed priors); intrinsics-variance handling for
   YouTube-scale ingestion.
5. **Scenario mining & long-tail discovery.** Embedding-space novelty search on
   TanitScena; automatic scenario taxonomy induction vs our aligned vocabulary.
6. ⭐ **TanitSpear groundwork.** Survey generation/rendering/augmentation
   (GAIA-class, Vista, DriveDreamer, NuRec/gsplat) with the explicit goal of a
   RADICALLY CHEAPER path — small-scale proofs first (reduced resolution, fewer
   scenes, clever tricks from other disciplines). Includes neural rendering on
   our own recorded clips (we have gsplat at 492 FPS on Thor already).

## Field 2 — Architecture & Inference for AI & autonomous driving

1. ⭐ **Anti-collapse representation learning** (the live thread): what keeps a
   trunk full-rank AND decodable — two-term objectives (LeWM), subspace
   regularization (Sub-JEPA), prediction-horizon effects (our k=1 result),
   input-stacking effects (H-RANK-8); publish our elimination tree.
2. **Hierarchical world models.** Multi-timescale latents; goal-conditioned
   planning; our 4B architecture vs flat baselines with the C6-confound
   discipline; the strategic layer's own predictor.
3. ⭐ **Continuous learning for intelligent vehicles** (future-product
   groundwork): latent-RAG as a continuous-learning mechanism; the idle-time
   training layer (the car trains itself while parked); replay/rehearsal
   without catastrophic forgetting; self-monitoring and uncertainty (when does
   the model KNOW it doesn't know).
4. **Cross-discipline transfer watch.** Standing scan of LLM/foundation-model
   techniques with AD transfer potential: test-time compute/reasoning, MoE
   efficiency, state-space models for long horizons, distillation recipes,
   tokenizer lessons for continuous signals.
5. **Encoder question** (feeds PREREG_E_ENC_3WAY): scratch vs fine-tuned vs
   frozen foundation encoders for driving; what pretraining actually transfers;
   DINOv3/V-JEPA-2.1 dense-feature exploitation.
6. **Efficient inference architectures.** Sub-300M excellence: what capacity
   ratio encoder/predictor/planner wins at fixed total (our 7.79× finding says
   ratio alone is not it); early-exit; caching latents across ticks.

## Field 3 — Deployment & Optimization for AI & autonomous driving

1. **Quantization SOTA sweep.** INT8/FP8/INT4, QAT vs PTQ, per-layer
   sensitivity on OUR models; the paired-eval rule (a quantization without a
   paired eval is not a deployment).
2. **Compilation & kernels for edge.** TensorRT vs torch.compile vs manual CUDA
   graphs on Thor (Triton absent on the dev box — measured); where the 20-SM
   batch-8 saturation comes from and what it implies for architecture.
3. **Memory-bounded inference.** Unified-memory scheduling on Thor; KV/latent
   cache management for the hierarchy; the admissible-probe discipline.
4. **Hardware abstraction without efficiency loss** (TrainingFlyWheel mandate,
   Lab researches): one training/inference stack across Thor / dev box / Colab /
   HF; what abstraction layers cost in practice, measured.
5. **On-vehicle continuous-learning feasibility**: what training is affordable
   on embedded compute during idle time (links to Field 2's idle-layer).

## Field 4 — Opponent Analysis  *(separated from Benchmarks & Evals, PI 2026-08-22)*

1. **Opponent teardown series.** Per major stack (Wayve GAIA line, NVIDIA
   Alpamayo/Cosmos, Waymo EMMA-class, openpilot, UniAD/VAD lineage,
   DriveVLM-class): architecture, data strategy, eval claims, and what we
   adopt/refute. One teardown per week.
2. **Competitor data-strategy analysis.** What data moats the major stacks
   actually have, how they built them, and where ours can be cheaper.
3. **Capability-claim verification.** For each major opponent claim, the
   cheapest discriminating check; publish confirm/refute with evidence.
4. **Trend detection.** Which directions the field is converging on (VLM-
   planners, world-model pretraining, end-to-end vs modular) and where the
   contrarian opportunity lies for a sub-300M stack.

## Field 5 — Benchmarks & Evals  *(separated per PI 2026-08-22)*

1. ⭐ **Standard-benchmark adoption (MANDATORY): NavSim, nuScenes, and peers** —
   their exact protocols, metrics (PDMS, EPDMS, NC/DAC/TTC/EP), submission
   formats; what our stack needs to be comparable; run them verbatim and cited.
2. **Eval science.** Open-loop ↔ closed-loop divergence (our 4.05× is a
   finding); action-echo detection as a standard test; leakage taxonomies; the
   estimator discipline as a publishable methods contribution.
3. **Leaderboard integrity.** Reproducible third-party numbers: rerun published
   baselines under our harness where licences allow; publish claimed-vs-
   reproduced deltas.
4. **Metric design for hierarchies.** The four-families doctrine formalised;
   per-level credit assignment; seam metrics with valid estimators.
5. **Safety/regulation watch.** UNECE R157/ALKS-class requirements mapped to
   measurable eval criteria (regulation banked in the Library).

## Process note

Each item, when picked up, becomes a §3 work package (SPEC with both outcomes
first) and registers its hypotheses as `H-…` rows. A finding with "extremely
good results" triggers the §7 transfer handoff to the owning FlyWheel —
bidirectional, mandatory, never silent.
